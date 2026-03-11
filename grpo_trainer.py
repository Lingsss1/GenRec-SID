"""
GRPO Trainer for generative recommendation.

Adapted from MiniOneRec's ReReTrainer (minionerec_trainer.py).
Implements Group Relative Policy Optimization with constrained decoding
for SID-based sequential recommendation.
"""

import os
import math
import random
import warnings
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TemperatureLogitsWarper,
    LogitsProcessorList,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation import LogitsProcessor
from transformers.utils import is_peft_available

try:
    from trl import apply_chat_template, is_conversational, maybe_apply_chat_template
except ImportError:
    from trl import maybe_apply_chat_template
    is_conversational = None
    apply_chat_template = None

try:
    from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
except ImportError:
    from trl.experimental.utils import create_reference_model
    from trl.models import prepare_deepspeed, unwrap_model_for_generation

from trl import SyncRefModelCallback, GRPOConfig

try:
    from trl.trainer.utils import pad, selective_log_softmax
except ImportError:
    from trl.trainer.utils import pad
    def selective_log_softmax(logits, input_ids):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# ============================================================
# Constrained Logits Processor
# ============================================================

class ConstrainedLogitsProcessor(LogitsProcessor):
    """Constrains generation to valid SID token sequences via prefix hash lookup."""

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None,
        prefix_index: int = None,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0
        self.base_model = base_model or ""
        self.eos_token_id = eos_token_id
        if prefix_index is not None:
            self.prefix_index = prefix_index
        elif "gpt2" in self.base_model.lower():
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))

        for batch_id, beam_sent in enumerate(
            input_ids.view(-1, self._num_beams, input_ids.shape[-1])
        ):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index :]
                else:
                    hash_key = sent[-self.count :]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue

                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1
        scores = scores + mask
        return scores


# ============================================================
# Repeat Random Sampler
# ============================================================

class RepeatRandomSampler(Sampler):
    """Repeats each index N times in a shuffled order for GRPO group generation."""

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


# ============================================================
# GRPO Trainer
# ============================================================

class GRPORecTrainer(Trainer):
    """
    GRPO-based trainer for generative recommendation.

    Adapted from MiniOneRec's ReReTrainer to work with this project's
    SID-based data format (CSV with history_item_sid / item_sid columns).
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        base_model: str,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        # sampling
        beam_search: bool = False,
        length_penalty: float = 0.0,
        # eval during training
        test_during_training: bool = True,
        test_beam: int = 20,
        # loss variants
        dapo: bool = False,
        # constrained decoding
        info_file: str = None,
        sid_index_path: str = None,
        # prompt-target mapping
        prompt2history: dict[str, str] = None,
        history2target: dict[str, str] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes=None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config=None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            args = GRPOConfig(f"{model_name.split('/')[-1]}-GRPO")

        self.base_model = base_model
        model_init_kwargs = getattr(args, "model_init_kwargs", None) or {}

        if isinstance(model, str):
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            # 3090 显存优化：使用 float16 而非 bfloat16
            if torch_dtype is None and torch.cuda.is_available():
                if not torch.cuda.is_bf16_supported():
                    model_init_kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                base_model, **model_init_kwargs
            )
        elif not is_peft_model(model):
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(base_model, padding_side="left")
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        _reward_weights = getattr(args, "reward_weights", None)
        if _reward_weights is not None:
            if len(_reward_weights) != len(reward_funcs):
                raise ValueError("reward_weights length must match reward_funcs length")
            self.reward_weights = torch.tensor(_reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        for i, (rpc, rf) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(rf, PreTrainedModel) and rpc is None:
                rpc = AutoTokenizer.from_pretrained(rf.config._name_or_path)
                if rpc.pad_token_id is None:
                    rpc.pad_token = rpc.eos_token
                rf.config.pad_token_id = rpc.pad_token_id
                reward_processing_classes[i] = rpc
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        # Config (兼容不同版本 trl 的 GRPOConfig)
        self.max_prompt_length = getattr(args, "max_prompt_length", None)
        self.max_completion_length = getattr(args, "max_completion_length", 128)
        self.num_generations = getattr(args, "num_generations", 8)
        self.use_vllm = getattr(args, "use_vllm", False)
        self.beta = getattr(args, "beta", 0.04)

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)
        self.log_completions = getattr(args, "log_completions", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.prompt2history = prompt2history or {}
        self.history2target = history2target or {}
        self.beam_search = beam_search
        self.info_file = info_file
        self.sid_index_path = sid_index_path
        self.temperature = getattr(args, "temperature", 1.0)
        self.length_penalty = length_penalty
        self.test_during_training = test_during_training
        self.test_beam = test_beam
        self.dapo = dapo

        # Validate batch sizes
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [
            n for n in range(2, global_batch_size + 1) if global_batch_size % n == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"Global train batch size ({global_batch_size}) must be divisible by "
                f"num_generations ({self.num_generations}). Valid: {possible_values}"
            )

        set_seed(args.seed, device_specific=True)

        # Generation config
        if self.beam_search:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                num_beams=self.num_generations,
                num_return_sequences=self.num_generations,
                pad_token_id=processing_class.pad_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                do_sample=True,
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=processing_class.pad_token_id,
                eos_token_id=processing_class.eos_token_id,
            )

        self.model_accepts_loss_kwargs = False
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                # 尝试把 ref_model 放到空闲 GPU 上
                ref_device_str = os.environ.get("REF_MODEL_DEVICE", "")
                if ref_device_str:
                    ref_device = torch.device(ref_device_str)
                    self.ref_model = self.ref_model.to(ref_device)
                    self.ref_model.eval()
                    print(f"[GRPORecTrainer] ref_model placed on {ref_device}")
                else:
                    self.ref_model = self.accelerator.prepare_model(
                        self.ref_model, evaluation_mode=True
                    )

        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator)
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

        # Build hash_dict for constrained decoding
        self._build_hash_dict()

        # Test generation config
        self.test_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            num_beams=self.test_beam,
            num_return_sequences=self.test_beam,
            do_sample=False,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )

    def _build_hash_dict(self):
        """Build prefix hash dict from SID index for constrained decoding."""
        import json

        index_path = self.sid_index_path or self.info_file
        if index_path is None:
            raise ValueError("Either sid_index_path or info_file must be provided")

        if index_path.endswith(".json"):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)

            semantic_ids = []
            for item_id, tokens in index_data.items():
                if len(tokens) >= 3:
                    sid_str = "".join(tokens[:3])
                    semantic_ids.append(sid_str)

            info_semantic = [f"### Response:\n{sid}\n" for sid in semantic_ids]
        else:
            with open(index_path, "r") as f:
                info = f.readlines()
            semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]
            info_semantic = [f"### Response:\n{s}" for s in semantic_ids]

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # 动态计算 prefix_index：tokenize "### Response:\n" 得到的 token 数
        response_prefix = "### Response:\n"
        response_prefix_ids = tokenizer(response_prefix).input_ids
        prefix_index = len(response_prefix_ids)
        print(f"[GRPORecTrainer] prefix_index={prefix_index} "
              f"('{response_prefix.strip()}' -> {response_prefix_ids})")

        prefixID = [tokenizer(s).input_ids for s in info_semantic]

        self.hash_dict = {}
        for ID in prefixID:
            ID.append(tokenizer.eos_token_id)
            for i in range(prefix_index, len(ID)):
                if i == prefix_index:
                    hash_number = self._get_hash(ID[:i])
                else:
                    hash_number = self._get_hash(ID[prefix_index:i])
                if hash_number not in self.hash_dict:
                    self.hash_dict[hash_number] = set()
                self.hash_dict[hash_number].add(ID[i])

        for key in self.hash_dict:
            self.hash_dict[key] = list(self.hash_dict[key])

        # 保存 prefix_index 供 ConstrainedLogitsProcessor 使用
        self._prefix_index = prefix_index

        print(f"[GRPORecTrainer] hash_dict built: {len(self.hash_dict)} entries, "
              f"{len(semantic_ids)} SIDs")

    @staticmethod
    def _get_hash(x):
        return "-".join(str(v) for v in x)

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        hash_number = self._get_hash(input_ids)
        if hash_number in self.hash_dict:
            return self.hash_dict[hash_number]
        return []

    def make_prefix_allowed_fn(self, prompt_length):
        """Create a prefix_allowed_tokens_fn compatible with generate() native API.
        
        generate() passes (batch_id, full_input_ids_tensor) where full_input_ids
        includes the prompt. We need to extract only the generated part for hash lookup.
        """
        prefix_index = getattr(self, "_prefix_index", 3)
        hash_dict = self.hash_dict
        eos_id = self.processing_class.eos_token_id

        def _get_hash(x):
            return "-".join(str(v) for v in x)

        def _fn(batch_id, input_ids):
            generated = input_ids[prompt_length:]
            gen_len = len(generated)

            if gen_len == 0:
                key = _get_hash(input_ids[-prefix_index:].tolist())
            else:
                key = _get_hash(generated.tolist())

            if key in hash_dict:
                return hash_dict[key]
            if eos_id is not None:
                return [eos_id]
            return []

        return _fn

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        if train_dataset is None:
            train_dataset = self.train_dataset
        return RepeatRandomSampler(train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        try:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits
        except TypeError:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)

    def _prepare_inputs(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        if self.test_during_training:
            histories = [self.prompt2history.get(x["prompt"], "") for x in inputs]
            targets = [self.history2target.get(h, "") for h in histories]

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Build prefix_allowed_tokens_fn for constrained decoding (native generate API)
        prompt_len_for_constraint = prompt_ids.size(1)
        _prefix_fn = self.make_prefix_allowed_fn(prompt_len_for_constraint)

        # Test evaluation during training
        topk = [3, 5, 10, 20]
        ndcg = [0, 0, 0, 0]
        hr = [0, 0, 0, 0]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            # 覆盖模型自带的 generation_config，防止 top_p/repetition_penalty 干扰约束解码
            if hasattr(unwrapped_model, "generation_config"):
                unwrapped_model.generation_config.top_p = 1.0
                unwrapped_model.generation_config.top_k = 0
                unwrapped_model.generation_config.repetition_penalty = 1.0
                unwrapped_model.generation_config.temperature = self.temperature
                unwrapped_model.generation_config.do_sample = True

            if self.test_during_training:
                dedup_prompt = []
                dedup_mask = []
                dedup_target = []
                for i in range(len(prompt_ids)):
                    if i % self.num_generations == 0:
                        dedup_prompt.append(prompt_ids[i])
                        dedup_mask.append(prompt_mask[i])
                        if self.test_during_training:
                            dedup_target.append(targets[i])

                dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                dedup_prompt_mask = torch.stack(dedup_mask).to(device)

                _test_prefix_fn = self.make_prefix_allowed_fn(dedup_prompt_ids.size(1))
                with torch.no_grad():
                    test_completion_ids = unwrapped_model.generate(
                        dedup_prompt_ids,
                        attention_mask=dedup_prompt_mask,
                        generation_config=self.test_generation_config,
                        prefix_allowed_tokens_fn=_test_prefix_fn,
                    )

                test_completions = self.processing_class.batch_decode(
                    test_completion_ids, skip_special_tokens=True
                )
                test_completions = [c.split("Response:\n")[-1] for c in test_completions]
                test_comp_lis = [
                    test_completions[i : i + self.test_beam]
                    for i in range(0, len(test_completions), self.test_beam)
                ]

                for i, comp_lis in enumerate(test_comp_lis):
                    target = dedup_target[i]
                    for j, comp in enumerate(comp_lis):
                        if comp.strip('\n"') == target.strip('\n"'):
                            for index, k in enumerate(topk):
                                if j < k:
                                    hr[index] += 1
                                    ndcg[index] += 1 / math.log2(j + 2)
                            break

                if dedup_target:
                    hr = [e / len(dedup_target) for e in hr]
                    ndcg = [e / len(dedup_target) for e in ndcg]

            # Generate completions for training
            if self.beam_search:
                dedup_prompt_b = []
                dedup_mask_b = []
                for i in range(len(prompt_ids)):
                    if i % self.num_generations == 0:
                        dedup_prompt_b.append(prompt_ids[i])
                        dedup_mask_b.append(prompt_mask[i])
                dedup_prompt_ids_b = torch.stack(dedup_prompt_b).to(device)
                dedup_prompt_mask_b = torch.stack(dedup_mask_b).to(device)
                prompt_completion_ids = unwrapped_model.generate(
                    dedup_prompt_ids_b,
                    attention_mask=dedup_prompt_mask_b,
                    generation_config=self.generation_config,
                    prefix_allowed_tokens_fn=_prefix_fn,
                )
            else:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                    prefix_allowed_tokens_fn=_prefix_fn,
                )

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask after first EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_device = next(self.ref_model.parameters()).device
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids.to(ref_device),
                    attention_mask.to(ref_device),
                    logits_to_keep,
                ).to(device)
            elif is_peft_model(self.model):
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
            else:
                ref_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        completions = completions_text

        # Diversity metrics
        div_lis = [
            len(set(completions_text[i : i + self.num_generations])) / self.num_generations
            for i in range(0, len(completions_text), self.num_generations)
        ]
        cate_diversity = sum(div_lis) / len(div_lis) if div_lis else 0.0

        # Compute rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Group normalization
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        sliced_rewards = rewards[process_slice]

        # Log metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["categorical_diversity"].append(cate_diversity)

        if self.test_during_training:
            for i in range(len(topk)):
                self._metrics[f"NDCG@{topk[i]}"].append(ndcg[i])
                self._metrics[f"HR@{topk[i]}"].append(hr[i])

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "sliced_rewards": sliced_rewards,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPORecTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        ref_per_token_logps = inputs["ref_per_token_logps"]

        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        if self.dapo:
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        else:
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs, start_time=None):
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
