"""
Phase 6: GRPO 强化学习训练

基于 SFT 后的模型，使用 GRPO (Group Relative Policy Optimization) 进行
推荐导向的强化学习训练。

参照 MiniOneRec rl.py 结构，适配本项目的 SID 数据格式。

训练流程：
1. 加载 SFT 后的模型（phase5 产出）
2. 构建 RL 训练数据集（SidDataset + RLTitle2SidDataset + RLSeqTitle2SidDataset）
3. 定义奖励函数（rule / ranking / hierarchy）
4. 使用 GRPORecTrainer 进行 GRPO 训练
5. 保存 RL 后的模型
"""

import os
import sys
import json
import random
import math
import csv
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from fire import Fire

from grpo_trainer import GRPORecTrainer

os.environ["WANDB_MODE"] = "disabled"


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# RL 数据集类（返回 prompt/completion 对）
# ============================================================

class RLSidDataset(Dataset):
    """
    RL 主任务：history SID → target SID。
    返回 prompt + completion 对（不做 tokenize，由 Trainer 处理）。
    """

    def __init__(self, csv_file, sample=-1, seed=42):
        self.data = pd.read_csv(csv_file)
        if sample > 0 and len(self.data) > sample:
            self.data = self.data.sample(sample, random_state=seed)

        self.prompt2history = {}
        self.history2target = {}
        self.inputs = []
        self._preprocess_all()

    def _preprocess_all(self):
        for idx in tqdm(range(len(self.data)), desc="RLSidDataset", leave=False):
            item = self._preprocess_one(idx)
            if item is not None:
                self.inputs.append(item)

    def _preprocess_one(self, idx):
        row = self.data.iloc[idx]
        history_sids = eval(row["history_item_sid"])
        history_str = "::".join(history_sids)

        history_display = ", ".join(history_sids)
        input_text = (
            f"The user has interacted with items {history_display} "
            f"in chronological order. Can you predict the next possible "
            f"item that the user may expect?"
        )

        prompt = f"""### User Input: 
{input_text}

### Response:\n"""

        target_sid = str(row["item_sid"]) + "\n"

        self.prompt2history[prompt] = history_str
        self.history2target[history_str] = target_sid

        return {"prompt": prompt, "completion": target_sid}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class RLTitle2SidDataset(Dataset):
    """
    RL 辅助任务：title → SID。
    """

    def __init__(self, item_file, index_file, sample=-1, seed=42):
        with open(item_file, "r", encoding="utf-8") as f:
            self.item_feat = json.load(f)
        with open(index_file, "r") as f:
            self.indices = json.load(f)

        self.prompt2history = {}
        self.history2target = {}
        self.inputs = []

        data = []
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                title = self.item_feat[item_id]["title"]
                combined_sid = sids[0] + sids[1] + sids[2]
                data.append({"title": title, "sid": combined_sid})

        if sample > 0 and len(data) > sample:
            random.seed(seed)
            data = random.sample(data, sample)

        for item in tqdm(data, desc="RLTitle2SidDataset", leave=False):
            prompt = f"""### User Input: 
Which item has the title: {item['title']}?

### Response:\n"""
            target = item["sid"] + "\n"
            self.prompt2history[prompt] = item["title"]
            self.history2target[item["title"]] = target
            self.inputs.append({"prompt": prompt, "completion": target})

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class RLSeqTitle2SidDataset(Dataset):
    """
    RL 辅助任务：title 序列 → target SID。
    """

    def __init__(self, csv_file, sample=10000, seed=42):
        self.data = pd.read_csv(csv_file)
        if sample > 0 and len(self.data) > sample:
            self.data = self.data.sample(sample, random_state=seed)

        self.prompt2history = {}
        self.history2target = {}
        self.inputs = []
        self._preprocess_all()

    def _preprocess_all(self):
        for idx in tqdm(range(len(self.data)), desc="RLSeqTitle2SidDataset", leave=False):
            item = self._preprocess_one(idx)
            if item is not None:
                self.inputs.append(item)

    def _preprocess_one(self, idx):
        row = self.data.iloc[idx]
        try:
            history_titles = eval(row["history_item_title"])
        except Exception:
            return None

        inter_titles = ", ".join([f'"{t}"' for t in history_titles])
        history_str = "::".join(history_titles)
        target_sid = str(row["item_sid"]) + "\n"

        prompt = f"""### User Input: 
Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?

### Response:\n"""

        self.prompt2history[prompt] = history_str
        self.history2target[history_str] = target_sid

        return {"prompt": prompt, "completion": target_sid}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


# ============================================================
# CSV Logger (离线日志)
# ============================================================

class CsvLoggerCallback:
    """简单的 CSV 日志记录器。"""

    def __init__(self, log_file):
        self.log_file = log_file
        self._header_written = os.path.exists(log_file)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update(logs)
        write_header = not self._header_written
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ============================================================
# 奖励函数
# ============================================================

def make_rule_reward(prompt2history, history2target):
    """完全匹配奖励：生成 == ground truth → 1.0，否则 0.0。"""

    def rule_reward(prompts, completions):
        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            history = prompt2history.get(prompt, "")
            target = history2target.get(history, "")
            if completion.strip('\n" ') == target.strip('\n" '):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    rule_reward.__name__ = "rule_reward"
    return rule_reward


def make_ranking_reward(prompt2history, history2target, num_generations):
    """
    排序感知奖励：在组内，命中的 completion 得正分，
    未命中但排名靠前的得负分（NDCG 惩罚）。
    """
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    ndcg_rewards = [-e / sum(ndcg_rewards) for e in ndcg_rewards]

    def ranking_reward(prompts, completions):
        rewards = []
        flag = False
        lis = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            history = prompt2history.get(prompt, "")
            target = history2target.get(history, "")
            if completion.strip('\n"') == target.strip('\n"'):
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i % num_generations])

            if (i + 1) % num_generations == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * num_generations)
                flag = False
                lis = []
        return rewards

    ranking_reward.__name__ = "ranking_reward"
    return ranking_reward


def make_hierarchy_reward(prompt2history, history2target):
    """
    层级匹配奖励（HEPO 风格）：
    - 第一层 SID token 匹配：0.2
    - 第二层匹配：0.5
    - 第三层匹配（完全匹配）：1.0
    """

    def _parse_sid_tokens(sid_str):
        """从 '<a_X><b_Y><c_Z>' 中提取各层 token。"""
        import re
        tokens = re.findall(r"<[a-c]_\d+>", sid_str.strip('\n" '))
        return tokens

    def hierarchy_reward(prompts, completions):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            history = prompt2history.get(prompt, "")
            target = history2target.get(history, "")

            target_tokens = _parse_sid_tokens(target)
            pred_tokens = _parse_sid_tokens(completion)

            if not target_tokens or not pred_tokens:
                rewards.append(0.0)
                continue

            score = 0.0
            for level, (gt, pd_tok) in enumerate(zip(target_tokens, pred_tokens)):
                if gt == pd_tok:
                    if level == 0:
                        score = 0.2
                    elif level == 1:
                        score = 0.5
                    elif level == 2:
                        score = 1.0
                else:
                    break
            rewards.append(score)
        return rewards

    hierarchy_reward.__name__ = "hierarchy_reward"
    return hierarchy_reward


# ============================================================
# 主训练函数
# ============================================================

def train(
    # model
    model_path: str = "",
    base_model: str = "",
    seed: int = 42,
    # data
    train_file: str = "",
    eval_file: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    # training
    output_dir: str = "",
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    temperature: float = 1.0,
    num_generations: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    max_completion_length: int = 64,
    # sampling
    beam_search: bool = False,
    test_during_training: bool = True,
    test_beam: int = 20,
    # reward
    reward_type: str = "rule",
    # loss
    dapo: bool = False,
    # misc
    eval_step: float = 0.199,
    sample: int = -1,
    seq_title_sample: int = 10000,
    # config file
    config: str = None,
):
    """
    GRPO 强化学习训练主函数。

    参数说明：
    - model_path: SFT 后模型路径（phase5 产出的 final_checkpoint）
    - base_model: 基座模型路径（用于 tokenizer 和 ref model，如果为空则用 model_path）
    - train_file: 训练 CSV 文件
    - eval_file: 验证 CSV 文件
    - sid_index_path: SID 索引文件（.index.json）
    - item_meta_path: 商品元数据文件（.item.json）
    - reward_type: 奖励类型 (rule / ranking / hierarchy / rule+ranking)
    """
    # 从 config 文件加载参数
    if config and os.path.exists(config):
        print(f"[INFO] 从配置文件加载参数: {config}")
        with open(config, "r") as f:
            cfg = json.load(f)
        model_path = cfg.get("model_path", model_path)
        base_model = cfg.get("base_model", base_model)
        seed = cfg.get("seed", seed)
        train_file = cfg.get("train_file", train_file)
        eval_file = cfg.get("eval_file", eval_file)
        sid_index_path = cfg.get("sid_index_path", sid_index_path)
        item_meta_path = cfg.get("item_meta_path", item_meta_path)
        output_dir = cfg.get("output_dir", output_dir)
        train_batch_size = cfg.get("train_batch_size", train_batch_size)
        eval_batch_size = cfg.get("eval_batch_size", eval_batch_size)
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", gradient_accumulation_steps)
        temperature = cfg.get("temperature", temperature)
        num_generations = cfg.get("num_generations", num_generations)
        num_train_epochs = cfg.get("num_train_epochs", num_train_epochs)
        learning_rate = cfg.get("learning_rate", learning_rate)
        beta = cfg.get("beta", beta)
        max_completion_length = cfg.get("max_completion_length", max_completion_length)
        beam_search = cfg.get("beam_search", beam_search)
        test_during_training = cfg.get("test_during_training", test_during_training)
        test_beam = cfg.get("test_beam", test_beam)
        reward_type = cfg.get("reward_type", reward_type)
        dapo = cfg.get("dapo", dapo)
        eval_step = cfg.get("eval_step", eval_step)
        sample = cfg.get("sample", sample)
        seq_title_sample = cfg.get("seq_title_sample", seq_title_sample)

    if not base_model:
        base_model = model_path

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)

    print("=" * 60)
    print("Phase 6: GRPO 强化学习训练")
    print("=" * 60)
    print(f"SFT Model:    {model_path}")
    print(f"Base Model:   {base_model}")
    print(f"Train:        {train_file}")
    print(f"Eval:         {eval_file}")
    print(f"Index:        {sid_index_path}")
    print(f"Item Meta:    {item_meta_path}")
    print(f"Output:       {output_dir}")
    print(f"Reward Type:  {reward_type}")
    print(f"Batch Size:   {train_batch_size} (grad_accum={gradient_accumulation_steps})")
    print(f"LR:           {learning_rate}")
    print(f"Beta (KL):    {beta}")
    print(f"Generations:  {num_generations}")
    print(f"Temperature:  {temperature}")
    print(f"Beam Search:  {beam_search}")
    print(f"Test Beam:    {test_beam}")
    print(f"DAPO:         {dapo}")
    print("=" * 60)

    # ==========================================
    # 1. 构建数据集
    # ==========================================
    print("\n[1] 构建 RL 训练数据集...")

    train_datasets = []

    print("    ① RLSidDataset (主任务: history SID → target SID)")
    ds1 = RLSidDataset(train_file, sample=sample, seed=seed)
    train_datasets.append(ds1)
    print(f"       → {len(ds1)} 样本")

    print("    ② RLTitle2SidDataset (辅助: title → SID)")
    ds2 = RLTitle2SidDataset(item_meta_path, sid_index_path, sample=sample, seed=seed)
    train_datasets.append(ds2)
    print(f"       → {len(ds2)} 样本")

    print("    ③ RLSeqTitle2SidDataset (辅助: title 序列 → SID)")
    ds3 = RLSeqTitle2SidDataset(train_file, sample=seq_title_sample, seed=seed)
    train_datasets.append(ds3)
    print(f"       → {len(ds3)} 样本")

    train_data = ConcatDataset(train_datasets)
    eval_data = RLSidDataset(eval_file, sample=sample, seed=seed)

    # 合并 prompt2history / history2target
    prompt2history = {}
    history2target = {}
    for ds in train_datasets:
        prompt2history.update(ds.prompt2history)
        history2target.update(ds.history2target)
    prompt2history.update(eval_data.prompt2history)
    history2target.update(eval_data.history2target)

    # 转为 HuggingFace Dataset
    train_dataset = HFDataset.from_dict(
        {k: [elm[k] for elm in train_data] for k in train_data[0].keys()}
    )
    train_dataset = train_dataset.shuffle(seed=seed)

    eval_dataset = HFDataset.from_dict(
        {k: [elm[k] for elm in eval_data] for k in eval_data[0].keys()}
    )
    eval_dataset = eval_dataset.shuffle(seed=seed)

    print(f"\n    [OK] 总训练样本: {len(train_dataset)}")
    print(f"    [OK] 验证样本:   {len(eval_dataset)}")

    # ==========================================
    # 2. 构建奖励函数
    # ==========================================
    print(f"\n[2] 构建奖励函数: {reward_type}")

    if reward_type == "rule":
        reward_fun = make_rule_reward(prompt2history, history2target)
    elif reward_type == "ranking":
        reward_fun = [
            make_rule_reward(prompt2history, history2target),
            make_ranking_reward(prompt2history, history2target, num_generations),
        ]
    elif reward_type == "hierarchy":
        reward_fun = make_hierarchy_reward(prompt2history, history2target)
    elif reward_type == "rule+ranking":
        reward_fun = [
            make_rule_reward(prompt2history, history2target),
            make_ranking_reward(prompt2history, history2target, num_generations),
        ]
    elif reward_type == "rule+hierarchy":
        reward_fun = [
            make_rule_reward(prompt2history, history2target),
            make_hierarchy_reward(prompt2history, history2target),
        ]
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    print("    [OK]")

    # ==========================================
    # 3. 配置 GRPO Trainer
    # ==========================================
    print(f"\n[3] 配置 GRPORecTrainer...")

    os.makedirs(output_dir, exist_ok=True)

    # 自动检测 bf16 支持（3090 不支持 bf16，使用 fp16）
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    
    grpo_kwargs = dict(
        output_dir=output_dir,
        save_steps=0.25,
        save_total_limit=2,
        eval_strategy="no",
        load_best_model_at_end=False,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        temperature=temperature,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_step,
        logging_steps=1,
        learning_rate=learning_rate,
        beta=beta,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        report_to="none",
        run_name=os.path.basename(output_dir),
        gradient_checkpointing=True,
    )

    # 过滤掉当前 trl 版本 GRPOConfig 不支持的参数
    import inspect
    valid_fields = set()
    for cls in inspect.getmro(GRPOConfig):
        if hasattr(cls, "__dataclass_fields__"):
            valid_fields.update(cls.__dataclass_fields__.keys())
        if hasattr(cls, "__init__"):
            sig = inspect.signature(cls.__init__)
            valid_fields.update(sig.parameters.keys())
    if valid_fields:
        filtered = {k: v for k, v in grpo_kwargs.items() if k in valid_fields}
        skipped = set(grpo_kwargs.keys()) - set(filtered.keys())
        if skipped:
            print(f"    [WARN] GRPOConfig 不支持以下参数，已跳过: {skipped}")
        grpo_kwargs = filtered

    training_args = GRPOConfig(**grpo_kwargs)

    trainer = GRPORecTrainer(
        model=model_path,
        base_model=base_model,
        reward_funcs=reward_fun,
        args=training_args,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        dapo=dapo,
        sid_index_path=sid_index_path,
        prompt2history=prompt2history,
        history2target=history2target,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("    [OK]")

    # ==========================================
    # 4. 开始训练
    # ==========================================
    print(f"\n[4] 开始 GRPO 训练...")
    print("=" * 60)

    trainer.train()

    # ==========================================
    # 5. 保存模型
    # ==========================================
    print(f"\n[5] 保存模型...")
    trainer.save_model(output_dir)

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(final_dir)

    print(f"    [OK] 模型已保存至: {final_dir}")
    print("\n" + "=" * 60)
    print("GRPO 强化学习训练完成！")
    print("=" * 60)
    print(f"\n后续步骤：使用 eval/run_eval.py 评估 RL 模型：")
    print(f"  python eval/run_eval.py \\")
    print(f"    --model_dir {final_dir} \\")
    print(f"    --test_file <test.csv> \\")
    print(f"    --index_file {sid_index_path} \\")
    print(f"    --model_name RL_{os.path.basename(output_dir)} \\")
    print(f"    --N <N> --beam 50")


if __name__ == "__main__":
    Fire(train)
