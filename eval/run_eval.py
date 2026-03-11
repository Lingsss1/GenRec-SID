"""
eval/run_eval.py

Phase 6：约束解码 + HR@10 / NDCG@10 评估。
参照 evaluate.py + ConstrainedLogitsProcessor 实现。

用法（在 server_train 根目录下运行）：

    python eval/run_eval.py \
        --model_dir  train/runs/M_N_50K/final_checkpoint \
        --test_file  data/sequences/S2.5/test.csv \
        --index_file data/sequences/S2.5/S2.5.index.json \
        --model_name M_N_50K \
        --N          50000 \
        --beam       50 \
        --output_json analysis/results_summary.json
"""

import argparse
import ast
import json
import math
import os
import re
import sys
import warnings
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
)
from transformers.generation import LogitsProcessor
from transformers.utils import add_start_docstrings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ============================================================
# ConstrainedLogitsProcessor（对齐参考代码）
# ============================================================

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        eos_token_id: int = None,
        prefix_index: int = 3,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0
        self.eos_token_id = eos_token_id
        self.prefix_index = prefix_index

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))

        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key = sent[-self.count:]
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
# 数据准备
# ============================================================

def build_prompt(history_sids: List[str]) -> str:
    """与训练时 SidSFTDataset._preprocess_one 完全对齐的 prompt 构建。"""
    instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
    history_str = ", ".join(history_sids)
    prompt_input = f"The user has interacted with items {history_str} in chronological order. Can you predict the next possible item that the user may expect?"

    prompt = f"""### User Input: 
{prompt_input}

### Response:
"""
    return instruction + prompt


def load_test_data(test_file: str, max_samples: int = -1, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(test_file)
    if max_samples > 0 and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


# ============================================================
# 指标计算
# ============================================================

def compute_hr_ndcg(pred_list: List[str], ground_truth: str, k: int = 10) -> Tuple[float, float]:
    pred_k = pred_list[:k]
    if ground_truth in pred_k:
        rank = pred_k.index(ground_truth) + 1
        ndcg = 1.0 / math.log2(rank + 1)
        return 1.0, ndcg
    return 0.0, 0.0


# ============================================================
# hash_dict 构建（对齐参考代码）
# ============================================================

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)


def build_hash_dict(index_path: str, tokenizer, prefix_index: int = 3):
    """
    从 index.json 构建 hash_dict，对齐参考代码的方式：
    - 把每个 SID 拼成 "### Response:\\n<a_X><b_Y><c_Z>\\n"
    - 整体 tokenize
    - 从 prefix_index 开始构建前缀 -> 合法下一个 token 的映射
    """
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    semantic_ids = []
    item_ids = []
    for item_id, tokens in index.items():
        if len(tokens) >= 3:
            sid_str = "".join(tokens[:3])
            semantic_ids.append(sid_str)
            item_ids.append(item_id)

    info_semantic = [f"### Response:\n{sid}\n" for sid in semantic_ids]
    prefixID = [tokenizer(s).input_ids for s in info_semantic]

    hash_dict = {}
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict:
        hash_dict[key] = list(hash_dict[key])

    sid2item = {}
    for item_id, tokens in index.items():
        if len(tokens) >= 3:
            sid_str = "".join(tokens[:3])
            sid2item[sid_str] = item_id

    return hash_dict, sid2item, prefix_index


# ============================================================
# 推理主逻辑
# ============================================================

def run_evaluation(
    model_dir: str,
    test_file: str,
    index_file: str,
    trie_file: Optional[str],
    model_name: str,
    N: int,
    beam: int,
    topk: int = 10,
    batch_size: int = 4,
    max_samples: int = -1,
    device: str = "auto",
    output_json: str = "analysis/results_summary.json",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")

    print(f"[INFO] 加载模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )
    model.eval()
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    print(f"[INFO] 构建 hash_dict...")
    hash_dict, sid2item, prefix_index = build_hash_dict(index_file, tokenizer)
    print(f"  hash_dict 条目数: {len(hash_dict)}")
    print(f"  SID 总数: {len(sid2item)}")

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    df = load_test_data(test_file, max_samples)
    print(f"[INFO] 测试样本数: {len(df)}")

    # 构建所有编码
    encodings = []
    ground_truths = []
    for _, row in df.iterrows():
        history_sids = ast.literal_eval(row["history_item_sid"])
        prompt = build_prompt(history_sids)
        enc = tokenizer(prompt, truncation=True, max_length=512)
        encodings.append({"input_ids": enc["input_ids"]})

        gt_sid = str(row["item_sid"]).strip()
        gt_item = sid2item.get(gt_sid)
        ground_truths.append(gt_item)

    SID_PATTERN = re.compile(r"(<a_\d+>)(<b_\d+>)(<c_\d+>)")

    all_hr, all_ndcg = [], []
    num_beams = beam
    num_return = num_beams
    max_new_tokens = 64

    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for block_idx in tqdm(range(BLOCK), desc=f"{model_name} beam={beam}"):
        batch_enc = encodings[block_idx * batch_size: (block_idx + 1) * batch_size]
        batch_gt = ground_truths[block_idx * batch_size: (block_idx + 1) * batch_size]

        maxLen = max(len(e["input_ids"]) for e in batch_enc)
        padding_input_ids = []
        attention_mask = []
        for e in batch_enc:
            L = len(e["input_ids"])
            padding_input_ids.append([tokenizer.pad_token_id] * (maxLen - L) + e["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=0.0,
            num_return_sequences=num_return,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            top_k=None,
            top_p=None,
        )

        clp = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            eos_token_id=model.config.eos_token_id,
            prefix_index=prefix_index,
        )
        logits_processor = LogitsProcessorList([clp])

        with torch.no_grad():
            try:
                generation_output = model.generate(
                    torch.tensor(padding_input_ids).to(model.device),
                    attention_mask=torch.tensor(attention_mask).to(model.device),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    logits_processor=logits_processor,
                )
            except Exception as e:
                print(f"\n[WARN] batch {block_idx} 推理失败: {e}")
                for _ in batch_gt:
                    all_hr.append(0.0)
                    all_ndcg.append(0.0)
                continue

        batched_completions = generation_output.sequences[:, maxLen:]
        output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)

        real_outputs = [output[i * num_return: (i + 1) * num_return] for i in range(len(batch_enc))]

        for i, gt_item in enumerate(batch_gt):
            if gt_item is None:
                all_hr.append(0.0)
                all_ndcg.append(0.0)
                continue

            candidate_items = []
            for text in real_outputs[i]:
                text = text.strip()
                matches = SID_PATTERN.findall(text)
                for m in matches:
                    sid = "".join(m)
                    item = sid2item.get(sid)
                    if item and item not in candidate_items:
                        candidate_items.append(item)

            hr, ndcg = compute_hr_ndcg(candidate_items, gt_item, k=topk)
            all_hr.append(hr)
            all_ndcg.append(ndcg)

    avg_hr = sum(all_hr) / len(all_hr) if all_hr else 0.0
    avg_ndcg = sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0.0
    num_samples = len(all_hr)

    print(f"\n{'='*50}")
    print(f"模型: {model_name}  N={N}  Beam={beam}")
    print(f"HR@{topk}:   {avg_hr:.4f}")
    print(f"NDCG@{topk}: {avg_ndcg:.4f}")
    print(f"样本数: {num_samples}")
    print(f"{'='*50}\n")

    result_entry = {
        "model": model_name,
        "N": N,
        "beam": beam,
        f"HR@{topk}": round(avg_hr, 6),
        f"NDCG@{topk}": round(avg_ndcg, 6),
        "num_samples": num_samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    existing = []
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    updated = False
    for idx, entry in enumerate(existing):
        if entry.get("model") == model_name and entry.get("beam") == beam:
            existing[idx] = result_entry
            updated = True
            break
    if not updated:
        existing.append(result_entry)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[OK] 结果已写入: {output_json}")
    return result_entry


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6 约束解码评估")
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--test_file",    required=True)
    parser.add_argument("--index_file",   required=True)
    parser.add_argument("--trie_file",    default=None, help="已弃用，保留兼容性")
    parser.add_argument("--model_name",   required=True)
    parser.add_argument("--N",            required=True,  type=int)
    parser.add_argument("--beam",         required=True,  type=int)
    parser.add_argument("--topk",         default=10,     type=int)
    parser.add_argument("--batch_size",   default=4,      type=int)
    parser.add_argument("--max_samples",  default=-1,     type=int)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--output_json",  default="analysis/results_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        test_file=args.test_file,
        index_file=args.index_file,
        trie_file=args.trie_file,
        model_name=args.model_name,
        N=args.N,
        beam=args.beam,
        topk=args.topk,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        output_json=args.output_json,
    )
