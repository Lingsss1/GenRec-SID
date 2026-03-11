"""
eval/run_all_eval.py

批量运行所有可用模型 × 所有 Beam 档位的评估。
同一模型只加载一次，不同 beam 复用模型和 hash_dict。

用法（在 server_train 根目录下运行）：
    python eval/run_all_eval.py
    python eval/run_all_eval.py --only M_N_50K M_N_100K
    python eval/run_all_eval.py --beams 50 100
    python eval/run_all_eval.py --max_samples 500
"""

import argparse
import ast
import json
import math
import os
import re
import sys
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.run_eval import (
    ConstrainedLogitsProcessor,
    build_hash_dict,
    build_prompt,
    compute_hr_ndcg,
    get_hash,
    load_test_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="批量评估所有可用模型")
    parser.add_argument("--config",      default="eval/eval_config.json")
    parser.add_argument("--only",        nargs="+", default=None)
    parser.add_argument("--beams",       nargs="+", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--device",      default="auto")
    return parser.parse_args()


def evaluate_single_beam(
    model, tokenizer, hash_dict, sid2item, prefix_index,
    encodings, ground_truths,
    beam, topk, batch_size, model_name, N, output_json,
):
    """对已加载的模型，用指定 beam 评估。"""

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(get_hash(input_ids), [])

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

        with torch.no_grad():
            try:
                generation_output = model.generate(
                    torch.tensor(padding_input_ids).to(model.device),
                    attention_mask=torch.tensor(attention_mask).to(model.device),
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    logits_processor=LogitsProcessorList([clp]),
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
                for m in SID_PATTERN.findall(text.strip()):
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

    return result_entry


def main():
    args = parse_args()

    config_path = os.path.join(ROOT, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    beams = args.beams if args.beams else config["beams"]
    topk = config["topk"]
    batch_size = args.batch_size if args.batch_size else config["batch_size"]
    output_json = os.path.join(ROOT, config["output_json"])
    max_samples = args.max_samples if args.max_samples != -1 else config.get("max_samples", -1)

    models = config["models"]
    if args.only:
        models = [m for m in models if m["name"] in args.only]

    available = [m for m in models if m.get("available", False)]
    skipped = [m for m in models if not m.get("available", False)]

    print(f"=== 批量评估 ===")
    print(f"可用模型: {[m['name'] for m in available]}")
    print(f"跳过模型: {[m['name'] for m in skipped]} (available=false)")
    print(f"Beam 档位: {beams}")
    print(f"每模型样本数: {max_samples}")
    print(f"评估 HR@{topk} / NDCG@{topk}")
    print(f"结果写入: {output_json}")
    print()

    total = len(available) * len(beams)
    done = 0

    for model_cfg in available:
        model_dir = os.path.join(ROOT, model_cfg["model_dir"])
        if not os.path.exists(model_dir):
            print(f"[SKIP] {model_cfg['name']}: model_dir 不存在 ({model_dir})")
            done += len(beams)
            continue

        print(f"=== 加载模型: {model_cfg['name']} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
        )
        model.eval()
        model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id

        index_file = os.path.join(ROOT, model_cfg["index_file"])
        print(f"  构建 hash_dict: {index_file}")
        hash_dict, sid2item, prefix_index = build_hash_dict(index_file, tokenizer)
        print(f"  hash_dict 条目数: {len(hash_dict)}, SID 总数: {len(sid2item)}")

        test_file = os.path.join(ROOT, model_cfg["test_file"])
        df = load_test_data(test_file, max_samples)
        print(f"  测试样本数: {len(df)}")

        encodings = []
        ground_truths = []
        for _, row in df.iterrows():
            history_sids = ast.literal_eval(row["history_item_sid"])
            prompt = build_prompt(history_sids)
            enc = tokenizer(prompt, truncation=True, max_length=512)
            encodings.append({"input_ids": enc["input_ids"]})
            gt_sid = str(row["item_sid"]).strip()
            ground_truths.append(sid2item.get(gt_sid))

        for beam in beams:
            done += 1
            print(f"\n[{done}/{total}] {model_cfg['name']} beam={beam}")
            evaluate_single_beam(
                model=model,
                tokenizer=tokenizer,
                hash_dict=hash_dict,
                sid2item=sid2item,
                prefix_index=prefix_index,
                encodings=encodings,
                ground_truths=ground_truths,
                beam=beam,
                topk=topk,
                batch_size=batch_size,
                model_name=model_cfg["name"],
                N=model_cfg["N"],
                output_json=output_json,
            )

        del model
        torch.cuda.empty_cache()

    print("\n=== 全部评估完成 ===")
    print(f"结果文件: {output_json}")


if __name__ == "__main__":
    main()
