"""诊断脚本：检查 hash_dict 和 ConstrainedLogitsProcessor 的兼容性"""
import json
from transformers import AutoTokenizer

model_path = "train/runs/M_N_5K/final_checkpoint"
index_path = "data/sequences/S1/S1.index.json"

tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Tokenizer: {type(tokenizer).__name__}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

with open(index_path, "r", encoding="utf-8") as f:
    index_data = json.load(f)

items = list(index_data.items())[:3]
print(f"\n=== Sample SIDs ===")
for item_id, tokens in items:
    sid_str = "".join(tokens[:3])
    print(f"  {item_id}: {tokens[:3]} -> '{sid_str}'")

print(f"\n=== Tokenization of '### Response:\\n<sid>\\n' ===")
for item_id, tokens in items:
    sid_str = "".join(tokens[:3])
    info_str = f"### Response:\n{sid_str}\n"
    ids = tokenizer(info_str).input_ids
    print(f"\n  Input: {repr(info_str)}")
    print(f"  Token IDs ({len(ids)} tokens): {ids}")
    for i, tid in enumerate(ids):
        print(f"    [{i}] {tid} -> {repr(tokenizer.decode([tid]))}")

print(f"\n=== prefix_index check ===")
resp_str = "### Response:\n"
resp_ids = tokenizer(resp_str).input_ids
print(f"  '### Response:\\n' -> {len(resp_ids)} tokens: {resp_ids}")
for i, tid in enumerate(resp_ids):
    print(f"    [{i}] {tid} -> {repr(tokenizer.decode([tid]))}")

print(f"\n=== Hash dict first entry check ===")
item_id, tokens = items[0]
sid_str = "".join(tokens[:3])
info_str = f"### Response:\n{sid_str}\n"
ids = tokenizer(info_str).input_ids
ids.append(tokenizer.eos_token_id)

prefix_index = 3
print(f"  prefix_index = {prefix_index}")
print(f"  Full IDs: {ids}")
print(f"  First hash (ID[:prefix_index]): {ids[:prefix_index]} -> hash='{'-'.join(str(v) for v in ids[:prefix_index])}'")
for i in range(prefix_index, len(ids)):
    if i == prefix_index:
        h = ids[:i]
    else:
        h = ids[prefix_index:i]
    hash_str = '-'.join(str(v) for v in h)
    print(f"  Step {i-prefix_index}: hash='{hash_str}' -> next_token={ids[i]} ({repr(tokenizer.decode([ids[i]]))})")

print(f"\n=== Prompt end check ===")
prompt = """### User Input: 
The user has interacted with items <a_188><b_141><c_111> in chronological order. Can you predict the next possible item that the user may expect?

### Response:\n"""
prompt_ids = tokenizer(prompt).input_ids
print(f"  Prompt ends with (last 5 tokens):")
for i in range(max(0, len(prompt_ids)-5), len(prompt_ids)):
    print(f"    [{i}] {prompt_ids[i]} -> {repr(tokenizer.decode([prompt_ids[i]]))}")
print(f"  Last 3 token IDs: {prompt_ids[-3:]}")
print(f"  Hash of last 3: '{'-'.join(str(v) for v in prompt_ids[-3:])}'")

first_hash = '-'.join(str(v) for v in ids[:prefix_index])
prompt_hash = '-'.join(str(v) for v in prompt_ids[-3:])
print(f"\n=== MATCH CHECK ===")
print(f"  hash_dict first key: '{first_hash}'")
print(f"  prompt last-3 hash:  '{prompt_hash}'")
print(f"  MATCH: {first_hash == prompt_hash}")
