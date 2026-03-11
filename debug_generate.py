"""诊断脚本：测试单次 generate 调用的耗时"""
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
from grpo_trainer import ConstrainedLogitsProcessor

model_path = "train/runs/M_N_5K/final_checkpoint"
index_path = "data/sequences/S1/S1.index.json"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Build hash_dict
print("\nBuilding hash_dict...")
with open(index_path, "r", encoding="utf-8") as f:
    index_data = json.load(f)

semantic_ids = []
for item_id, tokens in index_data.items():
    if len(tokens) >= 3:
        sid_str = "".join(tokens[:3])
        semantic_ids.append(sid_str)

info_semantic = [f"### Response:\n{sid}\n" for sid in semantic_ids]
prefixID = [tokenizer(s).input_ids for s in info_semantic]
prefix_index = len(tokenizer("### Response:\n").input_ids)
print(f"prefix_index = {prefix_index}")

hash_dict = {}
for ID in prefixID:
    ID.append(tokenizer.eos_token_id)
    for i in range(prefix_index, len(ID)):
        if i == prefix_index:
            hash_number = "-".join(str(v) for v in ID[:i])
        else:
            hash_number = "-".join(str(v) for v in ID[prefix_index:i])
        if hash_number not in hash_dict:
            hash_dict[hash_number] = set()
        hash_dict[hash_number].add(ID[i])

for key in hash_dict:
    hash_dict[key] = list(hash_dict[key])
print(f"hash_dict: {len(hash_dict)} entries")

def prefix_allowed_tokens_fn(batch_id, input_ids):
    hash_number = "-".join(str(v) for v in input_ids)
    if hash_number in hash_dict:
        return hash_dict[hash_number]
    return []

# Test prompt
prompt = """### User Input: 
The user has interacted with items <a_188><b_141><c_111>, <a_118><b_76><c_200> in chronological order. Can you predict the next possible item that the user may expect?

### Response:\n"""

print(f"\nPrompt ({len(prompt)} chars):")
print(prompt[:100] + "...")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Input tokens: {inputs['input_ids'].shape}")

# Override model generation_config
model.generation_config.top_p = 1.0
model.generation_config.top_k = 0
model.generation_config.repetition_penalty = 1.0
model.generation_config.temperature = 1.0
model.generation_config.do_sample = True

gen_config = GenerationConfig(
    max_new_tokens=48,
    do_sample=True,
    temperature=1.0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Test WITHOUT constrained decoding first
print("\n=== Test 1: Generate WITHOUT constrained decoding ===")
t0 = time.time()
with torch.no_grad():
    out1 = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            max_new_tokens=10,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
    )
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
decoded = tokenizer.decode(out1[0], skip_special_tokens=True)
print(f"Output: ...{decoded[-100:]}")

# Test WITH constrained decoding
print("\n=== Test 2: Generate WITH constrained decoding (max_new_tokens=10) ===")
ccc = ConstrainedLogitsProcessor(
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    num_beams=1,
    base_model=model_path,
    eos_token_id=tokenizer.eos_token_id,
    prefix_index=prefix_index,
)
lp = LogitsProcessorList([ccc])

t0 = time.time()
with torch.no_grad():
    out2 = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            max_new_tokens=10,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        logits_processor=lp,
    )
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
decoded = tokenizer.decode(out2[0], skip_special_tokens=True)
print(f"Output: ...{decoded[-100:]}")
comp = tokenizer.decode(out2[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"Completion: '{comp}'")

# Test with batch_size=4
print("\n=== Test 3: Generate WITH constrained decoding, batch=4 ===")
batch_inputs = tokenizer([prompt]*4, return_tensors="pt", padding=True).to(model.device)
ccc2 = ConstrainedLogitsProcessor(
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    num_beams=1,
    base_model=model_path,
    eos_token_id=tokenizer.eos_token_id,
    prefix_index=prefix_index,
)
lp2 = LogitsProcessorList([ccc2])

t0 = time.time()
with torch.no_grad():
    out3 = model.generate(
        **batch_inputs,
        generation_config=GenerationConfig(
            max_new_tokens=10,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        logits_processor=lp2,
    )
t1 = time.time()
print(f"Time: {t1-t0:.2f}s")
for i in range(4):
    comp = tokenizer.decode(out3[i][batch_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"  Completion {i}: '{comp}'")

print("\n=== Done ===")
