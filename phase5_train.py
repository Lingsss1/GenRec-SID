"""
Phase 5: 多任务 SFT 训练

参照 MiniOneRec sft.py 结构，训练 Qwen2.5-0.5B-Instruct。

三个训练任务（ConcatDataset）：
1. SidSFTDataset: history SID → target SID (主任务)
2. SidItemFeatDataset: sid↔title 双向任务 (辅助)
3. FusionSeqRecDataset: history SID → target title (辅助)

训练配置（对齐实验蓝本）：
- batch=32, grad_accum=4 (等效 128)
- lr=3e-4, warmup=500, epochs=3
- bf16 + gradient_checkpointing
- early stopping patience=3
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset, ConcatDataset
from datasets import Dataset as HFDataset
from transformers import TrainerCallback


# ============================================================
# WandB 配置（训练结束后可手动执行 wandb sync 上传）
# ============================================================
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")


class CsvLoggerCallback(TrainerCallback):
    """
    将每步训练日志写入 CSV 文件，训练结束后再统一上传 wandb。
    完全本地，不依赖网络连接。
    """
    def __init__(self, log_file: str, run_name: str = ""):
        self.log_file = log_file
        self.run_name = run_name
        self._header_written = os.path.exists(log_file)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update(logs)
        import csv
        write_header = not self._header_written
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def upload_to_wandb(log_file: str, run_name: str, project: str = "cellphones_phase5"):
    """训练完成后调用，将 CSV 日志上传到 wandb。"""
    try:
        import wandb
        os.environ.setdefault("WANDB_SILENT", "true")
        wandb.login(key=WANDB_API_KEY)
        run = wandb.init(project=project, name=run_name, resume="allow")
        # 读 CSV 并上传每一条记录
        rows = pd.read_csv(log_file)
        for _, row in rows.iterrows():
            step = int(row.get("step", 0))
            metrics = {k: v for k, v in row.items() if k not in ("step", "epoch") and pd.notna(v)}
            if metrics:
                wandb.log(metrics, step=step)
        run.finish()
        print(f"[OK] wandb 上传完成: {project}/{run_name}")
    except Exception as e:
        print(f"[WARN] wandb 上传失败，日志已保存在 {log_file}: {e}")


# ============================================================
# 通用工具
# ============================================================

def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TokenExtender:
    """
    从 {S}.index.json 提取所有 unique SID tokens，
    用于扩展 tokenizer 词表。
    """
    def __init__(self, index_file_path: str):
        self.index_file_path = index_file_path
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(self.index_file_path, 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        if self.indices is None:
            self._load_data()

        self.new_tokens = set()
        for token_list in self.indices.values():
            for token in token_list:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens


class Tokenizer:
    """Tokenizer wrapper，处理 bos/eos"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t = self.tokenizer.encode(s)

        # 去除已有的 bos/eos
        while t and t[0] == self.bos_id:
            t = t[1:]
        while t and t[-1] == self.eos_id:
            t = t[:-1]

        # 按需添加
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


# ============================================================
# Dataset 类 (对齐 MiniOneRec data.py)
# ============================================================

class SidSFTDataset(Dataset):
    """
    主任务：history SID → target SID 预测。

    CSV 列：user_id, history_item_sid, item_sid, history_item_title, item_title
    history_item_sid / history_item_title 是 Python list 的 str repr。
    """
    def __init__(self, csv_file, tokenizer, max_len=512, sample=-1, seed=42, test=False):
        self.data = pd.read_csv(csv_file)
        if sample > 0 and len(self.data) > sample:
            self.data = self.data.sample(sample, random_state=seed)

        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.test = test
        self.inputs = []
        self._preprocess_all()

    def _preprocess_all(self):
        """预处理所有样本"""
        for idx in tqdm(range(len(self.data)), desc="SidSFTDataset", leave=False):
            self.inputs.append(self._preprocess_one(idx))

    def _preprocess_one(self, idx):
        """对齐参考代码的 pre() 函数"""
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        row = self.data.iloc[idx]
        history_sids = eval(row['history_item_sid'])  # Python list

        # 拼接 history: "<SID1>, <SID2>, ..."
        history_str = ", ".join(history_sids)
        prompt_input = f"The user has interacted with items {history_str} in chronological order. Can you predict the next possible item that the user may expect?"

        prompt = f"""### User Input: 
{prompt_input}

### Response:
"""
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        # 添加 target
        target_sid = str(row['item_sid'])
        target = target_sid + '\n'
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)

        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)

        # labels: input 部分全 -100，只监督 response
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        # 从左截断
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class SidItemFeatDataset(Dataset):
    """
    辅助任务：sid↔title 双向任务。

    从 item.json + index.json 构建 sid2title 和 title2sid 映射，
    生成两类样本：
    - sid2title: "What is the title of item '<a_3><b_12><c_7>'?" → "iPhone 12 Case"
    - title2sid: "Which item has the title: iPhone 12 Case?" → "<a_3><b_12><c_7>"
    """
    def __init__(self, item_file, index_file, tokenizer, max_len=512, sample=-1, seed=42, test=False):
        with open(item_file, 'r', encoding='utf-8') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.test = test

        # 构建 sid2title / title2sid
        self.sid2title = {}
        self.title2sid = {}

        for item_id, tokens in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                if len(tokens) >= 3:
                    # 拼接 SID str
                    combined_sid = tokens[0] + tokens[1] + tokens[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid

        # 生成样本
        self.data = []
        for sid, title in self.sid2title.items():
            self.data.append({'task': 'sid2title', 'input': sid, 'output': title})
        for title, sid in self.title2sid.items():
            self.data.append({'task': 'title2sid', 'input': title, 'output': sid})

        if sample > 0 and len(self.data) > sample:
            self.data = random.sample(self.data, sample)

        self.inputs = []
        self._preprocess_all()

    def _preprocess_all(self):
        for idx in tqdm(range(len(self.data)), desc="SidItemFeatDataset", leave=False):
            self.inputs.append(self._preprocess_one(idx))

    def _preprocess_one(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        data_point = self.data[idx]

        if data_point['task'] == 'title2sid':
            prompt_input = f"Which item has the title: {data_point['input']}?"
        else:  # sid2title
            prompt_input = f'What is the title of item "{data_point["input"]}"?'

        prompt = f"""### User Input: 
{prompt_input}

### Response:
"""
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        target = data_point['output'] + '\n'
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)

        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class FusionSeqRecDataset(Dataset):
    """
    辅助任务：history SID → target title 预测。

    融合 SID 历史和 item title 信息。
    """
    def __init__(self, csv_file, item_file, index_file, tokenizer, max_len=512, sample=-1, seed=42, test=False):
        self.data = pd.read_csv(csv_file)
        if sample > 0 and len(self.data) > sample:
            self.data = self.data.sample(sample, random_state=seed)

        with open(item_file, 'r', encoding='utf-8') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.test = test

        # 构建 sid2title
        self.sid2title = {}
        for item_id, tokens in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                if len(tokens) >= 3:
                    combined_sid = tokens[0] + tokens[1] + tokens[2]
                    self.sid2title[combined_sid] = title

        self.inputs = []
        self._preprocess_all()

    def _preprocess_all(self):
        for idx in tqdm(range(len(self.data)), desc="FusionSeqRecDataset", leave=False):
            item = self._preprocess_one(idx)
            if item is not None:
                self.inputs.append(item)

    def _preprocess_one(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        row = self.data.iloc[idx]
        history_sids = eval(row['history_item_sid'])
        history_str = ", ".join(history_sids)

        target_sid = str(row['item_sid'])
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid

        prompt_input = f"The user has sequentially interacted with items {history_str}. Can you recommend the next item for him? Tell me the title of the item"

        prompt = f"""### User Input: 
{prompt_input}

### Response:
"""
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        target = target_title + '\n'
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)

        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


# ============================================================
# 训练函数
# ============================================================

def train(
    # model/data params
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_file: str = "",
    eval_file: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    output_dir: str = "",
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    warmup_steps: int = 500,
    seed: int = 42,
    
    # optional
    sample: int = -1,
    category: str = "cell phones and accessories",
    resume_from_checkpoint: str = None,
    wandb_project: str = "",
    wandb_run_name: str = "",
    config: str = None,
):
    """
    主训练函数（对齐 MiniOneRec sft.py）
    
    如果指定 --config，从 JSON 文件加载参数（会覆盖命令行参数）。
    """
    # 从 config 文件加载参数
    if config and os.path.exists(config):
        print(f"[INFO] 从配置文件加载参数: {config}")
        with open(config, 'r') as f:
            config_dict = json.load(f)
        
        # 更新参数
        base_model = config_dict.get('base_model', base_model)
        train_file = config_dict.get('train_file', train_file)
        eval_file = config_dict.get('eval_file', eval_file)
        sid_index_path = config_dict.get('sid_index_path', sid_index_path)
        item_meta_path = config_dict.get('item_meta_path', item_meta_path)
        output_dir = config_dict.get('output_dir', output_dir)
        batch_size = config_dict.get('batch_size', batch_size)
        micro_batch_size = config_dict.get('micro_batch_size', micro_batch_size)
        num_epochs = config_dict.get('num_epochs', num_epochs)
        learning_rate = config_dict.get('learning_rate', learning_rate)
        cutoff_len = config_dict.get('cutoff_len', cutoff_len)
        warmup_steps = config_dict.get('warmup_steps', warmup_steps)
        seed = config_dict.get('seed', seed)
        sample = config_dict.get('sample', sample)
        category = config_dict.get('category', category)

    # 减少显存碎片，避免 OOM
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(seed)

    # 推导 run_name
    if not wandb_project:
        wandb_project = "cellphones_phase5"
    if not wandb_run_name:
        wandb_run_name = os.path.basename(output_dir)

    # 自动检测断点：如果 resume_from_checkpoint 未指定，自动查找 output_dir 内最新 checkpoint
    if resume_from_checkpoint is None:
        ckpt_dirs = [
            d for d in (
                [os.path.join(output_dir, x) for x in os.listdir(output_dir)]
                if os.path.exists(output_dir) else []
            )
            if os.path.isdir(d) and os.path.basename(d).startswith("checkpoint-")
        ]
        if ckpt_dirs:
            resume_from_checkpoint = max(ckpt_dirs, key=os.path.getmtime)
            print(f"[INFO] 检测到断点，将从 {resume_from_checkpoint} 续训")
        else:
            resume_from_checkpoint = False  # 无断点，从头训练

    gradient_accumulation_steps = batch_size // micro_batch_size
    
    print("="*60)
    print("Phase 5: 多任务 SFT 训练")
    print("="*60)
    print(f"Model: {base_model}")
    print(f"Train: {train_file}")
    print(f"Eval:  {eval_file}")
    print(f"Index: {sid_index_path}")
    print(f"Item:  {item_meta_path}")
    print(f"Output: {output_dir}")
    print(f"Batch: {batch_size} (micro={micro_batch_size}, accum={gradient_accumulation_steps})")
    print(f"LR: {learning_rate}, Warmup: {warmup_steps}, Epochs: {num_epochs}")
    print(f"Max length: {cutoff_len}, Seed: {seed}")
    print("="*60)
    
    # 加载模型
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    print(f"\n[1] 加载模型: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # 扩展词表（添加 SID tokens）
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"\n[2] 扩展词表（添加 SID tokens）")
        print(f"    从 {sid_index_path} 提取新 tokens")
        token_extender = TokenExtender(sid_index_path)
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"    添加 {len(new_tokens)} 个新 tokens")
            print(f"    示例: {new_tokens[:10]}")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
            print(f"    词表大小: {len(tokenizer)}")
    
    # 构建训练数据集（三个任务 ConcatDataset）
    print(f"\n[3] 构建训练数据集（三个任务）")
    train_datasets = []
    
    print("    ① SidSFTDataset (主任务)")
    train_data1 = SidSFTDataset(
        csv_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
    )
    train_datasets.append(train_data1)
    print(f"       → {len(train_data1)} 样本")
    
    print("    ② SidItemFeatDataset (辅助: sid↔title)")
    train_data2 = SidItemFeatDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
    )
    train_datasets.append(train_data2)
    print(f"       → {len(train_data2)} 样本")
    
    print("    ③ FusionSeqRecDataset (辅助: SID history→title)")
    train_data3 = FusionSeqRecDataset(
        csv_file=train_file,
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
    )
    train_datasets.append(train_data3)
    print(f"       → {len(train_data3)} 样本")
    
    train_data = ConcatDataset(train_datasets)
    print(f"    [OK] 总训练样本: {len(train_data)}")
    
    # 验证数据集（只用主任务）
    print(f"\n[4] 构建验证数据集（主任务）")
    val_data = SidSFTDataset(
        csv_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
    )
    print(f"    [OK] {len(val_data)} 样本")
    
    # 转换为 HuggingFace Dataset
    print(f"\n[5] 转换为 HuggingFace Dataset 格式")
    hf_train_dataset = HFDataset.from_dict(
        {k: [v[k] for v in train_data] for k in train_data[0].keys()}
    )
    hf_train_dataset = hf_train_dataset.shuffle(seed=seed)
    
    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in val_data] for k in val_data[0].keys()}
    )
    hf_val_dataset = hf_val_dataset.shuffle(seed=seed)
    
    print(f"    Train: {hf_train_dataset}")
    print(f"    Val:   {hf_val_dataset}")
    
    # 配置 Trainer
    print(f"\n[6] 配置 Trainer")
    os.makedirs(output_dir, exist_ok=True)
    
    eval_steps = 0.05
    training_args = TrainingArguments(
        run_name=wandb_run_name if wandb_run_name else os.path.basename(output_dir),
        output_dir=output_dir,
        
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        
        bf16=True,
        gradient_checkpointing=True,
        
        optim="adamw_torch",
        weight_decay=0.01,
        
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        
        ddp_find_unused_parameters=False if ddp else None,
        report_to="none",
    )

    # CSV 本地日志（完全不依赖网络）
    csv_log_file = os.path.join(output_dir, "train_log.csv")
    os.makedirs(output_dir, exist_ok=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            CsvLoggerCallback(log_file=csv_log_file, run_name=wandb_run_name),
        ],
    )
    
    model.config.use_cache = False
    
    # 开始训练
    print(f"\n[7] 开始训练")
    print("="*60)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存模型
    print(f"\n[8] 保存模型")
    trainer.save_model(output_dir)
    
    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"    [OK] 模型已保存至: {final_dir}")
    print(f"    [OK] 训练日志已保存至: {csv_log_file}")
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

    # 训练结束后统一上传到 wandb（网络可用时）
    print(f"\n[9] 上传训练日志到 wandb...")
    upload_to_wandb(log_file=csv_log_file, run_name=wandb_run_name, project=wandb_project)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import fire
    fire.Fire(train)
