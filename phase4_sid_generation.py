"""
Phase 4: 文本表征与 FAISS RQ - 生成 Item Semantic IDs (SID)

对照 MiniOneRec 参考实现完整修复版本：

修复点：
1. 仅对 S3（最大子集）训练 RQ-VAE，冻结后复用编码所有子集
2. Embedding 使用 attention_mask 做正确 masked mean pooling，排除 padding
3. 文本拼接 title + store + main_category，丰富语义输入
4. 元数据清洗 bug 修复：先按 parent_asin 去重，再处理空 title
5. 为 S1/S2/S2.5/S3 各自产出独立 SID 映射与 Prefix Trie
6. 输出路径与项目规划对齐：embeddings/rq_vae、embeddings/sids、trie/
7. SID token 格式使用 <a_N><b_N><c_N> 与参考 evaluate.py/data.py 完全兼容
8. Trie 通用化：支持任意 n_levels 层，不再硬编码 3 层
9. 支持断点续跑：embedding 已存在时跳过重新生成
10. 保存 asin→int_index 映射，方便后续离线查找

参考 MiniOneRec:
- rqkmeans_faiss.py: train_faiss_rq / encode_with_rq / unpack_rq_codes
- generate_indices.py: prefix = ["<a_{}>","<b_{}>","<c_{}>"] 格式
- data.py SidSFTDataset: history_item_sid 列表 -> 逗号拼接推理
- evaluate.py: hash_dict Trie 由 tokenizer ID 序列构建（Phase 6 用）
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import defaultdict
import faiss


# ============================================================
# 全局配置
# ============================================================

CONFIG = {
    # RQ-VAE 参数
    'codebook_size': 256,       # 每层 codebook 大小，256 = 2^8
    'n_levels': 3,              # 层数 L=3
    # Embedding 模型
    'embedding_model': 'Qwen/Qwen3-Embedding-0.6B',
    'batch_size': 64,
    'max_text_length': 512,
    # SID token 前缀（与 MiniOneRec generate_indices.py 一致）
    'sid_prefix': ['<a_{}>', '<b_{}>', '<c_{}>', '<d_{}>', '<e_{}>'],
    # 子集列表（处理顺序：大→小）
    'subsets': ['S3', 'S2.5', 'S2', 'S1'],
    # 路径
    'paths': {
        'metadata':      'data/raw/item_metadata.parquet',
        'subsets_dir':   'data/subsets',
        'embeddings_dir': 'embeddings',
        'rqvae_dir':     'embeddings/rq_vae',
        'sids_dir':      'embeddings/sids',
        'trie_dir':      'trie',
    }
}


# ============================================================
# 第 1 步：Item Embedding 生成
# ============================================================

class ItemEmbeddingGenerator:
    """
    使用 Qwen3-Embedding-0.6B 生成 item 文本嵌入。

    关键实现细节（来自官方 GitHub: QwenLM/Qwen3-Embedding）：
    1. padding_side='left'：decoder-only 模型必须左填充，右填充导致
       最后一个有效 token 位置计算错误。
    2. last_token_pool：Qwen3-Embedding 官方指定的池化方式，
       mean pooling 会静默产生错误的 embedding。
    3. attn_implementation="flash_attention_2" + torch_dtype=float16：
       官方推荐，速度约快 2x，显存节省 ~50%。
       若未安装 flash-attn，将自动 fallback 到标准 attention。
    """

    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device='cuda', batch_size=64):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        print(f"[INFO] 加载 Embedding 模型: {model_name}")
        print(f"       设备: {self.device}")

        # ① padding_side='left'（官方要求，decoder-only 标准）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side='left'
        )

        # ② flash_attention_2 + fp16（官方推荐；无 flash-attn 时自动降级）
        try:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
            ).to(self.device)
            print(f"       精度: fp16 + flash_attention_2")
        except Exception:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
            print(f"       精度: fp16（flash_attention_2 不可用，已降级）")

        self.model.eval()

        # 探测实际输出维度
        with torch.no_grad():
            _d = self.tokenizer("probe", return_tensors='pt').to(self.device)
            _o = self.model(**_d)
            self.embed_dim = _o.last_hidden_state.shape[-1]

        print(f"[OK] 模型加载完成（输出维度: {self.embed_dim}）")

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        """
        官方 last_token_pool（QwenLM/Qwen3-Embedding）。
        自动兼容左填充（padding_side='left'）和右填充两种情况：
        - 左填充：最后一列 attention_mask 全为 1 → 直接取最后一个 token
        - 右填充：按每条序列实际长度取最后一个有效 token
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode_texts(self, texts: list, show_progress: bool = True) -> torch.Tensor:
        """批量编码文本为 L2 归一化 embeddings，返回 fp32 CPU 张量 [N, D]"""
        all_embeddings = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        it = range(0, len(texts), self.batch_size)
        if show_progress:
            it = tqdm(it, desc="生成 embeddings", total=n_batches)

        with torch.no_grad():
            for i in it:
                batch = texts[i: i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=CONFIG['max_text_length'],
                    return_tensors='pt',
                ).to(self.device)

                outputs = self.model(**inputs)

                # ③ 官方 last_token_pool 替换 masked_mean_pool
                emb = self.last_token_pool(
                    outputs.last_hidden_state, inputs['attention_mask']
                )
                # fp16 → fp32 再做 L2 normalize，避免精度损失
                emb = F.normalize(emb.float(), p=2, dim=1)
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)   # [N, D]，fp32


# ============================================================
# 第 2 步：FAISS ResidualQuantizer
# ============================================================

def train_faiss_rq(data: np.ndarray, num_levels: int = 3,
                   codebook_size: int = 256, verbose: bool = True) -> faiss.ResidualQuantizer:
    """
    训练 FAISS ResidualQuantizer。
    严格对齐 MiniOneRec rqkmeans_faiss.py: train_faiss_rq()
    """
    N, d = data.shape
    nbits = int(np.log2(codebook_size))   # 256 -> 8 bits

    if verbose:
        print(f"\n{'='*60}")
        print(f"训练 FAISS ResidualQuantizer")
        print(f"{'='*60}")
        print(f"  数据: {N:,} items × {d}d")
        print(f"  Levels: {num_levels}, Codebook size: {codebook_size} ({nbits} bits)")
        print(f"  总 code 空间: {codebook_size ** num_levels:,}")

    rq = faiss.ResidualQuantizer(d, num_levels, nbits)
    rq.train_type   = faiss.ResidualQuantizer.Train_default   # k-means 初始化
    rq.max_beam_size = 1                                       # 贪心编码

    data_f32 = np.ascontiguousarray(data.astype(np.float32))
    print(f"\n[INFO] 开始训练（k-means 自动初始化）...")
    rq.train(data_f32)

    if verbose:
        print(f"[OK] 训练完成")
    return rq


def unpack_rq_codes(codes: np.ndarray, nbits: int, num_levels: int) -> np.ndarray:
    """
    解包 FAISS bit-packed codes → int32 索引矩阵 [N, num_levels]。
    完全对齐 MiniOneRec rqkmeans_faiss.py: unpack_rq_codes()
    """
    N = codes.shape[0]
    # FAISS Little-Endian packing
    packed = np.zeros(N, dtype=np.int64)
    for i in range(codes.shape[1]):
        packed |= codes[:, i].astype(np.int64) << (8 * i)

    mask = (1 << nbits) - 1
    out  = np.zeros((N, num_levels), dtype=np.int32)
    for i in range(num_levels):
        out[:, i] = (packed >> (i * nbits)) & mask
    return out


def encode_with_rq(rq: faiss.ResidualQuantizer, data: np.ndarray,
                   codebook_size: int, verbose: bool = True) -> np.ndarray:
    """
    将 embeddings 编码为整数 code 矩阵 [N, n_levels]。
    对齐 MiniOneRec rqkmeans_faiss.py: encode_with_rq()
    """
    nbits    = int(np.log2(codebook_size))
    data_f32 = np.ascontiguousarray(data.astype(np.float32))

    if verbose:
        print(f"\n[INFO] 编码 {data.shape[0]:,} 个 items...")

    codes_packed = rq.compute_codes(data_f32)

    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        codes = unpack_rq_codes(codes_packed, nbits, rq.M)

    codes = codes.astype(np.int32)

    if verbose:
        print(f"[OK] 编码完成, shape={codes.shape}")
    return codes


def int_codes_to_token_sids(codes: np.ndarray, n_levels: int) -> list:
    """
    将整数 code 矩阵转为带前缀的 token 字符串列表。
    对齐 MiniOneRec generate_indices.py:
        prefix = ["<a_{}>","<b_{}>","<c_{}>",...]
        code = [prefix[i].format(int(c)) for i, c in enumerate(index)]
    返回：[["<a_3>","<b_12>","<c_7>"], ...]  shape [N, n_levels]
    """
    prefix = CONFIG['sid_prefix']
    result = []
    for row in codes:
        tokens = [prefix[i].format(int(c)) for i, c in enumerate(row[:n_levels])]
        result.append(tokens)
    return result


def token_sids_to_str(token_list: list) -> str:
    """
    将 token 列表拼接为字符串，与 MiniOneRec evaluate.py 一致：
        semantic_id = tokens[0] + tokens[1] + tokens[2]   (无空格直接拼接)
    例：["<a_3>","<b_12>","<c_7>"] -> "<a_3><b_12><c_7>"
    """
    return ''.join(token_list)


def analyze_codes(codes: np.ndarray, title: str = "", verbose: bool = True) -> float:
    """分析 code 统计信息，返回碰撞率"""
    N, M = codes.shape
    collision_rate = 0.0
    if verbose:
        if title:
            print(f"\n{title}")
        print(f"  Total items: {N:,}")
        for l in range(M):
            unique = len(np.unique(codes[:, l]))
            print(f"  Level {l}: unique={unique}/{CONFIG['codebook_size']} "
                  f"({unique/CONFIG['codebook_size']*100:.1f}% utilization)")
        combos = len(set(map(tuple, codes)))
        collision_rate = 1.0 - combos / N
        print(f"  Unique full-paths: {combos:,}/{N:,}")
        print(f"  Collision rate: {collision_rate*100:.2f}%")
    return collision_rate


# ============================================================
# 第 3 步：Prefix Trie 构建（通用版，任意层数）
# ============================================================

def build_flat_trie(sid_mapping: dict, n_levels: int = 3) -> dict:
    """
    构建扁平 dict Trie，支持任意层数。

    输入 sid_mapping: { asin: ["<a_3>","<b_12>","<c_7>"] }   (token 列表)
    输出: { str(prefix_tuple): sorted_list_of_next_tokens }
    例：
      {"()"         : ["<a_0>","<a_1>",...]  }  Level 0 根节点
      {"('<a_3>',)" : ["<b_5>","<b_12>",...]  }  Level 1
      {"('<a_3>', '<b_12>')" : ["<c_7>"]      }  Level 2 叶子

    与 MiniOneRec evaluate.py 中 hash_dict 的 get_hash 语义对应：
      get_hash(x) = '-'.join([str(_) for _ in x])
    这里用 str(tuple) 作 key，Phase 6 LogitsProcessor 按相同格式查询即可。
    """
    flat_trie: dict[str, set] = defaultdict(set)

    for asin, tokens in sid_mapping.items():
        if len(tokens) != n_levels:
            continue
        for level in range(n_levels):
            prefix_key = str(tuple(tokens[:level]))   # "()" / "('<a_3>',)" / ...
            flat_trie[prefix_key].add(tokens[level])

    return {k: sorted(v) for k, v in flat_trie.items()}


# ============================================================
# 数据加载
# ============================================================

def load_subset_items(subset_name: str) -> set:
    """加载子集的 item 列表（parent_asin 字符串集合）"""
    filepath = os.path.join(CONFIG['paths']['subsets_dir'], f'{subset_name}_items.txt')
    with open(filepath, 'r', encoding='utf-8') as f:
        items = {line.strip() for line in f if line.strip()}
    print(f"[INFO] {subset_name}: {len(items):,} items")
    return items


def build_item_text(row: pd.Series) -> str:
    """
    拼接 item 文本：title + store + main_category。
    对应实验方案 Phase 4："对 S3 的 item 文本（title + brand + category）编码"
    """
    parts = []

    title = row.get('title', '')
    if pd.notna(title) and str(title).strip():
        parts.append(str(title).strip())
    else:
        parts.append(str(row['parent_asin']))   # fallback

    store = row.get('store', '')
    if pd.notna(store) and str(store).strip():
        parts.append(f"Brand: {str(store).strip()}")

    cat = row.get('main_category', '')
    if pd.notna(cat) and str(cat).strip():
        parts.append(f"Category: {str(cat).strip()}")

    return ' | '.join(parts)


def load_metadata_for_items(target_asins: set) -> dict:
    """
    加载 metadata，只保留 target_asins 中的 items。
    修复原代码 bug：先去重，再 fillna（原先 dropna 把 title=NaN 整行删掉，
    导致 fillna 无法生效）。
    """
    print(f"\n[INFO] 加载 item metadata（目标: {len(target_asins):,} items）...")

    df = pd.read_parquet(CONFIG['paths']['metadata'])

    # 只保留目标 items，按 parent_asin 去重（取第一条）
    df = df[df['parent_asin'].isin(target_asins)].copy()
    df = df.drop_duplicates(subset='parent_asin', keep='first')

    # 构建拼接文本
    df['_text'] = df.apply(build_item_text, axis=1)
    text_map = dict(zip(df['parent_asin'], df['_text']))

    # 补全 metadata 里没有的 items（用 asin 作 fallback）
    missing = target_asins - set(text_map.keys())
    if missing:
        print(f"[WARN] {len(missing)} 个 items 无 metadata，使用 asin 作 fallback")
        for asin in missing:
            text_map[asin] = asin

    print(f"[OK] 有效文本: {len(text_map):,} items")

    # 打印前 3 条示例
    for asin in list(text_map.keys())[:3]:
        print(f"  {asin}: {text_map[asin][:100]}")

    return text_map


# ============================================================
# 主流程函数
# ============================================================

def step1_generate_s3_embeddings(device: str) -> tuple:
    """
    步骤 1：加载 S3 items，生成（或加载已有）embeddings。
    支持断点续跑：pt 文件已存在则直接加载。
    """
    print(f"\n{'#'*60}")
    print(f"# 步骤 1: 加载 S3 并生成 Embeddings")
    print(f"{'#'*60}")

    s3_items = load_subset_items('S3')
    emb_path = os.path.join(CONFIG['paths']['embeddings_dir'], 'S3_item_embeddings.pt')

    if os.path.exists(emb_path):
        print(f"[INFO] 发现已有 embeddings，直接加载: {emb_path}")
        ckpt = torch.load(emb_path, map_location='cpu')
        s3_asins: list = ckpt['asins']
        s3_embeddings: torch.Tensor = ckpt['embeddings']
        print(f"[OK] 加载完成: {len(s3_asins):,} items, shape={s3_embeddings.shape}")
    else:
        text_map = load_metadata_for_items(s3_items)
        # 保证 asin 顺序确定（与 metadata 顺序一致）
        s3_asins = sorted(text_map.keys())
        texts   = [text_map[a] for a in s3_asins]

        print(f"\n[INFO] 开始生成 {len(s3_asins):,} 个 item 的 embeddings...")
        generator = ItemEmbeddingGenerator(
            CONFIG['embedding_model'], device, CONFIG['batch_size']
        )
        s3_embeddings = generator.encode_texts(texts, show_progress=True)
        print(f"[OK] Embeddings shape: {s3_embeddings.shape}")

        os.makedirs(CONFIG['paths']['embeddings_dir'], exist_ok=True)
        torch.save(
            {'asins': s3_asins, 'embeddings': s3_embeddings,
             'model_name': CONFIG['embedding_model']},
            emb_path
        )
        print(f"[OK] 已保存: {emb_path}")

    return s3_asins, s3_embeddings


def step2_train_rq_on_s3(s3_asins: list, s3_embeddings: torch.Tensor) -> faiss.ResidualQuantizer:
    """
    步骤 2：仅在 S3 embeddings 上训练一次 FAISS RQ，冻结后复用。
    这是与原始脚本最核心的差别——训练只发生在 S3 上，
    S1/S2/S2.5 的编码直接用同一个 RQ，保证 SID 空间一致。
    """
    print(f"\n{'#'*60}")
    print(f"# 步骤 2: 在 S3 上训练 FAISS RQ")
    print(f"{'#'*60}")

    emb_np = s3_embeddings.numpy()

    rq = train_faiss_rq(
        emb_np,
        num_levels=CONFIG['n_levels'],
        codebook_size=CONFIG['codebook_size'],
        verbose=True,
    )

    # 保存 RQ-VAE
    rqvae_dir = CONFIG['paths']['rqvae_dir']
    os.makedirs(rqvae_dir, exist_ok=True)

    nbits = int(np.log2(CONFIG['codebook_size']))
    faiss_path = os.path.join(rqvae_dir, 'faiss_rq.index')
    try:
        index = faiss.IndexResidualQuantizer(rq.d, rq.M, nbits)
        index.rq        = rq
        index.is_trained = True
        faiss.write_index(index, faiss_path)
        print(f"[OK] FAISS index 已保存: {faiss_path}")
    except Exception as e:
        print(f"[WARN] FAISS index 保存失败（不影响后续流程）: {e}")

    # 验证 S3 编码质量
    s3_codes = encode_with_rq(rq, emb_np, CONFIG['codebook_size'], verbose=True)
    s3_cr    = analyze_codes(s3_codes, title="S3 SID 统计信息")

    # 保存 RQ 配置
    cfg = {
        'embedding_model': CONFIG['embedding_model'],
        'input_dim':       int(rq.d),
        'codebook_size':   CONFIG['codebook_size'],
        'n_levels':        CONFIG['n_levels'],
        'trained_on':      'S3',
        'n_trained_items': len(s3_asins),
        's3_collision_rate': float(s3_cr),
    }
    with open(os.path.join(rqvae_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"[OK] RQ 配置已保存: {os.path.join(rqvae_dir, 'config.json')}")

    return rq


def step3_generate_subset_sids(
    rq: faiss.ResidualQuantizer,
    s3_asins: list,
    s3_embeddings: torch.Tensor,
) -> dict:
    """
    步骤 3：用已冻结的 RQ 为每个子集（S3/S2.5/S2/S1）生成 SID 映射 + Trie。

    实现细节：
    - 所有子集 items ⊆ S3，因此只需对 S3 编码一次，然后按子集过滤索引。
    - SID 格式对齐 MiniOneRec generate_indices.py：
        ["<a_3>", "<b_12>", "<c_7>"]  （token 字符串列表）
    - Trie key 格式：str(tuple(tokens[:level]))，与 Phase 6 LogitsProcessor 查询方式一致。
    - 同时保存 asin→int_index 映射（embeddings 矩阵的行索引），供后续离线检索用。
    """
    print(f"\n{'#'*60}")
    print(f"# 步骤 3: 为所有子集生成 SID 和 Trie")
    print(f"{'#'*60}")

    n_levels = CONFIG['n_levels']
    sids_dir = CONFIG['paths']['sids_dir']
    trie_dir = CONFIG['paths']['trie_dir']
    os.makedirs(sids_dir, exist_ok=True)
    os.makedirs(trie_dir, exist_ok=True)

    # ── 一次性编码 S3 全量 ──────────────────────────────────────────
    emb_np = s3_embeddings.numpy()
    s3_int_codes = encode_with_rq(rq, emb_np, CONFIG['codebook_size'], verbose=True)
    # int_codes: [N, n_levels]  整数矩阵

    # asin -> S3 行索引
    s3_asin2idx = {asin: idx for idx, asin in enumerate(s3_asins)}

    # 转为 token SID：["<a_3>","<b_12>","<c_7>"]
    all_token_sids = int_codes_to_token_sids(s3_int_codes, n_levels)

    all_subset_stats = {}

    for subset_name in CONFIG['subsets']:
        print(f"\n{'='*50}")
        print(f"处理 {subset_name}")
        print(f"{'='*50}")

        subset_items = load_subset_items(subset_name)

        # 构建 {asin: token_list} 映射
        sid_mapping: dict[str, list] = {}
        for asin in subset_items:
            if asin in s3_asin2idx:
                idx = s3_asin2idx[asin]
                sid_mapping[asin] = all_token_sids[idx]   # e.g. ["<a_3>","<b_12>","<c_7>"]
            # else: asin 不在 S3（理论上不应发生，因为 S1⊂S2⊂S2.5⊂S3）

        missing = len(subset_items) - len(sid_mapping)
        if missing:
            print(f"[WARN] {missing} 个 items 不在 S3（已跳过）")

        # 分析碰撞率
        subset_int_codes = s3_int_codes[
            [s3_asin2idx[a] for a in sid_mapping.keys()]
        ]
        cr = analyze_codes(subset_int_codes, title=f"{subset_name} SID 统计")

        # ── 保存 SID 映射 ─────────────────────────────────────────
        # 同时保存两种格式：
        #   token_list：供训练时直接读取，格式与 MiniOneRec index.json 一致
        #   str_sid   ：拼接字符串版，供 item-info 文件使用
        sid_out = {
            asin: {
                'token_list': tokens,           # ["<a_3>","<b_12>","<c_7>"]
                'str_sid':    token_sids_to_str(tokens),   # "<a_3><b_12><c_7>"
            }
            for asin, tokens in sid_mapping.items()
        }
        sid_path = os.path.join(sids_dir, f'{subset_name}_sid_mapping.json')
        with open(sid_path, 'w', encoding='utf-8') as f:
            json.dump(sid_out, f, ensure_ascii=False)
        print(f"[OK] SID 映射已保存: {sid_path}")

        # ── 构建并保存 Trie ────────────────────────────────────────
        flat_trie = build_flat_trie(sid_mapping, n_levels)

        trie_path = os.path.join(trie_dir, f'{subset_name}_prefix_trie.json')
        with open(trie_path, 'w', encoding='utf-8') as f:
            json.dump(flat_trie, f, ensure_ascii=False)
        print(f"[OK] Trie 已保存: {trie_path}")

        # Trie 统计
        root_choices = len(flat_trie.get('()', []))
        print(f"     Trie 节点数: {len(flat_trie):,}  |  Level-0 choices: {root_choices}")

        # ── 验证（抽样 3 条路径）────────────────────────────────────
        print(f"\n[验证] {subset_name} 随机 3 条 SID 约束路径:")
        for asin in list(sid_mapping.keys())[:3]:
            tokens = sid_mapping[asin]
            path_info = []
            for level in range(n_levels):
                prefix_key  = str(tuple(tokens[:level]))
                valid_next  = flat_trie.get(prefix_key, [])
                in_trie_sym = '✓' if tokens[level] in valid_next else '✗'
                path_info.append(f"L{level}[{len(valid_next)}]{in_trie_sym}")
            print(f"  {asin[:15]}... {tokens} → {' | '.join(path_info)}")

        all_subset_stats[subset_name] = {
            'n_items':       len(sid_mapping),
            'collision_rate': float(cr),
            'trie_nodes':    len(flat_trie),
            'level0_choices': root_choices,
        }

    # 保存汇总统计
    stats_path = os.path.join(sids_dir, 'subset_sid_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(all_subset_stats, f, indent=2)
    print(f"\n[OK] 子集统计已保存: {stats_path}")

    return all_subset_stats


def summarize_phase4(all_subset_stats: dict):
    """打印 Phase 4 完成总结"""
    print(f"\n{'='*60}")
    print(f"Phase 4 完成总结")
    print(f"{'='*60}")

    rq_cfg_path = os.path.join(CONFIG['paths']['rqvae_dir'], 'config.json')
    with open(rq_cfg_path) as f:
        rq_cfg = json.load(f)

    print(f"\n[RQ-VAE]")
    print(f"  训练集: {rq_cfg['trained_on']}  ({rq_cfg['n_trained_items']:,} items)")
    print(f"  Input dim: {rq_cfg['input_dim']}  |  Codebook: {rq_cfg['codebook_size']}  |  Levels: {rq_cfg['n_levels']}")
    print(f"  S3 碰撞率: {rq_cfg['s3_collision_rate']*100:.2f}%")

    print(f"\n[子集 SID 统计]")
    print(f"  {'子集':<8}  {'N':>8}  {'碰撞率':>8}  {'Trie节点':>10}  {'Level0选择':>12}")
    print(f"  {'-'*54}")
    for sn, st in all_subset_stats.items():
        print(f"  {sn:<8}  {st['n_items']:>8,}  {st['collision_rate']*100:>7.2f}%"
              f"  {st['trie_nodes']:>10,}  {st['level0_choices']:>12,}")

    print(f"\n[产出文件]")
    emb_dir  = CONFIG['paths']['embeddings_dir']
    rq_dir   = CONFIG['paths']['rqvae_dir']
    sid_dir  = CONFIG['paths']['sids_dir']
    tri_dir  = CONFIG['paths']['trie_dir']
    print(f"  {emb_dir}/S3_item_embeddings.pt")
    print(f"  {rq_dir}/faiss_rq.index  +  config.json")
    for sn in CONFIG['subsets']:
        print(f"  {sid_dir}/{sn}_sid_mapping.json")
        print(f"  {tri_dir}/{sn}_prefix_trie.json")

    print(f"\n[下一步]  Phase 5：模型训练（8 个模型配置）")


def main():
    print("="*60)
    print("Phase 4: 文本表征与 RQ-VAE 离散化")
    print("="*60)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] 使用设备: {device}")
    if device == 'cpu':
        print(f"[WARN] 未检测到 GPU，Embedding 生成会较慢")

    # 步骤 1：生成（或加载）S3 embeddings
    s3_asins, s3_embeddings = step1_generate_s3_embeddings(device)

    # 步骤 2：在 S3 上训练 FAISS RQ（仅一次，冻结）
    rq = step2_train_rq_on_s3(s3_asins, s3_embeddings)

    # 步骤 3：为 S3/S2.5/S2/S1 各自生成 SID + Trie
    all_subset_stats = step3_generate_subset_sids(rq, s3_asins, s3_embeddings)

    # 总结
    summarize_phase4(all_subset_stats)

    print(f"\n{'='*60}")
    print(f"Phase 4 全部完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
