"""
eval/build_trie.py

构建 SID Prefix Trie，用于约束解码（beam search 时限制只生成合法 SID）。

支持两种输入：
1. index.json  格式：{item_id: ["<a_X>", "<b_Y>", "<c_Z>"]}
2. prefix_trie.json 格式：{str(tuple): [token_str, ...]}（已预计算的 trie）

Trie 节点存储 token_id（整数），与 tokenizer 对齐。
"""

import json
import re
from typing import Dict, List, Optional, Tuple


class TrieNode:
    """Prefix Trie 节点，children key 为 token_id（整数）。"""

    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False


class SIDTrie:
    """
    SID Prefix Trie。

    每个 SID 由 3 个 token 组成：<a_X><b_Y><c_Z>。
    Trie 深度为 3，叶节点 is_end=True。

    用法：
        trie = SIDTrie.from_index_json(index_path, tokenizer)
        # 或
        trie = SIDTrie.from_trie_json(trie_path, tokenizer)

        # 在 beam search 的 prefix_allowed_tokens_fn 中：
        allowed = trie.get_next_tokens(generated_token_ids)
    """

    def __init__(self):
        self.root = TrieNode()
        self._token_str_to_id: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 构建方法
    # ------------------------------------------------------------------

    @classmethod
    def from_index_json(cls, index_path: str, tokenizer) -> "SIDTrie":
        """
        从 index.json 构建 Trie。

        index.json 格式：
            {item_id: ["<a_X>", "<b_Y>", "<c_Z>"], ...}
        """
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        trie = cls()
        trie._build_token_cache(index, tokenizer)

        for item_id, token_strs in index.items():
            if len(token_strs) < 3:
                continue
            token_ids = [trie._token_str_to_id.get(t) for t in token_strs[:3]]
            if any(tid is None for tid in token_ids):
                continue
            trie._insert(token_ids)

        return trie

    @classmethod
    def from_trie_json(cls, trie_path: str, tokenizer) -> "SIDTrie":
        """
        从预计算的 prefix_trie.json 构建 Trie。

        trie_json 格式：
            {
              "()": ["<a_0>", "<a_1>", ...],          # 根节点的合法第一个 token
              "('<a_0>',)": ["<b_5>", "<b_12>", ...], # 以 <a_0> 开头的合法第二个 token
              "('<a_0>', '<b_5>')": ["<c_3>", ...],    # 完整路径
              ...
            }
        """
        with open(trie_path, "r", encoding="utf-8") as f:
            trie_data = json.load(f)

        trie = cls()

        # 收集所有出现的 token 字符串，建立 token_str -> token_id 缓存
        all_tokens = set()
        for key_str, children_strs in trie_data.items():
            # key 是 Python tuple 的字符串表示，解析出其中的 token 字符串
            tokens_in_key = re.findall(r"'(<[abc]_\d+>)'", key_str)
            all_tokens.update(tokens_in_key)
            all_tokens.update(children_strs)

        trie._build_token_cache_from_set(all_tokens, tokenizer)

        # 遍历 trie_data，重建 TrieNode 结构
        # key 形如 "()" / "('<a_0>',)" / "('<a_0>', '<b_5>')"
        for key_str, children_strs in trie_data.items():
            prefix_tokens = re.findall(r"'(<[abc]_\d+>)'", key_str)
            prefix_ids = [trie._token_str_to_id.get(t) for t in prefix_tokens]
            if any(pid is None for pid in prefix_ids):
                continue

            # 导航到 prefix 对应的节点
            node = trie.root
            for pid in prefix_ids:
                if pid not in node.children:
                    node.children[pid] = TrieNode()
                node = node.children[pid]

            # 在该节点下添加 children
            for child_str in children_strs:
                child_id = trie._token_str_to_id.get(child_str)
                if child_id is None:
                    continue
                if child_id not in node.children:
                    node.children[child_id] = TrieNode()
                # 深度 3 的节点标记为叶节点
                if len(prefix_ids) == 2:
                    node.children[child_id].is_end = True

        return trie

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------

    def get_next_tokens(self, prefix_token_ids: List[int]) -> List[int]:
        """
        给定已生成的 SID token_id 前缀（长度 0/1/2），
        返回合法的下一个 token_id 列表。

        - 前缀长度 0：返回所有合法的第一个 token（<a_X>）
        - 前缀长度 1：返回以该 <a_X> 开头的合法第二个 token（<b_Y>）
        - 前缀长度 2：返回完整 SID 的第三个 token（<c_Z>）
        - 前缀长度 >= 3：SID 已完整，返回空列表（由 eos 处理）
        """
        if len(prefix_token_ids) >= 3:
            return []

        node = self.root
        for tid in prefix_token_ids:
            if tid not in node.children:
                return []
            node = node.children[tid]

        return list(node.children.keys())

    def contains(self, token_ids: List[int]) -> bool:
        """检查给定的 token_id 序列是否是一个完整合法的 SID。"""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                return False
            node = node.children[tid]
        return node.is_end

    def num_sids(self) -> int:
        """统计 Trie 中的 SID 总数（叶节点数）。"""
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_end:
                count += 1
            stack.extend(node.children.values())
        return count

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _build_token_cache(self, index: dict, tokenizer) -> None:
        """从 index.json 收集所有 SID token 字符串，编码为 token_id。"""
        all_tokens = set()
        for token_strs in index.values():
            all_tokens.update(token_strs)
        self._build_token_cache_from_set(all_tokens, tokenizer)

    def _build_token_cache_from_set(self, token_set: set, tokenizer) -> None:
        """将 token 字符串集合批量编码，填充 _token_str_to_id 缓存。"""
        for token_str in token_set:
            if token_str in self._token_str_to_id:
                continue
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) == 1:
                self._token_str_to_id[token_str] = ids[0]
            # 若 tokenizer 将 <a_X> 拆成多个 token，说明词表未正确扩展，跳过

    def _insert(self, token_ids: List[int]) -> None:
        """将一条 SID（3 个 token_id）插入 Trie。"""
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True


# ------------------------------------------------------------------
# 便捷工厂函数
# ------------------------------------------------------------------

def load_trie(index_or_trie_path: str, tokenizer, prefer_trie_json: bool = True) -> SIDTrie:
    """
    自动选择加载方式：
    - 若路径以 '_prefix_trie.json' 结尾，使用 from_trie_json
    - 否则使用 from_index_json
    """
    if prefer_trie_json and "_prefix_trie.json" in index_or_trie_path:
        return SIDTrie.from_trie_json(index_or_trie_path, tokenizer)
    else:
        return SIDTrie.from_index_json(index_or_trie_path, tokenizer)


# ------------------------------------------------------------------
# 快速验证（直接运行此文件时执行）
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # 默认测试路径（相对于 server_train 根目录运行）
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_cases = [
        {
            "name": "S2.5 (from index.json)",
            "index_path": os.path.join(BASE, "data", "sequences", "S2.5", "S2.5.index.json"),
            "trie_path": None,
        },
        {
            "name": "S2.5 (from trie.json)",
            "index_path": None,
            "trie_path": os.path.join(BASE, "trie", "S2.5_prefix_trie.json"),
        },
    ]

    # 加载 tokenizer（需要已训练的 checkpoint）
    checkpoint = os.path.join(
        BASE, "models", "results_20260306_142557", "train", "runs",
        "M_N_50K", "final_checkpoint"
    )
    if not os.path.exists(checkpoint):
        print(f"[SKIP] checkpoint 不存在: {checkpoint}")
        sys.exit(0)

    from transformers import AutoTokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    for tc in test_cases:
        print(f"\n=== 测试: {tc['name']} ===")
        if tc["trie_path"] and os.path.exists(tc["trie_path"]):
            trie = SIDTrie.from_trie_json(tc["trie_path"], tokenizer)
        elif tc["index_path"] and os.path.exists(tc["index_path"]):
            trie = SIDTrie.from_index_json(tc["index_path"], tokenizer)
        else:
            print("  [SKIP] 文件不存在")
            continue

        n = trie.num_sids()
        print(f"  SID 总数: {n}")

        # 测试根节点（空前缀）
        root_tokens = trie.get_next_tokens([])
        print(f"  根节点合法 token 数: {len(root_tokens)}")

        # 取第一个合法 a-token，测试第二层
        if root_tokens:
            first_a = root_tokens[0]
            b_tokens = trie.get_next_tokens([first_a])
            print(f"  第一个 a-token ({first_a}) 的合法 b-token 数: {len(b_tokens)}")

            if b_tokens:
                first_b = b_tokens[0]
                c_tokens = trie.get_next_tokens([first_a, first_b])
                print(f"  前缀 ({first_a}, {first_b}) 的合法 c-token 数: {len(c_tokens)}")

                if c_tokens:
                    first_c = c_tokens[0]
                    end_tokens = trie.get_next_tokens([first_a, first_b, first_c])
                    print(f"  完整 SID 后继续查询应返回 []: {end_tokens}")
                    valid = trie.contains([first_a, first_b, first_c])
                    print(f"  contains([a,b,c]) = {valid} (期望 True)")

    print("\n[OK] build_trie.py 验证完成")
