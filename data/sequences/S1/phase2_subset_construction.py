"""
Phase 2: 嵌套候选集 S1-S3 构造
- 统计 item popularity 并做 log-binning 分桶
- 按分层抽样从 116K items 中抽取 S3(100K)
- 在 S3 内嵌套抽样: S2.5(50K) → S2(20K) → S1(5K)
- 验证各子集 popularity 分布形状一致
- 保存各子集的 item 列表与统计
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_interactions():
    """加载 Phase 1 产出的交互数据"""
    print("="*60)
    print("Phase 2: 嵌套候选集构造")
    print("="*60)
    print("\n加载 Phase 1 数据...")
    
    df = pd.read_parquet('data/raw/interactions_5core.parquet')
    print(f"[OK] 加载了 {len(df)} 条交互，{df['parent_asin'].nunique()} 个 items")
    
    return df

def log_binning(popularity_values, num_bins=20):
    """对 popularity 做 log-binning 分桶"""
    print(f"\n执行 log-binning 分桶（{num_bins} 个桶）...")
    
    # 对数尺度分桶边界
    min_pop = popularity_values.min()
    max_pop = popularity_values.max()
    
    # 创建 log 尺度的桶边界
    bins = np.logspace(np.log10(min_pop), np.log10(max_pop), num_bins + 1)
    
    # 为每个 popularity 值分配桶 ID
    bin_ids = np.digitize(popularity_values, bins) - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)  # 确保在范围内
    
    # 统计每个桶的 item 数量
    unique, counts = np.unique(bin_ids, return_counts=True)
    
    print(f"[OK] 分桶完成，每个桶的 item 数:")
    for bin_id, count in zip(unique, counts):
        bin_range = f"[{bins[bin_id]:.1f}, {bins[bin_id+1]:.1f})"
        print(f"  桶 {bin_id:2d} ({bin_range:20s}): {count:6d} items")
    
    return bin_ids, bins

def stratified_sample(items_with_bins, target_size, seed=42):
    """按分桶比例分层抽样"""
    print(f"\n分层抽样目标: {target_size} items...")
    
    # 统计每个桶的 item 数
    bin_counts = defaultdict(list)
    for item, bin_id in items_with_bins:
        bin_counts[bin_id].append(item)
    
    total_items = sum(len(items) for items in bin_counts.values())
    
    # 按桶的比例计算每个桶应抽取的数量
    sampled_items = []
    np.random.seed(seed)
    
    for bin_id in sorted(bin_counts.keys()):
        items_in_bin = bin_counts[bin_id]
        bin_ratio = len(items_in_bin) / total_items
        n_to_sample = max(1, int(target_size * bin_ratio))  # 至少抽 1 个
        
        # 如果桶内 item 数少于目标数，全部取
        n_to_sample = min(n_to_sample, len(items_in_bin))
        
        sampled = np.random.choice(items_in_bin, size=n_to_sample, replace=False)
        sampled_items.extend(sampled)
    
    # 如果抽样数量不足，随机补充；如果超出，随机删减
    if len(sampled_items) < target_size:
        remaining = target_size - len(sampled_items)
        all_items = [item for items in bin_counts.values() for item in items]
        available = list(set(all_items) - set(sampled_items))
        if len(available) >= remaining:
            additional = np.random.choice(available, size=remaining, replace=False)
            sampled_items.extend(additional)
    elif len(sampled_items) > target_size:
        sampled_items = list(np.random.choice(sampled_items, size=target_size, replace=False))
    
    print(f"[OK] 分层抽样完成: {len(sampled_items)} items")
    
    return set(sampled_items)

def construct_nested_subsets(df):
    """构造嵌套候选集 S1 ⊂ S2 ⊂ S2.5 ⊂ S3"""
    print("\n构造嵌套候选集...")
    
    # 计算 item popularity
    item_popularity = df['parent_asin'].value_counts()
    all_items = item_popularity.index.tolist()
    popularity_values = item_popularity.values
    
    print(f"总 item 池: {len(all_items)} items")
    
    # Log-binning
    bin_ids, bins = log_binning(popularity_values, num_bins=20)
    
    # 创建 (item, bin_id) 列表
    items_with_bins = list(zip(all_items, bin_ids))
    
    # 目标规模
    targets = {
        'S3': 100000,
        'S2.5': 50000,
        'S2': 20000,
        'S1': 5000
    }
    
    # 分层抽样 S3
    print(f"\n{'='*60}")
    print("Step 1: 从全量 116K 抽样 S3")
    print(f"{'='*60}")
    S3 = stratified_sample(items_with_bins, targets['S3'], seed=42)
    
    # 在 S3 内嵌套抽样 S2.5
    print(f"\n{'='*60}")
    print("Step 2: 在 S3 内抽样 S2.5")
    print(f"{'='*60}")
    S3_items_with_bins = [(item, bin_id) for item, bin_id in items_with_bins if item in S3]
    S2_5 = stratified_sample(S3_items_with_bins, targets['S2.5'], seed=43)
    
    # 在 S2.5 内嵌套抽样 S2
    print(f"\n{'='*60}")
    print("Step 3: 在 S2.5 内抽样 S2")
    print(f"{'='*60}")
    S2_5_items_with_bins = [(item, bin_id) for item, bin_id in S3_items_with_bins if item in S2_5]
    S2 = stratified_sample(S2_5_items_with_bins, targets['S2'], seed=44)
    
    # 在 S2 内嵌套抽样 S1
    print(f"\n{'='*60}")
    print("Step 4: 在 S2 内抽样 S1")
    print(f"{'='*60}")
    S2_items_with_bins = [(item, bin_id) for item, bin_id in S2_5_items_with_bins if item in S2]
    S1 = stratified_sample(S2_items_with_bins, targets['S1'], seed=45)
    
    # 验证嵌套关系
    print(f"\n{'='*60}")
    print("验证嵌套关系")
    print(f"{'='*60}")
    assert S1.issubset(S2), "S1 NOT subset of S2"
    assert S2.issubset(S2_5), "S2 NOT subset of S2.5"
    assert S2_5.issubset(S3), "S2.5 NOT subset of S3"
    print("[OK] S1 subset S2 subset S2.5 subset S3 验证通过")
    
    subsets = {
        'S1': S1,
        'S2': S2,
        'S2.5': S2_5,
        'S3': S3
    }
    
    return subsets, item_popularity

def analyze_subset_distributions(subsets, item_popularity):
    """分析并可视化各子集的 popularity 分布"""
    print(f"\n{'='*60}")
    print("分析各子集 popularity 分布")
    print(f"{'='*60}")
    
    subset_stats = {}
    
    for name in ['S1', 'S2', 'S2.5', 'S3']:
        items = subsets[name]
        popularity_values = item_popularity[list(items)].values
        
        stats = {
            'size': len(items),
            'mean': float(np.mean(popularity_values)),
            'median': float(np.median(popularity_values)),
            'std': float(np.std(popularity_values)),
            'min': int(np.min(popularity_values)),
            'max': int(np.max(popularity_values)),
            'p25': float(np.percentile(popularity_values, 25)),
            'p75': float(np.percentile(popularity_values, 75)),
        }
        
        subset_stats[name] = stats
        
        print(f"\n{name} ({stats['size']} items):")
        print(f"  均值={stats['mean']:.1f}, 中位数={stats['median']:.1f}, 标准差={stats['std']:.1f}")
        print(f"  范围=[{stats['min']}, {stats['max']}], IQR=[{stats['p25']:.1f}, {stats['p75']:.1f}]")
    
    return subset_stats

def plot_subset_distributions(subsets, item_popularity, output_dir='data/subsets'):
    """绘制各子集的 popularity 分布对比图"""
    print(f"\n绘制子集分布对比图...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'orange', 'red']
    subset_names = ['S1', 'S2', 'S2.5', 'S3']
    
    for idx, name in enumerate(subset_names):
        items = list(subsets[name])
        popularity_values = item_popularity[items].values
        
        # Rank-Frequency curve (log-log)
        sorted_pop = np.sort(popularity_values)[::-1]
        ranks = np.arange(1, len(sorted_pop) + 1)
        
        axes[idx].loglog(ranks, sorted_pop, color=colors[idx], alpha=0.7, linewidth=1.5)
        axes[idx].set_xlabel('Item Rank (log)', fontsize=10)
        axes[idx].set_ylabel('Popularity (log)', fontsize=10)
        axes[idx].set_title(f'{name} ({len(items)} items)', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'subsets_popularity_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 对比图已保存: {plot_path}")
    
    # 绘制叠加图（所有子集在一张图上）
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for idx, name in enumerate(subset_names):
        items = list(subsets[name])
        popularity_values = item_popularity[items].values
        sorted_pop = np.sort(popularity_values)[::-1]
        ranks = np.arange(1, len(sorted_pop) + 1)
        
        ax.loglog(ranks, sorted_pop, color=colors[idx], alpha=0.7, 
                 linewidth=2, label=f'{name} (N={len(items)})')
    
    ax.set_xlabel('Item Rank (log scale)', fontsize=12)
    ax.set_ylabel('Popularity (log scale)', fontsize=12)
    ax.set_title('Nested Subsets Popularity Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, 'subsets_overlay.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 叠加图已保存: {overlay_path}")

def save_subsets(subsets, subset_stats, output_dir='data/subsets'):
    """保存各子集的 item 列表与统计"""
    print(f"\n保存子集数据...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name in ['S1', 'S2', 'S2.5', 'S3']:
        # 保存 item 列表
        items = sorted(list(subsets[name]))
        item_list_path = os.path.join(output_dir, f'{name}_items.txt')
        with open(item_list_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(f"{item}\n")
        print(f"[OK] {name}: {item_list_path} ({len(items)} items)")
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, 'subset_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(subset_stats, f, indent=2, ensure_ascii=False)
    print(f"[OK] 统计信息: {stats_path}")

def main():
    # 1. 加载数据
    df = load_interactions()
    
    # 2. 计算 item popularity
    item_popularity = df['parent_asin'].value_counts()
    
    # 3. 构造嵌套子集
    subsets, item_popularity = construct_nested_subsets(df)
    
    # 4. 分析分布
    subset_stats = analyze_subset_distributions(subsets, item_popularity)
    
    # 5. 可视化对比
    plot_subset_distributions(subsets, item_popularity)
    
    # 6. 保存结果
    save_subsets(subsets, subset_stats)
    
    print("\n" + "="*60)
    print("Phase 2 完成！")
    print("="*60)
    print("\n产出文件:")
    print("  - data/subsets/S1_items.txt (5K items)")
    print("  - data/subsets/S2_items.txt (20K items)")
    print("  - data/subsets/S2.5_items.txt (50K items)")
    print("  - data/subsets/S3_items.txt (100K items)")
    print("  - data/subsets/subsets_popularity_comparison.png")
    print("  - data/subsets/subsets_overlay.png")
    print("  - data/subsets/subset_statistics.json")
    print("\n下一步：")
    print("  1. 检查 popularity 分布对比图，确认各子集形状一致")
    print("  2. 继续 Phase 3（为每个子集构造序列数据）")

if __name__ == "__main__":
    main()
