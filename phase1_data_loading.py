"""
Phase 1: 数据载入与基本清洗
- 使用 HuggingFace datasets 加载 Cell_Phones_and_Accessories
- 提取 user_id, parent_asin, timestamp
- 执行 5-core 过滤
- 统计基本信息并验证数据规模
- 生成 item popularity 分布直方图（log-log scale）
- 加载 item 元数据（title, store, category）
- 保存为 Parquet 格式
"""

import os
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

def load_review_data():
    """加载 review 数据"""
    print("正在加载 Amazon Reviews 2023 - Cell_Phones_and_Accessories...")
    print("(首次加载会从 HuggingFace 下载，约 1-2GB，请耐心等待)")
    
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Cell_Phones_and_Accessories",
        trust_remote_code=True
    )
    
    print(f"[OK] 成功加载 review 数据")
    print(f"  - 数据集包含 split: {list(dataset.keys())}")
    print(f"  - full split 记录数: {len(dataset['full'])}")
    
    return dataset

def extract_interactions(dataset):
    """提取交互数据（user_id, parent_asin, timestamp）"""
    print("\n正在提取交互数据...")
    
    data = dataset['full']
    interactions = []
    
    for item in tqdm(data, desc="提取字段"):
        interactions.append({
            'user_id': item['user_id'],
            'parent_asin': item['parent_asin'],
            'timestamp': item['timestamp'],
        })
    
    df = pd.DataFrame(interactions)
    
    # 删除缺失值
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len:
        print(f"[WARN] 删除了 {original_len - len(df)} 条缺失值记录")
    
    print(f"[OK] 提取完成：{len(df)} 条交互记录")
    return df

def compute_5core(df):
    """执行 5-core 过滤"""
    print("\n执行 5-core 过滤...")
    print(f"  原始数据: {df['user_id'].nunique()} users, {df['parent_asin'].nunique()} items, {len(df)} 交互")
    
    max_iterations = 20
    for iteration in range(max_iterations):
        prev_size = len(df)
        prev_users = df['user_id'].nunique()
        prev_items = df['parent_asin'].nunique()
        
        # 过滤 user（至少 5 条交互）
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        df = df[df['user_id'].isin(valid_users)]
        
        # 过滤 item（至少 5 条交互）
        item_counts = df['parent_asin'].value_counts()
        valid_items = item_counts[item_counts >= 5].index
        df = df[df['parent_asin'].isin(valid_items)]
        
        if len(df) == prev_size:
            print(f"[OK] 5-core 收敛于第 {iteration+1} 轮")
            break
        
        print(f"  迭代 {iteration+1}: {df['user_id'].nunique()} users, {df['parent_asin'].nunique()} items, {len(df)} 交互")
    
    print(f"\n[OK] 5-core 过滤完成:")
    print(f"  - {df['user_id'].nunique()} users")
    print(f"  - {df['parent_asin'].nunique()} items")
    print(f"  - {len(df)} 交互")
    
    # 验证 5-core 条件
    user_counts = df['user_id'].value_counts()
    item_counts = df['parent_asin'].value_counts()
    assert user_counts.min() >= 5, "User 5-core 条件未满足"
    assert item_counts.min() >= 5, "Item 5-core 条件未满足"
    print(f"[OK] 验证通过: 最少 user 交互数={user_counts.min()}, 最少 item 交互数={item_counts.min()}")
    
    return df

def analyze_and_plot_popularity(df, output_dir='data/raw'):
    """统计并绘制 item popularity 分布（log-log scale）"""
    print("\n统计 item popularity 分布...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算 popularity
    item_popularity = df['parent_asin'].value_counts()
    
    # 基本统计
    stats = {
        'num_users': int(df['user_id'].nunique()),
        'num_items': int(df['parent_asin'].nunique()),
        'num_interactions': int(len(df)),
        'avg_interactions_per_user': float(len(df) / df['user_id'].nunique()),
        'avg_interactions_per_item': float(len(df) / df['parent_asin'].nunique()),
        'median_item_popularity': float(item_popularity.median()),
        'mean_item_popularity': float(item_popularity.mean()),
        'std_item_popularity': float(item_popularity.std()),
        'min_item_popularity': int(item_popularity.min()),
        'max_item_popularity': int(item_popularity.max()),
    }
    
    print("\n" + "="*60)
    print("数据集基本统计")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value:,}")
    print("="*60)
    
    # Popularity 分布分析
    popularity_values = item_popularity.values
    
    # 检查长尾分布
    p99 = np.percentile(popularity_values, 99)
    p90 = np.percentile(popularity_values, 90)
    p50 = np.percentile(popularity_values, 50)
    
    top1_count = (popularity_values >= p99).sum()
    top10_count = (popularity_values >= p90).sum()
    tail50_count = (popularity_values <= p50).sum()
    
    print(f"\nPopularity 分布特征:")
    print(f"  Top 1% items (>= {p99:.0f} 交互): {top1_count} items")
    print(f"  Top 10% items (>= {p90:.0f} 交互): {top10_count} items")
    print(f"  Tail 50% items (<= {p50:.0f} 交互): {tail50_count} items")
    
    # 绘制 log-log 分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图 1: Rank vs Popularity (log-log)
    sorted_popularity = np.sort(popularity_values)[::-1]
    ranks = np.arange(1, len(sorted_popularity) + 1)
    
    axes[0].loglog(ranks, sorted_popularity, 'b-', alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Item Rank (log scale)', fontsize=12)
    axes[0].set_ylabel('Popularity (log scale)', fontsize=12)
    axes[0].set_title('Item Popularity Distribution (Rank-Frequency)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 子图 2: Histogram (log bins)
    log_bins = np.logspace(np.log10(popularity_values.min()), 
                           np.log10(popularity_values.max()), 
                           50)
    axes[1].hist(popularity_values, bins=log_bins, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Popularity (log scale)', fontsize=12)
    axes[1].set_ylabel('Number of Items (log scale)', fontsize=12)
    axes[1].set_title('Item Popularity Histogram', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'item_popularity_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 直方图已保存至: {plot_path}")
    
    # 判断是否适合分层抽样
    gini = compute_gini(popularity_values)
    print(f"\n长尾分布评估:")
    print(f"  Gini 系数: {gini:.3f}")
    if gini > 0.6:
        print(f"  [OK] 强长尾分布（Gini > 0.6），适合分层抽样")
    elif gini > 0.4:
        print(f"  [WARN] 中等长尾分布（0.4 < Gini < 0.6），分层抽样仍有意义")
    else:
        print(f"  [ERROR] 分布较均匀（Gini < 0.4），分层抽样意义不大")
    
    return stats, item_popularity

def compute_gini(values):
    """计算基尼系数（衡量不平等程度）"""
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

def load_item_metadata():
    """加载 item 元数据"""
    print("\n正在加载 item 元数据...")
    
    try:
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_Cell_Phones_and_Accessories",
            split="full",
            trust_remote_code=True
        )
        
        print(f"[OK] 加载了 {len(meta_dataset)} 条元数据")
        
        meta_data = []
        for item in tqdm(meta_dataset, desc="提取元数据"):
            meta_data.append({
                'parent_asin': item.get('parent_asin'),
                'title': item.get('title', ''),
                'store': item.get('store', ''),
                'categories': str(item.get('categories', [])),
                'average_rating': item.get('average_rating', 0),
                'rating_number': item.get('rating_number', 0),
                'main_category': item.get('main_category', ''),
            })
        
        df_meta = pd.DataFrame(meta_data)
        print(f"[OK] 处理元数据：{len(df_meta)} 个 items")
        return df_meta
        
    except Exception as e:
        print(f"[WARN] 元数据加载失败: {e}")
        print("将在后续需要时重新尝试")
        return None

def save_cleaned_data(df_interactions, df_meta, stats, output_dir='data/raw'):
    """保存清洗后的数据"""
    print("\n保存数据...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存交互数据
    interactions_path = os.path.join(output_dir, 'interactions_5core.parquet')
    df_interactions.to_parquet(interactions_path, index=False)
    print(f"[OK] 交互数据: {interactions_path}")
    
    # 保存元数据
    if df_meta is not None:
        meta_path = os.path.join(output_dir, 'item_metadata.parquet')
        df_meta.to_parquet(meta_path, index=False)
        print(f"[OK] 元数据: {meta_path}")
    
    # 保存统计信息
    stats_path = os.path.join(output_dir, 'data_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[OK] 统计信息: {stats_path}")

def main():
    print("="*60)
    print("Phase 1: 数据载入与基本清洗")
    print("="*60)
    
    # 1. 加载 review 数据
    dataset = load_review_data()
    
    # 2. 提取交互
    df_interactions = extract_interactions(dataset)
    
    # 3. 5-core 过滤
    df_interactions = compute_5core(df_interactions)
    
    # 4. 统计与可视化
    stats, item_popularity = analyze_and_plot_popularity(df_interactions)
    
    # 5. 加载元数据
    df_meta = load_item_metadata()
    
    # 6. 保存结果
    save_cleaned_data(df_interactions, df_meta, stats)
    
    print("\n" + "="*60)
    print("Phase 1 完成！")
    print("="*60)
    print("\n下一步：")
    print("  1. 检查 data/raw/item_popularity_distribution.png 验证长尾分布")
    print("  2. 确认数据规模是否符合预期（~381K users, 111.5K items, 2.8M reviews）")
    print("  3. 如验证通过，继续 Phase 2（嵌套候选集构造）")

if __name__ == "__main__":
    main()
