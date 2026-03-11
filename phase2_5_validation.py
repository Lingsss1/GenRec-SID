"""
Phase 2.5: 数据质量验证
在开始序列构造前，验证关键数据质量问题：
1. 元数据覆盖率（重试加载）
2. 时间戳格式确认
3. 重复交互检测与去重策略
4. 用户序列长度分布
5. Leave-one-out 切分策略验证
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from tqdm import tqdm
import json

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def retry_load_metadata():
    """重试加载元数据"""
    print("="*60)
    print("验证 1: 元数据覆盖率")
    print("="*60)
    
    try:
        print("\n重新尝试加载 item 元数据...")
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_Cell_Phones_and_Accessories",
            split="full",
            trust_remote_code=True
        )
        
        print(f"[OK] 成功加载 {len(meta_dataset)} 条元数据")
        
        meta_data = []
        for item in tqdm(meta_dataset, desc="提取元数据字段"):
            meta_data.append({
                'parent_asin': item.get('parent_asin'),
                'title': item.get('title', ''),
                'store': item.get('store', ''),
                'categories': str(item.get('categories', [])),
                'main_category': item.get('main_category', ''),
            })
        
        df_meta = pd.DataFrame(meta_data)
        
        # 保存
        meta_path = 'data/raw/item_metadata.parquet'
        df_meta.to_parquet(meta_path, index=False)
        print(f"[OK] 元数据已保存: {meta_path}")
        
        return df_meta
        
    except Exception as e:
        print(f"[WARN] 元数据加载仍失败: {e}")
        return None

def check_metadata_coverage(df_meta):
    """检查元数据覆盖率"""
    if df_meta is None:
        print("[WARN] 无元数据，跳过覆盖率检查")
        return
    
    # 加载 S3 items
    with open('data/subsets/S3_items.txt', 'r') as f:
        s3_items = set(line.strip() for line in f)
    
    print(f"\nS3 候选集: {len(s3_items)} items")
    print(f"元数据总数: {len(df_meta)} items")
    
    # 计算覆盖率
    meta_items = set(df_meta['parent_asin'].dropna())
    covered = s3_items & meta_items
    coverage_rate = len(covered) / len(s3_items)
    
    print(f"\n覆盖率分析:")
    print(f"  S3 中有元数据的 items: {len(covered)} / {len(s3_items)}")
    print(f"  覆盖率: {coverage_rate*100:.2f}%")
    
    if coverage_rate < 0.95:
        print(f"  [WARN] 覆盖率 < 95%，可能影响语义特征质量")
    else:
        print(f"  [OK] 覆盖率 >= 95%，满足要求")
    
    # 检查 title 缺失率
    df_meta_s3 = df_meta[df_meta['parent_asin'].isin(s3_items)]
    title_missing = df_meta_s3['title'].isna().sum() + (df_meta_s3['title'] == '').sum()
    title_missing_rate = title_missing / len(df_meta_s3)
    
    print(f"\nTitle 字段质量:")
    print(f"  缺失/空值: {title_missing} / {len(df_meta_s3)}")
    print(f"  缺失率: {title_missing_rate*100:.2f}%")
    
    if title_missing_rate > 0.05:
        print(f"  [WARN] Title 缺失率 > 5%")
    else:
        print(f"  [OK] Title 缺失率 < 5%")
    
    return coverage_rate, title_missing_rate

def check_timestamp_format():
    """确认时间戳格式"""
    print(f"\n{'='*60}")
    print("验证 2: 时间戳格式")
    print(f"{'='*60}")
    
    df = pd.read_parquet('data/raw/interactions_5core.parquet')
    
    # 采样检查
    sample_timestamps = df['timestamp'].head(10).values
    
    print(f"\n前 10 条时间戳样本:")
    for i, ts in enumerate(sample_timestamps[:5], 1):
        print(f"  {i}. {ts}")
    
    # 判断是秒还是毫秒
    # Unix 秒时间戳范围: 1996-05 ≈ 8.3亿, 2023-09 ≈ 16.9亿
    # Unix 毫秒时间戳范围: 1996-05 ≈ 8300亿, 2023-09 ≈ 16900亿
    
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    
    print(f"\n时间戳范围:")
    print(f"  最小: {min_ts}")
    print(f"  最大: {max_ts}")
    
    if max_ts > 1e12:  # > 1 trillion，肯定是毫秒
        ts_format = "milliseconds"
        print(f"[OK] 时间戳格式: Unix 毫秒（符合 Amazon 2023 规范）")
    else:
        ts_format = "seconds"
        print(f"[WARN] 时间戳格式: Unix 秒（需要转换）")
    
    # 转换为可读日期验证
    if ts_format == "milliseconds":
        sample_date = pd.to_datetime(min_ts, unit='ms')
    else:
        sample_date = pd.to_datetime(min_ts, unit='s')
    
    print(f"  最早日期示例: {sample_date}")
    
    return ts_format

def check_duplicate_interactions():
    """检测重复交互"""
    print(f"\n{'='*60}")
    print("验证 3: 重复交互检测")
    print(f"{'='*60}")
    
    df = pd.read_parquet('data/raw/interactions_5core.parquet')
    
    original_len = len(df)
    
    # 检测 (user_id, parent_asin) 重复
    duplicates = df.duplicated(subset=['user_id', 'parent_asin'], keep=False)
    n_duplicates = duplicates.sum()
    
    print(f"\n重复交互统计:")
    print(f"  总交互数: {original_len:,}")
    print(f"  重复交互数: {n_duplicates:,}")
    print(f"  重复率: {n_duplicates/original_len*100:.2f}%")
    
    if n_duplicates > 0:
        # 分析重复模式
        dup_df = df[duplicates].copy()
        dup_groups = dup_df.groupby(['user_id', 'parent_asin']).size()
        
        print(f"\n重复模式分析:")
        print(f"  重复的 (user, item) 对数: {len(dup_groups):,}")
        print(f"  最多重复次数: {dup_groups.max()}")
        print(f"  平均重复次数: {dup_groups.mean():.2f}")
        
        print(f"\n建议去重策略:")
        print(f"  [推荐] 保留最早一条（first）- 反映用户首次接触")
        print(f"  [备选] 保留最晚一条（last）- 反映用户最新偏好")
        
        # 执行去重（保留最早）
        df_dedup = df.sort_values('timestamp').drop_duplicates(
            subset=['user_id', 'parent_asin'], 
            keep='first'
        )
        
        print(f"\n去重后:")
        print(f"  剩余交互数: {len(df_dedup):,}")
        print(f"  删除了: {original_len - len(df_dedup):,} 条")
        
        # 保存去重后数据
        dedup_path = 'data/raw/interactions_5core_dedup.parquet'
        df_dedup.to_parquet(dedup_path, index=False)
        print(f"[OK] 去重数据已保存: {dedup_path}")
        
        return df_dedup, n_duplicates / original_len
    else:
        print("[OK] 无重复交互")
        return df, 0.0

def analyze_user_sequence_lengths(df):
    """分析用户序列长度分布"""
    print(f"\n{'='*60}")
    print("验证 4: 用户序列长度分布")
    print(f"{'='*60}")
    
    user_seq_lengths = df.groupby('user_id').size()
    
    print(f"\n序列长度统计:")
    print(f"  平均长度: {user_seq_lengths.mean():.2f}")
    print(f"  中位数长度: {user_seq_lengths.median():.0f}")
    print(f"  最小长度: {user_seq_lengths.min()}")
    print(f"  最大长度: {user_seq_lengths.max()}")
    
    # 各长度段的用户分布
    length_bins = [0, 3, 5, 10, 20, 50, 100, float('inf')]
    length_labels = ['<3', '3-5', '6-10', '11-20', '21-50', '51-100', '>100']
    
    print(f"\n序列长度分布:")
    for i in range(len(length_bins) - 1):
        count = ((user_seq_lengths > length_bins[i]) & (user_seq_lengths <= length_bins[i+1])).sum()
        pct = count / len(user_seq_lengths) * 100
        print(f"  {length_labels[i]:>8s}: {count:7,} users ({pct:5.2f}%)")
    
    # 建议过滤策略
    print(f"\nLeave-last-out 切分要求:")
    print(f"  - 序列长度 >= 3: train(>=1) + val(1) + test(1)")
    print(f"  - 当前 >=3 的用户: {(user_seq_lengths >= 3).sum():,} ({(user_seq_lengths >= 3).sum()/len(user_seq_lengths)*100:.2f}%)")
    print(f"  [建议] 过滤序列长度 < 3 的用户")
    
    # 如果使用截断 20 条
    truncated = user_seq_lengths.clip(upper=20)
    print(f"\n截断到 20 条后:")
    print(f"  平均有效长度: {truncated.mean():.2f}")
    print(f"  需要截断的用户数: {(user_seq_lengths > 20).sum():,} ({(user_seq_lengths > 20).sum()/len(user_seq_lengths)*100:.2f}%)")
    
    return user_seq_lengths

def visualize_timestamp_and_sequence():
    """可视化时间戳与序列长度"""
    print(f"\n绘制时间与序列分布图...")
    
    df = pd.read_parquet('data/raw/interactions_5core_dedup.parquet' 
                         if os.path.exists('data/raw/interactions_5core_dedup.parquet')
                         else 'data/raw/interactions_5core.parquet')
    
    user_seq_lengths = df.groupby('user_id').size()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图 1: 时间戳分布
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['year'] = df['datetime'].dt.year
    year_counts = df['year'].value_counts().sort_index()
    
    axes[0].bar(year_counts.index, year_counts.values, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Number of Interactions', fontsize=12)
    axes[0].set_title('Interaction Distribution by Year', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 子图 2: 用户序列长度分布
    axes[1].hist(user_seq_lengths.values, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=3, color='red', linestyle='--', linewidth=2, label='Min length=3')
    axes[1].axvline(x=20, color='orange', linestyle='--', linewidth=2, label='Truncate at 20')
    axes[1].set_xlabel('User Sequence Length', fontsize=12)
    axes[1].set_ylabel('Number of Users', fontsize=12)
    axes[1].set_title('User Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = 'data/raw/timestamp_and_sequence_validation.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 验证图已保存: {plot_path}")

def demonstrate_leave_one_out_split():
    """演示 leave-one-out 切分逻辑"""
    print(f"\n{'='*60}")
    print("验证 5: Leave-one-out 切分策略")
    print(f"{'='*60}")
    
    print("\n标准 leave-last-out 切分:")
    print("  全序列: [i1, i2, ..., i_{n-1}, i_n]  (按 timestamp 升序)")
    print("  ")
    print("  Train 部分: [i1, i2, ..., i_{n-2}]")
    print("    → 用滑动窗口生成训练样本:")
    print("       ([i1], i2), ([i1,i2], i3), ..., ([i1,...,i_{n-3}], i_{n-2})")
    print("  ")
    print("  Valid 样本: input=[i1, ..., i_{n-2}], target=i_{n-1}")
    print("  Test 样本:  input=[i1, ..., i_{n-1}], target=i_n")
    print("  ")
    print("关键: val 和 test 的输入序列长度不同")
    print("      val input 长度 = n-2")
    print("      test input 长度 = n-1")
    
    # 实际示例
    df = pd.read_parquet('data/raw/interactions_5core_dedup.parquet'
                         if os.path.exists('data/raw/interactions_5core_dedup.parquet')
                         else 'data/raw/interactions_5core.parquet')
    
    # 找一个序列长度适中的用户
    user_seq_lengths = df.groupby('user_id').size()
    target_users = user_seq_lengths[(user_seq_lengths >= 8) & (user_seq_lengths <= 12)].head(3)
    
    print(f"\n实际示例（展示 3 个用户）:")
    for user_id in target_users.index:
        user_df = df[df['user_id'] == user_id].sort_values('timestamp')
        items = user_df['parent_asin'].tolist()
        n = len(items)
        
        print(f"\n  User: {user_id[:20]}... (序列长度={n})")
        print(f"    完整序列: {items[:3]}...{items[-2:]}")
        print(f"    Train: {items[:-2][:3]}... ({n-2} items)")
        print(f"    Valid: input={items[:-2][:3]}...{items[-3:-2]}, target={items[-2]}")
        print(f"    Test:  input={items[:-1][:3]}...{items[-2:-1]}, target={items[-1]}")

def generate_validation_summary():
    """生成验证摘要报告"""
    print(f"\n{'='*60}")
    print("数据质量验证摘要")
    print(f"{'='*60}")
    
    # 加载验证结果
    summary = {
        "phase": "Phase 2.5 - Data Quality Validation",
        "checks": []
    }
    
    # 读取已有统计
    with open('data/raw/data_statistics.json', 'r') as f:
        stats = json.load(f)
    
    summary['checks'].append({
        "name": "数据规模",
        "status": "PASS",
        "details": f"{stats['num_users']:,} users, {stats['num_items']:,} items, {stats['num_interactions']:,} interactions"
    })
    
    # 保存摘要
    summary_path = 'data/raw/validation_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 验证摘要已保存: {summary_path}")

def main():
    print("="*60)
    print("Phase 2.5: 数据质量验证")
    print("="*60)
    
    # 1. 重试加载元数据
    df_meta = retry_load_metadata()
    
    # 2. 检查元数据覆盖率
    if df_meta is not None:
        check_metadata_coverage(df_meta)
    
    # 3. 确认时间戳格式
    ts_format = check_timestamp_format()
    
    # 4. 检测并去重
    df_dedup, dup_rate = check_duplicate_interactions()
    
    # 5. 分析序列长度
    user_seq_lengths = analyze_user_sequence_lengths(df_dedup)
    
    # 6. 可视化
    visualize_timestamp_and_sequence()
    
    # 7. 演示切分策略
    demonstrate_leave_one_out_split()
    
    # 8. 生成摘要
    generate_validation_summary()
    
    print("\n" + "="*60)
    print("Phase 2.5 完成！")
    print("="*60)
    print("\nPhase 3 准备就绪，确认以下事项后继续:")
    print("  1. 元数据覆盖率是否满足要求（>95%）")
    print("  2. 时间戳格式已确认（Unix 毫秒）")
    print("  3. 去重策略：保留 (user, item) 对的最早一条交互")
    print("  4. 序列过滤：保留长度 >= 3 的用户")
    print("  5. 截断策略：每用户最多保留最近 20 条交互")

if __name__ == "__main__":
    main()
