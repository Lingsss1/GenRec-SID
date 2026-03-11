"""
Phase 3: 为每个子集构造序列数据
对 S1/S2/S2.5/S3 各执行一遍:
1. 过滤：仅保留 parent_asin ∈ S_i 的交互
2. 按 user 分组、timestamp 排序
3. 截断到最近 20 条
4. Leave-last-out 划分
5. 滑动窗口生成训练样本
6. C=10 限流（user 多样性优先）
7. 统计 N、D、C̄
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json

def load_subset_items(subset_name):
    """加载子集的 item 列表"""
    filepath = f'data/subsets/{subset_name}_items.txt'
    with open(filepath, 'r') as f:
        items = set(line.strip() for line in f)
    return items

def construct_user_sequences(df, subset_items, subset_name, max_length=20, min_length=3):
    """为指定子集构造用户序列（向量化实现）"""
    print(f"\n{'='*60}")
    print(f"处理子集: {subset_name}")
    print(f"{'='*60}")
    
    # 1. 过滤：仅保留该子集的 items
    df_subset = df[df['parent_asin'].isin(subset_items)].copy()
    print(f"\n[1] 过滤后: {len(df_subset):,} 条交互，{df_subset['user_id'].nunique():,} users")
    
    # 2-3. 向量化：按 user 分组、排序、聚合为 list
    print(f"[2-3] 构造用户序列（向量化）...")
    df_subset = df_subset.sort_values(['user_id', 'timestamp'])
    grouped = df_subset.groupby('user_id')['parent_asin'].apply(list)
    
    # 截断到最近 20 条，过滤 < min_length
    filtered_sequences = {}
    truncated_count = 0
    
    for user_id, seq in grouped.items():
        if len(seq) < min_length:
            continue
        if len(seq) > max_length:
            seq = seq[-max_length:]
            truncated_count += 1
        filtered_sequences[user_id] = seq
    
    print(f"[OK] 有效用户: {len(filtered_sequences):,}/{len(grouped):,}")
    print(f"     截断用户: {truncated_count:,} ({truncated_count/len(grouped)*100:.2f}%)")
    
    return filtered_sequences

def leave_one_out_split(sequences):
    """Leave-last-out 划分（修正：train 滑动窗口包含 valid target）"""
    print(f"\n[4-5] Leave-last-out 划分 + 滑动窗口...")
    
    train_samples = []
    valid_samples = []
    test_samples = []
    
    for user_id, items in tqdm(sequences.items(), desc="划分数据"):
        n = len(items)
        
        if n < 3:
            continue
        
        # Valid: input=前 n-2, target=倒数第 2 个
        valid_samples.append({
            'user_id': user_id,
            'history': items[:-2],
            'target': items[-2],
            'seq_length': len(items[:-2])
        })
        
        # Test: input=前 n-1, target=最后 1 个
        test_samples.append({
            'user_id': user_id,
            'history': items[:-1],
            'target': items[-1],
            'seq_length': len(items[:-1])
        })
        
        # Train: 滑动窗口覆盖到 items[:-1]（包含 valid target）
        # 避免 train/valid 的 context 长度 gap
        train_items_for_sliding = items[:-1]  
        
        for i in range(1, len(train_items_for_sliding)):
            train_samples.append({
                'user_id': user_id,
                'history': train_items_for_sliding[:i],
                'target': train_items_for_sliding[i],
                'context_length': i
            })
    
    print(f"[OK] Train: {len(train_samples):,} 样本")
    print(f"     Valid: {len(valid_samples):,} 样本")
    print(f"     Test:  {len(test_samples):,} 样本")
    
    return train_samples, valid_samples, test_samples

def apply_c10_limit(train_samples):
    """应用 C=10 限流（主实验策略 - 简化版：user 多样性优先）"""
    print(f"\n[6] 应用 C=10 限流（user 多样性优先）...")
    print(f"   原始训练样本: {len(train_samples):,}")
    
    # 按 target_item 聚合
    item_samples = defaultdict(list)
    for sample in train_samples:
        item_samples[sample['target']].append(sample)
    
    print(f"   涉及 target items: {len(item_samples):,}")
    
    # 对每个 item 应用限流
    limited_samples = []
    
    for item, samples in tqdm(item_samples.items(), desc="限流处理"):
        if len(samples) <= 10:
            # 样本数 <= 10，全部保留
            limited_samples.extend(samples)
        else:
            # 样本数 > 10，优先选择不同 user 的样本
            # 策略：按 user 出现次数排序，优先选低频 user
            
            # 统计每个 user 在该 item 的样本数
            user_sample_count = defaultdict(int)
            for sample in samples:
                user_sample_count[sample['user_id']] += 1
            
            # 按 user 出现次数升序排序（低频 user 优先）
            # 次要：按 context_length 升序（覆盖不同长度）
            samples.sort(key=lambda s: (user_sample_count[s['user_id']], s['context_length']))
            
            # 贪心选择：优先选不同 user
            selected = []
            user_seen = set()
            
            for sample in samples:
                if sample['user_id'] not in user_seen:
                    user_seen.add(sample['user_id'])
                    selected.append(sample)
                    if len(selected) == 10:
                        break
            
            # 如果不同 user 数 < 10，再从重复 user 中补充
            if len(selected) < 10:
                for sample in samples:
                    if sample not in selected:
                        selected.append(sample)
                        if len(selected) == 10:
                            break
            
            limited_samples.extend(selected)
    
    print(f"[OK] 限流后训练样本: {len(limited_samples):,}")
    
    # 统计实际 C̄
    item_counts_after = defaultdict(int)
    for sample in limited_samples:
        item_counts_after[sample['target']] += 1
    
    c_values = list(item_counts_after.values())
    c_mean = np.mean(c_values)
    c_median = np.median(c_values)
    c_std = np.std(c_values)
    
    print(f"\n   实际每 item 样本数统计:")
    print(f"     平均 C_bar = {c_mean:.2f}")
    print(f"     中位数 = {c_median:.1f}")
    print(f"     标准差 = {c_std:.2f}")
    print(f"     范围 = [{min(c_values)}, {max(c_values)}]")
    
    return limited_samples, c_mean

def compute_statistics(train_samples, valid_samples, test_samples, subset_name, c_mean_from_limit):
    """计算统计信息（复用 C_bar）"""
    print(f"\n[7] 统计 {subset_name} 的指标...")
    
    # 实际 N（Train 中出现的 unique target items）
    train_items = set(sample['target'] for sample in train_samples)
    
    # Valid/Test 中的 target items
    valid_items = set(sample['target'] for sample in valid_samples)
    test_items = set(sample['target'] for sample in test_samples)
    
    # Test coverage（test target 在 train 中出现的比例）
    test_in_train = test_items & train_items
    test_coverage = len(test_in_train) / len(test_items) if test_items else 0
    
    # 训练样本数 D
    D = len(train_samples)
    
    # 平均 context 长度
    avg_context_len = np.mean([s['context_length'] for s in train_samples])
    
    # Valid/Test 平均 input 长度
    avg_valid_len = np.mean([s['seq_length'] for s in valid_samples])
    avg_test_len = np.mean([s['seq_length'] for s in test_samples])
    
    stats = {
        'subset_name': subset_name,
        'N_train': len(train_items),
        'D_train': D,
        'C_mean': float(c_mean_from_limit),
        'avg_context_length': float(avg_context_len),
        'n_valid': len(valid_samples),
        'n_test': len(test_samples),
        'avg_valid_input_length': float(avg_valid_len),
        'avg_test_input_length': float(avg_test_len),
        'test_coverage_in_train': float(test_coverage),
        'n_test_items': len(test_items),
        'n_test_items_in_train': len(test_in_train),
    }
    
    print(f"\n{subset_name} 统计:")
    print(f"  实际 N（Train 中出现的 items）: {stats['N_train']:,}")
    print(f"  训练样本数 D: {stats['D_train']:,}")
    print(f"  实际平均 C_bar: {stats['C_mean']:.2f}")
    print(f"  平均 context 长度: {stats['avg_context_length']:.2f}")
    print(f"  Valid 样本数: {stats['n_valid']:,}（平均 input={stats['avg_valid_input_length']:.2f}）")
    print(f"  Test 样本数: {stats['n_test']:,}（平均 input={stats['avg_test_input_length']:.2f}）")
    print(f"  Test coverage: {stats['test_coverage_in_train']*100:.2f}% ({stats['n_test_items_in_train']}/{stats['n_test_items']})")
    
    return stats

def save_sequence_data(train_samples, valid_samples, test_samples, stats, subset_name):
    """保存序列数据（只保存字符串格式的 history）"""
    print(f"\n[8] 保存 {subset_name} 数据...")
    
    output_dir = f'data/sequences/{subset_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 转为 DataFrame，history 转为空格分隔字符串
    train_data = []
    for s in train_samples:
        train_data.append({
            'user_id': s['user_id'],
            'history': ' '.join(s['history']),
            'target': s['target'],
            'context_length': s['context_length']
        })
    
    valid_data = []
    for s in valid_samples:
        valid_data.append({
            'user_id': s['user_id'],
            'history': ' '.join(s['history']),
            'target': s['target'],
            'seq_length': s['seq_length']
        })
    
    test_data = []
    for s in test_samples:
        test_data.append({
            'user_id': s['user_id'],
            'history': ' '.join(s['history']),
            'target': s['target'],
            'seq_length': s['seq_length']
        })
    
    df_train = pd.DataFrame(train_data)
    df_valid = pd.DataFrame(valid_data)
    df_test = pd.DataFrame(test_data)
    
    # 保存
    train_path = os.path.join(output_dir, 'train.parquet')
    valid_path = os.path.join(output_dir, 'valid.parquet')
    test_path = os.path.join(output_dir, 'test.parquet')
    
    df_train.to_parquet(train_path, index=False)
    df_valid.to_parquet(valid_path, index=False)
    df_test.to_parquet(test_path, index=False)
    
    print(f"[OK] Train: {train_path}")
    print(f"[OK] Valid: {valid_path}")
    print(f"[OK] Test:  {test_path}")
    
    # 保存统计
    stats_path = os.path.join(output_dir, 'statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[OK] 统计: {stats_path}")

def process_one_subset(df, subset_name, subset_items):
    """处理一个子集的完整流程"""
    print(f"\n\n{'#'*60}")
    print(f"# 开始处理: {subset_name} ({len(subset_items):,} items)")
    print(f"{'#'*60}")
    
    # 1-3. 构造序列（过滤、分组、截断）
    sequences = construct_user_sequences(df, subset_items, subset_name)
    
    # 4-5. Leave-one-out 划分 + 滑动窗口
    train_samples, valid_samples, test_samples = leave_one_out_split(sequences)
    
    # 6. C=10 限流
    train_samples, c_mean = apply_c10_limit(train_samples)
    
    # 7. 统计（复用 c_mean）
    stats = compute_statistics(train_samples, valid_samples, test_samples, subset_name, c_mean)
    
    # 8. 保存
    save_sequence_data(train_samples, valid_samples, test_samples, stats, subset_name)
    
    return stats

def generate_summary_table(all_stats):
    """生成所有子集的汇总表"""
    print(f"\n{'='*60}")
    print("所有子集统计汇总")
    print(f"{'='*60}")
    
    # 打印表格
    print(f"\n{'子集':<8} {'N(items)':<12} {'D(samples)':<12} {'C_bar':<8} {'Valid':<10} {'Test':<10} {'Coverage':<10}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['subset_name']:<8} "
              f"{stats['N_train']:<12,} "
              f"{stats['D_train']:<12,} "
              f"{stats['C_mean']:<8.2f} "
              f"{stats['n_valid']:<10,} "
              f"{stats['n_test']:<10,} "
              f"{stats['test_coverage_in_train']*100:<9.1f}%")
    
    # 保存汇总
    summary_path = 'data/sequences/summary_statistics.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 汇总统计已保存: {summary_path}")

def main():
    print("="*60)
    print("Phase 3: 序列数据构造（对 S1/S2/S2.5/S3 各执行一遍）")
    print("="*60)
    
    # 加载去重后的交互数据
    print("\n加载去重后的交互数据...")
    df = pd.read_parquet('data/raw/interactions_5core_dedup.parquet')
    print(f"[OK] 加载了 {len(df):,} 条交互")
    
    # 确保按时间排序
    df = df.sort_values(['user_id', 'timestamp'])
    
    # 加载各子集
    subsets = {
        'S1': load_subset_items('S1'),
        'S2': load_subset_items('S2'),
        'S2.5': load_subset_items('S2.5'),
        'S3': load_subset_items('S3'),
    }
    
    print(f"\n子集规模:")
    for name in ['S1', 'S2', 'S2.5', 'S3']:
        print(f"  {name}: {len(subsets[name]):,} items")
    
    # 处理每个子集
    all_stats = []
    
    for name in ['S1', 'S2', 'S2.5', 'S3']:
        stats = process_one_subset(df, name, subsets[name])
        all_stats.append(stats)
    
    # 生成汇总表
    generate_summary_table(all_stats)
    
    print("\n" + "="*60)
    print("Phase 3 完成！")
    print("="*60)
    print("\n产出:")
    print("  - data/sequences/S1/train.parquet, valid.parquet, test.parquet")
    print("  - data/sequences/S2/train.parquet, valid.parquet, test.parquet")
    print("  - data/sequences/S2.5/train.parquet, valid.parquet, test.parquet")
    print("  - data/sequences/S3/train.parquet, valid.parquet, test.parquet")
    print("  - data/sequences/summary_statistics.json")
    print("\n下一步:")
    print("  继续 Phase 4（文本表征与 RQ-VAE 离散化）")

if __name__ == "__main__":
    main()
