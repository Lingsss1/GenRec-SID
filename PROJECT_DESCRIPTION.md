# 生成式序列推荐系统（基于 MiniOneRec）

## 核心技术路线

基于 MiniOneRec 框架，构建了完整的生成式序列推荐 pipeline：

1. **语义 ID 构建**：利用 RQ-VAE 将商品文本嵌入离散化为 3 层语义 ID（SID）
2. **多任务 SFT**：微调 Qwen2.5-0.5B 使 LLM 根据用户交互历史直接生成目标商品 SID
3. **约束解码**：设计 Prefix Trie 约束解码保证输出合法性
4. **强化学习优化**：采用 GRPO（Group Relative Policy Optimization）进行推荐导向的强化学习训练，通过组内归一化 advantage 和 KL 散度惩罚优化生成策略

## 技术亮点

- **端到端生成**：将推荐任务建模为序列生成，直接输出商品 SID
- **多任务学习**：SID 预测 + SID↔Title 对齐 + 融合推荐三任务联合训练
- **约束解码**：基于 hash 表的高效前缀约束，保证 100% 生成合法 SID
- **GRPO 优化**：在 SFT 基础上通过强化学习进一步提升推荐准确率

## 实验设置

- **数据集**：Amazon Reviews 2023 - Cell Phones and Accessories
- **候选集规模**：5K / 20K / 50K / 100K（嵌套子集 S1⊂S2⊂S2.5⊂S3）
- **基座模型**：Qwen2.5-0.5B-Instruct（~500M 参数）
- **评估指标**：HR@10, NDCG@10

## 核心实现

### Phase 1-4：数据处理与 SID 生成
- 5-core 过滤 → 分层抽样构造嵌套子集 → 序列构造 → RQ-VAE 训练

### Phase 5：多任务 SFT
- 主任务：`history SID → target SID`
- 辅助任务 1：`SID ↔ Title` 双向对齐
- 辅助任务 2：`history SID → target Title` 融合推荐

### Phase 6：GRPO 强化学习（本次新增）
- 基于 SFT 模型进行 GRPO 训练
- 奖励函数：rule（完全匹配）/ ranking（排序感知）/ hierarchy（层级匹配）
- 约束解码：beam search + prefix hash 表
- 训练中实时评估 HR@K / NDCG@K

## 关键代码

| 文件 | 说明 |
|------|------|
| `phase5_train.py` | SFT 训练主脚本 |
| `phase6_rl_train.py` | GRPO RL 训练主脚本 |
| `grpo_trainer.py` | GRPO Trainer 实现 |
| `eval/run_eval.py` | 约束解码评估 |

## 训练流程

```bash
# SFT 训练
python phase5_train.py --config train/configs/M_N_5K.json

# RL 训练
python phase6_rl_train.py --config train/configs/RL_M_N_5K_minimal.json

# 评估
python eval/run_eval.py \
    --model_dir train/runs/RL_M_N_5K_minimal/final_checkpoint \
    --test_file data/sequences/S1/test.csv \
    --index_file data/sequences/S1/S1.index.json \
    --model_name RL_M_N_5K_minimal \
    --N 5000 --beam 20
```

## 预期效果

基于 MiniOneRec 论文，GRPO 训练后预期提升：
- HR@10: +2-5%
- NDCG@10: +3-8%
