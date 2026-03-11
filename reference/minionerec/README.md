# MiniOneRec 强化学习参考代码

本目录包含从 [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) GitHub 项目获取的 GRPO 强化学习训练相关代码，用于参考实现推荐系统的 RL 训练。

## 文件列表

| 文件 | 说明 |
|------|------|
| `rl.py` | RL 训练主脚本，使用 TRL 的 GRPOConfig 和 ReReTrainer |
| `rl.sh` | RL 训练 shell 脚本，配置 accelerate 多 GPU 启动 |
| `minionerec_trainer.py` | GRPO Trainer 实现，基于 TRL 的 GRPOTrainer 扩展，支持约束解码、beam search、动态采样等 |
| `LogitProcessor.py` | 约束解码，ConstrainedLogitsProcessor 限制生成 token 在有效候选集内 |
| `data.py` | 数据处理，包含 SidDataset、RLTitle2SidDataset、RLSeqTitle2SidDataset 等多种数据集 |
| `evaluate.py` | 评估代码，使用 beam search + 约束解码进行评估 |
| `rl_gpr.py` | GPR 版 RL 代码，增加 HEPO（Hierarchical Reward）层级奖励机制 |

## 核心组件

### 奖励函数 (Reward Types)
- **rule**: 精确匹配奖励 (0/1)
- **ranking**: rule + NDCG 排序奖励
- **ranking_only**: 仅 NDCG 奖励
- **semantic**: 基于 ADA 嵌入的余弦相似度奖励
- **sasrec**: 协同过滤模型预测分数 + HEPO 层级奖励 (rl_gpr.py)

### 训练配置
- 使用 `ReReTrainer` 继承 TRL `GRPOTrainer`
- 支持 beam search、dynamic_sampling、add_gt 等策略
- 通过 `info_file` 构建 semantic ID 的 hash 约束字典，实现约束解码

### 依赖
- trl (GRPOConfig)
- transformers
- datasets
- accelerate
- fire

## 来源
- 仓库: https://github.com/AkaliKong/MiniOneRec
- 获取时间: 2025-03
