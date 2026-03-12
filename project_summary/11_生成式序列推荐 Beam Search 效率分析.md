# 生成式序列推荐 Beam Search 效率分析

---

## 版本一：完整版（约 300 字）

### 生成式序列推荐系统的 Beam Search 效率分析与候选集可扩展性研究

**技术栈**：Qwen2.5-0.5B / RQ-VAE / FAISS / Prefix Trie 约束解码 / Beam Search / SciPy 幂律拟合 / PyTorch

- 基于 Qwen2.5-0.5B + RQ-VAE 构建端到端生成式序列推荐管线，通过 3 层残差量化将商品嵌入离散化为语义 ID（SID），设计 Prefix Trie 约束 Beam Search 解码保证输出合法性，在 Amazon Reviews 2023（Cell Phones，100K+ 商品）上完成系统搭建与评估
- 设计嵌套候选集实验（5K⊂20K⊂50K⊂100K）× 3 档 Beam Size（20/50/100），共 12 组实验，系统分析了 Beam Search 在不同候选集规模下的推荐精度与计算效率的权衡关系；发现 Beam=50 为最优性价比点——相比 Beam=20 平均提升 HR@10 约 15%，而 Beam=100 仅在 Beam=50 基础上额外提升 3%~5%，边际收益显著递减
- 量化了候选集大小 N 对 Beam Search 效果的影响规律：HR@10(N) = αN^(-β) + γ（R²=0.995），衰减指数 β 在 3 个 Beam 档位下高度一致（均值 0.64，标准差 0.047），证明候选集规模是制约生成式推荐性能的核心瓶颈，且该瓶颈无法通过增大 Beam Size 缓解
- 基于拟合结果反推业务可行域：当前 0.5B 模型 + Beam=50 配置下，候选集上限约 6K 可满足 HR@10≥0.04 的召回要求；提出分层架构方案（传统召回截断至 20K + 生成式 Beam Search 精排）作为工程落地路径

---

## 版本二：精简版（约 200 字）

### 生成式序列推荐 — Beam Search 效率分析

**技术栈**：Qwen2.5-0.5B / RQ-VAE / Prefix Trie / Beam Search / 幂律拟合

- 构建端到端生成式推荐管线（RQ-VAE 语义 ID → 多任务 SFT 微调 Qwen2.5 → Prefix Trie 约束 Beam Search 解码），在 Amazon 100K 商品数据上完成系统搭建
- 设计 4 候选集规模 × 3 Beam Size 共 12 组实验，系统分析 Beam Search 精度-效率权衡：Beam=50 为最优性价比点（相比 Beam=20 提升 15%，Beam=100 仅额外提升 3%~5%，边际收益递减）
- 量化候选集规模对 Beam Search 的影响：HR@10 ∝ N^(-0.64)，R²=0.995；衰减指数跨 Beam 一致（std=0.047），证明候选集规模是核心瓶颈且无法通过增大 Beam 缓解
- 反推业务可行域（Beam=50 下候选集上限约 6K），提出传统召回 + 生成式 Beam Search 精排的分层落地方案

---
