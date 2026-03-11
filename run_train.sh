#!/bin/bash
# 一键训练 M_N_50K 和 M_N_100K
# 用法: bash run_train.sh [可选: nohup]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================="
echo "Phase 5 Server Training"
echo "=============================="
echo "工作目录: $SCRIPT_DIR"
echo "开始时间: $(date)"
echo ""

# 检查 GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
echo ""

# 安装依赖（如果需要）
# pip install -r requirements.txt

# ==============================
# 实验 1: M_N_50K
# ==============================
echo "=============================="
echo "开始训练: M_N_50K"
echo "时间: $(date)"
echo "=============================="

python phase5_train.py --config train/configs/M_N_50K.json

echo ""
echo "✅ M_N_50K 训练完成: $(date)"
echo ""

# ==============================
# 实验 2: M_N_100K
# ==============================
echo "=============================="
echo "开始训练: M_N_100K"
echo "时间: $(date)"
echo "=============================="

python phase5_train.py --config train/configs/M_N_100K.json

echo ""
echo "✅ M_N_100K 训练完成: $(date)"
echo ""

echo "=============================="
echo "所有训练完成！"
echo "结束时间: $(date)"
echo "=============================="

# 打包结果
echo "打包模型输出..."
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz \
    train/runs/M_N_50K/final_checkpoint \
    train/runs/M_N_50K/train_log.csv \
    train/runs/M_N_100K/final_checkpoint \
    train/runs/M_N_100K/train_log.csv \
    2>/dev/null || echo "[WARN] 打包时部分文件不存在，已跳过"

echo "✅ 结果已打包到 results_*.tar.gz，可下载回本地"
