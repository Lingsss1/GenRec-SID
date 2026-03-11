#!/bin/bash
# GRPO 强化学习训练脚本
#
# 用法:
#   bash run_rl_train.sh                    # 训练所有 RL 模型
#   bash run_rl_train.sh RL_M_N_5K          # 只训练指定配置
#   bash run_rl_train.sh RL_M_N_5K ranking  # 指定配置 + 奖励类型
#
# 多 GPU (accelerate):
#   accelerate launch --num_processes 4 phase6_rl_train.py --config train/configs/RL_M_N_5K.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================="
echo "Phase 6: GRPO RL Training"
echo "=============================="
echo "工作目录: $SCRIPT_DIR"
echo "开始时间: $(date)"
echo ""

# 检查 GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[WARN] nvidia-smi 不可用"
echo ""

# 参数解析
CONFIG_NAME="${1:-all}"
REWARD_TYPE="${2:-}"

run_single() {
    local config_name="$1"
    local config_file="train/configs/${config_name}.json"

    if [ ! -f "$config_file" ]; then
        echo "[ERROR] 配置文件不存在: $config_file"
        return 1
    fi

    echo "=============================="
    echo "开始 RL 训练: $config_name"
    echo "配置文件: $config_file"
    echo "时间: $(date)"
    echo "=============================="

    if [ -n "$REWARD_TYPE" ]; then
        python phase6_rl_train.py --config "$config_file" --reward_type "$REWARD_TYPE"
    else
        python phase6_rl_train.py --config "$config_file"
    fi

    echo ""
    echo "✅ $config_name RL 训练完成: $(date)"
    echo ""
}

if [ "$CONFIG_NAME" = "all" ]; then
    # 按顺序训练所有配置
    for cfg in RL_M_N_5K RL_M_N_20K RL_M_N_50K RL_M_N_100K; do
        config_file="train/configs/${cfg}.json"
        if [ -f "$config_file" ]; then
            run_single "$cfg"
        else
            echo "[SKIP] $cfg: 配置文件不存在"
        fi
    done
else
    run_single "$CONFIG_NAME"
fi

echo "=============================="
echo "所有 RL 训练完成！"
echo "结束时间: $(date)"
echo "=============================="

# 打包结果
echo "打包 RL 模型输出..."
tar -czf rl_results_$(date +%Y%m%d_%H%M%S).tar.gz \
    train/runs/RL_M_N_*/final_checkpoint \
    2>/dev/null || echo "[WARN] 打包时部分文件不存在，已跳过"

echo "✅ 结果已打包到 rl_results_*.tar.gz"
