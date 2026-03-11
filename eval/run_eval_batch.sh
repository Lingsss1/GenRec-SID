#!/bin/bash
#SBATCH -p q-3090 --gres=gpu:1 --cpus-per-task=8 --time=12:00:00 --mail-type=ALL

echo "=== 评估任务开始: $(date) ==="
echo "节点: $(hostname)"
nvidia-smi

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate onerec

cd ~/server_train

rm -f analysis/results_summary.json

python eval/run_all_eval.py

echo "=== 评估任务完成: $(date) ==="
