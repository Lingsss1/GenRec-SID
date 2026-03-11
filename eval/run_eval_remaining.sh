#!/bin/bash
#SBATCH -p q-3090-batch --gres=gpu:1 --cpus-per-task=8 --time=4:00:00 --mail-type=ALL

echo "=== 评估剩余任务开始: $(date) ==="
echo "节点: $(hostname)"
nvidia-smi

. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate onerec

cd ~/server_train

# M_N_50K 只跑 beam=100，M_N_100K 跑全部 beam
python eval/run_all_eval.py --only M_N_50K --beams 100
python eval/run_all_eval.py --only M_N_100K

echo "=== 评估任务完成: $(date) ==="
