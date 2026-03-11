"""
将本地训练日志上传到 wandb 新项目 cellphones_phase5_local
"""
import os
import pandas as pd
import wandb

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "YOUR_WANDB_API_KEY")
PROJECT = "cellphones_phase5_local"

BASE = os.environ.get("TRAIN_RUNS_DIR", "models/results/train/runs")

RUNS = [
    ("M_N_50K",  os.path.join(BASE, "M_N_50K",  "train_log.csv")),
    ("M_N_100K", os.path.join(BASE, "M_N_100K", "train_log.csv")),
]


def load_log(log_file: str) -> pd.DataFrame:
    """读取 CSV，过滤掉 DDP 重复写入的 header 行"""
    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line.split(",")
                continue
            # 跳过重复 header 行
            if line.startswith("step,"):
                continue
            parts = line.split(",")
            if len(parts) == len(header):
                rows.append(parts)
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def upload_run(run_name: str, log_file: str):
    print(f"\n{'='*50}")
    print(f"上传: {run_name}  ({log_file})")
    df = load_log(log_file)
    print(f"共 {len(df)} 行记录，列: {list(df.columns)}")

    # 分离训练步骤行和评估行（eval 行 loss 列值通常很大或有 eval_* 列）
    train_df = df[df.columns[:5]].dropna(subset=["step", "loss"])
    # 过滤掉 eval summary 行（loss > 10 通常是 eval_loss 或 runtime 等）
    train_df = train_df[train_df["loss"] < 10].drop_duplicates(subset=["step"])

    print(f"训练步骤行: {len(train_df)}")

    run = wandb.init(
        project=PROJECT,
        name=run_name,
        config={
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "dataset": run_name,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "epochs": 3,
        },
        reinit=True,
    )

    for _, row in train_df.iterrows():
        step = int(row["step"])
        metrics = {
            "train/loss": row["loss"],
            "train/grad_norm": row["grad_norm"],
            "train/learning_rate": row["learning_rate"],
            "train/epoch": row["epoch"],
        }
        metrics = {k: v for k, v in metrics.items() if pd.notna(v)}
        wandb.log(metrics, step=step)

    run.finish()
    print(f"[OK] {run_name} 上传完成")


if __name__ == "__main__":
    wandb.login(key=WANDB_API_KEY)
    for run_name, log_file in RUNS:
        upload_run(run_name, log_file)
    print("\n全部上传完成！")
    print(f"查看地址: https://wandb.ai/home → 项目 {PROJECT}")
