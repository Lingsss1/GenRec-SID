"""
analysis/plot_results.py

Phase 7：读取 results_summary.json，生成分析图表和文本结论。

输出：
  analysis/fig1_hr_vs_N.png       — HR@10 vs N 衰减曲线（含拟合线）
  analysis/fig2_beam_N_heatmap.png — Beam × N 热力图
  analysis/conclusion.txt          — 文本结论

用法（在 server_train 根目录下运行）：
    python analysis/plot_results.py
    python analysis/plot_results.py --results analysis/results_summary.json
    python analysis/plot_results.py --topk 10 --show
"""

import argparse
import json
import math
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[WARN] seaborn 未安装，热力图将使用 matplotlib 替代")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(ROOT, "analysis")

# N 变化实验的标准顺序
N_ORDER = [5000, 20000, 50000, 100000]
N_LABELS = {5000: "5K", 20000: "20K", 50000: "50K", 100000: "100K"}
BEAM_ORDER = [20, 50, 100]
BEAM_COLORS = {20: "#2196F3", 50: "#FF9800", 100: "#4CAF50"}


# ============================================================
# 数据加载
# ============================================================

def load_results(path: str) -> pd.DataFrame:
    """加载 results_summary.json，返回 DataFrame。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"结果文件不存在: {path}\n请先运行 eval/run_all_eval.py 生成评估结果。")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("results_summary.json 为空，请先完成评估。")

    df = pd.DataFrame(data)
    print(f"[INFO] 加载 {len(df)} 条评估结果")
    print(df.to_string(index=False))
    return df


def get_hr_col(df: pd.DataFrame, topk: int) -> str:
    """自动检测 HR 列名（HR@10 或 HR@K）。"""
    col = f"HR@{topk}"
    if col not in df.columns:
        hr_cols = [c for c in df.columns if c.startswith("HR@")]
        if hr_cols:
            col = hr_cols[0]
            print(f"[WARN] 未找到 HR@{topk}，使用 {col}")
        else:
            raise ValueError(f"结果中没有 HR 列，现有列: {list(df.columns)}")
    return col


def get_ndcg_col(df: pd.DataFrame, topk: int) -> str:
    col = f"NDCG@{topk}"
    if col not in df.columns:
        ndcg_cols = [c for c in df.columns if c.startswith("NDCG@")]
        if ndcg_cols:
            col = ndcg_cols[0]
    return col if col in df.columns else None


# ============================================================
# 图 1：HR@10 vs N 衰减曲线
# ============================================================

def plot_hr_vs_N(df: pd.DataFrame, topk: int, output_path: str, show: bool = False):
    """
    图 1：HR@K vs N（log scale），每个 Beam 档位一条线 + 拟合曲线。
    """
    hr_col = get_hr_col(df, topk)

    # 只保留 N 变化实验数据
    plot_df = df[df["N"].isin(N_ORDER)].copy()
    if plot_df.empty:
        print("[WARN] 图1：没有 N 变化实验数据，跳过")
        return

    available_beams = sorted(plot_df["beam"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for beam in available_beams:
        beam_df = plot_df[plot_df["beam"] == beam].sort_values("N")
        if beam_df.empty:
            continue

        xs = beam_df["N"].values
        ys = beam_df[hr_col].values
        color = BEAM_COLORS.get(beam, None)
        label = f"Beam={beam}"

        # 实际数据点
        ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=7, zorder=3)

        # 在 log(N) 上做线性拟合（至少需要 2 个点）
        if len(xs) >= 2:
            log_xs = np.log10(xs)
            coeffs = np.polyfit(log_xs, ys, 1)
            fit_fn = np.poly1d(coeffs)
            # 在更密集的 x 范围上绘制拟合线
            x_fit = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 100)
            y_fit = fit_fn(np.log10(x_fit))
            ax.plot(x_fit, y_fit, "--", color=color, alpha=0.5, linewidth=1.5)

    # 坐标轴设置
    ax.set_xscale("log")
    ax.set_xticks(N_ORDER)
    ax.set_xticklabels([N_LABELS[n] for n in N_ORDER])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("候选集大小 N", fontsize=12)
    ax.set_ylabel(f"HR@{topk}", fontsize=12)
    ax.set_title(f"HR@{topk} vs 候选集大小 N（N 变化实验）", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OK] 图1 已保存: {output_path}")
    if show:
        plt.show()
    plt.close()


# ============================================================
# 图 2：Beam × N 热力图
# ============================================================

def plot_beam_N_heatmap(df: pd.DataFrame, topk: int, output_path: str, show: bool = False):
    """
    图 2：Beam × N 热力图，单元格为 HR@K。
    行：N（5K, 20K, 50K, 100K）；列：Beam（20, 50, 100）
    """
    hr_col = get_hr_col(df, topk)

    plot_df = df[df["N"].isin(N_ORDER)].copy()
    if plot_df.empty:
        print("[WARN] 图2：没有 N 变化实验数据，跳过")
        return

    available_Ns = [n for n in N_ORDER if n in plot_df["N"].values]
    available_beams = [b for b in BEAM_ORDER if b in plot_df["beam"].values]

    # 构建热力图矩阵
    matrix = np.full((len(available_Ns), len(available_beams)), np.nan)
    for i, n in enumerate(available_Ns):
        for j, beam in enumerate(available_beams):
            row = plot_df[(plot_df["N"] == n) & (plot_df["beam"] == beam)]
            if not row.empty:
                matrix[i, j] = row[hr_col].values[0]

    row_labels = [N_LABELS[n] for n in available_Ns]
    col_labels = [f"Beam={b}" for b in available_beams]

    fig, ax = plt.subplots(figsize=(6, max(3, len(available_Ns) * 1.2)))

    if HAS_SEABORN:
        heat_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        sns.heatmap(
            heat_df,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": f"HR@{topk}"},
        )
    else:
        # matplotlib fallback
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label=f"HR@{topk}")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(len(available_Ns)):
            for j in range(len(available_beams)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9)

    ax.set_title(f"HR@{topk} 热力图（N × Beam）", fontsize=13)
    ax.set_xlabel("Beam Size", fontsize=11)
    ax.set_ylabel("候选集大小 N", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OK] 图2 已保存: {output_path}")
    if show:
        plt.show()
    plt.close()


# ============================================================
# 文本结论
# ============================================================

def print_conclusions(df: pd.DataFrame, topk: int, output_path: str):
    """生成文本结论，打印到 stdout 并写入文件。"""
    hr_col = get_hr_col(df, topk)
    ndcg_col = get_ndcg_col(df, topk)

    lines = []
    lines.append("=" * 60)
    lines.append("Cell_Phones_and_Accessories 生成式推荐实验结论")
    lines.append("=" * 60)
    lines.append("")

    # ---- N 衰减分析 ----
    plot_df = df[df["N"].isin(N_ORDER)].copy()
    if not plot_df.empty:
        lines.append("【问题1】HR@10 随候选集大小 N 的衰减")
        lines.append("-" * 40)

        for beam in sorted(plot_df["beam"].unique()):
            beam_df = plot_df[plot_df["beam"] == beam].sort_values("N")
            if len(beam_df) < 2:
                continue
            hr_vals = beam_df[hr_col].values
            n_vals = beam_df["N"].values
            lines.append(f"  Beam={beam}:")
            for n, hr in zip(n_vals, hr_vals):
                lines.append(f"    N={N_LABELS[n]:>5s}  HR@{topk}={hr:.4f}")

            # 计算衰减比例
            hr_min_n = hr_vals[0]  # 最小 N 对应的 HR
            hr_max_n = hr_vals[-1]  # 最大 N 对应的 HR
            if hr_min_n > 0:
                drop_pct = (hr_min_n - hr_max_n) / hr_min_n * 100
                lines.append(
                    f"    → N 从 {N_LABELS[n_vals[0]]} 增大到 {N_LABELS[n_vals[-1]]}，"
                    f"HR@{topk} 下降 {drop_pct:.1f}%"
                )
            lines.append("")

        # Beam 收益分析
        lines.append("【附加】Beam 增大对 HR@10 的收益")
        lines.append("-" * 40)
        for n in N_ORDER:
            n_df = plot_df[plot_df["N"] == n].sort_values("beam")
            if n_df.empty:
                continue
            lines.append(f"  N={N_LABELS[n]}:")
            for _, row in n_df.iterrows():
                lines.append(f"    Beam={int(row['beam']):>3d}  HR@{topk}={row[hr_col]:.4f}")
            beams_sorted = n_df["beam"].values
            hrs_sorted = n_df[hr_col].values
            if len(hrs_sorted) >= 2 and hrs_sorted[0] > 0:
                gain = (hrs_sorted[-1] - hrs_sorted[0]) / hrs_sorted[0] * 100
                lines.append(
                    f"    → Beam {beams_sorted[0]}→{beams_sorted[-1]} 收益: +{gain:.1f}%"
                )
            lines.append("")

    # ---- 业务向总结 ----
    lines.append("【业务结论】")
    lines.append("-" * 40)

    # 找到 beam=50 时各 N 的 HR
    b50 = plot_df[plot_df["beam"] == 50].sort_values("N") if not plot_df.empty else pd.DataFrame()
    if not b50.empty:
        hr_5k_row = b50[b50["N"] == 5000]
        hr_100k_row = b50[b50["N"] == 100000]
        if not hr_5k_row.empty and not hr_100k_row.empty:
            hr_5k = hr_5k_row[hr_col].values[0]
            hr_100k = hr_100k_row[hr_col].values[0]
            threshold = 0.05  # 业务可用阈值（可调整）
            lines.append(
                f"在 Cell_Phones_and_Accessories 类目下（Beam=50）："
            )
            lines.append(f"  N=5K   HR@{topk}={hr_5k:.4f}")
            lines.append(f"  N=100K HR@{topk}={hr_100k:.4f}")
            if hr_5k >= threshold:
                lines.append(
                    f"  → N=5K 时 HR@{topk}={hr_5k:.4f} ≥ {threshold}，生成式推荐在小候选集下可用。"
                )
            if hr_100k < threshold:
                lines.append(
                    f"  → N=100K 时 HR@{topk}={hr_100k:.4f} < {threshold}，"
                    f"大候选集下性能显著下降，建议配合召回截断（N≤20K）使用。"
                )
    else:
        lines.append("  （数据不足，无法生成业务结论，请完成全部模型评估后重新运行）")

    lines.append("")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[OK] 结论已写入: {output_path}")


# ============================================================
# CLI 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7 分析与可视化")
    parser.add_argument("--results",  default="analysis/results_summary.json", help="评估结果 JSON 路径")
    parser.add_argument("--topk",     default=10, type=int, help="HR@K 和 NDCG@K 的 K 值")
    parser.add_argument("--show",     action="store_true", help="显示图表（需要 GUI 环境）")
    parser.add_argument("--out_dir",  default="analysis", help="图表输出目录")
    return parser.parse_args()


def main():
    args = parse_args()

    results_path = os.path.join(ROOT, args.results)
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.show:
        matplotlib.use("TkAgg")  # 切换到交互后端

    df = load_results(results_path)

    plot_hr_vs_N(
        df, args.topk,
        output_path=os.path.join(out_dir, "fig1_hr_vs_N.png"),
        show=args.show,
    )

    plot_beam_N_heatmap(
        df, args.topk,
        output_path=os.path.join(out_dir, "fig2_beam_N_heatmap.png"),
        show=args.show,
    )

    print_conclusions(
        df, args.topk,
        output_path=os.path.join(out_dir, "conclusion.txt"),
    )


if __name__ == "__main__":
    main()
