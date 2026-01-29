"""Demo: plot one or multiple CSVs where `step` is x-axis and each other column is a curve.

Enhancements:
  - multiple CSVs -> one figure with subplots (like plot_demo.py)
  - per-column missing values are filled by interpolation
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.file.file_util import get_root_dir


def _fill_missing_by_interp(df: pd.DataFrame, x_col: str = "step") -> pd.DataFrame:
    """对每个数值列补齐缺失：线性插值 + 首尾用 ffill/bfill。"""
    out = df.copy()

    # 确保 x 轴存在且升序（便于插值更合理）
    if x_col in out.columns:
        out = out.sort_values(by=x_col, kind="mergesort").reset_index(drop=True)

    # 把空字符串等转成 NaN，统一数值 dtype
    for c in out.columns:
        if c == x_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    value_cols = [c for c in out.columns if c != x_col]
    if not value_cols:
        return out

    # 逐列插值：中间空洞用线性插值，边界空洞用前后填充
    out[value_cols] = (
        out[value_cols]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    return out


def plot_csvs_as_subplots(
    csv_paths: Sequence[str],
    *,
    titles: Optional[Sequence[str]] = None,
    xlabel: str = "Steps",
    ylabel: str = "Value",
    steps_scale: Optional[float] = 1000,
    save_path: Optional[str] = None,
    dpi: int = 300,
    # NEW: y-label 左对齐位置（越小越靠左，越远离图）
    ylabel_x: float = 10,
    # NEW: 每条线颜色映射（key 为 CSV header/列名）
    line_colors: Optional[dict[str, str]] = None,
    # NEW: 每条线绘制优先级（key 为 CSV header/列名；值越大越后画 -> 更显眼）
    line_priority: Optional[dict[str, int]] = None,
) -> None:
    n = len(csv_paths)
    if n == 0:
        raise ValueError("csv_paths 不能为空")

    fig, axs = plt.subplots(n, 1, figsize=(6, max(3, 3 * n)), sharex=False)
    if n == 1:
        axs = [axs]

    fig.subplots_adjust(hspace=0.23)
    ylabels = [r"Task Success Rate(%)", "Episode Return"]

    line_colors = line_colors or {}
    line_priority = line_priority or {}

    for i, (ax, csv_path) in enumerate(zip(axs, csv_paths)):
        df = pd.read_csv(csv_path)
        df = _fill_missing_by_interp(df, x_col="step")

        if "step" not in df.columns:
            raise ValueError(f"CSV 缺少 step 列: {csv_path}")

        x = df["step"].to_numpy()
        if steps_scale:
            x = x / float(steps_scale)

        # NEW: 按优先级排序后绘制（未配置的默认 0）
        value_cols = [c for c in df.columns if c != "step"]
        value_cols_sorted = sorted(value_cols, key=lambda c: line_priority.get(str(c), 0))

        for col in value_cols_sorted:
            y = df[col].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                linewidth=1.0,
                label=str(col),
                color=line_colors.get(str(col), None),
            )

        t = (titles[i] if titles and i < len(titles) else Path(csv_path).stem)
        # ax.set_title(t)
        ax.set_ylabel(ylabels[i] if i < len(ylabels) else ylabel)

        # NEW: y-label 左对齐 + 可调距离
        ax.yaxis.set_label_coords(ylabel_x, 0.5)

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        # ax.legend(loc="best", fontsize=10)

    axs[-1].set_xlabel(xlabel if not steps_scale else f"{xlabel}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 多个 CSV 合并成一个 figure 的多个子图（参考 plot_demo.py 的结构）
    plot_csvs_as_subplots(
        csv_paths=[
            r"D:\bad reweard function\successrate.csv",
            r"D:\bad reweard function\episode_return.csv",
        ],
        titles=[
            "successrate",
        ],
        xlabel="Steps (in thousands)",
        ylabel="Return",
        save_path=r"D:\temp.pdf",
        dpi=500,
        # NEW: y-label 更靠左（更远离图）
        ylabel_x=-0.11,
        # NEW: header->颜色
        line_colors={
            "Brave": "#ab3a29",
            "ExploRS": "#207d4c",
            "Relara":  "#15689f",
            "Standard":  "#e5a3c4",
            "RND":   "#d0802e",
            # 示例：把 CSV 列名映射到颜色
            # "disable BRB": "#1565c0",
            # "enable BRB": "#df6172",
        },
        # NEW: header->优先级（越大越后画更显眼）
        line_priority={
            "Brave":110,
            "Explorse":10,
            "Relara":9,
            "Standard":8,
            "RND":7,
            # 示例：让 enable 盖在 disable 上面
            # "disable BRB": 0,
            # "enable BRB": 10,
        },
    )
