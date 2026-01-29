from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np

# 仅做类型引用/构造用：避免循环 import 放函数内


def read_steps_and_runs_csv(
    csv_path: str | Path,
    *,
    delimiter: str = ",",
    skip_header: bool = True,
    # 新增：当同一个 step 出现多行时如何处理
    # - "mean": 对重复 step 的各行取均值（推荐）
    # - "last": 取最后一行
    # - "first": 取第一行
    dedup: str = "mean",
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    读取格式：
      - 第1列: steps
      - 第2..N列: 多次 runs（每列一条 run）
    返回：
      steps: shape (T,)
      runs: List[np.ndarray]，每个 shape (T,)
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    # 用 genfromtxt 支持缺失值/混合类型；默认跳过表头
    data = np.genfromtxt(
        p,
        delimiter=delimiter,
        skip_header=1 if skip_header else 0,
        dtype=float,
        invalid_raise=False,
    )

    if data.ndim == 1:
        # 单行/异常：至少要两列（steps + 1 run）
        if data.size < 2:
            raise ValueError(f"CSV 至少需要两列(steps + run)：{p}")
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        raise ValueError(f"CSV 至少需要两列(steps + run)：{p}")

    steps = data[:, 0]
    runs_mat = data[:, 1:]

    # 去掉全 nan 的列
    keep_cols = []
    for j in range(runs_mat.shape[1]):
        col = runs_mat[:, j]
        if np.isfinite(col).any():
            keep_cols.append(j)

    if not keep_cols:
        raise ValueError(f"CSV 没有可用的 run 列：{p}")

    runs_mat = runs_mat[:, keep_cols]

    # 对齐长度：以 steps / runs 同时非 nan 的前缀长度为准（更稳）
    finite_steps = np.isfinite(steps)
    finite_any_run = np.isfinite(runs_mat).any(axis=1)
    mask = finite_steps & finite_any_run
    if not mask.any():
        raise ValueError(f"CSV 没有有效数据行：{p}")

    # 取有效行（保留原顺序）
    steps = steps[mask]
    runs_mat = runs_mat[mask, :]

    # ===== 新增：按 step 排序（避免乱序 step 影响绘图/平滑/locator）=====
    order = np.argsort(steps, kind="mergesort")
    steps = steps[order]
    runs_mat = runs_mat[order, :]

    # ===== 新增：重复 step 去重（可选）=====
    # 注意：重复 step 常见于日志重复写入；不去重会导致折线来回折返或平滑异常
    if steps.size > 1:
        # 找到每个唯一 step 的边界
        uniq_steps, idx_start, counts = np.unique(steps, return_index=True, return_counts=True)
        if np.any(counts > 1):
            if dedup not in ("mean", "last", "first"):
                raise ValueError(f"dedup 仅支持 mean/last/first，得到: {dedup}")

            if dedup == "mean":
                agg = np.empty((uniq_steps.size, runs_mat.shape[1]), dtype=float)
                for i, (s, c) in enumerate(zip(idx_start, counts)):
                    block = runs_mat[s : s + c, :]
                    agg[i, :] = np.nanmean(block, axis=0)
                steps = uniq_steps
                runs_mat = agg
            elif dedup == "last":
                take = idx_start + counts - 1
                steps = steps[take]
                runs_mat = runs_mat[take, :]
            else:  # "first"
                take = idx_start
                steps = steps[take]
                runs_mat = runs_mat[take, :]

    # 生成 runs list（每列一条 run）
    runs: List[np.ndarray] = [runs_mat[:, j].astype(float, copy=False) for j in range(runs_mat.shape[1])]
    return steps.astype(float, copy=False), runs


def align_runs_by_steps(
    steps: np.ndarray,
    runs: Sequence[np.ndarray],
    *,
    # True: 丢弃 mean 不可计算（该 step 全部为 NaN）的点
    drop_all_nan_steps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    你的数据前提：所有 runs 共用同一列 steps（同一行同一 step），因此不需要“对齐”。
    这里仅把 runs 堆成 (N,T)，并可选丢弃全 NaN 的 step。
    """
    s = np.asarray(steps, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("steps 不能为空")

    rr = [np.asarray(r, dtype=float).reshape(-1) for r in runs]
    if not rr:
        raise ValueError("runs 不能为空")

    # 防御：若某些列意外长度不同，按最短截断（不做任何 step 对齐）
    T = min(s.size, *(r.size for r in rr))
    s = s[:T]
    rr = [r[:T] for r in rr]

    runs2d = np.stack(rr, axis=0)  # (N,T)

    if not drop_all_nan_steps:
        return s, runs2d

    # 丢弃该 step 在所有 run 上都为 NaN 的点（否则 mean 为 NaN，画图会断）
    keep = np.isfinite(s) & np.isfinite(runs2d).any(axis=0)
    if not keep.any():
        raise ValueError("没有任何 step 存在可用的 run 值（全为 NaN）")
    return s[keep], runs2d[:, keep]


def _default_color_cycle() -> List[str]:
    # matplotlib tab10-ish（无需依赖 mpl）
    # return [
    #     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    #     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    # ]
    return[
        "#ab3a29","#1e7c4a","#13679e","#d07f2c","#e5a2c4"
    ]


def _round_up_to_base(value: float, base: float) -> float:
    if base <= 0:
        raise ValueError("x_axis_round_base 必须为正数")
    if not math.isfinite(value) or value <= 0:
        return base
    return max(base, math.ceil(value / base) * base)


def build_specs_from_root(
    root: str | Path,
    *,
    ylabel: str = "",
    xlabel_text: str = "Steps (in thousands)",
    smooth_window: int = 1000,
    mean_alpha: float = 0.25,
    smooth_alpha: float = 0.99,
    steps_in_thousands: bool = True,
    env_allowlist: Optional[Sequence[str]] = None,
    algo_allowlist: Optional[Sequence[str]] = None,
    algo_display_name: Optional[Dict[str, str]] = None,
    algo_color: Optional[Dict[str, str]] = None,
    x_axis_round_base: float = 1e5,
):
    """
    扫描目录结构：
      root/
        envA/
          algo1.csv
          algo2.csv
        envB/
          algo1.csv
          ...
    规则：
      - 每个 env => 一个 SubplotSpec（title=env 名）
      - 每个 algo.csv => 一个 CurveSpec（label=algo 名）
      - CSV: steps + 多列 runs
    """
    from .plot_train_data import AxisLimit, CurveSpec, SubplotSpec  # 延迟 import

    r = Path(root)
    if not r.exists():
        raise FileNotFoundError(str(r))

    env_dirs = sorted([p for p in r.iterdir() if p.is_dir()], key=lambda p: p.name)
    if env_allowlist is not None:
        allow = set(env_allowlist)
        env_dirs = [p for p in env_dirs if p.name in allow]

    colors = _default_color_cycle()
    algo_display_name = algo_display_name or {}
    algo_color = algo_color or {}

    specs = []
    scale = 1000.0 if steps_in_thousands else 1.0
    for env_dir in env_dirs:
        csv_files = sorted(env_dir.glob("*.csv"), key=lambda p: p.stem)
        if algo_allowlist is not None:
            allow_algo = set(algo_allowlist)
            csv_files = [p for p in csv_files if p.stem in allow_algo]

        curves = []
        x_reference = None
        max_step_in_env = 0.0
        for k, csv_path in enumerate(csv_files):
            steps, runs = read_steps_and_runs_csv(csv_path)
            x_scaled = steps / scale
            if x_reference is None:
                x_reference = x_scaled
            max_step_in_env = max(max_step_in_env, float(np.nanmax(steps)))
            algo = csv_path.stem
            label = algo_display_name.get(algo, algo)
            color = algo_color.get(algo, colors[k % len(colors)])
            is_brave = label.strip().lower() == "brave" or algo.strip().lower() == "brave"

            curves.append(
                CurveSpec(
                    label=label,
                    color=color,
                    runs=runs,
                    smooth_window=smooth_window,
                    mean_alpha=mean_alpha,
                    smooth_alpha=smooth_alpha,
                    steps=x_scaled,
                    priority=100 if is_brave else 0,
                )
            )
        if not curves:
            continue
        rounded_max = _round_up_to_base(max_step_in_env, x_axis_round_base)
        xlim = (0.0, rounded_max / scale)
        specs.append(
            SubplotSpec(
                title=env_dir.name,
                ylabel=ylabel,
                curves=curves,
                xlabel_text=xlabel_text,
                x=x_reference,
                limits=AxisLimit(xlim=xlim),
            )
        )

    return specs


def read_step_and_columns_csv(
    csv_path: str | Path,
    *,
    step_col: str = "step",
    delimiter: str = ",",
    skip_header: bool = True,
    dedup: str = "last",
    steps_scale: float = 1.0,
    columns_allowlist: Optional[Sequence[str]] = None,
    columns_blocklist: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Read a CSV where one file corresponds to one figure.

    Expected format:
      - One column named by `step_col` (default: 'step') for x-axis.
      - Every other numeric column is a curve (no averaging).

    Returns:
      steps: shape (T,), sorted ascending and scaled by `steps_scale`.
      series: dict[name -> values], each shape (T,), aligned with steps.

    Notes:
      - `step` does NOT need to be contiguous.
      - If duplicated steps exist, `dedup` controls how to resolve: mean/last/first.
      - Non-numeric columns are ignored.
    """
    import pandas as pd

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if dedup not in ("mean", "last", "first"):
        raise ValueError(f"dedup 仅支持 mean/last/first，得到: {dedup}")

    df = pd.read_csv(p, delimiter=delimiter)
    if df.empty:
        raise ValueError(f"CSV 为空: {p}")

    # Normalize column names a bit (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    if step_col not in df.columns:
        raise ValueError(f"CSV 缺少 step 列 '{step_col}': {p}")

    # Keep numeric columns only (besides step)
    step = pd.to_numeric(df[step_col], errors="coerce")

    cols = [c for c in df.columns if c != step_col]
    if columns_allowlist is not None:
        allow = set(columns_allowlist)
        cols = [c for c in cols if c in allow]
    if columns_blocklist is not None:
        block = set(columns_blocklist)
        cols = [c for c in cols if c not in block]

    if not cols:
        raise ValueError(f"CSV 中除了 '{step_col}' 外没有可用列: {p}")

    series_df = df[cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows where step is NaN or all series are NaN
    mask = step.notna() & series_df.notna().any(axis=1)
    if not mask.any():
        raise ValueError(f"CSV 没有有效数据行: {p}")

    step = step[mask]
    series_df = series_df.loc[mask]

    # Sort by step (stable)
    order = step.argsort(kind="mergesort")
    step = step.iloc[order].reset_index(drop=True)
    series_df = series_df.iloc[order].reset_index(drop=True)

    # Dedup by step if needed
    if step.duplicated().any():
        temp = pd.concat({step_col: step, "__idx__": pd.RangeIndex(len(step))}, axis=1)
        temp = pd.concat([temp, series_df], axis=1)
        if dedup == "mean":
            grouped = temp.groupby(step_col, as_index=False).mean(numeric_only=True)
        elif dedup == "first":
            grouped = temp.groupby(step_col, as_index=False).first()
        else:  # last
            grouped = temp.groupby(step_col, as_index=False).last()
        step = grouped[step_col]
        series_df = grouped[cols]

    steps_np = step.to_numpy(dtype=float) / float(steps_scale)
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        col = series_df[c].to_numpy(dtype=float)
        # keep columns that have at least one finite value
        if np.isfinite(col).any():
            out[c] = col

    if not out:
        raise ValueError(f"CSV 没有任何可用的数值列(除了 '{step_col}'): {p}")

    return steps_np, out
