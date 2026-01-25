"""
Summarize RL training curves (success rate 0-1) from multiple CSV files,
and explicitly mark which one is "my algorithm" for later comparison/write-up.

Per CSV:
- first column = step; remaining columns = multiple runs/groups (values)
  - if step is empty -> treat as missing row; ignored automatically
- steps strictly increasing (guarded)
- step grids across groups may differ -> align via interpolation

Outputs:
- summary_table.csv : one row per algorithm with key metrics + relative deltas vs my algorithm
- curves_mean.json / curves_stats.json / summaries_full.json
- optional plots

Usage:
  python rl_successrate_summary.py --input_dir /path/to/csvs --my_algo_file my_algo.csv
  # or identify by algorithm name (derived from filename without .csv):
  python rl_successrate_summary.py --input_dir /path/to/csvs --my_algo_name MyMethod
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class GroupCurve:
    steps: np.ndarray
    values: np.ndarray


def _read_groups_from_csv(csv_path: str) -> List[GroupCurve]:
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1, how="all")

    ncols = df.shape[1]
    if ncols < 2:
        raise ValueError(f"{csv_path}: need at least 2 columns (step + >=1 group). Got {ncols}.")

    def _postprocess_one(steps: np.ndarray, values: np.ndarray) -> Optional[GroupCurve]:
        steps = pd.to_numeric(steps, errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

        # drop missing step/value rows (step empty => ignored)
        mask = np.isfinite(steps) & np.isfinite(values)
        steps = steps[mask]
        values = values[mask]
        if steps.size == 0:
            return None

        # enforce strictly increasing (drop any non-increasing points)
        keep = np.ones_like(steps, dtype=bool)
        keep[1:] = steps[1:] > steps[:-1]
        steps = steps[keep]
        values = values[keep]

        values = np.clip(values, 0.0, 1.0)
        if steps.size >= 1:  # NOTE: allow single-point group
            return GroupCurve(steps=steps, values=values)
        return None

    # NEW-only: first col = step, remaining cols = values per group
    step_col = df.iloc[:, 0]
    groups: List[GroupCurve] = []
    for j in range(1, ncols):
        g = _postprocess_one(step_col, df.iloc[:, j])
        if g is not None:
            groups.append(g)

    if not groups:
        raise ValueError(f"{csv_path}: no valid groups found (after dropping missing steps/values).")
    return groups


def _common_grid_for_groups(groups: List[GroupCurve], grid_points: int = 2000) -> np.ndarray:
    min_end = min(g.steps[-1] for g in groups)
    max_start = max(g.steps[0] for g in groups)

    if min_end > max_start:
        start, end = max_start, min_end  # intersection range
    else:
        # fallback to union range
        start = min(g.steps[0] for g in groups)
        end = max(g.steps[-1] for g in groups)

    grid_points = max(10, int(grid_points))
    return np.linspace(start, end, grid_points, dtype=float)


def _interp_to_grid(group: GroupCurve, grid: np.ndarray) -> np.ndarray:
    x, y = group.steps, group.values
    if x.size == 1:
        return np.full_like(grid, float(y[0]), dtype=float)
    g = np.clip(grid, x[0], x[-1])
    return np.interp(g, x, y)


def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x)) if x.size >= 2 else float("nan")


def _steps_to_threshold(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    if x.size < 2:
        return float("nan")
    idx = np.where(y >= thr)[0]
    if idx.size == 0:
        return float("nan")
    k = int(idx[0])
    if k == 0:
        return float(x[0])

    x0, y0 = x[k - 1], y[k - 1]
    x1, y1 = x[k], y[k]
    if y1 == y0:
        return float(x1)
    t = (thr - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return float(x0 + t * (x1 - x0))


def _tail_mean(y: np.ndarray, tail_frac: float = 0.05) -> float:
    n = y.size
    if n == 0:
        return float("nan")
    k = max(1, int(math.ceil(n * tail_frac)))
    return float(np.mean(y[-k:]))


def _max_drawdown(y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    peak = -np.inf
    mdd = 0.0
    for v in y:
        if v > peak:
            peak = v
        mdd = max(mdd, peak - v)
    return float(mdd)


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    return x[idx], y[idx]


def summarize_algorithm(csv_path: str, grid_points: int, tail_frac: float) -> Dict:
    groups = _read_groups_from_csv(csv_path)
    grid = _common_grid_for_groups(groups, grid_points=grid_points)

    Ys = np.vstack([_interp_to_grid(g, grid) for g in groups])  # (G,T)
    mean_y = Ys.mean(axis=0)
    std_y = Ys.std(axis=0, ddof=1) if Ys.shape[0] >= 2 else np.zeros_like(mean_y)

    n = Ys.shape[0]
    sem_y = std_y / math.sqrt(n) if n >= 2 else np.zeros_like(mean_y)
    ci95_y = 1.96 * sem_y  # normal approx

    auc = _auc_trapz(grid, mean_y)
    auc_norm = auc / (grid[-1] - grid[0]) if grid[-1] > grid[0] else float("nan")

    final_mean = _tail_mean(mean_y, tail_frac=tail_frac)
    best = float(np.max(mean_y))
    best_step = float(grid[int(np.argmax(mean_y))])

    steps_50 = _steps_to_threshold(grid, mean_y, 0.50)
    steps_70 = _steps_to_threshold(grid, mean_y, 0.70)
    steps_80 = _steps_to_threshold(grid, mean_y, 0.80)
    steps_90 = _steps_to_threshold(grid, mean_y, 0.90)

    mdd = _max_drawdown(mean_y)
    avg_std = float(np.mean(std_y))
    tail_std = float(np.std(mean_y[-max(2, int(mean_y.size * tail_frac)):])) if mean_y.size >= 2 else float("nan")

    x_ds, mean_ds = _downsample_curve(grid, mean_y, max_points=400)
    _, std_ds = _downsample_curve(grid, std_y, max_points=400)
    _, ci95_ds = _downsample_curve(grid, ci95_y, max_points=400)

    algo = os.path.splitext(os.path.basename(csv_path))[0]
    return {
        "file": os.path.basename(csv_path),
        "algorithm": algo,
        "num_groups": int(n),
        "grid_start_step": float(grid[0]),
        "grid_end_step": float(grid[-1]),
        "metrics": {
            "final_tail_mean": final_mean,
            "best_mean": best,
            "best_step": best_step,
            "auc": auc,
            "auc_normalized": auc_norm,
            "steps_to_50": steps_50,
            "steps_to_70": steps_70,
            "steps_to_80": steps_80,
            "steps_to_90": steps_90,
            "max_drawdown": mdd,
            "avg_std_across_groups": avg_std,
            "tail_std_of_mean_curve": tail_std,
        },
        "curve_downsampled": {
            "step": x_ds.tolist(),
            "mean": mean_ds.tolist(),
            "std": std_ds.tolist(),
            "ci95": ci95_ds.tolist(),
        },
    }


def _resolve_my_algo_name(
    summaries: List[Dict],
    my_algo_file: Optional[str],
    my_algo_name: Optional[str],
) -> str:
    if bool(my_algo_file) == bool(my_algo_name):
        raise SystemExit("Please provide exactly ONE of --my_algo_file or --my_algo_name.")

    if my_algo_file:
        # match by filename (with or without path)
        target = os.path.basename(my_algo_file)
        for s in summaries:
            if s["file"] == target:
                return s["algorithm"]
        available = ", ".join(sorted({s["file"] for s in summaries}))
        raise SystemExit(f"--my_algo_file {target} not found. Available files: {available}")

    # my_algo_name
    target = my_algo_name
    for s in summaries:
        if s["algorithm"] == target:
            return s["algorithm"]
    available = ", ".join(sorted({s["algorithm"] for s in summaries}))
    raise SystemExit(f"--my_algo_name {target} not found. Available algorithms: {available}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Folder containing CSV files (each file = one algorithm).")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for CSV files.")
    parser.add_argument("--grid_points", type=int, default=2000, help="Interpolation grid size for aligning groups inside one algorithm.")
    parser.add_argument("--tail_frac", type=float, default=0.05, help="Fraction of tail points for final performance average.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder (default: input_dir).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--my_algo_file", type=str, help="Which CSV file is YOUR method (e.g., MyMethod.csv).")
    group.add_argument("--my_algo_name", type=str, help="Which algorithm name is YOUR method (derived from filename without .csv).")
    parser.add_argument("--plots", action="store_true", help="If set, save comparison plots (requires matplotlib).")
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.out_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)


    csv_files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {in_dir} with pattern {args.pattern}")

    summaries: List[Dict] = []
    for p in csv_files:
        summaries.append(summarize_algorithm(p, grid_points=args.grid_points, tail_frac=args.tail_frac))

    my_algo = _resolve_my_algo_name(summaries, args.my_algo_file, args.my_algo_name)

    # Build table
    rows = []
    for s in summaries:
        m = s["metrics"]
        rows.append({
            "algorithm": s["algorithm"],
            "file": s["file"],
            "is_mine": (s["algorithm"] == my_algo),
            "num_groups": s["num_groups"],
            "start_step_used": s["grid_start_step"],
            "end_step_used": s["grid_end_step"],

            "final_tail_mean": m["final_tail_mean"],
            "best_mean": m["best_mean"],
            "best_step": m["best_step"],
            "auc_normalized": m["auc_normalized"],

            "steps_to_50": m["steps_to_50"],
            "steps_to_70": m["steps_to_70"],
            "steps_to_80": m["steps_to_80"],
            "steps_to_90": m["steps_to_90"],

            "max_drawdown": m["max_drawdown"],
            "avg_std_across_groups": m["avg_std_across_groups"],
            "tail_std_of_mean_curve": m["tail_std_of_mean_curve"],
        })

    table = pd.DataFrame(rows)

    # Add deltas vs my algorithm (good for academic writing: "relative improvement")
    mine = table.loc[table["is_mine"]].iloc[0]

    def rel_delta(a: float, b: float) -> float:
        # (a - b) / (|b| + eps)
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        eps = 1e-12
        return float((a - b) / (abs(b) + eps))

    # Higher is better:
    table["delta_final_vs_mine"] = table["final_tail_mean"].apply(lambda v: v - mine["final_tail_mean"])
    table["reldelta_final_vs_mine"] = table["final_tail_mean"].apply(lambda v: rel_delta(v, mine["final_tail_mean"]))
    table["delta_auc_vs_mine"] = table["auc_normalized"].apply(lambda v: v - mine["auc_normalized"])
    table["reldelta_auc_vs_mine"] = table["auc_normalized"].apply(lambda v: rel_delta(v, mine["auc_normalized"]))

    # Lower is better for steps_to_*:
    for col in ["steps_to_50", "steps_to_70", "steps_to_80", "steps_to_90"]:
        table[f"delta_{col}_vs_mine"] = table[col].apply(lambda v: v - mine[col])
        table[f"reldelta_{col}_vs_mine"] = table[col].apply(lambda v: rel_delta(v, mine[col]))

    # Stability (lower is better):
    for col in ["max_drawdown", "avg_std_across_groups"]:
        table[f"delta_{col}_vs_mine"] = table[col].apply(lambda v: v - mine[col])
        table[f"reldelta_{col}_vs_mine"] = table[col].apply(lambda v: rel_delta(v, mine[col]))

    # Rankings (NaNs worst)
    def _rank(series: pd.Series, ascending: bool) -> pd.Series:
        s = series.copy()
        if ascending:
            fill = np.nanmax(s.to_numpy()) + 1 if np.isfinite(np.nanmax(s.to_numpy())) else 1e18
        else:
            fill = np.nanmin(s.to_numpy()) - 1 if np.isfinite(np.nanmin(s.to_numpy())) else -1e18
        s = s.fillna(fill)
        return s.rank(ascending=ascending, method="min").astype(int)

    table["rank_final"] = _rank(table["final_tail_mean"], ascending=False)
    table["rank_auc"] = _rank(table["auc_normalized"], ascending=False)
    table["rank_steps_to_80"] = _rank(table["steps_to_80"], ascending=True)
    table["rank_stability_std"] = _rank(table["avg_std_across_groups"], ascending=True)
    table["rank_drawdown"] = _rank(table["max_drawdown"], ascending=True)

    # Write outputs
    summary_csv = os.path.join(out_dir, "summary_table.csv")
    table.to_csv(summary_csv, index=False)

    curves_mean = {s["algorithm"]: {"step": s["curve_downsampled"]["step"], "mean": s["curve_downsampled"]["mean"]} for s in summaries}
    curves_stats = {s["algorithm"]: s["curve_downsampled"] for s in summaries}

    with open(os.path.join(out_dir, "curves_mean.json"), "w", encoding="utf-8") as f:
        json.dump(curves_mean, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "curves_stats.json"), "w", encoding="utf-8") as f:
        json.dump(curves_stats, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "summaries_full.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "my_algorithm.json"), "w", encoding="utf-8") as f:
        json.dump({"my_algorithm": my_algo, "my_file": str(mine["file"])}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Processed {len(csv_files)} CSV files.")
    print(f"[OK] My algorithm identified as: {my_algo} (file={mine['file']})")
    print(f"[OK] Wrote: {summary_csv}")
    print(f"[OK] Wrote: {os.path.join(out_dir, 'curves_mean.json')}")
    print(f"[OK] Wrote: {os.path.join(out_dir, 'curves_stats.json')}")
    print(f"[OK] Wrote: {os.path.join(out_dir, 'summaries_full.json')}")
    print(f"[OK] Wrote: {os.path.join(out_dir, 'my_algorithm.json')}")

    if args.plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not installed; skipping plots. (pip install matplotlib)")
            return

        # Mean curves plot
        plt.figure(figsize=(10, 6))
        for s in summaries:
            step = np.array(s["curve_downsampled"]["step"], dtype=float)
            mean = np.array(s["curve_downsampled"]["mean"], dtype=float)
            label = s["algorithm"] + (" (mine)" if s["algorithm"] == my_algo else "")
            plt.plot(step, mean, label=label, linewidth=2.5 if s["algorithm"] == my_algo else 1.6)
        plt.xlabel("Global Steps")
        plt.ylabel("Success Rate")
        plt.title("Mean Success Curves (Aligned within Algorithm)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "mean_curves.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Wrote: {out_path}")

        # Efficiency vs final scatter
        plt.figure(figsize=(8, 6))
        for _, r in table.iterrows():
            if np.isfinite(r["steps_to_80"]):
                plt.scatter(
                    r["steps_to_80"],
                    r["final_tail_mean"],
                    s=70 if r["is_mine"] else 35,
                    marker="*" if r["is_mine"] else "o",
                )
                plt.text(r["steps_to_80"], r["final_tail_mean"], str(r["algorithm"]), fontsize=9)
        plt.xlabel("Steps to 80% Success (lower is better)")
        plt.ylabel(f"Final Tail Mean (last {args.tail_frac:.0%})")
        plt.title("Sample Efficiency vs Final Performance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "efficiency_vs_final.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()