from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, List, Literal  # <- add Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

from utils.plot.csv_util import build_specs_from_root
from utils.plot.csv_util import align_runs_by_steps  # <- 新增

# 参考 plot_demo.py 的全局风格
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# mpl.rcParams["font.size"] = 24  # <- 改为由 plot_training_curves_grid 的 font_sizes.base 控制

LEGEND_SIZE = 20
LABEL_SIZE = 24
LINE_WIDTH = 1.0

GRID_COLOR = "#cfcfcf"
GRID_STYLE = "--"


@dataclass(frozen=True)
class AxisLimit:
    """统一设置坐标范围：传 None 表示不限制。"""
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class CurveSpec:
    """一条曲线 = 多个 runs，绘制 mean(更淡) + smooth(mean)(更实)。"""
    label: str
    color: str
    runs: Sequence[np.ndarray]  # 每个 run: shape (T,)
    smooth_window: int = 21     # odd 更自然；<=1 表示不平滑
    mean_alpha: float = 0.25
    smooth_alpha: float = 0.99
    mean_lw: float = LINE_WIDTH
    smooth_lw: float = LINE_WIDTH * 2


@dataclass(frozen=True)
class SubplotSpec:
    title: str
    ylabel: str
    curves: Sequence[CurveSpec]
    # 约定：为空字符串/全空白 => 不显示
    xlabel_text: str = "Steps (in thousands)"  # 放在轴下方的说明文本
    # 新增：单子图强制控制是否显示 xlabel_text（None=交给全局策略）
    show_xlabel_text: Optional[bool] = None
    # 新增：控制 xlabel_text 在 Axes 坐标系中的 y 位置（越小越往下，间距越大）
    # 例如：-0.12 更贴近图；-0.25 更远（需要更多底部留白）
    xlabel_text_y: Optional[float] = -0.05

    x: Optional[np.ndarray] = None              # 默认用 [0..T-1]
    limits: Optional[AxisLimit] = None          # 子图单独覆盖全局限制（可选）
    y_major_step: Optional[float] = None        # 如需固定 y 主刻度间隔
    x_major_step: Optional[float] = None        # 如需固定 x 主刻度间隔


@dataclass(frozen=True)
class FontSizes:
    """统一控制图内所有文本字号（不传则使用旧常量/默认）。"""
    base: int = 24                 # 对应 rcParams['font.size']
    title: int = LABEL_SIZE
    label: int = LABEL_SIZE        # x/y label（这里主要用在 ylabel 和 xlabel_text）
    tick: int = 20                 # 坐标轴刻度数字
    legend: int = LEGEND_SIZE
    xlabel_text: int = 20  # ax.text 的轴下方说明文本
    offset: int = 20               # 科学计数法 offset 文本（如 1e3）


def _ensure_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    return a


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = _ensure_1d(y)
    if window is None or window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = max(1, (len(y) // 2) * 2 + 1)
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / window
    # reflect padding，尽量避免边界下陷
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="reflect")
    return np.convolve(ypad, kernel, mode="valid")


def _apply_limits(ax: plt.Axes, global_limits: Optional[AxisLimit], local_limits: Optional[AxisLimit]) -> None:
    lim = local_limits or global_limits
    if lim is None:
        return
    if lim.xlim is not None:
        ax.set_xlim(*lim.xlim)
    if lim.ylim is not None:
        ax.set_ylim(*lim.ylim)


def _draw_reference_grid(ax: plt.Axes, x_step: Optional[float] = None, y_step: Optional[float] = None) -> None:
    # 参考 plot_demo.py：用 axhline/axvline 画虚线网格（而不是 ax.grid）
    if y_step is not None and y_step > 0:
        y_min, y_max = ax.get_ylim()
        start = np.floor(y_min / y_step) * y_step
        for y in np.arange(start, y_max + 1e-12, y_step):
            ax.axhline(y=y, color=GRID_COLOR, linestyle=GRID_STYLE, zorder=0)

    if x_step is not None and x_step > 0:
        x_min, x_max = ax.get_xlim()
        start = np.floor(x_min / x_step) * x_step
        for x in np.arange(start, x_max + 1e-12, x_step):
            ax.axvline(x=x, color=GRID_COLOR, linestyle=GRID_STYLE, zorder=0)


def plot_training_curves_grid(
    specs: Sequence[SubplotSpec],
    nrows: int,
    ncols: int,
    *,
    figsize: Tuple[float, float] = (12, 8),
    global_limits: Optional[AxisLimit] = None,
    hspace: float = 0.15,
    wspace: float = 0.20,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    sharex: bool = False,
    sharey: bool = False,
    font_sizes: Optional[FontSizes] = None,
    default_xlabel_text_y: float = -0.18,
    # 新增：xlabel 说明文本的显示策略
    # - "all": 每个子图都显示（保持旧行为）
    # - "leftmost": 只在最左列子图显示（你的需求）
    # - "none": 全部不显示
    xlabel_text_mode: Literal["all", "leftmost", "none"] = "all",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    - n*m 网格子图
    - 每子图：title 在上，坐标轴在下，横轴说明文本用 ax.text 放在轴下方
    - global_limits 统一设置 xlim/ylim；子图可用 SubplotSpec.limits 覆盖
    - xlabel_text 为空字符串/全空白时不显示
    """
    if len(specs) > nrows * ncols:
        raise ValueError("specs 数量超过 nrows*ncols")

    fs = font_sizes or FontSizes()
    with mpl.rc_context({"font.size": fs.base}):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        for idx, ax in enumerate(axs.flat):
            if idx >= len(specs):
                ax.axis("off")
                continue

            sp = specs[idx]
            row = idx // ncols
            col = idx % ncols  # 预留：如你后续想按列控制显示

            ax.set_title(sp.title, fontsize=fs.title, pad=10)
            ax.set_ylabel(sp.ylabel, fontsize=fs.label)

            # 统一控制 tick 字号（x/y）
            ax.tick_params(axis="both", which="both", labelsize=fs.tick)

            # x 轴数值：必须来自 sp.x（它就是 steps），不要用“曲线长度”推断
            if sp.x is None:
                raise ValueError("SubplotSpec.x 不能为空（应传入 steps 或其缩放值）")
            x_full = _ensure_1d(sp.x)

            for c in sp.curves:
                # 所有 runs 共用同一列 steps：无需对齐；仅堆叠并丢弃全 NaN 的 step
                xx, runs2d = align_runs_by_steps(x_full, c.runs, drop_all_nan_steps=True)

                # 每个 step：忽略 NaN；一行多个值 => 取平均
                mean = np.nanmean(runs2d, axis=0)
                smooth = _moving_average(mean, c.smooth_window)

                ax.plot(xx, mean, color=c.color, linewidth=c.mean_lw, alpha=c.mean_alpha, label=None)
                ax.plot(xx, smooth, color=c.color, linewidth=c.smooth_lw, alpha=c.smooth_alpha, label=c.label)

            # 坐标范围：先应用，再画参考线（参考线需要用最终范围）
            _apply_limits(ax, global_limits, sp.limits)

            if sp.y_major_step is not None:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(sp.y_major_step))
            if sp.x_major_step is not None:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(sp.x_major_step))

            # 科研常用：y 轴支持科学计数（与 plot_demo.py 类似）
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
            # 控制 offset 文本（如 1e3）的字号
            ax.yaxis.get_offset_text().set_fontsize(fs.offset)

            # 先让 matplotlib 自动定一下范围，避免 get_xlim/get_ylim 是默认(0,1)
            ax.relim()
            ax.autoscale_view()

            # 若用了 global/local lim，再次确保生效
            _apply_limits(ax, global_limits, sp.limits)

            _draw_reference_grid(ax, x_step=sp.x_major_step, y_step=sp.y_major_step)

            # 子图底部说明：支持全局策略 + 单图 override
            should_show = True
            if xlabel_text_mode == "none":
                should_show = False
            elif xlabel_text_mode == "leftmost":
                should_show = (col == 0)

            if sp.show_xlabel_text is not None:
                should_show = bool(sp.show_xlabel_text)

            if should_show and sp.xlabel_text and sp.xlabel_text.strip():
                y_text = default_xlabel_text_y if sp.xlabel_text_y is None else float(sp.xlabel_text_y)
                ax.text(
                    0.5, y_text, sp.xlabel_text,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=fs.xlabel_text,
                )

            # 若不是最后一行，隐藏 x tick labels（保持上面干净）
            if row < nrows - 1:
                ax.tick_params(axis="x", which="both", labelbottom=False)

            if show_legend:
                ax.legend(loc=legend_loc, fontsize=fs.legend, frameon=False)

    return fig, axs


if __name__ == "__main__":
    # ===== 从 root/env/algo.csv 扫描读取并绘图 =====

    root = r"D:\test"  # TODO: 替换为你的数据根目录：root/envnamesxxx/algonamesxxx.csv

    specs = build_specs_from_root(
        root,
        ylabel="",  # 可按需填，比如 "Return" / "Loss"
        xlabel_text="Steps (in thousands)",
        smooth_window=21,
        mean_alpha=0.25,
        smooth_alpha=0.99,
        steps_in_thousands=True,
        # env_allowlist=[...],  # 可选
        # algo_allowlist=[...],  # 可选
        # algo_display_name={"algoA": "Algo A"},  # 可选
        # algo_color={"algoA": "#1565c0"},  # 可选
    )

    if not specs:
        raise RuntimeError(f"未在 root 下找到可用的 env/algo csv：{root}")

    # 自动网格：默认 3 列（你可改成 2/4 或者按需传参）
    ncols = 3
    nrows = (len(specs) + ncols - 1) // ncols

    # 可选：从所有子图的 x 推出全局 xlim（steps 已按 thousands 缩放）
    all_xmins = [float(sp.x[0]) for sp in specs if sp.x is not None and len(sp.x) > 0]
    all_xmaxs = [float(sp.x[-1]) for sp in specs if sp.x is not None and len(sp.x) > 0]
    global_limits = AxisLimit(
        xlim=(min(all_xmins), max(all_xmaxs)) if all_xmins and all_xmaxs else None,
        ylim=None,  # y 轴通常不强行统一；如需要可填
    )

    fig, _ = plot_training_curves_grid(
        specs,
        nrows=nrows,
        ncols=ncols,
        figsize=( 10.0 * ncols, 8 * nrows),
        global_limits=global_limits,
        hspace=0.35,
        wspace=0.15,
        show_legend=True,
        legend_loc="upper right",
        xlabel_text_mode="leftmost",  # 关键：只在最左列显示横轴说明 text
    )
    plt.savefig("train_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
