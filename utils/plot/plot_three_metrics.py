from __future__ import annotations

"""Plot 3 training metrics (step + 3 columns) with dual y-axes + optional zoom inset.

Data format (single CSV):
    col0: step
    col1: metric1
    col2: metric2
    col3: metric3  (range is usually smaller; drawn on right y-axis)

This module is designed to match the style used in `utils/plot/plot_demo.py`:
- Times New Roman
- dashed reference grid via axhline/axvline (instead of ax.grid)
- translucent raw/mean line + thicker smooth line
- optional zoom inset via inset_axes + mark_inset

Typical usage:
    from utils.plot.plot_three_metrics import plot_three_metrics_from_csv
    plot_three_metrics_from_csv(
        csv_path="assets/data/metrics.csv",
        metric_names=("Loss", "Entropy", "Success"),
        left_ylabel="Loss / Entropy",
        right_ylabel="Success",
        zoom=ZoomSpec(enabled=True, xlim=(1e5, 2e5)),
        save_path="out.png",
    )

"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


GRID_COLOR = "#cfcfcf"
GRID_STYLE = "--"


def _default_colors() -> List[str]:
    # Close to the repo's palette; also readable on white.
    return ["#1565c0", "#2e7d32", "#df6172"]


@dataclass(frozen=True)
class PlotStyle:
    font_family: str = "Times New Roman"
    base_font_size: int = 20
    label_size: int = 20
    legend_size: int = 20
    line_width: float = 1.0
    dpi: int = 500


@dataclass(frozen=True)
class ZoomSpec:
    enabled: bool = False
    # xlim in *step units* (before x_scale) unless `x_is_scaled=True`.
    xlim: Optional[Tuple[float, float]] = None
    # If xlim is None and auto=True: pick a window near the end.
    auto: bool = True
    # inset placement, in Axes fraction (same meaning as plot_demo's bbox_to_anchor)
    bbox_to_anchor: Tuple[float, float, float, float] = (0.40, 0.75, 0.35, 0.35)
    width: str = "120%"
    height: str = "120%"
    borderpad: float = 2.0
    link_loc1: int = 2
    link_loc2: int = 4
    hide_ticklabels: bool = True


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if window is None or window <= 1 or y.size <= 2:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    if window > y.size:
        window = max(1, (y.size // 2) * 2 + 1)
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="reflect")
    return np.convolve(ypad, kernel, mode="valid")


def _draw_reference_grid(ax: plt.Axes, *, x_step: Optional[float], y_step: Optional[float]) -> None:
    # plot_demo.py style: draw dashed reference lines.
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


def _auto_ylim(values: np.ndarray, *, pad_ratio: float = 0.05) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmin == vmax:
        # Small symmetric padding.
        pad = 1.0 if vmin == 0 else abs(vmin) * pad_ratio
        return (vmin - pad, vmax + pad)
    pad = (vmax - vmin) * pad_ratio
    return (vmin - pad, vmax + pad)


def _pick_auto_zoom_xlim(x: np.ndarray) -> Tuple[float, float]:
    """Pick a default zoom window: last 10%-25% of the curve."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        return (float(x[0]) if x.size else 0.0, float(x[0]) if x.size else 1.0)
    x1 = x[int(x.size * 0.65)]
    x2 = x[int(x.size * 0.85)]
    if x2 <= x1:
        x1, x2 = x[0], x[-1]
    return (float(x1), float(x2))


def read_three_metrics_csv(
    csv_path: str | Path,
    *,
    delimiter: str = ",",
    skip_header: bool = True,
    dedup: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a 4-column CSV: step + 3 metrics.

    Returns:
        steps: (T,)
        y: (T,3)
    """
    from utils.plot.csv_util import read_steps_and_runs_csv

    # We can reuse the robust step sorting/dedup by treating the 3 metrics as 3 "runs".
    steps, runs = read_steps_and_runs_csv(
        csv_path,
        delimiter=delimiter,
        skip_header=skip_header,
        dedup=dedup,
    )

    if len(runs) != 3:
        raise ValueError(
            f"Expected exactly 3 metrics columns after steps, got {len(runs)} in {csv_path}. "
            "If your CSV has more columns, please pre-filter it or extend this reader."
        )

    y = np.stack([np.asarray(r, dtype=float).reshape(-1) for r in runs], axis=1)
    T = min(steps.size, y.shape[0])
    return steps[:T], y[:T, :]


def plot_three_metrics_from_csv(
    *,
    csv_path: str | Path,
    metric_names: Tuple[str, str, str] = ("Metric 1", "Metric 2", "Metric 3"),
    left_ylabel: str = "",
    right_ylabel: str = "",
    title: Optional[str] = None,
    xlabel: str = "Step",
    # smoothing: applied to each metric line
    smooth_window: int = 1,
    mean_alpha: float = 0.25,
    smooth_alpha: float = 0.99,
    # x scaling
    x_scale: float = 1.0,
    x_formatter: Optional[ticker.Formatter] = None,
    # reference grid spacing (optional)
    x_grid_step: Optional[float] = None,
    y_grid_step_left: Optional[float] = None,
    y_grid_step_right: Optional[float] = None,
    # axis limits (optional)
    xlim: Optional[Tuple[float, float]] = None,
    ylim_left: Optional[Tuple[float, float]] = None,
    ylim_right: Optional[Tuple[float, float]] = None,
    # zoom
    zoom: Optional[ZoomSpec] = None,
    style: Optional[PlotStyle] = None,
    # legend
    show_legend: bool = True,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (12, 4),
    # layout
    bottom: Optional[float] = None,  # 新增：控制 subplots_adjust(bottom=...)
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Plot 3 metrics with dual y-axis.

    - metric1/metric2 -> left axis
    - metric3 -> right axis
    """

    st = style or PlotStyle()
    zoom = zoom or ZoomSpec(enabled=False)


    steps, y = read_three_metrics_csv(csv_path)

    x = steps.astype(float) / float(x_scale)
    m1 = y[:, 0]
    m2 = y[:, 1]
    m3 = y[:, 2]

    c1, c2, c3 = _default_colors()

    with mpl.rc_context(
        {
            "font.family": st.font_family,
            "font.size": st.base_font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax_left = plt.subplots(1, 1, figsize=figsize)

        if bottom is not None:
            fig.subplots_adjust(bottom=float(bottom))

        ax_right = ax_left.twinx()

        # Keep right-axis background from covering anything, but DO NOT change axes zorder
        # (changing axes zorder can hide right-axis lines/ticks behind other axes/artists).
        ax_right.patch.set_alpha(0.0)

        lw = st.line_width

        # Left axis metrics
        s1 = _moving_average(m1, smooth_window)
        s2 = _moving_average(m2, smooth_window)
        s3 = _moving_average(m3, smooth_window)

        # draw raw (light) + smooth (strong)
        ax_left.plot(x, m1, color=c1, linewidth=lw, alpha=mean_alpha, label=None)
        h1 = ax_left.plot(x, s1, color=c1, linewidth=lw * 1.8, alpha=smooth_alpha, label=metric_names[0])[0]

        ax_left.plot(x, m2, color=c2, linewidth=lw, alpha=mean_alpha, label=None)
        h2 = ax_left.plot(x, s2, color=c2, linewidth=lw * 1.8, alpha=smooth_alpha, label=metric_names[1])[0]

        ax_right.plot(x, m3, color=c3, linewidth=lw, alpha=mean_alpha, label=None)
        h3 = ax_right.plot(x, s3, color=c3, linewidth=lw * 1.8, alpha=smooth_alpha, label=metric_names[2])[0]

        if title:
            ax_left.set_title(title, fontsize=st.label_size)

        ax_left.set_xlabel(xlabel, fontsize=st.label_size)
        ax_left.set_ylabel(left_ylabel, fontsize=st.label_size)
        ax_right.set_ylabel(right_ylabel, fontsize=st.label_size)

        # tick sizes
        ax_left.tick_params(axis="both", which="both", labelsize=st.base_font_size - 4)
        ax_right.tick_params(axis="y", which="both", labelsize=st.base_font_size - 4)

        # Apply x formatter if provided
        if x_formatter is not None:
            ax_left.xaxis.set_major_formatter(x_formatter)

        # autoscale then optional overrides
        ax_left.relim(); ax_left.autoscale_view()
        ax_right.relim(); ax_right.autoscale_view()

        if xlim is not None:
            ax_left.set_xlim(*xlim)
        if ylim_left is None:
            ax_left.set_ylim(*_auto_ylim(np.concatenate([m1, m2], axis=0)))
        else:
            ax_left.set_ylim(*ylim_left)
        if ylim_right is None:
            ax_right.set_ylim(*_auto_ylim(m3))
        else:
            ax_right.set_ylim(*ylim_right)

        _draw_reference_grid(ax_left, x_step=x_grid_step, y_step=y_grid_step_left)
        _draw_reference_grid(ax_right, x_step=None, y_step=y_grid_step_right)

        # legend: merge left and right handles
        if show_legend:
            handles = [h1, h2, h3]
            labels = [metric_names[0], metric_names[1], metric_names[2]]
            leg = ax_left.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=st.legend_size,
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="0.2",
            )
            # Highest priority: keep legend (colors + text) above everything.
            leg.set_zorder(10_000)
            leg.set_clip_on(False)
            frame = leg.get_frame()
            if frame is not None:
                frame.set_zorder(10_000)
                frame.set_clip_on(False)
            # Also raise legend contents explicitly (some backends treat them separately)
            for t in leg.get_texts():
                t.set_zorder(10_001)
                t.set_clip_on(False)
            for h in leg.legend_handles:
                try:
                    h.set_zorder(10_001)
                    h.set_clip_on(False)
                except Exception:
                    pass

        # ---- zoom inset (on left axis) ----
        if zoom.enabled:
            axins_l = inset_axes(
                ax_left,
                width=zoom.width,
                height=zoom.height,
                bbox_to_anchor=zoom.bbox_to_anchor,
                bbox_transform=ax_left.transAxes,
                borderpad=zoom.borderpad,
            )
            axins_l.set_zorder(100)
            axins_l.patch.set_alpha(0.0)
            for spine in axins_l.spines.values():
                spine.set_linewidth(1)

            axins_r = axins_l.twinx()
            # Keep inset right-axis background transparent; do not change axes zorder.
            axins_r.patch.set_alpha(0.0)
            for spine in axins_r.spines.values():
                spine.set_linewidth(1)

            # Repeat smooth curves in inset (cleaner than raw)
            axins_l.plot(x, s1, color=c1, linewidth=lw * 1.8, alpha=smooth_alpha)
            axins_l.plot(x, s2, color=c2, linewidth=lw * 1.8, alpha=smooth_alpha)
            axins_r.plot(x, s3, color=c3, linewidth=lw * 1.8, alpha=smooth_alpha)

            if zoom.xlim is not None:
                zx1, zx2 = zoom.xlim
                zx1 = zx1 / float(x_scale)
                zx2 = zx2 / float(x_scale)
            elif zoom.auto:
                zx1, zx2 = _pick_auto_zoom_xlim(x)
            else:
                zx1, zx2 = x[0], x[-1]

            axins_l.set_xlim(zx1, zx2)

            # Auto y-lims within zoom window
            mask = (x >= zx1) & (x <= zx2)
            if np.any(mask):
                axins_l.set_ylim(*_auto_ylim(np.concatenate([s1[mask], s2[mask]], axis=0)))
                axins_r.set_ylim(*_auto_ylim(s3[mask]))

            if zoom.hide_ticklabels:
                axins_l.set_xticklabels([])
                axins_l.set_yticklabels([])
                axins_r.set_yticklabels([])

            # Use mark_inset to connect main left axis and left inset
            # Note: it connects to axins_l only; that's OK visually.
            pp, p1, p2 = mark_inset(
                ax_left,
                axins_l,
                loc1=zoom.link_loc1,
                loc2=zoom.link_loc2,
                fc="none",
                ec="0.5",
                linewidth=1.5,
            )
            for artist in (pp, p1, p2):
                try:
                    artist.set_zorder(5)
                    artist.set_clip_on(True)
                except Exception:
                    pass

        if save_path is not None:
            fig.savefig(str(save_path), dpi=st.dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax_left, ax_right
