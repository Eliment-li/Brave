from __future__ import annotations

"""Minimal demo for `utils.plot.plot_three_metrics`.

Edit `CSV_PATH` to your training log CSV and run this file.
"""

from pathlib import Path

from utils.plot.plot_three_metrics import PlotStyle, ZoomSpec, plot_three_metrics_from_csv


def main() -> None:
    # TODO: change to your csv.
    CSV_PATH = Path(r"D:\sync\brave_data\hdr.csv")

    plot_three_metrics_from_csv(
        csv_path=CSV_PATH,
        metric_names=("HDR", "Bouns", "Distance to Goal"),
        left_ylabel="Value of HDR and Bouns",
        right_ylabel="Distance to Goal",
        title=None,
        xlabel="Steps",
        # If you want to show 'Steps (in thousands)', set x_scale=1000.
        # x_scale=1000.0,
        smooth_window=1,
        #zoom=ZoomSpec(enabled=True, auto=True),
        #zoom=ZoomSpec(enabled=True, xlim=(0,20)),
        style=PlotStyle(),
        show_legend=False,  # 设为 False 可隐藏 legend
        save_path=Path("three_metrics.pdf"),
        bottom=0.18,  # 新增：按需调大/调小（典型范围 0.10~0.25）
        show=True,
    )


if __name__ == "__main__":
    main()
