import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import to_rgb
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from utils.file.csv_util import load_series_from_csv
from utils.file.file_util import get_root_dir

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
legend_size = 20
mpl.rcParams['font.size'] = 24
label_size = 26
line_width = 1.8
dpi = 500


def replace_nan_with_average(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    for idx, value in enumerate(arr):
        if np.isnan(value):
            prev_val = arr[idx - 1] if idx > 0 else np.nan
            next_val = arr[idx + 1] if idx < len(arr) - 1 else np.nan
            if not np.isnan(prev_val) and not np.isnan(next_val):
                arr[idx] = (prev_val + next_val) / 2
            elif not np.isnan(prev_val):
                arr[idx] = prev_val
            elif not np.isnan(next_val):
                arr[idx] = next_val
    return arr


def lighten_color(color: str, factor: float) -> Tuple[float, float, float]:
    factor = float(np.clip(factor, 0.0, 1.0))
    base = np.array(to_rgb(color))
    return tuple(base + (1 - base) * factor)


def plot_training_curves(
    csv_files: Iterable[str | Path],
    labels: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
    shade_alpha: float = 0.3,
    shade_lighten: float = 0.5,
    linewidth: float = line_width,
):
    """
    基于多个 CSV 绘制训练曲线。每个 CSV：
      第 1 列为 step，其余列为同一次实验下的多次测量。
    主线为逐步平均值，阴影为 mean±std，颜色与主线一致但更浅。
    """
    csv_files = list(csv_files)
    if labels and len(labels) != len(csv_files):
        raise ValueError('labels 数量需要与 csv_files 对齐')

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        created_fig = True

    color_cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', ['#1565c0', '#df6172'])

    for idx, csv_path in enumerate(csv_files):
        steps, values = load_series_from_csv(csv_path)
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(values, axis=1)
            std = np.nanstd(values, axis=1)
        color = color_cycle[idx % len(color_cycle)]
        label = labels[idx] if labels else Path(csv_path).stem

        valid_mask = ~np.isnan(mean)
        if not np.any(valid_mask):
            continue
        steps_valid = steps[valid_mask]
        mean_valid = mean[valid_mask]
        std_valid = std[valid_mask]

        ax.plot(steps_valid, mean_valid, color=color, linewidth=linewidth, label=label, alpha=0.95)
        ax.fill_between(
            steps_valid,
            mean_valid - std_valid,
            mean_valid + std_valid,
            color=lighten_color(color, shade_lighten),
            alpha=shade_alpha,
            linewidth=0,
        )

    ax.set_xlabel('Step', fontsize=label_size)
    ax.set_ylabel('Score', fontsize=label_size)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    ax.grid(ls='--', color='#cfcfcf', alpha=0.6)
    ax.legend(fontsize=legend_size, frameon=False)

    if created_fig:
        return ax
    return ax


if __name__ == '__main__':
    csvs = [
        Path(r'D:\paper\icml2026\data\mountain_car_train\mountain_car_continous_brave_r80.csv'),
        Path(r'D:\paper\icml2026\data\mountain_car_train\mountain_car_continous_standerd.csv'),
    ]
    ax = plot_training_curves(csvs, labels=['Brave', 'PPO'])
    root = Path(get_root_dir())
    out_path = root / 'results' / 'fig' / 'train_curves.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.show()

