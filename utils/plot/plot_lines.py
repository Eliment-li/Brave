import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Union

# 全局风格设置（参考用户代码）
plt.rcParams["font.family"] = "Arial"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 24

legend_size = 20
label_size = 26
line_width = 1.5
default_dpi = 500

def plot_lines(
    lines: Sequence[Sequence[float]],
    names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = default_dpi,
    xlabel: str = '',
    ylabel: str = '',
    title: str = '',
    show_mean: bool = True,
    grid_x_step: int = 20,
    grid_y_step: Optional[float] = None,
    legend_loc: str = 'upper right',
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制二维曲线。
    - lines: 可迭代对象，每个元素为一维数组（或可转为 numpy 的序列），表示一条曲线。
    - names: 与 lines 等长的名字列表；若存在重复名字，则按名字分组：绘制每条组内曲线（alpha 低）并绘制该组均值（粗线）。
             若为 None，则逐条绘制，不计算组均值。
    - colors: 可选颜色列表；如果提供且组名少于颜色数，将按组或按曲线循环使用颜色。
    - show_mean: 对分组时是否绘制均值线（粗线）。
    - grid_x_step / grid_y_step: 参考线间隔（None 表示自动选择 y 间隔）。
    - 返回 (fig, ax)。
    """
    # prepare data
    lines_np = [np.array(l, dtype=float) for l in lines]
    n = len(lines_np)
    if n == 0:
        raise ValueError("lines 不能为空")

    # x 序列
    max_len = max(len(l) for l in lines_np)
    x = np.arange(max_len)

    # pad shorter lines with NaN 保持对齐
    padded = []
    for l in lines_np:
        if len(l) < max_len:
            arr = np.full(max_len, np.nan)
            arr[:len(l)] = l
            padded.append(arr)
        else:
            padded.append(l)
    data = np.vstack(padded)  # shape (n, max_len)

    # 分组逻辑
    grouped = None
    if names is not None:
        if len(names) != n:
            raise ValueError("names 长度必须等于 lines 长度")
        grouped = defaultdict(list)
        for idx, name in enumerate(names):
            grouped[name].append(data[idx])
        groups = list(grouped.items())  # list of (name, [arrays])
    else:
        # treat each line as its own group with unique name
        groups = [(f'line_{i}', [data[i]]) for i in range(n)]

    # 颜色分配
    cmap = plt.get_cmap('tab10')
    if colors is None:
        colors = [cmap(i % 10) for i in range(len(groups))]
    # ensure enough colors
    if len(colors) < len(groups):
        colors = list(colors) + [cmap(i % 10) for i in range(len(groups) - len(colors))]

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=label_size)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)

    # plot each group: individual faint lines + group mean (粗)
    for i, (gname, arrs) in enumerate(groups):
        color = colors[i]
        for arr in arrs:
            ax.plot(x, arr, color=color, linewidth=line_width, alpha=0.25, zorder=0)
        if show_mean and len(arrs) > 1:
            mean_arr = np.nanmean(np.vstack(arrs), axis=0)
            ax.plot(x, mean_arr, color=color, linewidth=line_width * 2, alpha=0.99, label=gname, zorder=5)
        else:
            # single-line group -> plot as normal thicker
            ax.plot(x, arrs[0], color=color, linewidth=line_width * 1.5, alpha=0.99, label=gname, zorder=5)

    # 设置参考线（水平与垂直）
    y_min, y_max = ax.get_ylim()
    # 自动选择 y step
    if grid_y_step is None:
        # 尝试根据范围选择合适的步长（简单启发式）
        rng = y_max - y_min if (y_max - y_min) > 0 else 1.0
        grid_y_step = float(np.round(rng / 8, 2))
        if grid_y_step == 0:
            grid_y_step = rng / 8 if rng > 0 else 1.0

    # horizontal lines
    y_start = y_min - 1e-9
    y_end = y_max + 1e-9
    ys = np.arange(y_start, y_end + grid_y_step, grid_y_step)
    for yy in ys:
        ax.axhline(y=yy, color='#cfcfcf', linestyle='--', zorder=0)

    # vertical lines
    for xx in np.arange(0, max_len, grid_x_step):
        ax.axvline(x=xx, color='#cfcfcf', linestyle='--', zorder=0)

    # 优化刻度格式
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.grid(False)  # 使用自定义参考线，不启用默认网格

    # 图例
    ax.legend(loc=legend_loc, fontsize=legend_size)

    # 美学微调：y label 左对齐
    try:
        ax.yaxis.set_label_coords(-0.12, 0.5)
    except Exception:
        pass

    fig.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax

if __name__ == '__main__':
    import numpy as np
    data = [np.sin(np.linspace(0, 4*np.pi, 200)) + 0.1*np.random.randn(200) for _ in range(6)]
    names = ['A1','A2','A3','B','B','B']
    plot_lines(data, names=names, ylabel='Value', xlabel='X', title='example', save_path=None)

