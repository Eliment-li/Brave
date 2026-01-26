import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

from utils.file.csv_util import to_dataframe
from utils.file.file_util import get_root_dir

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# from matplotlib import font_manager
#
# # 列出可用字体
# available_fonts = sorted([f.name for f in font_manager.fontManager.ttflist])
# print(available_fonts)
legend_size=20
mpl.rcParams['font.size'] = 24
label_size = 26
line_width = 1.5
v6=[]
v7=[]
dpi = 500
# Custom formatter function
def time_formatter(x,pos):
    if x <3600:
        # Format as hours and minutes if more than 60 minutes
        return f'{(x/60).__round__(1)}min'
    else:
        # Format as minutes otherwise
        return f'{(x/3600).__round__(1)}hr'
# 定义格式化函数，将秒转换为小时格式
def seconds_to_hours(x, pos):
    # 将秒转换为小时
    hours = x / 3600
    return f'{hours:.1f}'


fig, axs = plt.subplots(4, 1, figsize=(10,16))
#设置子图上下之间的间距
fig.subplots_adjust(hspace=0.1)

def plot1():
    ax = axs[0]
    data_path = f'assets\\data\\BRB_policy_loss.csv'
    disable_brb,enable_brb = get_brb_policy_loss_data(data_path)

    for i  in range(len(disable_brb)):
        data = disable_brb[i]
        label = 'disable BRB'
        ax.plot(data,color='#1565c0',linewidth=line_width ,alpha=0.25)
    mean1 = np.mean(disable_brb, axis=0)
    ax.plot(mean1,color='#1565c0',linewidth=line_width*1.5, label=label,alpha=0.99)

    for i  in range(len(enable_brb)):
        data = enable_brb[i]
        label = 'enable BRB'
        ax.plot(data,color='#df6172',linewidth=line_width,alpha=0.25)
    mean2 = np.mean(enable_brb, axis=0)
    ax.plot(mean2,color='#df6172',linewidth=line_width*1.5, label=label,alpha=0.99)
    # 设置y轴为对数刻度
    #ax.set_yscale('log',base=10)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.05):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #x轴添加参考线
    for x_coord in np.arange(0, len(mean1), 20):
        ax.axvline(x=x_coord, color='#cfcfcf', linestyle='--', zorder=0 )

    #plt.title('amplitude_estimation')
    #ax.set_xlabel('Training Iteration',fontsize = label_size)
    ax.set_xticklabels([])
    ax.set_ylabel('Policy loss',fontsize = label_size)
    # 5 和  0 对应的是x轴和y轴的坐标值
    ax.text(1, -0.08, 'a)', fontsize=label_size, color='black', transform=ax.transData)

    from matplotlib.ticker import ScalarFormatter
    #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.legend(loc='upper right', fontsize=legend_size)

#total loss
def plot2():
    ax = axs[1]
    data_path = f'assets\\data\\BRB_vf_loss.csv'
    disable_brb,enable_brb = get_brb_policy_loss_data(data_path)
    x = range(0, len(disable_brb[0]))
    for i  in range(len(disable_brb)):
        data = disable_brb[i]
        label='disable BRB'
        ax.plot(x,data,color='#1565c0',linewidth=line_width, alpha=0.25)
    mean1 = np.mean(disable_brb, axis=0)
    ax.plot(x,mean1,color='#1565c0',linewidth=line_width*1.5, label=label,alpha=0.99)
    for i  in range(len(enable_brb)):
        data = enable_brb[i]
        label = 'enable BRB'
        ax.plot(x,data,color='#df6172',linewidth=line_width, alpha=0.25)
    mean2 = np.mean(enable_brb, axis=0)
    ax.plot(x,mean2,color='#df6172',linewidth=line_width*1.5, label=label,alpha=0.99)
    # 设置y轴为对数刻度
    #ax.set_yscale('log',base=10)
    # 5 和  0 对应的是x轴和y轴的坐标值
    ax.text(1, 0.2, 'b)', fontsize=label_size, color='black', transform=ax.transData)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 0.5):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )
        # x轴添加参考线
    for x_coord in np.arange(0, len(mean1), 20):
        ax.axvline(x=x_coord, color='#cfcfcf', linestyle='--', zorder=0)
   # plt.title('amplitude_estimation')
    ax.set_xlabel('',fontsize = label_size)
    ax.set_ylabel('Value Function loss',fontsize = label_size)

    #hide x axis labels
    ax.set_xticklabels([])

   # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.legend(loc='upper right', fontsize=legend_size)

    # ====== 添加局部放大图 ======
    # 1. 创建 inset axes
    axins = inset_axes(ax, width="120%", height="120%",bbox_to_anchor=(0.35, 0.8, 0.35, 0.35),bbox_transform=ax.transAxes, borderpad=2)
    # 设置 inset 边框线宽
    for spine in axins.spines.values():
        spine.set_linewidth(1)  # 设置边框线宽为 2
    # 2. 在 inset 上画同样的数据
    for i in range(len(disable_brb)):
        data = disable_brb[i]
        axins.plot(x, data, color='#1565c0', linewidth=line_width, alpha=0.25)
    axins.plot(x, mean1, color='#1565c0', linewidth=line_width * 2, alpha=0.99)

    for i in range(len(enable_brb)):
        data = enable_brb[i]
        axins.plot(x, data, color='#df6172', linewidth=line_width, alpha=0.25)
    axins.plot(x, mean2, color='#df6172', linewidth=line_width * 2, alpha=0.99)

    # 3. 设置放大区域范围
    x1, x2 = 10,20
    axins.set_xlim(x1, x2)

    # 4. 自动适应y轴
    all_data = np.concatenate([disable_brb, enable_brb], axis=0)
    y_data_in_zoom = all_data[:, x1:x2 + 1]
    y1, y2 = np.min(y_data_in_zoom), np.max(y_data_in_zoom)
    axins.set_ylim(y1, y2)

    # 5. 去掉 inset 的刻度标签
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    # # 6. 画参考线（风格一致）
    # for y_coord in np.arange(y1, y2, 0.2):
    #     axins.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0)

    # 7. 用 mark_inset 连接主图和 inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=1.5)

def get_episode_return_data(path):
    data = to_dataframe(relative_path=path)
    group1 = []
    group2 = []
    # iter by column and get index for each column
    i = 0
    for column in data.columns:
        col_data = data[column].values
        col_data = replace_nan_with_average(col_data)
        if i ==1:
            group1.append(col_data)
        elif i==2:
            group2.append(col_data)
        i += 1
    return group1, group2
def plot4():
    ax = axs[3]
    data_path = f'assets\\data\\episode_return_mean.csv'
    disable_brb, enable_brb = get_brb_policy_loss_data(data_path)
    x = range(0, len(disable_brb[0]))
    for i in range(len(disable_brb)):
        data = disable_brb[i]
        label = 'disable BRB'
        ax.plot(x, data, color='#1565c0', linewidth=line_width*1.5,label=label, alpha=0.99)
    for i in range(len(enable_brb)):
        data = enable_brb[i]
        label = 'enable BRB'
        ax.plot(x, data, color='#df6172', linewidth=line_width*1.5,label=label, alpha=0.99)

    ax.text(1, -1, 'd)', fontsize=label_size, color='black', transform=ax.transData)
    y_min, y_max =  ax.get_ylim()    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(y_min, y_max, 5):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )
    # x轴添加参考线
    for x_coord in np.arange(0, len(disable_brb[0]), 20):
        ax.axvline(x=x_coord, color='#cfcfcf', linestyle='--', zorder=0)
    #plt.title('amplitude_estimation')


    ax.set_xlabel('Training iteration', fontsize=label_size)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    ax.set_ylabel('Sample Time (s)',fontsize = label_size)


    ax.set_ylabel('Episode return ',fontsize = label_size)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.legend(loc='lower right', fontsize=legend_size)

def get_brb_policy_loss_data(path):
    data=to_dataframe(relative_path=path)
    group1=[]
    group2=[]
    #iter by column and get index for each column
    i = 0
    for column in data.columns:
        col_data=data[column].values
        col_data =replace_nan_with_average(col_data)
        if i % 2 == 0:
            group1.append(col_data)
        else:
            group2.append(col_data)
        i+=1
    return group1,group2


def replace_nan_with_average(arr):
    #Replace NaN values in the array with the average of the previous and next values.

    arr = np.array(arr, dtype=float)  # Ensure the input is a numpy array of floats

    # Iterate through the array and replace NaN values
    for i in range(len(arr)):
        if np.isnan(arr[i]) or arr[i]=='nan':
            prev_value = arr[i - 1] if i > 0 else None
            next_value = arr[i + 1] if i < len(arr) - 1 else None

            # Calculate the average of previous and next values
            if prev_value is not None and next_value is not None and not np.isnan(prev_value) and not np.isnan(
                    next_value):
                arr[i] = (prev_value + next_value) / 2
            elif prev_value is not None and not np.isnan(prev_value):  # Only previous value exists
                arr[i] = prev_value
            elif next_value is not None and not np.isnan(next_value):  # Only next value exists
                arr[i] = next_value
            else:  # Both are NaN or missing, leave the value as NaN
                arr[i] = np.nan

    return arr

def get_brb_policy_entropy_data(data_path):
    data = to_dataframe(relative_path=data_path)
    group1 = []
    group2 = []
    # iter by column and get index for each column
    i = 0
    for column in data.columns:
        col_data = data[column].values
        col_data = replace_nan_with_average(col_data)
        if str(column).startswith('disable'):
            group1.append(col_data)
        elif str(column).startswith('enable'):
            group2.append(col_data)
        i += 1
    return group1, group2
def plot3():
    ax = axs[2]
    data_path = f'assets\\data\\policy_entropy.csv'
    disable_brb, enable_brb = get_brb_policy_entropy_data(data_path)
    x = range(0, len(disable_brb[0]))
    for i in range(len(disable_brb)):
        data = disable_brb[i]
        label = 'disable BRB'
        ax.plot(x, data, color='#1565c0', linewidth=line_width, alpha=0.3)
    mean1 = np.mean(disable_brb, axis=0)
    ax.plot(x, mean1, color='#1565c0', linewidth=line_width*2, label=label, alpha=0.99)

    for i in range(len(enable_brb)):
        data = enable_brb[i]
        label = 'enable BRB'
        ax.plot(x, data, color='#df6172', linewidth=line_width, alpha=0.3)
    mean2 = np.mean(enable_brb, axis=0)
    ax.plot(x, mean2, color='#df6172', linewidth=line_width*2, label=label, alpha=0.99)

    ax.text(1, 2.3, 'c)', fontsize=label_size, color='black', transform=ax.transData)

    y_min, y_max =  ax.get_ylim()
    #循环遍历 y 轴坐标值，为每个 y 坐标值添加参考线
    for y_coord in np.arange(2, y_max, 0.5):
        ax.axhline(y=y_coord, color='#cfcfcf', linestyle='--', zorder=0 )
    # x轴添加参考线
    for x_coord in np.arange(0, len(disable_brb[0]), 20):
        ax.axvline(x=x_coord, color='#cfcfcf', linestyle='--', zorder=0)
    #plt.title('amplitude_estimation')
    ax.set_ylabel('Sample Time (s)',fontsize = label_size)

    # ustom formatte
    plt.subplots_adjust(bottom=0.2)

    ax.set_ylabel('Policy entropy',fontsize = label_size)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.legend(loc='upper right', fontsize=legend_size)
 # hide x axis labels
    ax.set_xticklabels([])
# 设置所有子图的 ylabel 左对齐
def align_ylabels_left(axs, label_x_position=-0.1):
    for ax in axs:
        ax.yaxis.set_label_coords(label_x_position, 0.5)  # 设置 y-label 的 x 坐标和 y 坐标

if __name__ == '__main__':
    plot1()
    plot2()
    plot3()
    plot4()

    align_ylabels_left(axs)
    rootdir = get_root_dir()
    path = rootdir + '\\results\\fig\\brb.png'
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.show()

    # print(replace_nan_with_average(group1[0]))