import math
from collections import deque
from sklearn.preprocessing import MinMaxScaler

class ZScoreNormalizer:
    '''
    #test code


    # 示例数据流
    data = [10, 20, 15, 30, 25]

    # 初始化动态归一化器
    normalizer = ZScoreNormalizer()

    # 动态处理数据
    print("Original -> Z-Score Normalized:")
    for x in data:
        z_score = normalizer.update(x)
        print(f"{x} -> {z_score}")

    '''
    def __init__(self):
        self.mean = 0  # 均值
        self.var = 0   # 方差
        self.n = 0     # 数据点数量

    def update(self, x):
        self.n += 1
        old_mean = self.mean
        # 更新均值
        self.mean += (x - self.mean) / self.n
        # 更新方差
        self.var += (x - old_mean) * (x - self.mean)
        # 计算标准差
        std = math.sqrt(self.var / self.n) if self.n > 1 else 0
        # 返回 Z-Score 标准化值
        return (x - self.mean) / std if std > 0 else 0


def normalize_MinMaxScaler(array:list):
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(array)
    return normalized_array



def calculate_conv_output_size(input_size, conv_spec):
    """
                计算卷积层输出的数据维度（支持 padding 参数）

                参数:
                    input_size: 输入图像的尺寸 (height, width)
                    conv_spec: 卷积层配置列表，每个元素为 [过滤器数量, 卷积核大小, 步幅, padding模式]
                              padding模式可以是 'valid' 或 'same'（默认为 'valid'）

                返回:
                    输出数据的维度 (channels, height, width)

                测试代码:
                        # 测试案例1：混合padding模式
                        conv_spec = [
                            [16, 3, 1, 'same'],  # 保持尺寸
                            [32, 3, 2, 'valid'],  # 尺寸减半
                            [64, 5, 1, 'same'],  # 保持尺寸
                        ]
                        input_size = (10, 10)
                        print("\n测试案例1: 混合padding模式")
                        final = CNN.calculate_conv_output_size(input_size, conv_spec)
                        print("最终输出维度:", final)

                        # 测试案例2：全valid模式（兼容旧版）
                        conv_spec_old = [
                            [16, 2, 1, ],  # 过滤器数量，卷积核大小 步幅
                            [32, 3, 1],  # 过滤器数量，卷积核大小 步幅
                            [64, 3, 1],  # 过滤器数量，卷积核大小 步幅
                        ]
                        print("\n测试案例2: 全valid模式")
                        final = CNN.calculate_conv_output_size((10, 10), conv_spec_old)
                        print("最终输出维度:", final)
                """
    channels = 1  # 初始输入通道数（假设是灰度图像）
    height, width = input_size

    for i, layer in enumerate(conv_spec):
        # 解析层配置（兼容旧版缺少padding的情况）
        if len(layer) == 3:
            filters, kernel_size, stride = layer
            padding = 'valid'  # 默认值
        else:
            filters, kernel_size, stride, padding = layer

        # 计算输出高度和宽度
        if padding == 'valid':
            height = (height - kernel_size) // stride + 1
            width = (width - kernel_size) // stride + 1
        elif padding == 'same':
            # 'same' 填充的目标是使输出尺寸 = ceil(输入尺寸 / stride)
            height = (height + stride - 1) // stride
            width = (width + stride - 1) // stride

            # 当 stride=1 时，输出尺寸与输入相同
            if stride == 1:
                height = height
                width = width
        else:
            raise ValueError(f"未知的padding模式: {padding} (必须是 'valid' 或 'same')")

        channels = filters  # 更新通道数为当前层的过滤器数量

        print(f"层 {i + 1} 配置 {layer}: 输出尺寸 ({channels}, {height}, {width})")

    print(f'最终数据维度（通道×高度×宽度）: {channels}×{height}×{width} = {channels * height * width}')
    return channels, height, width




'''
test code
this is for evn to calc rescent dist
    # 测试基本功能
    print("=== 测试基本功能 ===")
    ma = SlideWindow(3)
    print(f"添加1，平均值: {ma.next(1):.2f}")  # 期望: 1.00
    print(f"添加2，平均值: {ma.next(2):.2f}")  # 期望: 1.50
    print(f"添加3，平均值: {ma.next(3):.2f}")  # 期望: 2.00
    print(f"添加4，平均值: {ma.next(4):.2f}")  # 期望: 3.00 (1被移除)

    # 测试重置功能
    print("\n=== 测试重置功能 ===")
    ma.reset()
    print(f"重置后当前平均值: {ma.current_avg:.2f}")  # 期望: 0.00
    print(f"添加5，平均值: {ma.next(5):.2f}")  # 期望: 5.00
    print(f"添加10，平均值: {ma.next(10):.2f}")  # 期望: 7.50

    # 测试边界情况
    print("\n=== 测试边界情况 ===")
    ma = SlideWindow(2)
    print(f"添加0，平均值: {ma.next(0):.2f}")  # 期望: 0.00
    print(f"添加-1，平均值: {ma.next(-1):.2f}")  # 期望: -0.50
    print(f"添加1，平均值: {ma.next(1):.2f}")  # 期望: 0.00 (-1和1)

    # 测试窗口大小为1
    print("\n=== 测试窗口大小为1 ===")
    ma = SlideWindow(1)
    print(f"添加10，平均值: {ma.next(10):.2f}")  # 期望: 10.00
    print(f"添加20，平均值: {ma.next(20):.2f}")  # 期望: 20.00

    # 测试无效窗口大小
    print("\n=== 测试无效窗口大小 ===")
    try:
        ma = SlideWindow(0)
    except ValueError as e:
        print(f"捕获到预期错误: {e}")  # 期望: 窗口大小必须为正整数
'''
class SlideWindow:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.total = 0.0
        self.avg = 0.0

    def next(self, val):
        if len(self.queue) == self.size:
            self.total -= self.queue.popleft()
        self.queue.append(val)
        self.total += val
        return self.current_avg

    def reset(self):
        self.queue.clear()
        self.total = 0.0

    @property
    def current_avg(self):
        if self.queue:
            return self.total / len(self.queue)
        else:
            return 0


if __name__ == '__main__':
    input_size = (18, 18)
    conv_filters = [
        [32, 3, 1],  # 过滤器数量，卷积核大小 步幅
        [64, 3,2],  # 过滤器数量，卷积核大小 步幅
        [128, 3, 2],  # 过滤器数量，卷积核大小 步幅
    ]
    final = calculate_conv_output_size(input_size, conv_filters)