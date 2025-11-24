# 配置系统使用指南 - Configuration System Usage Guide

本文档详细介绍如何使用 Brave 框架的配置系统。

This document provides detailed instructions on how to use the Brave framework's configuration system.

## 目录 (Table of Contents)

1. [快速开始](#快速开始-quick-start)
2. [配置类详解](#配置类详解-configuration-classes-explained)
3. [命令行使用](#命令行使用-command-line-usage)
4. [程序化使用](#程序化使用-programmatic-usage)
5. [扩展新配置](#扩展新配置-extending-with-new-configurations)
6. [最佳实践](#最佳实践-best-practices)
7. [常见问题](#常见问题-faq)

---

## 快速开始 (Quick Start)

### 安装依赖 (Install Dependencies)

```bash
pip install -r requirements.txt
```

### 运行示例 (Run Examples)

```bash
# 查看配置演示
python demo_config.py

# 测试配置系统
python test_config.py

# 运行 Atari 训练（使用默认配置）
python train_atari.py

# 运行 CartPole 训练（使用默认配置）
python train_cartpole.py
```

---

## 配置类详解 (Configuration Classes Explained)

### BaseArgs - 基础配置类

所有实验配置的基类，包含通用参数。

Base class for all experiment configurations, containing common parameters.

**主要参数 (Main Parameters):**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int | 1 | 随机种子 |
| `cuda` | bool | True | 是否使用 CUDA |
| `env_id` | str | "BreakoutNoFrameskip-v4" | 环境 ID |
| `num_envs` | int | 8 | 并行环境数量 |
| `total_timesteps` | int | 10,000,000 | 总训练步数 |
| `learning_rate` | float | 2.5e-4 | 学习率 |
| `gamma` | float | 0.99 | 折扣因子 |
| `gae_lambda` | float | 0.95 | GAE lambda |
| `clip_coef` | float | 0.1 | PPO clip 系数 |
| `ent_coef` | float | 0.01 | 熵系数 |

**完整参数列表：** 运行 `python train_atari.py --help` 查看所有参数

**Full parameter list:** Run `python train_atari.py --help` to see all parameters

### AtariPPOArgs - Atari PPO 配置

专为 Atari 游戏优化的 PPO 配置。

PPO configuration optimized for Atari games.

**关键特性 (Key Features):**
- 8 个并行环境 (8 parallel environments)
- 128 步更新 (128 steps per update)
- 学习率 2.5e-4 (Learning rate 2.5e-4)
- 支持帧堆叠 (Frame stacking support)
- 10M 训练步数 (10M training timesteps)

**使用场景 (Use Cases):**
- Atari 游戏训练 (Atari game training)
- 需要帧堆叠的视觉环境 (Visual environments requiring frame stacking)
- 离散动作空间 (Discrete action spaces)

### CartPolePPOArgs - CartPole PPO 配置

专为 CartPole 经典控制环境优化的 PPO 配置。

PPO configuration optimized for CartPole classic control environment.

**关键特性 (Key Features):**
- 4 个并行环境 (4 parallel environments)
- 256 步更新 (256 steps per update)
- 学习率 2.5e-3（更高）(Learning rate 2.5e-3, higher)
- 500K 训练步数（更少）(500K training timesteps, fewer)
- 零熵系数 (Zero entropy coefficient)

**使用场景 (Use Cases):**
- 快速原型验证 (Quick prototype validation)
- 简单环境训练 (Simple environment training)
- 算法调试 (Algorithm debugging)

---

## 命令行使用 (Command Line Usage)

### 查看帮助 (View Help)

```bash
# 查看所有可配置参数
python train_atari.py --help
python train_cartpole.py --help
```

### 使用默认配置 (Use Default Configuration)

```bash
python train_atari.py
python train_cartpole.py
```

### 覆盖单个参数 (Override Single Parameter)

```bash
# 改变随机种子
python train_atari.py --seed 42

# 改变环境
python train_atari.py --env-id PongNoFrameskip-v4

# 改变学习率
python train_cartpole.py --learning-rate 0.001
```

### 覆盖多个参数 (Override Multiple Parameters)

```bash
python train_atari.py \
    --env-id PongNoFrameskip-v4 \
    --seed 123 \
    --num-envs 16 \
    --learning-rate 1e-4 \
    --total-timesteps 20000000
```

### 布尔参数 (Boolean Parameters)

```bash
# 启用参数（默认为 False 的参数）
python train_atari.py --track              # 启用 wandb 跟踪
python train_atari.py --capture-video      # 启用视频捕获
python train_atari.py --save-model         # 启用模型保存

# 禁用参数（默认为 True 的参数）
python train_atari.py --no-cuda            # 禁用 CUDA
python train_atari.py --no-anneal-lr       # 禁用学习率衰减
python train_atari.py --no-norm-adv        # 禁用优势归一化
```

### 参数命名约定 (Parameter Naming Convention)

支持两种风格：

Two styles are supported:

```bash
# 下划线风格 (Underscore style)
python train_atari.py --total_timesteps 5000000

# 连字符风格（推荐）(Hyphen style - recommended)
python train_atari.py --total-timesteps 5000000
```

### 实验跟踪 (Experiment Tracking)

```bash
# 使用 wandb 跟踪
python train_atari.py \
    --track \
    --wandb-project-name my-atari-project \
    --wandb-entity your-team-name

# 保存模型
python train_atari.py \
    --save-model \
    --save-freq 100000  # 每 100k 步保存一次
```

---

## 程序化使用 (Programmatic Usage)

### 方式 1：使用 tyro 解析命令行 (Method 1: Parse CLI with tyro)

```python
import tyro
from configs.args import AtariPPOArgs

if __name__ == "__main__":
    # tyro 会自动解析命令行参数并创建配置对象
    args = tyro.cli(AtariPPOArgs)
    
    # 使用配置
    print(f"Training on {args.env_id}")
    print(f"Learning rate: {args.learning_rate}")
```

### 方式 2：直接创建配置对象 (Method 2: Create Configuration Object Directly)

```python
from configs.args import CartPolePPOArgs

# 使用默认值
args = CartPolePPOArgs()

# 覆盖特定参数
custom_args = CartPolePPOArgs(
    seed=42,
    num_envs=8,
    total_timesteps=1_000_000,
    learning_rate=0.001
)

# 使用配置
print(f"Environment: {args.env_id}")
print(f"Total timesteps: {custom_args.total_timesteps}")
```

### 方式 3：从字典创建 (Method 3: Create from Dictionary)

```python
from configs.args import AtariPPOArgs

config_dict = {
    "env_id": "PongNoFrameskip-v4",
    "seed": 42,
    "num_envs": 16,
    "learning_rate": 1e-4
}

args = AtariPPOArgs(**config_dict)
```

### 方式 4：配置继承和修改 (Method 4: Configuration Inheritance and Modification)

```python
from dataclasses import dataclass, replace
from configs.args import AtariPPOArgs

# 基于现有配置创建变体
base_args = AtariPPOArgs()

# 创建修改后的副本
modified_args = replace(
    base_args,
    env_id="PongNoFrameskip-v4",
    learning_rate=1e-3
)
```

---

## 扩展新配置 (Extending with New Configurations)

### 示例 1：添加新环境配置 (Example 1: Add New Environment Configuration)

```python
from dataclasses import dataclass
from configs.args import BaseArgs

@dataclass
class MountainCarPPOArgs(BaseArgs):
    """MountainCar PPO 配置"""
    
    # 覆盖环境
    env_id: str = "MountainCar-v0"
    
    # MountainCar 特定优化
    num_envs: int = 4
    total_timesteps: int = 300_000
    learning_rate: float = 1e-3
    num_steps: int = 256
```

### 示例 2：添加新算法配置 (Example 2: Add New Algorithm Configuration)

```python
from dataclasses import dataclass
from configs.args import BaseArgs

@dataclass
class AtariDQNArgs(BaseArgs):
    """Atari DQN 配置"""
    
    env_id: str = "BreakoutNoFrameskip-v4"
    num_envs: int = 1  # DQN 通常单环境
    
    # DQN 特有参数
    buffer_size: int = 1_000_000
    learning_starts: int = 80_000
    batch_size: int = 32
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
```

### 示例 3：多级继承 (Example 3: Multi-level Inheritance)

```python
from dataclasses import dataclass
from configs.args import AtariPPOArgs

@dataclass
class PongPPOArgs(AtariPPOArgs):
    """基于 Atari PPO 的 Pong 特定配置"""
    
    env_id: str = "PongNoFrameskip-v4"
    num_envs: int = 4  # Pong 较简单，用更少环境
    total_timesteps: int = 5_000_000  # 需要更少的步数
    learning_rate: float = 3e-4  # 稍高的学习率
```

---

## 最佳实践 (Best Practices)

### 1. 配置组织 (Configuration Organization)

✅ **推荐 (Recommended):**
```python
# configs/args.py - 通用配置
class BaseArgs: ...
class AtariPPOArgs(BaseArgs): ...
class CartPolePPOArgs(BaseArgs): ...

# extended_configs.py - 扩展配置
class MuJoCoPPOArgs(BaseArgs): ...
class AtariDQNArgs(BaseArgs): ...
```

❌ **不推荐 (Not Recommended):**
```python
# 所有配置混在一个超大文件中
# All configurations mixed in one huge file
```

### 2. 参数覆盖 (Parameter Override)

✅ **推荐 (Recommended):**
```python
@dataclass
class MyArgs(BaseArgs):
    # 只覆盖需要改变的参数
    env_id: str = "MyEnv-v1"
    learning_rate: float = 1e-3
```

❌ **不推荐 (Not Recommended):**
```python
@dataclass
class MyArgs(BaseArgs):
    # 重复定义所有参数（即使使用相同值）
    seed: int = 1
    cuda: bool = True
    env_id: str = "MyEnv-v1"
    # ... 大量重复 ...
```

### 3. 类型提示 (Type Hints)

✅ **推荐 (Recommended):**
```python
from typing import Optional

@dataclass
class MyArgs(BaseArgs):
    learning_rate: float = 1e-3
    wandb_entity: Optional[str] = None
```

❌ **不推荐 (Not Recommended):**
```python
@dataclass
class MyArgs(BaseArgs):
    learning_rate = 1e-3  # 缺少类型提示
    wandb_entity = None   # 缺少 Optional
```

### 4. 文档字符串 (Docstrings)

✅ **推荐 (Recommended):**
```python
@dataclass
class MyArgs(BaseArgs):
    """My experiment configuration
    
    Optimized for specific use case...
    """
    
    learning_rate: float = 1e-3
    """Learning rate for optimizer"""
```

### 5. 默认值选择 (Default Value Selection)

✅ **推荐 (Recommended):**
- 使用经过验证的默认超参数
- 参考论文或成功的实验
- 为不同环境/任务选择合适的默认值

❌ **不推荐 (Not Recommended):**
- 随意设置默认值
- 所有实验使用相同的默认值

---

## 常见问题 (FAQ)

### Q1: 如何查看配置对象的所有参数？

```python
from configs.args import AtariPPOArgs
import dataclasses

args = AtariPPOArgs()
print(dataclasses.asdict(args))
```

### Q2: 如何保存和加载配置？

```python
import json
from configs.args import AtariPPOArgs
import dataclasses

# 保存配置
args = AtariPPOArgs(seed=42)
with open('config.json', 'w') as f:
    json.dump(dataclasses.asdict(args), f, indent=2)

# 加载配置
with open('config.json', 'r') as f:
    config_dict = json.load(f)
args = AtariPPOArgs(**config_dict)
```

### Q3: 如何在多个实验中共享配置？

```python
# base_config.py
base_config = {
    "seed": 42,
    "num_envs": 16,
    "learning_rate": 1e-4
}

# experiment1.py
from configs.args import AtariPPOArgs
from base_config import base_config

args = AtariPPOArgs(
    env_id="PongNoFrameskip-v4",
    **base_config
)

# experiment2.py
args = AtariPPOArgs(
    env_id="BreakoutNoFrameskip-v4",
    **base_config
)
```

### Q4: 配置类可以包含方法吗？

✅ 可以，但不推荐在配置类中放复杂逻辑：

```python
@dataclass
class MyArgs(BaseArgs):
    num_steps: int = 128
    num_envs: int = 8
    
    @property
    def batch_size(self) -> int:
        """计算批次大小"""
        return self.num_steps * self.num_envs
```

### Q5: 如何处理环境特定的参数？

创建专门的配置类：

```python
@dataclass
class AtariArgs(BaseArgs):
    """Atari 环境通用配置"""
    frame_stack: int = 4
    noop_max: int = 30
    
@dataclass  
class AtariPPOArgs(AtariArgs):
    """Atari PPO 配置"""
    # PPO 特定参数
    ...
    
@dataclass
class AtariDQNArgs(AtariArgs):
    """Atari DQN 配置"""
    # DQN 特定参数
    ...
```

---

## 更多资源 (More Resources)

- **示例脚本 (Example Scripts):**
  - `demo_config.py` - 配置系统演示
  - `test_config.py` - 配置系统测试
  - `extended_configs.py` - 扩展配置示例

- **主文档 (Main Documentation):**
  - `README.md` - 项目总览
  - `USAGE_GUIDE.md` - 本文档

- **训练脚本 (Training Scripts):**
  - `train_atari.py` - Atari PPO 训练
  - `train_cartpole.py` - CartPole PPO 训练

---

## 反馈与贡献 (Feedback and Contribution)

如有问题或建议，欢迎提交 Issue 或 Pull Request！

If you have any questions or suggestions, feel free to submit an Issue or Pull Request!
