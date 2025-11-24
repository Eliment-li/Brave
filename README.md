# Brave - 强化学习实验框架 (Reinforcement Learning Experiment Framework)

这是一个基于"基础配置 + 实验配置"继承模式的强化学习实验框架，使用 dataclass 管理配置参数。

This is a reinforcement learning experiment framework based on "base configuration + experiment configuration" inheritance pattern, using dataclass to manage configuration parameters.

## 项目结构 (Project Structure)

```
Brave/
├── configs/                  # 配置模块目录 (Configuration module directory)
│   ├── __init__.py          # 模块初始化文件 (Module initialization)
│   └── args.py              # 配置类定义 (Configuration class definitions)
├── train_atari.py           # Atari 训练脚本 (Atari training script)
├── train_cartpole.py        # CartPole 训练脚本 (CartPole training script)
├── requirements.txt         # 依赖包列表 (Dependency list)
└── README.md               # 项目文档 (Project documentation)
```

## 配置架构 (Configuration Architecture)

### 设计理念 (Design Philosophy)

采用**继承模式**组织配置类，将配置分为两层：

Using **inheritance pattern** to organize configuration classes, divided into two layers:

1. **BaseArgs** - 基础配置类：包含所有实验通用的参数
   - Base configuration class: Contains parameters common to all experiments

2. **实验特定配置类** - Experiment-specific configuration classes：继承 BaseArgs，覆盖或添加特定实验的参数
   - Inherit from BaseArgs, override or add parameters specific to certain experiments

### 优势 (Advantages)

✅ **可读性好** - 每个实验的配置清晰明了  
✅ **可维护性强** - 通用参数集中管理，减少重复  
✅ **易扩展** - 添加新实验只需继承 BaseArgs 并覆盖特定参数  
✅ **类型安全** - 使用 dataclass 提供类型提示和验证  
✅ **命令行友好** - 与 tyro 等工具无缝集成，支持命令行参数覆盖  

✅ **Good Readability** - Configuration for each experiment is clear  
✅ **High Maintainability** - Common parameters centrally managed, reducing duplication  
✅ **Easy to Extend** - Adding new experiments only requires inheriting BaseArgs and overriding specific parameters  
✅ **Type Safe** - Using dataclass provides type hints and validation  
✅ **CLI Friendly** - Seamlessly integrates with tools like tyro, supporting command line parameter overrides  

## 配置类说明 (Configuration Classes)

### BaseArgs

基础配置类，包含所有实验通用的参数：

Base configuration class containing parameters common to all experiments:

- **基础设置** (Basic Settings): `seed`, `cuda`, `torch_deterministic`
- **环境设置** (Environment Settings): `env_id`, `num_envs`
- **训练超参数** (Training Hyperparameters): `total_timesteps`, `learning_rate`, `num_steps`, `gamma`, `gae_lambda`, etc.
- **PPO 特定参数** (PPO Specific Parameters): `clip_coef`, `ent_coef`, `vf_coef`, etc.
- **日志和跟踪** (Logging and Tracking): `track`, `wandb_project_name`, `capture_video`
- **保存和加载** (Save and Load): `save_model`, `save_freq`

### AtariPPOArgs

Atari 游戏环境的 PPO 配置类：

PPO configuration class for Atari game environments:

- 继承自 `BaseArgs`
- 覆盖环境默认为 `BreakoutNoFrameskip-v4`
- 优化的超参数适合 Atari 游戏（8 个并行环境，学习率 2.5e-4 等）
- 新增 `frame_stack` 参数用于帧堆叠

- Inherits from `BaseArgs`
- Overrides default environment to `BreakoutNoFrameskip-v4`
- Optimized hyperparameters for Atari games (8 parallel environments, learning rate 2.5e-4, etc.)
- Adds `frame_stack` parameter for frame stacking

### CartPolePPOArgs

CartPole 经典控制环境的 PPO 配置类：

PPO configuration class for CartPole classic control environment:

- 继承自 `BaseArgs`
- 覆盖环境默认为 `CartPole-v1`
- 较少的总步数（500,000 vs 10,000,000）
- 更高的学习率（2.5e-3 vs 2.5e-4）
- 更少的并行环境（4 vs 8）

- Inherits from `BaseArgs`
- Overrides default environment to `CartPole-v1`
- Fewer total timesteps (500,000 vs 10,000,000)
- Higher learning rate (2.5e-3 vs 2.5e-4)
- Fewer parallel environments (4 vs 8)

## 使用方法 (Usage)

### 安装依赖 (Install Dependencies)

```bash
pip install -r requirements.txt
```

### 运行示例 (Run Examples)

#### 1. Atari PPO 训练 (Atari PPO Training)

使用默认配置：

Using default configuration:

```bash
python train_atari.py
```

覆盖特定参数：

Overriding specific parameters:

```bash
python train_atari.py --env-id PongNoFrameskip-v4 --learning-rate 1e-4 --seed 42
```

查看所有可配置参数：

View all configurable parameters:

```bash
python train_atari.py --help
```

#### 2. CartPole PPO 训练 (CartPole PPO Training)

使用默认配置：

Using default configuration:

```bash
python train_cartpole.py
```

覆盖特定参数：

Overriding specific parameters:

```bash
python train_cartpole.py --num-envs 8 --total-timesteps 1000000 --seed 123
```

### 在代码中使用配置 (Using Configuration in Code)

#### 方法 1：导入并使用配置类 (Method 1: Import and Use Configuration Class)

```python
import tyro
from configs.args import AtariPPOArgs

def main(args: AtariPPOArgs):
    print(f"Training on {args.env_id}")
    print(f"Learning rate: {args.learning_rate}")
    # 使用配置进行训练 (Use configuration for training)
    # ...

if __name__ == "__main__":
    args = tyro.cli(AtariPPOArgs)
    main(args)
```

#### 方法 2：程序化创建配置实例 (Method 2: Programmatically Create Configuration Instance)

```python
from configs.args import CartPolePPOArgs

# 使用默认值 (Using default values)
args = CartPolePPOArgs()

# 覆盖特定参数 (Override specific parameters)
args = CartPolePPOArgs(
    seed=42,
    num_envs=8,
    total_timesteps=1_000_000
)

# 使用配置 (Use configuration)
print(f"Environment: {args.env_id}")
print(f"Learning rate: {args.learning_rate}")
```

## 扩展新实验 (Extending with New Experiments)

添加新实验配置非常简单，只需继承 `BaseArgs` 并覆盖或添加特定参数：

Adding new experiment configurations is simple, just inherit from `BaseArgs` and override or add specific parameters:

### 示例：添加 MuJoCo 环境配置 (Example: Adding MuJoCo Environment Configuration)

在 `configs/args.py` 中添加：

Add to `configs/args.py`:

```python
@dataclass
class MuJoCoPPOArgs(BaseArgs):
    """MuJoCo PPO 配置类 - MuJoCo PPO Configuration Class"""
    
    # 覆盖环境设置 (Override environment settings)
    env_id: str = "HalfCheetah-v4"
    num_envs: int = 1
    
    # MuJoCo 特定的总步数 (MuJoCo specific total timesteps)
    total_timesteps: int = 1_000_000
    
    # MuJoCo 优化的学习率 (Optimized learning rate for MuJoCo)
    learning_rate: float = 3e-4
    
    # 其他 MuJoCo 特定参数 (Other MuJoCo specific parameters)
    num_steps: int = 2048
    num_minibatches: int = 32
```

创建新的训练脚本 `train_mujoco.py`:

Create new training script `train_mujoco.py`:

```python
import tyro
from configs.args import MuJoCoPPOArgs

def main(args: MuJoCoPPOArgs):
    # 训练逻辑 (Training logic)
    pass

if __name__ == "__main__":
    args = tyro.cli(MuJoCoPPOArgs)
    main(args)
```

## 配置文件组织建议 (Configuration File Organization Recommendations)

当实验数量增多时，可以进一步细分配置文件：

When the number of experiments increases, you can further subdivide configuration files:

```
configs/
├── __init__.py
├── base.py              # BaseArgs
├── atari.py            # AtariPPOArgs, AtariDQNArgs, etc.
├── classic_control.py  # CartPolePPOArgs, MountainCarArgs, etc.
└── mujoco.py          # MuJoCoPPOArgs, etc.
```

## 命令行参数覆盖示例 (Command Line Parameter Override Examples)

tyro 支持灵活的命令行参数覆盖：

tyro supports flexible command line parameter overrides:

```bash
# 基础参数覆盖 (Basic parameter override)
python train_atari.py --seed 42 --cuda False

# 使用下划线或连字符 (Using underscores or hyphens)
python train_atari.py --total-timesteps 5000000
python train_atari.py --total_timesteps 5000000

# 布尔值参数 (Boolean parameters)
python train_atari.py --track  # 启用 wandb 跟踪 (Enable wandb tracking)
python train_atari.py --no-anneal-lr  # 禁用学习率衰减 (Disable learning rate annealing)

# 组合多个参数 (Combining multiple parameters)
python train_atari.py \
    --env-id PongNoFrameskip-v4 \
    --seed 123 \
    --num-envs 16 \
    --learning-rate 1e-4 \
    --total-timesteps 20000000
```

## 最佳实践 (Best Practices)

1. **只在派生类中覆盖必要的参数** - 保持 BaseArgs 的默认值作为通用默认值
   - Only override necessary parameters in derived classes - Keep BaseArgs defaults as common defaults

2. **使用类型提示** - 确保所有字段都有明确的类型
   - Use type hints - Ensure all fields have explicit types

3. **添加文档字符串** - 为每个配置类和重要字段添加说明
   - Add docstrings - Add descriptions for each configuration class and important fields

4. **保持配置简洁** - 避免在配置类中添加业务逻辑
   - Keep configurations concise - Avoid adding business logic in configuration classes

5. **使用合理的默认值** - 为每个实验选择经过验证的默认超参数
   - Use reasonable defaults - Choose validated default hyperparameters for each experiment

## 许可证 (License)

MIT License

## 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！

Welcome to submit Issues and Pull Requests!
