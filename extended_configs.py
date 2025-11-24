"""扩展示例：添加新实验配置 - Extension Example: Adding New Experiment Configuration

本文件演示如何扩展配置系统，添加新的实验配置类。
This file demonstrates how to extend the configuration system and add new experiment configuration classes.
"""

from dataclasses import dataclass
from configs.args import BaseArgs


# ============================================================================
# 示例 1：添加 MuJoCo 环境配置
# Example 1: Adding MuJoCo Environment Configuration
# ============================================================================

@dataclass
class MuJoCoPPOArgs(BaseArgs):
    """MuJoCo PPO 配置类 - MuJoCo PPO Configuration Class
    
    专门为 MuJoCo 连续控制环境优化的 PPO 算法配置。
    PPO algorithm configuration optimized for MuJoCo continuous control environments.
    """
    
    # 覆盖环境设置 (Override environment settings)
    env_id: str = "HalfCheetah-v4"
    """MuJoCo 环境 ID - MuJoCo environment ID"""
    
    num_envs: int = 1
    """MuJoCo 并行环境数量（通常为 1）- Number of parallel MuJoCo environments (typically 1)"""
    
    # MuJoCo 特定的训练步数
    # MuJoCo specific training steps
    total_timesteps: int = 1_000_000
    """MuJoCo 总训练步数 - Total training timesteps for MuJoCo"""
    
    num_steps: int = 2048
    """MuJoCo 每次更新的步数 - Number of steps per update for MuJoCo"""
    
    # MuJoCo 优化的超参数
    # Hyperparameters optimized for MuJoCo
    learning_rate: float = 3e-4
    """MuJoCo PPO 学习率 - Learning rate for MuJoCo PPO"""
    
    num_minibatches: int = 32
    """MuJoCo PPO minibatch 数量 - Number of minibatches for MuJoCo PPO"""
    
    update_epochs: int = 10
    """MuJoCo PPO 更新 epoch 数 - Number of update epochs for MuJoCo PPO"""
    
    clip_coef: float = 0.2
    """MuJoCo PPO clip 系数 - PPO clip coefficient for MuJoCo"""
    
    ent_coef: float = 0.0
    """MuJoCo PPO 熵系数 - Entropy coefficient for MuJoCo"""
    
    gae_lambda: float = 0.95
    """MuJoCo GAE lambda 参数 - GAE lambda parameter for MuJoCo"""


# ============================================================================
# 示例 2：添加 DQN 算法配置
# Example 2: Adding DQN Algorithm Configuration
# ============================================================================

@dataclass
class AtariDQNArgs(BaseArgs):
    """Atari DQN 配置类 - Atari DQN Configuration Class
    
    专门为 Atari 游戏环境优化的 DQN 算法配置。
    DQN algorithm configuration optimized for Atari game environments.
    """
    
    # 覆盖环境设置
    # Override environment settings
    env_id: str = "BreakoutNoFrameskip-v4"
    """Atari 环境 ID - Atari environment ID"""
    
    num_envs: int = 1
    """DQN 通常使用单个环境 - DQN typically uses a single environment"""
    
    # DQN 特定的训练步数
    # DQN specific training steps
    total_timesteps: int = 10_000_000
    """DQN 总训练步数 - Total training timesteps for DQN"""
    
    # DQN 特有参数
    # DQN specific parameters
    buffer_size: int = 1_000_000
    """经验回放缓冲区大小 - Experience replay buffer size"""
    
    learning_starts: int = 80_000
    """开始学习的步数 - Steps before learning starts"""
    
    train_frequency: int = 4
    """训练频率 - Training frequency"""
    
    target_network_frequency: int = 1_000
    """目标网络更新频率 - Target network update frequency"""
    
    batch_size: int = 32
    """批次大小 - Batch size"""
    
    exploration_fraction: float = 0.1
    """探索衰减比例 - Exploration fraction"""
    
    exploration_initial_eps: float = 1.0
    """初始探索率 - Initial exploration rate"""
    
    exploration_final_eps: float = 0.01
    """最终探索率 - Final exploration rate"""
    
    # DQN 学习率通常较低
    # DQN learning rate is typically lower
    learning_rate: float = 1e-4
    """DQN 学习率 - Learning rate for DQN"""


# ============================================================================
# 示例 3：基于现有配置的变体
# Example 3: Variant Based on Existing Configuration
# ============================================================================

@dataclass
class PongPPOArgs(BaseArgs):
    """Pong PPO 配置类 - Pong PPO Configuration Class
    
    专门为 Pong 游戏优化的 PPO 配置，继承自 BaseArgs 并调整参数。
    PPO configuration optimized specifically for Pong game, inherits from BaseArgs with adjusted parameters.
    """
    
    # 指定 Pong 环境
    # Specify Pong environment
    env_id: str = "PongNoFrameskip-v4"
    """Pong 环境 ID - Pong environment ID"""
    
    # Pong 相对简单，可以用更少的环境和步数
    # Pong is relatively simple, can use fewer environments and steps
    num_envs: int = 4
    """Pong 并行环境数量 - Number of parallel Pong environments"""
    
    total_timesteps: int = 5_000_000
    """Pong 总训练步数（比通用 Atari 少）- Total training timesteps for Pong (less than general Atari)"""
    
    # Pong 可以用稍高的学习率
    # Pong can use slightly higher learning rate
    learning_rate: float = 3e-4
    """Pong PPO 学习率 - Learning rate for Pong PPO"""


# ============================================================================
# 使用示例
# Usage Examples
# ============================================================================

def demo_extended_configs():
    """演示扩展配置的使用"""
    print("=" * 70)
    print("扩展配置示例 (Extended Configuration Examples)")
    print("=" * 70)
    
    # MuJoCo 配置
    print("\n1. MuJoCo PPO 配置 (MuJoCo PPO Configuration):")
    mujoco_args = MuJoCoPPOArgs()
    print(f"   - Environment: {mujoco_args.env_id}")
    print(f"   - Num Steps: {mujoco_args.num_steps}")
    print(f"   - Learning Rate: {mujoco_args.learning_rate}")
    print(f"   - Num Minibatches: {mujoco_args.num_minibatches}")
    
    # Atari DQN 配置
    print("\n2. Atari DQN 配置 (Atari DQN Configuration):")
    dqn_args = AtariDQNArgs()
    print(f"   - Environment: {dqn_args.env_id}")
    print(f"   - Buffer Size: {dqn_args.buffer_size:,}")
    print(f"   - Learning Starts: {dqn_args.learning_starts:,}")
    print(f"   - Batch Size: {dqn_args.batch_size}")
    
    # Pong PPO 配置
    print("\n3. Pong PPO 配置 (Pong PPO Configuration):")
    pong_args = PongPPOArgs()
    print(f"   - Environment: {pong_args.env_id}")
    print(f"   - Num Envs: {pong_args.num_envs}")
    print(f"   - Total Timesteps: {pong_args.total_timesteps:,}")
    print(f"   - Learning Rate: {pong_args.learning_rate}")
    
    print("\n" + "=" * 70)
    print("所有配置都继承自 BaseArgs，共享通用参数！")
    print("All configurations inherit from BaseArgs, sharing common parameters!")
    print("=" * 70)


def demo_usage_in_script():
    """演示如何在训练脚本中使用"""
    print("\n" + "=" * 70)
    print("在训练脚本中使用示例 (Usage in Training Script)")
    print("=" * 70)
    
    print("""
# 文件: train_mujoco.py

import tyro
from extended_configs import MuJoCoPPOArgs

def train_mujoco_ppo(args: MuJoCoPPOArgs):
    # 使用配置进行训练
    print(f"Training on {args.env_id}")
    # ... 训练逻辑 ...
    pass

if __name__ == "__main__":
    # 使用 tyro 解析命令行参数
    args = tyro.cli(MuJoCoPPOArgs)
    train_mujoco_ppo(args)

# 运行示例 (Run examples):
# python train_mujoco.py                           # 使用默认配置
# python train_mujoco.py --env-id Hopper-v4        # 覆盖环境
# python train_mujoco.py --learning-rate 1e-3      # 覆盖学习率
    """)


if __name__ == "__main__":
    demo_extended_configs()
    demo_usage_in_script()
    
    print(f"""
扩展新配置的步骤 (Steps to Extend with New Configuration):
{'=' * 70}

1. 在此文件或 configs/args.py 中定义新的配置类
   Define a new configuration class in this file or configs/args.py
   
   @dataclass
   class YourExperimentArgs(BaseArgs):
       # 覆盖或添加字段
       env_id: str = "YourEnv-v1"
       # ...

2. 创建对应的训练脚本
   Create corresponding training script
   
   # train_your_experiment.py
   import tyro
   from configs.args import YourExperimentArgs
   
   def main(args: YourExperimentArgs):
       # 训练逻辑
       pass
   
   if __name__ == "__main__":
       args = tyro.cli(YourExperimentArgs)
       main(args)

3. 运行训练
   Run training
   
   python train_your_experiment.py --help
   python train_your_experiment.py

{'=' * 70}
    """)
