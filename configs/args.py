"""实验配置类 - Experiment Configuration Classes

基于继承/组合模式的实验参数配置，所有配置类使用 dataclass 定义。
Experiment argument configurations based on inheritance/composition pattern, all configuration classes defined using dataclass.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseArgs:
    """基础配置类 - Base Configuration Class
    
    包含所有实验通用的字段，如种子、CUDA 设置、默认环境 ID、训练总步数等。
    Contains all common fields for all experiments, such as seed, CUDA settings, default env_id, total training steps, etc.
    """
    
    # 基础设置 - Basic Settings
    seed: int = 1
    """随机种子 - Random seed"""
    
    cuda: bool = True
    """是否使用 CUDA - Whether to use CUDA"""
    
    torch_deterministic: bool = True
    """是否使用 PyTorch 确定性算法 - Whether to use PyTorch deterministic algorithms"""
    
    # 环境设置 - Environment Settings
    env_id: str = "BreakoutNoFrameskip-v4"
    """环境 ID - Environment ID"""
    
    num_envs: int = 8
    """并行环境数量 - Number of parallel environments"""
    
    # 训练超参数 - Training Hyperparameters
    total_timesteps: int = 10_000_000
    """总训练步数 - Total training timesteps"""
    
    learning_rate: float = 2.5e-4
    """学习率 - Learning rate"""
    
    num_steps: int = 128
    """每次更新的步数 - Number of steps per update"""
    
    anneal_lr: bool = True
    """是否线性衰减学习率 - Whether to anneal learning rate"""
    
    gamma: float = 0.99
    """折扣因子 - Discount factor"""
    
    gae_lambda: float = 0.95
    """GAE lambda 参数 - GAE lambda parameter"""
    
    num_minibatches: int = 4
    """minibatch 数量 - Number of minibatches"""
    
    update_epochs: int = 4
    """每次更新的 epoch 数 - Number of epochs per update"""
    
    # PPO 特定参数 - PPO Specific Parameters
    norm_adv: bool = True
    """是否归一化优势函数 - Whether to normalize advantages"""
    
    clip_coef: float = 0.1
    """PPO clip 系数 - PPO clip coefficient"""
    
    clip_vloss: bool = True
    """是否 clip value loss - Whether to clip value loss"""
    
    ent_coef: float = 0.01
    """熵系数 - Entropy coefficient"""
    
    vf_coef: float = 0.5
    """值函数系数 - Value function coefficient"""
    
    max_grad_norm: float = 0.5
    """最大梯度范数 - Maximum gradient norm"""
    
    target_kl: Optional[float] = None
    """目标 KL 散度 - Target KL divergence (optional)"""
    
    # 日志和实验跟踪 - Logging and Experiment Tracking
    track: bool = False
    """是否使用 wandb/swanlab 跟踪 - Whether to track with wandb/swanlab"""
    
    wandb_project_name: str = "Brave"
    """wandb 项目名称 - Wandb project name"""
    
    wandb_entity: Optional[str] = None
    """wandb 实体 - Wandb entity (optional)"""
    
    capture_video: bool = False
    """是否捕获视频 - Whether to capture video"""
    
    # 保存和加载 - Save and Load
    save_model: bool = False
    """是否保存模型 - Whether to save model"""
    
    save_freq: int = 100_000
    """保存频率（步数）- Save frequency (in steps)"""


@dataclass
class AtariPPOArgs(BaseArgs):
    """Atari PPO 配置类 - Atari PPO Configuration Class
    
    专门为 Atari 游戏环境优化的 PPO 算法配置。
    PPO algorithm configuration optimized for Atari game environments.
    """
    
    # 覆盖基础配置中的环境设置
    # Override environment settings from base configuration
    env_id: str = "BreakoutNoFrameskip-v4"
    """Atari 环境 ID - Atari environment ID"""
    
    num_envs: int = 8
    """Atari 并行环境数量 - Number of parallel Atari environments"""
    
    num_steps: int = 128
    """Atari 每次更新的步数 - Number of steps per update for Atari"""
    
    # Atari 特定设置
    # Atari specific settings
    frame_stack: int = 4
    """帧堆叠数量 - Number of frames to stack"""
    
    # Atari PPO 优化的超参数
    # Hyperparameters optimized for Atari PPO
    learning_rate: float = 2.5e-4
    """Atari PPO 学习率 - Learning rate for Atari PPO"""
    
    num_minibatches: int = 4
    """Atari PPO minibatch 数量 - Number of minibatches for Atari PPO"""
    
    update_epochs: int = 4
    """Atari PPO 更新 epoch 数 - Number of update epochs for Atari PPO"""
    
    clip_coef: float = 0.1
    """Atari PPO clip 系数 - PPO clip coefficient for Atari"""
    
    ent_coef: float = 0.01
    """Atari PPO 熵系数 - Entropy coefficient for Atari"""


@dataclass
class CartPolePPOArgs(BaseArgs):
    """CartPole PPO 配置类 - CartPole PPO Configuration Class
    
    专门为 CartPole 经典控制环境优化的 PPO 算法配置。
    PPO algorithm configuration optimized for CartPole classic control environment.
    """
    
    # 覆盖基础配置，为 CartPole 设置合适的默认值
    # Override base configuration with appropriate defaults for CartPole
    env_id: str = "CartPole-v1"
    """CartPole 环境 ID - CartPole environment ID"""
    
    num_envs: int = 4
    """CartPole 并行环境数量 - Number of parallel CartPole environments"""
    
    total_timesteps: int = 500_000
    """CartPole 总训练步数（相比 Atari 较少）- Total training timesteps for CartPole (less than Atari)"""
    
    num_steps: int = 256
    """CartPole 每次更新的步数 - Number of steps per update for CartPole"""
    
    # CartPole 优化的超参数
    # Hyperparameters optimized for CartPole
    learning_rate: float = 2.5e-3
    """CartPole PPO 学习率（相比 Atari 更高）- Learning rate for CartPole PPO (higher than Atari)"""
    
    num_minibatches: int = 2
    """CartPole PPO minibatch 数量 - Number of minibatches for CartPole PPO"""
    
    update_epochs: int = 4
    """CartPole PPO 更新 epoch 数 - Number of update epochs for CartPole PPO"""
    
    clip_coef: float = 0.2
    """CartPole PPO clip 系数 - PPO clip coefficient for CartPole"""
    
    ent_coef: float = 0.0
    """CartPole PPO 熵系数 - Entropy coefficient for CartPole"""
    
    gae_lambda: float = 0.95
    """CartPole GAE lambda 参数 - GAE lambda parameter for CartPole"""
