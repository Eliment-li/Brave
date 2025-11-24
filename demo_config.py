"""配置使用演示 - Configuration Usage Demo

演示如何使用配置类，不需要安装完整的依赖。
Demonstrates how to use configuration classes without installing full dependencies.
"""

import tyro
from configs.args import AtariPPOArgs, CartPolePPOArgs


def demo_atari_config():
    """演示 Atari 配置"""
    print("\n" + "=" * 60)
    print("Atari PPO 配置演示 (Atari PPO Configuration Demo)")
    print("=" * 60)
    
    # 方式 1: 使用默认配置
    args = AtariPPOArgs()
    
    print(f"\n使用默认配置 (Using default configuration):")
    print(f"  - 环境 (Environment): {args.env_id}")
    print(f"  - 并行环境数 (Num Envs): {args.num_envs}")
    print(f"  - 总步数 (Total Steps): {args.total_timesteps:,}")
    print(f"  - 学习率 (Learning Rate): {args.learning_rate}")
    print(f"  - 帧堆叠 (Frame Stack): {args.frame_stack}")
    
    # 方式 2: 覆盖部分配置
    custom_args = AtariPPOArgs(
        env_id="PongNoFrameskip-v4",
        seed=42,
        num_envs=16,
        learning_rate=1e-4
    )
    
    print(f"\n使用自定义配置 (Using custom configuration):")
    print(f"  - 环境 (Environment): {custom_args.env_id}")
    print(f"  - 随机种子 (Seed): {custom_args.seed}")
    print(f"  - 并行环境数 (Num Envs): {custom_args.num_envs}")
    print(f"  - 学习率 (Learning Rate): {custom_args.learning_rate}")


def demo_cartpole_config():
    """演示 CartPole 配置"""
    print("\n" + "=" * 60)
    print("CartPole PPO 配置演示 (CartPole PPO Configuration Demo)")
    print("=" * 60)
    
    # 使用默认配置
    args = CartPolePPOArgs()
    
    print(f"\n使用默认配置 (Using default configuration):")
    print(f"  - 环境 (Environment): {args.env_id}")
    print(f"  - 并行环境数 (Num Envs): {args.num_envs}")
    print(f"  - 总步数 (Total Steps): {args.total_timesteps:,}")
    print(f"  - 学习率 (Learning Rate): {args.learning_rate}")
    print(f"  - 熵系数 (Entropy Coef): {args.ent_coef}")


def demo_cli_usage():
    """演示命令行使用"""
    print("\n" + "=" * 60)
    print("命令行使用示例 (CLI Usage Examples)")
    print("=" * 60)
    
    print(f"""
1. 查看帮助 (View help):
   python train_atari.py --help
   python train_cartpole.py --help

2. 使用默认配置 (Use default configuration):
   python train_atari.py
   python train_cartpole.py

3. 覆盖特定参数 (Override specific parameters):
   python train_atari.py --env-id PongNoFrameskip-v4 --seed 42
   python train_cartpole.py --num-envs 8 --learning-rate 0.001

4. 组合多个参数 (Combine multiple parameters):
   python train_atari.py \\
       --env-id PongNoFrameskip-v4 \\
       --seed 123 \\
       --num-envs 16 \\
       --learning-rate 1e-4 \\
       --total-timesteps 20000000

5. 启用实验跟踪 (Enable experiment tracking):
   python train_atari.py --track --wandb-project-name my-project

6. 布尔参数 (Boolean parameters):
   python train_atari.py --no-cuda        # 禁用 CUDA
   python train_atari.py --capture-video  # 启用视频捕获
    """)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Brave 配置系统使用演示")
    print("Brave Configuration System Usage Demo")
    print("=" * 60)
    
    demo_atari_config()
    demo_cartpole_config()
    demo_cli_usage()
    
    print("\n" + "=" * 60)
    print("演示完成！(Demo completed!)")
    print("=" * 60)
    
    print(f"""
下一步 (Next steps):
1. 查看 README.md 了解更多使用方法
   See README.md for more usage information
   
2. 运行训练脚本查看完整配置
   Run training scripts to see full configuration:
   - python train_atari.py --help
   - python train_cartpole.py --help
   
3. 根据需要扩展新的实验配置
   Extend with new experiment configurations as needed
    """)


if __name__ == "__main__":
    main()
