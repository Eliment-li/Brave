"""配置系统测试脚本 - Configuration System Test Script

测试配置类的创建、继承和使用。
Test configuration class creation, inheritance, and usage.
"""

from configs.args import BaseArgs, AtariPPOArgs, CartPolePPOArgs


def test_base_args():
    """测试 BaseArgs 基础配置类"""
    print("=" * 60)
    print("测试 BaseArgs (Testing BaseArgs)")
    print("=" * 60)
    
    args = BaseArgs()
    
    print(f"✓ 成功创建 BaseArgs 实例 (Successfully created BaseArgs instance)")
    print(f"  - seed: {args.seed}")
    print(f"  - cuda: {args.cuda}")
    print(f"  - env_id: {args.env_id}")
    print(f"  - total_timesteps: {args.total_timesteps}")
    print(f"  - learning_rate: {args.learning_rate}")
    print(f"  - num_envs: {args.num_envs}")
    print()
    
    # 测试参数覆盖
    custom_args = BaseArgs(seed=42, learning_rate=1e-3)
    print(f"✓ 成功创建自定义 BaseArgs 实例 (Successfully created custom BaseArgs instance)")
    print(f"  - seed: {custom_args.seed} (覆盖为 42)")
    print(f"  - learning_rate: {custom_args.learning_rate} (覆盖为 0.001)")
    print()


def test_atari_ppo_args():
    """测试 AtariPPOArgs Atari PPO 配置类"""
    print("=" * 60)
    print("测试 AtariPPOArgs (Testing AtariPPOArgs)")
    print("=" * 60)
    
    args = AtariPPOArgs()
    
    print(f"✓ 成功创建 AtariPPOArgs 实例 (Successfully created AtariPPOArgs instance)")
    print(f"  - env_id: {args.env_id} (Atari 默认)")
    print(f"  - num_envs: {args.num_envs} (Atari 优化)")
    print(f"  - num_steps: {args.num_steps} (Atari 优化)")
    print(f"  - learning_rate: {args.learning_rate} (Atari 优化)")
    print(f"  - frame_stack: {args.frame_stack} (Atari 特有)")
    print(f"  - clip_coef: {args.clip_coef}")
    print(f"  - ent_coef: {args.ent_coef}")
    print()
    
    # 测试继承
    print(f"✓ 验证继承关系 (Verify inheritance)")
    print(f"  - isinstance(args, AtariPPOArgs): {isinstance(args, AtariPPOArgs)}")
    print(f"  - isinstance(args, BaseArgs): {isinstance(args, BaseArgs)}")
    print()


def test_cartpole_ppo_args():
    """测试 CartPolePPOArgs CartPole PPO 配置类"""
    print("=" * 60)
    print("测试 CartPolePPOArgs (Testing CartPolePPOArgs)")
    print("=" * 60)
    
    args = CartPolePPOArgs()
    
    print(f"✓ 成功创建 CartPolePPOArgs 实例 (Successfully created CartPolePPOArgs instance)")
    print(f"  - env_id: {args.env_id} (CartPole 默认)")
    print(f"  - num_envs: {args.num_envs} (CartPole 优化)")
    print(f"  - total_timesteps: {args.total_timesteps} (CartPole 优化)")
    print(f"  - learning_rate: {args.learning_rate} (CartPole 优化)")
    print(f"  - num_steps: {args.num_steps}")
    print(f"  - clip_coef: {args.clip_coef}")
    print(f"  - ent_coef: {args.ent_coef}")
    print()
    
    # 测试继承
    print(f"✓ 验证继承关系 (Verify inheritance)")
    print(f"  - isinstance(args, CartPolePPOArgs): {isinstance(args, CartPolePPOArgs)}")
    print(f"  - isinstance(args, BaseArgs): {isinstance(args, BaseArgs)}")
    print()


def test_comparison():
    """测试不同配置类的对比"""
    print("=" * 60)
    print("配置对比 (Configuration Comparison)")
    print("=" * 60)
    
    atari = AtariPPOArgs()
    cartpole = CartPolePPOArgs()
    
    print(f"参数对比 (Parameter Comparison):")
    print(f"  {'参数 (Parameter)':<25} {'Atari':<20} {'CartPole':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'env_id':<25} {atari.env_id:<20} {cartpole.env_id:<20}")
    print(f"  {'num_envs':<25} {atari.num_envs:<20} {cartpole.num_envs:<20}")
    print(f"  {'total_timesteps':<25} {atari.total_timesteps:<20} {cartpole.total_timesteps:<20}")
    print(f"  {'learning_rate':<25} {atari.learning_rate:<20} {cartpole.learning_rate:<20}")
    print(f"  {'num_steps':<25} {atari.num_steps:<20} {cartpole.num_steps:<20}")
    print(f"  {'clip_coef':<25} {atari.clip_coef:<20} {cartpole.clip_coef:<20}")
    print(f"  {'ent_coef':<25} {atari.ent_coef:<20} {cartpole.ent_coef:<20}")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("配置系统测试 (Configuration System Tests)")
    print("=" * 60 + "\n")
    
    test_base_args()
    test_atari_ppo_args()
    test_cartpole_ppo_args()
    test_comparison()
    
    print("=" * 60)
    print("✓ 所有测试通过！(All tests passed!)")
    print("=" * 60)


if __name__ == "__main__":
    main()
