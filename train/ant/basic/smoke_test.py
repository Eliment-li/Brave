import tyro
from gymnasium import register

from train.ant.basic.ant_train import Args, train_and_evaluate
# from stable_baselines3.common.logger import configure
#
# # 彻底禁用 SB3 控制台输出
# configure(folder=None, format_strings=[])
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.task='stand'
    args.reward_mode='rnd'
    args.reward_type = 'sparse'
    args.r_wrapper_ver=5
    args.early_break=False
    args.track=True
    args.ctrl_cost_weight=0.05
    args.total_timesteps=30_0000
    args.max_episode_steps=200
    args.finalize()
    register(id="AntStand-v0", entry_point="envs.ant.ant_tasks:AntStand", max_episode_steps=args.max_episode_steps)
    register(id="AntSpeed-v0", entry_point="envs.ant.ant_tasks:AntSpeed", max_episode_steps=args.max_episode_steps)
    register(id="AntFar-v0", entry_point="envs.ant.ant_tasks:AntFar", max_episode_steps=args.max_episode_steps)
    train_and_evaluate(args)