import tyro

from application.ant.basic.ant_train import Args, train_and_evaluate
# from stable_baselines3.common.logger import configure
#
# # 彻底禁用 SB3 控制台输出
# configure(folder=None, format_strings=[])
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.task='speed'
    args.reward_mode='brave'
    args.reward_type = 'dense'
    args.r_wrapper_ver=4
    args.track=False
    args.finalize()
    train_and_evaluate(args)