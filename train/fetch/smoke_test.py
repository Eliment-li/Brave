import gymnasium_robotics
import tyro

from train.fetch.fetch_train import Args, train_and_evaluate

if __name__ == "__main__":
    args = tyro.cli(Args)

    # 覆写为 smoke test 配置：快速跑通训练->评估->保存->录视频
    args.env_id = "FetchReach-v4"
    args.total_timesteps = 5_000
    args.learning_starts = 1_000
    args.n_eval_episodes = 1
    args.repeat = 1
    args.track = False
    args.seed = 0

    # 让 episode 更短，加快 smoke test（<=0 将使用环境默认）
    args.max_episode_steps = 50

    args.finalize()
    from gymnasium.envs.registration import register, register_envs
    register_envs(gymnasium_robotics)
    train_and_evaluate(args)

