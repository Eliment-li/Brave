import gymnasium_robotics
import tyro

from train.fetch.fetch_train import Args, train_and_evaluate

if __name__ == "__main__":
    args = tyro.cli(Args)

    # 覆写为 smoke test 配置：快速跑通训练->评估->保存->录视频
    args.env_id = "Reach"
    args.task='reach'
    args.total_timesteps = 2000
    args.learning_starts = 1_000
    args.n_eval_episodes = 1
    args.repeat = 1
    args.track = True
    args.reward_mode='brave'
    args.brs_beta=1.1

    # 让 episode 更短，加快 smoke test（<=0 将使用环境默认）
    args.max_episode_steps = 1000

    args.finalize()
    from gymnasium.envs.registration import register
    register(id="Reach", entry_point="envs.fetch.reach:MujocoFetchReachEnv", max_episode_steps=args.max_episode_steps)
    register(id="Push", entry_point="envs.fetch.push:MujocoFetchPushEnv", max_episode_steps=args.max_episode_steps)
    register(id="Slide", entry_point="envs.fetch.slide:MujocoFetchSlideEnv", max_episode_steps=args.max_episode_steps)
    register(id="PickAndPlace", entry_point="envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",max_episode_steps=args.max_episode_steps)

    train_and_evaluate(args)

