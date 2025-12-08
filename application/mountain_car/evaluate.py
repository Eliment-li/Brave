import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from pathlib import Path

from application.mountain_car.dqn_mountain_car import QNetwork, ProcessObsInputEnv
from configs.dqn_args import DqnArgs


def make_eval_env(args: DqnArgs, video_dir: str | None = None):
    env = ProcessObsInputEnv(gym.make(args.env_id, render_mode="rgb_array"))
    if args.enable_brave:
        from core.brs_mountaincar_wrapper import BRSRewardWrapper
        env = BRSRewardWrapper(env)
    env = RecordEpisodeStatistics(env)
    if video_dir is not None:
        env = RecordVideo(env, video_dir, episode_trigger=lambda episode_id: True)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(args.seed)
    return env


def evaluate_checkpoint(
    ckpt_path: str,
    episodes: int = 5,
    seed: int | None = None,
    device: str | None = None,
    save_video: bool = True,
):
    args = DqnArgs().finalize()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() and args.cuda else "cpu"))
    video_dir = None
    if save_video:
        video_dir = Path(args.root_path) / "results" / "videos" / "eval"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_dir = video_dir.as_posix()
    env = make_eval_env(args, video_dir=video_dir)

    obs_dim = int(np.array(env.observation_space.shape).prod())
    action_dim = env.action_space.n
    q_network = QNetwork(obs_dim, action_dim).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    q_network.load_state_dict(checkpoint["model_state_dict"])
    q_network.eval()

    base_seed = seed if seed is not None else args.seed
    returns: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            with torch.no_grad():
                action = torch.argmax(q_network(obs_tensor), dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(ep_return)
        print(f"Episode {ep + 1}: return = {ep_return:.2f}")

    env.close()
    avg_return = float(np.mean(returns))
    print(f"Average return over {episodes} episodes: {avg_return:.2f}")
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_video", action="store_true", help="Do not save evaluation videos")
    args = parser.parse_args()

    evaluate_checkpoint(
        ckpt_path=args.ckpt_path,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        save_video=True,
    )
