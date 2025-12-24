import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import swanlab
from configs.base_args import get_root_path
from configs.dqn_args import DqnArgs

def one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    duration = max(duration, 1)
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class ProcessObsInputEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = None
        if isinstance(self.env.observation_space, Discrete):
            self.n = self.env.observation_space.n
            self.observation_space = Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

    def observation(self, obs):
        if self.n is not None:
            return one_hot(int(obs), self.n)
        return obs

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(s_next), np.array(done, dtype=np.float32)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)

def main():
    args = DqnArgs().finalize()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        tb_dir = Path(args.root_path) / "results" / "runs" / run_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            group=args.swanlab_group,
            config=asdict(args),
            experiment_name=args.experiment_name,
            settings=swanlab.Settings(backup=False),
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = ProcessObsInputEnv(gym.make(args.env_id, render_mode="rgb_array"))
    if args.enable_brave:
        from bak.ppo.brs_mountaincar_wrapper import BRSRewardWrapper
        env = BRSRewardWrapper(env)
    env = RecordEpisodeStatistics(env)
    if args.capture_video:
        video_dir = Path(get_root_path()) / "results" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, video_dir.as_posix(), episode_trigger=lambda episode_id: episode_id % 10 == 0)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    obs_shape = int(np.array(env.observation_space.shape).prod())
    action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(args.buffer_size)
    q_network = QNetwork(obs_shape, action_dim).to(device)
    target_network = QNetwork(obs_shape, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    best_episode_return = -float("inf")

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)
            q_values = q_network(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.put((obs, action, reward, next_obs, float(done)))

        obs = next_obs
        episode_reward += reward

        if (
            global_step >= args.learning_starts
            and global_step % args.train_frequency == 0
            and len(replay_buffer) >= args.batch_size
        ):
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = replay_buffer.sample(args.batch_size)
            s_obs = torch.tensor(s_obs, dtype=torch.float32, device=device).view(args.batch_size, -1)
            s_next_obses = torch.tensor(s_next_obses, dtype=torch.float32, device=device).view(args.batch_size, -1)
            s_actions = torch.tensor(s_actions, dtype=torch.long, device=device)
            s_rewards = torch.tensor(s_rewards, dtype=torch.float32, device=device)
            s_dones = torch.tensor(s_dones, dtype=torch.float32, device=device)

            with torch.no_grad():
                target_max = target_network(s_next_obses).max(dim=1)[0]
                td_target = s_rewards + args.gamma * target_max * (1 - s_dones)

            current_q = q_network(s_obs).gather(1, s_actions.view(-1, 1)).squeeze(1)
            loss = loss_fn(td_target, current_q)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
            optimizer.step()

            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            if args.track  and global_step % 100 == 0:

                swanlab.log({"losses/td_loss": loss.item()}, step=global_step)

        if done:
            if args.track :
                swanlab.log(
                    {
                        "charts/episode_reward": episode_reward,
                        "charts/epsilon": epsilon,
                    },
                    step=global_step,
                )

            if episode_reward > best_episode_return:
                best_episode_return = episode_reward
                checkpoint_path = args.checkpoint_dir / f"{args.experiment_name}_step{global_step}.pt"
                torch.save(
                    {
                        "global_step": global_step,
                        "episode_reward": episode_reward,
                        "model_state_dict": q_network.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": asdict(args),
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint: {checkpoint_path}")

            obs, _ = env.reset()
            episode_reward = 0.0

    env.close()
    if args.track:
        swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="repeat train_and_evaluate")
    args = parser.parse_args()
    for _ in range(args.repeat):
        main()
        #make sure all resource closed
        time.sleep(60)
