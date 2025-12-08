import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from configs.ppo_args import PpoAtariArgs
from utils.model.checkpoint import load_checkpoint
from application.ppo_mountain_car import Agent as TrainAgent, make_env


def make_eval_envs(env_id: str, run_name: str = "eval_dummy"):
    # 与训练时相同的封装，只是 num_envs=1，且不录视频
    return gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, True, run_name,0.99)]
    )


class EvalAgent(TrainAgent):
    def act(self, obs_tensor: torch.Tensor) -> int:
        # obs_tensor 形状应为 [B, 4, 84, 84]
        with torch.no_grad():
            action, _, _, _ = self.get_action_and_value(obs_tensor)
        return int(action.cpu().numpy()[0])


def agent_factory(device: torch.device):
    envs = make_eval_envs('MountainCar-v0')
    agent = EvalAgent(envs).to(device)
    return agent


def optimizer_factory(agent):
    return optim.Adam(agent.parameters(), lr=2.5e-4)


def evaluate_model(
    ckpt_path: str,
    episodes: int = 5,
    seed: int = 0,
    device: str | None = None,
):
    args = PpoAtariArgs().finalize()
    args.env_id ='MountainCar-v0'
    device = torch.device(device or ("cuda" if torch.cuda.is_available() and args.cuda else "cpu"))

    # 创建与训练一致的 envs（这里只用 1 个 env）
    envs = make_eval_envs(args.env_id, run_name="eval_run")

    agent = EvalAgent(envs).to(device)
    optimizer = optimizer_factory(agent)
    load_checkpoint(ckpt_path, agent, optimizer, map_location=device)
    agent.eval()

    returns = []
    for ep in range(episodes):
        obs, _ = envs.reset(seed=seed + ep)
        # obs 形状: [1, 4, 84, 84]
        done = False
        ep_return = 0.0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            # obs_tensor 形状仍是 [1, 4, 84, 84]，直接喂给模型
            action = agent.act(obs_tensor)
            obs, reward, terminations, truncations, infos = envs.step([action])
            done = bool(terminations[0] or truncations[0])
            ep_return += float(reward[0])

        returns.append(ep_return)
        print(f"Episode {ep + 1}: return = {ep_return:.2f}")

    envs.close()
    avg_return = sum(returns) / len(returns)
    print(f"Average return over {episodes} episodes: {avg_return:.2f}")
    return returns


if __name__ == "__main__":
    ckpt_path = r"D:\project\Brave\results\checkpoints\MountainCar-v0__ppo_atari__3851__1764768735\checkpoint_iter78_step9856.pt"
    evaluate_model(
        ckpt_path=ckpt_path,
        episodes=6,
        seed=3851,
    )
