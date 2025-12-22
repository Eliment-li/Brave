import torch
import torch.nn as nn
from gymnasium import Wrapper, spaces

def _flatten_observation_space(observation_space) -> int:
    # 递归展开组合观测空间，得到总维度
    if isinstance(observation_space, spaces.Dict):
        return sum(_flatten_observation_space(space) for space in observation_space.spaces.values())
    if isinstance(observation_space, spaces.Tuple):
        return sum(_flatten_observation_space(space) for space in observation_space.spaces)
    if isinstance(observation_space, (spaces.Box, spaces.Discrete)):
        return spaces.flatdim(observation_space)
    raise TypeError("Unsupported observation space for RND.")

def _flatten_observation(observation_space, observation):
    # 将原生观测转换为统一的扁平张量，便于输入 MLP
    if isinstance(observation_space, spaces.Dict):
        return torch.cat(
            [
                _flatten_observation(sub_space, observation[key])
                for key, sub_space in observation_space.spaces.items()
            ],
            dim=-1,
        )
    if isinstance(observation_space, spaces.Tuple):
        return torch.cat(
            [
                _flatten_observation(sub_space, obs_part)
                for sub_space, obs_part in zip(observation_space.spaces, observation)
            ],
            dim=-1,
        )
    if isinstance(observation_space, spaces.Box):
        return torch.as_tensor(observation, dtype=torch.float32).view(1, -1)
    if isinstance(observation_space, spaces.Discrete):
        one_hot = torch.zeros(1, observation_space.n, dtype=torch.float32)
        one_hot[0, observation] = 1.0
        return one_hot
    raise TypeError("Unsupported observation space for RND.")

class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers = []
        last_dim = input_dim
        # 构建多层感知器：线性层 + ReLU
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#gymnasium.Wrapper
class RNDRewardWrapper(Wrapper):
    def __init__(
        self,
        env,
        feature_dim: int = 128,
        hidden_sizes: tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-4,
        beta: float = 1.0,
        device: str | None = None,
    ):
        super().__init__(env)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_dim = _flatten_observation_space(env.observation_space)
        # 目标网络参数固定，预测网络可训练
        self.target = _MLP(self.obs_dim, feature_dim, hidden_sizes).to(self.device)
        self.predictor = _MLP(self.obs_dim, feature_dim, hidden_sizes).to(self.device)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.beta = beta

    def reset(self, **kwargs):
        # 重置环境，保持原始返回
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        observation, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        # 将观测扁平化后送入 RND
        obs_tensor = _flatten_observation(self.observation_space, observation).to(self.device)
        with torch.no_grad():
            target_features = self.target(obs_tensor)
        predicted_features = self.predictor(obs_tensor)
        # 预测误差即内在奖励
        intrinsic_reward = torch.mean((predicted_features - target_features) ** 2, dim=-1)
        loss = intrinsic_reward.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        intrinsic_value = intrinsic_reward.item()
        total_reward = extrinsic_reward + self.beta * intrinsic_value
        # 记录额外信息，便于调试
        info["rnd_intrinsic_reward"] = intrinsic_value
        info["rnd_total_reward"] = total_reward
        return observation, total_reward, terminated, truncated, info