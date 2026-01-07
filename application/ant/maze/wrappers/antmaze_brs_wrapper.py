import gymnasium as gym
import numpy as np

class AntMazeBRSRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        beta: float = 1.1,
        min_bonus: float = 0.01,
    ):
        super().__init__(env)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.min_bonus = float(min_bonus)
        self.metric_max = 0.0
        self.rdcr = 0.0
        self.rdcr_max = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.metric_max = self._current_metric()
        self.rdcr = 0.0
        self.rdcr_max = 0.0
        info = dict(info or {})
        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max
        info["distance_to_goal"] = self._current_distance()
        info["metric_max_distance"] = self.metric_max
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        metric = self._current_metric()
        distance = -metric
        bonus = 0.0
        if metric > self.metric_max:
            self.metric_max = metric
            bonus = self.beta * (self.rdcr_max - self.gamma * self.rdcr) + self.min_bonus
            reward = bonus
            self.rdcr = self.gamma * self.rdcr + bonus
            assert self.rdcr > self.rdcr_max, f"rdcr did not increase: {self.rdcr} <= {self.rdcr_max}"
        else:
            self.rdcr = self.gamma * self.rdcr + reward
        self.rdcr_max = max(self.rdcr_max, self.rdcr)

        info = dict(info or {})
        info["brs_bonus"] = bonus
        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max
        info["distance_to_goal"] = distance
        info["metric_max_distance"] = self.metric_max
        return obs, reward, terminated, truncated, info

    def _current_metric(self) -> float:
        return -self._current_distance()

    def _current_distance(self) -> float:
        agent_xy = self._agent_xy()
        goal_xy = self._goal_xy()
        return float(np.linalg.norm(agent_xy - goal_xy))

    def _agent_xy(self) -> np.ndarray:
        data = self.env.unwrapped.data
        qpos = np.asarray(data.qpos).ravel()
        if qpos.size < 2:
            raise ValueError("AntMaze environment qpos must contain at least 2 elements.")
        return qpos[:2]

    def _goal_xy(self) -> np.ndarray:
        env = self.env.unwrapped
        for attr in ("goal_xy", "goal_pos", "goal", "target_goal"):
            if hasattr(env, attr):
                value = getattr(env, attr)
                if value is not None:
                    arr = np.asarray(value, dtype=np.float64).ravel()
                    if arr.size >= 2:
                        return arr[:2]
        raise AttributeError("Cannot locate goal position on the AntMaze environment.")

