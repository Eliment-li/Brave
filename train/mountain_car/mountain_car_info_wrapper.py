
import numpy as np
import gymnasium as gym

class MountainCarRewardStatsWrapper(gym.Wrapper):
    """
    统计 MountainCarContinuous 的 reward 在不同 x 位置上的记录（不累加）。
    - x 从 env.observation_space.low[0] 到 high[0]，按 bin_size=0.001 分箱
    - 每一步把 reward 追加到对应 bin 的列表
    - reset() 不重置统计数据
    - rewards_2d() 返回二维数组: [x_center, reward]（逐条记录展平）
    """

    def __init__(self, env, bin_size: float = 0.001):
        super().__init__(env)
        if bin_size <= 0:
            raise ValueError("bin_size must be > 0")

        self.bin_size = float(bin_size)

        obs_space = getattr(env, "observation_space", None)
        if obs_space is None or not hasattr(obs_space, "low") or not hasattr(obs_space, "high"):
            raise ValueError("env must have a Box-like observation_space with low/high")

        self.x_min = float(obs_space.low[0])
        self.x_max = float(obs_space.high[0])

        n_bins = int(np.floor((self.x_max - self.x_min) / self.bin_size)) + 1
        self.n_bins = max(1, n_bins)

        # 每个 bin 一个 list，存该位置上出现过的 reward（不等长）
        self.rewards_by_bin = [[] for _ in range(self.n_bins)]
        self.count_by_bin = np.zeros(self.n_bins, dtype=np.int64)

        self._last_x = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        self._last_x = float(obs[0])
        return out

    def step(self, action):
        out = self.env.step(action)

        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
        else:
            raise ValueError("Unexpected step() return format")

        x = float(obs[0])
        idx = self._x_to_bin(x)
        r = float(reward)
        self.rewards_by_bin[idx].append(r)
        self.count_by_bin[idx] += 1
        self._last_x = x

        return out

    def _x_to_bin(self, x: float) -> int:
        x_clamped = min(max(x, self.x_min), self.x_max)
        idx = int(np.floor((x_clamped - self.x_min) / self.bin_size))
        if idx < 0:
            return 0
        if idx >= self.n_bins:
            return self.n_bins - 1
        return idx

    def clear_stats(self):
        """显式清空统计数据（reset 不会清）。"""
        for lst in self.rewards_by_bin:
            lst.clear()
        self.count_by_bin.fill(0)

    def rewards_2d(self) -> np.ndarray:
        """
        返回二维数组 shape=(N, 2):
        - 第 0 列: bin 中心点 x
        - 第 1 列: 对应的一条 reward
        其中 N 为所有 bin 的 reward 记录总数（展平）。
        """
        rows = []
        for i, rewards in enumerate(self.rewards_by_bin):
            if not rewards:
                continue
            x_center = self.x_min + (i + 0.5) * self.bin_size
            if x_center > self.x_max:
                x_center = self.x_max
            for r in rewards:
                rows.append((x_center, r))
        if not rows:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def rewards_by_position(self):
        """
        返回不等长结构，便于按位置直接取：
        List[Tuple[x_center, List[reward]]]
        """
        out = []
        for i, rewards in enumerate(self.rewards_by_bin):
            x_center = self.x_min + (i + 0.5) * self.bin_size
            if x_center > self.x_max:
                x_center = self.x_max
            out.append((x_center, list(rewards)))
        return out

    def stats_dict(self):
        """返回便于序列化的统计信息。"""
        return {
            "bin_size": self.bin_size,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "n_bins": self.n_bins,
            "count_by_bin": self.count_by_bin.copy(),
            # 注意：每个 bin 不等长
            "rewards_by_bin": [list(v) for v in self.rewards_by_bin],
        }