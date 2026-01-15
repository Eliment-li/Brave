import numpy as np
import gymnasium as gym

from brs.brs_wrapper import BRSRewardWrapperBase, BRSRewardWrapperBaseV2


class FetchTaskBRSRewardWrapper(BRSRewardWrapperBaseV2):
    """
    一个 wrapper 覆盖 FetchReach / FetchPush：
    - metric 统一用 goal distance 的相反数（越大越好）
    - task 可显式传入，也可从 env.spec.id 推断
    """

    SUPPORTED_TASKS = {"reach", "push"}

    def __init__(
        self,
        env: gym.Env,
        task: str | None = None,
        gamma: float = 0.99,
        beta: float = 1.1,
        min_bonus: float = 0.01,
        use_global_max_bonus: bool = False,
        global_bonus: float | None = None,
    ):
        self.task = task

        def _get_obs_dict(e: gym.Env):
            # Monitor/VecEnv 之外的常规 gym env：取最近一步/当前观测最可靠的是 step/reset 返回值；
            # 但 wrapper 内拿不到它，所以退而求其次用 env.unwrapped 的内部观测接口时常不统一。
            # 因此这里直接走 env.unwrapped.compute_reward 体系的标准字段：achieved_goal/desired_goal
            # 从 env.unwrapped._get_obs() 拿 dict（Fetch 系列通常具备）。
            u = e.unwrapped
            if hasattr(u, "_get_obs"):
                obs = u._get_obs()
                if isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs:
                    return obs
            # 兜底：尝试从 e 直接拿 last obs（若外部 wrapper 存过）
            obs = getattr(e, "_brs_last_obs", None)
            if isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs:
                return obs
            raise RuntimeError(
                "Cannot access Fetch dict observation with achieved_goal/desired_goal. "
                "Ensure the env is a standard Fetch* goal-based env."
            )

        def metric_fn(e: gym.Env) -> float:
            obs = _get_obs_dict(e)
            ag = np.asarray(obs["achieved_goal"], dtype=np.float32).ravel()
            dg = np.asarray(obs["desired_goal"], dtype=np.float32).ravel()
            dist = float(np.linalg.norm(ag - dg))
            # BRS 假设 “metric 越大越好”，所以取负距离
            return -dist

        def info_fn(e: gym.Env, metric: float, metric_max: float):
            # metric = -dist
            dist = -float(metric)
            return {
                "task": self.task,
                "distance_to_goal": dist,
                self.task: metric,
                f"metric_max_{self.task}": metric_max,
            }

        super().__init__(
            env=env,
            metric_fn=metric_fn,
            metric_name=self.task,
            info_fn=info_fn,
            gamma=gamma,
            beta=beta,
            min_bonus=min_bonus,
            use_global_max_bonus=use_global_max_bonus,
            global_bonus=global_bonus,
        )



class _FetchObsCacheWrapper(gym.Wrapper):
    """
    可选：若你的 Fetch env 不暴露 unwrapped._get_obs()，用这个缓存最近一次 obs，
    供 FetchTaskBRSRewardWrapper 的兜底路径读取。
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        setattr(self, "_brs_last_obs", obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        setattr(self, "_brs_last_obs", obs)
        return obs, reward, terminated, truncated, info

