import gymnasium as gym
import numpy as np

from brs.brs_wrapper import BRSRewardWrapperBaseV2


class HumanoidStandupBRSRewardWrapper(BRSRewardWrapperBaseV2):
    """BRS wrapper for HumanoidStandup.

    metric_fn (bigger is better) =
        height_term + upright_term + low_speed_term

    - height_term: torso_z (bigger is better)
    - upright_term: alignment of torso's local +Z axis with world +Z axis (dot product)
      Range ~[-1, 1], bigger is more upright.
    - low_speed_term: exp(-k * speed) where speed is a combination of linear & angular speed.
      Range (0, 1], bigger is slower/more stable.

    Notes:
    - We intentionally keep this metric single-step (instantaneous) for real-time monitoring/
      reward shaping in BRS.
    - Uses MuJoCo "torso" body (in v5 it is body id 0). We resolve by name for robustness.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        gamma: float = 0.99,
        beta: float = 1.1,
        min_bonus: float = 0.01,
        use_global_max_bonus: bool = False,
        global_bonus: float | None = None,
        # metric weights
        w_height: float = 1.0,
        w_upright: float = 1.0,
        w_stability: float = 1.0,
        # stability shaping
        stability_k: float = 1.0,
        stability_lin_w: float = 1.0,
        stability_ang_w: float = 0.25,
        torso_body_name: str = "torso",
    ):
        self.w_height = float(w_height)
        self.w_upright = float(w_upright)
        self.w_stability = float(w_stability)

        self.stability_k = float(stability_k)
        self.stability_lin_w = float(stability_lin_w)
        self.stability_ang_w = float(stability_ang_w)
        self.torso_body_name = str(torso_body_name)

        def _get_torso_id(e: gym.Env) -> int:
            u = e.unwrapped
            if not hasattr(u, "model"):
                raise RuntimeError("HumanoidStandupBRSRewardWrapper requires a MuJoCo-based env with .model")
            model = u.model
            # mujoco>=2.x: model.body(name).id works; fallback to name2id
            if hasattr(model, "body"):
                try:
                    return int(model.body(self.torso_body_name).id)
                except Exception:
                    pass
            if hasattr(model, "body_name2id"):
                return int(model.body_name2id(self.torso_body_name))
            raise RuntimeError("Cannot resolve torso body id from model")

        def metric_fn(e: gym.Env) -> float:
            u = e.unwrapped
            data = getattr(u, "data", None)
            if data is None:
                raise RuntimeError("HumanoidStandupBRSRewardWrapper requires env.unwrapped.data")
            return float(self.w_height * data.qpos[2])

        def info_fn(e: gym.Env, metric: float, metric_max: float):
            u = e.unwrapped
            data = getattr(u, "data", None)
            model = getattr(u, "model", None)
            if data is None or model is None:
                return {
                   # "pos_after": float(metric),
                    # "humanoid_metric_max": float(metric_max),
                }

            # recompute terms for logging
            try:
                torso_id = int(model.body(self.torso_body_name).id) if hasattr(model, "body") else int(model.body_name2id(self.torso_body_name))
                torso_z = float(data.xpos[torso_id][2])
                xmat = np.asarray(data.xmat[torso_id], dtype=np.float32).reshape(3, 3)
                upright = float(np.clip(xmat[:, 2][2], -1.0, 1.0))
                lin_vel = np.asarray(data.cvel[torso_id][:3], dtype=np.float32)
                ang_vel = np.asarray(data.cvel[torso_id][3:6], dtype=np.float32)
                speed = float(self.stability_lin_w * np.linalg.norm(lin_vel) + self.stability_ang_w * np.linalg.norm(ang_vel))
                stability = float(np.exp(-self.stability_k * speed))
            except Exception:
                torso_z, upright, speed, stability = np.nan, np.nan, np.nan, np.nan

            return {
                #"pos_after": float(metric),
                "humanoid_height_torso_z": float(torso_z),
                "humanoid_upright": float(upright),
                "humanoid_speed": float(speed),
                "humanoid_stability": float(stability),
                # "humanoid_metric_max": float(metric_max),
            }

        super().__init__(
            env=env,
            metric_fn=metric_fn,
            metric_name="pos_after",
            info_fn=info_fn,
            gamma=gamma,
            beta=beta,
            min_bonus=min_bonus,
            use_global_max_bonus=use_global_max_bonus,
            global_bonus=global_bonus,
        )
