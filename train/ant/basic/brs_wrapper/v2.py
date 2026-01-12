from application.ant.basic.wrappers.brs_wrapper.v1 import AntBRSRewardWrapperV1, NovelRewardConfig
from typing import Dict, Optional
import gymnasium as gym

class AntBRSRewardWrapperV2(AntBRSRewardWrapperV1):
    """
    Ant bonus wrapper that keeps V1's best-step bonuses while also granting
    a one-time per-episode reward as soon as each threshold is first met.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[NovelRewardConfig] = None,
        initial_records: Optional[Dict[str, int]] = None,
    ):
        super().__init__(env, config=config, initial_records=initial_records)
        self._episode_bonus_paid: set[str] = set()
        self.bouns_cnt = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._episode_bonus_paid.clear()
        return obs, info

    def _maybe_break_record_and_bonus(self, key: str, steps_now: int, weight: float) -> float:
        episode_bonus = 0.0
        if weight > 0 and key not in self._episode_bonus_paid:
            episode_bonus = float(weight)
            self._episode_bonus_paid.add(key)
            self.bouns_cnt +=1
            # if self.bouns_cnt %100==0:
            #     print(f"episode bonus cnt:{self.bouns_cnt} key:{key} step:{steps_now}")
        record_bonus = super()._maybe_break_record_and_bonus(key, steps_now, weight)
        return episode_bonus + record_bonus
