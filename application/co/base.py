from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class StepInfo:
    cost: float
    best_cost: float
    delta: float
    improved: bool


class CandidateLocalSearchEnv(gym.Env):
    """Base class for local-search environments with a fixed candidate set.

    Action space is `Discrete(K)` where each action picks one of K *valid* moves.

    Observation is a `spaces.Dict` with:
      - global: shape (G,)
      - candidates: shape (K, C)

    Subclasses implement:
      - _reset_instance
      - _current_cost
      - _make_candidates
      - _apply_candidate

    Reward is delegated to a reward_fn: reward_fn(cost_t, cost_tp1, best_cost_before, best_cost_after, step, done)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        k: int = None,
        max_steps: int = None,
        reward_mode: str = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode
        self.potential_gamma = 1.0

        self._rng = np.random.default_rng(seed)

        # Placeholders; subclasses should call _set_spaces after they know dims.
        self.observation_space: spaces.Space
        self.action_space = spaces.Discrete(self.k)

        self._step = 0
        self._cost = np.inf
        self._best_cost = np.inf
        self.rdcr = 0.0
        self.rdcr_max = 0.0
        self.gamma = 0.99

    # --- Gymnasium API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step = 0
        self._reset_instance(options=options or {})
        self._cost = float(self._current_cost())
        if self._best_cost == np.inf:
            self._best_cost = float(self._cost)

        obs = self._get_obs()
        info = {"cost": self._cost,
                "best_cost": self._best_cost,
                "rdcr": 0.0,
                "rdcr_max": 0.0
                }
        self.rdcr = 0.0
        self.rdcr_max = 0.0
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        candidates, _ = self._make_candidates()
        cand = candidates[int(action)]

        cost_before = self._cost
        best_before = self._best_cost

        self._apply_candidate(cand)
        cost_after = float(self._current_cost())
        self._cost = cost_after
        self._best_cost = float(min(self._best_cost, self._cost))

        delta = cost_after - cost_before
        improved = cost_after < cost_before

        self._step += 1
        terminated = False
        truncated = self._step >= self.max_steps

        info = {
            "cost": cost_after,
            "best_cost": self._best_cost,
            "delta": delta,
            "improved": improved,
        }
        reward = float(
            self._compute_reward(
                cost_before=cost_before,
                cost_after=cost_after,
                best_before=best_before,
                best_after=self._best_cost,
                delta=delta,
                step=self._step,
                done=terminated or truncated,
                info=info,
            )
        )

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    # --- Reward shaping ---
    def _compute_reward(
        self,
        *,
        cost_before: float,
        cost_after: float,
        best_before: float,
        best_after: float,
        delta: float,
        step: int,
        done: bool,
        info: Dict[str, Any],
    ) -> float:
        # Minimization convention: lower cost is better.
        if self.reward_mode == "delta":
            return -delta

        if self.reward_mode == "terminal":
            return -cost_after if done else 0.0

        if self.reward_mode == "improve_only":
            return float(max(0.0, -delta))

        if self.reward_mode == "normalized_delta":
            denom = max(1e-9, abs(best_before))
            return -delta / denom

        if self.reward_mode == "brave":
            # Reward = -delta, but if we set a new best, add a bonus proportional to improvement.
            base = -delta
            bonus = 0.0
            if best_after < best_before:
                # improvement = best_before - best_after
                # bonus = 1.0 + improvement / max(1e-9, abs(best_before))
                bonus = 1.000001 * (self.rdcr_max - self.gamma * self.rdcr) + 0.5
                self.rdcr = self.gamma * self.rdcr + bonus
                assert self.rdcr > self.rdcr_max, f"rdcr did not increase: {self.rdcr} <= {self.rdcr_max}"
                return  bonus
            else:
                self.rdcr = self.gamma * self.rdcr + base
                return base
            info["brs_bonus"] = bonus
            info["rdcr"] = self.rdcr
            info["rdcr_max"] = self.rdcr_max


        if self.reward_mode == "potential":
            gamma = getattr(self, "potential_gamma", 1.0)
            base = -delta
            phi_before = -best_before
            phi_after = -best_after
            return base + gamma * phi_after - phi_before

        raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    # --- Observation ---
    def _set_spaces(self, global_dim: int, candidate_dim: int):
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=-np.inf, high=np.inf, shape=(global_dim,), dtype=np.float32),
                "candidates": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.k, candidate_dim), dtype=np.float32
                ),
            }
        )

    def _get_obs(self) -> Dict[str, np.ndarray]:
        candidates, cand_features = self._make_candidates()
        global_features = self._global_features().astype(np.float32)
        cand_features = cand_features.astype(np.float32)

        # Guard: enforce fixed shapes.
        if cand_features.shape[0] != self.k:
            raise RuntimeError(f"Candidate feature count mismatch: {cand_features.shape[0]} vs K={self.k}")

        return {"global": global_features, "candidates": cand_features}

    # --- Hooks for subclasses ---
    def _reset_instance(self, options: Dict[str, Any]):
        raise NotImplementedError

    def _current_cost(self) -> float:
        raise NotImplementedError

    def _global_features(self) -> np.ndarray:
        # Default: step, current cost, best cost
        return np.asarray([self._step, self._cost, self._best_cost], dtype=np.float32)

    def _make_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (candidates, features).

        candidates: array-like of length K, opaque dtype per-env.
        features: float array with shape (K, C).
        """
        raise NotImplementedError

    def _apply_candidate(self, cand: Any):
        raise NotImplementedError
