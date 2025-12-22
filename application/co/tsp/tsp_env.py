from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from application.co.base import CandidateLocalSearchEnv


@dataclass(frozen=True)
class TSP2OptMove:
    i: int
    j: int


class TSP2OptEnv(CandidateLocalSearchEnv):
    """Local-search TSP env using 2-opt moves and a K-candidate action set."""

    def __init__(
        self,
        n: int = 100,
        k: int = 64,
        max_steps: int = 200,
        reward_mode: str = "delta",
        seed: Optional[int] = None,
    ):
        self.n = int(n)
        self.coords: np.ndarray
        self.tour: np.ndarray
        self._dist: np.ndarray
        super().__init__(k=k, max_steps=max_steps, reward_mode=reward_mode, seed=seed)

        # global: step, cost, best, n
        # candidate: delta_est, i_norm, j_norm, span_norm
        self._set_spaces(global_dim=4, candidate_dim=4)

    def _reset_instance(self, options: Dict[str, Any]):
        n = int(options.get("n", self.n))
        self.n = n
        self.coords = self._rng.random((n, 2), dtype=np.float64)
        self._dist = self._pairwise_dist(self.coords)
        self.tour = np.arange(n, dtype=np.int32)
        self._rng.shuffle(self.tour)

    def _pairwise_dist(self, coords: np.ndarray) -> np.ndarray:
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def _current_cost(self) -> float:
        idx = self.tour
        nxt = np.roll(idx, -1)
        return float(np.sum(self._dist[idx, nxt]))

    def _global_features(self) -> np.ndarray:
        return np.asarray([self._step, self._cost, self._best_cost, self.n], dtype=np.float32)

    def _make_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        # Sample K 2-opt moves.
        moves = np.empty((self.k, 2), dtype=np.int32)
        feats = np.empty((self.k, 4), dtype=np.float64)

        for t in range(self.k):
            # ensure i<j and non-identical
            i = int(self._rng.integers(0, self.n))
            j = int(self._rng.integers(0, self.n))
            if i == j:
                j = (j + 1) % self.n
            if i > j:
                i, j = j, i

            delta = self._delta_2opt(i, j)
            moves[t, 0] = i
            moves[t, 1] = j
            span = j - i
            feats[t] = np.asarray(
                [delta, i / max(1, self.n - 1), j / max(1, self.n - 1), span / max(1, self.n - 1)],
                dtype=np.float64,
            )

        return moves, feats

    def _delta_2opt(self, i: int, j: int) -> float:
        # Edges: (i-1)->i and j->(j+1) are removed; (i-1)->j and i->(j+1) are added.
        n = self.n
        tour = self.tour

        a = tour[(i - 1) % n]
        b = tour[i % n]
        c = tour[j % n]
        d = tour[(j + 1) % n]

        before = self._dist[a, b] + self._dist[c, d]
        after = self._dist[a, c] + self._dist[b, d]
        return float(after - before)

    def _apply_candidate(self, cand: Any):
        i = int(cand[0])
        j = int(cand[1])
        if i == j:
            return
        if i > j:
            i, j = j, i
        # Reverse segment [i, j]
        self.tour[i : j + 1] = self.tour[i : j + 1][::-1]
