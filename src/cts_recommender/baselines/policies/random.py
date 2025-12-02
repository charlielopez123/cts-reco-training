"""
Random baseline policy.

Uniform random selection from candidates - serves as lower bound and sanity check.
"""

from typing import List, Tuple, Dict
import numpy as np

from .base import BasePolicy


class RandomPolicy(BasePolicy):
    """Uniform random baseline - selects candidates uniformly at random"""

    def __init__(self):
        self.rng = None

    def reset(self, seed: int):
        """Initialize random state"""
        self.rng = np.random.RandomState(seed)

    def score_slate(
        self,
        context_features: np.ndarray,
        candidates: List[str],
        candidate_features: Dict[str, np.ndarray],
        value_signals: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Assign random scores to all candidates.

        Returns:
            List of (catalog_id, random_score) tuples, sorted by score descending
        """
        # Generate random scores
        scores = self.rng.rand(len(candidates))

        # Create list of (catalog_id, score) tuples
        scored_slate = list(zip(candidates, scores))

        # Sort by score descending
        scored_slate.sort(key=lambda x: x[1], reverse=True)

        return scored_slate

    def select_action(
        self,
        context_features: np.ndarray,
        candidates: List[str],
        candidate_features: Dict[str, np.ndarray],
        value_signals: Dict[str, Dict[str, float]],
        K: int = 10
    ) -> List[str]:
        """
        Select top-K actions uniformly at random.

        Returns:
            List of K catalog_ids (randomly selected)
        """
        scored_slate = self.score_slate(
            context_features, candidates, candidate_features, value_signals
        )

        # Return top-K catalog_ids
        return [catalog_id for catalog_id, _ in scored_slate[:K]]

    def update(
        self,
        context_features: np.ndarray,
        action: str,
        reward: float,
        action_features: np.ndarray,
        value_signals: Dict[str, float]
    ):
        """
        No-op: Random policy has no parameters to update.
        """
        pass
