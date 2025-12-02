"""
Static scalarization baseline policy.

Uses value signals with a fixed global weight vector - no context-dependence,
no bandit learning.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from .base import BasePolicy


class StaticScalarizationPolicy(BasePolicy):
    """
    Static global scalarization baseline.

    Scores candidates using fixed weights: u(a, c_t) = w_0^T φ(a, c_t)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize static scalarization policy.

        Args:
            weights: Dict mapping signal names to weights (should sum to ~1.0)
                    If None, uses equal weights for all signals
        """
        self.weights = weights
        self.rng = None

    def reset(self, seed: int):
        """Initialize random state (not used, but required by interface)"""
        self.rng = np.random.RandomState(seed)

    def score_slate(
        self,
        context_features: np.ndarray,
        candidates: List[str],
        candidate_features: Dict[str, np.ndarray],
        value_signals: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Score candidates using fixed scalarization: u(a) = w_0^T φ(a).

        Returns:
            List of (catalog_id, score) tuples, sorted by score descending
        """
        scored_slate = []

        for catalog_id in candidates:
            signals = value_signals[catalog_id]

            # If no weights provided, use equal weights
            if self.weights is None:
                score = np.mean(list(signals.values()))
            else:
                # Weighted sum: w_0^T φ(a)
                score = sum(
                    self.weights.get(signal_name, 0.0) * signal_value
                    for signal_name, signal_value in signals.items()
                )

            scored_slate.append((catalog_id, score))

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
        Select top-K actions using static scalarization.

        Returns:
            List of K catalog_ids (top-K by scalarized score)
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
        No-op: Static policy has no parameters to update.
        """
        pass

    def set_weights(self, weights: Dict[str, float]):
        """
        Update the weight vector (useful for testing different weight configurations).

        Args:
            weights: Dict mapping signal names to weights
        """
        self.weights = weights
