"""
Audience-only baseline policy.

Single-objective baseline that maximizes only audience potential signal.
Demonstrates the value of multi-objective optimization by showing what happens
when we ignore other objectives (diversity, novelty, rights urgency, etc.).
"""

from typing import List, Tuple, Dict
import numpy as np

from .base import BasePolicy


class AudienceOnlyPolicy(BasePolicy):
    """
    Single-objective audience maximization baseline.

    Ranks candidates purely by audience potential signal, ignoring all other objectives.
    """

    def __init__(self, audience_signal_name: str = 'audience'):
        """
        Initialize audience-only policy.

        Args:
            audience_signal_name: Name of audience signal in value_signals dict
                                 (default: 'audience')
        """
        self.audience_signal_name = audience_signal_name
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
        Score candidates using only audience signal: u(a) = φ_audience(a).

        Returns:
            List of (catalog_id, score) tuples, sorted by audience score descending
        """
        scored_slate = []

        for catalog_id in candidates:
            signals = value_signals[catalog_id]

            # Extract audience signal only
            audience_score = signals.get(self.audience_signal_name, 0.0)

            scored_slate.append((catalog_id, audience_score))

        # Sort by audience score descending
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
        Select top-K actions by audience potential only.

        Returns:
            List of K catalog_ids (top-K by audience score)
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
        No-op: Audience-only policy has no parameters to update.
        """
        pass
