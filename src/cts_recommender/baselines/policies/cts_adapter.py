"""
CTS Policy Adapter.

Wraps the main ContextualThompsonSampler model to match the BasePolicy interface
used in baseline evaluation.
"""

from typing import List, Tuple, Dict
import numpy as np

from .base import BasePolicy
from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler


class CTSPolicyAdapter(BasePolicy):
    """
    Adapter to wrap the main CTS model as a BasePolicy.

    Converts between the evaluation interface (catalog_ids, value_signals dicts)
    and the CTS model interface (indices, S_matrix numpy arrays).
    """

    def __init__(self, cts_model: ContextualThompsonSampler, signal_names: List[str]):
        """
        Initialize CTS adapter.

        Args:
            cts_model: Instance of ContextualThompsonSampler
            signal_names: Ordered list of signal names matching CTS model's num_signals
                         e.g., ['audience', 'competition', 'diversity', 'novelty', 'rights']
        """
        self.cts = cts_model
        self.signal_names = signal_names

    def reset(self, seed: int):
        """
        Reset CTS model state.

        Note: CTS model is initialized with its own random_state in __init__.
        For evaluation, create a fresh CTS instance for each seed instead of resetting.
        """
        # The CTS model doesn't have a reset method
        # For multi-seed evaluation, create a fresh CTS instance instead
        pass

    def _build_signal_matrix(
        self,
        candidates: List[str],
        value_signals: Dict[str, Dict[str, float]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert value_signals dict to S_matrix expected by CTS.

        Args:
            candidates: List of catalog_ids
            value_signals: {catalog_id: {signal_name: value}}

        Returns:
            S_matrix: (M candidates, num_signals) array
            catalog_ids: Ordered list matching S_matrix rows
        """
        M = len(candidates)
        num_signals = len(self.signal_names)

        S_matrix = np.zeros((M, num_signals))
        catalog_ids = []

        for i, catalog_id in enumerate(candidates):
            catalog_ids.append(catalog_id)
            signals = value_signals[catalog_id]

            for j, signal_name in enumerate(self.signal_names):
                S_matrix[i, j] = signals.get(signal_name, 0.0)

        return S_matrix, catalog_ids

    def score_slate(
        self,
        context_features: np.ndarray,
        candidates: List[str],
        candidate_features: Dict[str, np.ndarray],
        value_signals: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Score all candidates using CTS model.

        Returns:
            List of (catalog_id, score) tuples, sorted by score descending
        """
        # Convert to S_matrix format
        S_matrix, catalog_ids = self._build_signal_matrix(candidates, value_signals)

        # Get CTS scores (using Thompson sampling)
        _, _, _, scores = self.cts.score_candidates(
            c_t=context_features,
            S_matrix=S_matrix,
            K=len(candidates)  # Get scores for all candidates
        )

        # Create scored slate
        scored_slate = [(catalog_ids[i], scores[i]) for i in range(len(catalog_ids))]

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
        Select top-K actions using CTS model.

        Returns:
            List of K catalog_ids (top-K by CTS score)
        """
        # Convert to S_matrix format
        S_matrix, catalog_ids = self._build_signal_matrix(candidates, value_signals)

        # Get top-K from CTS
        topk_indices, _, _, _ = self.cts.score_candidates(
            c_t=context_features,
            S_matrix=S_matrix,
            K=min(K, len(candidates))
        )

        # Convert indices back to catalog_ids
        topk_catalog_ids = [catalog_ids[i] for i in topk_indices]

        return topk_catalog_ids

    def update(
        self,
        context_features: np.ndarray,
        action: str,
        reward: float,
        action_features: np.ndarray,
        value_signals: Dict[str, float]
    ):
        """
        Update CTS model parameters.

        Args:
            context_features: Context vector
            action: Selected catalog_id
            reward: Binary reward (1 if matched curator, 0 otherwise)
            action_features: Movie features (not used by CTS)
            value_signals: Value signals for selected action
        """
        # Convert value_signals dict to s_chosen array
        s_chosen = np.array([
            value_signals.get(signal_name, 0.0)
            for signal_name in self.signal_names
        ])

        # Update CTS model
        self.cts.update(
            c_t=context_features,
            s_chosen=s_chosen,
            r=reward,
            y_signals=None  # No curator signal guidance in offline replay
        )

    def save(self, path: str):
        """Save CTS model state"""
        self.cts.save(path)

    def load(self, path: str):
        """Load CTS model state"""
        self.cts.load(path)
