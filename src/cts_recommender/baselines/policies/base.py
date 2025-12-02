"""
Base policy interface for baseline evaluation.

All policies (baselines, ablations, and CTS adapter) must implement this interface
to work with the offline replay engine.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np


class BasePolicy(ABC):
    """Base interface for all policies in offline evaluation"""

    @abstractmethod
    def reset(self, seed: int):
        """
        Initialize internal state and set random seed.

        Args:
            seed: Random seed for reproducibility
        """
        pass

    @abstractmethod
    def score_slate(
        self,
        context_features: np.ndarray,
        candidates: List[str],
        candidate_features: Dict[str, np.ndarray],
        value_signals: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Score all candidates.

        Args:
            context_features: 18D context vector
            candidates: List of catalog_ids
            candidate_features: {catalog_id: movie_features}
            value_signals: {catalog_id: {signal_name: value}}

        Returns:
            List of (catalog_id, score) tuples, sorted by score (descending)
        """
        pass

    @abstractmethod
    def select_action(
        self,
        context_features: np.ndarray,
        candidates: List[str],
        candidate_features: Dict[str, np.ndarray],
        value_signals: Dict[str, Dict[str, float]],
        K: int = 10
    ) -> List[str]:
        """
        Select top-K actions.

        Args:
            context_features: 18D context vector
            candidates: List of catalog_ids
            candidate_features: {catalog_id: movie_features}
            value_signals: {catalog_id: {signal_name: value}}
            K: Number of actions to select

        Returns:
            List of catalog_ids (top-K, ordered by score)
        """
        pass

    @abstractmethod
    def update(
        self,
        context_features: np.ndarray,
        action: str,
        reward: float,
        action_features: np.ndarray,
        value_signals: Dict[str, float]
    ):
        """
        Update policy parameters (no-op for non-learning policies).

        Args:
            context_features: 18D context vector
            action: Selected catalog_id
            reward: Binary reward (1 if matched curator, 0 otherwise)
            action_features: Movie features for selected action
            value_signals: Value signals for selected action
        """
        pass

    def save(self, path: str):
        """
        Save policy state (optional).

        Args:
            path: File path to save state
        """
        pass

    def load(self, path: str):
        """
        Load policy state (optional).

        Args:
            path: File path to load state from
        """
        pass
