"""
Ranking evaluation metrics for baseline comparison.

Implements Hit@K and NDCG@K metrics that work with both relevance modes:
- Historical-Choice: R_t = {a*_t} (strict - curator's exact choice)
- Context-Relevant: R_t = {all appropriate items} (relaxed - expert rules)
"""

from typing import List, Set
import numpy as np


def hit_at_k(
    predictions: List[List[str]],
    relevant_sets: List[Set[str]],
    K: int
) -> float:
    """
    Compute Hit@K metric (works for both relevance modes).

    Measures how often at least one relevant item appears in the top-K slate.

    Args:
        predictions: List of top-K slates (one per decision)
                    Each slate is a list of catalog_ids
        relevant_sets: List of relevant item sets (one per decision)
                      - Historical-Choice mode: R_t = {a*_t} (single item)
                      - Context-Relevant mode: R_t = {all appropriate items}
        K: Slate size

    Returns:
        Hit@K score (fraction of times at least one relevant item in top-K)
        Range: [0.0, 1.0]

    Example:
        >>> predictions = [['a', 'b', 'c'], ['d', 'e', 'f']]
        >>> relevant_sets = [{'a'}, {'x', 'y'}]
        >>> hit_at_k(predictions, relevant_sets, K=3)
        0.5  # First decision hit, second missed
    """
    if len(predictions) != len(relevant_sets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(relevant_sets)} relevant sets"
        )

    if len(predictions) == 0:
        return 0.0

    hits = 0
    for pred, relevant in zip(predictions, relevant_sets):
        # Check if any relevant item is in top-K
        if len(set(pred[:K]) & relevant) > 0:  # R_t ∩ TopK_t ≠ ∅
            hits += 1

    return hits / len(predictions)


def ndcg_at_k(
    predictions: List[List[str]],
    relevant_sets: List[Set[str]],
    K: int
) -> float:
    """
    Compute NDCG@K metric (works for both relevance modes).

    Measures the quality of ranking by considering the position of relevant items.
    Items ranked higher contribute more to the score.

    Args:
        predictions: List of top-K slates (one per decision)
                    Each slate is a list of catalog_ids (ordered by score)
        relevant_sets: List of relevant item sets (one per decision)
                      - Historical-Choice mode: R_t = {a*_t}
                      - Context-Relevant mode: R_t = {all appropriate items}
        K: Slate size

    Returns:
        NDCG@K score (averaged over all decisions)
        Range: [0.0, 1.0]

    Formula:
        For each decision t:
        - DCG@K = Σ_{i=1}^{K} [I(a_i ∈ R_t) / log₂(1 + i)]
        - IDCG@K = Σ_{i=1}^{L} [1 / log₂(1 + i)] where L = min(K, |R_t|)
        - NDCG@K = DCG@K / IDCG@K

    Example:
        >>> predictions = [['a', 'b', 'c'], ['x', 'd', 'e']]
        >>> relevant_sets = [{'a', 'c'}, {'d', 'e'}]
        >>> ndcg_at_k(predictions, relevant_sets, K=3)
        # First: DCG = 1/log₂(2) + 1/log₂(4) = 1.0 + 0.5 = 1.5
        #        IDCG = 1/log₂(2) + 1/log₂(3) = 1.0 + 0.63 = 1.63
        #        NDCG = 1.5 / 1.63 = 0.92
        # Second: DCG = 1/log₂(3) + 1/log₂(4) = 0.63 + 0.5 = 1.13
        #         IDCG = 1/log₂(2) + 1/log₂(3) = 1.0 + 0.63 = 1.63
        #         NDCG = 1.13 / 1.63 = 0.69
        # Average: (0.92 + 0.69) / 2 = 0.805
    """
    if len(predictions) != len(relevant_sets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(relevant_sets)} relevant sets"
        )

    if len(predictions) == 0:
        return 0.0

    ndcg_scores = []

    for pred, relevant in zip(predictions, relevant_sets):
        if len(relevant) == 0:
            # Skip decisions with no relevant items
            # (Alternative: count as 0.0 - document chosen convention)
            continue

        # Compute DCG@K
        dcg = 0.0
        for i, item in enumerate(pred[:K], start=1):
            if item in relevant:
                # Discount by position (log₂(1 + i))
                dcg += 1.0 / np.log2(1 + i)

        # Compute IDCG@K (ideal DCG - best possible ranking)
        L = min(K, len(relevant))
        idcg = sum(1.0 / np.log2(1 + i) for i in range(1, L + 1))

        # Normalized DCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def precision_at_k(
    predictions: List[List[str]],
    relevant_sets: List[Set[str]],
    K: int
) -> float:
    """
    Compute Precision@K metric (fraction of retrieved items that are relevant).

    Args:
        predictions: List of top-K slates (one per decision)
        relevant_sets: List of relevant item sets (one per decision)
        K: Slate size

    Returns:
        Precision@K score (averaged over all decisions)
        Range: [0.0, 1.0]

    Formula:
        P@K = (1/T) × Σ_t [|R_t ∩ TopK_t| / K]

    Example:
        >>> predictions = [['a', 'b', 'c'], ['d', 'e', 'f']]
        >>> relevant_sets = [{'a', 'c'}, {'d'}]
        >>> precision_at_k(predictions, relevant_sets, K=3)
        # First: 2/3 = 0.667
        # Second: 1/3 = 0.333
        # Average: (0.667 + 0.333) / 2 = 0.5
    """
    if len(predictions) != len(relevant_sets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(relevant_sets)} relevant sets"
        )

    if len(predictions) == 0:
        return 0.0

    precision_scores = []

    for pred, relevant in zip(predictions, relevant_sets):
        # Count relevant items in top-K
        num_relevant_in_topk = len(set(pred[:K]) & relevant)
        precision_scores.append(num_relevant_in_topk / K)

    return np.mean(precision_scores)


def recall_at_k(
    predictions: List[List[str]],
    relevant_sets: List[Set[str]],
    K: int
) -> float:
    """
    Compute Recall@K metric (fraction of relevant items that are retrieved).

    Args:
        predictions: List of top-K slates (one per decision)
        relevant_sets: List of relevant item sets (one per decision)
        K: Slate size

    Returns:
        Recall@K score (averaged over all decisions)
        Range: [0.0, 1.0]

    Formula:
        R@K = (1/T) × Σ_t [|R_t ∩ TopK_t| / |R_t|]

    Example:
        >>> predictions = [['a', 'b', 'c'], ['d', 'e', 'f']]
        >>> relevant_sets = [{'a', 'c', 'x', 'y'}, {'d'}]
        >>> recall_at_k(predictions, relevant_sets, K=3)
        # First: 2/4 = 0.5
        # Second: 1/1 = 1.0
        # Average: (0.5 + 1.0) / 2 = 0.75
    """
    if len(predictions) != len(relevant_sets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(relevant_sets)} relevant sets"
        )

    if len(predictions) == 0:
        return 0.0

    recall_scores = []

    for pred, relevant in zip(predictions, relevant_sets):
        if len(relevant) == 0:
            # Skip decisions with no relevant items
            continue

        # Count relevant items in top-K
        num_relevant_in_topk = len(set(pred[:K]) & relevant)
        recall_scores.append(num_relevant_in_topk / len(relevant))

    return np.mean(recall_scores) if recall_scores else 0.0
