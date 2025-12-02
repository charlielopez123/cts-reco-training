"""
Value signal aggregation for multi-objective trade-off analysis.

Analyzes how policies make trade-offs between competing objectives
(audience potential, novelty, diversity, rights urgency, competition).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def aggregate_value_signals(
    policy_log: List[Dict[str, Any]],
    signal_names: List[str],
    context_classes: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate value signals by context class.

    Computes mean value signals for selected actions, both globally and
    stratified by context class (Saturday family, Saturday action, etc.).

    Args:
        policy_log: List of decision records, each containing:
            - 'context_class': str (context class name)
            - 'value_signals_top1': Dict[str, float] (signals for top-1 action)
            - 'selected_action': str (catalog_id)
            - 'matched_curator': bool (whether top-1 matched curator choice)
        signal_names: List of signal names to aggregate (e.g., ['audience', 'competition', ...])
        context_classes: Optional list of context class names to include
                        If None, aggregates over all context classes found in log

    Returns:
        Dict mapping context class names (+ 'global') to mean signals:
        {
            'global': {
                'audience': 0.72,
                'competition': 0.45,
                ...
            },
            'saturday_family': {
                'audience': 0.78,
                ...
            },
            ...
        }

    Example:
        >>> from policies.random import RandomPolicy
        >>> # ... run replay, get policy_log ...
        >>> signals = aggregate_value_signals(
        ...     policy_log,
        ...     signal_names=['audience', 'competition', 'diversity', 'novelty', 'rights']
        ... )
        >>> print(f"Global avg audience: {signals['global']['audience']:.3f}")
    """
    if len(policy_log) == 0:
        return {}

    # Initialize result dict
    result = {'global': {signal: [] for signal in signal_names}}

    # Collect signals by context class
    context_signals = {}

    for record in policy_log:
        context_class = record.get('context_class', 'other')
        signals_top1 = record.get('value_signals_top1', {})

        # Add to global
        for signal in signal_names:
            if signal in signals_top1:
                result['global'][signal].append(signals_top1[signal])

        # Add to context-specific
        if context_class not in context_signals:
            context_signals[context_class] = {signal: [] for signal in signal_names}

        for signal in signal_names:
            if signal in signals_top1:
                context_signals[context_class][signal].append(signals_top1[signal])

    # Compute means
    for key in result:
        for signal in signal_names:
            values = result[key][signal]
            result[key][signal] = float(np.mean(values)) if len(values) > 0 else 0.0

    for context_class, signals_dict in context_signals.items():
        result[context_class] = {}
        for signal in signal_names:
            values = signals_dict[signal]
            result[context_class][signal] = float(np.mean(values)) if len(values) > 0 else 0.0

    # Filter by requested context classes if specified
    if context_classes is not None:
        result = {
            k: v for k, v in result.items()
            if k == 'global' or k in context_classes
        }

    return result


def aggregate_topk_value_signals(
    policy_log: List[Dict[str, Any]],
    signal_names: List[str],
    K: int = 10,
    context_classes: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate value signals across entire Top-K slates (not just top-1).

    Useful for understanding the overall quality/characteristics of
    recommendations beyond just the top choice.

    Args:
        policy_log: List of decision records, each containing:
            - 'context_class': str
            - 'value_signals_topk': List[Dict[str, float]] (signals for top-K)
        signal_names: List of signal names to aggregate
        K: Slate size (default: 10)
        context_classes: Optional list of context classes to include

    Returns:
        Dict mapping context class names to mean signals across Top-K:
        {
            'global': {
                'audience': 0.70,  # Mean across all top-K items
                ...
            },
            ...
        }

    Example:
        >>> signals = aggregate_topk_value_signals(policy_log, signal_names, K=10)
        >>> print(f"Avg audience in Top-10: {signals['global']['audience']:.3f}")
    """
    if len(policy_log) == 0:
        return {}

    # Initialize result dict
    result = {'global': {signal: [] for signal in signal_names}}

    # Collect signals by context class
    context_signals = {}

    for record in policy_log:
        context_class = record.get('context_class', 'other')
        signals_topk = record.get('value_signals_topk', [])

        # signals_topk is a list of dicts (one per top-K item)
        for signals_dict in signals_topk[:K]:
            # Add to global
            for signal in signal_names:
                if signal in signals_dict:
                    result['global'][signal].append(signals_dict[signal])

            # Add to context-specific
            if context_class not in context_signals:
                context_signals[context_class] = {signal: [] for signal in signal_names}

            for signal in signal_names:
                if signal in signals_dict:
                    context_signals[context_class][signal].append(signals_dict[signal])

    # Compute means
    for key in result:
        for signal in signal_names:
            values = result[key][signal]
            result[key][signal] = float(np.mean(values)) if len(values) > 0 else 0.0

    for context_class, signals_dict in context_signals.items():
        result[context_class] = {}
        for signal in signal_names:
            values = signals_dict[signal]
            result[context_class][signal] = float(np.mean(values)) if len(values) > 0 else 0.0

    # Filter by requested context classes if specified
    if context_classes is not None:
        result = {
            k: v for k, v in result.items()
            if k == 'global' or k in context_classes
        }

    return result


def compare_policies_value_signals(
    policy_logs: Dict[str, List[Dict[str, Any]]],
    signal_names: List[str],
    context_class: str = 'global'
) -> pd.DataFrame:
    """
    Compare value signals across multiple policies for a specific context.

    Creates a comparison table showing how different policies trade off
    between objectives.

    Args:
        policy_logs: Dict mapping policy names to their policy logs
        signal_names: List of signal names to compare
        context_class: Context class to analyze (default: 'global')

    Returns:
        DataFrame with policies as rows, signals as columns:

                            audience  competition  diversity  novelty  rights
        Random              0.156     0.200        0.300      0.750    0.130
        AudienceOnly        0.188     0.200        0.300      0.750    0.240
        StaticScalarization 0.175     0.200        0.300      0.750    0.600
        CTS                 0.175     0.200        0.300      0.750    0.600

    Example:
        >>> df = compare_policies_value_signals(
        ...     {'Random': log1, 'AudienceOnly': log2},
        ...     signal_names=['audience', 'competition', 'diversity']
        ... )
        >>> print(df.round(3))
    """
    comparison_data = []

    for policy_name, policy_log in policy_logs.items():
        signals_by_context = aggregate_value_signals(policy_log, signal_names)

        if context_class in signals_by_context:
            row = {'Policy': policy_name}
            row.update(signals_by_context[context_class])
            comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    if 'Policy' in df.columns:
        df = df.set_index('Policy')

    return df


def compute_signal_statistics(
    policy_log: List[Dict[str, Any]],
    signal_name: str,
    context_class: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute detailed statistics for a single value signal.

    Args:
        policy_log: List of decision records
        signal_name: Name of signal to analyze (e.g., 'audience')
        context_class: Optional context class to filter by

    Returns:
        Dict with statistics:
        {
            'mean': 0.72,
            'std': 0.15,
            'min': 0.42,
            'max': 0.95,
            'q25': 0.65,
            'q50': 0.73,
            'q75': 0.82,
            'count': 1193
        }

    Example:
        >>> stats = compute_signal_statistics(policy_log, 'audience', 'saturday_family')
        >>> print(f"Audience: {stats['mean']:.2f} ± {stats['std']:.2f}")
    """
    # Filter by context class if specified
    if context_class is not None:
        policy_log = [r for r in policy_log if r.get('context_class') == context_class]

    # Collect signal values
    values = []
    for record in policy_log:
        signals_top1 = record.get('value_signals_top1', {})
        if signal_name in signals_top1:
            values.append(signals_top1[signal_name])

    if len(values) == 0:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'q25': 0.0, 'q50': 0.0, 'q75': 0.0, 'count': 0
        }

    values_array = np.array(values)

    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q50': float(np.percentile(values_array, 50)),
        'q75': float(np.percentile(values_array, 75)),
        'count': len(values)
    }
