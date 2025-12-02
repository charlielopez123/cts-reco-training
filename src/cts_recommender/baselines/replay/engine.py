"""
Offline historical replay engine for policy evaluation.

Evaluates policies by replaying historical RTS programming decisions
and comparing policy recommendations against curator choices.

Adapted from cts_recommender.imitation_learning.IL_training.HistoricalDataProcessor
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from cts_recommender.baselines.metrics.context_classifier import classify_context, get_relevant_movies_for_context
from cts_recommender.baselines.metrics.ranking import hit_at_k, ndcg_at_k
from cts_recommender.baselines.policies.base import BasePolicy

# Import from main codebase
from cts_recommender.environments.schemas import Context, Season, Channel
from cts_recommender.RTS_constants import INTEREST_CHANNELS


class OfflineReplayEngine:
    """
    Offline historical replay for policy evaluation.

    Replays historical programming decisions chronologically, asking each policy
    to recommend movies and comparing against actual curator choices.

    Pattern adapted from HistoricalDataProcessor in IL_training.py.
    """

    def __init__(
        self,
        environment,  # TVProgrammingEnvironment
        historical_data: pd.DataFrame,
        catalog: pd.DataFrame
    ):
        """
        Initialize replay engine.

        Args:
            environment: TVProgrammingEnvironment instance (contains reward calculator)
            historical_data: Historical programming DataFrame with columns:
                - 'date': pd.Timestamp
                - 'channel': str
                - 'catalog_id': str
                - 'hour': int
                - 'weekday': int
                - Plus others from HISTORICAL_PROGRAMMING_DTYPES
            catalog: Catalog DataFrame (indexed by catalog_id)
        """
        self.env = environment
        self.reward_calc = environment.reward  # RewardCalculator is inside environment
        self.historical = historical_data
        self.catalog = catalog

    def _create_context_from_row(self, row: pd.Series) -> Context:
        """
        Create Context object from historical data row.

        Adapted from HistoricalDataProcessor._create_context_from_row()

        Args:
            row: Historical programming row with hour, weekday, channel, etc.

        Returns:
            Context object
        """
        # Get temporal features from row
        hour = row.get('hour')
        day_of_week = row.get('weekday')

        # Get date to extract month
        date = pd.Timestamp(row['date'])
        month = date.month

        # Map month to season
        if month in [3, 4, 5]:
            season = Season.SPRING
        elif month in [6, 7, 8]:
            season = Season.SUMMER
        elif month in [9, 10, 11]:
            season = Season.AUTUMN
        else:
            season = Season.WINTER

        # Map channel string to Channel enum
        channel_str = row.get('channel')
        if channel_str == 'RTS 1':
            channel = Channel.RTS1
        elif channel_str == 'RTS 2':
            channel = Channel.RTS2
        else:
            raise ValueError(f"Unknown channel: {channel_str}. Expected 'RTS 1' or 'RTS 2'")

        return Context(
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            season=season,
            channel=channel
        )

    def replay_policy(
        self,
        policy: BasePolicy,
        K: int = 10,
        n_seeds: int = 1,
        verbose: bool = True,
        max_decisions: Optional[int] = None
    ) -> Dict:
        """
        Run offline replay for a single policy.

        Args:
            policy: Policy to evaluate (must implement BasePolicy interface)
            K: Slate size for top-K recommendations
            n_seeds: Number of random seeds (for stochastic policies)
            verbose: Show progress bar
            max_decisions: Optional limit on number of decisions (for testing)

        Returns:
            Dict with:
            {
                'predictions': List[List[str]],  # Top-K slates
                'targets': List[str],             # Curator choices
                'context_classes': List[str],     # Context class per decision
                'value_signal_log': List[Dict],   # Per-decision value signals
                'metrics': {
                    'historical_choice': {        # Strict relevance
                        'hit_at_k': float,
                        'ndcg_at_k': float
                    },
                    'context_relevant': {         # Relaxed relevance
                        'hit_at_k': float,
                        'ndcg_at_k': float
                    }
                },
                'n_decisions': int,               # Total decisions evaluated
                'n_seeds': int                    # Number of seeds used
            }

        Example:
            >>> from policies.random import RandomPolicy
            >>> policy = RandomPolicy()
            >>> results = engine.replay_policy(policy, K=10, n_seeds=3)
            >>> print(f"Hit@10: {results['metrics']['historical_choice']['hit_at_k']:.3f}")
        """
        results_by_seed = []

        for seed in range(n_seeds):
            policy.reset(seed)

            predictions = []
            targets = []
            context_classes = []
            value_signal_log = []

            # Filter to RTS channels only (same as IL training)
            interest_channel_historical = self.historical[
                self.historical['channel'].isin(INTEREST_CHANNELS)
            ].copy()

            # Filter out records without valid catalog_id
            initial_count = len(interest_channel_historical)
            interest_channel_historical = interest_channel_historical[
                interest_channel_historical['catalog_id'].notna()
            ].copy()
            filtered_count = initial_count - len(interest_channel_historical)

            if verbose and seed == 0:
                print(f"\nFiltered {filtered_count} records without valid catalog_id")
                print(f"Evaluating on {len(interest_channel_historical)} decisions")

            # Sort chronologically
            interest_channel_historical = interest_channel_historical.sort_values('date')

            # Limit decisions if specified (for testing)
            if max_decisions:
                interest_channel_historical = interest_channel_historical.head(max_decisions)

            # Reset memory (same as IL training)
            self.env.memory = []
            self.env.reward.memory = self.env.memory

            # Progress bar
            iterator = tqdm(
                interest_channel_historical.iterrows(),
                total=len(interest_channel_historical),
                desc=f"Replaying (seed {seed})"
            ) if verbose else interest_channel_historical.iterrows()

            for idx, row in iterator:
                # 1. Create context from row (same as IL training)
                context = self._create_context_from_row(row)

                # Get context features
                context_features, context_cache_key = self.env.get_context_features(context)

                # 2. Get available candidates (same as IL training)
                air_date = row['date']
                self.env.get_available_movies(air_date)
                candidates = list(self.env.available_movies)

                if len(candidates) == 0:
                    # Skip if no candidates available
                    continue

                # 3. Compute value signals for all candidates
                candidate_features = {}
                value_signals = {}

                for catalog_id in candidates:
                    # Get movie features
                    movie_features = self.env.get_movie_features(catalog_id)
                    candidate_features[catalog_id] = movie_features

                    # Compute value signals (same as IL training for negative samples)
                    signals = self.reward_calc.compute_total_reward(
                        catalog_id=catalog_id,
                        air_date=air_date,
                        context=context,
                        times_shown_tracker=None  # Use static catalog values
                    )
                    value_signals[catalog_id] = signals

                # 4. Policy selects top-K
                top_k = policy.select_action(
                    context_features=context_features,
                    candidates=candidates,
                    candidate_features=candidate_features,
                    value_signals=value_signals,
                    K=K
                )

                # 5. Record prediction and target
                predictions.append(top_k)
                target = row['catalog_id']
                targets.append(target)

                # Update memory with curator's actual choice (same as IL training)
                self.env.update_memory(target)

                # 6. Classify context (for stratified analysis)
                # Create broadcast datetime using hour column (handles broadcast hours 6-29)
                date_part = pd.Timestamp(row['date'])
                hour = int(row['hour'])

                # If hour >= 24, it represents next day (broadcast format)
                if hour >= 24:
                    broadcast_datetime = date_part + pd.Timedelta(days=1, hours=hour-24)
                else:
                    broadcast_datetime = date_part + pd.Timedelta(hours=hour)

                channel = row['channel']
                context_class = classify_context(broadcast_datetime, channel)
                context_classes.append(context_class)

                # 7. Build relevant sets for both relevance modes
                # Mode 1: Historical-Choice (strict) - curator's exact choice
                relevant_historical = {target}

                # Mode 2: Context-Relevant (relaxed) - expert rules
                # Get all available movies that match context requirements
                available_ids = set(candidates)
                relevant_contextual = get_relevant_movies_for_context(
                    self.catalog,
                    context_class,
                    available_ids
                )

                # 8. Log value signals for top-1 AND full top-K
                selected = top_k[0]  # Top-1 for bandit update
                top_k_signals = [value_signals[cid] for cid in top_k]

                value_signal_log.append({
                    't': idx,
                    'date': broadcast_datetime,
                    'channel': channel,
                    'context_class': context_class,
                    'selected_action': selected,
                    'target_action': target,
                    'matched_curator': (selected == target),
                    'top_k_slate': top_k,
                    'value_signals_top1': value_signals[selected],
                    'value_signals_topk': top_k_signals,
                    'relevant_historical': relevant_historical,
                    'relevant_contextual': relevant_contextual
                })

                # 9. Update policy (for bandit methods)
                reward = 1.0 if selected == target else 0.0
                policy.update(
                    context_features=context_features,
                    action=selected,
                    reward=reward,
                    action_features=candidate_features[selected],
                    value_signals=value_signals[selected]
                )

            # Compute metrics for this seed
            # Extract relevant sets for each mode
            relevant_historical_sets = [log['relevant_historical'] for log in value_signal_log]
            relevant_contextual_sets = [log['relevant_contextual'] for log in value_signal_log]

            metrics = {
                'historical_choice': {
                    'hit_at_k': hit_at_k(predictions, relevant_historical_sets, K),
                    'ndcg_at_k': ndcg_at_k(predictions, relevant_historical_sets, K)
                },
                'context_relevant': {
                    'hit_at_k': hit_at_k(predictions, relevant_contextual_sets, K),
                    'ndcg_at_k': ndcg_at_k(predictions, relevant_contextual_sets, K)
                }
            }

            results_by_seed.append({
                'predictions': predictions,
                'targets': targets,
                'context_classes': context_classes,
                'value_signal_log': value_signal_log,
                'metrics': metrics,
                'n_decisions': len(predictions)
            })

        # If multiple seeds, average metrics
        if n_seeds == 1:
            result = results_by_seed[0]
            result['n_seeds'] = 1
            return result
        else:
            # Average metrics across seeds
            avg_metrics = {
                'historical_choice': {
                    'hit_at_k': np.mean([r['metrics']['historical_choice']['hit_at_k'] for r in results_by_seed]),
                    'ndcg_at_k': np.mean([r['metrics']['historical_choice']['ndcg_at_k'] for r in results_by_seed])
                },
                'context_relevant': {
                    'hit_at_k': np.mean([r['metrics']['context_relevant']['hit_at_k'] for r in results_by_seed]),
                    'ndcg_at_k': np.mean([r['metrics']['context_relevant']['ndcg_at_k'] for r in results_by_seed])
                }
            }

            # Return first seed's data with averaged metrics
            result = results_by_seed[0].copy()
            result['metrics'] = avg_metrics
            result['n_seeds'] = n_seeds

            return result

    def replay_multiple_policies(
        self,
        policies: Dict[str, BasePolicy],
        K: int = 10,
        n_seeds: int = 1,
        verbose: bool = True,
        max_decisions: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Run offline replay for multiple policies.

        Args:
            policies: Dict mapping policy names to policy instances
            K: Slate size
            n_seeds: Number of seeds per policy
            verbose: Show progress
            max_decisions: Optional limit (for testing)

        Returns:
            Dict mapping policy names to their results:
            {
                'RandomPolicy': {results from replay_policy()},
                'AudienceOnly': {results from replay_policy()},
                ...
            }

        Example:
            >>> policies = {
            ...     'Random': RandomPolicy(),
            ...     'AudienceOnly': AudienceOnlyPolicy()
            ... }
            >>> all_results = engine.replay_multiple_policies(policies, K=10)
            >>> for name, results in all_results.items():
            ...     print(f"{name}: Hit@10 = {results['metrics']['historical_choice']['hit_at_k']:.3f}")
        """
        all_results = {}

        for policy_name, policy in policies.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating: {policy_name}")
                print('='*60)

            results = self.replay_policy(
                policy=policy,
                K=K,
                n_seeds=n_seeds,
                verbose=verbose,
                max_decisions=max_decisions
            )

            all_results[policy_name] = results

        return all_results
