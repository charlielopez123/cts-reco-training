import pandas as pd
import logging
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union
import numpy as np



from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.environments.schemas import Context, CurtainContext, ContextMode, Season, Channel
from cts_recommender.environments.curtains import get_curtain_for_broadcast
from cts_recommender.RTS_constants import INTEREST_CHANNELS
from cts_recommender.features.whatson_schema import SHOWINGS_COLUMNS
from cts_recommender.imitation_learning.IL_constants import PSEUDO_REWARD_WEIGHTS
from cts_recommender.imitation_learning.times_shown_tracker import TimesShownTracker


logger = logging.getLogger(__name__)

class HistoricalDataProcessor:
    """
    Processes historical programming data for offline training, using past programming decisions
    to train the curator model and warm start the Contextual Thompson Sampler weights
    """
    def __init__(self,
                environment: TVProgrammingEnvironment,
                historical_data: pd.DataFrame,
                gamma: float = 0.6,
                negative_sampling_ratio: float = 5,
                time_split_date: str = None,
                context_mode: ContextMode = ContextMode.GENERAL):

        self.env = environment
        self.historical_df = historical_data
        self.gamma = gamma
        self.negative_sampling_ratio = negative_sampling_ratio
        self.time_split_date = time_split_date
        self.context_mode = context_mode


    def extract_training_samples(self) -> List[Dict]:
        """
        Extract training samples from historical programming data for imitation learning.

        Returns:
            List of dictionaries with training samples, each containing:
                - context_features: Features representing the programming context
                - movie_features: Features of the movie shown or considered
                - movie_id: ID of the movie
                - selected: 1 if the movie was shown, 0 if it was a negative sample
                - reward: Computed reward for the programming decision
                - value_signals: Breakdown of reward components
                - date: Date of the programming decision
                - context_cache_key: Key for caching context features
                - current_memory: Current memory state of the environment
        """

        logger.info("Processing historical programming data for RTS 1 and RTS 2...")

        # Keep historical only from RTS channels of interest
        interest_channel_historical_df: pd.DataFrame = self.historical_df[self.historical_df['channel'].isin(INTEREST_CHANNELS)].copy()

        # Filter out records without valid catalog_id (movies not in catalog)
        initial_count = len(interest_channel_historical_df)
        interest_channel_historical_df = interest_channel_historical_df[interest_channel_historical_df['catalog_id'].notna()].copy()
        filtered_count = initial_count - len(interest_channel_historical_df)

        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} records without valid catalog_id (movies broadcast but not in WhatsOn catalog)")
            logger.info(f"  Filtered: {filtered_count} ({100*filtered_count/initial_count:.1f}%)")
            logger.info(f"  Remaining: {len(interest_channel_historical_df)} ({100*len(interest_channel_historical_df)/initial_count:.1f}%)")

        logger.info(f"Processing {len(interest_channel_historical_df)} programming decisions for {INTEREST_CHANNELS}")

        # Initialize times_shown tracker for dynamic computation during IL training
        times_shown_tracker = TimesShownTracker(
            catalog_df=self.env.catalog_df,
            historical_df=self.historical_df,
            interest_channels=INTEREST_CHANNELS
        )
        logger.info("Initialized TimesShownTracker for dynamic times_shown computation")

        # Prepare positive samples (movies that were actually shown)
        all_samples = []
        # Keep track of number of positive samples considered
        num_successful_samples = 0

        # reset memory
        self.env.memory = []
        self.env.reward.memory = self.env.memory

        # Track skipped broadcasts for curtain mode
        skipped_outside_curtain = 0

        # Iterate through each row of the interest channel's historical decisions
        for _, row in tqdm(interest_channel_historical_df.iterrows(), total=len(interest_channel_historical_df), desc="Processing rows"):

            # Create context based on mode
            if self.context_mode == ContextMode.RTS_CURTAIN:
                context = self._create_curtain_context_from_row(row)
                if context is None:
                    skipped_outside_curtain += 1
                    continue
            else:
                context = self._create_context_from_row(row)

            air_date = row['date']

            movie_id: str = row['catalog_id']

            # Store movie ID in memory
            self.env.update_memory(movie_id)

            if movie_id not in self.env.catalog_df.index:
                    logger.info(f"Movie ID {movie_id} not found in catalog, skipping row")
                    continue

            context_features, context_cache_key = self.env.get_context_features(context)
            movie_features = self.env.get_movie_features(movie_id)

            # Calculate reward for this programming decision using tracker
            rewards = self.env.reward.compute_rewards_for_historical_row(row, times_shown_tracker)
            pseudo_reward = sum(rewards[component] * PSEUDO_REWARD_WEIGHTS[component] for component in PSEUDO_REWARD_WEIGHTS.keys())

            pos_sample = {
                    'context_features': context_features,
                    'movie_features': movie_features,
                    'movie_id': movie_id,
                    'selected': 1,  # Positive sample
                    'reward': (self.gamma * 1 + (1-self.gamma) * pseudo_reward), # weight the weighted reward factors with the actual selection of the curator
                    'value_signals': rewards, # detailed breakdown of reward components
                    'date': air_date,
                    'context_cache_key': context_cache_key,
                    'current_memory': self.env.memory
                }
            all_samples.append(pos_sample)
            num_successful_samples += 1

            # Generate negative samples (movies that could have been shown but weren't)
            neg_samples = self.generate_negative_samples(pos_sample, row, context)
            all_samples.extend(neg_samples)

        logger.info(f"Created {num_successful_samples} positive samples")
        if self.context_mode == ContextMode.RTS_CURTAIN:
            logger.info(f"Skipped {skipped_outside_curtain} broadcasts outside curtain time windows")

        return all_samples

    def _create_context_from_row(self, row: pd.Series) -> Context:
        """Create Context object from historical data row"""

        # Map time to time slot
        hour = row.get('hour')
        day_of_week = row.get('weekday')

        # Map month to season
        month = row.get('month')
        if month in [3, 4, 5]:
            season = Season.SPRING
        elif month in [6, 7, 8]:
            season = Season.SUMMER
        elif month in [9, 10, 11]:
            season = Season.AUTUMN
        else:
            season = Season.WINTER

        # Map channel string to Channel enum (match RTS_constants.INTEREST_CHANNELS)
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
            channel=channel,
        )

    def _create_curtain_context_from_row(self, row: pd.Series) -> Optional[CurtainContext]:
        """Create CurtainContext from historical data row, or None if outside curtain times."""
        curtain_id = get_curtain_for_broadcast(
            row['start_time'],
            row['channel'],
            row['weekday']
        )
        if curtain_id is None:
            return None

        air_date = row['date'].date() if hasattr(row['date'], 'date') else row['date']
        return self.env.create_curtain_context(air_date, curtain_id, row['channel'])

    def generate_negative_samples(self,
                                pos_sample: Dict,
                                row: pd.Series,
                                context: Union[Context, CurtainContext],
                                times_shown_tracker: TimesShownTracker | None = None) -> List[Dict]:
        """
        Generate negative samples for a given positive sample by randomly selecting movies that were NOT chosen.

        Args:
            pos_sample: The positive sample dictionary
            row: The historical data row corresponding to the positive sample
            context: The Context or CurtainContext object for the programming decision
            times_shown_tracker: Optional TimesShownTracker for dynamic times_shown computation
        Returns:
            List of negative sample dictionaries
        """

        negative_samples = []

        # Update available movies based on historical data up to the air date
        self.env.get_available_movies(row['date'])

        # Remove the actual movie that was shown
        if pos_sample['movie_id'] in self.env.available_movies:
            self.env.available_movies.remove(pos_sample['movie_id'])

        # Sample negative movies
        negative_movie_ids = np.random.choice(
            self.env.available_movies, size=self.negative_sampling_ratio, replace=False
        )

        # For reward calculation, always use general Context (has hour for audience model)
        reward_context = self._create_context_from_row(row)

        context_features = pos_sample['context_features']
        for neg_movie_id in negative_movie_ids:
            movie_features = self.env.get_movie_features(neg_movie_id)
            rewards = self.env.reward.compute_total_reward(neg_movie_id, row['date'], reward_context, times_shown_tracker)
            pseudo_reward = sum(rewards[component] * PSEUDO_REWARD_WEIGHTS[component] for component in PSEUDO_REWARD_WEIGHTS.keys())
            negative_samples.append({
                        'context_features': context_features,
                        'movie_features': movie_features,
                        'movie_id': neg_movie_id,
                        'selected': 0,  # Negative sample
                        'reward': (1- self.gamma) * pseudo_reward,
                        'value_signals': rewards, # detailed breakdown of reward components
                        'date': pos_sample['date'],
                        'context_cache_key': pos_sample['context_cache_key'],
                        'current_memory': self.env.memory
                    })
            
        return negative_samples

    def _convert_samples_to_training_data(self, all_samples: List[Dict], time_split_date: str = None):
        """
        Convert raw training samples to numpy arrays for model training.

        Args:
            all_samples: List of sample dictionaries
            time_split_date: Optional date string for train/val split (YYYY-MM-DD)

        Returns:
            Dictionary with training data arrays
        """
        context_features = np.array([s['context_features'] for s in all_samples])
        movie_features = np.array([s['movie_features'] for s in all_samples])
        curator_targets = np.array([s['selected'] for s in all_samples])
        reward_targets = np.array([s['reward'] for s in all_samples])
        dates = np.array([s['date'] for s in all_samples])
        context_cache_keys = np.array([s['context_cache_key'] for s in all_samples])
        current_memories = np.array([s['current_memory'] for s in all_samples])

        # Split into training and validation sets based on time_split_date if provided
        if time_split_date:
            split_date = pd.to_datetime(time_split_date)
            train_mask = dates < split_date

            train_data = {
                'context_features': context_features[train_mask],
                'movie_features': movie_features[train_mask],
                'curator_targets': curator_targets[train_mask],
                'reward_targets': reward_targets[train_mask],
                'context_cache_keys': context_cache_keys[train_mask],
                'current_memories': current_memories[train_mask]
            }

            val_data = {
                'context_features': context_features[~train_mask],
                'movie_features': movie_features[~train_mask],
                'curator_targets': curator_targets[~train_mask],
                'reward_targets': reward_targets[~train_mask],
                'context_cache_keys': context_cache_keys[~train_mask],
                'current_memories': current_memories[~train_mask]
            }

            return {'train': train_data, 'val': val_data}

        else:
            return {
                'context_features': context_features,
                'movie_features': movie_features,
                'curator_targets': curator_targets,
                'reward_targets': reward_targets,
                'context_cache_keys': context_cache_keys,
                'current_memories': current_memories,
            }

    def prepare_training_data(self):
        """
        Prepare training data from historical programming decisions

        Returns:
            Dictionary with training data arrays
        """
        all_samples = self.extract_training_samples()
        return self._convert_samples_to_training_data(all_samples, self.time_split_date)