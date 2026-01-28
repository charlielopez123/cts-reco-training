import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cts_recommender.RTS_constants import COMPETITOR_CHANNELS, INTEREST_CHANNELS
from cts_recommender.competition.competitor import CompetitorDataManager
from cts_recommender.environments.curtains import (
    get_curtain_definition,
    get_curtain_type_one_hot,
    validate_curtain_for_channel,
    validate_curtain_for_day,
)
from cts_recommender.environments.reward import RewardCalculator
from cts_recommender.environments.schemas import (
    Channel,
    Context,
    ContextMode,
    CurtainContext,
    Season,
    TimeSlot,
)
from cts_recommender.models.audience_regression.audience_ratings_regressor import (
    AudienceRatingsRegressor,
)
from cts_recommender.utils.dates import get_season
from cts_recommender.utils.scalers import make_safe_positive_pipeline

logger = logging.getLogger(__name__)

class TVProgrammingEnvironment:
    def __init__(
            self,
            catalog_df: pd.DataFrame,
            historical_programming_df: Optional[pd.DataFrame] = None,
            audience_model: AudienceRatingsRegressor = None,
            context_mode: ContextMode = ContextMode.GENERAL,
            ):
        self.catalog_df = catalog_df
        self.historical_programming_df = historical_programming_df
        self.audience_model = audience_model

        # Context mode determines feature dimensions
        self.context_mode = context_mode
        if context_mode == ContextMode.RTS_CURTAIN:
            # 7 curtain_type + 7 day_of_week + 1 weekend + 4 season + 2 channel = 21
            self.context_dim = 21
        else:
            # 4 time_slot + 7 day_of_week + 1 weekend + 4 season + 2 channel = 18
            self.context_dim = 18

        # Separate historical programming by interest channels with competitors
        if self.historical_programming_df is not None:
            self.competition_historical_programming_df = self.historical_programming_df[self.historical_programming_df['channel'].isin(COMPETITOR_CHANNELS)]
            self.interest_historical_programming_df = self.historical_programming_df[self.historical_programming_df['channel'].isin(INTEREST_CHANNELS)]

        self.memory_size = 100  # Max memory size
        self.memory = []

        # Movies available to be shown, initialized to empty
        self.available_movies = []

        # State tracking
        self.context_features_cache: Dict[Tuple, np.ndarray] = {}

        # Competitor data manager
        logger.info("Setting up CompetitorDataManager...")
        self.competitor_manager = CompetitorDataManager(self.competition_historical_programming_df)

        logger.info("Setting up Scalers...")

        self.scaler_dict = {
            "revenue": make_safe_positive_pipeline(log_compress=True).fit(
                self.catalog_df[["revenue"]].replace([np.inf, -np.inf], np.nan)
            ),
            "popularity": make_safe_positive_pipeline(log_compress=True).fit(
                self.catalog_df[["popularity"]].replace([np.inf, -np.inf], np.nan)
            ),
            "movie_age": make_safe_positive_pipeline(log_compress=True).fit(
                self.catalog_df[["movie_age"]].replace([np.inf, -np.inf], np.nan)
            ),
            "duration": make_safe_positive_pipeline(log_compress=False).fit(
                self.catalog_df[["duration_min"]].replace([np.inf, -np.inf], np.nan)
            ),
            "vote_average": make_safe_positive_pipeline(log_compress=False).fit(
                self.catalog_df[["vote_average"]].replace([np.inf, -np.inf], np.nan)
            ),
            "rt_m": make_safe_positive_pipeline(log_compress=True).fit(
                self.historical_programming_df[["rt_m"]].replace([np.inf, -np.inf], np.nan)
            ),
        }

        logger.info("Setting up RewardCalculator...")
        self.reward = RewardCalculator(
            catalog_df=self.catalog_df,
            historical_df=self.historical_programming_df,
            audience_model=self.audience_model,
            competition_manager=self.competitor_manager,
            memory=self.memory,
            interest_historical_df=self.interest_historical_programming_df,
            competition_historical_df=self.competition_historical_programming_df,
            scaler_dict=self.scaler_dict
        )


    def get_available_movies(self, date: Union[datetime, date], times_shown_tracker=None):
        """
        Get movies available for the given context (rights not expired), updates self.available_movies.

        Args:
            date: Date to check availability (datetime or date object)
            times_shown_tracker: Optional TimesShownTracker for dynamic broadcast quota checking.
                If provided, filters out movies that have exhausted their broadcast quota
                at this date (total_broadcasts - times_shown <= 0).
                If None, only filters by TV rights validity.

        Note:
            Converts input to pd.Timestamp to ensure compatibility with datetime64[ns] columns
            after enforce_dtypes() is applied to the catalog.
        """
        # Convert to pandas Timestamp for comparison with datetime64[ns] columns
        date_ts = pd.Timestamp(date)

        # Base filter: TV rights validity
        available_mask = ((self.catalog_df['tv_rights_end'] > date_ts) &
                            (self.catalog_df['tv_rights_start'] < date_ts))

        available_ids = self.catalog_df[available_mask].index.tolist()

        # If tracker provided, filter by remaining broadcast quota at decision time
        if times_shown_tracker is not None:
            available_ids = [
                movie_id for movie_id in available_ids
                if (self.catalog_df.loc[movie_id, 'total_broadcasts'] -
                    times_shown_tracker.get_times_shown(movie_id, date_ts)) > 0
            ]

        self.available_movies = available_ids


    def update_memory(self, catalog_id: str):
        """Update self.memory with the given catalog_id, maintaining the memory size limit"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(catalog_id)

    def get_context_features(
        self, context: Union[Context, CurtainContext, Tuple]
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Convert context to feature vector.

        For Context (general mode): 18 dims
            [time_slot(4), day(7), weekend(1), season(4), channel(2)]

        For CurtainContext (rts_curtain mode): 21 dims
            [curtain_type(7), day(7), weekend(1), season(4), channel(2)]
        """
        if isinstance(context, tuple):
            if context in self.context_features_cache:
                return self.context_features_cache[context], context
            raise ValueError(f"Cache key {context} not found")

        if isinstance(context, CurtainContext):
            return self._encode_curtain_context(context)
        else:
            return self._encode_general_context(context)

    def _encode_general_context(self, context: Context) -> Tuple[np.ndarray, Tuple]:
        """Encode Context (general mode) to 18-dim feature vector."""
        cache_key = (
            context.hour,
            context.day_of_week,
            context.month,
            context.season.value,
            context.channel.value,
        )

        if cache_key in self.context_features_cache:
            return self.context_features_cache[cache_key], cache_key

        features: List[int] = []

        # Time slot one-hot (4 dims)
        time_slot = self.get_time_slot(context.hour)
        time_slot_features = [1 if ts == time_slot else 0 for ts in TimeSlot]
        features.extend(time_slot_features)

        # Day-of-week one-hot (7 dims)
        features.extend([1 if context.day_of_week == i else 0 for i in range(7)])

        # Weekend flag (1 dim)
        features.append(1 if context.day_of_week >= 5 else 0)

        # Season one-hot (4 dims)
        features.extend([1 if s == context.season else 0 for s in Season])

        # Channel one-hot (2 dims)
        features.extend([1 if c == context.channel else 0 for c in Channel])

        feature_vector = np.array(features, dtype=np.float32)
        self.context_features_cache[cache_key] = feature_vector
        return feature_vector, cache_key

    def _encode_curtain_context(self, context: CurtainContext) -> Tuple[np.ndarray, Tuple]:
        """Encode CurtainContext (rts_curtain mode) to 21-dim feature vector."""
        cache_key = (
            "curtain",  # Prefix to distinguish from general mode cache keys
            context.curtain_type.value,
            context.day_of_week,
            context.month,
            context.season.value,
            context.channel.value,
        )

        if cache_key in self.context_features_cache:
            return self.context_features_cache[cache_key], cache_key

        features: List[int] = []

        # Curtain type one-hot (7 dims)
        features.extend(get_curtain_type_one_hot(context.curtain_type))

        # Day-of-week one-hot (7 dims)
        features.extend([1 if context.day_of_week == i else 0 for i in range(7)])

        # Weekend flag (1 dim)
        features.append(1 if context.day_of_week >= 5 else 0)

        # Season one-hot (4 dims)
        features.extend([1 if s == context.season else 0 for s in Season])

        # Channel one-hot (2 dims)
        features.extend([1 if c == context.channel else 0 for c in Channel])

        feature_vector = np.array(features, dtype=np.float32)
        self.context_features_cache[cache_key] = feature_vector
        return feature_vector, cache_key

    def create_curtain_context(
        self, air_date: date, curtain_id: str, channel: str
    ) -> CurtainContext:
        """Create CurtainContext for RTS curtain mode from a date and curtain ID."""
        validate_curtain_for_channel(curtain_id, channel)
        validate_curtain_for_day(curtain_id, air_date.weekday())

        curtain_def = get_curtain_definition(curtain_id)

        return CurtainContext(
            curtain_type=curtain_def.curtain_type,
            curtain_id=curtain_id,
            day_of_week=air_date.weekday(),
            month=air_date.month,
            season=Season(get_season(air_date)),
            channel=Channel(channel),
        )

    def create_context(self, air_date: date, hour: int, channel: str) -> Context:
        """Create Context for general mode from a date and hour."""
        return Context(
            hour=hour,
            day_of_week=air_date.weekday(),
            month=air_date.month,
            season=Season(get_season(air_date)),
            channel=Channel(channel),
        )


    def get_movie_features(self, catalog_id: str) -> np.ndarray:
        """
        Get normalized movie features for the given catalog_id to be used to compute reward/value signals.

        Args:
            catalog_id: The catalog ID (string) to retrieve features for

        Returns:
            np.ndarray: Normalized movie features

        Raises:
            KeyError: If catalog_id not found in catalog
        """
        try:
            film_catalog_row: pd.Series = self.catalog_df.loc[catalog_id]
        except KeyError:
            logger.error(f"Catalog ID {catalog_id} not found in catalog")
            raise KeyError(f"Catalog ID {catalog_id} not found in catalog")

        features = pd.DataFrame({
        'norm_revenue': self.scaler_dict['revenue'].transform(pd.DataFrame({'revenue': [film_catalog_row['revenue']]}))[0][0],
        'norm_vote_avg': self.scaler_dict['vote_average'].transform(pd.DataFrame({'vote_average': [film_catalog_row['vote_average']]}))[0][0],
        'norm_popularity': self.scaler_dict['popularity'].transform(pd.DataFrame({'popularity': [film_catalog_row['popularity']]}))[0][0],
        'norm_duration': self.scaler_dict['duration'].transform(pd.DataFrame({'duration_min': [film_catalog_row['duration_min']]}))[0][0],
        'norm_movie_age': self.scaler_dict['movie_age'].transform(pd.DataFrame({'movie_age': [film_catalog_row['movie_age']]}))[0][0],
        }, index=[0])

        genre_prefix = 'genre_'
        # Identify expected dummy columns from model
        audience_model_col_names = self.audience_model.get_feature_names()
        genre_dummies = [x for x in audience_model_col_names if x.startswith(genre_prefix)]
        # Add all dummy columns with 0 initially
        for col in genre_dummies:
            features[col] = 0
        
        movie_genre_list = [genre['name'] for genre in film_catalog_row['genres']]
        for genre in movie_genre_list:
            col_name = f"{genre_prefix}{genre}"
            if col_name in features.columns:
                features[col_name] = 1  # Set genre presence to 1

        assert not features.isna().any().any(), f"NaN values found in movie features for {catalog_id}, \n {features}"

        movie_features = np.squeeze(np.array(features, dtype=np.float32))
        return movie_features

    def get_time_slot(self, hour: int) -> TimeSlot:
        """
        Map hour of day to time slot.

        Hours 0-5 (midnight to 6am) are considered LATE_NIGHT (continuation from previous day).
        """
        if 6 <= hour < 14:
            return TimeSlot.MORNING
        elif 14 <= hour < 19:
            return TimeSlot.AFTERNOON
        elif 19 <= hour < 22:
            return TimeSlot.PRIME_TIME
        elif 22 <= hour < 26:  # 22-26 handles up to 2am as hour 24, 25, 26
            return TimeSlot.LATE_NIGHT
        elif 0 <= hour < 6:  # Midnight to 6am is also late night
            return TimeSlot.LATE_NIGHT
        else:
            raise ValueError(f"Invalid hour: {hour}. Expected 0-26.")