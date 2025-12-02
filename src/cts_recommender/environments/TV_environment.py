import pandas as pd
from typing import Optional
from datetime import datetime, date
from typing import Union, Tuple, Dict
import numpy as np
import logging

from cts_recommender.RTS_constants import COMPETITOR_CHANNELS, INTEREST_CHANNELS
from cts_recommender.environments.reward import RewardCalculator
from cts_recommender.models.audience_regression.audience_ratings_regressor import AudienceRatingsRegressor
from cts_recommender.competition.competitor import CompetitorDataManager
from cts_recommender.environments.schemas import Context, TimeSlot, Season, Channel
from cts_recommender.utils.scalers import make_safe_positive_pipeline

logger = logging.getLogger(__name__)

class TVProgrammingEnvironment:
    def __init__(
            self,
            catalog_df: pd.DataFrame,
            historical_programming_df: Optional[pd.DataFrame] = None,
            audience_model: AudienceRatingsRegressor = None,
            ):
        
        self.catalog_df = catalog_df
        self.historical_programming_df = historical_programming_df
        self.audience_model = audience_model

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


    def get_available_movies(self, date: Union[datetime, date]):
        """
        Get movies available for the given context (rights not expired), updates self.available_movies.

        Args:
            date: Date to check availability (datetime or date object)

        Note:
            Converts input to pd.Timestamp to ensure compatibility with datetime64[ns] columns
            after enforce_dtypes() is applied to the catalog.
        """
        # Convert to pandas Timestamp for comparison with datetime64[ns] columns
        date_ts = pd.Timestamp(date)

        available_mask = ((self.catalog_df['tv_rights_end'] > date_ts) &
                            (self.catalog_df['tv_rights_start'] < date_ts) &
                            (self.catalog_df['available_broadcasts'] > 0))

        self.available_movies = self.catalog_df[available_mask].index.tolist()


    def update_memory(self, catalog_id: str):
        """Update self.memory with the given catalog_id, maintaining the memory size limit"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(catalog_id)

    def get_context_features(self, context: Union[Context, Tuple]) -> Tuple[np.ndarray, Tuple]:

        """
        Convert context to feature vector from either a given Context or previous context_cache_key

        feature_vector = [time_slot_hot (4,), day_of_week_hot (7,), is_weekend (1,), season_one_hot (4,), channel_one_hot (2,)] -> shape: (18,)

        Channel is now included in the feature vector to allow CTS to learn channel-specific signal preferences.
        """
        # Cache key for efficiency (includes channel)
        if isinstance(context, tuple):
            cache_key = context
        else:
            cache_key = (context.hour, context.day_of_week,
                        context.month, context.season.value, context.channel.value)

        if cache_key in self.context_features_cache:
            return self.context_features_cache[cache_key], cache_key

        # Extract values from context (handle both Context object and tuple)
        if isinstance(context, tuple):
            hour, day_of_week, _, season_value, channel_value = context
        else: # Context object
            hour = context.hour
            day_of_week = context.day_of_week
            season_value = context.season.value
            channel_value = context.channel.value

        features = []

        # Slot hour of showing into 4 different TimeSlot
        time_slot_value = self.get_time_slot(hour).value
        time_slots = [time_slot.value for time_slot in TimeSlot]
        time_slot_features = [1 if time_slot == time_slot_value else 0 for time_slot in time_slots]
        features.extend(time_slot_features)

        # Day-of-week one-hot (0=Monday ... 6=Sunday)
        dow_one_hot = [1 if day_of_week == i else 0 for i in range(7)]
        features.extend(dow_one_hot)

        # Weekend flag
        is_weekend = 1 if day_of_week >= 5 else 0
        features.append(is_weekend)

        # Season one-hot
        seasons = [season.value for season in Season]
        season_features = [1 if season == season_value else 0 for season in seasons]
        features.extend(season_features)

        # Channel one-hot (RTS1, RTS2)
        channels = [channel.value for channel in Channel]
        channel_features = [1 if channel == channel_value else 0 for channel in channels]
        features.extend(channel_features)

        feature_vector = np.array(features, dtype=np.float32)
        self.context_features_cache[cache_key] = feature_vector
        return feature_vector, cache_key


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