"""
Reward calculation for TV programming environment.

Computes rewards based on:
    - Audience ratings predictions (via trained audience model)
    - Competition effects (optional)
    - Diversity
    - Novelty
    - Rights urgency

Designed to work with TVProgrammingEnvironment via shared references
(no data duplication).

"""
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, List, Optional
import numpy as np
from sklearn.pipeline import Pipeline

from cts_recommender.imitation_learning.times_shown_tracker import TimesShownTracker
from cts_recommender.models.audience_regression.audience_ratings_regressor import AudienceRatingsRegressor
from cts_recommender.competition.competitor import CompetitorDataManager, MovieCompetitorContext
from cts_recommender.environments.reward_components import diversity
from cts_recommender.environments.schemas import Context, Channel
from cts_recommender.utils import dates

logger = logging.getLogger(__name__)

class RewardCalculator:
    """
    Calculates rewards for TV programming decisions.

    Designed to work with TVProgrammingEnvironment via shared references
    (no data duplication).
    """
    def __init__(self,
                catalog_df: pd.DataFrame,
                historical_df: Optional[pd.DataFrame] = None,
                audience_model: AudienceRatingsRegressor = None,
                competition_manager: Optional[CompetitorDataManager] = None,
                memory: Optional[List[str]] = None,
                interest_historical_df: Optional[pd.DataFrame] = None,
                competition_historical_df: Optional[pd.DataFrame] = None,
                scaler_dict: Optional[Dict[str, Pipeline]] = None,
                holiday_index: Optional[Dict] = None,
                ):
        """
        Args:
            catalog_df: Reference to environment's catalog_df (shared, not copied)
            audience_model: Trained model for predicting audience ratings
            historical_df: Historical programming data for all channels
            competition_manager: Manager for competitor data (optional)
            memory: Reference to environment's memory (shared list)
            interest_historical_df: Reference to interest channels historical programming data
            competition_historical_df: Reference to competition channels historical programming data
            scaler_dict: Reference to environment's scalers (shared dict)
            holiday_index: Holiday index from dates.build_holiday_index (optional)
        """

        # Store references (no copies)
        self.catalog_df = catalog_df
        self.historical_df = historical_df
        self.audience_model = audience_model
        self.competition_manager = competition_manager
        self.memory = memory if memory is not None else []
        self.interest_historical_df = interest_historical_df
        self.competition_historical_df = competition_historical_df
        self.scaler_dict = scaler_dict
        self.holiday_index = holiday_index

        logger.info(f"RewardCalculator initialized successfully.")

    def compute_total_reward(self,
                            catalog_id: str,
                            air_date: pd.Timestamp,
                            context: Context,
                            times_shown_tracker: TimesShownTracker | None = None
                            ) -> Dict[str, float]:
        """
        Calculate multi-objective reward for programming a movie given a context.

        Args:
            catalog_id: Movie catalog identifier
            air_date: Broadcast date
            context: Context object with temporal features (includes channel)
            times_shown_tracker: Optional tracker for temporal coherence (IL training)

        Returns:
            Dictionary with individual reward components

        Temporal Coherence Note:
            - Audience, competition, diversity, rights: Temporally coherent
            - Novelty: Temporally coherent ONLY with times_shown_tracker (IL training)
                      Without tracker, uses static catalog values (future data leak)
        """

        # Fetch movie details
        movie_catalog_row = self.catalog_df.loc[catalog_id]
        rewards = {}
        rewards['audience'] = self._calculate_audience_reward(context, movie_catalog_row, air_date)

        # 2. COMPETITIVE ADVANTAGE REWARD
        rewards['competition'] = self._calculate_competition_reward(air_date, movie_catalog_row)

        # 3. DIVERSITY REWARD
        rewards['diversity'] = self._calculate_diversity_reward(movie_catalog_row)

        # 4. NOVELTY REWARD
        rewards['novelty'] = self._calculate_novelty_reward(air_date, movie_catalog_row, times_shown_tracker)

        # 5. RIGHTS URGENCY REWARD
        rewards['rights'] = self._calculate_rights_urgency_reward(air_date, movie_catalog_row)

        return rewards

    def compute_rewards_for_historical_row(self,
                                            historical_row: pd.Series,
                                            times_shown_tracker: TimesShownTracker | None = None) -> Dict[str, float]:
        """Compute reward for a historical programming decision row.

        Args:
            historical_row: Row from historical programming DataFrame
            times_shown_tracker: Optional tracker for dynamic times_shown computation during IL training

        Returns:
            Dictionary with individual reward components
        """
        air_date: pd.Timestamp = historical_row['date']
        movie_catalog_row: pd.Series = self.catalog_df.loc[historical_row['catalog_id']]
        rewards = {}

        # 1. AUDIENCE APPEAL REWARD
        # Audience ratings directly accessible from historical data
        # (predicted via audience model during simulation)
        audience = historical_row.get('rt_m')
        rewards['audience'] = float(self.scaler_dict['rt_m'].transform(pd.DataFrame({'rt_m': [audience]})).squeeze())

        # 2. COMPETITIVE ADVANTAGE REWARD
        rewards['competition'] = self._calculate_competition_reward(air_date, movie_catalog_row)

        # 3. DIVERSITY REWARD
        rewards['diversity'] = self._calculate_diversity_reward(movie_catalog_row)

        # 4. NOVELTY REWARD
        rewards['novelty'] = self._calculate_novelty_reward(air_date, movie_catalog_row, times_shown_tracker)

        # 5. RIGHTS URGENCY REWARD
        rewards['rights'] = self._calculate_rights_urgency_reward(air_date, movie_catalog_row)

        return rewards

    def _calculate_audience_reward(self, context: Context, movie: pd.Series, air_date: pd.Timestamp, return_raw_prediction: bool = False):
        """Calculate reward based on predicted audience ratings.

        Uses trained audience model to predict rt_m for a broadcast context + movie,
        then normalizes to [0, 1] via rt_m scaler.

        Args:
            context: Context with temporal features (hour, day_of_week, month, season, channel)
            movie: Movie catalog row with TMDB metadata
            air_date: Broadcast date (for holiday lookup)
            return_raw_prediction: If True, return tuple of (normalized_reward, raw_rt_m_prediction)

        Returns:
            If return_raw_prediction=False: Normalized audience reward in [0.0, 1.0]
            If return_raw_prediction=True: Tuple of (normalized_reward, raw_rt_m_prediction)

        Raises:
            ValueError: If audience model not trained/loaded
        """
        if self.audience_model is None or not self.audience_model.is_trained:
            raise ValueError("Audience model must be trained before prediction")

        # Build feature vector matching model's expected order
        feature_names = self.audience_model.get_feature_names()
        features = {}

        # 1. Base temporal/broadcast features
        features['hour'] = context.hour
        features['weekday'] = context.day_of_week
        features['is_weekend'] = int(context.day_of_week >= 5)
        features['duration_min'] = movie['duration_min']
        features['public_holiday'] = int(dates.is_holiday_indexed(air_date, self.holiday_index)) if self.holiday_index else 0

        # 2. Categorical features (one-hot encode)
        # Season one-hot encoding
        season_cols = [col for col in feature_names if col.startswith('season_')]
        for season_col in season_cols:
            features[season_col] = 0
        season_value = context.season.value  # e.g., 'winter', 'spring', 'summer', 'fall'
        season_col_name = f'season_{season_value}'
        if season_col_name in feature_names:
            features[season_col_name] = 1

        # Channel one-hot encoding
        channel_cols = [col for col in feature_names if col.startswith('channel_')]
        for channel_col in channel_cols:
            features[channel_col] = 0
        channel_value = context.channel.value  # e.g., 'RTS 1', 'RTS 2'
        channel_col_name = f'channel_{channel_value}'
        if channel_col_name in feature_names:
            features[channel_col_name] = 1

        # Original language one-hot encoding
        lang_cols = [col for col in feature_names if col.startswith('original_language_')]
        for lang_col in lang_cols:
            features[lang_col] = 0
        lang_value = movie['original_language']
        lang_col_name = f'original_language_{lang_value}'
        if lang_col_name in feature_names:
            features[lang_col_name] = 1

        # 3. TMDB numerical features
        features['popularity'] = movie['popularity']
        features['revenue'] = movie['revenue']
        features['vote_average'] = movie['vote_average']

        # 4. Boolean features
        features['adult'] = int(movie['adult'])
        features['missing_release_date'] = int(movie['missing_release_date'])
        features['missing_tmdb'] = int(movie['missing_tmdb'])
        features['is_movie'] = int(movie['is_movie'])

        # 5. Computed features
        features['movie_age'] = movie['movie_age']

        # 6. Genre one-hot features (dummify on-demand from list format)
        genre_cols = [col for col in feature_names if col.startswith('genre_')]
        # Initialize all genre columns to 0
        for genre_col in genre_cols:
            features[genre_col] = 0
        # Set to 1 for genres present in the movie
        movie_genres = movie.get('genres', [])
        if isinstance(movie_genres, list):
            for genre_dict in movie_genres:
                genre_name = genre_dict.get('name')
                if genre_name:
                    genre_col = f'genre_{genre_name}'
                    if genre_col in features:
                        features[genre_col] = 1

        # Build DataFrame with correct column order
        feature_df = pd.DataFrame([features])[feature_names]

        # Predict and normalize
        predicted_rt_m = self.audience_model.model.predict(feature_df)[0]
        normalized_reward = float(self.scaler_dict['rt_m'].transform(pd.DataFrame({'rt_m': [predicted_rt_m]})).squeeze())

        if return_raw_prediction:
            return np.clip(normalized_reward, 0.0, 1.0), predicted_rt_m
        else:
            return np.clip(normalized_reward, 0.0, 1.0)

    def _calculate_competition_reward(self, air_date: pd.Timestamp, movie: pd.Series) -> float:
        """Calculate reward based on competitive advantage.

        Args:
            air_date: Date when RTS airs the movie
            movie: Movie catalog row

        Returns:
            Competition reward in range [0.0, 1.0]

        Note:
            d = (air_date - competitor_air_date).days
            Negative d: RTS airs BEFORE competitor (good)
            Positive d: RTS airs AFTER competitor (bad, but recovers over time)
        """
        movie_competition_context: MovieCompetitorContext = self.competition_manager.get_movie_competitor_context(
            movie_id=movie.name, reference_date=air_date
        )

        # No competitor info → neutral reward
        if len(movie_competition_context.competitor_showings) == 0 or movie.name is None:
            return 0.2

        # Aggregate rewards from all competitor showings (take max to find best advantage)
        rewards = []
        for comp_showing in movie_competition_context.competitor_showings:
            comp_date: datetime = comp_showing['air_date']
            d = (air_date - comp_date).days

            # Apply reward function based on timing difference
            if 0 <= d <= 183:
                # 0-6 months after: no advantage
                reward = 0.0
            elif 183 < d <= 365:
                # 6 months to 1 year after: linear recovery to 0.2
                reward = 0.2 * (d - 183) / 183.0
            elif d > 365:
                # Over 1 year after: neutral
                reward = 0.2
            elif -4 <= d <= -1:
                # 1-4 days before: peak advantage
                reward = 1.0
            elif -21 < d < -4:
                # 4-21 days before: exponential rise from 0.4 to 1.0
                a = 0.25
                num = np.exp(a * (d + 21)) - 1.0
                den = np.exp(a * 17.0) - 1.0
                reward = 0.4 + 0.6 * (num / den)
            elif d <= -21:
                # More than 21 days before: neutral
                reward = 0.2
            elif -1 < d < 0:
                # Less than 1 day before: linear decay from 1.0 to 0.2
                t = d + 1.0
                reward = 1.0 * (1 - t) + 0.2 * t
            elif d == 0:
                # Same day: neutral
                reward = 0.2
            else:
                # Default fallback
                reward = 0.2

            rewards.append(reward)

        # Return capped sum of rewards (max 1.0)
        return min(sum(rewards), 1)

    def _calculate_diversity_reward(self, movie: pd.Series) -> float:
        """Calculate reward for programming diversity.

        Temporal Coherence: ✅ Uses only static catalog features (no temporal leak)
        Memory contains past programming decisions only.
        """

        if self.memory is None or len(self.memory) <= 1:
            return 0.3  # Neutral if no history

        # Look at recent programming (e.g., last 25 movies)
        recent_programming = self.memory[-1:-25:-1]
        # Filter to only include movies still in catalog (handle expired rights)
        recent_programming = [mid for mid in recent_programming if mid in self.catalog_df.index]

        # If no valid recent movies, return neutral
        if not recent_programming:
            return 0.3

        # Calculate diversity metrics
        diversity_scores = []

        # 1. Genre diversity
        # Extract genres from list format (catalog has 'genres' as list of dicts)
        current_genres: List[str] = [g['name'] for g in movie.get('genres', [])] if isinstance(movie.get('genres', []), list) else []

        recent_genres: List[List[str]] = []
        for movie_id in recent_programming:
            recent_movie = self.catalog_df.loc[movie_id]
            recent_movie_genres = [g['name'] for g in recent_movie.get('genres', [])] if isinstance(recent_movie.get('genres', []), list) else []
            recent_genres.append(recent_movie_genres)

        genre_diversity = diversity.compute_genre_diversity(current_genres, recent_genres)
        diversity_scores.append(genre_diversity)

        # 2. Country/Language diversity
        current_lang = movie['original_language']
        recent_langs = [self.catalog_df.loc[movie_id]['original_language'] for movie_id in recent_programming]
        language_diversity = diversity.calculate_language_diversity(current_lang, recent_langs)
        diversity_scores.append(language_diversity)

        # 3. Temporal diversity
        current_release_date = movie['release_date']
        current_release_year = pd.to_datetime(current_release_date).year if pd.notna(current_release_date) else None
        recent_release_years = [
            pd.to_datetime(self.catalog_df.loc[movie_id]['release_date']).year
            if pd.notna(self.catalog_df.loc[movie_id]['release_date'])
            else None
            for movie_id in recent_programming
        ]
        temporal_diversity = diversity.calculate_temporal_diversity(current_release_year, recent_release_years)
        diversity_scores.append(temporal_diversity)

        # 4. Voting average diversity
        current_vote_avg = movie['vote_average']
        recent_vote_avgs = [self.catalog_df.loc[movie_id]['vote_average'] for movie_id in recent_programming]
        vote_avg_diversity = diversity.calculate_voting_average_diversity(current_vote_avg, recent_vote_avgs)
        diversity_scores.append(vote_avg_diversity)

        # 5. Revenue diversity
        current_revenue = movie['revenue']
        recent_revenues = [self.catalog_df.loc[movie_id]['revenue'] for movie_id in recent_programming]
        revenue_diversity = diversity.calculate_revenue_diversity(current_revenue, recent_revenues)
        diversity_scores.append(revenue_diversity)

        # Average diversity scores
        diversity_reward = np.mean(diversity_scores)
        return diversity_reward

    def _calculate_novelty_reward(
        self,
        air_date: pd.Timestamp,
        movie: pd.Series,
        times_shown_tracker: TimesShownTracker | None = None
    ) -> float:

        """Calculate reward for programming novelty/freshness.

        Args:
            air_date: Date of the broadcast
            movie: Movie catalog row
            times_shown_tracker: Optional tracker for IL training (computes times_shown dynamically)
                                If None, uses static values from catalog (for online/simulation)

        Returns:
            Novelty reward in range [0.0, 1.0]

        Temporal Coherence:
            - Cross-channel repetition: Filters historical_df by date < air_date (line 187)
            - Channel-specific repetition:
                * With times_shown_tracker: Dynamically computed (temporally coherent)
                * Without tracker: <!> Uses catalog values computed from all WhatsOn data
                (includes future broadcasts - acceptable for IL training, NOT for online sim)
        """
        # 0) Early exit if no memory
        if self.memory is None or len(self.memory) == 0:
            return 0.75  # High novelty if no history

        # 1) Channel-specific repetition & time-decay score
        if times_shown_tracker:
            # Use tracker for IL training (dynamic computation)
            times_shown = times_shown_tracker.get_times_shown(movie.name, air_date)
            date_last_diff = times_shown_tracker.get_last_broadcast_date(movie.name, air_date)
        else:
            # Use catalog values for online/simulation (static values)
            times_shown = movie.get('times_shown', 0)
            date_last_diff = movie.get('last_broadcast_date', None)

        if times_shown == 0:
            channel_score = 1.0
        else:
            # Base repetition penalty
            rep_score = max(0.0, 1.0 - 0.25 * times_shown)
            # Linear time-decay over 365 days
            if date_last_diff is None:
                time_decay = 1.0
            else:
                days_since = air_date - date_last_diff
                days_since = min(days_since.days, 365)
                time_decay = days_since / 365.0
            channel_score = rep_score * time_decay

        # Collect scores
        novelty_scores = [channel_score]

        # 2) Cross-channel historical repetition

        if self.historical_df is not None:
            last_showing = None

            # Look back over last 300 historical showings to find last showing date of considered movie
            available_historical_df = self.historical_df[self.historical_df['date'] < air_date].copy()
            history_slice = available_historical_df.iloc[-300:]
            for _, hist_row in history_slice.iterrows():
                if pd.notna(hist_row['catalog_id']) and pd.notna(movie.name) and hist_row['catalog_id'] == movie.name:
                    hist_date = hist_row['date']
                    if last_showing is None or hist_date > last_showing:
                        last_showing: pd.Timestamp | None = hist_date
            if last_showing is None:
                cross_score = 1.0
            else:
                days = (air_date - last_showing).days
                cross_score = min(1.0, days / 365.0)
            novelty_scores.append(cross_score)

        # Final novelty reward is the mean of all components
        novelty_reward = np.mean(novelty_scores)
        return float(np.clip(novelty_reward, 0.0, 1.0))

    def _calculate_rights_urgency_reward(self, air_date: pd.Timestamp, movie: pd.Series) -> float:
        end_date: pd.Timestamp = movie['tv_rights_end']

        diff_days = (end_date - air_date).days

        # 0-30 days (critical urgency)
        if 0 <= diff_days <= 30: #Critical urgency
            reward = 1
        # 30-90 days (high urgency)
        elif 30 < diff_days <= 90:
            reward = 0.8  # smoother decay
        # 90-365 days (medium urgency)
        elif 90 < diff_days < 365//2:
            reward = 0.6
        # 6 months to a year
        elif 365//2 <= diff_days < 365:
            reward = 0.3
        # year to 2 years
        elif 365 <= diff_days < 2*365:
            reward = 0.1
        else:
            reward = 0  # No effect if too far before or after
        return reward