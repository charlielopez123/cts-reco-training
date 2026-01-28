"""
Imitation Learning Training Data Extraction Pipeline

Orchestrates the extraction of training data for imitation learning from historical
TV programming decisions. This data is used to:
1. Train a curator logistic regression model (learn from past programming decisions)
2. Warm-start the Contextual Thompson Sampler with historical data

This pipeline:
1. Loads historical programming data (facts about what was broadcast)
2. Loads WhatsOn catalog (movie features)
3. Loads trained audience ratings model
4. Creates TVProgrammingEnvironment for context and feature extraction
5. Uses HistoricalDataProcessor to extract positive/negative training samples
6. Prepares training data with context features, movie features, and rewards
7. Optionally splits data by time for temporal validation

Output includes:
- Training samples (positive: actual programming decisions by RTS curators)
- Negative samples (movies that could have been chosen but weren't)
- Context features (time, channel, season)
- Movie features (from catalog)
- Pseudo-rewards (weighted combination of audience, competition, diversity, novelty, rights)
- Curator selection targets (1 for selected, 0 for not selected)

The curator logistic regression learns to mimic curator decisions while considering
pseudo-rewards, enabling offline learning from historical programming.
"""

from pathlib import Path
import pandas as pd
import logging
import joblib
from typing import Optional

from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.environments.schemas import ContextMode
from cts_recommender.models.audience_regression.audience_ratings_regressor import AudienceRatingsRegressor
from cts_recommender.imitation_learning.IL_training import HistoricalDataProcessor
from cts_recommender.io import readers
from cts_recommender.settings import get_settings
from cts_recommender.features.catalog_schema import CATALOG_DTYPES, enforce_dtypes

cfg = get_settings()
logger = logging.getLogger(__name__)


def run_IL_training_data_pipeline(
    historical_programming_file: Optional[Path] = None,
    catalog_file: Optional[Path] = None,
    audience_model_file: Optional[Path] = None,
    out_file: Optional[Path] = None,
    gamma: float = 0.6,
    negative_sampling_ratio: int = 5,
    time_split_date: Optional[str] = None,
    context_mode: ContextMode = ContextMode.GENERAL
) -> tuple[dict, Path]:
    """
    Extract training data for imitation learning from historical programming decisions.

    This pipeline processes historical TV programming to create training samples for:
    1. Curator logistic regression model (learns to mimic curator selection decisions)
    2. Contextual Thompson Sampler warm-start (initializes bandit with historical data)

    Parameters:
    -----------
    historical_programming_file : Path, optional
        Path to historical_programming.parquet
        Default: data/processed/programming/historical_programming.parquet

    catalog_file : Path, optional
        Path to whatson_catalog.parquet
        Default: data/processed/whatson/whatson_catalog.parquet

    audience_model_file : Path, optional
        Path to trained audience ratings model (.joblib)
        Default: data/models/audience_ratings_model.joblib

    out_file : Path, optional
        Path to save training data (.joblib)
        Default: data/processed/IL/training_data.joblib

    gamma : float, default=0.6
        Weighting factor for curator selection signal vs pseudo-reward signals
        reward = gamma * selected + (1-gamma) * pseudo_reward
        Higher gamma gives more weight to actual curator decisions (imitation learning)
        Lower gamma gives more weight to computed rewards (reinforcement learning)

    negative_sampling_ratio : int, default=5
        Number of negative samples per positive sample
        Negative samples are randomly chosen from available movies that weren't selected
        Higher ratios help the model learn what NOT to select

    time_split_date : str, optional
        Date to split train/validation data (format: 'YYYY-MM-DD')
        If not provided, automatically calculates 80/20 split based on historical programming dates
        Returns {'train': {...}, 'val': {...}} with 80% of dates for training, 20% for validation

    Returns:
    --------
    tuple[dict, Path]
        Training data dictionary and output file path

        If time_split_date is None:
            {
                'context_features': ndarray,      # Context vectors (time, channel, etc.)
                'movie_features': ndarray,        # Movie feature vectors
                'curator_targets': ndarray,       # 1 for selected, 0 for not
                'reward_targets': ndarray,        # Pseudo-reward values
                'context_cache_keys': ndarray,    # For caching context computations
                'current_memories': ndarray       # Memory state (recently shown movies)
            }

        If time_split_date is provided:
            {
                'train': {context_features, movie_features, ...},
                'val': {context_features, movie_features, ...}
            }
    """
    # Set default paths
    if historical_programming_file is None:
        historical_programming_file = cfg.processed_dir / "programming" / "historical_programming.parquet"
    if catalog_file is None:
        catalog_file = cfg.processed_dir / "whatson" / "whatson_catalog.parquet"
    if audience_model_file is None:
        audience_model_file = cfg.models_dir / "audience_ratings_model.joblib"
    if out_file is None:
        out_file = cfg.processed_dir / "IL" / "training_data.joblib"

    logger.info("=== Imitation Learning Training Data Extraction Pipeline ===")
    logger.info(f"Historical programming: {historical_programming_file}")
    logger.info(f"Catalog: {catalog_file}")
    logger.info(f"Audience model: {audience_model_file}")
    logger.info(f"Gamma (curator weight): {gamma}")
    logger.info(f"Negative sampling ratio: {negative_sampling_ratio}")
    logger.info(f"Context mode: {context_mode.value}")

    # Load data
    logger.info("Loading historical programming data...")
    historical_df = readers.read_parquet(historical_programming_file)
    logger.info(f"Loaded {len(historical_df)} historical programming records")

    # Calculate 80/20 split date if not provided
    if time_split_date is None:
        # Get unique dates and sort them
        unique_dates = pd.Series(historical_df['date'].unique()).sort_values()
        # Calculate 80th percentile (80% for training, 20% for validation)
        split_idx = int(len(unique_dates) * 0.8)
        split_date = unique_dates.iloc[split_idx]
        time_split_date = split_date.strftime('%Y-%m-%d')
        logger.info(f"Auto-calculated 80/20 time split date: {time_split_date}")
        logger.info(f"  Training dates: {unique_dates.min().strftime('%Y-%m-%d')} to {time_split_date}")
        logger.info(f"  Validation dates: {time_split_date} to {unique_dates.max().strftime('%Y-%m-%d')}")
        logger.info(f"  Total unique dates: {len(unique_dates)}, Train: {split_idx}, Val: {len(unique_dates) - split_idx}")
    else:
        logger.info(f"Using provided time split date: {time_split_date}")

    logger.info("Loading catalog data...")
    catalog_df = readers.read_parquet(catalog_file)
    # Enforce correct dtypes (especially datetime columns)
    catalog_df = enforce_dtypes(catalog_df, CATALOG_DTYPES, skip_missing=True)
    logger.info(f"Loaded {len(catalog_df)} catalog movies")

    logger.info("Loading audience ratings model...")
    audience_model = AudienceRatingsRegressor()
    audience_model.load_model(audience_model_file)
    logger.info("Audience ratings model loaded successfully")

    # Create TV programming environment
    logger.info("Initializing TV programming environment...")
    env = TVProgrammingEnvironment(
        catalog_df=catalog_df,
        historical_programming_df=historical_df,
        audience_model=audience_model,
        context_mode=context_mode
    )
    logger.info(f"Environment initialized with context_dim={env.context_dim}")

    # Create historical data processor
    logger.info("Initializing historical data processor...")
    processor = HistoricalDataProcessor(
        environment=env,
        historical_data=historical_df,
        gamma=gamma,
        negative_sampling_ratio=negative_sampling_ratio,
        time_split_date=time_split_date,
        context_mode=context_mode
    )
    logger.info("Processor initialized")

    # Extract training samples (raw dictionaries for visualization)
    logger.info("Extracting training samples (this may take a while)...")
    training_samples = processor.extract_training_samples()

    # Save raw samples for visualization (used by CTS warm-start pipeline)
    # Derive samples filename from output file (training_data.joblib -> training_samples.joblib)
    samples_filename = out_file.name.replace("training_data", "training_samples")
    samples_file = out_file.parent / samples_filename
    logger.info(f"Saving raw training samples to {samples_file}...")
    joblib.dump(training_samples, samples_file)
    logger.info(f"Saved {len(training_samples)} raw samples")

    # Convert to training data (numpy arrays)
    logger.info("Converting samples to training data...")
    training_data = processor._convert_samples_to_training_data(
        training_samples,
        time_split_date
    )

    # Log statistics
    if time_split_date:
        train_size = len(training_data['train']['curator_targets'])
        val_size = len(training_data['val']['curator_targets'])
        train_pos = training_data['train']['curator_targets'].sum()
        val_pos = training_data['val']['curator_targets'].sum()

        logger.info(f"\nTraining set:")
        logger.info(f"  Total samples: {train_size}")
        logger.info(f"  Positive samples: {train_pos} ({100*train_pos/train_size:.1f}%)")
        logger.info(f"  Negative samples: {train_size - train_pos} ({100*(train_size-train_pos)/train_size:.1f}%)")

        logger.info(f"\nValidation set:")
        logger.info(f"  Total samples: {val_size}")
        logger.info(f"  Positive samples: {val_pos} ({100*val_pos/val_size:.1f}%)")
        logger.info(f"  Negative samples: {val_size - val_pos} ({100*(val_size-val_pos)/val_size:.1f}%)")
    else:
        total_samples = len(training_data['curator_targets'])
        positive_samples = training_data['curator_targets'].sum()

        logger.info(f"\nTotal samples: {total_samples}")
        logger.info(f"Positive samples: {positive_samples} ({100*positive_samples/total_samples:.1f}%)")
        logger.info(f"Negative samples: {total_samples - positive_samples} ({100*(total_samples-positive_samples)/total_samples:.1f}%)")

    # Ensure output directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save training data
    logger.info(f"\nSaving training data to {out_file}...")
    joblib.dump(training_data, out_file)
    logger.info("Training data saved successfully")

    return training_data, out_file
