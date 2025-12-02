"""
Catalog Feature Processing for WhatsOn Movie Catalog

This module provides feature processing for movie catalog representation,
separate from the ML training features used for audience ratings regression.

Key differences from ML_feature_processing.py:
- catalog_features.py: Process features for catalog representation (WhatsOn movies)
- ML_feature_processing.py: Process features for model training (historical programming data)

While they share transformation logic (via feature_transformations.py),
they serve different purposes in the system architecture.
"""

import pandas as pd
import logging
from cts_recommender.preprocessing import feature_transformations

logger = logging.getLogger(__name__)


def finalize_catalog_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final processing: convert booleans to int and log any remaining NaNs.

    Note: Unlike ML features, catalog features may contain NaNs in non-feature
    columns (e.g., broadcast dates, rights info). This is acceptable for catalog
    representation.

    Parameters:
    df (pd.DataFrame): DataFrame with all features

    Returns:
    pd.DataFrame: DataFrame ready for catalog storage
    """
    # Convert booleans using shared utility
    df = feature_transformations.convert_bool_to_int(df)

    # Log any remaining NaNs (for monitoring, not enforcement)
    num_nans = df.isna().sum().sum()
    if num_nans > 0:
        logger.info(f"Catalog contains {num_nans} NaN values in non-feature columns (expected for catalog metadata).")
        nan_cols = df.isna().sum()[df.isna().sum() > 0]
        logger.debug(f"Columns with NaNs:\n{nan_cols}")

    return df


def build_catalog_features(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build catalog features for WhatsOn movies.

    This function processes TMDB-enriched movie data into catalog features
    for recommendation system representation. Unlike ML training features,
    catalog features focus on movie metadata representation rather than
    preparing data for model training.

    Parameters:
    enriched_df (pd.DataFrame): The enriched movie features DataFrame with TMDB data.

    Returns:
    pd.DataFrame: The feature-ready DataFrame for catalog storage.
    """
    # Process TMDB features (all catalog entries are movies)
    df = feature_transformations.process_tmdb_features(enriched_df, is_movie=True)

    # Extract keyword names from TMDB keywords structure
    df = feature_transformations.extract_keyword_names(df)

    # Add movie age
    df = feature_transformations.add_movie_age_feature(df)

    # Final cleanup
    df = finalize_catalog_features(df)

    logger.info(f"Built catalog features: shape={df.shape}")
    return df
