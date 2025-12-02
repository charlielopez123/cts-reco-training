"""
Shared Feature Transformation Functions

This module contains common feature transformation logic used by both:
- ML_feature_processing.py: For audience ratings model training
- catalog_features.py: For WhatsOn movie catalog representation

These functions handle TMDB metadata processing, genre encoding, and date features.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def process_tmdb_features(df: pd.DataFrame, is_movie: bool = True) -> pd.DataFrame:
    """
    Process TMDB features with consistent handling of missing values.

    This is the core TMDB processing logic shared by both ML training
    and catalog feature pipelines.

    Parameters:
    df (pd.DataFrame): DataFrame with movie data (potentially with TMDB features)
    is_movie (bool): Whether this dataframe contains actual movies (True) or non-movies (False)

    Returns:
    pd.DataFrame: DataFrame with processed TMDB features
    """
    df = df.copy()

    # Adult flag
    if is_movie:
        df['adult'] = np.where(df['adult'].isna(), False, df['adult']).astype(bool)
    else:
        df.loc[:, 'adult'] = False

    # Original language
    if is_movie:
        df['original_language'] = np.where(
            df['original_language'].isna(), 'unknown', df['original_language']
        )
    else:
        df.loc[:, 'original_language'] = 'unknown'

    # Genres
    if is_movie:
        df['genres'] = df['genres'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
        )
    else:
        df.loc[:, 'genres'] = [[] for _ in range(len(df))]

    # Keywords
    if is_movie:
        if 'keywords' in df.columns:
            df['keywords'] = df['keywords'].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
            )
        else:
            df.loc[:, 'keywords'] = [[] for _ in range(len(df))]
    else:
        df.loc[:, 'keywords'] = [[] for _ in range(len(df))]

    # Release date and missing flag
    if is_movie:
        df['release_date'] = np.where(
            df['release_date'].isna(), '1900-01-01', df['release_date']
        )
        df.loc[:, 'missing_release_date'] = (
            df['release_date'].isna() | (df['release_date'] == '')
        )
        df.loc[:, 'release_date'] = df['release_date'].fillna('1900-01-01').replace('', '1900-01-01')
    else:
        df.loc[:, 'release_date'] = '1900-01-01'
        df.loc[:, 'missing_release_date'] = True

    # Revenue
    if is_movie:
        df['revenue'] = df['revenue'].fillna(0)
    else:
        df.loc[:, 'revenue'] = 0

    # Missing TMDB ID flag
    if is_movie:
        df.loc[:, 'missing_tmdb'] = np.where(
            df['tmdb_id'].isna(), True, False
        )
    else:
        df.loc[:, 'missing_tmdb'] = True

    # Vote average
    if is_movie:
        df.loc[:, 'vote_average'] = df['vote_average'].fillna(0)
    else:
        df.loc[:, 'vote_average'] = 0

    # Popularity
    if is_movie:
        df.loc[:, 'popularity'] = df['popularity'].fillna(0)
    else:
        df.loc[:, 'popularity'] = 0

    # Movie flag
    df.loc[:, 'is_movie'] = is_movie

    return df


def add_genre_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode movie genres.

    Shared by both ML training and catalog pipelines.

    Parameters:
    df (pd.DataFrame): DataFrame with 'genres' column (list of dicts with 'name' key)

    Returns:
    pd.DataFrame: DataFrame with genre_* one-hot encoded columns
    """
    df = df.copy()

    # Extract genre names from list of dicts
    df.loc[:, 'genre_names'] = df['genres'].apply(lambda lst: [d['name'] for d in lst])

    # Explode and one-hot encode
    exploded = df.explode(['genre_names'])
    dummies = pd.get_dummies(exploded["genre_names"], prefix="genre")
    genre_dummies = dummies.groupby(level=0).sum()

    # Merge back
    df = pd.concat([df, genre_dummies], axis=1)

    # Drop intermediate columns
    df = df.drop(columns=['genres', 'genre_names'])

    return df


def extract_keyword_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract keyword names from TMDB keywords list.

    Keywords come as list of dicts: [{"id": 123, "name": "space"}]
    This function extracts just the names as a list of strings.

    Note: Unlike genres, we keep keywords as a list column rather than
    one-hot encoding, as there are thousands of possible keywords.

    Parameters:
    df (pd.DataFrame): DataFrame with 'keywords' column (list of dicts with 'name' key)

    Returns:
    pd.DataFrame: DataFrame with 'keywords' column converted to list of strings
    """
    df = df.copy()

    # Extract keyword names from list of dicts
    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].apply(
            lambda lst: [d['name'] for d in lst] if isinstance(lst, list) else []
        )

    return df


def add_movie_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate movie age from release date.

    Shared by both ML training and catalog pipelines.

    Parameters:
    df (pd.DataFrame): DataFrame with 'release_date' column (YYYY-MM-DD format)

    Returns:
    pd.DataFrame: DataFrame with 'movie_age' column added (release_date is kept for diversity calculations)
    """
    df = df.copy()

    # Convert to datetime
    df["release_date_dt"] = pd.to_datetime(df["release_date"], format="%Y-%m-%d")

    # Calculate age
    df['movie_age'] = pd.Timestamp('today').year - df["release_date_dt"].dt.year

    # Drop only the intermediate datetime column, keep release_date string for diversity reward calculations
    df = df.drop(columns=['release_date_dt'])

    return df


def convert_bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all boolean columns to integers (0/1).

    Shared utility for final processing in both pipelines.

    Parameters:
    df (pd.DataFrame): DataFrame with potential boolean columns

    Returns:
    pd.DataFrame: DataFrame with booleans converted to int
    """
    df = df.copy()
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    return df
