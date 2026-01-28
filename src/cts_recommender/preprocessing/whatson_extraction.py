from pathlib import Path
import pandas as pd
import numpy as np
import re

from cts_recommender.io import readers
from cts_recommender.settings import get_settings
from cts_recommender.preprocessing.whatson_constants import *
from cts_recommender import RTS_constants
from cts_recommender.utils import dates


def load_whatson_catalog_csv(csv_path: Path) -> pd.DataFrame: 
    whatson_df = readers.read_csv(csv_path, sep='\t')
    return whatson_df

# Load the preprocessed programming file into a DataFrame
def load_whatson_movie_catalogue(data_path: Path) -> pd.DataFrame:
    df = readers.read_parquet(data_path)
    return df

def is_episode_title(title: str) -> bool:
    """Check if a title matches episode patterns."""
    if pd.isna(title):
        return False

    title_str = str(title).strip()

    # Check all episode patterns
    for pattern in EPISODE_PATTERNS:
        if re.search(pattern, title_str, re.IGNORECASE):
            return True

    return False

def has_ignore_pattern(title: str) -> bool:
    """Check if a title contains ignore patterns (doublon, serie, etc.)."""
    if pd.isna(title):
        return False

    title_str = str(title).strip()

    # Check all ignore patterns
    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, title_str, re.IGNORECASE):
            return True

    return False

def has_date_pattern(title: str) -> bool:
    """Check if title contains date patterns like YYYY-MM-DD, YY-MM-DD, YY.MM.DD."""
    if pd.isna(title):
        return False
    
    title_str = str(title).strip()
    
    # Check all date patterns
    for pattern in DATE_PATTERNS:
        if re.search(pattern, title_str):
            return True
    
    return False

def parse_duration_minutes(duration_str: str) -> int | None:
    """
    Parse duration string (format: HH:MM:SS or MM:SS) and return total minutes.
    Returns None if parsing fails.
    """
    if pd.isna(duration_str):
        return None
    
    duration = str(duration_str).strip()
    
    try:
        # Split by ':'
        parts = duration.split(':')
        
        if len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 60 + minutes
        elif len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes
        else:
            return None
    except:
        return None

def is_film_collection(collection) -> bool:
    """
    Check if a collection name suggests it's a film collection.
    Uses both exact matches and keyword patterns.
    """
    if pd.isna(collection):
        return False
    
    collection_str = str(collection).strip()
    
    # Check exact matches (case-sensitive for specific names)
    if collection_str in FILM_COLLECTIONS:
        return True
    
    # Check keywords (case-insensitive)
    collection_lower = collection_str.lower()
    for keyword in FILM_COLLECTION_KEYWORDS:
        if keyword in collection_lower:
            return True
    
    return False

# Function to select the best title
def select_best_title(row: pd.Series) -> str:
    """
    Select the best title from original_title, title, or collection.
    Priority: 
    1. original_title (if not an episode pattern)
    2. title (if not an episode pattern)
    3. collection (fallback, even if it might be generic like 'Film')
    """
    title = row['title']
    original = row['original_title']
    collection = row['collection']
    
    # Priority 1: Use original_title if present, not empty, and not an episode
    if pd.notna(original) and str(original).strip() and str(original).strip().lower() != 'nan':
        if not is_episode_title(original):
            return str(original).strip()
    
    # Priority 2: Use title if not an episode marker
    if pd.notna(title) and str(title).strip():
        if not is_episode_title(title):
            return str(title).strip()
    
    # Priority 3: Use collection as fallback (even if generic like 'Film')
    if pd.notna(collection) and str(collection).strip():
        return str(collection).strip()
    
    # Last resort: return title even if it's episode-like
    return str(title) if pd.notna(title) else ''


def should_keep_as_movie(row: pd.Series, title_counts: pd.Series, collection_counts: pd.Series) -> bool:
    """
    Determine if a row should be kept as a movie based on the defined rules.

    

    ## MOVIE FILTERING LOGIC
        Following the specific rules:
        1. If episode pattern in title or original_title -> NOT a movie
        2. If date pattern in title (XX-XX-XX or XX.XX.XX) -> NOT a movie
        3. If title appears more than 3 times in dataset -> NOT a movie (likely series)
        4. If collection appears more than twice AND doesn't have film keyword -> NOT a movie (likely series collection)
        5. If duration is present (not 00:00:00) but < 35 min -> NOT a movie
        6. If collection == title AND collection matches film keywords -> NOT a movie (generic)
        7. If collection == title, check duration (must be >= 35 min or 00:00:00) -> else NOT a movie
        8. If title == original_title (exact match) -> IS a movie
        9. If title is substring of collection -> NOT a movie
        10. If collection matches film patterns -> IS a movie
        11. Otherwise -> NOT a movie (conservative approach)
    """
    title = row['title']
    original_title = row['original_title']
    collection = row['collection']
    duration = row['duration']
    
    # If title is unavailble, ignore
    if pd.isna(title):
        return False
    
    # Rule 1: Check for episode patterns in title or original_title
    if is_episode_title(title) or is_episode_title(original_title):
        return False

    # Rule 1b: Check for ignore patterns (doublon, serie, etc.) in title
    if has_ignore_pattern(title):
        return False

    # Rule 2: Check for date patterns in title
    if has_date_pattern(title):
        return False
    
    # Rule 3: If title appears more than 3 times -> likely a series
    if pd.notna(title):
        if title_counts.get(str(title).strip(), 0) > 3:
            return False
    
    # Rule 4: If collection appears more than twice AND doesn't have film keyword -> NOT movie
    if pd.notna(collection):
        collection_str = str(collection).strip()
        if collection_counts.get(collection_str, 0) > 2:
            if not is_film_collection(collection):
                return False
    
    # Rule 5: If duration is present (not 00:00:00) and < 35 minutes -> NOT movie
    duration_min = parse_duration_minutes(duration)
    if duration_min is not None and duration_min > 0 and duration_min < 35:
        return False
    
    # Rule 6: If collection == title AND collection is a generic film keyword -> NOT movie
    if pd.notna(collection) and pd.notna(title):
        if str(collection).strip() == str(title).strip():
            if is_film_collection(collection):
                return False
    
    # Rule 7: If collection == title (and NOT film keyword), check duration
    if pd.notna(collection) and pd.notna(title):
        if str(collection).strip() == str(title).strip():
            # Duration must be >= 35 min or 00:00:00 (unknown)
            if duration_min is not None and duration_min > 0 and duration_min < 35:
                return False
            # If duration is None, 0, or >= 35, continue to other checks
    
    # Rule 8: If title == original_title (exact match) -> IS movie
    if pd.notna(title) and pd.notna(original_title):
        if str(title).strip() == str(original_title).strip():
            return True
    
    # Rule 9: If title is substring of collection -> NOT movie (likely series)
    if pd.notna(title) and pd.notna(collection):
        title_str = str(title).strip().lower()
        collection_str = str(collection).strip().lower()
        # Avoid false positives from very short titles
        if len(title_str) >= 3 and title_str != collection_str:
            if title_str in collection_str:
                return False
    
    # Rule 10: Check if collection matches known film patterns
    if is_film_collection(collection):
        return True
    
    # Rule 11: Default to NOT movie (conservative approach)
    return False


def create_whatson_catalog(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a deduplicated movie catalog with catalog IDs from enriched WhatsOn data.

    This function:
    1. Deduplicates movies by processed_title (keeping row with most information)
    2. Assigns catalog_id (uses tmdb_id when available, custom ID otherwise)
    3. Converts date columns to datetime format
    4. Adds computed features (duration_min, times_shown, movie_age)
    5. Returns catalog indexed by catalog_id

    Parameters:
    enriched_df (pd.DataFrame): TMDB-enriched movie data

    Returns:
    pd.DataFrame: Deduplicated catalog indexed by catalog_id
    """
    df = enriched_df.copy()

    # Step 1: Count non-null entries in each row for deduplication scoring
    df['info_score'] = df.notna().sum(axis=1)

    # Step 2: Define deduplication key
    # Use processed_title as primary key (this comes from TMDB enrichment)
    group_cols = ['processed_title']

    # Keep row with max info_score within each group
    catalog_df = df.sort_values('info_score', ascending=False).drop_duplicates(subset=group_cols, keep='first')

    # Drop helper column
    catalog_df = catalog_df.drop(columns='info_score')

    # Step 3: Assign catalog_id
    # Base ID on TMDB if available, else assign a unique fallback
    catalog_df['catalog_id'] = catalog_df['tmdb_id'].astype('Int64').astype(str)

    # Assign fallback ID for missing tmdb_id
    missing_mask = catalog_df['missing_tmdb_id'] == True
    catalog_df.loc[missing_mask, 'catalog_id'] = [
        f"{RTS_constants.WHATSON_CATALOG_ID_PREFIX}{i:06d}"
        for i in range(missing_mask.sum())
    ]

    # Step 4: Convert date columns to datetime format
    for col in DATE_COLS:
        if col in catalog_df.columns:
            catalog_df[col] = pd.to_datetime(catalog_df[col], errors='coerce')

    # Step 5: Add computed features

    # duration_min - parse duration string to minutes
    if 'duration' in catalog_df.columns:
        catalog_df['duration_min'] = catalog_df['duration'].apply(parse_duration_minutes).astype('float64')

    # times_shown - count non-null broadcast dates (date_diff_1 and date_rediff_1-4)
    broadcast_date_cols = ['date_diff_1', 'date_rediff_1', 'date_rediff_2', 'date_rediff_3', 'date_rediff_4']
    available_broadcast_cols = [col for col in broadcast_date_cols if col in catalog_df.columns]
    catalog_df['times_shown'] = catalog_df[available_broadcast_cols].notna().sum(axis=1).astype('Int64')

    # movie_age - calculate from release_date
    if 'release_date' in catalog_df.columns:
        release_dates = pd.to_datetime(catalog_df['release_date'], errors='coerce')
        catalog_df['movie_age'] = ((pd.Timestamp.now() - release_dates).dt.days // 365).astype('Int64')

    # Step 6: Reset index and set catalog_id as index
    catalog_df = catalog_df.reset_index(drop=True)
    catalog_df = catalog_df.set_index('catalog_id')

    return catalog_df

