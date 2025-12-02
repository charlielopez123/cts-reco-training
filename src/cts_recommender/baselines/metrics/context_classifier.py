"""
Context classification for stratified evaluation.

Classifies programming decisions into key RTS timeslots for context-stratified analysis.
Based on RTS editorial briefs with ±20 minute flexibility for time windows.

NOTE: Uses actual RTS catalog schema with French genre names.
Age ratings are not available (RTS uses "Achats UAP", etc.), so we use the 'adult' flag from TMDB.
"""

from typing import Literal, Dict, Any, Set
import pandas as pd
import numpy as np


# Type alias for context classes
ContextClass = Literal[
    'saturday_family',
    'saturday_action',
    'wednesday_classics',
    'friday_youth',
    'other'
]


def classify_context(
    broadcast_datetime: pd.Timestamp,
    channel: str,
    movie_metadata: Dict[str, Any] = None
) -> ContextClass:
    """
    Classify context into one of 4 key RTS timeslots.

    Uses day-of-week, time (±20 min flexibility), and channel to assign contexts.

    Args:
        broadcast_datetime: Timestamp of broadcast
        channel: Channel name ('RTS 1' or 'RTS 2')
        movie_metadata: Optional movie metadata (not used for classification)

    Returns:
        Context class: one of 'saturday_family', 'saturday_action',
                      'wednesday_classics', 'friday_youth', or 'other'

    Key Timeslots (START TIME windows):
        - Saturday Family:    20:00-21:08 start, RTS 1
        - Saturday Action:    22:00-23:00 start, RTS 1
        - Wednesday Classics: 20:30-22:00 start, RTS 2
        - Friday Youth:       20:30-22:00 start, RTS 2

    Example:
        >>> dt = pd.Timestamp('2024-01-27 20:45:00')  # Saturday 20:45 start
        >>> classify_context(dt, 'RTS 1')
        'saturday_family'
    """
    day_of_week = broadcast_datetime.dayofweek  # 0=Monday, 5=Saturday
    hour = broadcast_datetime.hour
    minute = broadcast_datetime.minute

    # Convert to decimal time (e.g., 20:40 → 20.67)
    time_slot = hour + minute / 60.0

    # Normalize channel name (handle variations like 'RTS 1', 'RTS1', etc.)
    channel_normalized = channel.strip().upper().replace(' ', '')

    # Saturday Family: 20:00-21:08 start window, RTS 1
    # 20:00 = 20.0, 21:08 = 21.133
    if (day_of_week == 5 and
        20.0 <= time_slot <= 21.133 and
        'RTS1' in channel_normalized):
        return 'saturday_family'

    # Saturday Action: 22:00-23:00 start window, RTS 1
    # 22:00 = 22.0, 23:00 = 23.0
    if (day_of_week == 5 and
        22.0 <= time_slot <= 23.0 and
        'RTS1' in channel_normalized):
        return 'saturday_action'

    # Wednesday Classics: 20:30-22:00 start window, RTS 2
    # 20:30 = 20.5, 22:00 = 22.0
    if (day_of_week == 2 and
        20.5 <= time_slot <= 22.0 and
        'RTS2' in channel_normalized):
        return 'wednesday_classics'

    # Friday Youth: 20:30-22:00 start window, RTS 2
    # 20:30 = 20.5, 22:00 = 22.0
    if (day_of_week == 4 and
        20.5 <= time_slot <= 22.0 and
        'RTS2' in channel_normalized):
        return 'friday_youth'

    return 'other'


def extract_genre_names(genres_array: np.ndarray) -> Set[str]:
    """
    Extract genre names from catalog genres array.

    Args:
        genres_array: numpy array of dicts like [{'id': 27, 'name': 'Horreur'}, ...]

    Returns:
        Set of genre names (in French)

    Example:
        >>> genres = np.array([{'id': 27, 'name': 'Horreur'}, {'id': 18, 'name': 'Drame'}])
        >>> extract_genre_names(genres)
        {'Horreur', 'Drame'}
    """
    if genres_array is None or len(genres_array) == 0:
        return set()

    genre_names = set()
    for genre_dict in genres_array:
        if isinstance(genre_dict, dict) and 'name' in genre_dict:
            genre_names.add(genre_dict['name'])

    return genre_names


def is_movie_relevant_for_context(
    movie_metadata: Dict[str, Any],
    context_class: ContextClass
) -> bool:
    """
    Determine if a movie is relevant for a given context class.

    Uses expert rules based on RTS editorial briefs.
    NOTE: RTS doesn't have standard age ratings, so we use 'adult' flag from TMDB.

    Args:
        movie_metadata: Movie metadata dict with keys:
            - 'genres': np.ndarray of dicts [{'id': int, 'name': str}, ...]
            - 'production_year': float (release year, can be NaN)
            - 'original_language': str (language code: 'fr', 'en', 'unknown', etc.)
            - 'adult': bool (TMDB adult content flag)

    Returns:
        True if movie is appropriate for this context, False otherwise

    Editorial Rules:
        - saturday_family: adult=False, year 1980-2025, genres in {Familial, Comédie, Animation},
                          language in {fr, en}
        - saturday_action: adult=False, genres contain "Action"
        - wednesday_classics: year 1990-2015 (classics era)
        - friday_youth: genres in {Action, Science-Fiction, Aventure, Fantastique}
        - other: No constraints (all movies relevant)

    Example:
        >>> movie = {
        ...     'genres': np.array([{'id': 28, 'name': 'Action'}]),
        ...     'production_year': 2015,
        ...     'original_language': 'en',
        ...     'adult': False
        ... }
        >>> is_movie_relevant_for_context(movie, 'saturday_action')
        True
    """
    if context_class == 'other':
        return True  # No specific constraints

    # Extract metadata with defaults
    genres_array = movie_metadata.get('genres', np.array([]))
    genre_names = extract_genre_names(genres_array)

    year = movie_metadata.get('production_year', np.nan)
    if pd.isna(year):
        year = 0

    language = movie_metadata.get('original_language', 'unknown')
    adult = movie_metadata.get('adult', False)

    # Saturday Family (20:00-21:08 start, RTS 1)
    # Editorial brief: Family-friendly movie for broad audience
    # Examples: "Mystère à Saint-Tropez", "La fine fleur", "Les Tuche 4",
    #           "Qu'est-ce qu'on a fait au Bon Dieu", "Belle and Sebastian"
    if context_class == 'saturday_family':
        # Must be family-appropriate (not adult content)
        if adult:
            return False

        # Year range: 1980-2025 (relatively modern)
        if not (1980 <= year <= 2025):
            return False

        # Genre: Must have at least one family-friendly genre (French names)
        family_genres = {'Familial', 'Comédie', 'Animation', 'Aventure'}
        if not (genre_names & family_genres):
            return False

        return True

    # Saturday Action (22:00-23:00 start, RTS 1)
    # Editorial brief: Action-oriented for slightly later audience
    # Examples: "Jason Bourne", "Equalizer", "Ryan Initiative", "Northman", "Dune"
    if context_class == 'saturday_action':
        # Must be family-appropriate (RTS policy - no explicit content)
        if adult:
            return False

        # Must have Action or Thriller genre
        action_genres = {'Action', 'Thriller'}
        if not (genre_names & action_genres):
            return False

        return True

    # Wednesday Classics (20:30-22:00 start, RTS 2)
    # Editorial brief: TV classics for adult audience (30-49)
    # Year range defines "classics" era
    if context_class == 'wednesday_classics':
        # Year range: 1975-2015 (classic era as defined by RTS)
        if not (1975 <= year <= 2015):
            return False

        return True

    # Friday Youth (20:30-22:00 start, RTS 2)
    # Editorial brief: Youth-oriented, high-energy (superhero, Marvel-type)
    # Examples: "Edge of Tomorrow", "Kingsman", "Spider-Verse"
    if context_class == 'friday_youth':
        # Genre: Youth-oriented action/sci-fi (French names)
        # Note: TMDB doesn't have "Superhero" as a genre, but these films
        # are typically tagged as Action/Science-Fiction/Aventure/Fantastique
        youth_genres = {
            'Action', 'Science-Fiction', 'Aventure', 'Fantastique'
        }
        if not (genre_names & youth_genres):
            return False

        return True

    return False


def get_context_class_stats(
    historical_df: pd.DataFrame,
    catalog_df: pd.DataFrame = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistics about context class distribution in historical data.

    Args:
        historical_df: Historical programming DataFrame with columns:
            - 'date': pd.Timestamp
            - 'channel': str
            - 'catalog_id': str (optional)
        catalog_df: Optional catalog DataFrame (not currently used)

    Returns:
        Dict mapping context class names to stats:
        {
            'saturday_family': {
                'count': int,
                'fraction': float,
                'example_dates': List[pd.Timestamp]
            },
            ...
        }

    Example:
        >>> stats = get_context_class_stats(historical, catalog)
        >>> print(f"Saturday family: {stats['saturday_family']['count']} decisions")
    """
    stats = {
        'saturday_family': {'count': 0, 'dates': []},
        'saturday_action': {'count': 0, 'dates': []},
        'wednesday_classics': {'count': 0, 'dates': []},
        'friday_youth': {'count': 0, 'dates': []},
        'other': {'count': 0, 'dates': []}
    }

    for _, row in historical_df.iterrows():
        # Create broadcast datetime using hour column (handles broadcast hours 6-29)
        # Note: start_time uses broadcast format (e.g., "25:09:42" for late-night)
        # We use the 'hour' column which is already extracted from start_time
        date_part = pd.Timestamp(row['date'])
        hour = int(row['hour'])

        # Create datetime - if hour >= 24, it represents next day
        if hour >= 24:
            broadcast_datetime = date_part + pd.Timedelta(days=1, hours=hour-24)
        else:
            broadcast_datetime = date_part + pd.Timedelta(hours=hour)

        channel = row['channel']

        context_class = classify_context(broadcast_datetime, channel)
        stats[context_class]['count'] += 1
        stats[context_class]['dates'].append(broadcast_datetime)

    # Compute fractions
    total = len(historical_df)
    for context_class in stats:
        stats[context_class]['fraction'] = stats[context_class]['count'] / total
        # Keep only first 5 example dates
        stats[context_class]['example_dates'] = stats[context_class]['dates'][:5]
        del stats[context_class]['dates']

    return stats


def get_relevant_movies_for_context(
    catalog_df: pd.DataFrame,
    context_class: ContextClass,
    available_catalog_ids: Set[str] = None
) -> Set[str]:
    """
    Get all movies from catalog that are relevant for a given context.

    Args:
        catalog_df: Catalog DataFrame with movie metadata (indexed by catalog_id)
        context_class: Context class to filter for
        available_catalog_ids: Optional set of available catalog IDs to filter

    Returns:
        Set of catalog_ids that are relevant for this context

    Example:
        >>> relevant = get_relevant_movies_for_context(
        ...     catalog, 'saturday_family', available_ids
        ... )
        >>> print(f"Found {len(relevant)} family-friendly movies")
    """
    if available_catalog_ids is None:
        available_catalog_ids = set(catalog_df.index)

    relevant_ids = set()

    for catalog_id in available_catalog_ids:
        if catalog_id not in catalog_df.index:
            continue

        movie_row = catalog_df.loc[catalog_id]

        # Convert row to metadata dict
        movie_metadata = {
            'genres': movie_row.get('genres', np.array([])),
            'production_year': movie_row.get('production_year', np.nan),
            'original_language': movie_row.get('original_language', 'unknown'),
            'adult': movie_row.get('adult', False)
        }

        if is_movie_relevant_for_context(movie_metadata, context_class):
            relevant_ids.add(catalog_id)

    return relevant_ids
