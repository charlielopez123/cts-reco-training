# Feature extraction specifications for TMDB movie details as described https://developer.themoviedb.org/reference/movie-details
from typing import Any, Callable, Dict, Iterable, Optional, List
from datetime import date

# 1) Describe how to extract each feature: (path_in_details, transform, default)
# - path can be a key or a dotted path for nested dicts
FeatureSpec = Dict[str, tuple[str, Optional[Callable[[Any], Any]], Any]]

BASIC_FEATURES: FeatureSpec = {
    "adult":               ("adult",               bool,                         False),
    "original_language":   ("original_language",   str,                          'unknown'),
    "popularity":          ("popularity",          float,                        0.0),
    "poster_path":         ("poster_path",         str,                          None),
    "release_date":        ("release_date",        str,                          None),
    "revenue":             ("revenue",             int,                          0),
    "vote_average":        ("vote_average",        float,                        0.0),
    "genres":              ("genres",              List[Dict[str, Any]],          []),
    "keywords":            ("keywords",            List[Dict[str, Any]],          []),  # Extracted from nested keywords.keywords
}

