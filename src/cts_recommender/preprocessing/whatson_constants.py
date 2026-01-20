# ROBUST EPISODE DETECTION PATTERNS
EPISODE_PATTERNS = [
    r'[eéè]p\.\s*\d+',           # ép.1, ep.1, ép. 2, ep. 2, etc.
    r'[eéè]pisode\s*\d+',        # épisode 1, episode 1, èpisode 2
    r'^E\d+',                    # E1, E23, etc. (at start)
    r'S\d+E\d+',                 # S01E01, S1E2, etc.
    r'^\d+$',                    # Just a number (like "1", "2")
    r'^\d+\s*-\s*',              # "1 - Something", "2 - Title"
    r'partie\s*\d+',             # partie 1, partie 2
    r'\bpart\s*\d+',             # part 1, part 2
]

# IGNORE PATTERNS - titles containing these words should NOT be considered movies
IGNORE_PATTERNS = [
    r'\bdoublons?\b',            # doublon, doublons (word boundary)
    r'\bs[eéè]ries?\b',          # serie, series, série, séries (word boundary)
]

# Date patterns in title
DATE_PATTERNS = [
    r'\d{4}-\d{2}-\d{2}',    # YYYY-MM-DD
    r'\d{2}-\d{2}-\d{2}',    # YY-MM-DD or DD-MM-YY
    r'\d{2}\.\d{2}\.\d{2}',  # YY.MM.DD or DD.MM.YY
    r'\d{2}/\d{2}/\d{2}',    # YY/MM/DD or DD/MM/YY
]

# Known film collections (whitelist)
FILM_COLLECTIONS = [
    'Film',
    'Téléfilm', 
    'Fiction',
    'Fiction\\Achats',
    'Cinéma',
    'Film d\'action',
    'Emotions fortes',
    'Comédie',
    'Film Jeunesse',
    'Film de minuit',
    'Ecran TV',
    'Les classiques du cinéma',
    'Court métrage',
    'Nocturne',
    'Box office'
]

# Keywords that suggest a collection is film-related
FILM_COLLECTION_KEYWORDS = [
    'film', 'cinéma', 'cinema', 'fiction', 'téléfilm',
    'comédie', 'action', 'classique', 'nocturne', 'écran'
]

DATE_COLS = [
        'tv_rights_start', 'tv_rights_end',
        'first_broadcast_date', 'last_broadcast_date',
        'rebroadcast_date_1', 'rebroadcast_date_2', 'rebroadcast_date_3', 'rebroadcast_date_4'
    ]