# Constants specific to RTS Analytics department for parsing programming schedules, extraction of movie specific data

# Catalog ID prefix for WhatsOn movies without TMDB IDs
WHATSON_CATALOG_ID_PREFIX = "WO_"

RTS_MOVIE_CLASSKEYS = ['71', '72', '761']
RTS_WHATSON_CLASSKEYS = ['72 - Téléfilms', '71 - Films de cinéma', "761 - Longs métrages d'animation"]
RTS_COMPETITOR_MOVIE_CODES = ['AAA', 'AAB', 'AAC', 'AAD', 'AAF', 'AAG', 'AAH', 'AAL', 'ABA', 'ABB' 'ABC', 'ABD', 'ABF', 'ABM']
ALL_MOVIE_CLASSKEYS = RTS_MOVIE_CLASSKEYS + RTS_COMPETITOR_MOVIE_CODES + RTS_WHATSON_CLASSKEYS

# Specific titles where it is not the actual title of the content but a Special categorisation, in these cases retrieve from description column
RTS_SPECIAL_MOVIE_NAMES = ['Le film de minuit, 40 ans de grands frissons: Film de minuit', 'Edition spéciale Noël', 'Box office', 'Nocturne', 'Film de minuit',
       'Les classiques du cinéma', 'Ramdam, les films...',
       'Ramdam - La culture en films', 'Le film de minuit, 40 ans de grands frissons: Film de minuit']


HISTORICAL_PROGRAMMING_COLUMNS = [
        'catalog_id',
        'date',
        'start_time',
        'channel',
        'title',
        'processed_title',
        'duration_min',
        'rt_m',
        'pdm',
        'hour',
        'weekday',
        'is_weekend',
        'season',
        'public_holiday',
        'tmdb_id',  # Keep for matching purposes, but features should come from catalog
        'missing_tmdb_id',  # Data quality flag
    ]

INTEREST_CHANNELS = ['RTS 1', 'RTS 2']

# General mode context features (18 dims)
# time_slot(4) + day(7) + weekend(1) + season(4) + channel(2)
CONTEXT_FEATURE_NAMES = [
    "prime_time",
    "late_night",
    "afternoon",
    "morning",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "is_weekend",
    "spring",
    "summer",
    "autumn",
    "winter",
    "RTS 1",
    "RTS 2",
]

# RTS curtain mode context features (21 dims)
# curtain_type(7) + day(7) + weekend(1) + season(4) + channel(2)
CURTAIN_CONTEXT_FEATURE_NAMES = [
    "fiction_matin",
    "apres_midi_1",
    "apres_midi_2",
    "rideau_1",
    "rideau_2",
    "rideau_3",
    "rideau_4",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "is_weekend",
    "spring",
    "summer",
    "autumn",
    "winter",
    "RTS 1",
    "RTS 2",
]

# Competition Channels
COMPETITOR_CHANNELS = [
    "France 2", "France 3", "M6_T_PL", "TF1_T_PL"
]

# M6 and TF1 use different classification codes than RTS:
# Code '1' = General programming (includes movies, TV shows, reality TV, etc.)
# Code '2' = Shopping/infomercials (M6 BOUTIQUE, TELE SHOPPING)
# For M6/TF1 movie extraction, use Code '1' + duration-based filtering (>=75 min)
M6_TF1_CLASSKEYS = ['1', '2']
M6_TF1_GENERAL_PROGRAMMING = '1'
M6_TF1_SHOPPING = '2'
