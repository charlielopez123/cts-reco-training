PSEUDO_REWARD_WEIGHTS = {
    'audience': 0.4,       # Prioritizes strong viewership, but with room for public service trade-offs
    'competition': 0.15,   # Avoids clashing with competitors, but not overly reactive
    'diversity': 0.1,      # Promotes balanced representation across genres, demographics, etc.
    'novelty': 0.2,        # Encourages programming that is fresh or less repetitive
    'rights': 0.15         # Incentivizes using content with soon-to-expire rights
}

# Signal names in canonical order (matches production's all_reward_feature_names)
# Index 0: curator, Index 1-5: value signals
# IMPORTANT: This order must match production signal extraction in contextual_thompson.py
SIGNAL_NAMES = ['curator', 'audience', 'competition', 'diversity', 'novelty', 'rights']

# Target signal weights for CTS initialization (all 6 signals, must sum to 1.0)
# These are prior beliefs about signal importance; the model learns from data.
# Order matches SIGNAL_NAMES: [curator, audience, competition, diversity, novelty, rights]
TARGET_SIGNAL_WEIGHTS = {
    'curator': 0.30,       # Curator expertise signal (from logistic regression model)
    'audience': 0.28,      # 0.4 * 0.7 (scaled to leave room for curator)
    'competition': 0.105,  # 0.15 * 0.7
    'diversity': 0.07,     # 0.1 * 0.7
    'novelty': 0.14,       # 0.2 * 0.7
    'rights': 0.105        # 0.15 * 0.7
}