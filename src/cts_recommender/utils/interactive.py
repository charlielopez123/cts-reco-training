"""
Interactive testing utilities for CTS recommendation system.

This module provides display and UI helper functions for interactive testing
of the Contextual Thompson Sampler with human feedback.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from cts_recommender.environments.schemas import Context
from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.imitation_learning.IL_constants import SIGNAL_NAMES


def display_recommendations(
    recommendations: List[Dict[str, Any]],
    context: Context,
    test_date: pd.Timestamp,
    catalog_df: pd.DataFrame,
    env: TVProgrammingEnvironment,
    signal_names: Optional[List[str]] = None
) -> None:
    """
    Display recommendation slate with signal breakdowns.

    Args:
        recommendations: List of recommendation dictionaries containing catalog_id, score, signals, weights
        context: Context object with hour, day_of_week, channel, season
        test_date: The date for this recommendation context
        catalog_df: DataFrame containing movie catalog information
        env: TVProgrammingEnvironment instance (for audience prediction)
        signal_names: Optional list of signal names (defaults to SIGNAL_NAMES)
    """
    if signal_names is None:
        signal_names = SIGNAL_NAMES

    print(f"\n{'='*80}")
    print(f"DATE: {test_date.strftime('%Y-%m-%d')}")
    print(f"CONTEXT: {context.hour}:00 on {context.day_of_week_name()} | {context.channel.value} | {context.season.value}")
    print(f"{'='*80}\n")

    for i, rec in enumerate(recommendations, 1):
        movie = catalog_df.loc[rec['catalog_id']]

        # Get predicted audience rating (raw rt_m value, not normalized)
        _, predicted_rt_m = env.reward._calculate_audience_reward(
            context=context,
            movie=movie,
            air_date=test_date,
            return_raw_prediction=True
        )

        # Get rights end date
        rights_end = movie.get('tv_rights_end', pd.NaT)
        if pd.notna(rights_end):
            rights_end_str = pd.Timestamp(rights_end).strftime('%Y-%m-%d')
        else:
            rights_end_str = 'N/A'

        print(f"[{i}] {movie['title']} ({movie.get('release_date', 'N/A')[:4]})")
        print(f"    ID: {rec['catalog_id']}")
        print(f"    Overall Score: {rec['score']:.4f}")
        print(f"    Predicted Audience: {predicted_rt_m:.2f} | Rights End: {rights_end_str}\n")

        # Display signal contributions
        print("    Signal Contributions (weighted by CTS):")
        for idx, (sig_name, sig_val, weight) in enumerate(zip(signal_names, rec['signals'], rec['weights'])):
            contribution = sig_val * weight
            print(f"      [{idx}] {sig_name:15s}: {sig_val:.3f} × {weight:.3f} = {contribution:.4f}")
        print()


def get_user_choice() -> str:
    """
    Display options and get user input for recommendation selection.

    Returns:
        User input string (lowercased and stripped)
    """
    print("\nOptions:")
    print("  - Enter recommendation number [1-5] to select")
    print("  - Enter 'r' to regenerate without rejected movies")
    print("  - Enter 's' to skip this context")
    print("  - Enter 'q' to quit and end testing")

    return input("\nYour choice: ").strip().lower()


def get_signal_feedback(
    chosen_movie_title: str,
    signal_names: Optional[List[str]] = None
) -> Optional[np.ndarray]:
    """
    Ask user which signals were pertinent for their selection.

    Args:
        chosen_movie_title: Title of the selected movie
        signal_names: Optional list of signal names (defaults to SIGNAL_NAMES)

    Returns:
        Binary numpy array indicating which signals were pertinent, or None if no feedback
    """
    if signal_names is None:
        signal_names = SIGNAL_NAMES

    print(f"\nYou selected: {chosen_movie_title}")
    print("\nWhich value signals were pertinent for your selection?")
    print("Enter signal indices separated by spaces (e.g., '0 2 4' for audience, diversity, rights)")

    # Print signal index mapping
    for idx, sig_name in enumerate(signal_names):
        print(f"  {idx}={sig_name}", end="")
        if idx < len(signal_names) - 1:
            print(", ", end="")
    print()
    print("Or press Enter to indicate reward only (no signal feedback)")

    signal_input = input("\nSignal indices: ").strip()

    # Parse signal feedback
    if not signal_input:
        return None

    try:
        signal_indices = [int(x) for x in signal_input.split()]
        y_signals = np.zeros(len(signal_names))
        for idx in signal_indices:
            if 0 <= idx < len(signal_names):
                y_signals[idx] = 1.0
            else:
                print(f"Warning: Signal index {idx} out of range, ignoring")
        return y_signals
    except ValueError:
        print("Invalid signal input, proceeding without signal feedback")
        return None


def compute_signal_vector(
    catalog_id: str,
    test_date: pd.Timestamp,
    context: Context,
    context_features: np.ndarray,
    env,
    curator_model
) -> np.ndarray:
    """
    Compute the full signal vector for a given movie in a context.

    Args:
        catalog_id: Movie catalog ID
        test_date: Air date for the movie
        context: Context object
        context_features: Pre-computed context feature vector
        env: TVProgrammingEnvironment instance
        curator_model: Curator acceptance probability model

    Returns:
        Signal vector in SIGNAL_NAMES order: [curator, audience, competition, diversity, novelty, rights]
    """
    # Compute standard rewards
    rewards = env.reward.compute_total_reward(
        catalog_id=catalog_id,
        air_date=test_date,
        context=context,
        times_shown_tracker=None
    )

    # Compute curator acceptance probability
    movie_features = env.get_movie_features(catalog_id)
    combined_features = np.concatenate([context_features, movie_features])
    curator_prob = float(curator_model.predict_proba(combined_features.reshape(1, -1))[0, 1])

    # Build signal vector in SIGNAL_NAMES order (curator first)
    return np.array([
        curator_prob,           # curator FIRST
        rewards['audience'],
        rewards['competition'],
        rewards['diversity'],
        rewards['novelty'],
        rewards['rights'],
    ])


def print_summary(recommendation_history: List[Dict[str, Any]], signal_names: Optional[List[str]] = None) -> None:
    """
    Print summary statistics of the recommendation session.

    Args:
        recommendation_history: List of recommendation history dictionaries
        signal_names: Optional list of signal names (defaults to SIGNAL_NAMES)
    """
    if signal_names is None:
        signal_names = SIGNAL_NAMES

    print(f"\n{'='*80}")
    print(f"Completed {len(recommendation_history)} recommendations")
    print(f"{'='*80}")

    if len(recommendation_history) > 0:
        # Show which signals were indicated as pertinent
        signal_feedback_count = np.zeros(len(signal_names))
        for rec in recommendation_history:
            if rec.get('y_signals') is not None:
                signal_feedback_count += rec['y_signals']

        if signal_feedback_count.sum() > 0:
            print("\nSignals marked as pertinent:")
            for i, sig_name in enumerate(signal_names):
                if signal_feedback_count[i] > 0:
                    print(f"  - {sig_name}: {int(signal_feedback_count[i])} times")
