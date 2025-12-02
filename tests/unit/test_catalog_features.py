"""Tests for catalog feature processing."""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from cts_recommender.preprocessing.catalog_features import (
    finalize_catalog_features,
    build_catalog_features,
)
from cts_recommender.preprocessing.feature_transformations import (
    process_tmdb_features,
    add_genre_features,
    add_movie_age_feature,
)


class TestProcessTmdbFeaturesForCatalog:
    """Tests for process_tmdb_features function (catalog use case)."""

    def test_basic_processing(self):
        """Test basic TMDB feature processing."""
        df = pd.DataFrame({
            'tmdb_id': [123, 456],
            'adult': [False, True],
            'original_language': ['en', 'fr'],
            'genres': [[], []],
            'release_date': ['2020-01-01', '2021-06-15'],
            'revenue': [1000000, 2000000],
            'vote_average': [7.5, 8.2],
            'popularity': [100.5, 200.3],
        })

        result = process_tmdb_features(df, is_movie=True)

        assert 'is_movie' in result.columns
        assert result['is_movie'].all()
        assert 'missing_tmdb' in result.columns
        assert not result['missing_tmdb'].any()

    def test_missing_tmdb_id(self):
        """Test handling of missing TMDB IDs."""
        df = pd.DataFrame({
            'tmdb_id': [123, np.nan],
            'adult': [False, False],
            'original_language': ['en', 'fr'],
            'genres': [[], []],
            'release_date': ['2020-01-01', '2021-06-15'],
            'revenue': [1000000, 0],
            'vote_average': [7.5, 0],
            'popularity': [100.5, 0],
        })

        result = process_tmdb_features(df, is_movie=True)

        assert result.loc[0, 'missing_tmdb'] == False
        assert result.loc[1, 'missing_tmdb'] == True

    def test_missing_values_filled(self):
        """Test that missing values are properly filled."""
        df = pd.DataFrame({
            'tmdb_id': [123],
            'adult': [np.nan],
            'original_language': [np.nan],
            'genres': [[]],
            'release_date': [np.nan],
            'revenue': [np.nan],
            'vote_average': [np.nan],
            'popularity': [np.nan],
        })

        result = process_tmdb_features(df, is_movie=True)

        assert result['adult'].iloc[0] == False
        assert result['original_language'].iloc[0] == 'unknown'
        assert result['release_date'].iloc[0] == '1900-01-01'
        assert result['revenue'].iloc[0] == 0
        assert result['vote_average'].iloc[0] == 0
        assert result['popularity'].iloc[0] == 0


class TestAddGenreFeatures:
    """Tests for add_genre_features function."""

    def test_single_genre(self):
        """Test one-hot encoding with single genre."""
        df = pd.DataFrame({
            'genres': [[{'name': 'Action'}]]
        })

        result = add_genre_features(df)

        assert 'genre_Action' in result.columns
        assert result['genre_Action'].iloc[0] == 1
        assert 'genres' not in result.columns

    def test_multiple_genres(self):
        """Test one-hot encoding with multiple genres."""
        df = pd.DataFrame({
            'genres': [
                [{'name': 'Action'}, {'name': 'Comedy'}],
                [{'name': 'Drama'}]
            ]
        })

        result = add_genre_features(df)

        assert 'genre_Action' in result.columns
        assert 'genre_Comedy' in result.columns
        assert 'genre_Drama' in result.columns
        assert result.loc[0, 'genre_Action'] == 1
        assert result.loc[0, 'genre_Comedy'] == 1
        assert result.loc[1, 'genre_Drama'] == 1


class TestAddMovieAgeFeature:
    """Tests for add_movie_age_feature function."""

    def test_movie_age_calculation(self):
        """Test movie age is calculated correctly."""
        current_year = datetime.now().year
        df = pd.DataFrame({
            'release_date': ['2020-01-01', '2000-06-15']
        })

        result = add_movie_age_feature(df)

        assert 'movie_age' in result.columns
        assert result['movie_age'].iloc[0] == current_year - 2020
        assert result['movie_age'].iloc[1] == current_year - 2000
        # release_date is kept for diversity reward calculations
        assert 'release_date' in result.columns


class TestFinalizeCatalogFeatures:
    """Tests for finalize_catalog_features function."""

    def test_bool_to_int_conversion(self):
        """Test boolean columns converted to integers."""
        df = pd.DataFrame({
            'is_movie': [True, False, True],
            'missing_tmdb': [False, True, False],
            'title': ['Movie A', 'Movie B', 'Movie C']
        })

        result = finalize_catalog_features(df)

        assert result['is_movie'].dtype == 'int64'
        assert result['missing_tmdb'].dtype == 'int64'
        assert result['is_movie'].tolist() == [1, 0, 1]


class TestBuildCatalogFeatures:
    """Tests for build_catalog_features function."""

    def test_complete_pipeline(self):
        """Test complete catalog feature building pipeline."""
        df = pd.DataFrame({
            'tmdb_id': [123, 456],
            'adult': [False, True],
            'original_language': ['en', 'fr'],
            'genres': [
                [{'name': 'Action'}],
                [{'name': 'Drama'}]
            ],
            'release_date': ['2020-01-01', '2019-06-15'],
            'revenue': [1000000, 2000000],
            'vote_average': [7.5, 8.2],
            'popularity': [100.5, 200.3],
        })

        result = build_catalog_features(df)

        # Check genres are kept as list (not one-hot encoded for catalog)
        assert 'genres' in result.columns
        assert isinstance(result['genres'].iloc[0], list)

        # Check keywords column was added (even if empty)
        assert 'keywords' in result.columns

        # Check movie age was calculated
        assert 'movie_age' in result.columns

        # Check booleans converted to int
        assert result['is_movie'].dtype == 'int64'

        # Check tmdb_id is preserved (needed for catalog)
        assert 'tmdb_id' in result.columns

        # Check genres and release_date are preserved (not one-hot encoded for catalog)
        assert 'genres' in result.columns
        assert 'release_date' in result.columns
