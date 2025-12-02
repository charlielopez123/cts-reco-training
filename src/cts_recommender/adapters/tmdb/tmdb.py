# Folder in charge of TMDB API interactions
from cts_recommender.adapters.tmdb.client import TMDB_APIClient
from requests.exceptions import HTTPError, Timeout, RequestException
from typing import Any, Callable, Dict, Optional
from cts_recommender.utils import text_cleaning
from cts_recommender.features.tmdb_extract import BASIC_FEATURES
import logging
import re


logger = logging.getLogger(__name__)

def detect_title_language(title: str) -> str:
    """
    Detect the likely language of a movie title to optimize TMDB search.

    Strategy:
    - Check for common language indicators in first 5 words (most reliable)
    - Supports: English, French, Italian, German, Spanish
    - Default to French for RTS catalog context (Swiss broadcaster)

    Args:
        title: The movie title to analyze

    Returns:
        Language code: 'en', 'fr', 'it', 'de', or 'es'

    Examples:
        "The Dark Knight" -> 'en'
        "Le Fabuleux Destin d'Amélie Poulain" -> 'fr'
        "La vita è bella" -> 'it'
        "Der Untergang" -> 'de'
        "El laberinto del fauno" -> 'es'
    """
    if not title:
        return 'fr'

    title_lower = title.lower()
    words = title_lower.split()

    # Common language indicators (articles, prepositions, conjunctions)
    # These are words that are distinctive and unlikely to appear in other languages
    english_indicators = ['the', 'of', 'and', 'you', 'an', 'in', 'to', 'for', 'with', 'at', 'from', 'about', 'aka', 'on']
    french_indicators = ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'dans', 'pour', 'avec', 'quand', 'comme']
    italian_indicators = ['il', 'lo', 'gli', 'uno', 'del', 'della', 'dello', 'degli', 'delle', 'di', 'che', 'per', 'con', 'come', 'dove', 'quando']
    german_indicators = ['der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem', 'eines', 'und', 'von', 'mit', 'für', 'aus']
    spanish_indicators = ['el', 'los', 'un', 'unos', 'unas', 'del', 'al', 'y', 'en', 'con', 'para', 'por', 'como', 'que', 'donde']

    # Check first 5 words for language indicators (most reliable)
    first_words = words[:5]

    # Count matches for each language
    english_matches = sum(1 for word in first_words if word in english_indicators)
    french_matches = sum(1 for word in first_words if word in french_indicators)
    italian_matches = sum(1 for word in first_words if word in italian_indicators)
    german_matches = sum(1 for word in first_words if word in german_indicators)
    spanish_matches = sum(1 for word in first_words if word in spanish_indicators)

    # Create a dictionary of language matches
    language_scores = {
        'en': english_matches,
        'fr': french_matches,
        'it': italian_matches,
        'de': german_matches,
        'es': spanish_matches
    }

    # Find the language with the most matches
    max_matches = max(language_scores.values())

    # If no matches found, default to French (RTS context)
    if max_matches == 0:
        return 'fr'

    # Return the language with the highest score
    # In case of ties, priority order: en, fr, it, de, es
    for lang in ['en', 'fr', 'it', 'de', 'es']:
        if language_scores[lang] == max_matches:
            return lang

    # Fallback (should never reach here)
    return 'fr'

class TMDB_API():
    """Wrapper class for TMDB API interactions"""
    def __init__(self):
        self._client:  TMDB_APIClient = TMDB_APIClient()


    def search_movie(self, title: str, language: str = None) -> Dict[str, Any]:
        """
        Searches for a movie by title and returns the different potential TMDB ids.

        Args:
            title: Movie title to search for
            language: Language code ('en', 'fr', etc.). If None, auto-detects from title.

        Returns:
            TMDB API response with search results
        """
        if language is None:
            language = detect_title_language(title)
            logger.debug(f"Auto-detected language '{language}' for title: {title}")

        response = self._client.get('search/movie', params={'query': title, 'include_adult': 'true', 'language': language})
        return response

    def get_movie_details(self, movie_id: str) -> Dict[str, Any] | None:
        """Fetches full movie details from a given movie_id"""
        try:
            response = self._client.get(f'movie/{movie_id}', params={
                'language': 'fr',
                'append_to_response': 'keywords'
            })
            return response
        except HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Movie ID {movie_id} not found (404)")
                return None
            raise
    
    def get_movie_title(self, movie_id: str) -> str | None:
        """Fetches the title of a movie from a given movie_id"""
        details = self.get_movie_details(movie_id)
        if details is None or not bool(details):
            title = None
        else:
            title = details["title"]
        return title
    
    def find_best_match(self, title: str, known_runtime, top_n = 10, language: str = None) -> int | None:
        """
        Searches for a movie by title and finds the best match based on title and uses movie duration as a secondary factor.

        Args:
            title: Movie title to search for
            known_runtime: Expected runtime in minutes (used for tie-breaking)
            top_n: Number of top results to consider (default: 10)
            language: Language code ('en', 'fr', etc.). If None, auto-detects from title.

        Returns:
            TMDB movie ID of the best match, or None if no match found
        """


        response = self.search_movie(title, language=language)
        num_results = len(response['results'])



        # If no results found initially try decomposing the title
        if num_results == 0:
            best_id = self.find_best_id_decomposed(title, known_runtime, language=language)
            return best_id
        
        # If only one result is found return it
        if num_results==1:
            best_id = response["results"][0]['id']
            return best_id

        # Retrieve the top movie ids from the search results
        logger.debug(f"Found {num_results} results for title '{title}'")
        logger.debug(f"length of results: {len(response['results'])}")
        top_movie_ids = [response["results"][i]['id'] for i in range(num_results)]

        #Search for top_n ids if there are more results than top_n
        if top_n > num_results:
            top_n = num_results

        # if translated title is the same as the provided title return the corresponding movie_id
        for movie in response["results"][:top_n]:
            n_title = text_cleaning.normalize(title)
            n_tmdb_title = text_cleaning.normalize(movie["title"])
            if ( (n_title in n_tmdb_title) or (n_tmdb_title in n_title) ):
                best_id = movie['id']
                return best_id
            
        
        # Assuming it is the most unlikely case that the first movie has a runtime of 0 and the match is still wrong
        if num_results==1:
            best_id = response["results"][0]['id']
            movie_details = self.get_movie_details(best_id)
            print(f"Single result found for '{title}': {movie_details}")
            if movie_details["runtime"] == 0 or known_runtime == 0:
                return best_id 

        # Compare the runtime of the top_n movies with the known runtime and find the closest match
        top_n_movie_ids = top_movie_ids[:top_n]
        best_id = None
        lowest_diff = float("inf")
        for i, id in enumerate(top_n_movie_ids):
            details = self.get_movie_details(id)
            if details is None:
                logger.warning(f"Skipping movie ID {id} - details not found")
                continue
            runtime = details["runtime"]
            diff = abs(runtime - known_runtime)
            if diff < lowest_diff:
                lowest_diff = diff
                best_id = id
        return best_id
    
    def find_best_id_decomposed(self, title:str, known_runtime: int, language: str = None) -> int | None:
        """
        If no results are found for the full title, try to decompose the title by common separators.
        Handles titles like:
        - 'Comme je ferme les yeux (The Shameless)'
        - 'Birds of prey (et la fabuleuse histoire de Harley Quinn)'
        - 'Movie: Subtitle'
        - 'Title - Another Title'

        Args:
            title: Movie title to decompose and search
            known_runtime: Expected runtime in minutes
            language: Language code for TMDB search. If None, auto-detects for each part.
        """

        # Try extracting content from parentheses first (e.g., 'Title (English Title)')
        if "(" in title and ")" in title:
            match = re.match(r'^(.+?)\s*\((.+?)\)\s*$', title)
            if match:
                title_before = match.group(1).strip()
                title_inside = match.group(2).strip()

                # Skip if content inside parentheses is a year or date (e.g., '(2023)' or '(1995)')
                if not re.match(r'^\d{4}$', title_inside):
                    # Try inside parentheses first (often the English title), then the part before
                    # Each part will auto-detect its own language
                    for part in (title_inside, title_before):
                        best_id = self.find_best_match(part, known_runtime, language=None)
                        if best_id:
                            return best_id

        # try splitting by common separators and searching each part
        if ":" in title:
            left, right = title.split(":", 1)
            for part in (left.strip(), right.strip()):
                best_id = self.find_best_match(part, known_runtime, language=None)
                if best_id:
                    return best_id

        if "-" in title:
            left, right = title.split("-", 1)
            for part in (left.strip(), right.strip()):
                best_id = self.find_best_match(part, known_runtime, language=None)
                if best_id:
                    return best_id

        # give up
        return None
    
    def get_movie_features(self, movie_id: str) -> Dict[str, Any] | None:

        details = self.get_movie_details(movie_id)

        if details is None:
            return None

        movie_features = {k: v for k, v in details.items() if k in BASIC_FEATURES.keys()}

        # Extract nested keywords list from keywords.keywords structure
        if 'keywords' in movie_features and isinstance(movie_features['keywords'], dict):
            movie_features['keywords'] = movie_features['keywords'].get('keywords', [])

        return movie_features