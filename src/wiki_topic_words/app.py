import requests
import argparse
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import sys
import time
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# --- Configuration ---
# 환경 변수 또는 별도의 설정 파일에서 User-Agent를 로드하는 것이 더 안전합니다.
# 예시: os.environ.get("WIKI_BOT_USER_AGENT", "MyWikiWordFreqBot/1.1 (https://example.com/bot; myemail@example.com) requests")
DEFAULT_USER_AGENT = (
    "MyWikiWordFreqBot/1.1 (https://example.com/bot; myemail@example.com) requests"
)
DEFAULT_CACHE_TTL = 24 * 60 * 60  # Default cache validity: 24 hours in seconds


class Config:
    """Class to hold configuration parameters."""

    API_URL: str = "https://en.wikipedia.org/w/api.php"
    HEADERS: Dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}
    BATCH_SIZE_PAGES: int = 50
    RETRY_DELAY: int = 5
    CACHE_DIR: Path = Path(
        "./.wiki_cache"
    )  # Store cache files in a hidden subdirectory
    DEFAULT_CACHE_TTL: int = DEFAULT_CACHE_TTL


# --- NLTK Stopwords Setup ---
class NLTKInitializer:
    """Handles the download and loading of NLTK stopwords."""

    _STOP_WORDS: Optional[set] = None

    @classmethod
    def get_stopwords(cls) -> set:
        """
        Ensures NLTK stopwords are downloaded and loaded, then returns them.
        Uses a class-level cache to avoid re-downloading/reloading.
        """
        if cls._STOP_WORDS is None:
            cls._download_and_load_stopwords()
        return cls._STOP_WORDS

    @classmethod
    def _download_and_load_stopwords(cls):
        """Downloads the NLTK stopwords corpus if not already available and loads them."""
        print("Attempting to initialize NLTK stopwords...")
        try:
            # 먼저 stopwords가 있는지 확인
            nltk.data.find("corpora/stopwords")
            print("NLTK 'stopwords' corpus found locally.")
        except LookupError:
            print(
                "NLTK 'stopwords' corpus not found locally. Attempting to download...",
                file=sys.stderr,
            )
            try:
                # 다운로드 시도
                # quiet=True를 사용하여 다운로드 중 발생하는 불필요한 메시지를 줄일 수 있습니다.
                nltk.download("stopwords", quiet=False)
                print("NLTK stopwords corpus downloaded successfully.")
            except Exception as e:
                # 다운로드 실패 시 더 구체적인 오류 메시지 출력 후 종료
                print(f"Failed to download NLTK stopwords corpus: {e}", file=sys.stderr)
                print(
                    "Please try running 'python -m nltk.downloader stopwords' manually.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except Exception as e:  # 다른 예상치 못한 예외 처리
            print(
                f"An unexpected error occurred during NLTK stopwords check: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 다운로드 성공 또는 이미 존재할 경우 stopwords 로드
        try:
            cls._STOP_WORDS = set(stopwords.words("english"))
            print("NLTK stopwords corpus is available and loaded into memory.")
        except LookupError:  # 다운로드 성공했으나 로드 실패하는 경우 (경로 문제 등)
            print(
                "Error: NLTK stopwords corpus was downloaded but could not be loaded.",
                file=sys.stderr,
            )
            print("Please check NLTK data paths and permissions.", file=sys.stderr)
            print(f"NLTK search paths: {nltk.data.path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:  # 로드 중 발생하는 기타 예외
            print(
                f"An unexpected error occurred while loading NLTK stopwords: {e}",
                file=sys.stderr,
            )
            sys.exit(1)


# --- Cache Management ---
class CacheManager:
    """Manages reading from and writing to JSON cache files."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(
            parents=True, exist_ok=True
        )  # Ensure cache directory exists

    def _sanitize_filename(self, name: str) -> str:
        """Removes or replaces characters problematic for filenames."""
        name = re.sub(r'[\\/*?:"<>|\s]+', "_", name)
        name = re.sub(r"[^\w\-\_]+", "", name)
        return name[:100].strip("_")

    def get_cache_path(self, base_name: str, category_name: str) -> Path:
        """Generates a sanitized cache file path."""
        sanitized_category = self._sanitize_filename(category_name)
        return self.cache_dir / f"{base_name}_{sanitized_category}.json"

    def read_cache(self, path: Path, max_age_seconds: int) -> Optional[Any]:
        """Reads cache data from a JSON file if it exists and is not expired."""
        if not path.exists():
            return None

        try:
            file_mod_time = path.stat().st_mtime
            if time.time() - file_mod_time > max_age_seconds:
                print(f"  Cache file expired: {path.name}")
                path.unlink()
                return None

            with path.open("r", encoding="utf-8") as f:
                cache_data = json.load(f)

            if "data" in cache_data:
                print(f"  Cache hit: {path.name}")
                return cache_data["data"]
            else:
                print(
                    f"  Cache file format error (missing 'data' key): {path.name}. Ignoring cache.",
                    file=sys.stderr,
                )
                return None

        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(
                f"  Error reading cache file {path.name}: {e}. Ignoring cache.",
                file=sys.stderr,
            )
            try:
                path.unlink()
            except OSError:
                pass  # Ignore if file cannot be unlinked
            return None

    def write_cache(self, path: Path, data: Any):
        """Writes data to a JSON cache file with a timestamp."""
        try:
            cache_content = {
                "timestamp": time.time(),
                "data": data,
            }
            with path.open("w", encoding="utf-8") as f:
                json.dump(cache_content, f, ensure_ascii=False, indent=2)
            print(f"  Cache written: {path.name}")
        except IOError as e:
            print(f"  Error writing cache file {path.name}: {e}", file=sys.stderr)


# --- Wikipedia Analyzer ---
class WikipediaCategoryAnalyzer:
    """
    Analyzes word frequencies within a specified Wikipedia category.
    Handles API requests, caching, and text processing.
    """

    def __init__(
        self, session: requests.Session, config: Config, cache_manager: CacheManager
    ):
        self.session = session
        self.config = config
        self.cache_manager = cache_manager
        self.stop_words = NLTKInitializer.get_stopwords()

    def _fetch_api_data(
        self, params: Dict[str, Any], endpoint: str = Config.API_URL
    ) -> Optional[Dict[str, Any]]:
        """
        Generic method to fetch data from the Wikipedia API with retries.
        """
        retries = 3
        while retries > 0:
            try:
                response = self.session.get(
                    endpoint, params=params, headers=self.config.HEADERS, timeout=30
                )
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                if "error" in data:
                    print(
                        f"API Error: {data['error'].get('info', 'Unknown error')}",
                        file=sys.stderr,
                    )
                    return None
                return data
            except requests.exceptions.Timeout:
                retries -= 1
                print(
                    f"Timeout fetching data. Retries left: {retries}",
                    file=sys.stderr,
                )
                time.sleep(self.config.RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                retries -= 1
                print(
                    f"Network error: {e}. Retries left: {retries}",
                    file=sys.stderr,
                )
                time.sleep(self.config.RETRY_DELAY)
            except json.JSONDecodeError:
                retries -= 1
                print(
                    "API response was not valid JSON. Retrying...",
                    file=sys.stderr,
                )
                time.sleep(self.config.RETRY_DELAY)
            except Exception as e:
                print(
                    f"An unexpected error occurred during API fetch: {e}",
                    file=sys.stderr,
                )
                return None
        print("Max retries exceeded. Aborting API request.", file=sys.stderr)
        return None

    def get_pages_in_category(
        self, category_title: str, use_cache: bool, cache_ttl: int
    ) -> Optional[List[str]]:
        """Fetches all page titles within a given Wikipedia category, using cache if enabled."""
        cache_path = self.cache_manager.get_cache_path("pages", category_title)

        if use_cache:
            cached_data = self.cache_manager.read_cache(cache_path, cache_ttl)
            if isinstance(cached_data, list):
                print(f"Using cached page list for Category:{category_title}")
                return cached_data
            elif cached_data is not None:  # Not None but not list
                print(
                    f"  Cached page data for '{category_title}' has unexpected format. Ignoring cache.",
                    file=sys.stderr,
                )

        print(f"Fetching pages in Category:{category_title} from API...")
        pages = []
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_title}",
            "cmlimit": "max",
            "cmtype": "page",
        }
        last_continue = {}

        while True:
            current_params = params.copy()
            current_params.update(last_continue)
            data = self._fetch_api_data(current_params)

            if data is None:
                return (
                    None  # _fetch_api_data already handled errors and printed messages
                )

            if "query" in data and "categorymembers" in data["query"]:
                found_pages = data["query"]["categorymembers"]
                page_titles = [page["title"] for page in found_pages]
                pages.extend(page_titles)
                print(f"  Fetched {len(found_pages)} pages (Total: {len(pages)})")

            if "continue" in data:
                last_continue = data["continue"]
                time.sleep(0.5)
            else:
                break

        print(f"Finished fetching pages. Found {len(pages)} total pages.")

        if pages and use_cache:  # Only write if pages were successfully fetched
            self.cache_manager.write_cache(cache_path, pages)

        return pages

    def get_pages_content(self, page_titles: List[str]) -> str:
        """Fetches plain text content for a list of page titles in batches."""
        all_text = ""
        print(f"Fetching content for {len(page_titles)} pages...")

        for i in range(0, len(page_titles), self.config.BATCH_SIZE_PAGES):
            batch_titles = page_titles[i : i + self.config.BATCH_SIZE_PAGES]
            print(
                f"  Fetching batch {i // self.config.BATCH_SIZE_PAGES + 1}/{(len(page_titles) + self.config.BATCH_SIZE_PAGES - 1) // self.config.BATCH_SIZE_PAGES} ({len(batch_titles)} titles)..."
            )

            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "titles": "|".join(batch_titles),
                "explaintext": True,
                "exintro": False,
                "exlimit": "max",
            }

            data = self._fetch_api_data(params)
            if data is None:
                print(
                    f"  Failed to fetch content for batch starting with '{batch_titles[0]}'. Skipping batch.",
                    file=sys.stderr,
                )
                continue  # Skip to next batch if fetching failed

            if "query" in data and "pages" in data["query"]:
                pages_data = data["query"]["pages"]
                for page_id, page_info in pages_data.items():
                    if page_info.get("missing") is None and "extract" in page_info:
                        all_text += page_info["extract"] + "\n"

            if i + self.config.BATCH_SIZE_PAGES < len(page_titles):
                time.sleep(0.5)

        print(
            f"Finished fetching content. Total text length: {len(all_text)} characters."
        )
        return all_text

    def calculate_word_frequencies(self, text: str) -> Counter:
        """Calculates the frequency of non-common words in the text."""
        print("Processing text and calculating frequencies...")
        words = re.findall(r"\b\w+\b", text.lower())
        filtered_words = [
            word for word in words if word not in self.stop_words and len(word) > 1
        ]
        word_counts = Counter(filtered_words)
        print(
            f"Found {len(words)} total words, {len(filtered_words)} non-common words, {len(word_counts)} unique non-common words."
        )
        return word_counts

    def display_results(self, word_counts: Counter, category_name: str):
        """Prints the word frequency results."""
        print(
            f"\n--- Top 100 Cumulative Non-Common Word Frequencies for Category:{category_name} ---"
        )
        if not word_counts:
            print("No frequencies to display.")
            return
        for word, count in word_counts.most_common(100):
            print(f"{word}: {count}")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Calculate cumulative word frequency for non-common words in a Wikipedia category, with caching."
    )
    parser.add_argument(
        "category", help="The Wikipedia category name (e.g., 'Large_language_models')"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass all caches and force API fetches/reprocessing.",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=Config.DEFAULT_CACHE_TTL,
        help=f"Cache time-to-live in seconds (default: {Config.DEFAULT_CACHE_TTL} - 24 hours)",
    )
    args = parser.parse_args()

    # Initialize NLTK stopwords early
    NLTKInitializer.get_stopwords()

    use_cache = not args.no_cache
    cache_ttl = args.cache_ttl

    display_category_name = args.category
    api_category_name = args.category.replace(" ", "_")

    cache_manager = CacheManager(Config.CACHE_DIR)
    results_cache_path = cache_manager.get_cache_path("results", api_category_name)

    word_counts: Optional[Counter] = None

    # --- Check Results Cache First ---
    if use_cache:
        print("Checking for cached results...")
        cached_results = cache_manager.read_cache(results_cache_path, cache_ttl)
        if isinstance(cached_results, dict):
            print(f"\nUsing cached results for Category:{display_category_name}")
            word_counts = Counter(cached_results)
        elif cached_results is not None:  # Not None but not dict
            print(
                f"  Cached results data for '{display_category_name}' has unexpected format. Ignoring cache.",
                file=sys.stderr,
            )

    if word_counts is not None:
        # If cached results were found and valid, display them and exit
        WikipediaCategoryAnalyzer(None, Config(), cache_manager).display_results(
            word_counts, display_category_name
        )
        sys.exit(0)

    # --- Results Cache Miss or bypassed - Proceed with full workflow ---
    print("Proceeding with full analysis (fetching/processing)...")
    with requests.Session() as session:
        analyzer = WikipediaCategoryAnalyzer(session, Config(), cache_manager)

        # --- Step 1: Get Page List ---
        page_titles = analyzer.get_pages_in_category(
            api_category_name, use_cache, cache_ttl
        )
        if page_titles is None:
            print("Could not retrieve page list. Exiting.", file=sys.stderr)
            sys.exit(1)
        if not page_titles:
            print(f"No pages found in Category:{display_category_name}. Exiting.")
            sys.exit(0)

        # --- Step 2: Get Page Content ---
        full_text = analyzer.get_pages_content(page_titles)
        if not full_text:
            print("No text content could be retrieved from the pages. Exiting.")
            sys.exit(1)

        # --- Step 3: Calculate Frequencies ---
        word_counts = analyzer.calculate_word_frequencies(full_text)

        # --- Step 4: Write Results Cache if enabled ---
        if use_cache and word_counts is not None:
            cache_manager.write_cache(results_cache_path, dict(word_counts))

    # --- Step 5: Display final results ---
    if word_counts is not None:
        analyzer.display_results(word_counts, display_category_name)
    else:
        print("Analysis complete, but no word counts were generated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
