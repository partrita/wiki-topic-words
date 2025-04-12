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
from pathlib import Path # For easier path manipulation

# --- Configuration ---
API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {'User-Agent': 'MyWikiWordFreqBot/1.1 (https://example.com/bot; myemail@example.com) requests'} # Updated version
BATCH_SIZE_PAGES = 50
RETRY_DELAY = 5
CACHE_DIR = Path("./.wiki_cache") # Store cache files in a hidden subdirectory
DEFAULT_CACHE_TTL = 24 * 60 * 60 # Default cache validity: 24 hours in seconds

# --- NLTK Stopwords Setup ---
try:
    nltk.data.find('corpora/stopwords')
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("NLTK 'stopwords' corpus not found.", file=sys.stderr)
    print("Please run: python -m nltk.downloader stopwords", file=sys.stderr)
    sys.exit(1)

# --- Helper Functions ---

def sanitize_filename(name):
    """Removes or replaces characters problematic for filenames."""
    # Replace common problematic chars with underscore
    name = re.sub(r'[\\/*?:"<>|\s]+', '_', name)
    # Remove any remaining non-alphanumeric characters (except hyphen and underscore)
    name = re.sub(r'[^\w\-\_]+', '', name)
    # Limit length potentially
    return name[:100].strip('_') # Limit length and remove leading/trailing underscores

def read_cache(path, max_age_seconds):
    """Reads cache data from a JSON file if it exists and is not expired."""
    if not path.exists():
        return None # Cache miss - file doesn't exist

    try:
        file_mod_time = path.stat().st_mtime
        if time.time() - file_mod_time > max_age_seconds:
            print(f"  Cache file expired: {path.name}")
            path.unlink() # Remove expired cache file
            return None # Cache miss - expired

        with path.open('r', encoding='utf-8') as f:
            cache_data = json.load(f)
        # We previously stored {"timestamp": ..., "data": ...}, retrieve 'data'
        if 'data' in cache_data:
             print(f"  Cache hit: {path.name}")
             return cache_data['data']
        else:
             print(f"  Cache file format error (missing 'data' key): {path.name}. Ignoring cache.", file=sys.stderr)
             return None

    except (IOError, json.JSONDecodeError, KeyError) as e:
        print(f"  Error reading cache file {path.name}: {e}. Ignoring cache.", file=sys.stderr)
        try:
            path.unlink() # Attempt to remove corrupted cache file
        except OSError:
            pass
        return None # Cache miss on error

def write_cache(path, data):
    """Writes data to a JSON cache file with a timestamp."""
    try:
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_content = {
            "timestamp": time.time(), # Store timestamp for potential future use (though we use file mod time now)
            "data": data
        }
        with path.open('w', encoding='utf-8') as f:
            json.dump(cache_content, f, ensure_ascii=False, indent=2) # Use indent for readability
        print(f"  Cache written: {path.name}")
    except IOError as e:
        print(f"  Error writing cache file {path.name}: {e}", file=sys.stderr)


def get_pages_in_category(session, category_title, use_cache, cache_ttl):
    """Fetches all page titles within a given Wikipedia category, using cache if enabled."""
    sanitized_category = sanitize_filename(category_title)
    cache_path = CACHE_DIR / f"pages_{sanitized_category}.json"

    if use_cache:
        cached_data = read_cache(cache_path, cache_ttl)
        if cached_data is not None:
            if isinstance(cached_data, list): # Basic validation
                print(f"Using cached page list for Category:{category_title}")
                return cached_data
            else:
                print(f"  Cached page data for '{category_title}' has unexpected format. Ignoring cache.", file=sys.stderr)

    # --- Cache miss or invalid ---
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
    retries = 3

    while True:
        try:
            current_params = params.copy()
            current_params.update(last_continue)
            response = session.get(API_URL, params=current_params, headers=HEADERS, timeout=20)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                print(f"API Error fetching pages: {data['error'].get('info', 'Unknown error')}", file=sys.stderr)
                return None # Indicate failure

            if 'query' in data and 'categorymembers' in data['query']:
                found_pages = data['query']['categorymembers']
                page_titles = [page['title'] for page in found_pages]
                pages.extend(page_titles)
                print(f"  Fetched {len(found_pages)} pages (Total: {len(pages)})")

            if 'continue' in data:
                last_continue = data['continue']
                time.sleep(0.5)
            else:
                break
            retries = 3 # Reset retries on success

        except requests.exceptions.RequestException as e:
            retries -= 1
            print(f"Network error fetching category members: {e}. Retries left: {retries}", file=sys.stderr)
            if retries <= 0:
                print("Max retries exceeded fetching pages. Aborting.", file=sys.stderr)
                return None # Indicate failure
            time.sleep(RETRY_DELAY)
        except Exception as e:
             print(f"An unexpected error occurred during page fetch: {e}", file=sys.stderr)
             return None # Indicate failure

    print(f"Finished fetching pages. Found {len(pages)} total pages.")

    # Write to cache if successful and caching is enabled
    if pages is not None and use_cache:
        write_cache(cache_path, pages)

    return pages

def get_pages_content(session, page_titles):
    """Fetches plain text content for a list of page titles in batches."""
    # --- This function remains largely unchanged ---
    # (No caching needed here as it depends on the page list from the previous step)
    all_text = ""
    retries = 3

    print(f"Fetching content for {len(page_titles)} pages...")
    for i in range(0, len(page_titles), BATCH_SIZE_PAGES):
        batch_titles = page_titles[i:i + BATCH_SIZE_PAGES]
        print(f"  Fetching batch {i//BATCH_SIZE_PAGES + 1}/{ (len(page_titles) + BATCH_SIZE_PAGES - 1)//BATCH_SIZE_PAGES } ({len(batch_titles)} titles)...")

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": "|".join(batch_titles),
            "explaintext": True,
            "exintro": False,
            "exlimit": "max"
        }

        fetch_attempt = 0
        while fetch_attempt < retries:
            try:
                response = session.get(API_URL, params=params, headers=HEADERS, timeout=30)
                response.raise_for_status()
                data = response.json()

                if 'error' in data:
                    print(f"API Error fetching content: {data['error'].get('info', 'Unknown error')}", file=sys.stderr)
                    break

                if 'query' in data and 'pages' in data['query']:
                    pages_data = data['query']['pages']
                    for page_id, page_info in pages_data.items():
                        # Handle cases where a page might be missing or invalid
                        if page_info.get('missing') is None and 'extract' in page_info:
                             all_text += page_info['extract'] + "\n"
                    break # Successfully processed batch

            except requests.exceptions.Timeout:
                fetch_attempt += 1
                print(f"Timeout fetching content batch. Retrying ({fetch_attempt}/{retries})...", file=sys.stderr)
                time.sleep(RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                fetch_attempt += 1
                print(f"Network error fetching content: {e}. Retrying ({fetch_attempt}/{retries})...", file=sys.stderr)
                time.sleep(RETRY_DELAY)
            except Exception as e:
                 print(f"An unexpected error occurred during content fetch: {e}", file=sys.stderr)
                 break # Non-retryable error for this batch

        if fetch_attempt >= retries:
            print(f"Failed to fetch content for batch starting with '{batch_titles[0]}' after {retries} retries.", file=sys.stderr)

        if i + BATCH_SIZE_PAGES < len(page_titles):
            time.sleep(0.5)

    print(f"Finished fetching content. Total text length: {len(all_text)} characters.")
    return all_text

def calculate_word_frequencies(text):
    """Calculates the frequency of non-common words in the text."""
    # --- This function remains unchanged ---
    print("Processing text and calculating frequencies...")
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [
        word for word in words if word not in STOP_WORDS and len(word) > 1
    ]
    word_counts = Counter(filtered_words)
    print(f"Found {len(words)} total words, {len(filtered_words)} non-common words, {len(word_counts)} unique non-common words.")
    return word_counts

def display_results(word_counts, category_name):
    """Prints the word frequency results."""
    print(f"\n--- Top 100 Cumulative Non-Common Word Frequencies for Category:{category_name} ---")
    if not word_counts:
        print("No frequencies to display.")
        return
    # Print sorted by frequency, descending
    for word, count in word_counts.most_common(100): # Print top 100
        print(f"{word}: {count}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Calculate cumulative word frequency for non-common words in a Wikipedia category, with caching.")
    parser.add_argument("category", help="The Wikipedia category name (e.g., 'Large_language_models')")
    parser.add_argument("--no-cache", action="store_true", help="Bypass all caches and force API fetches/reprocessing.")
    parser.add_argument("--cache-ttl", type=int, default=DEFAULT_CACHE_TTL,
                        help=f"Cache time-to-live in seconds (default: {DEFAULT_CACHE_TTL} - 24 hours)")
    args = parser.parse_args()

    use_cache = not args.no_cache
    cache_ttl = args.cache_ttl
    # Use original name for display, sanitized name for files/API
    display_category_name = args.category
    api_category_name = args.category.replace(" ", "_")
    sanitized_category = sanitize_filename(api_category_name)
    results_cache_path = CACHE_DIR / f"results_{sanitized_category}.json"

    # --- Check Results Cache (Cache 2) First ---
    if use_cache:
        print("Checking for cached results...")
        cached_results = read_cache(results_cache_path, cache_ttl)
        if cached_results is not None:
            if isinstance(cached_results, dict): # Basic validation
                print(f"\nUsing cached results for Category:{display_category_name}")
                word_counts = Counter(cached_results) # Recreate counter for most_common
                display_results(word_counts, display_category_name)
                sys.exit(0) # Exit early, successful run from cache
            else:
                 print(f"  Cached results data for '{display_category_name}' has unexpected format. Ignoring cache.", file=sys.stderr)


    # --- Results Cache Miss or bypassed - Proceed with full workflow ---
    print("Proceeding with full analysis (fetching/processing)...")
    word_counts = None # Initialize word_counts

    with requests.Session() as session:
        # --- Step 1: Get Page List (potentially uses Cache 1) ---
        page_titles = get_pages_in_category(session, api_category_name, use_cache, cache_ttl)

        if page_titles is None:
            print("Could not retrieve page list. Exiting.", file=sys.stderr)
            sys.exit(1)
        if not page_titles:
            print(f"No pages found in Category:{display_category_name}. Exiting.")
            sys.exit(0)

        # --- Step 2: Get Page Content (no caching here) ---
        full_text = get_pages_content(session, page_titles)

        if not full_text:
            print("No text content could be retrieved from the pages. Exiting.")
            # Optionally write an empty results cache? For now, just exit.
            sys.exit(1)

        # --- Step 3: Calculate Frequencies ---
        word_counts = calculate_word_frequencies(full_text)

        # --- Step 4: Write Results Cache (Cache 2) if enabled ---
        if use_cache and word_counts is not None:
             # Convert Counter to simple dict for JSON serialization
             results_to_cache = dict(word_counts)
             write_cache(results_cache_path, results_to_cache)

    # --- Step 5: Display final results ---
    if word_counts is not None:
        display_results(word_counts, display_category_name)
    else:
        print("Analysis complete, but no word counts were generated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
