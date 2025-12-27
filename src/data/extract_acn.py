"""ACN (Adaptive Charging Network) data extraction module.

This module fetches EV charging session data from the ACN API.
Data source: https://ev.caltech.edu/
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_TOKEN = os.getenv("ACN_API_TOKEN")

# Available sites: 'caltech', 'jpl', 'office001'
SITE = "office001"
BASE_URL = "https://ev.caltech.edu/api/v1/sessions/"

# Split by year to reduce server load
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def get_sessions_by_year(site: str, year: int, token: str) -> list[dict]:
    """Fetch all charging sessions for a given site and year.

    Args:
        site: The ACN site identifier ('caltech', 'jpl', 'office001').
        year: The year to fetch data for.
        token: API authentication token.

    Returns:
        List of session dictionaries.
    """
    auth = (token, "")

    # Date format required by ACN API (RFC 1123)
    start_date = f"Mon, 01 Jan {year} 00:00:00 GMT"
    end_date = f"Mon, 01 Jan {year + 1} 00:00:00 GMT"

    # Build query filter
    query = (
        f'{site}?where=connectionTime>="{start_date}" and connectionTime<"{end_date}"'
    )
    url = BASE_URL + query

    sessions_year = []
    print(f"\n--- Downloading year {year} ---")

    while url:
        try:
            response = requests.get(url, auth=auth, timeout=30)

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                time.sleep(2)
                continue

            data = response.json()
            items = data.get("_items", [])
            sessions_year.extend(items)

            print(f"Year {year} - Total: {len(sessions_year)} sessions...", end="\r")

            # Handle pagination
            links = data.get("_links", {})
            next_link = links.get("next", {}).get("href")

            if next_link:
                url = "https://ev.caltech.edu/api/v1/" + next_link
            else:
                url = None

        except requests.RequestException as e:
            print(f"Error: {e}")
            break

    print(f"\nCompleted {year}: {len(sessions_year)} sessions total.")
    return sessions_year


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_history = []

    for year in YEARS:
        year_data = get_sessions_by_year(SITE, year, API_TOKEN)
        full_history.extend(year_data)

        # Intermediate save (safety checkpoint)
        filename = OUTPUT_DIR / f"acn_{SITE}_{year}.json"
        with open(filename, "w") as f:
            json.dump(year_data, f)
            print(f"-> Saved to {filename}")

    # Final merged save
    print(f"\n--- FINAL TOTAL: {len(full_history)} sessions ---")
    with open(OUTPUT_DIR / f"acn_{SITE}_FULL_2018-2024.json", "w") as f:
        json.dump(full_history, f)
