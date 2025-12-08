import pandas as pd
import requests
import feedparser
import json

print("--- 1. CHECKING CSV FILE ---")
try:
    df = pd.read_csv("sources.csv")
    print(f"Columns found: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    # Check for whitespace or capitalization issues
    # specific check for the 'type' column
    if "type" in df.columns:
        unique_types = df["type"].unique()
        print(f"Unique types found: {unique_types}")

        # Count matches
        journal_count = len(df[df["type"].str.strip().str.lower() == "journal"])
        rss_count = len(df[df["type"].str.strip().str.lower() == "rss"])
        print(f"Rows identified as Journals: {journal_count}")
        print(f"Rows identified as RSS: {rss_count}")
    else:
        print("ERROR: 'type' column is missing from sources.csv")

except Exception as e:
    print(f"CRITICAL ERROR reading CSV: {e}")

print("\n--- 2. CHECKING OPENALEX CONNECTION ---")
# Try to fetch ONE known journal without any date filters to see if the name is correct
test_journal = "Minerals Engineering"
url = f'https://api.openalex.org/works?filter=primary_location.source.display_name:"{test_journal}"&per-page=1'
print(f"Testing OpenAlex query for: {test_journal}")

try:
    r = requests.get(url)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        count = data.get("meta", {}).get("count", 0)
        print(f"Total papers found in database for this journal: {count}")
        if count == 0:
            print("WARNING: Journal name match failed. OpenAlex returned 0 results.")
    else:
        print("API Error.")
except Exception as e:
    print(f"Connection Error: {e}")

print("\n--- 3. CHECKING RSS CONNECTION ---")
# Try to fetch ONE known RSS feed
test_rss = "https://www.mdpi.com/rss/journal/minerals"
print(f"Testing RSS feed: {test_rss}")
try:
    feed = feedparser.parse(test_rss)
    print(f"Feed Status: {feed.get('status', 'Unknown')}")
    print(f"Entries found: {len(feed.entries)}")
    if len(feed.entries) > 0:
        print(f"First title: {feed.entries[0].title}")
except Exception as e:
    print(f"RSS Error: {e}")
