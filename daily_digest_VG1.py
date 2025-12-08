import os
import json
import datetime
import time
import io
import concurrent.futures

import pandas as pd
import feedparser
import requests
import google.generativeai as genai

# Try to load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------- CONFIGURATION ----------
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("config.json not found. Using defaults.")
    config = {}

LOOKBACK_DAYS = config.get("lookback_days", 7)
MAX_ITEMS = config.get("max_items_per_feed", 10)
EMAIL_SUBJECT = config.get("email_subject", "Mining Digest")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Cache for OpenAlex journal IDs
JOURNAL_ID_CACHE = {}


def truncate_text(text: str, max_chars: int = 800) -> str:
    """Truncate long text to save tokens."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text[:max_chars] + "..." if len(text) > max_chars else text


def get_journal_id(journal_name: str):
    """Resolve journal display name to OpenAlex source ID."""
    if journal_name in JOURNAL_ID_CACHE:
        return JOURNAL_ID_CACHE[journal_name]

    url = "https://api.openalex.org/sources"
    params = {"filter": f'display_name.search:"{journal_name}"', "per-page": 1}

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                # OpenAlex returns https://openalex.org/S123... -> we need just S123...
                short_id = results[0]["id"].split("/")[-1]
                JOURNAL_ID_CACHE[journal_name] = short_id
                print(f"[OpenAlex] ID found: {journal_name} -> {short_id}")
                return short_id
    except Exception as e:
        print(f"[OpenAlex] Error resolving {journal_name}: {e}")

    return None


def reconstruct_abstract(inverted_index):
    """Rebuild abstract text from OpenAlex's inverted index structure."""
    if not inverted_index:
        return "Abstract not available."
    word_list = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos, word))
    return " ".join(word for _, word in sorted(word_list, key=lambda x: x[0]))


def fetch_openalex_journal_watch(sources_df: pd.DataFrame):
    """
    OPTIMIZED: Groups journal IDs into chunks to fetch many journals in one request.
    """
    print(f"\n[STEP] Querying OpenAlex (Optimized Batch Mode)...")
    papers = []
    journal_rows = sources_df[sources_df["type"] == "journal"]

    if journal_rows.empty:
        return papers

    # 1. Resolve all Journal IDs first
    valid_journals = []
    for _, row in journal_rows.iterrows():
        jid = get_journal_id(row["identifier"])
        if jid:
            valid_journals.append({"id": jid, "name": row["identifier"]})

    if not valid_journals:
        print("[OpenAlex] No valid journal IDs found.")
        return papers

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=LOOKBACK_DAYS)

    # 2. Process in chunks of 15 journals per API call (URL length safety)
    chunk_size = 15
    for i in range(0, len(valid_journals), chunk_size):
        chunk = valid_journals[i : i + chunk_size]

        # Create OR filter: primary_location.source.id:ID1|ID2|ID3
        ids_or_string = "|".join([j["id"] for j in chunk])

        print(f"[OpenAlex] Fetching batch of {len(chunk)} journals...")

        base_url = "https://api.openalex.org/works"
        params = {
            "filter": f"primary_location.source.id:{ids_or_string},from_publication_date:{start_date}",
            "per-page": 50,  # Get up to 50 recent papers for this group
            "select": "id,title,doi,abstract_inverted_index,primary_location",
        }

        try:
            r = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                results = r.json().get("results", [])
                for work in results:
                    source_id = (
                        work.get("primary_location", {})
                        .get("source", {})
                        .get("id", "")
                        .split("/")[-1]
                    )
                    # Find which journal name this ID belongs to
                    source_name = next(
                        (j["name"] for j in chunk if j["id"] == source_id),
                        "Academic Source",
                    )

                    papers.append(
                        {
                            "title": work.get("title", "No title"),
                            "url": work.get("doi") or work.get("id"),
                            "abstract": truncate_text(
                                reconstruct_abstract(
                                    work.get("abstract_inverted_index")
                                )
                            ),
                            "source": source_name,
                            "topic": "Journal Watch",
                            "type": "Academic",
                        }
                    )
            else:
                print(f"[OpenAlex] Batch failed with status {r.status_code}")

            time.sleep(0.5)
        except Exception as e:
            print(f"[OpenAlex] Error in batch: {e}")

    print(f"[OpenAlex] Total papers collected: {len(papers)}")
    return papers


def fetch_single_rss(row):
    """Helper function to fetch one RSS feed (for parallel execution)."""
    name = row["name"]
    url = row["identifier"]
    items = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        content = io.BytesIO(response.content) if response.status_code == 200 else url
        feed = feedparser.parse(content)

        for entry in feed.entries[:MAX_ITEMS]:
            abstract = entry.get("summary") or entry.get("description") or ""
            items.append(
                {
                    "title": entry.get("title", "No title"),
                    "url": entry.get("link", ""),
                    "abstract": truncate_text(abstract),
                    "source": name,
                    "topic": "Industry News",
                    "type": "News",
                }
            )
    except Exception as e:
        print(f"[RSS] Failed {name}: {e}")
    return items


def fetch_rss_feeds_parallel(sources_df: pd.DataFrame):
    """OPTIMIZED: Fetches all RSS feeds in parallel using threads."""
    print("\n[STEP] Fetching RSS feeds (Parallel)...")
    rss_rows = sources_df[sources_df["type"] == "rss"]
    all_items = []

    if rss_rows.empty:
        return all_items

    # Use ThreadPoolExecutor to run multiple fetches at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_single_rss, row) for _, row in rss_rows.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            all_items.extend(future.result())

    print(f"[RSS] Total news items collected: {len(all_items)}")
    return all_items


def summarize_with_ai(items, keywords_context, batch_size: int = 5):
    """
    Analyzes items with Gemini. Includes RETRY logic to prevent data loss.
    """
    if not items:
        return []

    print(f"\n[STEP] Analyzing {len(items)} items with Gemini...")
    newsletter_items = []
    interests = ", ".join(keywords_context.values())

    # Chunk items into batches
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    for i, batch in enumerate(batches):
        print(f"[AI] Processing Batch {i + 1}/{len(batches)} ({len(batch)} items)...")

        # Prepare Prompt
        batch_text = "\n\n".join(
            [
                f"Item {idx + 1}:\nTitle: {item['title']}\nAbstract: {item['abstract']}"
                for idx, item in enumerate(batch)
            ]
        )

        prompt = f"""
        You are a Mining Research Assistant. My interests: {interests}.
        Analyze these {len(batch)} items.
        
        For EACH item, output exactly one line:
        INDEX | YES/NO | CATEGORY | PRIORITY (1=High, 2=Low) | HIGHLIGHT (1 sentence)

        Items:
        {batch_text}
        """

        # --- RETRY LOOP ---
        retries = 3
        while retries > 0:
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()

                # Parse Response
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                for line in lines:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        try:
                            # Parse index (1-based) to list index (0-based)
                            idx = int(parts[0].strip()) - 1
                            if 0 <= idx < len(batch):
                                if parts[1].strip().lower().startswith("yes"):
                                    item = batch[idx]
                                    item["topic"] = parts[2].strip()
                                    item["priority"] = parts[3].strip()
                                    item["highlight"] = parts[4].strip()
                                    newsletter_items.append(item)
                                    print(f"  -> Approved: {item['title'][:40]}...")
                        except ValueError:
                            continue
                break  # Success, exit retry loop

            except Exception as e:
                if "429" in str(e):
                    print(
                        f"  [!] Rate Limit Hit. Waiting 60s... (Retries left: {retries})"
                    )
                    time.sleep(60)
                    retries -= 1
                else:
                    print(f"  [!] Error: {e}")
                    break  # Unknown error, skip batch

        # --- SPEED LIMIT ---
        # 4 seconds sleep guarantees < 15 requests/minute
        time.sleep(4)

    return newsletter_items


def build_newsletter_html(items):
    if not items:
        return "<html><body>No items.</body></html>"

    df = pd.DataFrame(items)
    # Sort: Priority 1 first, then by Topic
    df["p_rank"] = df["priority"].apply(lambda x: 1 if "1" in str(x) else 2)
    df.sort_values(by=["p_rank", "topic"], inplace=True)

    html = """<html><body style="font-family:sans-serif; color:#333;">
    <div style="background:#2c3e50; padding:20px; text-align:center; color:white;">
    <h2>⛏️ Mining Intelligence Digest</h2></div><div style="padding:20px;">"""

    for _, row in df.iterrows():
        color = "#c0392b" if row["p_rank"] == 1 else "#2980b9"
        badge = "🔥 MUST READ" if row["p_rank"] == 1 else "ℹ️ Worth Scanning"

        html += f"""
        <div style="border-left:5px solid {color}; padding-left:15px; margin-bottom:20px;">
            <div style="color:{color}; font-weight:bold; font-size:12px;">{badge} | {row["topic"]}</div>
            <a href="{row["url"]}" style="font-size:16px; font-weight:bold; color:#2c3e50; text-decoration:none;">{row["title"]}</a>
            <div style="font-size:12px; color:#777;">Source: {row["source"]}</div>
            <div style="background:#f4f6f7; padding:10px; margin-top:5px; border-radius:5px;">💡 {row["highlight"]}</div>
        </div>
        """
    html += "</div></body></html>"
    return html


def send_email(html: str):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("[EMAIL] Config missing. Skipping.")
        return

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg["Subject"] = f"{EMAIL_SUBJECT} - {datetime.date.today()}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("[EMAIL] Sent successfully.")
    except Exception as e:
        print(f"[EMAIL] Failed: {e}")


if __name__ == "__main__":
    if not os.path.exists("keywords.csv") or not os.path.exists("sources.csv"):
        print("Error: Missing CSV files.")
    else:
        keywords_df = pd.read_csv("keywords.csv")
        sources_df = pd.read_csv("sources.csv")
        keywords_context = dict(zip(keywords_df.topic, keywords_df.keywords))

        # 1. Collect Data (Optimized)
        papers = fetch_openalex_journal_watch(sources_df)
        news = fetch_rss_feeds_parallel(sources_df)
        all_items = papers + news

        # 2. AI Analysis
        if all_items:
            final_items = summarize_with_ai(all_items, keywords_context)
            if final_items:
                html = build_newsletter_html(final_items)
                send_email(html)
                # Optional: Save to file for debugging
                with open("latest_newsletter.html", "w", encoding="utf-8") as f:
                    f.write(html)
            else:
                print("No relevant items found after AI analysis.")
        else:
            print("No new raw items found.")
