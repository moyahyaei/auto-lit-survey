import os
import json
import datetime
import time
import io

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

# Load config.json if present
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
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Cache for OpenAlex journal IDs
JOURNAL_ID_CACHE = {}


def get_journal_id(journal_name: str):
    """Resolve journal display name to OpenAlex source ID."""
    if journal_name in JOURNAL_ID_CACHE:
        return JOURNAL_ID_CACHE[journal_name]

    url = "https://api.openalex.org/sources"
    params = {
        "filter": f'display_name.search:"{journal_name}"',
        "per-page": 1,
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            full_id = results[0]["id"]  # e.g. https://openalex.org/S123456789
            short_id = full_id.replace("https://openalex.org/", "")
            JOURNAL_ID_CACHE[journal_name] = short_id
            print(f"[OpenAlex] ID found: {journal_name} -> {short_id}")
            return short_id
        else:
            print(f"[OpenAlex] No results for journal: {journal_name}")
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
    sorted_words = sorted(word_list, key=lambda x: x[0])
    return " ".join(word for _, word in sorted_words)


def fetch_openalex_journal_watch(sources_df: pd.DataFrame):
    """Fetch recent papers from selected journals via OpenAlex."""
    print(f"\n[STEP] Querying OpenAlex for last {LOOKBACK_DAYS} days...")
    papers = []

    journal_rows = sources_df[sources_df["type"] == "journal"]
    if journal_rows.empty:
        print("[OpenAlex] No journals configured in sources.csv.")
        return papers

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=LOOKBACK_DAYS)

    for _, row in journal_rows.iterrows():
        journal_name = row["identifier"]
        print(f"\n[OpenAlex] Processing journal: {journal_name}")

        journal_id = get_journal_id(journal_name)
        if not journal_id:
            print(f"[OpenAlex] Skipping {journal_name} (ID not found).")
            continue

        base_url = "https://api.openalex.org/works"
        filter_str = (
            f"primary_location.source.id:{journal_id},"
            f"from_publication_date:{start_date}"
        )
        params = {
            "filter": filter_str,
            "per-page": MAX_ITEMS,
            "select": "id,title,doi,abstract_inverted_index,primary_location",
        }

        try:
            r = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            print(f"[OpenAlex] Found {len(results)} works for {journal_name}")

            for work in results:
                abstract_text = reconstruct_abstract(
                    work.get("abstract_inverted_index")
                )
                url = work.get("doi") or work.get("id") or ""
                papers.append(
                    {
                        "title": work.get("title", "No title"),
                        "url": url,
                        "abstract": abstract_text,
                        "source": journal_name,
                        "topic": "Journal Watch",
                        "type": "Academic",
                    }
                )

            time.sleep(0.2)  # be polite to API
        except Exception as e:
            print(f"[OpenAlex] Error fetching works for {journal_name}: {e}")

    print(f"[OpenAlex] Total journal papers collected: {len(papers)}")
    return papers


def fetch_rss_feeds(sources_df: pd.DataFrame):
    """Fetch recent items from configured RSS feeds."""
    print("\n[STEP] Fetching RSS feeds...")
    news_items = []
    rss_rows = sources_df[sources_df["type"] == "rss"]

    if rss_rows.empty:
        print("[RSS] No RSS sources configured in sources.csv.")
        return news_items

    for _, row in rss_rows.iterrows():
        name = row["name"]
        url = row["identifier"]
        print(f"[RSS] Fetching: {name} ({url})")

        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            if response.status_code != 200:
                print(f"[RSS] Error {name}: status {response.status_code}")
                continue

            feed = feedparser.parse(io.BytesIO(response.content))
            if not feed.entries:
                # Fallback: let feedparser fetch directly from URL
                feed = feedparser.parse(url)

            entries = feed.entries[:MAX_ITEMS]
            print(f"[RSS] {name}: {len(entries)} items")

            for entry in entries:
                abstract_text = (
                    entry.get("summary")
                    or entry.get("description")
                    or "No summary provided."
                )
                news_items.append(
                    {
                        "title": entry.get("title", "No title"),
                        "url": entry.get("link", ""),
                        "abstract": abstract_text[:1500],
                        "source": name,
                        "topic": "Industry News",
                        "type": "News",
                    }
                )
        except Exception as e:
            print(f"[RSS] Error fetching {name}: {e}")

    print(f"[RSS] Total news items collected: {len(news_items)}")
    return news_items


def summarize_with_ai(items, keywords_context):
    """Use Gemini to filter, categorise and highlight items."""
    if not items:
        print("[AI] No items to analyse.")
        return []

    print(f"\n[STEP] Analysing {len(items)} items with Gemini...")
    newsletter_items = []
    interests = ", ".join(keywords_context.values())

    for i, item in enumerate(items):
        print(f"[AI] {i + 1}/{len(items)}: {item['title'][:60]}...")

        prompt = f"""
You are a Mining & Mineral Processing Research Assistant.

My technical interests: {interests}.

Evaluate this item and answer in ONE line:
Title: {item["title"]}
Abstract: {item["abstract"]}

1. RELEVANCE: Is this technically significant for those interests? (Yes/No)
2. CATEGORY: Short category (e.g., Comminution, ESG, Automation, Hydrometallurgy).
3. PRIORITY: 1 = must-read, 2 = worth scanning.
4. HIGHLIGHT: One sentence on the core finding/insight.

Output format (exactly):
Yes/No | [Category] | [Priority: 1 or 2] | [Highlight]
"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            print(f"[AI] Raw response: {text}")

            if text.lower().startswith("yes"):
                parts = [p.strip() for p in text.split("|")]
                if len(parts) >= 4:
                    item["topic"] = parts[1]
                    item["priority"] = parts[2]
                    item["highlight"] = parts[3]
                    newsletter_items.append(item)
                    print("[AI] -> APPROVED")
                else:
                    print("[AI] -> Malformed response, skipping")
            else:
                print("[AI] -> Not relevant, skipped")

        except Exception as e:
            print(f"[AI] Error on item {i + 1}: {e}")
            # crude rate-limit handling
            if "429" in str(e):
                print("[AI] Hit rate limit, sleeping 60 seconds...")
                time.sleep(60)

        # free tier: be gentle
        time.sleep(4)

    print(f"[AI] Total items approved: {len(newsletter_items)}")
    return newsletter_items


def build_newsletter_html(items):
    """Build grouped HTML newsletter from selected items."""
    if not items:
        return "<html><body><p>No relevant items this week.</p></body></html>"

    df = pd.DataFrame(items)

    # Default values if missing
    if "topic" not in df.columns:
        df["topic"] = "General"
    if "priority" not in df.columns:
        df["priority"] = "2"  # treat as 'worth scanning'

    # Sort by topic then priority (1 before 2)
    df["priority_num"] = (
        df["priority"].astype(str).str.extract("(\d)").fillna("2").astype(int)
    )
    df.sort_values(by=["topic", "priority_num"], inplace=True)

    today_str = datetime.date.today().isoformat()

    html = f"""
<html>
  <body style="font-family: Arial, sans-serif; color: #333;">
    <div style="background-color: #2c3e50; padding: 16px; text-align: center;">
      <h2 style="color: #ecf0f1; margin: 0;">⛏️ Weekly Mining & Processing Digest</h2>
      <div style="color: #bdc3c7; font-size: 12px;">{today_str}</div>
    </div>
    <div style="padding: 20px;">
      <p>This digest includes AI-filtered publications and news relevant to mining, mineral processing, and the resources industry.</p>
"""

    current_topic = None
    for _, row in df.iterrows():
        topic = row.get("topic", "General")
        priority = str(row.get("priority", "2")).strip()
        title = row.get("title", "No title")
        url = row.get("url", "#")
        source = row.get("source", "Unknown source")
        highlight = row.get("highlight", "").strip()

        if topic != current_topic:
            current_topic = topic
            html += f"""
      <h3 style="color: #d35400; border-bottom: 2px solid #d35400; padding-bottom: 4px; margin-top: 24px;">
        {current_topic}
      </h3>
"""

        badge_text = "Must read" if priority.startswith("1") else "Worth scanning"
        badge_color = "#c0392b" if priority.startswith("1") else "#2980b9"

        html += f"""
      <div style="margin-bottom: 16px; padding-left: 10px; border-left: 4px solid #2980b9;">
        <div style="font-size: 12px; margin-bottom: 4px;">
          <span style="background-color: {badge_color}; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 11px;">{badge_text}</span>
          <span style="color: #7f8c8d; margin-left: 8px;">Source: {source}</span>
        </div>
        <a href="{url}" style="font-weight: bold; font-size: 15px; color: #2c3e50; text-decoration: none;">
          {title}
        </a>
        <div style="background-color: #f8f9f9; padding: 8px; border-radius: 4px; font-style: italic; color: #444; margin-top: 4px;">
          <span style="font-weight:bold; color: #d35400;">AI Highlight:</span> {highlight}
        </div>
      </div>
"""

    html += """
    </div>
    <div style="text-align: center; font-size: 11px; color: #aaa; padding: 16px;">
      Generated automatically by your Mining Digest agent.
    </div>
  </body>
</html>
"""
    return html


def save_html(html: str):
    """Save newsletter HTML to output folder."""
    os.makedirs("output", exist_ok=True)
    today_str = datetime.date.today().isoformat()
    path = os.path.join("output", f"newsletter_{today_str}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OUTPUT] Newsletter HTML saved to: {path}")


def send_email(html: str):
    """Send newsletter via email (Gmail). Skips if EMAIL_* not configured."""
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("[EMAIL] Email settings not complete. Skipping email send.")
        return

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg["Subject"] = f"{EMAIL_SUBJECT} - {datetime.date.today().isoformat()}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("[EMAIL] Newsletter email sent successfully.")


if __name__ == "__main__":
    # Basic file existence checks
    if not os.path.exists("keywords.csv") or not os.path.exists("sources.csv"):
        print("Error: keywords.csv or sources.csv not found in the project folder.")
        raise SystemExit(1)

    # Load configs
    keywords_df = pd.read_csv("keywords.csv")
    sources_df = pd.read_csv("sources.csv")
    keywords_context = dict(zip(keywords_df.topic, keywords_df.keywords))

    print("[STEP] Starting collection pipeline...")
    journal_papers = fetch_openalex_journal_watch(sources_df)
    rss_news = fetch_rss_feeds(sources_df)

    all_raw_items = journal_papers + rss_news
    print(f"\n[INFO] Total raw items collected: {len(all_raw_items)}")

    if all_raw_items:
        selected_items = summarize_with_ai(all_raw_items, keywords_context)

        # OPTIONAL: apply a simple cap on total items
        if selected_items:
            # Build and save HTML
            html = build_newsletter_html(selected_items)
            save_html(html)

            # Try to send email (if configured)
            send_email(html)
        else:
            print("[INFO] No items passed AI relevance filter.")
    else:
        print(f"[INFO] No new items found in the last {LOOKBACK_DAYS} days.")
