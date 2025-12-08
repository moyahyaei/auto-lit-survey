import os
import json
import datetime
import time
import io
from collections import Counter, defaultdict

import pandas as pd
import feedparser
import requests
import google.generativeai as genai

# Try to load .env if present (for local runs)
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
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it to your .env file or GitHub Secrets."
    )

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


def truncate_text(text: str, max_chars: int = 800) -> str:
    """Truncate long text to avoid huge prompts while preserving meaning."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


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


def fetch_openalex_journal_watch(sources_df: pd.DataFrame, start_date: datetime.date):
    """Fetch recent papers from selected journals via OpenAlex."""
    print(
        f"\n[STEP] Querying OpenAlex for last {LOOKBACK_DAYS} days (from {start_date})..."
    )
    papers = []

    journal_rows = sources_df[sources_df["type"] == "journal"]
    if journal_rows.empty:
        print("[OpenAlex] No journals configured in sources.csv.")
        return papers

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
                abstract_text = truncate_text(abstract_text, max_chars=800)
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
                abstract_text = truncate_text(abstract_text, max_chars=800)
                news_items.append(
                    {
                        "title": entry.get("title", "No title"),
                        "url": entry.get("link", ""),
                        "abstract": abstract_text,
                        "source": name,
                        "topic": "Industry News",
                        "type": "News",
                    }
                )
        except Exception as e:
            print(f"[RSS] Error fetching {name}: {e}")

    print(f"[RSS] Total news items collected: {len(news_items)}")
    return news_items


def summarize_with_ai(items, keywords_context, batch_size: int = 3):
    """
    Use Gemini to annotate items in batches.

    We NO LONGER gate on relevance; assume all items are at least somewhat relevant.
    The model's job is to:
      - assign Category,
      - assign Priority (1 = must read, 2 = worth scanning),
      - generate a Highlight.

    Expected line format per item:
      INDEX | Category | Priority | Highlight
    """
    if not items:
        print("[AI] No items to analyse.")
        return []

    print(f"\n[STEP] Analysing {len(items)} items with Gemini (batched)...")
    annotated_items = []
    interests = ", ".join(keywords_context.values())

    # Helper to chunk the list
    def chunk_list(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    batch_number = 0

    for batch in chunk_list(items, batch_size):
        batch_number += 1
        print(f"\n[AI] Processing batch {batch_number} with {len(batch)} items...")

        # Build the prompt with numbered items
        items_text_lines = []
        for idx, item in enumerate(batch, start=1):
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            items_text_lines.append(f"{idx}. Title: {title}\n   Abstract: {abstract}")
        items_block = "\n\n".join(items_text_lines)

        prompt = f"""
You are an expert research assistant in mining, mineral processing, and the resources industry.

My current technical interests are:
{interests}

You will receive a batch of {len(batch)} candidate items (papers or news). For EACH item, you must:
- Assume it is at least somewhat relevant to my interests (do NOT reject or skip any item).
- Assign a CATEGORY from this list (choose the closest match):
  - Comminution
  - Flotation
  - Hydrometallurgy
  - Pyrometallurgy
  - Geometallurgy & Ore Characterisation
  - Mineral Processing Modelling & Simulation
  - Automation & Control
  - Data Science & AI
  - ESG & Sustainability
  - Equipment & Operations
  - Other

- Assign a PRIORITY:
  - 1 = must-read (highly central to the interests above, conceptually novel, or practically important)
  - 2 = worth scanning (supporting, incremental, or more peripheral, but still relevant)

- Write a one-sentence HIGHLIGHT capturing the main idea, contribution, or finding.
  - The highlight must be understandable in isolation.
  - Do NOT just repeat the title.

OUTPUT RULES (very important):
- You MUST return EXACTLY one line per item.
- You MUST keep the original order of items (1, 2, 3, ...).
- You MUST NOT add any commentary before, between, or after the lines.
- Each line MUST have this exact format:

  INDEX | Category | Priority | Highlight

Where:
- INDEX is the item number I gave you (1, 2, 3, ...).
- Category is one of the labels listed above.
- Priority is 1 or 2.
- Highlight is one sentence.

Example of valid output for 2 items:
1 | Flotation | 1 | This paper proposes a new reagent strategy that significantly improves rougher recovery in complex sulphide ores.
2 | Automation & Control | 2 | The article describes a case study of implementing model predictive control to stabilise a grinding circuit under variable feed conditions.

Now process the following items:

{items_block}
"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            print(f"[AI] Raw batch response:\n{text}\n")

            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if len(lines) != len(batch):
                print(
                    f"[AI] WARNING: Expected {len(batch)} lines, but got {len(lines)}. "
                    "Some items may fall back to default metadata."
                )

            # Build a map from index -> parsed fields
            parsed_by_index = {}

            for line in lines:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 4:
                    print(f"[AI] Malformed line, skipping: {line}")
                    continue

                index_str, category, priority, highlight = parts[:4]

                try:
                    item_index = int(index_str)
                    if not (1 <= item_index <= len(batch)):
                        print(
                            f"[AI] Invalid index {item_index} for batch, skipping line."
                        )
                        continue
                except ValueError:
                    print(f"[AI] Invalid index '{index_str}', skipping line.")
                    continue

                parsed_by_index[item_index] = {
                    "category": category,
                    "priority": priority,
                    "highlight": highlight,
                }

            # Apply parsed annotations back to items
            for idx, item in enumerate(batch, start=1):
                ann = parsed_by_index.get(idx, None)
                if ann:
                    item["topic"] = ann["category"]
                    item["priority"] = ann["priority"]
                    item["highlight"] = ann["highlight"]
                else:
                    # Fallback defaults if AI did not return a line for this item
                    item.setdefault("topic", item.get("topic", "General"))
                    item.setdefault("priority", "2")
                    item.setdefault(
                        "highlight",
                        truncate_text(item.get("abstract", ""), max_chars=200),
                    )
                annotated_items.append(item)

        except Exception as e:
            print(f"[AI] Error on batch {batch_number}: {e}")
            if "429" in str(e):
                print("[AI] Hit rate limit, sleeping 60 seconds...")
                time.sleep(60)

            # If the batch fails completely, fall back for all items
            for item in batch:
                item.setdefault("topic", item.get("topic", "General"))
                item.setdefault("priority", "2")
                item.setdefault(
                    "highlight",
                    truncate_text(item.get("abstract", ""), max_chars=200),
                )
                annotated_items.append(item)

        # Gentle pacing between batches (reduces rate-limit risk)
        time.sleep(3)

    print(f"\n[AI] Total items annotated across all batches: {len(annotated_items)}")
    return annotated_items


def generate_journal_summaries(items):
    """
    Generate brief per-journal summaries based on the annotated items.

    Only considers items of type 'Academic'.
    """
    print("\n[STEP] Generating per-journal summaries...")
    journal_items = defaultdict(list)
    for it in items:
        if it.get("type") == "Academic":
            journal_items[it.get("source", "Unknown")].append(it)

    summaries = {}
    for journal, j_items in journal_items.items():
        print(f"[AI] Summarising journal: {journal} ({len(j_items)} items)")
        titles_and_highlights = []
        for idx, it in enumerate(j_items, start=1):
            titles_and_highlights.append(
                f"{idx}. Title: {it.get('title', '')}\n   Highlight: {it.get('highlight', '')}"
            )
        block = "\n".join(titles_and_highlights)

        prompt = f"""
You are summarising recent publications in the journal '{journal}' 
for a mining & mineral processing researcher.

Here are some recent papers from this journal (each with title and a short highlight):

{block}

Write a concise overview of the main technical themes and trends represented by these papers.

Requirements:
- Length: 2–3 sentences total.
- Focus on themes and trends (e.g., focus areas, methods, types of ore, process challenges).
- Do NOT list individual paper titles.
- Do NOT use bullet points or numbered lists.
- Write in plain, professional, neutral language.

Now provide the 2–3 sentence summary.
"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            summaries[journal] = text
        except Exception as e:
            print(f"[AI] Error generating summary for journal {journal}: {e}")
            summaries[journal] = "Summary unavailable due to an AI error."

        time.sleep(2)

    return summaries


def build_newsletter_html(items, review_stats, journal_summaries):
    """Build grouped HTML newsletter from annotated items."""
    today_str = review_stats["end_date"]
    start_str = review_stats["start_date"]

    if not items:
        # Minimal skeleton when no items or no relevant items
        return f"""
<html>
  <body style="font-family: Arial, sans-serif; color: #333;">
    <div style="background-color: #2c3e50; padding: 16px; text-align: center;">
      <h2 style="color: #ecf0f1; margin: 0;">⛏️ Weekly Mining & Processing Digest</h2>
      <div style="color: #bdc3c7; font-size: 12px;">Coverage: {start_str} → {today_str}</div>
    </div>
    <div style="padding: 20px;">
      <p>No new items were identified for this period.</p>
    </div>
    <div style="text-align: center; font-size: 11px; color: #aaa; padding: 16px;">
      Generated automatically by your Mining Digest agent.
    </div>
  </body>
</html>
"""

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

    total_raw = review_stats["total_raw"]
    total_annotated = len(items)
    per_source = review_stats["per_source"]

    # Build coverage and source stats block
    sources_html_lines = []
    for src, stats in per_source.items():
        sources_html_lines.append(
            f"<li><b>{src}</b>: {stats['raw']} collected, {stats['annotated']} included</li>"
        )
    sources_html = "\n".join(sources_html_lines)

    # Journal summaries block
    journal_summaries_html_lines = []
    for journal, summary in journal_summaries.items():
        journal_summaries_html_lines.append(
            f"<h4 style='margin-bottom:4px;'>{journal}</h4>"
            f"<p style='margin-top:0;'>{summary}</p>"
        )
    journal_summaries_html = (
        "\n".join(journal_summaries_html_lines)
        or "<p>No academic journal items this period.</p>"
    )

    html = f"""
<html>
  <body style="font-family: Arial, sans-serif; color: #333;">
    <div style="background-color: #2c3e50; padding: 16px; text-align: center;">
      <h2 style="color: #ecf0f1; margin: 0;">⛏️ Weekly Mining & Processing Digest</h2>
      <div style="color: #bdc3c7; font-size: 12px;">Coverage: {start_str} → {today_str}</div>
    </div>
    <div style="padding: 20px;">

      <h3>Overview</h3>
      <p>
        This digest covers literature and news discovered between <b>{start_str}</b> and <b>{today_str}</b>.
        A total of <b>{total_raw}</b> items were collected from your configured journals and news sources,
        of which <b>{total_annotated}</b> are summarised below.
      </p>

      <h3>Coverage & Sources</h3>
      <ul>
        {sources_html}
      </ul>

      <h3>Journal Overviews</h3>
      {journal_summaries_html}

      <h3>Detailed Items</h3>
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

    # Define review window explicitly
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=LOOKBACK_DAYS)

    print("[STEP] Starting collection pipeline...")
    journal_papers = fetch_openalex_journal_watch(sources_df, start_date)
    rss_news = fetch_rss_feeds(sources_df)

    all_raw_items = journal_papers + rss_news
    print(f"\n[INFO] Total raw items collected: {len(all_raw_items)}")

    # Build raw stats
    source_counts_raw = Counter(it.get("source", "Unknown") for it in all_raw_items)

    if not all_raw_items:
        print(f"[INFO] No new items found in the last {LOOKBACK_DAYS} days.")
        review_stats = {
            "start_date": start_date.isoformat(),
            "end_date": today.isoformat(),
            "total_raw": 0,
            "per_source": {},
        }
        html = build_newsletter_html([], review_stats, journal_summaries={})
        save_html(html)
        send_email(html)
        raise SystemExit(0)

    # Use AI to annotate and prioritise (no more hard Yes/No gating)
    annotated_items = summarize_with_ai(all_raw_items, keywords_context, batch_size=3)

    # Build stats for annotated items
    source_counts_annotated = Counter(
        it.get("source", "Unknown") for it in annotated_items
    )
    per_source_stats = {}
    for src, raw_count in source_counts_raw.items():
        per_source_stats[src] = {
            "raw": raw_count,
            "annotated": source_counts_annotated.get(src, 0),
        }

    review_stats = {
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
        "total_raw": len(all_raw_items),
        "per_source": per_source_stats,
    }

    # Generate per-journal summaries based on annotated items
    journal_summaries = generate_journal_summaries(annotated_items)

    # Build, save, and send newsletter
    html = build_newsletter_html(annotated_items, review_stats, journal_summaries)
    save_html(html)
    send_email(html)
