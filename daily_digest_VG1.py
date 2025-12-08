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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

JOURNAL_ID_CACHE = {}


def truncate_text(text: str, max_chars: int = 800) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text[:max_chars] + "..." if len(text) > max_chars else text


def get_journal_id(journal_name: str):
    if journal_name in JOURNAL_ID_CACHE:
        return JOURNAL_ID_CACHE[journal_name]
    url = "https://api.openalex.org/sources"
    params = {"filter": f'display_name.search:"{journal_name}"', "per-page": 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                short_id = results[0]["id"].split("/")[-1]
                JOURNAL_ID_CACHE[journal_name] = short_id
                print(f"[OpenAlex] ID found: {journal_name} -> {short_id}")
                return short_id
    except Exception as e:
        print(f"[OpenAlex] Error resolving {journal_name}: {e}")
    return None


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return "Abstract not available."
    word_list = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos, word))
    return " ".join(word for _, word in sorted(word_list, key=lambda x: x[0]))


def fetch_openalex_journal_watch(sources_df: pd.DataFrame):
    print(f"\n[STEP] Querying OpenAlex (Optimized Batch Mode)...")
    papers = []
    journal_rows = sources_df[sources_df["type"] == "journal"]
    if journal_rows.empty:
        return papers

    valid_journals = []
    for _, row in journal_rows.iterrows():
        jid = get_journal_id(row["identifier"])
        if jid:
            valid_journals.append({"id": jid, "name": row["identifier"]})

    if not valid_journals:
        return papers

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=LOOKBACK_DAYS)

    chunk_size = 15
    for i in range(0, len(valid_journals), chunk_size):
        chunk = valid_journals[i : i + chunk_size]
        ids_or_string = "|".join([j["id"] for j in chunk])

        base_url = "https://api.openalex.org/works"
        params = {
            "filter": f"primary_location.source.id:{ids_or_string},from_publication_date:{start_date}",
            "per-page": 50,
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
            time.sleep(0.5)
        except Exception as e:
            print(f"[OpenAlex] Error in batch: {e}")
    return papers


def fetch_single_rss(row):
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
    print("\n[STEP] Fetching RSS feeds (Parallel)...")
    rss_rows = sources_df[sources_df["type"] == "rss"]
    all_items = []
    if rss_rows.empty:
        return all_items

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_single_rss, row) for _, row in rss_rows.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            all_items.extend(future.result())
    return all_items


def summarize_with_ai(items, keywords_context, batch_size: int = 5):
    if not items:
        return []
    print(f"\n[STEP] Analyzing {len(items)} items with Gemini...")
    newsletter_items = []
    interests = ", ".join(keywords_context.values())

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    for i, batch in enumerate(batches):
        print(f"[AI] Processing Batch {i + 1}/{len(batches)} ({len(batch)} items)...")
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

        retries = 3
        while retries > 0:
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                for line in lines:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        try:
                            idx = int(parts[0].strip()) - 1
                            if 0 <= idx < len(batch):
                                decision = (
                                    parts[1]
                                    .strip()
                                    .lower()
                                    .replace("*", "")
                                    .replace("[", "")
                                    .replace("]", "")
                                )
                                if "yes" in decision:
                                    item = batch[idx]
                                    item["topic"] = parts[2].strip()
                                    item["priority"] = parts[3].strip()
                                    item["highlight"] = parts[4].strip()
                                    newsletter_items.append(item)
                                    print(f"  -> Approved: {item['title'][:40]}...")
                        except ValueError:
                            continue
                break
            except Exception as e:
                if "429" in str(e):
                    print(f"  [!] Rate Limit Hit. Waiting 60s...")
                    time.sleep(60)
                    retries -= 1
                else:
                    print(f"  [!] Error: {e}")
                    break
        time.sleep(4)

    return newsletter_items


def build_newsletter_html(items, total_checked=0, is_fallback=False):
    today_str = datetime.date.today().isoformat()

    html = f"""<html><body style="font-family:sans-serif; color:#333;">
    <div style="background:#2c3e50; padding:20px; text-align:center; color:white;">
    <h2>⛏️ Mining Intelligence Digest</h2>
    <p style="font-size:12px;">Date: {today_str}</p></div><div style="padding:20px;">"""

    if is_fallback:
        html += f"""
        <div style="border: 2px solid #e74c3c; padding: 15px; margin-bottom: 20px; text-align: center; border-radius: 5px; background-color: #fdedec;">
            <h3 style="color: #c0392b; margin:0;">⚠️ Fallback Mode Active</h3>
            <p>We collected <strong>{total_checked}</strong> raw items, but the AI filter rejected all of them (or failed). 
            Below is the <strong>unfiltered list</strong> so you don't miss anything.</p>
        </div>
        """

    if not items:
        html += (
            "<p>No items found (Total vacuum). Check your sources configuration.</p>"
        )
    else:
        df = pd.DataFrame(items)
        # Handle cases where priority might be missing in fallback mode
        df["priority"] = df.get("priority", "2")
        df["p_rank"] = df["priority"].apply(lambda x: 1 if "1" in str(x) else 2)
        df.sort_values(by=["p_rank", "topic"], inplace=True)

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


def save_html(html: str):
    """Saves HTML to output folder so GitHub Actions can find it."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    path = os.path.join(output_dir, f"newsletter.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OUTPUT] Newsletter saved to: {path}")


def send_email(html: str):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("[EMAIL] Config missing. Skipping.")
        return

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

        # 1. Collect Data
        papers = fetch_openalex_journal_watch(sources_df)
        news = fetch_rss_feeds_parallel(sources_df)
        all_items = papers + news

        print(f"\n[INFO] Total raw items collected: {len(all_items)}")

        # 2. AI Analysis
        final_items = []
        is_fallback = False

        if all_items:
            final_items = summarize_with_ai(all_items, keywords_context)

            # --- FALLBACK LOGIC ---
            # If we found raw items but AI approved NONE, use the raw list.
            if not final_items:
                print(
                    "[WARN] AI approved 0 items. Generating FALLBACK newsletter with all raw items."
                )
                is_fallback = True
                final_items = all_items

                # Apply defaults so HTML builder doesn't crash
                for item in final_items:
                    item["priority"] = "2"  # Default to 'Worth Scanning'
                    if "topic" not in item:
                        item["topic"] = "Raw Feed (Unfiltered)"
                    if "highlight" not in item:
                        item["highlight"] = item.get("abstract", "")[:200] + "..."

        # 3. Generate Output (Guaranteed to run if all_items > 0)
        html_content = build_newsletter_html(
            final_items, total_checked=len(all_items), is_fallback=is_fallback
        )

        save_html(html_content)
        send_email(html_content)
