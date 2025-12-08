import os
import json
import datetime
import pandas as pd
import feedparser
import requests
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import io

# Load Environment Variables (for local testing)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --- CONFIGURATION ---
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("config.json not found. Using defaults.")
    config = {}

LOOKBACK_DAYS = config.get("lookback_days", 60)
MAX_ITEMS = config.get("max_items_per_feed", 5)
EMAIL_SUBJECT = config.get("email_subject", "Mining Digest")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- CACHE FOR IDS ---
JOURNAL_ID_CACHE = {}


def get_journal_id(journal_name):
    """
    Finds the OpenAlex ID for a journal name.
    Example: 'Nature Geoscience' -> 'S12345678'
    """
    if journal_name in JOURNAL_ID_CACHE:
        return JOURNAL_ID_CACHE[journal_name]

    url = "https://api.openalex.org/sources"
    params = {"filter": f'display_name.search:"{journal_name}"', "per-page": 1}

    try:
        r = requests.get(url, params=params, headers=HEADERS)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                full_id = results[0]["id"]
                short_id = full_id.replace("https://openalex.org/", "")
                JOURNAL_ID_CACHE[journal_name] = short_id
                print(f"  [ID Found] {journal_name} -> {short_id}")
                return short_id
    except Exception as e:
        print(f"  [ID Error] Could not resolve {journal_name}: {e}")

    return None


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return "Abstract not available in database."
    word_list = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos, word))
    sorted_words = sorted(word_list, key=lambda x: x[0])
    return " ".join([word for _, word in sorted_words])


def fetch_openalex_journal_watch(sources_df):
    print(f"Querying OpenAlex (Last {LOOKBACK_DAYS} days)...")
    papers = []

    target_journals = sources_df[sources_df["type"] == "journal"]["identifier"].tolist()

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=LOOKBACK_DAYS)

    for journal in target_journals:
        # STEP 1: RESOLVE NAME TO ID
        journal_id = get_journal_id(journal)

        if not journal_id:
            print(f"  > Skipping {journal} (ID not found)")
            continue

        # STEP 2: QUERY BY ID
        base_url = "https://api.openalex.org/works"
        filter_str = f"primary_location.source.id:{journal_id},from_publication_date:{start_date}"

        params = {
            "filter": filter_str,
            "per-page": MAX_ITEMS,
            "select": "id,title,doi,abstract_inverted_index,primary_location",
        }

        try:
            r = requests.get(base_url, params=params, headers=HEADERS)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                if results:
                    print(f"  > Found {len(results)} in {journal}")

                for work in results:
                    abstract_text = reconstruct_abstract(
                        work.get("abstract_inverted_index")
                    )
                    papers.append(
                        {
                            "title": work.get("title"),
                            "url": work.get("doi") or work.get("id"),
                            "abstract": abstract_text,
                            "source": journal,
                            "topic": "Journal Watch",
                            "type": "Academic",
                        }
                    )
            else:
                print(f"  > Failed {journal}: Status {r.status_code}")

            time.sleep(0.2)
        except Exception as e:
            print(f"Skipping {journal}: {e}")

    return papers


def fetch_rss_feeds(sources_df):
    print("Fetching RSS feeds...")
    news_items = []
    rss_sources = sources_df[sources_df["type"] == "rss"]

    for _, row in rss_sources.iterrows():
        try:
            response = requests.get(row["identifier"], headers=HEADERS, timeout=15)
            if response.status_code != 200:
                print(f"  > RSS Error {row['name']}: Status {response.status_code}")
                continue

            feed = feedparser.parse(io.BytesIO(response.content))

            if not feed.entries:
                feed = feedparser.parse(row["identifier"])

            if feed.entries:
                print(f"  > Found {len(feed.entries[:MAX_ITEMS])} in {row['name']}")

            for entry in feed.entries[:MAX_ITEMS]:
                abstract_text = (
                    entry.get("summary")
                    or entry.get("description")
                    or "No summary provided."
                )
                news_items.append(
                    {
                        "title": entry.title,
                        "url": entry.link,
                        "abstract": abstract_text[:1500],
                        "source": row["name"],
                        "topic": "Industry News",
                        "type": "News",
                    }
                )
        except Exception as e:
            print(f"Error fetching RSS {row['name']}: {e}")
    return news_items


def summarize_with_ai(items, keywords_context):
    print(f"\nAnalyzing {len(items)} items with AI...")
    newsletter_content = []
    interests = ", ".join(keywords_context.values())

    for i, item in enumerate(items):
        # Progress Counter
        print(f"Processing {i + 1}/{len(items)}: {item['title'][:30]}...")

        prompt = f"""
        You are a Mining Industry Research Assistant. 
        My Technical Interests: {interests}.
        
        Evaluate this publication:
        Title: {item["title"]}
        Abstract: {item["abstract"]}
        
        1. RELEVANCE: Is this paper technically significant for my interests? (Yes/No)
        2. CATEGORY: Classify it (e.g., Comminution, ESG, Automation, Hydrometallurgy).
        3. HIGHLIGHT: Summarize the core finding or innovation in 1 sentence (based on the abstract).
        
        Output format: Yes/No | [Category] | [Highlight]
        """

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            if text.lower().startswith("yes"):
                parts = text.split("|")
                if len(parts) >= 3:
                    item["topic"] = parts[1].strip()
                    item["highlight"] = parts[2].strip()
                    newsletter_content.append(item)
                    print(f"   > APPROVED: {item['title'][:30]}...")
            else:
                print(f"   > Skipped (Irrelevant)")

        except Exception as e:
            print(f"   > AI Error on item {i}: {e}")
            # Rate Limit Logic
            if "429" in str(e):
                print("   > Hit Rate Limit. Cooling down for 60 seconds...")
                time.sleep(60)

        # THROTTLE: Wait 4 seconds between requests to respect Free Tier limits
        time.sleep(4)

    return newsletter_content


def send_email(content):
    if not content:
        print("No content to send.")
        return

    html = """
    <html>
    <body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333;">
        <div style="background-color: #2c3e50; padding: 20px; text-align: center;">
            <h2 style="color: #ecf0f1; margin: 0;">⛏️ Mining Research Digest</h2>
        </div>
        <div style="padding: 20px;">
    """

    df = pd.DataFrame(content)
    if "topic" in df.columns:
        df.sort_values(by="topic", inplace=True)

    current_topic = ""
    for _, item in df.iterrows():
        if item.get("topic") != current_topic:
            current_topic = item.get("topic", "General")
            html += f"<h3 style='color: #d35400; border-bottom: 2px solid #d35400; padding-bottom: 5px; margin-top: 25px;'>{current_topic}</h3>"

        html += f"""
        <div style="margin-bottom: 20px; padding-left: 10px; border-left: 4px solid #2980b9;">
            <a href="{item["url"]}" style="font-weight: bold; font-size: 16px; color: #2c3e50; text-decoration: none;">{item["title"]}</a>
            <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px; margin-bottom: 6px;">Source: {item["source"]}</div>
            <div style="background-color: #f8f9f9; padding: 8px; border-radius: 4px; font-style: italic; color: #444;">
                <span style="font-weight:bold; color: #d35400;">AI Highlight:</span> {item["highlight"]}
            </div>
        </div>
        """

    html += "</div><div style='text-align: center; font-size: 12px; color: #aaa; padding: 20px;'>Generated by AI Agent</div></body></html>"

    msg = MIMEMultipart()
    msg["Subject"] = f"{EMAIL_SUBJECT} - {datetime.date.today()}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    print("Email sent successfully.")


if __name__ == "__main__":
    if not os.path.exists("keywords.csv") or not os.path.exists("sources.csv"):
        print("Error: keywords.csv or sources.csv not found.")
    else:
        keywords_df = pd.read_csv("keywords.csv")
        sources_df = pd.read_csv("sources.csv")
        keywords_context = dict(zip(keywords_df.topic, keywords_df.keywords))

        print("Starting collection...")
        journal_papers = fetch_openalex_journal_watch(sources_df)
        rss_news = fetch_rss_feeds(sources_df)

        all_raw_items = journal_papers + rss_news

        if all_raw_items:
            final_digest = summarize_with_ai(all_raw_items, keywords_context)
            send_email(final_digest)
        else:
            print(f"No new items found in the last {LOOKBACK_DAYS} days.")
