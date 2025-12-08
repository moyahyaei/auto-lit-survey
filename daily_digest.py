import os
import datetime
import pandas as pd
import feedparser
import requests
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EMAIL_SENDER = os.environ["EMAIL_SENDER"]
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]
EMAIL_RECEIVER = os.environ["EMAIL_RECEIVER"]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def fetch_openalex_journal_watch(sources_df):
    """Fetches papers specifically from the high-impact journals listed in sources.csv"""
    print("Querying OpenAlex for specific journals...")
    papers = []

    # Filter for sources marked as 'journal'
    target_journals = sources_df[sources_df["type"] == "journal"]["identifier"].tolist()

    today = datetime.date.today()
    seven_days_ago = today - datetime.timedelta(days=7)
    date_filter = f"from_publication_date:{seven_days_ago}"

    # We query OpenAlex for each journal.
    # Note: To be respectful of free API limits, we add a small sleep.
    for journal in target_journals:
        # Construct query: filter by venue name and date
        # We replace spaces with %20 for URL encoding
        journal_query = f'primary_location.source.display_name:"{journal}"'
        url = f"https://api.openalex.org/works?filter={journal_query},{date_filter}&per-page=3&select=id,title,doi,abstract_inverted_index,primary_location"

        try:
            r = requests.get(url)
            if r.status_code == 200:
                data = r.json()
                for work in data.get("results", []):
                    # Reconstruct abstract
                    abstract_text = "No abstract available."
                    if work.get("abstract_inverted_index"):
                        # Simplified reconstruction (OpenAlex stores abstracts as inverted indexes)
                        # For the AI summary, often just the title + venue is enough if abstract is complex
                        abstract_text = "Abstract available via link."

                    papers.append(
                        {
                            "title": work.get("title"),
                            "url": work.get("doi") or work.get("id"),
                            "abstract": abstract_text,
                            "source": journal,
                            "topic": "Journal Watch",  # We group these together
                            "type": "Academic",
                        }
                    )
            time.sleep(0.5)  # Politeness delay
        except Exception as e:
            print(f"Skipping {journal}: {e}")

    return papers


def fetch_rss_feeds(sources_df):
    """Fetches from the RSS URLs provided."""
    print("Fetching RSS feeds...")
    news_items = []

    rss_sources = sources_df[sources_df["type"] == "rss"]

    for _, row in rss_sources.iterrows():
        try:
            feed = feedparser.parse(row["identifier"])
            for entry in feed.entries[:3]:  # Top 3 per feed
                news_items.append(
                    {
                        "title": entry.title,
                        "url": entry.link,
                        "abstract": entry.get("summary", "")[:500],
                        "source": row["name"],
                        "topic": "Industry News",
                        "type": "News",
                    }
                )
        except Exception as e:
            print(f"Error fetching RSS {row['name']}: {e}")
    return news_items


def summarize_with_ai(items, keywords_context):
    """
    Uses Gemini to filter and summarize.
    We inject the user's specific keywords into the prompt so the AI knows what to prioritize.
    """
    print(f"Analyzing {len(items)} items with AI...")
    newsletter_content = []

    # Create a string of user interests for the prompt
    interests = ", ".join(keywords_context.values())

    for item in items:
        prompt = f"""
        You are a Mining Industry Expert. I am interested in: {interests}.
        
        Analyze this publication:
        Source: {item["source"]}
        Title: {item["title"]}
        Abstract Snippet: {item["abstract"]}
        
        Tasks:
        1. RELEVANCE: Is this highly relevant to my interests? (Yes/No)
        2. HIGHLIGHT: Write one sentence explaining the *technical* value (e.g., "Proposes a new collector for sulfide flotation" or "Discusses ESG risks in tailing dams").
        3. CATEGORY: Pick the best matching category from my interests (e.g., "Comminution", "ESG", "Automation").
        
        Output format: Yes/No | [Category] | [Highlight]
        """

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            if text.lower().startswith("yes"):
                parts = text.split("|")
                if len(parts) >= 3:
                    item["topic"] = parts[
                        1
                    ].strip()  # Overwrite generic topic with AI-selected category
                    item["highlight"] = parts[2].strip()
                    newsletter_content.append(item)
                    print(f" > Added: {item['title'][:30]}...")
        except Exception as e:
            print(f"AI Error: {e}")

    return newsletter_content


def send_email(content):
    if not content:
        print("No content to send.")
        return

    html = """
    <html>
    <body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333;">
        <div style="background-color: #2c3e50; padding: 20px; text-align: center;">
            <h2 style="color: #ecf0f1; margin: 0;">⛏️ Mining & Mineral Processing Weekly</h2>
        </div>
        <div style="padding: 20px;">
    """

    # Group by AI-assigned Topic
    df = pd.DataFrame(content)
    # Sort by Topic to keep email organized
    df.sort_values(by="topic", inplace=True)

    current_topic = ""
    for _, item in df.iterrows():
        if item["topic"] != current_topic:
            current_topic = item["topic"]
            html += f"<h3 style='color: #d35400; border-bottom: 2px solid #d35400; padding-bottom: 5px; margin-top: 25px;'>{current_topic}</h3>"

        html += f"""
        <div style="margin-bottom: 15px; padding-left: 10px; border-left: 3px solid #eee;">
            <a href="{item["url"]}" style="font-weight: bold; font-size: 16px; color: #2980b9; text-decoration: none;">{item["title"]}</a>
            <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">Source: {item["source"]}</div>
            <div style="font-style: italic; margin-top: 5px;">💡 {item["highlight"]}</div>
        </div>
        """

    html += "</div><div style='text-align: center; font-size: 12px; color: #aaa; padding: 20px;'>Generated by AI Agent</div></body></html>"

    msg = MIMEMultipart()
    msg["Subject"] = (
        f"Mining Intel: {len(content)} New Papers ({datetime.date.today()})"
    )
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
    print("Email sent.")


# --- MAIN ---
if __name__ == "__main__":
    # Load Data
    keywords_df = pd.read_csv("keywords.csv")
    sources_df = pd.read_csv("sources.csv")

    keywords_context = dict(zip(keywords_df.topic, keywords_df.keywords))

    # Fetch
    print("Starting collection...")
    journal_papers = fetch_openalex_journal_watch(sources_df)
    rss_news = fetch_rss_feeds(sources_df)

    all_raw_items = journal_papers + rss_news

    # AI Process
    if all_raw_items:
        final_digest = summarize_with_ai(all_raw_items, keywords_context)
        send_email(final_digest)
    else:
        print("No new items found in the last 7 days.")
