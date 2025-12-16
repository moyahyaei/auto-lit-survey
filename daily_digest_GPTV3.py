import os
import re
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup

import google.generativeai as genai

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv


# =========================
# Utilities
# =========================


def utc_today_date() -> dt.date:
    # Use UTC to avoid local timezone surprises in GitHub Actions
    return dt.datetime.utcnow().date()


def parse_date_flexible(s: str) -> Optional[dt.date]:
    """
    Tries to parse a date from various formats.
    Returns None if parsing fails.
    """
    if not s:
        return None
    s = s.strip()

    # common: "Sep 1, 2024"
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    # RSS often uses RFC822 or ISO-ish; feedparser may already parse into struct_time
    # We handle simple ISO substrings as fallback:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    return None


def in_window(d: Optional[dt.date], start: dt.date, end: dt.date) -> bool:
    """
    Inclusive window: start <= d <= end.
    If d is None => False.
    """
    if d is None:
        return False
    return start <= d <= end


def safe_get_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# =========================
# Data model
# =========================


@dataclass
class Item:
    title: str
    url: str
    source: str  # journal / rss / repository name
    source_type: str  # "Academic" | "News" | "Repository"
    published_date: Optional[dt.date]
    authors: Optional[str] = None
    abstract: Optional[str] = None

    # Filled by AI
    decision: Optional[str] = None  # "MUST_READ" | "SCAN" | "SKIP"
    category: Optional[str] = None
    highlight: Optional[str] = None
    rationale: Optional[str] = None


# =========================
# Config & Environment
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_PATH = os.path.join(SCRIPT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)
print(f"[ENV] Loaded .env from: {ENV_PATH} (exists={os.path.exists(ENV_PATH)})")

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"[CONFIG] Loaded config.json from: {CONFIG_PATH}")
else:
    print("[CONFIG] config.json not found. Using defaults.")
    config = {
        "lookback_days": 7,
        "max_items_per_feed": 5,
        "email_subject": "Weekly Mining & Processing Digest",
    }

LOOKBACK_DAYS = int(config.get("lookback_days", 7))
MAX_ITEMS_PER_FEED = int(config.get("max_items_per_feed", 5))
EMAIL_SUBJECT = str(config.get("email_subject", "Weekly Mining & Processing Digest"))

TODAY = utc_today_date()
START_DATE = TODAY - dt.timedelta(days=LOOKBACK_DAYS)
END_DATE = TODAY  # inclusive end date

print(f"[WINDOW] Coverage: {START_DATE} -> {END_DATE} (lookback_days={LOOKBACK_DAYS})")

# Env vars (local via .env; GitHub Actions via secrets)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

print(f"[ENV CHECK] GEMINI_API_KEY present? {bool(GEMINI_API_KEY)}")
print(f"[ENV CHECK] EMAIL_SENDER present? {bool(EMAIL_SENDER)}")
print(f"[ENV CHECK] EMAIL_RECEIVER present? {bool(EMAIL_RECEIVER)}")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Set it in .env (local) or GitHub Secrets (Actions)."
    )

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.0-pro"
print(f"[GEMINI] Using model: {GEMINI_MODEL}")
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MiningDigestBot/1.0; +https://example.com/bot)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# =========================
# Load Inputs
# =========================


def load_keywords_csv(path: str) -> Dict[str, str]:
    """
    keywords.csv expected format:
    topic,keywords
    Mineral Processing,"flotation, comminution, grinding"
    """
    df = pd.read_csv(path)
    if "topic" not in df.columns or "keywords" not in df.columns:
        raise ValueError("keywords.csv must have columns: topic, keywords")
    mapping = dict(zip(df["topic"].astype(str), df["keywords"].astype(str)))
    return mapping


def load_sources_csv(path: str) -> pd.DataFrame:
    """
    sources.csv expected columns:
    name,type,identifier

    type can include:
      - journal     (OpenAlex journal/source name)
      - rss         (RSS url)
      - repository  (OneMine, etc.)
    """
    df = pd.read_csv(path)
    needed = {"name", "type", "identifier"}
    if not needed.issubset(set(df.columns)):
        raise ValueError("sources.csv must have columns: name, type, identifier")
    df["name"] = df["name"].astype(str)
    df["type"] = df["type"].astype(str).str.lower()
    df["identifier"] = df["identifier"].astype(str)
    return df


# =========================
# OpenAlex: Journals
# =========================

OPENALEX_SOURCE_CACHE: Dict[str, str] = {}


def openalex_resolve_source_id(journal_name: str) -> Optional[str]:
    """
    Resolve an OpenAlex 'source' id from a display name.
    Returns short id like 'S123456789'.
    """
    j = journal_name.strip()
    if j in OPENALEX_SOURCE_CACHE:
        return OPENALEX_SOURCE_CACHE[j]

    url = "https://api.openalex.org/sources"
    params = {"filter": f'display_name.search:"{j}"', "per-page": 1}
    r = SESSION.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print(f"[OpenAlex][ID] Failed for '{j}' status={r.status_code}")
        return None

    results = r.json().get("results", [])
    if not results:
        print(f"[OpenAlex][ID] No match for '{j}'")
        return None

    full_id = results[0].get("id", "")
    short_id = full_id.replace("https://openalex.org/", "").strip()
    if not short_id:
        print(f"[OpenAlex][ID] Bad id for '{j}'")
        return None

    OPENALEX_SOURCE_CACHE[j] = short_id
    print(f"[OpenAlex][ID] {j} -> {short_id}")
    return short_id


def openalex_reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    if not inverted_index:
        return ""
    words = []
    for w, positions in inverted_index.items():
        for p in positions:
            words.append((p, w))
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


def fetch_openalex_journal_items(
    sources_df: pd.DataFrame, start: dt.date, end: dt.date
) -> List[Item]:
    """
    Fetch items from OpenAlex for all journal entries in sources.csv.
    Enforces date filtering in query AND again locally (safety).
    """
    out: List[Item] = []

    journals = sources_df[sources_df["type"] == "journal"]["identifier"].tolist()
    if not journals:
        print("[OpenAlex] No journals configured.")
        return out

    print(f"[OpenAlex] Querying journals for {start} -> {end} ...")

    for jname in journals:
        jid = openalex_resolve_source_id(jname)
        if not jid:
            print(f"[OpenAlex] Skipping journal '{jname}' (no source id).")
            continue

        # OpenAlex supports from_publication_date and to_publication_date in filters.
        base_url = "https://api.openalex.org/works"
        filter_str = (
            f"primary_location.source.id:{jid},"
            f"from_publication_date:{start},"
            f"to_publication_date:{end}"
        )
        params = {
            "filter": filter_str,
            "per-page": MAX_ITEMS_PER_FEED,
            "sort": "publication_date:desc",
            "select": "id,doi,title,publication_date,abstract_inverted_index,authorships,primary_location",
        }

        try:
            r = SESSION.get(base_url, params=params, timeout=45)
            if r.status_code != 200:
                print(f"[OpenAlex] {jname}: HTTP {r.status_code}")
                continue
            data = r.json()
            results = data.get("results", [])
            print(f"[OpenAlex] {jname}: fetched {len(results)} candidates")

            for w in results:
                pub = parse_date_flexible(w.get("publication_date", ""))
                if not in_window(pub, start, end):
                    # Extra safety: don't include out-of-window even if API returned it
                    continue

                title = normalize_ws(w.get("title", ""))
                doi = w.get("doi")
                wid = w.get("id")
                url = doi if doi else wid

                # Authors
                auths = []
                for a in w.get("authorships") or []:
                    name = ((a.get("author") or {}).get("display_name") or "").strip()
                    if name:
                        auths.append(name)
                authors = ", ".join(auths) if auths else None

                abstract = openalex_reconstruct_abstract(
                    w.get("abstract_inverted_index")
                )
                abstract = normalize_ws(abstract)

                out.append(
                    Item(
                        title=title,
                        url=url,
                        source=jname,
                        source_type="Academic",
                        published_date=pub,
                        authors=authors,
                        abstract=abstract,
                    )
                )

            time.sleep(0.2)  # be polite
        except Exception as e:
            print(f"[OpenAlex] {jname}: error {e}")

    print(f"[OpenAlex] Total in-window items: {len(out)}")
    return out


# =========================
# RSS
# =========================


def parse_rss_entry_date(entry) -> Optional[dt.date]:
    # feedparser provides published_parsed/updated_parsed as time.struct_time
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return dt.date(
            entry.published_parsed.tm_year,
            entry.published_parsed.tm_mon,
            entry.published_parsed.tm_mday,
        )
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return dt.date(
            entry.updated_parsed.tm_year,
            entry.updated_parsed.tm_mon,
            entry.updated_parsed.tm_mday,
        )

    # fallback string
    s = getattr(entry, "published", "") or getattr(entry, "updated", "") or ""
    return parse_date_flexible(s)


def fetch_rss_items(
    sources_df: pd.DataFrame, start: dt.date, end: dt.date
) -> List[Item]:
    out: List[Item] = []
    rss_df = sources_df[sources_df["type"] == "rss"]
    if rss_df.empty:
        print("[RSS] No rss sources configured.")
        return out

    print(f"[RSS] Fetching RSS for {start} -> {end} ...")

    for _, row in rss_df.iterrows():
        name = row["name"].strip()
        rss_url = row["identifier"].strip()

        try:
            resp = SESSION.get(rss_url, timeout=30)
            if resp.status_code != 200:
                print(f"[RSS] {name}: HTTP {resp.status_code}")
                continue

            feed = feedparser.parse(resp.content)
            entries = feed.entries[
                : MAX_ITEMS_PER_FEED * 5
            ]  # take more; filter by dates afterwards
            kept = 0

            for e in entries:
                pub = parse_rss_entry_date(e)
                if not in_window(pub, start, end):
                    continue

                title = normalize_ws(getattr(e, "title", "") or "")
                link = getattr(e, "link", "") or ""

                # For RSS, "summary/description" is all we have; do NOT truncate
                summary = (
                    getattr(e, "summary", "") or getattr(e, "description", "") or ""
                )
                summary = normalize_ws(summary)

                out.append(
                    Item(
                        title=title,
                        url=link,
                        source=name,
                        source_type="News",
                        published_date=pub,
                        abstract=summary,
                    )
                )
                kept += 1
                if kept >= MAX_ITEMS_PER_FEED:
                    break

            print(f"[RSS] {name}: kept {kept} in-window items")
        except Exception as e:
            print(f"[RSS] {name}: error {e}")

    print(f"[RSS] Total in-window items: {len(out)}")
    return out


# =========================
# OneMine Repository Scraper (Option A)
# =========================


def onemine_build_search_url(
    organization: str,
    sort_by: str,
    start: dt.date,
    end: dt.date,
    page: int = 1,
    page_size: int = 20,
    keywords: Optional[str] = None,
) -> str:
    """
    Build OneMine search URL that supports:
      - Organization
      - SortBy=MostRecent
      - DateFrom / DateTo
      - Keywords (optional)
    """
    base = "https://www.onemine.org/search"
    params = {
        "Organization": organization,
        "SortBy": sort_by,
        "page": str(page),
        "pageSize": str(page_size),
        "DateFrom": start.isoformat(),
        "DateTo": end.isoformat(),
    }
    if keywords:
        params["Keywords"] = keywords
        params["SearchField"] = "All"
    # manual query build (avoid adding extra deps)
    q = "&".join([f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()])
    return f"{base}?{q}"


def onemine_parse_list_page(html: str) -> List[dict]:
    """
    Parse OneMine search results list page.
    Returns list of dicts with: title, url, author_text, date_text, snippet
    """
    soup = BeautifulSoup(html, "html.parser")
    items = []
    # Each result is <li class="item-list__item">
    for li in soup.select("ul.item-list li.item-list__item"):
        a_title = li.select_one("a.item-list__title")
        if not a_title:
            continue
        title = safe_get_text(a_title)
        href = a_title.get("href", "").strip()
        if href.startswith("/"):
            url = "https://www.onemine.org" + href
        else:
            url = href

        # Author line like: <p class="item-list__date">By ...</p> (first occurrence)
        # Published date line like: <p class="item-list__date">Sep 1, 2024</p> (later occurrence)
        date_ps = li.select("p.item-list__date")
        author_text = safe_get_text(date_ps[0]) if len(date_ps) >= 1 else ""
        pub_text = safe_get_text(date_ps[-1]) if len(date_ps) >= 2 else ""

        # snippet is visible <p> ... with hidden span. We’ll later fetch full page anyway.
        p_desc = li.select_one("p")
        snippet = safe_get_text(p_desc)

        items.append(
            {
                "title": normalize_ws(title),
                "url": url,
                "author_text": normalize_ws(author_text),
                "date_text": normalize_ws(pub_text),
                "snippet": normalize_ws(snippet),
            }
        )
    return items


def onemine_fetch_document_abstract(
    doc_url: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch a OneMine document page and extract:
      - authors (if available)
      - abstract/description full text (best-effort)
    """
    try:
        r = SESSION.get(doc_url, timeout=45)
        if r.status_code != 200:
            return None, None
        soup = BeautifulSoup(r.text, "html.parser")

        # Heuristics: many OneMine docs have main content in article/section blocks.
        # We try a few likely selectors.
        # 1) meta description as fallback
        meta_desc = soup.select_one('meta[name="description"]')
        meta_abstract = meta_desc.get("content", "").strip() if meta_desc else ""

        # 2) Look for a prominent paragraph block
        # Often description is inside main content area, may include <p> with substantial text.
        main = soup.select_one("div.content[role='main']") or soup
        paragraphs = [
            normalize_ws(p.get_text(" ", strip=True)) for p in main.select("p")
        ]

        # Remove short boilerplate lines
        paragraphs = [p for p in paragraphs if len(p) > 80]

        # Build abstract as the longest paragraph or concatenation of first few
        abstract = ""
        if paragraphs:
            # choose top 2-3 long paragraphs
            paragraphs.sort(key=len, reverse=True)
            abstract = " ".join(paragraphs[:3]).strip()
        elif meta_abstract:
            abstract = meta_abstract

        # Authors: sometimes shown near top, but list pages already provide "By ..."
        # We keep it best-effort; if not found, caller uses list page.
        # Try to find “By …” patterns
        text = main.get_text(" ", strip=True)
        m = re.search(r"\bBy\s+([A-Z][^|•\n]{3,200})", text)
        authors = m.group(0).strip() if m else None

        return normalize_ws(authors or ""), normalize_ws(abstract or "")
    except Exception:
        return None, None


def fetch_onemine_repository_items(
    sources_df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    max_pages: int = 3,
    page_size: int = 20,
    keywords: Optional[str] = None,
) -> List[Item]:
    """
    Scrape OneMine search pages for configured repository sources.

    sources.csv: include a row like:
      name,type,identifier
      OneMine AUSIMM,repository,onemine:Organization=AUSIMM

    Supported identifier formats:
      - onemine:Organization=AUSIMM
      - onemine:Organization=SME
    """
    out: List[Item] = []
    repo_df = sources_df[sources_df["type"] == "repository"]
    if repo_df.empty:
        print("[Repo] No repository sources configured.")
        return out

    # Find OneMine entries
    onemine_rows = []
    for _, row in repo_df.iterrows():
        ident = row["identifier"].strip()
        if ident.lower().startswith("onemine:"):
            onemine_rows.append(row)

    if not onemine_rows:
        print(
            "[Repo] No OneMine repository entries found (identifier must start with 'onemine:')."
        )
        return out

    print(f"[Repo][OneMine] Fetching OneMine for {start} -> {end} ...")

    for row in onemine_rows:
        name = row["name"].strip()
        ident = row["identifier"].strip()

        # parse organization
        # ident like "onemine:Organization=AUSIMM"
        org = None
        m = re.search(r"Organization=([A-Za-z0-9_]+)", ident, flags=re.IGNORECASE)
        if m:
            org = m.group(1)
        if not org:
            print(
                f"[Repo][OneMine] {name}: cannot parse Organization from identifier='{ident}'"
            )
            continue

        kept = 0
        candidates = []

        for page in range(1, max_pages + 1):
            url = onemine_build_search_url(
                organization=org,
                sort_by="MostRecent",
                start=start,
                end=end,
                page=page,
                page_size=page_size,
                keywords=keywords,
            )
            print(f"[Repo][OneMine] GET {name}: page={page} url={url}")

            r = SESSION.get(url, timeout=45)
            if r.status_code != 200:
                print(f"[Repo][OneMine] {name}: HTTP {r.status_code} on page {page}")
                break

            parsed = onemine_parse_list_page(r.text)
            print(
                f"[Repo][OneMine] {name}: parsed {len(parsed)} list items on page {page}"
            )

            if not parsed:
                break

            candidates.extend(parsed)
            time.sleep(0.3)

        # Turn candidates into Items, fetch full abstract per document page
        for c in candidates:
            pub = parse_date_flexible(c.get("date_text", ""))
            if not in_window(pub, start, end):
                continue

            title = c["title"]
            url = c["url"]
            author_text = c.get("author_text", "")
            authors = (
                author_text.replace("By ", "").strip()
                if author_text.lower().startswith("by ")
                else author_text
            )

            # Fetch full doc page to get better abstract
            doc_auth, abstract = onemine_fetch_document_abstract(url)
            if doc_auth and len(doc_auth) > len(authors or ""):
                authors = (
                    doc_auth.replace("By ", "").strip()
                    if doc_auth.lower().startswith("by ")
                    else doc_auth
                )

            if not abstract:
                # fallback to snippet
                abstract = c.get("snippet", "")

            out.append(
                Item(
                    title=title,
                    url=url,
                    source=name,
                    source_type="Repository",
                    published_date=pub,
                    authors=authors or None,
                    abstract=abstract or "",
                )
            )
            kept += 1

        print(f"[Repo][OneMine] {name}: kept {kept} in-window items")

    print(f"[Repo][OneMine] Total in-window items: {len(out)}")
    return out


# =========================
# AI Scoring (Robust + Retry)
# =========================

AI_JSON_SCHEMA = """
Return ONLY valid JSON with these keys:
{
  "decision": "MUST_READ" | "SCAN" | "SKIP",
  "category": "short label",
  "highlight": "one sentence highlight",
  "rationale": "one short sentence why"
}
"""


def ai_rate_limit_sleep(e: Exception) -> None:
    msg = str(e)
    # Gemini sometimes provides retry_delay seconds in the exception string
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
    if m:
        delay = int(m.group(1))
        delay = max(delay, 10)
        print(f"[AI] Rate limit hinted: sleeping {delay}s ...")
        time.sleep(delay)
        return
    # fallback
    print("[AI] Hit rate limit or transient error: sleeping 60s ...")
    time.sleep(60)


def gemini_call_with_retry(prompt: str, max_attempts: int = 4) -> str:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            txt = (resp.text or "").strip()
            if not txt:
                raise RuntimeError("Empty Gemini response.")
            return txt
        except Exception as e:
            last_err = e
            print(f"[AI] Error attempt {attempt}/{max_attempts}: {e}")
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                ai_rate_limit_sleep(e)
            else:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Gemini failed after {max_attempts} attempts: {last_err}")


def parse_json_strict(text: str) -> Optional[dict]:
    """
    Extract and parse JSON strictly.
    Handles cases where model wraps JSON in code fences.
    """
    t = text.strip()
    # remove code fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()

    # attempt direct parse
    try:
        return json.loads(t)
    except Exception:
        # try to find a JSON object substring
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def ai_classify_items(
    items: List[Item], keywords_context: Dict[str, str]
) -> List[Item]:
    """
    Classify each item using Gemini.
    NO abstract truncation. We send the full abstract we have.
    """
    if not items:
        return []

    interests = "; ".join([f"{k}: {v}" for k, v in keywords_context.items()])

    rated: List[Item] = []
    print(f"[AI] Classifying {len(items)} items with Gemini ({GEMINI_MODEL}) ...")

    for idx, it in enumerate(items, start=1):
        print(f"[AI] {idx}/{len(items)}: {it.title[:80]}")

        abstract = normalize_ws(it.abstract or "")
        if not abstract:
            abstract = "(No abstract/description available.)"

        prompt = f"""
You are an expert research triage assistant for Mining, Mineral Processing, Extractive Metallurgy, and Automation/Control.

My interests (topics -> keywords):
{interests}

Task:
Evaluate the item below strictly against these interests.
Be tolerant to spelling variants (US/UK), capitalization, minor typos, and terminology differences.
If it's remotely relevant, do NOT over-filter.

Decision rules:
- MUST_READ: highly relevant or impactful for my interests; likely worth deep reading.
- SCAN: relevant enough to skim or keep; may inform trend awareness.
- SKIP: not relevant.

Item:
Title: {it.title}
Source: {it.source} ({it.source_type})
Published date: {it.published_date}
Authors: {it.authors or "Unknown"}
Abstract/Description:
{abstract}

{AI_JSON_SCHEMA}
""".strip()

        try:
            text = gemini_call_with_retry(prompt, max_attempts=4)
            data = parse_json_strict(text)
            if not data:
                print("[AI] Bad JSON response. Marking as SCAN with fallback.")
                it.decision = "SCAN"
                it.category = "Unparsed"
                it.highlight = (
                    "AI returned an unparseable response; included for manual review."
                )
                it.rationale = "Fallback inclusion to avoid false negatives."
                rated.append(it)
                continue

            decision = (data.get("decision") or "").strip().upper()
            if decision not in {"MUST_READ", "SCAN", "SKIP"}:
                decision = "SCAN"

            it.decision = decision
            it.category = normalize_ws(str(data.get("category") or "General"))
            it.highlight = normalize_ws(str(data.get("highlight") or ""))
            it.rationale = normalize_ws(str(data.get("rationale") or ""))

            # keep MUST_READ and SCAN
            if it.decision in {"MUST_READ", "SCAN"}:
                rated.append(it)
            else:
                pass

        except Exception as e:
            print(f"[AI] Failed on item '{it.title[:50]}...': {e}")
            # Safety bias: include as SCAN rather than drop it
            it.decision = "SCAN"
            it.category = "AI_Error"
            it.highlight = "AI error occurred; included for manual review."
            it.rationale = "Fail-open to avoid missing relevant items."
            rated.append(it)

        # Throttle slightly to reduce rate limit pain
        time.sleep(1.0)

    print(f"[AI] Kept {len(rated)} items (MUST_READ/SCAN) after AI triage.")
    return rated


# =========================
# Journal / Source Overviews (No AI needed)
# =========================


def build_source_overviews(items: List[Item]) -> Dict[str, dict]:
    """
    Builds simple, deterministic overviews per source:
    - counts
    - decision breakdown
    - top categories (from AI)
    """
    by_source = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)

    out = {}
    for src, arr in by_source.items():
        decisions = Counter([a.decision or "UNKNOWN" for a in arr])
        cats = Counter([a.category or "General" for a in arr]).most_common(5)

        out[src] = {
            "total": len(arr),
            "decisions": dict(decisions),
            "top_categories": cats,
        }
    return out


# =========================
# Newsletter Rendering (HTML)
# =========================


def render_newsletter_html(
    start: dt.date,
    end: dt.date,
    collected_counts: Dict[str, int],
    included_counts: Dict[str, int],
    source_overviews: Dict[str, dict],
    items: List[Item],
) -> str:
    """
    Simple HTML (no external assets). Items grouped by source, then decision (MUST_READ first).
    """

    # Sort items
    def decision_rank(d: str) -> int:
        return {"MUST_READ": 0, "SCAN": 1, "SKIP": 2}.get(d or "SCAN", 1)

    items_sorted = sorted(
        items,
        key=lambda x: (
            x.source,
            decision_rank(x.decision or "SCAN"),
            x.published_date or dt.date(1900, 1, 1),
        ),
        reverse=False,
    )

    html = f"""
<html>
<body style="font-family: Arial, Helvetica, sans-serif; color: #222;">
  <div style="background:#1f2d3d; padding: 18px;">
    <h2 style="color:#fff; margin:0;">⛏️ Weekly Mining & Processing Digest</h2>
    <div style="color:#cfd8dc; margin-top:6px;">Coverage: {start} → {end}</div>
  </div>

  <div style="padding: 18px;">
    <h3 style="margin-top:0;">Overview</h3>
    <p>This digest covers literature and news discovered between <b>{start}</b> and <b>{end}</b>.</p>

    <h3>Coverage & Sources</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 900px;">
      <tr>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Source</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Collected</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Included</th>
      </tr>
    """

    # Keep stable ordering
    all_sources = sorted(
        set(list(collected_counts.keys()) + list(included_counts.keys()))
    )
    for s in all_sources:
        html += f"""
      <tr>
        <td style="border-bottom:1px solid #eee; padding:6px;">{s}</td>
        <td style="border-bottom:1px solid #eee; padding:6px;">{collected_counts.get(s, 0)}</td>
        <td style="border-bottom:1px solid #eee; padding:6px;">{included_counts.get(s, 0)}</td>
      </tr>
        """
    html += "</table>"

    html += "<h3 style='margin-top:26px;'>Source Overviews</h3>"
    for src in all_sources:
        ov = source_overviews.get(src)
        if not ov:
            continue

        decisions = ov.get("decisions", {})
        topcats = ov.get("top_categories", [])
        topcats_str = (
            ", ".join([f"{c} ({n})" for c, n in topcats]) if topcats else "N/A"
        )

        html += f"""
        <div style="margin-top:14px; padding:10px; border:1px solid #eee; border-radius:8px;">
          <div style="font-weight:bold;">{src}</div>
          <div style="font-size:13px; color:#555; margin-top:4px;">
            Included: {ov.get("total", 0)} |
            MUST_READ: {decisions.get("MUST_READ", 0)} |
            SCAN: {decisions.get("SCAN", 0)}
          </div>
          <div style="font-size:13px; color:#555; margin-top:4px;">
            Top categories: {topcats_str}
          </div>
        </div>
        """

    html += "<h3 style='margin-top:26px;'>Detailed Items</h3>"

    current_source = None
    for it in items_sorted:
        if it.source != current_source:
            current_source = it.source
            html += f"<h4 style='margin-top:22px; border-bottom:2px solid #efefef; padding-bottom:6px;'>{current_source}</h4>"

        badge = it.decision or "SCAN"
        badge_color = {
            "MUST_READ": "#b71c1c",
            "SCAN": "#ef6c00",
            "SKIP": "#757575",
        }.get(badge, "#ef6c00")

        pub = it.published_date.isoformat() if it.published_date else "Unknown date"
        authors = it.authors or "Unknown"
        cat = it.category or "General"
        highlight = it.highlight or ""
        rationale = it.rationale or ""

        html += f"""
        <div style="margin:12px 0; padding:12px; border-left:6px solid {badge_color}; background:#fafafa; border-radius:6px;">
          <div style="font-size:12px; color:#444; margin-bottom:6px;">
            <span style="display:inline-block; padding:2px 8px; border-radius:10px; background:{badge_color}; color:#fff; font-weight:bold;">
              {badge}
            </span>
            <span style="margin-left:8px;">{pub}</span>
            <span style="margin-left:8px; color:#666;">[{cat}]</span>
          </div>

          <div style="font-size:16px; font-weight:bold; margin-bottom:6px;">
            <a href="{it.url}" style="color:#1f2d3d; text-decoration:none;">{it.title}</a>
          </div>

          <div style="font-size:13px; color:#666; margin-bottom:8px;">
            Source: {it.source_type} | Authors: {authors}
          </div>

          <div style="font-size:13px; color:#333;">
            <b>AI highlight:</b> {highlight}
          </div>
          <div style="font-size:12px; color:#555; margin-top:6px;">
            {rationale}
          </div>
        </div>
        """

    html += """
  </div>

  <div style="padding: 18px; text-align:center; font-size: 12px; color:#999;">
    Generated automatically.
  </div>
</body>
</html>
"""
    return html


# =========================
# Email
# =========================


def send_email(html: str) -> None:
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        raise ValueError(
            "EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER missing. Set them in .env or GitHub Secrets."
        )

    msg = MIMEMultipart()
    msg["Subject"] = f"{EMAIL_SUBJECT} - {TODAY}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("[EMAIL] Sent successfully.")


# =========================
# Tests (run locally)
# =========================


def test_env() -> None:
    print("\n[TEST] Environment sanity")
    assert bool(GEMINI_API_KEY), "GEMINI_API_KEY missing"
    print("[TEST] GEMINI_API_KEY OK")
    # Email vars may be optional if not testing send:
    print(f"[TEST] EMAIL_SENDER present? {bool(EMAIL_SENDER)}")
    print(f"[TEST] EMAIL_RECEIVER present? {bool(EMAIL_RECEIVER)}")


def test_openalex_one_journal(journal_name: str) -> None:
    print(f"\n[TEST] OpenAlex journal test: {journal_name}")
    sid = openalex_resolve_source_id(journal_name)
    assert sid, f"Could not resolve OpenAlex source id for {journal_name}"
    items = fetch_openalex_journal_items(
        pd.DataFrame([{"type": "journal", "identifier": journal_name}]),
        START_DATE,
        END_DATE,
    )
    print(f"[TEST] Retrieved {len(items)} items in-window for {journal_name}")
    if items:
        print("[TEST] Sample:", items[0].title, items[0].published_date, items[0].url)


def test_rss_one_feed(name: str, url: str) -> None:
    print(f"\n[TEST] RSS test: {name}")
    df = pd.DataFrame([{"name": name, "type": "rss", "identifier": url}])
    items = fetch_rss_items(df, START_DATE, END_DATE)
    print(f"[TEST] RSS kept {len(items)} in-window items")
    if items:
        print("[TEST] Sample:", items[0].title, items[0].published_date, items[0].url)


def test_onemine(org: str = "AUSIMM") -> None:
    print(f"\n[TEST] OneMine test org={org}")
    df = pd.DataFrame(
        [
            {
                "name": f"OneMine {org}",
                "type": "repository",
                "identifier": f"onemine:Organization={org}",
            }
        ]
    )
    items = fetch_onemine_repository_items(
        df, START_DATE, END_DATE, max_pages=2, page_size=20, keywords=None
    )
    print(f"[TEST] OneMine kept {len(items)} in-window items")
    if items:
        print("[TEST] Sample:", items[0].title, items[0].published_date, items[0].url)
        print("[TEST] Abstract length:", len(items[0].abstract or ""))


# =========================
# Main Pipeline
# =========================


def run_pipeline(
    keywords_csv: str = "keywords.csv",
    sources_csv: str = "sources.csv",
    do_send_email: bool = True,
) -> None:
    # Inputs
    keywords_path = os.path.join(SCRIPT_DIR, keywords_csv)
    sources_path = os.path.join(SCRIPT_DIR, sources_csv)

    if not os.path.exists(keywords_path):
        raise FileNotFoundError(f"Missing {keywords_path}")
    if not os.path.exists(sources_path):
        raise FileNotFoundError(f"Missing {sources_path}")

    keywords_context = load_keywords_csv(keywords_path)
    sources_df = load_sources_csv(sources_path)

    print("\n[PIPELINE] Step 1: Collect items (OpenAlex + RSS + Repositories)")

    collected: List[Item] = []

    # Journals via OpenAlex
    journal_items = fetch_openalex_journal_items(sources_df, START_DATE, END_DATE)
    collected.extend(journal_items)

    # RSS
    rss_items = fetch_rss_items(sources_df, START_DATE, END_DATE)
    collected.extend(rss_items)

    # Repositories (OneMine); optional: use a broad keyword string to focus
    # If you want keyword filtering at OneMine search level, build it from your topics:
    # onemine_keywords = " OR ".join(set(",".join(keywords_context.values()).split(",")))  # too broad; not recommended
    # For now, pull by date window only (best to avoid missing)
    repo_items = fetch_onemine_repository_items(
        sources_df, START_DATE, END_DATE, max_pages=3, page_size=20, keywords=None
    )
    collected.extend(repo_items)

    print(f"[PIPELINE] Collected in-window total: {len(collected)}")

    # Coverage counts (collected)
    collected_counts = Counter([i.source for i in collected])

    print("\n[PIPELINE] Step 2: AI triage (Gemini) on full abstracts (no truncation)")
    included = ai_classify_items(collected, keywords_context)

    included_counts = Counter([i.source for i in included])

    print("\n[PIPELINE] Step 3: Build deterministic source overviews")
    overviews = build_source_overviews(included)

    print("\n[PIPELINE] Step 4: Render newsletter")
    html = render_newsletter_html(
        start=START_DATE,
        end=END_DATE,
        collected_counts=dict(collected_counts),
        included_counts=dict(included_counts),
        source_overviews=overviews,
        items=included,
    )

    # Save local HTML for inspection
    out_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"digest_{TODAY}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[PIPELINE] Saved HTML to: {out_html}")

    print("\n[PIPELINE] Step 5: Send email")
    if do_send_email:
        send_email(html)
    else:
        print("[PIPELINE] Email sending disabled (do_send_email=False).")


if __name__ == "__main__":
    """
    Local usage examples (VS Code terminal):

    1) Run tests:
       python daily_digest_GPTV3.py

    2) Run pipeline without sending email:
       python daily_digest_GPTV3.py --no-email

    3) If you want to run only OneMine test quickly, comment out others or add an arg parser.
    """
    import sys

    # Basic arg flag
    no_email = "--no-email" in sys.argv

    # ===== Tests =====
    test_env()

    # Optional targeted tests (safe defaults)
    # If you want to test a journal that was missing (e.g., Minerals Engineering):
    try:
        test_openalex_one_journal("Minerals Engineering")
    except AssertionError as e:
        print(f"[TEST] OpenAlex journal test warning: {e}")

    # Example RSS test (only if you have at least one RSS source in sources.csv)
    # Uncomment and set an RSS url if needed:
    # test_rss_one_feed("Mining.com", "https://www.mining.com/feed/")

    # OneMine test (AUSIMM)
    test_onemine("AUSIMM")

    # ===== Run full pipeline =====
    run_pipeline(do_send_email=not no_email)
