import os
import re
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from urllib.parse import urlparse, parse_qsl

import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup

import google.generativeai as genai

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from dotenv import load_dotenv


# =========================
# Utilities
# =========================


def utc_today_date() -> dt.date:
    # timezone-aware UTC date (avoid deprecation warnings)
    return dt.datetime.now(dt.timezone.utc).date()


def parse_date_flexible(s: str) -> Optional[dt.date]:
    if not s:
        return None
    s = s.strip()

    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def in_window(d: Optional[dt.date], start: dt.date, end: dt.date) -> bool:
    if d is None:
        return False
    return start <= d <= end


def safe_get_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def normalize_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# =========================
# Data model
# =========================


@dataclass
class Item:
    title: str
    url: str
    source: str
    source_type: str  # "Academic" | "News" | "Repository"
    published_date: Optional[dt.date]
    authors: Optional[str] = None
    abstract: Optional[str] = None

    decision: Optional[str] = None
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

# Repositories always scan +30 days
REPO_EXTRA_DAYS = 30

TODAY = utc_today_date()
START_DATE = TODAY - dt.timedelta(days=LOOKBACK_DAYS)
END_DATE = TODAY
REPO_START_DATE = START_DATE - dt.timedelta(days=REPO_EXTRA_DAYS)

print(
    f"[WINDOW] Coverage (Journals/RSS): {START_DATE} -> {END_DATE} (lookback_days={LOOKBACK_DAYS})"
)
print(
    f"[WINDOW] Coverage (Repositories): {REPO_START_DATE} -> {END_DATE} (lookback_days+{REPO_EXTRA_DAYS})"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Set it in .env or GitHub Secrets.")

genai.configure(api_key=GEMINI_API_KEY)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MiningDigestBot/1.0)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# =========================
# Gemini: auto-discovery + pacing
# =========================

# Default pacing: 2 seconds between Gemini calls (prevents 429s)
GEMINI_MIN_INTERVAL_SECONDS = float(config.get("gemini_min_interval_seconds", 2.0))
# If any error happens (non-429), wait this long before retrying
GEMINI_ERROR_SLEEP_SECONDS = float(config.get("gemini_error_sleep_seconds", 10.0))

# Optional override (if you ever want to force a model temporarily)
GEMINI_MODEL_OVERRIDE = (os.environ.get("GEMINI_MODEL_OVERRIDE") or "").strip() or str(
    config.get("gemini_model_override", "") or ""
).strip()

_LAST_GEMINI_CALL_TS = 0.0
_GEMINI_MODEL_OBJS: Dict[str, genai.GenerativeModel] = {}
_LAST_PRINTED_MODEL: Optional[str] = None


def _strip_models_prefix(name: str) -> str:
    name = (name or "").strip()
    return name.replace("models/", "", 1) if name.startswith("models/") else name


def list_available_gemini_models() -> List[str]:
    """
    Returns model names (without 'models/' prefix) that:
    - contain 'gemini'
    - support generateContent
    """
    candidates = []
    try:
        for m in genai.list_models():
            mname = _strip_models_prefix(getattr(m, "name", "") or "")
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" not in methods:
                continue
            if "gemini" not in mname.lower():
                continue
            candidates.append(mname)
    except Exception as e:
        print(
            f"[GEMINI] list_models() failed (non-fatal). Will fall back to common defaults. Error: {e}"
        )
    return candidates


def _parse_version(model_name: str) -> Tuple[int, int]:
    """
    Extracts (major, minor) from things like:
    gemini-3.0-..., gemini-2.0-..., gemini-1.5-...
    Unknown -> (0, 0)
    """
    s = model_name.lower()
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", s)
    if not m:
        return (0, 0)
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return (major, minor)


def _model_rank_key(model_name: str) -> Tuple[int, int, int, int, str]:
    """
    Higher is better (we'll sort descending):
    - version major/minor
    - prefer 'pro' over 'flash' (capability)
    - prefer non-vision? (neutral)
    - prefer 'latest' or 'stable' keywords slightly (optional)
    """
    name = model_name.lower()
    major, minor = _parse_version(name)

    pro_rank = 2 if "-pro" in name else (1 if "-flash" in name else 0)
    stability_rank = 2 if "latest" in name else (1 if "stable" in name else 0)
    return (major, minor, pro_rank, stability_rank, name)


def choose_best_models() -> List[str]:
    """
    Returns a sorted list of candidate models, newest-first.
    """
    models = list_available_gemini_models()

    # If list_models fails or returns nothing, fall back to sane defaults
    if not models:
        models = [
            "gemini-2.0-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

    # De-dup while preserving sort later
    models = list(dict.fromkeys(models))

    models_sorted = sorted(models, key=_model_rank_key, reverse=True)

    # Apply override (if provided) by putting it first
    if GEMINI_MODEL_OVERRIDE:
        override = _strip_models_prefix(GEMINI_MODEL_OVERRIDE)
        if override in models_sorted:
            models_sorted.remove(override)
        models_sorted.insert(0, override)

    # Show top few for debugging
    print("[GEMINI] Candidate models (newest-first):")
    for m in models_sorted[:10]:
        print(f"  - {m}")

    return models_sorted


def get_gemini_model_obj(model_name: str) -> genai.GenerativeModel:
    model_name = _strip_models_prefix(model_name)
    if model_name not in _GEMINI_MODEL_OBJS:
        _GEMINI_MODEL_OBJS[model_name] = genai.GenerativeModel(model_name)
    return _GEMINI_MODEL_OBJS[model_name]


def log_gemini_model(model_name: str) -> None:
    global _LAST_PRINTED_MODEL
    if _LAST_PRINTED_MODEL != model_name:
        print(f"[GEMINI] Using model: {model_name}")
        _LAST_PRINTED_MODEL = model_name


def enforce_min_interval() -> None:
    global _LAST_GEMINI_CALL_TS
    now = time.time()
    wait = (_LAST_GEMINI_CALL_TS + GEMINI_MIN_INTERVAL_SECONDS) - now
    if wait > 0:
        time.sleep(wait)
    _LAST_GEMINI_CALL_TS = time.time()


def ai_rate_limit_sleep_from_error(e: Exception, base: int = 60) -> None:
    """
    If Gemini returns a retry_delay hint, honor it; else sleep base seconds.
    """
    msg = str(e)
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
    if m:
        delay = max(base, int(m.group(1)))
    else:
        delay = base
    print(f"[AI] Rate limit/transient: sleeping {delay}s ...")
    time.sleep(delay)


def parse_json_strict(text: str) -> Optional[dict]:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# Build model candidates at startup
GEMINI_MODELS_TRY = choose_best_models()


def gemini_generate_with_retry(prompt: str, max_attempts_per_model: int = 3) -> str:
    """
    Uses newest-available model list from list_models(), tries in order.
    Adds:
    - enforced pacing between calls (min interval)
    - model fallback if 404/unsupported or persistent 429
    """
    last_err = None

    for model_name in GEMINI_MODELS_TRY:
        model_obj = get_gemini_model_obj(model_name)

        for attempt in range(1, max_attempts_per_model + 1):
            try:
                enforce_min_interval()
                log_gemini_model(model_name)
                resp = model_obj.generate_content(prompt)
                txt = (getattr(resp, "text", "") or "").strip()
                if not txt:
                    raise RuntimeError("Empty Gemini response.")
                return txt

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                print(
                    f"[AI] Error model={model_name} attempt {attempt}/{max_attempts_per_model}: {e}"
                )

                # If model truly not available/unsupported -> try next model immediately
                if (
                    "404" in msg and ("not found" in msg or "not supported" in msg)
                ) or "unsupported" in msg:
                    break

                # Rate limit -> back off, then retry; if still failing after attempts, fall back to next model
                if (
                    "429" in msg
                    or "resource exhausted" in msg
                    or "quota" in msg
                    or "rate" in msg
                ):
                    ai_rate_limit_sleep_from_error(e, base=60)
                else:
                    # Generic errors: small wait, then retry
                    time.sleep(GEMINI_ERROR_SLEEP_SECONDS)

        # next model

    raise RuntimeError(
        f"Gemini failed for all candidate models. Last error: {last_err}"
    )


# =========================
# Load Inputs
# =========================


def load_keywords_csv(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    if "topic" not in df.columns or "keywords" not in df.columns:
        raise ValueError("keywords.csv must have columns: topic, keywords")
    return dict(zip(df["topic"].astype(str), df["keywords"].astype(str)))


def load_sources_csv(path: str) -> pd.DataFrame:
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
    Fixes:
    - Commas in journal names: uses `search=` instead of filter=display_name.search:"..."
    - Sloppy first-hit mapping: chooses best match (prevents Minerals Engineering -> Minerals)
    """
    q = (journal_name or "").strip()
    if not q:
        return None
    if q in OPENALEX_SOURCE_CACHE:
        return OPENALEX_SOURCE_CACHE[q]

    url = "https://api.openalex.org/sources"
    params = {"search": q, "per-page": 25, "select": "id,display_name,issn_l,issn"}
    r = SESSION.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print(f"[OpenAlex][ID] Failed for '{q}' status={r.status_code}")
        return None

    results = r.json().get("results", []) or []
    if not results:
        print(f"[OpenAlex][ID] No match for '{q}'")
        return None

    qn = normalize_name(q)

    def score(res: dict) -> float:
        name = (res.get("display_name") or "").strip()
        nn = normalize_name(name)
        if not nn:
            return 1e9
        if nn == qn:
            return 0.0

        length_penalty = 0.0
        if len(nn) < max(5, int(0.7 * len(qn))):
            length_penalty += 2.0

        q_tokens = set(qn.split())
        n_tokens = set(nn.split())
        inter = len(q_tokens & n_tokens)
        union = max(1, len(q_tokens | n_tokens))
        jaccard = inter / union
        sim_penalty = (1.0 - jaccard) * 3.0

        prefix_bonus = 0.0
        if nn.startswith(qn) or qn.startswith(nn):
            prefix_bonus -= 0.5
        if qn in nn:
            prefix_bonus -= 0.3

        return 1.0 + sim_penalty + length_penalty + prefix_bonus

    best = min(results, key=score)
    full_id = (best.get("id") or "").strip()
    short_id = full_id.replace("https://openalex.org/", "").strip()
    if not short_id:
        return None

    OPENALEX_SOURCE_CACHE[q] = short_id
    print(f"[OpenAlex][ID] {q} -> {short_id}")
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
            "select": "id,doi,title,publication_date,abstract_inverted_index,authorships",
        }

        try:
            r = SESSION.get(base_url, params=params, timeout=45)
            if r.status_code != 200:
                print(f"[OpenAlex] {jname}: HTTP {r.status_code}")
                continue
            results = (r.json() or {}).get("results", []) or []
            print(f"[OpenAlex] {jname}: fetched {len(results)} candidates")

            for w in results:
                pub = parse_date_flexible(w.get("publication_date", ""))
                if not in_window(pub, start, end):
                    continue

                title = normalize_ws(w.get("title", ""))
                doi = w.get("doi")
                wid = w.get("id")
                url = doi if doi else wid or ""

                auths = []
                for a in w.get("authorships") or []:
                    name = ((a.get("author") or {}).get("display_name") or "").strip()
                    if name:
                        auths.append(name)
                authors = ", ".join(auths) if auths else None

                abstract = normalize_ws(
                    openalex_reconstruct_abstract(w.get("abstract_inverted_index"))
                )

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

            time.sleep(0.15)
        except Exception as e:
            print(f"[OpenAlex] {jname}: error {e}")

    print(f"[OpenAlex] Total in-window items: {len(out)}")
    return out


# =========================
# RSS
# =========================


def parse_rss_entry_date(entry) -> Optional[dt.date]:
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
            entries = feed.entries[: MAX_ITEMS_PER_FEED * 6]
            kept = 0

            for e in entries:
                pub = parse_rss_entry_date(e)
                if not in_window(pub, start, end):
                    continue

                title = normalize_ws(getattr(e, "title", "") or "")
                link = getattr(e, "link", "") or ""

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
# OneMine Repository Scraper
# =========================


def onemine_parse_identifier(ident: str) -> Dict[str, str]:
    raw = (ident or "").strip()
    if not raw.lower().startswith("onemine:"):
        return {}
    rest = raw.split(":", 1)[1].strip()
    if not rest:
        return {}
    if rest.lower().startswith("http"):
        u = urlparse(rest)
        return dict(parse_qsl(u.query, keep_blank_values=True))
    params = dict(parse_qsl(rest, keep_blank_values=True))
    if not params and "=" in rest:
        k, v = rest.split("=", 1)
        params = {k.strip(): v.strip()}
    return params


def onemine_build_params(
    base_params: Dict[str, str], page: int, page_size: int
) -> Dict[str, str]:
    params = dict(base_params or {})
    params["SortBy"] = params.get("SortBy", "MostRecent")
    params["page"] = str(page)
    params["pageSize"] = str(page_size)
    return params


def onemine_parse_list_page(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for li in soup.select("ul.item-list li.item-list__item"):
        a_title = li.select_one("a.item-list__title")
        if not a_title:
            continue

        title = safe_get_text(a_title)
        href = (a_title.get("href") or "").strip()
        url = ("https://www.onemine.org" + href) if href.startswith("/") else href

        date_ps = li.select("p.item-list__date")
        author_text = safe_get_text(date_ps[0]) if len(date_ps) >= 1 else ""
        pub_text = safe_get_text(date_ps[-1]) if len(date_ps) >= 2 else ""

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
    try:
        r = SESSION.get(doc_url, timeout=45)
        if r.status_code != 200:
            return None, None
        soup = BeautifulSoup(r.text, "html.parser")

        meta_desc = soup.select_one('meta[name="description"]')
        meta_abstract = meta_desc.get("content", "").strip() if meta_desc else ""

        main = soup.select_one("div.content[role='main']") or soup
        paragraphs = [
            normalize_ws(p.get_text(" ", strip=True)) for p in main.select("p")
        ]
        paragraphs = [p for p in paragraphs if len(p) > 80]

        abstract = ""
        if paragraphs:
            paragraphs.sort(key=len, reverse=True)
            abstract = " ".join(paragraphs[:3]).strip()
        elif meta_abstract:
            abstract = meta_abstract

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
    max_pages: int = 6,
    page_size: int = 20,
) -> List[Item]:
    out: List[Item] = []
    repo_df = sources_df[sources_df["type"] == "repository"]
    if repo_df.empty:
        print("[Repo] No repository sources configured.")
        return out

    onemine_rows = repo_df[repo_df["identifier"].str.lower().str.startswith("onemine:")]
    if onemine_rows.empty:
        print(
            "[Repo] No OneMine repository entries found (identifier must start with 'onemine:')."
        )
        return out

    print(f"[Repo][OneMine] Fetching OneMine for {start} -> {end} ...")

    base_url = "https://www.onemine.org/search"

    for _, row in onemine_rows.iterrows():
        name = row["name"].strip()
        ident = row["identifier"].strip()
        base_params = onemine_parse_identifier(ident)

        # defensive cleanup if someone accidentally nested "onemine:" into keywords
        if "keywords" in base_params and str(
            base_params["keywords"]
        ).lower().startswith("onemine:"):
            base_params["keywords"] = str(base_params["keywords"])[7:]

        candidates: List[dict] = []

        for page in range(1, max_pages + 1):
            params = onemine_build_params(base_params, page=page, page_size=page_size)
            print(
                f"[Repo][OneMine] GET {name}: page={page} url={base_url} params={params}"
            )

            r = SESSION.get(base_url, params=params, timeout=45)
            if r.status_code != 200:
                print(f"[Repo][OneMine] {name}: HTTP {r.status_code} on page {page}")
                break

            parsed = onemine_parse_list_page(r.text)
            print(
                f"[Repo][OneMine] {name}: parsed {len(parsed)} list items on page {page}"
            )
            if not parsed:
                break

            page_dates = [parse_date_flexible(p.get("date_text", "")) for p in parsed]
            page_dates = [d for d in page_dates if d]

            if page_dates:
                page_newest = max(page_dates)
                page_oldest = min(page_dates)

                if page_newest < start and page == 1:
                    print(
                        f"[Repo][OneMine][NOTE] {name}: newest item seen is {page_newest} < START_DATE={start}. Expect 0 in-window results."
                    )
                    break

                candidates.extend(parsed)

                # early-stop once we've passed start
                if page_oldest < start:
                    break
            else:
                candidates.extend(parsed)

            time.sleep(0.25)

        kept = 0
        for c in candidates:
            pub = parse_date_flexible(c.get("date_text", ""))
            if not in_window(pub, start, end):
                continue

            authors = c.get("author_text", "")
            if authors.lower().startswith("by "):
                authors = authors[3:].strip()

            doc_auth, abstract = onemine_fetch_document_abstract(c["url"])
            if doc_auth and len(doc_auth) > len(authors or ""):
                authors = (
                    doc_auth[3:].strip()
                    if doc_auth.lower().startswith("by ")
                    else doc_auth
                )
            if not abstract:
                abstract = c.get("snippet", "")

            out.append(
                Item(
                    title=c["title"],
                    url=c["url"],
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
# Dedup + CSV outputs
# =========================


def dedupe_items(items: List[Item]) -> List[Item]:
    seen = set()
    out = []
    for it in items:
        u = (it.url or "").strip().lower()
        key = u if u else (normalize_name(it.title), str(it.published_date or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def write_raw_csv(path: str, items: List[Item]) -> None:
    rows = []
    for it in items:
        rows.append(
            {
                "source_type": it.source_type,
                "source": it.source,
                "published_date": it.published_date.isoformat()
                if it.published_date
                else "",
                "title": it.title,
                "authors": it.authors or "",
                "url": it.url,
                "abstract": it.abstract or "",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def write_triaged_csv(path: str, items: List[Item]) -> None:
    rows = []
    for it in items:
        rows.append(
            {
                "source_type": it.source_type,
                "source": it.source,
                "published_date": it.published_date.isoformat()
                if it.published_date
                else "",
                "decision": it.decision or "",
                "category": it.category or "",
                "title": it.title,
                "authors": it.authors or "",
                "url": it.url,
                "highlight": it.highlight or "",
                "rationale": it.rationale or "",
                "abstract": it.abstract or "",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


# =========================
# AI
# =========================

AI_TRIAGE_SCHEMA = """
Return ONLY valid JSON in this exact shape:
{
  "items": [
    {
      "id": <int>,
      "decision": "MUST_READ" | "GOOD_TO_READ" | "NOT_SURE",
      "category": "short label",
      "highlight": "one sentence highlight",
      "rationale": "one short sentence why"
    }
  ]
}
""".strip()


def ai_classify_items_batched(
    items: List[Item], keywords_context: Dict[str, str], batch_size: int = 6
) -> List[Item]:
    if not items:
        return []

    interests = "; ".join([f"{k}: {v}" for k, v in keywords_context.items()])
    print(
        f"[AI] Classifying {len(items)} Academic/Repository items with Gemini (batch_size={batch_size}) ..."
    )

    id_to_item: Dict[int, Item] = {}
    packed: List[dict] = []
    for i, it in enumerate(items, start=1):
        id_to_item[i] = it
        packed.append(
            {
                "id": i,
                "title": it.title,
                "source": it.source,
                "published_date": it.published_date.isoformat()
                if it.published_date
                else "",
                "authors": it.authors or "",
                "abstract": normalize_ws(it.abstract or "")
                or "(No abstract/description available.)",
                "url": it.url,
            }
        )

    rated: List[Item] = []

    for batch_start in range(0, len(packed), batch_size):
        batch = packed[batch_start : batch_start + batch_size]

        prompt = f"""
You are an expert research triage assistant for Mining, Mineral Processing, Extractive Metallurgy, and Automation/Control.

My interests (topics -> keywords):
{interests}

Task:
For each item below, classify relevance for my review.

Decision rules:
- MUST_READ: highly relevant or impactful.
- GOOD_TO_READ: relevant enough to read/skim.
- NOT_SURE: uncertain relevance; keep as "needs manual check".

Return decisions for ALL items you are given (do not skip any).

Items (JSON):
{json.dumps(batch, ensure_ascii=False)}

{AI_TRIAGE_SCHEMA}
""".strip()

        text = gemini_generate_with_retry(prompt, max_attempts_per_model=3)
        data = parse_json_strict(text)

        if not data or "items" not in data:
            print(
                "[AI] Bad JSON response for batch. Marking all in batch as NOT_SURE (fail-open)."
            )
            for b in batch:
                it = id_to_item[b["id"]]
                it.decision = "NOT_SURE"
                it.category = "Unparsed"
                it.highlight = (
                    "AI returned an unparseable response; included for manual review."
                )
                it.rationale = "Fallback inclusion to avoid false negatives."
                rated.append(it)
            continue

        for r in data.get("items", []):
            try:
                rid = int(r.get("id"))
            except Exception:
                continue
            it = id_to_item.get(rid)
            if not it:
                continue

            decision = (r.get("decision") or "").strip().upper()
            if decision not in {"MUST_READ", "GOOD_TO_READ", "NOT_SURE"}:
                decision = "NOT_SURE"

            it.decision = decision
            it.category = normalize_ws(str(r.get("category") or "General"))
            it.highlight = normalize_ws(str(r.get("highlight") or ""))
            it.rationale = normalize_ws(str(r.get("rationale") or ""))

            rated.append(it)

        # Extra gentle spacing on top of the global pacing (keeps you well under tight quotas)
        time.sleep(0.5)

    print(f"[AI] Classified {len(rated)} Academic/Repository items.")
    return rated


TREND_SCHEMA = """
Return ONLY valid JSON:
{
  "trends": {
    "<source_name>": "one short paragraph about the week's themes/trends for this source"
  }
}
""".strip()


def ai_trends_one_call(
    items: List[Item], max_titles_per_source: int = 12
) -> Dict[str, str]:
    by_source = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)

    payload = {}
    for src, arr in by_source.items():

        def rank(d: str) -> int:
            return {"MUST_READ": 0, "GOOD_TO_READ": 1, "NOT_SURE": 2}.get(
                d or "NOT_SURE", 2
            )

        arr_sorted = sorted(
            arr,
            key=lambda x: (rank(x.decision), x.published_date or dt.date(1900, 1, 1)),
        )
        cats = Counter([a.category or "General" for a in arr_sorted]).most_common(6)

        payload[src] = {
            "n_items": len(arr_sorted),
            "decision_counts": dict(
                Counter([a.decision or "NOT_SURE" for a in arr_sorted])
            ),
            "top_categories": cats,
            "top_titles": [a.title for a in arr_sorted[:max_titles_per_source]],
        }

    prompt = f"""
You are an expert mining/mineral processing research analyst.

Task:
For each source in the JSON input, write ONE short paragraph describing the dominant themes/trends across the listed titles/categories.
Do NOT invent details not supported by the titles/categories. If a source has mixed topics, say so.

Input (JSON):
{json.dumps(payload, ensure_ascii=False)}

{TREND_SCHEMA}
""".strip()

    try:
        text = gemini_generate_with_retry(prompt, max_attempts_per_model=3)
        data = parse_json_strict(text) or {}
        trends = (data.get("trends") or {}) if isinstance(data, dict) else {}
        out = {}
        for k, v in (trends or {}).items():
            out[str(k)] = normalize_ws(str(v))
        return out
    except Exception as e:
        print(f"[AI] Trend summary failed (non-fatal): {e}")
        return {}


# =========================
# Deterministic source overviews
# =========================


def build_source_overviews(items: List[Item]) -> Dict[str, dict]:
    by_source = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)

    out = {}
    for src, arr in by_source.items():
        decisions = Counter([a.decision or "NOT_SURE" for a in arr])
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
    repo_start: dt.date,
    collected_counts: Dict[str, int],
    included_counts: Dict[str, int],
    source_overviews: Dict[str, dict],
    source_trends: Dict[str, str],
    items: List[Item],
) -> str:
    def decision_rank(d: str) -> int:
        return {"MUST_READ": 0, "GOOD_TO_READ": 1, "NOT_SURE": 2}.get(
            d or "NOT_SURE", 2
        )

    items_sorted = sorted(
        items,
        key=lambda x: (
            x.source,
            decision_rank(x.decision or "NOT_SURE"),
            x.published_date or dt.date(1900, 1, 1),
        ),
        reverse=False,
    )

    html = f"""
<html>
<body style="font-family: Arial, Helvetica, sans-serif; color: #222;">
  <div style="background:#1f2d3d; padding: 18px;">
    <h2 style="color:#fff; margin:0;">⛏️ Weekly Mining & Processing Digest</h2>
    <div style="color:#cfd8dc; margin-top:6px;">
      Coverage (Journals/RSS): {start} → {end}<br/>
      Coverage (Repositories): {repo_start} → {end}
    </div>
  </div>

  <div style="padding: 18px;">
    <h3 style="margin-top:0;">Overview</h3>
    <p>
      This digest covers journals/RSS between <b>{start}</b> and <b>{end}</b>.
      Repository sources are scanned over an extended window back to <b>{repo_start}</b> to avoid missing infrequent updates.
    </p>

    <h3>Coverage & Sources</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 900px;">
      <tr>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Source</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Collected</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Included</th>
      </tr>
    """

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
        trend = source_trends.get(src, "")

        html += f"""
        <div style="margin-top:14px; padding:10px; border:1px solid #eee; border-radius:8px;">
          <div style="font-weight:bold;">{src}</div>
          <div style="font-size:13px; color:#555; margin-top:4px;">
            Included: {ov.get("total", 0)} |
            MUST_READ: {decisions.get("MUST_READ", 0)} |
            GOOD_TO_READ: {decisions.get("GOOD_TO_READ", 0)} |
            NOT_SURE: {decisions.get("NOT_SURE", 0)}
          </div>
          <div style="font-size:13px; color:#555; margin-top:4px;">
            Top categories: {topcats_str}
          </div>
          {f'<div style="font-size:13px; color:#333; margin-top:8px;"><b>Trend:</b> {trend}</div>' if trend else ""}
        </div>
        """

    html += "<h3 style='margin-top:26px;'>Detailed Items</h3>"

    current_source = None
    for it in items_sorted:
        if it.source != current_source:
            current_source = it.source
            html += f"<h4 style='margin-top:22px; border-bottom:2px solid #efefef; padding-bottom:6px;'>{current_source}</h4>"

        badge = it.decision or "NOT_SURE"
        badge_color = {
            "MUST_READ": "#b71c1c",
            "GOOD_TO_READ": "#ef6c00",
            "NOT_SURE": "#757575",
        }.get(badge, "#757575")

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


def send_email(html: str, attachments: List[str]) -> None:
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        raise ValueError(
            "EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER missing. Set them in .env or GitHub Secrets."
        )

    msg = MIMEMultipart()
    msg["Subject"] = f"{EMAIL_SUBJECT} - {TODAY}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html, "html"))

    for path in attachments:
        if not path or not os.path.exists(path):
            continue
        part = MIMEBase("application", "octet-stream")
        with open(path, "rb") as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition", f'attachment; filename="{os.path.basename(path)}"'
        )
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("[EMAIL] Sent successfully (with attachments).")


# =========================
# Main Pipeline
# =========================


def run_pipeline(
    keywords_csv: str = "keywords.csv",
    sources_csv: str = "sources.csv",
    do_send_email: bool = True,
    batch_size: int = 6,
) -> None:
    keywords_path = os.path.join(SCRIPT_DIR, keywords_csv)
    sources_path = os.path.join(SCRIPT_DIR, sources_csv)

    if not os.path.exists(keywords_path):
        raise FileNotFoundError(f"Missing {keywords_path}")
    if not os.path.exists(sources_path):
        raise FileNotFoundError(f"Missing {sources_path}")

    keywords_context = load_keywords_csv(keywords_path)
    sources_df = load_sources_csv(sources_path)

    print("\n[PIPELINE] Step 1: Collect items (OpenAlex + RSS + OneMine)")

    collected: List[Item] = []
    collected.extend(fetch_openalex_journal_items(sources_df, START_DATE, END_DATE))
    collected.extend(fetch_rss_items(sources_df, START_DATE, END_DATE))
    collected.extend(
        fetch_onemine_repository_items(
            sources_df, REPO_START_DATE, END_DATE, max_pages=6, page_size=20
        )
    )

    collected = dedupe_items(collected)
    print(f"[PIPELINE] Collected total (deduped): {len(collected)}")

    # Raw CSV for Academic + Repository
    raw_items = [i for i in collected if i.source_type in {"Academic", "Repository"}]
    out_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    raw_csv = os.path.join(out_dir, f"raw_academic_repository_{TODAY}.csv")
    write_raw_csv(raw_csv, raw_items)
    print(f"[PIPELINE] Wrote raw CSV: {raw_csv}")

    print(
        "\n[PIPELINE] Step 2: AI triage ONLY for Academic + Repository (token optimisation)"
    )
    triaged = ai_classify_items_batched(
        raw_items, keywords_context, batch_size=batch_size
    )

    triaged_csv = os.path.join(out_dir, f"triaged_academic_repository_{TODAY}.csv")
    write_triaged_csv(triaged_csv, triaged)
    print(f"[PIPELINE] Wrote triaged CSV: {triaged_csv}")

    print("\n[PIPELINE] Step 3: Build deterministic source overviews")
    overviews = build_source_overviews(triaged)

    print("\n[PIPELINE] Step 3b: AI trend paragraph per source (single-call, capped)")
    trends = ai_trends_one_call(triaged, max_titles_per_source=12)

    print("\n[PIPELINE] Step 4: Render newsletter")
    collected_counts = Counter([i.source for i in collected])
    included_counts = Counter([i.source for i in triaged])

    html = render_newsletter_html(
        start=START_DATE,
        end=END_DATE,
        repo_start=REPO_START_DATE,
        collected_counts=dict(collected_counts),
        included_counts=dict(included_counts),
        source_overviews=overviews,
        source_trends=trends,
        items=triaged,
    )

    out_html = os.path.join(out_dir, f"digest_{TODAY}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[PIPELINE] Saved HTML to: {out_html}")

    print("\n[PIPELINE] Step 5: Send email (with CSV attachments)")
    if do_send_email:
        send_email(html, attachments=[raw_csv, triaged_csv])
    else:
        print("[PIPELINE] Email sending disabled (do_send_email=False).")


if __name__ == "__main__":
    import sys

    no_email = "--no-email" in sys.argv

    batch = 6
    for arg in sys.argv:
        if arg.startswith("--batch="):
            try:
                batch = int(arg.split("=", 1)[1])
            except Exception:
                pass

    run_pipeline(do_send_email=not no_email, batch_size=batch)
