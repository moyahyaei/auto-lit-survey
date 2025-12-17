# coding: utf-8
"""
Auto Literature Survey Digest

Collects:
- Academic items via OpenAlex
- News items via RSS (optional)
- Repository items via OneMine search

Outputs:
- output/raw_academic_repository_<YYYY-MM-DD>.csv
- output/triaged_academic_repository_<YYYY-MM-DD>.csv
- output/digest_<YYYY-MM-DD>.html
Email:
- Sends HTML digest + attaches HTML + CSVs.

Key improvements incorporated:
- OpenAlex source resolution uses `search=` (handles commas) and improved matching to avoid wrong journal IDs.
- Gemini model selection is dynamic via `genai.list_models()` (no fixed model name).
- AI triage runs only for Academic + Repository (token optimisation).
- Batched triage + explicit pacing + retry/backoff for 429 rate limits.
- OneMine repository window is always extended by +30 days relative to config lookback.
- Uses timezone-aware UTC dates (avoids utcnow deprecation).
"""

import os
import re
import json
import time
import hashlib
import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlparse

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
    # timezone-aware UTC for forward compatibility
    return dt.datetime.now(dt.timezone.utc).date()


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def safe_get_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def parse_date_flexible(s: str) -> Optional[dt.date]:
    if not s:
        return None
    s = str(s).strip()

    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    try:
        ss = s.replace("Z", "+00:00")
        dtx = dt.datetime.fromisoformat(ss)
        return dtx.date()
    except Exception:
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


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


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
    decision: Optional[str] = None  # MUST_READ | GOOD_TO_READ | NOT_SURE
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

# Config (optional)
# Prefer config.yaml (supports comments). We also support config.json for backward compatibility.

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


def _simple_yaml_load(text: str) -> Dict[str, object]:
    """Very small YAML subset parser: key: value pairs, # comments, and simple scalars.
    Used only if PyYAML isn't installed.
    """
    out: Dict[str, object] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # strip inline comments (best-effort)
        if "#" in line:
            before = line.split("#", 1)[0].rstrip()
            if before:
                line = before
            else:
                continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = v.strip()
        if not key:
            continue

        # remove quotes
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            out[key] = val[1:-1]
            continue

        low = val.lower()
        if low in ("true", "false"):
            out[key] = low == "true"
            continue
        if low in ("null", "none", "~", ""):
            out[key] = None
            continue

        # numbers
        try:
            if re.fullmatch(r"[-+]?\d+", val):
                out[key] = int(val)
                continue
            if re.fullmatch(r"[-+]?\d*\.\d+", val):
                out[key] = float(val)
                continue
        except Exception:
            pass

        out[key] = val
    return out


def load_config(script_dir: Path) -> Tuple[Dict[str, object], Optional[Path]]:
    script_dir = Path(script_dir)  # allow caller to pass str or Path
    candidates = [
        script_dir / "config.yaml",
        script_dir / "config.yml",
        script_dir / "config.json",
        script_dir / "config.jason",  # common typo
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix in (".yaml", ".yml"):
                raw = p.read_text(encoding="utf-8")
                if yaml is not None:
                    cfg = yaml.safe_load(raw) or {}
                    if isinstance(cfg, dict):
                        return cfg, p
                    return {}, p
                return _simple_yaml_load(raw), p
            else:
                with p.open("r", encoding="utf-8") as f:
                    cfg = json.load(f) or {}
                return cfg, p
        except Exception as e:
            print(f"[CONFIG] Failed to load {p}: {e}")
            return {}, p
    return {}, None


config, CONFIG_PATH_USED = load_config(SCRIPT_DIR)
if CONFIG_PATH_USED:
    print(f"[CONFIG] Loaded {CONFIG_PATH_USED.name} from: {CONFIG_PATH_USED}")
else:
    print(
        f"[CONFIG] No config.yaml/config.json found in: {SCRIPT_DIR} (using defaults)"
    )

LOOKBACK_DAYS = int(config.get("lookback_days", 7))
MAX_ITEMS_PER_FEED = int(config.get("max_items_per_feed", 5))
EMAIL_SUBJECT = str(config.get("email_subject", "Weekly Mining & Processing Digest"))

AI_BATCH_SIZE = int(config.get("ai_batch_size", 6))
AI_SLEEP_SECONDS = float(config.get("ai_sleep_seconds", 2.0))
AI_MIN_INTERVAL_SECONDS = float(
    config.get(
        "ai_min_interval_seconds", config.get("gemini_min_interval_seconds", 2.0)
    )
)
AI_ERROR_SLEEP_SECONDS = float(
    config.get("ai_error_sleep_seconds", config.get("gemini_error_sleep_seconds", 10.0))
)
REPOSITORY_EXTRA_DAYS = int(config.get("repository_extra_days", 30))

# Simple global throttling to reduce 429s (rate limits)
_AI_LAST_CALL_MONO = 0.0


def ai_throttle():
    """Ensure at least AI_MIN_INTERVAL_SECONDS between Gemini requests."""
    global _AI_LAST_CALL_MONO
    if AI_MIN_INTERVAL_SECONDS <= 0:
        return
    now = time.monotonic()
    elapsed = now - _AI_LAST_CALL_MONO
    if elapsed < AI_MIN_INTERVAL_SECONDS:
        time.sleep(AI_MIN_INTERVAL_SECONDS - elapsed)
    _AI_LAST_CALL_MONO = time.monotonic()


AI_ABSTRACT_MAX_CHARS = int(config.get("ai_abstract_max_chars", 6000))
TREND_ITEMS_CAP = int(config.get("trend_items_cap", 12))

TODAY = utc_today_date()
START_DATE = TODAY - dt.timedelta(days=LOOKBACK_DAYS)
END_DATE = TODAY

REPO_START_DATE = TODAY - dt.timedelta(days=LOOKBACK_DAYS + REPOSITORY_EXTRA_DAYS)
REPO_END_DATE = END_DATE

print(
    f"[WINDOW] Academic/News coverage: {START_DATE} -> {END_DATE} (lookback_days={LOOKBACK_DAYS})"
)
print(
    f"[WINDOW] Repository coverage (OneMine): {REPO_START_DATE} -> {REPO_END_DATE} (lookback_days+repository_extra_days={REPOSITORY_EXTRA_DAYS})"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Set it in .env or GitHub Secrets.")

genai.configure(api_key=GEMINI_API_KEY)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MiningDigestBot/1.1)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# =========================
# Gemini model discovery (robust)
# =========================


def _strip_models_prefix(name: str) -> str:
    return name.replace("models/", "").strip()


def _parse_version_tokens(name: str) -> Tuple[int, int]:
    m = re.search(r"\bgemini-(\d+)(?:\.(\d+))?\b", name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


def _is_text_generation_model(name: str) -> bool:
    n = name.lower()
    if not n.startswith("models/gemini"):
        return False
    blocked = [
        "image",
        "tts",
        "robotics",
        "computer-use",
        "deep-research",
        "exp-image-generation",
    ]
    return not any(b in n for b in blocked)


def _rank_model(name: str, prefer: str) -> tuple:
    n = name.lower()
    major, minor = _parse_version_tokens(n)

    tag = 0
    if "latest" in n:
        tag = 60
    elif "preview" in n:
        tag = 30
    elif "-exp" in n or "gemini-exp" in n:
        tag = 20
    elif "lite" in n:
        tag = 10

    if prefer == "pro":
        fam = 40 if "pro" in n else (20 if "flash" in n else 0)
    else:
        fam = 40 if "flash" in n else (20 if "pro" in n else 0)

    rev = 5 if re.search(r"-\d{3}\b", n) else 0
    return (tag, major, minor, fam, rev)


def select_best_model(prefer: str = "flash") -> str:
    candidates = []
    for m in genai.list_models():
        if "generateContent" not in getattr(m, "supported_generation_methods", []):
            continue
        if not _is_text_generation_model(m.name):
            continue
        candidates.append(m.name)

    if not candidates:
        return "gemini-2.0-flash"

    best = max(candidates, key=lambda x: _rank_model(x, prefer=prefer))
    return _strip_models_prefix(best)


GEMINI_MODEL_TRIAGE = select_best_model(prefer="flash")
GEMINI_MODEL_TRENDS = select_best_model(prefer="pro")

print(f"[GEMINI] Selected triage model: {GEMINI_MODEL_TRIAGE}")
print(f"[GEMINI] Selected trends model: {GEMINI_MODEL_TRENDS}")

gemini_model_triage = genai.GenerativeModel(GEMINI_MODEL_TRIAGE)
gemini_model_trends = genai.GenerativeModel(GEMINI_MODEL_TRENDS)


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
    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["identifier"] = df["identifier"].astype(str)
    return df


# =========================
# OpenAlex
# =========================

OPENALEX_SOURCE_CACHE: Dict[str, str] = {}


def _norm_journal_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def openalex_resolve_source_id(journal_name: str) -> Optional[str]:
    j_raw = (journal_name or "").strip()
    if not j_raw:
        return None
    if j_raw in OPENALEX_SOURCE_CACHE:
        return OPENALEX_SOURCE_CACHE[j_raw]

    url = "https://api.openalex.org/sources"
    params = {"search": j_raw, "per-page": 25}
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    r = SESSION.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print(f"[OpenAlex][ID] Failed for '{j_raw}' status={r.status_code}")
        return None

    results = r.json().get("results", []) or []
    if not results:
        print(f"[OpenAlex][ID] No match for '{j_raw}'")
        return None

    target = _norm_journal_name(j_raw)

    def score(res) -> float:
        name = _norm_journal_name(res.get("display_name", ""))
        if not name:
            return -1.0
        if name == target:
            return 1e6
        sim = SequenceMatcher(None, target, name).ratio()
        t_tokens = set(target.split())
        n_tokens = set(name.split())
        jac = (
            (len(t_tokens & n_tokens) / len(t_tokens | n_tokens))
            if (t_tokens and n_tokens)
            else 0.0
        )
        prefix_penalty = (
            0.25 if (target.startswith(name) and len(name) < len(target)) else 0.0
        )
        coverage_bonus = 0.2 if (t_tokens and t_tokens.issubset(n_tokens)) else 0.0
        return sim + 0.8 * jac + coverage_bonus - prefix_penalty

    best = max(results, key=score)
    full_id = (best.get("id") or "").strip()
    short_id = full_id.replace("https://openalex.org/", "").strip()
    if not short_id:
        return None

    OPENALEX_SOURCE_CACHE[j_raw] = short_id
    print(f"[OpenAlex][ID] {j_raw} -> {short_id}")
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
        if OPENALEX_MAILTO:
            params["mailto"] = OPENALEX_MAILTO

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
                url = doi if doi else (wid or "")

                auths = []
                for a in w.get("authorships") or []:
                    name = ((a.get("author") or {}).get("display_name") or "").strip()
                    if name:
                        auths.append(name)

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
                        authors=", ".join(auths) if auths else None,
                        abstract=abstract,
                    )
                )
            time.sleep(0.2)
        except Exception as e:
            print(f"[OpenAlex] {jname}: error {e}")

    print(f"[OpenAlex] Total in-window items (pre-dedupe): {len(out)}")
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
                summary = normalize_ws(
                    getattr(e, "summary", "") or getattr(e, "description", "") or ""
                )

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

    print(f"[RSS] Total in-window items (pre-dedupe): {len(out)}")
    return out


# =========================
# OneMine Repository
# =========================


def onemine_parse_identifier(identifier: str) -> Dict[str, str]:
    """
    Parses repository identifier into base OneMine query params.

    Supported forms:
      - "onemine:"  or "onemine"                      -> {}
      - "onemine:Organization=AUSIMM"                 -> {"Organization": "AUSIMM"}
      - "onemine:keywords=SME Annual Conference&searchfield=all"
      - "onemine:?keywords=World Gold Conference&searchfield=all"
      - Full URL: "https://www.onemine.org/search?keywords=...&searchfield=all"

    Returns dict of params to include in OneMine search.
    """
    ident = (identifier or "").strip()
    if not ident:
        return {}

    # Full URL support (robust to future sources.csv conventions)
    if (
        ident.lower().startswith(("http://", "https://"))
        and "onemine.org" in ident.lower()
    ):
        try:
            parsed = urlparse(ident)
            pairs = dict(parse_qsl(parsed.query, keep_blank_values=True))
        except Exception:
            pairs = {}
    else:
        if not ident.lower().startswith("onemine"):
            return {}
        rest = ident[len("onemine") :].lstrip(":").strip()
        if rest.startswith("?"):
            rest = rest[1:].strip()
        if not rest:
            pairs = {}
        else:
            pairs = dict(parse_qsl(rest, keep_blank_values=True))
            if not pairs and "=" in rest:
                k, v = rest.split("=", 1)
                pairs = {k.strip(): v.strip()}

    norm: Dict[str, str] = {}
    for k, v in (pairs or {}).items():
        kk, vv = (k or "").strip(), (v or "").strip()
        if not kk:
            continue
        if kk.lower() == "searchfield":
            norm["searchfield"] = vv
        elif kk.lower() == "keywords":
            norm["keywords"] = vv
        elif kk.lower() == "organization":
            norm["Organization"] = vv
        else:
            norm[kk] = vv

    return norm


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

        snippet_span = li.select_one("span.item-list__description")
        snippet = (
            snippet_span.get_text(" ", strip=True)
            if snippet_span
            else safe_get_text(li.select_one("div.item-list__content p"))
        )

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

        abstract = " ".join(paragraphs[:3]).strip() if paragraphs else meta_abstract

        text = main.get_text(" ", strip=True)
        m = re.search(r"\bBy\s+([A-Z][^|•\n]{3,200})", text)
        authors = m.group(0).strip() if m else ""

        return normalize_ws(authors), normalize_ws(abstract)
    except Exception:
        return None, None


def fetch_onemine_repository_items(
    sources_df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    max_pages: int = 3,
    page_size: int = 20,
) -> List[Item]:
    out: List[Item] = []
    repo_df = sources_df[sources_df["type"] == "repository"]
    if repo_df.empty:
        print("[Repo] No repository sources configured.")
        return out

    onemine_rows = []
    for _, row in repo_df.iterrows():
        ident = row["identifier"].strip()
        # Accept both the old "onemine:..." convention and a full OneMine URL
        if ident.lower().startswith("onemine") or ("onemine.org" in ident.lower()):
            onemine_rows.append(row)

    if not onemine_rows:
        print("[Repo] No OneMine repository entries found.")
        return out

    print(f"[Repo][OneMine] Fetching OneMine for {start} -> {end} ...")
    base_url = "https://www.onemine.org/search"

    for row in onemine_rows:
        name = row["name"].strip()
        ident = row["identifier"].strip()
        base_params = onemine_parse_identifier(ident)
        if "keywords" in base_params and "searchfield" not in base_params:
            base_params["searchfield"] = "all"

        candidates = []
        newest_seen: Optional[dt.date] = None

        for page in range(1, max_pages + 1):
            params = dict(base_params)
            params.update(
                {
                    "SortBy": "MostRecent",
                    "page": str(page),
                    "pageSize": str(page_size),
                    "DateFrom": start.isoformat(),
                    "DateTo": end.isoformat(),
                }
            )
            print(f"[Repo][OneMine] GET {name}: page={page} params={params}")

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

            if newest_seen is None:
                newest_seen = parse_date_flexible(parsed[0].get("date_text", ""))

            candidates.extend(parsed)
            time.sleep(0.3)

            last_date = parse_date_flexible(parsed[-1].get("date_text", ""))
            if last_date and last_date < start:
                break

        if newest_seen and newest_seen < start:
            print(
                f"[Repo][OneMine][NOTE] {name}: newest item seen is {newest_seen} < START_DATE={start}. Expect 0 in-window results."
            )

        kept = 0
        for c in candidates:
            pub = parse_date_flexible(c.get("date_text", ""))
            if not in_window(pub, start, end):
                continue

            authors = c.get("author_text", "")
            if authors.lower().startswith("by "):
                authors = authors[3:].strip()

            doc_auth, abstract = onemine_fetch_document_abstract(c["url"])
            if doc_auth:
                da = doc_auth
                if da.lower().startswith("by "):
                    da = da[3:].strip()
                if len(da) > len(authors):
                    authors = da

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

    print(f"[Repo][OneMine] Total in-window items (pre-dedupe): {len(out)}")
    return out


# =========================
# Dedupe
# =========================


def dedupe_items(items: List[Item]) -> List[Item]:
    seen = set()
    out = []
    for it in items:
        url = (it.url or "").strip()
        title = normalize_ws(it.title)
        date_s = it.published_date.isoformat() if it.published_date else ""
        doi_key = ""
        if url.lower().startswith("https://doi.org/") or url.lower().startswith(
            "http://doi.org/"
        ):
            doi_key = url.lower().replace("http://", "https://")
        key = doi_key or url.lower() or (title.lower() + "|" + date_s)
        key = key.strip() or sha1(title + "|" + date_s + "|" + (it.source or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# =========================
# AI helpers
# =========================


def ai_rate_limit_sleep(e: Exception) -> None:
    msg = str(e)
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
    if m:
        delay = max(int(m.group(1)), 10)
        print(f"[AI] Rate limit hinted: sleeping {delay}s ...")
        time.sleep(delay)
        return
    print(f"[AI] Rate limit/transient error: sleeping {AI_ERROR_SLEEP_SECONDS}s ...")
    time.sleep(AI_ERROR_SLEEP_SECONDS)


def gemini_call_with_retry(
    model: genai.GenerativeModel, prompt: str, max_attempts: int = 4
) -> str:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            ai_throttle()
            resp = model.generate_content(prompt)
            txt = (getattr(resp, "text", "") or "").strip()
            if not txt:
                raise RuntimeError("Empty Gemini response.")
            return txt
        except Exception as e:
            last_err = e
            print(f"[AI] Error attempt {attempt}/{max_attempts}: {e}")
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                ai_rate_limit_sleep(e)
            elif "503" in str(e) or "unavailable" in str(e).lower():
                time.sleep(AI_ERROR_SLEEP_SECONDS)
            else:
                time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"Gemini failed after {max_attempts} attempts: {last_err}")


def parse_json_strict(text: str) -> Optional[dict]:
    t = text.strip()
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


def clamp_for_ai(text: str, max_chars: int) -> str:
    t = normalize_ws(text or "")
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 40].rstrip() + " ... [truncated for AI to reduce tokens]"


AI_TRIAGE_SCHEMA = """
Return ONLY valid JSON:
{
  "results": [
    {
      "id": "<id>",
      "decision": "MUST_READ" | "GOOD_TO_READ" | "NOT_SURE",
      "category": "short label",
      "highlight": "one sentence highlight",
      "rationale": "one short sentence why"
    }
  ]
}
"""


def ai_triage_items_batched(
    items: List[Item], keywords_context: Dict[str, str]
) -> List[Item]:
    if not items:
        return []

    interests = "; ".join([f"{k}: {v}" for k, v in keywords_context.items()])
    print(
        f"[AI] Classifying {len(items)} Academic/Repository items with Gemini (batch_size={AI_BATCH_SIZE}) ..."
    )

    id_map: Dict[str, Item] = {}
    payload = []
    for i, it in enumerate(items, start=1):
        pid = f"it{i}"
        id_map[pid] = it
        payload.append(
            {
                "id": pid,
                "title": it.title,
                "source": it.source,
                "source_type": it.source_type,
                "published_date": it.published_date.isoformat()
                if it.published_date
                else "",
                "authors": it.authors or "",
                "abstract": clamp_for_ai(it.abstract or "", AI_ABSTRACT_MAX_CHARS),
                "url": it.url,
            }
        )

    for batch_start in range(0, len(payload), AI_BATCH_SIZE):
        batch = payload[batch_start : batch_start + AI_BATCH_SIZE]

        prompt = f"""
You are an expert research triage assistant for Mining, Mineral Processing, Extractive Metallurgy, and Automation/Control.

My interests (topics -> keywords):
{interests}

Classify each item into exactly one decision:
- MUST_READ
- GOOD_TO_READ
- NOT_SURE

Items (JSON):
{json.dumps(batch, ensure_ascii=False)}

{AI_TRIAGE_SCHEMA}
""".strip()

        try:
            txt = gemini_call_with_retry(gemini_model_triage, prompt, max_attempts=4)
            data = parse_json_strict(txt)
            if not data or "results" not in data:
                raise RuntimeError("Bad JSON from Gemini (missing results).")

            got = {r.get("id"): r for r in (data.get("results") or [])}
            for b in batch:
                pid = b["id"]
                it = id_map[pid]
                r = got.get(pid, {}) or {}

                dec = (r.get("decision") or "NOT_SURE").strip().upper()
                if dec not in {"MUST_READ", "GOOD_TO_READ", "NOT_SURE"}:
                    dec = "NOT_SURE"

                it.decision = dec
                it.category = normalize_ws(str(r.get("category") or "General"))
                it.highlight = normalize_ws(str(r.get("highlight") or ""))
                it.rationale = normalize_ws(str(r.get("rationale") or ""))
        except Exception as e:
            print(f"[AI] Batch error ({batch_start}..): {e}")
            for b in batch:
                it = id_map[b["id"]]
                it.decision = "NOT_SURE"
                it.category = "AI_Error"
                it.highlight = "AI error; included for manual review."
                it.rationale = "Fail-open to avoid missing relevant items."

        time.sleep(AI_SLEEP_SECONDS)

    print(f"[AI] Classified {len(items)} Academic/Repository items.")
    return items


def ai_trend_paragraphs_per_source(
    items: List[Item], max_items_per_source: int
) -> Dict[str, str]:
    by_source = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)

    def rank(d: str) -> int:
        return {"MUST_READ": 0, "GOOD_TO_READ": 1, "NOT_SURE": 2}.get(
            d or "NOT_SURE", 2
        )

    out: Dict[str, str] = {}
    for src, arr in sorted(by_source.items(), key=lambda x: x[0].lower()):
        arr_sorted = sorted(
            arr,
            key=lambda x: (rank(x.decision), x.published_date or dt.date(1900, 1, 1)),
        )
        take = arr_sorted[:max_items_per_source]

        mini = [
            {
                "title": it.title,
                "decision": it.decision,
                "category": it.category or "",
                "abstract_snippet": clamp_for_ai(it.abstract or "", 450),
            }
            for it in take
        ]

        prompt = f"""
Write ONE paragraph (3–5 sentences) summarising dominant themes for this source in the coverage window.

Source: {src}
Coverage: {START_DATE} to {END_DATE}

Items (JSON):
{json.dumps(mini, ensure_ascii=False)}

Return only the paragraph text (no headings).
""".strip()

        try:
            txt = gemini_call_with_retry(gemini_model_trends, prompt, max_attempts=3)
            out[src] = normalize_ws(txt)
        except Exception as e:
            print(f"[AI][Trends] {src}: {e}")
            out[src] = ""

        time.sleep(AI_SLEEP_SECONDS)

    return out


# =========================
# Overviews (deterministic)
# =========================


def build_source_overviews(items: List[Item]) -> Dict[str, dict]:
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
# CSV
# =========================


def items_to_dataframe(items: List[Item]) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append(
            {
                "title": it.title,
                "authors": it.authors or "",
                "published_date": it.published_date.isoformat()
                if it.published_date
                else "",
                "source": it.source,
                "source_type": it.source_type,
                "url": it.url,
                "abstract": it.abstract or "",
                "decision": it.decision or "",
                "category": it.category or "",
                "highlight": it.highlight or "",
                "rationale": it.rationale or "",
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =========================
# Newsletter rendering
# =========================


def render_newsletter_html(
    start: dt.date,
    end: dt.date,
    repo_start: dt.date,
    repo_end: dt.date,
    collected_counts: Dict[str, int],
    source_overviews: Dict[str, dict],
    items: List[Item],
    trend_notes: Dict[str, str],
    attachment_names: List[str],
) -> str:
    def decision_rank(d: str) -> int:
        return {"MUST_READ": 0, "GOOD_TO_READ": 1, "NOT_SURE": 2}.get(
            d or "NOT_SURE", 2
        )

    items_sorted = sorted(
        items,
        key=lambda x: (
            x.source.lower(),
            decision_rank(x.decision or "NOT_SURE"),
            x.published_date or dt.date(1900, 1, 1),
        ),
        reverse=False,
    )

    all_sources = sorted(set(collected_counts.keys()), key=lambda s: s.lower())

    html = f"""
<html>
<body style="font-family: Arial, Helvetica, sans-serif; color: #222;">
  <div style="background:#1f2d3d; padding: 18px;">
    <h2 style="color:#fff; margin:0;">⛏️ Weekly Mining & Processing Digest</h2>
    <div style="color:#cfd8dc; margin-top:6px;">
      Academic/News coverage: {start} → {end} <br/>
      Repository coverage (OneMine): {repo_start} → {repo_end}
    </div>
  </div>

  <div style="padding: 18px;">
    <h3 style="margin-top:0;">Overview</h3>
    <p>Repository sources are searched over an extended window (+30 days) to compensate for slower repository publishing/update cycles.</p>

    <h3>Collected by Source</h3>
    <table style="border-collapse: collapse; width: 100%; max-width: 900px;">
      <tr>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Source</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Collected</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">MUST_READ</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">GOOD_TO_READ</th>
        <th style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">NOT_SURE</th>
      </tr>
    """

    for src in all_sources:
        ov = source_overviews.get(src, {})
        decs = ov.get("decisions", {})
        html += f"""
      <tr>
        <td style="border-bottom:1px solid #eee; padding:6px;">{src}</td>
        <td style="border-bottom:1px solid #eee; padding:6px;">{collected_counts.get(src, 0)}</td>
        <td style="border-bottom:1px solid #eee; padding:6px;">{decs.get("MUST_READ", 0)}</td>
        <td style="border-bottom:1px solid #eee; padding:6px;">{decs.get("GOOD_TO_READ", 0)}</td>
        <td style="border-bottom:1px solid #eee; padding:6px;">{decs.get("NOT_SURE", 0)}</td>
      </tr>
        """

    html += "</table>"

    if attachment_names:
        html += "<h3 style='margin-top:22px;'>Attachments</h3><ul>"
        for a in attachment_names:
            html += f"<li>{a}</li>"
        html += "</ul>"

    html += "<h3 style='margin-top:26px;'>Source Trend Notes</h3>"
    for src in all_sources:
        note = (trend_notes.get(src) or "").strip()
        if not note:
            continue
        html += f"""
        <div style="margin-top:14px; padding:10px; border:1px solid #eee; border-radius:8px;">
          <div style="font-weight:bold;">{src}</div>
          <div style="font-size:13px; color:#333; margin-top:6px; line-height:1.4;">
            {note}
          </div>
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
            "NOT_SURE": "#546e7a",
        }.get(badge, "#546e7a")

        pub = it.published_date.isoformat() if it.published_date else "Unknown date"
        authors = it.authors or "Unknown"
        cat = it.category or "General"
        highlight = it.highlight or ""
        rationale = it.rationale or ""

        html += f"""
        <div style="margin:12px 0; padding:12px; border-left:6px solid {badge_color}; background:#fafafa; border-radius:6px;">
          <div style="font-size:12px; color:#444; margin-bottom:6px;">
            <span style="display:inline-block; padding:2px 8px; border-radius:10px; background:{badge_color}; color:#fff; font-weight:bold;">{badge}</span>
            <span style="margin-left:8px;">{pub}</span>
            <span style="margin-left:8px; color:#666;">[{cat}]</span>
          </div>

          <div style="font-size:16px; font-weight:bold; margin-bottom:6px;">
            <a href="{it.url}" style="color:#1f2d3d; text-decoration:none;">{it.title}</a>
          </div>

          <div style="font-size:13px; color:#666; margin-bottom:8px;">
            Source: {it.source_type} | Authors: {authors}
          </div>

          <div style="font-size:13px; color:#333;"><b>AI highlight:</b> {highlight}</div>
          <div style="font-size:12px; color:#555; margin-top:6px;">{rationale}</div>
        </div>
        """

    html += """
  </div>
  <div style="padding: 18px; text-align:center; font-size: 12px; color:#999;">Generated automatically.</div>
</body>
</html>
"""
    return html


# =========================
# Email
# =========================


def attach_file(msg: MIMEMultipart, filepath: str) -> None:
    with open(filepath, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    filename = os.path.basename(filepath)
    part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
    msg.attach(part)


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

    for fp in attachments:
        if fp and os.path.exists(fp):
            attach_file(msg, fp)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("[EMAIL] Sent successfully (with attachments).")


# =========================
# Main pipeline
# =========================


def run_pipeline(
    keywords_csv: str = "keywords.csv",
    sources_csv: str = "sources.csv",
    do_send_email: bool = True,
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
            sources_df, REPO_START_DATE, REPO_END_DATE, max_pages=3, page_size=20
        )
    )
    collected = dedupe_items(collected)
    print(f"[PIPELINE] Collected total (deduped): {len(collected)}")

    # AI only for Academic + Repository
    ai_items = [it for it in collected if it.source_type in {"Academic", "Repository"}]

    out_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, f"raw_academic_repository_{TODAY}.csv")
    write_csv(items_to_dataframe(ai_items), raw_path)
    print(f"[PIPELINE] Wrote raw CSV: {raw_path}")

    print(
        "\n[PIPELINE] Step 2: AI triage ONLY for Academic + Repository (token optimisation)"
    )
    triaged = ai_triage_items_batched(ai_items, keywords_context)

    triaged_path = os.path.join(out_dir, f"triaged_academic_repository_{TODAY}.csv")
    write_csv(items_to_dataframe(triaged), triaged_path)
    print(f"[PIPELINE] Wrote triaged CSV: {triaged_path}")

    print("\n[PIPELINE] Step 3: Build deterministic source overviews")
    overviews = build_source_overviews(triaged)

    print("\n[PIPELINE] Step 3b: AI trend paragraph per source (capped)")
    trend_notes = ai_trend_paragraphs_per_source(
        triaged, max_items_per_source=TREND_ITEMS_CAP
    )

    print("\n[PIPELINE] Step 4: Render newsletter")
    collected_counts = Counter([i.source for i in triaged])
    attachment_paths = [raw_path, triaged_path]
    attachment_names = [os.path.basename(p) for p in attachment_paths]

    html = render_newsletter_html(
        start=START_DATE,
        end=END_DATE,
        repo_start=REPO_START_DATE,
        repo_end=REPO_END_DATE,
        collected_counts=dict(collected_counts),
        source_overviews=overviews,
        items=triaged,
        trend_notes=trend_notes,
        attachment_names=attachment_names,
    )

    out_html = os.path.join(out_dir, f"digest_{TODAY}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[PIPELINE] Saved HTML to: {out_html}")

    print("\n[PIPELINE] Step 5: Send email (with attachments)")
    if do_send_email:
        send_email(html, attachments=[out_html] + attachment_paths)
    else:
        print("[PIPELINE] Email sending disabled (do_send_email=False).")


if __name__ == "__main__":
    import sys

    run_pipeline(do_send_email=("--no-email" not in sys.argv))
