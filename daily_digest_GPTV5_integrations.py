#!/usr/bin/env python3
"""
Recent Publications Digest (with optional integrations)

Fixes vs. the broken integration draft:
- load_integrations_config is defined before it is called (no NameError)
- integration step is inside run_pipeline (no indentation/flow corruption)
- missing helpers (ai_throttle, parse_date) are implemented
- integration Item creation matches the Item schema
- integrations config is optional and supports multiple filenames
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Optional YAML (preferred). We also include a tiny fallback parser.
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Gemini SDK (old one). We keep google-genai optional, but we don't require it.
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore


# ----------------------------
# Data model
# ----------------------------


@dataclass
class Item:
    title: str
    url: str
    source: str
    source_type: str  # "academic" | "rss" | "repository" | "integration"
    published_date: str  # YYYY-MM-DD where possible
    authors: str = ""
    abstract: str = ""
    doi: str = ""


# ----------------------------
# Utilities
# ----------------------------

USER_AGENT = "auto-lit-survey/1.0 (+https://github.com/moyahyaei/auto-lit-survey; contact: see README)"

DEFAULT_CONFIG: Dict[str, Any] = {
    "lookback_days": 7,
    "max_items_per_feed": 5,
    "email_subject": "Recent Publications Digest",
    "gemini_min_interval_seconds": 2.0,
    "gemini_error_sleep_seconds": 10.0,
    "repository_extra_days": 30,
}

DEFAULT_INTEGRATIONS: Dict[str, Any] = {
    "openalex": {"enabled": True, "per_journal_max_results": 5, "timeout_seconds": 30},
    "crossref": {"enabled": False, "max_items_per_topic": 5, "timeout_seconds": 30},
    "arxiv": {"enabled": False, "max_items_per_topic": 5, "timeout_seconds": 30},
    "ausimm": {
        "enabled": False,
        "max_items_per_topic": 5,
        "timeout_seconds": 30,
        "rss_url": "",
    },
    "semantic_scholar": {
        "enabled": False,
        "max_items_per_topic": 5,
        "timeout_seconds": 30,
    },
    "lens": {"enabled": False, "max_items_per_topic": 5, "timeout_seconds": 30},
}


def utc_today() -> dt.date:
    # Avoid deprecated utcnow()
    return dt.datetime.now(dt.timezone.utc).date()


def iso_date(d: dt.date) -> str:
    return d.isoformat()


def parse_date_flexible(s: str) -> Optional[dt.date]:
    if not s:
        return None
    s = s.strip()
    # Common formats
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",  # RSS
        "%d %b %Y",
        "%b %d, %Y",
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).date()
        except Exception:
            pass
    # Try feedparser parsed time format (e.g., "2025-12-17T00:00:00Z")
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except Exception:
        return None


def in_window(d: Optional[dt.date], start: dt.date, end: dt.date) -> bool:
    if d is None:
        return False
    return start <= d <= end


def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def dedupe_items(items: List[Item]) -> List[Item]:
    seen = set()
    out: List[Item] = []
    for it in items:
        key = (
            norm_text(it.title),
            (it.doi or "").lower(),
            (it.url or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def safe_get(
    url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30
) -> requests.Response:
    return requests.get(
        url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )


def safe_post(
    url: str,
    json_payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    return requests.post(url, json=json_payload, headers=hdrs, timeout=timeout)


def _simple_yaml_load(raw: str) -> Dict[str, Any]:
    """
    Minimal YAML subset loader: handles 'key: value' and nested via indentation.
    Only for emergencies if PyYAML is unavailable.
    """
    result: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, result)]
    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            continue
        key, val = line.strip().split(":", 1)
        val = val.strip()
        # walk stack
        while stack and indent < stack[-1][0]:
            stack.pop()
        cur = stack[-1][1] if stack else result
        if val == "":
            cur[key] = {}
            stack.append((indent + 2, cur[key]))
        else:
            # coerce types
            if val.lower() in ("true", "false"):
                cur[key] = val.lower() == "true"
            else:
                try:
                    if "." in val:
                        cur[key] = float(val)
                    else:
                        cur[key] = int(val)
                except Exception:
                    cur[key] = val.strip('"').strip("'")
    return result


def load_yaml_file(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(raw) or {}
    return _simple_yaml_load(raw)


def load_config(script_dir: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    candidates = [
        script_dir / "config.yaml",
        script_dir / "config.yml",
        script_dir / "config.json",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".json":
                cfg = json.loads(p.read_text(encoding="utf-8"))
            else:
                cfg = load_yaml_file(p)
            merged = dict(DEFAULT_CONFIG)
            merged.update(cfg or {})
            return merged, p
    return dict(DEFAULT_CONFIG), None


def load_integrations_config(script_dir: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    candidates = [
        script_dir / "config_integrations.yaml",
        script_dir / "config_integrations.yml",
        script_dir / "config_integration.yaml",
        script_dir / "config_integration.yml",
    ]
    for p in candidates:
        if p.exists():
            cfg = load_yaml_file(p)
            merged = dict(DEFAULT_INTEGRATIONS)
            # shallow merge per key
            for k, v in (cfg or {}).items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            return merged, p
    return dict(DEFAULT_INTEGRATIONS), None


def load_sources_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing sources.csv at: {path}")
    df = pd.read_csv(path)
    # Normalize expected columns
    for col in ["type", "name"]:
        if col not in df.columns:
            raise ValueError(
                f"sources.csv must include column '{col}'. Found: {list(df.columns)}"
            )
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["name"] = df["name"].astype(str).str.strip()
    return df


def load_keywords_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    # Accept "keyword" or first column
    if "keyword" in df.columns:
        vals = df["keyword"].astype(str).tolist()
    else:
        vals = df.iloc[:, 0].astype(str).tolist()
    out = []
    for v in vals:
        v = v.strip()
        if v and not v.startswith("#"):
            out.append(v)
    return out


# ----------------------------
# Gemini model selection + throttling
# ----------------------------


def _is_text_generation_model(name: str) -> bool:
    n = name.lower()
    return n.startswith("models/gemini") or n.startswith("gemini")


def _strip_models_prefix(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("models/") else name


def _model_score(name: str, prefer: str) -> Tuple[int, int, int, int, int]:
    """
    Ranking heuristic:
    - prefer stable-ish aliases (latest) over previews/exp when possible
    - prefer family (flash vs pro) depending on prefer
    - then prefer higher major/minor (2.5 > 2.0)
    """
    n = name.lower()
    tag = 0
    if "latest" in n:
        tag = 60
    elif "preview" in n:
        tag = 30
    elif "-exp" in n or "experimental" in n:
        tag = 20
    elif re.search(r"-\d{3}\b", n):  # revision like -001
        tag = 10

    m = re.search(r"gemini-(\d+)\.(\d+)", n)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if m else 0

    fam = 0
    if prefer == "flash":
        fam = 40 if "flash" in n else 20 if "pro" in n else 0
    else:
        fam = 40 if "pro" in n else 20 if "flash" in n else 0

    rev = 0
    m2 = re.search(r"-(\d{3})\b", n)
    if m2:
        rev = int(m2.group(1))

    return (tag, major, minor, fam, rev)


def pick_latest_model(prefer: str = "flash") -> Optional[str]:
    if genai is None:
        return None
    try:
        models = list(genai.list_models())
    except Exception:
        return None

    candidates = []
    for m in models:
        name = getattr(m, "name", "")
        methods = getattr(m, "supported_generation_methods", []) or []
        if not name:
            continue
        if "generateContent" not in methods:
            continue
        if not _is_text_generation_model(name):
            continue
        candidates.append(name)

    if not candidates:
        return None

    best = sorted(candidates, key=lambda n: _model_score(n, prefer), reverse=True)[0]
    return _strip_models_prefix(best)


_AI_LAST_CALL_MONO = 0.0


def ai_throttle(min_interval_seconds: float) -> None:
    global _AI_LAST_CALL_MONO
    now = time.monotonic()
    if _AI_LAST_CALL_MONO <= 0:
        _AI_LAST_CALL_MONO = now
        return
    wait = min_interval_seconds - (now - _AI_LAST_CALL_MONO)
    if wait > 0:
        time.sleep(wait)
    _AI_LAST_CALL_MONO = time.monotonic()


def init_gemini() -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (sdk_name, model_name).
    """
    if genai is None:
        return None, None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "google-generativeai", None
    genai.configure(api_key=api_key)
    # Prefer flash for triage by default.
    model = pick_latest_model("flash") or "gemini-2.0-flash"
    return "google-generativeai", model


# ----------------------------
# OpenAlex (journals)
# ----------------------------

OPENALEX_BASE = "https://api.openalex.org"


def _similarity(a: str, b: str) -> float:
    # lightweight similarity score
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    # token overlap + prefix bonus
    aset, bset = set(a.split()), set(b.split())
    jacc = len(aset & bset) / max(1, len(aset | bset))
    prefix = 0.15 if a == b else (0.05 if a.startswith(b) or b.startswith(a) else 0.0)
    return min(1.0, jacc + prefix)


_OPENALEX_SOURCE_ID_CACHE: Dict[str, str] = {}


def resolve_openalex_source_id(
    journal_name: str, mailto: str = "", timeout: int = 30
) -> Optional[str]:
    if journal_name in _OPENALEX_SOURCE_ID_CACHE:
        return _OPENALEX_SOURCE_ID_CACHE[journal_name]

    params = {"search": journal_name, "per-page": 10}
    if mailto:
        params["mailto"] = mailto
    url = f"{OPENALEX_BASE}/sources"
    r = safe_get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []

    best_id = None
    best_score = -1.0
    for src in results:
        disp = src.get("display_name") or ""
        sid = src.get("id") or ""
        score = _similarity(journal_name, disp)
        if score > best_score:
            best_score = score
            best_id = sid

    if not best_id:
        return None

    # Guardrail: avoid bad matches like "Minerals" vs "Minerals Engineering"
    if best_score < 0.55:
        print(
            f"[OpenAlex][WARN] Low-confidence match for '{journal_name}'. Best='{best_id}' score={best_score:.2f}. Skipping."
        )
        return None

    _OPENALEX_SOURCE_ID_CACHE[journal_name] = best_id
    return best_id


def fetch_openalex_journal_items(
    journal_name: str,
    start_date: dt.date,
    end_date: dt.date,
    max_results: int,
    mailto: str = "",
    timeout: int = 30,
) -> List[Item]:
    sid = resolve_openalex_source_id(journal_name, mailto=mailto, timeout=timeout)
    if not sid:
        return []

    # OpenAlex uses full IDs like "https://openalex.org/Sxxxx". Accept both.
    sid_norm = sid.split("/")[-1] if sid.startswith("http") else sid

    params = {
        "filter": f"from_publication_date:{iso_date(start_date)},to_publication_date:{iso_date(end_date)},primary_location.source.id:{sid_norm}",
        "sort": "publication_date:desc",
        "per-page": max_results,
    }
    if mailto:
        params["mailto"] = mailto

    url = f"{OPENALEX_BASE}/works"
    r = safe_get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    works = data.get("results") or []

    out: List[Item] = []
    for w in works:
        title = (w.get("title") or "").strip()
        if not title:
            continue
        pub = (w.get("publication_date") or "").strip()
        doi = (w.get("doi") or "").strip()
        url_ = ""
        # primary location
        pl = w.get("primary_location") or {}
        if isinstance(pl, dict):
            url_ = (pl.get("landing_page_url") or pl.get("pdf_url") or "").strip()
        if not url_:
            url_ = (w.get("id") or "").strip()

        authors = ""
        auths = []
        for a in w.get("authorships") or []:
            try:
                auths.append(a["author"]["display_name"])
            except Exception:
                pass
        if auths:
            authors = ", ".join(auths[:15])

        abstract = ""
        inv = w.get("abstract_inverted_index")
        if isinstance(inv, dict):
            # reconstruct quickly (best-effort)
            pairs = []
            for word, positions in inv.items():
                for p in positions:
                    pairs.append((p, word))
            pairs.sort(key=lambda x: x[0])
            abstract = " ".join([w for _, w in pairs])

        out.append(
            Item(
                title=title,
                url=url_,
                source=journal_name,
                source_type="academic",
                published_date=pub or "",
                authors=authors,
                abstract=abstract,
                doi=doi,
            )
        )
    return out


# ----------------------------
# RSS
# ----------------------------


def fetch_rss_items(
    feed_url: str,
    feed_name: str,
    start_date: dt.date,
    end_date: dt.date,
    max_items: int,
) -> List[Item]:
    parsed = feedparser.parse(feed_url)
    out: List[Item] = []
    for entry in parsed.entries[: max_items * 5]:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        if not title or not link:
            continue
        # date
        pub = ""
        if hasattr(entry, "published"):
            pub = entry.published
        d = parse_date_flexible(pub) if pub else None
        if not in_window(d, start_date, end_date):
            continue

        summary = getattr(entry, "summary", "") or ""
        out.append(
            Item(
                title=title,
                url=link,
                source=feed_name,
                source_type="rss",
                published_date=iso_date(d) if d else "",
                abstract=html.unescape(re.sub(r"<[^>]+>", "", summary)).strip(),
            )
        )
        if len(out) >= max_items:
            break
    return out


# ----------------------------
# OneMine repository (scrape)
# ----------------------------


def _onemine_search_url(keywords: str, page: int) -> str:
    # Your code already uses a search URL with many query params.
    # OneMine sometimes normalizes the URL on redirect; we keep it simple and stable.
    base = "https://www.onemine.org/search"
    params = {
        "SortBy": "MostRecent",
        "page": str(page),
        "pageSize": "20",
        "keywords": keywords,
        "searchfield": "all",
    }
    # We'll let requests handle params when calling safe_get.
    return base, params


def _parse_onemine_date(text_: str) -> Optional[dt.date]:
    # OneMine date formats vary; try flexible parsing.
    return parse_date_flexible(text_)


def fetch_onemine_items(
    start_date: dt.date,
    end_date: dt.date,
    max_pages: int = 2,
    categories: Optional[List[Tuple[str, str]]] = None,
    timeout: int = 30,
) -> List[Item]:
    if categories is None:
        categories = [
            ("OneMine (global)", "onemine:"),
            (
                "SME Annual Conference",
                "onemine:keywords=SME%20Annual%20Conference&searchfield=all",
            ),
            (
                "International Mineral Processing Congress",
                "onemine:keywords=International%20Mineral%20Processing%20Congress&searchfield=all",
            ),
            (
                "World Gold Conference",
                "onemine:keywords=World%20Gold%20Conference&searchfield=all",
            ),
            (
                "Iron Ore Conference",
                "onemine:keywords=Iron%20Ore%20Conference&searchfield=all",
            ),
            (
                "Copper International Conference",
                "onemine:keywords=Copper%20International%20Conference&searchfield=all",
            ),
            ("AusIMM Bulletin", "onemine:keywords=AusIMM%20Bulletin&searchfield=all"),
        ]

    out: List[Item] = []

    for cat_name, kw in categories:
        newest_seen: Optional[dt.date] = None
        kept = 0

        for page in range(1, max_pages + 1):
            base, params = _onemine_search_url(kw, page)
            print(
                f"[Repo][OneMine] GET {cat_name}: page={page} (keywords='{kw}') url={base}"
            )
            r = safe_get(base, params=params, timeout=timeout)
            if r.status_code != 200:
                print(
                    f"[Repo][OneMine][WARN] HTTP {r.status_code} for {cat_name} page={page}"
                )
                break

            soup = BeautifulSoup(r.text, "html.parser")
            # OneMine listings appear as cards; we use heuristics.
            cards = (
                soup.select("div.search-result, li.search-result, div.result, article")
                or []
            )
            if not cards:
                # fallback: find links to /document/
                cards = soup.find_all("a", href=re.compile(r"/document/"))

            parsed_count = 0
            for card in cards:
                parsed_count += 1
                # link
                link = ""
                title = ""
                if hasattr(card, "select_one"):
                    a = card.select_one("a[href]")
                    if a:
                        link = a.get("href", "").strip()
                        title = a.get_text(" ", strip=True)
                if not link and getattr(card, "get", None):
                    link = card.get("href", "").strip()
                    title = (
                        card.get_text(" ", strip=True)
                        if hasattr(card, "get_text")
                        else ""
                    )

                if not link:
                    continue
                if link.startswith("/"):
                    link = "https://www.onemine.org" + link
                if not title:
                    continue

                # date (look for something that looks like a date near the card)
                date_text = ""
                txt = (
                    card.get_text(" ", strip=True) if hasattr(card, "get_text") else ""
                )
                m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", txt)
                if m:
                    date_text = m.group(1)
                else:
                    # other common format e.g. "Oct 2, 2025"
                    m2 = re.search(r"\b([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b", txt)
                    if m2:
                        date_text = m2.group(1)

                d = _parse_onemine_date(date_text) if date_text else None
                if d and (newest_seen is None or d > newest_seen):
                    newest_seen = d

                if not in_window(d, start_date, end_date):
                    continue

                out.append(
                    Item(
                        title=title,
                        url=link,
                        source=cat_name,
                        source_type="repository",
                        published_date=iso_date(d) if d else "",
                    )
                )
                kept += 1

            print(
                f"[Repo][OneMine] {cat_name}: parsed {parsed_count} list items on page {page}"
            )

            # early stop if newest seen is already older than start_date
            if newest_seen and newest_seen < start_date:
                print(
                    f"[Repo][OneMine][NOTE] {cat_name}: newest item seen is {iso_date(newest_seen)} < START_DATE={iso_date(start_date)}. Expect 0 in-window results."
                )
                break

        print(f"[Repo][OneMine] {cat_name}: kept {kept} in-window items")

    return out


# ----------------------------
# Optional integrations (Option A keyword scan)
# ----------------------------


def _integration_enabled(cfg: Dict[str, Any], name: str) -> bool:
    v = cfg.get(name, {})
    return bool(v.get("enabled", False))


def _integration_max_items(cfg: Dict[str, Any], name: str, default: int) -> int:
    v = cfg.get(name, {})
    try:
        return int(v.get("max_items_per_topic", default))
    except Exception:
        return default


def _integration_timeout(cfg: Dict[str, Any], name: str, default: int = 30) -> int:
    v = cfg.get(name, {})
    try:
        return int(v.get("timeout_seconds", default))
    except Exception:
        return default


def _integration_extra_days(
    cfg_main: Dict[str, Any], cfg_int: Dict[str, Any], name: str
) -> int:
    v = cfg_int.get(name, {})
    try:
        return int(v.get("extra_days", cfg_main.get("repository_extra_days", 30)))
    except Exception:
        return int(cfg_main.get("repository_extra_days", 30))


def fetch_crossref_for_topic(
    topic: str,
    start_date: dt.date,
    end_date: dt.date,
    max_items: int,
    timeout: int = 30,
    mailto: str = "",
) -> List[Item]:
    """
    Crossref REST API: https://api.crossref.org/works
    """
    url = "https://api.crossref.org/works"
    params: Dict[str, Any] = {
        "query": topic,
        "rows": max(5, min(50, max_items * 3)),
        "sort": "published",
        "order": "desc",
        "filter": f"from-pub-date:{iso_date(start_date)},until-pub-date:{iso_date(end_date)}",
        "select": "DOI,title,URL,author,issued,published,created,abstract",
    }
    if mailto:
        params["mailto"] = mailto

    r = safe_get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = (data.get("message") or {}).get("items") or []
    out: List[Item] = []

    for it in items:
        title = ""
        if isinstance(it.get("title"), list) and it["title"]:
            title = str(it["title"][0]).strip()
        elif isinstance(it.get("title"), str):
            title = it["title"].strip()
        if not title:
            continue

        doi = (it.get("DOI") or "").strip()
        link = (it.get("URL") or "").strip() or (
            f"https://doi.org/{doi}" if doi else ""
        )

        # date
        d: Optional[dt.date] = None
        for k in ("published", "issued", "created"):
            v = it.get(k)
            if isinstance(v, dict):
                parts = (v.get("date-parts") or [[]])[0]
                if parts:
                    try:
                        y = int(parts[0])
                        m = int(parts[1]) if len(parts) > 1 else 1
                        dd = int(parts[2]) if len(parts) > 2 else 1
                        d = dt.date(y, m, dd)
                        break
                    except Exception:
                        pass

        if not in_window(d, start_date, end_date):
            continue

        # authors
        auths = []
        for a in it.get("author") or []:
            fam = a.get("family") or ""
            giv = a.get("given") or ""
            nm = (giv + " " + fam).strip()
            if nm:
                auths.append(nm)
        authors = ", ".join(auths[:15])

        abstract = it.get("abstract") or ""
        if abstract:
            abstract = re.sub(r"<[^>]+>", "", str(abstract))
            abstract = html.unescape(abstract).strip()

        out.append(
            Item(
                title=title,
                url=link,
                source="Crossref",
                source_type="integration",
                published_date=iso_date(d) if d else "",
                authors=authors,
                abstract=abstract,
                doi=doi,
            )
        )
        if len(out) >= max_items:
            break

    return out


def fetch_arxiv_for_topic(
    topic: str,
    start_date: dt.date,
    end_date: dt.date,
    max_items: int,
    timeout: int = 30,
) -> List[Item]:
    """
    arXiv API (Atom): http://export.arxiv.org/api/query
    Date filter uses submittedDate range query syntax.
    """
    # arXiv uses UTC timestamps without separators
    start_q = start_date.strftime("%Y%m%d") + "0000"
    end_q = end_date.strftime("%Y%m%d") + "2359"
    search_query = f'(all:"{topic}") AND submittedDate:[{start_q} TO {end_q}]'
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max(5, min(50, max_items * 3)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = safe_get(url, params=params, timeout=timeout)
    r.raise_for_status()
    feed = feedparser.parse(r.text)

    out: List[Item] = []
    for entry in feed.entries:
        title = getattr(entry, "title", "").strip().replace("\n", " ")
        link = getattr(entry, "link", "").strip()
        if not title or not link:
            continue
        pub_s = getattr(entry, "published", "") or getattr(entry, "updated", "")
        d = parse_date_flexible(pub_s)
        if not in_window(d, start_date, end_date):
            continue
        authors = ", ".join(
            [a.name for a in getattr(entry, "authors", []) if getattr(a, "name", "")][
                :15
            ]
        )
        abstract = getattr(entry, "summary", "") or ""
        abstract = html.unescape(re.sub(r"<[^>]+>", "", abstract)).strip()

        out.append(
            Item(
                title=title,
                url=link,
                source="arXiv",
                source_type="integration",
                published_date=iso_date(d) if d else "",
                authors=authors,
                abstract=abstract,
            )
        )
        if len(out) >= max_items:
            break
    return out


def fetch_ausimm_from_rss(
    rss_url: str,
    start_date: dt.date,
    end_date: dt.date,
    max_items: int,
    timeout: int = 30,
) -> List[Item]:
    """
    AusIMM Digital Library: safest integration is via RSS if you have a feed URL.
    If you don't have an RSS URL, leave it blank and this integration will no-op.
    """
    if not rss_url:
        print("[Integrations][AusIMM] rss_url not configured; skipping.")
        return []
    parsed = feedparser.parse(rss_url)
    out: List[Item] = []
    for entry in parsed.entries[: max_items * 5]:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", "").strip()
        if not title or not link:
            continue
        pub = getattr(entry, "published", "") or getattr(entry, "updated", "")
        d = parse_date_flexible(pub)
        if not in_window(d, start_date, end_date):
            continue
        summary = getattr(entry, "summary", "") or ""
        out.append(
            Item(
                title=title,
                url=link,
                source="AusIMM Digital Library",
                source_type="integration",
                published_date=iso_date(d) if d else "",
                abstract=html.unescape(re.sub(r"<[^>]+>", "", summary)).strip(),
            )
        )
        if len(out) >= max_items:
            break
    return out


def collect_integrations(
    config: Dict[str, Any],
    integrations: Dict[str, Any],
    keywords: List[str],
    start_date: dt.date,
    end_date: dt.date,
) -> List[Item]:
    """
    Option A: run keyword-based scans against enabled sources and merge.
    """
    items: List[Item] = []
    topics = keywords[:]
    if not topics:
        return items

    # Crossref
    if _integration_enabled(integrations, "crossref"):
        extra = _integration_extra_days(config, integrations, "crossref")
        s = start_date - dt.timedelta(days=extra)
        timeout = _integration_timeout(integrations, "crossref", 30)
        max_per = _integration_max_items(integrations, "crossref", 5)
        mailto = (
            os.getenv(integrations.get("crossref", {}).get("mailto_env", "") or "")
            or ""
        )
        print(
            f"[Integrations] Crossref enabled. Window: {iso_date(s)} -> {iso_date(end_date)}."
        )
        for topic in topics:
            try:
                items.extend(
                    fetch_crossref_for_topic(
                        topic, s, end_date, max_per, timeout=timeout, mailto=mailto
                    )
                )
            except Exception as e:
                print(f"[Integrations][Crossref][WARN] topic='{topic}': {e}")

    # arXiv
    if _integration_enabled(integrations, "arxiv"):
        extra = _integration_extra_days(config, integrations, "arxiv")
        s = start_date - dt.timedelta(days=extra)
        timeout = _integration_timeout(integrations, "arxiv", 30)
        max_per = _integration_max_items(integrations, "arxiv", 5)
        print(
            f"[Integrations] arXiv enabled. Window: {iso_date(s)} -> {iso_date(end_date)}."
        )
        for topic in topics:
            try:
                items.extend(
                    fetch_arxiv_for_topic(topic, s, end_date, max_per, timeout=timeout)
                )
            except Exception as e:
                print(f"[Integrations][arXiv][WARN] topic='{topic}': {e}")

    # AusIMM (RSS only)
    if _integration_enabled(integrations, "ausimm"):
        extra = _integration_extra_days(config, integrations, "ausimm")
        s = start_date - dt.timedelta(days=extra)
        timeout = _integration_timeout(integrations, "ausimm", 30)
        max_per = _integration_max_items(integrations, "ausimm", 5)
        rss_url = integrations.get("ausimm", {}).get("rss_url", "") or ""
        print(
            f"[Integrations] AusIMM enabled (RSS). Window: {iso_date(s)} -> {iso_date(end_date)}."
        )
        try:
            items.extend(
                fetch_ausimm_from_rss(rss_url, s, end_date, max_per, timeout=timeout)
            )
        except Exception as e:
            print(f"[Integrations][AusIMM][WARN] {e}")

    # Semantic Scholar and Lens are intentionally no-op until you add API keys and enable them.
    return items


# ----------------------------
# AI triage (classification/summarisation)
# ----------------------------


def build_triage_prompt(batch: List[Item]) -> str:
    # Keep prompt compact to reduce tokens
    lines = []
    for i, it in enumerate(batch, start=1):
        lines.append(
            f"{i}. {it.title}\n   Source: {it.source}\n   URL: {it.url}\n   Date: {it.published_date}\n"
        )
    return (
        "You are triaging research items for mining/mineral processing/process control/sustainability.\n"
        "For each item, output a JSON array of objects with keys: index, keep (true/false), reason (<=20 words), tags (array of <=6 short tags).\n"
        "Be conservative: keep only items likely relevant.\n\n"
        "Items:\n" + "\n".join(lines)
    )


def gemini_generate_json(model: Any, prompt: str) -> Any:
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", "") or ""
    # try to extract JSON
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    return json.loads(m.group(0))


def triage_items_with_gemini(
    items: List[Item],
    gemini_model_name: Optional[str],
    min_interval_seconds: float,
    error_sleep_seconds: float,
    batch_size: int = 6,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    if genai is None or not gemini_model_name:
        print("[AI][WARN] Gemini not configured; skipping triage.")
        return []

    model = genai.GenerativeModel(gemini_model_name)
    triaged_rows: List[Dict[str, Any]] = []

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    print(
        f"[AI] Classifying {len(items)} items with Gemini (batch_size={batch_size}) ..."
    )

    for b in batches:
        prompt = build_triage_prompt(b)

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                ai_throttle(min_interval_seconds)
                result = gemini_generate_json(model, prompt)
                if not isinstance(result, list):
                    raise ValueError("No JSON array found in model response.")
                # Map back to items
                for obj in result:
                    idx = int(obj.get("index", 0))
                    if idx <= 0 or idx > len(b):
                        continue
                    it = b[idx - 1]
                    triaged_rows.append(
                        {
                            "title": it.title,
                            "url": it.url,
                            "source": it.source,
                            "source_type": it.source_type,
                            "published_date": it.published_date,
                            "authors": it.authors,
                            "doi": it.doi,
                            "keep": bool(obj.get("keep", False)),
                            "reason": str(obj.get("reason", ""))[:200],
                            "tags": ",".join(obj.get("tags", []) or [])[:200],
                        }
                    )
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                # crude rate-limit detection
                if "429" in msg or "Resource exhausted" in msg:
                    print(
                        f"[AI] Rate limit: sleeping {max(60, int(error_sleep_seconds))}s ..."
                    )
                    time.sleep(max(60, int(error_sleep_seconds)))
                else:
                    time.sleep(error_sleep_seconds)

        if last_err is not None:
            print(f"[AI][WARN] Failed batch after {max_retries} attempts: {last_err}")

    return triaged_rows


# ----------------------------
# Render + email (email sending left unchanged; you already have working SMTP bits)
# ----------------------------


def write_csv(path: Path, items: List[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(asdict(items[0]).keys())
            if items
            else list(asdict(Item("", "", "", "", "")).keys()),
        )
        w.writeheader()
        for it in items:
            w.writerow(asdict(it))


def write_triaged_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write header-only
        cols = [
            "title",
            "url",
            "source",
            "source_type",
            "published_date",
            "authors",
            "doi",
            "keep",
            "reason",
            "tags",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_html_digest(
    items: List[Item],
    triaged: List[Dict[str, Any]],
    window_start: dt.date,
    window_end: dt.date,
    title: str,
) -> str:
    kept = [r for r in triaged if r.get("keep")]
    kept_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in kept:
        kept_by_source.setdefault(r.get("source", "Unknown"), []).append(r)

    def esc(s: str) -> str:
        return html.escape(s or "")

    parts = []
    parts.append(
        f"<html><head><meta charset='utf-8'><title>{esc(title)}</title></head><body>"
    )
    parts.append(f"<h2>{esc(title)}</h2>")
    parts.append(
        f"<p><b>Coverage:</b> {esc(iso_date(window_start))} → {esc(iso_date(window_end))} (UTC)</p>"
    )
    parts.append(
        f"<p><b>Total items collected:</b> {len(items)} &nbsp; | &nbsp; <b>Kept after triage:</b> {len(kept)}</p>"
    )

    if not kept:
        parts.append("<p><i>No items were kept by the triage filter this run.</i></p>")
    else:
        for src, rows in sorted(kept_by_source.items(), key=lambda x: x[0].lower()):
            parts.append(f"<h3>{esc(src)}</h3><ul>")
            for r in rows:
                parts.append(
                    "<li>"
                    f"<a href='{esc(r.get('url', ''))}'>{esc(r.get('title', ''))}</a>"
                    f" <small>({esc(r.get('published_date', ''))})</small><br/>"
                    f"<small>{esc(r.get('reason', ''))}</small>"
                    "</li>"
                )
            parts.append("</ul>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ----------------------------
# Pipeline
# ----------------------------


def run_pipeline(script_dir: Path, no_email: bool = False) -> None:
    # Load env + config
    env_path = script_dir / ".env"
    load_dotenv(dotenv_path=env_path if env_path.exists() else None)
    print(f"[ENV] Loaded .env from: {env_path} (exists={env_path.exists()})")

    config, cfg_path = load_config(script_dir)
    print(f"[CONFIG] Loaded config from: {cfg_path if cfg_path else '(defaults)'}")

    integrations, int_path = load_integrations_config(script_dir)
    print(
        f"[INTEGRATIONS] Loaded integrations config from: {int_path if int_path else '(defaults)'}"
    )

    # Date window
    lookback_days = int(config.get("lookback_days", 7))
    end_date = utc_today()
    start_date = end_date - dt.timedelta(days=lookback_days)
    print(
        f"[WINDOW] Coverage: {iso_date(start_date)} -> {iso_date(end_date)} (lookback_days={lookback_days})"
    )

    # Gemini
    sdk_name, gemini_model_name = init_gemini()
    if sdk_name and gemini_model_name:
        print(f"[GEMINI] SDK={sdk_name} model={gemini_model_name}")
    elif sdk_name:
        print(f"[GEMINI] SDK={sdk_name} but no model available (missing API key?)")
    else:
        print("[GEMINI] SDK not available (install google-generativeai).")

    # Inputs
    sources_path = script_dir / "sources.csv"
    keywords_path = script_dir / "keywords.csv"
    sources_df = load_sources_csv(sources_path)
    keywords = load_keywords_csv(keywords_path)

    max_items_per_feed = int(config.get("max_items_per_feed", 5))

    collected: List[Item] = []

    print(
        "\n[PIPELINE] Step 1: Collect items (OpenAlex + RSS + OneMine + Integrations)"
    )
    # OpenAlex (journals)
    if bool(integrations.get("openalex", {}).get("enabled", True)):
        mailto_env = (
            integrations.get("openalex", {}).get("polite_pool_email_env", "") or ""
        )
        mailto = os.getenv(mailto_env, "") if mailto_env else ""
        oa_timeout = int(integrations.get("openalex", {}).get("timeout_seconds", 30))
        per_journal = int(
            integrations.get("openalex", {}).get(
                "per_journal_max_results", max_items_per_feed
            )
        )

        journal_sources = sources_df[sources_df["type"] == "journal"]["name"].tolist()
        for jname in journal_sources:
            try:
                print(f"[OpenAlex] Querying: {jname} ...")
                items = fetch_openalex_journal_items(
                    jname,
                    start_date,
                    end_date,
                    per_journal,
                    mailto=mailto,
                    timeout=oa_timeout,
                )
                collected.extend(items)
                print(f"[OpenAlex] {jname}: fetched {len(items)} candidates")
            except Exception as e:
                print(f"[OpenAlex][WARN] {jname}: {e}")
    else:
        print("[OpenAlex] Disabled by integrations config.")

    # RSS
    rss_sources = sources_df[sources_df["type"] == "rss"]
    if len(rss_sources) == 0:
        print("[RSS] No rss sources configured.")
    else:
        for _, row in rss_sources.iterrows():
            feed_name = str(row["name"]).strip()
            feed_url = str(row.get("url") or row.get("rss_url") or "").strip()
            if not feed_url:
                continue
            try:
                items = fetch_rss_items(
                    feed_url, feed_name, start_date, end_date, max_items_per_feed
                )
                collected.extend(items)
                print(f"[RSS] {feed_name}: fetched {len(items)}")
            except Exception as e:
                print(f"[RSS][WARN] {feed_name}: {e}")

    # OneMine (repositories) - ALWAYS extend by repository_extra_days
    repo_extra = int(config.get("repository_extra_days", 30))
    repo_start = start_date - dt.timedelta(days=repo_extra)
    try:
        onemine_items = fetch_onemine_items(
            repo_start, end_date, max_pages=2, timeout=30
        )
        collected.extend(onemine_items)
        print(f"[Repo][OneMine] Total in-window items: {len(onemine_items)}")
    except Exception as e:
        print(f"[Repo][OneMine][WARN] {e}")

    # Integrations (Crossref, arXiv, AusIMM RSS) - also use extended window per integration
    try:
        integ_items = collect_integrations(
            config, integrations, keywords, start_date, end_date
        )
        if integ_items:
            print(f"[Integrations] Collected: {len(integ_items)} items")
        collected.extend(integ_items)
    except Exception as e:
        print(f"[Integrations][WARN] {e}")

    collected = dedupe_items(collected)
    print(f"[PIPELINE] Collected in-window total (deduped): {len(collected)}")

    out_dir = script_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = out_dir / f"raw_academic_repository_{iso_date(end_date)}.csv"
    write_csv(raw_csv_path, collected)
    print(f"[PIPELINE] Wrote raw CSV: {raw_csv_path}")

    # Step 2: AI triage (Academic + Repository + Integrations)
    print(
        "\n[PIPELINE] Step 2: AI triage ONLY for Academic + Repository + Integrations (token optimisation)"
    )
    ai_items = [
        it
        for it in collected
        if it.source_type in ("academic", "repository", "integration")
    ]

    triaged_rows = triage_items_with_gemini(
        ai_items,
        gemini_model_name,
        min_interval_seconds=float(config.get("gemini_min_interval_seconds", 2.0)),
        error_sleep_seconds=float(config.get("gemini_error_sleep_seconds", 10.0)),
        batch_size=6,
        max_retries=3,
    )

    triaged_csv_path = out_dir / f"triaged_academic_repository_{iso_date(end_date)}.csv"
    write_triaged_csv(triaged_csv_path, triaged_rows)
    print(f"[PIPELINE] Wrote triaged CSV: {triaged_csv_path}")

    # Step 3: Render newsletter HTML
    print("\n[PIPELINE] Step 3: Render newsletter")
    title = str(config.get("email_subject", "Recent Publications Digest"))
    html_body = render_html_digest(
        collected, triaged_rows, start_date, end_date, title=title
    )
    html_path = out_dir / f"digest_{iso_date(end_date)}.html"
    html_path.write_text(html_body, encoding="utf-8")
    print(f"[PIPELINE] Saved HTML to: {html_path}")

    # Step 4: Email (left to your existing SMTP code; you can copy from your working version)
    if no_email:
        print("[PIPELINE] Email sending disabled (--no-email).")
        return

    # If you already have a working SMTP sender in your previous script, keep using it.
    # We deliberately avoid re-implementing SMTP here to reduce the risk of breaking your working setup.
    try:
        from email.message import EmailMessage
        import smtplib

        sender = os.getenv("EMAIL_SENDER", "")
        password = os.getenv("EMAIL_PASSWORD", "")
        receiver = os.getenv("EMAIL_RECEIVER", "")
        if not sender or not password or not receiver:
            print(
                "[EMAIL][WARN] Missing EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER in .env; skipping email."
            )
            return

        msg = EmailMessage()
        msg["Subject"] = title
        msg["From"] = sender
        msg["To"] = receiver
        msg.set_content(
            "HTML digest attached. If you can't view HTML, open it in a browser."
        )

        # Attach HTML + raw CSV
        msg.add_attachment(
            html_body.encode("utf-8"),
            maintype="text",
            subtype="html",
            filename=html_path.name,
        )
        msg.add_attachment(
            raw_csv_path.read_bytes(),
            maintype="text",
            subtype="csv",
            filename=raw_csv_path.name,
        )
        if triaged_csv_path.exists():
            msg.add_attachment(
                triaged_csv_path.read_bytes(),
                maintype="text",
                subtype="csv",
                filename=triaged_csv_path.name,
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("[EMAIL] Sent successfully (with attachments).")

    except Exception as e:
        print(f"[EMAIL][WARN] Email sending failed: {e}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    no_email = "--no-email" in sys.argv
    run_pipeline(script_dir, no_email=no_email)


if __name__ == "__main__":
    main()
