import datetime
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
}


def truncate_text(text: str, max_chars: int = 800) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def parse_onemine_date(date_text: str):
    """
    OneMine displays dates like: 'Sep 1, 2024'
    We convert them to datetime.date objects.
    """
    if not isinstance(date_text, str):
        return None
    date_text = date_text.strip()
    for fmt in ("%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.datetime.strptime(date_text, fmt).date()
        except ValueError:
            continue
    return None


def fetch_onemine_page(org: str, page: int = 1):
    """
    Fetch ONE page from OneMine for a given organization.
    Returns a list of raw item dicts (no date filtering yet).
    """
    base_url = "https://www.onemine.org/search"
    params = {
        "Organization": org,
        "SortBy": "MostRecent",
        "page": page,
    }
    print(f"[TEST] Fetching OneMine org={org}, page={page} ...")
    r = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
    print(f"[TEST] HTTP status: {r.status_code}")
    if r.status_code != 200:
        return []

    soup = BeautifulSoup(r.text, "html.parser")

    # This comes directly from the HTML you pasted:
    # <ul class="item-list ...">
    #   <li class="item-list__item"> ... </li>
    # </ul>
    results = soup.select("ul.item-list li.item-list__item")
    print(f"[TEST] Found {len(results)} list items on the page.")

    items = []

    for li in results:
        # Title and URL
        title_el = li.select_one("a.item-list__title")
        if not title_el or not title_el.get("href"):
            continue

        title = title_el.get_text(strip=True)
        href = title_el["href"]
        if href.startswith("/"):
            url = "https://www.onemine.org" + href
        else:
            url = href

        # Snippet / abstract: we use the hidden <span class="item-list__description">
        # If it's not there, fall back to the visible <p>.
        snippet_span = li.select_one("span.item-list__description")
        if snippet_span:
            snippet = snippet_span.get_text(" ", strip=True)
        else:
            # fallback: first <p> in the content
            first_p = li.select_one("div.item-list__content p")
            snippet = (
                first_p.get_text(" ", strip=True) if first_p else "No summary provided."
            )

        snippet = truncate_text(snippet, max_chars=800)

        # Authors + date:
        # HTML has two <p class="item-list__date">:
        #  - first: "By X"
        #  - second: "Sep 1, 2024"
        date_els = li.select("p.item-list__date")
        authors = ""
        pub_date = None

        if date_els:
            # First one usually has "By ..."
            authors_text = date_els[0].get_text(strip=True)
            if authors_text.lower().startswith("by"):
                authors = authors_text

            # Last one should be the actual date
            date_text = date_els[-1].get_text(strip=True)
            pub_date = parse_onemine_date(date_text)

        items.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "authors": authors,
                "pub_date": pub_date,
            }
        )

    return items


def main():
    # For the test, just pull one page of AUSIMM
    org = "AUSIMM"
    items = fetch_onemine_page(org, page=1)
    print(f"\n[TEST] Total items parsed: {len(items)}\n")

    for it in items[:5]:
        print("TITLE:", it["title"])
        print("URL:  ", it["url"])
        print("AUTH: ", it["authors"])
        print("DATE: ", it["pub_date"])
        print("SNIP: ", truncate_text(it["snippet"], 200))
        print("-" * 80)
        time.sleep(0.1)  # just so your console is readable


if __name__ == "__main__":
    main()
