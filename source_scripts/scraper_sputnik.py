"""
Sputnik.kg Kyrgyz News Scraper
Scrapes Kyrgyz-language articles from sputnik.kg
Label: 0 (Real News)
Target: 150 articles

Key fix from v1: Sputnik's structure changed. This version tries multiple
selector strategies and validates Kyrgyz URLs by checking the domain path.
Sputnik.kg (not ru.sputnik.kg) is the Kyrgyz-language version.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
import logging
from urllib.parse import urljoin

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sputnik] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/scraper_sputnik.log"), logging.StreamHandler()],
)
log = logging.getLogger("sputnik")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL     = "https://sputnik.kg"
CATEGORY_URL = "https://sputnik.kg/news/"
TARGET       = 250
MAX_PAGES    = 45
DELAY        = 5.0    # Sputnik is strict; 5s avoids rate-limiting

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8",
    "Referer": "https://sputnik.kg/",
}


# ── Link collection ───────────────────────────────────────────────────────────
def get_article_links(max_pages: int = MAX_PAGES) -> list[str]:
    """
    Collect Kyrgyz-language article URLs from sputnik.kg.

    Filtering rules:
      - Must end with .html (article pages)
      - Must contain /202 (year-based path — recent content)
      - Must NOT contain 'ru.sputnik.kg' (Russian site)
      - Must NOT contain '/sport/' or '/turizm/' (non-news verticals) — optional
    """
    links: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{CATEGORY_URL}?page={page}"
        log.info(f"Fetching page {page}: {url}")
        try:
            resp = requests.get(url, timeout=20, headers=HEADERS)
            if resp.status_code != 200:
                log.warning(f"  HTTP {resp.status_code} on page {page}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            found = 0

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if (
                    href.endswith(".html")
                    and "/202" in href
                    and "ru.sputnik.kg" not in href
                    and "sputnik.kg" not in href.split("/202")[0].replace("https://sputnik.kg", "")
                ):
                    full = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    # Keep only sputnik.kg (not subdomains)
                    if full.startswith("https://sputnik.kg/") or full.startswith("http://sputnik.kg/"):
                        if full not in links:
                            links.add(full)
                            found += 1

            log.info(f"  → {found} new links (total: {len(links)})")
            time.sleep(1.5)

        except Exception as exc:
            log.warning(f"Page {page} error: {exc}")

    log.info(f"Total article links: {len(links)}")
    return list(links)


# ── Article scraper ───────────────────────────────────────────────────────────
def scrape_article(url: str) -> dict | None:
    """
    Scrape a single sputnik.kg article.
    Tries multiple selector strategies in priority order.
    """
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        if resp.status_code != 200:
            return None

        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Headline ──────────────────────────────────────────────────────────
        h1 = (
            soup.find("h1", class_="article__title")
            or soup.find("h1", class_="b-article__title")
            or soup.find("h1")
        )
        headline = h1.get_text(strip=True) if h1 else ""
        if not headline:
            return None

        # ── Body text — multi-strategy ────────────────────────────────────────
        body_text = ""

        # Strategy 1: article__body with article__block divs (original approach)
        content_body = soup.find("div", class_="article__body")
        if content_body:
            blocks = content_body.find_all("div", class_="article__block")
            if blocks:
                seen = set()
                parts = []
                for block in blocks:
                    txt = block.get_text(strip=True)
                    if txt and txt not in seen:
                        seen.add(txt)
                        parts.append(txt)
                body_text = " ".join(parts)

        # Strategy 2: fallback to paragraph extraction inside article__body
        if not body_text and content_body:
            paras = [p.get_text(strip=True) for p in content_body.find_all("p") if p.get_text(strip=True)]
            body_text = " ".join(paras)

        # Strategy 3: b-article__text or article__text class
        if not body_text:
            fallback = (
                soup.find("div", class_="b-article__text")
                or soup.find("div", class_="article__text")
                or soup.find("article")
            )
            if fallback:
                paras = [p.get_text(strip=True) for p in fallback.find_all("p") if p.get_text(strip=True)]
                body_text = " ".join(paras)

        if len(body_text) < 300:
            return None          # too short — stub, ad page, or video-only

        # ── Date ──────────────────────────────────────────────────────────────
        date_tag = (
            soup.find("div", class_="article__info-date")
            or soup.find("time")
            or soup.find("div", class_="date")
        )
        date_str = date_tag.get_text(strip=True) if date_tag else None

        return {
            "id":               str(uuid.uuid4()),
            "headline":         headline,
            "body_text":        body_text,
            "source":           "sputnik.kg",
            "url":              url,
            "date":             date_str,
            "language":         "kyrgyz",
            "label":            0,           # Real news
            "label_confidence": "high",
            "source_type":      "news_site",
        }

    except Exception as exc:
        log.warning(f"Failed {url}: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("SPUTNIK.KG KYRGYZ SCRAPER")
    log.info("=" * 60)

    links   = get_article_links(max_pages=MAX_PAGES)
    results = []

    for link in links:
        if len(results) >= TARGET:
            break
        log.info(f"Scraping ({len(results)+1}/{TARGET}): {link}")
        record = scrape_article(link)
        if record:
            results.append(record)
            log.info(f"  ✓ {record['headline'][:70]}")
        else:
            log.info("  ✗ Skipped (too short or no content)")
        time.sleep(DELAY)

    df = pd.DataFrame(results)
    out = "data/sputnik_kyrgyz_data.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"\nSaved {len(df)} articles → {out}")
    return df


if __name__ == "__main__":
    main()
