"""
24.kg Kyrgyz News Scraper
Scrapes Kyrgyz-language articles from 24.kg/kyrgyzcha/
Label: 0 (Real News)
Target: 150 articles
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
    format="%(asctime)s [24kg] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/scraper_24kg.log"), logging.StreamHandler()],
)
log = logging.getLogger("24kg")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL      = "https://24.kg"
CATEGORY_URL  = "https://24.kg/kyrgyzcha/"          # Kyrgyz-only section
TARGET        = 150
DELAY         = 2.5                                  # seconds between article requests
MAX_PAGES     = 20                                   # pagination pages to crawl

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8",
}


# ── Link collection ───────────────────────────────────────────────────────────
def get_article_links(max_pages: int = MAX_PAGES) -> list[str]:
    """Collect unique article URLs from the Kyrgyz category pages."""
    links: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{CATEGORY_URL}page_{page}/"
        log.info(f"Fetching page {page}: {url}")
        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                # Pattern: /kyrgyzcha/123456_slug/ — must contain digits
                if "/kyrgyzcha/" in href and any(c.isdigit() for c in href):
                    full = urljoin(BASE_URL, href)
                    # Exclude category-level pages (no digits after last slash)
                    if full != CATEGORY_URL:
                        links.add(full)

            log.info(f"  → Running total: {len(links)} unique links")
            time.sleep(1)

        except Exception as exc:
            log.warning(f"Page {page} error: {exc}")

    log.info(f"Total links collected: {len(links)}")
    return list(links)


# ── Article scraper ───────────────────────────────────────────────────────────
def scrape_article(url: str) -> dict | None:
    """Scrape a single 24.kg article and return a structured record."""
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Headline ──────────────────────────────────────────────────────────
        # 24.kg uses <h1 class="text-big"> or a plain <h1>
        h1 = (
            soup.find("h1", class_="text-big")
            or soup.find("h1", class_="head")
            or soup.find("h1")
        )
        headline = h1.get_text(strip=True) if h1 else ""
        if not headline:
            return None                              # skip if no title

        # ── Body text ─────────────────────────────────────────────────────────
        # Priority order of known selectors on 24.kg
        content = (
            soup.find("div", class_="text")
            or soup.find("div", itemprop="articleBody")
            or soup.find("div", class_="article__text")
            or soup.find("article")
        )
        if not content:
            return None

        paragraphs = [p.get_text(strip=True) for p in content.find_all("p") if p.get_text(strip=True)]
        body_text = " ".join(paragraphs)

        # Require a minimum body length to filter out stubs
        if len(body_text) < 100:
            return None

        # ── Date ──────────────────────────────────────────────────────────────
        date_tag = (
            soup.find("time")
            or soup.find("span", class_="date")
            or soup.find("div", class_="date")
        )
        date_str = date_tag.get_text(strip=True) if date_tag else None

        return {
            "id":               str(uuid.uuid4()),
            "headline":         headline,
            "body_text":        body_text,
            "source":           "24.kg",
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
    log.info("24.KG KYRGYZ SCRAPER")
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
            log.info("  ✗ Skipped (no content or too short)")
        time.sleep(DELAY)

    df = pd.DataFrame(results)
    out = "data/24kg_kyrgyz_data.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"\nSaved {len(df)} articles → {out}")
    return df


if __name__ == "__main__":
    main()
