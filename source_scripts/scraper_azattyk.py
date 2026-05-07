"""
Azattyk.org Kyrgyz News Scraper
Scrapes Kyrgyz-language articles from azattyk.org
Label: 0 (Real News)
Target: 150 articles

Azattyk (Radio Free Europe/Radio Liberty Kyrgyz Service) publishes
high-quality journalism strictly in Kyrgyz. This is a premium real-news source.
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
    format="%(asctime)s [azattyk] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/scraper_azattyk.log"), logging.StreamHandler()],
)
log = logging.getLogger("azattyk")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL     = "https://www.azattyk.org"
# /kg/ is the Kyrgyz-language news listing; /a/ URLs are individual articles
NEWS_URL     = "https://www.azattyk.org/news"
TARGET       = 250
MAX_PAGES    = 50
DELAY        = 3.0    # azattyk can be slow; respect the server

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
    """
    Collect unique article URLs from azattyk.org.
    The site uses ?p=N pagination and individual articles are at /a/*.html
    """
    links: set[str] = set()

    for page in range(0, max_pages):
        url = f"{NEWS_URL}?p={page}"
        log.info(f"Fetching listing page {page}: {url}")
        try:
            resp = requests.get(url, timeout=20, headers=HEADERS)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            found_on_page = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # Azattyk article URLs: /a/NNNNNNN.html
                if href.startswith("/a/") and href.endswith(".html"):
                    full = urljoin(BASE_URL, href)
                    if full not in links:
                        links.add(full)
                        found_on_page += 1

            log.info(f"  → Found {found_on_page} new links (total: {len(links)})")

            if found_on_page == 0:
                log.info("  No new links on this page — stopping pagination.")
                break

            time.sleep(1.5)

        except Exception as exc:
            log.warning(f"Page {page} error: {exc}")

    log.info(f"Total article links: {len(links)}")
    return list(links)


# ── Article scraper ───────────────────────────────────────────────────────────
def scrape_article(url: str) -> dict | None:
    """Scrape a single azattyk.org article."""
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Headline ──────────────────────────────────────────────────────────
        h1 = (
            soup.find("h1", class_="title")
            or soup.find("h1", class_="article-title")
            or soup.find("h1")
        )
        headline = h1.get_text(strip=True) if h1 else ""
        if not headline:
            return None

        # ── Body text ─────────────────────────────────────────────────────────
        # Azattyk wraps article body in div.wsw or div.body-text
        content = (
            soup.find("div", class_="wsw")
            or soup.find("div", class_="body-text")
            or soup.find("div", class_="article__body")
            or soup.find("article")
        )
        if not content:
            return None

        paragraphs = [p.get_text(strip=True) for p in content.find_all("p") if p.get_text(strip=True)]
        body_text = " ".join(paragraphs)

        if len(body_text) < 100:
            return None

        # ── Date ──────────────────────────────────────────────────────────────
        date_tag = (
            soup.find("time")
            or soup.find("div", class_="published")
            or soup.find("span", class_="date")
        )
        date_str = (date_tag.get("datetime") or date_tag.get_text(strip=True)) if date_tag else None

        # ── Language check — confirm article is in Kyrgyz ────────────────────
        # Azattyk also has Russian content; Kyrgyz articles always have /a/ URLs
        # and the page's <html lang> will be "ky"
        html_tag  = soup.find("html")
        lang_attr = html_tag.get("lang", "") if html_tag else ""
        # Accept if lang is "ky" or empty (some pages omit it)
        if lang_attr and lang_attr not in ("ky", ""):
            log.info(f"  Skipping non-Kyrgyz article (lang={lang_attr}): {url}")
            return None

        return {
            "id":               str(uuid.uuid4()),
            "headline":         headline,
            "body_text":        body_text,
            "source":           "azattyk.org",
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
    log.info("AZATTYK.ORG KYRGYZ SCRAPER")
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
            log.info("  ✗ Skipped")
        time.sleep(DELAY)

    df = pd.DataFrame(results)
    out = "data/azattyk_kyrgyz_data.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"\nSaved {len(df)} articles → {out}")
    return df


if __name__ == "__main__":
    main()
