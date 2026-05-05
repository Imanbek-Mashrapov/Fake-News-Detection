"""
Kaktus.media Kyrgyz News Scraper
Scrapes Kyrgyz-language articles from kaktus.media
Label: 0 (Real News)
Target: 150 articles

Kaktus.media is a high-quality Kyrgyz news portal with broad coverage.
This is a new source to diversify your "real news" set beyond Azattyk/24kg/Sputnik.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
import logging
from urllib.parse import urljoin, urlparse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [kaktus] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("scraper_kaktus.log"), logging.StreamHandler()],
)
log = logging.getLogger("kaktus")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL  = "https://kaktus.media"
# Kaktus.media has a /ky/ section for Kyrgyz-language content
KY_URL    = "https://kaktus.media/ky/"
TARGET    = 150
MAX_PAGES = 20
DELAY     = 3.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8",
    "Referer": "https://kaktus.media/",
}


# ── Link collection ───────────────────────────────────────────────────────────
def get_article_links(max_pages: int = MAX_PAGES) -> list[str]:
    """Collect Kyrgyz article links from kaktus.media/ky/ pagination."""
    links: set[str] = set()

    for page in range(1, max_pages + 1):
        # kaktus.media uses ?page=N for pagination
        url = f"{KY_URL}?page={page}" if page > 1 else KY_URL
        log.info(f"Fetching page {page}: {url}")

        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            found = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full = urljoin(BASE_URL, href)
                parsed = urlparse(full)

                # Article URLs on kaktus contain /doc/ or a numeric ID pattern
                # and are under the same domain
                if (
                    "kaktus.media" in parsed.netloc
                    and (
                        "/doc/" in parsed.path
                        or "/ky/" in parsed.path
                    )
                    and any(c.isdigit() for c in parsed.path)
                    and full not in links
                    and full != KY_URL
                ):
                    links.add(full)
                    found += 1

            log.info(f"  → {found} new links (total: {len(links)})")

            if found == 0:
                log.info("No new links — stopping.")
                break

            time.sleep(1)

        except Exception as exc:
            log.warning(f"Page {page} error: {exc}")

    log.info(f"Total links: {len(links)}")
    return list(links)


# ── Article scraper ───────────────────────────────────────────────────────────
def scrape_article(url: str) -> dict | None:
    """Scrape a single kaktus.media article."""
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Headline ──────────────────────────────────────────────────────────
        h1 = soup.find("h1") or soup.find("h2", class_="article-title")
        headline = h1.get_text(strip=True) if h1 else ""
        if not headline:
            return None

        # ── Body text ─────────────────────────────────────────────────────────
        content = (
            soup.find("div", class_="article-body")
            or soup.find("div", class_="article__content")
            or soup.find("div", class_="text-content")
            or soup.find("div", class_="content")
            or soup.find("article")
        )
        if not content:
            return None

        paragraphs = [p.get_text(strip=True) for p in content.find_all("p") if p.get_text(strip=True)]
        body_text = " ".join(paragraphs)

        if len(body_text) < 100:
            return None

        # ── Date ──────────────────────────────────────────────────────────────
        date_tag = soup.find("time") or soup.find("span", class_="date")
        date_str = (
            date_tag.get("datetime") or date_tag.get_text(strip=True)
        ) if date_tag else None

        return {
            "id":               str(uuid.uuid4()),
            "headline":         headline,
            "body_text":        body_text,
            "source":           "kaktus.media",
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
    log.info("KAKTUS.MEDIA KYRGYZ SCRAPER")
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
    out = "kaktus_kyrgyz_data.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"\nSaved {len(df)} articles → {out}")
    return df


if __name__ == "__main__":
    main()
