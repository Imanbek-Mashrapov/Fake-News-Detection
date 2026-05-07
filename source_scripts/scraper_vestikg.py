"""
Vesti.kg Kyrgyz News Scraper
==============================
Scrapes Kyrgyz-language articles from vesti.kg/kg/
Label: 0 (Real News)
Target: 150 articles

Vesti.kg is a well-established Kyrgyz news portal with broad coverage
of politics, economy, and social issues.
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
    format="%(asctime)s [vestikg] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/scraper_vestikg.log"), logging.StreamHandler()],
)
log = logging.getLogger("vestikg")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL  = "https://vesti.kg"
KY_URL    = "https://vesti.kg/kg/"        # Kyrgyz-language section
TARGET    = 150
MAX_PAGES = 30
DELAY     = 2.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8",
    "Referer": "https://vesti.kg/",
}


# ── Kyrgyz language detection ─────────────────────────────────────────────────
def is_kyrgyz(text: str) -> bool:
    """Quick heuristic: Kyrgyz-specific characters not found in Russian."""
    kyrgyz_specific = set("өүңғ")
    return sum(1 for c in text.lower() if c in kyrgyz_specific) >= 3


# ── Link collection ───────────────────────────────────────────────────────────
def get_article_links(max_pages: int = MAX_PAGES) -> list[str]:
    """
    Collect Kyrgyz article links from vesti.kg/kg/ with pagination.
    vesti.kg uses ?PAGEN_1=N for pagination.
    """
    links: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{KY_URL}?PAGEN_1={page}" if page > 1 else KY_URL
        log.info(f"Fetching page {page}: {url}")

        try:
            resp = requests.get(url, timeout=20, headers=HEADERS)
            if resp.status_code == 404:
                log.info(f"  404 on page {page} — end of pagination.")
                break
            resp.raise_for_status()
            resp.encoding = "utf-8"

            soup = BeautifulSoup(resp.text, "html.parser")
            found = 0

            for a in soup.find_all("a", href=True):
                href = a["href"]
                full = urljoin(BASE_URL, href)
                parsed = urlparse(full)

                # Article URLs: must be under /kg/ and have numeric ID
                if (
                    "vesti.kg" in parsed.netloc
                    and "/kg/" in parsed.path
                    and any(c.isdigit() for c in parsed.path)
                    and full != KY_URL
                    and "?PAGEN" not in full
                    and full not in links
                    # Exclude category/tag pages
                    and parsed.path.count("/") >= 3
                ):
                    links.add(full)
                    found += 1

            log.info(f"  → {found} new links (total: {len(links)})")

            if found == 0:
                log.info("  No new links — stopping pagination.")
                break

            time.sleep(1)

        except Exception as exc:
            log.warning(f"Page {page} error: {exc}")

    # Also try category sub-pages for broader coverage
    categories = [
        "https://vesti.kg/kg/politics/",
        "https://vesti.kg/kg/society/",
        "https://vesti.kg/kg/economics/",
    ]
    for cat_url in categories:
        try:
            resp = requests.get(cat_url, timeout=20, headers=HEADERS)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full = urljoin(BASE_URL, href)
                parsed = urlparse(full)
                if (
                    "vesti.kg" in parsed.netloc
                    and "/kg/" in parsed.path
                    and any(c.isdigit() for c in parsed.path)
                    and full not in links
                    and parsed.path.count("/") >= 3
                ):
                    links.add(full)
            time.sleep(1)
        except Exception as exc:
            log.warning(f"Category {cat_url} error: {exc}")

    log.info(f"Total article links collected: {len(links)}")
    return list(links)


# ── Article scraper ───────────────────────────────────────────────────────────
def scrape_article(url: str) -> dict | None:
    """Scrape a single vesti.kg article."""
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Headline ──────────────────────────────────────────────────────────
        h1 = (
            soup.find("h1", class_="detail_text_title")
            or soup.find("h1", class_="article-title")
            or soup.find("h1", class_="title")
            or soup.find("h1")
        )
        headline = h1.get_text(strip=True) if h1 else ""
        if not headline:
            return None

        # ── Body text — multiple selector strategies ───────────────────────────
        content = (
            soup.find("div", class_="itemBody")
            or soup.find("div", class_="detail_text")
            or soup.find("div", class_="article-body")
            or soup.find("div", class_="news-text")
            or soup.find("div", itemprop="articleBody")
            or soup.find("article")
        )
        if not content:
            return None

        paragraphs = [p.get_text(strip=True) for p in content.find_all("p") if p.get_text(strip=True)]
        body_text = " ".join(paragraphs)

        # Fallback: get all text from the content div if no <p> tags found
        if len(body_text) < 100:
            body_text = content.get_text(separator=" ", strip=True)

        if len(body_text) < 100:
            return None

        # ── Language check: must be Kyrgyz ────────────────────────────────────
        if not is_kyrgyz(headline + " " + body_text[:500]):
            log.info(f"  Skipping non-Kyrgyz article: {url}")
            return None

        # ── Date ──────────────────────────────────────────────────────────────
        date_tag = (
            soup.find("time")
            or soup.find("span", class_="date")
            or soup.find("div", class_="detail_date")
            or soup.find("div", class_="article-date")
        )
        date_str = None
        if date_tag:
            date_str = date_tag.get("datetime") or date_tag.get_text(strip=True)

        return {
            "id":               str(uuid.uuid4()),
            "headline":         headline,
            "body_text":        body_text,
            "source":           "vesti.kg",
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
    log.info("VESTI.KG KYRGYZ SCRAPER")
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
            log.info("  ✗ Skipped (no Kyrgyz content / too short)")
        time.sleep(DELAY)

    df = pd.DataFrame(results)
    out = "data/vestikg_kyrgyz_data.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"\nSaved {len(df)} articles → {out}")
    return df


if __name__ == "__main__":
    main()