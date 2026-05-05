"""
FactCheck.kg Kyrgyz Fake News Scraper
======================================
Scrapes fact-checked articles from factcheck.kg and extracts the
ORIGINAL FAKE CLAIM being debunked, not the fact-checker's commentary.

Label: 1 (Fake News)
Target: 200–300 articles

Strategy:
  1. Crawl /category/factcheck/ pagination to collect article URLs
  2. For each article: extract (a) the original fake claim/headline,
     (b) the full article body, (c) the verdict
  3. Verdict is extracted from Kyrgyz keywords (жалган, чын, etc.)
     AND Russian keywords (as factcheck.kg publishes in both languages)
  4. Only keep articles with a clear FAKE (label=1) verdict

Language note:
  factcheck.kg publishes some articles in Kyrgyz and some in Russian.
  Both are included — but we add a `language` field so you can filter.
  For a purely Kyrgyz dataset, filter where language='kyrgyz'.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
import logging
import re
from urllib.parse import urljoin

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [factcheck] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("scraper_factcheck.log"), logging.StreamHandler()],
)
log = logging.getLogger("factcheck")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL     = "https://factcheck.kg"
CATEGORY_URL = "https://factcheck.kg/ky/category/factcheck/"
TARGET       = 300       # Aim for 300; keep all confirmed-fake ones
MAX_PAGES    = 100       # factcheck.kg has many archived pages
DELAY        = 4.0       # Be very respectful — this is a small NGO site

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8",
    "Referer": "https://factcheck.kg/",
}


# ── Verdict extraction ────────────────────────────────────────────────────────

# Kyrgyz fake/false indicators
KY_FAKE_KEYWORDS = [
    "жалган",           # false/fake
    "туура эмес",       # incorrect
    "ырасталган жок",   # not confirmed
    "далилденген жок",  # not proven
    "туура эмес маалымат",  # incorrect information
    "жаңылыш",          # wrong/erroneous
    "бурмаланган",      # distorted
    "жасалма",          # fabricated/artificial
    "фейк",
    "чындыкка жатпайт",
    "калп",
    "калп."
]

# Kyrgyz true/real indicators
KY_TRUE_KEYWORDS = [
    "чын",              # true
    "туура",            # correct
    "ырасталды",        # confirmed
    "далилденди",       # proven
    "чындык",           # truth
]

# Russian fake indicators
RU_FAKE_KEYWORDS = [
    "неправда", "ложь", "ложное", "ложью", "выдумка",
    "вымышленное", "не соответствует действительности",
    "не подтверждается", "ошибочное", "заблуждение",
    "миф", "мифом", "фейк", "дезинформация",
    "манипуляция", "недостоверно",
]

# Russian true indicators
RU_TRUE_KEYWORDS = [
    "правда", "истина", "подтверждается",
    "соответствует действительности",
    "верно", "истинно", "правдиво",
]


def detect_language(text: str) -> str:
    """Detect if text is primarily Kyrgyz or Russian based on character/word patterns."""
    # Kyrgyz-specific letters not in Russian
    kyrgyz_specific = set("өүңғ")
    kyrgyz_count = sum(1 for c in text.lower() if c in kyrgyz_specific)
    # If more than 5 Kyrgyz-specific chars → likely Kyrgyz
    if kyrgyz_count >= 5:
        return "kyrgyz"
    return "russian"


def extract_verdict(text: str, conclusion: str) -> tuple[int, float]:
    """
    Extract verdict from article text.
    Returns: (label, confidence)
      label: 1=fake, 0=real, -1=uncertain
      confidence: 0.0–1.0
    """
    search_text = (conclusion + " " + text[:3000]).lower()

    # Count keyword hits
    ky_fake  = sum(search_text.count(kw) for kw in KY_FAKE_KEYWORDS)
    ky_true  = sum(search_text.count(kw) for kw in KY_TRUE_KEYWORDS)
    ru_fake  = sum(search_text.count(kw) for kw in RU_FAKE_KEYWORDS)
    ru_true  = sum(search_text.count(kw) for kw in RU_TRUE_KEYWORDS)

    fake_score = ky_fake + ru_fake
    true_score = ky_true + ru_true

    if fake_score > true_score and fake_score > 0:
        conf = min(fake_score / max(fake_score + true_score, 1), 1.0)
        return 1, round(conf, 2)
    elif true_score > fake_score and true_score > 0:
        conf = min(true_score / max(fake_score + true_score, 1), 1.0)
        return 0, round(conf, 2)
    else:
        return -1, 0.0


def extract_conclusion(soup: BeautifulSoup, full_text: str) -> str:
    """
    Extract the verdict/conclusion section from the article.
    factcheck.kg typically has a conclusion paragraph with bold text.
    """
    # Strategy 1: Look for conclusion section by heading text
    conclusion_markers_ky = ["жыйынтык", "корутунду", "баа"]
    conclusion_markers_ru = ["вывод", "заключение", "итог", "результат"]

    for tag in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
        tag_text = tag.get_text(strip=True).lower()
        if any(m in tag_text for m in conclusion_markers_ky + conclusion_markers_ru):
            # Get the next sibling paragraph(s)
            sibling = tag.find_next_sibling()
            if sibling:
                return sibling.get_text(strip=True)[:800]

    # Strategy 2: Last 1000 chars of the article (usually contains conclusion)
    return full_text[-1000:] if len(full_text) > 1000 else full_text


def extract_original_claim(soup: BeautifulSoup, headline: str, full_text: str) -> str:
    """
    Extract the original FAKE CLAIM being debunked.
    factcheck.kg often quotes the original claim in blockquotes or special divs.
    The headline itself is usually the fake claim restatement.
    """
    # Look for blockquote — often contains the original fake claim
    blockquote = soup.find("blockquote")
    if blockquote:
        claim = blockquote.get_text(strip=True)
        if len(claim) > 30:
            return claim

    # Look for a special "claim" div
    for cls in ["claim", "fake-claim", "statement", "teza"]:
        div = soup.find("div", class_=cls)
        if div:
            return div.get_text(strip=True)

    # Fallback: headline is typically the fake claim on factcheck.kg
    return headline


# ── Link collection ───────────────────────────────────────────────────────────
def get_article_links(max_pages: int = MAX_PAGES) -> list[str]:
    """Collect all article links from factcheck.kg/category/factcheck/ pagination."""
    links: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{CATEGORY_URL}page/{page}/" if page > 1 else CATEGORY_URL
        log.info(f"Fetching listing page {page}: {url}")

        try:
            resp = requests.get(url, timeout=20, headers=HEADERS)

            # 404 = past the last page
            if resp.status_code == 404:
                log.info(f"  404 on page {page} — end of pagination.")
                break
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            found = 0

            # Primary: <article> tags with links
            for article in soup.find_all("article"):
                a = article.find("a", href=True)
                if a:
                    href = a["href"]
                    full = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    if "factcheck.kg" in full and full not in links and full != CATEGORY_URL:
                        links.add(full)
                        found += 1

            # Fallback: any link that looks like a post
            if found == 0:
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    full = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    if (
                        "factcheck.kg" in full
                        and "/category/" not in full
                        and "/tag/" not in full
                        and "/page/" not in full
                        and full != BASE_URL
                        and full not in links
                        and len(full) > len(BASE_URL) + 5
                    ):
                        links.add(full)
                        found += 1

            log.info(f"  → {found} new links (total: {len(links)})")

            if found == 0:
                log.info("  No new links — stopping pagination.")
                break

            time.sleep(1.5)

        except Exception as exc:
            log.warning(f"Page {page} error: {exc}")
            break

    log.info(f"Total article links: {len(links)}")
    return list(links)


# ── Article scraper ───────────────────────────────────────────────────────────
def scrape_article(url: str) -> dict | None:
    """Scrape a single factcheck.kg article and extract verdict."""
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Headline (= the fake claim title) ────────────────────────────────
        h1 = soup.find("h1") or soup.find("h2", class_="entry-title")
        headline = h1.get_text(strip=True) if h1 else ""
        if not headline:
            return None

        # ── Full article body ─────────────────────────────────────────────────
        content = (
            soup.find("div", class_="entry-content")
            or soup.find("article")
            or soup.find("div", class_="post-content")
            or soup.find("div", class_="content")
        )
        if not content:
            return None

        full_text = content.get_text(separator=" ", strip=True)
        if len(full_text) < 150:
            return None

        # ── Conclusion / verdict section ──────────────────────────────────────
        conclusion = extract_conclusion(soup, full_text)

        # ── Verdict ───────────────────────────────────────────────────────────
        label, confidence = extract_verdict(full_text, conclusion)

        # ── Original fake claim ───────────────────────────────────────────────
        original_claim = extract_original_claim(soup, headline, full_text)

        # ── Language detection ────────────────────────────────────────────────
        lang = detect_language(full_text)

        # ── Date ──────────────────────────────────────────────────────────────
        date_tag = soup.find("time") or soup.find("span", class_="date") or soup.find("div", class_="date")
        date_str = (
            date_tag.get("datetime") or date_tag.get_text(strip=True)
        ) if date_tag else None

        # ── Evidence URL (factcheck source link) ─────────────────────────────
        # factcheck.kg usually links to the source being debunked
        evidence_url = url   # The factcheck article itself is the evidence

        return {
            "id":               str(uuid.uuid4()),
            "headline":         original_claim,     # The FAKE claim as headline
            "body_text":        full_text[:3000],   # First 3000 chars of article
            "conclusion":       conclusion[:500],
            "source":           "factcheck.kg",
            "url":              url,
            "date":             date_str,
            "language":         lang,
            "label":            label,              # 1=fake, 0=real, -1=uncertain
            "label_confidence": confidence,
            "evidence_url":     evidence_url,
            "source_type":      "fact_check",
        }

    except Exception as exc:
        log.warning(f"Failed {url}: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("FACTCHECK.KG SCRAPER")
    log.info("=" * 60)

    links   = get_article_links(max_pages=MAX_PAGES)
    all_articles = []

    for link in links:
        log.info(f"Scraping ({len(all_articles)+1}): {link}")
        record = scrape_article(link)
        if record:
            verdict_str = {1: "FAKE", 0: "REAL", -1: "UNCERTAIN"}.get(record["label"], "?")
            log.info(f"  [{verdict_str}] {record['headline'][:65]}")
            all_articles.append(record)
        else:
            log.info("  ✗ Skipped")
        time.sleep(DELAY)

    df_all = pd.DataFrame(all_articles)
    log.info(f"\nTotal scraped: {len(df_all)}")

    if df_all.empty:
        log.error("No articles scraped. The site may be blocking requests.")
        return df_all

    # ── Save full results (all verdicts) ──────────────────────────────────────
    df_all.to_csv("factcheck_all.csv", index=False, encoding="utf-8-sig")
    log.info(f"Saved all {len(df_all)} articles → factcheck_all.csv")

    # ── Save FAKE ONLY (label=1) — the primary dataset for training ───────────
    df_fake = df_all[df_all["label"] == 1].copy()
    df_fake.to_csv("factcheck_fake_news.csv", index=False, encoding="utf-8-sig")
    log.info(f"Saved {len(df_fake)} FAKE articles → factcheck_fake_news.csv")

    # ── Save Kyrgyz-language fake news separately ─────────────────────────────
    df_fake_ky = df_fake[df_fake["language"] == "kyrgyz"].copy()
    df_fake_ky.to_csv("factcheck_fake_kyrgyz.csv", index=False, encoding="utf-8-sig")
    log.info(f"Saved {len(df_fake_ky)} Kyrgyz-language fake articles → factcheck_fake_kyrgyz.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Total articles: {len(df_all)}")
    log.info(f"  FAKE    (1): {(df_all['label'] == 1).sum()}")
    log.info(f"  REAL    (0): {(df_all['label'] == 0).sum()}")
    log.info(f"  UNCERTAIN(-1): {(df_all['label'] == -1).sum()}")
    log.info(f"Language breakdown of FAKE:")
    if not df_fake.empty:
        log.info(f"  Kyrgyz:  {(df_fake['language'] == 'kyrgyz').sum()}")
        log.info(f"  Russian: {(df_fake['language'] == 'russian').sum()}")

    return df_fake


if __name__ == "__main__":
    main()
