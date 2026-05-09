import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
import logging
import re
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [factcheck] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/scraper_factcheck.log"), logging.StreamHandler()],
)
log = logging.getLogger("factcheck")


BASE_URL     = "https://factcheck.kg"
CATEGORY_URL = "https://factcheck.kg/ky/category/factcheck/"
SCRAPE_TARGET = 400      
TARGET_FAKE  = 500       
MAX_PAGES    = 200
DELAY        = 4.0

MIN_BODY_LEN = 80
SPLIT_MIN_LEN = 300
BT_MIN_LEN   = 80

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8",
    "Referer": "https://factcheck.kg/",
}

STOP_PHRASES = [
    "редакциясы аныктады",
    "редакция аныктады",
    "выяснила редакция",
    "проверила редакция",
    "редакциябыз аныктады",
    "текшерип аныктады",
]

KY_FAKE_KEYWORDS = [
    "жалган", "туура эмес", "ырасталган жок", "далилденген жок",
    "туура эмес маалымат", "жаңылыш", "бурмаланган", "жасалма",
    "фейк", "чындыкка жатпайт", "калп",
]
KY_TRUE_KEYWORDS = ["чын", "туура", "ырасталды", "далилденди", "чындык"]




def detect_language(text: str) -> str:
    """Detect Kyrgyz vs Russian by Kyrgyz-specific Unicode characters."""
    kyrgyz_specific = set("өүңғ")
    if sum(1 for c in text.lower() if c in kyrgyz_specific) >= 5:
        return "kyrgyz"
    return "russian"


def extract_verdict(text: str, conclusion: str) -> tuple[int, float]:
    """Return (label, confidence): 1=fake, 0=real, -1=uncertain."""
    search = (conclusion + " " + text[:3000]).lower()
    ky_fake = sum(search.count(kw) for kw in KY_FAKE_KEYWORDS)
    ky_true = sum(search.count(kw) for kw in KY_TRUE_KEYWORDS)

    fake_score = ky_fake
    true_score = ky_true

    if fake_score > true_score and fake_score > 0:
        return 1, round(min(fake_score / max(fake_score + true_score, 1), 1.0), 2)
    elif true_score > fake_score and true_score > 0:
        return 0, round(min(true_score / max(fake_score + true_score, 1), 1.0), 2)
    return -1, 0.0


def extract_conclusion(soup: BeautifulSoup, full_text: str) -> str:
    markers = ["жыйынтык", "корутунду", "баа", "вывод", "заключение", "итог"]
    for tag in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
        if any(m in tag.get_text(strip=True).lower() for m in markers):
            sib = tag.find_next_sibling()
            if sib:
                return sib.get_text(strip=True)[:800]
    return full_text[-1000:] if len(full_text) > 1000 else full_text


def extract_fake_body(soup: BeautifulSoup) -> str:
    """
    Extract ONLY the fake-claim portion of the article —
    every paragraph before any stop-phrase appears.
    This avoids leaking the factcheckers' verification style into training data.
    """
    content_div = soup.find("div", class_="entry-content")
    if not content_div:
        return ""

    parts: list[str] = []
    for p in content_div.find_all("p"):
        text = p.get_text(strip=True)
        if not text:
            continue
        # Stop as soon as we hit a factchecker stop-phrase
        if any(sp in text for sp in STOP_PHRASES):
            break
        parts.append(text)

    return " ".join(parts)


def extract_original_claim(soup: BeautifulSoup, headline: str) -> str:
    """Return blockquote claim if available, else headline."""
    bq = soup.find("blockquote")
    if bq:
        claim = bq.get_text(strip=True)
        if len(claim) > 30:
            return claim
    for cls in ["claim", "fake-claim", "statement", "teza"]:
        div = soup.find("div", class_=cls)
        if div:
            return div.get_text(strip=True)
    return headline


# ══════════════════════════════════════════════════════════════════════════════
# Data Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def split_text_augment(row: dict) -> dict | None:
    """
    Text Splitting: if body_text is long enough, return a new record
    containing the second logical half of the text.
    The original row keeps the first half.
    """
    body = row["body_text"]
    if len(body) < SPLIT_MIN_LEN:
        return None

    # Split at mid-sentence boundary (nearest period/exclamation near midpoint)
    mid = len(body) // 2
    split_idx = body.rfind(".", 0, mid)
    if split_idx == -1 or split_idx < 50:
        split_idx = mid

    second_half = body[split_idx + 1:].strip()
    if len(second_half) < MIN_BODY_LEN:
        return None

    new_row = row.copy()
    new_row["id"] = str(uuid.uuid4())
    new_row["body_text"] = second_half
    new_row["headline"] = row["headline"] + " [split]"
    new_row["augmentation"] = "text_split"
    # Also trim the original to its first half
    row["body_text"] = body[:split_idx + 1].strip()
    row["augmentation"] = "text_split_orig"
    return new_row


def back_translate_augment(rows: list[dict], needed: int) -> list[dict]:
    """
    Back-translation: Kyrgyz → Russian → Kyrgyz via Helsinki-NLP MarianMT.
    Only runs if `transformers` is installed; otherwise logs a warning and skips.
    Returns `needed` augmented rows (or fewer if not enough qualify).
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        log.warning(
            "transformers not installed — skipping back-translation. "
            "Install with: pip install transformers sentencepiece"
        )
        return []

    log.info(f"Loading MarianMT models for back-translation (need {needed} rows)…")

    try:
        # Kyrgyz → Russian
        ky_ru_name = "Helsinki-NLP/opus-mt-ky-ru"
        ky_ru_tok  = MarianTokenizer.from_pretrained(ky_ru_name)
        ky_ru_mdl  = MarianMTModel.from_pretrained(ky_ru_name)

        # Russian → Kyrgyz
        ru_ky_name = "Helsinki-NLP/opus-mt-ru-ky"
        ru_ky_tok  = MarianTokenizer.from_pretrained(ru_ky_name)
        ru_ky_mdl  = MarianMTModel.from_pretrained(ru_ky_name)
    except Exception as exc:
        log.warning(f"Could not load MarianMT models: {exc}. Skipping back-translation.")
        return []

    def translate(text: str, tok, mdl, max_len: int = 512) -> str:
        """Translate a single text string."""
        # Truncate to avoid token-limit errors
        inputs = tok(
            text[:1000], return_tensors="pt", padding=True,
            truncation=True, max_length=max_len
        )
        translated = mdl.generate(**inputs)
        return tok.decode(translated[0], skip_special_tokens=True)

    augmented: list[dict] = []
    candidates = [r for r in rows if len(r.get("body_text", "")) >= BT_MIN_LEN]

    for row in candidates:
        if len(augmented) >= needed:
            break
        try:
            ru_text  = translate(row["body_text"], ky_ru_tok, ky_ru_mdl)
            ky_text  = translate(ru_text, ru_ky_tok, ru_ky_mdl)

            if len(ky_text) < MIN_BODY_LEN:
                continue

            new_row = row.copy()
            new_row["id"]          = str(uuid.uuid4())
            new_row["body_text"]   = ky_text
            new_row["headline"]    = row["headline"] + " [bt]"
            new_row["augmentation"] = "back_translation"
            augmented.append(new_row)
            log.info(f"  [BT] augmented: {new_row['headline'][:60]}")
        except Exception as exc:
            log.warning(f"  Back-translation failed for {row['url']}: {exc}")

    log.info(f"Back-translation produced {len(augmented)} augmented rows.")
    return augmented


# ══════════════════════════════════════════════════════════════════════════════
# Link collection
# ══════════════════════════════════════════════════════════════════════════════

def get_article_links(max_pages: int = MAX_PAGES) -> list[str]:
    links: set[str] = set()

    for page in range(1, max_pages + 1):
        url = f"{CATEGORY_URL}page/{page}/" if page > 1 else CATEGORY_URL
        log.info(f"Fetching listing page {page}: {url}")

        try:
            resp = requests.get(url, timeout=20, headers=HEADERS)
            if resp.status_code == 404:
                log.info(f"  404 — end of pagination.")
                break
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            found = 0

            for article in soup.find_all("article"):
                a = article.find("a", href=True)
                if a:
                    href = a["href"]
                    full = urljoin(BASE_URL, href) if not href.startswith("http") else href
                    if "factcheck.kg" in full and full not in links and full != CATEGORY_URL:
                        links.add(full)
                        found += 1

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
                log.info("  No new links — stopping.")
                break

            time.sleep(1.5)

        except Exception as exc:
            log.warning(f"Page {page} error: {exc}")
            break

    log.info(f"Total article links: {len(links)}")
    return list(links)


# ══════════════════════════════════════════════════════════════════════════════
# Article scraper
# ══════════════════════════════════════════════════════════════════════════════

def scrape_article(url: str) -> dict | None:
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Headline ──────────────────────────────────────────────────────────
        h1 = soup.find("h1") or soup.find("h2", class_="entry-title")
        headline = h1.get_text(strip=True) if h1 else ""
        if not headline:
            return None

        # ── Fake claim body (before stop-phrase) ──────────────────────────────
        body_text = extract_fake_body(soup)
        if len(body_text) < MIN_BODY_LEN:
            return None

        # ── Full article text (for verdict detection only) ────────────────────
        content = (
            soup.find("div", class_="entry-content")
            or soup.find("article")
            or soup.find("div", class_="post-content")
        )
        full_text = content.get_text(separator=" ", strip=True) if content else body_text
        if len(full_text) < 150:
            return None

        conclusion  = extract_conclusion(soup, full_text)
        label, conf = extract_verdict(full_text, conclusion)
        original    = extract_original_claim(soup, headline)
        lang        = detect_language(full_text)

        date_tag = soup.find("time") or soup.find("span", class_="date")
        date_str = (date_tag.get("datetime") or date_tag.get_text(strip=True)) if date_tag else None

        return {
            "id":               str(uuid.uuid4()),
            "headline":         original,
            "body_text":        body_text,
            "source":           "factcheck.kg",
            "url":              url,
            "date":             date_str,
            "language":         lang,
            "label":            label,
            "label_confidence": conf,
            "evidence_url":     url,
            "source_type":      "fact_check",
            "augmentation":     "original",
        }

    except Exception as exc:
        log.warning(f"Failed {url}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FACTCHECK.KG SCRAPER  v2  (with Augmentation)")
    log.info("=" * 60)

    # ── 1. Scrape ─────────────────────────────────────────────────────────────
    links = get_article_links(max_pages=MAX_PAGES)
    all_articles: list[dict] = []

    for link in links:
        if len(all_articles) >= SCRAPE_TARGET:
            break
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
    if df_all.empty:
        log.error("No articles scraped. The site may be blocking requests.")
        return df_all

    # Save full raw results
    df_all.to_csv("data/factcheck_all.csv", index=False, encoding="utf-8-sig")
    log.info(f"Saved all {len(df_all)} scraped articles → factcheck_all.csv")

    # ── 2. Filter: keep FAKE (label=1), Kyrgyz language only ─────────────────
    df_fake = df_all[(df_all["label"] == 1) & (df_all["language"] == "kyrgyz")].copy()
    log.info(f"Kyrgyz fake articles after filtering: {len(df_fake)}")

    # ── 3. Augmentation ───────────────────────────────────────────────────────
    augmented_rows: list[dict] = []

    # 3a. Text Splitting
    rows_for_split = df_fake.to_dict("records")
    split_extras: list[dict] = []
    for row in rows_for_split:
        extra = split_text_augment(row)
        if extra:
            split_extras.append(extra)

    # Update originals (body_text may have been trimmed in-place by split_text_augment)
    df_fake = pd.DataFrame(rows_for_split)
    augmented_rows.extend(split_extras)
    log.info(f"Text-split augmentation added {len(split_extras)} rows.")

    # Combine original + split
    df_augmented = pd.concat(
        [df_fake, pd.DataFrame(augmented_rows)], ignore_index=True
    )
    log.info(f"After text splitting: {len(df_augmented)} rows")

    # 3b. Back-translation if still below TARGET_FAKE
    still_needed = TARGET_FAKE - len(df_augmented)
    if still_needed > 0:
        log.info(f"Still need {still_needed} rows — attempting back-translation…")
        bt_rows = back_translate_augment(
            df_augmented.to_dict("records"), needed=still_needed
        )
        if bt_rows:
            df_augmented = pd.concat(
                [df_augmented, pd.DataFrame(bt_rows)], ignore_index=True
            )
            log.info(f"After back-translation: {len(df_augmented)} rows")
    else:
        log.info("Target already reached — skipping back-translation.")

    # ── 4. Final trimming / shuffling ─────────────────────────────────────────
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    # Cap at TARGET_FAKE to avoid overshoot
    df_final = df_augmented.head(TARGET_FAKE)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    df_final.to_csv("data/factcheck_fake_news.csv", index=False, encoding="utf-8-sig")
    log.info(f"\n✓ Saved {len(df_final)} FAKE rows → factcheck_fake_news.csv")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Raw scraped:         {len(df_all)}")
    log.info(f"Kyrgyz fake (raw):   {len(df_fake)}")
    log.info(f"After augmentation:  {len(df_final)}")
    aug_counts = df_final.get("augmentation", pd.Series()).value_counts()
    for aug_type, cnt in aug_counts.items():
        log.info(f"  {aug_type}: {cnt}")

    return df_final


if __name__ == "__main__":
    main()