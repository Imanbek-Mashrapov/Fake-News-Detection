import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
import logging
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [saat] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/scraper_saat.log"), logging.StreamHandler()],
)
log = logging.getLogger("saat")

BASE_URL = "https://saat.kg"

CATEGORIES = [
    "https://saat.kg/biylik/",
    "https://saat.kg/koom/",
    "https://saat.kg/okuia/",
    "https://saat.kg/ekonomika/",
]

TARGET = 150
DELAY = 2.5
MAX_PAGES = 20

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8",
}


def get_links():
    links = set()

    for category in CATEGORIES:
        for page in range(1, MAX_PAGES + 1):
            url = f"{category}page/{page}/" if page > 1 else category
            log.info(f"Fetching {url}")

            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 404:
                    break

                soup = BeautifulSoup(r.text, "html.parser")

                for a in soup.find_all("a", href=True):
                    href = a["href"]

                    # Pattern: /YYYY/MM/DD/slug/
                    if "/202" in href and href.count("/") >= 5:
                        full = urljoin(BASE_URL, href)
                        links.add(full)

                time.sleep(1)

            except Exception as e:
                log.warning(e)
                break

    return list(links)


def scrape_article(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        h1 = soup.find("h1")
        if not h1:
            return None

        headline = h1.get_text(strip=True)

        content = soup.find("div", class_="entry-content") or soup.find("article")
        if not content:
            return None

        paragraphs = [p.get_text(strip=True) for p in content.find_all("p")]
        body = " ".join(paragraphs)

        if len(body) < 100:
            return None

        return {
            "id": str(uuid.uuid4()),
            "headline": headline,
            "body_text": body,
            "source": "saat.kg",
            "url": url,
            "date": None,
            "language": "kyrgyz",
            "label": 0,
            "label_confidence": "medium",
            "source_type": "news_site",
        }

    except Exception as e:
        log.warning(e)
        return None


def main():
    links = get_links()
    results = []

    for link in links:
        if len(results) >= TARGET:
            break

        rec = scrape_article(link)
        if rec:
            results.append(rec)

        time.sleep(DELAY)

    df = pd.DataFrame(results)
    df.to_csv("data/saat_kyrgyz_data.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    main()