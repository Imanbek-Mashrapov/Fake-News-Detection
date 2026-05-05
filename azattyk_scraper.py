import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import uuid
from urllib.parse import urljoin

BASE_URL = "https://www.azattyk.org"
CATEGORY_URL = "https://www.azattyk.org/news"

def get_article_links(max_pages=1):
    links = set()
    for i in range(max_pages):
        # Azattyk archives often use ?p=0, ?p=1
        url = f"{CATEGORY_URL}?p={i}"
        print(f"Fetching links from: {url}")
    
        try:
            resp = requests.get(url, timeout=15)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('/a/') and '.html' in href:
                    full_url = urljoin(BASE_URL, href)
                    links.add(full_url)
        except Exception as e:
            print(f"Error on page {i}: {e}")
    print(f"Found {len(links)} unique article links.")
    return list(links)


def scrape_azattyk_article(url):
    try:
        resp = requests.get(url, timeout=15)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')


        headline = soup.find('h1', class_='title').get_text(strip=True)
        content_div = soup.find('div', class_='wsw')
        if not content_div:
            return None
        paragraphs = [p.get_text(strip=True) for p in content_div.find_all('p')]
        body_text = " ".join(paragraphs)
        return {
            "id": str(uuid.uuid4()),
            "headline": headline,
            "body_text": body_text,
            "source": "azattyk.org",
            "label": 0, # Real news
            "label_confidence": "high",
            "source_type": "news_site"
        }

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

# Execution logic
all_links = get_article_links(max_pages=10)
results = []

for link in all_links[:100]: # Test with 10 first
    print(f"Processing: ({len(results)+1}/100): {link}")
    data = scrape_azattyk_article(link)
    if data:
        results.append(data)
    time.sleep(4) # Respectful delay

df = pd.DataFrame(results)
df.to_csv("azattyk_fixed.csv", index=False, encoding='utf-8-sig')
print("Saved to azattyk_fixed.csv")
print(f"Done! Saved {len(df)} articles to azattyk_fixed.csv")