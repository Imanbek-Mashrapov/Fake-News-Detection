import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import uuid

class KyrgyzNewsScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'University Student Research Project (email@student.kg)'}
        self.results = []

    def get_azattyk_article(self, url):
        """Scraper for Azattyk.org"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Selectors (based on 2025/2026 structure)
            headline = soup.find('h1', class_='title').text.strip()
            # Azattyk body is usually inside div.wsw
            body = " ".join([p.text for p in soup.find('div', class_='wsw').find_all('p')])
            pub_date = soup.find('time')['datetime'] if soup.find('time') else "2026-05-01"

            return self.format_entry(headline, body, url, pub_date)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def format_entry(self, headline, body, url, date):
        return {
            "id": str(uuid.uuid4()),
            "headline": headline,
            "body_text": body,
            "source": "azattyk.org",
            "date": date,
            "label": 0, # REAL
            "label_confidence": "high",
            "source_type": "news_site"
        }

    def run_slowly(self, url_list):
        for url in url_list:
            print(f"Scraping: {url}")
            data = self.get_azattyk_article(url)
            if data:
                self.results.append(data)
            
            # CRITICAL: Wait 3-5 seconds to avoid IP block
            time.sleep(4) 

# Example usage
scraper = KyrgyzNewsScraper()
urls = ["https://www.azattyk.org/a/328...html"] # Replace with collected links
scraper.run_slowly(urls)

# Save to CSV
df = pd.DataFrame(scraper.results)
df.to_csv("kyrgyz_real_news.csv", index=False)