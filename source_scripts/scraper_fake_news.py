import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from urllib.parse import urljoin

# === КОНФИГУРАЦИЯ ===
SCRAPE_TARGET = 800     # Общее количество новостей для сбора
MAX_PAGES     = 90      # Макс. страниц для каждого сайта
DELAY         = 3.0     # Задержка между запросами
OUTPUT_FILE   = "fake_news_dataset.xlsx"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("scraper")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ky,ru;q=0.9,en;q=0.8"
}

KY_FAKE_KEYWORDS = ["жалган", "туура эмес", "ырасталган жок", "далилденген жок", "туура эмес маалымат", "жаңылыш", "бурмаланган", "жасалма", "фейк", "чындыкка жатпайт", "калп", "дипфейк"]
KY_TRUE_KEYWORDS = ["чын", "туура", "ырасталды", "далилденди", "чындык"]

def detect_language(text: str) -> str:
    kyrgyz_specific = set("өүңғ")
    if sum(1 for c in text.lower() if c in kyrgyz_specific) >= 5:
        return "kyrgyz"
    return "russian"

def extract_verdict(text: str):
    search = text.lower()
    ky_fake = sum(search.count(kw) for kw in KY_FAKE_KEYWORDS)
    ky_true = sum(search.count(kw) for kw in KY_TRUE_KEYWORDS)
    
    if ky_fake > ky_true and ky_fake > 0: return 1
    if ky_true > ky_fake and ky_true > 0: return 0
    return -1

def get_links(source_config):
    links = set()
    base_url = source_config['base_url']
    cat_url = source_config['cat_url']
    
    for page in range(1, MAX_PAGES + 1):
        if len(links) >= SCRAPE_TARGET: break
        
        url = f"{cat_url}page/{page}/" if page > 1 else cat_url
        log.info(f"Сбор ссылок с {source_config['name']} (Страница {page})")
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 404: break
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            found = 0
            for a in soup.select(source_config['link_selector']):
                href = a.get('href')
                if href:
                    full = urljoin(base_url, href) if not href.startswith("http") else href
                    if full not in links and len(full) > len(base_url) + 5:
                        links.add(full)
                        found += 1
            
            if found == 0: break
            time.sleep(1.0)
        except Exception as e:
            log.warning(f"Ошибка на странице {page}: {e}")
            break
            
    return list(links)

def extract_fake_body(soup, content_selector, source_name):
    """
    Умное извлечение текста в зависимости от источника:
    - PolitKlinika: Берем весь текст статьи.
    - Factcheck.kg: Берем только первый значащий абзац.
    """
    content_div = soup.select_one(content_selector)
    if not content_div: 
        return ""
    
    # Для ПолитКлиники берем весь текст
    if source_name == "PolitKlinika":
        # Убираем картинки и скрипты из текста
        for s in content_div(['script', 'style', 'figure', 'img']): s.decompose()
        return content_div.get_text(" ", strip=True)

    # Для Factcheck.kg берем только первый абзац
    for p in content_div.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) > 20:
            return text
            
    return content_div.get_text(" ", strip=True)

def scrape_article(url, source_name, content_selector, date_selector):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 1. Headline
        h_tag = soup.find("h1") or soup.find("h2", class_="entry-title") or soup.find("h3", class_="entry-title")
        headline = h_tag.get_text(strip=True) if h_tag else ""
        if not headline: return None
        
        # 2. Body Text (с учетом разной логики для сайтов)
        body_text = extract_fake_body(soup, content_selector, source_name)
        if len(body_text) < 50: return None
        
        # Полный текст нужен только для определения вердикта (лейбла) и языка
        content_div = soup.select_one(content_selector)
        full_text_for_label = content_div.get_text(" ", strip=True) if content_div else body_text
        
        # 3. Label & Language
        label = extract_verdict(full_text_for_label[-2000:])
        lang = detect_language(full_text_for_label)
        
        # 4. Date
        date_tag = soup.select_one(date_selector)
        date_str = date_tag.get_text(strip=True) if date_tag else "N/A"
        
        return {
            "headline": headline,
            "body_text": body_text,
            "source": source_name,
            "url": url,
            "date": date_str,
            "language": lang,
            "label": label
        }
    except Exception as e:
        log.warning(f"Сбой при загрузке {url}: {e}")
        return None

def main():
    sources = [
        {
            "name": "Factcheck.kg",
            "base_url": "https://factcheck.kg",
            "cat_url": "https://factcheck.kg/ky/category/factcheck/",
            "link_selector": "article a",
            "content_selector": ".entry-content",
            "date_selector": "span.posts-date a"
        },
        {
            "name": "PolitKlinika",
            "base_url": "https://pk.kg",
            "cat_url": "https://pk.kg/news/category/faktcheking/",
            # ИСПРАВЛЕННЫЕ СЕЛЕКТОРЫ ДЛЯ POLITKLINIKA:
            "link_selector": ".newsGrid-title a",
            "content_selector": ".newsElement-text",
            "date_selector": ".newsElement-params li"
        }
    ]
    
    all_results = []
    
    for src in sources:
        links = get_links(src)
        log.info(f"--- Найдено {len(links)} уникальных ссылок для {src['name']} ---")
        
        site_count = 0  # Счетчик скачанных статей именно для этого сайта
        
        for link in links:
            # Теперь мы проверяем лимит конкретного сайта
            if site_count >= SCRAPE_TARGET: 
                break
            
            record = scrape_article(link, src['name'], src['content_selector'], src['date_selector'])
            if record:
                all_results.append(record)
                site_count += 1
                log.info(f"[{record['label']}] Собрано: {record['headline'][:50]}... | Дата: {record['date']}")
            
            time.sleep(DELAY) 
            
    df = pd.DataFrame(all_results)
    df.to_excel(OUTPUT_FILE, index=False)
    
    log.info("=" * 50)
    log.info(f"ГОТОВО! Сохранено {len(df)} записей в {OUTPUT_FILE}")
    log.info("=" * 50)

if __name__ == "__main__":
    main()