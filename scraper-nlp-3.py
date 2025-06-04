import streamlit as st
import pickle
import os
import time
import urllib.parse
from urllib.parse import urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import locale
import random
from streamlit_option_menu import option_menu
import io

import newspaper
from newspaper import Article
from newspaper.configuration import Configuration
from transformers import BertTokenizer, EncoderDecoderModel
from transformers import pipeline
from tqdm import tqdm

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import yake

### ONLINE STREAMLIT DEPENDENCIES ###
import sys
import types
import torch

if isinstance(torch.classes, types.ModuleType):
 torch.classes.__path__ = []
## ONLINE STREAMLIT DEPENDENCIES ###

from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By

# Read media database
URL='https://docs.google.com/spreadsheets/d/e/2PACX-1vQwxy1jmenWfyv49wzEwrp3gYE__u5JdhvVjn1c0zMUxDL6DTaU_t4Yo03qRlS4JaJWE3nK9_dIQMYZ/pub?output=csv'.format()
media_db=pd.read_csv(URL).fillna(0)

# Set config for newspaper
config = Configuration() # Default config, akan di-override di get_exact_publish_date
config.request_timeout = 25 

# ================================
# ANTI-CAPTCHA & SCRAPING HELPERS
# ================================
USER_AGENTS = [
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0",
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/114.0",
"Mozilla/5.0 (X11; Linux i686; rv:109.0) Gecko/20100101 Firefox/114.0",
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/114.0.1823.51",
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
"Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
]

BASE_HEADERS = {
"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
"Accept-Language": "en-US,en;q=0.9,id;q=0.8",
"Accept-Encoding": "gzip, deflate, br",
"Connection": "keep-alive",
"Upgrade-Insecure-Requests": "1",
"Sec-Fetch-Dest": "document",
"Sec-Fetch-Mode": "navigate",
"Sec-Fetch-Site": "same-origin",
"Sec-Fetch-User": "?1",
"TE": "trailers"
}

# ================================
# FUNCTION ZONE
# ================================

def format_boolean_query(query):
    token_pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\(|\)|"[^"]+"|\S+)'
    tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)
    def parse_tokens(tokens_list): # Renamed parameter to avoid conflict
        output = []
        i = 0
        while i < len(tokens_list):
            token = tokens_list[i].upper()
            if token == "AND": i += 1; continue
            elif token == "OR": output.append("OR")
            elif token == "NOT":
                i += 1
                if i < len(tokens_list):
                    next_token = tokens_list[i]
                    if next_token.startswith("("):
                        group_tokens_list, paren_count, i_loop = [], 1, i + 1 # Renamed variables
                        while i_loop < len(tokens_list) and paren_count > 0:
                            if tokens_list[i_loop] == "(": paren_count += 1
                            elif tokens_list[i_loop] == ")": paren_count -= 1
                            if paren_count > 0: group_tokens_list.append(tokens_list[i_loop])
                            i_loop += 1
                        output.append(f'-({parse_tokens(group_tokens_list)})'); i = i_loop -1 # Update outer i
                    else: output.append(f'-{next_token}')
            else: output.append(tokens_list[i])
            i += 1
        return " ".join(output)
    return parse_tokens(tokens)

try:
    locale.setlocale(locale.LC_TIME, "id_ID.utf8")
except:
    locale.setlocale(locale.LC_TIME, "C")

def convert_relative_date(text):
    text = text.lower().strip().replace("yang", "").replace("  ", " ").strip()
    today = datetime.today()
    date_obj = None
    patterns = {
        r"(\d+)\s+hari": lambda m: today - timedelta(days=int(m.group(1))),
        r"(\d+)\s+jam": lambda m: today - timedelta(hours=int(m.group(1))),
        r"(\d+)\s+menit": lambda m: today, 
        r"kemarin": lambda m: today - timedelta(days=1),
        r"(\d+)\s+minggu": lambda m: today - timedelta(weeks=int(m.group(1))),
        r"(\d+)\s+bulan": lambda m: today - timedelta(days=int(m.group(1)) * 30),
        r"(\d+)\s+tahun": lambda m: today - timedelta(days=int(m.group(1)) * 365),
    }
    for pattern, func in patterns.items():
        match = re.search(pattern, text)
        if match: date_obj = func(match); break
    if not date_obj and re.match(r"\d{1,2}\s+\w+", text):
        try: date_obj = datetime.strptime(text + f" {today.year}", "%d %B %Y")
        except ValueError: return text
    return date_obj.strftime("%d %b %Y").lstrip("0") if date_obj else text

def extract_domain_from_url(url):
    netloc = urlparse(url).netloc
    return netloc[4:] if netloc.startswith("www.") else netloc

def load_schedules():
    if os.path.exists("schedules.pkl"):
        with open("schedules.pkl", "rb") as f: return pickle.load(f)
    return []

def save_schedules(schedules):
    with open("schedules.pkl", "wb") as f: pickle.dump(schedules, f)

def get_exact_publish_date(url, google_news_date_text):
    try:
        article_config = Configuration()
        article_config.browser_user_agent = random.choice(USER_AGENTS)
        article_config.request_timeout = 25
        
        article = Article(url, language="id", config=article_config)
        time.sleep(random.uniform(2.0, 5.0)) 
        article.download()
        article.parse()
        if article.publish_date:
            return article.publish_date.strftime("%d %b %Y").lstrip("0")
    except Exception as e:
        # print(f"Newspaper3k failed for {url}: {e}")
        pass
    return convert_relative_date(google_news_date_text)

def check_for_captcha_selenium(driver):
    title = driver.title.lower()
    if "captcha" in title or "verify you are human" in title or "verifikasi bahwa anda manusia" in title:
        print("CAPTCHA terdeteksi oleh Selenium!")
        captcha_elements = driver.find_elements(By.CSS_SELECTOR, "iframe[src*='recaptcha'], div.g-recaptcha, #captcha-form")
        if captcha_elements:
            return True
    return False

def scrape_with_bs4(base_url):
    news_results = []
    page = 0
    session = requests.Session() # Gunakan session

    while True:
        start = page * 10
        current_url = f"{base_url}&start={start}"
        
        # Sederhanakan header untuk BS4, biarkan session menangani cookies
        # Cukup rotasi User-Agent
        current_headers = {
            "User-Agent": random.choice(USER_AGENTS),
            # Anda bisa mencoba menambahkan Accept dan Accept-Language jika masih bermasalah
            # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            # "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
        }
        # Hapus pengaturan Referer dan Sec-Fetch-Site yang kompleks untuk BS4
        # if page > 0 : 
        #     current_headers["Referer"] = f"{base_url}&start={(page-1)*10}" # Dihapus
        #     current_headers["Sec-Fetch-Site"] = "same-origin" # Dihapus

        print(f"BS4: Requesting page {page+1} - URL: {current_url}")
        try:
            # Jeda yang lebih lama dan lebih bervariasi
            time.sleep(random.uniform(7.0, 18.0)) # Tingkatkan jeda untuk BS4
            response = session.get(current_url, headers=current_headers, timeout=30)
            response.raise_for_status() # Cek jika ada HTTP error (4xx atau 5xx)
        except requests.exceptions.RequestException as e:
            print(f"BS4: Request error on page {page+1}: {e}")
            # Jika error karena timeout atau koneksi, mungkin lebih baik break
            # Jika error karena status (misal 403 Forbidden), mungkin CAPTCHA atau blokir
            if response is not None and response.status_code == 429: # Too Many Requests
                print("BS4: Error 429 - Terlalu banyak request. Berhenti dan coba lagi nanti.")
                st.error("BS4: Terlalu banyak request (Error 429). Silakan coba lagi setelah beberapa saat.")
            break 

        soup = BeautifulSoup(response.content, "html.parser")

        # Deteksi CAPTCHA yang lebih sederhana, karena BS4 tidak bisa render JavaScript
        # Cari teks yang umum ada di halaman CAPTCHA Google
        page_text_lower = soup.get_text().lower()
        if "untuk melanjutkan, lengkapi captcha di bawah ini" in page_text_lower or \
        "our systems have detected unusual traffic" in page_text_lower or \
        "tentang laman ini" in page_text_lower and "tidak ada dokumen yang cocok dengan kueri penelusuran anda" not in page_text_lower: # "Tentang laman ini" kadang muncul di halaman CAPTCHA
            print(f"BS4: Kemungkinan CAPTCHA terdeteksi di halaman {page+1} berdasarkan konten teks.")
            st.warning(f"BS4: Kemungkinan terdeteksi CAPTCHA. Proses untuk keyword ini mungkin tidak lengkap.")
            break

        results_on_page = 0
        for el in soup.select("div.SoaBEf, div.Gx5Zad, div.xuvV6b, div.yDYNvb, div.MjjYud"): # Tambahkan selector umum lainnya untuk hasil Google
            try:
                link_tag = el.find("a", href=True)
                if not link_tag: continue
                link = link_tag["href"]
                
                # Coba beberapa selector untuk judul
                title_tag = el.select_one("div.MBeuO, h3, div.n0jPhd") # Tambahkan h3 dan selector lain
                judul = title_tag.get_text(strip=True) if title_tag else "N/A"

                snippet_tag = el.select_one(".GI74Re, .OSrXXb, .d4rhi") # Tambahkan selector lain
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else "N/A"
                
                date_tag = el.select_one(".LfVVr, .OSrXXb span, .dSbHq") # Tambahkan selector lain
                tanggal_google = date_tag.get_text(strip=True) if date_tag else "N/A"
                
                if judul == "N/A" and "google.com/url?q=" not in link: # Skip jika tidak ada judul dan bukan link redirect Google
                    # Kadang ada div kosong yang cocok dengan selector utama
                    continue

                tanggal_akurat = get_exact_publish_date(link, tanggal_google)
                news_results.append({"Link": link, "Judul": judul, "Snippet": snippet, "Tanggal": tanggal_akurat, "Media": extract_domain_from_url(link)})
                results_on_page += 1
            except Exception as e:
                # print(f"BS4: Error parsing element: {e} - Element HTML: {el.prettify()[:200]}") # Untuk debug
                continue
        
        print(f"BS4: Page {page+1} - Found {results_on_page} results.")
        if results_on_page == 0:
            # Sebelum break, cek apakah ini benar-benar akhir atau halaman kosong karena masalah
            if "tidak ada dokumen yang cocok dengan kueri penelusuran anda" in page_text_lower or \
            "no results found for your query" in page_text_lower:
                st.info(f"BS4: Tidak ada hasil lagi yang cocok dengan query setelah halaman {page}.")
            elif page == 0:
                st.info("BS4: Tidak ada hasil ditemukan di halaman pertama.")
            else:
                st.warning(f"BS4: Tidak ada hasil di halaman {page+1}, mungkin karena blokir atau akhir dari hasil yang diizinkan.")
            break
        page += 1
        # Pembatasan halaman dihapus

    news_results_df = pd.DataFrame(news_results, columns=['Link', 'Judul', 'Snippet','Tanggal','Media'])
    if not news_results_df.empty:
        news_results_df = news_results_df.merge(media_db, on='Media', how='left')
    return news_results_df

def setup_selenium_options(headless=True):
    options = FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    options.set_preference("dom.webdriver.enabled", False)
    options.set_preference('useAutomationExtension', False)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920,1080")
    return options

def scrape_with_selenium(base_url, run_interactively=False): 
    options = setup_selenium_options(headless=not run_interactively) 
    driver = webdriver.Firefox(options=options)

    news_results = []
    page = 0
    try:
        while True:
            start = page * 10
            current_url = f"{base_url}&start={start}"
            print(f"Selenium: Requesting page {page+1} - URL: {current_url}")
            
            time.sleep(random.uniform(6.0, 15.0)) 
            driver.get(current_url)
            time.sleep(random.uniform(3.0, 7.0)) 

            if check_for_captcha_selenium(driver):
                st.warning(f"Selenium: Terdeteksi CAPTCHA di halaman {page+1}.")
                if run_interactively:
                    try:
                        input("CAPTCHA terdeteksi. Selesaikan di browser, lalu tekan Enter di terminal ini untuk melanjutkan...")
                        driver.refresh()
                        time.sleep(5)
                        if check_for_captcha_selenium(driver): 
                            print("Selenium: CAPTCHA masih ada setelah intervensi manual. Menghentikan.")
                            break
                    except KeyboardInterrupt:
                        print("Selenium: Proses dihentikan oleh user saat menunggu CAPTCHA.")
                        break
                else: 
                    break 

            elements = driver.find_elements(By.CSS_SELECTOR, "div.SoaBEf")
            if not elements and page > 0 :
                print(f"Selenium: Tidak ada elemen berita ditemukan di halaman {page+1}, mungkin akhir hasil.")
                break
            elif not elements and page == 0:
                print(f"Selenium: Tidak ada elemen berita ditemukan di halaman pertama.")
                break

            results_on_page = 0
            for el in elements:
                try:
                    link_element = el.find_element(By.TAG_NAME, "a"); link = link_element.get_attribute("href")
                    title = el.find_element(By.CSS_SELECTOR, "div.MBeuO").text
                    snippet = el.find_element(By.CSS_SELECTOR, ".GI74Re").text
                    date_google = el.find_element(By.CSS_SELECTOR, ".LfVVr").text
                    date_akurat = get_exact_publish_date(link, date_google)
                    news_results.append({"Link": link, "Judul": title, "Snippet": snippet, "Tanggal": date_akurat, "Media": extract_domain_from_url(link)})
                    results_on_page +=1
                except Exception as e:
                    # print(f"Selenium: Error parsing element: {e}")
                    continue
            
            print(f"Selenium: Page {page+1} - Found {results_on_page} results.")
            if results_on_page == 0:
                if page == 0: st.info("Selenium: Tidak ada hasil ditemukan di halaman pertama.")
                else: st.info(f"Selenium: Tidak ada hasil lagi setelah halaman {page}.")
                break
            page += 1
            # Pembatasan halaman dihapus
    finally:
        driver.quit()

    news_results_df = pd.DataFrame(news_results, columns=['Link', 'Judul', 'Snippet','Tanggal','Media'])
    if not news_results_df.empty:
        news_results_df = news_results_df.merge(media_db, on='Media', how='left')
    return news_results_df

def scrape_duckduckgo(duck_url, run_interactively=False):
    options = setup_selenium_options(headless=not run_interactively)
    driver = webdriver.Firefox(options=options)

    news_results = []
    try:
        print(f"DuckDuckGo: Requesting URL: {duck_url}")
        time.sleep(random.uniform(5.0, 10.0))
        driver.get(duck_url)
        time.sleep(random.uniform(4.0, 8.0))

        if check_for_captcha_selenium(driver): 
            st.warning(f"DuckDuckGo: Terdeteksi CAPTCHA.")
            if run_interactively:
                input("CAPTCHA terdeteksi. Selesaikan di browser, lalu tekan Enter di terminal ini untuk melanjutkan...")
                driver.refresh(); time.sleep(5)
                if check_for_captcha_selenium(driver): print("DuckDuckGo: CAPTCHA masih ada. Menghentikan."); return pd.DataFrame()
            else: return pd.DataFrame()
        
        while True: 
            try:
                time.sleep(random.uniform(1.0, 3.0)) 
                load_more_selectors = [
                    "button[type='button'].BttQvzGBidWFHCQKHTdB", 
                    "button.result--more__btn",                   
                    "a.result--more__btn",                        
                    "input[value='More Results']"                 
                ]
                load_more_button = None
                for selector in load_more_selectors:
                    try:
                        btn = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        if btn.is_displayed() and btn.is_enabled():
                            load_more_button = btn
                            break
                    except:
                        continue
                
                if load_more_button:
                    print(f"DuckDuckGo: Clicking 'Load More'") 
                    driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", load_more_button)
                    time.sleep(random.uniform(4.0, 7.0)) 
                else:
                    print("DuckDuckGo: Tombol 'Load More' tidak ditemukan atau tidak aktif lagi. Menghentikan scroll.")
                    break 
            except Exception as e:
                print(f"DuckDuckGo: Error saat scroll atau klik 'Load More': {e}")
                break 
        
        print("DuckDuckGo: Selesai scroll, mengambil semua elemen...")
        elements = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result']")
        print(f"DuckDuckGo: Total elemen ditemukan setelah scroll: {len(elements)}")

        for el in elements:
            try:
                link_element = el.find_element(By.CSS_SELECTOR, "a[data-testid='result-title-a']")
                link = link_element.get_attribute("href")
                title = link_element.find_element(By.CSS_SELECTOR, "span").text
                snippet_element = el.find_element(By.CSS_SELECTOR, "div[data-testid='result-snippet']")
                snippet = snippet_element.text if snippet_element else "N/A"
                
                date_source_text = "N/A"
                try: 
                    date_source_element = el.find_element(By.CSS_SELECTOR, "div.result__extras__timestamp, span.result__timestamp, .result__age")
                    date_source_text = date_source_element.text
                except: pass 

                date_akurat = get_exact_publish_date(link, date_source_text)
                news_results.append({"Link": link, "Judul": title, "Snippet": snippet, "Tanggal": date_akurat, "Media": extract_domain_from_url(link)})
            except Exception as e:
                # print(f"DuckDuckGo: Error processing element: {e}")
                continue
    finally:
        driver.quit()

    news_results_df = pd.DataFrame(news_results, columns=['Link', 'Judul', 'Snippet','Tanggal','Media'])
    if not news_results_df.empty:
        news_results_df = news_results_df.merge(media_db, on='Media', how='left')
    return news_results_df

def get_news_data(method, start_date, end_date, keyword_query, run_interactively_selenium=False):
    keyword_query_formatted = format_boolean_query(keyword_query)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_obj = end_date.date() if isinstance(end_date, datetime) else end_date
    end_date_plus_one_str = (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date_str = end_date_obj.strftime('%Y-%m-%d')

    base_url_google = f"https://www.google.com/search?q={urllib.parse.quote(f'{keyword_query_formatted} after:{start_date_str} before:{end_date_plus_one_str}')}&gl=id&hl=id&lr=lang_id&tbm=nws&num=10"
    duck_url = f"https://duckduckgo.com/?q={urllib.parse.quote(keyword_query)}&t=h_&df={start_date_str}...{end_date_str}&iar=news&kl=id-id"

    news_df = pd.DataFrame()
    if method == "BeautifulSoup":
        news_df = scrape_with_bs4(base_url_google)
    elif method == "Selenium":
        news_df = scrape_with_selenium(base_url_google, run_interactively=run_interactively_selenium)
    elif method == "Selenium DuckDuckGo":
        news_df = scrape_duckduckgo(duck_url, run_interactively=run_interactively_selenium)
    else:
        raise ValueError("Invalid method")

    if not news_df.empty and 'Tanggal' in news_df.columns:
        news_df['SortableDate'] = pd.to_datetime(news_df['Tanggal'], errors='coerce')
        news_df = news_df.sort_values(by='SortableDate', ascending=False, na_position='last').drop(columns=['SortableDate'])
    return news_df

def enrich_with_nlp(df, selected_nlp=[]):
    if df.empty: st.warning("No data to process with NLP."); return df

    if "Article Content" in selected_nlp: df['Article Content'] = ''
    if "Summary" in selected_nlp: df['Summary'] = ''
    if "Sentiment" in selected_nlp: df['Sentiment'] = ''; df['SentimentScore'] = 0.0
    if "Keywords" in selected_nlp: df['Keywords'] = ''
    if "Author" in selected_nlp: df['Author'] = ''
    if "Exact Publish Date" in selected_nlp and 'Exact Date' not in df.columns: df['Exact Date'] = ''

    summarizer, sentiment_nlp, tokenizer = load_models()
    df['Status'] = ''
    error_count = 0
    media_name_regex = r"^[A-Z][\w\s]+?\s[-:]\s[\w\s,]+(?:\d{4})?"
    kw_extractor = yake.KeywordExtractor(lan="id", n=3, dedupLim=0.9, top=5, features=None)

    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_articles = len(df)

    for idx, row in tqdm(df.iterrows(), total=total_articles, desc="Running NLP"):
        progress_percent = (idx + 1) / total_articles
        progress_bar.progress(progress_percent)
        progress_text.markdown(f"**NLP Progress:** {int(progress_percent * 100)}%")

        url = row['Link']
        try:
            article_nlp_config = Configuration()
            article_nlp_config.browser_user_agent = random.choice(USER_AGENTS)
            article_nlp_config.request_timeout = 25
            
            article = Article(url, language="id", config=article_nlp_config)
            time.sleep(random.uniform(1.0, 3.0)) 
            article.download()
            article.parse()

            full_text = article.text.strip()
            full_text = re.sub(media_name_regex, "", full_text)

            if not full_text: raise ValueError("Article text empty after parsing and cleaning.")

            if "Article Content" in selected_nlp: df.at[idx, 'Article Content'] = full_text
            if "Summary" in selected_nlp:
                inputs = tokenizer(full_text[:1024], return_tensors="pt", padding=True, truncation=True, max_length=512)
                summary_ids = summarizer.generate(inputs.input_ids,max_length=150,min_length=40,length_penalty=2.0,num_beams=4,early_stopping=True)
                df.at[idx, 'Summary'] = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            if "Sentiment" in selected_nlp:
                sentiment_result = sentiment_nlp(full_text[:512])[0]
                df.at[idx, 'Sentiment'] = sentiment_result['label']; df.at[idx, 'SentimentScore'] = sentiment_result['score']
            if "Keywords" in selected_nlp:
                keywords = [kw for kw, _ in kw_extractor.extract_keywords(full_text)]
                df.at[idx, 'Keywords'] = ", ".join(keywords)
            if "Author" in selected_nlp: df.at[idx, 'Author'] = ", ".join(article.authors) if article.authors else "Unknown"
            if "Exact Publish Date" in selected_nlp: 
                exact_date_val = article.publish_date
                if exact_date_val: df.at[idx, 'Exact Date'] = exact_date_val.strftime("%d %b %Y").lstrip("0")
                elif 'Tanggal' in row and pd.notna(row['Tanggal']): df.at[idx, 'Exact Date'] = row['Tanggal']
                else: df.at[idx, 'Exact Date'] = "Unknown"
            df.at[idx, 'Status'] = "OK"
        except Exception as e:
            error_count += 1; err_msg = str(e)
            if "Article Content" in selected_nlp: df.at[idx, 'Article Content'] = "ERROR"
            if "Summary" in selected_nlp: df.at[idx, 'Summary'] = "ERROR"
            if "Sentiment" in selected_nlp: df.at[idx, 'Sentiment'] = "ERROR"; df.at[idx, 'SentimentScore'] = 0.0
            if "Keywords" in selected_nlp: df.at[idx, 'Keywords'] = "ERROR"
            if "Author" in selected_nlp: df.at[idx, 'Author'] = "ERROR"
            if "Exact Publish Date" in selected_nlp: df.at[idx, 'Exact Date'] = "ERROR"
            df.at[idx, 'Status'] = f"ERROR: {err_msg[:80]}"
    progress_bar.empty(); progress_text.empty()
    if error_count > 0: st.warning(f"⚠️ NLP selesai. {error_count} dari {total_articles} artikel gagal diproses.")
    else: st.success("✅ NLP selesai tanpa kesalahan.")
    return df

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Burson News Scraper", layout="wide")

@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    summarizer = EncoderDecoderModel.from_pretrained("cahya/bert2bert-indonesian-summarization")
    pretrained_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    sentiment_nlp = pipeline("sentiment-analysis",model=pretrained_name,tokenizer=pretrained_name)
    return summarizer, sentiment_nlp, tokenizer

with st.sidebar:
    menu = option_menu(
        menu_title = "Main Menu",
        options=["Scrape", "Queue", "NLP Tools", "Scheduler", "How to use", "About"],
        icons=["search", "list-check", "tools", "clock", "question-circle-fill", "diagram-3"],
        menu_icon="cast", default_index=0,
        styles={"icon": {"color": "orange"},"nav-link": {"--hover-color": "#eee"},"nav-link-selected": {"background-color": "green"},},)

if menu == "Scrape":
    st.title("📰 Burson News Scraper - v1.0.7 (No Page Limit)")
    st.markdown("Scrape berita dengan upaya menghindari CAPTCHA. **Pembatasan halaman telah dihapus - gunakan dengan hati-hati.**")

    run_selenium_interactively = False
    if st.checkbox("Jalankan Selenium secara interaktif (untuk debug/menyelesaikan CAPTCHA manual)", value=False, key="interactive_selenium"):
        run_selenium_interactively = True
        st.caption("Jika dicentang, browser Selenium akan terlihat dan Anda bisa diminta menyelesaikan CAPTCHA manual jika terdeteksi.")

    with st.form("scrape_form"):
        keyword = st.text_input("Masukkan keyword (gunakan AND, OR, NOT):", value="")
        col1, col2 = st.columns(2)
        with col1:
            default_start_date = datetime.now().date() - timedelta(days=7)
            start_date = st.date_input("Tanggal mulai", value=default_start_date)
        with col2:
            end_date = st.date_input("Tanggal akhir", value=datetime.now().date())
        method = st.radio("Metode Scraping:", ["BeautifulSoup", "Selenium", "Selenium DuckDuckGo"], horizontal=True, index=0)
        nlp_options = st.multiselect("Pilih fitur NLP (opsional):",["Article Content", "Summary", "Sentiment", "Keywords", "Author", "Exact Publish Date"])
        submitted = st.form_submit_button("Mulai Scrape")

    if submitted:
        if not keyword.strip(): st.error("Keyword tidak boleh kosong.")
        elif start_date > end_date: st.error("Tanggal mulai tidak boleh setelah tanggal akhir.")
        else:
            with st.spinner("Sedang scraping berita... Ini mungkin memakan waktu lebih lama dengan tindakan anti-CAPTCHA dan tanpa batas halaman."):
                results_df = get_news_data(method, start_date, end_date, keyword, run_interactively_selenium=run_selenium_interactively)
            
            if results_df.empty: st.warning("Tidak ada hasil berita ditemukan.")
            else:
                st.session_state.scraped_df = results_df
                st.session_state.filename = f"hasil_berita_{keyword.replace(' ','_')}_{start_date}_to_{end_date}.xlsx"
                st.session_state.nlp_done = False
                if len(nlp_options) > 0:
                    with st.spinner("Menjalankan proses NLP..."):
                        st.session_state.scraped_df = enrich_with_nlp(results_df.copy(), selected_nlp=nlp_options)
                        st.session_state.nlp_done = True
                st.success(f"{len(st.session_state.scraped_df)} berita berhasil di-scrape!")

    if "scraped_df" in st.session_state and not st.session_state.scraped_df.empty:
        df_to_display = st.session_state.scraped_df
        st.dataframe(df_to_display)
        filename = st.session_state.filename
        excel_buffer = io.BytesIO()
        st.session_state.scraped_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        st.download_button(label="📥 Download Hasil (Excel)",data=excel_buffer,file_name=filename,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click=lambda: [st.session_state.pop(key, None) for key in ["scraped_df", "filename", "nlp_done"]])
        if st.button("Hapus Hasil & Mulai Baru"):
            for key in ["scraped_df", "filename", "nlp_done"]: st.session_state.pop(key, None)
            st.rerun()

elif menu == "Queue":
    st.title("📋 Multiple Keyword Scraper (Queue)")
    st.markdown("Tambahkan beberapa keyword untuk di-scrape secara berurutan.")
    if "query_queue" not in st.session_state: st.session_state.query_queue = []
    st.subheader("➕ Tambah Keyword ke Antrian")
    with st.form("add_queue_form"):
        keyword_q = st.text_input("Masukkan keyword (gunakan AND, OR, NOT):")
        col1_q, col2_q = st.columns(2)
        with col1_q:
            default_start_date_q = datetime.now().date() - timedelta(days=7)
            start_date_q = st.date_input("Tanggal mulai", value=default_start_date_q, key="q_start")
        with col2_q:
            end_date_q = st.date_input("Tanggal akhir", value=datetime.now().date(), key="q_end")
        method_q = st.radio("Metode Scraping:", ["BeautifulSoup", "Selenium", "Selenium DuckDuckGo"], horizontal=True, key="q_method")
        nlp_options_q = st.multiselect("Pilih fitur NLP (opsional):",["Article Content", "Summary", "Sentiment", "Keywords", "Author", "Exact Publish Date"],key="q_nlp")
        add_button = st.form_submit_button("Tambahkan ke Antrian")
        if add_button:
            if not keyword_q.strip(): st.error("Keyword tidak boleh kosong.")
            elif start_date_q > end_date_q: st.error("Tanggal mulai tidak boleh setelah tanggal akhir.")
            else:
                st.session_state.query_queue.append({"keyword": keyword_q,"start_date": start_date_q,"end_date": end_date_q,"method": method_q,"nlp_options": nlp_options_q,"has_nlp": len(nlp_options_q) > 0})
                st.success(f"✅ Keyword '{keyword_q}' ditambahkan ke antrian.")
    st.subheader("🧾 Antrian Aktif")
    if st.session_state.query_queue:
        for i, item in enumerate(st.session_state.query_queue):
            col1_disp, col2_disp = st.columns([11, 1], vertical_alignment="center")
            with col1_disp:
                nlp_display = f" dengan NLP: {', '.join(item['nlp_options'])}" if item.get('has_nlp', False) else ""
                st.markdown(f"**{i+1}.** `{item['keyword']}` ({item['start_date']} - {item['end_date']}) via `{item['method']}`{nlp_display}")
            with col2_disp:
                if st.button("❌", key=f"delete_q_{i}", help="Hapus dari antrian"):
                    st.session_state.query_queue.pop(i); st.rerun()
    else: st.info("Tidak ada query dalam antrian.")
    if st.session_state.query_queue and st.button("🚀 Proses Semua Antrian", use_container_width=True, type="primary"):
        st.subheader("📤 Hasil Proses Scraping Antrian")
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_folder = os.path.join(desktop_path, "BursonScraperOutput")
        try: os.makedirs(output_folder, exist_ok=True); st.caption(f"File Excel akan disimpan di: `{output_folder}`.")
        except Exception as e: st.error(f"Tidak dapat membuat folder output: {e}."); output_folder = None
        processed_items_indices = []
        for idx, item in enumerate(st.session_state.query_queue):
            st.markdown(f"--- \n ### 🔄 Memproses Query #{idx+1}: `{item['keyword']}`")
            with st.spinner(f"Scraping untuk '{item['keyword']}'..."):
                result_df_q = get_news_data(item["method"], item["start_date"], item["end_date"], item["keyword"], run_interactively_selenium=False) # Queue selalu non-interaktif
            if not result_df_q.empty:
                st.success(f"✅ {len(result_df_q)} berita ditemukan untuk `{item['keyword']}`.")
                if item['has_nlp']:
                    with st.spinner(f"Menjalankan NLP untuk '{item['keyword']}'..."):
                        result_df_q = enrich_with_nlp(result_df_q.copy(), selected_nlp=item['nlp_options'])
                safe_keyword = re.sub(r"[^\w\s-]", "", item["keyword"]).replace(" ", "_")
                filename_q = f"berita_{safe_keyword}_{item['start_date']}_{item['end_date']}.xlsx"
                excel_buffer_q = io.BytesIO()
                result_df_q.to_excel(excel_buffer_q, index=False, engine='openpyxl'); excel_buffer_q.seek(0)
                if output_folder:
                    try:
                        with open(os.path.join(output_folder, filename_q), "wb") as f: f.write(excel_buffer_q.getbuffer())
                        st.caption(f"📂 File disimpan: `{os.path.join(output_folder, filename_q)}`")
                    except Exception as e: st.warning(f"Gagal menyimpan file otomatis: {e}")
                st.download_button(label=f"📥 Download hasil: '{item['keyword']}'",data=excel_buffer_q,file_name=filename_q,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"download_q_{idx}")
                processed_items_indices.append(idx)
            else: st.warning(f"⚠️ Tidak ada hasil ditemukan untuk `{item['keyword']}`.")
        for i in sorted(processed_items_indices, reverse=True): st.session_state.query_queue.pop(i)
        st.success("🎉 Semua antrian telah diproses.")
        if st.button("🔄 Bersihkan Antrian & Sembunyikan Hasil", use_container_width=True):
            st.session_state.query_queue.clear(); st.rerun()

elif menu == "NLP Tools":
    st.title("🛠️ NLP Processor (Upload File)")
    st.markdown("Upload file Excel berisi daftar URL berita (pastikan ada kolom 'Link'). Jalankan proses NLP untuk ekstraksi konten, ringkasan, sentimen, dll.")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
        df_nlp_tool = None
        if uploaded_file is not None:
            try:
                df_nlp_tool = pd.read_excel(uploaded_file)
                if 'Link' not in df_nlp_tool.columns: st.error("File Excel harus memiliki kolom 'Link'."); df_nlp_tool = None
                else:
                    with st.expander("📂 Preview File (5 baris pertama)", expanded=False): st.write(df_nlp_tool.head())
                    st.success(f"✅ File '{uploaded_file.name}' diupload ({len(df_nlp_tool)} baris).")
            except Exception as e: st.error(f"❌ Error memuat file: {e}"); df_nlp_tool = None
        st.divider()
        selected_nlp_tool = st.multiselect("Pilih fitur NLP:",["Article Content", "Summary", "Sentiment", "Keywords", "Author", "Exact Publish Date"],default=["Article Content", "Sentiment", "Author", "Exact Publish Date"],key="nlp_tool_select")
        run_nlp_button = False
        if df_nlp_tool is not None and selected_nlp_tool: run_nlp_button = st.button("🚀 Jalankan NLP pada File Terupload", type="primary")
    if df_nlp_tool is not None and selected_nlp_tool and run_nlp_button:
        with st.spinner("Memproses NLP pada file..."):
            st.session_state.processed_df_nlp_tool = enrich_with_nlp(df_nlp_tool.copy(), selected_nlp_tool)
        st.success("Proses NLP selesai!")
    if "processed_df_nlp_tool" in st.session_state:
        st.subheader("📊 Hasil Proses NLP")
        st.dataframe(st.session_state.processed_df_nlp_tool)
        to_download_nlp_tool = io.BytesIO()
        st.session_state.processed_df_nlp_tool.to_excel(to_download_nlp_tool, index=False, engine='openpyxl'); to_download_nlp_tool.seek(0)
        processed_filename = f"nlp_processed_{uploaded_file.name if uploaded_file else 'data.xlsx'}"
        st.download_button("📥 Download Hasil Proses NLP (Excel)",data=to_download_nlp_tool,file_name=processed_filename,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button("Hapus Hasil Proses NLP"): st.session_state.pop("processed_df_nlp_tool", None); st.rerun()

elif menu == "Scheduler":
    st.title("🗓️ Jadwal Scraping Otomatis")
    st.warning("🚧 Fitur Scheduler masih dalam tahap pengembangan dan belum fungsional sepenuhnya untuk eksekusi otomatis di Streamlit Cloud. Ini hanya untuk demonstrasi konsep.")
    schedules = load_schedules()
    with st.expander("➕ Tambah atau Ubah Jadwal (Konsep)"):
        with st.form("schedule_form_concept"):
            query_s = st.text_input("Keyword Boolean", key="sched_query")
            mode_s = st.selectbox("Pilih Mode Waktu", ["1 hari lalu", "Seminggu lalu", "Sebulan lalu", "Pilih tanggal"], key="sched_mode")
            col1_s, col2_s = st.columns(2); custom_start_s, custom_end_s = None, None
            if mode_s == "Pilih tanggal":
                with col1_s: custom_start_s = st.date_input("Tanggal mulai", key="sched_cstart")
                with col2_s: custom_end_s = st.date_input("Tanggal akhir", key="sched_cend")
            freq_s = st.selectbox("Frekuensi Scraping", ["Setiap hari", "Setiap minggu"], key="sched_freq")
            if freq_s == "Setiap hari": waktu_s = st.time_input("Jam scraping", key="sched_time_daily"); hari_s = None
            else: hari_s = st.selectbox("Hari dalam minggu", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], key="sched_day_weekly"); waktu_s = st.time_input("Jam scraping", key="sched_time_weekly")
            save_button_s = st.form_submit_button("Simpan Jadwal (Konsep)")
        if save_button_s:
            if not query_s.strip(): st.error("Keyword tidak boleh kosong.")
            else:
                new_schedule = {"query": query_s,"mode": mode_s,"start": custom_start_s.isoformat() if custom_start_s else None,"end": custom_end_s.isoformat() if custom_end_s else None,"frekuensi": freq_s,"hari": hari_s,"waktu": waktu_s.strftime("%H:%M"),"id": str(time.time())}
                schedules.append(new_schedule); save_schedules(schedules)
                st.success("Jadwal (konsep) berhasil disimpan!"); st.rerun()
    st.subheader("📋 Daftar Jadwal Aktif (Konsep)")
    if schedules:
        for i, sched in enumerate(schedules):
            st.markdown(f"**{i+1}. {sched['query']}** (ID: ...{sched.get('id', 'N/A')[-6:]})")
            details = f"- Mode: {sched['mode']}"
            if sched['mode'] == "Pilih tanggal": details += f", Dari: {sched.get('start', 'N/A')} s.d. {sched.get('end', 'N/A')}"
            if sched['frekuensi'] == "Setiap hari": details += f"\n- Frekuensi: Harian jam {sched['waktu']}"
            else: details += f"\n- Frekuensi: {sched['hari']} jam {sched['waktu']}"
            st.markdown(details)
            if st.button("Hapus Jadwal", key=f"del_sched_{sched.get('id', i)}"): schedules.pop(i); save_schedules(schedules); st.rerun()
            st.markdown("---")
    else: st.info("Belum ada jadwal scraping (konsep) yang ditambahkan.")

elif menu == "How to use":
    st.title("📖 Panduan Penggunaan Burson News Scraper")
    st.markdown("""
    Selamat datang di Burson News Scraper! Versi ini menyertakan beberapa upaya untuk mengurangi pemblokiran CAPTCHA dan **tidak lagi membatasi jumlah halaman scrape secara default.**

    ### Halaman Utama (`Scrape`)
    1.  **Mode Interaktif Selenium (Opsional)**:
        *   Di bagian atas halaman "Scrape", ada checkbox "Jalankan Selenium secara interaktif".
        *   Jika Anda memilih metode "Selenium" atau "Selenium DuckDuckGo" DAN mencentang kotak ini, browser akan terlihat saat scraping.
        *   Jika CAPTCHA terdeteksi, scraper akan berhenti sejenak dan meminta Anda untuk menyelesaikan CAPTCHA di browser tersebut. Setelah selesai, tekan Enter di terminal tempat Anda menjalankan Streamlit untuk melanjutkan.
        *   **Gunakan ini hanya jika Anda menjalankan Streamlit secara lokal dan bisa berinteraksi dengan browser.** Jangan centang jika berjalan di server.
    2.  **Masukkan Keyword**: Format Boolean (`"frasa" AND (kata1 OR kata2) NOT pengecualian`).
    3.  **Pilih Tanggal**.
    4.  **Metode Scraping**:
        *   **BeautifulSoup**: Cepat, lebih sedikit terdeteksi, tapi mungkin tidak bisa menangani semua situs.
        *   **Selenium / Selenium DuckDuckGo**: Lebih mampu, tapi lebih lambat dan lebih rentan terdeteksi.
    5.  **Fitur NLP (Opsional)**.
    6.  **Mulai Scrape**: Proses mungkin lebih lambat karena jeda acak yang ditambahkan untuk menghindari deteksi. **PERINGATAN:** Tanpa batas halaman, proses bisa sangat lama dan meningkatkan risiko blokir jika hasilnya banyak.

    ### Catatan Penting Anti-CAPTCHA & Tanpa Batas Halaman
    *   **Risiko Lebih Tinggi**: Menghapus batas halaman berarti scraper akan mencoba mengambil SEMUA hasil. Ini sangat meningkatkan risiko CAPTCHA atau blokir IP. Gunakan dengan sangat hati-hati.
    *   **Tidak Ada Jaminan 100%**: Upaya menghindari CAPTCHA tidak selalu berhasil.
    *   **Jeda Acak & Headers**: Scraper tetap menggunakan jeda waktu yang lebih lama dan acak antar request serta merotasi User-Agent & headers.

    (Sisa panduan sama seperti sebelumnya)
    """)

elif menu == "About":
    st.title("ℹ️ Tentang Burson News Scraper")
    st.markdown("""
    ### Versi: 1.0.7 (No Page Limit & Enhanced Anti-CAPTCHA)

    Alat ini dikembangkan untuk mempermudah proses pengumpulan dan analisis awal artikel berita dari web, dengan tambahan upaya untuk mengurangi masalah CAPTCHA. **Pembatasan jumlah halaman scrape telah dihapus secara default.**

    **Perubahan Terbaru (v1.0.7):**
    - **Penghapusan Batas Halaman Default**: Scraper kini akan mencoba mengambil semua hasil pencarian yang tersedia. Ini dapat menghasilkan lebih banyak data tetapi juga meningkatkan risiko deteksi.
    - Perbaikan kecil pada logika parsing boolean query.
    - Penyesuaian minor pada UI dan teks.

    **Fitur Utama yang Dipertahankan:**
    - Implementasi jeda acak yang signifikan antar request.
    - Rotasi User-Agent (10 variasi) dan penggunaan header browser yang lebih lengkap.
    - Penggunaan `requests.Session()` untuk metode BeautifulSoup.
    - Penambahan opsi untuk menjalankan Selenium secara interaktif guna menangani CAPTCHA manual.
    - Upaya dasar untuk menghindari deteksi headless pada Selenium.
    - Deteksi CAPTCHA sederhana pada BeautifulSoup dan Selenium.
    - Peningkatan pada logika scroll "Load More" DuckDuckGo.

    (Sisa catatan rilis dan detail sama seperti sebelumnya)

    ---

    **Dibuat oleh**: Jay dan Naomi ✨
    """)
