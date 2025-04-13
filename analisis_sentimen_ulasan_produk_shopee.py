# Import Library
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import json
import random
import time

#Download NLTK Stopwords
nltk.download('stopwords')

"""## Scraping Data"""

# Scraping Data Playsotre
def scrape_playstore_reviews(app_id, num_reviews=3000):
    reviews = []
    for page in range(1, num_reviews // 40 + 2):
        url = f"https://play.google.com/store/getreviews?authuser=0&reviewType=0&pageNum={page}&id={app_id}&reviewSortOrder=0&xhr=1"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
        }
        data = f'reviewType=0&pageNum={page}&id={app_id}&reviewSortOrder=0&xhr=1'
        response = requests.post(url, headers=headers, data=data)
        try:
            content = json.loads(response.text[6:])[0][2]
            soup = BeautifulSoup(content, 'html.parser')
            for div in soup.find_all('div', class_='review-body'):
                text = div.text.strip()
                if text:
                    reviews.append(text)
        except Exception:
            continue
        time.sleep(0.5)
        if len(reviews) >= num_reviews:
            break
    return pd.DataFrame(reviews[:num_reviews], columns=['review'])
