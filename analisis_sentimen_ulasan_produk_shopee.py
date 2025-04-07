"""## Import Library"""

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
from nltk.stem.snowball import SnowballStemmer
import nltk
import json
import random

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

"""## Preprocessing Text"""

# PREPROCESSING TEXT
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

"""## Feature Extraction & Data Split"""

# Menggunakan Dataset Ulasan Shopee Shopee
app_id = "com.shopee.id"
df = scrape_playstore_reviews(app_id)

# Tambahan data kompleks manual
additional_reviews = [
    "Aplikasi ini sangat membantu, tapi kadang lemot kalau sinyal buruk. Overall oke lah.",
    "Belanja pertama lancar, yang kedua barangnya lama banget dikirim. Kecewa sih.",
    "UI/UX sudah membaik dari versi sebelumnya, tapi sistem pencarian masih tidak akurat.",
    "Kenapa tiba-tiba aplikasi force close terus? Padahal sebelumnya lancar.",
    "Pengalaman belanja sangat menyenangkan, pengiriman cepat, CS responsif. Good job!",
    "Lumayan sih, kadang ada bug tapi sering update juga.",
    "Saya sudah dua kali beli di sini dan selalu memuaskan. Penjual responsif, pengiriman cepat.",
    "Setelah update terbaru, aplikasi sering ngelag. Harap segera diperbaiki.",
    "Barangnya tidak sesuai deskripsi, sangat mengecewakan dan CS tidak membantu.",
    "Fitur promo sering error saat checkout, padahal sinyal bagus dan aplikasi sudah diupdate.",
    "Packing rapi, barang aman sampai tujuan. Terima kasih Shopee!",
    "Cukup puas, hanya saja notifikasi suka telat muncul. Mohon ditingkatkan.",
    "Awalnya lancar, tapi sekarang sering keluar sendiri dari aplikasi."
]
df_complex = pd.DataFrame(additional_reviews, columns=['review'])
df = pd.concat([df, df_complex], ignore_index=True)

positive_keywords = ["bagus", "mantap", "cepat", "puas", "keren", "terbaik", "menyenangkan", "lancar", "responsif", "oke"]
negative_keywords = ["jelek", "lemot", "buruk", "error", "gagal", "parah", "kecewa", "bug", "force close"]

def label_sentiment(text):
    text = text.lower()
    if any(word in text for word in positive_keywords):
        return "positif"
    elif any(word in text for word in negative_keywords):
        return "negatif"
    else:
        return "netral"

df["label"] = df["review"].apply(label_sentiment)
df = df[df['label'].isin(['positif', 'netral', 'negatif'])]
df["clean_review"] = df["review"].apply(preprocess_text)

X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
