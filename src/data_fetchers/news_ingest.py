"""Scrape Yahoo Finance company page headlines for Deepak Nitrite and compute VADER sentiment.
Simple, lightweight, and works without paid APIs.
"""
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

RAW_DIR = os.path.join("..","..","data","raw")
os.makedirs(RAW_DIR, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def fetch_yahoo_headlines(ticker: str = "DEEPAKNTR.NS"):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    if resp.status_code != 200:
        print("Yahoo Finance request failed", resp.status_code)
        return pd.DataFrame()
    soup = BeautifulSoup(resp.text, "lxml")
    items = soup.select("li.js-stream-content")
    rows = []
    for it in items:
        try:
            title_tag = it.select_one("h3")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            # Yahoo often has time in <time> tag or relative times. We'll store fetch time.
            published = datetime.utcnow()
            vs = analyzer.polarity_scores(title)
            rows.append({"datetime_utc": published, "headline": title, "sentiment": vs})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
    out_path = os.path.join(RAW_DIR, "news_headlines_yahoo.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} headlines to {out_path}")
    return df


if __name__ == "__main__":
    fetch_yahoo_headlines()