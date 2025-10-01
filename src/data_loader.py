
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahooquery import Ticker
import requests
import tweepy   # ✅ make sure you install tweepy (pip install tweepy)
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env

FINNHUB_KEY = os.getenv("FINNHUB_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
BENZINGA_KEY = os.getenv("BENZINGA_KEY")
TWITTER_BEARER = os.getenv("TWITTER_BEARER")

# -------------------------
# 1️⃣ Load Stock Data
# -------------------------
def load_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    # Flatten MultiIndex columns
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock.columns]

    # Fix missing standard columns
    if "Close" not in stock.columns and "Adj Close" in stock.columns:
        stock["Close"] = stock["Adj Close"]
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in stock.columns:
            stock[col] = pd.NA

    stock = stock.reset_index()
    return stock

# -------------------------
# 2️⃣ Fetch Yahoo News
# -------------------------
def fetch_news_yahoo(ticker_symbol, start, end):
    ticker = Ticker(ticker_symbol)
    raw_news = ticker.news()
    news_data = []

    for article in raw_news:
        if not isinstance(article, dict):
            continue
        ts = article.get('providerPublishTime')
        title = article.get('title', '')
        if ts is None or title == '':
            continue
        date = pd.to_datetime(ts, unit='s')
        if pd.to_datetime(start) <= date <= pd.to_datetime(end):
            news_data.append({'date': date, 'headline': title})

    news_df = pd.DataFrame(news_data)
    if news_df.empty:
        news_df = pd.DataFrame({'date': [], 'headline': []})
    return news_df

# -------------------------
# 3️⃣ Fetch NewsAPI
# -------------------------
def fetch_news_newsapi(ticker_symbol, start, end, max_articles=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker_symbol,
        "from": start,
        "to": end,
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWSAPI_KEY
    }
    response = requests.get(url, params=params).json()
    articles = response.get("articles", [])

    news_data = []
    for a in articles:
        date = pd.to_datetime(a.get("publishedAt"), errors='coerce')
        headline = a.get("title", "")
        if headline and pd.notna(date):
            news_data.append({"date": date, "headline": headline})
    
    news_df = pd.DataFrame(news_data)
    if news_df.empty:
        news_df = pd.DataFrame({"date": [], "headline": []})
    return news_df

# -------------------------
# 4️⃣ Fetch Finnhub News
# -------------------------
def fetch_news_finnhub(ticker_symbol, start, end, max_articles=100):
    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker_symbol,
        "from": start,
        "to": end,
        "token": FINNHUB_KEY
    }
    response = requests.get(url, params=params).json()
    news_data = []

    for a in response[:max_articles]:
        date = pd.to_datetime(a.get("datetime"), unit='s', errors='coerce')
        headline = a.get("headline", "")
        if headline and pd.notna(date):
            news_data.append({"date": date, "headline": headline})

    news_df = pd.DataFrame(news_data)
    if news_df.empty:
        news_df = pd.DataFrame({"date": [], "headline": []})
    return news_df

# -------------------------
# 5️⃣ Fetch Benzinga News
# -------------------------
def fetch_news_benzinga(ticker_symbol, start, end, max_articles=100):
    url = "https://api.benzinga.com/api/v2/news"
    params = {
        "token": BENZINGA_KEY,
        "symbols": ticker_symbol,
        "from": start,
        "to": end,
        "limit": max_articles
    }
    try:
        response = requests.get(url, params=params).json()
        news_data = []
        for a in response.get("articles", []):
            date = pd.to_datetime(a.get("created"), errors="coerce")
            headline = a.get("title", "")
            if headline and pd.notna(date):
                news_data.append({"date": date, "headline": headline})
        news_df = pd.DataFrame(news_data)
    except Exception:
        news_df = pd.DataFrame({"date": [], "headline": []})
    return news_df

# -------------------------
# 6️⃣ Fetch Twitter (Cashtag Search)
# -------------------------
def fetch_news_twitter(ticker_symbol, start, end, max_tweets=100):
    client = tweepy.Client(bearer_token=TWITTER_BEARER)
    query = f"${ticker_symbol} -is:retweet lang:en"
    try:
        tweets = client.search_recent_tweets(
            query=query,
            max_results=min(max_tweets, 100),
            tweet_fields=["created_at", "text"],
        )
        news_data = []
        if tweets.data:
            for tw in tweets.data:
                date = pd.to_datetime(tw.created_at, errors="coerce")
                text = tw.text
                news_data.append({"date": date, "headline": text})
        news_df = pd.DataFrame(news_data)
    except Exception:
        news_df = pd.DataFrame({"date": [], "headline": []})
    return news_df

# -------------------------
# 7️⃣ Add Sentiment
# -------------------------
def add_sentiment(news_df):
    analyzer = SentimentIntensityAnalyzer()
    if news_df.empty:
        news_df['sentiment'] = 0.0
        return news_df
    news_df['sentiment'] = news_df['headline'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    news_df['sentiment'] = news_df['sentiment'].fillna(0)
    return news_df

# -------------------------
# 8️⃣ Fetch Combined News
# -------------------------
def fetch_combined_news(ticker_symbol, start, end):
    yahoo = fetch_news_yahoo(ticker_symbol, start, end)
    newsapi = fetch_news_newsapi(ticker_symbol, start, end)
    finnhub = fetch_news_finnhub(ticker_symbol, start, end)
    benzinga = fetch_news_benzinga(ticker_symbol, start, end)
    twitter = fetch_news_twitter(ticker_symbol, start, end)

    combined = pd.concat([yahoo, newsapi, finnhub, benzinga, twitter], ignore_index=True)

    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = combined.dropna(subset=['date', 'headline'])
    combined = combined.drop_duplicates(subset=['date', 'headline']).reset_index(drop=True)
    combined = add_sentiment(combined)
    return combined
