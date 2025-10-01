import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(news_df):
    news_df["sentiment"] = news_df["headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return news_df
