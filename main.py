

import os
import pandas as pd
from src import data_loader, feature_engineering, models, backtest
import matplotlib.pyplot as plt

TICKER = "MSFT"  # Change ticker here

def main(ticker=TICKER):
    os.makedirs("models", exist_ok=True)  # Ensure models folder exists

    # 1️⃣ Load stock data
    print("📥 Loading stock price data...")
    stock_data = data_loader.load_data(ticker=ticker, start="2025-01-01", end="2025-08-30")

    # 2️⃣ Fetch news
    print(f"📰 Fetching news for {ticker}...")
    sources = {}
    for source in ["yahoo", "newsapi", "finnhub", "benzinga", "twitter"]:
        df = getattr(data_loader, f"fetch_news_{source}")(ticker, "2025-01-01", "2025-08-30")
        df = data_loader.add_sentiment(df)
        sources[source] = df
        print(f"{source.capitalize()} articles fetched: {0 if df is None else len(df)}")
    
    # 3️⃣ Merge news
    news_data = pd.concat([df for df in sources.values() if df is not None and not df.empty], ignore_index=True)
    print("Total news articles fetched:", len(news_data))

    if news_data.empty:
        print("⚠️ No news data. Exiting...")
        return

    # 4️⃣ Feature engineering
    print("⚙️ Creating features...")
    dataset = feature_engineering.create_features(stock_data, news_data, ticker=ticker)
    if dataset.empty:
        print("⚠️ Dataset empty after feature engineering. Exiting...")
        return
    print("Dataset shape:", dataset.shape)

    # 5️⃣ Train model
    model_choice = "lstm"  # options: 'xgb', 'lgb', 'rf', 'lstm'
    print(f"🤖 Training ML model: {model_choice}...")
    #model, metrics = models.train_model(dataset, model_type=model_choice, ticker=ticker)
    #print("Metrics:", metrics)
    model, results = models.train_model(dataset, model_type="auto", ticker="MSFT")
    print("AutoML Results:", results)
    print("📈 Running backtest...")

    backtest_results = backtest.run_backtest(model, dataset, ticker="MSFT", model_type=results["best_model"])
    print("Backtest:", backtest_results)

    # 6️⃣ Backtest
    #print("📈 Running backtest...")
    #backtest_results = backtest.run_backtest(model, dataset, ticker=ticker)
    #backtest_results = backtest.run_backtest(model, dataset, ticker="MSFT", model_type="lstm", lookback=10)

    print("Backtest:", backtest_results)

if __name__ == "__main__":
    main()
