import pandas as pd
import numpy as np

# -------------------------
# Technical indicators
# -------------------------
def add_technical_indicators(df, price_col):
    """Add SMA, EMA, RSI, and MACD"""
    df = df.copy()
    
    # SMA & EMA
    df['SMA_3'] = df[price_col].rolling(3).mean()
    df['SMA_5'] = df[price_col].rolling(5).mean()
    df['EMA_3'] = df[price_col].ewm(span=3, adjust=False).mean()
    df['EMA_5'] = df[price_col].ewm(span=5, adjust=False).mean()
    
    # RSI
    delta = df[price_col].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(7).mean()
    roll_down = down.rolling(7).mean()
    RS = roll_up / (roll_down + 1e-6)
    df['RSI_7'] = 100 - (100 / (1 + RS))
    
    # MACD
    ema12 = df[price_col].ewm(span=12, adjust=False).mean()
    ema26 = df[price_col].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    return df

# -------------------------
# Merge stock + news + features
# -------------------------
import pandas as pd
import numpy as np

def create_features(stock_df, news_df, ticker='AAPL'):
    """Create ML-ready features combining stock + sentiment"""
    df = stock_df.copy()

    # -----------------------
    # 1️⃣ Ensure datetime index
    # -----------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            raise ValueError("Stock dataframe must have a 'Date' column or a DatetimeIndex")

    # -----------------------
    # 2️⃣ Identify price column
    # -----------------------
    price_col = f'Adj Close_{ticker}'
    if price_col not in df.columns:
        raise ValueError(f"Price column {price_col} not found in stock dataframe.")
    print(f"Using price column: {price_col}")

    # -----------------------
    # 3️⃣ Add technical indicators
    # -----------------------
    df['SMA_5'] = df[price_col].rolling(5).mean()
    df['SMA_10'] = df[price_col].rolling(10).mean()
    df['EMA_5'] = df[price_col].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df[price_col].ewm(span=10, adjust=False).mean()

    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Simple volatility: rolling std of returns
    df['Returns'] = df[price_col].pct_change()
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Volatility_10'] = df['Returns'].rolling(10).std()

    # -----------------------
    # 4️⃣ Aggregate news sentiment
    # -----------------------
    if news_df is not None and not news_df.empty and 'sentiment' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_daily = news_df.groupby(news_df['date'].dt.date)['sentiment'].mean().reset_index()
        news_daily.rename(columns={'sentiment': 'sentiment_avg'}, inplace=True)
        # merge on date
        df['date_only'] = df.index.date
        df = df.merge(news_daily, left_on='date_only', right_on='date', how='left')
        df['sentiment_avg'].fillna(0, inplace=True)
        df.drop(columns=['date', 'date_only'], inplace=True)
    else:
        df['sentiment_avg'] = 0

    # -----------------------
    # 5️⃣ Lagged features
    # -----------------------
    df['Returns_1'] = df['Returns'].shift(1)
    df['Returns_2'] = df['Returns'].shift(2)
    df['Sentiment_1'] = df['sentiment_avg'].shift(1)
    df['Sentiment_2'] = df['sentiment_avg'].shift(2)

    # -----------------------
    # 6️⃣ Target column
    # -----------------------
    df['target'] = (df['Returns'].shift(-1) > 0).astype(int)

    # -----------------------
    # 7️⃣ Drop rows with NaN values
    # -----------------------
    df.fillna(0, inplace=True)
    print(f"Dataset shape: {df.shape}")

    return df
