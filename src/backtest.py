

def run_backtest(model, dataset, ticker="AAPL", model_type="xgb", lookback=10):
    import numpy as np
    import joblib

    feature_cols = joblib.load(f"models/feature_cols_{ticker}.joblib")
    df = dataset.copy().dropna()

    # Returns for strategy performance
    price_col = f"Adj Close_{ticker}"
    df["returns"] = df[price_col].pct_change().fillna(0)

    if model_type == "lstm":
        # Build sequences like in train_lstm
        Xs, ys = [], []
        features = df[feature_cols].values
        for i in range(lookback, len(df)):
            Xs.append(features[i - lookback : i])
        Xs = np.array(Xs)

        preds = (model.predict(Xs) > 0.5).astype(int).flatten()
        df = df.iloc[lookback:].copy()
        df["prediction"] = preds

    else:
        # For RF/XGB/LGBM
        scaler = joblib.load(f"models/scaler_{ticker}.joblib")
        X_scaled = scaler.transform(df[feature_cols])
        preds = model.predict(X_scaled)
        df["prediction"] = preds

    # Strategy return
    df["strategy_return"] = df["prediction"].shift(1).fillna(0) * df["returns"]
    cumulative_return = (1 + df["strategy_return"]).cumprod().iloc[-1] - 1

    return {"cumulative_return": cumulative_return}
