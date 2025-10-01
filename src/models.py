


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# optional ML libraries (install if needed):
# pip install xgboost lightgbm tensorflow

# -------------------------
# XGBoost
# -------------------------

os.makedirs("models", exist_ok=True)
def train_xgboost(X, y, n_iter=25, random_state=42):
    import xgboost as xgb
    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    clf = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=random_state
    )

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X, y)

    best = search.best_estimator_
    return best, search.best_params_, search.cv_results_

# -------------------------
# LightGBM
# -------------------------
def train_lightgbm(X, y, n_iter=25, random_state=42):
    import lightgbm as lgb
    tscv = TimeSeriesSplit(n_splits=5)

    param_dist = {
        "n_estimators": [100, 200, 400],
        "num_leaves": [31, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [5, 10, 20],
    }

    clf = lgb.LGBMClassifier(random_state=random_state)

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X, y)

    best = search.best_estimator_
    return best, search.best_params_, search.cv_results_

# -------------------------
# Random Forest
# -------------------------
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return clf, {"accuracy": acc}

# -------------------------
# LSTM (sequence model)
# -------------------------
def train_lstm(dataset, feature_cols, lookback=10, epochs=20, batch_size=32):
    """
    dataset: DataFrame with Date index or column, feature_cols present, and 'target'
    Builds sliding windows of length lookback and trains a simple LSTM classifier.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    df = dataset.copy().dropna(subset=feature_cols + ["target"])

    # Build sequences
    Xs, ys = [], []
    features = df[feature_cols].values
    targets = df["target"].values
    for i in range(lookback, len(df)):
        Xs.append(features[i - lookback : i])
        ys.append(targets[i])
    Xs = np.array(Xs)
    ys = np.array(ys)

    # Train-test split (time-series)
    split = int(0.8 * len(Xs))
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]

    # Model
    model = Sequential(
        [
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return model, {"val_accuracy": float(acc)}




def train_model(dataset, model_type='xgb', ticker='AAPL', tune=True, lookback=10):
    """
    dataset: ML-ready dataframe
    model_type: 'xgb', 'lgb', 'rf', 'lstm', or 'auto'
    ticker: stock ticker
    tune: whether to run hyperparameter tuning (for xgb/lgb)
    """
    df = dataset.copy().dropna()
    drop_cols = ['Date', 'target', 'headline'] if 'headline' in df.columns else ['Date', 'target']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['target'].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"models/scaler_{ticker}.joblib")
    feature_cols = list(X.columns)
    joblib.dump(feature_cols, f"models/feature_cols_{ticker}.joblib")

    results = {}
    best_model = None
    best_score = -1
    best_type = None

    # ----------------------------
    # AUTO MODE → train all models
    # ----------------------------
    if model_type == "auto":
        candidate_models = ["xgb", "lgb", "rf", "lstm"]
        for m in candidate_models:
            print(f"\n🔍 Training candidate model: {m.upper()} ...")
            model, metrics = train_model(dataset, model_type=m, ticker=ticker, tune=tune, lookback=lookback)

            results[m] = metrics
            score = metrics.get("accuracy") or metrics.get("val_accuracy", 0)

            if score > best_score:
                best_score = score
                best_model = model
                best_type = m

        print(f"\n✅ AutoML selected best model: {best_type.upper()} with score {best_score:.4f}")
        joblib.dump(best_model, f"models/best_model_{ticker}.joblib")
        return best_model, {"best_model": best_type, "score": best_score, "all_results": results}

    # ----------------------------
    # SINGLE MODEL TRAINING
    # ----------------------------
    if model_type == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        metrics = {"accuracy": accuracy_score(y, preds)}

    elif model_type == 'lgb':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=200, random_state=42)
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        metrics = {"accuracy": accuracy_score(y, preds)}

    elif model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        metrics = {"accuracy": accuracy_score(y, preds)}

    elif model_type == 'lstm':
        model, metrics = train_lstm(dataset, feature_cols, lookback=lookback, epochs=10, batch_size=16)

    else:
        raise ValueError("Unknown model_type")

    # Save model
    joblib.dump(model, f"models/{model_type}_model_{ticker}.joblib")
    return model, metrics
