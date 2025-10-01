# 📈 Stock Market Sentiment + Price Prediction  

An **AI-powered stock forecasting system** that combines **news sentiment analysis** with **time-series price prediction**.  
The project leverages **NLP + ML/DL models** to predict stock movements and includes **multi-ticker support, technical indicators, and AutoML-style model selection**.  

---

## ✨ Features  

- 📰 **News Sentiment Analysis** (positive/negative/neutral) integrated into price prediction.  
- 💹 **Multi-Ticker Support** (analyze multiple stocks dynamically).  
- 📊 **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands, etc.  
- 🤖 **Multiple Models**: Random Forest, XGBoost, LightGBM, LSTM.  
- ⚡ **AutoML Pipeline**: Automatically trains/evaluates all models and picks the best one.  
- 🔁 **Backtesting Engine**: Simulates trading strategy with predictions for performance validation.  

---

## 🏗️ Project Structure  

stock-sentiment-prediction/
│── main.py                 # Entry point (train + backtest)
│── requirements.txt         # Dependencies
│── models/                  # Saved models + scalers
│── data/                    # Stock + news data
│── src/
│   ├── data_loader.py       # Fetch + preprocess stock & sentiment data
│   ├── feature_engineering.py # Add technical indicators
│   ├── models.py            # ML/DL training functions
│   ├── backtest.py          # Backtesting logic
│   └── utils.py             # Helper functions



git clone https://github.com/yourusername/stock-sentiment-prediction.git
cd stock-sentiment-prediction

pip install -r requirements.txt


python main.py --ticker AAPL --model xgb
python main.py --ticker TSLA --automl
python main.py --ticker MSFT --model lstm --backtest


Tech Stack

Python (Pandas, NumPy, Scikit-learn)

XGBoost, LightGBM, Random Forest

TensorFlow/Keras (LSTM)

NLTK / Hugging Face (sentiment)

Matplotlib (visualization)