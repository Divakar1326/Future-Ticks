# 📈 Future Ticks — Stock Price Prediction Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LSTM-Time%20Series%20Model-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="LSTM">
  <img src="https://img.shields.io/badge/yFinance-Market%20Data-6C47FF?style=for-the-badge" alt="yFinance">
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/NewsAPI-News%20Data-222222?style=for-the-badge" alt="NewsAPI">
</p>

<p align="center">
  <b>An interactive Streamlit dashboard for stock-price forecasting, technical analysis, performance evaluation, and stock-news sentiment.</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-workflow">Workflow</a> •
  <a href="#-technical-analysis">Technical Analysis</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-disclaimer">Disclaimer</a>
</p>

---

## ✨ Overview

**Future Ticks** is a Streamlit-based stock-analysis and forecasting application that combines historical market data, an **LSTM neural network**, technical indicators, interactive visualizations, and recent stock-related news.

The application uses **Yahoo Finance data** for historical prices and provides a user-defined forecasting horizon ranging from **5 to 365 days**.

It is designed as an educational project demonstrating how machine learning and financial-data analysis can be combined into an interactive dashboard.

---

# 🚀 Features

| Feature | Description |
|---|---|
| 📈 **Historical Stock Data** | Fetches up to 10 years of historical stock data |
| 🤖 **LSTM Forecasting** | Uses a Long Short-Term Memory neural network for future price prediction |
| 🔮 **Custom Forecast Horizon** | Supports user-defined future prediction periods |
| 📊 **RSI Analysis** | Displays Relative Strength Index information |
| 📉 **Moving Averages** | Visualizes moving-average trends |
| 📰 **News Integration** | Retrieves recent stock-related articles |
| 🧠 **News Sentiment** | Provides news-sentiment information for selected companies |
| 📋 **Model Metrics** | Reports MAE, MSE, RMSE, and R² |
| 🕯️ **Candlestick Charts** | Interactive price visualizations |
| 📊 **Forecast Charts** | Visualizes predicted price trends |
| 🎨 **Custom UI** | Styled Streamlit interface with dashboard-oriented presentation |

---

# 🔄 Workflow

```text
              ┌─────────────────────┐
              │   Select Stock      │
              │   / Enter Ticker    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Yahoo Finance     │
              │   Historical Data   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Preparation &  │
              │ Feature Processing  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     LSTM Model      │
              │ Training / Forecast │
              └──────────┬──────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌─────────┐ ┌──────────┐
        │ Forecast │ │ Metrics │ │Technical │
        │ Results  │ │ MAE etc.│ │ Analysis │
        └────┬─────┘ └────┬────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
               ┌────────────────────┐
               │ Streamlit Dashboard│
               └─────────┬──────────┘
                         │
                         ▼
               ┌────────────────────┐
               │ News & Sentiment   │
               └────────────────────┘
```

---

# 🤖 Forecasting Model

The project uses an **LSTM (Long Short-Term Memory)** neural network for stock-price forecasting.

LSTMs are designed to work with sequential data and can model patterns across historical observations.

The application trains the forecasting workflow using historical market data and produces predictions over a user-selected future horizon.

### 🔮 Forecast Horizon

The dashboard supports a user-defined prediction horizon from:

```text
5 days → 365 days
```

---

# 📊 Technical Analysis

The dashboard complements the forecasting model with traditional technical-analysis views.

## 📈 Moving Averages

Moving-average trends are visualized alongside historical prices to help inspect broader price movement patterns.

## 📊 RSI

The application displays the **Relative Strength Index (RSI)** to provide an additional momentum-oriented view of the selected stock.

## 🕯️ Candlestick Charts

Interactive candlestick charts provide a visual representation of historical price movement.

---

# 📰 News & Sentiment

The application integrates stock-related news through **NewsAPI**.

The dashboard can retrieve recent articles associated with the selected company and present news-sentiment information alongside the quantitative analysis.

```text
Stock Selection
      │
      ▼
NewsAPI Request
      │
      ▼
Recent Articles
      │
      ▼
Sentiment Information
      │
      ▼
Dashboard
```

---

# 📋 Model Evaluation

The project reports common regression metrics for evaluating the forecasting workflow:

| Metric | Purpose |
|---|---|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of Determination |

These metrics provide different views of prediction error and model fit.

---

# 🛠️ Tech Stack

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
<img src="https://img.shields.io/badge/yFinance-6C47FF?style=flat-square" alt="yFinance">
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib">
<img src="https://img.shields.io/badge/Requests-2CA5E0?style=flat-square&logo=python&logoColor=white" alt="Requests">
<img src="https://img.shields.io/badge/NewsAPI-222222?style=flat-square" alt="NewsAPI">

</p>

---

# 📁 Repository Structure

```text
Future-Ticks/
│
├── 📂 .devcontainer/
├── 🐍 Future_ticks.py
├── 📦 requirements.txt
├── ⚙️ runtime.txt
└── 📘 README.md
```

---

# ⚙️ Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Divakar1326/Future-Ticks.git
cd Future-Ticks
```

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Configure API Access

The project uses external services for market and news data.

Configure any required API credentials in environment variables rather than hard-coding secrets into source files.

## 5️⃣ Run the Application

```bash
streamlit run Future_ticks.py
```

---

# 🎮 How to Use

1. Launch the Streamlit application.
2. Select a supported stock ticker or enter a custom ticker.
3. Choose the desired future prediction horizon.
4. Run the forecasting workflow.
5. Explore:
   - 🕯️ Candlestick charts
   - 📈 Moving averages
   - 📊 RSI analysis
   - 🔮 Forecast results
   - 📋 Model metrics
   - 📰 Recent stock-related news
   - 🧠 News sentiment information

---

# 📦 Dependencies

The repository includes a `requirements.txt` file for the project's Python dependencies.

The documented technology stack includes:

```text
streamlit
tensorflow
scikit-learn
pandas
numpy
matplotlib
yfinance
plotly
requests
```

> Keep `requirements.txt` as the authoritative dependency list for the exact project environment.

---

# 🧠 Skills Demonstrated

- 🐍 Python application development
- 🖥️ Streamlit dashboard development
- 🧠 LSTM-based time-series forecasting
- 📊 Financial-data analysis
- 📈 Technical indicators
- 📰 News API integration
- 🧠 Sentiment-analysis integration
- 📉 Regression-model evaluation
- 📊 Interactive Plotly visualizations
- 🔌 External API integration

---

# 🎯 Project Value

This project demonstrates an end-to-end workflow that combines:

> **External data → preprocessing → sequential modeling → forecasting → quantitative evaluation → technical analysis → interactive visualization**

It is particularly useful as a portfolio example because it brings together **machine learning, data engineering, APIs, visualization, and application development** rather than presenting a standalone model notebook.

---

# 🔮 Future Improvements

Potential extensions include:

- 📊 Compare LSTM against GRU and classical forecasting methods
- 🧪 Add walk-forward validation
- 📉 Add richer volatility indicators
- 📈 Add additional technical indicators
- 📰 Improve news-sentiment analysis
- 🗃️ Add cached historical data
- 🚀 Deploy the dashboard publicly
- 🧪 Add automated tests
- 📊 Add backtesting functionality

---

# ⚠️ Disclaimer

This project is intended for **educational and experimental purposes only**.

Stock-price predictions are uncertain and should **not** be considered financial advice, investment recommendations, or guarantees of future market performance.

---

# 👨‍💻 Author
## Neha S 
**B.Tech CSE — Artificial Intelligence & Data Science**

## Divakar M

**B.Tech CSE — Artificial Intelligence & Data Science**

AI/ML • Generative AI • Python • Machine Learning

<p align="center">
  <a href="https://github.com/Divakar1326">
    <img src="https://img.shields.io/badge/GitHub-Divakar1326-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
</p>

---

<p align="center">
  ⭐ If you find this project useful, consider starring the repository.
</p>

<p align="center">
  <b>Predict 📈 • Analyze 📊 • Explore 🧠</b>
</p>
