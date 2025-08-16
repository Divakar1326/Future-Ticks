📊 Stock Price Prediction Dashboard 🔻

An interactive Streamlit web application that predicts stock prices using LSTM (Long Short-Term Memory Neural Networks) and provides technical analysis (RSI, Moving Averages) along with latest news sentiment for selected companies.

This project leverages Yahoo Finance API for historical stock data and integrates Plotly visualizations for interactive charts.

🚀 Features

📈 Stock Data Fetching – Pulls 10 years of stock history from Yahoo Finance.

🤖 LSTM Model – Predicts future stock prices based on past performance.

🔮 Future Forecasting – User-defined prediction horizon (5 to 365 days).

📊 RSI Analysis – Shows whether stock is overbought / oversold.

📰 Stock News Integration – Fetches latest articles using News API.

📑 Model Performance Metrics – MAE, MSE, RMSE, and R² score.

💹 Visualizations – Candlestick charts, RSI plots, prediction trends.

🎨 Custom UI – Background image, styled sidebar, metric cards.

🛠️ Tech Stack

Frontend: Streamlit

Data: yFinance for stock market data

ML Model: TensorFlow (LSTM)

Data Processing: Pandas, NumPy, Scikit-learn

Visualization: Plotly, Matplotlib

API: NewsAPI for real-time stock news

📂 Project Structure
📦 Stock-Prediction-Dashboard
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
└── README.md             # Documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction

2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py

📦 requirements.txt

Make sure you include these dependencies in requirements.txt:

streamlit
tensorflow
scikit-learn
pandas
numpy
matplotlib
yfinance
plotly
requests

🎮 How to Use

Launch the app with streamlit run app.py.

Select a stock ticker from the sidebar (AAPL, MSFT, TSLA, etc.) or enter your own.

Choose the number of days for future predictions.

Click Run Prediction 🚀 to see:

Candlestick charts & Moving Averages

RSI Analysis

Model predictions & forecast table

Key performance metrics

Latest stock-related news

📌 Example Dashboard

⚠️ Disclaimer

This project is built for educational purposes only.
Stock predictions are not financial advice. Please consult a professional before making investment decisions.
