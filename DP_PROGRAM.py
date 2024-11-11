import warnings     #warnings library to avoid potentisl warnings
warnings.filterwarnings('ignore')
# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import plotly.graph_objects as go
from datetime import timedelta 
import requests
# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Stock Price Prediction Dashboard",page_icon='🔻')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Custom styling
st.markdown("""
    <style>
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0066cc;
    }
    .metric-label {
        font-size: 16px;
        color: #666666;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)
# function to calculate rsi value
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
#using chche
@st.cache_data
#function to generate future dates
def generate_future_dates(last_date, n_days):
    future_dates = []
    current_date = last_date
    for _ in range(n_days):
        current_date = current_date + timedelta(days=1)
        future_dates.append(current_date)
    return pd.DatetimeIndex(future_dates)
# function to get news on stock
def get_news_sentiment(stock_ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={stock_ticker}&apiKey=5438841ffa714285bcb0399f0bec8558"
        response = requests.get(url).json()
        return response['articles']
    except Exception as e:
        st.write("Error retrieving news data. Check API key or connection.")
        return []
# function to predict future values
def predict_future(model, last_sequence, scaler, n_steps, look_back):
    future_predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(n_steps):
        current_sequence_reshaped = current_sequence.reshape((1, look_back, 1))
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions)
# setting background
bg_image_url = "https://i.ibb.co/N2SVhh9/JN-3522-Future-of-Energy-Trading-ETRM-open-graph.jpg"
#bg_image_url = "https://i.ibb.co/2StYcDC/finalpp.jpg"
# CSS to set the background image and sidebar transparency
# CSS to set the background image
st.markdown(
    f"""
    <style>
    /* Background Image */
    .stApp {{
        background-image: url("{bg_image_url}");
        background-attachment: fixed;
        background-size: cover;
    }}
       /* Sidebar Styling for Matte Finish */
    [data-testid="stSidebar"] {{
        background: rgba(0, 0, 0, 0.5);  /* Dark gray with 85% opacity */
        color: white;  /* Text color */
    }}
    .stPlotlyChart {{
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .metric-card {{
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        color: #0066cc;
    }}
    .metric-label {{
        font-size: 16px;
        color: #666666;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 24px;
        font-weight: bold;
    }}
    div[data-testid="stMetricDelta"] {{
        font-size: 16px;
    }}
       </style>
    """,
    unsafe_allow_html=True
)
stocks = ('AAPL','MSFT','GOOGL','AMZN','TSLA','NFLX','NVDA','DIS','V','JNJ','PYPL','INTC','CSCO','ADBE','💼 Others')
company_name=()
# Sidebar for inputs
with st.sidebar:
    st.title('📊 Stock prediction')
    st.subheader('Stock Selection 🔍',divider='green')
    ticker_symbols = st.selectbox('Select Stocks to Analyze ✅',stocks)
    if ticker_symbols == "💼 Others": # custom stock input
        ticker_symbols = st.sidebar.text_input("(Check help❔) Enter stock here 👇 :",help=" Try Removing letters after spacil character with character itself for better understanding")# other stock input
    stock = yf.Ticker(ticker_symbols)
    company_name = stock.info.get("longName", ticker_symbols)  # longName usually contains the full company name
    st.markdown(f"## 🏢 Company Name:           {company_name}")
    st.write("---")
    future_days = st.slider('Number of days to predict 📅', 5, 365, 10)
    run_analysis = st.button('🚀 Run Prediction', use_container_width=True)
    st.write("---")
    st.subheader("🌐 App Information",divider='red') #side bar information
    st.write("📧 This app predicts stock prices using live Yahoo Finance data.")
    st.write("💡 Try selecting popular stocks or entering your own stock ticker!")
    st.write("---")
    st.sidebar.markdown("""### ⚠️ Please note that predictions are for educational purposes only.""",unsafe_allow_html=True)
# extracting stock data
try:
        stock_data = yf.Ticker(ticker_symbols)
        data = stock_data.history(period="10y") # getting 10 year stock data
        if data.empty:
                st.error(f"No data available for {company_name} try other stocks")
                pass
except Exception as e:
        st.error(f"Error fetching data : {str(e)}")
# Main content
st.title('🎯 Stock Price Prediction Dashboard') #page tittle
st.markdown('### Using LSTM with RSI Analysis 📝')
st.subheader("📄 Stock Summary",divider='green') 
try:
    summary = stock_data.info['longBusinessSummary']
    short_summary = summary[:500] + "..." if len(summary) > 500 else summary
except KeyError:
    summary = "Summary information is not available."
    short_summary = summary
# Display the short summary
st.write(short_summary)
# Add a "Read more" option if the summary is longer than 500 characters
if len(summary) > 500:
    with st.expander("read more"):
        st.write(summary)
st.write('---')
st.subheader("📰 News On Stock", divider='red')
news_articles = get_news_sentiment(company_name)  # getting news on stock
# Creating two columns for displaying two articles side by side
col1, col2 = st.columns(2)
# Display the first article in the first column
with col1:
        article = news_articles[1]
        st.write(f"**{article['title']}** - {article['source']['name']}")
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
# Display the second article in the second column
with col2:
        article = news_articles[2]
        st.write(f"**{article['title']}** - {article['source']['name']}")
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
st.write("---")
#initialize values
predictions_value={}
fig=go.Figure()
fig_pred = go.Figure()
fig_future = go.Figure()
st.markdown(f"### Analysis for {company_name} 📝") 
# Yesterday's metrics display
try:
        yesterday_data = data.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #333; margin-bottom: 15px;">📈 Yesterday's Close</h4>
                    </div>
                    """, unsafe_allow_html=True)
            st.metric(
                        "Close Price", #yesterday close price diffrance
                        f"${yesterday_data['Close']:.2f}",
                        f"{((yesterday_data['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):+.2f}%"
                    )
        with col2:
            st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #333; margin-bottom: 15px;">📊 Yesterday's Volume</h4>
                    </div>
                    """, unsafe_allow_html=True)
            st.metric(
                        "Volume", # previous day volume diffrance
                        f"{yesterday_data['Volume']:,.0f}",
                        f"{((yesterday_data['Volume'] - data['Volume'].iloc[-2]) / data['Volume'].iloc[-2] * 100):+.2f}%"
                    )
        with col3:
            st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #333; margin-bottom: 15px;">⚖️ Yesterday's Range</h4>
                    </div>
                    """, unsafe_allow_html=True)
            st.metric(
                        "High/Low",   # previous day high and low
                        f"${yesterday_data['High']:.2f}",
                        f"Low: ${yesterday_data['Low']:.2f}",
                        delta_color="off"
                    )
        with col4:
            st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #333; margin-bottom: 15px;">📉 Moving Average</h4>
                    </div>
                    """, unsafe_allow_html=True)
            ma_7 = data['Close'].rolling(window=7).mean().iloc[-1]  #calculating past 7 days average
            st.metric( 
                        "7-day MA",
                        f"${ma_7:.2f}",
                        f"{((yesterday_data['Close'] - ma_7) / ma_7 * 100):+.2f}% vs close"
                    )
except Exception as e:
    # Handle the case where data is empty or out-of-bounds
    st.error(f"Data not available.")
st.write("---")
if run_analysis:
        try:    
                # Store original data before scaling
                original_data = data.copy() 
                original_data['MA_7'] = original_data['Close'].rolling(window=7).mean()
                # Data preprocessing
                data.index = pd.to_datetime(data.index)
                data = data.dropna()
                scaler = MinMaxScaler(feature_range=(0, 1))
                data['Close'] = scaler.fit_transform(data[['Close']])
                data['RSI'] = calculate_rsi(data['Close'])
                # Prepare training/test datasets
                close_prices = data['Close'].values
# Perform train-test split (80% train, 20% test)
                split_date = data.index[-int(len(data) * 0.2)]
                train_data = data[data.index < split_date]
                test_data = data[data.index >= split_date]
                y_train, y_test = train_test_split(close_prices, test_size=0.2, shuffle=False)
                # LSTM model preparation and training
                n_features = 1
                y_train = train_data['Close'].values.reshape(-1, n_features)
                y_test = y_test.reshape(-1, n_features)
                look_back = min(20, len(train_data) // 2)
                batch_size = min(10, len(train_data) // 4)
                # evaluating train and test data
                train_generator = TimeseriesGenerator(y_train, y_train, length=look_back, batch_size=batch_size)
                test_generator = TimeseriesGenerator(y_test, y_test, length=look_back, batch_size=batch_size)
                model = Sequential([
    LSTM(50, input_shape=(look_back, n_features), return_sequences=True),
    LSTM(50),
    Dense(25),
    Dense(1)
])
                model.compile(optimizer='adam', loss='mse')
                # Training progress
                with st.expander("View Training Progress ⌛", expanded=False):
                    # Initialize the progress bar
                    progress_bar = st.progress(0)  # Assuming you're using Streamlit
                    progress_text = st.empty()  # Empty element to display progress text
# Create a variable to track early stopping
                    best_loss = float('inf')
                    no_improvement_epochs = 0
                    epochs = 100
                    start_time = time.time()
                    #progress bass for loop
                    for epoch in range(epochs):
                        history = model.fit(train_generator, epochs=1, verbose=0)
    # Get the training loss or validation loss (depending on your setup)
                        loss = history.history['loss'][0]  # Assuming 'loss' is being tracked
    # Early stopping logic
                        if loss < best_loss:
                            best_loss = loss
                            no_improvement_epochs = 0
                        else:
                            no_improvement_epochs += 1
    # Update the progress bar
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        progress_text.text(f'Training Progress: {progress * 100:.1f}%')
                        if no_improvement_epochs >= 10:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            progress_bar.progress(100)
                            progress_text.text(f'Training Progress: {1 * 100:.1f}%')
                            break
                        else:
                             epochs+=1
                    print(f"epoch:{epochs}")
                    end_time = time.time()
                print(f"total time :{end_time - start_time}")
                # Make predictions
                test_predictions = model.predict(test_generator)
                test_predictions_inv = scaler.inverse_transform(test_predictions)
                y_test_inv = scaler.inverse_transform(y_test[look_back:])
                # Generate future predictions
                last_sequence = y_test[-look_back:].reshape(-1, 1)
                future_pred = predict_future(model, last_sequence, scaler, future_days, look_back)
                future_dates = generate_future_dates(data.index[-1], future_days)
                # Store predictions
                predictions_value = {
                    'test_predictions': test_predictions_inv,
                    'actual_values': y_test_inv,
                    'test_dates': test_data.index[look_back:],
                    'future_predictions': future_pred,
                    'future_dates': future_dates,
                    'metrics': {
                        'mae': mean_absolute_error(y_test_inv, test_predictions_inv),
                        'mse': mean_squared_error(y_test_inv, test_predictions_inv),
                        'rmse': np.sqrt(mean_squared_error(y_test_inv, test_predictions_inv)),
                        'r2': r2_score(y_test_inv, test_predictions_inv)
                    }
                }
                # Display model performance metrics
                st.header("📑 Model Performance Metrics",divider='green')
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metrics = predictions_value['metrics']
                with metric_col1:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                with metric_col2:
                    st.metric("MSE", f"{metrics['mse']:.4f}")
                with metric_col3:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with metric_col4:
                    st.metric("R² Score", f"{metrics['r2']:.4f}")
                # Plotting
                st.write("---")
                st.header("💵 Price Analysis",divider='red')
                st.markdown("<br>", unsafe_allow_html=True)
                tab1, tab2, tab3, tab4 = st.tabs(["💹 Price Charts", "📍 RSI Analysis", "⏸️ Model Fit", "🎖️ Prediction"])
                with tab1:
                    st.markdown("<div style='padding: 10px;'>", unsafe_allow_html=True)
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=original_data.index,
                        open=original_data['Open'],
                        high=original_data['High'],
                        low=original_data['Low'],
                        close=original_data['Close'],
                        name="OHLC"
                    ))
                    # 7 days moving average
                    fig.add_trace(go.Scatter(
    x=original_data.index,
    y=original_data['MA_7'],
    mode='lines',
    name='MA_7',
    line=dict(color='white', width=2)
))
                    fig.update_layout(
                        title=dict(
                            text=f'{company_name} Price Movement',
                            font=dict(size=24)
                        ),
                        yaxis_title='Price ($)',
                        xaxis_title='Date',
                        template='plotly_white',
                        height=600,
                        margin=dict(l=50, r=50, t=80, b=50),
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with tab2:
                    # rassi plot
                    fig_rsi = plt.figure(figsize=(14, 6))
                    plt.plot(data.index, data['RSI'], label='RSI', color='blue', linewidth=2)
                    plt.axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought (70)')
                    plt.axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold (30)')
                    # protting overbrought and underbrought
                    plt.fill_between(data.index, data['RSI'], 70, where=(data['RSI'] >= 70), color='red', alpha=0.3)
                    plt.fill_between(data.index, data['RSI'], 30, where=(data['RSI'] <= 30), color='green', alpha=0.3)
                    plt.title(f'RSI Analysis for {company_name}')
                    plt.xlabel('Date')
                    plt.ylabel('RSI')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    st.pyplot(fig_rsi)

                # Update the predictions tab to include future predictions
                with tab3:
                    # Plot actual values
                    fig_pred.add_trace(go.Scatter(
                        x=predictions_value['test_dates'],
                        y=predictions_value['actual_values'].flatten(),
                        name='Actual',
                        mode='lines',
                        line=dict(color='blue')
                    ))
                    # Plot test predictions
                    fig_pred.add_trace(go.Scatter(
                        x=predictions_value['test_dates'],
                        y=predictions_value['test_predictions'].flatten(),
                        name='Past Predictions',
                        mode='lines',
                        line=dict(color='red', dash='dot')
                    ))                          
                    st.plotly_chart(fig_pred, use_container_width=True)
                with tab4:
                        # plot future predictions with dates
                        fig_future.add_trace(go.Scatter(
                        x=predictions_value['future_dates'],
                        y=predictions_value['future_predictions'].flatten(),
                        name='Future Predictions',
                        mode='lines',
                        line=dict(color='green', dash='dot')
                    ))
                    # Plot future predictions
                        fig_future.update_layout(
                        title=dict(
                            text=f'{company_name} Price Predictions (Including Future)',
                            font=dict(size=24)
                        ),
                        yaxis_title='Price ($)',
                        xaxis_title='Date',
                        template='plotly_white',
                        height=600,
                        margin=dict(l=50, r=50, t=80, b=50),
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                        st.plotly_chart(fig_future, use_container_width=True)
                    # plot future predictions in a table
                st.write("---")
                st.header("🔮 Future Price Predictions",divider='green')
                #storing future prediction with dates
                future_df = pd.DataFrame({
                        'Date': predictions_value['future_dates'],
                        'Predicted Price': predictions_value['future_predictions'].flatten().round(2)
                    })
                future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
                future_df.set_index('Date', inplace=True)
                st.dataframe(future_df, use_container_width=True)
                st.markdown("---")
                st.subheader("🌟 Key Matrices",divider="red")
                col1,col2,col3=st.columns(3)
                with col1: 
                    #plotting close prize
                    st.metric(
                        "Close Price 🚩",
                        f"${yesterday_data['Close']:.2f}",
                        f"{((yesterday_data['Close'] - original_data['Close'].iloc[-2]) / original_data['Close'].iloc[-2] * 100):+.2f}%"
                    )
                with col2:
                    # plotting rsi values
                    rsi = data['RSI'].iloc[-1]  # Get the last RSI value
                    st.metric(label="RSI 📍", value=f"{float(rsi):.2f}", delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
                with col3:    
                    predicted_price = predictions_value['future_predictions'][-1].item()
# Calculate the percentage change relative to yesterday's closing price
                    predicted_change = ((predicted_price - yesterday_data['Close']) / yesterday_data['Close']) * 100
                    predicted_change = predicted_change.item()  # Ensure it is a scalar for formatting

# Display the metric in Streamlit, with formatted values
                    st.metric(
    label=f"🗓️ LSTM Predicted Price ({future_days} days)",
    value=f"${predicted_price:.2f}",
    delta=f"{predicted_change:.2f}%"
)
                st.write("---")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

### APPLY FOOTER
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #B3B3B2;
        color: black;
        text-align: center;
        padding: 2px;
        line-height: 1.2;
    }
    </style>
    <div class="footer">
        <p>Developed by DIVAKAR ,NEHA, LAXNYA | © 2024 All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)