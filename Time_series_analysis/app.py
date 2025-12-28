import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
import streamlit as st
from PIL import Image
warnings.filterwarnings('ignore')

# streamlit configuration
st.set_page_config(layout='wide')
st.markdown("<style>.main {padding-top: 0px;} </style>", unsafe_allow_html=True)

# title
st.sidebar.image("bit.png", use_container_width = True )
st.image("bit2.png", use_container_width = True)

# add main title
st.markdown("<h1 style='text-align: center; margin-top:-20px;' >TIME SERIES ANALYSIS WITH CRPYTOCURRENCY</h1>", unsafe_allow_html=True)

# sidebar inputs
st.sidebar.header("Model Parameters")
crypto_symbol = st.sidebar.text_input("Crptocurrency Symbol", "BTC-USD")
prediction_ahead = st.sidebar.number_input("Prediction Horizon (Days)", min_value=1,max_value=30, value=15, step=1)
show_volatility = st.sidebar.checkbox("Show Volatility Chart")
if st.sidebar.button("Predict"):

  # pull crypto data for last 1 year
  btc_data = yf.download(crypto_symbol, period='1y', interval = '1d')
  btc_data = btc_data[['Close']].dropna()
  # prepare the data for LSTM
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(btc_data)

  # correct split for training and tetsing dataset
  train_size = int(len(scaled_data) * 0.8)
  train_data = scaled_data[:train_size]
  test_data = scaled_data[train_size:]

  def create_dataset(data, time_step=1):
    X,y = [],[]
    for i in range(len(data) - time_step):
      X.append(data[i:(i+time_step), 0])
      y.append(data[i + time_step ,0 ])
    return np.array(X), np.array(y)

  # use 80% of the total data for training and 20% for testing
  time_step = 60
  X_train, y_train = create_dataset(scaled_data[:train_size], time_step)
  X_test, y_test = create_dataset(scaled_data[train_size - time_step:], time_step)

  # reshape input to be [samples, time steps,features]
  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

  # X_train[0:10]
  # y_train[0:10]

  # building LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(X_train, y_train, batch_size = 1, epochs = 5, verbose = 1)

  # make predictions
  train_predictions = model.predict(X_train)
  test_predictions = model.predict(X_test)

  # inverse transform predictions and actual values
  train_predictions = scaler.inverse_transform(train_predictions)
  y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
  test_predictions = scaler.inverse_transform(test_predictions)
  y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

  # forecasting ahead
  last_60_days = scaled_data[-time_step:]
  future_input = last_60_days.reshape(1, time_step,1)
  future_forecast = []
  for _ in range(prediction_ahead):
    next_pred = model.predict(future_input)[0,0]
    future_forecast.append(next_pred)
    next_input = np.append(future_input[0,1:], [[next_pred]], axis = 0)
    future_input = next_input.reshape(1, time_step, 1)

  future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

  # latest price and las predicted price
  latest_close_price = float(btc_data['Close'].iloc[-1])
  last_predicted_price = float(future_forecast[-1])

  # Centered layout for metrics
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
      st.markdown(
          f"""
            <div style="display: flex; justify-content: space-around;">
                <div style="background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center;">
                    <h3>Latest Price</h3>
                    <p style="font-size: 20px;">${latest_close_price:,.2f}</p>
                </div>
                <div style="background-color: #d5f5d5; color: black; padding: 10px; border-radius: 10px; text-align: center;">
                    <h3>Price After {prediction_ahead} Days</h3>
                    <p style="font-size: 20px;">${last_predicted_price:,.2f}</p>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
        )
  # plot the result
  plt.figure(figsize=(14, 5))
  plt.plot(btc_data.index, btc_data['Close'], label='Actual', color='blue')
  plt.axvline(x= btc_data.index[train_size], color= 'gray',linestyle='--',label ="Train/Test Split")

  # Train/Test predictions
  train_range = btc_data.index[time_step : train_size]
  test_range = btc_data.index[train_size : train_size + len(test_predictions)]
  plt.plot(train_range, train_predictions, label='Train Predictions', color='green')
  plt.plot(test_range, test_predictions, label='Test predictions', color='orange')

  # future predictions
  future_index = pd.date_range(start=btc_data.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
  plt.plot(future_index, future_forecast, label=f'{prediction_ahead}-Day Forecast', color='red')

  plt.title(f"{crypto_symbol} LSTM Model Predictions")
  plt.xlabel('Date')
  plt.ylabel('Price (USD)')
  plt.legend()
  st.pyplot(plt)
  #volatility chart
  if show_volatility:
    btc_data['Returns'] = btc_data['Close'].pct_change()
    rolling_window = 20
    btc_data['Volatility'] = btc_data['Returns'].rolling(window=rolling_window).std() * np.sqrt(252)

    fig_vol, ax_vol = plt.subplots(figsize=(14,5))
    ax_vol.plot(btc_data.index, btc_data['Volatility'], color='purple', label=f'{rolling_window}-Day Rolling Volatility')
    ax_vol.set_title(f'{crypto_symbol} Volatility Analysis')
    ax_vol.set_xlabel('Date')
    ax_vol.set_ylabel('Volatility (Annualized)')
    ax_vol.legend()
    st.pyplot(fig_vol)





