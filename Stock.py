import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to prepare data for LSTM
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)) 

    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to visualize results with date on x-axis
def visualize_results(dates, actual, predicted):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Actual Price', color='blue', linewidth=2)
    plt.plot(dates, predicted, label='Predicted Price', color='orange', linewidth=2)
    plt.title('Stock Price Prediction using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)  
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Fetch data
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-08-31'
    data = get_stock_data(ticker, start_date, end_date)

    # Prepare data for LSTM
    look_back = 60  
    X, y, scaler = prepare_data(data, look_back)

    # Extract dates and split them into train and test sets
    dates = data.index[look_back:]
    split = int(len(X) * 0.8)
    dates_train, dates_test = dates[:split], dates[split:]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Predict on test data
    predicted = model.predict(X_test)

    # Inverse transform predictions and actual data back to original scale
    predicted_prices = scaler.inverse_transform(predicted)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate model performance
    mse = mean_squared_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    # Visualize results with date on x-axis
    visualize_results(dates_test, actual_prices, predicted_prices)



