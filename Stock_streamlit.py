import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime as dt
import streamlit as st



# User input
stock_symbol = st.text_input("Enter a stock symbol (e.g. AAPL):", "AAPL")

# Data collection
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=30)
df = yf.download(stock_symbol, start=start_date, end=end_date, interval='1d')

# Data preprocessing
df['Day'] = range(1, len(df) + 1)
X = df[['Day']]
y = df['Close']

# Model training
model = LinearRegression()
model.fit(X, y)

# Prediction
future_days = 30
future_dates = pd.date_range(start=end_date, periods=future_days)
future_df = pd.DataFrame({'Day': range(len(df) + 1, len(df) + future_days + 1)})
future_X = future_df[['Day']]
future_y = model.predict(future_X)

# Output
st.write("Next 30 days' predicted closing prices:")
predicted_df = pd.DataFrame({'Date': future_dates, 'Close': future_y})
predicted_df.set_index('Date', inplace=True)
st.line_chart(predicted_df)
