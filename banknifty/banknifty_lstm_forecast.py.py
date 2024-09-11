import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from pandas.tseries.offsets import DateOffset
from datetime import timedelta

filename = 'combined_bank_and_index_data.csv'

df = pd.read_csv(filename, index_col='Date', parse_dates=True)
df.index = pd.DatetimeIndex(df.index).to_period('D')
df['day_of_week_int'] = df.index.day_of_week
df['month'] = df.index.month
df['year'] = df.index.year
df = df.fillna(df.mean())

y_train_columns = ['BANKNIFTY_Open', 'BANKNIFTY_High', 'BANKNIFTY_Low', 'BANKNIFTY_Close']

def preprocess_data(df, columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[columns].values)
    return scaled_data, scaler

# Scale the entire dataset (all columns)
scaled_data_all, scaler_all = preprocess_data(df, df.columns)

# Scale only the target columns (y_train)
scaled_y_train, scaler_y = preprocess_data(df, y_train_columns)


# Exclude the last row for x_train
x_train = scaled_data_all[:-1]  # All rows except the last one
# Reshape to 3D: (samples, timesteps, features), with timesteps=1
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

# Exclude the first row for y_train
y_train = scaled_y_train[1:]   # All rows except the first one

# Ensure y_train is correctly shaped
print('Shape of y_train:', y_train.shape)

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=12, return_sequences=True))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dense(units=len(y_train_columns)))  # Output layer matches the number of features
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build model with the updated input shape
model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]))

# Train the model
history = model.fit(x_train, y_train, batch_size=5, epochs=50)

# Make predictions for the next day
test_data = scaled_data_all[-1].reshape(1, 1, -1)  # Reshape to 3D for prediction
predicted_next_day_scaled = model.predict(test_data)

# Check predictions
print('Predicted next day scaled:', predicted_next_day_scaled)

# Inverse transform the scaled prediction back to the original scale
predicted_next_day = scaler_y.inverse_transform(predicted_next_day_scaled)

# Print the predicted values for the next day
print("Predicted values for the next day:")
for i, col in enumerate(y_train_columns):
    print(f"{col}: {predicted_next_day[0, i]}")
