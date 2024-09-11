import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ModelProcessor:
    def __init__(self, filename, target_columns):
        self.filename = filename
        self.df = self.load_and_preprocess_data()
        self.target_columns = target_columns
        self.x_train, self.y_train, self.scaler_y = self.prepare_train_data()

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.filename, index_col='Date', parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).to_period('D')
        df['day_of_week_int'] = df.index.day_of_week
        df = df.fillna(df.mean())  # Replace NaN with column means
        return df

    def preprocess_data(self, df, columns):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[columns].values)
        return scaled_data, scaler

    def prepare_train_data(self):
        # Scale the entire dataset
        scaled_data_all, _ = self.preprocess_data(self.df, self.df.columns)
        
        # Scale only the target columns
        scaled_y_train, scaler_y = self.preprocess_data(self.df, self.target_columns)

        # Prepare x_train and y_train
        x_train = scaled_data_all[:-1]  # All rows except the last one
        y_train = scaled_y_train[1:]  # Exclude the first row for target data
        return x_train, y_train, scaler_y

    def train_random_forest(self):
        print("Training RandomForestRegressor...")
        rf_model = RandomForestRegressor(max_depth=30, max_features='sqrt', min_samples_leaf=1,min_samples_split=2, n_estimators=200)
        rf_model.fit(self.x_train, self.y_train)
        return rf_model

    def train_gradient_boosting(self):
        print("Training GradientBoostingRegressor...")
        gb_models = []
        for i in range(len(self.target_columns)):
            gb_model = GradientBoostingRegressor(
                n_estimators=1000,        # Increased number of trees
                learning_rate=0.01,       # Increased learning rate
                max_depth=6,              # Slightly deeper trees
                min_samples_split=5,      # Minimum samples required to split a node
                min_samples_leaf=4,       # Minimum samples required at a leaf node
                subsample=0.8,            # Fraction of samples used for fitting each base learner
                max_features='sqrt',      # Use a subset of features for finding the best split
                random_state=42
            )
            gb_model.fit(self.x_train, self.y_train[:, i])
            gb_models.append(gb_model)
        return gb_models

    def train_arima_models(self):
        print("Training ARIMA models...")
        arima_models = []
        for col in self.target_columns:
            train = self.df[col].dropna()
            arima_model = ARIMA(train, order=(4, 0, 4))  # Adjust order as needed
            arima_models.append(arima_model.fit())
        return arima_models

    def train_sarima_models(self):
        print("Training SARIMA models...")
        sarima_models = []
        for col in self.target_columns:
            train = self.df[col].dropna()
            sarima_model = SARIMAX(train, order=(2, 1, 3), seasonal_order=(0, 1, 0, 12),
                                   enforce_stationarity=False, enforce_invertibility=True)
            sarima_models.append(sarima_model.fit(disp=False, maxiter=500))
        return sarima_models

    def predict(self, models, test_data, model_type="forest"):
        print(f"Predicting with {model_type} models...")
        if model_type == "forest":
            rf_model = models
            predicted_scaled = rf_model.predict(test_data)
            return self.scaler_y.inverse_transform(predicted_scaled)
        elif model_type == "gb":
            gb_models = models
            predicted_gb = np.array([gb_model.predict(test_data) for gb_model in gb_models]).T
            return self.scaler_y.inverse_transform(predicted_gb)
        elif model_type == "arima":
            arima_predictions = np.array([model.forecast(steps=1) for model in models]).reshape(1, -1)
            return arima_predictions
        elif model_type == "sarima":
            sarima_predictions = np.array([model.forecast(steps=1) for model in models]).reshape(1, -1)
            return sarima_predictions

    def run_all_models(self):
        # Parallel processing for different models
        with ThreadPoolExecutor() as executor:
            future_rf = executor.submit(self.train_random_forest)
            future_gb = executor.submit(self.train_gradient_boosting)
            future_arima = executor.submit(self.train_arima_models)
            future_sarima = executor.submit(self.train_sarima_models)
            
            rf_model = future_rf.result()
            gb_models = future_gb.result()
            arima_models = future_arima.result()
            sarima_models = future_sarima.result()
            
            return rf_model, gb_models, arima_models, sarima_models

    def combine_predictions(self, rf_pred, gb_pred, arima_pred, sarima_pred):
        combined_predictions = np.mean([rf_pred, gb_pred, arima_pred, sarima_pred], axis=0)
        return combined_predictions

    def predict_next_day(self):
        test_data = self.x_train[-1].reshape(1, -1)  # Last row for prediction

        # Get trained models
        rf_model, gb_models, arima_models, sarima_models = self.run_all_models()

        # Parallel predictions
        with ThreadPoolExecutor() as executor:
            future_rf_pred = executor.submit(self.predict, rf_model, test_data, model_type="forest")
            future_gb_pred = executor.submit(self.predict, gb_models, test_data, model_type="gb")
            future_arima_pred = executor.submit(self.predict, arima_models, test_data, model_type="arima")
            future_sarima_pred = executor.submit(self.predict, sarima_models, test_data, model_type="sarima")
            
            rf_pred = future_rf_pred.result()
            gb_pred = future_gb_pred.result()
            arima_pred = future_arima_pred.result()
            sarima_pred = future_sarima_pred.result()

        # Combine predictions
        combined_predictions = self.combine_predictions(rf_pred, gb_pred, arima_pred, sarima_pred)

        # Print results
        print(rf_pred, gb_pred, arima_pred, sarima_pred)
        print("Combined predicted values for the next day:")
        for i, col in enumerate(self.target_columns):
            print(f"{col}: {combined_predictions[0, i]}")

# Usage
if __name__ == "__main__":
    target_columns = ['NIFTY50_Open', 'NIFTY50_High', 'NIFTY50_Low', 'NIFTY50_Close']
    processor = ModelProcessor('combined_nifty50_and_index_data.csv', target_columns)
    processor.predict_next_day()
