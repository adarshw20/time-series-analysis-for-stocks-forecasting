import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ForecastingModels:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = MinMaxScaler()

    def _calculate_metrics(self, y_true, y_pred) -> dict:
        """Evaluate predictions using multiple metrics."""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R²': r2_score(y_true, y_pred)
        }

    def _split_data(self, ts: pd.Series, split_ratio: float = 0.8):
        """Split time series data into training and testing sets."""
        split = int(len(ts) * split_ratio)
        return ts[:split], ts[split:]

    def arima_forecast(self, forecast_days: int = 30):
        """Apply ARIMA model to forecast."""
        try:
            ts = self.data['Close'].dropna()
            train, test = self._split_data(ts)

            model = ARIMA(train, order=(5, 1, 0))
            fit = model.fit()

            test_pred = fit.forecast(steps=len(test))
            metrics = self._calculate_metrics(test, test_pred)

            future_forecast = fit.forecast(steps=forecast_days)
            return future_forecast.tolist(), metrics

        except Exception as e:
            raise Exception(f"ARIMA forecasting failed: {e}")

    def sarima_forecast(self, forecast_days: int = 30):
        """Apply SARIMA model for seasonal forecasting."""
        try:
            ts = self.data['Close'].dropna()
            train, test = self._split_data(ts)

            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fit = model.fit()

            test_pred = fit.forecast(steps=len(test))
            metrics = self._calculate_metrics(test, test_pred)

            future_forecast = fit.forecast(steps=forecast_days)
            return future_forecast.tolist(), metrics

        except Exception as e:
            raise Exception(f"SARIMA forecasting failed: {e}")

    def prophet_forecast(self, forecast_days: int = 30):
        """Use Facebook Prophet for time series forecasting."""
        try:
            df = self.data[['Close']].reset_index()
            df.columns = ['ds', 'y']
            train, test = self._split_data(df)

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(train)

            test_forecast = model.predict(test[['ds']])
            metrics = self._calculate_metrics(test['y'], test_forecast['yhat'])

            future = model.make_future_dataframe(periods=forecast_days)
            full_forecast = model.predict(future)
            future_values = full_forecast['yhat'].tail(forecast_days)

            return future_values.tolist(), metrics

        except Exception as e:
            raise Exception(f"Prophet forecasting failed: {e}")

    def lstm_forecast(self, forecast_days: int = 30, look_back: int = 60):
        """Use LSTM for deep learning-based forecasting."""
        try:
            data = self.data['Close'].values.reshape(-1, 1)
            scaled = self.scaler.fit_transform(data)

            X, y = [], []
            for i in range(look_back, len(scaled)):
                X.append(scaled[i - look_back:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, look_back, 1)
            y = np.array(y)

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

            pred = model.predict(X_test, verbose=0)
            pred_inv = self.scaler.inverse_transform(pred)
            y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))

            metrics = self._calculate_metrics(y_test_inv.flatten(), pred_inv.flatten())

            # Future forecasting
            last_seq = scaled[-look_back:]
            future = []

            for _ in range(forecast_days):
                next_val = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)[0, 0]
                future.append(next_val)
                last_seq = np.append(last_seq[1:], [[next_val]], axis=0)

            future_inv = self.scaler.inverse_transform(np.array(future).reshape(-1, 1))

            return future_inv.flatten().tolist(), metrics

        except Exception as e:
            raise Exception(f"LSTM forecasting failed: {e}")
