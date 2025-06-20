# Stock Market Time Series Analysis and Forecasting Dashboard

A comprehensive web application for analyzing and forecasting stock market trends using multiple time series models.

## Features

- **Interactive Stock Selection**: Choose from popular stocks or enter custom ticker symbols
- **Multiple Forecasting Models**: 
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - Prophet (Facebook's forecasting tool)
  - LSTM (Long Short-Term Memory neural networks)
- **Real-time Data**: Fetch live stock data from Yahoo Finance
- **Technical Analysis**: RSI, Bollinger Bands, Moving Averages
- **Model Comparison**: Side-by-side performance metrics
- **Beautiful Visualizations**: Interactive charts with Plotly
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-market-forecasting
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Select a Stock**: Choose from popular stocks or enter a custom ticker symbol
2. **Set Date Range**: Select the historical data period for analysis
3. **Configure Forecasting**: Set forecast horizon and select models to run
4. **Load Data**: Click "Load Data & Run Analysis" to fetch data and generate forecasts
5. **Explore Results**: Navigate through different tabs to view:
   - Stock overview with key metrics
   - Forecasting results from different models
   - Model performance comparison
   - Technical analysis indicators

## Models

### ARIMA
AutoRegressive Integrated Moving Average model for non-seasonal time series forecasting.

### SARIMA
Seasonal ARIMA extends ARIMA to handle seasonal patterns in the data.

### Prophet
Facebook's forecasting tool designed for business time series with strong seasonal effects.

### LSTM
Deep learning approach using Long Short-Term Memory neural networks for sequence prediction.

## Metrics

The application evaluates model performance using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## Technical Analysis

- **Moving Averages**: 10, 20, and 50-day moving averages
- **RSI**: Relative Strength Index for momentum analysis
- **Bollinger Bands**: Volatility bands around moving average

## Data Source

Stock data is fetched from Yahoo Finance using the `yfinance` library, providing:
- Historical OHLCV (Open, High, Low, Close, Volume) data
- Real-time stock information
- Company fundamentals

## Requirements

See `requirements.txt` for complete list of dependencies.

## License

This project is open source and available under the MIT License.# Stock-forecasting
# forecasting-stocks
