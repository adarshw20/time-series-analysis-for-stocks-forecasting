import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

class DataLoader:
    def __init__(self):
        self.data = None

    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame | None:
        """
        Fetch and preprocess stock data from Yahoo Finance.

        Parameters:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (datetime): Start of date range
            end_date (datetime): End of date range

        Returns:
            pd.DataFrame: Preprocessed stock data or None if fetching fails
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                st.error(f"⚠️ No data found for symbol: {symbol}")
                return None

            return self._preprocess_data(data)

        except Exception as e:
            st.error(f"❌ Error fetching data for {symbol}: {str(e)}")
            return None

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the stock data: clean, enrich with indicators.

        Parameters:
            data (pd.DataFrame): Raw stock data

        Returns:
            pd.DataFrame: Cleaned and feature-enriched data
        """
        # Remove timezone
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Fill missing values
        data = data.fillna(method='ffill')

        # Add features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Returns'].rolling(window=30).std()

        # Moving averages
        for ma in [10, 20, 50]:
            data[f'MA_{ma}'] = data['Close'].rolling(window=ma).mean()

        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])

        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(data['Close'])
        data['BB_Upper'] = bb_upper
        data['BB_Lower'] = bb_lower

        return data

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).

        Parameters:
            prices (pd.Series): Closing prices
            window (int): Window length for RSI

        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = -delta.clip(upper=0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Parameters:
            prices (pd.Series): Closing prices
            window (int): Moving average window
            num_std (int): Number of standard deviations for bands

        Returns:
            Tuple of upper and lower Bollinger Bands
        """
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, lower_band

    def get_stock_info(self, symbol: str) -> dict:
        """
        Retrieve basic metadata for the stock.

        Parameters:
            symbol (str): Stock ticker

        Returns:
            dict: Company name, sector, industry, and financials
        """
        try:
            info = yf.Ticker(symbol).info
            return {
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }
        except Exception:
            return {
                'company_name': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 'N/A',
                'pe_ratio': 'N/A',
                'dividend_yield': 'N/A'
            }
