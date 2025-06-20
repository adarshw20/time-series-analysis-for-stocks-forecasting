import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common regression evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary containing MAE, RMSE, MAPE, and R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100  # prevent division by 0
    r2 = r2_score(y_true, y_pred)

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}

def format_currency(value: float) -> str:
    """Format number as USD currency string."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format number as percentage string."""
    return f"{value:.2f}%"

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.01) -> float:
    """
    Calculate the annualized Sharpe ratio.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 0.01)

    Returns:
        Sharpe ratio
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = excess_returns.mean() / (excess_returns.std() + 1e-9)  # prevent division by 0
    return sharpe * np.sqrt(252)

def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate the maximum drawdown.

    Args:
        prices: Series of closing prices

    Returns:
        Maximum drawdown value (negative float)
    """
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.min()

def detect_trend(prices: pd.Series, window: int = 20) -> str:
    """
    Detect trend direction using moving average slope.

    Args:
        prices: Series of prices
        window: Rolling window for moving average

    Returns:
        "Upward", "Downward", or "Sideways"
    """
    ma = prices.rolling(window=window).mean()
    slope = ma.diff().iloc[-1]

    if slope > 0:
        return "Upward"
    elif slope < 0:
        return "Downward"
    return "Sideways"

def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate annualized rolling volatility.

    Args:
        returns: Series of daily returns
        window: Rolling window size

    Returns:
        Series of rolling volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def support_resistance_levels(prices: pd.Series, window: int = 20, min_touches: int = 2) -> Tuple[List[float], List[float]]:
    """
    Detect support and resistance levels based on price touches.

    Args:
        prices: Series of prices
        window: Rolling window for high/low calculation
        min_touches: Minimum touches to qualify as a level

    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    highs = prices.rolling(window=window).max()
    lows = prices.rolling(window=window).min()

    support = [level for level in lows.unique() if np.sum(lows == level) >= min_touches]
    resistance = [level for level in highs.unique() if np.sum(highs == level) >= min_touches]

    return sorted(support), sorted(resistance, reverse=True)
