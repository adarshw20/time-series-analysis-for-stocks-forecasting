import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import timedelta

class Visualizer:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#9467bd',
            'light': '#e6f3ff',
            'dark': '#2c3e50'
        }

    def plot_price_history(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Plot candlestick chart with volume."""
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol} Stock Price", "Volume")
        )

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='OHLC',
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger']
            ),
            row=1, col=1
        )

        bar_colors = ['green' if close >= open_ else 'red'
                      for close, open_ in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=bar_colors,
                name='Volume'
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f"{symbol} Price & Volume",
            yaxis_title="Price ($)",
            yaxis2_title="Volume",
            height=600,
            showlegend=False
        )
        return fig

    def plot_moving_averages(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Plot price with MA(10, 20, 50)."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name='Close Price',
            line=dict(color=self.colors['primary'], width=2)
        ))

        for ma, color in zip([10, 20, 50], [self.colors['secondary'], self.colors['success'], self.colors['danger']]):
            ma_col = f"MA_{ma}"
            if ma_col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[ma_col],
                    name=f"MA {ma}", line=dict(color=color, width=1)
                ))

        fig.update_layout(
            title=f"{symbol} Moving Averages",
            xaxis_title="Date", yaxis_title="Price ($)",
            height=500
        )
        return fig

    def plot_volume(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Plot volume with its moving average."""
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume', marker_color=self.colors['info']
        ))

        vol_ma = data['Volume'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=data.index, y=vol_ma,
            name='Volume MA(20)',
            line=dict(color=self.colors['danger'], width=2)
        ))

        fig.update_layout(
            title=f"{symbol} Volume & MA",
            xaxis_title="Date", yaxis_title="Volume",
            height=400
        )
        return fig

    def plot_rsi(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Plot Relative Strength Index."""
        fig = go.Figure()

        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['RSI'],
                name='RSI', line=dict(color=self.colors['primary'], width=2)
            ))

            for level, label, color in [(70, "Overbought", "red"), (30, "Oversold", "green"), (50, "Neutral", "gray")]:
                fig.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label)

        fig.update_layout(
            title=f"{symbol} RSI Indicator",
            xaxis_title="Date", yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        return fig

    def plot_bollinger_bands(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Plot Bollinger Bands with price."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name='Close Price', line=dict(color=self.colors['primary'], width=2)
        ))

        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'],
                name='Upper Band', line=dict(color=self.colors['danger'], width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'],
                name='Lower Band', line=dict(color=self.colors['danger'], width=1),
                fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)'
            ))

        if 'MA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MA_20'],
                name='Middle Band (MA20)',
                line=dict(color=self.colors['secondary'], width=1, dash='dash')
            ))

        fig.update_layout(
            title=f"{symbol} Bollinger Bands",
            xaxis_title="Date", yaxis_title="Price ($)",
            height=500
        )
        return fig

    def plot_forecast_comparison(self, historical_data: pd.DataFrame, forecasts: dict, symbol: str) -> go.Figure:
        """Plot future forecasts from multiple models."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='Historical Price',
            line=dict(color=self.colors['dark'], width=2)
        ))

        last_date = historical_data.index[-1]
        color_cycle = [self.colors['primary'], self.colors['secondary'],
                       self.colors['success'], self.colors['info']]

        for i, (model_name, result) in enumerate(forecasts.items()):
            forecast_values = result['forecast']
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast_values))

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                name=f"{model_name} Forecast",
                line=dict(color=color_cycle[i % len(color_cycle)], width=2, dash='dash'),
                mode='lines+markers'
            ))

        fig.update_layout(
            title=f"{symbol} Model Forecast Comparison",
            xaxis_title="Date", yaxis_title="Price ($)",
            height=600
        )
        return fig
