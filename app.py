import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner UI

# Custom modules
from data_loader import DataLoader              # Handles data retrieval
from models import ForecastingModels            # Forecasting models: ARIMA, SARIMA, etc.
from visualizations import Visualizer           # For plotting charts
from utils import calculate_metrics, format_currency  # Utility functions

# Streamlit page configuration
st.set_page_config(
    page_title="Stock Market Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling UI components
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-metric {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .warning-metric {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    }
    .error-metric {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
</style>
""", unsafe_allow_html=True)

def display_overview(df, symbol):
    st.subheader(f"📊 Stock Overview: {symbol}")

    if df is None or df.empty:
        st.warning("No data available to display.")
        return

    st.markdown(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")
    st.markdown(f"**Latest Closing Price:** ${df['Close'].iloc[-1]:,.2f}")
    st.markdown(f"**Average Volume:** {df['Volume'].mean():,.0f}")
    st.markdown(f"**Daily Return Volatility:** {df['Returns'].std():.2%}")
    st.markdown(f"**Daily Return Standard Deviation:** {df['Returns'].std():.2%}")
    st.markdown(f"**Daily Return Variance:** {df['Returns'].var():.2%}")


def display_forecasts():
    st.subheader("🔮 Forecasting Results")
    
    if not st.session_state.get("forecasts"):
        st.info("👆 Please load data and run analysis first.")
        return
    
    from visualizations import Visualizer
    visualizer = Visualizer()
    
    # Forecast comparison chart
    fig = visualizer.plot_forecast_comparison(
        st.session_state.stock_data,
        st.session_state.forecasts,
        st.session_state.stock_symbol
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual model forecasts
    for model_name, result in st.session_state.forecasts.items():
        with st.expander(f"📦 {model_name} Details"):
            col1, col2 = st.columns(2)

            # Forecast Table
            with col1:
                forecast_df = pd.DataFrame({
                    'Date': pd.date_range(
                        start=st.session_state.stock_data.index[-1] + timedelta(days=1),
                        periods=len(result['forecast']),
                        freq='D'
                    ),
                    'Forecast': result['forecast']
                }).head(10)
                st.dataframe(forecast_df)

            # Metrics
            with col2:
                for metric, value in result['metrics'].items():
                    st.metric(label=metric, value=f"{value:.4f}")


def display_model_comparison():
    st.subheader("📊 Model Performance Comparison")
    
    if not st.session_state.get("forecasts"):
        st.info("👆 Please run the models first to view comparison.")
        return
    
    # Create a comparison DataFrame
    rows = []
    for model_name, result in st.session_state.forecasts.items():
        row = {'Model': model_name}
        row.update(result['metrics'])
        rows.append(row)
    
    metrics_df = pd.DataFrame(rows).set_index('Model')
    
    # Display as a formatted dataframe
    st.dataframe(metrics_df.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "MAPE": "{:.2f}%",
        "R²": "{:.4f}"
    }).background_gradient(cmap='Blues'), height=300)


def display_technical_analysis(df, symbol):
    st.subheader("📈 Technical Indicators")

    viz = Visualizer()
    
    st.plotly_chart(viz.plot_moving_averages(df, symbol), use_container_width=True)
    st.plotly_chart(viz.plot_rsi(df, symbol), use_container_width=True)
    st.plotly_chart(viz.plot_bollinger_bands(df, symbol), use_container_width=True)
    st.plotly_chart(viz.plot_volume(df, symbol), use_container_width=True)


def main():
    st.markdown('<h1 class="main-header">📈 Stock Market Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize session state variables
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}

    # Sidebar configuration section
    st.sidebar.header("🔧 Configuration")
    st.sidebar.subheader("Stock Selection")

    # Dropdown for popular stocks
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
    selected_stock = st.sidebar.selectbox("Choose a stock symbol", popular_stocks)

    # Optional custom stock input
    custom_stock = st.sidebar.text_input("Or enter custom symbol", placeholder="e.g., AAPL, GOOGL")
    if custom_stock:
        selected_stock = custom_stock.upper()

    # Date range selection
    st.sidebar.subheader("Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    date_range = st.sidebar.date_input("Select date range", value=(start_date, end_date))

    # Forecast configuration
    st.sidebar.subheader("Forecasting Parameters")
    forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 365, 30, step=7)

    # Forecasting model selections
    st.sidebar.subheader("Models to Run")
    run_arima = st.sidebar.checkbox("ARIMA", value=True)
    run_sarima = st.sidebar.checkbox("SARIMA", value=True)
    run_prophet = st.sidebar.checkbox("Prophet", value=True)
    run_lstm = st.sidebar.checkbox("LSTM", value=True)

    # Run analysis button
    if st.sidebar.button("🔄 Load Data & Run Analysis", type="primary"):
        with st.spinner("Loading stock data..."):
            data_loader = DataLoader()
            try:
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df = data_loader.fetch_stock_data(selected_stock, start_date, end_date)
                    if df is not None and not df.empty:
                        st.session_state.stock_data = df
                        st.session_state.stock_symbol = selected_stock
                        st.session_state.data_loaded = True
                        st.success(f"✅ Successfully loaded {len(df)} days of data for {selected_stock}")
                    else:
                        st.error("❌ Failed to load stock data. Please check the symbol and try again.")
                        return
                else:
                    st.error("❌ Please select both start and end dates.")
                    return
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return

        # Run forecasting models
        if st.session_state.data_loaded:
            models = ForecastingModels(st.session_state.stock_data)
            st.session_state.forecasts = {}

            progress_bar = st.progress(0)
            status_text = st.empty()
            total_models = sum([run_arima, run_sarima, run_prophet, run_lstm])
            completed = 0

            if run_arima:
                status_text.text("Running ARIMA model...")
                try:
                    forecast, metrics = models.arima_forecast(forecast_days)
                    st.session_state.forecasts['ARIMA'] = {'forecast': forecast, 'metrics': metrics}
                    completed += 1
                    progress_bar.progress(completed / total_models)
                except Exception as e:
                    st.warning(f"ARIMA model failed: {str(e)}")

            if run_sarima:
                status_text.text("Running SARIMA model...")
                try:
                    forecast, metrics = models.sarima_forecast(forecast_days)
                    st.session_state.forecasts['SARIMA'] = {'forecast': forecast, 'metrics': metrics}
                    completed += 1
                    progress_bar.progress(completed / total_models)
                except Exception as e:
                    st.warning(f"SARIMA model failed: {str(e)}")

            if run_prophet:
                status_text.text("Running Prophet model...")
                try:
                    forecast, metrics = models.prophet_forecast(forecast_days)
                    st.session_state.forecasts['Prophet'] = {'forecast': forecast, 'metrics': metrics}
                    completed += 1
                    progress_bar.progress(completed / total_models)
                except Exception as e:
                    st.warning(f"Prophet model failed: {str(e)}")

            if run_lstm:
                status_text.text("Running LSTM model...")
                try:
                    forecast, metrics = models.lstm_forecast(forecast_days)
                    st.session_state.forecasts['LSTM'] = {'forecast': forecast, 'metrics': metrics}
                    completed += 1
                    progress_bar.progress(completed / total_models)
                except Exception as e:
                    st.warning(f"LSTM model failed: {str(e)}")

            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")

    # Show results after analysis
    if st.session_state.data_loaded:
        display_results()


def display_results():
    df = st.session_state.stock_data
    symbol = st.session_state.stock_symbol
    tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔮 Forecasts",
    "📈 Model Comparison",
    "📋 Technical Analysis"
    ])

    with tab1:
        display_overview(df, symbol)
    with tab2:
        display_forecasts()
    with tab3:
        display_model_comparison()
    with tab4:
        display_technical_analysis(df, symbol)

# Overview metrics and price chart
# Forecast chart and metrics
# Model performance comparisons
# Technical indicators

# If run as script, launch the app
if __name__ == "__main__":
    main()
