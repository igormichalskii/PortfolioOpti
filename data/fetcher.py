import pandas as pd
import yfinance as yf
import streamlit as st
import time

@st.cache_data(ttl=3600)
def fetch_market_data(tickers, start_date, end_date):
    """Pulls historical closing prices and calculates daily returns."""
    if not tickers:
        st.error("No ticker provided")
        return pd.DataFrame()
    
    if isinstance(tickers, str):
        tickers = tickers.split()

    for attempt in range(3):
        try:
            prices = yf.download(tickers, start_date, end_date, progress=False)['Close']

            if prices.empty:
                if attempt < 2:
                    time.sleep(2)
                    continue
                st.error("No data avilable for the selected tickers.")
                return pd.DataFrame()

            if isinstance(prices, pd.Series):
                prices = prices.to_frame(tickers[0])

            prices.index = pd.to_datetime(prices.index)

            missing_data = prices.isnull().sum()
            if missing_data.any():
                st.warning(f"Filling missing data with forward/backward fill.")
                for ticker in missing_data[missing_data > 0].index:
                    st.info(f"{ticker} has {missing_data[ticker]} missing data points")

                prices = prices.fillna(method='ffill').fillna(method='bfill').dropna()

            return prices, prices.pct_change().dropna()

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

def fetch_asset_info(tickers):
    """
    Scrapes Yahoo Finance for sectors and dividend yields.
    Expected this to be slow.
    """
    sector_map = {}
    div_yields = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            sector_map[t] = info.get('sector', 'Other')
            # yfinance sometimes returns None instead of 0.0, which breaks the math later
            div_yields[t] = info.get('trailingAnnualDividendYield', 0.0) or 0.0
        except Exception:
            sector_map[t] = 'Other'
            div_yields[t] = 0.0

    return sector_map, div_yields