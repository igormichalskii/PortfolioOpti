import streamlit as st
import pandas as pd


def create_sidebar():
    TOP_100_US = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'BRK-B', 'LLY', 'AVGO', 'JPM', 
        'TSLA', 'UNH', 'V', 'XOM', 'MA', 'JNJ', 'PG', 'HD', 'COST', 'MRK', 'ABBV', 'CRM', 
        'CVX', 'AMD', 'NFLX', 'PEP', 'KO', 'BAC', 'WMT', 'TMO', 'MCD', 'CSCO', 'ACN', 
        'LIN', 'INTC', 'ABT', 'ORCL', 'CMCSA', 'DIS', 'TXN', 'DHR', 'PFE', 'AMGN', 'VZ', 
        'NKE', 'IBM', 'NEE', 'PM', 'UNP', 'HON', 'SPGI', 'RTX', 'QCOM', 'INTU', 'CAT', 
        'GE', 'NOW', 'AMAT', 'UBER', 'GS', 'BA', 'MS', 'BKNG', 'T', 'AXP', 'ISRG', 'SYK', 
        'BLK', 'MDLZ', 'TJX', 'MMC', 'C', 'VRTX', 'REGN', 'LMT', 'ADI', 'PGR', 'CVS', 
        'ZTS', 'BSX', 'CB', 'CI', 'FI', 'ETN', 'SLB', 'GILD', 'BDX', 'DE', 'MU', 'SO'
    ]

    # --- Sidebar Inputs ---
    st.sidebar.header("Parameters")

    selected_list = st.sidebar.multiselect(
        "Select from Top 100 US Stocks",
        TOP_100_US,
        default=['NVDA', 'AAPL']
    )

    custom_tickers = st.sidebar.text_input("Add Custom Tickers (comma separated)", "SPY")

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

    risk_profile = st.sidebar.selectbox(
        "Risk Profile",
        ["Conservative", "Balanced", "Aggressive"]
    )

    model_choice = st.sidebar.selectbox(
        "Optimization Model",
        ["Markowitz (Max Sharpe)", "Hierarchical Risk Parity", "Risk Parity", "Black-Litterman"]
    )

    apply_constraints = st.sidebar.checkbox("Apply Sector Constraints (Max 30%)", value=False)
    return custom_tickers, selected_list, model_choice, apply_constraints, start_date, end_date