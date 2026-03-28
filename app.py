import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetcher import fetch_market_data, fetch_asset_info
from models.markowitz import optimize_markowitz, optimize_markowitz_constrained
from models.black_litterman import optimize_black_litterman
from models.hrp import optimize_hrp
from models.risk_parity import optimzie_risk_parity
from data_pipeline import (
    generate_export_report, 
    plot_monte_carlo_ef, 
    plot_backtest,
    calculate_portfolio_dividend
)

# --- Page Config ---
st.set_page_config(page_title="Portfolio Optimization Tool", layout="wide")
st.title("Portfolio Optimization Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Portfolio Parameters")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL, MSFT, GOOG, TSLA")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
apply_constraints = st.sidebar.checkbox("Apply Sector Constraints (Max 30%)", value=False)
model_choice = st.sidebar.selectbox(
    "Optimization Model", 
    [
        'Markowitz (Max Sharpe)', 
        'Risk Parity', 
        'Hierarchical Risk Parity', 
        'Black-Litterman'
    ]
)

# --- Main Execution ---
if st.sidebar.button("Optimize"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]

    with st.spinner("Crunching the numbers so you don't have to..."):
        # 1. Fetch Data
        prices, returns = fetch_market_data(tickers, start_date, end_date)
        spy_prices, _ = fetch_market_data(['SPY'], start_date, end_date)
        sector_map, div_yields = fetch_asset_info(tickers)

        # 2. Route to Model
        if model_choice == "Markowitz (Max Sharpe)":
            if apply_constraints:
                weights, performance = optimize_markowitz_constrained(prices, sector_map)
            else:
                weights, performance = optimize_markowitz(prices)

        elif model_choice == "Hierarchical Risk Parity":
            weights, performance = optimize_hrp(prices, returns)

        elif model_choice == "Risk Parity":
            weights, performance = optimzie_risk_parity(prices)

        elif model_choice == "Black-Litterman":
            st.info("Using baseline market caps and neutral 5% views for demo stability.")
            mcaps = {t: 1000000000 for t in tickers}
            views = {t: 0.05 for t in tickers}

            weights, performance = optimize_black_litterman(prices, spy_prices, mcaps, views)

        # 3. KPI Cards
        port_div_yield = calculate_portfolio_dividend(weights, div_yields)
        expected_return, volatility, sharpe = performance
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
        col2.metric("Annual Volatility", f"{volatility*100:.2f}%")
        col3.metric("Sharpe Ration", f"{sharpe:.2f}")
        col4.metric("Dividend Yield", f"{port_div_yield*100:.2f}%")

        # 4. Visuals
        st.markdown("---")
        chart_col, weight_col = st.columns([2, 1])

        with chart_col:
            st.subheader("Asset Correlation Matrix")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            st.subheader("Efficient Frontier")
            ef_fig = plot_monte_carlo_ef(prices)
            st.pyplot(ef_fig)

        with weight_col:
            st.subheader("Optimal Weights")
            st.dataframe(weights, width=True)

            csv_data = generate_export_report(weights, performance, 0.0)
            st.download_button(
                label="Download Report (CSV)",
                data=csv_data,
                file_name="optimized_portfolio.csv",
                mime="text/csv"
            )

        # 5. Backtest    
        st.markdown("---")
        st.subheader("Historical Performance Simulation")
        backtest_fig = plot_backtest(returns, weights, spy_prices)
        st.pyplot(backtest_fig)