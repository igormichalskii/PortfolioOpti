import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import fetch_market_data, optimize_markowitz, generate_export_report, optimize_hrp, optimize_black_litterman

# --- Page Config ---
st.set_page_config(page_title="Portfolio Optimization Tool", layout="wide")
st.title("Portfolio Optimization Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Portfolio Parameters")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL, MSFT, GOOG, TSLA")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
model_choice = st.sidebar.selectbox("Optimization Model", ['Markowitz (Max Sharpe)', 'Hierarchical Risk Parity', 'Black-Litterman'])

# --- Main Execution ---
if st.sidebar.button("Optimize"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]

    with st.spinner("Crunching the numbers so you don't have to..."):
        # Fetch Data
        prices, returns = fetch_market_data(tickers, start_date, end_date)

        # Optimize based on selection
        if model_choice == "Markowitz (Max Sharpe)":
            weights, performance = optimize_markowitz(prices)

        elif model_choice == "Hierarchical Risk Parity":
            weights, performance = optimize_hrp(prices, returns)

        elif model_choice == "Black-Litterman":
            st.info("Using baseline market caps and neutral 5% views for demo stability.")
            mcaps = {t: 1000000000 for t in tickers}
            views = {t: 0.05 for t in tickers}

            spy_prices, _ = fetch_market_data(['SPY'], start_date, end_date)

            weights, performance = optimize_black_litterman(prices, spy_prices, mcaps, views)

        expected_return, volatility, sharpe = performance

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
        col2.metric("Annual Volatility", f"{volatility*100:.2f}%")
        col3.metric("Sharpe Ration", f"{sharpe:.2f}")

        # Visuals & Weights Layout
        st.markdown("---")
        chart_col, weight_col = st.columns([2, 1])

        with chart_col:
            st.subheader("Asset Correlation Matrix")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        with weight_col:
            st.subheader("Optimal Weights")
            st.dataframe(weights, use_container_width=True)

            csv_data = generate_export_report(weights, performance, 0.0)
            st.download_button(
                label="Download Report (CSV)",
                data=csv_data,
                file_name="optimized_portfolio.csv",
                mime="text/csv"
            )