import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetcher import fetch_market_data, fetch_asset_info
from models.black_litterman import optimize_black_litterman
from models.hrp import optimize_hrp
from models.markowitz import optimize_markowitz, optimize_markowitz_constrained
from models.risk_parity import optimzie_risk_parity
from plots.standard_plots import plot_backtest, plot_monte_carlo_ef

def validate_tickers(tickers):
    valid, invalid = [], []
    for t in tickers:
        if not t:
            continue

        try:
            if not yf.Ticker(t).history(period="1d").empty:
                valid.append(t)
            else:
                invalid.append(t)
        except Exception as e:
            invalid.append(t)
    
    return valid, invalid

def execution(custom_tickers, selected_list, model_choice, apply_constraints, start_date, end_date):
    raw_custom = [t.strip().upper() for t in custom_tickers.split(',')] if custom_tickers else []
    all_tickers = list(set(selected_list + raw_custom))
    all_tickers = [t for t in all_tickers if t]

    with st.spinner("Checking if your custom tickers actually exist..."):
        valid_tickers, invalid_tickers = validate_tickers(all_tickers)

    if invalid_tickers:
        st.sidebar.warning(f"Ignored invalid or delisted tickers: {', '.join(invalid_tickers)}")

    if len(valid_tickers) < 2:
        st.sidebar.error("You need at least 2 valid tickers to run an optimization. Try again.")
        st.stop()

    with st.spinner("Running matrix algebra. Please hold..."):
        prices, returns = fetch_market_data(valid_tickers, start_date, end_date)
        spy_prices, _ = fetch_market_data(['SPY'], start_date, end_date)
        sector_map, div_yields = fetch_asset_info(valid_tickers)

        # 2. Route to Model
        if model_choice == 'model_markowitz' or model_choice == 'model_min_vol' or model_choice == 'model_max_quad':
            if apply_constraints:
                weights, performance = optimize_markowitz_constrained(prices, model_choice, sector_map)
            else:
                weights, performance = optimize_markowitz(prices, model_choice)

        elif model_choice == 'model_hrp':
            weights, performance = optimize_hrp(prices, returns)

        elif model_choice == 'model_risk_parity':
            weights, performance = optimzie_risk_parity(prices)

        elif model_choice == 'model_black_litterman':
            st.info("Using baseline market caps and neutral 5% views for demo stability.")
            mcaps = {t: 1000000000 for t in valid_tickers}
            views = {t: 0.05 for t in valid_tickers}

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

def calculate_portfolio_dividend(weights, div_yields):
    """Calculates the weighted average dividend yield."""
    return sum(weights.get(t, 0) * div_yields.get(t, 0) for t in weights)

def generate_export_report(weights, performance, total_div_yield):
    """Packages the weights and matrics into a CSV-ready string."""
    df_weights = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Allocation'])
    df_weights['Allocation'] = df_weights['Allocation'].apply(lambda x: f"{x*100:.2f}%")

    metrics = [
        {"Ticker": "Expected Return", "Allocation": f"{performance[0]*100:.2f}%"},
        {"Ticker": "Volatility", "Allocation": f"{performance[1]*100:.2f}%"},
        {"Ticker": "Sharpe Ratio", "Allocation": f"{performance[2]*100:.2f}"},
        {"Ticker": "Dividend Yield", "Allocation": f"{total_div_yield*100:.2f}%"}
    ]

    df_metrics = pd.DataFrame(metrics)

    report = pd.concat([df_weights, df_metrics], ignore_index=True)
    return report.to_csv(index=False).encode('utf-8')