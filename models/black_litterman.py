import pandas as pd
from pypfopt import EfficientFrontier, risk_models, black_litterman, BlackLittermanModel
import streamlit as st

def optimize_black_litterman(prices, benchmark_prices, mcaps, views):
    """Blends market consensus with subjective investor view."""
    try:
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Prices must be a pandas DataFrame")
        if prices.empty:
            raise ValueError("Price data is empty")
        
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        delta = black_litterman.market_implied_risk_aversion(benchmark_prices)
        prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

        bl = BlackLittermanModel(S, pi=prior, absolute_views=views)
        ef = EfficientFrontier(bl.bl_returns(), bl.bl_cov())

        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    except Exception as e:
        st.error(f"Error in portfolio computation: {str(e)}")
        n_assets = len(prices.columns)
        equal_weights = {col: 1.0/n_assets for col in prices.columns}
        return equal_weights, (0, 0, 0)