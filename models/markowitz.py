import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
import streamlit as st

def optimize_markowitz(prices):
    """Classic Mean-Variance Optimization. Maximizes the Sharpe Ratio."""
    try:
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Prices must be a pandas DataFrame")
        if prices.empty:
            raise ValueError("Price data is empty")
        
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()

        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    
    except Exception as e:
        st.error(f"Error in portfolio computation: {str(e)}")
        n_assets = len(prices.columns)
        equal_weights = {col: 1.0/n_assets for col in prices.columns}
        return equal_weights, (0, 0, 0)
    
def optimize_markowitz_constrained(prices, sector_map, max_weight=0.3):
    """
    Markowitz, but forces diversification.
    Caps any single sector at a maximum weight (default 30%).
    """
    try:
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Prices must be a pandas DataFrame")
        if prices.empty:
            raise ValueError("Price data is empty")
        
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        ef = EfficientFrontier(mu, S)

        unique_sectors = set(sector_map.values())

        min_required_weight = 1.0 / len(unique_sectors)

        if max_weight < min_required_weight:
            max_weight = min_required_weight + 0.001

        sector_lower = {sec: 0.0 for sec in unique_sectors}
        sector_upper = {sec: max_weight for sec in unique_sectors}

        ef.add_sector_constraints(sector_map, sector_lower, sector_upper)

        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    
    except Exception as e:
        st.error(f"Error in portfolio computation: {str(e)}")
        n_assets = len(prices.columns)
        equal_weights = {col: 1.0/n_assets for col in prices.columns}
        return equal_weights, (0, 0, 0)