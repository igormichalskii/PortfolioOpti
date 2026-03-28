import pandas as pd
from pypfopt import HRPOpt 
import streamlit as st

def optimize_hrp(prices):
    """Hierarchical Risk Parity: Machine learning clustering for risk distribution."""
    try:
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Prices is not a pandas DataFrame")
        if prices.empty:
            raise ValueError("Price data is empty")
        
        returns = prices.pct_change().dropna()
        hrp = HRPOpt(returns)
        hrp.optimize()
        return hrp.clean_weights(), hrp.portfolio_performance(verbose=False)
    except Exception as e:
        st.error(f"Error in portfolio computatiton: {str(e)}")
        n_assets = len(prices.columns)
        equal_weights = {col: 1.0/n_assets for col in prices.columns}
        return equal_weights, (0, 0, 0)