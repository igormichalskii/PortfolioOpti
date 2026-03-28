import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pypfopt import risk_models, expected_returns 
import streamlit as st

def optimzie_risk_parity(prices):
    """
    Standard Risk Parity (Equal Risk Contribution).
    Forces every asset to contribute the exact same amount of volatility.
    """
    try:
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("Prices is not a pandas DataFrame")
        if prices.empty:
            raise ValueError("Price data is empty")
        
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        n = len(prices.columns)
        cov_matrix = S.values

        def risk_budget_objective(weights):
            port_variance = weights.T @ cov_matrix @ weights
            risk_contrib = np.multiply(weights, cov_matrix @ weights) / port_variance
            return np.sum(np.square(risk_contrib - (1 / n)))
    
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) 
        bounds = tuple((0, 1) for _ in range(n))

        init_guess = np.ones(n) / n

        result = minimize(risk_budget_objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        raw_weights = result.x
        raw_weights[raw_weights < 1e-4] = 0
        raw_weights /= np.sum(raw_weights)

        weights = dict(zip(prices.columns, raw_weights))

        annual_return = np.dot(raw_weights, mu.values)
        volatility = np.sqrt(raw_weights.T @ cov_matrix @ raw_weights)
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0.0

        return weights, (annual_return, volatility, sharpe)
    except Exception as e:
        st.error(f"Error in portfolio computation: {str(e)}")
        n_assets = len(prices.columns)
        equal_weights = {col: 1.0/n_assets for col in prices.columns}
        return equal_weights, (0, 0, 0)