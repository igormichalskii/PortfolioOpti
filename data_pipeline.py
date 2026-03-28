import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns, risk_models, black_litterman, plotting
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.hierarchical_portfolio import HRPOpt


def fetch_market_data(tickers, start_date, end_date):
    """Pulls historical closing prices and calculates daily returns."""
    if isinstance(tickers, str):
        tickers = tickers.split()
    
    prices = yf.download(tickers, start_date, end_date)['Close']

    if isinstance(prices, pd.Series):
        prices = prices.toframe(tickers[0])

    prices = prices.ffill().dropna()
    returns = prices.pct_change().dropna()

    return prices, returns

def fetch_asset_info(tickers):
    """
    Scrapes Yahoo Finance for sectors and dividend yields.
    Expected this to be slow.
    """
    sector_map = {}
    div_yields = {}
    for t in tickers:
        try:
            info = stock.info
            sector_map[t] = info.get('sector', 'Other')
            # yfinance sometimes returns None instead of 0.0, which breaks the math later
            div_yields[t] = info.get('trailingAnnualDividendYield', 0.0) or 0.0
        except Exception:
            sector_map[t] = 'Other'
            div_yields[t] = 0.0

    return sector_map, div_yields

def calculate_portfolio_dividend(weights, div_yields):
    """Calculates the weighted average dividend yield."""
    return sum(weights.get(t, 0) * div_yields.get(t, 0) for t in weights)

def optimize_markowitz(prices):
    """Classic Mean-Variance Optimization. Maximizes the Sharpe Ratio."""
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    return ef.clean_weights(), ef.portfolio_performance(verbose=False)

def optimize_markowitz_constrained(prices, sector_map, max_weight=0.3):
    """
    Markowitz, but forces diversification.
    Caps any single sector at a maximum weight (default 30%).
    """
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

def optimize_black_litterman(prices, benchmark_prices, mcaps, views):
    """Blends market consensus with subjective investor view."""
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(benchmark_prices)
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    bl = BlackLittermanModel(S, pi=prior, absolute_views=views)
    ef = EfficientFrontier(bl.bl_returns(), bl.bl_cov())

    weights = ef.max_sharpe()
    return ef.clean_weights(), ef.portfolio_performance(verbose=False)

def optimize_hrp(prices):
    """Hierarchical Risk Parity: Machine learning clustering for risk distribution."""
    returns = prices.pct_change().dropna()
    hrp = HRPOpt(returns)
    hrp.optimize()
    return hrp.clean_weights(), hrp.portfolio_performance(verbose=False)

def optimzie_risk_parity(prices):
    """
    Standard Risk Parity (Equal Risk Contribution).
    Forces every asset to contribute the exact same amount of volatility.
    """
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

def plot_monte_carlo_ef(prices, n_portfolios=3000):
    """Draws the Efficient Frontier curve over a cloud of random portfolios."""
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(10,6))

    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

    n_assets = len(prices.columns)
    random_returns, random_volatility = [], []

    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        random_returns.append(np.dot(weights, mu))
        random_volatility.append(np.sqrt(np.dot(weights.T, np.dot(S, weights))))

    sharpe_ratios = np.array(random_returns) / np.array(random_volatility)
    sc = ax.scatter(
        random_volatility,
        random_returns,
        marker='.',
        c=sharpe_ratios,
        cmap='viridis',
        alpha=0.3,
        zorder=0
    )

    plt.colorbar(sc, label='Sharpe Ratio')
    ax.set_title("Efficient Frontier vs. 3,000 Random Portfolios")
    ax.set_xlabel("Expected Volatility (Risk)")
    ax.set_ylabel("Expected Return")

    return fig

def plot_backtest(returns, weights, benchmark_prices):
    """Simulates historical performance of the optimized weights against SPY."""
    weight_array = np.array([weights.get(ticker, 0.0) for ticker in returns.columns])
    port_daily = returns.dot(weight_array)
    port_cum = (1 + port_daily).cumprod() - 1

    bench_returns = benchmark_prices.pct_change().dropna()
    bench_returns = bench_returns.reindex(port_daily.index).fillna(0)
    bench_cum = (1 + bench_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(port_cum.index, port_cum * 100, label='Optimized Portfolio', color='#00ffcc', linewidth=2)
    ax.plot(bench_cum.index, bench_cum * 100, label='Benchmark (SPY)', color='gray', linestyle='--')

    ax.set_title("Historical Backtest: You vs. The Market")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    return fig

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