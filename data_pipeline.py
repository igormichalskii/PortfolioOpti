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
    # Download the data. We only care about Close, since now Close column is adjusted for splits and dividends.
    print(f"Fetching data for: {tickers}.")
    data = yf.download(tickers, start_date, end_date)['Close']

    # If the user only passed one ticker, yfinance returns a Series. Force it to DataFrame.
    if isinstance(data, pd.Series):
        data = data.toframe(tickers[0])

    # Scrub the data: forward-fill missing days (weekends/holidays/glitches), then drop whatever is still broken
    prices = data.ffill().dropna()
    # Calculate daily percentage returns
    returns = prices.pct_change().dropna()

    return prices, returns

def fetch_asset_info(tickers):
    sector_map = {}
    div_yields = {}

    for t in tickers:
        stock = yf.Ticker(t)
        info = stock.info
        # Default to 'Other' and 0.0 if YF hides the data
        sector_map[t] = info.get('sector', 'Other')
        div_yields[t] = info.get('trailingAnnualDividendYield', 0.0)

    return sector_map, div_yields

def optimize_with_sectors(prices, sector_map, sector_lower, sector_upper):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)

    ef.add_sector_constraints(sector_map, sector_lower, sector_upper)

    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)
    
    return cleaned_weights, performance

def optimize_markowitz(prices):
    # Expected Returns: We assume that the past predicts the future.
    mu = expected_returns.mean_historical_return(prices)

    # Risk Model: Ledoit-Wolf, a mathematical trick to smooth out extreme outliers so the optimizer doesn't hallucinate.
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # The Optimizer
    ef = EfficientFrontier(mu, S)

    raw_weights = ef.max_sharpe()
    # Clean the weights: Round off microscopic fractions. Zeros out anything under 1%.
    cleaned_weights = ef.clean_weights()

    # Extract the metrics: Expected Annual Return, Annual Volatility, Sharpe Ratio
    performance = ef.portfolio_performance(verbose=False)

    return cleaned_weights, performance

def optimize_black_litterman(prices, benchmark_prices, mcaps, views):
    # Basic Covariance
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Market Implied Risk Aversion
    # Calculates how much risk the overall market currently tolerates.
    delta = black_litterman.market_implied_risk_aversion(benchmark_prices)

    # What the market expects these stocks to return based on their size and risk
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    # Mixes the prior with absolute views
    bl = BlackLittermanModel(S, pi=prior, absolute_views=views)
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()

    # Optimize using the new BL returns, and covariance
    ef = EfficientFrontier(bl_returns, bl_cov)
    raw_weights = ef.max_sharpe()

    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)

    return cleaned_weights, performance

def optimize_hrp(prices, returns):
    hrp = HRPOpt(returns)
    hrp.optimize()

    cleaned_weights = hrp.clean_weights()
    performance = hrp.portfolio_performance(verbose=False)

    return cleaned_weights, performance

def optimzie_risk_parity(prices):
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

def run_backtest(returns, weights, benchmark_returns):
    weight_array = np.array([weights.get(ticker, 0.0) for ticker in returns.columns])

    port_daily_returns = returns.dot(weight_array)

    port_cummulative = (1 + port_daily_returns).cumprod() - 1
    bench_cummulative = (1 + benchmark_returns).cumprod() - 1

    return port_cummulative, bench_cummulative

def calculate_portfolio_dividend(weights, div_yields):
    total_yield = sum(weights.get(t, 0) * div_yields.get(t, 0) for t in weights)
    return total_yield

def generate_export_report(weights, performance, total_div_yield):
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

def plot_monte_carlo_ef(prices, n_portfolios=3000):
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

if __name__ == "__main__":
    # Example of usage
    prices, returns = fetch_market_data(['SPY', 'AAPL', 'TSLA', 'QQQ'], '2020-01-01', '2024-01-01')
    print(f"Printing prices: \n{prices}.\nPrinting returns: \n{returns}")