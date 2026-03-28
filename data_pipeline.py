import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns, risk_models, plotting



def calculate_portfolio_dividend(weights, div_yields):
    """Calculates the weighted average dividend yield."""
    return sum(weights.get(t, 0) * div_yields.get(t, 0) for t in weights)

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