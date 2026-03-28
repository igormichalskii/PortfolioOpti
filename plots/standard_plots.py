import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pypfopt import EfficientFrontier, expected_returns, risk_models, plotting

def plot_normalized_prices(prices):
    """Plot normalized price chart"""
    fig, ax = plt.subplots(figsize=(14,8))

    if prices.empty or len(prices) == 0:
        ax.text(0.5, 0.5, 'No price data available',
                horizontalalignment='center',
                verticalalignment='center', fontsize=14)
        ax.set_title('Price Chart - No Data')
        plt.tight_layout()
        return fig
    
    try:
        normalized = prices.div(prices.iloc[0]) * 100

        palette = sns.color_palette("husl", len(normalized.columns))

        for i, column in enumerate(normalized.columns):
            ax.plot(normalized.index, normalized[column], label=column, linewidth=2, color=palette[i])

        ax.set_title('Normalized Price (Base=100)', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)

        if len(normalized.columns) > 10:

            ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5),
                      ncol=max(1, len(normalized.columns) // 20))
            
        else:
            ax.legend(fontsize=10)

        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    except Exception as e:
        import traceback
        st.error(f"Error in plot_normalized_prices: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        ax.text(0.5, 0.5, f'Error generating chart: {str(e)}',
                horizontalalignment='center',
                verticalalignment='center', fontsize=12, color='red')
        
    plt.tight_layout()
    return fig

def plot_correlation_matrix(returns, figsize=(14,10)):
    """Plot correlation heatmap"""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='coolwarrm', ax=ax, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()

    return fig

def plot_portfolio_weights(weights_df, figsize=(14,8)):
    """Plot portfolio weights bar chart"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(weights_df['Assets'], weights_df['Weight'])
    ax.set_ylabel('Weigth (%)')
    ax.set_title('Portfolio Allocation')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

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
