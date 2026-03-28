import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, black_litterman, HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier, EfficientRiskParity

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
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    erp = EfficientRiskParity(cov_matrix=S)
    erp.optimize()

    cleaned_weights = erp.clean_weights()
    performance = erp.portfolio_performance(verbose=False)

    return cleaned_weights, performance

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
if __name__ == "__main__":
    # Example of usage
    prices, returns = fetch_market_data(['SPY', 'AAPL', 'TSLA', 'QQQ'], '2020-01-01', '2024-01-01')
    print(f"Printing prices: \n{prices}.\nPrinting returns: \n{returns}")