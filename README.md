# Quantitative Portfolio Optimization Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imichalskiportfolioopti.streamlit.app/)

## Overview
A dynamic, real-time web application built in Python that allocates capital across financial assets using advanced quantitative models. This tool ingests historical market data, calculates covariance matrices with shrinkage estimators, and outputs optimal portfolio weights to maximize risk-adjusted returns. 

It upgrades standard Modern Portfolio Theory by integrating sector constraints, subjective market views (Black-Litterman), and machine-learning-based risk distribution (Hierarchical Risk Parity).

## Core Features

* **Multi-Model Optimization:** Choose between Markowitz (Max Sharpe), Black-Litterman, Hierarchical Risk Parity (HRP), and standard Risk Parity.
* **Dynamic Data Ingestion:** Real-time historical price scraping and automatic missing-data imputation via `yfinance`.
* **Sector-Level Constraints:** Bypasses "infeasible solver" errors with dynamic allocation limits to enforce diversification.
* **Interactive Visualizations:** Generates Monte Carlo Efficient Frontier distributions and asset correlation heatmaps using `matplotlib` and `seaborn`.
* **Historical Backtesting:** Simulates optimized portfolio performance against the S&P 500 benchmark.
* **Exportable Reporting:** One-click CSV generation containing final weights, expected returns, volatility, Sharpe ratios, and dividend yields.

## The Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Reactive UI and interactive dashboarding. |
| **Math Engine** | PyPortfolioOpt, SciPy | Core convex optimization and matrix algebra. |
| **Data Pipeline** | pandas, NumPy, yfinance | Time-series manipulation and live market data scraping. |
| **Visualization** | Matplotlib, Seaborn | Rendering the Efficient Frontier and correlation matrices. |

## Running Locally

To run this matrix-crunching machine on your own hardware:

1. Clone the repository:
   ```bash
   git clone https://github.com/igormichalskii/PortfolioOpti.git
   cd PortgolioOpti 
   ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Boot the Streamlit server:
    ```bash
    streamlit run app.py
    ```

## Disclaimer
This tool is for educational purposes only. It heavily relies o nhistorical data to estimate future covariance and expected returns. As any competent quant analyst knows, the past does not perfectly predict future. Do not blindly dump your net worth into a single tech stock just because the Max Sharpe algorithm told you to.
