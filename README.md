# 🏛️ Systematic Quant Fund: AI Alpha + CVaR Risk Optimizer

## 📌 Overview
An institutional-grade systematic trading pipeline that dynamically allocates capital across a universe of S&P 500 equities. This system uses Machine Learning (XGBoost) to predict positive expected returns and complex calculus (SciPy SLSQP) to dynamically optimize portfolio weights while capping extreme tail-risk (Conditional Value at Risk / Expected Shortfall).

## 🛠️ Tech Stack
*   **Data & Web Scraping:** `yfinance`, `requests`, `pandas` (Bypassing anti-bot walls via `read_html`)
*   **Machine Learning:** `xgboost`, `scikit-learn`
*   **Risk Optimization:** `scipy.optimize`, `numpy`
*   **Backtesting:** Custom historical simulator with Timezone/UTC normalization.

## 🏗️ System Architecture
This fund operates on a strict 5-phase daily cycle:

1.  **Dynamic Data Pipeline (`1_advanced_data.py`):** Bypasses Wikipedia bot-blocks to dynamically scrape the live S&P 500 roster, preventing survivorship bias. Downloads historical price/volume data to create a massive panel dataset.
2.  **Cross-Sectional Feature Engineering (`2_feature_engineering.py`):** Calculates technical momentum and volatility indicators (SMA, RSI). **Note:** Implements strict `.groupby('Ticker')` logic to prevent data leakage and mathematical bleeding between distinct assets in the panel.
3.  **Alpha Generation (`3_model_training.py`):** An XGBoost engine trained on historical data. Instead of binary outputs, it utilizes `.predict_proba()` to generate a statistical confidence score for tomorrow's performance. Strict chronological splitting is enforced to prevent Look-Ahead Bias.
4.  **Risk Desk (`4_portfolio_optimizer.py`):** Takes the AI's highest-probability targets and passes them to a SciPy optimizer. The optimizer allocates portfolio weights to maximize the AI's expected return while strictly capping the 95% Expected Shortfall (CVaR) at a maximum 5.0% daily loss.
5.  **Historical Simulator (`5_historical_simulator.py`):** A custom backtesting engine that steps through time chronologically. It simulates end-of-day execution, matches timezone-stripped dates against the S&P 500 (SPY) benchmark, and calculates relative performance.

## 📊 Backtesting Realities & Lessons Learned
In a recent 60-day Out-of-Sample simulation, the fund underperformed the broader S&P 500 bull market. This provided vital insights into real-world quantitative trading:
*   **Risk vs. Reward:** A strict CVaR threshold during a high-velocity tech rally causes the optimizer to favor cash and low-beta assets, effectively acting as an "emergency brake" that protects capital but severely limits upside capture.
*   **Conviction Thresholds:** The AI must be forced to only act on high-probability signals (>55%). Trading on 51% probability creates excessive portfolio turnover and "death by a thousand cuts" during sideways market chop.
*   **The Engine is Ready:** The core infrastructure (Dynamic Universe -> ML Probabilities -> SciPy Optimization) is complete. Future iterations will focus on injecting alternative datasets and macroeconomic indicators to improve the XGBoost model's predictive edge.

## 🚀 How to Run Locally
1. Clone the repository: 
   ```bash
   git clone [https://github.com/yourusername/systematic-quant-allocator.git](https://github.com/yourusername/systematic-quant-allocator.git)