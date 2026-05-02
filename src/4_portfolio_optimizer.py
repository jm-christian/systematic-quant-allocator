import pandas as pd
import numpy as np
from scipy.optimize import minimize

def calculate_cvar(weights, historical_returns, confidence_level=0.95):
    """Calculates Expected Shortfall (CVaR) for the given portfolio weights."""
    portfolio_returns = historical_returns.dot(weights)
    var_threshold = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    worst_days = portfolio_returns[portfolio_returns <= var_threshold]
    
    # If no days are worse than VaR (rare edge case), return 0 risk
    if len(worst_days) == 0:
        return 0
    return -worst_days.mean()

def optimize_portfolio(expected_returns, historical_returns, max_cvar=0.02):
    """Finds the optimal weights to maximize AI score while capping CVaR risk."""
    num_assets = len(expected_returns)
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    # Maximize returns (minimize negative returns)
    def objective(weights):
        return -(weights.dot(expected_returns))
    
    # Weights must sum to 1.0 (100% of our money)
    def weight_constraint(weights):
        return np.sum(weights) - 1.0
        
    # Risk must be below our max_cvar
    def risk_constraint(weights):
        return max_cvar - calculate_cvar(weights, historical_returns)
    
    constraints = [
        {'type': 'eq', 'fun': weight_constraint},
        {'type': 'ineq', 'fun': risk_constraint}
    ]
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else np.zeros(num_assets)

def run_daily_trading_desk():
    print("Initializing Quantitative Risk Desk...")
    
    # 1. Load the AI's Predictions and Historical Data
    predictions = pd.read_csv("data/sp500_predictions.csv", index_col=0, parse_dates=True)
    history = pd.read_csv("data/sp500_engineered.csv", index_col=0, parse_dates=True)
    
    # 2. Get the "Current Day" (The very last day in our dataset)
    latest_date = predictions.index.max()
    print(f"\nTrading Date: {latest_date.date()}")
    
    todays_predictions = predictions[predictions.index == latest_date].copy()
    
    # 3. Alpha Screening: Keep only the Top 20 stocks the AI is most confident in
    top_20 = todays_predictions.nlargest(20, 'AI_Probability')
    target_tickers = top_20['Ticker'].tolist()
    
    print(f"AI identified {len(target_tickers)} high-probability targets. Passing to Risk Manager...")
    
    # 4. Prepare Risk Data: Get the last 252 trading days (1 year) of returns for these 20 stocks
    one_year_ago = latest_date - pd.Timedelta(days=365)
    risk_data = history[(history.index >= one_year_ago) & (history.index <= latest_date)]
    
    # Pivot the data so columns are Tickers and rows are Daily Returns
    risk_matrix = risk_data.pivot(columns='Ticker', values='Daily_Return').fillna(0)
    
    # Ensure our risk matrix only contains the Top 20 tickers
    risk_matrix = risk_matrix[target_tickers]
    
    # Convert AI Probabilities to an "Expected Score" (subtract 0.5 so <50% is negative)
    expected_scores = top_20['AI_Probability'].values - 0.5
    
    # 5. Run the Optimizer! (Strict Rule: Do not let daily CVaR exceed a 2.5% loss)
    print("Running Scipy SLSQP CVaR Optimizer...")
    optimal_weights = optimize_portfolio(expected_scores, risk_matrix, max_cvar=0.025)
    
    # 6. Output the Final Trade Orders
    print("\n--- ⚖️ OFFICIAL PORTFOLIO WEIGHTS FOR TOMORROW ---")
    for ticker, weight in zip(target_tickers, optimal_weights):
        if weight > 0.01: # Only print if the optimizer allocated more than 1%
            prob = top_20[top_20['Ticker'] == ticker]['AI_Probability'].values[0]
            print(f"BUY {ticker:<5} | Allocation: {weight*100:>5.1f}% | AI Confidence: {prob*100:.1f}%")

if __name__ == "__main__":
    run_daily_trading_desk()