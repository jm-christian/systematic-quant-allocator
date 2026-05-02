import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore') 

def calculate_cvar(weights, historical_returns, confidence_level=0.95):
    portfolio_returns = historical_returns.dot(weights)
    var_threshold = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    worst_days = portfolio_returns[portfolio_returns <= var_threshold]
    if len(worst_days) == 0: return 0
    return -worst_days.mean()

# CHANGED: Default max_cvar is now 5% (0.05) instead of 2.5%
def optimize_portfolio(expected_returns, historical_returns, max_cvar=0.05):
    num_assets = len(expected_returns)
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    def objective(weights): return -(weights.dot(expected_returns))
    def weight_constraint(weights): return np.sum(weights) - 1.0
    def risk_constraint(weights): return max_cvar - calculate_cvar(weights, historical_returns)
    
    constraints = [{'type': 'eq', 'fun': weight_constraint}, {'type': 'ineq', 'fun': risk_constraint}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else np.zeros(num_assets)

def run_simulation(days_to_simulate=60, starting_capital=10000):
    print(f"Loading data for a {days_to_simulate}-day historical simulation...")
    predictions = pd.read_csv("data/sp500_predictions.csv", index_col=0, parse_dates=True)
    history = pd.read_csv("data/sp500_engineered.csv", index_col=0, parse_dates=True)
    
    unique_dates = sorted(predictions.index.unique())
    simulation_dates = unique_dates[-days_to_simulate:]
    
    start_date = simulation_dates[0]
    end_date = simulation_dates[-1]
    
    print(f"Simulation Period: {start_date.date()} to {end_date.date()}")
    
    portfolio_value = starting_capital
    portfolio_history = []
    
    print("Running daily SciPy Optimizations with 55% Conviction Threshold...")
    
    for i in range(len(simulation_dates) - 1):
        current_date = simulation_dates[i]
        next_date = simulation_dates[i+1]
        
        todays_preds = predictions[predictions.index == current_date]
        
        # THE FIX: Only look at stocks where the AI is > 55% confident
        high_conviction = todays_preds[todays_preds['AI_Probability'] > 0.55]
        top_picks = high_conviction.nlargest(20, 'AI_Probability')
        target_tickers = top_picks['Ticker'].tolist()
        
        # If the AI hates the market today, hold 100% cash (0% return)
        if not target_tickers:
            portfolio_history.append({'Date': next_date, 'Fund_Value': portfolio_value})
            continue
            
        one_year_ago = current_date - pd.Timedelta(days=365)
        risk_data = history[(history.index >= one_year_ago) & (history.index <= current_date)]
        risk_matrix = risk_data.pivot(columns='Ticker', values='Daily_Return').fillna(0)
        
        valid_tickers = [t for t in target_tickers if t in risk_matrix.columns]
        if not valid_tickers:
            portfolio_history.append({'Date': next_date, 'Fund_Value': portfolio_value})
            continue
            
        risk_matrix = risk_matrix[valid_tickers]
        expected_scores = top_picks[top_picks['Ticker'].isin(valid_tickers)]['AI_Probability'].values - 0.5
        
        # CHANGED: Passing our looser 5% max_cvar
        weights = optimize_portfolio(expected_scores, risk_matrix, max_cvar=0.05)
        
        next_day_data = history[history.index == next_date]
        daily_portfolio_return = 0
        
        for ticker, weight in zip(valid_tickers, weights):
            if weight > 0.01:
                ticker_next_day = next_day_data[next_day_data['Ticker'] == ticker]
                if not ticker_next_day.empty:
                    actual_return = ticker_next_day['Daily_Return'].values[0]
                    daily_portfolio_return += (weight * actual_return)
        
        portfolio_value = portfolio_value * (1 + daily_portfolio_return)
        portfolio_history.append({'Date': next_date, 'Fund_Value': portfolio_value})
    
    fund_df = pd.DataFrame(portfolio_history).set_index('Date')
    
    print("\nFetching S&P 500 Benchmark data...")
    spy = yf.download("SPY", start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
    spy['SPY_Return'] = spy['Close'].pct_change().fillna(0)
    spy['SPY_Value'] = starting_capital * (1 + spy['SPY_Return']).cumprod()
    
    # Safe timezone strip
    fund_df.index = pd.to_datetime(fund_df.index, utc=True).normalize().tz_localize(None)
    spy.index = pd.to_datetime(spy.index, utc=True).normalize().tz_localize(None)
    
    comparison_df = fund_df.join(spy['SPY_Value'], how='inner')
    
    fund_final = comparison_df['Fund_Value'].iloc[-1]
    spy_final = comparison_df['SPY_Value'].iloc[-1]
    
    print("\n--- 📈 60-DAY RELATIVE PERFORMANCE ---")
    print(f"Starting Capital:   ${starting_capital:,.2f}")
    print(f"S&P 500 Benchmark:  ${spy_final:,.2f} ({((spy_final/starting_capital)-1)*100:.2f}%)")
    print(f"AI Quant Fund:      ${fund_final:,.2f} ({((fund_final/starting_capital)-1)*100:.2f}%)")
    
    if fund_final > spy_final:
        print("\n🏆 Result: The AI Fund beat the market!")
    else:
        print("\n📉 Result: The AI Fund underperformed the market.")
        
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df.index, comparison_df['Fund_Value'], label='AI Quant Fund (CVaR Optimized)', color='blue', linewidth=2)
    plt.plot(comparison_df.index, comparison_df['SPY_Value'], label='S&P 500 (Benchmark)', color='gray', linestyle='--')
    plt.title('AI Systematic Fund vs S&P 500 (Last 60 Days)')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('data/relative_performance.png')
    print("\nChart saved locally as data/relative_performance.png")

if __name__ == "__main__":
    run_simulation(days_to_simulate=60)