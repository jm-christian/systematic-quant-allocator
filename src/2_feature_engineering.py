import pandas as pd
import numpy as np
import os

def calculate_rsi(series, window=14):
    """Calculates RSI, but now designed to be passed into a groupby function."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def engineer_features(file_path):
    print(f"Loading raw multi-stock data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    print("Calculating Technical Indicators per stock (preventing data bleed)...")
    
    # 1. Moving Averages (Grouped by Ticker!)
    df['SMA_10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())
    df['SMA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())

    # 2. RSI (Grouped by Ticker)
    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(calculate_rsi)

    # 3. Daily Returns
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()

    # 4. Target Variable (Shifted per stock!)
    # If we don't group by ticker here, Apple's last day will look at Amazon's first day to decide if it went up.
    df['Target'] = df.groupby('Ticker')['Close'].transform(lambda x: (x.shift(-1) > x).astype(int))

    # Drop missing values (the first 50 days OF EACH STOCK will be dropped)
    print("Cleaning up missing data...")
    df = df.dropna()

    return df

if __name__ == "__main__":
    raw_data_path = "data/sp500_raw_data.csv"
    engineered_df = engineer_features(raw_data_path)
    
    # Let's verify our tickers didn't get mixed up
    print("\nFeature Engineering Complete! Sample data:")
    print(engineered_df[['Ticker', 'Close', 'SMA_10', 'Target']].head())
    
    # Save for the AI
    output_path = "data/sp500_engineered.csv"
    engineered_df.to_csv(output_path)
    print(f"\nSaved {len(engineered_df)} perfectly grouped rows to {output_path}")