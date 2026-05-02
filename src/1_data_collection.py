import yfinance as yf
import pandas as pd
import time
import os
import requests

def get_sp500_tickers(limit=100):
    """Scrapes Wikipedia for the current S&P 500 tickers."""
    print("Scraping Wikipedia for S&P 500 tickers...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # We forge a fake "User-Agent" so Wikipedia thinks we are a normal Google Chrome browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Fetch the webpage using our fake browser identity
    response = requests.get(url, headers=headers)
    
    # Hand the raw HTML string to Pandas instead of a URL
    table = pd.read_html(response.text)[0]
    tickers = table['Symbol'].tolist()
    
    # Clean up weird ticker names (e.g., BRK.B needs to be BRK-B for yfinance)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    
    print(f"Found {len(tickers)} tickers. Limiting to top {limit} for testing...")
    return tickers[:limit]

def download_multi_stock_data(tickers, years=5):
    """Downloads historical data and stacks it into a single master dataset."""
    all_data = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Fetching {years} years of data for {ticker}...")
        try:
            # Download the data
            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{years}y")
            
            if df.empty:
                print(f"   Warning: No data found for {ticker}. Skipping.")
                continue
                
            # Keep only the essential columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # CRITICAL: Add a column so the AI knows WHICH stock this is
            df['Ticker'] = ticker
            
            all_data.append(df)
            
            # Sleep for 0.5 seconds to avoid Yahoo Finance rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   Failed to download {ticker}: {e}")
            
    # Combine everything into one massive DataFrame
    print("\nMerging all stocks into a master database...")
    master_df = pd.concat(all_data)
    master_df = master_df.dropna()
    return master_df

if __name__ == "__main__":
    # Create a data folder to hold our CSVs cleanly
    if not os.path.exists("data"):
        os.makedirs("data")
        
    sp500_tickers = get_sp500_tickers(limit=10)
    master_dataset = download_multi_stock_data(sp500_tickers, years=5)
    
    print("\nSample of Master Dataset:")
    print(master_dataset[['Ticker', 'Close']].head())
    
    # Save it to our new data folder
    file_path = "data/sp500_raw_data.csv"
    master_dataset.to_csv(file_path)
    print(f"\nSaved {len(master_dataset)} rows of data to {file_path}")