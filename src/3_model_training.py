import pandas as pd
import xgboost as xgb

def train_multi_stock_model(file_path):
    print(f"Loading engineered panel data from {file_path}...")
    # Read the CSV and set the Date as the index
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # CRITICAL: Sort by Date! 
    # With 100 stocks stacked, we must sort by time so we don't accidentally train on 
    # Apple's 2023 data to predict Amazon's 2020 data.
    df = df.sort_index()

    features = ['SMA_10', 'SMA_50', 'RSI_14', 'Daily_Return']
    
    # Find the exact date that represents the 80% mark of our timeline
    unique_dates = df.index.unique()
    split_date_idx = int(len(unique_dates) * 0.8)
    cutoff_date = unique_dates[split_date_idx]
    
    print(f"Chronological Split: Training before {cutoff_date.date()}, Testing after...")
    
    # Split the dataset
    train_df = df[df.index < cutoff_date]
    test_df = df[df.index >= cutoff_date]
    
    X_train, y_train = train_df[features], train_df['Target']
    X_test, y_test = test_df[features], test_df['Target']

    print(f"Training AI on {len(X_train)} historical days across 100 stocks...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model trained! Generating probability rankings for the unseen future data...")
    
    # THE UPGRADE: Instead of .predict(), we use .predict_proba()
    # It returns an array of [Probability of 0, Probability of 1]. We slice [:, 1] to keep the '1'.
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Attach the probabilities to our test dataset
    results_df = test_df.copy()
    results_df['AI_Probability'] = probabilities
    
    # Keep only what the Risk Optimizer needs for Phase 4
    final_output = results_df[['Ticker', 'Close', 'Daily_Return', 'Target', 'AI_Probability']]
    
    return final_output

if __name__ == "__main__":
    raw_data_path = "data/sp500_engineered.csv"
    results = train_multi_stock_model(raw_data_path)
    
    print("\nSample of AI Probabilities:")
    print(results[['Ticker', 'AI_Probability']].head())
    
    output_path = "data/sp500_predictions.csv"
    results.to_csv(output_path)
    print(f"\nSaved {len(results)} predictions to {output_path}")