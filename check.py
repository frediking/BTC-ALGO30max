import pandas as pd

def merge_sp500_features(X_df, sp500_data_path='spX_prepared.csv'):
    """
    Merge selected S&P 500 features into the Bitcoin feature DataFrame with timezone handling.
    """
    # Load S&P 500 data
    sp500_df = pd.read_csv(sp500_data_path)
    
    # Convert dates to timezone-naive datetime
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date']).dt.tz_localize(None)
    sp500_df.set_index('Date', inplace=True)
    
    # Add lagged return features
    sp500_df['SP500_Daily_Return_Lag1'] = sp500_df['SP500_Daily_Return'].shift(1)
    sp500_df['SP500_Daily_Return_Lag3'] = sp500_df['SP500_Daily_Return'].shift(3)
    
    # Select relevant features
    sp500_features = sp500_df[[
        'SP500_Daily_Return',
        'SP500_Daily_Return_Lag1',
        'SP500_Daily_Return_Lag3',
        'SP500_Return_20d',
        'SP500_Rolling_Vol_20d'
    ]]
    
    # Load Bitcoin data and convert to timezone-naive
    X_df = pd.read_csv('Xdate_trim.csv')
    X_df['Date'] = pd.to_datetime(X_df['Date']).dt.tz_localize(None)
    X_df.set_index('Date', inplace=True)
    
    # Merge with Bitcoin data (align on date index)
    X_df = X_df.merge(sp500_features, left_index=True, right_index=True, how='left')
    
    # Fill missing values
    X_df = X_df.fillna(method='ffill').fillna(method='bfill')
    
    return X_df

if __name__ == "__main__":
    # Merge features
    X_df = merge_sp500_features(X_df='Xdate_trim.csv', sp500_data_path='spX_prepared.csv')
    
    # Save updated dataset
    X_df.to_csv('X_MACRO.csv')
    print("Features merged and saved to X_MACRO.csv")
    print(f"Final dataset shape: {X_df.shape}")