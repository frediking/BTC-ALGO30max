import yfinance as yf
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_sp500_data(start_date='2019-01-01', end_date=None):
    """Fetch S&P 500 data and calculate returns"""
    try:
        # Add buffer days for calculating initial returns
        start = pd.to_datetime(start_date) - timedelta(days=10)
        end = pd.to_datetime(end_date) if end_date else datetime.now()
        
        # Create data directory if it doesn't exist
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
        
        # Construct file path with date range
        file_name = f"sp500_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        sp500_path = os.path.join(data_dir, file_name)
        
        # Check if recent data exists
        if os.path.exists(sp500_path):
            last_modified = datetime.fromtimestamp(os.path.getmtime(sp500_path))
            if (datetime.now() - last_modified).days < 1:
                print(f"Loading cached S&P 500 data from {sp500_path}")
                sp500 = pd.read_csv(sp500_path, index_col=0, parse_dates=True)
                return sp500[['sp500_return', 'sp500_volatility', 'sp500_trend', 'sp500_rsi']]
        
        # Fetch new data if needed
        print(f"Downloading S&P 500 data from {start} to {end}")
        sp500 = yf.download('^GSPC', start=start, end=end)
        
        # Calculate returns and volatility
        sp500['sp500_return'] = sp500['Close'].pct_change()
        sp500['sp500_volatility'] = sp500['sp500_return'].rolling(window=7).std()
        sp500['sp500_ma20'] = sp500['Close'].rolling(window=20).mean()
        sp500['sp500_rsi'] = calculate_rsi(sp500['Close'])
        
        # Calculate correlation features
        sp500['sp500_trend'] = np.where(sp500['Close'] > sp500['sp500_ma20'], 1, -1)
        
        # Drop any rows with NaN values
        sp500 = sp500.dropna()

        # Save complete data with all columns
        sp500.to_csv(sp500_path)
        print(f"S&P 500 data saved to {sp500_path}")
        
        return sp500[['sp500_return', 'sp500_volatility', 'sp500_trend', 'sp500_rsi']]
        
    except Exception as e:
        print(f"Error fetching S&P 500 data: {str(e)}")
        raise
    

def calculate_rsi(prices, period=14):
    """Calculate RSI for S&P 500"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def merge_sp500_features(bitcoin_df):
    """Merge S&P 500 features with Bitcoin data"""
    try:
        # Get date range from Bitcoin data
        start_date = bitcoin_df.index[0]
        end_date = bitcoin_df.index[-1]
        
        # Fetch S&P 500 data
        sp500_features = fetch_sp500_data(start_date, end_date)
        
        # Merge on index (dates)
        merged_df = bitcoin_df.join(sp500_features, how='left')
        
        # Forward fill any missing values (e.g., from market holidays)
        sp500_cols = ['sp500_return', 'sp500_volatility', 'sp500_trend', 'sp500_rsi']
        merged_df[sp500_cols] = merged_df[sp500_cols].fillna(method='ffill')
        
        print(f"Added S&P 500 features: {sp500_cols}")
        return merged_df
        
    except Exception as e:
        print(f"Error merging S&P 500 features: {str(e)}")
        raise