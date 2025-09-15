# import pandas as pd
# import numpy as np
# import yfinance as yf
# from datetime import datetime, timedelta
# import os
# import time
# import warnings
# warnings.filterwarnings("ignore")

# def fetch_new_data():
#     # Read existing data
#     existing_df = pd.read_csv('Bitcoin_Historical_Data_Enhanced.csv')
#     existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    
#     # Get the last date in the existing data
#     last_date = existing_df['Date'].max()
    
#     # Fetch new data from the day after the last date
#     start_date = last_date + pd.Timedelta(days=1)
#     end_date = '2025-05-08'
    
#     # Fetch Bitcoin data from Yahoo Finance
#     btc = yf.Ticker("BTC-USD")
#     new_data = btc.history(start=start_date, end=end_date)
    
#     # Reset index to make Date a column
#     new_data = new_data.reset_index()
    
#     # Rename columns to match existing data
#     new_data = new_data.rename(columns={
#         'Date': 'Date',
#         'Open': 'Open',
#         'High': 'High',
#         'Low': 'Low',
#         'Close': 'Close',
#         'Adj Close': 'Adj Close',
#         'Volume': 'Volume'
#     })
    
    
#     # Convert Date to string format matching existing data
#     new_data['Date'] = new_data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    
#     # Append new data to existing data
#     combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    
#     # Save the combined dataset
#     combined_df.to_csv('Bitcoin_Historical_Data_Enhanced.csv', index=False)
#     print(f"Successfully added data from {start_date.date()} to {end_date}")
#     print(f"Total rows in extended dataset: {len(combined_df)}")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import time
import warnings
warnings.filterwarnings("ignore")

def fetch_with_retry(symbol, start_date, end_date, max_retries=5, initial_delay=10):
    """Fetch data with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            # Calculate delay with exponential backoff
            delay = initial_delay * (2 ** attempt)
            
            if attempt > 0:
                print(f"Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            
            print(f"Fetching {symbol} data (attempt {attempt + 1}/{max_retries})")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                return data
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
    
    return pd.DataFrame()  # Return empty DataFrame if all attempts fail

def fetch_new_data():
    """Fetch and update Bitcoin data"""
    try:
        # Read existing data
        existing_df = pd.read_csv('Bitcoin_Historical_Data_Enhanced.csv')
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        # Get the last date in the existing data
        last_date = existing_df['Date'].max()
        
        # Fetch new data from the day after the last date
        start_date = last_date + pd.Timedelta(days=1)
        end_date = datetime.now()
        
        print(f"Fetching Bitcoin data from {start_date.date()} to {end_date.date()}")
        
        # Use retry logic to fetch data
        new_data = fetch_with_retry("BTC-USD", start_date, end_date)
        
        if new_data.empty:
            print("No new data to add")
            return existing_df
        
        # Reset index and process data
        new_data = new_data.reset_index()
        
        # ...rest of your existing processing code...
        
        print(f"\nData Summary:")
        print(f"Last existing date: {last_date.date()}")
        print(f"New data range: {new_data['Date'].min().date()} to {new_data['Date'].max().date()}")
        print(f"New rows added: {len(new_data)}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error in fetch_new_data: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting Bitcoin data download...")
    try:
        updated_df = fetch_new_data()
        print("\nUpdate completed successfully!")
    except Exception as e:
        print(f"\nFailed to update data: {str(e)}")
        print("Try again later when rate limit resets (usually after 1 hour)")

    


# def calculate_rsi(series, window=14):
#     """Calculate Relative Strength Index (RSI)"""
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))

# def fetch_sp500_data(max_retries=3, delay_between_retries=5):
#     """Fetch S&P 500 data with retry logic and proper date handling"""
#     try:
#         # Set explicit date range
#         start_date = '2020-01-01'
#         end_date = datetime.now().strftime('%Y-%m-%d')
        
#         # Try to load existing data first
#         output_file = 'SP500_Historical_Data.csv'
#         if os.path.exists(output_file):
#             print(f"Found existing S&P 500 data, checking if update needed...")
            
#             # Load existing data with explicit date parsing
#             existing_data = pd.read_csv(output_file)
#             existing_data['Date'] = pd.to_datetime(existing_data['Date'], errors='coerce')
            
#             # Remove any rows with invalid dates
#             existing_data = existing_data.dropna(subset=['Date'])
            
#             if len(existing_data) > 0:
#                 last_date = existing_data['Date'].max()
                
#                 # Verify last_date is valid
#                 if pd.notna(last_date):
#                     if last_date.date() >= datetime.now().date() - timedelta(days=1):
#                         print(f"Data is up to date (last date: {last_date.date()})")
#                         return existing_data
                    
#                     start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
#                     print(f"Fetching new data from {start_date}")
#                 else:
#                     print("Invalid last date found, redownloading all data")
#             else:
#                 print("No valid data found, downloading complete dataset")
        
#         # Download with retries
#         for attempt in range(max_retries):
#             try:
#                 print(f"\nFetching S&P 500 data (attempt {attempt + 1}/{max_retries})")
#                 sp500 = yf.download('^GSPC', 
#                                   start=start_date, 
#                                   end=end_date,
#                                   progress=True,
#                                   interval='1d')
                
#                 if not sp500.empty:
#                     break
                    
#                 print(f"Attempt {attempt + 1} failed, waiting {delay_between_retries} seconds...")
#                 time.sleep(delay_between_retries)
                
#             except Exception as e:
#                 print(f"Error on attempt {attempt + 1}: {str(e)}")
#                 if attempt < max_retries - 1:
#                     time.sleep(delay_between_retries)
#                 else:
#                     raise
        
#         # Verify downloaded data
#         if sp500.empty:
#             raise ValueError("Failed to download S&P 500 data after all retries")
            
#         print(f"Successfully downloaded {len(sp500):,} rows")
        
#         # Calculate features
#         print("\nCalculating features...")
#         sp500['sp500_return'] = sp500['Close'].pct_change()
#         sp500['sp500_volatility'] = sp500['sp500_return'].rolling(window=7).std()
#         sp500['sp500_ma20'] = sp500['Close'].rolling(window=20).mean()
#         sp500['sp500_trend'] = np.where(sp500['Close'] > sp500['sp500_ma20'], 1, -1)
#         sp500['sp500_rsi'] = calculate_rsi(sp500['Close'])
        
#         # Reset index and ensure Date column
#         sp500 = sp500.reset_index()
#         sp500['Date'] = pd.to_datetime(sp500['Date'], errors='coerce')
#         sp500 = sp500.dropna(subset=['Date'])
        
#         # Merge with existing data if available
#         if os.path.exists(output_file) and len(existing_data) > 0:
#             sp500 = pd.concat([existing_data, sp500], ignore_index=True)
#             sp500 = sp500.drop_duplicates(subset=['Date']).sort_values('Date')
        
#         # Final validation
#         sp500 = sp500.dropna(subset=['Date', 'Close'])
        
#         # Save and verify
#         sp500.to_csv(output_file, index=False)
#         print(f"\nSaved {len(sp500):,} rows to {output_file}")
#         print(f"Date range: {sp500['Date'].min().date()} to {sp500['Date'].max().date()}")
        
#         return sp500
        
#     except Exception as e:
#         print(f"\nError fetching S&P 500 data:")
#         print(f"Error type: {type(e).__name__}")
#         print(f"Error message: {str(e)}")
#         raise

# if __name__ == "__main__":
#     print("Starting data download...")
    # fetch_sp500_data()