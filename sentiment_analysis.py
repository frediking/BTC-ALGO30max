import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_sentiment(file_path='data/all-data.csv'):
    """Load and preprocess sentiment data from CSV"""
    try:
        # Load data with correct encoding and proper column names
        sentiment_df = pd.read_csv(
            file_path, 
            encoding='iso-8859-1',
            names=['sentiment', 'text'],  # Add column names
            quoting=1  # Handle quoted text properly
        )
        print("Loaded sentiment data successfully")
        
        # Convert sentiment to numeric values
        sentiment_map = {
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }
        
        # Apply sentiment mapping
        sentiment_df['sentiment_score'] = sentiment_df['sentiment'].map(sentiment_map)
        
        # Use actual date range from price data
        start_date = pd.Timestamp('2020-01-01')  # Adjust based on your price data
        end_date = pd.Timestamp('2023-12-31')    # Adjust based on your price data
        
        sentiment_df['date'] = pd.date_range(
            start=start_date,
            # end=start_date + pd.Timedelta(days=len(sentiment_df)-1),
            periods=len(sentiment_df),
            freq='D'
        )
        
        print(f"Processed {len(sentiment_df)} sentiment records")
        
        # Verify data
        print("\nSentiment distribution:")
        print(sentiment_df['sentiment'].value_counts())
        
        return sentiment_df
        
    except Exception as e:
        print(f"Error loading sentiment data: {str(e)}")
        raise

def aggregate_daily_sentiment(sentiment_df, price_df=None):
    """Aggregate sentiment scores by date"""
    try:
        # Ensure date column exists
        if 'date' not in sentiment_df.columns:
            start_date = price_df.index[0] if price_df is not None else '2023-01-01'
            sentiment_df['date'] = pd.date_range(
                start=start_date,  # Use price data start date or default
                periods=len(sentiment_df),
                freq='D'
            )
        
        # Group by date and calculate statistics
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_score': [
                'mean',
                'std',
                'count'
            ]
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = [
            'sentiment_mean',
            'sentiment_std',
            'sentiment_count'
        ]
        
        # Set date as index
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        print(f"Created daily sentiment features for {len(daily_sentiment)} days")
        return daily_sentiment
        
    except Exception as e:
        print(f"Error in aggregate_daily_sentiment: {str(e)}")
        print("Daily sentiment structure:")
        print(daily_sentiment.head())
        raise

def merge_with_price_data(price_df, sentiment_df, suffixes=('', '_sent')):
    """Merge price and sentiment data with suffix handling"""
    try:
        # Reset index to avoid duplicates
        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()
        
        # Ensure datetime index
        price_df.index = pd.to_datetime(price_df.index)
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
        
        # Merge with suffixes
        df_merged = price_df.merge(
            sentiment_df,
            left_index=True,
            right_index=True,
            how='left',
            suffixes=suffixes
        )
        
        print(f"Merged shape: {df_merged.shape}")
        return df_merged
        
    except Exception as e:
        print(f"Merge error: {str(e)}")
        print("\nPrice data columns:", price_df.columns)
        print("Sentiment data columns:", sentiment_df.columns)
        raise

def create_sentiment_features(df):
    """Create additional sentiment features"""
    df['sentiment_ma7'] = df['sentiment_mean'].rolling(7).mean()
    df['sentiment_ma30'] = df['sentiment_mean'].rolling(30).mean()
    df['sentiment_momentum'] = df['sentiment_mean'].diff()
    
    return df

def plot_sentiment_analysis(df):
    """Plot sentiment analysis results"""
    plt.figure(figsize=(15, 10))
    
    # Price and Sentiment
    plt.subplot(2,1,1)
    plt.plot(df.index, df['Close'], label='Price')
    plt.plot(df.index, df['sentiment_ma7'], label='7-day Sentiment MA')
    plt.title('Bitcoin Price vs Sentiment')
    plt.legend()
    
    # Sentiment Distribution
    plt.subplot(2,1,2)
    plt.hist(df['sentiment_mean'], bins=50)
    plt.title('Sentiment Distribution')
    
    plt.tight_layout()
    plt.show()