import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

def calculate_sp500_returns():
    """Calculate daily and cumulative returns for S&P500"""
    
    # Read the CSV file
    df = pd.read_csv('spx15-25.csv')
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date in ascending order
    df = df.sort_values('Date')
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close/Last'].pct_change() * 100
    
    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Daily_Return']/100).cumprod() - 1
    
    # Calculate summary statistics
    stats = {
        'Mean Daily Return': df['Daily_Return'].mean(),
        'Daily Return Std': df['Daily_Return'].std(),
        'Annual Volatility': df['Daily_Return'].std() * np.sqrt(252),
        'Total Return': df['Cumulative_Return'].iloc[-1] * 100,
        'Sharpe Ratio': (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252)
    }
    
    # Plot returns
    plt.figure(figsize=(15, 10))
    
    # Plot daily returns
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['Daily_Return'])
    plt.title('S&P 500 Daily Returns')
    plt.ylabel('Return (%)')
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    plt.plot(df['Date'], df['Cumulative_Return'] * 100)
    plt.title('S&P 500 Cumulative Returns')
    plt.ylabel('Cumulative Return (%)')
    
    plt.tight_layout()
    plt.savefig('sp500_returns.png')
    
    return df, stats

def calculate_sp500_features():
    """Calculate and add S&P500 features to existing dataset"""
    
    # Read the CSV file
    df = pd.read_csv('spx15-25.csv')
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date in ascending order
    df = df.sort_values('Date')
    
    # Calculate return features
    df['SP500_Daily_Return'] = df['Close/Last'].pct_change()
    df['SP500_Return_Lag1'] = df['SP500_Daily_Return'].shift(1)
    df['SP500_Return_Lag3'] = df['SP500_Daily_Return'].shift(3)
    df['SP500_Daily_Return_Pct'] = df['SP500_Daily_Return'] * 100
    df['SP500_Cumulative_Return'] = (1 + df['SP500_Daily_Return']).cumprod() - 1
    
    # Add volatility features
    df['SP500_Rolling_Vol_5d'] = df['SP500_Daily_Return'].rolling(window=5).std() * np.sqrt(252)
    df['SP500_Rolling_Vol_20d'] = df['SP500_Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Add momentum features
    df['SP500_Return_5d'] = df['Close/Last'].pct_change(periods=5)
    df['SP500_Return_20d'] = df['Close/Last'].pct_change(periods=20)
    
    # Add moving averages
    df['SP500_MA_5d'] = df['Close/Last'].rolling(window=5).mean()
    df['SP500_MA_20d'] = df['Close/Last'].rolling(window=20).mean()
    
    # Calculate summary statistics
    stats = {
        'Mean_Daily_Return': df['SP500_Daily_Return'].mean(),
        'Daily_Return_Std': df['SP500_Daily_Return'].std(),
        'Annual_Volatility': df['SP500_Daily_Return'].std() * np.sqrt(252),
        'Total_Return': df['SP500_Cumulative_Return'].iloc[-1],
        'Sharpe_Ratio': (df['SP500_Daily_Return'].mean() / df['SP500_Daily_Return'].std()) * np.sqrt(252)
    }
    
    # Plot key metrics
    plt.figure(figsize=(15, 12))
    
    # Daily returns
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['SP500_Daily_Return_Pct'])
    plt.title('S&P 500 Daily Returns')
    plt.ylabel('Return (%)')
    
    # Cumulative returns
    plt.subplot(3, 1, 2)
    plt.plot(df['Date'], df['SP500_Cumulative_Return'] * 100)
    plt.title('S&P 500 Cumulative Returns')
    plt.ylabel('Cumulative Return (%)')
    
    # Volatility
    plt.subplot(3, 1, 3)
    plt.plot(df['Date'], df['SP500_Rolling_Vol_20d'], label='20-day Vol')
    plt.plot(df['Date'], df['SP500_Rolling_Vol_5d'], label='5-day Vol')
    plt.title('S&P 500 Rolling Volatility')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sp500_analysis.png')
    
    return df, stats

if __name__ == "__main__":
    df, stats = calculate_sp500_features()
    
    print("\nS&P 500 Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # Save enhanced dataset
    df.to_csv('SPXenhanced.csv', index=False)


def prepare_features():
    """Prepare features with KNN imputation for missing values"""
    # Read the enhanced dataset
    df = pd.read_csv('SPXenhanced.csv')
    
    # Select relevant features
    X_features = df[[
        'Date',
        'SP500_Daily_Return',
        'SP500_Daily_Return_Pct',
        'SP500_Cumulative_Return',
        'SP500_Rolling_Vol_5d',
        'SP500_Rolling_Vol_20d',
        'SP500_Return_5d',
        'SP500_Return_20d',
        'SP500_MA_5d',
        'SP500_MA_20d'
    ]]
    
    # Set Date as index
    X_features['Date'] = pd.to_datetime(X_features['Date'])
    X_features.set_index('Date', inplace=True)
    
    # Initialize KNN imputer
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    
    # Perform KNN imputation
    features_imputed = imputer.fit_transform(X_features)
    
    # Convert back to DataFrame with original index and column names
    X_features_clean = pd.DataFrame(
        features_imputed, 
        index=X_features.index,
        columns=X_features.columns
    )
    
    # Save to CSV
    X_features_clean.to_csv('spX_prepared.csv')
    
    print(f"Shape before imputation: {X_features.shape}")
    print(f"Shape after imputation: {X_features_clean.shape}")
    print(f"\nMissing values before imputation: {X_features.isna().sum().sum()}")
    print(f"Missing values after imputation: {X_features_clean.isna().sum().sum()}")
    
    return X_features_clean

if __name__ == "__main__":
    X_features_clean = prepare_features()
    print("Features shape:", X_features_clean.shape)
    print("\nFirst few rows:")
    print(X_features_clean.head())