import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt

# Read the Bitcoin historical data
# Read the CSV file
df = pd.read_csv('Bitcoin_Historical_Data.csv')
    
# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
    
# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())
    
# Display basic statistics
print("\nBasic statistics of the dataset:")
print(df.describe())

# Display the number of missing values in each column

# 1. Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 2. Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# 3. Check and remove duplicates
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate entries: {duplicate_count}")
df = df.drop_duplicates()

# 4. Clean column names - convert to lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace('%', 'percent').str.replace(' ', '_')

# 5. Convert price columns to numeric, removing any currency symbols and commas
price_columns = ['open', 'high', 'low', 'close', 'volume']
for col in price_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

# Display the cleaned dataset info
print("\nCleaned Dataset Info:")
print(df.info())
print("\nFirst 5 rows of cleaned data:")
print(df.head())



# Data Analysis and Visualization

# Set the date as index
df.set_index('date', inplace=True)

# 1. Calculate daily returns
df['daily_returns'] = df['close'].pct_change()

# 2. Create price trend visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'])
plt.title('Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Volume analysis
plt.figure(figsize=(12, 6))
plt.bar(df.index, df['volume'])
plt.title('Bitcoin Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Basic statistics of daily returns
print("\nDaily Returns Statistics:")
print(df['daily_returns'].describe())

# 5. Rolling average
df['MA50'] = df['close'].rolling(window=50).mean()
df['MA200'] = df['close'].rolling(window=200).mean()

# Plot price with moving averages
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='Price')
plt.plot(df.index, df['MA50'], label='50-day MA')
plt.plot(df.index, df['MA200'], label='200-day MA')
plt.title('Bitcoin Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Additional Analysis
# 1. Set date as index
df.set_index('date', inplace=True)

# 2. Calculate additional metrics
df['daily_returns'] = df['close'].pct_change()
df['volatility'] = df['daily_returns'].rolling(window=30).std() * np.sqrt(252)  # Annualized volatility
df['price_range'] = df['high'] - df['low']
df['price_change'] = df['close'] - df['open']

# 3. Correlation analysis
correlation_matrix = df[['close', 'volume', 'daily_returns', 'volatility']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 4. Advanced Visualizations
plt.style.use('seaborn')

# 4.1 Price and Volume subplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(df.index, df['close'], label='Close Price')
ax1.set_title('Bitcoin Price and Volume Analysis')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True)

ax2.bar(df.index, df['volume'], color='gray', alpha=0.5, label='Volume')
ax2.set_ylabel('Volume')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4.2 Volatility Analysis
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['volatility'], color='red', alpha=0.7)
plt.title('30-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4.3 Daily Returns Distribution
plt.figure(figsize=(10, 6))
df['daily_returns'].hist(bins=50)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Summary Statistics
print("\nSummary Statistics:")
print("\nVolatility Statistics:")
print(df['volatility'].describe())
print("\nDaily Returns Statistics:")
print(df['daily_returns'].describe())

# 6. Risk Metrics
var_95 = np.percentile(df['daily_returns'].dropna(), 5)
print(f"\nValue at Risk (95% confidence): {var_95:.4f}")
max_drawdown = (df['close'] / df['close'].expanding(min_periods=1).max() - 1).min()
print(f"Maximum Drawdown: {max_drawdown:.4f}")


# 7. Trading Signals Analysis

# Calculate technical indicators
# RSI (Relative Strength Index)
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['close'])

# MACD (Moving Average Convergence Divergence)
df['EMA12'] = df['close'].ewm(span=12).mean()
df['EMA26'] = df['close'].ewm(span=26).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

df.head()


# Visualize Technical Indicators
plt.figure(figsize=(12, 8))

# Plot RSI
plt.subplot(2, 1, 1)
plt.plot(df.index, df['RSI'], label='RSI', color='purple')
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
plt.title('Relative Strength Index (RSI)')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)

# Plot MACD
plt.subplot(2, 1, 2)
plt.plot(df.index, df['MACD'], label='MACD', color='blue')
plt.plot(df.index, df['Signal_Line'], label='Signal Line', color='orange')
plt.title('Moving Average Convergence Divergence (MACD)')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print Trading Signals Summary
print("\nTrading Signals Analysis:")
print("\nCurrent RSI:", df['RSI'].iloc[-1].round(2))
print("Current MACD:", df['MACD'].iloc[-1].round(2))
print("Current Signal Line:", df['Signal_Line'].iloc[-1].round(2))

# Generate simple trading signals
df['RSI_Signal'] = np.where(df['RSI'] > 70, 'Overbought', 
                           np.where(df['RSI'] < 30, 'Oversold', 'Neutral'))
df['MACD_Signal'] = np.where(df['MACD'] > df['Signal_Line'], 'Buy', 'Sell')

# Display last 5 days of signals
print("\nLast 5 days of trading signals:")
print(df[['RSI_Signal', 'MACD_Signal']].tail())

print("\nlast 4 years of trading signals:")
print(df[['RSI_Signal', 'MACD_Signal']].tail(1464))