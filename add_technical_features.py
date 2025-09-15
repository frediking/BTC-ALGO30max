# Standard libraries
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict
import json

# Statistical and ML libraries
from scipy.stats import ks_2samp
from sklearn.model_selection import TimeSeriesSplit

# Technical Analysis libraries
from ta.trend import (
    EMAIndicator, 
    MACD, 
    CCIIndicator, 
    ADXIndicator
)
from ta.momentum import (
    RSIIndicator, 
    StochasticOscillator,
    WilliamsRIndicator
)
from ta.volatility import (
    BollingerBands, 
    AverageTrueRange
)
from ta.volume import OnBalanceVolumeIndicator

def add_technical_features(df):
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Basic price metrics
    df['daily_returns'] = df['Close'].pct_change()
    df['percentreturn'] = df['daily_returns'] * 100
    df['price_range'] = df['High'] - df['Low']
    df['price_change'] = df['Close'] - df['Open']
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Volatility
    df['volatility'] = df['price_range'] / df['Open']
    
    # RSI
    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi.rsi()

    # Commodity Channel Index (CCI)
    cci = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20)
    df['CCI'] = cci.cci()

    # Williams %R
    williams = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14)
    df['Williams_%R'] = williams.williams_r()

    # Average Directional Index (ADX)
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_pos'] = adx.adx_pos()
    df['ADX_neg'] = adx.adx_neg()   
    
    # EMAs
    ema12 = EMAIndicator(close=df['Close'], window=12)
    ema26 = EMAIndicator(close=df['Close'], window=26)
    df['EMA12'] = ema12.ema_indicator()
    df['EMA26'] = ema26.ema_indicator()
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['Close'], window=21).ema_indicator()
    df['EMA50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()    
    df['EMA12_over_EMA26'] = df['EMA12'] / df['EMA26']
    df['EMA9_over_EMA21'] = df['EMA9'] / df['EMA21']
    df['EMA12_over_EMA50'] = df['EMA12'] / df['EMA50']
    df['EMA9_over_EMA50'] = df['EMA9'] / df['EMA50']
    df['EMA21_over_EMA50'] = df['EMA21'] / df['EMA50']
    df['EMA12_over_EMA21'] = df['EMA12'] / df['EMA21']
    df['EMA9_over_EMA26'] = df['EMA9'] / df['EMA26']
    df['EMA21_over_EMA26'] = df['EMA21'] / df['EMA26']

    df['PPO'] = ((ema12.ema_indicator() - ema26.ema_indicator()) / ema26.ema_indicator()) * 100
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['Signal_Line'] = macd.macd_signal()
    df['MACD_histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_std'] = bb.bollinger_mavg()
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # ATR
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
    df['ATR'] = atr.average_true_range()
    
    # OBV
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    
    # VWAP
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # Rolling statistics
    for window in [7, 30]:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_vol_{window}'] = df['daily_returns'].rolling(window=window).std()
        df[f'rolling_return_{window}'] = df['daily_returns'].rolling(window=window).mean()
        df[f'return_ma_{window}'] = df['daily_returns'].rolling(window=window).mean()
        df[f'return_std_{window}'] = df['daily_returns'].rolling(window=window).std()
    
    # Rolling max/min
    df['rolling_max_7'] = df['Close'].rolling(window=7).max()
    df['rolling_min_7'] = df['Close'].rolling(window=7).min()
    df['rolling_vol_7'] = df['Close'].pct_change().rolling(7).std()
    df['rolling_return_7'] = df['Close'].pct_change().rolling(7).mean()
    
    # Lagged prices
    for lag in [1, 3, 7, 30]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    
    # Trend strength
    df['trend_strength'] = (df['Close'] - df['MA50']) / df['MA50']
    
    # Volume volatility ratio
    df['volatility_volume_ratio'] = df['volatility'] / (df['Volume'] / df['Volume'].rolling(window=30).mean())
    
    # Next day predictions
    df['next_day_price'] = df['Close'].shift(-1)
    df['next_day_return'] = df['daily_returns'].shift(-1)
    
    # Return statistics
    df['pos_returns'] = df['daily_returns'].rolling(window=30).apply(lambda x: (x > 0).sum())
    df['neg_returns'] = df['daily_returns'].rolling(window=30).apply(lambda x: (x < 0).sum())
    
    # Time features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['month'] = df['Date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    rcma1 = df['Close'].diff(10).rolling(10).mean()
    rcma2 = df['Close'].diff(15).rolling(10).mean()
    rcma3 = df['Close'].diff(20).rolling(10).mean()
    rcma4 = df['Close'].diff(30).rolling(15).mean()
    df['KST'] = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)

    # Volatility windows
    for window in [7, 14, 21, 30]:
        df[f'Volatility_{window}'] = df['Close'].pct_change().rolling(window).std() * np.sqrt(252)
    
    # Keltner Channel Width
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=20)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    ema = typical_price.ewm(span=20).mean()
    df['KC_Width'] = ((ema + (2 * atr.average_true_range())) - 
                           (ema - (2 * atr.average_true_range()))) / ema * 100
    
    # Volume-price indicators
    # CMF (Chaikin Money Flow)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfm * df['Volume']
    df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # Force Index
    df['Force_Index'] = df['Close'].diff(1) * df['Volume']
    
    # MFI (Money Flow Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (positive_flow / negative_flow)))
    
    # VWAP
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Momentum windows
    for window in [10, 20, 30, 60]:
        df[f'Momentum_{window}'] = df['Close'].diff(window)
    
    # DPO (Detrended Price Oscillator)
    df['DPO'] = df['Close'] - df['Close'].rolling(20).mean().shift(11)
    
    # Aroon Oscillator
    def aroon_up(high, window=25):
        return 100 * high.rolling(window + 1).apply(lambda x: x.argmax()) / window
    
    def aroon_down(low, window=25):
        return 100 * low.rolling(window + 1).apply(lambda x: x.argmin()) / window
    
    df['Aroon_Osc'] = aroon_up(df['High']) - aroon_down(df['Low'])
    
    # Pattern recognition
    # Higher Highs to Lower Lows ratio
    highs = df['High'].rolling(20).max()
    lows = df['Low'].rolling(20).min()
    df['HH_LL_Ratio'] = (df['High'] > highs.shift(1)).rolling(20).sum() / \
                             ((df['Low'] < lows.shift(1)).rolling(20).sum() + 1)
    
    # Price Rate of Change
    df['Price_ROC'] = df['Close'].pct_change(12) * 100
    
    # RSI Divergence
    rsi = df['RSI']
    price_momentum = df['Close'].diff()
    df['RSI_Divergence'] = (rsi - rsi.shift(1)) - (price_momentum - price_momentum.shift(1))
    
    # Volatility-volume ratio
    df['volatility_volume_ratio'] = (df['Volume'] / df['Volume'].rolling(20).mean()) / \
                                        (df['Volatility_21'] / df['Volatility_21'].rolling(20).mean())

    
    
    return df

def assess_feature_stability(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Dict]:
    """
    Assess feature stability across time-based cross-validation splits.
    
    Args:
        df: DataFrame containing features
        n_splits: Number of time-based CV splits
    
    Returns:
        Dictionary with stability scores for each feature
    """
    # Exclude non-numeric and special columns
    exclude_columns = ['Date', 'next_day_price', 'next_day_return', 'day_of_week', 
                      'week_of_year', 'month', 'is_weekend']
    features = [col for col in df.columns if col not in exclude_columns]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    stability_scores = defaultdict(list)
    
    for train_idx, test_idx in tscv.split(df):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        for feature in features:
            if df[feature].dtype in ['float64', 'int64']:
                # Normalize feature values to [-1, 1] range for fair comparison
                train_norm = (train_data[feature] - train_data[feature].mean()) / train_data[feature].std()
                test_norm = (test_data[feature] - test_data[feature].mean()) / test_data[feature].std()
                
                psi = calculate_psi(train_norm, test_norm)
                ks_stat, _ = ks_2samp(train_norm.dropna(), test_norm.dropna())
                
                # Calculate distribution shifts
                mean_shift = abs(train_data[feature].mean() - test_data[feature].mean())
                std_shift = abs(train_data[feature].std() - test_data[feature].std())
                
                # Calculate stability score
                stability = 1 - (0.3 * psi + 0.2 * ks_stat + 
                                0.25 * mean_shift + 0.25 * std_shift)
                
                stability_scores[feature].append(stability)
    
    # Calculate final stability metrics
    final_scores = {
        feature: {
            'mean_stability': np.mean(scores),
            'std_stability': np.std(scores),
            'min_stability': np.min(scores),
            'max_stability': np.max(scores)
        }
        for feature, scores in stability_scores.items()
    }
    
    return final_scores

def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index between two distributions.
    
    Args:
        expected: Reference distribution
        actual: Comparison distribution
        bins: Number of bins for distribution comparison
    
    Returns:
        PSI value
    """
    try:
        # Create bins based on expected distribution
        bins_range = pd.qcut(expected, q=bins, duplicates='drop', labels=False)
        expected_bins = pd.value_counts(bins_range, normalize=True)
        
        # Map actual values to same bins
        actual_bins = pd.value_counts(
            pd.cut(actual, bins=expected_bins.index.max() + 1, labels=False),
            normalize=True
        )
        
        # Calculate PSI
        psi = sum((actual_bins - expected_bins) * 
                  np.log(actual_bins / expected_bins))
        
        return psi
    except Exception as e:
        # Return a high PSI value if calculation fails
        return 1.0

def visualize_feature_stability(stability_scores: Dict[str, Dict], save_path: str = None):
    """
    Visualize feature stability analysis results.
    
    Args:
        stability_scores: Dictionary containing stability scores
        save_path: Optional path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Prepare data for plotting
    features = list(stability_scores.keys())
    mean_scores = [scores['mean_stability'] for scores in stability_scores.values()]
    std_scores = [scores['std_stability'] for scores in stability_scores.values()]
    
    # Sort by mean stability
    sorted_idx = np.argsort(mean_scores)
    features = [features[i] for i in sorted_idx]
    mean_scores = [mean_scores[i] for i in sorted_idx]
    std_scores = [std_scores[i] for i in sorted_idx]
    
    # Plot mean stability with error bars
    ax1.barh(range(len(features)), mean_scores, xerr=std_scores, 
             alpha=0.6, capsize=5)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features, fontsize=8)
    ax1.set_xlabel('Stability Score')
    ax1.set_title('Feature Stability Analysis')
    ax1.axvline(x=0.7, color='r', linestyle='--', label='Stability Threshold')
    ax1.legend()
    
    # Plot stability distribution
    stability_values = [score for scores in stability_scores.values() 
                       for score in [scores['min_stability'], scores['mean_stability'], 
                                   scores['max_stability']]]
    ax2.hist(stability_values, bins=30, alpha=0.6)
    ax2.set_xlabel('Stability Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Stability Scores')
    ax2.axvline(x=0.7, color='r', linestyle='--', label='Stability Threshold')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def get_feature_groups():
    """
    Organize stable and important features into logical groups for modeling.
    """
    feature_groups = {
        'return_based': [
            'Daily Returns', '%Return', 'daily_returns', 'percentreturn',
            'Profit/Loss_binary', 'Profit/Loss_target_mean'
        ],
        'oscillators': [
            'RSI', 'CCI', 'Williams_%R', 'stoch_k', 'RSI_Divergence'
        ],
        'volatility': [
            'BB_width', 'Volatility_7', 'Volatility_14', 'Volatility_21', 
            'Volatility_30', 'rolling_vol_7', 'rolling_vol_30', 
            'volatility_volume_ratio'
        ],
        'trend_ratios': [
            'EMA12_over_EMA26', 'EMA9_over_EMA21', 'EMA12_over_EMA50',
            'EMA9_over_EMA50', 'EMA21_over_EMA50', 'trend_strength'
        ]
    }
    return feature_groups

def main():
    # Load the extended dataset
    df = pd.read_csv('Bitcoin_Historical_Data_Enhanced.csv')
    
    # Add technical features
    df_with_features = add_technical_features(df)
    
    # Assess feature stability
    print("Assessing feature stability...")
    stability_scores = assess_feature_stability(df_with_features)
    
    # Visualize stability
    print("Generating stability visualization...")
    visualize_feature_stability(stability_scores, 'feature_stability.png')
    
    # Calculate feature importance based on correlation with returns
    correlations = df_with_features.corr()['percentreturn'].abs()
    
    # Weight stability scores by correlation importance
    weighted_stable_features = [
        feature for feature, scores in stability_scores.items()
        if scores['mean_stability'] >= 0.5 or correlations.get(feature, 0) > 0.3
    ]
    
    print(f"\nStable and important features ({len(weighted_stable_features)}):", weighted_stable_features)
    
    # After calculating weighted_stable_features
    feature_groups = get_feature_groups()
    
    # Save feature groups for later use
    with open('feature_groups.json', 'w') as f:
        json.dump(feature_groups, f)
    
    print("\nFeature Groups:")
    for group, features in feature_groups.items():
        print(f"\n{group.upper()}:")
        print(", ".join(features))
    
    # Save the enhanced dataset
    df_with_features.to_csv(('encoded_bitcoin_data.csv'), index=False)
    print("\nSuccessfully added technical features to the dataset")
    print(f"Total features: {len(df_with_features.columns)}")
    print("Features added:", list(df_with_features.columns))

if __name__ == "__main__":
    main() 