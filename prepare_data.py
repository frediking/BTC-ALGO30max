import pandas as pd
from typing import Tuple
import logging
import numpy as np
from deep_learning_btc import EnhancedPreprocessor, get_numeric_X_y
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def calculate_market_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """Calculate market volatility using rolling standard deviation of returns"""
    if 'Close' in df.columns:
        returns = df['Close'].pct_change()
        current_volatility = returns.rolling(window=window).std().iloc[-1]
        return current_volatility
    return 0.0

def adaptive_anomaly_detection(df: pd.DataFrame, col: str, market_volatility: float, 
                             threshold: float = 0.02) -> dict:
    """
    Adjust anomaly detection based on market conditions.
    
    Args:
        df: Input DataFrame
        col: Column to analyze
        market_volatility: Current market volatility
        threshold: Volatility threshold for switching methods
    """
    if market_volatility > threshold:
        # Use enhanced version for high volatility
        q1, q3 = df[col].quantile([0.15, 0.85])
        multiplier = 2.0
        logging.info(f"High volatility detected ({market_volatility:.4f}). Using enhanced bounds for {col}")
    else:
        # Use traditional for normal conditions
        q1, q3 = df[col].quantile([0.25, 0.75])
        multiplier = 1.5
        logging.info(f"Normal volatility ({market_volatility:.4f}). Using standard bounds for {col}")
    
    iqr = q3 - q1
    bounds = {
        'lower': q1 - multiplier * iqr,
        'upper': q3 + multiplier * iqr
    }
    return bounds

def load_and_preprocess_data(file_path: str, input_features: list, target_column: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load and preprocess Bitcoin data using EnhancedPreprocessor.
    
    Args:
        file_path (str): Path to the input CSV.
        input_features (list): List of feature columns to retain.
        target_column (str): Name of the target column.
    
    Returns:
        Tuple[pd.DataFrame, dict]: Preprocessed DataFrame and preprocessing stats.
    """
    logging.info("Loading and preprocessing data...")

    # Load raw data
    df = pd.read_csv(file_path)
    
    # 7. Drop unnecessary columns
    df = df.drop(columns=['Daily Returns','%Return'])
    logging.info("columns dropped")

    # Convert Date to datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        logging.warning("Date column not found. Proceeding without setting index.")

    # Log all columns
    logging.info(f"Initial columns in DataFrame: {list(df.columns)}")
    logging.info(f"Initial shape: {df.shape}")

    # Verify target column
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")
        raise KeyError(f"Target column '{target_column}' not found in DataFrame")

    # Initialize EnhancedPreprocessor with valid parameters
    preprocessor = EnhancedPreprocessor(
        quality_threshold=70,
        smoothing=10,
        decay_factor=0.95,
        n_splits=5,
        stability_window=20,
        drift_threshold=0.1,
        correlation_threshold=0.85,
        min_iv_threshold=0.02
    )

    # Manual anomaly handling with adaptive thresholds
    anomaly_counts = {'price_anomalies_handled': 0, 'volume_anomalies_handled': 0}
    price_cols = ['Close', 'High', 'Low', 'Open', 'Adj Close']
    volume_col = 'Volume'

    df_clean = df.copy()
    
    # Calculate market volatility
    market_volatility = calculate_market_volatility(df_clean)
    logging.info(f"Current market volatility: {market_volatility:.4f}")

    # Handle price columns with adaptive thresholds
    for col in price_cols:
        if col in df_clean.columns:
            bounds = adaptive_anomaly_detection(df_clean, col, market_volatility)
            outliers = (df_clean[col] < bounds['lower']) | (df_clean[col] > bounds['upper'])
            anomaly_counts['price_anomalies_handled'] += outliers.sum()
            df_clean[col] = df_clean[col].clip(bounds['lower'], bounds['upper'])
            logging.info(f"Handled {outliers.sum()} anomalies in {col}")

    # Handle volume with traditional method
    if volume_col in df_clean.columns:
        bounds = adaptive_anomaly_detection(df_clean, volume_col, market_volatility, threshold=0.03)
        outliers = (df_clean[volume_col] < bounds['lower']) | (df_clean[volume_col] > bounds['upper'])
        anomaly_counts['volume_anomalies_handled'] += outliers.sum()
        df_clean[volume_col] = df_clean[volume_col].clip(bounds['lower'], bounds['upper'])
        logging.info(f"Handled {outliers.sum()} anomalies in {volume_col}")

    logging.info(f"Manual anomalies handled: {anomaly_counts['price_anomalies_handled']} price, "
                 f"{anomaly_counts['volume_anomalies_handled']} volume")

    # Preprocess data with EnhancedPreprocessor
    initial_rows = len(df_clean)
    missing_before = df_clean.isna().sum().sum()
    df_processed, preprocessing_info = preprocessor.preprocess_data(df_clean)
    missing_after = df_processed.isna().sum().sum()

    # Update preprocessing stats
    preprocessing_info['preprocessing_stats']['gap_handling'] = {
        'gaps_filled': missing_before - missing_after
    }
    preprocessing_info['preprocessing_stats']['anomaly_handling'] = anomaly_counts
    logging.info(f"Missing values filled: {missing_before - missing_after}")

    # Handle duplicates
    df_processed = df_processed.drop_duplicates()
    logging.info(f"Removed {initial_rows - len(df_processed)} duplicate rows")

    # Remove duplicate columns
    if df_processed.columns.duplicated().any():
        logging.warning("Duplicate column names found. Keeping first occurrence.")
        df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]

    # Fallback imputation for any remaining NaN values
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    if df_processed[numeric_cols].isna().sum().sum() > 0:
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        logging.info("Applied mean imputation for remaining NaN values in numeric columns")
    
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    if categorical_cols.any():
        df_processed[categorical_cols] = df_processed[categorical_cols].fillna(method='ffill').fillna(method='bfill')
        logging.info("Applied ffill/bfill for remaining NaN values in categorical columns")

    # Verify target and features after preprocessing
    if target_column not in df_processed.columns:
        logging.error(f"Target column '{target_column}' dropped during preprocessing.")
        raise KeyError(f"Target column '{target_column}' not found after preprocessing")

    available_features = [f for f in input_features if f in df_processed.columns]
    missing_features = [f for f in input_features if f not in df_processed.columns]
    if missing_features:
        logging.warning(f"Missing features after preprocessing: {missing_features}")
    else:
        logging.info("All specified features retained after preprocessing.")

    # Update preprocessing stats
    preprocessing_info['preprocessing_stats']['feature_selection']['selected_features'] = available_features

    # Calculate data quality score
    missing_ratio = missing_after / (df_processed.shape[0] * df_processed.shape[1])
    anomaly_ratio = sum(anomaly_counts.values()) / df_processed.shape[0]
    quality_score = max(0, 100 * (1 - missing_ratio - anomaly_ratio))
    preprocessing_info['quality_report']['overall_score'] = round(quality_score, 2)
    logging.info(f"Data Quality Score: {quality_score}/100")

    logging.info(f"Preprocessing complete. Final shape: {df_processed.shape}")
    logging.info(f"Final columns: {list(df_processed.columns)}")

    return df_processed, preprocessing_info

def main():
    ENCODED_CSV_PATH = 'encoded_bitcoin_data.csv'
    TARGET_COLUMN = 'percentreturn'
    # Define input features (all relevant technical indicators and features from CSV)
    input_features = [
    'stoch_k',
    'stoch_d',
    # 'return_ma_7',
    # 'rolling_return_7',
    'rolling_vol_7',
    'Volume',
    # 'return_std_7',
    'Price_ROC',
    'Momentum_10',
    # 'BB_width',
    # 'KC_Width',
    'ATR',
    'CCI',
    # 'ADX_neg',
    'price_change',
    'volatility',
    # 'Williams_%R',
    'RSI',
    'RSI_Divergence',
    # 'MACD_histogram',
    'Close',
    'Force_Index',
    'close_lag_1'
    

        # 'Close', 'High', 'Low', 'Open', 'Volume', 'price_range', 'price_change',
        # 'MA50', 'MA200', 'volatility', 'RSI', 'EMA12', 'EMA26', 'MACD', 'Signal_Line',
        # 'MACD_histogram', 'BB_middle', 'BB_std', 'BB_upper', 'BB_lower', 'BB_width',
        # 'stoch_k', 'stoch_d', 'ATR', 'OBV', 'vwap', 'rolling_mean_7', 'rolling_vol_7',
        # 'rolling_return_7', 'return_ma_7', 'return_std_7', 'rolling_mean_30',
        # 'rolling_vol_30', 'rolling_return_30', 'return_ma_30', 'return_std_30',
        # 'rolling_max_7', 'rolling_min_7', 'close_lag_1', 'close_lag_3', 'close_lag_7',
        # 'close_lag_30', 'trend_strength', 'volatility_volume_ratio', 'next_day_price',
        # 'next_day_return', 'pos_returns', 'neg_returns', 'day_of_week', 'week_of_year',
        # 'month', 'is_weekend', 'CCI', 'Williams_%R', 'ADX', 'ADX_pos', 'ADX_neg',
        # 'EMA9', 'EMA21', 'EMA50', 'EMA12_over_EMA26', 'EMA9_over_EMA21',
        # 'EMA12_over_EMA50', 'EMA9_over_EMA50', 'EMA21_over_EMA50', 'EMA12_over_EMA21',
        # 'EMA9_over_EMA26', 'EMA21_over_EMA26', 'PPO', 'KST', 'Volatility_7',
        # 'Volatility_14', 'Volatility_21', 'Volatility_30', 'KC_Width', 'CMF',
        # 'Force_Index', 'MFI', 'VWAP', 'Momentum_10', 'Momentum_20', 'Momentum_30',
        # 'Momentum_60', 'DPO', 'Aroon_Osc', 'HH_LL_Ratio', 'Price_ROC', 'RSI_Divergence'
    ]

    # Remove target-like columns from input features
    target_like = [ 'daily_returns', 'percentreturn',
                   'Profit/Loss_binary', 'Profit/Loss_target_mean']
    input_features = [f for f in input_features if f not in target_like]

    print("Loading and preprocessing data...")
    df_processed, preprocessing_stats = load_and_preprocess_data(ENCODED_CSV_PATH, input_features, TARGET_COLUMN)

    # Get X and y
    available_features = [f for f in input_features if f in df_processed.columns]
    if not available_features:
        logging.error("No input features available after preprocessing")
        raise ValueError("No input features found in DataFrame")

    X, y = get_numeric_X_y(df_processed, available_features, TARGET_COLUMN)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")

    # Save outputs
    X.to_csv('Xdate_prepared.csv', index=True)
    # y.to_csv('ydate_prepared.csv', index=True)
    # y.to_csv('y1_prepared.csv', index=False)
    print("Saved All .csv Files.")

if __name__ == "__main__":
    main()