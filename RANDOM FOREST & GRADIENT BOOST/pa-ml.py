import numpy as np
import pandas as pd

# Helper Functions: Data Cleaning, Preprocessing, and Validation
def validate_data(df, stage=""):
    """Validate data at different stages of processing"""
    print(f"\nValidation at {stage}:")
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Contains infinity: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Contains NaN: {df.isna().sum().sum()}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        max_val = df[col].abs().max()
        if max_val > 1e6:
            print(f"Warning: Large values in {col}: {max_val}")

def clean_and_prepare_data(df):
    """Clean and prepare data for modeling by removing non-numeric (if needed) and handling missing values."""
    df_clean = df.copy()
    # Keep only numeric columns—adjust if you need to keep non-numeric ones too.
    numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
    df_clean = df_clean[numeric_columns]
    # Forward-fill missing values
    df_clean = df_clean.fillna(method='ffill')
    return df_clean

def preprocess_data(df):
    """Handle infinite values, extreme outliers, scale numbers, and ensure float32 compatibility."""
    df_processed = df.copy()
    
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_columns] = df_processed[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
    for column in numeric_columns:
        q1 = df_processed[column].quantile(0.25)
        q3 = df_processed[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df_processed[column] = df_processed[column].clip(lower=lower_bound, upper=upper_bound)
        
        max_val = df_processed[column].abs().max()
        if max_val > 1e6:
            scale_factor = 1e6 / max_val
            df_processed[column] = df_processed[column] * scale_factor
        
        df_processed[column] = df_processed[column].astype('float32')
    
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    return df_processed


# TECHNICAL INDICATORS
def calculate_atr(df, window):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

def calculate_rsi(prices, periods=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12):
    slow_ema = prices.ewm(span=slow).mean()
    fast_ema = prices.ewm(span=fast).mean()
    return fast_ema - slow_ema

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band


 
# 2.BUILD ADVANCE FEATURES
def build_advanced_features(df):
    """Create comprehensive technical features from the cleaned data."""
    # Assume df is already cleaned and preprocessed
    df_features = df.copy()
    
    try:
        # Basic price features
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log1p(df_features['returns'].fillna(0))
        
        # Trend Indicators: Moving Averages and Return over different windows
        for window in [7, 14, 21, 50, 200]:
            df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
            df_features[f'ema_{window}'] = df_features['close'].ewm(span=window).mean()
            df_features[f'return_{window}d'] = df_features['close'].pct_change(window)
        
        # Volatility Indicators
        for window in [7, 14, 21]:
            df_features[f'volatility_{window}d'] = df_features['returns'].rolling(window=window).std()
            df_features[f'atr_{window}'] = calculate_atr(df_features, window)
        
        # Volume Indicators (if volume column exists)
        if 'volume' in df_features.columns:
            df_features['volume_price_ratio'] = df_features['volume'] * df_features['close']
            df_features['volume_ma_ratio'] = df_features['volume'] / df_features['volume'].rolling(window=20).mean()
        
        # Technical Indicators
        df_features['rsi'] = calculate_rsi(df_features['close'])
        df_features['macd'] = calculate_macd(df_features['close'])
        df_features['bollinger_upper'], df_features['bollinger_lower'] = calculate_bollinger_bands(df_features['close'])
        
        # Final fill for any remaining NaNs
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
        print("Features created successfully")
        return df_features
        
    except Exception as e:
        print(f"Error in feature creation: {str(e)}")
        raise



# 3.TRAIN ENSEMBLE MODEL
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

def train_ensemble_model(df_features, target_days=30):
    """Train an ensemble model with advanced techniques."""
    try:
        # Verify numeric features
        numeric_features = df_features.select_dtypes(include=['float64', 'int64'])
        print(f"Number of features: {len(numeric_features.columns)}")
        
        # Prepare target variable: future returns (percentage change) over target_days
        df_features['target'] = df_features['close'].shift(-target_days).pct_change(target_days)
        
        # Select features to exclude target and raw price columns (which may be redundant)
        feature_cols = [col for col in numeric_features.columns 
                        if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
        X = df_features[feature_cols].dropna()
        y = df_features['target'].dropna()
        print(f"Training data shape: X: {X.shape}, y: {y.shape}")
        
        # Initialize ensemble models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        ensemble_model = VotingRegressor(estimators=[('rf', rf_model), ('gb', gb_model)])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            ensemble_model.fit(X_train_scaled, y_train)
            y_pred = ensemble_model.predict(X_test_scaled)
            cv_scores.append(r2_score(y_test, y_pred))
        
        for i, score in enumerate(cv_scores, 1):
            print(f"Fold {i}: R² score = {score:.4f}")
        print(f"Average R² score: {np.mean(cv_scores):.4f}")
        
        return ensemble_model, cv_scores
    
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise


# # Execution Pipeline

# # Step 1: Validate input data (df is your original DataFrame)
# validate_data(df, "input")

# # Step 2: Clean data
# df_clean = clean_and_prepare_data(df)

# # Step 3: Preprocess data (handle infinities, outliers, scaling, and imputation)
# df_processed = preprocess_data(df_clean)
# validate_data(df_processed, "after preprocessing")

# # Step 4: Build advanced features
# df_features = build_advanced_features(df_processed)

# # Optionally, display the head of the feature DataFrame
# print("Head of advanced features:")
# print(df_features.head())

# # Step 5: Train the ensemble model and evaluate using TimeSeriesSplit
# ensemble_model, cv_scores = train_ensemble_model(df_features)

# # (Optional) Plot the cross-validation R² scores
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o')
# plt.title('R² Scores Across Folds')
# plt.xlabel('Fold')
# plt.ylabel('R² Score')
# plt.grid(True)
# plt.show()



































# def preprocess_data(df):
#     """Handle infinite values, large numbers, and ensure float32 compatibility"""
#     df_processed = df.copy()
    
#     # First, identify numeric columns
#     numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    
#     # Handle infinities and NaN values
#     df_processed[numeric_columns] = df_processed[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
#     for column in numeric_columns:
#         # Calculate robust statistics using percentiles
#         q1 = df_processed[column].quantile(0.25)
#         q3 = df_processed[column].quantile(0.75)
#         iqr = q3 - q1
#         lower_bound = q1 - 1.5 * iqr
#         upper_bound = q3 + 1.5 * iqr
        
#         # Clip outliers
#         df_processed[column] = df_processed[column].clip(lower=lower_bound, upper=upper_bound)
        
#         # Scale large values if needed
#         max_val = df_processed[column].abs().max()
#         if max_val > 1e6:
#             scale_factor = 1e6 / max_val
#             df_processed[column] = df_processed[column] * scale_factor
        
#         # Convert to float32
#         df_processed[column] = df_processed[column].astype('float32')
    
#     # Fill remaining NaN values
#     df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
#     return df_processed

# # Add data validation function
# def validate_data(df, stage=""):
#     """Validate data at different stages of processing"""
#     print(f"\nValidation at {stage}:")
#     print(f"Shape: {df.shape}")
#     print(f"Data types:\n{df.dtypes}")
#     print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
#     print(f"Contains infinity: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
#     print(f"Contains NaN: {df.isna().sum().sum()}")
    
#     # Check value ranges
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     for col in numeric_cols:
#         max_val = df[col].abs().max()
#         if max_val > 1e6:
#             print(f"Warning: Large values in {col}: {max_val}")

# # Modified execution pipeline
# # First, verify input data
# validate_data(df, "input")

# # Clean and preprocess
# df_clean = clean_and_prepare_data(df)
# df_processed = preprocess_data(df_clean)
# validate_data(df_processed, "after preprocessing")

# # Continue with existing pipeline...





# # Prepare the data
# df_ml = df.copy()
# df_ml['Price'] = df_ml['close']
# df_ml['Volume'] = df_ml['volume']
# df_ml['Returns'] = df_ml['daily_returns']
# df_ml['Volatility'] = df_ml['volatility']
# df_ml['RSI'] = df_ml['RSI']
# df_ml['MACD'] = df_ml['MACD']
# df_ml['Signal Line'] = df_ml['Signal_Line']
# df_ml['Position'] = df_ml['Position']
# df_ml['Strategy Returns'] = df_ml['Strategy_Returns']
# df_ml['ma_ratio'] = df_ml['close'] / df_ml['close'].rolling(window=50).mean()
# df_ml['volume_ma_ratio'] = df_ml['volume'] / df_ml['volume'].rolling(window=20).mean()


# # Drop missing values
# df_ml = df_ml.dropna()  

# df_ml.head()





# def preprocess_data(df):
#     """Handle infinite values, large numbers, and ensure float32 compatibility"""
#     df_processed = df.copy()
    
#     # First, identify numeric columns
#     numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    
#     # Handle infinities and NaN values
#     df_processed[numeric_columns] = df_processed[numeric_columns].replace([np.inf, -np.inf], np.nan)
    
#     for column in numeric_columns:
#         # Calculate robust statistics using percentiles
#         q1 = df_processed[column].quantile(0.25)
#         q3 = df_processed[column].quantile(0.75)
#         iqr = q3 - q1
#         lower_bound = q1 - 1.5 * iqr
#         upper_bound = q3 + 1.5 * iqr
        
#         # Clip outliers
#         df_processed[column] = df_processed[column].clip(lower=lower_bound, upper=upper_bound)
        
#         # Scale large values if needed
#         max_val = df_processed[column].abs().max()
#         if max_val > 1e6:
#             scale_factor = 1e6 / max_val
#             df_processed[column] = df_processed[column] * scale_factor
        
#         # Convert to float32
#         df_processed[column] = df_processed[column].astype('float32')
    
#     # Fill remaining NaN values
#     df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
#     return df_processed

# # Add data validation function
# def validate_data(df, stage=""):
#     """Validate data at different stages of processing"""
#     print(f"\nValidation at {stage}:")
#     print(f"Shape: {df.shape}")
#     print(f"Data types:\n{df.dtypes}")
#     print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
#     print(f"Contains infinity: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
#     print(f"Contains NaN: {df.isna().sum().sum()}")
    
#     # Check value ranges
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     for col in numeric_cols:
#         max_val = df[col].abs().max()
#         if max_val > 1e6:
#             print(f"Warning: Large values in {col}: {max_val}")

# # Modified execution pipeline
# # First, verify input data
# validate_data(df, "input")

# # Clean and preprocess
# df_clean = clean_and_prepare_data(df)
# df_processed = preprocess_data(df_clean)
# validate_data(df_processed, "after preprocessing")

# # Continue with existing pipeline...





# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
# from sklearn.feature_selection import SelectFromModel
# from sklearn.preprocessing import RobustScaler
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import r2_score, mean_squared_error
# import numpy as np

# def clean_and_prepare_data(df):
#     """Clean and prepare data for modeling"""
#     df_clean = df.copy()
    
#     # Remove any string columns or convert them if needed
#     numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
#     df_clean = df_clean[numeric_columns]
    
#     # Handle any missing values
#     df_clean = df_clean.fillna(method='ffill')
    
#     return df_clean

# def build_advanced_features(df):
#     """Create comprehensive technical features"""
#     # First clean the data
#     df_clean = clean_and_prepare_data(df)
#     df_features = df_clean.copy()
    
#     try:
#         # Basic price features
#         df_features['returns'] = df_features['close'].pct_change()
#         df_features['log_returns'] = np.log1p(df_features['returns'].fillna(0))
        
#         # Trend indicators
#         for window in [7, 14, 21, 50, 200]:
#             df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
#             df_features[f'ema_{window}'] = df_features['close'].ewm(span=window).mean()
#             df_features[f'return_{window}d'] = df_features['close'].pct_change(window)
        
#         # Volatility indicators
#         for window in [7, 14, 21]:
#             df_features[f'volatility_{window}d'] = df_features['returns'].rolling(window=window).std()
#             df_features[f'atr_{window}'] = calculate_atr(df_features, window)
        
#         # Volume indicators
#         if 'volume' in df_features.columns:
#             df_features['volume_price_ratio'] = df_features['volume'] * df_features['close']
#             df_features['volume_ma_ratio'] = df_features['volume'] / df_features['volume'].rolling(window=20).mean()
        
#         # Technical indicators
#         df_features['rsi'] = calculate_rsi(df_features['close'])
#         df_features['macd'] = calculate_macd(df_features['close'])
#         df_features['bollinger_upper'], df_features['bollinger_lower'] = calculate_bollinger_bands(df_features['close'])
        
#         # Fill any remaining NaN values
#         df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
#         print("Features created successfully")
#         return df_features
        
#     except Exception as e:
#         print(f"Error in feature creation: {str(e)}")
#         raise

# def train_ensemble_model(df_features, target_days=30):
#     """Train an ensemble model with advanced techniques"""
#     try:
#         # Verify features are numeric
#         numeric_features = df_features.select_dtypes(include=['float64', 'int64'])
#         print(f"Number of features: {len(numeric_features.columns)}")
        
#         # Prepare target variable
#         df_features['target'] = df_features['close'].shift(-target_days).pct_change(target_days)
        
#         # Select features and target
#         feature_cols = [col for col in numeric_features.columns 
#                        if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
        
#         X = df_features[feature_cols].dropna()
#         y = df_features['target'].dropna()
        
#         print(f"Training data shape: X: {X.shape}, y: {y.shape}")
        
         
#         # Initialize models
#         rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#         gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#         ensemble_model = VotingRegressor(estimators=[('rf', rf_model), ('gb', gb_model)])

#         # Time series split
#         tscv = TimeSeriesSplit(n_splits=5)

#         # Cross-validation scores
#         cv_scores = []
#         predictions = []

#         for train_idx, test_idx in tscv.split(X):
#             X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#             # Scale features
#             scaler = RobustScaler()
#             X_train_scaled = scaler.fit_transform(X_train)
#             X_test_scaled = scaler.transform(X_test)

#             # Train and predict
#             ensemble_model.fit(X_train_scaled, y_train)
#             pred = ensemble_model.predict(X_test_scaled)

#             # Store results
#             cv_scores.append(r2_score(y_test, pred))
#             predictions.extend(pred)

#         print("\nCross-validation R² scores:")
#         for i, score in enumerate(cv_scores, 1):
#             print(f"Fold {i}: {score:.4f}")
#         print(f"Average R² score: {np.mean(cv_scores):.4f}")

#         return ensemble_model, cv_scores
    
#     except Exception as e:
#         print(f"Error in model training: {str(e)}")
#         raise
    
# clean_and_prepare_data(df)
# df_features = build_advanced_features(df)
# ensemble_model, cv_scores = train_ensemble_model(df_features)

