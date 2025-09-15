import numpy as np
import pandas as pd
import scipy
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import shap
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler  
from sentiment_analysis import (
    load_and_preprocess_sentiment,
    aggregate_daily_sentiment,
    merge_with_price_data,
    create_sentiment_features,
    plot_sentiment_analysis
)
import logging
import pandas_ta as ta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def verify_data_quality(X_df, y):
    """Verify data quality and alignment"""
    checks = {
        "Data shapes match": len(X_df) == len(y),
        "No missing values": not X_df.isnull().any().any(),
        "Required columns present": all(col in X_df.columns for col in ['Close', 'RSI', 'MACD']),
        "Target variable valid": not np.isnan(y).any()
    }
    
    failed_checks = [k for k, v in checks.items() if not v]
    if failed_checks:
        raise ValueError(f"Data quality checks failed: {failed_checks}")
    
    print("All data quality checks passed")
    return True



# Replace the data loading and initial processing section
try:
    # Load data with explicit column names
    X_df = pd.read_csv('X_prepared.csv')
    print(f"Loaded X_df shape: {X_df.shape}")
    y_df = pd.read_csv('y1_prepared.csv')
    print(f"Loaded y_df shape: {y_df.shape}")

    # Drop rows with all NaN values
    X_df = X_df.dropna(how='all')
    y_df = y_df.dropna(how='all')

    print("After dropna:")
    print("X_df shape:", X_df.shape)
    print("y_df shape:", y_df.shape)
    print(X_df.head())
    print(y_df.head())
    
    # Initial data validation
    print("\nInitial data shapes:")
    print(f"X_df: {X_df.shape}, columns: {list(X_df.columns)}")
    print(f"y_df: {y_df.shape}, columns: {list(y_df.columns)}")
    
    
    #  Create datetime index
    date_range = pd.date_range(
        start='2020-01-01',
        periods=len(X_df),
        freq='D'
    )
    
    # 5. Set index and verify
    X_df.index = date_range
    y_df.index = date_range

    # Extract target variable
    y = y_df['percentreturn'].values

    print("\nFirst 5 rows of X_df:")
    print(X_df.head())
    print("\nFirst 5 values of y:")
    print(y[:5])
    
    # 6. Process sentiment data
    print("\nProcessing sentiment data...")
    sentiment_df = load_and_preprocess_sentiment('data/all-data.csv')
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    
    # 7. Merge sentiment with price data
    X_df = merge_with_price_data(X_df, daily_sentiment, suffixes=('', '_sent'))
    print(f"\nShape after sentiment merge: {X_df.shape}")
    
    # 8. Create features
    X_df = create_sentiment_features(X_df)
    print(f"Shape after feature creation: {X_df.shape}")
    
    # 9. Final validation
    if X_df.empty or len(y) == 0:
        raise ValueError("Empty dataset after processing")
        
except Exception as e:
    print(f"Error in data processing pipeline: {str(e)}")
    raise

# Winsorize percent returns at ±10%
print("Winsorizing percent returns...")
y = winsorize(y, limits=[0.10, 0.10])  # 10% on both tails
print(f"Original range: [{np.min(y):.4f}, {np.max(y):.4f}]")

# Log winsorization statistics
logger.info(f"Winsorized returns statistics:")
logger.info(f"Mean: {np.mean(y):.4f}")
logger.info(f"Std: {np.std(y):.4f}")
logger.info(f"Skewness: {scipy.stats.skew(y):.4f}")
logger.info(f"Kurtosis: {scipy.stats.kurtosis(y):.4f}")

# Feature engineering
core_features = [
    'Close', 'volatility_volume_ratio', 'rolling_vol_7', 'rolling_return_7',
    'CCI', 'ADX_pos', 'price_range', 'volatility', 'RSI', 'MACD'
]

# Add sentiment features to core_features
sentiment_features = ['sentiment_mean', 'sentiment_ma7', 'sentiment_ma30']

# Verify sentiment features
print("\nSentiment features added:")
sentiment_cols = [col for col in X_df.columns if 'sentiment' in col]
print(X_df[sentiment_cols].head())

# Add interaction features
X_df['sentiment_price_impact'] = X_df['sentiment_mean'] * X_df['Close']
# X_df['sentiment_volatility'] = X_df['sentiment_std'] * X_df['volatility']

# Add to sentiment_features list
sentiment_features.extend(['sentiment_price_impact', 'sentiment_volatility'])

# Add verification code here
print("\nVerifying sentiment integration:")
print(f"Original features: {len(core_features)}")
print(f"Added sentiment features: {len(sentiment_features)}")
print(f"Total features: {X_df.shape[1]}")

# Replace the existing Bollinger Bands calculation with:
print("\nCalculating Bollinger Bands...")
try:
    bb = ta.bbands(X_df['Close'], length=20, std=2)
    if bb is not None:
        X_df['BB_upper'] = bb['BBU_20_2.0']
        X_df['BB_middle'] = bb['BBM_20_2.0']
        X_df['BB_lower'] = bb['BBL_20_2.0']
        X_df['BB_width'] = (X_df['BB_upper'] - X_df['BB_lower']) / X_df['BB_middle']
        print("Bollinger Bands calculated successfully")
    else:
        print("Warning: Bollinger Bands calculation returned None")
except Exception as e:
    print(f"Error calculating Bollinger Bands: {str(e)}")
    # Use alternative calculation or skip
    pass

# Add BB features to sentiment_features list
bb_features = ['BB_upper', 'BB_middle', 'BB_lower', 'BB_width']
sentiment_features.extend(bb_features)

core_features.extend([f for f in sentiment_features if f in X_df.columns])

extended_features = ['High', 'Low', 'ATR', 'Momentum_10']
features_to_use = [f for f in core_features + extended_features if f in X_df.columns]
X_df = X_df[features_to_use]

# Add data quality check
print("\nChecking for remaining null values:")
null_check = X_df.isnull().sum()
if null_check.any():
    print("Warning: Found null values in:")
    print(null_check[null_check > 0])
    # Handle remaining nulls
    # X_df = X_df.fillna(method='ffill').fillna(method='bfill')

# Add momentum divergence and interaction term
X_df['momentum_divergence'] = X_df['Close'].diff(5) - X_df['Close'].diff(20)
if 'MACD' in X_df.columns and 'Momentum_10' in X_df.columns:
    X_df['macd_momentum'] = X_df['MACD'] * X_df['Momentum_10']

# Add lags and moving averages for key features
key_features = ['Close', 'MACD', 'RSI', 'Momentum_10', 'ATR'] if 'Momentum_10' in X_df.columns and 'ATR' in X_df.columns else ['Close', 'MACD', 'RSI']
for col in [c for c in key_features if c in X_df.columns]:
    for lag in range(1, 4):
        X_df[f'{col}_lag_{lag}'] = X_df[col].shift(lag)
    X_df[f'{col}_ma_5'] = X_df[col].rolling(window=5).mean()

# Remove features that are all NaN
for col in X_df.columns:
    if X_df[col].isnull().all():
        print(f"Dropping all-NaN feature: {col}")
        X_df = X_df.drop(columns=[col])

# Drop rows with NaN only in required/rolling columns
rolling_cols = ['sentiment_ma7', 'sentiment_ma30', 'BB_upper', 'BB_middle', 'BB_lower', 'BB_width']
X_df = X_df.dropna(subset=rolling_cols)

# Now align y
y = y_df.loc[X_df.index, 'percentreturn'].values

# Convert to numpy array
X = X_df.values

# Add after data loading
verify_data_quality(X_df, y)


# Add these debug prints after each major data transformation
print("\nData shape after initial load:")
print(f"X_df shape: {X_df.shape}")
print(f"y shape: {len(y)}")


# After feature selection
X_df = X_df[features_to_use]
print("\nData shape after feature selection:")
print(f"X_df shape: {X_df.shape}")

# Before train_test_split
print("\nFinal shapes before split:")
print(f"X shape: {X.shape}")
print(f"y shape: {len(y)}")

# Add data validation check
if len(X_df) == 0:
    raise ValueError("DataFrame is empty after preprocessing. Check data transformations.")

if len(y) == 0:
    raise ValueError("Target variable array is empty. Check index alignment.")


# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Preprocessing pipeline
preprocessor = make_pipeline(
    KNNImputer(n_neighbors=5),
    RobustScaler()             # Changed to RobustScaler for better handling of outliers
)
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Create sequences for RNN (T=10)
def create_sequences(X, y, T):
    Xs, ys = [], []
    for i in range(len(X) - T):
        Xs.append(X[i:(i+T), :])
        ys.append(y[i+T])
    return np.array(Xs), np.array(ys)

T = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, T)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, T)

# RNN model builder
def create_rnn_model(trial, input_shape):
    units1 = trial.suggest_int('units1', 16, 64)
    units2 = trial.suggest_int('units2', 8, 32)
    dropout = trial.suggest_float('dropout', 0.4, 0.6)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = Sequential([
        LSTM(units1, input_shape=input_shape, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(dropout),
        LSTM(units2, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

# Optuna objective for RNN
def objective_rnn(trial):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_seq, y_train_seq, test_size=0.2, shuffle=False)
    model = create_rnn_model(trial, input_shape=(T, X_train_seq.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
    preds = model.predict(X_val, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    tf.keras.backend.clear_session()
    return rmse

# Tune RNN
study = optuna.create_study(direction='minimize')
study.optimize(objective_rnn, n_trials=200)  # Increased trials
best_rnn_params = study.best_trial.params
rnn_cv_rmse = study.best_value

# Train best RNN
best_rnn = create_rnn_model(study.best_trial, input_shape=(T, X_train_seq.shape[2]))
best_rnn.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)], verbose=0)
rnn_preds = best_rnn.predict(X_test_seq, verbose=0).flatten()

# SHAP feature selection for tree-based models
X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(X_train_scaled, y_train, test_size=0.1, shuffle=False)
lgbm_initial = LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=5)
lgbm_initial.fit(X_tr_final, y_tr_final)
explainer = shap.TreeExplainer(lgbm_initial)
shap_values = explainer.shap_values(X_tr_final)
shap_importances = np.abs(shap_values).mean(axis=0)
feature_indices = np.argsort(shap_importances)[::-1]
top_features = feature_indices[:int(0.85 * len(feature_indices))]

# Select top features
X_train_scaled_selected = X_train_scaled[:, top_features]
X_test_scaled_selected = X_test_scaled[:, top_features]
X_tr_final_selected = X_tr_final[:, top_features]
X_val_final_selected = X_val_final[:, top_features]

# Print feature importances
feature_names = X_df.columns.tolist()
selected_feature_names = [feature_names[i] for i in top_features if i < len(feature_names)]
print("Feature Importances (Top Features based on SHAP):")
for name, imp in zip(selected_feature_names, shap_importances[top_features[:len(selected_feature_names)]]):
    print(f"{name}: {imp}")

# Replace the XGBoost tuning section with:
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'random_state': 42
    }
    
    model = XGBRegressor(**params)
    scores = cross_val_score(
        model, 
        X_train_scaled_selected, 
        y_train, 
        cv=TimeSeriesSplit(n_splits=7),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    return -np.mean(scores)

# XGBoost optimization
study_xgb = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5
    )
)
study_xgb.optimize(objective_xgb, n_trials=200)

# Get best XGBoost model
best_xgb = XGBRegressor(**study_xgb.best_params)
best_xgb.fit(X_train_scaled_selected, y_train)
xgb_cv_rmse = np.sqrt(-study_xgb.best_value)
xgb_preds = best_xgb.predict(X_test_scaled_selected[T:])

# Replace the LightGBM tuning section with:
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        # Fixed reg_lambda parameter suggestion
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'random_state': 42
    }
    
    model = LGBMRegressor(**params)
    scores = cross_val_score(
        model, 
        X_train_scaled_selected, 
        y_train, 
        cv=TimeSeriesSplit(n_splits=7),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    return -np.mean(scores)

# LightGBM optimization
studylgbm = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5
    )
)
studylgbm.optimize(objective_lgbm, n_trials=100)

# Get best LightGBM model
best_lgbm = LGBMRegressor(**studylgbm.best_params)
best_lgbm.fit(X_train_scaled_selected, y_train)
lgbm_cv_rmse = np.sqrt(-studylgbm.best_value)
lgbm_preds = best_lgbm.predict(X_test_scaled_selected[T:])

# Replace the existing ensemble weighting section with:
def optimize_ensemble_weights(predictions, y_true):
    """
    Optimize ensemble weights using RidgeCV
    """
    # Stack predictions
    stacked_preds = np.column_stack(predictions)
    
    # Create and fit meta-learner
    meta_learner = RidgeCV(
        alphas=[0.1, 1.0, 10.0],
        cv=TimeSeriesSplit(n_splits=5)
    )
    meta_learner.fit(stacked_preds, y_true)
    
    # Get optimal weights
    weights = meta_learner.coef_
    weights = np.maximum(weights, 0)  # Ensure non-negative weights
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    return weights, meta_learner

# Optimize weights
print("\nOptimizing ensemble weights...")
weights, meta_learner = optimize_ensemble_weights(
    [rnn_preds, xgb_preds, lgbm_preds],
    y_test_seq
)

# Create weighted predictions
ensemble_preds = meta_learner.predict(
    np.column_stack([rnn_preds, xgb_preds, lgbm_preds])
)

# Update weight reporting
print(f"Optimized Ensemble Weights:")
print(f"RNN: {weights[0]:.4f}")
print(f"XGBoost: {weights[1]:.4f}")
print(f"LightGBM: {weights[2]:.4f}")

# Add after weight optimization
def cv_ensemble_performance(X, y, n_splits=5):
    """Cross-validate ensemble performance"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
        meta.fit(X_train, y_train)
        pred = meta.predict(X_test)
        
        score = np.sqrt(mean_squared_error(y_test, pred))
        cv_scores.append(score)
    
    print(f"\nCross-validation RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


# Weighted ensemble
ensemble_rmse = np.sqrt(mean_squared_error(y_test_seq, ensemble_preds))
ensemble_mae = mean_absolute_error(y_test_seq, ensemble_preds)
y_test_clipped = np.where(np.abs(y_test_seq) < 1e-6, 1e-6, y_test_seq)
ensemble_mape = np.mean(np.abs((ensemble_preds - y_test_seq) / y_test_clipped)) * 100
ensemble_r2 = r2_score(y_test_seq, ensemble_preds)
directional_accuracy = np.mean(np.sign(ensemble_preds) == np.sign(y_test_seq)) * 100
max_error = np.max(np.abs(ensemble_preds - y_test_seq))

# Print results
print(f"RNN CV RMSE: {rnn_cv_rmse:.4f}")
print(f"XGBoost CV RMSE: {xgb_cv_rmse:.4f}")
print(f"LightGBM CV RMSE: {lgbm_cv_rmse:.4f}")
print(f"Weighted Ensemble RMSE: {ensemble_rmse:.4f}")
print(f"Weighted Ensemble MAE: {ensemble_mae:.4f}")
print(f"Weighted Ensemble MAPE (clipped): {ensemble_mape:.2f}%")
print(f"Weighted Ensemble R²: {ensemble_r2:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Maximum Error: {max_error:.4f}")

# Save models and preprocessor
best_rnn.save('bestrnn41.h5')
joblib.dump(best_xgb, 'bestxgb41.pkl')
joblib.dump(best_lgbm, 'bestlgbm41.pkl')
joblib.dump(preprocessor, 'preprocessor41.pkl')
joblib.dump(meta_learner, 'meta_learner41.pkl')