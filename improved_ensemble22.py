import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
import lightgbm as lgb
import pandas_ta as ta
from scipy.stats.mstats import winsorize
import mlflow
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import VotingRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Start MLflow run
mlflow.start_run()

# Load data
X_df = pd.read_csv('X_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# Advanced outlier handling using IQR method before winsorizing
def remove_outliers_iqr(data, column, multiplier=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return np.clip(data, lower_bound, upper_bound)

# Apply outlier removal before winsorizing
y = remove_outliers_iqr(y, 'percentreturn', multiplier=2.0)

# Apply winsorization with tighter bounds
y = winsorize(y, limits=[0.025, 0.025])

# Enhanced feature engineering
def engineer_features(df):
    """Enhanced feature engineering with validation"""
    # Check for required columns
    required_columns = ['Close', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        return df
    
    df = df.copy()
    # Price-based indicators
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Bollinger Bands
    bbands_df = ta.bbands(df['Close'], length=20, std=2)
    df['BB_upper'] = bbands_df['BBU_20_2.0']
    df['BB_middle'] = bbands_df['BBM_20_2.0']
    df['BB_lower'] = bbands_df['BBL_20_2.0']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # RSI with multiple periods
    for period in [6, 14, 21]:
        df[f'RSI_{period}'] = ta.rsi(df['Close'], length=period)
    
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    
    # Average True Range for volatility
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Rate of Change
    for period in [5, 10, 20]:
        df[f'ROC_{period}'] = ta.roc(df['Close'], length=period)
    
    # Moving Averages and crosses
    for period in [10, 20, 50]:
        df[f'SMA_{period}'] = ta.sma(df['Close'], length=period)
        df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)
    
    # Volume indicators
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['Volume_SMA_5'] = ta.sma(df['Volume'], length=5)
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_5']
    
    # Price distance from moving averages
    for ma in [20, 50]:
        df[f'Close_SMA{ma}_ratio'] = df['Close'] / df[f'SMA_{ma}']
    
    # Add lagged variables with expanded range
    for col in df.select_dtypes(include=[np.number]).columns:
        # Add lags
        for lag in range(1, 6):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Add moving averages
        for window in [5, 10, 20]:
            df[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
        
        # Add momentum features
        df[f'{col}_mom_5'] = df[col] - df[col].shift(5)
    
    # Add day of week and month cyclical features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    
    # Feature interactions
    df['close_volume_ratio'] = df['Close'] / (df['Volume'] + 1)  # Avoid division by zero
    
    return df

# Apply enhanced feature engineering
X_df = engineer_features(X_df)

# Add preprocessing sanity checks
nan_count_before = X_df.isna().sum().sum()
print(f"Total NaN values before preprocessing: {nan_count_before}")

# Store original index before dropping NaNs
original_index = X_df.index.copy()
X_df = X_df.dropna()
print(f"Dropped {len(original_index) - len(X_df)} rows with NaN values")

# Update y to match the filtered X_df
y = y[X_df.index]

# Convert to numpy array
X = X_df.values

# Split into train, validation, and test sets (70% train, 15% validation, 15% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, shuffle=False)  # ~15% of original data

# Save column names for feature importance analysis
feature_names = X_df.columns.tolist()
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))

# Modify the preprocessing pipeline section:
def create_and_fit_preprocessors(X_train, X_val, X_test):
    """Create and fit preprocessing pipelines"""
    
    # RNN preprocessor
    preprocessor_rnn = make_pipeline(
        KNNImputer(n_neighbors=5),
        QuantileTransformer(output_distribution='normal'),
        PCA(n_components=0.95, random_state=RANDOM_SEED)
    )
    
    # Tree-based models preprocessor
    preprocessor_tree = make_pipeline(
        KNNImputer(n_neighbors=5),
        RobustScaler()
    )
    
    # Fit and transform training data
    X_train_scaled_rnn = preprocessor_rnn.fit_transform(X_train)
    X_train_scaled_tree = preprocessor_tree.fit_transform(X_train)
    
    # Transform validation and test data using fitted preprocessors
    X_val_scaled_rnn = preprocessor_rnn.transform(X_val)
    X_test_scaled_rnn = preprocessor_rnn.transform(X_test)
    
    X_val_scaled_tree = preprocessor_tree.transform(X_val)
    X_test_scaled_tree = preprocessor_tree.transform(X_test)
    
    return {
        'rnn': (X_train_scaled_rnn, X_val_scaled_rnn, X_test_scaled_rnn),
        'tree': (X_train_scaled_tree, X_val_scaled_tree, X_test_scaled_tree),
        'preprocessors': {
            'rnn': preprocessor_rnn,
            'tree': preprocessor_tree
        }
    }

# Replace the preprocessing section with:
print("Preprocessing data...")
processed_data = create_and_fit_preprocessors(X_train, X_val, X_test)

# Get scaled data
X_train_scaled_rnn, X_val_scaled_rnn, X_test_scaled_rnn = processed_data['rnn']
X_train_scaled_tree, X_val_scaled_tree, X_test_scaled_tree = processed_data['tree']

# Save preprocessors
joblib.dump(processed_data['preprocessors'], 'preprocessors3.pkl')

# Add verification
print("\nVerifying preprocessed data shapes:")
print(f"RNN Training data shape: {X_train_scaled_rnn.shape}")
print(f"RNN Validation data shape: {X_val_scaled_rnn.shape}")
print(f"RNN Test data shape: {X_test_scaled_rnn.shape}")
print(f"Tree Training data shape: {X_train_scaled_tree.shape}")

# Create sequences for RNN with variable sequence length optimization
def create_sequences(data, target, T):
    X_seq, y_seq = [], []
    for i in range(len(data) - T):
        X_seq.append(data[i:i+T, :])
        y_seq.append(target[i+T])
    return np.array(X_seq), np.array(y_seq)

# Optimize sequence length with Optuna
def objective_sequence_length(trial):
    seq_length = trial.suggest_int('seq_length', 3, 15)
    X_tr_seq, y_tr_seq = create_sequences(X_train_scaled_rnn, y_train, seq_length)
    split = int(0.8 * len(X_tr_seq))
    X_tr, X_val = X_tr_seq[:split], X_tr_seq[split:]
    y_tr, y_val = y_tr_seq[:split], y_tr_seq[split:]
    
    # Simple model for sequence length optimization
    model = Sequential([
        LSTM(32, input_shape=(seq_length, X_tr.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[es], verbose=0)
    preds = model.predict(X_val, verbose=0).flatten()
    return np.sqrt(mean_squared_error(y_val, preds))

# Tune sequence length
study_seq = optuna.create_study(direction='minimize')
study_seq.optimize(objective_sequence_length, n_trials=30)
best_seq_length = study_seq.best_trial.params['seq_length']
mlflow.log_param('best_sequence_length', best_seq_length)

# Use optimized sequence length
T = best_seq_length
X_train_seq, y_train_seq = create_sequences(X_train_scaled_rnn, y_train, T)
X_val_seq, y_val_seq = create_sequences(X_val_scaled_rnn, y_val, T)
X_test_seq, y_test_seq = create_sequences(X_test_scaled_rnn, y_test, T)

# Improved RNN model builder with bidirectional option and batch normalization
def create_rnn_model(trial, input_shape):
    rnn_type = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU'])
    units = trial.suggest_int('units', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    reg_strength = trial.suggest_loguniform('reg_strength', 1e-5, 1e-3)

    model = Sequential()
    
    for i in range(n_layers):
        # First layer needs input shape
        if i == 0:
            if bidirectional:
                if rnn_type == 'LSTM':
                    model.add(Bidirectional(LSTM(units, return_sequences=i < n_layers-1,
                                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
                                          input_shape=input_shape))
                else:  # GRU
                    model.add(Bidirectional(GRU(units, return_sequences=i < n_layers-1,
                                              kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
                                          input_shape=input_shape))
            else:
                if rnn_type == 'LSTM':
                    model.add(LSTM(units, return_sequences=i < n_layers-1,
                                 kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
                                 input_shape=input_shape))
                else:  # GRU
                    model.add(GRU(units, return_sequences=i < n_layers-1,
                                kernel_regularizer=tf.keras.regularizers.l2(reg_strength),
                                input_shape=input_shape))
        else:
            # Middle/last layers
            if bidirectional:
                if rnn_type == 'LSTM':
                    model.add(Bidirectional(LSTM(units, return_sequences=i < n_layers-1,
                                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
                else:  # GRU
                    model.add(Bidirectional(GRU(units, return_sequences=i < n_layers-1,
                                              kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
            else:
                if rnn_type == 'LSTM':
                    model.add(LSTM(units, return_sequences=i < n_layers-1,
                                 kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
                else:  # GRU
                    model.add(GRU(units, return_sequences=i < n_layers-1,
                                kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
        
        # Add batch normalization if enabled
        if batch_norm:
            model.add(BatchNormalization())
            
        # Add dropout after each layer
        model.add(Dropout(dropout))
    
    # Add a dense hidden layer
    model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
    model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile with optimized learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

# Optuna objective for improved RNN
def objective_rnn(trial):
    model = create_rnn_model(trial, X_train_seq.shape[1:])
    
    # Add learning rate scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Train with validation data
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
        callbacks=[es, reduce_lr],
        verbose=0
    )
    
    # Predict on validation set
    val_preds = model.predict(X_val_seq, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_val_seq, val_preds))
    return rmse

# Tune RNN with more trials
study_rnn = optuna.create_study(direction='minimize')
study_rnn.optimize(objective_rnn, n_trials=100)
best_params_rnn = study_rnn.best_trial.params
mlflow.log_params(best_params_rnn)

# Train best RNN
model_rnn = create_rnn_model(study_rnn.best_trial, X_train_seq.shape[1:])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model_rnn.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=100,
    batch_size=best_params_rnn.get('batch_size', 32),
    callbacks=[es, reduce_lr],
    verbose=1
)
model_rnn.save('rnn10_improved.h5')

# Advanced objective for LightGBM with focal parameters
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'subsample': trial.suggest_uniform('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'random_state': RANDOM_SEED,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    
    for train_idx, val_idx in tscv.split(X_train_scaled_tree):
        X_tr, X_val = X_train_scaled_tree[train_idx], X_train_scaled_tree[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        # Corrected fit parameters
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmses)

# Tune LightGBM with more trials
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=100)
best_params_lgbm = study_lgbm.best_trial.params
mlflow.log_params(best_params_lgbm)

# Train best LightGBM
model_lgbm = lgb.LGBMRegressor(**best_params_lgbm)
model_lgbm.fit(
    X_train_scaled_tree, y_train,
    eval_set=[(X_val_scaled_tree, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)
    ]
)

# Save model
joblib.dump(model_lgbm, 'lgbm10_improved.pkl')

# Log feature importances
lgbm_feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model_lgbm.feature_importances_
}).sort_values('Importance', ascending=False).head(30)
lgbm_feature_importance.to_csv('lgbm_feature_importance.csv', index=False)
mlflow.log_artifact('lgbm_feature_importance.csv')

# CatBoost implementation
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-2, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_seed': RANDOM_SEED,
        'verbose': False
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    
    for train_idx, val_idx in tscv.split(X_train_scaled_tree):
        X_tr, X_val = X_train_scaled_tree[train_idx], X_train_scaled_tree[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmses)

# Tune CatBoost
study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(objective_catboost, n_trials=50)
best_params_catboost = study_catboost.best_trial.params
mlflow.log_params(best_params_catboost)

# Train best CatBoost
model_catboost = CatBoostRegressor(**best_params_catboost)
model_catboost.fit(
    X_train_scaled_tree, y_train,
    eval_set=[(X_val_scaled_tree, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Save model
joblib.dump(model_catboost, 'catboost10_improved.pkl')

# XGBoost with Optuna instead of GridSearchCV
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'random_state': RANDOM_SEED
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    
    for train_idx, val_idx in tscv.split(X_train_scaled_tree):
        X_tr, X_val = X_train_scaled_tree[train_idx], X_train_scaled_tree[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(rmses)

# Tune XGBoost with more trials
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=75)
best_params_xgb = study_xgb.best_trial.params
mlflow.log_params(best_params_xgb)

# Train best XGBoost
model_xgb = XGBRegressor(**best_params_xgb)
model_xgb.fit(
    X_train_scaled_tree, y_train,
    eval_set=[(X_val_scaled_tree, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Save model
joblib.dump(model_xgb, 'xgb10_improved.pkl')

# MLP Neural Network
def objective_mlp(trial):
    params = {
        'hidden_layer_sizes': tuple(trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(
            trial.suggest_int('n_layers', 1, 3))),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-2),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-2),
        'max_iter': 1000,
        'early_stopping': True,
        'random_state': RANDOM_SEED
    }
    
    model = MLPRegressor(**params)
    model.fit(X_train_scaled_tree, y_train)
    preds = model.predict(X_val_scaled_tree)
    
    return np.sqrt(mean_squared_error(y_val, preds))

# Tune MLP
study_mlp = optuna.create_study(direction='minimize')
study_mlp.optimize(objective_mlp, n_trials=50)
best_params_mlp = study_mlp.best_trial.params
mlflow.log_params(best_params_mlp)

# Train best MLP
model_mlp = MLPRegressor(**best_params_mlp)
model_mlp.fit(X_train_scaled_tree, y_train)

# Save model
joblib.dump(model_mlp, 'mlp10_improved.pkl')

# Gradient Boosting Regressor
model_gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    random_state=RANDOM_SEED
)
model_gbr.fit(X_train_scaled_tree, y_train)
joblib.dump(model_gbr, 'gbr10_improved.pkl')

# Generate predictions from all models
rnn_val_preds = model_rnn.predict(X_val_seq, verbose=0).flatten()
lgbm_val_preds = model_lgbm.predict(X_val_scaled_tree)
xgb_val_preds = model_xgb.predict(X_val_scaled_tree)
catboost_val_preds = model_catboost.predict(X_val_scaled_tree)
mlp_val_preds = model_mlp.predict(X_val_scaled_tree)
gbr_val_preds = model_gbr.predict(X_val_scaled_tree)

# Generate test predictions
preds_test_rnn = model_rnn.predict(X_test_seq, verbose=0).flatten()
preds_test_lgbm = model_lgbm.predict(X_test_scaled_tree)
preds_test_xgb = model_xgb.predict(X_test_scaled_tree)
preds_test_catboost = model_catboost.predict(X_test_scaled_tree)
preds_test_mlp = model_mlp.predict(X_test_scaled_tree)
preds_test_gbr = model_gbr.predict(X_test_scaled_tree)

# Align validation predictions (ensure they match with y_val_seq)
val_size_seq = len(y_val_seq)

# Truncate if necessary to align length
lgbm_val_preds_aligned = lgbm_val_preds[-val_size_seq:] if len(lgbm_val_preds) > val_size_seq else lgbm_val_preds
xgb_val_preds_aligned = xgb_val_preds[-val_size_seq:] if len(xgb_val_preds) > val_size_seq else xgb_val_preds
catboost_val_preds_aligned = catboost_val_preds[-val_size_seq:] if len(catboost_val_preds) > val_size_seq else catboost_val_preds
mlp_val_preds_aligned = mlp_val_preds[-val_size_seq:] if len(mlp_val_preds) > val_size_seq else mlp_val_preds
gbr_val_preds_aligned = gbr_val_preds[-val_size_seq:] if len(gbr_val_preds) > val_size_seq else gbr_val_preds

# Align test predictions
test_size_seq = len(y_test_seq)
test_offset = len(y_test) - test_size_seq

preds_test_lgbm_aligned = preds_test_lgbm[-test_size_seq:] if len(preds_test_lgbm) >= test_size_seq else preds_test_lgbm
preds_test_xgb_aligned = preds_test_xgb[-test_size_seq:] if len(preds_test_xgb) >= test_size_seq else preds_test_xgb
preds_test_catboost_aligned = preds_test_catboost[-test_size_seq:] if len(preds_test_catboost) >= test_size_seq else preds_test_catboost
preds_test_mlp_aligned = preds_test_mlp[-test_size_seq:] if len(preds_test_mlp) >= test_size_seq else preds_test_mlp
preds_test_gbr_aligned = preds_test_gbr[-test_size_seq:] if len(preds_test_gbr) >= test_size_seq else preds_test_gbr

# Align y_test to match the sequence predictions
y_test_aligned = y_test[-test_size_seq:]

# Enhanced stacking approach: Create multiple meta-models and ensemble them
# Create DataFrame with base model predictions for validation set
val_preds_df = pd.DataFrame({
    'rnn': rnn_val_preds,
    'lgbm': lgbm_val_preds_aligned,
    'xgb': xgb_val_preds_aligned,
    'catboost': catboost_val_preds_aligned,
    'mlp': mlp_val_preds_aligned,
    'gbr': gbr_val_preds_aligned
})

# Calculate correlation matrix to assess model diversity
correlation_matrix = val_preds_df.corr()
print("Model Prediction Correlation Matrix:")
print(correlation_matrix)
correlation_matrix.to_csv('model_correlations.csv')
mlflow.log_artifact('model_correlations.csv')

# Create DataFrames with base model predictions for test set
test_preds_df = pd.DataFrame({
    'rnn': preds_test_rnn,
    'lgbm': preds_test_lgbm_aligned,
    'xgb': preds_test_xgb_aligned,
    'catboost': preds_test_catboost_aligned,
    'mlp': preds_test_mlp_aligned,
    'gbr': preds_test_gbr_aligned
})

# Calculate individual model performance on validation set
model_val_metrics = {}
for model_name in val_preds_df.columns:
    preds = val_preds_df[model_name].values
    rmse = np.sqrt(mean_squared_error(y_val_seq, preds))
    mae = mean_absolute_error(y_val_seq, preds)
    r2 = r2_score(y_val_seq, preds)
    model_val_metrics[model_name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"{model_name} Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Create multiple meta-learners with different approaches
meta_models = {
    'linear': LinearRegression(),
    'huber': HuberRegressor(epsilon=1.35, alpha=0.0001),
    'ridge': Ridge(alpha=0.5, random_state=RANDOM_SEED),
    'lasso': Lasso(alpha=0.001, random_state=RANDOM_SEED),
    'elastic': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_SEED)
}

# Add cross-validation for meta-models
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
meta_cv_scores = {}

for name, model in meta_models.items():
    cv_scores = []
    for train_idx, val_idx in kf.split(val_preds_df):
        X_train_meta, X_val_meta = val_preds_df.iloc[train_idx], val_preds_df.iloc[val_idx]
        y_train_meta, y_val_meta = y_val_seq[train_idx], y_val_seq[val_idx]
        
        model.fit(X_train_meta, y_train_meta)
        preds = model.predict(X_val_meta)
        cv_scores.append(np.sqrt(mean_squared_error(y_val_meta, preds)))
    
    meta_cv_scores[name] = np.mean(cv_scores)
    print(f"Meta-model {name} CV RMSE: {meta_cv_scores[name]:.4f}")
    mlflow.log_metric(f"meta_{name}_cv_rmse", meta_cv_scores[name])

# Train all meta-models
try:
    for name, model in meta_models.items():
        model.fit(val_preds_df, y_val_seq)
        joblib.dump(model, f'meta_model_{name}.pkl')
except Exception as e:
    print(f"Error training meta-models: {e}")
    mlflow.log_param("meta_model_error", str(e))

# Create weighted average ensemble based on validation performance
def inverse_rmse_weights(metrics_dict):
    # Extract RMSE values and convert to inverse (higher value = better model)
    rmse_values = np.array([metrics['RMSE'] for metrics in metrics_dict.values()])
    inverse_rmse = 1 / rmse_values
    # Normalize to sum to 1
    weights = inverse_rmse / inverse_rmse.sum()
    return dict(zip(metrics_dict.keys(), weights))

model_weights = inverse_rmse_weights(model_val_metrics)
print("Model weights based on validation performance:")
for model_name, weight in model_weights.items():
    print(f"{model_name}: {weight:.4f}")
    mlflow.log_param(f"weight_{model_name}", weight)

# Function to create weighted ensemble predictions
def weighted_ensemble_prediction(predictions_df, weights):
    weighted_sum = np.zeros(len(predictions_df))
    for model_name, weight in weights.items():
        weighted_sum += predictions_df[model_name].values * weight
    return weighted_sum

# Make ensemble predictions on test set using different approaches
ensemble_predictions = {}

# 1. Simple average
ensemble_predictions['simple_avg'] = test_preds_df.mean(axis=1).values

# 2. Weighted average based on validation performance
ensemble_predictions['weighted_avg'] = weighted_ensemble_prediction(test_preds_df, model_weights)

# 3. Meta-model predictions
for name, model in meta_models.items():
    ensemble_predictions[f'meta_{name}'] = model.predict(test_preds_df)

# 4. Additional ensemble strategy - stacked meta-ensemble
# First get predictions from all meta-models
meta_preds_df = pd.DataFrame({
    f'meta_{name}': model.predict(test_preds_df) for name, model in meta_models.items()
})

# Then do a weighted average of meta-models
meta_ensemble = meta_preds_df.mean(axis=1).values
ensemble_predictions['meta_ensemble'] = meta_ensemble

# Evaluate all ensemble strategies
ensemble_metrics = {}
for name, preds in ensemble_predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test_aligned, preds))
    mae = mean_absolute_error(y_test_aligned, preds)
    
    # Handle zero values in y_test for MAPE calculation
    epsilon = 1e-10  # Small constant to avoid division by zero
    mape = np.mean(np.abs((preds - y_test_aligned) / (np.abs(y_test_aligned) + epsilon))) * 100
    
    r2 = r2_score(y_test_aligned, preds)
    directional_accuracy = np.mean(np.sign(preds) == np.sign(y_test_aligned)) * 100
    max_error = np.max(np.abs(preds - y_test_aligned))
    
    # Simulate trading strategy for Sharpe ratio
    simulated_returns = np.where(preds > 0.01, 1, -1) * y_test_aligned
    sharpe_ratio = np.mean(simulated_returns) / np.std(simulated_returns) if np.std(simulated_returns) > 0 else 0
    
    ensemble_metrics[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Max_Error': max_error,
        'Sharpe_Ratio': sharpe_ratio
    }
    
    # Print results
    print(f"\n{name.upper()} ENSEMBLE METRICS:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"Maximum Error: {max_error:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Log metrics to MLflow
    for metric_name, value in ensemble_metrics[name].items():
        mlflow.log_metric(f"{name}_{metric_name}", value)

# Find the best ensemble method
best_ensemble = min(ensemble_metrics.items(), key=lambda x: x[1]['RMSE'])[0]
print(f"\nBest Ensemble Method: {best_ensemble}")
mlflow.log_param("best_ensemble_method", best_ensemble)

# Save the final best predictions
best_preds = ensemble_predictions[best_ensemble]
pd.DataFrame({
    'actual': y_test_aligned,
    'predicted': best_preds,
    'error': best_preds - y_test_aligned
}).to_csv('best_ensemble_predictions.csv', index=False)
mlflow.log_artifact('best_ensemble_predictions.csv')

# Further analyze errors
errors = best_preds - y_test_aligned
error_percentiles = np.percentile(np.abs(errors), [25, 50, 75, 90, 95, 99])
print("\nError Analysis:")
print(f"25th percentile: {error_percentiles[0]:.4f}")
print(f"Median error: {error_percentiles[1]:.4f}")
print(f"75th percentile: {error_percentiles[2]:.4f}")
print(f"90th percentile: {error_percentiles[3]:.4f}")
print(f"95th percentile: {error_percentiles[4]:.4f}")
print(f"99th percentile: {error_percentiles[5]:.4f}")

# Check for outliers in the best predictions
z_scores = stats.zscore(errors)
outliers_mask = np.abs(z_scores) > 3
outlier_count = np.sum(outliers_mask)
print(f"Number of outlier predictions (|z| > 3): {outlier_count} ({outlier_count/len(errors)*100:.2f}%)")

# Get final metrics for the best ensemble
best_metrics = ensemble_metrics[best_ensemble]
print("\nFINAL BEST ENSEMBLE METRICS:")
for metric_name, value in best_metrics.items():
    print(f"{metric_name}: {value:.4f}")

# End MLflow run
mlflow.end_run()