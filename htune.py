import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import shap
from sklearn.model_selection import GridSearchCV
from sp500_features import merge_sp500_features


# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load data
X_df = pd.read_csv('X_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

#----------------------------------------------------------------------------------
# Add S&P 500 features
print("\nAdding S&P 500 features...")
X_df = merge_sp500_features(X_df)
    
    # Verify new features
print("\nNew feature correlations with target:")
correlations = X_df[['sp500_return', 'sp500_volatility', 'sp500_trend', 'sp500_rsi']].corrwith(y['percentreturn'])
print(correlations)
#----------------------------------------------------------------------------------

# Check for duplicate rows
if X_df.duplicated().any():
    print("Duplicate rows found in features. Dropping duplicates...")
    X_df.drop_duplicates(inplace=True)
# Check for outliers
if (X_df < -1e6).any().any() or (X_df > 1e6).any().any():
    print("Outliers found in features. Clipping values...")
    X_df = np.clip(X_df, -1e6, 1e6)
# Check for multicollinearity
correlation_matrix = X_df.corr()
highly_correlated_features = set()
threshold = 0.9
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)
            print(f"High correlation found: {colname} with {correlation_matrix.columns[j]}")
            X_df.drop(colname, axis=1, inplace=True)

#-----------------------------------------------------------------------------------

# Feature engineering
core_features = [
    'Close', 'volatility_volume_ratio', 'rolling_vol_7', 'rolling_return_7',
    'CCI', 'ADX_pos', 'price_range', 'volatility', 'RSI', 'MACD'
]
extended_features = ['High', 'Low', 'ATR', 'Momentum_10']
features_to_use = [f for f in core_features + extended_features if f in X_df.columns]
X_df = X_df[features_to_use]

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

# Drop rows with NaN values
X_df = X_df.dropna()
y = y[X_df.index]

# Convert to numpy array
X = X_df.values

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Preprocessing pipeline
preprocessor = make_pipeline(
    KNNImputer(n_neighbors=5),
    RobustScaler()
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
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
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
study.optimize(objective_rnn, n_trials=100)  # Increased trials
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
feature_names = X_df.columns
print("Feature Importances (Top Features based on SHAP):")
for idx, imp in zip(top_features, shap_importances[top_features]):
    print(f"{feature_names[idx]}: {imp}")

# XGBoost tuning
xgb = XGBRegressor(random_state=42)
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [1, 10, 20]
}
tscv = TimeSeriesSplit(n_splits=7)  # Increased splits
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled_selected, y_train)
best_xgb = grid_xgb.best_estimator_
xgb_cv_rmse = np.sqrt(-grid_xgb.best_score_)
xgb_preds = best_xgb.predict(X_test_scaled_selected[T:])

# LightGBM tuning
lgbm = LGBMRegressor(random_state=42)
param_grid_lgbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 30, 40],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [1, 10, 20]
}
tscv = TimeSeriesSplit(n_splits=7)  # Increased splits
grid_lgbm = GridSearchCV(lgbm, param_grid_lgbm, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_lgbm.fit(X_train_scaled_selected, y_train)
best_lgbm = grid_lgbm.best_estimator_
lgbm_cv_rmse = np.sqrt(-grid_lgbm.best_score_)
lgbm_preds = best_lgbm.predict(X_test_scaled_selected[T:])

# Compute weights for ensemble
w_rnn = 1 / rnn_cv_rmse
w_xgb = 1 / xgb_cv_rmse
w_lgbm = 1 / lgbm_cv_rmse
total_w = w_rnn + w_xgb + w_lgbm
w_rnn /= total_w
w_xgb /= total_w
w_lgbm /= total_w

# Weighted ensemble
ensemble_preds = w_rnn * rnn_preds + w_xgb * xgb_preds + w_lgbm * lgbm_preds
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
print(f"Ensemble Weights: RNN={w_rnn:.4f}, XGBoost={w_xgb:.4f}, LightGBM={w_lgbm:.4f}")
print(f"Weighted Ensemble RMSE: {ensemble_rmse:.4f}")
print(f"Weighted Ensemble MAE: {ensemble_mae:.4f}")
print(f"Weighted Ensemble MAPE (clipped): {ensemble_mape:.2f}%")
print(f"Weighted Ensemble R²: {ensemble_r2:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Maximum Error: {max_error:.4f}")

# Save models and preprocessor
best_rnn.save('bestrnn61.h5')
joblib.dump(best_xgb, 'bestxgb61.pkl')
joblib.dump(best_lgbm, 'bestlgbm61.pkl')
joblib.dump(preprocessor, 'preprocessor11.pkl')