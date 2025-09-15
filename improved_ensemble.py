import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
import arch
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import pandas_ta as ta
from scipy.stats.mstats import winsorize
import mlflow
from sklearn.linear_model import LinearRegression

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Start MLflow run
mlflow.start_run()

# Load data
X_df = pd.read_csv('X_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

print(f"Initial data shape: {X_df.shape}")

# Winsorize target to handle outliers
y = winsorize(y, limits=[0.05, 0.05])


# Feature engineering: add Bollinger Bands
bbands_df = ta.bbands(X_df['Close'], length=20, std=2)
X_df['BB_upper'] = bbands_df['BBU_20_2.0']
X_df['BB_middle'] = bbands_df['BBM_20_2.0']
X_df['BB_lower'] = bbands_df['BBL_20_2.0']
X_df['BB_width'] = (X_df['BB_upper'] - X_df['BB_lower']) / X_df['BB_middle']

# # Step 1: Preprocessing - Compute log returns
# X_df['Log_Returns'] = np.log1p(X_df['Close'].pct_change())

# # Step 2: Differencing for stationarity
# X_df['Diff_Close'] = X_df['Close'].diff()

# # Step 3: Volatility feature with GARCH
# vreturns = X_df['Log_Returns'].dropna() * 100  # Scale for GARCH
# garch_model = arch_model(vreturns, vol='Garch', p=1, q=1)
# garch_fit = garch_model.fit(disp='off')
# X_df['GARCH_Volatility'] = np.sqrt(garch_fit.conditional_volatility) / 100  # Rescale

# Add lagged variables and moving averages
for col in X_df.columns:
    for lag in range(1, 4):
        X_df[f'{col}_lag_{lag}'] = X_df[col].shift(lag)
    X_df[f'{col}_ma_5'] = X_df[col].rolling(window=5).mean()

print("Feature engineering completed.")

# Drop rows with NaN values
X_df = X_df.dropna()
y = y[X_df.index]

# Convert to numpy array
X = X_df.values

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Preprocessing pipelines
preprocessor_rnn = make_pipeline(
    KNNImputer(n_neighbors=5),
    RobustScaler(),
    PCA(n_components=0.95)
)
preprocessor_tree = make_pipeline(
    KNNImputer(n_neighbors=5),
    RobustScaler()
)

# Scale data
X_train_scaled_rnn = preprocessor_rnn.fit_transform(X_train)
X_test_scaled_rnn = preprocessor_rnn.transform(X_test)
X_train_scaled_tree = preprocessor_tree.fit_transform(X_train)
X_test_scaled_tree = preprocessor_tree.transform(X_test)

# Create sequences for RNN
T = 5
def create_sequences(data, target, T):
    X_seq, y_seq = [], []
    for i in range(len(data) - T):
        X_seq.append(data[i:i+T, :])
        y_seq.append(target[i+T])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled_rnn, y_train, T)
X_test_seq, y_test_seq = create_sequences(X_test_scaled_rnn, y_test, T)

# RNN model builder
def create_rnn_model(trial, input_shape):
    units = trial.suggest_int('units', 16, 64)
    n_layers = trial.suggest_int('n_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    reg_strength = trial.suggest_loguniform('reg_strength', 1e-4, 1e-2)

    model = Sequential()
    first_layer = True
    for i in range(n_layers):
        if first_layer:
            model.add(LSTM(units, input_shape=input_shape, return_sequences=True if i < n_layers - 1 else False,
                           kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
            first_layer = False
        else:
            model.add(LSTM(units, return_sequences=True if i < n_layers - 1 else False,
                           kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

# Optuna objective for RNN
def objective_rnn(trial):
    split = int(0.8 * len(X_train_seq))
    X_tr, y_tr = X_train_seq[:split], y_train_seq[:split]
    X_val, y_val = X_train_seq[split:], y_train_seq[split:]
    model = create_rnn_model(trial, X_tr.shape[1:])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[es], verbose=0)
    preds = model.predict(X_val, verbose=0).flatten()
    return np.sqrt(mean_squared_error(y_val, preds))

# Tune RNN
study_rnn = optuna.create_study(direction='minimize')
study_rnn.optimize(objective_rnn, n_trials=200)
best_params_rnn = study_rnn.best_trial.params
mlflow.log_params(best_params_rnn)

# Train best RNN
model_rnn = create_rnn_model(study_rnn.best_trial, X_train_seq.shape[1:])
model_rnn.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, callbacks=[EarlyStopping(monitor='loss', patience=10)], verbose=0)
model_rnn.save('rnn10.h5')

# LightGBM Optuna objective
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10)
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    for train_idx, val_idx in tscv.split(X_train_scaled_tree):
        X_tr, X_val = X_train_scaled_tree[train_idx], X_train_scaled_tree[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(rmses)

# Tune LightGBM
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=100)
best_params_lgbm = study_lgbm.best_trial.params
mlflow.log_params(best_params_lgbm)

# Train best LightGBM
model_lgbm = lgb.LGBMRegressor(**best_params_lgbm)
model_lgbm.fit(X_train_scaled_tree, y_train)
joblib.dump(model_lgbm, 'lgbm10.pkl')

# XGBoost GridSearchCV
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [1, 10, 20]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_xgb = GridSearchCV(XGBRegressor(), param_grid_xgb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled_tree, y_train)
model_xgb = grid_xgb.best_estimator_
joblib.dump(model_xgb, 'xgb10.pkl')

# Generate validation predictions
val_size_seq = int(0.2 * len(X_train_seq))
X_val_seq, y_val_seq = X_train_seq[-val_size_seq:], y_train_seq[-val_size_seq:]
rnn_val_preds = model_rnn.predict(X_val_seq, verbose=0).flatten()
lgbm_val_preds = model_lgbm.predict(X_train_scaled_tree[-val_size_seq:])
xgb_val_preds = model_xgb.predict(X_train_scaled_tree[-val_size_seq:])

# Generate test predictions
preds_test_rnn = model_rnn.predict(X_test_seq, verbose=0).flatten()
preds_test_lgbm = model_lgbm.predict(X_test_scaled_tree)
preds_test_xgb = model_xgb.predict(X_test_scaled_tree)

# Align predictions
start_idx = T - 1
preds_test_lgbm_aligned = preds_test_lgbm[start_idx:start_idx + len(preds_test_rnn)]
preds_test_xgb_aligned = preds_test_xgb[start_idx:start_idx + len(preds_test_rnn)]

# Stacking: Train meta-model
val_preds = pd.DataFrame({
    'rnn': rnn_val_preds,
    'lgbm': lgbm_val_preds,
    'xgb': xgb_val_preds
})
meta_model = LinearRegression()
meta_model.fit(val_preds, y_val_seq)

# Stacking: Combine test predictions
test_preds = pd.DataFrame({
    'rnn': preds_test_rnn,
    'lgbm': preds_test_lgbm_aligned,
    'xgb': preds_test_xgb_aligned
})
ensemble_preds = meta_model.predict(test_preds)
y_test_aligned = y_test[start_idx:start_idx + len(preds_test_rnn)]

# Evaluate ensemble
ensemble_rmse = np.sqrt(mean_squared_error(y_test_aligned, ensemble_preds))
ensemble_mae = mean_absolute_error(y_test_aligned, ensemble_preds)
y_test_clipped = np.where(np.abs(y_test_aligned) < 0.01, 0.01, y_test_aligned)
ensemble_mape = np.mean(np.abs((ensemble_preds - y_test_aligned) / y_test_clipped)) * 100
ensemble_r2 = r2_score(y_test_aligned, ensemble_preds)
directional_accuracy = np.mean(np.sign(ensemble_preds) == np.sign(y_test_aligned)) * 100
max_error = np.max(np.abs(ensemble_preds - y_test_aligned))

# Simulate trading strategy for Sharpe ratio
simulated_returns = np.where(ensemble_preds > 0.01, 1, -1) * y_test_aligned
sharpe_ratio = np.mean(simulated_returns) / np.std(simulated_returns)

# Print results
print(f"Weighted Ensemble RMSE: {ensemble_rmse:.4f}")
print(f"Weighted Ensemble MAE: {ensemble_mae:.4f}")
print(f"Weighted Ensemble MAPE (clipped): {ensemble_mape:.2f}%")
print(f"Weighted Ensemble R²: {ensemble_r2:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Maximum Error: {max_error:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# Log metrics to MLflow
mlflow.log_metric("RMSE", ensemble_rmse)
mlflow.log_metric("MAE", ensemble_mae)
mlflow.log_metric("MAPE", ensemble_mape)
mlflow.log_metric("R2", ensemble_r2)
mlflow.log_metric("Directional Accuracy", directional_accuracy)
mlflow.log_metric("Maximum Error", max_error)
mlflow.log_metric("Sharpe Ratio", sharpe_ratio)

# End MLflow run
mlflow.end_run()