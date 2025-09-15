import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sp500_features import merge_sp500_features

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load data
X_df = pd.read_csv('X_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# Feature engineering: lagged variables and moving averages
for col in X_df.columns:
    for lag in range(1, 4):
        X_df[f'{col}_lag_{lag}'] = X_df[col].shift(lag)
    X_df[f'{col}_ma_5'] = X_df[col].rolling(window=5).mean()

#----------------------------------------------------------------------------------
# # Add S&P 500 features
# print("\nAdding S&P 500 features...")
# X_df = merge_sp500_features(X_df)
    
#     # Verify new features
# print("\nNew feature correlations with target:")
# correlations = X_df[['sp500_return', 'sp500_volatility', 'sp500_trend', 'sp500_rsi']].corrwith(y['percentreturn'])
# print(correlations)
#----------------------------------------------------------------------------------

# Check for duplicate rows
if X_df.duplicated().any():
    print("Duplicate rows found in features. Dropping duplicates...")
    X_df.drop_duplicates(inplace=True)
# Check for outliers
if (X_df < -1e6).any().any() or (X_df > 1e6).any().any():
    print("Outliers found in features. Clipping values...")
    X_df = np.clip(X_df, -1e6, 1e6)

# # Check for multicollinearity
# correlation_matrix = X_df.corr()
# highly_correlated_features = set()
# threshold = 0.9
# for i in range(len(correlation_matrix.columns)):
#     for j in range(i):
#         if abs(correlation_matrix.iloc[i, j]) > threshold:
#             colname = correlation_matrix.columns[i]
#             highly_correlated_features.add(colname)
#             print(f"High correlation found: {colname} with {correlation_matrix.columns[j]}")
#             X_df.drop(colname, axis=1, inplace=True)

# Drop rows with NaN values and align y
X_df = X_df.dropna()
y = y[X_df.index]
X = X_df.values

# Split into train and test sets (80% train, 20% test, time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Preprocessing for RNN
preprocessor_rnn = make_pipeline(
    KNNImputer(n_neighbors=5),
    RobustScaler(),
    PCA(n_components=0.95)
)
X_train_scaled_rnn = preprocessor_rnn.fit_transform(X_train)
X_test_scaled_rnn = preprocessor_rnn.transform(X_test)

# Preprocessing for tree-based models
preprocessor_tree = make_pipeline(
    KNNImputer(n_neighbors=5),
    RobustScaler()
)
X_train_scaled_tree = preprocessor_tree.fit_transform(X_train)
X_test_scaled_tree = preprocessor_tree.transform(X_test)

# Create sequences for RNN
T = 5


def create_sequences(data, target, T):
    X_seq, y_seq = [], []
    for i in range(len(data) - T + 1):
        X_seq.append(data[i:i+T, :])
        y_seq.append(target[i+T-1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled_rnn, y_train, T)
X_test_seq, y_test_seq = create_sequences(X_test_scaled_rnn, y_test, T)

# RNN model builder
def create_rnn_model(trial, input_shape):
    model_type = trial.suggest_categorical('model_type', ['LSTM', 'BidirectionalLSTM'])
    units = trial.suggest_categorical('units', [32, 64])
    n_layers = trial.suggest_int('n_layers', 1, 2)
    dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-3)
    reg_strength = trial.suggest_loguniform('reg_strength', 1e-4, 1e-2)

    model = Sequential()
    first_layer = True
    for i in range(n_layers):
        if model_type == 'LSTM':
            if first_layer:
                model.add(LSTM(units, input_shape=input_shape, return_sequences=True if i < n_layers - 1 else False,
                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
                first_layer = False
            else:
                model.add(LSTM(units, return_sequences=True if i < n_layers - 1 else False,
                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
        else:
            if first_layer:
                model.add(Bidirectional(LSTM(units, input_shape=input_shape, return_sequences=True if i < n_layers - 1 else False,
                                             kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
                first_layer = False
            else:
                model.add(Bidirectional(LSTM(units, return_sequences=True if i < n_layers - 1 else False,
                                             kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['RootMeanSquaredError'])
    return model

# RNN Optuna objective
def objective_rnn(trial):
    split = int(0.8 * len(X_train_seq))
    X_tr, y_tr = X_train_seq[:split], y_train_seq[:split]
    X_val, y_val = X_train_seq[split:], y_train_seq[split:]
    model = create_rnn_model(trial, X_tr.shape[1:])
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=batch_size, callbacks=[es], verbose=0)
    preds = model.predict(X_val, verbose=0).flatten()
    return np.sqrt(mean_squared_error(y_val, preds))

# Train RNN
study_rnn = optuna.create_study(direction='minimize')
study_rnn.optimize(objective_rnn, n_trials=200)
best_params_rnn = study_rnn.best_trial.params
model_rnn = create_rnn_model(study_rnn.best_trial, X_train_seq.shape[1:])
model_rnn.fit(X_train_seq, y_train_seq, epochs=50, batch_size=best_params_rnn['batch_size'],
              callbacks=[EarlyStopping(monitor='loss', patience=10)], verbose=0)
preds_test_rnn = model_rnn.predict(X_test_seq, verbose=0).flatten()
model_rnn.save('rnn4.h5')

# LightGBM Optuna objective
def objective_lgbm(trial):
    params = {
        'objective': 'huber',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 5e-4, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
        'random_state': 42
    }
    learning_rate_penalty = params['learning_rate'] * 100
    tscv = TimeSeriesSplit(n_splits=3)
    rmses = []
    for train_idx, val_idx in tscv.split(X_train_scaled_tree):
        X_tr, X_val = X_train_scaled_tree[train_idx], X_train_scaled_tree[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(params, train_data, valid_sets=[valid_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)])
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(rmses) + learning_rate_penalty

# Train LightGBM
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=150)
best_params_lgbm = study_lgbm.best_trial.params
val_size = int(0.1 * len(X_train_scaled_tree))
X_tr_final, y_tr_final = X_train_scaled_tree[:-val_size], y_train[:-val_size]
X_val_final, y_val_final = X_train_scaled_tree[-val_size:], y_train[-val_size:]
train_data_initial = lgb.Dataset(X_tr_final, label=y_tr_final)
valid_data_initial = lgb.Dataset(X_val_final, label=y_val_final, reference=train_data_initial)
initial_model_lgbm = lgb.train(best_params_lgbm, train_data_initial, valid_sets=[valid_data_initial],
                               callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)])
importances = initial_model_lgbm.feature_importance(importance_type='gain')
top_features = np.argsort(importances)[::-1][:int(0.7 * len(importances))]
X_tr_final_selected = X_tr_final[:, top_features]
X_val_final_selected = X_val_final[:, top_features]
X_test_selected_lgbm = X_test_scaled_tree[:, top_features]
train_data_final = lgb.Dataset(X_tr_final_selected, label=y_tr_final)
valid_data_final = lgb.Dataset(X_val_final_selected, label=y_val_final, reference=train_data_final)
model_lgbm = lgb.train(best_params_lgbm, train_data_final, valid_sets=[valid_data_final],
                       callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)])
preds_test_lgbm = model_lgbm.predict(X_test_selected_lgbm, num_iteration=model_lgbm.best_iteration)
joblib.dump(model_lgbm, 'lgbm4.pkl')

# XGBoost training
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.6, 0.8],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 10]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train_scaled_tree, y_train)
model_xgb = grid_xgb.best_estimator_
preds_test_xgb = model_xgb.predict(X_test_scaled_tree)
joblib.dump(model_xgb, 'xgb4.pkl')

# Ensemble calculation
X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled_tree, y_train, test_size=0.2, shuffle=False)
val_size_seq = int(0.2 * len(X_train_seq))
X_val_seq, y_val_seq = X_train_seq[-val_size_seq:], y_train_seq[-val_size_seq:]
rnn_val_preds = model_rnn.predict(X_val_seq, verbose=0).flatten()
X_val_selected_lgbm = X_val[:, top_features]
lgbm_val_preds = model_lgbm.predict(X_val_selected_lgbm, num_iteration=model_lgbm.best_iteration)
xgb_val_preds = model_xgb.predict(X_val)

# Compute weights
rnn_rmse_val = np.sqrt(mean_squared_error(y_val_seq, rnn_val_preds))
lgbm_rmse_val = np.sqrt(mean_squared_error(y_val, lgbm_val_preds))
xgb_rmse_val = np.sqrt(mean_squared_error(y_val, xgb_val_preds))
total_rmse = rnn_rmse_val + lgbm_rmse_val + xgb_rmse_val
w_rnn = (1 / rnn_rmse_val) / (1 / rnn_rmse_val + 1 / lgbm_rmse_val + 1 / xgb_rmse_val)
w_lgbm = (1 / lgbm_rmse_val) / (1 / rnn_rmse_val + 1 / lgbm_rmse_val + 1 / xgb_rmse_val)
w_xgb = (1 / xgb_rmse_val) / (1 / rnn_rmse_val + 1 / lgbm_rmse_val + 1 / xgb_rmse_val)

# Generate test predictions
preds_test_rnn = model_rnn.predict(X_test_seq, verbose=0).flatten()  # Shape: (772,)
preds_test_lgbm = model_lgbm.predict(X_test_selected_lgbm)  # Shape: (776,)
preds_test_xgb = model_xgb.predict(X_test_scaled_tree)  # Shape: (776,)

# Align predictions to the RNN's range (this is where the modification begins)
start_idx = T - 1  # For T=5, start_idx=4
preds_test_lgbm_aligned = preds_test_lgbm[start_idx:]  # Shape: (772,)
preds_test_xgb_aligned = preds_test_xgb[start_idx:]    # Shape: (772,)


# Ensemble predictions
ensemble_preds = w_rnn * preds_test_rnn + w_lgbm * preds_test_lgbm_aligned + w_xgb * preds_test_xgb_aligned

# Align true target values
y_test_aligned =  y_test[start_idx:]

# Metrics
# Compute metrics with aligned true values
ensemble_rmse = np.sqrt(mean_squared_error(y_test_aligned, ensemble_preds))
ensemble_mae = mean_absolute_error(y_test_aligned, ensemble_preds)
y_test_clipped = np.where(np.abs(y_test_aligned) < 1e-6, 1e-6, y_test_aligned)
ensemble_mape = np.mean(np.abs((ensemble_preds - y_test_aligned) / y_test_clipped)) * 100
ensemble_r2 = r2_score(y_test_aligned, ensemble_preds)
directional_accuracy = np.mean(np.sign(ensemble_preds) == np.sign(y_test_aligned)) * 100
max_error = np.max(np.abs(ensemble_preds - y_test_aligned))

# Output results

print(f"Weighted Ensemble RMSE: {ensemble_rmse:.4f}")
print(f"Weighted Ensemble MAE: {ensemble_mae:.4f}")
print(f"Weighted Ensemble MAPE (clipped): {ensemble_mape:.2f}%")
print(f"Weighted Ensemble R²: {ensemble_r2:.4f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Maximum Error: {max_error:.4f}")