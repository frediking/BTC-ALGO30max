

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib

# Load your prepared data
X_df = pd.read_csv('Xl_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# Feature engineering: add lagged variables and moving average for all columns
# for col in X_df.columns:
#     for lag in range(1, 4):
#         X_df[f'{col}_lag_{lag}'] = X_df[col].shift(lag)
#     X_df[f'{col}_ma_5'] = X_df[col].rolling(window=5).mean()

# Drop rows with NaN values due to lagging and moving average
X_df = X_df.dropna()
y = y[X_df.index]  # Align y with the reduced X_df

# Convert X_df to numpy array
X = X_df.values

# Split into train and test sets (80% train, 20% test, no shuffling for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Preprocessing pipeline: impute missing values and scale features
preprocessor = make_pipeline(
    KNNImputer(n_neighbors=5),
    RobustScaler()
)
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Define Optuna objective function with early stopping
def objective(trial):
    params = {
        'objective': 'huber',  # Use Huber loss for robustness to outliers
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

    # Penalize large learning rates to favor smaller ones
    learning_rate_penalty = params['learning_rate'] * 100
    tscv = TimeSeriesSplit(n_splits=3)
    rmses = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)]
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)
    return np.mean(rmses) + learning_rate_penalty

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Output best hyperparameters and cross-validation RMSE
best_params = study.best_trial.params
print("Best Hyperparameters:", best_params)
print("Cross-Validation RMSE:", study.best_value)

# Split training data for final model training with early stopping
val_size = int(0.1 * len(X_train_scaled))
X_tr_final = X_train_scaled[:-val_size]
y_tr_final = y_train[:-val_size]
X_val_final = X_train_scaled[-val_size:]
y_val_final = y_train[-val_size:]

# Train an initial model to get feature importance
train_data_initial = lgb.Dataset(X_tr_final, label=y_tr_final)
valid_data_initial = lgb.Dataset(X_val_final, label=y_val_final, reference=train_data_initial)
initial_model = lgb.train(
    best_params,
    train_data_initial,
    valid_sets=[valid_data_initial],
    callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)]
)

# Select top 70% of features based on importance
importances = initial_model.feature_importance(importance_type='gain')
feature_indices = np.argsort(importances)[::-1]
top_features = feature_indices[:int(0.7 * len(feature_indices))]
X_tr_final_selected = X_tr_final[:, top_features]
X_val_final_selected = X_val_final[:, top_features]
X_test_selected = X_test_scaled[:, top_features]

# Print feature importances for analysis
feature_names = X_df.columns
print("Feature Importances (Top Features):")
for idx, imp in zip(top_features, importances[top_features]):
    print(f"{feature_names[idx]}: {imp}")

# Train the best model with early stopping on selected features
train_data_final = lgb.Dataset(X_tr_final_selected, label=y_tr_final)
valid_data_final = lgb.Dataset(X_val_final_selected, label=y_val_final, reference=train_data_final)
best_model = lgb.train(
    best_params,
    train_data_final,
    valid_sets=[valid_data_final],
    callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)]
)
print(f"Best iteration: {best_model.best_iteration}")

# Evaluate on test set with multiple metrics
preds_test = best_model.predict(X_test_selected, num_iteration=best_model.best_iteration)
test_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
test_mae = mean_absolute_error(y_test, preds_test)
# Clip y_test to avoid division by near-zero for MAPE
y_test_clipped = np.where(np.abs(y_test) < 1e-6, 1e-6, y_test)
test_mape = np.mean(np.abs((preds_test - y_test) / y_test_clipped)) * 100
test_r2 = r2_score(y_test, preds_test)
directional_accuracy = np.mean(np.sign(preds_test) == np.sign(y_test)) * 100
max_error = np.max(np.abs(preds_test - y_test))

print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")
print(f"Test MAPE (clipped): {test_mape:.2f}%")
print(f"Test R²: {test_r2}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Maximum Error: {max_error}")

# Save the optimized model and preprocessor
joblib.dump(best_model, 'lightmodel2110.pkl')
joblib.dump(preprocessor, 'preprocessorl.pkl')









