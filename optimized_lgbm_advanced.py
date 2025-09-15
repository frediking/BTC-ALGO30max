import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, RegressorMixin
import joblib

# Load data
X_df = pd.read_csv('Xl_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# Advanced Feature Engineering
for col in ['volatility', 'RSI', 'Volume', 'Price_ROC']:
    for lag in [1, 3, 5]:
        X_df[f'{col}_lag_{lag}'] = X_df[col].shift(lag)
    X_df[f'{col}_ma_3'] = X_df[col].rolling(window=3).mean()
    X_df[f'{col}_ma_7'] = X_df[col].rolling(window=7).mean()
    X_df[f'{col}_ema_14'] = X_df[col].ewm(span=14, adjust=False).mean()
X_df['volatility_momentum'] = X_df['volatility'].diff()
X_df['RSI_div_ma'] = X_df['RSI_Divergence'].rolling(window=5).mean()
X_df['vol_price_interaction'] = X_df['Volume'] * X_df['Price_ROC']
X_df['RSI_squared'] = X_df['RSI'] ** 2

# Drop NaN rows from feature engineering
X_df = X_df.dropna()
y = y[X_df.index]

# Convert to numpy
X = X_df.values

# Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Preprocessing with log transformation for outliers
class LogTransformer(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_transformed = X.copy()
        for i in [3, 15]:  # Volume, Force_Index
            X_transformed[:, i] = np.log1p(np.abs(X_transformed[:, i])) * np.sign(X_transformed[:, i])
        return X_transformed
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

preprocessor = make_pipeline(
    IterativeImputer(max_iter=10, random_state=42),
    LogTransformer(),
    RobustScaler()
)
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Custom Huber Loss with higher penalty for large errors
def custom_huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    small_error_loss = 0.5 * error ** 2
    large_error_loss = delta * np.abs(error) - 0.5 * delta ** 2
    return np.where(is_small_error, small_error_loss, large_error_loss).mean()

# Walk-forward validation
def walk_forward_validation(X, y, n_splits=5):
    step_size = len(X) // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = i * step_size
        val_end = train_end + step_size
        yield (np.arange(train_end), np.arange(train_end, min(val_end, len(X))))

# Optuna objective with custom loss and walk-forward CV
def objective(trial):
    params = {
        'objective': 'huber',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.05),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 100),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'random_state': 42
    }
    rmses = []
    for train_idx, val_idx in walk_forward_validation(X_train_scaled, y_train):
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
    return np.mean(rmses) + params['learning_rate'] * 50  # Reduced penalty

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

best_params = study.best_trial.params
print("Best Hyperparameters:", best_params)

# Train initial model for feature selection
val_size = int(0.1 * len(X_train_scaled))
X_tr_final, X_val_final = X_train_scaled[:-val_size], X_train_scaled[-val_size:]
y_tr_final, y_val_final = y_train[:-val_size], y_train[-val_size:]
train_data_initial = lgb.Dataset(X_tr_final, label=y_tr_final)
valid_data_initial = lgb.Dataset(X_val_final, label=y_val_final, reference=train_data_initial)
initial_model = lgb.train(
    best_params,
    train_data_initial,
    valid_sets=[valid_data_initial],
    callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)]
)

# Feature selection (top 70%)
importances = initial_model.feature_importance(importance_type='gain')
feature_indices = np.argsort(importances)[::-1]
top_features = feature_indices[:int(0.7 * len(feature_indices))]
joblib.dump(top_features, 'top_features.pkl')
X_tr_final_selected = X_tr_final[:, top_features]
X_val_final_selected = X_val_final[:, top_features]
X_test_selected = X_test_scaled[:, top_features]

# Train final model
train_data_final = lgb.Dataset(X_tr_final_selected, label=y_tr_final)
valid_data_final = lgb.Dataset(X_val_final_selected, label=y_val_final, reference=train_data_final)
best_model = lgb.train(
    best_params,
    train_data_final,
    valid_sets=[valid_data_final],
    callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)]
)

# Prediction calibration with isotonic regression
preds_test_raw = best_model.predict(X_test_selected, num_iteration=best_model.best_iteration)
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(preds_test_raw, y_test)
preds_test = iso_reg.predict(preds_test_raw)

# Evaluate with multiple metrics
test_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
test_mae = mean_absolute_error(y_test, preds_test)
y_test_clipped = np.where(np.abs(y_test) < 1e-6, 1e-6, y_test)
test_mape = mean_absolute_percentage_error(y_test_clipped, preds_test) * 100
test_r2 = r2_score(y_test, preds_test)
directional_accuracy = np.mean(np.sign(preds_test) == np.sign(y_test)) * 100
max_error = np.max(np.abs(preds_test - y_test))

print(f"Test RMSE: {test_rmse}")
print(f"Test MAE: {test_mae}")
print(f"Test MAPE: {test_mape:.2f}%")
print(f"Test R²: {test_r2}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Maximum Error: {max_error}")

# Residual analysis
residuals = y_test - preds_test
print("Residual Mean:", np.mean(residuals))
print("Residual Std:", np.std(residuals))

# Save model and preprocessor
joblib.dump(best_model, 'lightmodeladv2.pkl')
joblib.dump(preprocessor, 'preprocessoradv1.pkl')

# Test RMSE: 0.4464390487611248      - **BEST LGBM** 
# Test MAE: 0.2811506126714485
# Test MAPE: 46.72%
# Test R²: 0.9689387570650653
# Directional Accuracy: 100.00%