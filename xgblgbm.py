


# 2025-05-18 08:48:26,426 INFO Single Model Test RMSE: 0.5182885684103425


import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import optuna
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Custom Outlier Clipper
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds = None
        self.upper_bounds = None
    
    def fit(self, X, y=None):
        self.lower_bounds = X.quantile(self.lower_quantile)
        self.upper_bounds = X.quantile(self.upper_quantile)
        return self
    
    def transform(self, X):
        X_clipped = X.copy()
        for col in X.columns:
            X_clipped[col] = np.clip(X_clipped[col], self.lower_bounds[col], self.upper_bounds[col])
        return X_clipped

# Load data
logging.info("Loading datasets")
X_df = pd.read_csv('Xg_prepared.csv')
y_df = pd.read_csv('y1_prepared.csv')

# Ensure data alignment
logging.info("Aligning datasets")
X_df['index'] = X_df.index
y_df['index'] = y_df.index
data = X_df.merge(y_df[['index', 'percentreturn']], on='index', how='inner')
X_df = data.drop(columns=['index', 'percentreturn'])
y = data['percentreturn'].values

# Define features
features = ['stoch_k', 'rolling_vol_7', 'Momentum_10', 'ATR', 'CCI', 
            'price_change', 'volatility', 'RSI', 'RSI_Divergence', 'Force_Index']
X_df = X_df[features]

# Split into train and test sets (80% train, 20% test, no shuffling)
train_size = int(0.8 * len(X_df))
X_train, X_test = X_df.iloc[:train_size], X_df.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define preprocessor
logging.info("Setting up preprocessor")
preprocessor = make_pipeline(
    OutlierClipper(lower_quantile=0.01, upper_quantile=0.99),
    StandardScaler()
)

# Apply preprocessor
logging.info("Preprocessing data")
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Set up TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Define Optuna objective
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    
    rmses = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Convert to DMatrix for xgboost.train
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping
        evals = [(dval, 'eval')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=trial.suggest_int('n_estimators', 100, 2000),
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)
    
    return np.mean(rmses)

# Run Optuna optimization
logging.info("Starting Optuna optimization")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=200)

# Get best parameters
best_params = study.best_params
logging.info(f"Best Hyperparameters: {best_params}")
logging.info(f"Best CV RMSE: {study.best_value}")

# Train final model with best parameters
logging.info("Training final model")
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)
evals = [(dtest, 'test')]
best_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_params['n_estimators'],
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=10
)

# Predict and evaluate single model
y_pred = best_model.predict(dtest)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
logging.info(f"Single Model Test RMSE: {test_rmse}")

# Ensemble with multiple seeds
logging.info("Training ensemble models")
seeds = [42, 123, 456]
predictions = []
for seed in seeds:
    params_with_seed = best_params.copy()
    params_with_seed['seed'] = seed
    model = xgb.train(
        params_with_seed,
        dtrain,
        num_boost_round=best_params['n_estimators'],
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    preds = model.predict(dtest)
    predictions.append(preds)

# Average predictions
ensemble_preds = np.mean(predictions, axis=0)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
logging.info(f"Ensemble Test RMSE: {ensemble_rmse}")

# Save the best model and preprocessor
logging.info("Saving model and preprocessor")
joblib.dump(best_model, 'xghost1.pkl')
joblib.dump(preprocessor, 'preprocessorx.pkl')

