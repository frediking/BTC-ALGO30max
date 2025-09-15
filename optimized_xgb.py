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
    def __init__(self, lower_quantile=0.005, upper_quantile=0.995):
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

# Target Transformer
class TargetTransformer(BaseEstimator, TransformerMixin):
    def fit(self, y):
        return self
    
    def transform(self, y):
        return np.log1p(np.abs(y)) * np.sign(y)  # Handle negative returns
    
    def inverse_transform(self, y):
        return np.sign(y) * (np.expm1(np.abs(y)))

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

# Transform target
logging.info("Transforming target")
target_transformer = TargetTransformer()
y_transformed = target_transformer.transform(y)

# Define features
features = ['stoch_k', 'rolling_vol_7', 'Momentum_10', 'ATR', 'CCI', 
            'price_change', 'volatility', 'RSI', 'RSI_Divergence', 'Force_Index']
X_df = X_df[features]

# Add interaction terms
X_df['stoch_k_RSI'] = X_df['stoch_k'] * X_df['RSI']
X_df['Momentum_ATR'] = X_df['Momentum_10'] * X_df['ATR']

# Split into train and test sets (80% train, 20% test, no shuffling)
train_size = int(0.8 * len(X_df))
X_train, X_test = X_df.iloc[:train_size], X_df.iloc[train_size:]
y_train, y_test = y_transformed[:train_size], y_transformed[train_size:]

# Define preprocessor
logging.info("Setting up preprocessor")
preprocessor = make_pipeline(
    OutlierClipper(lower_quantile=0.005, upper_quantile=0.995),
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
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }
    
    rmses = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dval, 'eval')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=trial.suggest_int('n_estimators', 100, 5000),
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        preds = target_transformer.inverse_transform(preds)
        y_val_orig = target_transformer.inverse_transform(y_val)
        rmse = np.sqrt(mean_squared_error(y_val_orig, preds))
        rmses.append(rmse)
    
    return np.mean(rmses)

# Run Optuna optimization
logging.info("Starting Optuna optimization")
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=500)

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

# Predict and evaluate
y_pred = best_model.predict(dtest)
y_pred = target_transformer.inverse_transform(y_pred)
y_test_orig = target_transformer.inverse_transform(y_test)
test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
logging.info(f"Test RMSE: {test_rmse}")

# Save the best model and preprocessor
logging.info("Saving model and preprocessor")
joblib.dump(best_model, 'xghost22.pkl')
joblib.dump(preprocessor, 'preprocessorx.pkl')
joblib.dump(target_transformer, 'target_transformer.pkl')


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import RobustScaler, PowerTransformer
# from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# import optuna
# import joblib
# import logging
# import matplotlib.pyplot as plt
# logging.info(f"XGBoost version: {xgb.__version__}")

# # Set random seed for reproducibility
# np.random.seed(42)

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# # Define TargetTransformer using Yeo-Johnson
# class TargetTransformer:
#     def __init__(self):
#         self.pt = PowerTransformer(method='yeo-johnson')
    
#     def fit(self, y):
#         self.pt.fit(y.reshape(-1, 1))
#         return self
    
#     def transform(self, y):
#         return self.pt.transform(y.reshape(-1, 1)).flatten()
    
#     def inverse_transform(self, y):
#         return self.pt.inverse_transform(y.reshape(-1, 1)).flatten()

# # Load and align data
# logging.info("Loading datasets")
# X_df = pd.read_csv('Xg_prepared.csv')
# y_df = pd.read_csv('y1_prepared.csv')

# logging.info("Aligning datasets")
# X_df['index'] = X_df.index
# y_df['index'] = y_df.index
# data = X_df.merge(y_df[['index', 'percentreturn']], on='index', how='inner')
# X_df = data.drop(columns=['index', 'percentreturn'])
# y = data['percentreturn'].values

# # Select features and add polynomial terms
# features = ['stoch_k', 'rolling_vol_7', 'Momentum_10', 'ATR', 'CCI', 
#             'price_change', 'volatility', 'RSI', 'RSI_Divergence', 'Force_Index']
# X_df = X_df[features]
# X_df['RSI_squared'] = X_df['RSI'] ** 2
# X_df['volatility_squared'] = X_df['volatility'] ** 2
# X_df['stoch_k_RSI'] = X_df['stoch_k'] * X_df['RSI']
# X_df['Momentum_ATR'] = X_df['Momentum_10'] * X_df['ATR']

# logging.info("Correlation Analysis")
# print(X_df[['stoch_k', 'RSI', 'stoch_k_RSI']].corr())
# print(X_df[['Momentum_10', 'ATR', 'Momentum_ATR']].corr())

# # Split data into train and test sets
# train_size = int(0.8 * len(X_df))
# X_train, X_test = X_df.iloc[:train_size], X_df.iloc[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Transform target variable (fit on train only)
# logging.info("Transforming target")
# target_transformer = TargetTransformer()
# target_transformer.fit(y_train)
# y_train_transformed = target_transformer.transform(y_train)
# y_test_transformed = target_transformer.transform(y_test)

# # Preprocess features with RobustScaler
# logging.info("Setting up preprocessor")
# preprocessor = RobustScaler()
# logging.info("Preprocessing data")
# X_train_scaled = preprocessor.fit_transform(X_train)
# X_test_scaled = preprocessor.transform(X_test)

# # Set up TimeSeriesSplit for cross-validation
# tscv = TimeSeriesSplit(n_splits=5)

# # Define Optuna objective function
# def objective(trial):
#     params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
#     }
    
#     rmses = []
#     for train_idx, val_idx in tscv.split(X_train_scaled):
#         X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
#         y_tr, y_val = y_train_transformed[train_idx], y_train[val_idx]
#         y_val_orig = y_train[val_idx]
        
#         dtrain = xgb.DMatrix(X_tr, label=y_tr)
#         dval = xgb.DMatrix(X_val, label=y_val)
        
#         evals = [(dval, 'eval')]
#         model = xgb.train(
#             params,
#             dtrain,
#             num_boost_round=trial.suggest_int('n_estimators', 100, 10000),
#             evals=evals,
#             early_stopping_rounds=100,
#             verbose_eval=False
#         )
        
#         preds = model.predict(dval)
#         preds = target_transformer.inverse_transform(preds)
#         rmse = np.sqrt(mean_squared_error(y_val_orig, preds))
#         rmses.append(rmse)
    
#     return np.mean(rmses)

# # Run Optuna optimization
# logging.info("Starting Optuna optimization")
# study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
# study.optimize(objective, n_trials=500)

# # Retrieve best parameters
# best_params = study.best_params
# logging.info(f"Best Hyperparameters: {best_params}")
# logging.info(f"Best CV RMSE: {study.best_value}")

# # Train final model
# logging.info("Training final model")
# dtrain = xgb.DMatrix(X_train_scaled, label=y_train_transformed)
# dtest = xgb.DMatrix(X_test_scaled, label=y_test_transformed)
# evals = [(dtest, 'test')]
# best_model = xgb.train(
#     best_params,
#     dtrain,
#     num_boost_round=best_params['n_estimators'],
#     evals=evals,
#     early_stopping_rounds=100,
#     verbose_eval=10
# )

# # Predict and evaluate on test set
# y_pred_transformed = best_model.predict(dtest)
# y_pred = target_transformer.inverse_transform(y_pred_transformed)
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# logging.info(f"Test RMSE: {test_rmse}")

# # Feature importance
# logging.info("Computing feature importance")
# importance = best_model.get_score(importance_type='gain')
# feature_names = X_train.columns
# importance = {feature_names[int(k.replace('f', ''))]: v for k, v in importance.items()}
# logging.info(f"Feature Importance: {importance}")



# # Save model and preprocessors
# logging.info("Saving model, preprocessor, and target transformer")
# joblib.dump(best_model, 'xghost3.pkl')
# joblib.dump(preprocessor, 'preprocessorxi.pkl')
# joblib.dump(target_transformer, 'target_transformer.pkl')

# # Plot residuals
# logging.info("Plotting residuals")
# residuals = y_test - y_pred
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.scatter(y_pred, residuals, alpha=0.5)
# plt.axhline(0, color='red', linestyle='--')
# plt.title('Residuals vs Predicted')
# plt.xlabel('Predicted')
# plt.ylabel('Residuals')

# plt.subplot(1, 2, 2)
# plt.plot(residuals, alpha=0.5)
# plt.axhline(0, color='red', linestyle='--')
# plt.title('Residuals over Time')
# plt.xlabel('Time Index')
# plt.ylabel('Residuals')
# plt.tight_layout()
# plt.show()
