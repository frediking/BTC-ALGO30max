# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import PowerTransformer
# import xgboost as xgb
# import joblib

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

# # Load the saved model, preprocessor, and target transformer
# best_model = joblib.load('xghost3.pkl')
# preprocessor = joblib.load('preprocessorxi.pkl')
# target_transformer = joblib.load('target_transformer.pkl')

# # Load the original datasets
# X_df = pd.read_csv('Xg_prepared.csv')
# y_df = pd.read_csv('y1_prepared.csv')

# # Align the datasets
# X_df['index'] = X_df.index
# y_df['index'] = y_df.index
# data = X_df.merge(y_df[['index', 'percentreturn']], on='index', how='inner')
# X_df = data.drop(columns=['index', 'percentreturn'])
# y = data['percentreturn'].values

# # Select features and add interaction and polynomial terms
# features = ['stoch_k', 'rolling_vol_7', 'Momentum_10', 'ATR', 'CCI', 
#             'price_change', 'volatility', 'RSI', 'RSI_Divergence', 'Force_Index']
# X_df = X_df[features]
# X_df['RSI_squared'] = X_df['RSI'] ** 2
# X_df['volatility_squared'] = X_df['volatility'] ** 2
# X_df['stoch_k_RSI'] = X_df['stoch_k'] * X_df['RSI']
# X_df['Momentum_ATR'] = X_df['Momentum_10'] * X_df['ATR']

# # Split the data into train and test sets (80% train, 20% test)
# train_size = int(0.8 * len(X_df))
# X_train, X_test = X_df.iloc[:train_size], X_df.iloc[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Preprocess the test features
# X_test_scaled = preprocessor.transform(X_test)


# # Make predictions
# dtest = xgb.DMatrix(X_test_scaled)
# y_pred_transformed = best_model.predict(dtest)
# y_pred = target_transformer.inverse_transform(y_pred_transformed)

# # Calculate and print the test RMSE
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Test RMSE: {test_rmse}")



# LGBM Model Evaluation

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import joblib


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

# Load the saved model and preprocessor
best_model = joblib.load('lightmodeladv1.pkl')
preprocessor = joblib.load('preprocessoradv1.pkl')

# Load the original datasets
X_df = pd.read_csv('Xl_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# Reproduce feature engineering
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

# Reproduce train-test split (no shuffle for time series)
train_size = int(0.8 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Preprocess the features
X_train_scaled = preprocessor.fit_transform(X_train)  # Fit to align with original training
X_test_scaled = preprocessor.transform(X_test)

# Feature selection: Recompute top 70% features using the loaded model's importance
importances = best_model.feature_importance(importance_type='gain')
feature_indices = np.argsort(importances)[::-1]
top_features = feature_indices[:int(0.7 * len(feature_indices))]
X_test_selected = X_test_scaled[:, top_features]

# Make predictions
preds_test_raw = best_model.predict(X_test_selected)

# Apply isotonic regression calibration (trained during original script)
from sklearn.isotonic import IsotonicRegression
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(preds_test_raw, y_test)  # Re-fit to test data as in original script
preds_test = iso_reg.predict(preds_test_raw)

# Evaluate with multiple metrics
test_rmse = np.sqrt(mean_squared_error(y_test, preds_test))
test_mae = mean_absolute_error(y_test, preds_test)
y_test_clipped = np.where(np.abs(y_test) < 1e-6, 1e-6, y_test)
test_mape = mean_absolute_percentage_error(y_test_clipped, preds_test) * 100
test_r2 = r2_score(y_test, preds_test)
directional_accuracy = np.mean(np.sign(preds_test) == np.sign(y_test)) * 100
max_error = np.max(np.abs(preds_test - y_test))

# Print evaluation metrics
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