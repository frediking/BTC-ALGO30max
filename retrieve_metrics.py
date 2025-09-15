# retrieve_metrics.py

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
import logging

# Suppress LightGBM logs about no further splits and info messages
warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# 1. Load prepared data
X = pd.read_csv('X_prepared.csv').values
y = pd.read_csv('y_prepared.csv').values.flatten()

# 2. Scale & reshape for RNN
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 3. Load models (skip compile for Keras) and silence LightGBM at model level
rnn  = load_model('best_rnn_model.h5', compile=False)
xgb  = joblib.load('best_xgb_model.pkl')
lgbm = joblib.load('best_lgbm_model.pkl')
lgbm.set_params(verbose=-1)

# 4. Point‐estimate RMSE on full set
pred_rnn  = rnn.predict(X_rnn).flatten()
pred_xgb  = xgb.predict(X_scaled)
pred_lgbm = lgbm.predict(X_scaled)

print("=== Train‐set RMSE ===")
for name, pred in zip(['RNN','XGB','LGBM'],
                      [pred_rnn, pred_xgb, pred_lgbm]):
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"{name}: {rmse:.4f}")

ensemble3 = (pred_rnn + pred_xgb + pred_lgbm) / 3
print(f"3‑model Ensemble: {np.sqrt(mean_squared_error(y, ensemble3)):.4f}")

# 5. Quick 5‑fold CV RMSE (re‑fits each model 5 times)
print("\n=== 5‑fold CV RMSE ===")
tscv = TimeSeriesSplit(n_splits=5)
for model, name in [(xgb,'XGB'), (lgbm,'LGBM')]:
    scores = cross_val_score(model, X_scaled, y,
                             cv=tscv,
                             scoring='neg_mean_squared_error',
                             n_jobs=-1)
    cv_rmse = np.mean(np.sqrt(-scores))
    print(f"{name} CV: {cv_rmse:.4f}")

# (Optional) If you want RF or RF‑based metrics on your top‑10 feature set, 
# you can similarly load/run cross_val_score on RandomForestRegressor