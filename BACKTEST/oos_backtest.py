# oos_backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Configuration
FEATURES = [
    'volatility_volume_ratio',
    'stoch_k',
    'stoch_d',
    'rolling_vol_7',
    'rolling_return_7',
    'CCI',
    'ADX_pos',
    'price_range',
    'volatility',
    'Williams_%R'
]
TARGET = 'percentreturn'     # matches y_prepared.csv
TRAIN_RATIO = 0.8            # 80% train, 20% hold‑out

# 2. Load data
df = pd.read_csv('encoded_output.csv', parse_dates=['Date'], index_col='Date')
df = df[FEATURES + [TARGET]].dropna()

# 3. Train/Test split
split = int(len(df) * TRAIN_RATIO)
train, test = df.iloc[:split], df.iloc[split:]

X_train, y_train = train[FEATURES].values, train[TARGET].values
X_test,  y_test  = test[FEATURES].values,  test[TARGET].values

# 4. Scale features on train only
scaler = MinMaxScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# 5. Load optimized models (skip compile to avoid missing metric errors)
rnn = load_model('best_rnn_model.h5', compile=False)
xgb = joblib.load('best_xgb_model.pkl')

# 6. Predict
# 6a. RNN expects 3D [samples, timesteps=1, features]
X_test_rnn = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
pred_rnn = rnn.predict(X_test_rnn).flatten()
pred_xgb = xgb.predict(X_test_s)

# 6b. Ensemble (simple average)
pred_ens = (pred_rnn + pred_xgb) / 2

# 7. Metrics
rmse = np.sqrt(mean_squared_error(y_test, pred_ens))
mae  = mean_absolute_error(y_test, pred_ens)
print(f"Hold‑out RMSE: {rmse:.4f}")
print(f"Hold‑out MAE : {mae:.4f}")

# 8. Backtest PnL (long/short by sign of prediction)
positions = np.where(pred_ens >= 0, 1, -1)
pnl = positions * y_test
cum_pnl = (1 + pnl).cumprod() - 1
cum_pnl = pd.Series(cum_pnl, index=test.index)

# 9. Save & plot
results = pd.DataFrame({
    'actual':        y_test,
    'pred_rnn':      pred_rnn,
    'pred_xgb':      pred_xgb,
    'pred_ensemble': pred_ens,
    'position':      positions,
    'pnl':           pnl
}, index=test.index)
results.to_csv('oos_backtest_results.csv')
cum_pnl.to_frame('cum_pnl').to_csv('oos_cum_pnl.csv')

plt.figure(figsize=(10, 5))
cum_pnl.plot(title='OOS Cumulative P&L')
plt.ylabel('Cumulative Return')
plt.tight_layout()
plt.savefig('oos_cum_pnl.png')
plt.show()