import os
os.environ["TF_MPS_ENABLED"] = "0"

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load data
X = pd.read_csv('X_prepared.csv').values
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values.flatten()

# 2. Train/hold‑out split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# 3. Scale + reshape for RNN
scaler = MinMaxScaler().fit(X_train)
X_tr_s = scaler.transform(X_train)
X_te_s = scaler.transform(X_test)
X_tr_rnn = X_tr_s.reshape((X_tr_s.shape[0], 1, X_tr_s.shape[1]))
X_te_rnn = X_te_s.reshape((X_te_s.shape[0], 1, X_te_s.shape[1]))

# 4. Load tuned models
rnn  = load_model('rnnmod_enhanced3.h5', compile=False)
xgb  = joblib.load('xghost1.pkl')
lgbm = joblib.load('lightmodeladv1.pkl')

# 5. Level‑1 predictions
pred_rnn_tr  = rnn.predict(X_tr_rnn).flatten()
pred_xgb_tr  = xgb.predict(X_tr_s)
pred_lgbm_tr = lgbm.predict(X_tr_s)
stack_tr = np.vstack([pred_rnn_tr, pred_xgb_tr, pred_lgbm_tr]).T

pred_rnn_te  = rnn.predict(X_te_rnn).flatten()
pred_xgb_te  = xgb.predict(X_te_s)
pred_lgbm_te = lgbm.predict(X_te_s)
stack_te = np.vstack([pred_rnn_te, pred_xgb_te, pred_lgbm_te]).T

# 6. Meta‑learner (RidgeCV)
meta = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
meta.fit(stack_tr, y_train)
final_pred = meta.predict(stack_te)

# 7. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, final_pred))
print(f"Stacking Regressor RMSE (hold‑out): {rmse:.4f}")

# 8. Save artifacts
joblib.dump(meta, 'stacking_meta.pkl')
pd.DataFrame({
    'rnn':  pred_rnn_te,
    'xgb':  pred_xgb_te,
    'lgbm': pred_lgbm_te,
    'stack': final_pred
}).to_csv('stacking_pred.csv', index=False)