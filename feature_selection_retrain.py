"""
Feature Selection and RMSE Results (2025-04-20)

Model: Random Forest Regressor (n_estimators=100, random_state=42)
Cross-validation: 5-fold

Feature set RMSEs:
- Top 3 features:   3.03
- Top 5 features:   2.04
- Top 10 features:  1.95  <-- best
- Top 15 features:  1.98
- Top 20 features:  1.99
- All features:     2.02

Conclusion:
- Best performance is achieved using the top 10 features.
- Using more than 10 features does not improve performance.
- The top features are all engineered technical indicators.
- Use these results to guide future modeling and feature engineering.


XGBoost Feature Importances:
 stoch_k                    0.351572
stoch_d                    0.176030
volatility_volume_ratio    0.166732
rolling_vol_7              0.099454
CCI                        0.056386
ADX_pos                    0.043124
price_range                0.036430
rolling_return_7           0.035311
volatility                 0.034962
Williams_%R                0.000000


LightGBM Feature Importances:
 volatility_volume_ratio    547
stoch_k                    421
stoch_d                    418
rolling_vol_7              360
rolling_return_7           348
ADX_pos                    239
CCI                        237
volatility                 223
price_range                184
Williams_%R                 23

Update this documentation after major pipeline/model changes.
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb


# Load data
X = pd.read_csv('encoded_bitcoin_data.csv')
X = X.drop(columns=['Date', 'percentreturn','next_day_price','next_day_return','daily_returns','Profit/Loss_binary', 'Daily Returns', 'Volatitlity','%Return','Profit/Loss_target_mean'])

y = pd.read_csv('y1_prepared.csv')['percentreturn'].squeeze()

# Fit Random Forest and compute feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Define range of N to test
N_values = [3, 5, 10, 15, 17, 20, 25, len(X.columns)]
rmse_scores = []

for N in N_values:
    if N == len(X.columns):
        features = X.columns.tolist()
        label = "All features"
    else:
        features = rf_importance.head(N).index.tolist()
        label = f"Top {N}"
    X_subset = X[features]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(rf_model, X_subset, y, cv=7, scoring='neg_mean_squared_error')
    rmse = np.mean(np.sqrt(-scores))
    rmse_scores.append(rmse)
    print(f"Cross-validated RMSE ({label}): {rmse:.4f}")

# # Plot RMSE vs N
# plt.figure(figsize=(8,5))
# plt.plot([str(n) for n in N_values[:-1]] + ["All"], rmse_scores, marker='o')
# plt.xlabel("Number of Features")
# plt.ylabel("Cross-validated RMSE")
# plt.title("RMSE vs Number of Top Features")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("rmse_vs_n_features.png")
# plt.show()




# XGBoost and LightGBM analysis


# Use top N features (replace with your actual top 10 list if you want)
top_features = [
    'stoch_k',
    'stoch_d',
    'return_ma_7',
    'rolling_return_7',
    'rolling_vol_7',
    'Volume',
    'return_std_7',
    'Price_ROC',
    'Momentum_10',
    'BB_width',
    'KC_Width',
    'ATR',
    'CCI',
    'ADX_neg',
    'price_change',
    'volatility',
    'Williams_%R',
    'RSI',
    'RSI_Divergence',
    'MACD_histogram',
    'Force_Index',
]
X_top = X[top_features]

# Refit Random Forest on top features for proper SHAP and permutation analysis
rf_top = RandomForestRegressor(n_estimators=100, random_state=42)
rf_top.fit(X_top, y)

# --- SHAP Analysis (Random Forest) ---
import shap

print("\n--- SHAP Analysis (Random Forest) ---")
explainer_rf = shap.TreeExplainer(rf_top, feature_perturbation="interventional")
shap_values_rf = explainer_rf.shap_values(X_top, check_additivity=False)


# # Bar plot (global feature importance)
# shap.summary_plot(shap_values_rf, X_top, plot_type="bar", show=False)
# plt.title("Random Forest SHAP Feature Importance")
# plt.tight_layout()
# plt.savefig("randomforest_shap_feature_importance.png")
# plt.show()

# # Beeswarm plot (detailed global/local impact)
# shap.summary_plot(shap_values_rf, X_top, show=False)
# plt.title("Random Forest SHAP Beeswarm")
# plt.tight_layout()
# plt.savefig("randomforest_shap_beeswarm.png")
# plt.show()

# --- Permutation Importance (Random Forest) ---
from sklearn.inspection import permutation_importance

print("\n--- Permutation Importance (Random Forest) ---")
perm_result_rf = permutation_importance(rf_top, X_top, y, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')
perm_importances_rf = pd.Series(perm_result_rf.importances_mean, index=X_top.columns).sort_values(ascending=False)
print("Random Forest Permutation Importances:\n", perm_importances_rf)

perm_importances_rf.plot(kind='bar', title='Random Forest Permutation Importance')
plt.tight_layout()
plt.savefig("randomforest_permutation_importance.png")
plt.show()

#--------------------------------------------------------------------------------------

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_scores = cross_val_score(xgb_model, X_top, y, cv=5, scoring='neg_mean_squared_error')
xgb_rmse = np.mean(np.sqrt(-xgb_scores))
print(f"XGBoost Cross-validated RMSE (Top 10): {xgb_rmse:.4f}")


# Fit and get feature importances
xgb_model.fit(X_top, y)
xgb_importance = pd.Series(xgb_model.feature_importances_, index=top_features).sort_values(ascending=False)
print("XGBoost Feature Importances:\n", xgb_importance)

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_scores = cross_val_score(lgb_model, X_top, y, cv=5, scoring='neg_mean_squared_error')
lgb_rmse = np.mean(np.sqrt(-lgb_scores))
print(f"LightGBM Cross-validated RMSE (Top 10): {lgb_rmse:.4f}")

# Fit and get feature importances
lgb_model.fit(X_top, y)
lgb_importance = pd.Series(lgb_model.feature_importances_, index=top_features).sort_values(ascending=False)
print("LightGBM Feature Importances:\n", lgb_importance)

# Optional: Plot feature importances
xgb_importance.plot(kind='bar', title='XGBoost Feature Importance')
plt.tight_layout()
plt.show()

lgb_importance.plot(kind='bar', title='LightGBM Feature Importance')
plt.tight_layout()
plt.show()


# --- SHAP Analysis (XGBoost and LightGBM) ---
import shap

print("\n--- SHAP Analysis (XGBoost) ---")
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_top)

# Bar plot (global feature importance)
shap.summary_plot(shap_values_xgb, X_top, plot_type="bar", show=False)
plt.title("XGBoost SHAP Feature Importance")
plt.tight_layout()
plt.savefig("xgboost_shap_feature_importance.png")
plt.show()

# Beeswarm plot (detailed global/local impact)
shap.summary_plot(shap_values_xgb, X_top, show=False)
plt.title("XGBoost SHAP Beeswarm")
plt.tight_layout()
plt.savefig("xgboost_shap_beeswarm.png")
plt.show()

print("\n--- SHAP Analysis (LightGBM) ---")
explainer_lgb = shap.TreeExplainer(lgb_model)
shap_values_lgb = explainer_lgb.shap_values(X_top)

shap.summary_plot(shap_values_lgb, X_top, plot_type="bar", show=False)
plt.title("LightGBM SHAP Feature Importance")
plt.tight_layout()
plt.savefig("lightgbm_shap_feature_importance.png")
plt.show()

shap.summary_plot(shap_values_lgb, X_top, show=False)
plt.title("LightGBM SHAP Beeswarm")
plt.tight_layout()
plt.savefig("lightgbm_shap_beeswarm.png")
plt.show()

# --- Permutation Importance (XGBoost example, repeat for LightGBM if desired) ---
from sklearn.inspection import permutation_importance

print("\n--- Permutation Importance (XGBoost) ---")
perm_result_xgb = permutation_importance(xgb_model, X_top, y, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')
perm_importances_xgb = pd.Series(perm_result_xgb.importances_mean, index=X_top.columns).sort_values(ascending=False)
print("XGBoost Permutation Importances:\n", perm_importances_xgb)

perm_importances_xgb.plot(kind='bar', title='XGBoost Permutation Importance')
plt.tight_layout()
plt.savefig("xgboost_permutation_importance.png")
plt.show()

print("\n--- Permutation Importance (LightGBM) ---")
perm_result_lgb = permutation_importance(lgb_model, X_top, y, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')
perm_importances_lgb = pd.Series(perm_result_lgb.importances_mean, index=X_top.columns).sort_values(ascending=False)
print("LightGBM Permutation Importances:\n", perm_importances_lgb)

perm_importances_lgb.plot(kind='bar', title='LightGBM Permutation Importance')
plt.tight_layout()
plt.savefig("lightgbm_permutation_importance.png")
plt.show()