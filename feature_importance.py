import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 1. Load prepared features and target
X = pd.read_csv('encoded_bitcoin_data.csv')
X = X.drop(columns=['Date', 'percentreturn','next_day_price','next_day_return','daily_returns','Profit/Loss_binary', 'Daily Returns', 'Volatitlity','%Return','Profit/Loss_target_mean'])

y = pd.read_csv('y1_prepared.csv').squeeze()  # .squeeze() to get a Series

# 2. Fit a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 3. Random Forest built-in feature importance
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Random Forest Feature Importance:")
print(rf_importance.head(20))

# 4. Permutation importance
perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nPermutation Importance:")
print(perm_importance.head(20))

# # 5. SHAP values (optional, but insightful)
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X)
# shap.summary_plot(shap_values, X, plot_type="bar", show=False)
# plt.title("SHAP Feature Importance")
# plt.tight_layout()
# plt.savefig("shap_feature_importance.png")
# plt.show()

# # 6. Plot Random Forest and Permutation Importance
# plt.figure(figsize=(10, 5))
# rf_importance.head(20).plot(kind='bar', color='skyblue', alpha=0.7, label='RF Importance')
# plt.title('Top 20 Random Forest Feature Importances')
# plt.ylabel('Importance')
# plt.tight_layout()
# plt.savefig("rf_feature_importance.png")
# plt.show()

# plt.figure(figsize=(10, 5))
# perm_importance.head(20).plot(kind='bar', color='orange', alpha=0.7, label='Permutation Importance')
# plt.title('Top 20 Permutation Feature Importances')
# plt.ylabel('Importance')
# plt.tight_layout()
# plt.savefig("permutation_feature_importance.png")
# plt.show()