import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
import joblib

# Load data
X_df = pd.read_csv('Xg_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# # Feature Engineering
# for col in ['volatility', 'RSI', 'Volume', 'Price_ROC']:
#     for lag in [1, 3, 5]:
#         X_df[f'{col}_lag_{lag}'] = X_df[col].shift(lag)
#     X_df[f'{col}_ma_3'] = X_df[col].rolling(window=3).mean()
#     X_df[f'{col}_ma_7'] = X_df[col].rolling(window=7).mean()
#     X_df[f'{col}_ema_14'] = X_df[col].ewm(span=14, adjust=False).mean()
# X_df['volatility_momentum'] = X_df['volatility'].diff()
# X_df['RSI_div_ma'] = X_df['RSI_Divergence'].rolling(window=5).mean()
# X_df['vol_price_interaction'] = X_df['Volume'] * X_df['Price_ROC']
# X_df['RSI_squared'] = X_df['RSI'] ** 2

# Drop NaN rows from feature engineering
X_df = X_df.dropna()
y = y[X_df.index]

# Convert to numpy array
X = X_df.values

# Split into train and test sets (80% train, 20% test, no shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define LogTransformer
# Define LogTransformer with column names
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.col_indices = None

    def fit(self, X, y=None):
        self._set_col_indices(X)
        return self

    def transform(self, X):
        # Defensive: ensure col_indices is set
        if self.col_indices is None:
            self._set_col_indices(X)
        X_transformed = X.copy()
        if hasattr(X_transformed, 'columns'):
            for col in self.columns:
                if col in X_transformed.columns:
                    X_transformed[col] = np.log1p(np.abs(X_transformed[col])) * np.sign(X_transformed[col])
        else:
            for idx in self.col_indices:
                X_transformed[:, idx] = np.log1p(np.abs(X_transformed[:, idx])) * np.sign(X_transformed[:, idx])
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def _set_col_indices(self, X):
        if hasattr(X, 'columns'):
            self.col_indices = [X.columns.get_loc(col) for col in self.columns if col in X.columns]
        else:
            self.col_indices = self.columns

# Define preprocessor
force_index_col = X_df.columns.get_loc('Force_Index')
log_transformer = LogTransformer(columns=[force_index_col])
preprocessor = make_pipeline(
    log_transformer,
    IterativeImputer(max_iter=10, random_state=42),
    RobustScaler()
)

# Apply preprocessor
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Set up TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
folds = list(tscv.split(X_train_scaled))

# Define Optuna objective
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_jobs': -1,
        'seed': 42,
        'eta': trial.suggest_loguniform('eta', 1e-4, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-2, 10),
    }
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        folds=folds,
        metrics='rmse',
        early_stopping_rounds=50,
        seed=42,
        verbose_eval=False
    )
    return cv_results['test-rmse-mean'].min()

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

# Get best parameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
print("Best RMSE from Cross-Validation:", study.best_value)

# Split training data for final model training
val_size = int(0.1 * len(X_train_scaled))
X_tr = X_train_scaled[:-val_size]
y_tr = y_train[:-val_size]
X_val = X_train_scaled[-val_size:]
y_val = y_train[-val_size:]

# Prepare DMatrix
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)

# Ensure fixed parameters
best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'rmse'

# Train final model with early stopping
evals = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=10
)

# Predict on test set
dtest = xgb.DMatrix(X_test_scaled)
preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

# Evaluate
test_rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Test RMSE: {test_rmse}")

# Save model
joblib.dump(model, 'xghost.pkl')