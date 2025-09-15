import os
import time
import numpy as np
import pandas as pd
import optuna
import gc
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import json

# ── MUST GO FIRST ───────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.random.set_seed(42)
tf.config.run_functions_eagerly(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ── DATA LOADING AND SPLITTING ──────────────────────────────────────
X = pd.read_csv('X_prepared.csv').values
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values.flatten()

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Preprocessor for training data
preprocessor = make_pipeline(
    KNNImputer(n_neighbors=5),
    MinMaxScaler(),
    PCA(n_components=0.95)
)

# Fit and transform training data
X_train_scaled = preprocessor.fit_transform(X_train)
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

# Transform test data (using training fit)
X_test_scaled = preprocessor.transform(X_test)
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ─── RNN MODEL BUILDER WITH INCREASED REGULARIZATION ────────────────
def create_rnn_model(trial, input_shape):
    units = trial.suggest_int('units', 8, 32)
    dropout = trial.suggest_float('dropout', 0.4, 0.7)  # Increased dropout range
    lr = trial.suggest_float('lr', 1e-4, 1e-3)

    model = Sequential([
        Input(shape=input_shape),
        LSTM(
            units,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)  # Increased L2 regularization
        ),
        Dropout(dropout),
        Dense(1, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer, 
        loss='mse',
        metrics=['RootMeanSquaredError']
    )
    return model

# ─── OPTUNA OBJECTIVE WITH TRAIN-VAL SPLIT ──────────────────────────
def objective(trial):
    try:
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
        
        # Further split training data into train and validation sets
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_rnn, y_train, test_size=0.2, shuffle=False)
        
        model = create_rnn_model(trial, input_shape=(1, X_train_rnn.shape[2]))
        
        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=batch_size,
            callbacks=[es],
            verbose=2
        )
        
        preds = model.predict(X_val, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        tf.keras.backend.clear_session()
        gc.collect()
        return rmse
    except Exception as e:
        tf.keras.backend.clear_session()
        gc.collect()
        return float('inf')

# ─── RUN OPTUNA STUDY ───────────────────────────────────────────────
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, timeout=3600)

    print("Best RNN trial:", study.best_trial.params, "CV RMSE:", study.best_value)

    # Refit best RNN on full training data
    best_params = study.best_trial.params
    tf.keras.backend.clear_session()
    best_model = create_rnn_model(study.best_trial, input_shape=(1, X_train_rnn.shape[2]))
    best_model.fit(X_train_rnn, y_train, epochs=5, batch_size=best_params['batch_size'], verbose=2)
    best_model.save('RMSE/001rnn.h5')

    # ========== XGBOOST & LIGHTGBM WITH INCREASED REGULARIZATION ==========
    print("\n→ Tuning XGBoost via GridSearchCV with increased regularization:")
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.6, 0.8],  # Lowered from [0.7, 1.0]
        'reg_alpha': [0, 0.1, 1],  # L1 regularization
        'reg_lambda': [1, 10]     # L2 regularization
    }

    tscv = TimeSeriesSplit(n_splits=3)
    grid_xgb = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid_xgb,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_xgb.fit(X_train_scaled, y_train)
    best_xgb = grid_xgb.best_estimator_
    xgb_rmse = np.sqrt(-grid_xgb.best_score_)
    print("Best XGB params:", grid_xgb.best_params_)
    print("Best XGB RMSE:", xgb_rmse)
    joblib.dump(best_xgb, 'RMSE/001xgb.pkl')

    print("\n→ Tuning LightGBM via GridSearchCV with increased regularization:")
    lgbm = LGBMRegressor(random_seed=42, n_jobs=-1)
    param_grid_lgbm = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [31, 50],
        'max_depth': [-1, 5],
        'subsample': [0.6, 0.8],  # Lowered
        'colsample_bytree': [0.7],
        'min_child_samples': [20, 50],
        'reg_alpha': [0, 0.1, 1],  # L1
        'reg_lambda': [1, 10]     # L2
    }

    grid_lgbm = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid_lgbm,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_lgbm.fit(X_train_scaled, y_train)
    best_lgbm = grid_lgbm.best_estimator_
    lgbm_rmse = np.sqrt(-grid_lgbm.best_score_)
    print("Best LGBM params:", grid_lgbm.best_params_)
    print("Best LGBM RMSE:", lgbm_rmse)
    joblib.dump(best_lgbm, 'RMSE/001lgbm.pkl')

    # ─── FEATURE SELECTION USING XGBOOST IMPORTANCES ─────────────────
    importances = best_xgb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_n = 20  # Select top 20 features
    top_features = sorted_idx[:top_n]
    
    # Reduce training and test data to top features
    X_train_selected = X_train_scaled[:, top_features]
    X_test_selected = X_test_scaled[:, top_features]
    X_train_rnn_selected = X_train_selected.reshape((X_train_selected.shape[0], 1, X_train_selected.shape[1]))
    X_test_rnn_selected = X_test_selected.reshape((X_test_selected.shape[0], 1, X_test_selected.shape[1]))

    # Refit models on selected features
    best_model.fit(X_train_rnn_selected, y_train, epochs=5, batch_size=best_params['batch_size'], verbose=2)
    best_xgb.fit(X_train_selected, y_train)
    best_lgbm.fit(X_train_selected, y_train)

    # ─── COMPUTE VALIDATION RMSE FOR WEIGHTED ENSEMBLE ───────────────
    # Split training data further for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_rnn_selected, y_train, test_size=0.2, shuffle=False)
    X_tr_scaled, X_val_scaled = X_train_selected[:-len(X_val)], X_train_selected[-len(X_val):]

    # Predict on validation set
    rnn_val_preds = best_model.predict(X_val).flatten()
    xgb_val_preds = best_xgb.predict(X_val_scaled)
    lgbm_val_preds = best_lgbm.predict(X_val_scaled)

    # Compute individual RMSEs
    rnn_rmse = np.sqrt(mean_squared_error(y_val, rnn_val_preds))
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))
    lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_val_preds))

    # Compute weights inversely proportional to RMSE
    total_rmse = rnn_rmse + xgb_rmse + lgbm_rmse
    w_rnn = (1 / rnn_rmse) / (1 / rnn_rmse + 1 / xgb_rmse + 1 / lgbm_rmse)
    w_xgb = (1 / xgb_rmse) / (1 / rnn_rmse + 1 / xgb_rmse + 1 / lgbm_rmse)
    w_lgbm = (1 / lgbm_rmse) / (1 / rnn_rmse + 1 / xgb_rmse + 1 / lgbm_rmse)

    # ─── ENSEMBLE ON TEST SET ────────────────────────────────────────
    rnn_test_preds = best_model.predict(X_test_rnn_selected).flatten()
    xgb_test_preds = best_xgb.predict(X_test_selected)
    lgbm_test_preds = best_lgbm.predict(X_test_selected)

    print(f"RNN RMSE: {rnn_rmse:.4f}")
    print(f"XGB RMSE: {xgb_rmse:.4f}")    
    print(f"LGBM RMSE: {lgbm_rmse:.4f}")

    # Weighted ensemble predictions
    ensemble_preds = w_rnn * rnn_test_preds + w_xgb * xgb_test_preds + w_lgbm * lgbm_test_preds
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    print(f"Weighted 3-model Ensemble RMSE: {ensemble_rmse:.4f}")

    # Save results
    pd.DataFrame({
        'rnn': rnn_test_preds,
        'xgb': xgb_test_preds,
        'lgbm': lgbm_test_preds,
        'ensemble': ensemble_preds
    }).to_csv('RMSE/ensemble_3model_predct.csv', index=False)
    print("Saved ensemble predictions to 'ensemble_3model_predct.csv'")

    # Save final results
    results = {
        'best_rnn_params': study.best_trial.params,
        'best_rnn_value': study.best_trial.value,
        'best_xgb_params': grid_xgb.best_params_,
        'best_lgbm_params': grid_lgbm.best_params_,
        'ensemble_rmse': ensemble_rmse
    }
    with open('RMSE/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)