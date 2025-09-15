#----------------------------------------------------------------------------------------------------------------------------------------
import os
# ── MUST GO FIRST ───────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]   = ""
# ── NORMAL IMPORTS ──────────────────────────────────────────────────
import time
import numpy as np
np.random.seed(42)
import pandas as pd
import optuna
import gc
import tensorflow as tf
tf.random.set_seed(42)
tf.config.run_functions_eagerly(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("ENV threads:",
      os.environ["OMP_NUM_THREADS"],
      os.environ["MKL_NUM_THREADS"],
      os.environ["TF_NUM_INTEROP_THREADS"],
      os.environ["TF_NUM_INTRAOP_THREADS"])
print("TF eager:", tf.executing_eagerly())

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, GRU, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from deep_learning_btc import (
    build_lstm_model, 
    build_bidirectional_lstm_model, 
    build_gru_model
)
import joblib
from scipy import stats  # Used in validate_model_parity
import matplotlib.pyplot as plt  # If you plan to visualize results
import psutil
import logging
import json


# ======== ENHANCED DATA PREP ========
class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Add time-aware features to capture temporal relationships"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Add cyclical time features, rolling stats, etc.
        X_new = np.hstack([
            X,
            np.sin(2 * np.pi * X[:, [0]] / 24),  # Example: hour of day
            np.cos(2 * np.pi * X[:, [0]] / 24),
            self._rolling_stats(X[:, 1], window=3),
            self._rolling_stats(X[:, 2], window=5)
        ])
        return X_new
    
    def _rolling_stats(self, series, window):
        return np.concatenate([
            np.full((window-1, 1), np.nan),
            np.lib.stride_tricks.sliding_window_view(series, window).mean(axis=1)[:, None]
        ])


# ─── 2. Data prep ───────────────────────────────────────────────────────────────
X = pd.read_csv('X_prepared.csv').values
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values.flatten()

#validate data
if np.isnan(X).any() or np.isinf(X).any():
        logging.error("NaN or Inf values found in X data")
        raise ValueError("NaN or Inf values in X data")
if np.isnan(y).any() or np.isinf(y).any():
        logging.error("NaN or Inf values found in y data")
        raise ValueError("NaN or Inf values in y data")

# Enhanced preprocessing pipeline
preprocessor = make_pipeline(
    TemporalFeatures(),  # Your custom transformer
    KNNImputer(n_neighbors=5),  # Add KNNImputer before scaling
    MinMaxScaler(),
    PCA(n_components=0.95)
)
X_processed = preprocessor.fit_transform(X)

# After preprocessing pipeline
X_processed_2d = preprocessor.fit_transform(X)  # 2D for XGBoost/LightGBM

# Reshape data to 3D for RNN (samples, timesteps, features)
X_processed = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))

# Define AttentionLayer for the AttentionLSTM model type
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

# ─── 3. Model builder with timing & SimpleRNN ─────────────────────────────────
def create_rnn_model(trial, input_shape):
    model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU', 'BiLSTM', 'AttentionLSTM'])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = trial.suggest_int('units', 32, 256, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    clipnorm = trial.suggest_float('clipnorm', 0.5, 5.0) if model_type == 'AttentionLSTM' else None
    
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if model_type == 'AttentionLSTM':
        model.add(LSTM(units, return_sequences=True))
        model.add(AttentionLayer())
        model.add(Dense(1))
    else:
        for i in range(n_layers):
            return_seq = (i < n_layers - 1)
            if model_type == 'LSTM':
                model.add(LSTM(units, return_sequences=return_seq))
            elif model_type == 'GRU':
                model.add(GRU(units, return_sequences=return_seq))
            elif model_type == 'BiLSTM':
                model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))
            model.add(Dropout(dropout))
        
        # Add batch normalization
        model.add(BatchNormalization())
        model.add(Dense(1))
    
    # Enhanced optimizer configuration
    if model_type == 'AttentionLSTM':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate=lr,
            clipnorm=clipnorm
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
    
    logging.info(f"Model created: type={model_type}, layers={n_layers}, units={units}, dropout={dropout}, lr={lr}")
    return model

def validate_model_parity(X_train, X_val, y_train, y_val, trial, n_runs=3):
    """
    Validates that the enhanced model maintains performance parity with original model.
    
    Args:
        X_train, X_val, y_train, y_val: Training and validation data
        trial: Optuna trial object
        n_runs: Number of runs for statistical significance
    
    Returns:
        dict: Comparison metrics and statistical test results
    """
    results = {
        'original': {'rmse': [], 'train_time': []},
        'enhanced': {'rmse': [], 'train_time': []}
    }
    
    for run in range(n_runs):
        print(f"\nValidation Run {run + 1}/{n_runs}")
        tf.keras.backend.clear_session()
        
        # 1. Train original model
        t0 = time.time()
        original_model = {
            'LSTM': build_lstm_model,
            'BiLSTM': build_bidirectional_lstm_model,
            'GRU': build_gru_model
        }[trial.params['model_type']](input_shape=X_train.shape[1:], output_dim=1)
        
        original_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
            verbose=0
        )
        
        original_time = time.time() - t0
        original_preds = original_model.predict(X_val)
        original_rmse = np.sqrt(mean_squared_error(y_val, original_preds))
        
        # 2. Train enhanced model
        t0 = time.time()
        enhanced_model = create_rnn_model(trial, input_shape=X_train.shape[1:])
        
        enhanced_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=trial.params['batch_size'],
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
            verbose=0
        )
        
        enhanced_time = time.time() - t0
        enhanced_preds = enhanced_model.predict(X_val)
        enhanced_rmse = np.sqrt(mean_squared_error(y_val, enhanced_preds))
        
        # Store results
        results['original']['rmse'].append(original_rmse)
        results['original']['train_time'].append(original_time)
        results['enhanced']['rmse'].append(enhanced_rmse)
        results['enhanced']['train_time'].append(enhanced_time)
    
    # Calculate statistics
    stats_results = {
        'rmse_mean_diff': np.mean(results['enhanced']['rmse']) - np.mean(results['original']['rmse']),
        'rmse_std_diff': np.std(results['enhanced']['rmse']) - np.std(results['original']['rmse']),
        'time_mean_diff': np.mean(results['enhanced']['train_time']) - np.mean(results['original']['train_time']),
        'rmse_ttest': stats.ttest_ind(results['enhanced']['rmse'], results['original']['rmse']),
        'original_mean_rmse': np.mean(results['original']['rmse']),
        'enhanced_mean_rmse': np.mean(results['enhanced']['rmse']),
        'original_mean_time': np.mean(results['original']['train_time']),
        'enhanced_mean_time': np.mean(results['enhanced']['train_time'])
    }
    
    # Print comparison
    print("\n=== Model Parity Validation Results ===")
    print(f"Original RMSE: {stats_results['original_mean_rmse']:.4f}")
    print(f"Enhanced RMSE: {stats_results['enhanced_mean_rmse']:.4f}")
    print(f"RMSE Difference: {stats_results['rmse_mean_diff']:.4f}")
    print(f"T-test p-value: {stats_results['rmse_ttest'].pvalue:.4f}")
    print(f"\nTraining Time Comparison:")
    print(f"Original: {stats_results['original_mean_time']:.2f}s")
    print(f"Enhanced: {stats_results['enhanced_mean_time']:.2f}s")
    
    # Determine if parity is maintained
    is_parity = (
        stats_results['rmse_ttest'].pvalue > 0.05  # No significant difference
        or stats_results['rmse_mean_diff'] < 0     # Enhanced is better
    )
    
    print(f"\nParity Maintained: {is_parity}")
    return stats_results

# ======== IMPROVED VALIDATION ========
def temporal_blocking_split(data, n_splits=3):
    """Time-aware cross-validation that prevents lookahead bias"""
    splits = []
    n_samples = len(data)
    k_fold_size = n_samples // (n_splits + 1)
    
    for i in range(n_splits):
        test_start = i * k_fold_size
        test_end = (i + 1) * k_fold_size
        train_end = test_start
        splits.append((
            np.arange(0, train_end),
            np.arange(test_start, test_end)
        ))
    return splits

# ─── 4. Single hold-out Optuna objective with timing logs ───────────────────────
def objective(trial):
    try:
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
        X_tr, X_val, y_tr, y_val = train_test_split(X_processed, y, test_size=0.2, shuffle=False)
        
        model = create_rnn_model(trial, input_shape=(X_processed.shape[1], X_processed.shape[2]))
        
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
        logging.info(f"Trial {trial.number}: RMSE {rmse}")
        
        # Clean up
        tf.keras.backend.clear_session()
        gc.collect()
        
        return rmse
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {str(e)}")
        tf.keras.backend.clear_session()
        gc.collect()
        return float('inf')

# ======== HYBRID ENSEMBLE STRATEGY ========
class HybridEnsemble:
    """Combine predictions using learned weights"""
    def __init__(self, models):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # Initialize equal weights
        
    def fit(self, X, y):
        # Optimize weights using validation data
        preds = np.array([model.predict(X) for model in self.models]).T
        self.weights = self._optimize_weights(preds, y)
        
    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        return np.dot(self.weights, preds)
    
    def _optimize_weights(self, preds, y_true):
        # Use constrained optimization for weight learning
        from scipy.optimize import minimize
        def objective(w):
            return np.sqrt(mean_squared_error(y_true, np.dot(preds, w)))
        
        bounds = [(0,1)] * len(self.models)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        initial_guess = self.weights
        
        result = minimize(objective, initial_guess, 
                         bounds=bounds, constraints=constraints)
        return result.x

# ======== POST-ENSEMBLE OPTIMIZATION ========
def optimize_ensemble_predictions(y_true, preds_df):
    """Use constrained optimization to find optimal ensemble weights"""
    from scipy.optimize import differential_evolution, minimize
    
    def loss(weights):
        combined = np.dot(preds_df.values, weights)
        return np.sqrt(mean_squared_error(y_true, combined))
    
    bounds = [(0,1)] * preds_df.shape[1]
    # Correct constraint format
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    initial_guess = np.ones(preds_df.shape[1]) / preds_df.shape[1]

    result = minimize(
        loss,
        initial_guess,
        bounds=bounds,
        constraints=constraints,
        # polish=True
    )
    return result.x

# ─── 5. Run study ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Create results directory
    results_dir = 'hyperparameter_tuning_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=f'{results_dir}/hyperparameter_tuning.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Memory management setup
    def get_memory_usage():
        return psutil.Process().memory_info().rss / 1024 / 1024  # in MB

    # ======== ENHANCED TUNING CONFIG ========
    # Update Optuna study configuration
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            reduction_factor=3
        )
    )
    
    # Balanced settings for M1 Pro 16GB
    n_trials = 50  # Good balance for your parameter space
    timeout = 2400  # 1.5 hours
    memory_threshold = 12000  # 12GB (leaving headroom for system)

    try:
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        study_start_time = time.time()
        study_checkpoint_callback = lambda study, trial: joblib.dump(
            study, f'study_checkpoint_{len(study.trials)}_trials.pkl'
        )
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout,
            catch=(Exception,),
            show_progress_bar=True,
            callbacks=[
                # Memory check callback
                lambda study, trial: (
                    trial.set_user_attr('memory_peak', get_memory_usage()),
                    print(f"\nMemory usage: {get_memory_usage():.2f} MB"),
                    gc.collect() if get_memory_usage() > memory_threshold else None
                ),
                study_checkpoint_callback
            ]
        )

    except KeyboardInterrupt:
        print("\nStudy interrupted. Saving current state...")
        joblib.dump(study, 'optuna_study_interrupted/1.pkl')
    except MemoryError:
        print("\nMemory limit reached. Saving current state...")
        gc.collect()
        joblib.dump(study, 'optuna_study_memory_limit/1.pkl')
    finally:
        # Clean up
        tf.keras.backend.clear_session()
        gc.collect()

    print("✨ Best trial:", study.best_trial.number,
          "params:", study.best_trial.params,
          "RMSE:", study.best_trial.value)

    # Refit best RNN on full data
    best_params = study.best_trial.params
    tf.keras.backend.clear_session()
    best_model = create_rnn_model(study.best_trial, input_shape=(X_processed.shape[1], X_processed.shape[2]))
    best_model.fit(X_processed, y, epochs=5, batch_size=study.best_trial.params['batch_size'], verbose=2)
    best_model.save('best_rnnmodel1.h5')

    # ========== XGBOOST & LIGHTGBM (SAFE PARALLELISM) ==========
    print("\n→ Tuning XGBoost via GridSearchCV:")
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    # Enhanced XGBoost/LightGBM parameter grids
    param_grid_xgb = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.005, 0.01, 0.05],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
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
    grid_xgb.fit(X_processed_2d, y)
    best_xgb = grid_xgb.best_estimator_
    xgb_rmse = np.sqrt(-grid_xgb.best_score_)
    print("Best XGB params:", grid_xgb.best_params_)
    print("Best XGB RMSE:", xgb_rmse)
    joblib.dump(best_xgb, 'best_xgbmodel1.pkl')

    print("\n→ Tuning LightGBM via GridSearchCV:")
    lgbm = LGBMRegressor(random_seed=42, n_jobs=-1)
    
    # Enhanced LightGBM parameter grid
    param_grid_lgbm = {
        'num_leaves': [31, 50, 100],
        'min_data_in_leaf': [20, 50, 100],
        'feature_fraction': [0.7, 0.9, 1.0],
        'bagging_fraction': [0.7, 0.9, 1.0],
        'lambda_l1': [0, 0.1, 0.5]
    }
    
    grid_lgbm = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid_lgbm,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_lgbm.fit(X_processed_2d, y)
    best_lgbm = grid_lgbm.best_estimator_
    lgbm_rmse = np.sqrt(-grid_lgbm.best_score_)
    print("Best LGBM params:", grid_lgbm.best_params_)
    print("Best LGBM RMSE:", lgbm_rmse)
    joblib.dump(best_lgbm, 'best_lgbmmodel1.pkl')

    # ========== ENSEMBLE ==========
    print("\n→ Diversity-based Ensemble (RNN + XGB + LGBM):")
    rnn_preds = best_model.predict(X_processed).flatten()
    xgb_preds = best_xgb.predict(X_processed_2d)
    lgbm_preds = best_lgbm.predict(X_processed_2d)
    
    # Create a DataFrame for ensemble predictions
    preds_df = pd.DataFrame({
        'rnn': rnn_preds,
        'xgb': xgb_preds,
        'lgbm': lgbm_preds
    })
    
    # Apply optimized weights instead of simple averaging
    ensemble_weights = optimize_ensemble_predictions(y, preds_df)
    print(f"Optimized ensemble weights: {ensemble_weights}")
    
    # Apply weights to get final predictions
    ensemble_preds = np.dot(preds_df.values, ensemble_weights)
    ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_preds))
    
    print(f"Optimized Ensemble RMSE: {ensemble_rmse:.4f}")
    
    # Add ensemble predictions to DataFrame and save
    preds_df['optimized_ensemble'] = ensemble_preds
    preds_df['simple_ensemble'] = (rnn_preds + xgb_preds + lgbm_preds) / 3
    preds_df.to_csv('ensemble_predictionshpt.csv', index=False)
    print("Saved ensemble_predictionshpt.csv")

    print("\n✅ Hyperparameter tuning complete.")

    # Save final results
    if study.best_trial:
        results = {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'n_trials': len(study.trials),
            'study_duration': time.time() - study_start_time,
            'ensemble_weights': ensemble_weights.tolist() if 'ensemble_weights' in locals() else None,
            'model_rmse': {
                'rnn': float(np.sqrt(mean_squared_error(y, rnn_preds))),
                'xgb': float(xgb_rmse),
                'lgbm': float(lgbm_rmse),
                'ensemble': float(ensemble_rmse)
            }
        }
        with open(f'{results_dir}/final_results.json', 'w') as f:
            json.dump(results, f, indent=4)