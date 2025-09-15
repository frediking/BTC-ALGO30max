# # ========== DATA LOADING ==========
# X = pd.read_csv('X_prepared.csv').values
# y = pd.read_csv('y_prepared.csv').values.flatten()

# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# # For debugging: limit data size (remove/comment this to use all data)
# # X_rnn = X_rnn[:200]
# # y = y[:200]

# # ========== RNN MODEL BUILDER ==========
# def create_rnn_model(trial, input_shape):
#     model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU', 'BiLSTM', 'BiGRU'])
#     n_layers = trial.suggest_int('n_layers', 1, 2)  # 2 layers max for speed/robustness
#     units = trial.suggest_int('units', 8, 64, log=True)  # Keep units small for now
#     dropout = trial.suggest_float('dropout', 0.0, 0.5)
#     lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

#     model = Sequential([Input(shape=input_shape),
#                         SimpleRNN(units, return_sequences=False),
#                         Dense(1)])
#     model.add(Input(shape=input_shape))
#     for i in range(n_layers):
#         return_seq = (i < n_layers - 1)
#         if model_type == 'LSTM':
#             model.add(LSTM(units, return_sequences=return_seq))
#         elif model_type == 'GRU':
#             model.add(GRU(units, return_sequences=return_seq))
#         elif model_type == 'BiLSTM':
#             model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))
#         else:
#             model.add(Bidirectional(GRU(units, return_sequences=return_seq)))
#         model.add(Dropout(dropout))
#     model.add(Dense(1))
#     optimizer = Adam(learning_rate=lr)
#     model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
#     return model

# # ========== OPTUNA OBJECTIVE ==========
# def objective(trial):
#     batch_size = trial.suggest_categorical('batch_size', [16, 32])
#     tscv = TimeSeriesSplit(n_splits=3)  # Use 3 splits for speed/robustness
#     val_rmses = []
#     fold = 0

#     for train_idx, val_idx in tscv.split(X_rnn):
#         fold += 1
#         print(f"\n--- Fold {fold} for trial {trial.number} ---")
#         tf.keras.backend.clear_session()
#         assert set(train_idx).isdisjoint(set(val_idx)), f"Leakage detected in fold {fold}!"

#         X_tr, X_val = X_rnn[train_idx], X_rnn[val_idx]
#         y_tr, y_val = y[train_idx], y[val_idx]

#         print("Any NaN in X_tr?", np.isnan(X_tr).any())
#         print("Any NaN in y_tr?", np.isnan(y_tr).any())
#         print("Any Inf in X_tr?", np.isinf(X_tr).any())
#         print("Any Inf in y_tr?", np.isinf(y_tr).any())

        

#         # Data checks
#         assert not np.isnan(X_tr).any(), "NaN in X_tr"
#         assert not np.isnan(y_tr).any(), "NaN in y_tr"
#         assert not np.isinf(X_tr).any(), "Inf in X_tr"
#         assert not np.isinf(y_tr).any(), "Inf in y_tr"

#         es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
#         model_fold = create_rnn_model(trial, input_shape=(1, X_rnn.shape[2]))
#         print(f"Before fit: X_tr {X_tr.shape}, y_tr {y_tr.shape}, X_val {X_val.shape}, y_val {y_val.shape}")
#         history = model_fold.fit(
#             X_tr, y_tr,
#             validation_data=(X_val, y_val),
#             epochs=5,  # Keep small for robustness; increase after confirming stability
#             batch_size=batch_size,
#             callbacks=[es],
#             verbose=2
#         )
#         print(f"After fit: fold {fold}, trial {trial.number}")
#         preds = model_fold.predict(X_val)
#         rmse = np.sqrt(mean_squared_error(y_val, preds))
#         val_rmses.append(rmse)
#         print(f"Fold {fold} RMSE: {rmse:.4f}")

#     avg_rmse = np.mean(val_rmses)
#     print(f"Trial {trial.number}: CV RMSEs: {val_rmses} | Avg: {avg_rmse:.4f}")
#     return avg_rmse

# # ========== MAIN ==========
# if __name__ == '__main__':
#     study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
#     study.optimize(objective, n_trials=2, timeout=600)  # Start with 2 trials for robustness

#     print("Best RNN trial:", study.best_trial.params, "CV RMSE:", study.best_value)

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

# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# # reshape → (samples, timesteps=1, features)
# X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

preprocessor = make_pipeline(  # Your custom transformer
    KNNImputer(n_neighbors=5),  # Add KNNImputer before scaling
    MinMaxScaler(),
    PCA(n_components=0.95))

X_scaled = preprocessor.fit_transform(X)
X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# class AttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)
        
#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
#                                  initializer="normal")
#         self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
#                                  initializer="zeros")
#         super(AttentionLayer, self).build(input_shape)
        
#     def call(self, x):
#         e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
#         a = tf.nn.softmax(e, axis=1)
#         output = x * a
#         return tf.reduce_sum(output, axis=1)
    
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[2]
    

# def create_rnn_model(trial, input_shape):
#     model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU', 'BiLSTM', 'AttentionLSTM'])
#     n_layers = trial.suggest_int('n_layers', 1, 3)
#     units = trial.suggest_int('units', 32, 256, log=True)
#     dropout = trial.suggest_float('dropout', 0.0, 0.5)
#     lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
#     clipnorm = trial.suggest_float('clipnorm', 0.5, 5.0) if model_type == 'AttentionLSTM' else None
    
#     model = Sequential()
#     model.add(Input(shape=input_shape))
    
#     if model_type == 'AttentionLSTM':
#         model.add(LSTM(units, return_sequences=True))
#         model.add(AttentionLayer())
#         model.add(Dense(1))
#     else:
#         for i in range(n_layers):
#             return_seq = (i < n_layers - 1)
#             if model_type == 'LSTM':
#                 model.add(LSTM(units, return_sequences=return_seq))
#             elif model_type == 'GRU':
#                 model.add(GRU(units, return_sequences=return_seq))
#             elif model_type == 'BiLSTM':
#                 model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))
#             model.add(Dropout(dropout))
        
#         # Add batch normalization
#         model.add(BatchNormalization())
#         model.add(Dense(1))
    
#     # Enhanced optimizer configuration
#     if model_type == 'AttentionLSTM':
#         optimizer = tf.keras.optimizers.Nadam(
#             learning_rate=lr,
#             clipnorm=clipnorm
#         )
#     else:
#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
#     model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
    
#     logging.info(f"Model created: type={model_type}, layers={n_layers}, units={units}, dropout={dropout}, lr={lr}")
#     return model


#─── 3. Model builder with timing & SimpleRNN ─────────────────────────────────
# def create_rnn_model(trial, input_shape):
#     model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU', 'BiLSTM'])
#     n_layers = trial.suggest_int('n_layers', 1, 2)
#     units = trial.suggest_int('units', 8, 64, log=True)
#     dropout = trial.suggest_float('dropout', 0.0, 0.5)
#     lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
#     model = Sequential()
#     model.add(Input(shape=input_shape))
    
#     for i in range(n_layers):
#         return_seq = (i < n_layers - 1)
#         if model_type == 'LSTM':
#             model.add(LSTM(units, return_sequences=return_seq))
#         elif model_type == 'GRU':
#             model.add(GRU(units, return_sequences=return_seq))
#         elif model_type == 'BiLSTM':
#             model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))
#         model.add(Dropout(dropout))
    
#     model.add(Dense(1))
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
    
#     logging.info(f"Model created: type={model_type}, layers={n_layers}, units={units}, dropout={dropout}, lr={lr}")
#     return model


#─── 3. Model builder with timing & SimpleRNN ─────────────────────────────────
# def create_rnn_model(trial, input_shape):
#     model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU'])  # Removed BiLSTM
#     n_layers = trial.suggest_int('n_layers', 1, 1)  # Force single layer
#     units = trial.suggest_int('units', 32, 64, step=16)  # 32 or 48 units
#     dropout = trial.suggest_float('dropout', 0.2, 0.4)  # Higher dropout for regularization
#     lr = trial.suggest_float('lr', 3e-4, 1e-3)  # Narrower learning rate range
    
#     model = Sequential()
#     model.add(Input(shape=input_shape))
    
#     # Single layer architecture
#     if model_type == 'LSTM':
#         model.add(LSTM(units))
#     else:  # GRU
#         model.add(GRU(units))
    
#     model.add(Dropout(dropout))
#     model.add(BatchNormalization())  # Added for stability
#     model.add(Dense(1))
    
#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=lr,
#         clipvalue=0.5  # Gradient clipping
#     )
#     model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
    
#     return model

def create_rnn_model(trial, input_shape):
    # Hyperparameter search space
    units = trial.suggest_int('units', 8, 32)
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    lr = trial.suggest_float('lr', 1e-4, 1e-3)

    model = Sequential([
        Input(shape=input_shape),
        LSTM(
            units,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
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
            epochs=10,
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
            epochs=10,
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

# ─── 4. Single hold-out Optuna objective with timing logs ───────────────────────
def objective(trial):
    try:
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
        X_tr, X_val, y_tr, y_val = train_test_split(X_rnn, y, test_size=0.2, shuffle=False)
        
        model = create_rnn_model(trial, input_shape=(1, X_rnn.shape[2]))
        
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
    

 # ─── 4. Single hold-out Optuna objective with timing logs ───────────────────────
# def objective(trial):
#     try:
#         batch_size = trial.suggest_categorical('batch_size', [32])  # Fixed batch size
#         tscv = TimeSeriesSplit(n_splits=3)
#         val_rmses = []
        
#         for fold, (train_idx, val_idx) in enumerate(tscv.split(X_rnn)):
#             X_tr, X_val = X_rnn[train_idx], X_rnn[val_idx]
#             y_tr, y_val = y[train_idx], y[val_idx]
            
#             model = create_rnn_model(trial, input_shape=(1, X_rnn.shape[2]))
            
#             es = EarlyStopping(
#                 monitor='val_loss',
#                 patience=3,  # Increased patience
#                 restore_best_weights=True
#             )
            
#             history = model.fit(
#                 X_tr, y_tr,
#                 validation_data=(X_val, y_val),
#                 epochs=15,  # Increased epochs
#                 batch_size=batch_size,
#                 callbacks=[es],
#                 verbose=0
#             )
            
#             preds = model.predict(X_val, verbose=0)
#             rmse = np.sqrt(mean_squared_error(y_val, preds))
#             val_rmses.append(rmse)
        
#         return np.mean(val_rmses)
#     except Exception as e:
#         return float('inf')   

    

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

    #More conservative study setup for M1 Pro
    # study = optuna.create_study(
    #     direction='minimize',
    #     pruner=optuna.pruners.MedianPruner(
    #         n_startup_trials=5,
    #         n_warmup_steps=3,
    #         interval_steps=1
    #     )
    # )
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    # Balanced settings for M1 Pro 16GB
    n_trials = 50 # Good balance for your parameter space
    timeout = 3600  # 1.5 hours
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
        joblib.dump(study, 'optuna_study_interrupted.pkl')
    except MemoryError:
        print("\nMemory limit reached. Saving current state...")
        gc.collect()
        joblib.dump(study, 'optuna_study_memory_limit.pkl')
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
    best_model = create_rnn_model(study.best_trial, input_shape=(1, X_rnn.shape[2]))
    best_model.fit(X_rnn, y, epochs=5, batch_size=study.best_trial.params['batch_size'], verbose=2)
    best_model.save('RMSE/1rnn2.h5')

    # ========== XGBOOST & LIGHTGBM (SAFE PARALLELISM) ==========
    print("\n→ Tuning XGBoost via GridSearchCV:")
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 1.0]

    }

    # param_grid_xgb = {
    # 'n_estimators': [100, 200, 300],
    # 'max_depth': [3, 5],
    # 'learning_rate': [0.01, 0.05],
    # 'colsample_bytree': [0.7, 0.9],
    # 'gamma': [0, 0.1, 0.2]

    # }

    tscv = TimeSeriesSplit(n_splits=3)
    grid_xgb = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid_xgb,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_xgb.fit(X_scaled, y)
    best_xgb = grid_xgb.best_estimator_
    xgb_rmse = np.sqrt(-grid_xgb.best_score_)
    print("Best XGB params:", grid_xgb.best_params_)
    print("Best XGB RMSE:", xgb_rmse)
    joblib.dump(best_xgb, 'RMSE/01xgb2.pkl')

    print("\n→ Tuning LightGBM via GridSearchCV:")
    lgbm = LGBMRegressor(random_seed=42, n_jobs=-1)
    param_grid_lgbm = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [31, 50],
        'max_depth': [-1, 5],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7],
        'min_child_samples': [20, 50],
        'feature_fraction': [0.7,  0.9]
    }
        #---------------------------,
        # 'min_data_in_leaf': [20, 50, 100],
        # 'feature_fraction': [0.7, 0.9, 1.0],
        # 'bagging_fraction': [0.7, 0.9, 1.0],
        # 'lambda_l1': [0, 0.1, 0.5]


    


    grid_lgbm = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid_lgbm,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_lgbm.fit(X_scaled, y)
    best_lgbm = grid_lgbm.best_estimator_
    lgbm_rmse = np.sqrt(-grid_lgbm.best_score_)
    print("Best LGBM params:", grid_lgbm.best_params_)
    print("Best LGBM RMSE:", lgbm_rmse)
    joblib.dump(best_lgbm, 'RMSE/01lgbm2.pkl')

    # ========== ENSEMBLE ==========
    print("\n→ Diversity-based Ensemble (RNN + XGB + LGBM):")
    rnn_preds = best_model.predict(X_rnn).flatten()
    xgb_preds = best_xgb.predict(X_scaled)
    lgbm_preds = best_lgbm.predict(X_scaled)
    ensemble3_preds = (rnn_preds + xgb_preds + lgbm_preds) / 3
    ensemble3_rmse = np.sqrt(mean_squared_error(y, ensemble3_preds))
    print(f"3-model Ensemble RMSE: {ensemble3_rmse:.4f}")
    pd.DataFrame({
        'rnn': rnn_preds,
        'xgb': xgb_preds,
        'lgbm': lgbm_preds,
        'ensemble': ensemble3_preds
    }).to_csv('RMSE/ensemble_3model_pred03.csv', index=False)
    print("Saved ensemble_3model_predictions03.csv")

    print("\n✅ Hyperparameter tuning complete.")

    # Save final results
    if study.best_trial:
        results = {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'n_trials': len(study.trials),
            'study_duration': time.time() - study_start_time
        }
        with open(f'{results_dir}/final_results.json', 'w') as f:
            json.dump(results, f, indent=4)













# import os
# import time
# import numpy as np
# import pandas as pd
# import optuna
# import tensorflow as tf
# from tensorflow.keras import Sequential, Input
# from tensorflow.keras.layers import SimpleRNN, Dense
# from tensorflow.keras.callbacks import EarlyStopping, Callback
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import gc

# # ── ENVIRONMENT SETUP ───────────────────────────────────────────
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU explicitly
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN warnings

# # ── TF CONFIGURATION FOR STABILITY ───────────────────────────────────
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.run_functions_eagerly(True)

# print("─" * 60)
# print("Environment configured:")
# print(f"  TF eager mode: {tf.executing_eagerly()}")
# print(f"  Thread settings: OMP={os.environ['OMP_NUM_THREADS']}, "
#       f"MKL={os.environ['MKL_NUM_THREADS']}, "
#       f"TF inter/intra={os.environ['TF_NUM_INTEROP_THREADS']}/{os.environ['TF_NUM_INTRAOP_THREADS']}")
# print("─" * 60)

# # ─── TIMEOUT AND MONITORING CALLBACKS ─────────────────────────────────────────
# class TimeoutCallback(Callback):
#     def __init__(self, timeout_seconds=60):
#         super().__init__()
#         self.timeout_seconds = timeout_seconds
#         self.start_time = None
        
#     def on_train_begin(self, logs=None):
#         self.start_time = time.time()
        
#     def on_batch_begin(self, batch, logs=None):
#         if time.time() - self.start_time > self.timeout_seconds:
#             print(f"\n⚠️ Timeout reached after {self.timeout_seconds} seconds!")
#             self.model.stop_training = True
            
# class ProgressLogger(Callback):
#     def __init__(self, trial_num):
#         super().__init__()
#         self.trial_num = trial_num
#         self.batch_times = []
#         self.start_time = None
#         self.batch_start = None
        
#     def on_train_begin(self, logs=None):
#         self.start_time = time.time()
#         print(f"[Trial {self.trial_num}] Training started")
        
#     def on_batch_begin(self, batch, logs=None):
#         print(f"[Trial {self.trial_num}] Batch {batch+1} begin")  # Debug
#         self.batch_start = time.time()
        
#     def on_batch_end(self, batch, logs=None):
#         print(f"[Trial {self.trial_num}] Batch {batch+1} end, batch_start={self.batch_start}")  # Debug
#         if self.batch_start is not None:
#             batch_time = time.time() - self.batch_start
#             self.batch_times.append(batch_time)
#             if (batch + 1) % 10 == 0:  # Log every 10 batches
#                 loss = logs.get('loss', 0)
#                 print(f"[Trial {self.trial_num}] Batch {batch+1} completed in {batch_time:.2f}s - loss: {loss:.4f}")
#         else:
#             print(f"[Trial {self.trial_num}] Warning: batch_start is None for batch {batch+1}")
        
#     def on_train_end(self, logs=None):
#         total_time = time.time() - self.start_time
#         print(f"[Trial {self.trial_num}] Training completed in {total_time:.2f}s")
#         if self.batch_times:
#             print(f"[Trial {self.trial_num}] Avg batch time: {np.mean(self.batch_times):.2f}s")

# # ─── DATA LOADING AND PREPARATION ───────────────────────────────────────────────
# try:
#     print("Loading and preparing data...")
#     t0 = time.time()
#     X = pd.read_csv('X_prepared.csv').values
#     y = pd.read_csv('y_prepared.csv').values.flatten()
    
#     print(f"X shape: {X.shape}, y shape: {y.shape}")
#     assert not np.isnan(X).any(), "NaN values found in X data"
#     assert not np.isnan(y).any(), "NaN values found in y data"
#     assert not np.isinf(X).any(), "Inf values found in X data"
#     assert not np.isinf(y).any(), "Inf values found in y data"
    
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
#     print(f"Data prep completed in {time.time() - t0:.2f}s")
#     print(f"X_rnn shape: {X_rnn.shape}, y shape: {y.shape}")
    
# except Exception as e:
#     print(f"❌ Error during data preparation: {str(e)}")
#     raise

# # ─── SIMPLIFIED MODEL BUILDER ─────────────────────────────────────────────────
# def create_rnn_model(trial, input_shape):
#     t0 = time.time()
    
#     units = trial.suggest_int('units', 8, 16, log=True)  # Reduced range for stability
    
#     try:
#         model = Sequential([
#             Input(shape=input_shape),
#             SimpleRNN(units),
#             Dense(1)
#         ])
        
#         lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
#         model.compile(
#             optimizer=optimizer,
#             loss='mse',
#             metrics=[tf.keras.metrics.RootMeanSquaredError()]
#         )
        
#         build_time = time.time() - t0
#         print(f"[Trial {trial.number}] Model built in {build_time:.2f}s - units={units}, lr={lr:.6f}")
        
#         return model
    
#     except Exception as e:
#         print(f"❌ Error building model: {str(e)}")
#         raise

# # ─── ROBUST OBJECTIVE FUNCTION WITH MANUAL TRAINING LOOP ───────────────────────
# def objective(trial):
#     print(f"\n{'=' * 50}")
#     print(f"Starting Trial {trial.number}")
#     print(f"{'=' * 50}")
#     trial_start = time.time()
    
#     try:
#         X_tr, X_val, y_tr, y_val = train_test_split(
#             X_rnn, y, test_size=0.2, shuffle=False
#         )
        
#         print(f"[Trial {trial.number}] Train shape={X_tr.shape}, Val shape={X_val.shape}")
        
#         model = create_rnn_model(trial, input_shape=X_rnn.shape[1:])
        
#         batch_size = trial.suggest_categorical('batch_size', [4, 8])  # Smaller batch sizes
#         epochs = trial.suggest_int('epochs', 2, 4)  # Fewer epochs
#         timeout = 120 if trial.number == 0 else 90  # Reduced timeout
        
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True, verbose=1),
#             TimeoutCallback(timeout_seconds=timeout),
#             ProgressLogger(trial.number)
#         ]
        
#         print(f"[Trial {trial.number}] Testing single batch training...")
#         ds = tf.data.Dataset.from_tensor_slices((X_tr[:batch_size], y_tr[:batch_size])).batch(batch_size)
#         xb, yb = next(iter(ds))
#         t0 = time.time()
#         loss = model.train_on_batch(xb, yb)
#         batch_time = time.time() - t0
#         loss_scalar = loss[0] if isinstance(loss, list) else loss
#         print(f"[Trial {trial.number}] Single batch test - loss={loss_scalar:.4f}, time={batch_time:.2f}s")
        
#         if batch_time > 3:  # Warn if batch takes too long
#             print(f"⚠️ Warning: Single batch training took {batch_time:.2f}s - model may be slow")

#         valid_callbacks = [cb for cb in callbacks if cb is not None]

#         # Call on_train_begin for all callbacks *before* the loop
#         for cb in valid_callbacks:
#             if hasattr(cb, 'set_model'): cb.set_model(model) # Set model for callbacks
#             if hasattr(cb, 'on_train_begin'): cb.on_train_begin(logs={})

#         print(f"[Trial {trial.number}] Starting manual training loop: epochs={epochs}, batch_size={batch_size}")
#         with tf.device("/CPU:0"):
#             n_samples = X_tr.shape[0]
#             steps_per_epoch = n_samples // batch_size

#             for epoch in range(epochs):
#                 # Call on_epoch_begin for all callbacks
#                 epoch_logs = {}
#                 for cb in valid_callbacks:
#                     if hasattr(cb, 'on_epoch_begin'): cb.on_epoch_begin(epoch, logs=epoch_logs)

#                 epoch_loss = 0.0
#                 model.stop_training = False # Reset stop_training flag per epoch

#                 for step in range(steps_per_epoch):
#                     # Call on_batch_begin for all callbacks
#                     batch_logs = {}
#                     for cb in valid_callbacks:
#                         if hasattr(cb, 'on_batch_begin'): cb.on_batch_begin(step, logs=batch_logs)

#                     # Check if timeout callback stopped training
#                     if model.stop_training:
#                         print(f"[Trial {trial.number}] Stopping epoch early due to callback signal (likely Timeout).")
#                         break

#                     start = step * batch_size
#                     end = start + batch_size
#                     X_batch = X_tr[start:end]
#                     y_batch = y_tr[start:end]
                    

#                     # try:
#                     #     loss = model.train_on_batch(X_batch, y_batch)
        
#                     #     # Handle different types of loss
#                     #     if loss is None:
#                     #         print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned None!")
#                     #         loss_scalar = 0.0
#                     #     elif isinstance(loss, list):
#                     #         if not loss:
#                     #             print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned an empty list!")
#                     #             loss_scalar = 0.0
#                     #         else:
#                     #             # Check if the first element (loss) is None
#                     #             if loss[0] is None:
#                     #                 print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned list with None loss: {loss}")
#                     #                 loss_scalar = 0.0
#                     #             else:
#                     #                 # Handle scalar or scalar array
#                     #                 if isinstance(loss[0], (float, int, np.number)):
#                     #                     loss_scalar = loss[0]
#                     #                 elif isinstance(loss[0], np.ndarray) and loss[0].size == 1:
#                     #                     loss_scalar = float(loss[0])
#                     #                 else:
#                     #                     print(f"⚠️ [Trial {trial.number}] Step {step+1}: Unexpected type for loss[0]: {type(loss[0])}")
#                     #                     loss_scalar = 0.0
#                     #     elif isinstance(loss, (float, int, np.number)):
#                     #         loss_scalar = loss
#                     #     else:
#                     #         print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned unexpected type: {type(loss)}, value: {loss}")
#                     #         loss_scalar = 0.0
                        
#                     #     # Final check: Ensure loss_scalar is finite
#                     #     if isinstance(loss_scalar, (float, int)) and np.isfinite(loss_scalar):
#                     #         epoch_loss += loss_scalar
#                     #     else:
#                     #         print(f"🚫 [Trial {trial.number}] Step {step+1}: Skipping epoch_loss update due to invalid loss_scalar: {loss_scalar}")
                    
#                     # except Exception as batch_exception:
#                     #     print(f"❌ [Trial {trial.number}] Step {step+1}: Exception during train_on_batch or loss handling: {batch_exception}")
#                     #     tf.keras.backend.clear_session()
#                     #     gc.collect()
#                     #     return float('inf')


#                     # Robust loss checking
#                     try:
#                         loss = model.train_on_batch(X_batch, y_batch)
#                         # print(f"[Trial {trial.number}] Step {step+1}: train_on_batch returned: {loss}") # Optional: Add for detailed debugging

#                         loss_scalar = None # Initialize loss_scalar to None

#                         if loss is None:
#                              print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned None!")
#                              # Assign a default float value to avoid immediate TypeError
#                              loss_scalar = 0.0
#                         elif isinstance(loss, list):
#                             if not loss: # Check if list is empty
#                                  print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned an empty list!")
#                                  loss_scalar = 0.0
#                             else:
#                                 # Check if the first element (presumed loss) is None
#                                 if loss[0] is None:
#                                     print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned list with None loss: {loss}")
#                                     loss_scalar = 0.0
#                                 else:
#                                     loss_scalar = loss[0] # Assign the loss value
#                         elif isinstance(loss, (float, int, np.number)):
#                             loss_scalar = loss # It returned a single numeric value
#                         else:
#                              # Handle unexpected return types
#                              print(f"⚠️ [Trial {trial.number}] Step {step+1}: train_on_batch returned unexpected type: {type(loss)}, value: {loss}")
#                              loss_scalar = 0.0

#                         # Final check: Ensure loss_scalar is a float before adding
#                         if loss_scalar is None:
#                              print(f"🔥 [Trial {trial.number}] Step {step+1}: loss_scalar ended up as None unexpectedly!")
#                              loss_scalar = 0.0 # Assign default to prevent TypeError

#                         # Perform addition only if loss_scalar is valid number
#                         if isinstance(loss_scalar, (float, int, np.number)) and np.isfinite(loss_scalar):
#                              epoch_loss += loss_scalar
#                         else:
#                              print(f"🚫 [Trial {trial.number}] Step {step+1}: Skipping epoch_loss update due to invalid loss_scalar: {loss_scalar}")


#                     except Exception as batch_exception:
#                         print(f"❌ [Trial {trial.number}] Step {step+1}: Exception during train_on_batch or loss handling: {batch_exception}")
#                         # Returning inf tells Optuna the trial failed severely
#                         tf.keras.backend.clear_session()
#                         gc.collect()
#                         return float('inf')
#                     # --- End of replacement block ---

#                     # Update how logs are passed to callbacks to use the safe loss_scalar
#                     logs_for_callbacks = {'loss': loss_scalar if loss_scalar is not None else 0.0}
#                     for cb in callbacks:
#                         # Make sure cb.on_batch_end exists before calling
#                         if hasattr(cb, 'on_batch_end'):
#                              cb.on_batch_end(step, logs=logs_for_callbacks)
                    
#                     if model.stop_training:
#                         print(f"[Trial {trial.number}] Early stopping triggered")
#                         break
#                     if step % 10 == 0:
#                         print(f"[Trial {trial.number}] Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}, Batch loss: {loss_scalar:.4f}")
                
#                 if model.stop_training:
#                     break
#                 avg_loss = epoch_loss / steps_per_epoch
#                 val_loss = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)[0]
#                 print(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs} completed, Avg loss: {avg_loss:.4f}, Val loss: {val_loss:.4f}")
        
#         preds = model.predict(X_val, batch_size=batch_size, verbose=0)
#         rmse = np.sqrt(mean_squared_error(y_val, preds))
        
#         trial_time = time.time() - trial_start
#         print(f"[Trial {trial.number}] Completed in {trial_time:.2f}s - RMSE: {rmse:.4f}")
        
#         tf.keras.backend.clear_session()
#         gc.collect()
        
#         return rmse
    
#     except Exception as e:
#         print(f"❌ Trial {trial.number} failed with error: {str(e)}")
#         trial_time = time.time() - trial_start
#         print(f"[Trial {trial.number}] Failed after {trial_time:.2f}s")
#         tf.keras.backend.clear_session()
#         gc.collect()
#         return float('inf')

# # ─── MAIN EXECUTION ─────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     try:
#         print("\n🔍 Starting hyperparameter optimization")
#         total_start = time.time()
        
#         study = optuna.create_study(
#             direction='minimize',
#             pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1)
#         )
        
#         n_trials = 3
#         timeout = 240  # 4 minutes max
        
#         study.optimize(
#             objective, 
#             n_trials=n_trials,
#             timeout=timeout,
#             catch=(Exception,)
#         )
        
#         total_time = time.time() - total_start
#         print("\n" + "=" * 60)
#         print(f"✅ Optimization completed in {total_time:.2f}s")
#         print(f"Total trials: {len(study.trials)}")
#         print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
#         print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
#         if study.best_trial.value < float('inf'):
#             print("\n🏆 Best trial:")
#             print(f"  Trial number: {study.best_trial.number}")
#             print(f"  Parameters: {study.best_trial.params}")
#             print(f"  RMSE: {study.best_trial.value:.4f}")
            
#             print("\n🔄 Retraining best model on full dataset...")
#             tf.keras.backend.clear_session()
            
#             best_trial = optuna.trial.FixedTrial(study.best_trial.params)
#             best_model = create_rnn_model(best_trial, input_shape=(1, X_rnn.shape[2]))
            
#             batch_size = study.best_trial.params['batch_size']
#             epochs = 3  # Fixed for stability
#             n_samples = X_rnn.shape[0]
#             steps_per_epoch = n_samples // batch_size
#             print(f"[INFO] Training best model with manual train_on_batch loop: batch_size={batch_size}, epochs={epochs}")
#             with tf.device("/CPU:0"):
#                 for epoch in range(epochs):
#                     epoch_loss = 0.0
#                     for step in range(steps_per_epoch):
#                         start = step * batch_size
#                         end = start + batch_size
#                         X_batch = X_rnn[start:end]
#                         y_batch = y[start:end]
#                         loss = best_model.train_on_batch(X_batch, y_batch)
#                         loss_scalar = loss[0] if isinstance(loss, list) else loss
#                         epoch_loss += loss_scalar
#                         if step % 10 == 0:
#                             print(f"  Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}, Batch loss: {loss_scalar:.4f}")
#                     avg_loss = epoch_loss / steps_per_epoch
#                     print(f"[INFO] Epoch {epoch+1}/{epochs} completed, Avg loss: {avg_loss:.4f}")
            
#             model_path = 'best_rnn_model.h5'
#             best_model.save(model_path)
#             print(f"✅ Best model saved to {model_path}")
        
#         else:
#             print("\n❌ No successful trials completed.")
        
#         tf.keras.backend.clear_session()
#         gc.collect()
        
#     except KeyboardInterrupt:
#         print("\n⚠️ Process interrupted by user")
#     except Exception as e:
#         print(f"\n❌ Error in main execution: {str(e)}")
#         tf.keras.backend.clear_session()
#         gc.collect()