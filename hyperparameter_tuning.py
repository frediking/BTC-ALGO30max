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
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, GRU, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
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
y = pd.read_csv('y_prepared.csv').values.flatten()
X_scaled = MinMaxScaler().fit_transform(X)
# reshape → (samples, timesteps=1, features)
X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# ─── 3. Model builder with timing & SimpleRNN ─────────────────────────────────
def create_rnn_model(trial, input_shape):
    t0 = time.time()
    
    # 1. Get base model configuration
    model_type = trial.suggest_categorical('model_type', ['LSTM', 'BiLSTM', 'GRU'])
    
    # 2. Get base model using existing builders
    base_model = {
        'LSTM': build_lstm_model,
        'BiLSTM': build_bidirectional_lstm_model,
        'GRU': build_gru_model
    }[model_type](input_shape=input_shape, output_dim=1)
    
    # 3. Extract layers from base model for modification
    layers = base_model.layers
    
    # 4. Advanced parameters that enhance but don't break existing functionality
    use_advanced = trial.suggest_categorical('use_advanced_features', [True, False])
    
    if use_advanced:
        # 4a. Enhanced model configuration
        model = Sequential()
        
        # First layer (RNN layer)
        if isinstance(layers[0], (LSTM, GRU, Bidirectional)):
            rnn_units = trial.suggest_int('rnn_units', 32, 256, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.4)
            recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.4)
            
            if isinstance(layers[0], Bidirectional):
                model.add(Bidirectional(
                    LSTM(rnn_units, 
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         input_shape=input_shape)
                ))
            elif isinstance(layers[0], LSTM):
                model.add(LSTM(rnn_units, 
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout,
                             input_shape=input_shape))
            else:  # GRU
                model.add(GRU(rnn_units, 
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            input_shape=input_shape))
        
        # Dense layers with enhanced parameters
        dense_units = trial.suggest_int('dense_units', 16, 128, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'selu', 'elu'])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        
        model.add(Dense(dense_units, activation=activation))
        
        # Optional: Add regularization
        if trial.suggest_categorical('use_regularization', [True, False]):
            l1_reg = trial.suggest_float('l1_reg', 1e-7, 1e-5, log=True)
            l2_reg = trial.suggest_float('l2_reg', 1e-7, 1e-5, log=True)
            model.add(Dense(1, 
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
        else:
            model.add(Dense(1))
    else:
        # Use original model if not using advanced features
        model = base_model
    
    # 5. Enhanced training parameters (compatible with both basic and advanced models)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # 6. Enhanced optimizer selection
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'AdamW'])
    optimizer_params = {
        'learning_rate': learning_rate
    }
    
    if optimizer_name == 'Adam':
        beta1 = trial.suggest_float('adam_beta1', 0.9, 0.999)
        beta2 = trial.suggest_float('adam_beta2', 0.999, 0.9999)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta1, beta_2=beta2
        )
    elif optimizer_name == 'RMSprop':
        rho = trial.suggest_float('rms_rho', 0.8, 0.99)
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate, rho=rho
        )
    else:  # AdamW
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    
    # 7. Compile with enhanced metrics
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mse', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    print(f"[Trial {trial.number}] build time: {time.time()-t0:.2f}s "
          f"─ type={model_type} advanced={use_advanced}")
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

# ─── 4. Single hold-out Optuna objective with timing logs ───────────────────────
def objective(trial):
    try:
        # Memory check at start of trial
        if get_memory_usage() > memory_threshold:
            print("Memory threshold exceeded, triggering cleanup...")
            tf.keras.backend.clear_session()
            gc.collect()

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_rnn, y, test_size=0.2, shuffle=False
        )
        
        model = create_rnn_model(trial, input_shape=X_rnn.shape[1:])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Removed 128 for memory
        
        # Use smaller validation set for parity checking
        if trial.params.get('use_advanced_features', False):
            val_size = min(len(X_val), 1000)  # Limit validation size
            X_val = X_val[:val_size]
            y_val = y_val[:val_size]

        t0 = time.time()
        loss = model.train_on_batch(X_tr, y_tr)
        print(f"train_on_batch done, loss={loss:.4f}, time={time.time()-t0:.2f}s")

        t1 = time.time()
        with tf.device("/CPU:0"):
            checkpoint = ModelCheckpoint(
                f'checkpoint_trial_{trial.number}.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min'
            )
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=5,
                batch_size=batch_size,
                callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True), checkpoint],
                verbose=2
            )
        print(f"[Trial {trial.number}] fit time: {time.time()-t1:.2f}s")

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"[Trial {trial.number}] RMSE: {rmse:.4f}")

        # Validate parity if using advanced features
        if trial.params.get('use_advanced_features', False):
            parity_results = validate_model_parity(X_tr, X_val, y_tr, y_val, trial)
            
            # Optionally: penalize the objective if parity is not maintained
            if not (parity_results['rmse_ttest'].pvalue > 0.05 or parity_results['rmse_mean_diff'] < 0):
                rmse *= 1.1  # 10% penalty for breaking parity

            tf.keras.backend.clear_session()
            gc.collect()
        
        # Clean up after each trial
        tf.keras.backend.clear_session()
        gc.collect()
        
        return rmse

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

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

    # More conservative study setup for M1 Pro
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
    )

    # Balanced settings for M1 Pro 16GB
    n_trials = 75  # Good balance for your parameter space
    timeout = 5400  # 1.5 hours
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
    best_model.fit(
        X_rnn, y,
        epochs=5,
        batch_size=best_params['batch_size'],
        callbacks=[EarlyStopping(monitor='loss', patience=2)],
        verbose=2
    )
    best_model.save('best_rnn_model.h5')

    # ========== XGBOOST & LIGHTGBM (SAFE PARALLELISM) ==========
    print("\n→ Tuning XGBoost via GridSearchCV:")
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 1.0]
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
    grid_xgb.fit(X_scaled, y)
    best_xgb = grid_xgb.best_estimator_
    xgb_rmse = np.sqrt(-grid_xgb.best_score_)
    print("Best XGB params:", grid_xgb.best_params_)
    print("Best XGB RMSE:", xgb_rmse)
    joblib.dump(best_xgb, 'best_xgb_model.pkl')

    print("\n→ Tuning LightGBM via GridSearchCV:")
    lgbm = LGBMRegressor(random_seed=42, n_jobs=-1)
    param_grid_lgbm = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'num_leaves': [31, 50],
        'max_depth': [-1, 5],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'min_child_samples': [20, 50]
    }
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
    joblib.dump(best_lgbm, 'best_lgbm_model.pkl')

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
    }).to_csv('ensemble_3model_predictions1.csv', index=False)
    print("Saved ensemble_3model_predictions1.csv")

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