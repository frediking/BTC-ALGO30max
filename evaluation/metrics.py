import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import tensorflow as tf
from deep_learning_btc import (
    build_lstm_model, 
    build_bidirectional_lstm_model, 
    build_gru_model
)
import time

def evaluate_sequence_model(model, data, trial_number):
    """
    Evaluate sequence model with multiple metrics.
    """
    predictions = model.predict(data['X_val'])
    
    # 1. Basic Metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(data['y_val'], predictions)),
        'mae': mean_absolute_error(data['y_val'], predictions),
        'r2': r2_score(data['y_val'], predictions)
    }
    
    # 2. Directional Accuracy
    actual_direction = np.sign(np.diff(data['y_val']))
    pred_direction = np.sign(np.diff(predictions.flatten()))
    metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction)
    
    # 3. Volatility Capture
    actual_vol = np.std(data['y_val'])
    pred_vol = np.std(predictions)
    metrics['volatility_ratio'] = pred_vol / actual_vol
    
    # 4. Maximum Drawdown
    def calculate_max_drawdown(returns):
        cum_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - running_max
        return np.min(drawdown)
    
    actual_returns = np.diff(data['y_val'])
    pred_returns = np.diff(predictions.flatten())
    metrics['actual_max_drawdown'] = calculate_max_drawdown(actual_returns)
    metrics['pred_max_drawdown'] = calculate_max_drawdown(pred_returns)
    
    # 5. Save Predictions
    pd.DataFrame({
        'actual': data['y_val'],
        'predicted': predictions.flatten(),
        'error': data['y_val'] - predictions.flatten()
    }).to_csv(f'predictions_trial_{trial_number}.csv')
    
    return metrics

def validate_model_parity(X_train, X_val, y_train, y_val, trial, n_runs=3):
    """
    Validates model performance parity.
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
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)],
            verbose=0
        )
        
        original_time = time.time() - t0
        original_preds = original_model.predict(X_val)
        original_rmse = np.sqrt(mean_squared_error(y_val, original_preds))
        
        # Store results and calculate statistics
        results['original']['rmse'].append(original_rmse)
        results['original']['train_time'].append(original_time)
    
    # Calculate final statistics
    stats_results = {
        'rmse_mean_diff': np.mean(results['enhanced']['rmse']) - np.mean(results['original']['rmse']),
        'rmse_ttest': stats.ttest_ind(results['enhanced']['rmse'], results['original']['rmse']),
        'original_mean_rmse': np.mean(results['original']['rmse']),
        'enhanced_mean_rmse': np.mean(results['enhanced']['rmse'])
    }
    
    return stats_results 