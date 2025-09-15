import os
import time
import logging
import json
import optuna
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Local imports
from models.sequence_model import create_sequence_rnn_model
from data.data_preparation import prepare_sequence_data
from evaluation.metrics import evaluate_sequence_model
from models.model_utils import get_memory_usage, cleanup_memory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Environment setup
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure TensorFlow
tf.config.run_functions_eagerly(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def objective(trial):
    """Optuna objective function for sequence-based model optimization."""
    try:
        # Memory Management
        if get_memory_usage() > memory_threshold:
            cleanup_memory()
        
        # Data Preparation
        data = prepare_sequence_data(
            X=X,
            y=y,
            window_size=trial.suggest_int('window_size', 20, 50),
            val_split=0.2
        )
        
        # Model Creation
        model = create_sequence_rnn_model(
            trial=trial,
            input_shape=(data['window_size'], X.shape[1]),
            advanced_features=True
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=trial.suggest_int('patience', 5, 10),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                f'model_trial_{trial.number}.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Training
        history = model.fit(
            data['X_train'],
            data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=trial.suggest_int('epochs', 50, 200),
            batch_size=trial.suggest_int('batch_size', 16, 128, log=True),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluation
        metrics = evaluate_sequence_model(model, data, trial.number)
        
        # Save Trial Results
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('history', history.history)
        
        # Cleanup
        cleanup_memory()
        
        return metrics['rmse']
    
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.exceptions.TrialPruned()

if __name__ == '__main__':
    # Load data
    X = pd.read_csv('X_prepared.csv').values
    y = pd.read_csv('y1_prepared.csv').values.flatten()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Study setup
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        ),
        study_name='bitcoin_sequence_prediction'
    )
    
    # Memory threshold (12GB for M1 Pro)
    memory_threshold = 12000
    
    try:
        study_start_time = time.time()
        study.optimize(
            objective,
            n_trials=100,
            timeout=7200,
            gc_after_trial=True,
            show_progress_bar=True
        )
        
    except KeyboardInterrupt:
        logging.info("Study interrupted. Saving current state...")
        joblib.dump(study, 'results/interrupted_study.pkl')
    
    finally:
        # Save results
        results = {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'n_trials': len(study.trials),
            'study_duration': time.time() - study_start_time
        }
        
        with open('results/final_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save study
        joblib.dump(study, 'results/final_study.pkl')
        
        # Plot results
        optuna.visualization.plot_optimization_history(study).write_image('results/optimization_history.png')
        optuna.visualization.plot_param_importances(study).write_image('results/param_importances.png') 