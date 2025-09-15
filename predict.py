import os
# Disable GPU at the very start before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_DISABLE_MPS_FALLBACK'] = '1'  # Disable Metal GPU fallback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.losses import MeanSquaredError

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

def load_and_predict():
    """Load model and make predictions using CPU only"""
    try:
        # Define custom objects for model loading
        custom_objects = {
            'mse': MeanSquaredError(),
            'mean_squared_error': MeanSquaredError()
        }
        
        print("Loading model...")
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model('rnnmod_optimized.h5', 
                                             custom_objects=custom_objects)
        
        # Get input shape from model
        input_shape = model.input_shape
        expected_features = input_shape[-1]
        print(f"Expected features: {expected_features}")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        X_new = pd.read_csv('Xdate_prepared.csv')
        
        # Store dates before feature selection
        dates = X_new['Date'].copy()
        
        # Ensure we use only the expected features in the correct order
        feature_columns = [
            'stoch_k', 'stoch_d', 'rolling_vol_7', 'Volume', 'Price_ROC', 'Momentum_10', 'ATR',
          'CCI', 'price_change', 'volatility', 'RSI', 'RSI_Divergence',  'Close', 'Force_Index', 'close_lag_1'
        ][:expected_features]
        
        print(f"Using features: {feature_columns}")
        X_new = X_new[feature_columns]
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_new)
        
        # Create sequences with correct dimensions
        sequence_length = 5
        batch_size = 32  # Match the model's expected batch size
        all_predictions = []
        
        for i in range(0, len(X_scaled) - sequence_length + 1, batch_size):
            batch_end = min(i + batch_size, len(X_scaled) - sequence_length + 1)
            X_batch = []
            
            for j in range(i, batch_end):
                X_batch.append(X_scaled[j:(j + sequence_length)])
            
            # Pad batch if necessary
            if len(X_batch) < batch_size:
                padding = batch_size - len(X_batch)
                X_batch.extend([X_batch[-1]] * padding)
            
            X_batch = np.array(X_batch)
            
            with tf.device('/CPU:0'):
                batch_predictions = model.predict(X_batch, verbose=0)
                
            # Remove predictions for padded sequences
            valid_predictions = batch_predictions[:batch_end-i]
            all_predictions.extend(valid_predictions.flatten())
            
            # Show progress
            batch_num = i // batch_size + 1
            total_batches = (len(X_scaled) - sequence_length + 1) // batch_size
            print(f"Processed batch {batch_num}/{total_batches}", end='\r')

        # Create results DataFrame using stored dates
        results = pd.DataFrame({
            'Date': dates.iloc[sequence_length-1:len(all_predictions)+sequence_length-1],
            'Predicted_Return': all_predictions
        })
        
        # Add cumulative weekly returns
        results['Date'] = pd.to_datetime(results['Date'])
        results['Week'] = results['Date'].dt.isocalendar().week
        results['Cumulative_Weekly_Return'] = results.groupby('Week')['Predicted_Return'].transform(
            lambda x: (1 + x/100).cumprod() - 1
        )
        
        # Save predictions
        results.to_csv('rnn_predictions.csv', index=False)
        print("\n=== Bitcoin Return Predictions ===")
        print(f"Total predictions: {len(results)}")
        print(f"Date range: {results['Date'].min()} to {results['Date'].max()}")
        
        return results
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    predictions = load_and_predict()
    
    if predictions is not None:
        # Display weekly summary
        weekly_summary = predictions.groupby('Week').agg({
            'Date': 'first',
            'Cumulative_Weekly_Return': 'last'
        }).reset_index()
        
        print("\n=== Weekly Return Summary ===")
        for _, week in weekly_summary.iterrows():
            print(f"\nWeek of {week['Date'].strftime('%B %d')}:")
            print(f"Expected Return: {week['Cumulative_Weekly_Return']*100:.2f}%")