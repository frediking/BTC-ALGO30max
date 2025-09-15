import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_sequence_data(X, y, window_size=30, val_split=0.2):
    """
    Prepare windowed sequences for time series prediction.
    
    Args:
        X (np.ndarray): Feature matrix [N, n_features]
        y (np.ndarray): Target vector [N]
        window_size (int): Number of time steps in each sequence
        val_split (float): Validation split ratio
        
    Returns:
        dict: Contains training and validation sequences with their shapes
    """
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences with validation
    X_sequences = np.array([
        X_scaled[i:i+window_size] 
        for i in range(len(X_scaled) - window_size + 1)
    ])
    y_sequences = y[window_size - 1:]
    
    # Ensure no data leakage in validation split
    split_idx = int(len(X_sequences) * (1 - val_split))
    
    # Split preserving temporal order
    X_train = X_sequences[:split_idx]
    X_val = X_sequences[split_idx:]
    y_train = y_sequences[:split_idx]
    y_val = y_sequences[split_idx:]
    
    data_info = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'scaler': scaler,
        'window_size': window_size,
        'shapes': {
            'X_train': X_train.shape,
            'X_val': X_val.shape,
            'y_train': y_train.shape,
            'y_val': y_val.shape
        }
    }
    
    return data_info 