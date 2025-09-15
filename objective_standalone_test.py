
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

X = np.random.rand(32, 50, 16).astype(np.float32)
y = np.random.rand(32).astype(np.float32)

print("Starting minimal TF fit (at very top)...")
model = Sequential([Flatten(input_shape=(50, 16)), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=1, batch_size=8, verbose=2)
print("Minimal TF fit at top successful!")





import os
import pandas as pd
import numpy as np
# import multiprocessing as mp

# mp.set_start_method('spawn', force=True)
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import logging
logging.basicConfig(level=logging.INFO)

from deep_learning_btc import load_and_preprocess_data, ModelTuner

# --- 1. Load processed features and target ---
df_processed, preprocessing_info = load_and_preprocess_data('encoded_output.csv')
y = pd.read_csv('y_prepared.csv').values.squeeze()
X = df_processed.values


# --- 2. Reshape for LSTM ---
timesteps = 50
if X.shape[0] != y.shape[0]:
    raise ValueError(f"X and y must have same number of samples. X: {X.shape[0]}, y: {y.shape[0]}")
num_samples = X.shape[0] - timesteps
if num_samples <= 0:
    raise ValueError("Not enough data to create LSTM sequences with the given timesteps.")

X_lstm = np.array([X[i:i+timesteps] for i in range(num_samples)])
y_lstm = y[timesteps:]

# --- 3. Train/val split ---
split = int(0.8 * len(X_lstm))
X_train, y_train = X_lstm[:split], y_lstm[:split]
X_val, y_val = X_lstm[split:], y_lstm[split:]

# --- 4. Tiny dummy fit sanity check (optional, can comment out) ---
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

X_train_debug = X_train[:32].astype(np.float32)
y_train_debug = y_train[:32].astype(np.float32)

print("X_train_debug shape:", X_train_debug.shape)
print("y_train_debug shape:", y_train_debug.shape)
print("X_train_debug dtype:", X_train_debug.dtype)
print("y_train_debug dtype:", y_train_debug.dtype)
print("Total elements in X_train_debug:", np.prod(X_train_debug.shape))
print("Total elements in y_train_debug:", np.prod(y_train_debug.shape))

print("Sample X_train_debug[0]:", X_train_debug[0])
print("Sample y_train_debug[0]:", y_train_debug[0])

# --- 5. DummyTrial for Optuna-like interface ---
class DummyTrial:
    def __init__(self):
        self.number = 0
    def suggest_categorical(self, name, choices):
        print(f"DummyTrial.suggest_categorical({name}, {choices}) -> {choices[0]}")
        return choices[0]
    def suggest_float(self, name, low, high, log=False):
        print(f"DummyTrial.suggest_float({name}, {low}, {high}, log={log}) -> {low}")
        return low
    def suggest_int(self, name, low, high):
        print(f"DummyTrial.suggest_int({name}, {low}, {high}) -> {low}")
        return low

# --- 6. Instantiate ModelTuner and run objective ---
tuner = ModelTuner(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    # Add any other required arguments here if ModelTuner needs them!
)

result = tuner.objective(
    DummyTrial(),
    model_type='LSTM',  # or 'GRU', etc. as appropriate
    input_shape=X_train.shape[1:],
    output_dim=1
)
print("Objective function result:", result)