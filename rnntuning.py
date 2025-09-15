import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load data
X = pd.read_csv('X_prepared.csv').values
y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# Split into train and test sets (80% train, 20% test)
N = len(X)
train_size = int(0.8 * N)
X_train_original = X[0:train_size, :]
y_train_original = y[0:train_size]
X_test_original = X[train_size:, :]
y_test_original = y[train_size:]

# Define preprocessor
preprocessor = make_pipeline(
    KNNImputer(n_neighbors=5),
    MinMaxScaler(),
    PCA(n_components=0.95)
)
preprocessor.fit(X_train_original)
X_train_scaled = preprocessor.transform(X_train_original)
X_test_scaled = preprocessor.transform(X_test_original)

# Define sequence length (T) for RNN
T = 5

# Function to create sequences
def create_sequences(data, target, T):
    X_seq = []
    y_seq = []
    for i in range(len(data) - T + 1):
        X_seq.append(data[i:i+T, :])
        y_seq.append(target[i+T-1])
    return np.array(X_seq), np.array(y_seq)

# Create sequences for train and test
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_original, T)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_original, T)

# Define RNN model builder with hyperparameter tuning
def create_rnn_model(trial, input_shape):
    model_type = trial.suggest_categorical('model_type', ['LSTM', 'BidirectionalLSTM'])
    units = trial.suggest_categorical('units', [32, 64])
    n_layers = trial.suggest_int('n_layers', 1, 2)
    dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-3)
    reg_strength = trial.suggest_loguniform('reg_strength', 1e-4, 1e-2)

    model = Sequential()
    first_layer = True
    for _ in range(n_layers):
        if model_type == 'LSTM':
            if first_layer:
                model.add(LSTM(units, input_shape=input_shape, return_sequences=True if _ < n_layers - 1 else False,
                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
                first_layer = False
            else:
                model.add(LSTM(units, return_sequences=True if _ < n_layers - 1 else False,
                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
        elif model_type == 'BidirectionalLSTM':
            if first_layer:
                model.add(Bidirectional(LSTM(units, input_shape=input_shape, return_sequences=True if _ < n_layers - 1 else False,
                                             kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
                first_layer = False
            else:
                model.add(Bidirectional(LSTM(units, return_sequences=True if _ < n_layers - 1 else False,
                                             kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
    return model

# Optuna objective function
def objective(trial):
    # Split train into train-train and validation
    split = int(0.8 * len(X_train_seq))
    X_tr = X_train_seq[:split]
    y_tr = y_train_seq[:split]
    X_val = X_train_seq[split:]
    y_val = y_train_seq[split:]

    input_shape = X_tr.shape[1:]
    model = create_rnn_model(trial, input_shape)

    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    es = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=batch_size,
              callbacks=[es], verbose=0)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds.flatten()))
    return rmse

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get best hyperparameters
best_params = study.best_trial.params
print("Best RNN trial:", best_params, "CV RMSE:", study.best_value)

# Retrain best RNN on full training data
input_shape = X_train_seq.shape[1:]
model = create_rnn_model(study.best_trial, input_shape)
batch_size = best_params['batch_size']
es = EarlyStopping(monitor='loss', patience=10)
model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=batch_size, callbacks=[es], verbose=2)

# Evaluate on test set
preds_test = model.predict(X_test_seq)
test_rmse = np.sqrt(mean_squared_error(y_test_seq, preds_test.flatten()))
print(f"Test RMSE: {test_rmse}")

# Save the best model
model.save('rnnmod.h5')
joblib.dump(test_rmse, 'rnnmod.h5')