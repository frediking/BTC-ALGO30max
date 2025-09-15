# import numpy as np
# import pandas as pd
# import optuna
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, LayerNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
# from sklearn.pipeline import make_pipeline
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.decomposition import PCA
# import joblib

# # Set random seeds for reproducibility
# tf.random.set_seed(42)
# np.random.seed(42)

# # Configure TensorFlow to use M1 Pro GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# # Load data
# X = pd.read_csv('Xrn_prepared.csv').values
# y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# # Split into train and test sets (80% train, 20% test)
# N = len(X)
# train_size = int(0.8 * N)
# X_train_original = X[:train_size, :]
# y_train_original = y[:train_size]
# X_test_original = X[train_size:, :]
# y_test_original = y[train_size:]

# # Define preprocessor
# preprocessor = make_pipeline(
#     IterativeImputer(max_iter=10, random_state=42),
#     StandardScaler(),
#     PCA(n_components=0.95)
# )
# preprocessor.fit(X_train_original)
# X_train_scaled = preprocessor.transform(X_train_original)
# X_test_scaled = preprocessor.transform(X_test_original)

# # Define sequence length (T) for RNN (5-10)
# T = 5

# # Function to create sequences
# def create_sequences(data, target, T):
#     X_seq, y_seq = [], []
#     for i in range(len(data) - T + 1):
#         X_seq.append(data[i:i+T, :])
#         y_seq.append(target[i+T-1])
#     return np.array(X_seq), np.array(y_seq)

# # Create sequences for train and test
# X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_original, T)
# X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_original, T)

# # Define RNN model builder
# def create_rnn_model(trial, input_shape):
#     model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU'])
#     units = trial.suggest_categorical('units', [64, 128])  # Focus on higher capacity
#     n_layers = trial.suggest_int('n_layers', 1, 2)  # Reduced max layers
#     dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3])
#     lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
#     reg_strength = trial.suggest_loguniform('reg_strength', 1e-5, 1e-3)
#     n_dense = trial.suggest_int('n_dense', 0, 1)  # Limited dense layers
#     dense_units = trial.suggest_categorical('dense_units', [16, 32]) if n_dense > 0 else None

#     model = Sequential()
#     first_layer = True
#     for i in range(n_layers):
#         return_sequences = i < n_layers - 1
#         if model_type == 'LSTM':
#             if first_layer:
#                 model.add(LSTM(units, input_shape=input_shape, return_sequences=return_sequences,
#                                kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
#                 first_layer = False
#             else:
#                 model.add(LSTM(units, return_sequences=return_sequences,
#                                kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
#         elif model_type == 'GRU':
#             if first_layer:
#                 model.add(GRU(units, input_shape=input_shape, return_sequences=return_sequences,
#                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
#                 first_layer = False
#             else:
#                 model.add(GRU(units, return_sequences=return_sequences,
#                               kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
#         model.add(LayerNormalization())
#         model.add(Dropout(dropout_rate))

#     # Add optional dense layer
#     if n_dense > 0:
#         model.add(Dense(dense_units, activation='relu'))
#         model.add(Dropout(dropout_rate))

#     model.add(Dense(1))

#     optimizer = Adam(learning_rate=lr)  # Simplified to Adam
#     model.compile(optimizer=optimizer, loss='mse',
#                   metrics=['RootMeanSquaredError', 'MeanAbsolutePercentageError'])
#     return model

# # Optuna objective function
# def objective(trial):
#     split = int(0.8 * len(X_train_seq))
#     X_tr = X_train_seq[:split]
#     y_tr = y_train_seq[:split]
#     X_val = X_train_seq[split:]
#     y_val = y_train_seq[split:]

#     input_shape = X_tr.shape[1:]
#     model = create_rnn_model(trial, input_shape)

#     batch_size = trial.suggest_categorical('batch_size', [32, 64])
#     es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

#     model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=batch_size,
#               callbacks=[es, reduce_lr], verbose=0)

#     preds = model.predict(X_val, verbose=0)
#     rmse = np.sqrt(mean_squared_error(y_val, preds.flatten()))
#     return rmse

# # Run Optuna study with reduced trials
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)  # Reduced from 200 to 50

# # Get best hyperparameters
# best_params = study.best_trial.params
# print("Best RNN trial:", best_params, "CV RMSE:", study.best_value)

# # Retrain best model on full training data
# input_shape = X_train_seq.shape[1:]
# model = create_rnn_model(study.best_trial, input_shape)
# batch_size = best_params['batch_size']
# es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)
# model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=batch_size,
#           callbacks=[es, reduce_lr], verbose=2)

# # Evaluate on test set
# preds_test = model.predict(X_test_seq, verbose=0)
# test_rmse = np.sqrt(mean_squared_error(y_test_seq, preds_test.flatten()))
# test_mape = mean_absolute_percentage_error(y_test_seq, preds_test.flatten())
# test_r2 = r2_score(y_test_seq, preds_test.flatten())

# print(f"Test RMSE: {test_rmse}")
# print(f"Test MAPE: {test_mape}")
# print(f"Test R²: {test_r2}")

# # Save the best model and metrics
# model.save('rnnmod_enhanced3.h5')
# joblib.dump({'rmse': test_rmse, 'mape': test_mape, 'r2': test_r2}, 'rnnmod_metrics.pkl')


# # Test RMSE: 0.4647909343017401


import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, LayerNormalization, Input, Attention, Conv1D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
from tensorflow.keras.models import Model
import joblib

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configure TensorFlow for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load data
X = pd.read_csv('Xrn_prepared.csv')
y = pd.read_csv('y1_prepared.csv')['percentreturn']

# Add lagged features
X['price_change_lag1'] = X['price_change'].shift(1)
X['volume_lag1'] = X['Volume'].shift(1)

# Exponential moving averages
X['ema_fast'] = X['Close'].ewm(span=12).mean()

X = X.dropna().values
y = y.iloc[1:].values  # Align with lagged features
y = winsorize(y, limits=[0.10, 0.10])  # 10% on both tails

# Split into train and test sets
N = len(X)
train_size = int(0.8 * N)
X_train_raw = X[:train_size, :]
y_train_raw = y[:train_size]
X_test_raw = X[train_size:, :]
y_test_raw = y[train_size:]

# Preprocessing pipeline
preprocessor = make_pipeline(
    IterativeImputer(max_iter=10, random_state=42),
    MinMaxScaler(),  # Switch to MinMaxScaler
    PCA(n_components=0.99)  # Retain more variance
)
preprocessor.fit(X_train_raw)
X_train_scaled = preprocessor.transform(X_train_raw)
X_test_scaled = preprocessor.transform(X_test_raw)

# Outlier clipping
y_train_raw = np.clip(y_train_raw, np.percentile(y_train_raw, 1), np.percentile(y_train_raw, 99))

# Sequence creation
T = 10  # Increased sequence length
def create_sequences(data, target, T):
    X_seq, y_seq = [], []
    for i in range(len(data) - T + 1):
        X_seq.append(data[i:i+T, :])
        y_seq.append(target[i+T-1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, T)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, T)

# Attention layer definition
def attention_layer(inputs):
    query = Dense(inputs.shape[-1])(inputs)
    value = Dense(inputs.shape[-1])(inputs)
    attention = Attention()([query, value])
    return attention

# RNN model builder
def create_rnn_model(model_type, units, n_layers, dropout_rate, lr, reg_strength, use_attention, input_shape):

    # Define input layer
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # Add RNN layers
    for i in range(n_layers):
        return_sequences = (i < n_layers - 1) or use_attention  # Return sequences for all but last layer unless using attention
        if model_type == 'LSTM':
            x = Bidirectional(LSTM(units, return_sequences=return_sequences,
                                   kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))(x)
        else:
            x = Bidirectional(GRU(units, return_sequences=return_sequences,
                                  kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    # Apply attention if enabled
    if use_attention:
        # Use last hidden state as query, full sequence as value
        query = x[:, -1, :]  # Shape: (batch, units*2)
        query = Lambda(lambda t: tf.expand_dims(t, axis=1))(query)  # Shape: (batch, 1, units*2)
        attention_output = Attention()([query, x])  # Shape: (batch, 1, units*2)
        x = Lambda(lambda t: tf.squeeze(t, axis=1))(attention_output)  # Shape: (batch, units*2)
    else:
        if n_layers > 0 :
            x = x  # Shape: (batch, units*2) since last layer doesn't return sequences

    # Output layer
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['RootMeanSquaredError', 'MeanAbsolutePercentageError'])
    return model

# Optuna objective
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU'])
    units = trial.suggest_categorical('units', [64, 128, 256])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Updated to suggest_float
    reg_strength = trial.suggest_float('reg_strength', 1e-5, 1e-2, log=True)  # Updated to suggest_float
    use_attention = trial.suggest_categorical('use_attention', [True, False])

    split = int(0.8 * len(X_train_seq))
    X_tr, y_tr = X_train_seq[:split], y_train_seq[:split]
    X_val, y_val = X_train_seq[split:], y_train_seq[split:]

    input_shape = X_tr.shape[1:]
    model = create_rnn_model(
        model_type=model_type,
        units=units,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        lr=lr,
        reg_strength=reg_strength,
        use_attention=use_attention,
        input_shape=input_shape
    )

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Add noise to training data
    noise = np.random.normal(0, 0.01, X_tr.shape)
    X_tr_noisy = X_tr + noise

    model.fit(X_tr_noisy, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=batch_size,
              callbacks=[es, reduce_lr], verbose=0)

    preds = model.predict(X_val, verbose=0)
    rmse = np.sqrt(mean_squared_error(y_val, preds.flatten()))
    return rmse

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # Increased trials

# Best hyperparameters
best_params = study.best_trial.params
print("Best RNN trial:", best_params, "CV RMSE:", study.best_value)

# Ensemble training
n_ensemble = 5
test_preds = []
for seed in range(n_ensemble):
    tf.random.set_seed(42 + seed)
    np.random.seed(42 + seed)
    model = create_rnn_model(
        model_type=best_params['model_type'],
        units=best_params['units'],
        n_layers=best_params['n_layers'],
        dropout_rate=best_params['dropout_rate'],
        lr=best_params['lr'],
        reg_strength=best_params['reg_strength'],
        use_attention=best_params['use_attention'],
        input_shape=X_train_seq.shape[1:]
    )
    model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=best_params['batch_size'],
              callbacks=[EarlyStopping(monitor='loss', patience=15, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6)],
              verbose=2)
    preds = model.predict(X_test_seq, verbose=0)
    test_preds.append(preds.flatten())

# Average ensemble predictions
final_preds = np.mean(test_preds, axis=0)

# Evaluate
test_rmse = np.sqrt(mean_squared_error(y_test_seq, final_preds))
test_mape = mean_absolute_percentage_error(y_test_seq, final_preds)
test_r2 = r2_score(y_test_seq, final_preds)

print(f"Test RMSE: {test_rmse}")
print(f"Test MAPE: {test_mape}")
print(f"Test R²: {test_r2}")

# Save model and metrics
model.save('rnnmod_optimized.h5')
joblib.dump({'rmse': test_rmse, 'mape': test_mape, 'r2': test_r2}, 'rnnmod_optimized_metrics.pkl')