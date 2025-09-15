import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, 
                                   MultiHeadAttention, GlobalAveragePooling1D,
                                   Conv1D, LSTM, GRU, Concatenate, Lambda)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from scipy import stats
from scipy.stats import mstats
import optuna  # Added for hyperparameter tuning
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class CryptoPredictor:
    def __init__(self, sequence_length=15, validation_method='walk_forward'):
        self.sequence_length = sequence_length
        self.validation_method = validation_method
        self.scaler = RobustScaler()
        self.feature_names = None
        self.models = {}
        
    def engineer_features(self, df):
        """
        Advanced feature engineering based on crypto market research
        """
        df = df.copy()
        
        # Basic price features (already provided)
        required_cols = ['Close', 'Volume', 'RSI', 'ATR', 'CCI', 'Momentum_10']
        
        # 1. Volatility regime features
        df['vol_regime_high'] = (df['rolling_vol_7'] > 
                                df['rolling_vol_7'].rolling(30).quantile(0.8)).astype(int)
        
        # 2. Multi-timeframe momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['Close'].pct_change(period)
            df[f'volume_momentum_{period}d'] = df['Volume'].pct_change(period)
        
        # 3. Technical strength indicators
        df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        df['rsi_momentum'] = df['RSI'].diff()
        
        # 4. Volume-price relationship
        df['volume_price_trend'] = np.sign(df['price_change']) * np.sign(df['Volume'].pct_change())
        df['price_change_lag1'] = df['price_change'].shift(1)
        df['volume_lag1'] = df['Volume'].shift(1)
        
        # 5. Market structure features
        df['price_position'] = (df['Close'] - df['Close'].rolling(20).min()) / \
                              (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
        
        # 6. Volatility clustering (GARCH-like features)
        df['vol_clustering'] = df['rolling_vol_7'].rolling(5).std()
        
        # 7. Regime detection features
        df['trend_strength'] = df['Close'].rolling(10).mean() / df['Close'].rolling(50).mean()
        df['volatility_breakout'] = (df['rolling_vol_7'] > 
                                   df['rolling_vol_7'].rolling(20).quantile(0.95)).astype(int)
        
        df['ema_fast'] = df['Close'].ewm(span=12).mean()
        df['ema_slow'] = df['Close'].ewm(span=26).mean()
        
        # New features: Lagged returns and rolling volatility
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag{lag}'] = df['percentreturn'].shift(lag)
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['percentreturn'].rolling(window).std().shift(1)
        
        # Drop close and close_lag_1 columns
        if 'Close' in df.columns:
            df = df.drop('Close', axis=1)
        if 'close_lag_1' in df.columns:
            df = df.drop('close_lag_1', axis=1)
        
        return df
    
    def feature_selection(self, X, y, k=15):
        """
        Select top k features based on mutual information
        """
        mi_scores = mutual_info_regression(X, y)
        return np.argsort(mi_scores)[::-1][:k]
    
    def create_sequences(self, X, y, sequence_length):
        """
        Create sequences for time series modeling
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def calculate_financial_metrics(self, y_true, y_pred):
        """
        Calculate key financial performance metrics
        """
        returns = y_pred
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        cumulative_returns = np.cumsum(returns)
        max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
        
        total_return = cumulative_returns[-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'volatility': volatility
        }
    
    def build_hybrid_model(self, input_shape, lstm_units1=64, lstm_units2=32, dropout_rate=0.2, num_heads=2, d_model=32):
        """
        Build hybrid model combining LSTM and Transformer with tunable hyperparameters
        """
        inputs = Input(shape=input_shape)
        
        # LSTM branch
        lstm_out = LSTM(lstm_units1, return_sequences=True, dropout=dropout_rate)(inputs)
        lstm_out = LSTM(lstm_units2, dropout=dropout_rate)(lstm_out)
        
        # Transformer branch
        x = Dense(d_model)(inputs)
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        transformer_out = GlobalAveragePooling1D()(x)
        
        # Combine branches
        combined = Concatenate()([lstm_out, transformer_out])
        combined = Dense(32, activation='relu')(combined)
        combined = Dropout(dropout_rate)(combined)
        outputs = Dense(1)(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def fit_and_evaluate(self, X, y, feature_names=None):
        """
        Complete model training and evaluation pipeline with hyperparameter tuning
        """
        self.feature_names = feature_names
        
        # Check data quality
        print("Data Quality Check:")
        print(f"Shape: {X.shape}")
        print(f"NaN values: {np.isnan(X).sum()}")
        print(f"Target NaN values: {np.isnan(y).sum()}")
        
        # Remove NaN values
        self.pipeline0 = make_pipeline(KNNImputer(n_neighbors=5))
        X_clean = self.pipeline0.fit_transform(X)

        # Handle NaN in target separately
        y_clean = y  # Still need to handle NaN in target separately
        print(f"Clean data shape: {X_clean.shape}")

        # Feature selection
        top_features = self.feature_selection(X_clean, y_clean, k=15)
        X_selected = X_clean[:, top_features]
        
        # Scale features
        self.pipeline2 = make_pipeline(RobustScaler())
        X_scaled = self.pipeline2.fit_transform(X_selected)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_clean, self.sequence_length)
        print(f"Sequence data shape: {X_seq.shape}")
        
        # Split data (80% train, 20% test)
        split_idx = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Further split train into train and validation for hyperparameter tuning
        val_split_idx = int(0.8 * len(X_train))
        X_tr, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_tr, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        # Hyperparameter tuning with Optuna
        def objective(trial):
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
            lstm_units1 = trial.suggest_int('lstm_units1', 32, 128)
            lstm_units2 = trial.suggest_int('lstm_units2', 16, 64)
            num_heads = trial.suggest_int('num_heads', 2, 8)
            d_model = trial.suggest_int('d_model', 16, 64)

            model = self.build_hybrid_model(X_tr.shape[1:], 
                                            lstm_units1=lstm_units1,
                                            lstm_units2=lstm_units2,
                                            dropout_rate=dropout_rate,
                                            num_heads=num_heads,
                                            d_model=d_model)

            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), 
                                epochs=50, batch_size=64, verbose=0, callbacks=[es])

            return min(history.history['val_loss'])

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

        best_params = study.best_params
        print("Best hyperparameters:", best_params)

        # Train final model with best hyperparameters
        final_model = self.build_hybrid_model(X_train.shape[1:], 
                                              lstm_units1=best_params['lstm_units1'],
                                              lstm_units2=best_params['lstm_units2'],
                                              dropout_rate=best_params['dropout_rate'],
                                              num_heads=best_params['num_heads'],
                                              d_model=best_params['d_model'])

        optimizer = Adam(learning_rate=best_params['learning_rate'])
        final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Train on full X_train with validation split
        val_split_idx_final = int(0.8 * len(X_train))
        X_tr_final, X_val_final = X_train[:val_split_idx_final], X_train[val_split_idx_final:]
        y_tr_final, y_val_final = y_train[:val_split_idx_final], y_train[val_split_idx_final:]

        es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

        history = final_model.fit(X_tr_final, y_tr_final, 
                                  validation_data=(X_val_final, y_val_final),
                                  epochs=200, batch_size=64, verbose=1, 
                                  callbacks=[es, reduce_lr])

        # Final evaluation on test set
        print("\nFinal Test Set Evaluation:")
        y_pred_test = final_model.predict(X_test, verbose=0).flatten()
        
        # Statistical metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Correlation and R²
        correlation = np.corrcoef(y_test, y_pred_test)[0, 1]
        r2_score = correlation**2
        
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test Correlation: {correlation:.4f}")
        print(f"Test R²: {r2_score:.4f}")
        
        # Financial metrics
        financial_metrics = self.calculate_financial_metrics(y_test, y_pred_test)
        print(f"\nFinancial Performance:")
        print(f"Sharpe Ratio: {financial_metrics['sharpe_ratio']:.4f}")
        print(f"Hit Rate: {financial_metrics['hit_rate']:.4f}")
        print(f"Max Drawdown: {financial_metrics['max_drawdown']:.4f}")
        print(f"Total Return: {financial_metrics['total_return']:.4f}")
        print(f"Volatility: {financial_metrics['volatility']:.4f}")
        
        # Directional accuracy
        directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred_test))
        print(f"Directional Accuracy: {directional_accuracy:.4f}")
        
        # Store model and results
        self.models['final'] = final_model
        self.results = {
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'correlation': correlation,
            'r2_score': r2_score,
            'directional_accuracy': directional_accuracy,
            'financial_metrics': financial_metrics,
            'feature_indices': top_features
        }
        
        return self.results

def main():
    """
    Main execution function
    """
    print("Loading data...")
    
    # Load data
    X_df = pd.read_csv('Xrn_prepared.csv')
    y_df = pd.read_csv('y1_prepared.csv')
    
    print("Original feature columns:", X_df.columns.tolist())
    
    # Initialize predictor
    predictor = CryptoPredictor(sequence_length=15, validation_method='walk_forward')
    
    # Engineer additional features
    print("Engineering features...")
    
    # Create a combined dataframe for feature engineering
    combined_df = X_df.copy()
    combined_df['percentreturn'] = y_df['percentreturn']
    
    # Apply feature engineering
    enhanced_df = predictor.engineer_features(combined_df)
    
    # Separate features and target
    feature_cols = [col for col in enhanced_df.columns if col != 'percentreturn']
    X_enhanced = enhanced_df[feature_cols].values
    y = enhanced_df['percentreturn'].values
    
    # Winsorize target
    from scipy.stats import mstats
    y = mstats.winsorize(y, limits=[0.10, 0.10])  # 10% on both tails
    
    # Train and evaluate model
    print("\nStarting model training and evaluation...")
    results = predictor.fit_and_evaluate(X_enhanced, y, feature_names=feature_cols)
    
    # Risk analysis
    print("\n" + "="*50)
    print("OVERFITTING ANALYSIS")
    print("="*50)
    
    realistic_r2_range = (0.02, 0.15)
    realistic_sharpe_range = (0.3, 1.2)
    
    print(f"R² Score: {results['r2_score']:.4f}")
    if realistic_r2_range[0] <= results['r2_score'] <= realistic_r2_range[1]:
        print("✓ R² is within realistic range for financial time series")
    else:
        print("⚠ R² may indicate overfitting - consider simplifying model")
    
    print(f"Sharpe Ratio: {results['financial_metrics']['sharpe_ratio']:.4f}")
    if realistic_sharpe_range[0] <= results['financial_metrics']['sharpe_ratio'] <= realistic_sharpe_range[1]:
        print("✓ Sharpe ratio is realistic")
    else:
        print("⚠ Sharpe ratio may be unrealistic")
    
    print(f"Directional Accuracy: {results['directional_accuracy']:.4f}")
    if 0.51 <= results['directional_accuracy'] <= 0.65:
        print("✓ Directional accuracy is realistic and valuable")
    else:
        print("⚠ Check directional accuracy - may indicate issues")
    
    # Save results
    import joblib
    joblib.dump(predictor, 'transformerlstm.pkl')
    joblib.dump(results, 'transformerlstmresults.pkl')
    
    print(f"\nModel and results saved successfully!")
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()

























# import numpy as np
# import pandas as pd
# import optuna
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import mean_squared_error
# from sklearn.pipeline import make_pipeline
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
# import joblib

# # Set random seed for reproducibility
# tf.random.set_seed(42)

# # Load data
# X = pd.read_csv('X_prepared.csv').values
# y = pd.read_csv('y1_prepared.csv')['percentreturn'].values

# # Split into train and test sets (80% train, 20% test)
# N = len(X)
# train_size = int(0.8 * N)
# X_train_original = X[0:train_size, :]
# y_train_original = y[0:train_size]
# X_test_original = X[train_size:, :]
# y_test_original = y[train_size:]

# # Define preprocessor
# preprocessor = make_pipeline(
#     KNNImputer(n_neighbors=5),
#     MinMaxScaler(),
#     PCA(n_components=0.95)
# )
# preprocessor.fit(X_train_original)
# X_train_scaled = preprocessor.transform(X_train_original)
# X_test_scaled = preprocessor.transform(X_test_original)

# # Define sequence length (T) for RNN
# T = 5

# # Function to create sequences
# def create_sequences(data, target, T):
#     X_seq = []
#     y_seq = []
#     for i in range(len(data) - T + 1):
#         X_seq.append(data[i:i+T, :])
#         y_seq.append(target[i+T-1])
#     return np.array(X_seq), np.array(y_seq)

# # Create sequences for train and test
# X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_original, T)
# X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_original, T)

# # Define RNN model builder with hyperparameter tuning
# def create_rnn_model(trial, input_shape):
#     model_type = trial.suggest_categorical('model_type', ['LSTM', 'BidirectionalLSTM'])
#     units = trial.suggest_categorical('units', [32, 64])
#     n_layers = trial.suggest_int('n_layers', 1, 2)
#     dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3])
#     lr = trial.suggest_loguniform('lr', 1e-4, 1e-3)
#     reg_strength = trial.suggest_loguniform('reg_strength', 1e-4, 1e-2)

#     model = Sequential()
#     first_layer = True
#     for _ in range(n_layers):
#         if model_type == 'LSTM':
#             if first_layer:
#                 model.add(LSTM(units, input_shape=input_shape, return_sequences=True if _ < n_layers - 1 else False,
#                                kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
#                 first_layer = False
#             else:
#                 model.add(LSTM(units, return_sequences=True if _ < n_layers - 1 else False,
#                                kernel_regularizer=tf.keras.regularizers.l2(reg_strength)))
#         elif model_type == 'BidirectionalLSTM':
#             if first_layer:
#                 model.add(Bidirectional(LSTM(units, input_shape=input_shape, return_sequences=True if _ < n_layers - 1 else False,
#                                              kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
#                 first_layer = False
#             else:
#                 model.add(Bidirectional(LSTM(units, return_sequences=True if _ < n_layers - 1 else False,
#                                              kernel_regularizer=tf.keras.regularizers.l2(reg_strength))))
#         model.add(Dropout(dropout_rate))
#     model.add(Dense(1))

#     optimizer = Adam(learning_rate=lr)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
#     return model

# # Optuna objective function
# def objective(trial):
#     # Split train into train-train and validation
#     split = int(0.8 * len(X_train_seq))
#     X_tr = X_train_seq[:split]
#     y_tr = y_train_seq[:split]
#     X_val = X_train_seq[split:]
#     y_val = y_train_seq[split:]

#     input_shape = X_tr.shape[1:]
#     model = create_rnn_model(trial, input_shape)

#     batch_size = trial.suggest_categorical('batch_size', [32, 64])
#     es = EarlyStopping(monitor='val_loss', patience=10)
#     model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=50, batch_size=batch_size,
#               callbacks=[es], verbose=0)

#     preds = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, preds.flatten()))
#     return rmse

# # Run Optuna study
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)

# # Get best hyperparameters
# best_params = study.best_trial.params
# print("Best RNN trial:", best_params, "CV RMSE:", study.best_value)

# # Retrain best RNN on full training data
# input_shape = X_train_seq.shape[1:]
# model = create_rnn_model(study.best_trial, input_shape)
# batch_size = best_params['batch_size']
# es = EarlyStopping(monitor='loss', patience=10)
# model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=batch_size, callbacks=[es], verbose=2)

# # Evaluate on test set
# preds_test = model.predict(X_test_seq)
# test_rmse = np.sqrt(mean_squared_error(y_test_seq, preds_test.flatten()))
# print(f"Test RMSE: {test_rmse}")

# # Save the best model
# model.save('rnnmod.h5')
# joblib.dump(test_rmse, 'rnnmod.h5')