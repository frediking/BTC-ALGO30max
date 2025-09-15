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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class CryptoPredictor:
    def __init__(self, sequence_length=20, validation_method='walk_forward'):
        self.sequence_length = sequence_length
        self.validation_method = validation_method
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
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
        # df['volume_weighted_return'] = df['percentreturn'] * np.log1p(df['Volume'])
        
        # 5. Market structure features
        df['price_position'] = (df['Close'] - df['Close'].rolling(20).min()) / \
                              (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
        
        # 6. Volatility clustering (GARCH-like features)
        df['vol_clustering'] = df['rolling_vol_7'].rolling(5).std()
        # df['return_volatility_ratio'] = np.abs(df['percentreturn']) / df['rolling_vol_7']
        
        # 7. Regime detection features
        df['trend_strength'] = df['Close'].rolling(10).mean() / df['Close'].rolling(50).mean()
        df['volatility_breakout'] = (df['rolling_vol_7'] > 
                                   df['rolling_vol_7'].rolling(20).quantile(0.95)).astype(int)
        
        return df
    
    def check_stationarity(self, series, name):
        """
        Check stationarity using Augmented Dickey-Fuller test
        """
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(series.dropna())
        print(f'{name} ADF Statistic: {result[0]:.6f}')
        print(f'{name} p-value: {result[1]:.6f}')
        
        if result[1] <= 0.05:
            print(f'{name} is stationary')
            return True
        else:
            print(f'{name} is non-stationary - consider differencing')
            return False
    
    def feature_selection(self, X, y, k=15):
        """
        Select top k features using mutual information
        """
        # Remove NaN values for feature selection
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
        
        # Select top k features
        top_features = np.argsort(mi_scores)[-k:]
        
        print("Top features by mutual information:")
        if self.feature_names is not None:
            for i, idx in enumerate(reversed(top_features)):
                print(f"{i+1}. {self.feature_names[idx]}: {mi_scores[idx]:.4f}")
        
        return top_features
    
    def create_sequences(self, data, target, sequence_length):
        """
        Create sequences for time series prediction with proper temporal ordering
        """
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(data)):
            # Use only past data (no future leakage)
            X_seq.append(data[i-sequence_length:i])
            y_seq.append(target[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_transformer_model(self, input_shape, d_model=64, num_heads=4, 
                              num_layers=2, ff_dim=128, dropout_rate=0.1):
        """
        Build Transformer model for time series prediction
        """
        inputs = Input(shape=input_shape)
        
        # Initial projection to d_model dimension
        x = Dense(d_model)(inputs)
        
        # Positional encoding
        seq_len = input_shape[0]
        position_encoding = self.get_positional_encoding(seq_len, d_model)
        x = x + position_encoding
        
        # Transformer blocks
        for _ in range(num_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=d_model,
                dropout=dropout_rate
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed Forward
            ffn_output = self.feed_forward_network(x, ff_dim, dropout_rate, d_model)
            
            # Add & Norm
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def get_positional_encoding(self, seq_len, d_model):
        """
        Create positional encoding for transformer
        """
        pos_encoding = np.zeros((seq_len, d_model))
        
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def feed_forward_network(self, x, ff_dim, dropout_rate, d_model):
        """
        Feed forward network for transformer
        Args:
            x: Input tensor
            ff_dim: Dimension of the feed-forward layer
            dropout_rate: Dropout rate
            d_model: Model dimension to project back to
        
        Returns:
            Tensor with shape [None, seq_len, d_model]
        """
        x = Dense(ff_dim, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(d_model)(x)
        return x
    
    def build_hybrid_model(self, input_shape, dropout_rate=0.2):
        """
        Build hybrid model combining LSTM and Transformer
        """
        inputs = Input(shape=input_shape)
        
        # LSTM branch
        lstm_out = LSTM(64, return_sequences=True, dropout=dropout_rate)(inputs)
        lstm_out = LSTM(32, dropout=dropout_rate)(lstm_out)
        
        # Transformer branch
        transformer_out = self.build_transformer_block(inputs, d_model=32, num_heads=2)
        transformer_out = GlobalAveragePooling1D()(transformer_out)
        
        # Combine branches
        combined = Concatenate()([lstm_out, transformer_out])
        combined = Dense(32, activation='relu')(combined)
        combined = Dropout(dropout_rate)(combined)
        outputs = Dense(1)(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_transformer_block(self, inputs, d_model, num_heads):
        """
        Single transformer block
        """
        x = Dense(d_model)(inputs)
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Dense(d_model * 2, activation='relu')(x)
        ffn_output = Dense(d_model)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    def walk_forward_validation(self, X, y, model_builder, n_splits=5):
        """
        Walk-forward validation for time series
        """
        n_samples = len(X)
        fold_size = n_samples // (n_splits + 1)  # Leave room for test set
        
        results = []
        
        for i in range(n_splits):
            # Define train and validation sets
            train_end = (i + 1) * fold_size
            val_start = train_end
            val_end = train_end + fold_size // 2
            
            if val_end >= n_samples:
                break
                
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            # Build and train model
            model = model_builder(X_train.shape[1:])
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='mse', metrics=['mae'])
            
            # Train with early stopping
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            
            history = model.fit(X_train, y_train, 
                              validation_data=(X_val, y_val),
                              epochs=100, batch_size=32, verbose=0,
                              callbacks=[es, reduce_lr])
            
            # Evaluate
            y_pred = model.predict(X_val, verbose=0).flatten()
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            # Directional accuracy
            directional_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
            
            results.append({
                'fold': i,
                'rmse': rmse,
                'mae': mae,
                'directional_accuracy': directional_acc,
                'train_size': len(X_train),
                'val_size': len(X_val)
            })
            
            print(f"Fold {i+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, Dir.Acc={directional_acc:.3f}")
        
        return results
    
    def calculate_financial_metrics(self, y_true, y_pred):
        """
        Calculate financial performance metrics
        """
        # Convert predictions to trading signals
        signals = np.sign(y_pred)
        returns = y_true * signals  # Strategy returns
        
        # Risk-adjusted metrics
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdown)
        
        # Hit rate
        hit_rate = np.mean((y_true > 0) == (y_pred > 0))
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'total_return': np.prod(1 + returns) - 1,
            'volatility': np.std(returns) * np.sqrt(252)
        }
    
    def fit_and_evaluate(self, X, y, feature_names=None):
        """
        Complete model training and evaluation pipeline
        """
        self.feature_names = feature_names
        
        # Check data quality
        print("Data Quality Check:")
        print(f"Shape: {X.shape}")
        print(f"NaN values: {np.isnan(X).sum()}")
        print(f"Target NaN values: {np.isnan(y).sum()}")
        
        # Remove NaN values
        # mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        # X_clean = X
        # y_clean = y
        # print(f"Clean data shape: {X_clean.shape}")
        # print(f"Rows removed due to NaNs: {len(X) - len(X_clean)}")

        # if len(X_clean) <= self.sequence_length:
        #     raise ValueError("Insufficient data after NaN removal to create sequences.")
        
        # Check stationarity of target
        # self.check_stationarity(pd.Series(y_clean), 'Target (percentreturn)')
        
        # # Feature selection
        # top_features = self.feature_selection(X_clean, y_clean, k=12)
        # X_selected = X_clean[:, top_features]
        
        # # Scale features
        # X_scaled = self.scaler.fit_transform(X_selected)

        # # Create sequences
        # X_seq, y_seq = self.create_sequences(X_scaled, y_clean, self.sequence_length)
        # print(f"Sequence data shape: {X_seq.shape}")
        
        # # Split data (80% train, 20% test)
        # split_idx = int(0.8 * len(X_seq))
        # X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        # y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        

        # Use the entire dataset (IterativeImputer will handle NaNs)
        X_clean = X
        y_clean = y
        
        # Feature selection (select top k features)
        top_features = self.feature_selection(X_clean, y_clean, k=12)
        X_selected = X_clean[:, top_features]
        
        # Split data into training and test sets (80% train, 20% test)
        split_idx = int(0.8 * len(X_selected))
        X_train_raw, X_test_raw = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        # Define and apply the preprocessing pipeline
        self.preprocessor = make_pipeline(
            IterativeImputer(max_iter=10, random_state=42),
            RobustScaler(),  # Switch to MinMaxScaler
            PCA(n_components=0.99)  # Retain 99% of variance
        )
        
        # Fit the pipeline on the training data to avoid data leakage
        self.preprocessor.fit(X_train_raw)
        
        # Transform both training and test data
        X_train_scaled = self.preprocessor.transform(X_train_raw)
        X_test_scaled = self.preprocessor.transform(X_test_raw)
        
        # Create sequences from the preprocessed data
        X_seq, y_seq = self.create_sequences(
            np.vstack((X_train_scaled, X_test_scaled)),
            np.concatenate((y_train, y_test)),
            self.sequence_length
        )
        
        # Split sequences into training and test sets
        seq_split_idx = int(0.8 * len(X_seq))
        X_train_seq, X_test_seq = X_seq[:seq_split_idx], X_seq[seq_split_idx:]
        y_train_seq, y_test_seq = y_seq[:seq_split_idx], y_seq[seq_split_idx:]
        
        print(f"Train sequence shape: {X_train_seq.shape}, Test sequence shape: {X_test_seq.shape}")
        
        print(f"Train shape: {X_train_seq.shape}, Test shape: {X_test_seq.shape}")


        
        # Walk-forward validation on training data
        print("\nWalk-Forward Validation:")
        
        # Test different model architectures
        model_configs = {
            'transformer': lambda shape: self.build_transformer_model(shape, d_model=32, num_heads=2),
            'hybrid': lambda shape: self.build_hybrid_model(shape, dropout_rate=0.3)
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model_builder in model_configs.items():
            print(f"\nTesting {name} model:")
            cv_results = self.walk_forward_validation(X_train_seq, y_train_seq, model_builder, n_splits=3)
            
            avg_rmse = np.mean([r['rmse'] for r in cv_results])
            avg_dir_acc = np.mean([r['directional_accuracy'] for r in cv_results])
            
            print(f"Average CV RMSE: {avg_rmse:.4f}")
            print(f"Average Directional Accuracy: {avg_dir_acc:.3f}")
            
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_model = (name, model_builder)
        
        # Train final model on all training data
        print(f"\nTraining final {best_model[0]} model...")
        final_model = best_model[1](X_train_seq.shape[1:])
        final_model.compile(optimizer=Adam(learning_rate=0.001), 
                           loss='mse', metrics=['mae'])
        
        # Use 20% of training data for validation
        val_split_idx = int(0.8 * len(X_train_seq))
        X_tr, X_val = X_train_seq[:val_split_idx], X_train_seq[val_split_idx:]
        y_tr, y_val = y_train_seq[:val_split_idx], y_train_seq[val_split_idx:]
        
        # Train with early stopping
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
        
        history = final_model.fit(X_tr, y_tr,
                                 validation_data=(X_val, y_val),
                                 epochs=200, batch_size=32, verbose=1,
                                 callbacks=[es, reduce_lr])
        
        # Final evaluation on test set
        print("\nFinal Test Set Evaluation:")
        y_pred_test = final_model.predict(X_test_seq, verbose=0).flatten()
        
        # Statistical metrics
        test_rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred_test))
        test_mae = mean_absolute_error(y_test_seq, y_pred_test)
        
        # Correlation and R²
        correlation = np.corrcoef(y_test_seq, y_pred_test)[0, 1]
        r2_score = correlation**2
        
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test Correlation: {correlation:.4f}")
        print(f"Test R²: {r2_score:.4f}")
        
        # Financial metrics
        financial_metrics = self.calculate_financial_metrics(y_test_seq, y_pred_test)
        print(f"\nFinancial Performance:")
        print(f"Sharpe Ratio: {financial_metrics['sharpe_ratio']:.4f}")
        print(f"Hit Rate: {financial_metrics['hit_rate']:.4f}")
        print(f"Max Drawdown: {financial_metrics['max_drawdown']:.4f}")
        print(f"Total Return: {financial_metrics['total_return']:.4f}")
        print(f"Volatility: {financial_metrics['volatility']:.4f}")
        
        # Directional accuracy
        directional_accuracy = np.mean(np.sign(y_test_seq) == np.sign(y_pred_test))
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
    
    def predict(self, X_new):
        """
        Make predictions on new data using the trained preprocessing pipeline
        """
        if 'final' not in self.models:
            raise ValueError("Model must be trained first")
        
        # Apply same feature selection
        X_selected = X_new[:, self.results['feature_indices']]
        
        # Transform new data using fitted preprocessor
        X_scaled = self.preprocessor.transform(X_selected)
        
        # Create sequences
        X_seq, _ = self.create_sequences(X_scaled, 
                                       np.zeros(len(X_scaled)), 
                                       self.sequence_length)
        
        # Make predictions
        predictions = self.models['final'].predict(X_seq, verbose=0).flatten()
        
        return predictions


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
    
    print(f"Enhanced features: {len(feature_cols)}")
    print("New feature columns:", feature_cols)
    
    # Outlier treatment for target
    q1, q99 = np.percentile(y, [1, 99])
    y_clipped = np.clip(y, q1, q99)
    print(f"Target clipped from range [{y.min():.4f}, {y.max():.4f}] to [{y_clipped.min():.4f}, {y_clipped.max():.4f}]")
    
    # Train and evaluate model
    print("\nStarting model training and evaluation...")
    results = predictor.fit_and_evaluate(X_enhanced, y_clipped, feature_names=feature_cols)
    
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
    joblib.dump(predictor, 'crypto_predictor_optimized.pkl')
    joblib.dump(results, 'crypto_results_optimized.pkl')
    
    print(f"\nModel and results saved successfully!")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()