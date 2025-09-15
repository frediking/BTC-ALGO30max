import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, 
                                   MultiHeadAttention, GlobalAveragePooling1D,
                                   Conv1D, LSTM, GRU, Concatenate, Lambda, 
                                   BatchNormalization, LeakyReLU, Add)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class OptimizedCryptoPredictor:
    def __init__(self, sequence_length=30, validation_method='purged_walk_forward'):
        self.sequence_length = sequence_length
        self.validation_method = validation_method
        self.feature_scaler = RobustScaler()
        self.target_scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
        self.feature_names = None
        self.models = {}
        self.feature_importance = {}
        
    def advanced_feature_engineering(self, df):
        """
        Enhanced feature engineering with financial domain knowledge
        """
        df = df.copy()
        
        # 1. Temporal features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['hour'] = df['date'].dt.hour if hasattr(df['date'].dt, 'hour') else 0
        
        # 2. Price-based features with multiple timeframes
        price_col = 'Close' if 'Close' in df.columns else [col for col in df.columns if 'close' in col.lower()][0]
        if price_col in df.columns:
            for window in [3, 5, 10, 20, 50]:
                # Returns and momentum
                df[f'return_{window}d'] = df[price_col].pct_change(window)
                df[f'log_return_{window}d'] = np.log(df[price_col] / df[price_col].shift(window))
                
                # Moving averages and ratios
                df[f'ma_{window}'] = df[price_col].rolling(window).mean()
                df[f'price_to_ma_{window}'] = df[price_col] / df[f'ma_{window}']
                
                # Volatility measures
                df[f'volatility_{window}d'] = df[price_col].pct_change().rolling(window).std()
                df[f'realized_vol_{window}d'] = np.sqrt(df[f'return_{window}d'].rolling(window).var() * 252)
        
        # 3. Advanced technical indicators
        if 'Volume' in df.columns:
            # Volume-weighted features
            df['vwap_20'] = (df[price_col] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
            df['price_to_vwap'] = df[price_col] / df['vwap_20']
            
            # Volume momentum and anomalies
            df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).quantile(0.95)).astype(int)
            
            # On-balance volume
            df['obv'] = (df['Volume'] * np.sign(df[price_col].diff())).cumsum()
            df['obv_ma'] = df['obv'].rolling(20).mean()
        
        # 4. Enhanced RSI features
        if 'RSI' in df.columns:
            df['rsi_momentum'] = df['RSI'].diff()
            df['rsi_divergence'] = (df['RSI'].diff() * df[price_col].pct_change() < 0).astype(int)
            df['rsi_extreme'] = ((df['RSI'] < 20) | (df['RSI'] > 80)).astype(int)
            df['rsi_normalized'] = (df['RSI'] - 50) / 50  # Normalize to [-1, 1]
        
        # 5. Volatility regime detection
        if price_col in df.columns:
            returns = df[price_col].pct_change()
            df['vol_regime'] = returns.rolling(20).std().rolling(5).mean()
            df['vol_breakout'] = (df['vol_regime'] > df['vol_regime'].rolling(50).quantile(0.8)).astype(int)
            df['garch_vol'] = returns.ewm(alpha=0.06).std()  # GARCH-like volatility
        
        # 6. Market microstructure features
        if 'High' in df.columns and 'Low' in df.columns:
            # True Range and ATR improvements
            df['hl_ratio'] = df['High'] / df['Low']
            df['price_range_pct'] = (df['High'] - df['Low']) / df[price_col]
            
            if 'ATR' in df.columns:
                df['atr_ratio'] = df['ATR'] / df[price_col]
                df['atr_momentum'] = df['ATR'].pct_change()
        
        # 7. Cross-asset and macro features (if available)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != price_col and not col.startswith(('return_', 'ma_', 'vol_')):
                # Feature interactions
                df[f'{col}_momentum'] = df[col].pct_change()
                df[f'{col}_zscore'] = (df[col] - df[col].rolling(20).mean()) / df[col].rolling(20).std()
        
        # 8. Regime and trend features
        if price_col in df.columns:
            # Trend strength
            short_ma = df[price_col].rolling(10).mean()
            long_ma = df[price_col].rolling(50).mean()
            df['trend_strength'] = (short_ma - long_ma) / long_ma
            df['trend_direction'] = np.sign(df['trend_strength'])
            
            # Support and resistance levels
            df['resistance_level'] = df[price_col].rolling(20).max()
            df['support_level'] = df[price_col].rolling(20).min()
            df['price_position'] = (df[price_col] - df['support_level']) / (df['resistance_level'] - df['support_level'])
        
        # 9. Lagged features for temporal dependencies
        important_features = [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'volume', 'return', 'volatility'])]
        for col in important_features[:5]:  # Limit to avoid too many features
            if df[col].dtype in [np.float64, np.int64]:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)
        
        # Clean up - remove original Close if it exists
        columns_to_drop = ['Close', 'close_lag_1', 'date'] if 'date' in df.columns else ['Close', 'close_lag_1']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1, errors='ignore')
        
        return df
    
    def robust_feature_selection(self, X, y, max_features=20):
        """
        Multi-method feature selection for robustness
        """
        print("Performing robust feature selection...")
        
        # Remove samples with NaN values
        # self.pipeline0 = make_pipeline(
        #     KNNImputer(n_neighbors=3))
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 100:
            print("Warning: Very few samples after cleaning")
            return np.arange(min(max_features, X.shape[1]))
        
        # Method 1: Mutual Information
        mi_selector = SelectKBest(mutual_info_regression, k=min(max_features, X_clean.shape[1]))
        mi_selector.fit(X_clean, y_clean)
        mi_scores = mi_selector.scores_
        
        # Method 2: F-statistics
        f_selector = SelectKBest(f_regression, k=min(max_features, X_clean.shape[1]))
        f_selector.fit(X_clean, y_clean)
        f_scores = f_selector.scores_
        
        # Method 3: Correlation (Spearman for non-linear relationships)
        corr_scores = []
        for i in range(X_clean.shape[1]):
            corr, _ = spearmanr(X_clean[:, i], y_clean)
            corr_scores.append(abs(corr) if not np.isnan(corr) else 0)
        corr_scores = np.array(corr_scores)
        
        # Combine scores with weighted average
        # Normalize scores to [0, 1] range
        mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
        f_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
        corr_norm = corr_scores / (corr_scores.max() + 1e-8)
        
        # Weighted combination
        combined_scores = 0.4 * mi_norm + 0.3 * f_norm + 0.3 * corr_norm
        
        # Select top features
        top_features = np.argsort(combined_scores)[-max_features:]
        
        # Print feature importance
        print("Top features by combined score:")
        if self.feature_names is not None:
            for i, idx in enumerate(reversed(top_features)):
                print(f"{i+1}. {self.feature_names[idx]}: {combined_scores[idx]:.4f}")
        
        self.feature_importance = {
            'mi_scores': mi_scores,
            'f_scores': f_scores, 
            'corr_scores': corr_scores,
            'combined_scores': combined_scores,
            'selected_features': top_features
        }
        
        return top_features
    
    def create_sequences_with_gaps(self, data, target, sequence_length, gap=1):
        """
        Create sequences with temporal gaps to reduce overfitting
        """
        X_seq, y_seq = [], []
        
        for i in range(sequence_length + gap, len(data)):
            # Use data from [i-sequence_length-gap:i-gap] to predict target at i
            X_seq.append(data[i-sequence_length-gap:i-gap])
            y_seq.append(target[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_residual_block(self, x, filters, dropout_rate=0.1):
        """
        Residual block for deeper networks
        """
        # First layer
        y = Dense(filters, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        y = BatchNormalization()(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(dropout_rate)(y)
        
        # Second layer
        y = Dense(filters, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(y)
        y = BatchNormalization()(y)
        
        # Skip connection if dimensions match
        if x.shape[-1] == filters:
            y = Add()([x, y])
        
        y = LeakyReLU(alpha=0.1)(y)
        return y
    
    def build_improved_model(self, input_shape, model_type='ensemble'):
        """
        Build improved model architecture
        """
        inputs = Input(shape=input_shape)
        
        if model_type == 'cnn_lstm':
            # CNN for local patterns
            x = Conv1D(64, 3, padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = Dropout(0.2)(x)
            
            x = Conv1D(32, 3, padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            
            # LSTM for temporal dependencies
            x = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(x)
            x = LSTM(32, dropout=0.2, recurrent_dropout=0.1)(x)
            
        elif model_type == 'attention':
            # Multi-head attention
            x = Dense(64)(inputs)
            attn = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
            x = LayerNormalization()(Add()([x, attn]))
            
            # Feed forward
            ff = Dense(128, activation='relu')(x)
            ff = Dropout(0.2)(ff)
            ff = Dense(64)(ff)
            x = LayerNormalization()(Add()([x, ff]))
            
            x = GlobalAveragePooling1D()(x)
            
        elif model_type == 'ensemble':
            # Branch 1: CNN
            cnn_branch = Conv1D(32, 3, padding='same')(inputs)
            cnn_branch = BatchNormalization()(cnn_branch)
            cnn_branch = LeakyReLU(alpha=0.1)(cnn_branch)
            cnn_branch = GlobalAveragePooling1D()(cnn_branch)
            
            # Branch 2: LSTM
            lstm_branch = LSTM(32, dropout=0.2)(inputs)
            
            # Branch 3: Dense with residual connections
            dense_branch = GlobalAveragePooling1D()(inputs)
            dense_branch = self.build_residual_block(dense_branch, 32, 0.2)
            
            # Combine branches
            x = Concatenate()([cnn_branch, lstm_branch, dense_branch])
        
        else:  # Simple dense
            x = GlobalAveragePooling1D()(inputs)
            x = self.build_residual_block(x, 64, 0.3)
            x = self.build_residual_block(x, 32, 0.2)
        
        # Output layers with residual connection
        x = Dense(16, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(1, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def purged_walk_forward_validation(self, X, y, model_builder, n_splits=5, purge_length=5):
        """
        Purged walk forward validation to prevent lookahead bias
        """
        n_samples = len(X)
        fold_size = n_samples // (n_splits + 2)  # Leave room for purging and test
        
        results = []
        
        for i in range(n_splits):
            # Define boundaries
            train_start = 0
            train_end = (i + 1) * fold_size
            
            # Purge period to prevent lookahead
            val_start = train_end + purge_length
            val_end = val_start + fold_size // 2
            
            if val_end >= n_samples:
                break
            
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            print(f"Fold {i+1}: Train[{train_start}:{train_end}], Val[{val_start}:{val_end}]")
            
            # Build model
            model = model_builder(X_train.shape[1:])
            model.compile(
                optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                loss='huber',  # More robust to outliers than MSE
                metrics=['mae']
            )
            
            # Callbacks
            es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-6)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-7)
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200, batch_size=64, verbose=0,
                callbacks=[es, reduce_lr]
            )
            
            # Predict and evaluate
            y_pred = model.predict(X_val, verbose=0).flatten()
            
            # Multiple metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            # Robust correlation (Spearman)
            corr_pearson = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0
            corr_spearman, _ = spearmanr(y_val, y_pred) if len(y_val) > 1 else (0, 1)
            
            # Directional accuracy
            dir_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
            
            # Information Coefficient (rank correlation)
            ic = corr_spearman
            
            results.append({
                'fold': i,
                'rmse': rmse,
                'mae': mae,
                'pearson_corr': corr_pearson if not np.isnan(corr_pearson) else 0,
                'spearman_corr': corr_spearman if not np.isnan(corr_spearman) else 0,
                'directional_accuracy': dir_acc,
                'information_coefficient': ic if not np.isnan(ic) else 0,
                'train_size': len(X_train),
                'val_size': len(X_val)
            })
            
            print(f"  RMSE: {rmse:.4f}, Spearman: {corr_spearman:.4f}, Dir.Acc: {dir_acc:.3f}, IC: {ic:.4f}")
        
        return results
    

    def calculate_enhanced_financial_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive financial performance metrics
        """
        import numpy as np
        
        # Convert to trading signals (long/short based on prediction sign)
        signals = np.sign(y_pred)
        
        # Strategy returns
        strategy_returns = y_true * signals
        
        # Handle edge cases
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'information_ratio': 0.0,
                'max_drawdown': 0.0,
                'hit_rate': 0.5,
                'profit_factor': 1.0,
                'total_return': 0.0,
                'volatility': 0.0,
                'sortino_ratio': 0.0
            }
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = -np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Basic metrics
        total_return = cumulative_returns[-1] - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        mean_return = np.mean(strategy_returns) * 252
        
        # Sharpe ratio
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        
        # Sortino ratio (downside risk)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = mean_return / downside_volatility if downside_volatility > 0 else 0.0
        
        # # Maximum drawdown
        # cumulative_returns = np.cumprod(1 + strategy_returns)
        # rolling_max = np.maximum.accumulate(cumulative_returns)
        # drawdown = (cumulative_returns - rolling_max) / rolling_max
        # max_drawdown = -np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Hit rate
        hit_rate = np.mean((y_true > 0) == (y_pred > 0))
        
        # Profit factor
        positive_returns = strategy_returns[strategy_returns > 0].sum()
        negative_returns = -strategy_returns[strategy_returns < 0].sum()
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else 1.0
        
        # Information ratio (assuming y_true as benchmark returns)
        excess_returns = strategy_returns - y_true
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = (mean_return - np.mean(y_true) * 252) / tracking_error if tracking_error > 0 else 0.0
        
        # Return metrics in a dictionary
        return {
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'volatility': volatility,
            'sortino_ratio': sortino_ratio
        }

    def evaluate_realism(self, results):
        """
        Evaluate the realism of the model's performance metrics.
        """
        # Calculate R² from Pearson correlation
        r2_score = results['pearson_correlation']**2
        
        # Define realistic ranges for financial metrics
        realistic_r2_range = (0.02, 0.15)
        realistic_sharpe_range = (0.3, 1.2)
        realistic_directional_accuracy = (0.51, 0.65)
        
        print("\n" + "="*50)
        print("REALISM CHECK")
        print("="*50)
        
        # R² analysis
        print(f"R² Score: {r2_score:.4f}")
        if realistic_r2_range[0] <= r2_score <= realistic_r2_range[1]:
            print("✓ R² is within realistic range for financial time series")
        else:
            print("⚠ R² may indicate overfitting or underfitting - review model complexity")
        
        # Sharpe Ratio analysis
        sharpe_ratio = results['financial_metrics']['sharpe_ratio']
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        if realistic_sharpe_range[0] <= sharpe_ratio <= realistic_sharpe_range[1]:
            print("✓ Sharpe ratio is realistic")
        else:
            print("⚠ Sharpe ratio may be unrealistic")


    def evaluate_realism(self, results):
        """
        Evaluate the realism of the model's performance metrics.
        """
        # Calculate R² from Pearson correlation
        r2_score = results['pearson_correlation']**2
        
        # Define realistic ranges for financial metrics
        realistic_r2_range = (0.02, 0.15)
        realistic_sharpe_range = (0.3, 1.2)
        realistic_directional_accuracy = (0.51, 0.65)
        
        print("\n" + "="*50)
        print("REALISM CHECK")
        print("="*50)
        
        # R² analysis
        print(f"R² Score: {r2_score:.4f}")
        if realistic_r2_range[0] <= r2_score <= realistic_r2_range[1]:
            print("✓ R² is within realistic range for financial time series")
        else:
            print("⚠ R² may indicate overfitting or underfitting - review model complexity")
        
        # Sharpe Ratio analysis
        sharpe_ratio = results['financial_metrics']['sharpe_ratio']
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        if realistic_sharpe_range[0] <= sharpe_ratio <= realistic_sharpe_range[1]:
            print("✓ Sharpe ratio is realistic")
        else:
            print("⚠ Sharpe ratio may be unrealistic")
        
        # Directional Accuracy analysis
        directional_accuracy = results['directional_accuracy']
        print(f"Directional Accuracy: {directional_accuracy:.4f}")
        if realistic_directional_accuracy[0] <= directional_accuracy <= realistic_directional_accuracy[1]:
            print("✓ Directional accuracy is realistic and valuable")
        else:
            print("⚠ Check directional accuracy - may indicate issues")

    
    def fit_and_evaluate(self, X, y, feature_names=None):
        """
        Complete optimized training pipeline
        """
        self.feature_names = feature_names
        
        print("="*60)
        print("OPTIMIZED CRYPTO PREDICTOR - TRAINING PIPELINE")
        print("="*60)
        
        # Data quality analysis
        print("\n1. DATA QUALITY ANALYSIS:")
        print(f"Original shape: {X.shape}")
        print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"NaN in features: {np.isnan(X).sum()}")
        print(f"NaN in target: {np.isnan(y).sum()}")
        
        # Handle missing values more carefully
        X = np.where(np.isinf(X), np.nan, X)

        # Clip extremely large values to a finite range
        X = np.clip(X, -1e10, 1e10)

        imputer = KNNImputer(n_neighbors=3, weights='distance')
        X_imputed = imputer.fit_transform(X)
        
        # Remove samples where target is NaN
        valid_mask = ~np.isnan(y)
        X_clean = X_imputed[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"Clean data shape: {X_clean.shape}")
        
        # Target preprocessing - use quantile transformation for better normalization
        y_reshaped = y_clean.reshape(-1, 1)
        y_normalized = self.target_scaler.fit_transform(y_reshaped).flatten()
        
        # Clip extreme outliers (beyond 3 standard deviations)
        y_std = np.std(y_normalized)
        y_clipped = np.clip(y_normalized, -3*y_std, 3*y_std)
        
        print(f"Target after normalization: [{y_clipped.min():.4f}, {y_clipped.max():.4f}]")
        
        # Feature selection
        print("\n2. FEATURE SELECTION:")
        top_features = self.robust_feature_selection(X_clean, y_clipped, max_features=15)
        X_selected = X_clean[:, top_features]
        
        # Feature scaling
        X_scaled = self.feature_scaler.fit_transform(X_selected)
        
        # Create sequences with gap to reduce overfitting
        print("\n3. SEQUENCE CREATION:")
        X_seq, y_seq = self.create_sequences_with_gaps(
            X_scaled, y_clipped, self.sequence_length, gap=2
        )
        print(f"Sequence data shape: {X_seq.shape}")
        
        if len(X_seq) < 100:
            print("Warning: Very few sequences created. Consider reducing sequence_length.")
            return None
        
        # Split data with proper temporal ordering
        train_pct = 0.7
        val_pct = 0.15
        # test_pct = 0.15
        
        train_size = int(train_pct * len(X_seq))
        val_size = int(val_pct * len(X_seq))
        
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]
        X_val = X_seq[train_size:train_size + val_size]
        y_val = y_seq[train_size:train_size + val_size]
        X_test = X_seq[train_size + val_size:]
        y_test = y_seq[train_size + val_size:]
        
        print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        # Model selection with cross-validation
        print("\n4. MODEL SELECTION & CROSS-VALIDATION:")
        
        model_configs = {
            'ensemble': lambda shape: self.build_improved_model(shape, 'ensemble'),
            'cnn_lstm': lambda shape: self.build_improved_model(shape, 'cnn_lstm'),
            'attention': lambda shape: self.build_improved_model(shape, 'attention'),
        }
        
        best_model_name = None
        best_ic = -1  # Information Coefficient
        cv_results_all = {}
        
        for name, model_builder in model_configs.items():
            print(f"\nTesting {name} model:")
            cv_results = self.purged_walk_forward_validation(
                X_train, y_train, model_builder, n_splits=3, purge_length=3
            )
            
            # Calculate average metrics
            avg_metrics = {
                'rmse': np.mean([r['rmse'] for r in cv_results]),
                'mae': np.mean([r['mae'] for r in cv_results]),
                'spearman_corr': np.mean([r['spearman_corr'] for r in cv_results]),
                'directional_accuracy': np.mean([r['directional_accuracy'] for r in cv_results]),
                'information_coefficient': np.mean([r['information_coefficient'] for r in cv_results])
            }
            
            cv_results_all[name] = avg_metrics
            
            print(f"Average IC: {avg_metrics['information_coefficient']:.4f}")
            print(f"Average Spearman: {avg_metrics['spearman_corr']:.4f}")
            print(f"Average Dir.Acc: {avg_metrics['directional_accuracy']:.3f}")
            
            # Select best model based on Information Coefficient
            if avg_metrics['information_coefficient'] > best_ic:
                best_ic = avg_metrics['information_coefficient']
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} (IC: {best_ic:.4f})")
        
        # Train final model
        print("\n5. FINAL MODEL TRAINING:")
        final_model = model_configs[best_model_name](X_train.shape[1:])
        final_model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss='huber',
            metrics=['mae']
        )
        
        # Callbacks for final training
        es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, min_delta=1e-6)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, min_lr=1e-7)
        checkpoint = ModelCheckpoint('best_crypto_model.h5', save_best_only=True, monitor='val_loss')
        
        # Train final model
        history = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300, batch_size=64, verbose=1,
            callbacks=[es, reduce_lr, checkpoint]
        )
        
        # Final evaluation
        print("\n6. FINAL TEST EVALUATION:")
        y_pred_test = final_model.predict(X_test, verbose=0).flatten()
        
        # Inverse transform predictions and targets for interpretable metrics
        y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = self.target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
        
        # Comprehensive metrics
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        test_mae = mean_absolute_error(y_test_orig, y_pred_orig)
        
        # Correlations
        pearson_corr = np.corrcoef(y_test_orig, y_pred_orig)[0, 1]
        spearman_corr, _ = spearmanr(y_test_orig, y_pred_orig)
        
        # Directional accuracy
        directional_accuracy = np.mean(np.sign(y_test_orig) == np.sign(y_pred_orig))
        
        # Financial metrics
        financial_metrics = self.calculate_enhanced_financial_metrics(y_test_orig, y_pred_orig)
        
        # Results summary
        results = {
            'model_name': best_model_name,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'pearson_correlation': pearson_corr if not np.isnan(pearson_corr) else 0,
            'spearman_correlation': spearman_corr if not np.isnan(spearman_corr) else 0,
            'directional_accuracy': directional_accuracy,
            'information_coefficient': spearman_corr if not np.isnan(spearman_corr) else 0,
            'financial_metrics': financial_metrics,
            'feature_indices': top_features,
            'cv_results': cv_results_all,
            'sequence_length': self.sequence_length
        }
        
        # Print results
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        print(f"Information Coefficient: {spearman_corr:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.4f}")
        
        print(f"\nFinancial Metrics:")
        for key, value in financial_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Realistic performance check
        self.evaluate_realism(results)
        
        # Store model and results
        self.models['final'] = final_model
        self.results = results
        
        return results
    


    

    def predict(self, X_new):
        """
        Make predictions on new data
        """
        if 'final' not in self.models:
            raise ValueError("Model must be trained first")
        
        # Apply same preprocessing
        X_selected = X_new[:, self.results['feature_indices']]
        X_scaled = self.scaler.transform(X_selected)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)), self.sequence_length)
        
        predictions = self.models['final'].predict(X_seq, verbose=0).flatten()
        return predictions
    

def main():
    """
    Main execution function for the OptimizedCryptoPredictor
    """
    print("Loading data...")

    # Load data
    X_df = pd.read_csv('Xrn_prepared.csv')
    y_df = pd.read_csv('y1_prepared.csv')

    print("Original feature columns:", X_df.columns.tolist())

    # Initialize predictor
    predictor = OptimizedCryptoPredictor(sequence_length=30, validation_method='purged_walk_forward')

    # Engineer additional features
    print("Engineering features...")

    # Create a combined dataframe for feature engineering
    combined_df = X_df.copy()
    combined_df['percentreturn'] = y_df['percentreturn']

    # Apply advanced feature engineering
    enhanced_df = predictor.advanced_feature_engineering(combined_df)

    print(f"Infinite values: {np.isinf(X_df).sum()}")
    print(f"Max value: {np.max(np.abs(X_df.values))}")

    # Separate features and target
    feature_cols = [col for col in enhanced_df.columns if col != 'percentreturn']
    X_enhanced = enhanced_df[feature_cols].values
    y = enhanced_df['percentreturn'].values

    print(f"Enhanced features: {len(feature_cols)}")
    print("New feature columns:", feature_cols)

    # Verify basic data quality before processing
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"NaN values in target: {np.isnan(y).sum()}")

    # Train and evaluate model
    print("\nStarting model training and evaluation...")
    results = predictor.fit_and_evaluate(X_enhanced, y, feature_names=feature_cols)

    # Overfitting analysis
    print("\n" + "="*50)
    print("OVERFITTING ANALYSIS")
    print("="*50)

    # Calculate R² from Pearson correlation
    r2_score = results['pearson_correlation']**2

    # Define realistic ranges for financial metrics
    realistic_r2_range = (0.02, 0.15)
    realistic_sharpe_range = (0.3, 1.2)

    # R² analysis
    print(f"R² Score: {r2_score:.4f}")
    if realistic_r2_range[0] <= r2_score <= realistic_r2_range[1]:
        print("✓ R² is within realistic range for financial time series")
    else:
        print("⚠ R² may indicate overfitting or underfitting - review model complexity")

    # Sharpe Ratio analysis
    print(f"Sharpe Ratio: {results['financial_metrics']['sharpe_ratio']:.4f}")
    if realistic_sharpe_range[0] <= results['financial_metrics']['sharpe_ratio'] <= realistic_sharpe_range[1]:
        print("✓ Sharpe ratio is realistic")
    else:
        print("⚠ Sharpe ratio may be unrealistic")

    # Directional Accuracy analysis
    print(f"Directional Accuracy: {results['directional_accuracy']:.4f}")
    if 0.51 <= results['directional_accuracy'] <= 0.65:
        print("✓ Directional accuracy is realistic and valuable")
    else:
        print("⚠ Check directional accuracy - may indicate issues")

    # Save predictor and results
    import joblib
    joblib.dump(predictor, 'crypto_predictor_enhanced.pkl')
    joblib.dump(results, 'crypto_results_enhanced.pkl')

    print(f"\nModel and results saved successfully!")

    return predictor, results

if __name__ == "__main__":
    predictor, results = main()
    
    