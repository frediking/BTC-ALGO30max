# Bitcoin Price Direction Prediction using CNN-LSTM with Boruta Feature Selection
# Based on Omole and Enke (2024) Financial Innovation study
# Target Accuracy: 80.44-82.44%
# pip install tensorflow pandas numpy scikit-learn boruta yfinance matplotlib seaborn

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Boruta feature selection
try:
    from boruta import BorutaPy
except ImportError:
    print("Installing BorutaPy...")
    import subprocess
    subprocess.check_call(["pip", "install", "boruta"])
    from boruta import BorutaPy

# Data manipulation and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BitcoinCNNLSTMPredictor:
    """
    CNN-LSTM model for Bitcoin price direction prediction using Boruta feature selection.
    
    This implementation follows the methodology from Omole and Enke (2024) study
    which achieved 82.44% accuracy with 6654% annual return.
    """
    
    def __init__(self, window_size=14, test_size=0.2):
        """
        Initialize the Bitcoin price predictor.
        
        Args:
            window_size (int): Number of past days to use for prediction (default: 14)
            test_size (float): Proportion of data to use for testing (default: 0.2)
        """
        self.window_size = window_size
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = []
        self.feature_names = [
            '90_day_coin_days_destroyed',
            'adjusted_sopr',
            'average_coin_dormancy',
            'average_spent_output_lifespan',
            'coin_days_destroyed',
            'coin_years_destroyed',
            'cumulative_value_days_destroyed',
            'difficulty_ribbon_compression',
            'entity_adjusted_dormancy_flow',
            'all_exchange_net_position_change_btc',
            'hodl_waves_1w_1m',
            'hodl_waves_24h',
            'hodl_waves_2y_3y',
            'hodl_waves_3m_6m',
            'inflation_rate',
            'issuance_btc',
            'median_spent_output_lifespan',
            'mvrv_z_score',
            'net_realized_profit_loss_usd',
            'all_exchanges_net_transfer_volume_btc',
            'net_unrealized_profit_loss',
            'percent_of_utxos_in_profit',
            'percent_of_supply_in_profit',
            'realized_loss_usd',
            'realized_profit_loss_ratio',
            'realized_profit_usd'
        ]
        
    def generate_synthetic_data(self, start_date='2013-02-06', end_date='2023-02-18'):
        """
        Generate synthetic Bitcoin data with on-chain metrics for demonstration.
        In production, replace this with actual Glassnode API data.
        
        Args:
            start_date (str): Start date for data generation
            end_date (str): End date for data generation
            
        Returns:
            pd.DataFrame: Synthetic Bitcoin dataset with price and on-chain metrics
        """
        print("Generating synthetic Bitcoin data...")
        print("Note: In production, replace this with actual Glassnode API data")
        
        # Get Bitcoin price data from Yahoo Finance
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
        dates = btc_data.index
        prices = btc_data['Close'].values
        
        # Generate synthetic on-chain metrics that correlate with price movements
        np.random.seed(42)
        n_days = len(prices)
        
        # Create base trends that correlate with price
        price_trend = np.gradient(prices)
        price_volatility = pd.Series(prices).rolling(30).std().fillna(method='bfill').values
        
        data = {
            'date': dates,
            'close_price': prices,
            'volume': btc_data['Volume'].values
        }
        
        # Generate synthetic on-chain metrics based on real Bitcoin characteristics
        for i, feature in enumerate(self.feature_names):
            if 'hodl' in feature.lower():
                # HODL metrics: trend following with some noise
                base_value = 0.1 + 0.05 * np.sin(np.arange(n_days) * 2 * np.pi / 365)  # Annual cycle
                noise = np.random.normal(0, 0.01, n_days)
                correlation_factor = 0.3 * (price_trend / np.std(price_trend))
                data[feature] = base_value + noise + correlation_factor * 0.01
                
            elif 'profit' in feature.lower() or 'loss' in feature.lower():
                # Profit/Loss metrics: inversely correlated with volatility
                base_value = np.random.uniform(1000, 100000, n_days)
                volatility_factor = -0.5 * (price_volatility / np.mean(price_volatility))
                data[feature] = base_value * (1 + volatility_factor + np.random.normal(0, 0.1, n_days))
                
            elif 'sopr' in feature.lower() or 'mvrv' in feature.lower():
                # Ratio metrics: centered around 1 with price correlation
                base_value = 1.0
                price_factor = 0.2 * (price_trend / np.std(price_trend))
                noise = np.random.normal(0, 0.1, n_days)
                data[feature] = base_value + price_factor + noise
                
            elif 'dormancy' in feature.lower() or 'lifespan' in feature.lower():
                # Time-based metrics: higher values during low activity periods
                base_value = np.random.uniform(10, 100)
                activity_factor = -0.3 * np.abs(price_trend) / np.std(price_trend)
                data[feature] = base_value * (1 + activity_factor + np.random.normal(0, 0.05, n_days))
                
            else:
                # General metrics: some correlation with price movements
                base_value = np.random.uniform(0.01, 1000, n_days)
                correlation = np.random.uniform(-0.3, 0.3)  # Random correlation
                price_factor = correlation * (price_trend / np.std(price_trend))
                data[feature] = base_value * (1 + price_factor + np.random.normal(0, 0.1, n_days))
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Generated {len(df)} days of synthetic data from {start_date} to {end_date}")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the Bitcoin dataset following the study methodology.
        
        Args:
            df (pd.DataFrame): Raw Bitcoin dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("Preprocessing data...")
        
        # Handle missing data
        # For MCAR (Missing Completely At Random): listwise deletion
        # For MNAR (Missing Not At Random): regression imputation
        
        # Calculate missing data percentage
        missing_pct = df.isnull().sum() / len(df) * 100
        
        # Apply listwise deletion for columns with <5% missing (MCAR assumption)
        cols_to_drop = missing_pct[missing_pct < 5].index
        df_clean = df.dropna(subset=cols_to_drop)
        
        # Apply regression imputation for remaining missing values (MNAR assumption)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use other numeric features to predict missing values
                predictor_cols = [c for c in numeric_cols if c != col and df_clean[c].isnull().sum() == 0]
                if len(predictor_cols) > 0:
                    # Fit regression model on non-missing data
                    non_missing_mask = ~df_clean[col].isnull()
                    if non_missing_mask.sum() > 0:
                        lr = LinearRegression()
                        X_train = df_clean.loc[non_missing_mask, predictor_cols[:5]]  # Use top 5 predictors
                        y_train = df_clean.loc[non_missing_mask, col]
                        lr.fit(X_train, y_train)
                        
                        # Predict missing values
                        missing_mask = df_clean[col].isnull()
                        if missing_mask.sum() > 0:
                            X_pred = df_clean.loc[missing_mask, predictor_cols[:5]]
                            df_clean.loc[missing_mask, col] = lr.predict(X_pred)
        
        # Create target variable: binary classification
        # 1 if next day's price > current day's price, 0 otherwise
        df_clean['target'] = (df_clean['close_price'].shift(-1) > df_clean['close_price']).astype(int)
        
        # Remove last row (no future price available)
        df_clean = df_clean[:-1]
        
        # Sort by date to maintain chronological order
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        print(f"Preprocessed data shape: {df_clean.shape}")
        print(f"Target distribution: {df_clean['target'].value_counts().to_dict()}")
        
        return df_clean
    
    def apply_boruta_selection(self, X, y, max_iter=100):
        """
        Apply Boruta feature selection algorithm.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            max_iter (int): Maximum iterations for Boruta
            
        Returns:
            tuple: (selected_features, feature_ranking)
        """
        print("Applying Boruta feature selection...")
        
        # Initialize Random Forest for Boruta
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=7
        )
        
        # Initialize Boruta
        boruta = BorutaPy(
            rf,
            n_estimators='auto',
            verbose=1,
            random_state=42,
            max_iter=max_iter
        )
        
        # Fit Boruta
        boruta.fit(X.values, y.values)
        
        # Get selected features
        selected_mask = boruta.support_
        selected_features = X.columns[selected_mask].tolist()
        
        # Get feature ranking
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': boruta.ranking_,
            'selected': boruta.support_
        }).sort_values('ranking')
        
        print(f"Boruta selected {len(selected_features)} features out of {len(X.columns)}")
        print("\nTop 10 selected features:")
        print(feature_ranking[feature_ranking['selected']].head(10))
        
        self.selected_features = selected_features
        return selected_features, feature_ranking
    
    def create_sequences(self, data, features, target_col):
        """
        Create sequences for CNN-LSTM input.
        
        Args:
            data (pd.DataFrame): Preprocessed data
            features (list): Selected feature names
            target_col (str): Target column name
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        print(f"Creating sequences with window size: {self.window_size}")
        
        X, y = [], []
        
        for i in range(self.window_size, len(data)):
            # Get sequence of past window_size days
            sequence = data[features].iloc[i-self.window_size:i].values
            target = data[target_col].iloc[i]
            
            X.append(sequence)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} sequences with shape: {X.shape}")
        return X, y
    
    def build_cnn_lstm_model(self, input_shape):
        """
        Build CNN-LSTM model architecture following the study.
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
            
        Returns:
            tf.keras.Model: Compiled CNN-LSTM model
        """
        print("Building CNN-LSTM model...")
        
        model = Sequential([
            # CNN Component
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            AveragePooling1D(pool_size=1),
            Dropout(0.5),
            
            # LSTM Component
            LSTM(128, activation='tanh', return_sequences=True),
            BatchNormalization(),
            Dropout(0.5),
            
            LSTM(80, activation='tanh', return_sequences=False),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output Layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(model.summary())
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the CNN-LSTM model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            dict: Training history
        """
        print("Training CNN-LSTM model...")
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_prob),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f} (Target: 0.8044-0.8244)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"MCC:       {metrics['mcc']:.4f}")
        print("="*50)
        
        return metrics, y_pred_prob
    
    def backtest_strategy(self, predictions, actual_prices, initial_capital=1000, 
                         tax_rate=0.30, transaction_cost=0.005):
        """
        Backtest trading strategy based on model predictions.
        
        Args:
            predictions: Model predictions (probabilities)
            actual_prices: Actual Bitcoin prices
            initial_capital: Starting capital
            tax_rate: Tax rate on profits
            transaction_cost: Transaction cost percentage
            
        Returns:
            dict: Backtesting results
        """
        print("Backtesting trading strategy...")
        
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        trades = []
        portfolio_values = [capital]
        
        for i in range(len(predictions)):
            current_price = actual_prices[i]
            prediction = predictions[i][0]  # Get probability
            
            # Trading signal: buy if prediction > 0.6, sell if < 0.4
            if prediction > 0.6 and position <= 0:  # Buy signal
                if position == -1:  # Close short position
                    profit = capital * (1 - current_price/entry_price)
                    capital += profit * (1 - tax_rate) - capital * transaction_cost
                
                # Open long position
                position = 1
                entry_price = current_price
                capital -= capital * transaction_cost
                
            elif prediction < 0.4 and position >= 0:  # Sell signal
                if position == 1:  # Close long position
                    profit = capital * (current_price/entry_price - 1)
                    capital += profit * (1 - tax_rate) - capital * transaction_cost
                
                # Open short position
                position = -1
                entry_price = current_price
                capital -= capital * transaction_cost
            
            portfolio_values.append(capital)
        
        # Calculate performance metrics
        total_return = (capital - initial_capital) / initial_capital
        annual_return = total_return * (365 / len(predictions))
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades)
        }
        
        print("\n" + "="*50)
        print("BACKTESTING RESULTS")
        print("="*50)
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital:   ${capital:,.2f}")
        print(f"Total Return:    {total_return*100:.2f}%")
        print(f"Annual Return:   {annual_return*100:.2f}%")
        print(f"Sharpe Ratio:    {sharpe_ratio:.4f}")
        print(f"Max Drawdown:    {max_drawdown*100:.2f}%")
        print("="*50)
        
        return results
    
    def run_complete_pipeline(self):
        """
        Run the complete CNN-LSTM pipeline from data generation to evaluation.
        """
        print("Starting CNN-LSTM Bitcoin Price Direction Prediction Pipeline")
        print("=" * 70)
        
        # Step 1: Generate/Load Data
        df = self.generate_synthetic_data()
        
        # Step 2: Preprocess Data
        df_processed = self.preprocess_data(df)
        
        # Step 3: Prepare features and target
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['date', 'close_price', 'target', 'volume']]
        X = df_processed[feature_cols]
        y = df_processed['target']
        
        # Step 4: Apply Boruta Feature Selection
        selected_features, feature_ranking = self.apply_boruta_selection(X, y)
        
        # Step 5: Create sequences for CNN-LSTM
        X_seq, y_seq = self.create_sequences(df_processed, selected_features, 'target')
        
        # Step 6: Scale features
        print("Scaling features...")
        X_seq_scaled = self.scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1]))
        X_seq_scaled = X_seq_scaled.reshape(X_seq.shape)
        
        # Step 7: Split data (maintaining chronological order)
        split_idx = int(len(X_seq_scaled) * (1 - self.test_size))
        
        X_train = X_seq_scaled[:split_idx]
        y_train = y_seq[:split_idx]
        X_test = X_seq_scaled[split_idx:]
        y_test = y_seq[split_idx:]
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Step 8: Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_cnn_lstm_model(input_shape)
        
        history = self.train_model(X_train, y_train, X_val, y_val)
        
        # Step 9: Evaluate model
        metrics, predictions = self.evaluate_model(X_test, y_test)
        
        # Step 10: Backtest strategy
        test_prices = df_processed['close_price'].iloc[split_idx + self.window_size:].values
        backtest_results = self.backtest_strategy(predictions, test_prices)
        
        return {
            'model': self.model,
            'metrics': metrics,
            'backtest_results': backtest_results,
            'selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'history': history
        }


# Usage Example
if __name__ == "__main__":
    # Initialize predictor
    predictor = BitcoinCNNLSTMPredictor(window_size=14, test_size=0.2)
    
    # Run complete pipeline
    results = predictor.run_complete_pipeline()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Display key results
    accuracy = results['metrics']['accuracy']
    annual_return = results['backtest_results']['annual_return_pct']
    
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print(f"Backtested Annual Return: {annual_return:.2f}%")
    
    if accuracy >= 0.8044:
        print("✅ Target accuracy achieved!")
    else:
        print("⚠️  Consider hyperparameter tuning to improve accuracy")