import logging
logging.basicConfig(level=logging.INFO)

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# ========================
# Constants and Parameters
# ========================

Z_SCORE_STD_THRESHOLD = 3
IQR_FACTOR = 1.5
PERCENTAGE_MULTIPLIER = 100
VOLATILITY_WINDOW = 30
TRADING_DAYS_PER_YEAR = 252
SHORT_MA_WINDOW = 20
LONG_MA_WINDOW = 50
HIGH_VOLATILITY_THRESHOLD = 0.5
VOLUME_TREND_WINDOW = 20
VOLUME_TREND_THRESHOLD = 0.1
WINDOWS_HIGH_VOL = [3, 7, 14, 30]
WINDOWS_UPDOWN = [30, 90, 180]
WINDOWS_MIXED = [7, 30, 90, 180]
DECAY_FACTOR_HIGH_VOL = 0.90
DECAY_FACTOR_UPDOWN = 0.98
DECAY_FACTOR_MIXED = 0.95
SMOOTHING_HIGH_VOL = 15
SMOOTHING_UPDOWN = 8
SMOOTHING_MIXED = 10
CV_SPLITS_DEFAULT = 5
STABILITY_WINDOW_DEFAULT = 20
DRIFT_THRESHOLD_DEFAULT = 0.1
CORRELATION_THRESHOLD_DEFAULT = 0.85
MIN_IV_THRESHOLD_DEFAULT = 0.02
WOE_BINS = 10
LAPLACE_SMOOTHING = 0.5
STABILITY_CV_THRESHOLD = 0.5
PSI_THRESHOLD = 0.25
AUTOCORR_LAGS = [1, 5, 10, 22]
SEASONAL_PERIOD_MAX = 22
SEASONALITY_THRESHOLD = 0.1
QUALITY_SCORE_WEIGHTS = {
    'information_value': 0.3,
    'stability': 0.3,
    'drift': 0.2,
    'predictive_power': 0.2
}
NOISE_SNR_THRESHOLD = 1.0
LOW_QUALITY_SCORE = 0.5

# Locked production features
FINAL_FEATURES = [
    'volatility_volume_ratio',
    'stoch_k',
    'stoch_d',
    'rolling_vol_7',
    'rolling_return_7',
    'CCI',
    'ADX_pos',
    'price_range',
    'volatility',
    'Williams_%R'
]
SEQUENCE_LENGTH = 50  # default lookback window for sequence generation

# Standard library
import warnings
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# Typing
from typing import Dict, List, Tuple

# Scientific/ML stack
import numpy as np
import pandas as pd
import time
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import IsolationForest
from tensorflow.keras import Sequential
from scipy.stats import ks_2samp

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.seasonal import seasonal_decompose
logging.basicConfig(level=logging.INFO)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization and explainability
import optuna
import shap
try:
    import mlflow
    import mlflow.keras
    MLFLOW_ENABLED = True
    mlflow.set_experiment("bitcoin_price_prediction")
except ImportError:
    MLFLOW_ENABLED = False
from contextlib import contextmanager

@contextmanager
def maybe_mlflow_run(run_name: str):
    if MLFLOW_ENABLED:
        with mlflow.start_run(run_name=run_name):
            yield
    else:
        yield

# Suppress warnings globally (consider local suppression for production)
warnings.filterwarnings('ignore')

def detect_outliers(df, columns, n_std=Z_SCORE_STD_THRESHOLD):
    """
    Detect outliers using multiple methods:
    1. Z-score method
    2. Interquartile Range (IQR) method
    3. Isolation Forest for complex outlier patterns
    
    Args:
        df: DataFrame
        columns: List of columns to check for outliers
        n_std: Number of standard deviations for Z-score method
    
    Returns:
        DataFrame with outlier masks and summary
    """
    outlier_masks = {}
    summary = {}
    
    for col in columns:
        # Z-score method
        z_scores = np.abs(stats.zscore(df[col]))
        z_mask = z_scores > n_std
        
        # IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_mask = (df[col] < (Q1 - IQR_FACTOR * IQR)) | (df[col] > (Q3 + IQR_FACTOR * IQR))
        
        # Combine methods (consider a point an outlier if both methods agree)
        combined_mask = z_mask & iqr_mask
        
        outlier_masks[col] = combined_mask
        summary[col] = {
            'total_outliers': combined_mask.sum(),
            'percentage_outliers': (combined_mask.sum() / len(df)) * PERCENTAGE_MULTIPLIER,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return outlier_masks, summary

def handle_outliers(df, outlier_masks, method='clip'):
    """
    Handle outliers using specified method
    
    Args:
        df: DataFrame
        outlier_masks: Dictionary of boolean masks for outliers
        method: 'clip' or 'winsorize'
    """
    df_cleaned = df.copy()
    
    for col, mask in outlier_masks.items():
        if method == 'clip':
            # Clip values to 3 standard deviations
            std = df[col].std()
            mean = df[col].mean()
            df_cleaned.loc[mask, col] = df[col].clip(lower=mean - Z_SCORE_STD_THRESHOLD*std, upper=mean + Z_SCORE_STD_THRESHOLD*std)
        elif method == 'winsorize':
            # Winsorize the data
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df_cleaned.loc[mask, col] = df[col].clip(lower=Q1 - IQR_FACTOR*IQR, upper=Q3 + IQR_FACTOR*IQR)
    
    return df_cleaned

def plot_outliers(df, column, outlier_mask):
    """Plot data distribution with outliers highlighted"""
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    
    plt.subplot(2, 1, 2)
    sns.histplot(data=df, x=column, hue=outlier_mask, multiple="stack")
    plt.title(f'Distribution of {column} with Outliers')
    plt.tight_layout()
    plt.show()

class MarketConditionAnalyzer:
    """Analyzes market conditions to suggest optimal encoding parameters."""
    
    def __init__(self, price_data: pd.Series, volume_data: pd.Series):
        self.price = price_data
        self.volume = volume_data
        
    def calculate_volatility(self, window: int = VOLATILITY_WINDOW) -> float:
        """Calculate rolling volatility."""
        returns = self.price.pct_change()
        volatility = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        return volatility.iloc[-1]
    
    def detect_trend(self, short_window: int = SHORT_MA_WINDOW, long_window: int = LONG_MA_WINDOW) -> str:
        """Detect market trend using moving averages."""
        short_ma = self.price.rolling(short_window).mean()
        long_ma = self.price.rolling(long_window).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 'uptrend'
        elif short_ma.iloc[-1] < long_ma.iloc[-1]:
            return 'downtrend'
        return 'ranging'
    
    def get_market_state(self) -> Dict:
        """Analyze current market state and suggest parameters."""
        volatility = self.calculate_volatility()
        trend = self.detect_trend()
        volume_trend = self.volume.pct_change(VOLUME_TREND_WINDOW).mean()
        
        # Define parameter sets based on market conditions
        if volatility > HIGH_VOLATILITY_THRESHOLD:  # High volatility
            return {
                'windows': WINDOWS_HIGH_VOL,
                'decay_factor': DECAY_FACTOR_HIGH_VOL,
                'smoothing': SMOOTHING_HIGH_VOL
            }
        elif trend in ['uptrend', 'downtrend'] and abs(volume_trend) > VOLUME_TREND_THRESHOLD:
            return {
                'windows': WINDOWS_UPDOWN,
                'decay_factor': DECAY_FACTOR_UPDOWN,
                'smoothing': SMOOTHING_UPDOWN
            }
        else:  # Mixed/Ranging market
            return {
                'windows': WINDOWS_MIXED,
                'decay_factor': DECAY_FACTOR_MIXED,
                'smoothing': SMOOTHING_MIXED
            }
class BatchLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            print(f"Batch {batch} finished. Loss: {logs.get('loss'):.4f}")

class EncodingValidator:
    """
    Comprehensive validator for categorical encodings with focus on time series data.
    
    Features:
    - Information Value (IV) and Weight of Evidence (WoE) calculations
    - Encoding stability metrics across time windows
    - Feature drift detection
    - Correlation analysis
    - Predictive power assessment
    - Cross-temporal validation
    - Noise and redundancy detection
    """
    
    def __init__(self, data: pd.DataFrame, 
                 time_column: str = 'Date',
                 target_column: str = 'Close',
                 cv_splits: int = CV_SPLITS_DEFAULT,
                 stability_window: int = STABILITY_WINDOW_DEFAULT,
                 drift_threshold: float = DRIFT_THRESHOLD_DEFAULT,
                 correlation_threshold: float = CORRELATION_THRESHOLD_DEFAULT,
                 min_iv_threshold: float = MIN_IV_THRESHOLD_DEFAULT):
        """
        Initialize the encoding validator.
        
        Args:
            data: DataFrame containing the original and encoded features
            time_column: Name of the timestamp column
            target_column: Name of the target variable
            cv_splits: Number of cross-validation splits
            stability_window: Window size for stability calculations
            drift_threshold: Threshold for feature drift detection
            correlation_threshold: Threshold for correlation analysis
            min_iv_threshold: Minimum information value threshold
        """
        self.data = data
        self.time_column = time_column
        self.target_column = target_column
        self.cv_splits = cv_splits
        self.stability_window = stability_window
        self.drift_threshold = drift_threshold
        self.correlation_threshold = correlation_threshold
        self.min_iv_threshold = min_iv_threshold
        
        # Initialize validation results storage
        self.validation_results = {}
        self.stability_metrics = {}
        self.drift_metrics = {}
        self.correlation_analysis = {}
        self.predictive_power = {}
        
    def validate_encoding(self, original_col: str, encoded_features: List[str]) -> Dict:
        """
        Perform comprehensive validation of encoded features.
        
        Args:
            original_col: Name of the original categorical column
            encoded_features: List of encoded feature names derived from original_col
            
        Returns:
            Dictionary containing all validation metrics
        """
        validation_results = {
            'information_value': self._calculate_information_value(original_col, encoded_features),
            'stability_metrics': self._assess_encoding_stability(encoded_features),
            'drift_analysis': self._detect_feature_drift(encoded_features),
            'correlation_analysis': self._analyze_correlations(encoded_features),
            'predictive_power': self._assess_predictive_power(encoded_features),
            'noise_analysis': self._analyze_noise(encoded_features),
            'temporal_validity': self._validate_temporal_consistency(encoded_features),
            'quality_scores': self._calculate_quality_scores(encoded_features)
        }
        
        # Store results
        self.validation_results[original_col] = validation_results
        
        return validation_results
    
    def _calculate_information_value(self, original_col: str, encoded_features: List[str]) -> Dict:
        """Calculate Information Value and Weight of Evidence for encodings."""
        iv_metrics = {}
        
        for feature in encoded_features:
            # Calculate WoE
            woe = self._calculate_woe(self.data[feature], self.data[self.target_column])
            
            # Calculate IV
            iv = np.sum(woe['woe'] * (woe['distr_positive'] - woe['distr_negative']))
            
            iv_metrics[feature] = {
                'iv': iv,
                'woe_bins': woe,
                'is_predictive': iv >= self.min_iv_threshold
            }
        
        return iv_metrics
    
    def _calculate_woe(self, feature: pd.Series, target: pd.Series) -> pd.DataFrame:
        """Calculate Weight of Evidence for a feature."""
        # Create bins for continuous features
        if feature.dtype in ['float64', 'float32']:
            feature = pd.qcut(feature, q=WOE_BINS, duplicates='drop')
            
        # Calculate distributions
        total_positive = (target > target.shift(1)).sum()
        total_negative = (target <= target.shift(1)).sum()
        
        grouped = pd.DataFrame({
            'total': feature.value_counts(),
            'positive': feature[target > target.shift(1)].value_counts(),
            'negative': feature[target <= target.shift(1)].value_counts()
        }).fillna(LAPLACE_SMOOTHING)  # Laplace smoothing
        
        # Calculate WoE and distributions
        grouped['distr_positive'] = grouped['positive'] / total_positive
        grouped['distr_negative'] = grouped['negative'] / total_negative
        grouped['woe'] = np.log(grouped['distr_positive'] / grouped['distr_negative'])
        
        return grouped
    
    def _assess_encoding_stability(self, encoded_features: List[str]) -> Dict:
        """Assess stability of encodings across time windows."""
        stability_metrics = {}
        
        for feature in encoded_features:
            # Calculate rolling statistics
            rolling_mean = self.data[feature].rolling(self.stability_window).mean()
            rolling_std = self.data[feature].rolling(self.stability_window).std()
            
            # Calculate stability metrics
            cv = rolling_std / rolling_mean  # Coefficient of variation
            psi = self._calculate_psi(self.data[feature])  # Population Stability Index
            
            stability_metrics[feature] = {
                'coefficient_of_variation': cv.mean(),
                'population_stability_index': psi,
                'is_stable': (cv.mean() < STABILITY_CV_THRESHOLD) and (psi < PSI_THRESHOLD)
            }
        
        return stability_metrics
    
    def _calculate_psi(self, feature: pd.Series) -> float:
        """Calculate Population Stability Index."""
        # Split data into reference and current
        mid_point = len(feature) // 2
        reference = feature.iloc[:mid_point]
        current = feature.iloc[mid_point:]
        
        # Create bins
        bins = pd.qcut(reference, q=WOE_BINS, duplicates='drop', labels=False)
        bins_current = pd.cut(current, bins=[-np.inf] + list(np.unique(bins)) + [np.inf])
        
        # Calculate distributions
        dist_reference = pd.value_counts(bins, normalize=True)
        dist_current = pd.value_counts(bins_current, normalize=True)
        
        # Calculate PSI
        psi = np.sum((dist_current - dist_reference) * np.log(dist_current / dist_reference))
        
        return psi
    
    def _detect_feature_drift(self, encoded_features: List[str]) -> Dict:
        """Detect and quantify feature drift over time."""
        drift_metrics = {}
        
        for feature in encoded_features:
            # Calculate drift metrics
            drift = self._calculate_drift_metrics(self.data[feature])
            
            drift_metrics[feature] = {
                'drift_score': drift['drift_score'],
                'drift_direction': drift['direction'],
                'drift_velocity': drift['velocity'],
                'requires_attention': drift['drift_score'] > self.drift_threshold
            }
        
        return drift_metrics
    
    def _calculate_drift_metrics(self, feature: pd.Series) -> Dict:
        """Calculate comprehensive drift metrics for a feature."""
        # Calculate trend
        trend = feature.rolling(window=self.stability_window).mean()
        trend_diff = trend.diff()
        
        # Calculate drift score
        drift_score = abs(trend_diff).mean() / feature.std()
        
        # Calculate drift direction and velocity
        direction = np.sign(trend_diff.mean())
        velocity = trend_diff.mean() / self.stability_window
        
        return {
            'drift_score': drift_score,
            'direction': direction,
            'velocity': velocity
        }
    
    def _analyze_correlations(self, encoded_features: List[str]) -> Dict:
        """Analyze correlations between encoded features."""
        # Calculate correlation matrix
        corr_matrix = self.data[encoded_features].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(encoded_features)):
            for j in range(i+1, len(encoded_features)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': encoded_features[i],
                        'feature2': encoded_features[j],
                        'correlation': correlation
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs,
            'avg_correlation': corr_matrix.abs().mean().mean(),
            'redundancy_detected': len(high_corr_pairs) > 0
        }
    
    def _assess_predictive_power(self, encoded_features: List[str]) -> Dict:
        """Assess predictive power of encoded features."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.ensemble import RandomForestRegressor
        
        predictive_metrics = {}
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        for feature in encoded_features:
            cv_scores = []
            feature_importance = []
            
            # Perform time series cross-validation
            for train_idx, test_idx in tscv.split(self.data):
                X_train = self.data.iloc[train_idx][[feature]]
                y_train = self.data.iloc[train_idx][self.target_column]
                X_test = self.data.iloc[test_idx][[feature]]
                y_test = self.data.iloc[test_idx][self.target_column]
                
                # Train a simple model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate metrics
                y_pred = model.predict(X_test)
                cv_scores.append({
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                })
                feature_importance.append(model.feature_importances_[0])
            
            predictive_metrics[feature] = {
                'avg_r2': np.mean([score['r2'] for score in cv_scores]),
                'avg_rmse': np.mean([score['rmse'] for score in cv_scores]),
                'feature_importance': np.mean(feature_importance),
                'cv_stability': np.std([score['r2'] for score in cv_scores])
            }
        
        return predictive_metrics
    
    def _analyze_noise(self, encoded_features: List[str]) -> Dict:
        """Analyze noise levels in encoded features."""
        noise_metrics = {}
        
        for feature in encoded_features:
            # Calculate signal-to-noise ratio
            signal = np.abs(self.data[feature].mean())
            noise = self.data[feature].std()
            snr = signal / noise if noise != 0 else np.inf
            
            # Calculate noise metrics
            noise_metrics[feature] = {
                'signal_to_noise_ratio': snr,
                'noise_level': 1 / snr if snr != 0 else 0,
                'is_noisy': snr < NOISE_SNR_THRESHOLD
            }
        
        return noise_metrics
    
    def _validate_temporal_consistency(self, encoded_features: List[str]) -> Dict:
        """Validate temporal consistency of encodings."""
        temporal_metrics = {}
        
        for feature in encoded_features:
            # Calculate temporal metrics
            temporal_metrics[feature] = {
                'autocorrelation': self._calculate_autocorrelation(self.data[feature]),
                'temporal_stability': self._calculate_temporal_stability(self.data[feature]),
                'seasonality': self._detect_seasonality(self.data[feature])
            }
        
        return temporal_metrics
    
    def _calculate_autocorrelation(self, feature: pd.Series, lags: List[int] = AUTOCORR_LAGS) -> Dict:
        """Calculate autocorrelation at different lags."""
        return {
            f'lag_{lag}': feature.autocorr(lag=lag)
            for lag in lags
        }
    
    def _calculate_temporal_stability(self, feature: pd.Series) -> Dict:
        """Calculate temporal stability metrics."""
        # Split into time windows
        windows = np.array_split(feature, self.cv_splits)
        
        # Calculate statistics for each window
        window_stats = [
            {
                'mean': window.mean(),
                'std': window.std(),
                'skew': window.skew(),
                'kurtosis': window.kurtosis()
            }
            for window in windows
        ]
        
        # Calculate stability metrics
        return {
            'mean_stability': np.std([stats['mean'] for stats in window_stats]),
            'variance_stability': np.std([stats['std'] for stats in window_stats]),
            'distribution_stability': np.mean([
                stats['skew'] * stats['kurtosis']
                for stats in window_stats
            ])
        }
    
    def _detect_seasonality(self, feature: pd.Series) -> Dict:
        """Detect seasonality patterns in the feature."""
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                feature,
                period=min(len(feature) // 2, SEASONAL_PERIOD_MAX),  # Use business month or half the series length
                extrapolate_trend='freq'
            )
            
            return {
                'seasonal_strength': np.std(decomposition.seasonal),
                'trend_strength': np.std(decomposition.trend),
                'has_seasonality': np.std(decomposition.seasonal) > SEASONALITY_THRESHOLD * np.std(feature)
            }
        except ValueError:
            # Handle cases where decomposition fails due to insufficient data or NaNs
            return {
                'seasonal_strength': 0,
                'trend_strength': 0,
                'has_seasonality': False
            }
        except Exception as e:
            logging.warning(f"Unexpected error in _detect_seasonality: {e}")
            return {
                'seasonal_strength': 0,
                'trend_strength': 0,
                'has_seasonality': False
            }
    
    def _calculate_quality_scores(self, encoded_features: List[str]) -> Dict:
        """Calculate overall quality scores for encoded features."""
        quality_scores = {}
        
        for feature in encoded_features:
            # Gather all metrics
            iv_score = self.validation_results.get(feature, {}).get('information_value', {}).get('iv', 0)
            stability_score = 1 - self.stability_metrics.get(feature, {}).get('coefficient_of_variation', 1)
            drift_score = 1 - self.drift_metrics.get(feature, {}).get('drift_score', 1)
            predictive_score = self.predictive_power.get(feature, {}).get('avg_r2', 0)
            
            # Calculate weighted quality score
            quality_scores[feature] = {
                'overall_score': np.mean([
                    iv_score * QUALITY_SCORE_WEIGHTS['information_value'],
                    stability_score * QUALITY_SCORE_WEIGHTS['stability'],
                    drift_score * QUALITY_SCORE_WEIGHTS['drift'],
                    predictive_score * QUALITY_SCORE_WEIGHTS['predictive_power']
                ]),
                'component_scores': {
                    'information_value': iv_score,
                    'stability': stability_score,
                    'drift_resistance': drift_score,
                    'predictive_power': predictive_score
                }
            }
        
        return quality_scores
    
    def get_validation_summary(self, original_col: str) -> Dict:
        """Get a comprehensive summary of validation results."""
        if original_col not in self.validation_results:
            raise ValueError(f"No validation results found for column {original_col}")
        
        results = self.validation_results[original_col]
        
        return {
            'overall_quality': np.mean([
                score['overall_score']
                for score in results['quality_scores'].values()
            ]),
            'problematic_features': [
                feature for feature, metrics in results['quality_scores'].items()
                if metrics['overall_score'] < LOW_QUALITY_SCORE
            ],
            'redundant_features': [
                pair['feature1'] for pair in results['correlation_analysis']['high_correlations']
            ],
            'unstable_features': [
                feature for feature, metrics in results['stability_metrics'].items()
                if not metrics['is_stable']
            ],
            'drifting_features': [
                feature for feature, metrics in results['drift_analysis'].items()
                if metrics['requires_attention']
            ],
            'recommendations': self._generate_recommendations(original_col)
        }
    
    def _generate_recommendations(self, original_col: str) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        results = self.validation_results[original_col]
        
        # Check for low quality scores
        low_quality_features = [
            feature for feature, metrics in results['quality_scores'].items()
            if metrics['overall_score'] < LOW_QUALITY_SCORE
        ]
        if low_quality_features:
            recommendations.append(
                f"Consider reviewing or removing low quality features: {', '.join(low_quality_features)}"
            )
        
        # Check for high correlations
        if results['correlation_analysis']['redundancy_detected']:
            recommendations.append(
                "Consider removing highly correlated features based on their importance scores"
            )
        
        # Check for stability issues
        unstable_features = [
            feature for feature, metrics in results['stability_metrics'].items()
            if not metrics['is_stable']
        ]
        if unstable_features:
            recommendations.append(
                f"Consider adjusting encoding parameters for unstable features: {', '.join(unstable_features)}"
            )
        
        # Check for drift
        drifting_features = [
            feature for feature, metrics in results['drift_analysis'].items()
            if metrics['requires_attention']
        ]
        if drifting_features:
            recommendations.append(
                f"Monitor and potentially retrain encodings for features with significant drift: {', '.join(drifting_features)}"
            )
        
        return recommendations

class FinancialMetrics:
    """Financial and trading specific metrics for model evaluation."""
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the directional accuracy of predictions."""
        y_true_direction = np.diff(y_true.reshape(-1)) > 0
        y_pred_direction = np.diff(y_pred.reshape(-1)) > 0
        return np.mean(y_true_direction == y_pred_direction)
    
    @staticmethod
    def maximum_drawdown(returns: np.ndarray) -> float:
        """Calculate the maximum drawdown from peak."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """Calculate the Sharpe ratio of returns."""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, window: int = 252) -> float:
        """Calculate the Calmar ratio (return/max drawdown)."""
        if len(returns) < window:
            window = len(returns)
        rolling_returns = returns[-window:]
        max_drawdown = abs(FinancialMetrics.maximum_drawdown(rolling_returns))
        if max_drawdown == 0:
            return 0
        return (np.mean(rolling_returns) * 252) / max_drawdown
    
    @staticmethod
    def volatility_adjusted_returns(returns: np.ndarray) -> float:
        """Calculate returns adjusted for volatility."""
        vol = np.std(returns) * np.sqrt(252)
        if vol == 0:
            return 0
        return np.mean(returns) * 252 / vol

class ParameterOptimizer:
    """Optimizes encoding parameters using Optuna."""
    
    def __init__(self, df: pd.DataFrame, target_col: str, categorical_col: str):
        self.df = df
        self.target_col = target_col
        self.categorical_col = categorical_col
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for parameter optimization."""
        # Define parameter search space
        params = {
            'windows': trial.suggest_categorical('windows', [
                [3, 7, 14, 30],
                [7, 30, 90, 180],
                [30, 90, 180, 360]
            ]),
            'decay_factor': trial.suggest_float('decay_factor', 0.9, 0.99),
            'smoothing': trial.suggest_int('smoothing', 5, 20)
        }
        
        # Create encoder with trial parameters
        encoder = RobustCategoryEncoder(
            n_splits=5,
            smoothing=params['smoothing'],
            decay_factor=params['decay_factor']
        )
        
        # Encode data
        encoded_features = encoder.fit_transform(
            self.df,
            self.categorical_col,
            self.target_col
        )
        
        # Calculate validation metrics
        validator = EncodingValidator()
        metrics = validator.calculate_metrics(
            self.df[self.target_col],
            encoded_features
        )
        
        # Calculate overall score (weighted average of metrics)
        score = np.mean([
            np.mean([m['correlation'] for m in metrics.values()]),
            np.mean([m['iv'] for m in metrics.values()]),
            np.mean([m['stability'] for m in metrics.values()])
        ])
        
        return score

    def optimize(self, n_trials: int = 100) -> Dict:
        """Run parameter optimization."""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value
        }

class RobustCategoryEncoder:
    """A robust category encoder with adaptive parameters."""
    
    def __init__(self, data,  n_splits=5, smoothing=10, decay_factor=0.95):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.decay_factor = decay_factor
        self.encodings = {}
        self.global_stats = {}
        self.market_analyzer = None
        self.validator = EncodingValidator(data)
    
    def adapt_parameters(self, df: pd.DataFrame, price_col: str, volume_col: str):
        """Adapt parameters based on market conditions."""
        self.market_analyzer = MarketConditionAnalyzer(
            df[price_col],
            df[volume_col]
        )
        market_params = self.market_analyzer.get_market_state()
        
        self.smoothing = market_params['smoothing']
        self.decay_factor = market_params['decay_factor']
        return market_params['windows']
    
    def target_mean_encode(self, df, col, target, fold_column='fold'):
        """
        Perform target mean encoding with cross-validation to prevent leakage.
        """
        df_encoded = df.copy()
        means = df.groupby(col)[target].mean()
        global_mean = df[target].mean()
        
        # Smooth means with global mean
        smoothed_means = (means * df.groupby(col).size() + global_mean * self.smoothing) / \
                        (df.groupby(col).size() + self.smoothing)
        
        for fold in df[fold_column].unique():
            train_idx = df[fold_column] != fold
            test_idx = df[fold_column] == fold
            
            # Calculate means excluding current fold
            fold_means = df[train_idx].groupby(col)[target].mean()
            # Apply smoothing
            fold_means = (fold_means * df[train_idx].groupby(col).size() + global_mean * self.smoothing) / \
                        (df[train_idx].groupby(col).size() + self.smoothing)
            
            # Fill in means for test fold
            df_encoded.loc[test_idx, f'{col}_target_mean'] = df.loc[test_idx, col].map(fold_means).fillna(global_mean)
            
        return df_encoded[f'{col}_target_mean']
    
    def impact_encode(self, df, col, target):
        """
        Encode based on the magnitude of impact on the target variable.
        """
        impact_values = {}
        for category in df[col].unique():
            cat_mask = df[col] == category
            # Calculate average absolute change in target when this category occurs
            target_changes = df[target].diff().abs()
            impact = target_changes[cat_mask].mean() / target_changes.mean()
            impact_values[category] = impact
        
        return df[col].map(impact_values)
    
    def rolling_mean_encode(self, df, col, target, windows=[7, 30, 90]):
        """
        Create rolling mean encodings with multiple window sizes.
        """
        encodings = {}
        for window in windows:
            # Calculate rolling means for each category
            rolling_means = df.groupby(col)[target].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            encodings[f'{col}_rolling_mean_{window}'] = rolling_means
        return pd.DataFrame(encodings)
    
    def time_decay_encode(self, df, col, target, decay_windows=[7, 30, 90]):
        """
        Create time-decay weighted encodings with multiple window sizes.
        """
        encodings = {}
        for window in decay_windows:
            weights = np.power(self.decay_factor, np.arange(window)[::-1])
            weights = weights / weights.sum()
            
            # Calculate weighted means for each category
            decay_means = df.groupby(col)[target].transform(
                lambda x: x.rolling(window, min_periods=1)
                          .apply(lambda w: (w * weights[:len(w)]).sum() / weights[:len(w)].sum())
            )
            encodings[f'{col}_decay_mean_{window}'] = decay_means
        return pd.DataFrame(encodings)
    
    def fit_transform(self, df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
        """Enhanced fit_transform with adaptive parameters and validation."""
        # Adapt parameters to market conditions
        windows = self.adapt_parameters(df, target, 'Volume')
        
        # Create folds for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        df['fold'] = -1
        
        for fold, (_, test_idx) in enumerate(tscv.split(df)):
            df.loc[df.index[test_idx], 'fold'] = fold
        
        # Apply encoding methods with adapted parameters
        encoded_features = pd.DataFrame(index=df.index)
        
        # Apply encodings with adapted windows
        encoded_features[f'{col}_target_mean'] = self.target_mean_encode(df, col, target)
        encoded_features[f'{col}_impact'] = self.impact_encode(df, col, target)
        
        rolling_features = self.rolling_mean_encode(df, col, target, windows=windows)
        encoded_features = pd.concat([encoded_features, rolling_features], axis=1)
        
        decay_features = self.time_decay_encode(df, col, target, decay_windows=windows)
        encoded_features = pd.concat([encoded_features, decay_features], axis=1)
        
        # Validate encodings
        validation_metrics = self.validator.calculate_metrics(df[target], encoded_features)
        
        # Store results
        self.encodings[col] = {
            'features': encoded_features.columns.tolist(),
            'metrics': validation_metrics,
            'parameters': {
                'windows': windows,
                'decay_factor': self.decay_factor,
                'smoothing': self.smoothing
            }
        }
        
        return encoded_features

class MarketRegimeAnalyzer:
    """Analyzes market conditions to determine regime and appropriate metric weights."""
    
    def __init__(self, price_data: np.ndarray, window_size: int = 30):
        self.price_data = price_data
        self.window_size = window_size
        self.regimes = {}
        
    def calculate_market_metrics(self) -> dict:
        """Calculate key market metrics for regime detection."""
        # flatten price_data to 1D close-price series
        arr = np.asarray(self.price_data)
        price_arr = arr[:, -1] if arr.ndim > 1 else arr
        # compute returns as percent change
        returns = price_arr[1:] / price_arr[:-1] - 1
        
        # Rolling volatility (annualized)
        rolling_vol = pd.Series(returns).rolling(self.window_size).std() * np.sqrt(252)
        
        # Trend strength using ADX-like calculation
        price_series = pd.Series(price_arr)
        ma_short = price_series.rolling(14).mean()
        ma_long = price_series.rolling(30).mean()
        trend_strength = abs(ma_short - ma_long) / ma_long
        
        # Market efficiency ratio (MER)
        directional_movement = abs(price_arr[-1] - price_arr[0])
        path_movement = np.sum(abs(returns))
        efficiency_ratio = directional_movement / path_movement if path_movement != 0 else 0
        
        return {
            'volatility': rolling_vol.iloc[-1],
            'trend_strength': trend_strength.iloc[-1],
            'efficiency_ratio': efficiency_ratio,
            'recent_returns': returns[-self.window_size:],
            'current_price': price_arr[-1]
        }
    
    def detect_regime(self) -> dict:
        """Detect current market regime based on metrics."""
        metrics = self.calculate_market_metrics()
        
        # Volatility regime
        high_vol_threshold = 0.4  # 40% annualized volatility
        low_vol_threshold = 0.15  # 15% annualized volatility
        
        # Trend regime
        strong_trend_threshold = 0.05  # 5% difference in moving averages
        
        # Market efficiency regime
        high_efficiency_threshold = 0.7
        low_efficiency_threshold = 0.3
        
        # Determine primary regime
        if metrics['volatility'] > high_vol_threshold:
            primary_regime = 'high_volatility'
        elif metrics['trend_strength'] > strong_trend_threshold and metrics['efficiency_ratio'] > high_efficiency_threshold:
            primary_regime = 'trending'
        elif metrics['volatility'] < low_vol_threshold and metrics['efficiency_ratio'] < low_efficiency_threshold:
            primary_regime = 'ranging'
        else:
            primary_regime = 'mixed'
        
        # Calculate regime characteristics
        regime_characteristics = {
            'primary_regime': primary_regime,
            'volatility_level': metrics['volatility'],
            'trend_strength': metrics['trend_strength'],
            'market_efficiency': metrics['efficiency_ratio']
        }
        
        self.regimes = regime_characteristics
        return regime_characteristics
    
    def get_regime_weights(self) -> dict:
        """Determine appropriate metric weights based on market regime."""
        if not self.regimes:
            self.detect_regime()
        
        # Base weights
        weights = {
            'val_loss': 0.3,
            'directional_accuracy': 0.3,
            'sharpe_ratio': 0.2,
            'volatility_adjusted_returns': 0.1,
            'calmar_ratio': 0.1
        }
        
        # Adjust weights based on regime
        if self.regimes['primary_regime'] == 'high_volatility':
            # Prioritize risk management in high volatility
            weights.update({
                'val_loss': 0.2,
                'directional_accuracy': 0.2,
                'sharpe_ratio': 0.2,
                'volatility_adjusted_returns': 0.2,
                'calmar_ratio': 0.2  # Increased weight on drawdown protection
            })
        
        elif self.regimes['primary_regime'] == 'trending':
            # Prioritize directional accuracy in trending markets
            weights.update({
                'val_loss': 0.25,
                'directional_accuracy': 0.35,  # Increased weight on direction
                'sharpe_ratio': 0.2,
                'volatility_adjusted_returns': 0.1,
                'calmar_ratio': 0.1
            })
        
        elif self.regimes['primary_regime'] == 'ranging':
            # Prioritize value prediction in ranging markets
            weights.update({
                'val_loss': 0.35,  # Increased weight on value prediction
                'directional_accuracy': 0.25,
                'sharpe_ratio': 0.2,
                'volatility_adjusted_returns': 0.1,
                'calmar_ratio': 0.1
            })
        
        return weights

class ModelTuner:
    """Optimizes hyperparameters for deep learning models using regime-aware multi-metric optimization."""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_models = {}
        self.best_params = {}
        self.metrics = FinancialMetrics()
        self.regime_analyzer = MarketRegimeAnalyzer(y_val)
        self.metric_weights = self.regime_analyzer.get_regime_weights()
        
        # Log initial regime and weights
        self.current_regime = self.regime_analyzer.regimes
        logging.info("\nCurrent Market Regime Analysis:")
        logging.info(json.dumps(self.current_regime, indent=2))
        logging.info("\nOptimization Metric Weights:")
        logging.info(json.dumps(self.metric_weights, indent=2))
    
    def create_model(self, trial: optuna.Trial, model_type: str, input_shape: tuple, output_dim: int) -> tf.keras.Model:
        """Creates a model with trial hyperparameters."""
        
        # Common hyperparameters
        n_layers = trial.suggest_int('n_layers', 2, 4)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Model-specific hyperparameters
        if model_type == 'LSTM':
            return self._create_lstm_model(trial, n_layers, dropout_rate, learning_rate, input_shape, output_dim)
        elif model_type == 'BiLSTM':
            return self._create_bilstm_model(trial, n_layers, dropout_rate, learning_rate, input_shape, output_dim)
        else:  # GRU
            return self._create_gru_model(trial, n_layers, dropout_rate, learning_rate, input_shape, output_dim)
    
    def _create_lstm_model(self, trial, n_layers, dropout_rate, learning_rate, input_shape, output_dim):
        model = Sequential()
        
        # First layer
        units = trial.suggest_int('lstm_units_1', 32, 256)
        model.add(LSTM(units, return_sequences=True if n_layers > 1 else False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Middle layers
        for i in range(1, n_layers - 1):
            units = trial.suggest_int(f'lstm_units_{i+1}', 32, 256)
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        
        # Last LSTM layer (if more than one layer)
        if n_layers > 1:
            units = trial.suggest_int(f'lstm_units_{n_layers}', 32, 256)
            model.add(LSTM(units))
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        n_dense = trial.suggest_int('n_dense_layers', 1, 3)
        for i in range(n_dense):
            units = trial.suggest_int(f'dense_units_{i+1}', 16, 128)
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(output_dim))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model
    
    def _create_bilstm_model(self, trial, n_layers, dropout_rate, learning_rate, input_shape, output_dim):
        model = Sequential()
        
        # First layer
        units = trial.suggest_int('bilstm_units_1', 32, 256)
        model.add(Bidirectional(LSTM(units, return_sequences=True if n_layers > 1 else False), input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Middle layers
        for i in range(1, n_layers - 1):
            units = trial.suggest_int(f'bilstm_units_{i+1}', 32, 256)
            model.add(Bidirectional(LSTM(units, return_sequences=True)))
            model.add(Dropout(dropout_rate))
        
        # Last BiLSTM layer (if more than one layer)
        if n_layers > 1:
            units = trial.suggest_int(f'bilstm_units_{n_layers}', 32, 256)
            model.add(Bidirectional(LSTM(units)))
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        n_dense = trial.suggest_int('n_dense_layers', 1, 3)
        for i in range(n_dense):
            units = trial.suggest_int(f'dense_units_{i+1}', 16, 128)
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(output_dim))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model
    
    def _create_gru_model(self, trial, n_layers, dropout_rate, learning_rate, input_shape, output_dim):
        model = Sequential()
        
        # First layer
        units = trial.suggest_int('gru_units_1', 32, 256)
        model.add(GRU(units, return_sequences=True if n_layers > 1 else False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Middle layers
        for i in range(1, n_layers - 1):
            units = trial.suggest_int(f'gru_units_{i+1}', 32, 256)
            model.add(GRU(units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        
        # Last GRU layer (if more than one layer)
        if n_layers > 1:
            units = trial.suggest_int(f'gru_units_{n_layers}', 32, 256)
            model.add(GRU(units))
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        n_dense = trial.suggest_int('n_dense_layers', 1, 3)
        for i in range(n_dense):
            units = trial.suggest_int(f'dense_units_{i+1}', 16, 128)
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(output_dim))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model
    
    def calculate_financial_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate comprehensive financial metrics for model evaluation."""
        # Calculate returns
        true_returns = np.diff(y_true.reshape(-1)) / y_true[:-1].reshape(-1)
        pred_returns = np.diff(y_pred.reshape(-1)) / y_pred[:-1].reshape(-1)
        
        metrics = {
            'directional_accuracy': self.metrics.directional_accuracy(y_true, y_pred),
            'sharpe_ratio': self.metrics.sharpe_ratio(pred_returns),
            'volatility_adjusted_returns': self.metrics.volatility_adjusted_returns(pred_returns),
            'calmar_ratio': self.metrics.calmar_ratio(pred_returns)
        }
        
        return metrics
    
    def objective(self, trial: optuna.Trial, model_type: str, input_shape: tuple, output_dim: int) -> float:
        import numpy as np
        import tensorflow as tf
    
        print("DEBUG: Entered objective function")
    
        # === DATA VALIDATION ===
        print("DEBUG: Checking data types and shapes")
        assert hasattr(self, "X_train") and hasattr(self, "y_train"), "X_train/y_train not found"
        assert hasattr(self, "X_val") and hasattr(self, "y_val"), "X_val/y_val not found"
        assert isinstance(self.X_train, np.ndarray), "X_train is not a numpy array"
        assert isinstance(self.y_train, np.ndarray), "y_train is not a numpy array"
        assert not np.isnan(self.X_train).any(), "NaNs in X_train"
        assert not np.isnan(self.y_train).any(), "NaNs in y_train"
        assert not np.isinf(self.X_train).any(), "Infs in X_train"
        assert not np.isinf(self.y_train).any(), "Infs in y_train"
        assert self.X_train.shape[0] > 0, "X_train is empty"
        assert self.X_train.shape[0] == self.y_train.shape[0], "X_train and y_train sample count mismatch"
        if len(self.X_train.shape) != 3:
            raise ValueError(f"X_train must be 3D (samples, timesteps, features), got {self.X_train.shape}")
        if len(self.X_val.shape) != 3:
            raise ValueError(f"X_val must be 3D (samples, timesteps, features), got {self.X_val.shape}")
        print("DEBUG: Data validation passed")
    
        # === MINIMAL DATA SLICE FOR DEBUGGING ===
        print("DEBUG: Slicing minimal data for debug run")
        X_train_debug = self.X_train[:32]
        y_train_debug = self.y_train[:32]
        X_val_debug = self.X_val[:8]
        y_val_debug = self.y_val[:8]

        # === DEBUG DATA VALIDATION ===
        print("DEBUG: X_train_debug dtype:", X_train_debug.dtype)
        print("DEBUG: y_train_debug dtype:", y_train_debug.dtype)
        print("DEBUG: X_train_debug min/max:", np.min(X_train_debug), np.max(X_train_debug))
        print("DEBUG: y_train_debug min/max:", np.min(y_train_debug), np.max(y_train_debug))
        print("DEBUG: X_train_debug sample:", X_train_debug[0])
        print("DEBUG: y_train_debug sample:", y_train_debug[0])

        print("DEBUG: X_train_debug shape before reshape:", X_train_debug.shape)
        print("DEBUG: y_train_debug shape before reshape:", y_train_debug.shape)
        
        # Reshape if necessary
        if len(X_train_debug.shape) == 2:
            X_train_debug = X_train_debug.reshape((X_train_debug.shape[0], 1, X_train_debug.shape[1]))
        if len(X_val_debug.shape) == 2:
            X_val_debug = X_val_debug.reshape((X_val_debug.shape[0], 1, X_val_debug.shape[1]))
        if len(y_train_debug.shape) == 2 and y_train_debug.shape[1] == 1:
            y_train_debug = y_train_debug.ravel()
        if len(y_val_debug.shape) == 2 and y_val_debug.shape[1] == 1:
            y_val_debug = y_val_debug.ravel()
        
        print("DEBUG: X_train_debug shape after reshape:", X_train_debug.shape)
        print("DEBUG: y_train_debug shape after reshape:", y_train_debug.shape)

        # === ENSURE TARGET IS 1D (SINGLE OUTPUT) ===
        if len(y_train_debug.shape) == 2 and y_train_debug.shape[1] > 1:
            print("DEBUG: y_train_debug has >1 columns, selecting the first column for regression.")
            y_train_debug = y_train_debug[:, 0]
        if len(y_val_debug.shape) == 2 and y_val_debug.shape[1] > 1:
            print("DEBUG: y_val_debug has >1 columns, selecting the first column for regression.")
            y_val_debug = y_val_debug[:, 0]
        
        print("DEBUG: y_train_debug shape after column selection:", y_train_debug.shape)
        print("DEBUG: y_val_debug shape after column selection:", y_val_debug.shape)

        # Scale data to [-1, 1]
        X_train_debug = X_train_debug / np.max(np.abs(X_train_debug))
        y_train_debug = y_train_debug / np.max(np.abs(y_train_debug))
        print("DEBUG: X_train_debug min/max after scaling:", np.min(X_train_debug), np.max(X_train_debug))
        print("DEBUG: y_train_debug min/max after scaling:", np.min(y_train_debug), np.max(y_train_debug))
        
        print("Any NaNs in X_train_debug?", np.isnan(X_train_debug).any())
        print("Any Infs in X_train_debug?", np.isinf(X_train_debug).any())
        print("Any NaNs in y_train_debug?", np.isnan(y_train_debug).any())
        print("Any Infs in y_train_debug?", np.isinf(y_train_debug).any())

        # === TINY FIT GATEKEEPER ===

        print("DEBUG: Trying dummy model fit")
        
       
        
        dummy_model = Sequential([Flatten(input_shape=X_train_debug.shape[1:]), Dense(1)])
        dummy_model.compile(optimizer='adam', loss='mse')
        dummy_model.fit(X_train_debug, y_train_debug, epochs=1, batch_size=8, verbose=2)
        
        print("DEBUG: Dummy model tiny fit successful")
        # try:
        #     print("DEBUG: Before model creation")
        #     model = self.create_model(trial, model_type, input_shape, output_dim)
        #     print("DEBUG: Model created")
    
        #     print("DEBUG: Before tiny fit")
        #     model.fit(X_train_debug, y_train_debug, epochs=1, batch_size=8, verbose=2)
        #     print("DEBUG: Tiny fit successful")
        # except Exception as e:
        #     print("DEBUG: Tiny fit failed:", e)
        #     raise
    
        # === HYPERPARAMETER SUGGESTION ===
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 50, 200)
        patience = trial.suggest_int('patience', 10, 30)
        assert batch_size <= self.X_train.shape[0], "Batch size larger than dataset"
        print("DEBUG: Hyperparameters selected", batch_size, epochs, patience)
    
        # === MODEL TRAINING ===
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )

        # After selecting hyperparameters...
        if model_type == 'LSTM':
            model = Sequential()
            model.add(LSTM(64, input_shape=input_shape))  # adjust units/hyperparams as needed
            model.add(Dense(output_dim))
            model.compile(optimizer='adam', loss='mse')
        elif model_type == 'GRU':
            # ...similar for GRU
            pass
    
        print("DEBUG: Starting full model.fit")
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        print("DEBUG: model.fit finished")
    
        # === METRIC CALCULATION ===
        print("DEBUG: Predicting on validation set")
        val_pred = model.predict(self.X_val, verbose=0)
        val_loss = min(history.history['val_loss'])
        financial_metrics = self.calculate_financial_metrics(self.y_val, val_pred)
        print("DEBUG: Metrics calculated")
    
        # === REGIME-AWARE WEIGHTING ===
        if trial.number % 10 == 0:
            self.metric_weights = self.regime_analyzer.get_regime_weights()
        weighted_score = (
            self.metric_weights['val_loss'] * -val_loss +
            self.metric_weights['directional_accuracy'] * financial_metrics['directional_accuracy'] +
            self.metric_weights['sharpe_ratio'] * financial_metrics['sharpe_ratio'] +
            self.metric_weights['volatility_adjusted_returns'] * financial_metrics['volatility_adjusted_returns'] +
            self.metric_weights['calmar_ratio'] * financial_metrics['calmar_ratio']
        )
        print("DEBUG: Weighted score calculated")
    
        # === LOGGING & MLFLOW ===
        if MLFLOW_ENABLED:
            import mlflow
            mlflow.log_params({
                'batch_size': batch_size,
                'epochs': epochs,
                'patience': patience
            })
            mlflow.log_metrics({
                'val_loss': val_loss,
                'directional_accuracy': financial_metrics['directional_accuracy'],
                'sharpe_ratio': financial_metrics['sharpe_ratio'],
                'volatility_adjusted_returns': financial_metrics['volatility_adjusted_returns'],
                'calmar_ratio': financial_metrics['calmar_ratio'],
                'weighted_score': weighted_score
            })
            print("DEBUG: MLflow logging complete")
    
        # === OPTUNA TRIAL ATTRIBUTES ===
        trial.set_user_attr('val_loss', val_loss)
        trial.set_user_attr('directional_accuracy', financial_metrics['directional_accuracy'])
        trial.set_user_attr('sharpe_ratio', financial_metrics['sharpe_ratio'])
        trial.set_user_attr('volatility_adjusted_returns', financial_metrics['volatility_adjusted_returns'])
        trial.set_user_attr('calmar_ratio', financial_metrics['calmar_ratio'])
        trial.set_user_attr('market_regime', self.current_regime['primary_regime'])
        trial.set_user_attr('regime_volatility', self.current_regime['volatility_level'])
        print("DEBUG: Optuna trial attributes set")
    
        return weighted_score
    
    def optimize_model(self, model_type: str, input_shape: tuple, output_dim: int, n_trials: int = 100) -> dict:
        """Optimize hyperparameters using regime-aware multi-metric optimization."""
        
        run_name = f"{model_type}_optimization"
        with maybe_mlflow_run(run_name):
            if MLFLOW_ENABLED:
                mlflow.set_tag("model_type", model_type)
                mlflow.log_param("n_trials", n_trials)
            
            study_name = f"{model_type}_optimization"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Create partial objective function with fixed arguments
            objective_func = lambda trial: self.objective(trial, model_type, input_shape, output_dim)
            
            # Run optimization
            study.optimize(objective_func, n_trials=3, show_progress_bar=True)
            
            # Get best trial metrics and regime info
            best_metrics = {
                'val_loss': study.best_trial.user_attrs['val_loss'],
                'directional_accuracy': study.best_trial.user_attrs['directional_accuracy'],
                'sharpe_ratio': study.best_trial.user_attrs['sharpe_ratio'],
                'volatility_adjusted_returns': study.best_trial.user_attrs['volatility_adjusted_returns'],
                'calmar_ratio': study.best_trial.user_attrs['calmar_ratio'],
                'market_regime': study.best_trial.user_attrs['market_regime'],
                'regime_volatility': study.best_trial.user_attrs['regime_volatility']
            }
            
            # Store best model and parameters
            self.best_params[model_type] = study.best_params
            self.best_models[model_type] = self.create_model(
                optuna.trial.FixedTrial(study.best_params),
                model_type,
                input_shape,
                output_dim
            )
            
            # Log best params, metrics, and model to MLflow
            if MLFLOW_ENABLED:
                mlflow.log_params(study.best_params)
                for metric_name, metric_value in best_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                mlflow.keras.log_model(self.best_models[model_type], f"model_{model_type}")
            
            return {
                'best_params': study.best_params,
                'best_metrics': best_metrics,
                'best_value': study.best_value,
                'best_model': self.best_models[model_type],
                'market_regime': self.current_regime
            }

class ThresholdOptimizer:
    """Optimizes trading thresholds based on market regime and model predictions."""
    
    def __init__(self, model, regime_analyzer: MarketRegimeAnalyzer):
        self.model = model
        self.regime_analyzer = regime_analyzer
        self.optimal_thresholds = {}
        
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                entry_threshold: float, exit_threshold: float, 
                                stop_loss: float, take_profit: float) -> dict:
        """Calculate trading performance metrics for given thresholds."""
        
        returns = []
        position = 0  # -1: short, 0: neutral, 1: long
        entry_price = 0
        trades = []
        
        for i in range(1, len(y_true)):
            pred_return = (y_pred[i] - y_true[i-1]) / y_true[i-1]
            actual_return = (y_true[i] - y_true[i-1]) / y_true[i-1]
            
            # Exit conditions
            if position != 0:
                # Stop loss
                if position * actual_return < -stop_loss:
                    returns.append(-stop_loss)
                    trades.append({'type': 'stop_loss', 'return': -stop_loss})
                    position = 0
                # Take profit
                elif position * actual_return > take_profit:
                    returns.append(take_profit)
                    trades.append({'type': 'take_profit', 'return': take_profit})
                    position = 0
                # Regular exit
                elif abs(pred_return) < exit_threshold:
                    returns.append(position * actual_return)
                    trades.append({'type': 'regular_exit', 'return': position * actual_return})
                    position = 0
                else:
                    returns.append(position * actual_return)
            
            # Entry conditions
            elif abs(pred_return) > entry_threshold:
                position = 1 if pred_return > 0 else -1
                entry_price = y_true[i-1]
        
        returns = np.array(returns)
        
        if len(returns) == 0:
            return {
                'sharpe_ratio': -999,
                'max_drawdown': -999,
                'total_return': -999,
                'num_trades': 0
            }
        
        metrics = {
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns),
            'max_drawdown': FinancialMetrics.maximum_drawdown(returns),
            'total_return': np.sum(returns),
            'num_trades': len(trades),
            'win_rate': np.mean(returns > 0) if len(returns) > 0 else 0
        }
        
        return metrics
    
    def optimize_thresholds(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Optimize trading thresholds using Optuna."""
        
        def objective(trial):
            # Get current market regime
            regime = self.regime_analyzer.detect_regime()
            
            # Suggest thresholds based on regime
            if regime['primary_regime'] == 'high_volatility':
                entry_threshold = trial.suggest_float('entry_threshold', 0.015, 0.03)
                exit_threshold = trial.suggest_float('exit_threshold', 0.01, 0.02)
                stop_loss = trial.suggest_float('stop_loss', 0.02, 0.04)
                take_profit = trial.suggest_float('take_profit', 0.03, 0.06)
            elif regime['primary_regime'] == 'trending':
                entry_threshold = trial.suggest_float('entry_threshold', 0.01, 0.02)
                exit_threshold = trial.suggest_float('exit_threshold', 0.005, 0.015)
                stop_loss = trial.suggest_float('stop_loss', 0.015, 0.03)
                take_profit = trial.suggest_float('take_profit', 0.02, 0.04)
            else:  # ranging or mixed
                entry_threshold = trial.suggest_float('entry_threshold', 0.005, 0.015)
                exit_threshold = trial.suggest_float('exit_threshold', 0.003, 0.01)
                stop_loss = trial.suggest_float('stop_loss', 0.01, 0.02)
                take_profit = trial.suggest_float('take_profit', 0.015, 0.03)
            
            # Get model predictions
            y_pred = self.model.predict(X_val)
            
            # Calculate trading metrics
            metrics = self.calculate_trading_metrics(
                y_val, y_pred,
                entry_threshold, exit_threshold,
                stop_loss, take_profit
            )
            
            # Combine metrics based on regime
            if regime['primary_regime'] == 'high_volatility':
                score = (
                    0.4 * metrics['sharpe_ratio'] +
                    0.3 * (1 + metrics['max_drawdown']) +  # Convert to positive score
                    0.2 * metrics['total_return'] +
                    0.1 * metrics['win_rate']
                )
            elif regime['primary_regime'] == 'trending':
                score = (
                    0.3 * metrics['sharpe_ratio'] +
                    0.2 * (1 + metrics['max_drawdown']) +
                    0.4 * metrics['total_return'] +
                    0.1 * metrics['win_rate']
                )
            else:
                score = (
                    0.35 * metrics['sharpe_ratio'] +
                    0.25 * (1 + metrics['max_drawdown']) +
                    0.25 * metrics['total_return'] +
                    0.15 * metrics['win_rate']
                )
            
            # Store metrics for logging
            trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
            trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
            trial.set_user_attr('total_return', metrics['total_return'])
            trial.set_user_attr('num_trades', metrics['num_trades'])
            trial.set_user_attr('win_rate', metrics['win_rate'])
            trial.set_user_attr('regime', regime['primary_regime'])
            
            return score
        
        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        # Store optimal thresholds and metrics
        self.optimal_thresholds = {
            'parameters': study.best_params,
            'metrics': {
                'sharpe_ratio': study.best_trial.user_attrs['sharpe_ratio'],
                'max_drawdown': study.best_trial.user_attrs['max_drawdown'],
                'total_return': study.best_trial.user_attrs['total_return'],
                'num_trades': study.best_trial.user_attrs['num_trades'],
                'win_rate': study.best_trial.user_attrs['win_rate']
            },
            'regime': study.best_trial.user_attrs['regime']
        }
        
        return self.optimal_thresholds

class WalkForwardOptimizer:
    """Implements walk-forward optimization for time series models."""
    
    def __init__(self, data: pd.DataFrame, n_splits: int = 5, 
                 train_size: float = 0.7, validation_size: float = 0.15,
                 min_train_size: int = 252):  # Minimum 1 year of data
        """
        Initialize walk-forward optimizer.
        
        Args:
            data: Full dataset
            n_splits: Number of walk-forward splits
            train_size: Proportion of initial data for training
            validation_size: Proportion of remaining data for validation
            min_train_size: Minimum number of samples required for training
        """
        self.data = data
        self.n_splits = n_splits
        self.train_size = train_size
        self.validation_size = validation_size
        self.min_train_size = min_train_size
        self.splits = []
        self.regime_transitions = []
        
    def generate_splits(self) -> List[Dict]:
        """Generate walk-forward splits with regime analysis."""
        total_samples = len(self.data)
        
        # Ensure minimum training size
        initial_train_size = max(
            int(total_samples * self.train_size),
            self.min_train_size
        )
        
        # Calculate size of validation and test sets
        remaining_samples = total_samples - initial_train_size
        validation_samples = int(remaining_samples * self.validation_size)
        step_size = (total_samples - initial_train_size) // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * step_size
            train_end = initial_train_size + start_idx
            val_end = train_end + validation_samples
            test_end = val_end + step_size if i < self.n_splits - 1 else total_samples
            
            split = {
                'train': (start_idx, train_end),
                'validation': (train_end, val_end),
                'test': (val_end, test_end)
            }
            
            # Analyze regimes in each period
            train_regimes = self._analyze_regimes(start_idx, train_end)
            val_regimes = self._analyze_regimes(train_end, val_end)
            test_regimes = self._analyze_regimes(val_end, test_end)
            
            regime_info = {
                'train_regimes': train_regimes,
                'val_regimes': val_regimes,
                'test_regimes': test_regimes
            }
            
            self.splits.append({
                'indices': split,
                'regimes': regime_info
            })
        
        return self.splits
    
    def _analyze_regimes(self, start_idx: int, end_idx: int) -> Dict:
        """Analyze market regimes in a specific period."""
        period_data = self.data.iloc[start_idx:end_idx]
        
        # Create regime analyzer for this period
        regime_analyzer = MarketRegimeAnalyzer(period_data['Close'].values)
        regime_info = regime_analyzer.detect_regime()
        
        # Add additional regime statistics
        regime_info.update({
            'period_volatility': period_data['Close'].pct_change().std() * np.sqrt(252),
            'period_return': (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0]) - 1,
            'period_length': len(period_data)
        })
        
        return regime_info
    
    def get_split_data(self, split_idx: int) -> Tuple[Dict, Dict]:
        """Get data and regime information for a specific split."""
        if not self.splits:
            self.generate_splits()
        
        split = self.splits[split_idx]
        indices = split['indices']
        
        # Get data for each period
        data_splits = {
            'train': self.data.iloc[indices['train'][0]:indices['train'][1]],
            'validation': self.data.iloc[indices['validation'][0]:indices['validation'][1]],
            'test': self.data.iloc[indices['test'][0]:indices['test'][1]]
        }
        
        return data_splits, split['regimes']

class AdaptiveRetrainer:
    """Implements intelligent retraining triggers based on model performance and market conditions."""
    
    def __init__(self, 
                 performance_threshold: float = 0.15,  # 15% performance degradation
                 regime_change_threshold: float = 0.3,  # 30% regime metric change
                 volatility_threshold: float = 0.5,    # 50% volatility increase
                 min_samples: int = 30):  # Minimum samples before triggering
        
        self.performance_threshold = performance_threshold
        self.regime_change_threshold = regime_change_threshold
        self.volatility_threshold = volatility_threshold
        self.min_samples = min_samples
        self.baseline_metrics = None
        self.regime_metrics = None
        self.last_retrain_date = None
        self.monitoring_history = []
    
    def initialize_baseline(self, model_metrics: dict, regime_metrics: dict, date: pd.Timestamp):
        """Initialize baseline metrics after training."""
        self.baseline_metrics = model_metrics.copy()
        self.regime_metrics = regime_metrics.copy()
        self.last_retrain_date = date
    
    def check_retraining_triggers(self, 
                                current_metrics: dict,
                                current_regime: dict,
                                price_data: pd.Series,
                                date: pd.Timestamp) -> Tuple[bool, dict]:
        """Check if retraining is needed based on multiple triggers."""
        
        if len(price_data) < self.min_samples:
            return False, {"reason": "Insufficient data points"}
        
        triggers = {}
        
        # 1. Performance Degradation Check
        perf_degradation = self._check_performance_degradation(current_metrics)
        if perf_degradation['triggered']:
            triggers['performance'] = perf_degradation
        
        # 2. Regime Change Check
        regime_change = self._check_regime_change(current_regime)
        if regime_change['triggered']:
            triggers['regime'] = regime_change
        
        # 3. Volatility Change Check
        volatility_change = self._check_volatility_change(price_data)
        if volatility_change['triggered']:
            triggers['volatility'] = volatility_change
        
        # 4. Market Structure Change Check
        market_change = self._check_market_structure(price_data)
        if market_change['triggered']:
            triggers['market_structure'] = market_change
        
        # Log monitoring results
        self._log_monitoring(date, current_metrics, current_regime, triggers)
        
        # Determine if retraining is needed
        should_retrain = len(triggers) > 0
        
        return should_retrain, {
            'triggers': triggers,
            'days_since_last_retrain': (date - self.last_retrain_date).days,
            'current_regime': current_regime['primary_regime'],
            'monitoring_metrics': self._get_monitoring_summary()
        }
    
    def _check_performance_degradation(self, current_metrics: dict) -> dict:
        """Check for significant performance degradation."""
        degradation = {}
        triggered = False
        
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                change = (baseline - current_value) / baseline
                if abs(change) > self.performance_threshold:
                    degradation[metric] = {
                        'baseline': baseline,
                        'current': current_value,
                        'change': change
                    }
                    triggered = True
        
        return {
            'triggered': triggered,
            'details': degradation
        }
    
    def _check_regime_change(self, current_regime: dict) -> dict:
        """Check for significant market regime changes."""
        changes = {}
        triggered = False
        
        for metric, current_value in current_regime.items():
            if metric in self.regime_metrics:
                baseline = self.regime_metrics[metric]
                if isinstance(current_value, (int, float)):
                    change = abs((baseline - current_value) / baseline)
                    if change > self.regime_change_threshold:
                        changes[metric] = {
                            'baseline': baseline,
                            'current': current_value,
                            'change': change
                        }
                        triggered = True
                elif current_value != baseline:  # For categorical regime indicators
                    changes[metric] = {
                        'baseline': baseline,
                        'current': current_value
                    }
                    triggered = True
        
        return {
            'triggered': triggered,
            'details': changes
        }
    
    def _check_volatility_change(self, price_data: pd.Series) -> dict:
        """Check for significant volatility changes."""
        current_vol = price_data.pct_change().std() * np.sqrt(252)
        baseline_vol = self.regime_metrics.get('volatility_level', current_vol)
        
        vol_change = abs(current_vol - baseline_vol) / baseline_vol
        triggered = vol_change > self.volatility_threshold
        
        return {
            'triggered': triggered,
            'details': {
                'baseline_volatility': baseline_vol,
                'current_volatility': current_vol,
                'change': vol_change
            }
        }
    
    def _check_market_structure(self, price_data: pd.Series) -> dict:
        """Check for changes in market microstructure."""
        # Calculate market structure metrics
        returns = price_data.pct_change().dropna()
        
        current_metrics = {
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'autocorrelation': returns.autocorr(),
            'mean_reversion': self._calculate_hurst_exponent(returns)
        }
        
        if not hasattr(self, 'market_structure_baseline'):
            self.market_structure_baseline = current_metrics
            return {'triggered': False, 'details': {}}
        
        # Check for significant changes
        changes = {}
        triggered = False
        
        for metric, current_value in current_metrics.items():
            baseline = self.market_structure_baseline[metric]
            if baseline != 0:
                change = abs((current_value - baseline) / baseline)
                if change > self.regime_change_threshold:
                    changes[metric] = {
                        'baseline': baseline,
                        'current': current_value,
                        'change': change
                    }
                    triggered = True
        
        return {
            'triggered': triggered,
            'details': changes
        }
    
    def _calculate_hurst_exponent(self, returns: pd.Series, lags: list = None) -> float:
        """Calculate Hurst exponent for mean reversion analysis."""
        if lags is None:
            lags = [2, 5, 10, 20, 50]
        
        tau = np.array(lags)
        lagmat = np.zeros((len(tau), len(returns)-max(tau)))
        
        for idx, lag in enumerate(tau):
            lagmat[idx, :] = np.abs(returns[lag:].values - returns[:-lag].values).mean()
        
        m = np.polyfit(np.log10(tau), np.log10(lagmat), 1)
        hurst = m[0]  # Slope is the Hurst exponent
        
        return hurst
    
    def _log_monitoring(self, date: pd.Timestamp, metrics: dict, regime: dict, triggers: dict):
        """Log monitoring results for analysis."""
        self.monitoring_history.append({
            'date': date,
            'metrics': metrics,
            'regime': regime,
            'triggers': triggers
        })
    
    def _get_monitoring_summary(self) -> dict:
        """Get summary of recent monitoring results."""
        if not self.monitoring_history:
            return {}
        
        recent_history = self.monitoring_history[-30:]  # Last 30 observations
        
        return {
            'trigger_frequency': sum(1 for x in recent_history if x['triggers']) / len(recent_history),
            'dominant_regime': max(
                (x['regime']['primary_regime'] for x in recent_history),
                key=lambda x: sum(1 for y in recent_history if y['regime']['primary_regime'] == x)
            ),
            'avg_metrics': {
                metric: np.mean([x['metrics'][metric] for x in recent_history])
                for metric in recent_history[0]['metrics']
            }
        }

class DataQualityAnalyzer:
    """Analyzes and scores data quality for financial time series."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.quality_scores = {}
        self.anomalies = {}
        self.gaps = {}
        self.feature_importance = {}
        
    def analyze_data_quality(self) -> dict:
        """Comprehensive data quality analysis."""
        quality_report = {
            'completeness': self._analyze_completeness(),
            'consistency': self._analyze_consistency(),
            'anomalies': self._detect_anomalies(),
            'gaps': self._detect_gaps(),
            'feature_importance': self._analyze_feature_importance(),
            'timestamp_quality': self._analyze_timestamp_quality()
        }
        
        # Calculate overall quality score (0-100)
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'anomalies': 0.20,
            'gaps': 0.15,
            'timestamp_quality': 0.20
        }
        
        quality_score = sum(
            score['overall_score'] * weights[metric]
            for metric, score in quality_report.items()
            if metric in weights
        )
        
        quality_report['overall_score'] = quality_score
        self.quality_scores = quality_report
        
        return quality_report
    
    def _analyze_completeness(self) -> dict:
        """Analyze data completeness and missing values."""
        missing_stats = self.data.isnull().sum() / len(self.data) * 100
        
        completeness_score = 100 - missing_stats.mean()
        
        return {
            'overall_score': completeness_score,
            'missing_percentages': missing_stats.to_dict(),
            'columns_with_missing': missing_stats[missing_stats > 0].index.tolist()
        }
    
    def _analyze_consistency(self) -> dict:
        """Analyze data consistency and validity."""
        consistency_checks = {
            'price_validity': self._check_price_consistency(),
            'volume_validity': self._check_volume_consistency(),
            'return_validity': self._check_return_consistency()
        }
        
        # Calculate overall consistency score
        consistency_score = np.mean([
            check['score'] for check in consistency_checks.values()
        ])
        
        return {
            'overall_score': consistency_score,
            'checks': consistency_checks
        }
    
    def _check_price_consistency(self) -> dict:
        """Check price data consistency."""
        price_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in self.data.columns for col in price_cols):
            return {'score': 0, 'issues': ['Missing price columns']}
        
        issues = []
        deductions = []
        
        # Check High >= Low
        invalid_hl = (self.data['High'] < self.data['Low']).sum()
        if invalid_hl > 0:
            issues.append(f'Found {invalid_hl} cases where High < Low')
            deductions.append(20 * (invalid_hl / len(self.data)))
        
        # Check Open/Close within High/Low range
        invalid_range = (
            (self.data['Open'] > self.data['High']) |
            (self.data['Open'] < self.data['Low']) |
            (self.data['Close'] > self.data['High']) |
            (self.data['Close'] < self.data['Low'])
        ).sum()
        
        if invalid_range > 0:
            issues.append(f'Found {invalid_range} cases where Open/Close outside High/Low range')
            deductions.append(20 * (invalid_range / len(self.data)))
        
        # Check for negative prices
        negative_prices = (self.data[price_cols] <= 0).sum().sum()
        if negative_prices > 0:
            issues.append(f'Found {negative_prices} negative or zero prices')
            deductions.append(30 * (negative_prices / (len(self.data) * len(price_cols))))
        
        score = 100 - sum(deductions)
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _check_volume_consistency(self) -> dict:
        """Check volume data consistency."""
        if 'Volume' not in self.data.columns:
            return {'score': 0, 'issues': ['Missing volume data']}
        
        issues = []
        deductions = []
        
        # Check for negative volume
        negative_volume = (self.data['Volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f'Found {negative_volume} negative volume entries')
            deductions.append(30 * (negative_volume / len(self.data)))
        
        # Check for zero volume
        zero_volume = (self.data['Volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f'Found {zero_volume} zero volume entries')
            deductions.append(10 * (zero_volume / len(self.data)))
        
        # Check for volume outliers
        volume_zscore = stats.zscore(self.data['Volume'])
        volume_outliers = (np.abs(volume_zscore) > 5).sum()
        if volume_outliers > 0:
            issues.append(f'Found {volume_outliers} volume outliers (z-score > 5)')
            deductions.append(5 * (volume_outliers / len(self.data)))
        
        score = 100 - sum(deductions)
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _check_return_consistency(self) -> dict:
        """Check return data consistency."""
        returns = self.data['Close'].pct_change()
        issues = []
        deductions = []
        
        # Check for extreme returns
        extreme_returns = (np.abs(returns) > 0.5).sum()  # 50% price change
        if extreme_returns > 0:
            issues.append(f'Found {extreme_returns} extreme returns (>50%)')
            deductions.append(10 * (extreme_returns / len(returns)))
        
        # Check for return outliers
        returns_zscore = stats.zscore(returns.dropna())
        return_outliers = (np.abs(returns_zscore) > 5).sum()
        if return_outliers > 0:
            issues.append(f'Found {return_outliers} return outliers (z-score > 5)')
            deductions.append(5 * (return_outliers / len(returns)))
        
        score = 100 - sum(deductions)
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _detect_anomalies(self) -> dict:
        """Detect anomalies in the data using multiple methods."""
        anomaly_results = {}
        overall_anomalies = []
        
        # Price anomalies using Isolation Forest
        price_anomalies = self._detect_price_anomalies()
        anomaly_results['price'] = price_anomalies
        overall_anomalies.extend(price_anomalies['anomaly_indices'])
        
        # Volume anomalies
        volume_anomalies = self._detect_volume_anomalies()
        anomaly_results['volume'] = volume_anomalies
        overall_anomalies.extend(volume_anomalies['anomaly_indices'])
        
        # Calculate anomaly score (100 - percentage of anomalies)
        unique_anomalies = len(set(overall_anomalies))
        anomaly_score = 100 * (1 - unique_anomalies / len(self.data))
        
        return {
            'overall_score': anomaly_score,
            'details': anomaly_results
        }
    
    def _detect_price_anomalies(self) -> dict:
        """Detect price anomalies using Isolation Forest."""
        
        # Prepare features for anomaly detection
        price_features = self.data[['Open', 'High', 'Low', 'Close']].values
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(price_features)
        
        # Get anomaly indices
        anomaly_indices = np.where(anomalies == -1)[0]
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'count': len(anomaly_indices),
            'percentage': len(anomaly_indices) / len(self.data) * 100
        }
    
    def _detect_volume_anomalies(self) -> dict:
        """Detect volume anomalies using statistical methods."""
        volume = self.data['Volume'].values.reshape(-1, 1)
        
        # Calculate z-scores
        zscore = np.abs(stats.zscore(volume))
        
        # Consider points with z-score > 3 as anomalies
        anomaly_indices = np.where(zscore > 3)[0]
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'count': len(anomaly_indices),
            'percentage': len(anomaly_indices) / len(self.data) * 100
        }
    
    def _detect_gaps(self) -> dict:
        """Detect and analyze gaps in the time series."""
        # Convert index to datetime if needed
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        
        # Calculate expected dates
        date_range = pd.date_range(
            start=self.data.index.min(),
            end=self.data.index.max(),
            freq='D'
        )
        
        # Find missing dates
        missing_dates = date_range.difference(self.data.index)
        
        # Analyze gaps
        gaps = []
        if len(missing_dates) > 0:
            gap_starts = []
            gap_lengths = []
            current_gap = []
            
            for date in missing_dates:
                if not current_gap or (date - current_gap[-1]) == pd.Timedelta(days=1):
                    current_gap.append(date)
                else:
                    gap_starts.append(current_gap[0])
                    gap_lengths.append(len(current_gap))
                    current_gap = [date]
            
            if current_gap:
                gap_starts.append(current_gap[0])
                gap_lengths.append(len(current_gap))
            
            for start, length in zip(gap_starts, gap_lengths):
                gaps.append({
                    'start_date': start,
                    'end_date': start + pd.Timedelta(days=length-1),
                    'length': length
                })
        
        # Calculate gap score (100 - percentage of missing days)
        gap_score = 100 * (1 - len(missing_dates) / len(date_range))
        
        return {
            'overall_score': gap_score,
            'total_gaps': len(missing_dates),
            'gap_percentage': len(missing_dates) / len(date_range) * 100,
            'gaps': gaps
        }
    
    def _analyze_feature_importance(self) -> dict:
        """Analyze feature importance using multiple methods."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare features and target
        feature_cols = [col for col in self.data.columns if col != 'Close']
        X = self.data[feature_cols].fillna(method='ffill')
        y = self.data['Close']
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = dict(zip(feature_cols, rf.feature_importances_))
        
        # Correlation importance
        corr_importance = {col: y.corr(X[col]) for col in X.columns}
        
        # Combine importance scores
        combined_importance = {}
        for feature in feature_cols:
            combined_importance[feature] = {
                'rf_importance': rf_importance[feature],
                'correlation': corr_importance[feature],
                'combined_score': (rf_importance[feature] + abs(corr_importance[feature])) / 2
            }
        
        # Sort features by combined importance
        sorted_features = sorted(
            combined_importance.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        self.feature_importance = dict(sorted_features)
        return self.feature_importance
    
    def _analyze_timestamp_quality(self) -> dict:
        """Analyze quality of timestamp data."""
        if not isinstance(self.data.index, pd.DatetimeIndex):
            return {
                'overall_score': 0,
                'issues': ['Index is not DatetimeIndex']
            }
        
        issues = []
        deductions = []
        
        # Check for duplicates
        duplicates = self.data.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f'Found {duplicates} duplicate timestamps')
            deductions.append(20 * (duplicates / len(self.data)))
        
        # Check for ordering
        if not self.data.index.is_monotonic_increasing:
            issues.append('Timestamps are not monotonically increasing')
            deductions.append(20)
        
        # Check for timezone consistency
        if self.data.index.tz is None:
            issues.append('Timestamps have no timezone information')
            deductions.append(10)
        
        score = 100 - sum(deductions)
        score = max(0, min(100, score))
        
        return {
            'overall_score': score,
            'issues': issues
        }

class EnhancedPreprocessor:
    def __init__(self, 
                 quality_threshold: float = 70,
                 smoothing: float = 10,
                 decay_factor: float = 0.95,
                 n_splits: int = 5,
                 stability_window: int = 20,
                 drift_threshold: float = 0.1,
                 correlation_threshold: float = 0.85,
                 min_iv_threshold: float = 0.02):
        """
        Initialize the enhanced preprocessor with robust validation parameters.
        
        Args:
            quality_threshold: Minimum data quality score required
            smoothing: Smoothing factor for encodings
            decay_factor: Decay factor for time-based features
            n_splits: Number of cross-validation splits
            stability_window: Window size for stability calculations
            drift_threshold: Threshold for feature drift detection
            correlation_threshold: Threshold for correlation analysis
            min_iv_threshold: Minimum information value threshold
        """
        self.quality_threshold = quality_threshold
        self.quality_analyzer = None
        self.feature_importance = None
        self.preprocessing_stats = {
            'feature_engineering': {
                'total_features_created': 0,
                'features_removed_nan': 0,
                'timestamp': None
            },
            # ... other existing stats ...
        }
        
        # Category encoding parameters
        self.smoothing = smoothing
        self.decay_factor = decay_factor
        self.n_splits = n_splits
        self.encodings = {}
        self.market_analyzer = None
        
        # Technical indicator manager
        self.indicator_manager = None
        
        # Validation parameters
        self.stability_window = stability_window
        self.drift_threshold = drift_threshold
        self.correlation_threshold = correlation_threshold
        self.min_iv_threshold = min_iv_threshold
        
        # Bitcoin-specific encoding parameters
        self.volatility_regimes = {
            'low': (0, 0.5),
            'medium': (0.5, 1.5),
            'high': (1.5, float('inf'))
        }
        self.trend_regimes = {
            'strong_downtrend': -0.5,
            'downtrend': -0.2,
            'sideways': 0.2,
            'uptrend': 0.5,
            'strong_uptrend': float('inf')
        }
        
        # Initialize validation components
        self.encoding_validator = None
        self.validation_results = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Enhanced data preprocessing with technical indicator management and validation."""
        # Initialize technical indicator manager
        self.indicator_manager = TechnicalIndicatorManager(df)
        
        # Initialize quality analyzer
        self.quality_analyzer = DataQualityAnalyzer(df)
        quality_report = self.quality_analyzer.analyze_data_quality()
        
        if quality_report['overall_score'] < self.quality_threshold:
            logging.warning(f"Data quality score {quality_report['overall_score']} below threshold {self.quality_threshold}")
        
        # Initialize encoding validator
        self.encoding_validator = EncodingValidator(
            df,
            time_column='Date',
            target_column='Close',
            cv_splits=self.n_splits,
            stability_window=self.stability_window,
            drift_threshold=self.drift_threshold,
            correlation_threshold=self.correlation_threshold,
            min_iv_threshold=self.min_iv_threshold
        )
        
        # Setup cross-validation
        df = self._setup_cross_validation(df)
        
        # Handle gaps and anomalies
        df = self._handle_gaps(df)
        df = self._handle_anomalies(df)
        
        # Get technical features
        technical_features = self.indicator_manager.get_regime_features()
        df = pd.concat([df, technical_features], axis=1)
        
        # Select relevant features
        df = self._select_features(df)
        
        # Final cleaning
        df = self._final_cleaning(df)
        
        # Store preprocessing info
        self.preprocessing_stats.update({
            'quality_report': quality_report,
            'technical_features': technical_features.columns.tolist(),
            'validation_results': self.validation_results
        })
        
        return df, {
            'quality_report': quality_report,
            'preprocessing_stats': self.preprocessing_stats,
            'validation_results': self.validation_results
        }
    
    def _setup_cross_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Setup time series cross-validation splits."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        df['fold'] = -1
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Create folds
        for fold, (_, test_idx) in enumerate(tscv.split(df)):
            df.loc[df.index[test_idx], 'fold'] = fold
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical variables with advanced encoding techniques and validation."""
        # Get adaptive parameters from market conditions
        market_params = self.indicator_manager.get_adaptive_parameters()
        
        # Update encoding parameters
        self.smoothing = market_params['smoothing']
        self.decay_factor = market_params['decay_factor']
        windows = market_params['windows']
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Process each categorical column
        for col in categorical_cols:
            if col == 'Profit/Loss':
                encoded_features = self._encode_profit_loss(df, col, windows)
            elif col in ['Market_Regime', 'Volatility_Regime']:
                encoded_features = self._encode_market_regime(df, col, windows)
            else:
                encoded_features = self._encode_general_categorical(df, col, windows)
            
            # Validate encodings
            validation_results = self.encoding_validator.validate_encoding(col, encoded_features.columns)
            self.validation_results[col] = validation_results
            
            # Apply feature selection based on validation
            selected_features = self._select_validated_features(encoded_features, validation_results)
            df = pd.concat([df, selected_features], axis=1)
            
            # Store encoding information
            self.encodings[col] = {
                'features': selected_features.columns.tolist(),
                'parameters': {
                    'windows': windows,
                    'decay_factor': self.decay_factor,
                    'smoothing': self.smoothing
                },
                'validation_summary': self.encoding_validator.get_validation_summary(col)
            }
        
        return df
    
    def _select_validated_features(self, encoded_features: pd.DataFrame, 
                                 validation_results: Dict) -> pd.DataFrame:
        """Select features based on validation results."""
        # Get quality scores
        quality_scores = validation_results['quality_scores']
        
        # Remove low quality features
        high_quality_features = [
            feature for feature, metrics in quality_scores.items()
            if metrics['overall_score'] >= 0.5
        ]
        
        # Remove highly correlated features while keeping the most important ones
        correlation_analysis = validation_results['correlation_analysis']
        if correlation_analysis['redundancy_detected']:
            high_quality_features = self._remove_redundant_features(
                encoded_features[high_quality_features],
                correlation_analysis,
                quality_scores
            )
        
        return encoded_features[high_quality_features]
    
    def _remove_redundant_features(self, features: pd.DataFrame,
                                 correlation_analysis: Dict,
                                 quality_scores: Dict) -> List[str]:
        """Remove redundant features while keeping the most important ones."""
        selected_features = set(features.columns)
        
        # Sort features by quality score
        sorted_features = sorted(
            selected_features,
            key=lambda x: quality_scores[x]['overall_score'],
            reverse=True
        )
        
        # Remove highly correlated features
        for pair in correlation_analysis['high_correlations']:
            if pair['feature1'] in selected_features and pair['feature2'] in selected_features:
                # Keep the feature with higher quality score
                feature1_score = quality_scores[pair['feature1']]['overall_score']
                feature2_score = quality_scores[pair['feature2']]['overall_score']
                
                if feature1_score >= feature2_score:
                    selected_features.remove(pair['feature2'])
                else:
                    selected_features.remove(pair['feature1'])
        
        return list(selected_features)
    
    def _encode_profit_loss(self, df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
        """Encode profit/loss using cached technical indicators."""
        encoded_features = pd.DataFrame(index=df.index)
        
        # Use cached volatility for regime-specific encoding
        volatility = self.indicator_manager.get_volatility() if self.indicator_manager else None
        for regime, (lower, upper) in self.volatility_regimes.items():
            regime_mask = (volatility > lower) & (volatility <= upper)
            regime_data = df[regime_mask]
            if len(regime_data) > 0:
                regime_encoding = self._target_mean_encode(regime_data, col, 'Close')
                encoded_features[f'{col}_{regime}_mean'] = df[col].map(
                    regime_data.groupby(col)['Close'].mean()
                ).fillna(regime_encoding.mean())
        
        # Use cached trend for trend-aware encoding
        trend = self.indicator_manager.get_trend() if self.indicator_manager else None
        for trend_name, threshold in self.trend_regimes.items():
            if trend_name == 'strong_uptrend':
                trend_mask = trend > self.trend_regimes['uptrend']
            else:
                trend_mask = trend <= threshold
            trend_data = df[trend_mask]
            if len(trend_data) > 0:
                trend_encoding = self._target_mean_encode(trend_data, col, 'Close')
                encoded_features[f'{col}_{trend_name}_mean'] = df[col].map(
                    trend_data.groupby(col)['Close'].mean()
                ).fillna(trend_encoding.mean())
        
        # Use cached RSI for market sentiment encoding
        rsi = self.indicator_manager.get_rsi() if self.indicator_manager else None
        sentiment_regimes = pd.cut(rsi, bins=[0, 30, 70, 100], labels=['oversold', 'neutral', 'overbought'])
        for regime in sentiment_regimes.unique():
            regime_mask = sentiment_regimes == regime
            if regime_mask.any():
                regime_encoding = self._target_mean_encode(df[regime_mask], col, 'Close')
                encoded_features[f'{col}_sentiment_{regime}'] = df[col].map(
                    df[regime_mask].groupby(col)['Close'].mean()
                ).fillna(regime_encoding.mean())
        
        # Add momentum-based features
        for window in windows:
            momentum = self.indicator_manager.get_momentum(window) if self.indicator_manager else None
            vol_adj_momentum = self.indicator_manager.get_momentum(window, volume_adjusted=True) if self.indicator_manager else None
            
            encoded_features[f'{col}_momentum_{window}'] = df[col].map(
                df.groupby(col)['Close'].apply(lambda x: (x * momentum).mean())
            )
            encoded_features[f'{col}_vol_adj_momentum_{window}'] = df[col].map(
                df.groupby(col)['Close'].apply(lambda x: (x * vol_adj_momentum).mean())
            )
        
        return encoded_features
    
    def _encode_market_regime(self, df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
        """Encode market regime using cached technical indicators."""
        encoded_features = pd.DataFrame(index=df.index)
        
        # Get cached indicators
        volatility = self.indicator_manager.get_volatility() if self.indicator_manager else None
        trend = self.indicator_manager.get_trend() if self.indicator_manager else None
        momentum = self.indicator_manager.get_momentum() if self.indicator_manager else None
        rsi = self.indicator_manager.get_rsi() if self.indicator_manager else None
        
        # Create regime classification
        regime_df = pd.DataFrame({
            'volatility': volatility,
            'trend': trend,
            'momentum': momentum,
            'rsi': rsi
        })
        
        # Classify regimes using the same logic as before
        regimes = regime_df.apply(self._classify_regime, axis=1)
        regime_dummies = pd.get_dummies(regimes, prefix='regime')
        encoded_features = pd.concat([encoded_features, regime_dummies], axis=1)
        
        # Add regime stability metrics using cached values
        for window in windows:
            regime_changes = (regimes != regimes.shift(1)).rolling(window=window).mean()
            encoded_features[f'regime_change_freq_{window}'] = regime_changes
            
            current_regime_duration = regimes.groupby(
                (regimes != regimes.shift(1)).cumsum()
            ).cumcount()
            encoded_features[f'regime_persistence_{window}'] = current_regime_duration.rolling(window).mean()
            
            stability_score = 1 - regime_changes
            encoded_features[f'regime_stability_{window}'] = stability_score
        
        return encoded_features
    
    def _classify_regime(self, row: pd.Series) -> str:
        """Classify market regime based on multiple indicators."""
        if row['volatility'] > 0.5:
            if row['trend'] > 0.02:
                return 'volatile_uptrend'
            elif row['trend'] < -0.02:
                return 'volatile_downtrend'
            return 'high_volatility'
        else:
            if row['trend'] > 0.02:
                return 'stable_uptrend'
            elif row['trend'] < -0.02:
                return 'stable_downtrend'
            return 'ranging'
    
    def _encode_general_categorical(self, df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
        """Encode general categorical variables."""
        # For non-target related categoricals, use simpler encoding methods
        encoded_features = pd.DataFrame(index=df.index)
        
        # One-hot encoding for low cardinality
        if df[col].nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            encoded_features = pd.concat([encoded_features, dummies], axis=1)
        else:
            # Frequency encoding
            freq_encoding = df[col].map(df[col].value_counts(normalize=True))
            encoded_features[f'{col}_freq'] = freq_encoding
            
            # Rolling frequency encoding
            for window in windows:
                rolling_freq = df.groupby(col)[col].rolling(window).count() / window
                encoded_features[f'{col}_rolling_freq_{window}'] = rolling_freq
        
        return encoded_features
    
    def _target_mean_encode(self, df: pd.DataFrame, col: str, target: str) -> pd.Series:
        """Perform target mean encoding with cross-validation to prevent leakage."""
        df_encoded = df.copy()
        means = df.groupby(col)[target].mean()
        global_mean = df[target].mean()
        
        # Smooth means with global mean
        smoothed_means = (means * df.groupby(col).size() + global_mean * self.smoothing) / \
                        (df.groupby(col).size() + self.smoothing)
        
        for fold in df['fold'].unique():
            train_idx = df['fold'] != fold
            test_idx = df['fold'] == fold
            
            # Calculate means excluding current fold
            fold_means = df[train_idx].groupby(col)[target].mean()
            # Apply smoothing
            fold_means = (fold_means * df[train_idx].groupby(col).size() + global_mean * self.smoothing) / \
                        (df[train_idx].groupby(col).size() + self.smoothing)
            
            # Fill in means for test fold
            df_encoded.loc[test_idx, f'{col}_target_mean'] = df.loc[test_idx, col].map(fold_means).fillna(global_mean)
        
        return df_encoded[f'{col}_target_mean']
    
    def _impact_encode(self, df: pd.DataFrame, col: str, target: str) -> pd.Series:
        """Encode based on the magnitude of impact on the target variable."""
        impact_values = {}
        for category in df[col].unique():
            cat_mask = df[col] == category
            # Calculate average absolute change in target when this category occurs
            target_changes = df[target].diff().abs()
            impact = target_changes[cat_mask].mean() / target_changes.mean()
            impact_values[category] = impact
        
        return df[col].map(impact_values)
    
    def _rolling_mean_encode(self, df: pd.DataFrame, col: str, target: str, windows: List[int]) -> pd.DataFrame:
        """Create rolling mean encodings with multiple window sizes."""
        encodings = {}
        for window in windows:
            # Calculate rolling means for each category
            rolling_means = df.groupby(col)[target].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            encodings[f'{col}_rolling_mean_{window}'] = rolling_means
        return pd.DataFrame(encodings)
    
    def _time_decay_encode(self, df: pd.DataFrame, col: str, target: str, windows: List[int]) -> pd.DataFrame:
        """Create time-decay weighted encodings with multiple window sizes."""
        encodings = {}
        for window in windows:
            weights = np.power(self.decay_factor, np.arange(window)[::-1])
            weights = weights / weights.sum()
            
            # Calculate weighted means for each category
            decay_means = df.groupby(col)[target].transform(
                lambda x: x.rolling(window, min_periods=1)
                          .apply(lambda w: (w * weights[:len(w)]).sum() / weights[:len(w)].sum())
            )
            encodings[f'{col}_decay_mean_{window}'] = decay_means
        return pd.DataFrame(encodings)
    
    def _handle_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and gaps in the data using sophisticated imputation methods.
        
        Args:
            data: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        # Store original index for later
        original_index = data.index
        
        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Initialize results dictionary for quality tracking
        imputation_stats = {
            'gaps_detected': {},
            'imputation_methods': {},
            'quality_metrics': {}
        }
        
        # Handle numeric columns with KNN imputation
        if len(numeric_cols) > 0:
            # First pass: Simple forward fill for initialization
            data[numeric_cols] = data[numeric_cols].fillna(method='ffill')
            
            # Configure KNN imputer with adaptive n_neighbors
            n_samples = len(data)
            n_neighbors = min(max(int(np.sqrt(n_samples)), 5), 20)  # Between 5 and 20
            
            # Initialize KNN imputer with weights based on market volatility
            volatility = self.indicator_manager.get_volatility() if self.indicator_manager else None
            weights = 'distance' if volatility is not None and volatility.mean() > 0.2 else 'uniform'
            
            imputer = KNNImputer(
                n_neighbors=n_neighbors,
                weights=weights,
                metric='nan_euclidean'
            )
            
            # Prepare data for KNN imputation
            scaler = RobustScaler()  # Use RobustScaler to handle outliers
            scaled_data = scaler.fit_transform(data[numeric_cols])
            
            # Perform KNN imputation
            imputed_data = imputer.fit_transform(scaled_data)
            
            # Inverse transform the scaled data
            imputed_data = scaler.inverse_transform(imputed_data)
            
            # Create DataFrame with imputed values
            imputed_df = pd.DataFrame(
                imputed_data,
                columns=numeric_cols,
                index=original_index
            )
            
            # Validate imputation quality
            for col in numeric_cols:
                original = data[col]
                imputed = imputed_df[col]
                missing_mask = original.isna()
                
                if missing_mask.any():
                    # Calculate imputation statistics
                    imputation_stats['gaps_detected'][col] = missing_mask.sum()
                    imputation_stats['imputation_methods'][col] = 'KNN'
                    
                    # Analyze local statistics around imputed values
                    window_size = min(30, len(data) // 10)  # Adaptive window size
                    rolling_std = original[~missing_mask].rolling(window=window_size).std()
                    rolling_mean = original[~missing_mask].rolling(window=window_size).mean()
                    
                    # Check if imputed values are within reasonable bounds
                    imputed_values = imputed[missing_mask]
                    nearby_means = rolling_mean.fillna(method='ffill').fillna(method='bfill')
                    nearby_stds = rolling_std.fillna(method='ffill').fillna(method='bfill')
                    
                    common_index = imputed_values.index
                    nearby_means = nearby_means.reindex(common_index)
                    nearby_stds = nearby_stds.reindex(common_index)
                    within_bounds = np.abs(imputed_values - nearby_means) <= 3 * nearby_stds
                    
                    imputation_stats['quality_metrics'][col] = {
                        'within_bounds_percentage': within_bounds.mean() * 100,
                        'total_imputed': len(imputed_values),
                        'mean_deviation': np.abs(imputed_values - nearby_means).mean()
                    }
            
            # Update numeric columns with imputed values
            data[numeric_cols] = imputed_df
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                # For categorical data, use forward fill then backward fill
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                
                missing_mask = data[col].isna()
                if missing_mask.any():
                    imputation_stats['gaps_detected'][col] = missing_mask.sum()
                    imputation_stats['imputation_methods'][col] = 'ffill-bfill'
        
        # Store imputation statistics
        self.preprocessing_stats['imputation'] = imputation_stats
        
        # Handle any remaining infinities
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _handle_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle detected anomalies."""
        anomalies = self.quality_analyzer.quality_scores['anomalies']
        df = data.copy()
        
        # Handle price anomalies
        price_anomalies = set(anomalies['details']['price']['anomaly_indices'])
        if price_anomalies:
            # Use rolling median for price anomalies
            window = 5
            for col in ['Open', 'High', 'Low', 'Close']:
                rolling_median = df[col].rolling(window=window, center=True).median()
                # If price_anomalies are integer positions, convert to index labels for .loc
                if isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
                    anomaly_labels = df.index[list(price_anomalies)]
                    df.loc[anomaly_labels, col] = rolling_median.loc[anomaly_labels]
                else:
                    # fallback: if index is RangeIndex or integers
                    df.iloc[list(price_anomalies), df.columns.get_loc(col)] = rolling_median.iloc[list(price_anomalies)]
        
        # Handle volume anomalies
        volume_anomalies = set(anomalies['details']['volume']['anomaly_indices'])
        if volume_anomalies:
            # Use rolling median for volume
            window = 5
            rolling_median = df['Volume'].rolling(window=window, center=True).median()
            # If volume_anomalies are integer positions, convert to index labels for .loc
            if isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
                anomaly_labels = df.index[list(volume_anomalies)]
                df.loc[anomaly_labels, 'Volume'] = rolling_median.loc[anomaly_labels]
            else:
                # fallback: if index is RangeIndex or integers
                df.iloc[list(volume_anomalies), df.columns.get_loc('Volume')] = rolling_median.iloc[list(volume_anomalies)]
        
        self.preprocessing_stats['anomaly_handling'] = {
            'price_anomalies_handled': len(price_anomalies),
            'volume_anomalies_handled': len(volume_anomalies)
        }
        
        return df
    
    def _assess_feature_stability(self, data: pd.DataFrame, features: List[str], n_splits: int = 5) -> Dict[str, float]:
        """
        Assess feature stability across time-based cross-validation splits.
        
        Args:
            data: DataFrame containing features
            features: List of feature columns to assess
            n_splits: Number of time-based CV splits
        
        Returns:
            Dictionary with stability scores for each feature
        """
        # Create time-based splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        stability_scores = defaultdict(list)
        
        # Calculate stability metrics across splits
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            for feature in features:
                # Calculate Population Stability Index (PSI)
                psi = self._calculate_psi(train_data[feature], test_data[feature])
                
                # Calculate distribution similarity using KS test
                ks_stat, _ = ks_2samp(train_data[feature].dropna(), 
                                     test_data[feature].dropna())
                
                # Calculate mean and std deviation shift
                mean_shift = abs(train_data[feature].mean() - test_data[feature].mean())
                std_shift = abs(train_data[feature].std() - test_data[feature].std())
                
                # Combine metrics into stability score
                stability = 1 - (0.4 * psi + 0.3 * ks_stat + 
                               0.15 * mean_shift + 0.15 * std_shift)
                
                stability_scores[feature].append(stability)
        
        # Calculate final stability scores
        final_scores = {
            feature: {
                'mean_stability': np.mean(scores),
                'std_stability': np.std(scores),
                'min_stability': np.min(scores),
                'max_stability': np.max(scores)
            }
            for feature, scores in stability_scores.items()
        }
        
        return final_scores

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        # Create bins based on expected distribution
        bins_range = pd.qcut(expected, q=bins, duplicates='drop', labels=False)
        expected_bins = pd.value_counts(bins_range, normalize=True)
        
        # Map actual values to same bins
        actual_bins = pd.value_counts(
            pd.cut(actual, bins=expected_bins.index.max() + 1, labels=False),
            normalize=True
        )
        
        # Calculate PSI
        psi = sum((actual_bins - expected_bins) * 
                  np.log(actual_bins / expected_bins))
        
        return psi
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical and market regime features with stability analysis."""
        
        # Initialize TechnicalIndicatorManager with the data
        tim = TechnicalIndicatorManager(data)
        
        # Get all technical indicators with custom windows
        windows = {
            'volatility': [7, 14, 21, 30],
            'momentum': [10, 20, 30, 60],
            'trend': [50, 100, 200]
        }
        
        # Get all technical features
        technical_features = tim.get_regime_features(windows=windows)
        
        # Merge with original data
        enhanced_data = pd.concat([data, technical_features], axis=1)
        
        # Remove features with too many NaN values
        nan_threshold = 0.1
        valid_columns = enhanced_data.columns[
            enhanced_data.isna().mean() < nan_threshold
        ].tolist()
        
        # Keep only valid columns
        enhanced_data = enhanced_data[
            list(set(valid_columns + list(data.columns)))
        ]
        
        # Assess feature stability
        stability_scores = self._assess_feature_stability(
            enhanced_data,
            [col for col in enhanced_data.columns if col not in data.columns]
        )
        
        # Filter features based on stability
        stability_threshold = 0.7  # Features must be at least 70% stable
        stable_features = [
            feature for feature, scores in stability_scores.items()
            if scores['mean_stability'] >= stability_threshold
        ]
        
        # Keep only stable features and original columns
        final_columns = list(set(stable_features + list(data.columns)))
        enhanced_data = enhanced_data[final_columns]
        
        # Forward fill any remaining NaN values
        enhanced_data = enhanced_data.fillna(method='ffill').fillna(method='bfill')
        
        # Update preprocessing stats
        self.preprocessing_stats['feature_engineering'].update({
            'total_features_created': len(technical_features.columns),
            'features_removed_nan': len(technical_features.columns) - len(valid_columns),
            'features_removed_instability': len(valid_columns) - len(stable_features),
            'final_feature_count': len(stable_features),
            'stability_scores': stability_scores,
            'timestamp': pd.Timestamp.now()
        })
        
        return enhanced_data
    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Retain all features instead of selecting top 30."""
         # Keep all columns to avoid dropping percentreturn and technical indicators
        selected_columns = data.columns.tolist()
    
         # Update stats
        self.preprocessing_stats['feature_selection'] = {
            'total_features_selected': len(selected_columns),
            'importance_threshold': 0,
            'timestamp': pd.Timestamp.now()
        }
        
        return data[selected_columns]
    
    # def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """Select most important features after engineering."""
        
    #     # Calculate feature importance using correlation with target
    #     importance_scores = {}
    #     target_col = 'Close'  # or your specific target column
        
    #     for col in data.columns:
    #         if col != target_col:
    #             importance_scores[col] = abs(data[col].corr(data[target_col]))
        
    #     # Sort features by importance
    #     importance_df = pd.Series(importance_scores).sort_values(ascending=False)
        
    #     # Select top features (e.g., top 30)
    #     top_features = importance_df.head(30).index.tolist()
        
    #     # Add important original columns
    #     essential_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    #     selected_columns = list(set(top_features + essential_columns))
        
    #     # Update stats
    #     self.preprocessing_stats['feature_selection'] = {
    #         'total_features_selected': len(selected_columns),
    #         'importance_threshold': importance_df.iloc[30] if len(importance_df) > 30 else 0,
    #         'timestamp': pd.Timestamp.now()
    #     }
        
    #     return data[selected_columns]
    
    def _final_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform final data cleaning and validation."""
        # Get initial statistics
        initial_stats = {
            'rows_with_missing': data.isnull().any(axis=1).sum(),
            'columns_with_missing': data.isnull().any(axis=0).sum()
        }
        
        # Handle any remaining missing values with forward/backward fill
        # This serves as a fallback for any edge cases not handled by KNN imputation
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Final statistics after cleaning
        final_stats = {
            'rows_with_missing': data.isnull().any(axis=1).sum(),
            'columns_with_missing': data.isnull().any(axis=0).sum()
        }
        
        # Update preprocessing stats
        self.preprocessing_stats['final_cleaning'] = {
            'initial': initial_stats,
            'final': final_stats,
            'improvement': {
                'rows_fixed': initial_stats['rows_with_missing'] - final_stats['rows_with_missing'],
                'columns_fixed': initial_stats['columns_with_missing'] - final_stats['columns_with_missing']
            }
        }
        
        return data

    def visualize_encoding_analysis(self, col: str, save_path: str = None):
        """Generate comprehensive visualizations for encoding analysis results."""
        if not hasattr(self, 'encoding_importance') or not hasattr(self, 'encoding_correlations'):
            raise ValueError("No encoding analysis results available. Run encoding first.")
        
        analysis_results = self.get_encoding_analysis(col)
        
        # Create a figure with subplots
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # 1. Feature Importance Plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_feature_importance(analysis_results['importance_scores'], ax1)
        
        # 2. Correlation Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_correlation_heatmap(analysis_results['correlation_matrix'], ax2)
        
        # 3. Feature Clustering Dendrogram
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_feature_clusters(analysis_results['correlation_matrix'], ax3)
        
        # 4. SHAP Summary Plot
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_shap_summary(analysis_results['importance_scores'], ax4)
        
        # 5. Time Series of Top Features
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_top_features_timeline(col, analysis_results['top_features'], ax5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self, importance_scores: pd.DataFrame, ax: plt.Axes):
        """Plot feature importance scores."""
        # Sort by combined score
        sorted_scores = importance_scores.sort_values('combined_score', ascending=True)
        
        # Create stacked bar chart
        bottom = np.zeros(len(sorted_scores))
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        metrics = ['rf_importance', 'shap_importance', 'correlation']
        
        for metric, color in zip(metrics, colors):
            ax.barh(sorted_scores.index, sorted_scores[metric], left=bottom, 
                   color=color, alpha=0.7, label=metric)
            bottom += sorted_scores[metric]
        
        ax.set_title('Feature Importance Analysis', fontsize=12, pad=20)
        ax.set_xlabel('Importance Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add combined score as text
        for i, (idx, row) in enumerate(sorted_scores.iterrows()):
            ax.text(bottom[i], i, f' {row["combined_score"]:.3f}',
                   va='center', fontsize=8)
    
    def _plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, ax: plt.Axes):
        """Plot correlation heatmap with hierarchical clustering."""
        import seaborn as sns
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        
        # Convert correlation matrix to distance matrix
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(squareform(distance_matrix), method='complete')
        
        # Reorder matrix based on clustering
        idx = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
        reordered_matrix = correlation_matrix.iloc[idx, idx]
        
        # Plot heatmap
        sns.heatmap(reordered_matrix, ax=ax, cmap='RdYlBu_r', center=0,
                   xticklabels=True, yticklabels=True)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=12, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    def _plot_feature_clusters(self, correlation_matrix: pd.DataFrame, ax: plt.Axes):
        """Plot dendrogram of feature clusters."""
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        
        # Convert correlation matrix to distance matrix
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(squareform(distance_matrix), method='complete')
        
        # Plot dendrogram
        hierarchy.dendrogram(linkage, labels=correlation_matrix.columns,
                           leaf_rotation=45, ax=ax)
        
        ax.set_title('Feature Clustering Dendrogram', fontsize=12, pad=20)
    
    def _plot_shap_summary(self, importance_scores: pd.DataFrame, ax: plt.Axes):
        """Plot SHAP values summary."""
        # Create violin plot of SHAP importance distributions
        sorted_features = importance_scores.sort_values('shap_importance', ascending=True)
        
        violin_parts = ax.violinplot(
            [np.random.normal(0, v, 100) for v in sorted_features['shap_importance']],
            points=100, vert=False, showmeans=True
        )
        
        # Customize violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.7)
        
        ax.set_yticks(np.arange(1, len(sorted_features) + 1))
        ax.set_yticklabels(sorted_features.index)
        ax.set_title('SHAP Values Distribution', fontsize=12, pad=20)
        ax.set_xlabel('SHAP Value')
    
    def _plot_top_features_timeline(self, col: str, top_features: List[str], ax: plt.Axes):
        """Plot time series of top encoded features."""
        if not hasattr(self, 'encoded_features'):
            raise ValueError("No encoded features available")
        
        # Get encoded features data
        data = self.encoded_features[top_features]
        
        # Plot each feature
        for feature in top_features:
            # Normalize the feature for comparison
            normalized = (data[feature] - data[feature].mean()) / data[feature].std()
            ax.plot(normalized.index, normalized.values, label=feature, alpha=0.7)
        
        ax.set_title(f'Top Encoded Features Timeline for {col}', fontsize=12, pad=20)
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def visualize_market_conditions(self, save_path: str = None):
        """Visualize market conditions used in encoding."""
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, figure=fig)
        
        # 1. Liquidity Regimes
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_liquidity_regimes(ax1)
        
        # 2. Market Sentiment (RSI)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_market_sentiment(ax2)
        
        # 3. Market Efficiency
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_market_efficiency(ax3)
        
        # 4. Volatility Clusters
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_volatility_clusters(ax4)
        
        # 5. Combined Market Conditions
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_combined_conditions(ax5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_liquidity_regimes(self, ax: plt.Axes):
        """Plot liquidity regimes over time."""
        liquidity = self.data['Volume'].rolling(window=20).mean() / self.data['Close']
        liquidity_regimes = pd.qcut(liquidity, q=3, labels=['Low', 'Medium', 'High'])
        
        # Plot liquidity
        ax.plot(liquidity.index, liquidity.values, color='#3498db', alpha=0.7)
        
        # Color background by regime
        for regime in ['Low', 'Medium', 'High']:
            mask = liquidity_regimes == regime
            ax.fill_between(liquidity.index, liquidity.min(), liquidity.max(),
                          where=mask, alpha=0.2, label=f'{regime} Liquidity')
        
        ax.set_title('Liquidity Regimes', fontsize=12, pad=20)
        ax.set_ylabel('Volume/Price Ratio')
        ax.legend()
    
    def _plot_market_sentiment(self, ax: plt.Axes):
        """Plot RSI and sentiment regimes."""
        rsi = self._calculate_rsi(self.data['Close'])
        
        # Plot RSI
        ax.plot(rsi.index, rsi.values, color='#3498db', alpha=0.7)
        
        # Add overbought/oversold lines
        ax.axhline(y=70, color='#e74c3c', linestyle='--', alpha=0.5)
        ax.axhline(y=30, color='#2ecc71', linestyle='--', alpha=0.5)
        
        # Color regions
        ax.fill_between(rsi.index, 70, 100, alpha=0.1, color='#e74c3c', label='Overbought')
        ax.fill_between(rsi.index, 0, 30, alpha=0.1, color='#2ecc71', label='Oversold')
        
        ax.set_title('Market Sentiment (RSI)', fontsize=12, pad=20)
        ax.set_ylabel('RSI')
        ax.legend()
    
    def _plot_market_efficiency(self, ax: plt.Axes):
        """Plot market efficiency ratio."""
        efficiency = self._calculate_market_efficiency_ratio(self.data['Close'])
        efficiency_regimes = pd.qcut(efficiency, q=3, labels=['Low', 'Medium', 'High'])
        
        # Plot efficiency ratio
        ax.plot(efficiency.index, efficiency.values, color='#3498db', alpha=0.7)
        
        # Color background by regime
        for regime in ['Low', 'Medium', 'High']:
            mask = efficiency_regimes == regime
            ax.fill_between(efficiency.index, efficiency.min(), efficiency.max(),
                          where=mask, alpha=0.2, label=f'{regime} Efficiency')
        
        ax.set_title('Market Efficiency Ratio', fontsize=12, pad=20)
        ax.set_ylabel('Efficiency Ratio')
        ax.legend()
    
    def _plot_volatility_clusters(self, ax: plt.Axes):
        """Plot volatility clusters."""
        vol_clusters = self._detect_volatility_clusters(self.data['Close'])
        returns = self.data['Close'].pct_change()
        
        # Plot returns
        ax.plot(returns.index, returns.values, color='#3498db', alpha=0.7)
        
        # Color background by cluster
        for cluster in ['low_cluster', 'high_cluster']:
            mask = vol_clusters == cluster
            ax.fill_between(returns.index, returns.min(), returns.max(),
                          where=mask, alpha=0.2,
                          label='High Volatility' if cluster == 'high_cluster' else 'Low Volatility')
        
        ax.set_title('Volatility Clusters', fontsize=12, pad=20)
        ax.set_ylabel('Returns')
        ax.legend()
    
    def _plot_combined_conditions(self, ax: plt.Axes):
        """Plot combined market conditions."""
        # Normalize all conditions to [0, 1] range
        conditions = pd.DataFrame(index=self.data.index)
        
        # Add liquidity
        liquidity = self.data['Volume'].rolling(window=20).mean() / self.data['Close']
        conditions['Liquidity'] = (liquidity - liquidity.min()) / (liquidity.max() - liquidity.min())
        
        # Add RSI
        rsi = self._calculate_rsi(self.data['Close'])
        conditions['RSI'] = rsi / 100
        
        # Add efficiency
        efficiency = self._calculate_market_efficiency_ratio(self.data['Close'])
        conditions['Efficiency'] = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min())
        
        # Add volatility
        volatility = self.data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        conditions['Volatility'] = (volatility - volatility.min()) / (volatility.max() - volatility.min())
        
        # Plot all conditions
        for column in conditions.columns:
            ax.plot(conditions.index, conditions[column], label=column, alpha=0.7)
        
        ax.set_title('Combined Market Conditions', fontsize=12, pad=20)
        ax.set_ylabel('Normalized Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    

class TechnicalIndicatorManager:
    """Unified Technical Indicator Manager with comprehensive feature set."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Technical Indicator Manager.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
                Required columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.data = data
        self.cache = {}  # For storing computed values
        
    def get_regime_features(self, windows: Dict[str, List[int]] = None) -> pd.DataFrame:
        """Get all technical indicators and regime features."""
        if windows is None:
            windows = {
                'volatility': [7, 14, 21],
                'momentum': [10, 20, 30],
                'trend': [50, 100, 200]
            }
        
        features = pd.DataFrame(index=self.data.index)
        
        # 1. Core Price Indicators
        features['RSI'] = self.get_rsi()
        features['MACD'] = self.get_macd()
        features['PPO'] = self.calculate_ppo(self.data['Close'])
        features['KST'] = self.calculate_kst(self.data['Close'])
        
        # 2. Volatility Indicators
        for window in windows['volatility']:
            features[f'volatility_{window}'] = self.get_volatility(window)
        features['BB_Width'] = self.calculate_bb_width(self.data['Close'])
        features['KC_Width'] = self.calculate_keltner_width()
        features['ATR'] = self.calculate_atr()
        features['ATR_Ratio'] = self.calculate_atr_ratio()
        
        # 3. Volume-Price Indicators
        features['CMF'] = self.calculate_cmf()
        features['Force_Index'] = self.calculate_force_index()
        features['MFI'] = self.calculate_money_flow_index()
        features['VWAP'] = self.calculate_vwap()
        features['OBV'] = self.calculate_obv()
        
        # 4. Trend & Momentum Indicators
        for window in windows['momentum']:
            features[f'momentum_{window}'] = self.get_momentum(window)
        features['ADX'] = self.calculate_adx()
        features['DPO'] = self.calculate_dpo()
        features['Aroon_Osc'] = self.calculate_aroon_oscillator()
        
        # 5. Market Efficiency & Liquidity
        features['MER'] = self.get_market_efficiency_ratio()
        liquidity_metrics = self.get_liquidity_metrics()
        features = pd.concat([features, liquidity_metrics], axis=1)
        
        # 6. Pattern Recognition
        features['HH_LL_Ratio'] = self.calculate_hh_ll_ratio()
        features['Price_ROC'] = self.calculate_price_roc()
        features['RSI_Divergence'] = self.calculate_rsi_divergence()
        
        return features

    # === Core Technical Indicators ===
    def get_rsi(self, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD."""
        exp1 = self.data['Close'].ewm(span=fast).mean()
        exp2 = self.data['Close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line

    def calculate_ppo(self, close: pd.Series) -> pd.Series:
        """Calculate Price Percentage Oscillator."""
        ema26 = close.ewm(span=26).mean()
        ema12 = close.ewm(span=12).mean()
        return ((ema12 - ema26) / ema26) * 100

    def calculate_kst(self, close: pd.Series) -> pd.Series:
        """Calculate Know Sure Thing (KST)."""
        rcma1 = close.diff(10).rolling(10).mean()
        rcma2 = close.diff(15).rolling(10).mean()
        rcma3 = close.diff(20).rolling(10).mean()
        rcma4 = close.diff(30).rolling(15).mean()
        kst = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
        return kst

    # === Volatility Indicators ===
    def get_volatility(self, window: int = 20, annualized: bool = True) -> pd.Series:
        """Calculate price volatility."""
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        vol = log_returns.rolling(window=window).std()
        if annualized:
            vol = vol * np.sqrt(252)
        return vol

    def calculate_bb_width(self, close: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Bollinger Band Width."""
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        return ((sma + (2 * std)) - (sma - (2 * std))) / sma * 100

    def calculate_keltner_width(self, window: int = 20) -> pd.Series:
        """Calculate Keltner Channel Width."""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        atr = self.calculate_atr(window)
        ema = typical_price.ewm(span=window).mean()
        return ((ema + (2 * atr)) - (ema - (2 * atr))) / ema * 100

    def calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    def calculate_atr_ratio(self, window: int = 14) -> pd.Series:
        """Calculate ATR Ratio (ATR / Price)."""
        atr = self.calculate_atr(window)
        return atr / self.data['Close'] * 100

    # === Volume-Price Indicators ===
    def calculate_cmf(self, window: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        mfm = ((self.data['Close'] - self.data['Low']) - 
               (self.data['High'] - self.data['Close'])) / (self.data['High'] - self.data['Low'])
        mfv = mfm * self.data['Volume']
        return mfv.rolling(window=window).sum() / self.data['Volume'].rolling(window=window).sum()

    def calculate_force_index(self, window: int = 13) -> pd.Series:
        """Calculate Force Index."""
        return self.data['Close'].diff(1) * self.data['Volume']

    def calculate_money_flow_index(self, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        money_flow = typical_price * self.data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        return 100 - (100 / (1 + (positive_flow / negative_flow)))

    def calculate_vwap(self) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        return (typical_price * self.data['Volume']).cumsum() / self.data['Volume'].cumsum()

    def calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume."""
        return (np.sign(self.data['Close'].diff()) * self.data['Volume']).cumsum()

    # === Trend & Momentum Indicators ===
    def get_momentum(self, window: int = 20, volume_adjusted: bool = False) -> pd.Series:
        """Calculate price momentum."""
        momentum = self.data['Close'].diff(window)
        if volume_adjusted:
            momentum = momentum * self.data['Volume']
        return momentum

    def calculate_adx(self, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        plus_dm = self.data['High'].diff()
        minus_dm = self.data['Low'].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        tr = self.calculate_atr(window)
        plus_di = 100 * (plus_dm.rolling(window).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / tr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window).mean()

    def calculate_dpo(self, window: int = 20) -> pd.Series:
        """Calculate Detrended Price Oscillator."""
        shift = int(window/2 + 1)
        return self.data['Close'] - self.data['Close'].rolling(window).mean().shift(shift)

    def calculate_aroon_oscillator(self, window: int = 25) -> pd.Series:
        """Calculate Aroon Oscillator."""
        high = self.data['High'].rolling(window + 1)
        low = self.data['Low'].rolling(window + 1)
        high_days = high.apply(lambda x: x.argmax()) / window * 100
        low_days = low.apply(lambda x: x.argmin()) / window * 100
        return high_days - low_days

    # === Pattern Recognition ===
    def calculate_hh_ll_ratio(self, window: int = 20) -> pd.Series:
        """Calculate ratio of higher highs to lower lows."""
        highs = self.data['High'].rolling(window).max()
        lows = self.data['Low'].rolling(window).min()
        hh = (self.data['High'] > highs.shift(1)).rolling(window).sum()
        ll = (self.data['Low'] < lows.shift(1)).rolling(window).sum()
        return hh / (ll + 1)  # Add 1 to avoid division by zero

    def calculate_price_roc(self, window: int = 12) -> pd.Series:
        """Calculate Price Rate of Change."""
        return (self.data['Close'] - self.data['Close'].shift(window)) / self.data['Close'].shift(window) * 100

    def calculate_rsi_divergence(self, window: int = 14) -> pd.Series:
        """Calculate RSI Divergence."""
        rsi = self.get_rsi(window)
        price_momentum = self.get_momentum(window)
        return rsi - rsi.shift(1) - (price_momentum - price_momentum.shift(1))

    # === Market Efficiency & Liquidity ===
    def get_market_efficiency_ratio(self, window: int = 20) -> pd.Series:
        """Calculate Market Efficiency Ratio."""
        direction = abs(self.data['Close'] - self.data['Close'].shift(window))
        volatility = abs(self.data['Close'].diff()).rolling(window).sum()
        return direction / volatility

    def get_liquidity_metrics(self, window: int = 20) -> pd.DataFrame:
        """Calculate various liquidity metrics."""
        metrics = pd.DataFrame(index=self.data.index)
        
        # Amihud illiquidity
        metrics['illiquidity'] = abs(self.data['Close'].pct_change()) / self.data['Volume']
        
        # Volume-volatility ratio
        metrics['vol_vol_ratio'] = (self.data['Volume'] / self.data['Volume'].rolling(window).mean()) / \
                                  (self.get_volatility(window) / self.get_volatility(window).rolling(window).mean())
        
        return metrics

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, dict]:
    """Enhanced data loading with quality checks and preprocessing."""
    logging.info("\nLoading and preprocessing data...")
    
    # Load raw data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Subset to final features for production (include raw Volume for anomaly detection)
    df = df[['Open','High','Low','Volume'] + FINAL_FEATURES + ['Close']]
    
    # Initialize preprocessor
    preprocessor = EnhancedPreprocessor(quality_threshold=70)
    
    # Preprocess data
    df_processed, preprocessing_info = preprocessor.preprocess_data(df)
    
    # Print quality report
    quality_report = preprocessing_info['quality_report']
    logging.info(f"\nData Quality Score: {quality_report['overall_score']:.2f}/100")
    
    if quality_report['overall_score'] < preprocessor.quality_threshold:
        logging.warning("\nWarning: Data quality below threshold!")
        logging.info("\nQuality Issues:")
        for metric, score in quality_report.items():
            if isinstance(score, dict) and 'issues' in score:
                for issue in score['issues']:
                    logging.info(f"- {metric}: {issue}")
    
    # Print preprocessing statistics
    stats = preprocessing_info['preprocessing_stats']
    logging.info("\nPreprocessing Statistics:")
    logging.info(f"- Features engineered: {stats['feature_engineering']['engineered_features']}")
    logging.info(f"- Features selected: {stats['feature_selection']['selected_features']}")
    gaps_filled = stats.get('gap_handling', {}).get('gaps_filled', 0)
    logging.info(f"- Gaps filled: {gaps_filled}")
    logging.info(f"- Anomalies handled: {stats['anomaly_handling']['price_anomalies_handled']} price, "
                 f"{stats['anomaly_handling']['volume_anomalies_handled']} volume")
    
    return df_processed, preprocessing_info


def load_preprocess_data(file_path: str, feature_groups_path: str = 'feature_groups.json') -> Tuple[pd.DataFrame, dict]:
    """
    Enhanced data preprocessing pipeline with robust cleaning steps
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Initialize preprocessor
        preprocessor = EnhancedPreprocessor(
            quality_threshold=70,
            smoothing=10,
            n_splits=5,
            stability_window=20,
            drift_threshold=0.1,
            correlation_threshold=0.85
        )
    
        # Preprocess data using the unified preprocess_data method
        df_processed, preprocessing_info = preprocessor.preprocess_data(df)
        
        # Load and apply feature groups
        with open(feature_groups_path, 'r') as f:
            feature_groups = json.load(f)
            
        # Define base features
        base_features = [
            'volatility_volume_ratio', 'stoch_k', 'stoch_d', 
            'rolling_vol_7', 'rolling_return_7', 'CCI', 
            'ADX_pos', 'price_range', 'volatility',
            'Williams_%R', 'RSI', 'MACD'
        ]
        
        # Merge features
        selected_features = base_features.copy()
        for group in feature_groups.values():
            selected_features.extend(group)
        
        # Remove duplicates while preserving order
        selected_features = list(dict.fromkeys(selected_features))
        
        # Filter to selected features, keeping only those present in df_processed
        available_features = [f for f in selected_features if f in df_processed.columns]
        df_final = df_processed[available_features]
        
        # Collect preprocessing statistics
        preprocessing_stats = {
            'preprocessing_info': preprocessing_info,
            'selected_features': available_features,
            'feature_groups': feature_groups,
            'final_shape': df_final.shape
        }
        
        logging.info(f"Preprocessing complete. Final shape: {df_final.shape}")
        
        # Adjust return to match expected tuple length (df, preprocessing_stats, outlier_masks)
        # Since preprocess_data may not return outlier_masks, use a placeholder if needed
        outlier_masks = {}  # Placeholder; adjust based on actual preprocess_data output
        
        return df_final, preprocessing_stats, outlier_masks
        
    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {str(e)}")
        raise



def get_numeric_X_y(df: pd.DataFrame, feature_list, target_column):
    """
    Utility to extract numeric features and target for modeling.
    """
    X = df[feature_list].select_dtypes(include=['number']).copy()
    y = df[target_column].copy()
    return X, y

def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
    """Generate rolling sequences using positional indexing on numpy arrays."""
    # convert DataFrame or array-like to numpy for positional slicing
    arr = data.values if hasattr(data, 'values') else np.array(data)
    X, y = [], []
    for i in range(len(arr) - sequence_length):
        X.append(arr[i : i + sequence_length])
        y.append(arr[i + sequence_length])
    # return arrays of shape (num_samples, seq_len, features)
    return np.stack(X), np.stack(y)

def build_lstm_model(input_shape, output_dim):
    model = Sequential([
        LSTM(200, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(50),
        Dense(output_dim)
    ])
    return model

def build_bidirectional_lstm_model(input_shape, output_dim):
    model = Sequential([
        Bidirectional(LSTM(200, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50)),
        Dropout(0.2),
        Dense(50),
        Dense(output_dim)
    ])
    return model

def build_gru_model(input_shape, output_dim):
    model = Sequential([
        GRU(200, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(100, return_sequences=True),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(50),
        Dense(output_dim)
    ])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler_y, epochs=100, batch_size=32):
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions (only for the target variables)
    if y_train.shape[1] > 1:  # If we have multiple targets
        train_predict_price = scaler_y.inverse_transform(train_predict)
        test_predict_price = scaler_y.inverse_transform(test_predict)
        y_train_actual = scaler_y.inverse_transform(y_train)
        y_test_actual = scaler_y.inverse_transform(y_test)
    else:  # If we have a single target
        train_predict_price = scaler_y.inverse_transform(train_predict.reshape(-1, 1))
        test_predict_price = scaler_y.inverse_transform(test_predict.reshape(-1, 1))
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    metrics = {}
    for i in range(y_train_actual.shape[1]):
        metrics[f'target_{i}'] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_actual[:, i], train_predict_price[:, i])),
            'test_rmse': np.sqrt(mean_squared_error(y_test_actual[:, i], test_predict_price[:, i])),
            'train_r2': r2_score(y_train_actual[:, i], train_predict_price[:, i]),
            'test_r2': r2_score(y_test_actual[:, i], test_predict_price[:, i]),
            'train_mape': mean_absolute_percentage_error(y_train_actual[:, i], train_predict_price[:, i]) * 100,
            'test_mape': mean_absolute_percentage_error(y_test_actual[:, i], test_predict_price[:, i]) * 100
        }
    
    return metrics, history, test_predict_price, y_test_actual

class RiskManager:
    """Manages trading risk through position sizing, portfolio constraints, and correlation analysis."""
    
    def __init__(self,
                 max_position_size: float = 0.1,  # Maximum 10% of capital per position
                 max_portfolio_risk: float = 0.2,  # Maximum 20% portfolio risk
                 max_correlation_threshold: float = 0.7,  # Maximum 0.7 correlation threshold
                 risk_free_rate: float = 0.02,  # 2% risk-free rate
                 target_sharpe: float = 1.5,  # Target Sharpe ratio
                 max_drawdown_limit: float = 0.25):  # Maximum 25% drawdown
        
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation_threshold = max_correlation_threshold
        self.risk_free_rate = risk_free_rate
        self.target_sharpe = target_sharpe
        self.max_drawdown_limit = max_drawdown_limit
        self.position_history = []
        self.risk_metrics = {}
        
    def calculate_position_size(self,
                              capital: float,
                              volatility: float,
                              regime: str,
                              current_positions: dict = None) -> float:
        """Calculate optimal position size based on risk factors."""
        
        # Base position size using volatility targeting
        target_volatility = self.max_portfolio_risk / np.sqrt(252)  # Daily target vol
        position_volatility = volatility / np.sqrt(252)  # Daily asset vol
        
        # Kelly Criterion component
        win_rate = self._get_win_rate_from_history()
        avg_win = self._get_avg_win_from_history()
        avg_loss = self._get_avg_loss_from_history()
        
        if win_rate and avg_win and avg_loss:
            kelly_fraction = (win_rate / avg_loss - (1 - win_rate) / avg_win)
            # Use half-Kelly for conservatism
            kelly_size = kelly_fraction * 0.5
        else:
            kelly_size = self.max_position_size
        
        # Adjust for regime
        regime_factor = self._get_regime_factor(regime)
        
        # Calculate base position size
        base_size = min(
            target_volatility / position_volatility,
            kelly_size,
            self.max_position_size
        ) * regime_factor
        
        # Adjust for existing portfolio exposure
        if current_positions:
            total_exposure = sum(current_positions.values())
            remaining_capacity = 1 - total_exposure
            base_size = min(base_size, remaining_capacity)
        
        # Convert to actual position size
        position_size = capital * base_size
        
        return position_size
    
    def _get_regime_factor(self, regime: str) -> float:
        """Get position size adjustment factor based on market regime."""
        regime_factors = {
            'high_volatility': 0.5,    # Reduce size in high volatility
            'trending': 1.0,           # Full size in trending markets
            'ranging': 0.7,            # Reduce size in ranging markets
            'mixed': 0.8               # Slightly reduced size in mixed markets
        }
        return regime_factors.get(regime, 0.5)  # Default to conservative factor
    
    def _get_win_rate_from_history(self) -> float:
        """Calculate win rate from position history."""
        if not self.position_history:
            return None
        
        wins = sum(1 for pos in self.position_history if pos['return'] > 0)
        return wins / len(self.position_history)
    
    def _get_avg_win_from_history(self) -> float:
        """Calculate average winning trade from history."""
        winning_trades = [pos['return'] for pos in self.position_history if pos['return'] > 0]
        return np.mean(winning_trades) if winning_trades else None
    
    def _get_avg_loss_from_history(self) -> float:
        """Calculate average losing trade from history."""
        losing_trades = [abs(pos['return']) for pos in self.position_history if pos['return'] < 0]
        return np.mean(losing_trades) if losing_trades else None
    
    def calculate_portfolio_risk(self,
                               positions: dict,
                               correlations: pd.DataFrame,
                               volatilities: dict) -> float:
        """Calculate total portfolio risk considering correlations."""
        
        # Convert positions and volatilities to arrays
        assets = list(positions.keys())
        position_array = np.array([positions[asset] for asset in assets])
        volatility_array = np.array([volatilities[asset] for asset in assets])
        
        # Extract correlation matrix for relevant assets
        correlation_matrix = correlations.loc[assets, assets].values
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(
            np.dot(position_array * volatility_array,
                  correlation_matrix),
            position_array * volatility_array
        )
        
        return np.sqrt(portfolio_variance)
    
    def check_correlation_limits(self,
                               new_asset: str,
                               current_positions: list,
                               correlation_matrix: pd.DataFrame) -> bool:
        """Check if adding new asset violates correlation limits."""
        
        if not current_positions:
            return True
        
        # Check correlations with existing positions
        for asset in current_positions:
            correlation = abs(correlation_matrix.loc[new_asset, asset])
            if correlation > self.max_correlation_threshold:
                return False
        
        return True
    
    def calculate_risk_metrics(self,
                             returns: pd.Series,
                             capital: float) -> dict:
        """Calculate comprehensive risk metrics."""
        
        # Basic return metrics
        total_return = (returns + 1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate/252  # Daily risk-free rate
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        sortino = np.sqrt(252) * excess_returns.mean() / returns[returns < 0].std()
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Risk metrics relative to limits
        drawdown_utilization = abs(max_drawdown) / self.max_drawdown_limit
        sharpe_ratio_vs_target = sharpe / self.target_sharpe
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'drawdown_utilization': drawdown_utilization,
            'sharpe_ratio_vs_target': sharpe_ratio_vs_target
        }
        
        self.risk_metrics = metrics
        return metrics
    
    def adjust_for_drawdown(self,
                           position_size: float,
                           current_drawdown: float) -> float:
        """Adjust position size based on drawdown."""
        
        # Define drawdown thresholds and corresponding size multipliers
        drawdown_adjustments = [
            (-0.05, 1.0),    # Up to 5% drawdown: no adjustment
            (-0.10, 0.8),    # 5-10% drawdown: reduce to 80%
            (-0.15, 0.6),    # 10-15% drawdown: reduce to 60%
            (-0.20, 0.4),    # 15-20% drawdown: reduce to 40%
            (-0.25, 0.2),    # 20-25% drawdown: reduce to 20%
            (-float('inf'), 0.0)  # Beyond 25% drawdown: no new positions
        ]
        
        # Find appropriate multiplier
        for threshold, multiplier in drawdown_adjustments:
            if current_drawdown >= threshold:
                return position_size * multiplier
        
        return position_size
    
    def update_position_history(self, trade_result: dict):
        """Update position history with new trade result."""
        self.position_history.append(trade_result)
        
        # Keep only recent history (last 100 trades)
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]
    
    def get_risk_report(self) -> dict:
        """Generate comprehensive risk report."""
        
        report = {
            'current_metrics': self.risk_metrics,
            'position_metrics': {
                'win_rate': self._get_win_rate_from_history(),
                'avg_win': self._get_avg_win_from_history(),
                'avg_loss': self._get_avg_loss_from_history(),
                'profit_factor': self._calculate_profit_factor()
            },
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_correlation': self.max_correlation_threshold,
                'max_drawdown': self.max_drawdown_limit
            },
            'utilization': {
                'position_size': self._calculate_position_utilization(),
                'risk_budget': self._calculate_risk_budget_utilization(),
                'drawdown': self._calculate_drawdown_utilization()
            }
        }
        
        return report
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor from trade history."""
        if not self.position_history:
            return None
        
        gross_profits = sum(pos['return'] for pos in self.position_history if pos['return'] > 0)
        gross_losses = sum(abs(pos['return']) for pos in self.position_history if pos['return'] < 0)
        
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    def _calculate_position_utilization(self) -> float:
        """Calculate current position size utilization."""
        if not self.position_history:
            return 0.0
        
        recent_position_sizes = [abs(pos['size']) for pos in self.position_history[-10:]]
        avg_position_size = np.mean(recent_position_sizes)
        
        return avg_position_size / self.max_position_size
    
    def _calculate_risk_budget_utilization(self) -> float:
        """Calculate risk budget utilization."""
        if not self.risk_metrics:
            return 0.0
        
        current_volatility = self.risk_metrics.get('volatility', 0)
        return current_volatility / self.max_portfolio_risk
    
    def _calculate_drawdown_utilization(self) -> float:
        """Calculate drawdown limit utilization."""
        if not self.risk_metrics:
            return 0.0
        
        current_drawdown = abs(self.risk_metrics.get('max_drawdown', 0))
        return current_drawdown / self.max_drawdown_limit

class TechnicalIndicatorManager:
    """Centralized manager for technical indicators with efficient caching."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the technical indicator manager.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        self._cache = {}
        self._volatility_cache = {}
        self._momentum_cache = {}
        self._rsi_cache = {}
        self._trend_cache = {}
        
        # Default parameters
        self.default_windows = {
            'volatility': [20, 50, 100],
            'momentum': [5, 10, 20, 50],
            'rsi': [14, 21, 50],
            'trend': [20, 50, 100]
        }
        
    def get_volatility(self, window: int = 20, annualized: bool = True) -> pd.Series:
        """Get volatility with caching."""
        cache_key = f'vol_{window}_{annualized}'
        if cache_key not in self._volatility_cache:
            returns = self.data['Close'].pct_change()
            vol = returns.rolling(window).std()
            if annualized:
                vol = vol * np.sqrt(252)
            self._volatility_cache[cache_key] = vol
        return self._volatility_cache[cache_key]
    
    def get_momentum(self, window: int = 20, volume_adjusted: bool = False) -> pd.Series:
        """Get momentum with caching."""
        cache_key = f'mom_{window}_{volume_adjusted}'
        if cache_key not in self._momentum_cache:
            momentum = self.data['Close'].pct_change(window)
            if volume_adjusted:
                volume_sma = self.data['Volume'].rolling(window).mean()
                momentum = momentum * (self.data['Volume'] / volume_sma)
            self._momentum_cache[cache_key] = momentum
        return self._momentum_cache[cache_key]
    
    def get_rsi(self, window: int = 14) -> pd.Series:
        """Get RSI with caching."""
        cache_key = f'rsi_{window}'
        if cache_key not in self._rsi_cache:
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            self._rsi_cache[cache_key] = rsi
        return self._rsi_cache[cache_key]
    
    def get_trend(self, window: int = 50) -> pd.Series:
        """Get trend indicator with caching."""
        cache_key = f'trend_{window}'
        if cache_key not in self._trend_cache:
            ma = self.data['Close'].rolling(window=window).mean()
            trend = ma.pct_change(20)  # 20-day trend of moving average
            self._trend_cache[cache_key] = trend
        return self._trend_cache[cache_key]
    
    def get_regime_features(self, windows: Dict[str, List[int]] = None) -> pd.DataFrame:
        """Get comprehensive regime features."""
        if windows is None:
            windows = self.default_windows
        
        features = pd.DataFrame(index=self.data.index)
        
        # Add volatility features
        for window in windows['volatility']:
            features[f'volatility_{window}'] = self.get_volatility(window)
        
        # Add momentum features
        for window in windows['momentum']:
            features[f'momentum_{window}'] = self.get_momentum(window)
            features[f'vol_adj_momentum_{window}'] = self.get_momentum(window, volume_adjusted=True)
        
        # Add RSI features
        for window in windows['rsi']:
            features[f'rsi_{window}'] = self.get_rsi(window)
        
        # Add trend features
        for window in windows['trend']:
            features[f'trend_{window}'] = self.get_trend(window)
        
        return features
    
    def get_market_state(self) -> Dict[str, float]:
        """Get current market state metrics."""
        current_volatility = self.get_volatility().iloc[-1]
        current_momentum = self.get_momentum().iloc[-1]
        current_rsi = self.get_rsi().iloc[-1]
        current_trend = self.get_trend().iloc[-1]
        
        return {
            'volatility': current_volatility,
            'momentum': current_momentum,
            'rsi': current_rsi,
            'trend': current_trend
        }
    
    def get_adaptive_parameters(self) -> Dict[str, float]:
        """Get current adaptive parameters."""
        def safe_last(s):
            return s.iloc[-1] if len(s) > 0 else float('nan')
        return {
            'volatility': safe_last(self.get_volatility()),
            'momentum': safe_last(self.get_momentum()),
            'rsi': safe_last(self.get_rsi()),
            'trend': safe_last(self.get_trend()),
            'smoothing': 10,        # <-- add a default value
            'decay_factor': 0.95,   # <-- add a default value
            'n_splits': 5,           # <-- add a default value
            'windows': [5, 10, 20, 50],
        }

def main():
    logging.info("Starting main() - loading and preprocessing data...")
    # Load and preprocess data
    df, preprocessing_info = load_and_preprocess_data('Bitcoin_Historical_Data_Enhanced.csv')
    logging.info("Data loaded and preprocessed.")
    
    # Initialize walk-forward optimizer
    logging.info("Walk-forward optimizer initialized.")
    wfo = WalkForwardOptimizer(df)
    splits = wfo.generate_splits()
    logging.info("Walk-forward optimizer initialized and splits generated.")
    
    # Initialize adaptive retrainer and risk manager
    logging.info("Initializing retrainer and risk manager...")
    retrainer = AdaptiveRetrainer()
    risk_manager = RiskManager(
        max_position_size=0.1,
        max_portfolio_risk=0.2,
        max_correlation_threshold=0.7,
        max_drawdown_limit=0.25
    )
    logging.info("Setup complete. Beginning walk-forward process.")
    # Store results for each split
    all_results = []
    
    logging.info("\nStarting Walk-Forward Optimization with Risk Management...")
    
    for split_idx in range(len(splits)):
        logging.info(f"\nProcessing Split {split_idx + 1}/{len(splits)}")
        
        # Get data and regime information for this split
        data_splits, regime_info = wfo.get_split_data(split_idx)
        
        # Calculate risk metrics for the period
        returns = data_splits['train']['Close'].pct_change().dropna()
        risk_metrics = risk_manager.calculate_risk_metrics(returns, capital=100000)
        
        logging.info("\nRisk Metrics for Current Period:")
        logging.info("%s", json.dumps(risk_metrics, indent=2))
        
        # Initialize or check retraining triggers
        if split_idx > 0:
            should_retrain, trigger_info = retrainer.check_retraining_triggers(
                current_metrics=all_results[-1][next(iter(all_results[-1]))]['test_metrics'],
                current_regime=regime_info['train_regimes'],
                price_data=data_splits['train']['Close'],
                date=data_splits['train'].index[-1]
            )
            
            if should_retrain:
                logging.info("\nRetraining triggered:")
                logging.info("%s", json.dumps(trigger_info, indent=2))
        
        # Proceed with model optimization
        logging.info("\nRegime Analysis:")
        logging.info("Training period:\n%s", json.dumps(regime_info['train_regimes'], indent=2))
        logging.info("Validation period:\n%s", json.dumps(regime_info['val_regimes'], indent=2))
        logging.info("Test period:\n%s", json.dumps(regime_info['test_regimes'], indent=2))
        
        # Prepare sequences for each period
        X_train, y_train = create_sequences(data_splits['train'])
        X_val, y_val = create_sequences(data_splits['validation'])
        X_test, y_test = create_sequences(data_splits['test'])
        
        # Initialize model tuner with regime awareness
        tuner = ModelTuner(X_train, y_train, X_val, y_val)
        logging.info("ModelTuner initialized. Beginning model optimization loop.")
        
        # Optimize models
        optimization_results = {}
        for model_type in ['LSTM', 'BiLSTM', 'GRU']:
            logging.info(f"\nOptimizing {model_type}...")
            result = tuner.optimize_model(
                model_type,
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_dim=y_train.shape[1]
            )
            optimization_results[model_type] = result
            logging.info(f"Completed {model_type} optimization for split {split_idx+1}/{len(splits)}.")
            logging.info(f"Finished split {split_idx+1}/{len(splits)}.")
           
        
        # Optimize thresholds and apply risk management for each model
        split_results = {}
        for model_type, result in optimization_results.items():
            logging.info(f"\nOptimizing trading thresholds for {model_type}...")
            threshold_optimizer = ThresholdOptimizer(
                result['best_model'],
                tuner.regime_analyzer
            )
            logging.info("ThresholdOptimizer initialized. Beginning threshold optimization.")
            # Optimize thresholds
            threshold_results = threshold_optimizer.optimize_thresholds(X_val, y_val)
            
            # Apply risk management to test predictions
            test_predictions = result['best_model'].predict(X_test)
            
            # Calculate position sizes and apply risk constraints
            positions = []
            for i in range(len(test_predictions)):
                position_size = risk_manager.calculate_position_size(
                    capital=100000,  # Example initial capital
                    volatility=risk_metrics['volatility'],
                    regime=regime_info['test_regimes']['primary_regime']
                )
                
                # Adjust for current drawdown
                current_drawdown = risk_metrics['max_drawdown']
                adjusted_size = risk_manager.adjust_for_drawdown(position_size, current_drawdown)
                
                positions.append(adjusted_size)
            
            # Calculate test metrics with risk-adjusted positions
            test_metrics = threshold_optimizer.calculate_trading_metrics(
                y_test,
                test_predictions,
                **threshold_results['parameters']
            )
            
            # Update risk manager with trade results
            for i, (pred, actual) in enumerate(zip(test_predictions, y_test)):
                trade_result = {
                    'size': positions[i],
                    'return': (actual - pred) / pred,
                    'timestamp': data_splits['test'].index[i]
                }
                risk_manager.update_position_history(trade_result)
            
            split_results[model_type] = {
                'model_optimization': result,
                'threshold_optimization': threshold_results,
                'test_metrics': test_metrics,
                'regime_info': regime_info,
                'risk_metrics': risk_manager.get_risk_report()
            }
        
        # After getting results, initialize or update retrainer
        for model_type, results in split_results.items():
            if split_idx == 0:
                retrainer.initialize_baseline(
                    results['test_metrics'],
                    regime_info['train_regimes'],
                    data_splits['train'].index[-1]
                )
        
        all_results.append(split_results)

    logging.info("All walk-forward splits processed. Script complete.")
    
    # Analyze walk-forward results with risk metrics
    logging.info("\nWalk-Forward Optimization Results Summary:")
    for model_type in ['LSTM', 'BiLSTM', 'GRU']:
        logging.info(f"\n{model_type} Performance Across Splits:")
        
        # Calculate average metrics
        avg_metrics = defaultdict(list)
        regime_performance = defaultdict(lambda: defaultdict(list))
        risk_metrics_history = defaultdict(list)
        
        for split_idx, split_results in enumerate(all_results):
            results = split_results[model_type]
            metrics = results['test_metrics']
            regime = results['regime_info']['test_regimes']['primary_regime']
            risk_metrics = results['risk_metrics']
            
            # Aggregate metrics
            for metric, value in metrics.items():
                avg_metrics[metric].append(value)
                regime_performance[regime][metric].append(value)
            
            # Aggregate risk metrics
            for metric, value in risk_metrics['current_metrics'].items():
                risk_metrics_history[metric].append(value)
        
        # Print overall performance
        logging.info("\nOverall Performance:")
        for metric, values in avg_metrics.items():
            logging.info(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Print regime-specific performance
        logging.info("\nRegime-Specific Performance:")
        for regime, metrics in regime_performance.items():
            logging.info(f"\n{regime}:")
            for metric, values in metrics.items():
                logging.info(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Print risk metrics summary
        logging.info("\nRisk Metrics Summary:")
        for metric, values in risk_metrics_history.items():
            logging.info(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return all_results

if __name__ == "__main__":
    main()
