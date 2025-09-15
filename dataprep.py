import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
import json
from sklearn.preprocessing import (
    RobustScaler, 
    PowerTransformer,
    QuantileTransformer
)
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_union
from sklearn.compose import m  # Optional depending on setup
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Enhanced temporal feature engineering with market-specific transformations"""
    def __init__(self, time_col='Date'):
        self.time_col = time_col
        self.fitted_ = False
        
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Cyclical time features
        if self.time_col in X.columns:
            X['hour_sin'] = np.sin(2 * np.pi * X[self.time_col].dt.hour/24)
            X['hour_cos'] = np.cos(2 * np.pi * X[self.time_col].dt.hour/24)
            X['day_sin'] = np.sin(2 * np.pi * X[self.time_col].dt.dayofyear/365)
            X['day_cos'] = np.cos(2 * np.pi * X[self.time_col].dt.dayofyear/365)
        
        # Market session features
        X['is_london_session'] = ((X[self.time_col].dt.hour >= 7) & 
                                 (X[self.time_col].dt.hour < 16)).astype(int)
        X['is_newyork_session'] = ((X[self.time_col].dt.hour >= 13) & 
                                  (X[self.time_col].dt.hour < 22)).astype(int)
        
        # Advanced rolling features
        for window in [3, 7, 14, 21]:
            if 'Close' in X.columns:
                X[f'returns_{window}d'] = X['Close'].pct_change(window)
                X[f'volatility_{window}d'] = X['Close'].pct_change().rolling(window).std()
                X[f'zscore_{window}d'] = (
                    (X['Close'] - X['Close'].rolling(window).mean()) / 
                    X['Close'].rolling(window).std()
                )
        
        return X
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects specific columns from a DataFrame"""
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
    
class AdvancedAnomalyDetector(BaseEstimator, TransformerMixin):
    """Multi-modal anomaly detection with adaptive thresholds"""
    def __init__(self, sensitivity=1.5):
        self.sensitivity = sensitivity
        self.thresholds_ = {}
        
    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['float64', 'int64']):
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.thresholds_[col] = {
                'lower': q1 - self.sensitivity * iqr,
                'upper': q3 + self.sensitivity * iqr
            }
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, thresholds in self.thresholds_.items():
            if col in X.columns:
                X[col] = X[col].clip(thresholds['lower'], thresholds['upper'])
        return X

def calculate_feature_importance(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Calculate multiple feature importance metrics"""
    mi_scores = mutual_info_regression(X, y, random_state=42)
    corr_scores = X.corrwith(y).abs()
    return {
        'mutual_info': dict(zip(X.columns, mi_scores)),
        'correlation': dict(zip(X.columns, corr_scores))
    }

def optimize_feature_selection(
    X: pd.DataFrame, 
    y: pd.Series, 
    importance_scores: Dict,
    min_mi: float = 0.01,
    min_corr: float = 0.1
) -> List[str]:
    """Select features based on multiple importance metrics"""
    selected = []
    for feature in X.columns:
        mi_ok = importance_scores['mutual_info'][feature] >= min_mi
        corr_ok = abs(importance_scores['correlation'][feature]) >= min_corr
        if mi_ok and corr_ok:
            selected.append(feature)
    return selected

def create_preprocessing_pipeline(numerical_cols: List[str]) -> make_pipeline:
    """Create optimized preprocessing pipeline"""
    return make_pipeline(
        AdvancedAnomalyDetector(),
        make_union(
            make_pipeline(
                ColumnSelector(numerical_cols),
                KNNImputer(n_neighbors=5),
                PowerTransformer(method='yeo-johnson'),
                RobustScaler()
            ),
            TemporalFeatureEngineer()
        )
    )

def load_and_preprocess_data(
    file_path: str, 
    target_column: str,
    feature_config: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced data loading and preprocessing pipeline"""
    try:
        # Load data with optimized parsing
        df = pd.read_csv(
            file_path,
            parse_dates=['Date'],
            infer_datetime_format=True
        ).set_index('Date')
        
        # Initial cleaning
        df = df.dropna(how='all').drop_duplicates()
        
        # Feature engineering pipeline
        pipeline = create_preprocessing_pipeline(feature_config['numerical_features'])
        df_processed = pipeline.fit_transform(df)
        
        # Dynamic feature selection
        importance_scores = calculate_feature_importance(
            df_processed.drop(columns=[target_column]),
            df_processed[target_column]
        )
        selected_features = optimize_feature_selection(
            df_processed.drop(columns=[target_column]),
            df_processed[target_column],
            importance_scores
        )
        
        # Final output preparation
        X = df_processed[selected_features]
        y = df_processed[target_column]
        
        # Save preprocessing artifacts
        preprocessing_artifacts = {
            'selected_features': selected_features,
            'importance_scores': importance_scores,
            'pipeline_config': pipeline.get_params(),
            'data_stats': {
                'initial_shape': df.shape,
                'processed_shape': X.shape,
                'target_stats': y.describe().to_dict()
            }
        }
        
        return X, y, preprocessing_artifacts
        
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        raise

def main():
    # Configuration
    CONFIG = {
        'input_path': 'encoded_bitcoin_data.csv',
        'target_column': 'percentreturn',
        'feature_config': {
            'numerical_features': [
                'volatility_volume_ratio', 'stoch_k', 'stoch_d', 
                'rolling_vol_7', 'rolling_return_7', 'CCI', 
                'ADX_pos', 'price_range', 'volatility',
                'Williams_%R', 'RSI', 'MACD'
            ],
            'categorical_features': []
        },
        'output_files': {
            'features': 'X_optimized.csv',
            'target': 'y_optimized.csv',
            'metadata': 'preprocessing_metadata.json'
        }
    }
    
    try:
        logging.info("Starting optimized data preparation")
        Xx, yy, metadata = load_and_preprocess_data(
            CONFIG['input_path'],
            CONFIG['target_column'],
            CONFIG['feature_config']
        )
        
        # Save outputs
        Xx.to_csv(CONFIG['output_files']['features'])
        yy.to_csv(CONFIG['output_files']['target'])
        with open(CONFIG['output_files']['metadata'], 'w') as f:
            json.dump(metadata, f)
            
        logging.info(f"Data preparation complete. Files saved to: {CONFIG['output_files']}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()