import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

def check_temporal_dependency(X_df, y, n_splits=5):
    """Check if temporal effects are significant"""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(X_df)//10)
    base_model = LGBMRegressor(random_state=42)
    
    temporal_scores = []
    print("\nChecking temporal dependency...")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_df)):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        base_model.fit(X_train, y_train)
        score = base_model.score(X_test, y_test)
        temporal_scores.append(score)
        print(f"Fold {i+1} R² score: {score:.4f}")
    
    score_std = np.std(temporal_scores)
    print(f"\nTemporal score standard deviation: {score_std:.4f}")
    print(f"Mean R² score: {np.mean(temporal_scores):.4f}")
    
    needs_wfv = score_std > 0.1
    print(f"Need walk-forward validation: {needs_wfv}")
    return needs_wfv

if __name__ == "__main__":
    # Load only required data
    X_df = pd.read_csv('X_prepared.csv')
    y = pd.read_csv('y1_prepared.csv')['percentreturn'].values
    
    # Run check
    needs_walk_forward = check_temporal_dependency(X_df, y)