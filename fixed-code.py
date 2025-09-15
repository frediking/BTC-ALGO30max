# Here are the specific fixes to apply to the code:

# Fix 1: Correct the correlation matrix logging
# Line ~587: Replace this line:
mlflow.log_artifact(correlation_matrix.to_csv('model_correlations.csv'))

# With:
correlation_matrix.to_csv('model_correlations.csv')
mlflow.log_artifact('model_correlations.csv')

# Fix 2: Add checks for required columns in feature engineering
# Line ~120: Add this at the beginning of engineer_features():
def engineer_features(df):
    # Check for required columns
    required_columns = ['Close', 'High', 'Low', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        return df  # Return original dataframe if crucial columns are missing
    
    # Rest of function continues...

# Fix 3: Make preprocessing more robust to avoid NaN issues
# Line ~260: Before dropping NaN values, add more detailed logging:
# Count NaNs before preprocessing
nan_count_before = X_df.isna().sum().sum()
print(f"Total NaN values before preprocessing: {nan_count_before}")

# Store original index before dropping NaNs
original_index = X_df.index.copy()
X_df = X_df.dropna()
print(f"Dropped {len(original_index) - len(X_df)} rows with NaN values")

# Update y to match the filtered X_df
y = y[X_df.index]

# Fix 4: Improve consistency in sequence alignment
# Line ~551: Replace the start_idx calculation with:
# For test predictions, ensure proper alignment
test_size_seq = len(y_test_seq)
test_offset = len(y_test) - test_size_seq

# Align test predictions with correct indices
preds_test_lgbm_aligned = preds_test_lgbm[-test_size_seq:] if len(preds_test_lgbm) >= test_size_seq else preds_test_lgbm
preds_test_xgb_aligned = preds_test_xgb[-test_size_seq:] if len(preds_test_xgb) >= test_size_seq else preds_test_xgb
preds_test_catboost_aligned = preds_test_catboost[-test_size_seq:] if len(preds_test_catboost) >= test_size_seq else preds_test_catboost
preds_test_mlp_aligned = preds_test_mlp[-test_size_seq:] if len(preds_test_mlp) >= test_size_seq else preds_test_mlp
preds_test_gbr_aligned = preds_test_gbr[-test_size_seq:] if len(preds_test_gbr) >= test_size_seq else preds_test_gbr

# Fix 5: Add cross-validation for meta-model selection
# Line ~593: Before training meta-models, add:
# Add k-fold cross-validation for meta-models
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

meta_cv_scores = {}
for name, model in meta_models.items():
    cv_scores = []
    for train_idx, val_idx in kf.split(val_preds_df):
        X_train_meta, X_val_meta = val_preds_df.iloc[train_idx], val_preds_df.iloc[val_idx]
        y_train_meta, y_val_meta = y_val_seq[train_idx], y_val_seq[val_idx]
        
        model.fit(X_train_meta, y_train_meta)
        preds = model.predict(X_val_meta)
        cv_scores.append(np.sqrt(mean_squared_error(y_val_meta, preds)))
    
    meta_cv_scores[name] = np.mean(cv_scores)
    print(f"Meta-model {name} CV RMSE: {meta_cv_scores[name]:.4f}")
    mlflow.log_metric(f"meta_{name}_cv_rmse", meta_cv_scores[name])

# Fix 6: Add robust error handling
# Add try-except blocks around critical sections, for example:
# Around model training section (line ~600):
try:
    # Train all meta-models
    for name, model in meta_models.items():
        model.fit(val_preds_df, y_val_seq)
        joblib.dump(model, f'meta_model_{name}.pkl')
except Exception as e:
    print(f"Error training meta-models: {e}")
    mlflow.log_param("meta_model_error", str(e))

# Fix 7: Make better handling of MAPE calculation to avoid division by zero
# Line ~648: Replace the MAPE calculation with:
# Handle zero values in y_test for MAPE calculation - use epsilon approach
epsilon = 1e-10  # Small constant to avoid division by zero
mape = np.mean(np.abs((preds - y_test_aligned) / (np.abs(y_test_aligned) + epsilon))) * 100

# Fix 8: Add preprocessing sanity check
# Add after preprocessing and before model training:
# Check for any remaining NaN values after preprocessing
if np.isnan(X_train_scaled_rnn).any():
    print("Warning: NaN values detected in RNN preprocessed training data")
if np.isnan(X_train_scaled_tree).any():
    print("Warning: NaN values detected in tree preprocessed training data")

# Fix 9: Fix the PCA data leakage issue
# Line ~294: Move the PCA fit only on training data:
# Create separate preprocessor pipelines for each dataset
preprocessor_rnn_train = make_pipeline(
    KNNImputer(n_neighbors=5),
    QuantileTransformer(output_distribution='normal'),
    PCA(n_components=0.95, random_state=RANDOM_SEED)
)

preprocessor_rnn_val = make_pipeline(
    KNNImputer(n_neighbors=5),
    QuantileTransformer(output_distribution='normal')
)

# Fit and transform training data
X_train_scaled_rnn = preprocessor_rnn_train.fit_transform(X_train)

# Get the PCA component from the pipeline to apply to validation/test
pca_component = preprocessor_rnn_train.named_steps['pca']
preprocessor_rnn_val.steps.append(('pca', pca_component))

# Transform validation and test data using the pipeline with the fitted PCA
X_val_scaled_rnn = preprocessor_rnn_val.transform(X_val)
X_test_scaled_rnn = preprocessor_rnn_val.transform(X_test)
