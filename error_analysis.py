import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

# Sample data: actual vs predicted values
data = {'actual': [0.01, 0.5, 1.2, -0.05, 2.0], 'predicted': [0.02, 0.45, 1.1, 0.1, 1.8]}
df = pd.DataFrame(data)

# Calculate errors
df['abs_error'] = np.abs(df['actual'] - df['predicted'])
df['percentage_error'] = np.abs((df['actual'] - df['predicted']) / df['actual'].replace(0, np.finfo(float).eps)) * 100

# Identify large errors
threshold = 1.0  # Example threshold for large absolute errors
large_errors = df[df['abs_error'] > threshold]

# Print summary
print("Error Summary:")
print(f"MAE: {df['abs_error'].mean():.4f}")
print(f"MAPE: {mean_absolute_percentage_error(df['actual'], df['predicted'])*100:.2f}%")
print("\nInstances with Large Errors:")
print(large_errors)

# Suggested next steps based on analysis
if not large_errors.empty:
    print("\nRecommendations: Investigate volatility features or robust loss functions for outliers.")
else:
    print("\nErrors are within acceptable range; consider minor tuning.")