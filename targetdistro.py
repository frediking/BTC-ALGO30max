import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('y1_prepared.csv')
percentreturn = data['percentreturn']

# Summary statistics
print("Summary Statistics for percentreturn:")
print(percentreturn.describe())

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(percentreturn, bins=50, edgecolor='black')
plt.title('Distribution of percentreturn')
plt.xlabel('Percent Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('percentreturn_distribution.png')

# Identify near-zero values
near_zero = percentreturn[np.abs(percentreturn) < 0.1]
print(f"\nNumber of values near zero (< 0.1%): {len(near_zero)}")
print(f"Percentage of total: {len(near_zero) / len(percentreturn) * 100:.2f}%")