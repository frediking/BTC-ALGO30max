# %%


# %%
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('Bitcoin_Historical_Data1.csv')
print(df.head())
# %%
# Impute missing values using forward fill then backward fill
df = df.fillna(method='ffill').fillna(method='bfill')


# %%
df.columns 

# %%
# Display the count of NA values for each column
print("Number of NA values per column:")
print(df.isna().sum().sum())


# %%
# Get columns with NaN values and their count
nan_columns = df.columns[df.isna().any()].tolist()
nan_counts = df[nan_columns].isna().sum()

# Create a DataFrame to display results
nan_df = pd.DataFrame({
    'Column': nan_columns,
    'NaN Count': nan_counts,
    'NaN %': (nan_counts/len(df) * 100).round(2)
}).sort_values('NaN Count', ascending=False)

print("Columns containing NaN values:")
print(nan_df)



# %%
# Calculate the percentage of NA values
print("\nPercentage of NA values per column:")
print((df.isna().sum() / len(df) * 100).round(2))


# %%
# Define target variable
target = df['close']

# Define input features
non_features = ['date']
features = df.drop(columns=['close'] + non_features)
X = features
y = target

# Display features and target variable
print("Shape of X (features):", X.shape)
print("Shape of Y (target):", y.shape)

# %%
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Display the shape of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %%
# Import the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the MinMaxScaler object with the features data
scaler.fit(X_train)

# Transform the features data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the shape of the scaled data
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

# %%
# Import RandomForestRegressor

# Create and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared Score: {r2:.2f}')