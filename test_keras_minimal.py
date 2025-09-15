import numpy as np
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense

# Generate small random data
X = np.random.rand(200, 1, 16).astype(np.float32)
y = np.random.rand(200).astype(np.float32)

print("Starting minimal Keras fit...")
model = Sequential([Input(shape=(1, 16)), LSTM(8), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=3, batch_size=32, verbose=2)
print("Minimal Keras fit successful!")