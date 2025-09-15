import os
import sys
import time
import traceback
import gc
import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ── ENVIRONMENT SETUP AND DIAGNOSTICS ───────────────────────────────────────────
print("\n=== SYSTEM DIAGNOSTICS ===")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Optuna version: {optuna.__version__}")

# Set environment variables
os.environ["OMP_NUM_THREADS"] = "2"  # Increased from 1
os.environ["MKL_NUM_THREADS"] = "2"  # Increased from 1
os.environ["TF_NUM_INTEROP_THREADS"] = "2"  # Increased from 1
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"  # Increased from 1
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

# TF configuration with diagnostics
tf.config.threading.set_intra_op_parallelism_threads(2)  # Increased from 1
tf.config.threading.set_inter_op_parallelism_threads(2)  # Increased from 1
tf.config.run_functions_eagerly(True)

# Explicitly enable tf.data debug mode to address the warning
tf.data.experimental.enable_debug_mode()

# Print configuration for verification
print("\n=== CONFIGURATION ===")
print(f"TF eager mode: {tf.executing_eagerly()}")
print(f"Thread settings: OMP={os.environ['OMP_NUM_THREADS']}, "
      f"MKL={os.environ['MKL_NUM_THREADS']}, "
      f"TF inter/intra={os.environ['TF_NUM_INTEROP_THREADS']}/{os.environ['TF_NUM_INTRAOP_THREADS']}")
print("─" * 60)

# ─── DETAILED CALLBACKS FOR MONITORING ─────────────────────────────────────────
class DetailedProgressLogger(Callback):
    """Enhanced logger to track all stages of training with detailed timing"""
    
    def __init__(self, trial_num, log_interval=1):
        super().__init__()
        self.trial_num = trial_num
        self.epoch_times = []
        self.batch_times = []
        self.start_time = None
        self.epoch_start = None
        self.batch_start = None
        self.batch_count = 0
        self.log_interval = log_interval
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"[Trial {self.trial_num}] Training STARTED at {time.strftime('%H:%M:%S')}")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"[Trial {self.trial_num}] Epoch {epoch+1} STARTED at {time.strftime('%H:%M:%S')}")
        self.batch_count = 0
        
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()
        self.batch_count += 1
        if batch % self.log_interval == 0:
            print(f"[Trial {self.trial_num}] Batch {batch} started")
        
    def on_train_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start
        self.batch_times.append(batch_time)
        if batch % self.log_interval == 0:
            loss = logs.get('loss', 0)
            print(f"[Trial {self.trial_num}] Batch {batch} completed in {batch_time:.4f}s - loss: {loss:.4f}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        val_loss = logs.get('val_loss', 0)
        train_loss = logs.get('loss', 0)
        print(f"[Trial {self.trial_num}] Epoch {epoch+1} COMPLETED in {epoch_time:.2f}s")
        print(f"[Trial {self.trial_num}] Stats: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        print(f"[Trial {self.trial_num}] Processed {self.batch_count} batches, avg batch time: {np.mean(self.batch_times[-self.batch_count:]):.4f}s")
        
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"[Trial {self.trial_num}] Training COMPLETED in {total_time:.2f}s")
        if self.epoch_times:
            print(f"[Trial {self.trial_num}] Avg epoch time: {np.mean(self.epoch_times):.2f}s")
        print(f"[Trial {self.trial_num}] Memory usage: {get_memory_usage()}")

class TimeoutCallback(Callback):
    """Enhanced timeout callback with periodic status updates"""
    
    def __init__(self, timeout_seconds=60, status_interval=10):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.status_interval = status_interval
        self.start_time = None
        self.last_status = 0
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_batch_begin(self, batch, logs=None):
        elapsed = time.time() - self.start_time
        
        # Print periodic status updates
        if elapsed - self.last_status > self.status_interval:
            print(f"⏱️ Training running for {elapsed:.1f}s of {self.timeout_seconds}s timeout")
            self.last_status = elapsed
            
        # Check for timeout
        if elapsed > self.timeout_seconds:
            print(f"\n⚠️ TIMEOUT reached after {elapsed:.1f} seconds!")
            self.model.stop_training = True

# ─── MEMORY USAGE TRACKING ─────────────────────────────────────────────────────
def get_memory_usage():
    """Get current memory usage of the process"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return f"{mem_info.rss / (1024 * 1024):.1f} MB"
    except ImportError:
        return "psutil not available"

# ─── DATA LOADING WITH ROBUST ERROR HANDLING ─────────────────────────────────────
def load_data(debug_mode=True):
    """Load and prepare data with extensive validation and diagnostics"""
    print("\n=== LOADING DATA ===")
    try:
        t0 = time.time()
        
        # Check if files exist first
        for filename in ['X_prepared.csv', 'y_prepared.csv']:
            if not os.path.exists(filename):
                print(f"❌ ERROR: Data file '{filename}' not found")
                return None, None
        
        # Load data with diagnostic info
        print(f"Loading X data...")
        X = pd.read_csv('X_prepared.csv')
        print(f"X data loaded: {X.shape}, memory usage: {X.memory_usage().sum() / 1e6:.2f} MB")
        
        print(f"Loading y data...")
        y = pd.read_csv('y_prepared.csv')
        print(f"y data loaded: {y.shape}, memory usage: {y.memory_usage().sum() / 1e6:.2f} MB")
        
        # Convert to numpy
        X_np = X.values
        y_np = y.values.flatten()
        
        # Basic checks
        if debug_mode:
            print("\n--- Data Validation ---")
            print(f"X shape: {X_np.shape}, y shape: {y_np.shape}")
            print(f"X data types: {X.dtypes.value_counts().to_dict()}")
            print(f"X range: min={X_np.min()}, max={X_np.max()}")
            print(f"y range: min={y_np.min()}, max={y_np.max()}")
            print(f"X NaN count: {np.isnan(X_np).sum()}")
            print(f"y NaN count: {np.isnan(y_np).sum()}")
            
        # Check data integrity
        assert not np.isnan(X_np).any(), "NaN values found in X data"
        assert not np.isnan(y_np).any(), "NaN values found in y data"
        assert not np.isinf(X_np).any(), "Inf values found in X data"
        assert not np.isinf(y_np).any(), "Inf values found in y data"
        
        # Scale and reshape for RNN with explicit steps
        print("\nScaling and reshaping data...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_np)
        print(f"X scaled: min={X_scaled.min()}, max={X_scaled.max()}")
        
        # Reshape with careful verification
        X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        print(f"Reshaped X to: {X_rnn.shape}")
        
        print(f"Data preparation completed in {time.time() - t0:.2f}s")
        return X_rnn, y_np
        
    except Exception as e:
        print(f"❌ ERROR during data preparation: {str(e)}")
        print(traceback.format_exc())
        return None, None

# ─── MODEL CREATION WITH STEP-BY-STEP LOGGING ─────────────────────────────────
def create_rnn_model(units, learning_rate, input_shape, name="unnamed"):
    """Create an RNN model with detailed logging of each step"""
    t0 = time.time()
    print(f"\n=== BUILDING MODEL '{name}' ===")
    print(f"Parameters: units={units}, learning_rate={learning_rate:.6f}")
    print(f"Input shape: {input_shape}")
    
    try:
        # Clear any previous models
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Build model layer by layer with verification
        print("Creating input layer...")
        inputs = Input(shape=input_shape)
        
        print("Adding SimpleRNN layer...")
        rnn_layer = SimpleRNN(units)(inputs)
        
        print("Adding output Dense layer...")
        outputs = Dense(1)(rnn_layer)
        
        print("Assembling model...")
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Summary for verification
        print("Model structure:")
        model.summary(print_fn=lambda x: print(f"  {x}"))
        
        # Compile with optimizer
        print(f"Compiling model with Adam lr={learning_rate}...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        build_time = time.time() - t0
        print(f"Model built in {build_time:.2f}s")
        
        return model
    
    except Exception as e:
        print(f"❌ ERROR building model: {str(e)}")
        print(traceback.format_exc())
        return None

# ─── SIMPLIFIED DIRECT TRAINING FUNCTION ─────────────────────────────────────────
def train_simple_model(X_rnn, y):
    """Train a simple model with fixed parameters to debug the hanging issue"""
    print("\n" + "=" * 60)
    print("DIRECT MODEL TRAINING (SIMPLIFIED)")
    print("=" * 60)
    
    try:
        # Fixed parameters for debugging
        units = 12
        learning_rate = 0.001
        batch_size = 4  # Smaller batch size for debugging
        epochs = 2      # Fewer epochs for debugging
        
        # Split data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_rnn, y, test_size=0.2, shuffle=False
        )
        
        print(f"Train shape={X_tr.shape}, Val shape={X_val.shape}")
        
        # Create model
        model = create_rnn_model(
            units=units, 
            learning_rate=learning_rate,
            input_shape=X_rnn.shape[1:],
            name="debug_model"
        )
        
        if model is None:
            print("❌ Model creation failed")
            return
        
        # Test a single batch first
        print("\n=== TESTING SINGLE BATCH ===")
        single_batch_size = min(batch_size, len(X_tr))
        print(f"Taking first {single_batch_size} samples for batch test")
        
        print("Creating batch...")
        xb = X_tr[:single_batch_size]
        yb = y_tr[:single_batch_size]
        
        print(f"Single batch shapes: X={xb.shape}, y={yb.shape}")
        print("Training on single batch...")
        t0 = time.time()
        loss = model.train_on_batch(xb, yb)
        batch_time = time.time() - t0
        print(f"Single batch completed in {batch_time:.4f}s - loss={loss[0]:.4f}")
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=1, 
                restore_best_weights=True, 
                verbose=1
            ),
            DetailedProgressLogger(trial_num="debug", log_interval=1),
            TimeoutCallback(timeout_seconds=60, status_interval=5)
        ]
        
        # Train with very explicit approach - small batches and verbose output
        print("\n=== STARTING FULL TRAINING ===")
        print(f"Training with batch_size={batch_size}, epochs={epochs}")
        print("Memory usage before training:", get_memory_usage())
        
        # Use tf.data API explicitly for more control
        train_dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        train_dataset = train_dataset.batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        
        print("Dataset preparation complete")
        
        # Explicit training (not using fit() directly)
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            epoch_start = time.time()
            batch_count = 0
            
            # Manual training loop for maximum control
            for x_batch, y_batch in train_dataset:
                batch_start = time.time()
                print(f"  Batch {batch_count+1} starting...")
                # Force one batch at a time
                with tf.device("/CPU:0"):
                    loss = model.train_on_batch(x_batch, y_batch)
                batch_time = time.time() - batch_start
                print(f"  Batch {batch_count+1} completed in {batch_time:.4f}s - loss={loss[0]:.4f}")
                batch_count += 1
                # Early exit for debugging if needed
                if batch_count >= 5:  # Just process a few batches for quick debugging
                    print("  Early exit from training loop (debugging mode)")
                    break
            
            # Manual validation
            val_losses = []
            for x_val_batch, y_val_batch in val_dataset:
                val_loss = model.test_on_batch(x_val_batch, y_val_batch)
                val_losses.append(val_loss[0])
            
            avg_val_loss = np.mean(val_losses)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - val_loss: {avg_val_loss:.4f}")
        
        print("Manual training loop completed")
        print("Memory usage after training:", get_memory_usage())
        
        return model
        
    except Exception as e:
        print(f"❌ ERROR during training: {str(e)}")
        print(traceback.format_exc())
        return None

# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        print("\n🔍 STARTING DEBUG MODE EXECUTION")
        
        # Load and check data
        X_rnn, y = load_data(debug_mode=True)
        if X_rnn is None or y is None:
            print("❌ Data loading failed, exiting")
            sys.exit(1)
        
        # Train a simple model with explicit execution
        print("\nTraining a simple model for debugging...")
        model = train_simple_model(X_rnn, y)
        
        if model is not None:
            print("✅ Debug training completed successfully!")
        else:
            print("❌ Debug training failed")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR in main execution: {str(e)}")
        print(traceback.format_exc())