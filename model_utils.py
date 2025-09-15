import tensorflow as tf
import gc
import psutil

def get_memory_usage():
    """Get current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Cleanup GPU and system memory."""
    tf.keras.backend.clear_session()
    gc.collect() 