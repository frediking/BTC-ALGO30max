import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Bidirectional, Dropout, 
    LayerNormalization, Conv1D, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam, RectifiedAdam, AdamW

def create_sequence_rnn_model(trial, input_shape, advanced_features=True):
    """Create an RNN model optimized for sequence prediction."""
    model = Sequential()
    
    # 1. Architecture Selection
    architecture = trial.suggest_categorical('architecture', [
        'stacked_lstm',
        'bidirectional_lstm',
        'lstm_cnn_hybrid',
        'attention_lstm'
    ])
    
    # 2. Base Architecture Parameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    first_layer_units = trial.suggest_int('first_layer_units', 32, 256, log=True)
    
    # 3. Advanced Features
    if advanced_features:
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.3)
        l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        use_attention = trial.suggest_categorical('use_attention', [True, False])
        use_layer_norm = trial.suggest_categorical('use_layer_norm', [True, False])
    
    # 4. Build Architecture
    if architecture == 'stacked_lstm':
        model.add(LSTM(
            first_layer_units,
            input_shape=input_shape,
            return_sequences=n_layers > 1,
            dropout=dropout_rate if advanced_features else 0.0,
            recurrent_dropout=recurrent_dropout if advanced_features else 0.0,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if advanced_features else None
        ))
        
        if use_layer_norm and advanced_features:
            model.add(LayerNormalization())
        
        for i in range(1, n_layers):
            units = trial.suggest_int(f'layer_{i}_units', 16, first_layer_units, log=True)
            model.add(LSTM(
                units,
                return_sequences=i < n_layers - 1,
                dropout=dropout_rate if advanced_features else 0.0,
                recurrent_dropout=recurrent_dropout if advanced_features else 0.0
            ))
            
            if use_layer_norm and advanced_features:
                model.add(LayerNormalization())
    
    elif architecture == 'bidirectional_lstm':
        model.add(Bidirectional(LSTM(
            first_layer_units,
            return_sequences=n_layers > 1,
            dropout=dropout_rate if advanced_features else 0.0,
            recurrent_dropout=recurrent_dropout if advanced_features else 0.0
        ), input_shape=input_shape))
        
    elif architecture == 'lstm_cnn_hybrid':
        model.add(Conv1D(
            filters=trial.suggest_int('conv_filters', 16, 128),
            kernel_size=trial.suggest_int('kernel_size', 2, 5),
            activation='relu',
            input_shape=input_shape
        ))
        model.add(LSTM(first_layer_units))
        
    elif architecture == 'attention_lstm':
        if use_attention and advanced_features:
            model.add(MultiHeadAttention(
                num_heads=trial.suggest_int('attention_heads', 2, 8),
                key_dim=trial.suggest_int('key_dim', 16, 64),
                input_shape=input_shape
            ))
            model.add(LSTM(first_layer_units))
    
    # 5. Output Layer
    model.add(Dense(1))
    
    # 6. Compile with Advanced Options
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'radam', 'adamw'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'radam':
        optimizer = RectifiedAdam(learning_rate=learning_rate)
    else:
        optimizer = AdamW(
            learning_rate=learning_rate,
            weight_decay=trial.suggest_float('weight_decay', 1e-4, 1e-2)
        )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[
            'mse',
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError()
        ]
    )
    
    return model 