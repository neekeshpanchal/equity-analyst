# ml/lstm_model.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_lstm_data(df: pd.DataFrame, feature_col='close', window_size=20):
    """
    Prepare sequential data for LSTM from OHLCV dataframe.
    Args:
        df: pd.DataFrame with 'close' column (or any numeric feature)
        feature_col: which column to use as the target feature
        window_size: number of past days to use for prediction
    Returns:
        X, y: numpy arrays for model training
        scaler: fitted MinMaxScaler
    """
    values = df[[feature_col]].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # [samples, timesteps, features]
    return X, y, scaler

def build_lstm_model(input_shape):
    """
    Build a simple LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(df: pd.DataFrame, feature_col='close', window_size=20, epochs=50, batch_size=32):
    """
    Train LSTM model on given dataframe.
    """
    X, y, scaler = prepare_lstm_data(df, feature_col, window_size)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
    return model, scaler

def predict_lstm(model, scaler, df, feature_col='close', look_back=20):
    """
    Predict next price using trained LSTM.
    """
    data = df[[feature_col]].values
    if len(data) < look_back:
        raise ValueError(f"Not enough data to predict: required {look_back}, got {len(data)}")

    # Scale
    scaled_data = scaler.transform(data)

    # Prepare sequence
    X_test = []
    for i in range(look_back, len(scaled_data)):
        X_test.append(scaled_data[i-look_back:i, 0])
    X_test = np.array(X_test)

    # Reshape for LSTM input: (samples, timesteps, features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Predict
    predicted_scaled = model.predict(X_test, verbose=0)

    # Reverse scaling
    predicted = scaler.inverse_transform(predicted_scaled)
    return predicted.flatten()


def save_model(model, model_name: str):
    """
    Save model to disk
    """
    path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    model.save(path)

def load_lstm_model(model_name: str):
    """
    Load model from disk
    """
    path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LSTM model {model_name} not found")
    return load_model(path)
