import pytest
import numpy as np
from ml.lstm_model import build_lstm_model, train_lstm_model

def test_lstm_model_creation():
    """Test that LSTM model can be built with correct input shape"""
    input_shape = (10, 1)
    model = build_lstm_model(input_shape)
    assert model.input_shape[1:] == input_shape

def test_lstm_training():
    """Test that LSTM model can train on synthetic data"""
    X = np.random.rand(100, 10, 1)
    y = np.random.rand(100, 1)
    model = build_lstm_model((10, 1))
    history = train_lstm_model(model, X, y, epochs=1, batch_size=10)
    assert 'loss' in history.history
