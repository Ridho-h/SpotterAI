"""
spotter/model.py
----------------
Bi-LSTM + Attention model builder, extracted from app.py.
The @st.cache_resource decorator is applied here so the production app can cache
the loaded model. In tests, patch 'streamlit.cache_resource' before importing
this module to avoid the decorator being applied.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st  # noqa: E402  (side-effect: sets TF log level first)


def attention_block(inputs, time_steps):
    """
    Attention mechanism for the Bi-LSTM model.

    Args:
        inputs:     Keras tensor of shape (batch, time_steps, features).
        time_steps: int – sequence length (must match the time dimension).

    Returns:
        Keras tensor of the same shape as *inputs*, weighted by attention.
    """
    from tensorflow.keras.layers import Dense, Permute, multiply  # local import keeps TF lazy

    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation="softmax")(a)
    a_probs = Permute((2, 1), name="attention_vec")(a)
    output_attention_mul = multiply([inputs, a_probs], name="attention_mul")
    return output_attention_mul


@st.cache_resource
def build_model(
    hidden_units: int = 256,
    sequence_length: int = 30,
    num_input_values: int = 33 * 4,
    num_classes: int = 3,
    weights_path=None,
):
    """
    Build (and optionally load weights for) the Bi-LSTM + Attention model.

    Args:
        hidden_units     : LSTM hidden size (default 256).
        sequence_length  : Number of frames per sequence (default 30).
        num_input_values : Feature size per frame (default 132 = 33 x 4).
        num_classes      : Number of output classes (default 3).
        weights_path     : Path to an .h5 file. If None, no weights are loaded.

    Returns:
        Compiled keras.Model (weights are NOT compiled / optimized here).
    """
    from tensorflow.keras.layers import (  # local import keeps TF lazy
        Bidirectional,
        Dense,
        Dropout,
        Flatten,
        Input,
        LSTM,
    )
    from tensorflow.keras.models import Model

    inputs = Input(shape=(sequence_length, num_input_values))
    lstm_out = Bidirectional(LSTM(hidden_units, return_sequences=True))(inputs)
    attention_mul = attention_block(lstm_out, sequence_length)
    attention_mul = Flatten()(attention_mul)
    x = Dense(2 * hidden_units, activation="relu")(attention_mul)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=[inputs], outputs=x)

    if weights_path is not None:
        model.load_weights(weights_path)

    return model
