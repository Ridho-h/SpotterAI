"""
tests/test_model.py
-------------------
Unit tests for spotter.model – build_model() and attention_block().

Strategy
~~~~~~~~
* ``streamlit.cache_resource`` is patched to a pass-through decorator BEFORE
  the module is imported so the decorator has no caching side-effects.
* ``model.load_weights`` is patched via pytest-mock so no .h5 file is needed.
* TensorFlow is used directly because the model architecture itself is what we
  are testing; we want to exercise the real Keras layer graph.
* All tests are CPU-only and do not load any pre-trained weights.
"""

import sys
from unittest.mock import patch
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Patch streamlit.cache_resource to be a no-op decorator BEFORE the module
# under test is imported. This avoids Streamlit context requirements.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def patch_streamlit():
    """Replace st.cache_resource with an identity decorator for the test session."""
    with patch("streamlit.cache_resource", lambda f: f):
        # Force a fresh import of spotter.model with the patched decorator
        for key in list(sys.modules.keys()):
            if "spotter.model" in key or "spotter" == key:
                sys.modules.pop(key, None)
        yield


# Import AFTER the fixture has applied the patch (module-scope fixtures run
# before any test in the module).
@pytest.fixture(scope="module")
def model_module():
    with patch("streamlit.cache_resource", lambda f: f):
        import importlib
        import spotter.model as m
        importlib.reload(m)  # reload to pick up the patched decorator
        return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(model_module, **kwargs):
    """Call build_model with no weights_path (architecture-only)."""
    return model_module.build_model(**kwargs)


# ---------------------------------------------------------------------------
# attention_block
# ---------------------------------------------------------------------------

class TestAttentionBlock:
    def test_attention_block_output_shape(self, model_module):
        """
        attention_block should return a tensor with the same shape as its input.
        """
        import tensorflow as tf

        batch, time_steps, features = 2, 30, 512  # 512 = 2 * 256 Bi-LSTM units
        x = tf.random.uniform((batch, time_steps, features))
        out = model_module.attention_block(x, time_steps)
        assert out.shape == (batch, time_steps, features), (
            f"Expected shape {(batch, time_steps, features)}, got {out.shape}"
        )

    def test_attention_block_different_time_steps(self, model_module):
        """Attention block works with arbitrary (batch, T, F) shapes."""
        import tensorflow as tf

        for time_steps in [10, 20, 40]:
            x = tf.random.uniform((1, time_steps, 64))
            out = model_module.attention_block(x, time_steps)
            assert out.shape[-2] == time_steps


# ---------------------------------------------------------------------------
# build_model – architecture tests (no weights loaded)
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_model_input_shape(self, model_module):
        """Input layer should accept sequences of shape (None, 30, 132)."""
        model = _build(model_module)
        input_shape = model.input_shape  # (None, 30, 132)
        assert input_shape == (None, 30, 132), f"Unexpected input shape: {input_shape}"

    def test_model_output_shape(self, model_module):
        """Forward pass with shape (1, 30, 132) should produce output (1, 3)."""
        import numpy as np
        model = _build(model_module)
        x = np.random.rand(1, 30, 132).astype("float32")
        y = model.predict(x, verbose=0)
        assert y.shape == (1, 3), f"Expected (1, 3), got {y.shape}"

    def test_model_output_is_probability_distribution(self, model_module):
        """Softmax output: all values in [0,1] and sum ≈ 1."""
        import numpy as np
        model = _build(model_module)
        x = np.random.rand(1, 30, 132).astype("float32")
        y = model.predict(x, verbose=0)
        assert np.all(y >= 0) and np.all(y <= 1), "Values outside [0, 1]"
        assert abs(y.sum() - 1.0) < 1e-5, f"Probabilities do not sum to 1: {y.sum()}"

    def test_num_classes_configurable(self, model_module):
        """build_model(num_classes=5) should produce output of shape (1, 5)."""
        import numpy as np
        model = _build(model_module, num_classes=5)
        x = np.random.rand(1, 30, 132).astype("float32")
        y = model.predict(x, verbose=0)
        assert y.shape == (1, 5), f"Expected (1, 5), got {y.shape}"

    def test_num_classes_2(self, model_module):
        """build_model(num_classes=2) should produce output of shape (1, 2)."""
        import numpy as np
        model = _build(model_module, num_classes=2)
        x = np.random.rand(1, 30, 132).astype("float32")
        y = model.predict(x, verbose=0)
        assert y.shape == (1, 2)

    def test_model_has_lstm_layers(self, model_module):
        """The model must contain at least one Bidirectional LSTM layer."""
        from tensorflow.keras.layers import Bidirectional
        model = _build(model_module)
        bi_layers = [l for l in model.layers if isinstance(l, Bidirectional)]
        assert len(bi_layers) >= 1, "No Bidirectional layer found in model"

    def test_model_batch_inference(self, model_module):
        """Model must handle a batch larger than 1."""
        import numpy as np
        model = _build(model_module)
        x = np.random.rand(4, 30, 132).astype("float32")
        y = model.predict(x, verbose=0)
        assert y.shape == (4, 3)

    def test_load_weights_called_when_path_given(self, model_module):
        """When weights_path is provided, model.load_weights should be called once."""
        from unittest.mock import patch
        fake_path = "fake_model.h5"
        with patch("tensorflow.keras.models.Model.load_weights", return_value=None) as mock_load:
            _build(model_module, weights_path=fake_path)
            mock_load.assert_called_once_with(fake_path)

    def test_no_weights_loaded_when_path_is_none(self, model_module):
        """When weights_path=None (default), load_weights must NOT be called."""
        from unittest.mock import patch
        with patch("tensorflow.keras.models.Model.load_weights", return_value=None) as mock_load:
            _build(model_module, weights_path=None)
            mock_load.assert_not_called()
