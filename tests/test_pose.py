"""
tests/test_pose.py
------------------
Unit tests for spotter.pose.extract_keypoints().

No real camera, GPU, or MediaPipe installation required.
All MediaPipe objects are replaced with MagicMock instances.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from spotter.pose import extract_keypoints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LANDMARKS = 33
EXPECTED_SIZE  = NUM_LANDMARKS * 4  # 132


def _make_landmark(x, y, z, visibility):
    """Create a MagicMock that mimics a MediaPipe NormalizedLandmark."""
    lm = MagicMock()
    lm.x          = x
    lm.y          = y
    lm.z          = z
    lm.visibility = visibility
    return lm


def _make_results(landmarks=None):
    """
    Build a MagicMock that mimics a MediaPipe Pose result.

    Args:
        landmarks: list of 33 landmark mocks, or None to simulate no detection.
    """
    results = MagicMock()
    if landmarks is None:
        results.pose_landmarks = None
    else:
        results.pose_landmarks          = MagicMock()
        results.pose_landmarks.landmark = landmarks
    return results


def _make_33_landmarks(seed=42):
    """Return 33 deterministic fake landmarks with distinct x/y/z/visibility values."""
    rng = np.random.default_rng(seed)
    values = rng.random((NUM_LANDMARKS, 4)).tolist()
    return [_make_landmark(*v) for v in values], np.array(values)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractKeypoints:
    # ------------------------------------------------------------------
    # Shape tests
    # ------------------------------------------------------------------

    def test_extract_keypoints_returns_correct_shape(self):
        """With 33 landmarks the output must be a flat array of shape (132,)."""
        landmarks, _ = _make_33_landmarks()
        results = _make_results(landmarks)
        kp = extract_keypoints(results)
        assert kp.shape == (EXPECTED_SIZE,), f"Expected shape (132,), got {kp.shape}"

    def test_extract_keypoints_no_detection_returns_zeros(self):
        """If pose_landmarks is None, extract_keypoints must return zeros of shape (132,)."""
        results = _make_results(landmarks=None)
        kp = extract_keypoints(results)
        assert kp.shape == (EXPECTED_SIZE,), f"Expected shape (132,), got {kp.shape}"
        assert np.all(kp == 0), "Expected all zeros for missing detection"

    def test_extract_keypoints_zero_landmarks_returns_correct_size(self):
        """Zero landmarks should return zeros of the standard shape (132,)."""
        results = _make_results(landmarks=None)
        kp = extract_keypoints(results)
        assert len(kp) == EXPECTED_SIZE

    # ------------------------------------------------------------------
    # Value correctness
    # ------------------------------------------------------------------

    def test_keypoint_values_match_landmarks(self):
        """
        The extracted values must exactly match the mocked landmark attributes
        in the order: [x, y, z, visibility] per landmark, flattened.
        """
        landmarks, expected = _make_33_landmarks(seed=7)
        results = _make_results(landmarks)
        kp = extract_keypoints(results)

        expected_flat = expected.flatten()
        np.testing.assert_allclose(
            kp, expected_flat, rtol=1e-6,
            err_msg="Extracted keypoints do not match landmark values"
        )

    def test_first_landmark_values_in_correct_slots(self):
        """Spot-check: the very first landmark's x,y,z,vis are in slots [0:4]."""
        landmarks, _ = _make_33_landmarks(seed=0)
        results = _make_results(landmarks)
        kp = extract_keypoints(results)

        lm0 = landmarks[0]
        assert kp[0] == pytest.approx(lm0.x)
        assert kp[1] == pytest.approx(lm0.y)
        assert kp[2] == pytest.approx(lm0.z)
        assert kp[3] == pytest.approx(lm0.visibility)

    def test_last_landmark_values_in_correct_slots(self):
        """Spot-check: the 33rd landmark's x,y,z,vis are in the last four slots."""
        landmarks, _ = _make_33_landmarks(seed=1)
        results = _make_results(landmarks)
        kp = extract_keypoints(results)

        lm_last = landmarks[-1]
        assert kp[-4] == pytest.approx(lm_last.x)
        assert kp[-3] == pytest.approx(lm_last.y)
        assert kp[-2] == pytest.approx(lm_last.z)
        assert kp[-1] == pytest.approx(lm_last.visibility)

    def test_all_zero_landmarks(self):
        """When all landmark attributes are 0.0 the output should be all zeros."""
        landmarks = [_make_landmark(0.0, 0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS)]
        results = _make_results(landmarks)
        kp = extract_keypoints(results)
        assert np.all(kp == 0)

    def test_all_one_landmarks(self):
        """When all attribute values are 1.0 the output should be all ones."""
        landmarks = [_make_landmark(1.0, 1.0, 1.0, 1.0) for _ in range(NUM_LANDMARKS)]
        results = _make_results(landmarks)
        kp = extract_keypoints(results)
        assert np.all(kp == 1.0)

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_returns_numpy_array(self):
        """extract_keypoints must return a numpy ndarray (not a list)."""
        landmarks, _ = _make_33_landmarks()
        results = _make_results(landmarks)
        kp = extract_keypoints(results)
        assert isinstance(kp, np.ndarray), f"Expected np.ndarray, got {type(kp)}"

    def test_returns_numpy_array_on_no_detection(self):
        """extract_keypoints must return np.ndarray even when detection fails."""
        results = _make_results(landmarks=None)
        kp = extract_keypoints(results)
        assert isinstance(kp, np.ndarray)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_same_input_same_output(self):
        """Two calls with identical mocked landmarks must produce identical arrays."""
        landmarks, _ = _make_33_landmarks(seed=99)
        results = _make_results(landmarks)
        kp1 = extract_keypoints(results)
        kp2 = extract_keypoints(results)
        np.testing.assert_array_equal(kp1, kp2)
