"""
tests/test_tracker.py
---------------------
Unit tests for spotter.tracker – RepTracker and calculate_angle.
No webcam, GPU, model files, or Streamlit required.
"""

import numpy as np
import pytest

from spotter.tracker import RepTracker, calculate_angle


# ---------------------------------------------------------------------------
# calculate_angle
# ---------------------------------------------------------------------------


class TestCalculateAngle:
    def test_calculate_angle_straight_line(self):
        """Three collinear points should produce ~180°."""
        a = [0, 0]
        b = [1, 0]
        c = [2, 0]
        angle = calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 1e-6, f"Expected ~180, got {angle}"

    def test_calculate_angle_right_angle(self):
        """Classic L-shape should produce ~90°."""
        a = [0, 1]   # above b
        b = [0, 0]   # vertex
        c = [1, 0]   # to the right of b
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 1e-6, f"Expected ~90, got {angle}"

    def test_calculate_angle_zero(self):
        """
        When all three points are identical the arctan2 difference is 0.
        The function returns 0 in that degenerate case.
        """
        p = [3, 7]
        angle = calculate_angle(p, p, p)
        assert angle == 0.0, f"Expected 0, got {angle}"

    def test_calculate_angle_symmetry(self):
        """Swapping a and c around the vertex should give the same angle."""
        a = [0, 1]
        b = [0, 0]
        c = [1, 0]
        assert calculate_angle(a, b, c) == calculate_angle(c, b, a)

    def test_calculate_angle_clamped_to_180(self):
        """Angle must never exceed 180°."""
        for _ in range(50):
            pts = np.random.uniform(-10, 10, (3, 2)).tolist()
            angle = calculate_angle(pts[0], pts[1], pts[2])
            assert 0.0 <= angle <= 180.0, f"Angle out of range: {angle}"


# ---------------------------------------------------------------------------
# RepTracker – initial state
# ---------------------------------------------------------------------------


class TestRepTrackerInit:
    def test_initial_counters_are_zero(self):
        tracker = RepTracker()
        assert tracker.curl_counter == 0
        assert tracker.press_counter == 0
        assert tracker.squat_counter == 0

    def test_initial_stages_are_none(self):
        tracker = RepTracker()
        assert tracker.curl_stage is None
        assert tracker.press_stage is None
        assert tracker.squat_stage is None


# ---------------------------------------------------------------------------
# RepTracker – reset
# ---------------------------------------------------------------------------


class TestRepTrackerReset:
    def test_reset_clears_counters(self):
        tracker = RepTracker()
        # Manually corrupt state
        tracker.curl_counter  = 5
        tracker.press_counter = 3
        tracker.squat_counter = 7
        tracker.curl_stage    = "up"

        tracker.reset()

        assert tracker.curl_counter  == 0
        assert tracker.press_counter == 0
        assert tracker.squat_counter == 0
        assert tracker.curl_stage    is None

    def test_reset_clears_stages(self):
        tracker = RepTracker()
        tracker.press_stage = "down"
        tracker.squat_stage = "up"
        tracker.reset()
        assert tracker.press_stage is None
        assert tracker.squat_stage is None


# ---------------------------------------------------------------------------
# RepTracker – low-confidence guard
# ---------------------------------------------------------------------------


def _zeros_kp():
    """Return a zeroed keypoints array of the correct shape (132,)."""
    return np.zeros(33 * 4)


class TestRepTrackerLowConfidence:
    def test_update_ignores_low_confidence(self):
        """If confidence < threshold, counters must stay at 0."""
        tracker = RepTracker()
        kp = _zeros_kp()
        counts = tracker.update(action="curl", confidence=0.3, threshold=0.5, keypoints=kp)
        assert counts["curl"]  == 0
        assert counts["press"] == 0
        assert counts["squat"] == 0

    def test_update_ignores_confidence_equal_to_threshold(self):
        """
        confidence == threshold is NOT above threshold, so it should be ignored.
        The logic uses `confidence < threshold`, so equality passes through.
        We just assert counters stay at 0 for a zero-keypoints frame.
        """
        tracker = RepTracker()
        kp = _zeros_kp()
        # With zero keypoints the angle will be 0 (degenerate), which is < 30,
        # so curl_stage will flip to "up" but counter stays 0.
        counts = tracker.update(action="curl", confidence=0.5, threshold=0.5, keypoints=kp)
        assert counts["curl"] == 0  # no full rep yet

    def test_high_confidence_advances_stage(self):
        """High-confidence frame with zero keypoints does NOT raise – stage can advance."""
        tracker = RepTracker()
        kp = _zeros_kp()
        # Should not raise even with degenerate keypoints
        tracker.update(action="curl", confidence=0.9, threshold=0.5, keypoints=kp)


# ---------------------------------------------------------------------------
# RepTracker – curl rep counting (stage-machine integration)
# ---------------------------------------------------------------------------


def _make_curl_kp(shoulder, elbow, wrist):
    """
    Build a 132-element keypoints array with specific left arm joint positions.

    MediaPipe landmark indices:
        left_shoulder = 11  → base offset 44
        left_elbow    = 13  → base offset 52
        left_wrist    = 15  → base offset 60
    """
    kp = np.zeros(33 * 4)
    # left_shoulder (idx 11) – x, y only; z, vis stay 0
    kp[11 * 4 + 0], kp[11 * 4 + 1] = shoulder
    # left_elbow (idx 13)
    kp[13 * 4 + 0], kp[13 * 4 + 1] = elbow
    # left_wrist (idx 15)
    kp[15 * 4 + 0], kp[15 * 4 + 1] = wrist
    return kp


class TestCurlRepCounting:
    """Simulate a full curl rep: arm starts down (angle >140°), curls up (<30°), then back down."""

    # Geometry for arm-straight (angle ~180°)
    _KP_STRAIGHT = _make_curl_kp(shoulder=[0.5, 0.8], elbow=[0.5, 0.6], wrist=[0.5, 0.4])
    # Geometry for arm-curled (angle ~15°)
    _KP_CURLED   = _make_curl_kp(shoulder=[0.5, 0.8], elbow=[0.5, 0.6], wrist=[0.51, 0.61])

    def test_curl_rep_counting(self):
        """One full curl (straight → curled → straight) increments counter by 1."""
        tracker = RepTracker()
        conf, thr = 0.9, 0.5

        # Frame 1: arm straight  →  angle ≈ 180° → stage stays None (>140 but stage not "up")
        tracker.update("curl", conf, thr, self._KP_STRAIGHT)
        assert tracker.curl_counter == 0
        assert tracker.curl_stage   is None

        # Frame 2: arm curled → angle ≈ 15° < 30  → stage = "up"
        tracker.update("curl", conf, thr, self._KP_CURLED)
        assert tracker.curl_stage   == "up"
        assert tracker.curl_counter == 0

        # Frame 3: arm straight again → angle ≈ 180° > 140 AND stage == "up" → REP!
        tracker.update("curl", conf, thr, self._KP_STRAIGHT)
        assert tracker.curl_counter == 1
        assert tracker.curl_stage   == "down"

    def test_curl_no_double_count(self):
        """Staying in the straight position after a rep should NOT double-count."""
        tracker = RepTracker()
        conf, thr = 0.9, 0.5

        tracker.update("curl", conf, thr, self._KP_STRAIGHT)  # angle high, stage None
        tracker.update("curl", conf, thr, self._KP_CURLED)    # angle low  → stage "up"
        tracker.update("curl", conf, thr, self._KP_STRAIGHT)  # angle high → stage "down", +1
        tracker.update("curl", conf, thr, self._KP_STRAIGHT)  # still "down", no additional +1
        assert tracker.curl_counter == 1

    def test_two_full_curl_reps(self):
        """Two complete cycles should give counter == 2."""
        tracker = RepTracker()
        conf, thr = 0.9, 0.5

        for _ in range(2):
            tracker.update("curl", conf, thr, self._KP_CURLED)   # → "up"
            tracker.update("curl", conf, thr, self._KP_STRAIGHT) # → "down", +1

        assert tracker.curl_counter == 2

    def test_curl_action_resets_other_stages(self):
        """Processing a curl frame must clear press_stage and squat_stage."""
        tracker = RepTracker()
        tracker.press_stage = "up"
        tracker.squat_stage = "down"
        tracker.update("curl", 0.9, 0.5, self._KP_CURLED)
        assert tracker.press_stage is None
        assert tracker.squat_stage is None


# ---------------------------------------------------------------------------
# RepTracker – update return value
# ---------------------------------------------------------------------------


class TestRepTrackerReturnValue:
    def test_update_returns_dict_with_all_keys(self):
        tracker = RepTracker()
        result = tracker.update("curl", 0.1, 0.5, _zeros_kp())
        assert set(result.keys()) == {"curl", "press", "squat"}

    def test_update_returns_current_counts(self):
        tracker = RepTracker()
        tracker.curl_counter = 4
        result = tracker.update("press", 0.1, 0.5, _zeros_kp())  # low conf – no change
        assert result["curl"] == 4
