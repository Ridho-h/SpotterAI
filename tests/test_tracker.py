"""
tests/test_tracker.py
---------------------
Unit tests for spotter.tracker – RepTracker and calculate_angle.
"""

import numpy as np
import pytest

from spotter.tracker import RepTracker, calculate_angle


# ---------------------------------------------------------------------------
# calculate_angle
# ---------------------------------------------------------------------------

class TestCalculateAngle:
    def test_calculate_angle_straight_line(self):
        a = [0, 0]
        b = [1, 0]
        c = [2, 0]
        angle = calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 1e-6

    def test_calculate_angle_right_angle(self):
        a = [0, 1]
        b = [0, 0]
        c = [1, 0]
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 1e-6

    def test_calculate_angle_zero(self):
        p = [3, 7]
        angle = calculate_angle(p, p, p)
        assert angle == 0.0

    def test_calculate_angle_symmetry(self):
        a = [0, 1]
        b = [0, 0]
        c = [1, 0]
        assert calculate_angle(a, b, c) == calculate_angle(c, b, a)

    def test_calculate_angle_clamped_to_180(self):
        for _ in range(50):
            pts = np.random.uniform(-10, 10, (3, 2)).tolist()
            angle = calculate_angle(pts[0], pts[1], pts[2])
            assert 0.0 <= angle <= 180.0


# ---------------------------------------------------------------------------
# RepTracker – Initial state & Reset
# ---------------------------------------------------------------------------

class TestRepTrackerInitAndReset:
    def test_initial_counters_and_stages(self):
        tracker = RepTracker()
        assert tracker.curl_counter == 0
        assert tracker.press_counter == 0
        assert tracker.squat_counter == 0
        assert tracker.curl_stage is None
        assert tracker.press_stage is None
        assert tracker.squat_stage is None
        assert tracker.last_feedback == 'Good form'
        assert tracker.completed_reps == []

    def test_reset_clears_all(self):
        tracker = RepTracker()
        tracker.curl_counter = 5
        tracker.press_stage = "down"
        tracker.active_issues.add('incomplete_depth')
        tracker.completed_reps.append({'rep': 1})

        tracker.reset()

        assert tracker.curl_counter == 0
        assert tracker.press_stage is None
        assert len(tracker.active_issues) == 0
        assert len(tracker.completed_reps) == 0


# ---------------------------------------------------------------------------
# Helpers to construct 132-element arrays
# ---------------------------------------------------------------------------

def _zeros_kp():
    return np.zeros(33 * 4)

def _make_curl_kp(shoulder, elbow, wrist):
    kp = np.zeros(33 * 4)
    kp[11 * 4 + 0], kp[11 * 4 + 1] = shoulder
    kp[13 * 4 + 0], kp[13 * 4 + 1] = elbow
    kp[15 * 4 + 0], kp[15 * 4 + 1] = wrist
    return kp

def _make_squat_kp(hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r):
    kp = np.zeros(33 * 4)
    kp[11 * 4 + 0], kp[11 * 4 + 1] = [0.45, 0.2]  # l_shoulder
    kp[12 * 4 + 0], kp[12 * 4 + 1] = [0.55, 0.2]  # r_shoulder
    kp[23 * 4 + 0], kp[23 * 4 + 1] = hip_l
    kp[25 * 4 + 0], kp[25 * 4 + 1] = knee_l
    kp[27 * 4 + 0], kp[27 * 4 + 1] = ankle_l
    kp[24 * 4 + 0], kp[24 * 4 + 1] = hip_r
    kp[26 * 4 + 0], kp[26 * 4 + 1] = knee_r
    kp[28 * 4 + 0], kp[28 * 4 + 1] = ankle_r
    return kp

def _make_press_kp(shoulder, elbow, wrist, hip, knee):
    kp = np.zeros(33 * 4)
    kp[11 * 4 + 0], kp[11 * 4 + 1] = shoulder
    kp[13 * 4 + 0], kp[13 * 4 + 1] = elbow
    kp[15 * 4 + 0], kp[15 * 4 + 1] = wrist
    kp[23 * 4 + 0], kp[23 * 4 + 1] = hip
    kp[25 * 4 + 0], kp[25 * 4 + 1] = knee
    return kp


# ---------------------------------------------------------------------------
# RepTracker – Confidence & Keypoints Handling
# ---------------------------------------------------------------------------

class TestRepTrackerConfidenceAndInputs:
    def test_update_ignores_low_confidence(self):
        tracker = RepTracker()
        kp = _zeros_kp()
        counts = tracker.update(action="curl", confidence=0.3, threshold=0.5, keypoints=kp)
        assert counts["curl"] == 0
        assert counts["press"] == 0
        assert counts["squat"] == 0

    def test_keypoints_and_landmarks_compatibility(self):
        tracker = RepTracker()
        kp = _zeros_kp()
        # via keypoints argument
        tracker.update(action="curl", confidence=0.9, threshold=0.5, keypoints=kp)
        # via landmarks argument with mp_pose=None
        tracker.update(action="curl", confidence=0.9, threshold=0.5, landmarks=kp, mp_pose=None)


# ---------------------------------------------------------------------------
# RepTracker – Bicep Curl & Form Faults
# ---------------------------------------------------------------------------

class TestCurlTrackingAndForm:
    _KP_STRAIGHT = _make_curl_kp(shoulder=[0.5, 0.8], elbow=[0.5, 0.6], wrist=[0.5, 0.4])
    _KP_CURLED_GOOD = _make_curl_kp(
        shoulder=[0.5, 0.8],
        elbow=[0.5, 0.6],
        wrist=[0.5 + 0.2 * float(np.sin(np.radians(20))), 0.6 + 0.2 * float(np.cos(np.radians(20)))],
    )

    def test_curl_rep_counting_good_form(self):
        tracker = RepTracker()
        tracker.update("curl", 0.9, 0.5, self._KP_STRAIGHT)
        assert tracker.curl_counter == 0

        tracker.update("curl", 0.9, 0.5, self._KP_CURLED_GOOD)
        assert tracker.curl_stage == "up"

        tracker.update("curl", 0.9, 0.5, self._KP_STRAIGHT)
        assert tracker.curl_counter == 1
        assert tracker.curl_stage == "down"

        last_rep = tracker.get_last_completed_rep()
        assert last_rep is not None
        assert last_rep['exercise'] == 'curl'
        assert last_rep['is_correct'] is True
        assert "Great bicep contraction!" in last_rep['feedback_cue']

    def test_curl_incomplete_curl_fault(self):
        tracker = RepTracker()
        # Start curl with good position, then simulate shallow min_angle > 40
        tracker.update("curl", 0.9, 0.5, self._KP_CURLED_GOOD)
        tracker.min_angle = 48.0  # simulate shallow curl > 40
        tracker.update("curl", 0.9, 0.5, self._KP_STRAIGHT)

        last_rep = tracker.get_last_completed_rep()
        assert 'incomplete_curl' in last_rep['issues']
        assert last_rep['is_correct'] is False


# ---------------------------------------------------------------------------
# RepTracker – Squat & Form Faults
# ---------------------------------------------------------------------------

class TestSquatTrackingAndForm:
    _KP_STAND = _make_squat_kp(
        hip_l=[0.45, 0.5], knee_l=[0.45, 0.7], ankle_l=[0.45, 0.9],
        hip_r=[0.55, 0.5], knee_r=[0.55, 0.7], ankle_r=[0.55, 0.9]
    )
    _KP_SQUAT_DEEP = _make_squat_kp(
        hip_l=[0.45, 0.6], knee_l=[0.35, 0.7], ankle_l=[0.45, 0.9],
        hip_r=[0.55, 0.6], knee_r=[0.65, 0.7], ankle_r=[0.55, 0.9]
    )

    def test_squat_rep_counting_and_depth(self):
        tracker = RepTracker()
        tracker.update("squat", 0.9, 0.5, self._KP_STAND)
        assert tracker.squat_counter == 0

        tracker.update("squat", 0.9, 0.5, self._KP_SQUAT_DEEP)
        tracker.min_angle = 85.0  # below 105
        assert tracker.squat_stage == "down"

        tracker.update("squat", 0.9, 0.5, self._KP_STAND)
        assert tracker.squat_counter == 1
        assert tracker.squat_stage == "up"

        last_rep = tracker.get_last_completed_rep()
        assert last_rep['is_correct'] is True
        assert "Good squat depth!" in last_rep['feedback_cue']

    def test_squat_valgus_fault(self):
        tracker = RepTracker()
        # Knees caved in: knee distance = 0.02, ankle distance = 0.10 (< 0.82 * ankle)
        kp_valgus = _make_squat_kp(
            hip_l=[0.45, 0.6], knee_l=[0.49, 0.7], ankle_l=[0.40, 0.9],
            hip_r=[0.55, 0.6], knee_r=[0.51, 0.7], ankle_r=[0.60, 0.9]
        )
        tracker.update("squat", 0.9, 0.5, kp_valgus)
        assert 'knees_caving_in' in tracker.active_issues
        assert "Push knees outward" in tracker.last_feedback


# ---------------------------------------------------------------------------
# RepTracker – Overhead Press & Form Faults
# ---------------------------------------------------------------------------

class TestPressTrackingAndForm:
    _KP_PRESS_DOWN = _make_press_kp(
        shoulder=[0.5, 0.4], elbow=[0.5, 0.5], wrist=[0.5, 0.42],
        hip=[0.5, 0.7], knee=[0.5, 0.9]
    )
    _KP_PRESS_UP = _make_press_kp(
        shoulder=[0.5, 0.4], elbow=[0.5, 0.25], wrist=[0.5, 0.1],
        hip=[0.5, 0.7], knee=[0.5, 0.9]
    )

    def test_press_rep_counting_and_lockout(self):
        tracker = RepTracker()
        tracker.update("press", 0.9, 0.5, self._KP_PRESS_UP)
        assert tracker.press_stage == "up"

        tracker.update("press", 0.9, 0.5, self._KP_PRESS_DOWN)
        assert tracker.press_counter == 1
        assert tracker.press_stage == "down"

        last_rep = tracker.get_last_completed_rep()
        assert last_rep['is_correct'] is True
        assert "Solid overhead lockout!" in last_rep['feedback_cue']


# ---------------------------------------------------------------------------
# RepTracker – Return value & Telemetry methods
# ---------------------------------------------------------------------------

class TestRepTrackerReturnValueAndTelemetry:
    def test_state_contains_all_keys(self):
        tracker = RepTracker()
        result = tracker.update("curl", 0.1, 0.5, _zeros_kp())
        expected_keys = {"curl", "press", "squat", "curl_stage", "press_stage", "squat_stage", "feedback", "active_issues"}
        assert expected_keys.issubset(set(result.keys()))

    def test_pop_completed_reps(self):
        tracker = RepTracker()
        tracker.completed_reps = [{'rep': 1}, {'rep': 2}]
        popped = tracker.pop_completed_reps()
        assert len(popped) == 2
        assert len(tracker.completed_reps) == 0
