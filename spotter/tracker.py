"""
spotter/tracker.py
------------------
Stateful rep-counting logic for SpotterAI.

The ``RepTracker`` class tracks repetition counts and stage transitions for
three exercises:
  - **Bicep curl** — angle at the elbow (shoulder–elbow–wrist)
  - **Overhead press** — elbow angle + relative joint distances
  - **Squat** — knee and hip angles on both sides
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Pure geometry helper
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c) -> float:
    """
    Computes the 2-D joint angle (in degrees) formed at point *b* by the
    three points *a* → *b* → *c*.

    Uses ``arctan2`` to handle the full 360° range and then folds the result
    into ``[0°, 180°]``.

    Args:
        a (array-like): ``[x, y]`` coordinates of the first point.
        b (array-like): ``[x, y]`` coordinates of the vertex (mid) point.
        c (array-like): ``[x, y]`` coordinates of the third point.

    Returns:
        float: Angle in degrees, in the range ``[0, 180]``.

    Example::

        angle = calculate_angle([0, 1], [0, 0], [1, 0])  # 90.0
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return float(angle)


# ---------------------------------------------------------------------------
# RepTracker
# ---------------------------------------------------------------------------

class RepTracker:
    """
    Tracks repetition counts and stage transitions for three exercises.

    The tracker is entirely stateful — call :meth:`update` once per frame
    (or whenever a new prediction is available) and it accumulates counts
    across the session.

    Attributes:
        curl_counter (int): Total bicep-curl repetitions counted.
        press_counter (int): Total overhead-press repetitions counted.
        squat_counter (int): Total squat repetitions counted.
        curl_stage (str | None): Current curl phase: ``'up'``, ``'down'``,
            or ``None``.
        press_stage (str | None): Current press phase: ``'up'``, ``'down'``,
            or ``None``.
        squat_stage (str | None): Current squat phase: ``'up'``, ``'down'``,
            or ``None``.
    """

    def __init__(self):
        """Initialises all counters and stage variables."""
        self.curl_counter: int = 0
        self.press_counter: int = 0
        self.squat_counter: int = 0

        self.curl_stage = None   # 'up' | 'down' | None
        self.press_stage = None
        self.squat_stage = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_coord(landmarks, mp_pose, side: str, joint: str):
        """
        Retrieves the normalised ``[x, y]`` coordinates of a named landmark.

        Args:
            landmarks: Sequence of ``NormalizedLandmark`` objects from
                MediaPipe (i.e. ``results.pose_landmarks.landmark``).
            mp_pose: The ``mediapipe.solutions.pose`` module reference.
            side (str): ``'left'`` or ``'right'``.
            joint (str): One of ``'shoulder'``, ``'elbow'``, ``'wrist'``,
                ``'hip'``, ``'knee'``, ``'ankle'``.

        Returns:
            list[float]: ``[x, y]`` in normalised image coordinates.
        """
        enum_key = f"{side.upper()}_{joint.upper()}"
        coord = getattr(mp_pose.PoseLandmark, enum_key)
        return [landmarks[coord.value].x, landmarks[coord.value].y]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, action: str, confidence: float, threshold: float,
               landmarks=None, mp_pose=None) -> dict:
        """
        Updates rep counts and stage state for the currently predicted action.

        Call this once per inference step.  When *landmarks* and *mp_pose*
        are provided the method performs angle-based counting; otherwise the
        counters remain unchanged (safe fallback).

        Args:
            action (str): Predicted exercise label — one of
                ``'curl'``, ``'press'``, ``'squat'``, or ``''`` (unknown).
            confidence (float): Model softmax probability for *action*,
                in ``[0, 1]``.
            threshold (float): Minimum confidence required to update stage /
                counter state.
            landmarks: MediaPipe ``NormalizedLandmarkList.landmark`` sequence,
                or ``None`` if no pose was detected in this frame.
            mp_pose: The ``mediapipe.solutions.pose`` module, used to look up
                ``PoseLandmark`` enum values.  Required when *landmarks* is
                not ``None``.

        Returns:
            dict: Current state with keys:
                ``curl``, ``press``, ``squat``,
                ``curl_stage``, ``press_stage``, ``squat_stage``.
        """
        if confidence < threshold or not action or landmarks is None:
            return self.state()

        if action == 'curl':
            self._update_curl(landmarks, mp_pose)

        elif action == 'press':
            self._update_press(landmarks, mp_pose)

        elif action == 'squat':
            self._update_squat(landmarks, mp_pose)

        return self.state()

    def state(self) -> dict:
        """
        Returns the current rep-count and stage state as a plain dict.

        Returns:
            dict: Keys — ``curl``, ``press``, ``squat``,
                ``curl_stage``, ``press_stage``, ``squat_stage``.
        """
        return {
            'curl': self.curl_counter,
            'press': self.press_counter,
            'squat': self.squat_counter,
            'curl_stage': self.curl_stage,
            'press_stage': self.press_stage,
            'squat_stage': self.squat_stage,
        }

    def reset(self):
        """Resets all counters and stages back to their initial values."""
        self.__init__()

    # ------------------------------------------------------------------
    # Per-exercise private update methods
    # ------------------------------------------------------------------

    def _update_curl(self, landmarks, mp_pose):
        """
        Bicep-curl rep counting.

        Stage transitions:
          - **up**: elbow angle < 30°
          - **down** (counts +1): elbow angle > 140° after being ``'up'``

        Resets press and squat stages to avoid cross-contamination.
        """
        shoulder = self._get_coord(landmarks, mp_pose, 'left', 'shoulder')
        elbow    = self._get_coord(landmarks, mp_pose, 'left', 'elbow')
        wrist    = self._get_coord(landmarks, mp_pose, 'left', 'wrist')

        angle = calculate_angle(shoulder, elbow, wrist)

        if angle < 30:
            self.curl_stage = 'up'
        if angle > 140 and self.curl_stage == 'up':
            self.curl_stage = 'down'
            self.curl_counter += 1

        # Reset other stages
        self.press_stage = None
        self.squat_stage = None

    def _update_press(self, landmarks, mp_pose):
        """
        Overhead-press rep counting.

        Stage transitions (based on elbow angle and joint distances):
          - **up**: elbow angle > 130° and shoulder-to-elbow dist < shoulder-to-wrist dist
          - **down** (counts +1): elbow angle < 50° and shoulder-to-elbow dist >
            shoulder-to-wrist dist, after being ``'up'``

        Resets curl and squat stages.
        """
        shoulder = self._get_coord(landmarks, mp_pose, 'left', 'shoulder')
        elbow    = self._get_coord(landmarks, mp_pose, 'left', 'elbow')
        wrist    = self._get_coord(landmarks, mp_pose, 'left', 'wrist')

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        shoulder2elbow_dist = abs(math.dist(shoulder, elbow))
        shoulder2wrist_dist = abs(math.dist(shoulder, wrist))

        if (elbow_angle > 130) and (shoulder2elbow_dist < shoulder2wrist_dist):
            self.press_stage = 'up'
        if (
            (elbow_angle < 50)
            and (shoulder2elbow_dist > shoulder2wrist_dist)
            and (self.press_stage == 'up')
        ):
            self.press_stage = 'down'
            self.press_counter += 1

        # Reset other stages
        self.curl_stage = None
        self.squat_stage = None

    def _update_squat(self, landmarks, mp_pose):
        """
        Squat rep counting using bilateral knee and hip angles.

        Stage transitions:
          - **down**: all four angles (left/right knee + left/right hip) < 165°
          - **up** (counts +1): all four angles > 165° after being ``'down'``

        Resets curl and press stages.
        """
        left_shoulder  = self._get_coord(landmarks, mp_pose, 'left', 'shoulder')
        left_hip       = self._get_coord(landmarks, mp_pose, 'left', 'hip')
        left_knee      = self._get_coord(landmarks, mp_pose, 'left', 'knee')
        left_ankle     = self._get_coord(landmarks, mp_pose, 'left', 'ankle')

        right_shoulder = self._get_coord(landmarks, mp_pose, 'right', 'shoulder')
        right_hip      = self._get_coord(landmarks, mp_pose, 'right', 'hip')
        right_knee     = self._get_coord(landmarks, mp_pose, 'right', 'knee')
        right_ankle    = self._get_coord(landmarks, mp_pose, 'right', 'ankle')

        left_knee_angle  = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle   = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle  = calculate_angle(right_shoulder, right_hip, right_knee)

        thr = 165
        all_down = (
            left_knee_angle  < thr and right_knee_angle < thr
            and left_hip_angle < thr and right_hip_angle < thr
        )
        all_up = (
            left_knee_angle  > thr and right_knee_angle > thr
            and left_hip_angle > thr and right_hip_angle > thr
        )

        if all_down:
            self.squat_stage = 'down'
        if all_up and self.squat_stage == 'down':
            self.squat_stage = 'up'
            self.squat_counter += 1

        # Reset other stages
        self.curl_stage  = None
        self.press_stage = None
