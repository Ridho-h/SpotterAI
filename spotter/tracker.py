"""
spotter/tracker.py
------------------
Stateful rep-counting and form fault detection logic for SpotterAI.

The ``RepTracker`` class tracks repetition counts, stage transitions,
and real-time biomechanical form faults for three exercises:
  - **Bicep curl** — elbow angle and full range-of-motion extension
  - **Overhead press** — overhead elbow lockout and lumbar hyperextension
  - **Squat** — bilateral knee depth and knee valgus (knees caving in)
"""

import math
import time
from datetime import datetime, timezone
import numpy as np


# ---------------------------------------------------------------------------
# Landmark Map (safe fallback when mp_pose is None)
# ---------------------------------------------------------------------------

LANDMARK_MAP = {
    'NOSE': 0,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
}


# ---------------------------------------------------------------------------
# Pure geometry helper
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c) -> float:
    """
    Computes the 2-D joint angle (in degrees) formed at vertex *b* by
    the three points *a* -> *b* -> *c*.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


# ---------------------------------------------------------------------------
# RepTracker
# ---------------------------------------------------------------------------

class RepTracker:
    """
    Tracks repetition counts, stage transitions, and form faults for:
    - bicep curls
    - overhead presses
    - squats
    """

    def __init__(self):
        """Initialises all counters, stages, and telemetry attributes."""
        self.curl_counter: int = 0
        self.press_counter: int = 0
        self.squat_counter: int = 0

        self.curl_stage = None   # 'up' | 'down' | None
        self.press_stage = None
        self.squat_stage = None

        self.current_rep_start_time = None
        self.active_issues: set[str] = set()
        self.min_angle: float = float('inf')
        self.max_angle: float = float('-inf')
        self.last_feedback: str = 'Good form'
        self.completed_reps: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_coord(landmarks, mp_pose, side: str, joint: str):
        """
        Retrieves normalised [x, y] coordinates of a named landmark.

        Supports:
          - MediaPipe NormalizedLandmarkList or results.pose_landmarks
          - 1D NumPy array of shape (132,) or flat float list
          - Objects with .x and .y attributes
          - mp_pose=None fallback using hardcoded LANDMARK_MAP
        """
        side_clean = side.upper() if side else ""
        joint_clean = joint.upper() if joint else ""
        enum_key = f"{side_clean}_{joint_clean}" if side_clean else joint_clean

        if mp_pose is not None and hasattr(mp_pose, 'PoseLandmark'):
            try:
                idx = getattr(mp_pose.PoseLandmark, enum_key).value
            except AttributeError:
                idx = LANDMARK_MAP.get(enum_key, 0)
        else:
            idx = LANDMARK_MAP.get(enum_key, 0)

        # Extract landmark list if encapsulated
        if hasattr(landmarks, 'landmark'):
            landmarks = landmarks.landmark

        # Handle 1D array / flat list (len >= 132: x, y, z, v per joint)
        if isinstance(landmarks, np.ndarray):
            if landmarks.ndim == 1 and len(landmarks) >= 132:
                return [float(landmarks[idx * 4]), float(landmarks[idx * 4 + 1])]
            elif landmarks.ndim == 2 and landmarks.shape[0] > idx:
                return [float(landmarks[idx, 0]), float(landmarks[idx, 1])]
        elif isinstance(landmarks, (list, tuple)):
            if len(landmarks) >= 132 and isinstance(landmarks[0], (int, float, np.number)):
                return [float(landmarks[idx * 4]), float(landmarks[idx * 4 + 1])]
            elif len(landmarks) > idx:
                item = landmarks[idx]
                if hasattr(item, 'x') and hasattr(item, 'y'):
                    return [float(item.x), float(item.y)]
                elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                    return [float(item[0]), float(item[1])]

        try:
            item = landmarks[idx]
            if hasattr(item, 'x') and hasattr(item, 'y'):
                return [float(item.x), float(item.y)]
            return [float(item[0]), float(item[1])]
        except Exception:
            return [0.0, 0.0]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, action: str, confidence: float, threshold: float,
               landmarks=None, mp_pose=None, keypoints=None) -> dict:
        """
        Updates rep counts, stage, and form analysis for the predicted exercise.
        """
        if landmarks is None and keypoints is not None:
            landmarks = keypoints

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
        """Returns the current state dictionary."""
        return {
            'curl': self.curl_counter,
            'press': self.press_counter,
            'squat': self.squat_counter,
            'curl_stage': self.curl_stage,
            'press_stage': self.press_stage,
            'squat_stage': self.squat_stage,
            'feedback': self.last_feedback,
            'active_issues': list(self.active_issues),
        }

    def get_last_completed_rep(self) -> dict | None:
        """Returns the telemetry dictionary of the most recent completed rep."""
        return self.completed_reps[-1] if self.completed_reps else None

    def pop_completed_reps(self) -> list[dict]:
        """Pops and returns all completed reps logged so far."""
        reps = list(self.completed_reps)
        self.completed_reps = []
        return reps

    def reset(self):
        """Resets all counters, stages, and telemetry back to initial values."""
        self.__init__()

    # ------------------------------------------------------------------
    # Exercise Tracking Logic
    # ------------------------------------------------------------------

    def _record_rep(self, exercise: str, counter: int, secondary_angle: float, cue: str):
        """Helper to append a completed rep record."""
        now = datetime.now(timezone.utc).isoformat()
        duration = (time.time() - self.current_rep_start_time) if self.current_rep_start_time else 0.0
        rep_entry = {
            'exercise': exercise,
            'rep_number': counter,
            'timestamp': now,
            'duration_sec': round(duration, 2),
            'min_primary_angle': round(self.min_angle if self.min_angle != float('inf') else 0.0, 1),
            'max_primary_angle': round(self.max_angle if self.max_angle != float('-inf') else 0.0, 1),
            'secondary_angle': round(secondary_angle, 1),
            'is_correct': len(self.active_issues) == 0,
            'issues': sorted(list(self.active_issues)),
            'feedback_cue': cue,
        }
        self.completed_reps.append(rep_entry)
        self.current_rep_start_time = None

    def _update_curl(self, landmarks, mp_pose):
        """
        Bicep curl tracking:
          - Up: elbow angle < 50°
          - Down (counts +1): elbow angle > 140° after being 'up'
        Faults:
          - min_elbow_angle > 40°: 'incomplete_curl'
          - return angle < 145° before next rep: 'incomplete_extension'
        """
        shoulder = self._get_coord(landmarks, mp_pose, 'left', 'shoulder')
        elbow = self._get_coord(landmarks, mp_pose, 'left', 'elbow')
        wrist = self._get_coord(landmarks, mp_pose, 'left', 'wrist')
        hip = self._get_coord(landmarks, mp_pose, 'left', 'hip')

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        torso_angle = calculate_angle(hip, shoulder, elbow)

        if elbow_angle < 50 or (elbow_angle < 70 and self.curl_stage is None):
            if self.curl_stage != 'up':
                if self.curl_stage == 'down' and self.max_angle < 145:
                    self.active_issues.add('incomplete_extension')
                self.curl_stage = 'up'
                self.current_rep_start_time = time.time()
                self.active_issues = set()
                self.min_angle = elbow_angle
                self.max_angle = elbow_angle
            else:
                self.min_angle = min(self.min_angle, elbow_angle)
                self.max_angle = max(self.max_angle, elbow_angle)

        elif elbow_angle > 140 and self.curl_stage == 'up':
            self.curl_stage = 'down'
            self.curl_counter += 1
            self.max_angle = max(self.max_angle, elbow_angle)

            if self.min_angle > 40.0:
                self.active_issues.add('incomplete_curl')
                cue = "Curl higher for full contraction"
            elif 'incomplete_extension' in self.active_issues:
                cue = "Lower arm fully for full stretch"
            else:
                cue = "Great bicep contraction!"

            self.last_feedback = cue
            self._record_rep('curl', self.curl_counter, torso_angle, cue)

        elif self.curl_stage == 'down':
            self.max_angle = max(self.max_angle, elbow_angle)

        self.press_stage = None
        self.squat_stage = None

    def _update_press(self, landmarks, mp_pose):
        """
        Overhead press tracking:
          - Up: elbow angle > 120° and shoulder-to-elbow dist < shoulder-to-wrist dist
          - Down (counts +1): elbow angle < 60° after being 'up'
        Faults:
          - max_elbow_angle < 150°: 'incomplete_lockout'
          - hip angle < 155°: 'back_hyperextension'
        """
        shoulder = self._get_coord(landmarks, mp_pose, 'left', 'shoulder')
        elbow = self._get_coord(landmarks, mp_pose, 'left', 'elbow')
        wrist = self._get_coord(landmarks, mp_pose, 'left', 'wrist')
        hip = self._get_coord(landmarks, mp_pose, 'left', 'hip')
        knee = self._get_coord(landmarks, mp_pose, 'left', 'knee')

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        hip_angle = calculate_angle(shoulder, hip, knee)
        shoulder2elbow_dist = abs(math.dist(shoulder, elbow))
        shoulder2wrist_dist = abs(math.dist(shoulder, wrist))

        if (elbow_angle > 120) and (shoulder2elbow_dist < shoulder2wrist_dist or elbow_angle > 140):
            if self.press_stage != 'up':
                self.press_stage = 'up'
                self.current_rep_start_time = time.time()
                self.active_issues = set()
                self.min_angle = elbow_angle
                self.max_angle = elbow_angle
            else:
                self.max_angle = max(self.max_angle, elbow_angle)
                self.min_angle = min(self.min_angle, elbow_angle)

            if hip_angle < 155.0 and hip_angle > 20.0:
                self.active_issues.add('back_hyperextension')
                self.last_feedback = "Keep core tight: avoid arching back"

        elif elbow_angle < 60 and self.press_stage == 'up':
            self.press_stage = 'down'
            self.press_counter += 1

            if self.max_angle < 150.0:
                self.active_issues.add('incomplete_lockout')

            if 'back_hyperextension' in self.active_issues:
                cue = "Keep core tight: avoid arching back"
            elif 'incomplete_lockout' in self.active_issues:
                cue = "Lock out overhead: extend elbows fully"
            else:
                cue = "Solid overhead lockout!"

            self.last_feedback = cue
            self._record_rep('press', self.press_counter, hip_angle, cue)

        self.curl_stage = None
        self.squat_stage = None

    def _update_squat(self, landmarks, mp_pose):
        """
        Squat tracking:
          - Down: knee angles < 165°
          - Up (counts +1): knee angles > 165° after being 'down'
        Faults:
          - min_knee_angle > 105°: 'incomplete_depth'
          - knee_distance < 0.82 * ankle_distance: 'knees_caving_in' (knee valgus)
        """
        l_shoulder = self._get_coord(landmarks, mp_pose, 'left', 'shoulder')
        l_hip      = self._get_coord(landmarks, mp_pose, 'left', 'hip')
        l_knee     = self._get_coord(landmarks, mp_pose, 'left', 'knee')
        l_ankle    = self._get_coord(landmarks, mp_pose, 'left', 'ankle')

        r_shoulder = self._get_coord(landmarks, mp_pose, 'right', 'shoulder')
        r_hip      = self._get_coord(landmarks, mp_pose, 'right', 'hip')
        r_knee     = self._get_coord(landmarks, mp_pose, 'right', 'knee')
        r_ankle    = self._get_coord(landmarks, mp_pose, 'right', 'ankle')

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        l_hip_angle  = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle  = calculate_angle(r_shoulder, r_hip, r_knee)

        avg_knee = (l_knee_angle + r_knee_angle) / 2.0
        avg_hip = (l_hip_angle + r_hip_angle) / 2.0

        # Knee valgus ratio: distance between knees vs distance between ankles
        knee_dist = abs(r_knee[0] - l_knee[0])
        ankle_dist = max(abs(r_ankle[0] - l_ankle[0]), 1e-4)
        valgus_ratio = knee_dist / ankle_dist

        thr = 165
        all_down = (
            l_knee_angle < thr and r_knee_angle < thr
            and l_hip_angle < thr and r_hip_angle < thr
        )
        all_up = (
            l_knee_angle > thr and r_knee_angle > thr
            and l_hip_angle > thr and r_hip_angle > thr
        )

        if all_down:
            if self.squat_stage != 'down':
                self.squat_stage = 'down'
                self.current_rep_start_time = time.time()
                self.active_issues = set()
                self.min_angle = avg_knee
                self.max_angle = avg_knee
            else:
                self.min_angle = min(self.min_angle, avg_knee)
                self.max_angle = max(self.max_angle, avg_knee)

            if valgus_ratio < 0.82 and avg_knee < 140:
                self.active_issues.add('knees_caving_in')
                self.last_feedback = "Push knees outward over toes"

        elif all_up and self.squat_stage == 'down':
            self.squat_stage = 'up'
            self.squat_counter += 1

            if self.min_angle > 105.0:
                self.active_issues.add('incomplete_depth')

            if 'incomplete_depth' in self.active_issues:
                cue = "Squat deeper: target knee angle <100°"
            elif 'knees_caving_in' in self.active_issues:
                cue = "Push knees outward over toes"
            else:
                cue = "Good squat depth!"

            self.last_feedback = cue
            self._record_rep('squat', self.squat_counter, avg_hip, cue)

        self.curl_stage = None
        self.press_stage = None
