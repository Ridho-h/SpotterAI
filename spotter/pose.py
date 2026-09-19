"""
spotter/pose.py
---------------
MediaPipe keypoint extraction logic, extracted from app.py.
Importable without a live camera or GPU.
"""

import numpy as np


def extract_keypoints(results):
    """
    Convert a MediaPipe pose estimation result into a flat numpy array.

    Args:
        results: object with a ``pose_landmarks`` attribute.
                 Each landmark has .x, .y, .z, .visibility.

    Returns:
        np.ndarray: shape (132,) = 33 landmarks × 4 values.
                    Returns zeros if no pose was detected.
    """
    if results.pose_landmarks is None:
        return np.zeros(33 * 4)

    pose = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
    ).flatten()
    return pose
