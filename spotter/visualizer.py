"""
spotter/visualizer.py
---------------------
OpenCV drawing helpers for SpotterAI's live video overlay with
form quality colour coding.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Landmark & Skeleton drawing
# ---------------------------------------------------------------------------

def draw_form_skeleton(image, results, mp_drawing, mp_pose, active_issues=None, is_correct=True):
    """
    Draws MediaPipe pose landmarks and skeletal connections colour-coded by form:
      - Green (46, 204, 113) when good form.
      - Red/Amber (0, 50, 240) when active issues are detected.
    """
    if results is None or not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None:
        return

    has_issues = bool(active_issues) or (not is_correct)
    line_color = (0, 50, 240) if has_issues else (46, 204, 113)
    point_color = (0, 80, 255) if has_issues else (52, 231, 128)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=point_color, thickness=3, circle_radius=3),
        mp_drawing.DrawingSpec(color=line_color, thickness=3, circle_radius=2),
    )


def draw_landmarks(image, results, mp_drawing, mp_pose, active_issues=None, is_correct=True):
    """
    Backwards-compatible wrapper calling draw_form_skeleton.
    """
    draw_form_skeleton(image, results, mp_drawing, mp_pose, active_issues=active_issues, is_correct=is_correct)


# ---------------------------------------------------------------------------
# Info-box overlay
# ---------------------------------------------------------------------------

def draw_info_box(image, counters: dict, current_action: str,
                  prob: np.ndarray, colors: list,
                  feedback: str = None, active_issues: list = None) -> np.ndarray:
    """
    Draws exercise info, probability bars, rep counts, and real-time form feedback cue banner.
    """
    actions = ['curl', 'press', 'squat']
    output_frame = image.copy()
    h, w = output_frame.shape[:2]

    # ── Probability bar chart (below the top banner) ───────────────────
    for num, p in enumerate(prob):
        cv2.rectangle(
            output_frame,
            (0, 60 + num * 40),
            (int(p * 100), 90 + num * 40),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
        )

    # ── Top banner: Rep counts ─────────────────────────────────────────
    best_class_idx = int(np.argmax(prob))
    banner_color = colors[best_class_idx]

    cv2.rectangle(output_frame, (0, 0), (w, 45), banner_color, -1)

    cv2.putText(
        output_frame,
        f"curl {counters.get('curl', 0)}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        output_frame,
        f"press {counters.get('press', 0)}",
        (int(w * 0.38), 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        output_frame,
        f"squat {counters.get('squat', 0)}",
        (int(w * 0.72), 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # ── Bottom banner: Real-time Form Feedback Cue ────────────────────
    active_cue = feedback or counters.get('feedback', '')
    issues = active_issues if active_issues is not None else counters.get('active_issues', [])

    if active_cue:
        cue_bg = (0, 50, 240) if issues else (46, 204, 113)  # Red/Amber vs Green
        banner_h = 50
        cv2.rectangle(output_frame, (0, h - banner_h), (w, h), cue_bg, -1)
        cv2.putText(
            output_frame,
            f"FORM: {active_cue}",
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return output_frame
