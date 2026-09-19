"""
spotter/visualizer.py
---------------------
OpenCV drawing helpers for SpotterAI's live video overlay.

Two functions are provided:

- :func:`draw_landmarks` — renders MediaPipe skeleton on a BGR frame.
- :func:`draw_info_box` — overlays the rep-count banner, active exercise
  label, and a colour-coded probability bar chart.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Landmark drawing
# ---------------------------------------------------------------------------

def draw_landmarks(image, results, mp_drawing, mp_pose):
    """
    Draws MediaPipe pose landmarks and skeletal connections onto *image*.

    The function modifies *image* in-place and returns nothing.

    Args:
        image (numpy.ndarray): BGR frame to draw on (H × W × 3, ``uint8``).
        results: Pose estimation result from ``mp_pose.Pose.process()``.
            Must expose ``pose_landmarks`` (may be ``None``).
        mp_drawing: ``mediapipe.solutions.drawing_utils`` — provides the
            ``draw_landmarks`` helper.
        mp_pose: ``mediapipe.solutions.pose`` — provides ``POSE_CONNECTIONS``
            and landmark enum definitions.

    Example::

        draw_landmarks(frame, results, mp.solutions.drawing_utils,
                       mp.solutions.pose)
    """
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


# ---------------------------------------------------------------------------
# Info-box overlay
# ---------------------------------------------------------------------------

def draw_info_box(image, counters: dict, current_action: str,
                  prob: np.ndarray, colors: list) -> np.ndarray:
    """
    Draws a coloured information banner and probability bars on *image*.

    The overlay consists of:
      - A top banner (colour-coded to the highest-probability class) showing
        rep counts for all three exercises.
      - A small bar chart below the banner displaying the model's output
        probabilities for each class.

    Args:
        image (numpy.ndarray): BGR frame to annotate (H × W × 3, ``uint8``).
            This array is **not** modified in-place; a copy is operated on
            only for the probability bars, while the banner is drawn directly.
        counters (dict): Rep-count dictionary, expected keys:
            ``'curl'``, ``'press'``, ``'squat'`` (int values).
        current_action (str): The currently detected exercise label
            (``'curl'``, ``'press'``, ``'squat'``, or ``''``).
        prob (numpy.ndarray): Shape ``(num_classes,)`` — softmax output from
            the model.  Values should sum to 1.
        colors (list[tuple]): BGR colour tuples for each class in the same
            order as the model's class labels
            (e.g. ``[(245,117,16), (117,245,16), (16,117,245)]`` for
            curl / press / squat).

    Returns:
        numpy.ndarray: Annotated copy of the frame (BGR, same shape as input).

    Example::

        actions = ['curl', 'press', 'squat']
        colors  = [(245,117,16), (117,245,16), (16,117,245)]
        frame   = draw_info_box(frame, counters, 'curl', res, colors)
    """
    actions = ['curl', 'press', 'squat']
    output_frame = image.copy()

    # ── Probability bar chart (below the banner) ───────────────────────
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

    # ── Top banner ─────────────────────────────────────────────────────
    best_class_idx = int(np.argmax(prob))
    banner_color = colors[best_class_idx]

    cv2.rectangle(output_frame, (0, 0), (640, 40), banner_color, -1)

    cv2.putText(
        output_frame,
        f"curl {counters.get('curl', 0)}",
        (3, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        output_frame,
        f"press {counters.get('press', 0)}",
        (240, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        output_frame,
        f"squat {counters.get('squat', 0)}",
        (490, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return output_frame
