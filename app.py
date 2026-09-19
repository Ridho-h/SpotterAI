"""
app.py — SpotterAI Streamlit entry point
-----------------------------------------
Real-time AI Fitness Trainer powered by:
  • MediaPipe Pose  — human keypoint estimation
  • Bi-LSTM + Attention — exercise classification
  • OpenCV + streamlit-webrtc — live video pipeline
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
import av

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from spotter.model import build_model
from spotter.pose import extract_keypoints
from spotter.tracker import RepTracker
from spotter.visualizer import draw_landmarks, draw_info_box

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='SpotterAI',
    page_icon='🏋️',
    layout='wide',
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🏋️ SpotterAI')
    st.markdown('**Your Real-Time AI Fitness Trainer**')
    st.divider()

    st.subheader('🧠 Model Info')
    st.markdown(
        '- Architecture: **Bi-LSTM + Attention**\n'
        '- Input: 30 frames × 132 keypoints\n'
        '- Classes: Curl · Press · Squat'
    )
    st.divider()

    st.subheader('⚙️ Detection Settings')
    threshold1 = st.slider(
        'Min Keypoint Detection Confidence', 0.0, 1.0, 0.5, step=0.05,
        help='Minimum confidence for MediaPipe to detect a person in the frame.',
    )
    threshold2 = st.slider(
        'Min Tracking Confidence', 0.0, 1.0, 0.5, step=0.05,
        help='Minimum confidence to continue tracking detected landmarks.',
    )
    threshold3 = st.slider(
        'Min Activity Classification Confidence', 0.0, 1.0, 0.5, step=0.05,
        help='The model only counts reps when it is at least this confident.',
    )
    st.divider()

    st.markdown(
        '[![GitHub](https://img.shields.io/badge/GitHub-SpotterAI-181717?logo=github)]'
        '(https://github.com/Ridho-h/SpotterAI)'
    )
    st.caption('Built by **Ridho-h** 🚀')

# ──────────────────────────────────────────────────────────────────────────────
# Load model (cached — only runs once per session)
# ──────────────────────────────────────────────────────────────────────────────
model = build_model()

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
# ──────────────────────────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose       = mp_pose.Pose(
    min_detection_confidence=threshold1,
    min_tracking_confidence=threshold2,
)

# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────
st.title('SpotterAI 🏋️ — Real-Time AI Fitness Trainer')
st.markdown(
    'Point your camera at yourself and start exercising. '
    'SpotterAI will automatically detect your movement and count your reps '
    'for **bicep curls**, **overhead presses**, and **squats** in real time.'
)
st.info(
    '💡 **Tip**: Allow camera access when prompted, then press **START** below.',
    icon='📷',
)

# ──────────────────────────────────────────────────────────────────────────────
# VideoProcessor
# ──────────────────────────────────────────────────────────────────────────────
ACTIONS       = np.array(['curl', 'press', 'squat'])
COLORS        = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
SEQUENCE_LEN  = 30


class VideoProcessor:
    """
    streamlit-webrtc video processor.

    Per-frame pipeline:
      1. Convert BGR → RGB and run MediaPipe Pose.
      2. Draw pose landmarks.
      3. Append keypoints to the rolling sequence buffer.
      4. Once the buffer is full (30 frames), run model inference.
      5. Update rep counts via RepTracker.
      6. Render the info-box overlay and return the annotated frame.
    """

    def __init__(self):
        self.sequence: list      = []
        self.current_action: str = ''
        self.tracker             = RepTracker()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Receive a raw webcam frame, run the full AI pipeline, and return the
        annotated frame.

        Args:
            frame (av.VideoFrame): Raw frame from the browser webcam.

        Returns:
            av.VideoFrame: Annotated BGR frame with landmarks and rep overlay.
        """
        img = frame.to_ndarray(format='bgr24')

        # ── Pose estimation ───────────────────────────────────────────
        img.flags.writeable = False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        img.flags.writeable = True
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # ── Landmark drawing ──────────────────────────────────────────
        draw_landmarks(img, results, mp_drawing, mp_pose)

        # ── Keypoint extraction & sequence buffering ───────────────────
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))
        self.sequence = self.sequence[-SEQUENCE_LEN:]

        # ── Model inference (only once the buffer is full) ─────────────
        if len(self.sequence) == SEQUENCE_LEN:
            res = model.predict(
                np.expand_dims(self.sequence, axis=0), verbose=0
            )[0]

            best_idx            = int(np.argmax(res))
            self.current_action = ACTIONS[best_idx]
            confidence          = float(np.max(res))

            # Suppress low-confidence predictions
            if confidence < threshold3:
                self.current_action = ''

            # ── Rep counting ──────────────────────────────────────────
            try:
                landmarks = results.pose_landmarks.landmark
                self.tracker.update(
                    action=self.current_action,
                    confidence=confidence,
                    threshold=threshold3,
                    landmarks=landmarks,
                    mp_pose=mp_pose,
                )
            except Exception:
                pass

            # ── Overlay ───────────────────────────────────────────────
            counters = self.tracker.state()
            img = draw_info_box(img, counters, self.current_action, res, COLORS)

        return av.VideoFrame.from_ndarray(img, format='bgr24')


# ──────────────────────────────────────────────────────────────────────────────
# WebRTC streamer
# ──────────────────────────────────────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration(
    {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]}
)

webrtc_streamer(
    key='spotter-ai',
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={'video': True, 'audio': False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)