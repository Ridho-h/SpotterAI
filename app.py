"""
app.py — SpotterAI Streamlit entry point
-----------------------------------------
Real-time AI Fitness Coach & Form Analysis System:
  • Pose Tracking Engine: MediaPipe Pose + Bi-LSTM + Biomechanical Form Fault Detection
  • Session Log: SQLite structured persistence of every repetition & joint angle
  • Coaching Agent: Post-session debriefs with Form Quality Score & actionable cues
  • Chat Assistant: Natural language Q&A over workout history & trends
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import av
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from spotter.model import build_model
from spotter.pose import extract_keypoints
from spotter.tracker import RepTracker
from spotter.visualizer import draw_form_skeleton, draw_info_box
from spotter.database import WorkoutDatabase
from spotter.coach import CoachingAgent
from spotter.chat import WorkoutChatAssistant

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='SpotterAI — AI Fitness Coach',
    page_icon='🏋️‍♂️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ──────────────────────────────────────────────────────────────────────────────
# Database & Services Initialization
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_database():
    db = WorkoutDatabase()
    db.seed_demo_data_if_empty()
    return db

db = get_database()

# Session State
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'is_session_active' not in st.session_state:
    st.session_state.is_session_active = False
if 'last_finished_session_id' not in st.session_state:
    recent = db.get_recent_sessions(limit=1)
    st.session_state.last_finished_session_id = recent[0]['session_id'] if recent else None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi, I'm your **SpotterAI Workout Assistant**! Ask me anything about your workout history, like:\n- *'How has my squat depth changed this week?'*\n- *'Which exercise has the most form issues?'*\n- *'Show my stats for today'*"}
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🏋️‍♂️ SpotterAI')
    st.caption('**Real-Time Biomechanical Coach & Analytics**')
    st.divider()

    st.subheader('🤖 AI Coach Settings')
    api_key = st.text_input(
        'Gemini / OpenAI API Key (Optional)',
        type='password',
        help='Optional: Powers natural conversational tone for Coach & Chat. Without an API key, the system uses grounded deterministic rule-based analysis.'
    )
    if api_key:
        os.environ['GEMINI_API_KEY'] = api_key
        st.success('API Key loaded for enhanced dialogue!', icon='✨')

    coach = CoachingAgent(api_key=api_key if api_key else None)
    chat_assistant = WorkoutChatAssistant(db=db, api_key=api_key if api_key else None)

    st.divider()
    st.subheader('⚙️ Vision Detection Thresholds')
    threshold1 = st.slider('Min Keypoint Detection Confidence', 0.0, 1.0, 0.5, step=0.05)
    threshold2 = st.slider('Min Tracking Confidence', 0.0, 1.0, 0.5, step=0.05)
    threshold3 = st.slider('Min Activity Confidence', 0.0, 1.0, 0.5, step=0.05)

    st.divider()
    st.markdown(
        '[![GitHub](https://img.shields.io/badge/GitHub-SpotterAI-181717?logo=github)]'
        '(https://github.com/Ridho-h/SpotterAI)'
    )
    st.caption('Built by **Ridho-h** 🚀')

# ──────────────────────────────────────────────────────────────────────────────
# Global ML & Vision Models (Cached)
# ──────────────────────────────────────────────────────────────────────────────
model = build_model()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=threshold1, min_tracking_confidence=threshold2)

ACTIONS = np.array(['curl', 'press', 'squat'])
COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
SEQUENCE_LEN = 30

# Shared tracker instance across Streamlit rerenders
if 'tracker' not in st.session_state:
    st.session_state.tracker = RepTracker()
tracker: RepTracker = st.session_state.tracker

# ──────────────────────────────────────────────────────────────────────────────
# VideoProcessor for WebRTC
# ──────────────────────────────────────────────────────────────────────────────
class VideoProcessor:
    def __init__(self):
        self.sequence: list = []
        self.current_action: str = ''
        self.tracker = tracker

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format='bgr24')

        # Pose estimation
        img.flags.writeable = False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        img.flags.writeable = True
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Keypoint extraction & sequence buffer
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints.astype('float32', casting='same_kind'))
        self.sequence = self.sequence[-SEQUENCE_LEN:]

        res = np.zeros(len(ACTIONS), dtype='float32')
        # Model inference
        if len(self.sequence) == SEQUENCE_LEN:
            res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            best_idx = int(np.argmax(res))
            confidence = float(np.max(res))

            if confidence >= threshold3:
                self.current_action = ACTIONS[best_idx]
            else:
                self.current_action = ''

            # Update RepTracker with landmarks
            try:
                landmarks = results.pose_landmarks.landmark if (results and results.pose_landmarks) else None
                self.tracker.update(
                    action=self.current_action,
                    confidence=confidence,
                    threshold=threshold3,
                    landmarks=landmarks,
                    mp_pose=mp_pose,
                )

                # If a session is active, automatically flush completed reps to DB
                if st.session_state.get('is_session_active') and st.session_state.get('current_session_id'):
                    completed = self.tracker.pop_completed_reps()
                    for rep in completed:
                        db.log_rep(st.session_state.current_session_id, rep)
            except Exception:
                pass

        # Color-coded skeleton: green if good form, red if issue detected
        state = self.tracker.state()
        active_issues = state.get('active_issues', [])
        draw_form_skeleton(
            img, results, mp_drawing, mp_pose,
            active_issues=active_issues,
            is_correct=len(active_issues) == 0
        )

        # Render top rep banner + bottom real-time HUD cue
        img = draw_info_box(
            img, state, self.current_action, res, COLORS,
            feedback=state.get('feedback', ''),
            active_issues=active_issues
        )

        return av.VideoFrame.from_ndarray(img, format='bgr24')


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard Navigation Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_live, tab_coach, tab_chat, tab_analytics = st.tabs([
    "🎥 Live Workout & Form HUD",
    "📋 Session Coaching Review",
    "💬 Workout History Chat",
    "📊 Analytics & Trends"
])

# ==============================================================================
# TAB 1: Live Workout & Form HUD
# ==============================================================================
with tab_live:
    col_video, col_controls = st.columns([3, 1])

    with col_controls:
        st.subheader("🏋️ Workout Controls")
        if not st.session_state.is_session_active:
            if st.button("▶️ Start Workout Session", type="primary", use_container_width=True):
                session_id = db.start_session()
                st.session_state.current_session_id = session_id
                st.session_state.is_session_active = True
                tracker.reset()
                st.rerun()
        else:
            st.success(f"Session **{st.session_state.current_session_id}** Active", icon="🟢")
            if st.button("⏹️ Finish & Review Workout", type="primary", use_container_width=True):
                # Flush any remaining reps
                remaining_reps = tracker.pop_completed_reps()
                for rep in remaining_reps:
                    db.log_rep(st.session_state.current_session_id, rep)
                
                db.end_session(st.session_state.current_session_id)
                st.session_state.last_finished_session_id = st.session_state.current_session_id
                st.session_state.current_session_id = None
                st.session_state.is_session_active = False
                st.toast("Workout saved! Check the Session Coaching Review tab.", icon="✅")
                st.rerun()

        st.divider()
        st.markdown("### 📊 Live Counters")
        st_state = tracker.state()
        m1, m2, m3 = st.columns(3)
        m1.metric("Curls", st_state.get('curl', 0))
        m2.metric("Presses", st_state.get('press', 0))
        m3.metric("Squats", st_state.get('squat', 0))

        st.divider()
        st.markdown("### 🎯 Form HUD Status")
        cue = st_state.get('feedback', 'Ready')
        issues = st_state.get('active_issues', [])
        if issues:
            st.error(f"⚠️ **Issue**: {cue}")
        else:
            st.success(f"🟢 **Status**: {cue}")

        if st.button("🔄 Reset Counters", use_container_width=True):
            tracker.reset()
            st.rerun()

    with col_video:
        st.markdown("#### Real-Time Camera Stream")
        webrtc_streamer(
            key='spotter-live',
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]}),
            media_stream_constraints={'video': True, 'audio': False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

# ==============================================================================
# TAB 2: Post-Workout Coaching Review
# ==============================================================================
with tab_coach:
    target_session_id = st.session_state.last_finished_session_id
    recent_sessions = db.get_recent_sessions(limit=10)

    if not recent_sessions:
        st.info("No recorded workouts found. Start a workout session on Tab 1 or seed demo data.")
    else:
        session_options = {s['session_id']: f"{s['start_time'][:16]} — {s['total_reps']} reps ({s['accuracy_pct']}% form)" for s in recent_sessions}
        selected_id = st.selectbox(
            "Select Workout Session to Review:",
            options=list(session_options.keys()),
            format_func=lambda sid: session_options.get(sid, sid),
            index=list(session_options.keys()).index(target_session_id) if target_session_id in session_options else 0
        )

        session_data = db.get_session(selected_id)
        sess = session_data['session']
        reps = session_data['reps']

        # Metric summary row
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Reps", sess.get('total_reps', len(reps)))
        col_s2.metric("Clean Reps", sess.get('good_reps', 0))
        col_s3.metric("Form Faults", sess.get('faulty_reps', 0))
        acc = sess.get('accuracy_pct', 0.0)
        col_s4.metric("Form Quality Score", f"{acc:.1f}%")

        st.divider()

        col_coach_report, col_rep_table = st.columns([1, 1])

        with col_coach_report:
            st.markdown("### 🤖 Grounded Coach Debrief")
            feedback = coach.generate_session_feedback(session_data)
            st.markdown(feedback['markdown_report'])

        with col_rep_table:
            st.markdown("### 📋 Rep-by-Rep Biomechanical Telemetry")
            if reps:
                df_reps = pd.DataFrame(reps)[['rep_number', 'exercise', 'is_correct', 'min_primary_angle', 'issues', 'feedback_cue']]
                df_reps['is_correct'] = df_reps['is_correct'].apply(lambda x: '✅ Good' if x == 1 or x is True else '⚠️ Fault')
                df_reps['issues'] = df_reps['issues'].apply(lambda x: ', '.join(x) if isinstance(x, list) and x else 'None')
                df_reps.columns = ['Rep #', 'Exercise', 'Form', 'Min Angle (°)', 'Issues', 'Feedback']
                st.dataframe(df_reps, use_container_width=True, hide_index=True)
            else:
                st.write("No reps recorded in this session.")

# ==============================================================================
# TAB 3: Workout History Chat Assistant
# ==============================================================================
with tab_chat:
    st.markdown("### 💬 Ask SpotterAI About Your Workout History")
    st.caption("Ask questions about your form progression, depth trends, fault distributions, or volume.")

    # Quick suggestion prompt buttons
    st.markdown("**Quick Prompts:**")
    qp1, qp2, qp3, qp4 = st.columns(4)
    if qp1.button("📉 How is my squat depth changing?"):
        prompt_text = "How has my squat depth changed this week?"
    elif qp2.button("⚠️ Which exercise has the most issues?"):
        prompt_text = "Which exercise has the most form issues?"
    elif qp3.button("📊 Show my stats for recent workouts"):
        prompt_text = "Show my stats for recent workouts"
    elif qp4.button("💪 Tell me about my bicep curl form"):
        prompt_text = "How is my bicep curl form?"
    else:
        prompt_text = None

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User query handling
    user_input = st.chat_input("Ask a question about your workouts...") or prompt_text
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = chat_assistant.ask(user_input)
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

# ==============================================================================
# TAB 4: Analytics & Trends
# ==============================================================================
with tab_analytics:
    st.markdown("### 📈 Biomechanical History & Progression Trends")
    stats = db.get_overall_stats()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Workouts", stats.get('total_sessions', 0))
    a2.metric("Total Reps Logged", stats.get('total_reps', 0))
    a3.metric("Lifetime Clean Form", f"{stats.get('overall_accuracy', 0.0):.1f}%")
    a4.metric("Exercise Types", "Curl · Press · Squat")

    st.divider()
    sessions_history = db.get_recent_sessions(limit=15)
    if sessions_history:
        df_hist = pd.DataFrame(sessions_history)
        df_hist['date'] = df_hist['start_time'].apply(lambda x: x[:10])

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Rep Volume per Workout Session")
            chart_vol = df_hist[['date', 'total_reps']].set_index('date')
            st.bar_chart(chart_vol)

        with col_c2:
            st.markdown("#### Form Quality Accuracy (%) Trend")
            chart_acc = df_hist[['date', 'accuracy_pct']].set_index('date')
            st.line_chart(chart_acc)

        st.markdown("#### Workout Log Records")
        st.dataframe(
            df_hist[['session_id', 'start_time', 'total_reps', 'good_reps', 'faulty_reps', 'accuracy_pct']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No workout history available yet.")