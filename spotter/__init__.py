"""
spotter package
---------------
SpotterAI: Real-Time AI Fitness Coach & Form Analysis Engine
"""

from spotter.tracker import RepTracker, calculate_angle
from spotter.visualizer import draw_landmarks, draw_form_skeleton, draw_info_box
from spotter.model import build_model
from spotter.pose import extract_keypoints
from spotter.database import WorkoutDatabase
from spotter.coach import CoachingAgent
from spotter.chat import WorkoutChatAssistant

__all__ = [
    "RepTracker",
    "calculate_angle",
    "draw_landmarks",
    "draw_form_skeleton",
    "draw_info_box",
    "build_model",
    "extract_keypoints",
    "WorkoutDatabase",
    "CoachingAgent",
    "WorkoutChatAssistant",
]
