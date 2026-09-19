"""
tests/test_coach_chat.py
------------------------
Unit tests for SpotterAI CoachingAgent (spotter/coach.py) and
WorkoutChatAssistant (spotter/chat.py).
Operates 100% locally with mock data and in-memory SQLite (no API keys required).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
import pytest

from spotter.coach import CoachingAgent
from spotter.chat import WorkoutChatAssistant
from spotter.database import WorkoutDatabase


# ---------------------------------------------------------------------------
# Fixtures & Mock Telemetry Data
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session_data():
    """Returns a realistic mock workout session with squats and curls."""
    return {
        "session": {
            "total_reps": 10,
            "good_reps": 5,
            "faulty_reps": 5,
            "accuracy_pct": 50.0,
            "start_time": "2026-09-19T10:00:00",
        },
        "reps": [
            # 5 Squats: 2 clean (depth 78°, 80°), 3 faulty (depth 108°, 110°, 106° -> avg 108°)
            {
                "exercise": "squat",
                "rep_number": 1,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 78.0,
                "max_primary_angle": 170.0,
                "feedback_cue": "Good depth and tempo.",
            },
            {
                "exercise": "squat",
                "rep_number": 2,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 80.0,
                "max_primary_angle": 172.0,
                "feedback_cue": "Clean repetition.",
            },
            {
                "exercise": "squat",
                "rep_number": 3,
                "is_correct": False,
                "issues": ["incomplete depth"],
                "min_primary_angle": 108.0,
                "max_primary_angle": 168.0,
                "feedback_cue": "Sit deeper into the squat.",
            },
            {
                "exercise": "squat",
                "rep_number": 4,
                "is_correct": False,
                "issues": ["incomplete depth"],
                "min_primary_angle": 110.0,
                "max_primary_angle": 169.0,
                "feedback_cue": "Squat deeper below parallel.",
            },
            {
                "exercise": "squat",
                "rep_number": 5,
                "is_correct": False,
                "issues": ["incomplete depth"],
                "min_primary_angle": 106.0,
                "max_primary_angle": 167.0,
                "feedback_cue": "Hip crease must dip below knees.",
            },
            # 5 Bicep Curls: 3 clean, 2 faulty (missed full extension, bottom angle 130°, 134° < 140°)
            {
                "exercise": "curl",
                "rep_number": 6,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 35.0,
                "max_primary_angle": 165.0,
                "feedback_cue": "Full range of motion.",
            },
            {
                "exercise": "curl",
                "rep_number": 7,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 36.0,
                "max_primary_angle": 166.0,
                "feedback_cue": "Clean curl.",
            },
            {
                "exercise": "curl",
                "rep_number": 8,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 37.0,
                "max_primary_angle": 164.0,
                "feedback_cue": "Good peak contraction.",
            },
            {
                "exercise": "curl",
                "rep_number": 9,
                "is_correct": False,
                "issues": ["missed full extension"],
                "min_primary_angle": 40.0,
                "max_primary_angle": 130.0,
                "feedback_cue": "Extend elbow fully at bottom.",
            },
            {
                "exercise": "curl",
                "rep_number": 10,
                "is_correct": False,
                "issues": ["missed full extension"],
                "min_primary_angle": 42.0,
                "max_primary_angle": 134.0,
                "feedback_cue": "Extend arms completely.",
            },
        ],
    }


@pytest.fixture
def populated_db():
    """In-memory SQLite database populated with multi-day telemetry."""
    db = WorkoutDatabase(":memory:")
    now = datetime.now()

    # Session 1 (e.g. Monday - 5 days ago): 5 squats with bottom knee angle = 112°
    s1_time = (now - timedelta(days=5)).isoformat()
    s1_id = db.start_session(session_id="session-mon", start_time=s1_time)
    for i in range(1, 6):
        db.log_rep(
            s1_id,
            {
                "exercise": "squat",
                "rep_number": i,
                "is_correct": False,
                "issues": ["incomplete depth"],
                "min_primary_angle": 112.0,
                "max_primary_angle": 165.0,
                "feedback_cue": "Squat deeper",
                "timestamp": s1_time,
            },
        )
    db.end_session(s1_id)

    # Session 2 (e.g. Wednesday - 3 days ago): 5 squats with bottom knee angle = 98° (improved!)
    s2_time = (now - timedelta(days=3)).isoformat()
    s2_id = db.start_session(session_id="session-wed", start_time=s2_time)
    for i in range(1, 6):
        db.log_rep(
            s2_id,
            {
                "exercise": "squat",
                "rep_number": i,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 98.0,
                "max_primary_angle": 170.0,
                "feedback_cue": "Great depth achieved",
                "timestamp": s2_time,
            },
        )
    db.end_session(s2_id)

    # Session 3 (Yesterday - 1 day ago): 10 curls (7 good, 3 faulty with missed extension), 5 presses
    s3_time = (now - timedelta(days=1)).isoformat()
    s3_id = db.start_session(session_id="session-fri", start_time=s3_time)
    for i in range(1, 8):
        db.log_rep(
            s3_id,
            {
                "exercise": "curl",
                "rep_number": i,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 35.0,
                "max_primary_angle": 165.0,
                "feedback_cue": "Good rep",
                "timestamp": s3_time,
            },
        )
    for i in range(8, 11):
        db.log_rep(
            s3_id,
            {
                "exercise": "curl",
                "rep_number": i,
                "is_correct": False,
                "issues": ["missed full extension"],
                "min_primary_angle": 40.0,
                "max_primary_angle": 132.0,
                "feedback_cue": "Extend arms completely",
                "timestamp": s3_time,
            },
        )
    for i in range(1, 6):
        db.log_rep(
            s3_id,
            {
                "exercise": "press",
                "rep_number": i,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 70.0,
                "max_primary_angle": 175.0,
                "feedback_cue": "Solid lockout",
                "timestamp": s3_time,
            },
        )
    db.end_session(s3_id)

    return db


# ---------------------------------------------------------------------------
# Test CoachingAgent
# ---------------------------------------------------------------------------

class TestCoachingAgent:
    def test_generate_session_feedback_returns_all_keys(self, mock_session_data):
        """Verify generate_session_feedback returns score, strengths, issues, cues, and markdown report."""
        agent = CoachingAgent(api_key=None)
        feedback = agent.generate_session_feedback(mock_session_data)

        assert isinstance(feedback, dict)
        assert "score" in feedback
        assert "markdown_report" in feedback
        assert "strengths" in feedback
        assert "issues" in feedback
        assert "cues" in feedback

        # Validate types and values
        assert isinstance(feedback["score"], int)
        assert 0 <= feedback["score"] <= 100
        assert feedback["score"] == 50  # 5 clean / 10 total = 50%

        assert isinstance(feedback["markdown_report"], str)
        assert len(feedback["markdown_report"]) > 0

        assert isinstance(feedback["strengths"], list)
        assert len(feedback["strengths"]) > 0

        assert isinstance(feedback["issues"], list)
        assert len(feedback["issues"]) > 0

        assert isinstance(feedback["cues"], list)
        assert 1 <= len(feedback["cues"]) <= 2

    def test_recurring_issues_contain_actual_telemetry_numbers(self, mock_session_data):
        """Verify that recurring issues contain actual numbers from the mock reps (counts and angles)."""
        agent = CoachingAgent(api_key=None)
        feedback = agent.generate_session_feedback(mock_session_data)

        issues = feedback["issues"]
        issues_str = " ".join(str(i) for i in issues)
        report = feedback["markdown_report"]

        # Check squat depth issue: 3 of 5 squats, average bottom angle 108°
        squat_issue = next((i for i in issues if i.get("exercise") == "squat"), None)
        assert squat_issue is not None
        assert squat_issue["count"] == 3
        assert squat_issue["total"] == 5
        assert squat_issue["avg_angle"] == 108.0

        # Verify string description has exact numbers
        assert "3 of 5" in str(squat_issue)
        assert "108" in str(squat_issue)
        assert "108" in report

        # Check curl extension issue: 2 curls missed full extension (<140°)
        curl_issue = next((i for i in issues if i.get("exercise") == "curl"), None)
        assert curl_issue is not None
        assert curl_issue["count"] == 2
        assert "2" in str(curl_issue)
        assert "extension" in str(curl_issue).lower()

    def test_flawless_session_handling(self):
        """Verify CoachingAgent handles a 100% clean workout flawlessly."""
        agent = CoachingAgent(api_key=None)
        flawless_session = {
            "session": {
                "total_reps": 8,
                "good_reps": 8,
                "faulty_reps": 0,
                "accuracy_pct": 100.0,
            },
            "reps": [
                {
                    "exercise": "squat",
                    "rep_number": i,
                    "is_correct": True,
                    "issues": [],
                    "min_primary_angle": 75.0,
                    "max_primary_angle": 170.0,
                    "feedback_cue": "Perfect depth",
                }
                for i in range(1, 9)
            ],
        }
        feedback = agent.generate_session_feedback(flawless_session)
        assert feedback["score"] == 100
        assert len(feedback["issues"]) == 0
        assert any("100%" in s or "flawless" in s.lower() for s in feedback["strengths"])

    def test_provider_and_env_initialization(self, monkeypatch):
        """Verify API key auto-detection from environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_123")
        agent_gemini = CoachingAgent()
        assert agent_gemini.provider == "gemini"
        assert agent_gemini.api_key == "test_gemini_key_123"

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test_openai_key")
        agent_openai = CoachingAgent()
        assert agent_openai.provider == "openai"
        assert agent_openai.api_key == "sk-test_openai_key"


# ---------------------------------------------------------------------------
# Test WorkoutChatAssistant
# ---------------------------------------------------------------------------

class TestWorkoutChatAssistant:
    def test_ask_squat_depth_trend(self, populated_db):
        """Test asking about squat depth trend over the week."""
        assistant = WorkoutChatAssistant(db=populated_db)
        response = assistant.ask("How has my squat depth changed this week?")

        assert isinstance(response, str)
        # Should detect the progression from 112° to 98° (14° improvement)
        assert "112" in response
        assert "98" in response
        assert "14" in response
        assert "squat" in response.lower()

    def test_ask_which_exercise_has_most_issues(self, populated_db):
        """Test asking which exercise has the most form issues."""
        assistant = WorkoutChatAssistant(db=populated_db)
        response = assistant.ask("Which exercise has the most form issues?")

        assert isinstance(response, str)
        # In populated_db: Squat had 5 faults, Curl had 3 faults, Press had 0 faults
        assert "Squat" in response
        assert "5 faults" in response or "5" in response
        assert "incomplete depth" in response

    def test_ask_total_reps(self, populated_db):
        """Test asking about total reps completed all-time."""
        assistant = WorkoutChatAssistant(db=populated_db)
        response = assistant.ask("How many reps have I done?")

        assert isinstance(response, str)
        # Total reps in populated_db: 5 squats (session 1) + 5 squats (session 2) + 10 curls + 5 presses = 25 reps
        assert "25" in response
        assert "total" in response.lower()
        assert "Squat" in response
        assert "Curl" in response
        assert "Press" in response

    def test_ask_bicep_curl_form(self, populated_db):
        """Test asking about specific exercise form (bicep curl)."""
        assistant = WorkoutChatAssistant(db=populated_db)
        response = assistant.ask("How is my bicep curl form?")

        assert isinstance(response, str)
        assert "Curl" in response
        assert "10" in response  # 10 curl reps
        assert "missed full extension" in response or "extension" in response.lower()

    def test_ask_with_mock_db_object(self):
        """Test WorkoutChatAssistant compatibility with a custom mock DB object."""
        mock_db = MagicMock()
        mock_db.get_reps.return_value = [
            {
                "exercise": "squat",
                "rep_number": 1,
                "is_correct": False,
                "issues": ["incomplete depth"],
                "min_primary_angle": 105.0,
                "timestamp": "2026-09-18T12:00:00",
            },
            {
                "exercise": "squat",
                "rep_number": 2,
                "is_correct": True,
                "issues": [],
                "min_primary_angle": 95.0,
                "timestamp": "2026-09-19T12:00:00",
            },
        ]
        mock_db.get_total_reps.return_value = 2

        assistant = WorkoutChatAssistant(db=mock_db)
        res = assistant.ask("How has my squat depth changed?")
        assert "105" in res
        assert "95" in res
        assert "10" in res  # 10° improvement

    def test_ask_recent_stats(self, populated_db):
        """Test asking for recent workout stats."""
        assistant = WorkoutChatAssistant(db=populated_db)
        res = assistant.ask("Show my stats for today / recent workouts")
        assert "Workout Stats" in res
        assert "Total Reps" in res
        assert "Exercise Breakdown" in res

    def test_ask_fallback_overview(self, populated_db):
        """Test asking an unrecognised question returns the assistant overview."""
        assistant = WorkoutChatAssistant(db=populated_db)
        res = assistant.ask("What kind of things can I ask you?")
        assert "SpotterAI Workout History Assistant" in res
        assert "squat depth" in res.lower()

    def test_chat_llm_enhancement_and_fallback(self, populated_db, monkeypatch):
        """Test chat assistant calls LLM when key provided, and falls back on error."""
        # 1. Successful LLM response
        assistant = WorkoutChatAssistant(db=populated_db, api_key="fake-key")
        monkeypatch.setattr(
            assistant,
            "_enhance_with_llm",
            lambda q, ctx: "🌟 Polished LLM Coaching Summary: You improved your squat by 14°!",
        )
        res = assistant.ask("How has my squat depth changed this week?")
        assert "🌟 Polished LLM Coaching Summary" in res

        # 2. Failed LLM response falls back cleanly to deterministic response
        monkeypatch.setattr(
            assistant,
            "_enhance_with_llm",
            lambda q, ctx: None,
        )
        res_fallback = assistant.ask("How has my squat depth changed this week?")
        assert "Squat Depth Trend Analysis" in res_fallback
        assert "112" in res_fallback


class TestCoachingAgentLLM:
    def test_llm_enhancement_success(self, mock_session_data, monkeypatch):
        """Verify that successful LLM response enhances markdown_report."""
        agent = CoachingAgent(api_key="fake_key", provider="gemini")
        mock_report = "### 🏆 Elite Coach Debrief\n\nGreat work hitting 10 total reps with solid recovery."
        monkeypatch.setattr(agent, "_call_llm", lambda prompt: mock_report)

        feedback = agent.generate_session_feedback(mock_session_data)
        assert feedback["markdown_report"] == mock_report
        assert feedback["score"] == 50

    def test_llm_enhancement_fallback_on_exception(self, mock_session_data, monkeypatch):
        """Verify that LLM exception cleanly falls back to deterministic report."""
        agent = CoachingAgent(api_key="fake_key", provider="gemini")

        def raise_err(prompt):
            raise ConnectionError("Network unreachable")

        monkeypatch.setattr(agent, "_call_llm", raise_err)

        feedback = agent.generate_session_feedback(mock_session_data)
        assert "Post-Workout Coaching Debrief" in feedback["markdown_report"]
        assert "108" in feedback["markdown_report"]
        assert feedback["score"] == 50

    def test_empty_session_handling(self):
        """Verify CoachingAgent handles empty reps gracefully."""
        agent = CoachingAgent(api_key=None)
        feedback = agent.generate_session_feedback({"session": {}, "reps": []})
        assert feedback["score"] == 100
        assert feedback["issues"] == []
        assert isinstance(feedback["markdown_report"], str)

