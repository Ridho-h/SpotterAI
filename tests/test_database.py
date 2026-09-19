"""
tests/test_database.py
----------------------
Unit tests for spotter.database – WorkoutDatabase.
Tests are executed on an in-memory SQLite database (:memory:).
No external files, models, or network connections required.
"""

from datetime import datetime, timedelta
import json
import pytest

from spotter.database import WorkoutDatabase


@pytest.fixture
def db():
    """Provides a fresh in-memory WorkoutDatabase for each test."""
    database = WorkoutDatabase(db_path=":memory:")
    yield database
    database.close()


def test_create_tables(db):
    """Verifies that the sessions and reps tables and indexes are created properly."""
    cur = db.conn.cursor()

    # Verify tables exist
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'reps')"
    )
    tables = {row["name"] for row in cur.fetchall()}
    assert "sessions" in tables, "Table 'sessions' was not created"
    assert "reps" in tables, "Table 'reps' was not created"

    # Verify sessions schema columns
    cur.execute("PRAGMA table_info(sessions)")
    session_cols = {row["name"]: row["type"].upper() for row in cur.fetchall()}
    expected_session_cols = [
        "session_id",
        "start_time",
        "end_time",
        "total_reps",
        "good_reps",
        "faulty_reps",
        "accuracy_pct",
        "summary_json",
    ]
    for col in expected_session_cols:
        assert col in session_cols, f"Column '{col}' missing from 'sessions'"

    # Verify reps schema columns
    cur.execute("PRAGMA table_info(reps)")
    rep_cols = {row["name"]: row["type"].upper() for row in cur.fetchall()}
    expected_rep_cols = [
        "rep_id",
        "session_id",
        "exercise",
        "rep_number",
        "timestamp",
        "duration_sec",
        "min_primary_angle",
        "max_primary_angle",
        "secondary_angle",
        "is_correct",
        "issues_json",
        "feedback_cue",
    ]
    for col in expected_rep_cols:
        assert col in rep_cols, f"Column '{col}' missing from 'reps'"

    # Verify foreign keys are enabled
    cur.execute("PRAGMA foreign_keys")
    fk_enabled = cur.fetchone()[0]
    assert fk_enabled == 1, "Foreign keys should be enabled"


def test_start_and_end_session(db):
    """Tests starting a session, logging reps, and ending it with calculated accuracy."""
    session_id = db.start_session()
    assert isinstance(session_id, str)
    assert len(session_id) == 8

    # Verify initial session state in database
    session_data = db.get_session(session_id)
    assert session_data is not None
    assert session_data["session"]["session_id"] == session_id
    assert session_data["session"]["total_reps"] == 0
    assert session_data["session"]["accuracy_pct"] == 0.0
    assert session_data["session"]["end_time"] is None
    assert len(session_data["reps"]) == 0

    # Log 3 reps: 2 correct, 1 faulty
    db.log_rep(
        session_id,
        {
            "exercise": "squat",
            "rep_number": 1,
            "duration_sec": 2.5,
            "min_primary_angle": 75.0,
            "max_primary_angle": 170.0,
            "secondary_angle": 80.0,
            "is_correct": True,
            "issues": [],
            "feedback_cue": "Good depth!",
        },
    )
    db.log_rep(
        session_id,
        {
            "exercise": "squat",
            "rep_number": 2,
            "duration_sec": 2.4,
            "min_primary_angle": 105.0,
            "max_primary_angle": 168.0,
            "secondary_angle": 100.0,
            "is_correct": False,
            "issues": ["insufficient_depth"],
            "feedback_cue": "Squat deeper.",
        },
    )
    db.log_rep(
        session_id,
        {
            "exercise": "squat",
            "rep_number": 3,
            "duration_sec": 2.6,
            "min_primary_angle": 78.0,
            "max_primary_angle": 171.0,
            "secondary_angle": 82.0,
            "is_correct": True,
            "issues": [],
            "feedback_cue": "Solid rep.",
        },
    )

    # End session
    summary = db.end_session(session_id)
    assert summary["session_id"] == session_id
    assert summary["total_reps"] == 3
    assert summary["good_reps"] == 2
    assert summary["faulty_reps"] == 1
    assert summary["accuracy_pct"] == 66.7
    assert summary["end_time"] is not None

    # Verify persistent state via get_session
    updated_session = db.get_session(session_id)
    s = updated_session["session"]
    assert s["total_reps"] == 3
    assert s["good_reps"] == 2
    assert s["faulty_reps"] == 1
    assert s["accuracy_pct"] == 66.7
    assert s["end_time"] is not None
    assert len(updated_session["reps"]) == 3


def test_log_rep_and_retrieval(db):
    """Tests logging reps with detailed fields and retrieving session and recent sessions."""
    session_id = db.start_session()

    rep_payload = {
        "exercise": "curl",
        "rep_number": 1,
        "duration_sec": 1.85,
        "min_primary_angle": 35.2,
        "max_primary_angle": 165.8,
        "secondary_angle": 12.4,
        "is_correct": False,
        "issues": ["incomplete_extension", "elbow_drift"],
        "feedback_cue": "Keep elbow steady and fully extend arm.",
    }

    rep_id = db.log_rep(session_id, rep_payload)
    assert isinstance(rep_id, int)
    assert rep_id > 0

    session_info = db.get_session(session_id)
    assert session_info is not None
    assert len(session_info["reps"]) == 1

    rep = session_info["reps"][0]
    assert rep["rep_id"] == rep_id
    assert rep["session_id"] == session_id
    assert rep["exercise"] == "curl"
    assert rep["rep_number"] == 1
    assert rep["duration_sec"] == pytest.approx(1.85)
    assert rep["min_primary_angle"] == pytest.approx(35.2)
    assert rep["max_primary_angle"] == pytest.approx(165.8)
    assert rep["secondary_angle"] == pytest.approx(12.4)
    assert rep["is_correct"] is False
    assert rep["issues"] == ["incomplete_extension", "elbow_drift"]
    assert rep["feedback_cue"] == "Keep elbow steady and fully extend arm."

    # Test get_recent_sessions
    recent = db.get_recent_sessions(limit=5)
    assert len(recent) == 1
    assert recent[0]["session_id"] == session_id

    # Test export_session_json
    json_str = db.export_session_json(session_id)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["session"]["session_id"] == session_id
    assert len(parsed["reps"]) == 1
    assert parsed["reps"][0]["exercise"] == "curl"

    # Non-existent session
    assert db.get_session("non_existent_id") is None
    export_non_existent = db.export_session_json("non_existent_id")
    assert "error" in json.loads(export_non_existent)


def test_exercise_history(db):
    """Tests filtering exercise history by exercise name and date ranges."""
    now = datetime.now()

    # Session 1: Curls 10 days ago
    s1_id = db.start_session(start_time=(now - timedelta(days=10)).isoformat())
    db.log_rep(
        s1_id,
        {
            "exercise": "curl",
            "rep_number": 1,
            "duration_sec": 2.0,
            "min_primary_angle": 38.0,
            "max_primary_angle": 160.0,
            "secondary_angle": 10.0,
            "is_correct": True,
            "issues": [],
            "feedback_cue": "Good curl",
            "timestamp": (now - timedelta(days=10)).isoformat(),
        },
    )
    db.end_session(s1_id)

    # Session 2: Press 5 days ago
    s2_id = db.start_session(start_time=(now - timedelta(days=5)).isoformat())
    db.log_rep(
        s2_id,
        {
            "exercise": "press",
            "rep_number": 1,
            "duration_sec": 2.5,
            "min_primary_angle": 70.0,
            "max_primary_angle": 172.0,
            "secondary_angle": 170.0,
            "is_correct": True,
            "issues": [],
            "feedback_cue": "Good press",
            "timestamp": (now - timedelta(days=5)).isoformat(),
        },
    )
    db.end_session(s2_id)

    # Session 3: Old Squats 40 days ago
    s3_id = db.start_session(start_time=(now - timedelta(days=40)).isoformat())
    db.log_rep(
        s3_id,
        {
            "exercise": "squat",
            "rep_number": 1,
            "duration_sec": 3.0,
            "min_primary_angle": 80.0,
            "max_primary_angle": 170.0,
            "secondary_angle": 85.0,
            "is_correct": True,
            "issues": [],
            "feedback_cue": "Old squat",
            "timestamp": (now - timedelta(days=40)).isoformat(),
        },
    )
    db.end_session(s3_id)

    # Query last 30 days - should include s1 and s2, but exclude s3
    history_30d = db.get_exercise_history(days=30)
    assert len(history_30d) == 2
    exercises = [r["exercise"] for r in history_30d]
    assert "curl" in exercises
    assert "press" in exercises
    assert "squat" not in exercises

    # Filter by exercise: curl
    curl_history = db.get_exercise_history(exercise="curl", days=30)
    assert len(curl_history) == 1
    assert curl_history[0]["exercise"] == "curl"
    assert "session_start_time" in curl_history[0]

    # Filter by exercise: press
    press_history = db.get_exercise_history(exercise="press", days=30)
    assert len(press_history) == 1
    assert press_history[0]["exercise"] == "press"

    # Query last 60 days - should include all 3
    history_60d = db.get_exercise_history(days=60)
    assert len(history_60d) == 3


def test_seed_demo_data(db):
    """Verifies that seed_demo_data_if_empty populates 3 realistic sessions and reps."""
    # Ensure initially empty
    stats_initial = db.get_overall_stats()
    assert stats_initial["total_workouts"] == 0
    assert stats_initial["total_reps"] == 0

    # Seed demo data
    db.seed_demo_data_if_empty()

    # Check sessions count
    recent = db.get_recent_sessions(limit=10)
    assert len(recent) == 3

    # Check overall stats
    stats = db.get_overall_stats()
    assert stats["total_workouts"] == 3
    assert stats["total_reps"] == 28  # 10 squats + 10 curls + 8 presses
    assert stats["squat_reps"] == 10
    assert stats["curl_reps"] == 10
    assert stats["press_reps"] == 8
    assert stats["good_reps"] > 0
    assert stats["faulty_reps"] > 0
    assert 0.0 < stats["accuracy_pct"] < 100.0

    # Check that individual sessions have faulty reps with realistic issues
    squat_session = db.get_session("demo-sq1")
    assert squat_session is not None
    assert len(squat_session["reps"]) == 10
    faulty_squats = [r for r in squat_session["reps"] if not r["is_correct"]]
    assert len(faulty_squats) == 3
    assert any("insufficient_depth" in r["issues"] for r in faulty_squats)

    curl_session = db.get_session("demo-cr2")
    assert curl_session is not None
    assert len(curl_session["reps"]) == 10
    faulty_curls = [r for r in curl_session["reps"] if not r["is_correct"]]
    assert len(faulty_curls) == 3
    assert any("incomplete_extension" in r["issues"] for r in faulty_curls)

    press_session = db.get_session("demo-pr3")
    assert press_session is not None
    assert len(press_session["reps"]) == 8
    faulty_presses = [r for r in press_session["reps"] if not r["is_correct"]]
    assert len(faulty_presses) == 2
    assert any("incomplete_lockout" in r["issues"] for r in faulty_presses)

    # Verify idempotency: calling seed again should not duplicate data
    db.seed_demo_data_if_empty()
    assert len(db.get_recent_sessions(limit=10)) == 3
    assert db.get_overall_stats()["total_reps"] == 28
