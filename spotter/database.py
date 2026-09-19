"""
spotter/database.py
-------------------
Structured session logging and SQLite database engine for SpotterAI.

Manages workout sessions, rep-level telemetry (angles, durations, form issues,
and feedback cues), analytics aggregation, and demo data seeding.
"""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "spotter.db"


class WorkoutDatabase:
    """
    Manages SQLite database storage for workout sessions and individual rep logs.

    Parameters:
        db_path (str, optional): Path to the SQLite database file.
            Defaults to ``data/spotter.db`` relative to project root.
            Pass ``':memory:'`` for an in-memory test database.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = str(DEFAULT_DB_PATH)
        else:
            self.db_path = db_path

        # Create parent directory if storing to a file path
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initialise database schema and foreign key constraints."""
        with self.conn:
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_reps INTEGER DEFAULT 0,
                    good_reps INTEGER DEFAULT 0,
                    faulty_reps INTEGER DEFAULT 0,
                    accuracy_pct REAL DEFAULT 0.0,
                    summary_json TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS reps (
                    rep_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    exercise TEXT,
                    rep_number INTEGER,
                    timestamp TEXT,
                    duration_sec REAL DEFAULT 0.0,
                    min_primary_angle REAL DEFAULT 0.0,
                    max_primary_angle REAL DEFAULT 0.0,
                    secondary_angle REAL DEFAULT 0.0,
                    is_correct INTEGER DEFAULT 1,
                    issues_json TEXT DEFAULT '[]',
                    feedback_cue TEXT DEFAULT '',
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_reps_session_id ON reps(session_id);
                CREATE INDEX IF NOT EXISTS idx_reps_exercise ON reps(exercise);
                CREATE INDEX IF NOT EXISTS idx_reps_timestamp ON reps(timestamp);
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
                """
            )

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -----------------------------------------------------------------------
    # Session Management
    # -----------------------------------------------------------------------

    def start_session(
        self, session_id: Optional[str] = None, start_time: Optional[str] = None
    ) -> str:
        """
        Creates a new session record and returns the session_id.

        Args:
            session_id (str, optional): Custom ID. If not provided, generates 8-char hex UUID.
            start_time (str, optional): ISO timestamp. Defaults to current UTC/local ISO string.

        Returns:
            str: The session ID.
        """
        if not session_id:
            session_id = uuid.uuid4().hex[:8]
        if not start_time:
            start_time = datetime.now().isoformat()

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO sessions (session_id, start_time)
                VALUES (?, ?)
                """,
                (session_id, start_time),
            )
        return session_id

    def log_rep(self, session_id: str, rep_data: Dict[str, Any]) -> int:
        """
        Inserts a rep row into the database.

        Args:
            session_id (str): Associated session identifier.
            rep_data (dict): Dictionary with rep telemetry:
                - exercise (str)
                - rep_number (int)
                - duration_sec (float)
                - min_primary_angle (float)
                - max_primary_angle (float)
                - secondary_angle (float)
                - is_correct (bool or int, 1 for True, 0 for False)
                - issues (list of str, optional)
                - feedback_cue (str, optional)
                - timestamp (str, optional ISO timestamp)

        Returns:
            int: The created rep_id.
        """
        exercise = rep_data.get("exercise", "")
        rep_number = int(rep_data.get("rep_number", 1))
        timestamp = rep_data.get("timestamp") or datetime.now().isoformat()
        duration_sec = float(rep_data.get("duration_sec", 0.0))
        min_primary_angle = float(rep_data.get("min_primary_angle", 0.0))
        max_primary_angle = float(rep_data.get("max_primary_angle", 0.0))
        secondary_angle = float(rep_data.get("secondary_angle", 0.0))

        is_correct_val = rep_data.get("is_correct", 1)
        if isinstance(is_correct_val, bool):
            is_correct = 1 if is_correct_val else 0
        else:
            is_correct = 1 if int(is_correct_val) != 0 else 0

        # Handle issues list -> json string
        issues_val = rep_data.get("issues")
        if issues_val is not None:
            if isinstance(issues_val, list):
                issues_json = json.dumps(issues_val)
            elif isinstance(issues_val, str):
                issues_json = issues_val
            else:
                issues_json = "[]"
        elif "issues_json" in rep_data:
            issues_json = rep_data["issues_json"]
        else:
            issues_json = "[]"

        feedback_cue = rep_data.get("feedback_cue", "")

        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO reps (
                    session_id, exercise, rep_number, timestamp, duration_sec,
                    min_primary_angle, max_primary_angle, secondary_angle,
                    is_correct, issues_json, feedback_cue
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    exercise,
                    rep_number,
                    timestamp,
                    duration_sec,
                    min_primary_angle,
                    max_primary_angle,
                    secondary_angle,
                    is_correct,
                    issues_json,
                    feedback_cue,
                ),
            )
            rep_id = cur.lastrowid

        return rep_id

    def end_session(
        self, session_id: str, end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Computes summary metrics (total_reps, good_reps, faulty_reps, accuracy_pct)
        from the reps table for this session, updates the session record, sets end_time,
        and returns the updated session summary dict.

        Args:
            session_id (str): The session to finalize.
            end_time (str, optional): ISO timestamp. Defaults to current datetime.

        Returns:
            dict: Summary dictionary of the session.
        """
        if end_time is None:
            end_time = datetime.now().isoformat()

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT 
                COUNT(*) AS total_reps,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS good_reps,
                SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) AS faulty_reps
            FROM reps
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = cur.fetchone()
        total_reps = row["total_reps"] if row and row["total_reps"] is not None else 0
        good_reps = row["good_reps"] if row and row["good_reps"] is not None else 0
        faulty_reps = row["faulty_reps"] if row and row["faulty_reps"] is not None else 0

        accuracy_pct = (
            round((good_reps / total_reps * 100.0), 1) if total_reps > 0 else 0.0
        )

        cur.execute(
            """
            SELECT exercise, COUNT(*) as count,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as good
            FROM reps
            WHERE session_id = ?
            GROUP BY exercise
            """,
            (session_id,),
        )
        exercise_summary = {
            r["exercise"]: {
                "total": r["count"],
                "good": r["good"],
                "faulty": r["count"] - r["good"],
            }
            for r in cur.fetchall()
        }

        summary_data = {
            "total_reps": total_reps,
            "good_reps": good_reps,
            "faulty_reps": faulty_reps,
            "accuracy_pct": accuracy_pct,
            "exercise_breakdown": exercise_summary,
        }
        summary_json = json.dumps(summary_data)

        with self.conn:
            self.conn.execute(
                """
                UPDATE sessions
                SET end_time = ?,
                    total_reps = ?,
                    good_reps = ?,
                    faulty_reps = ?,
                    accuracy_pct = ?,
                    summary_json = ?
                WHERE session_id = ?
                """,
                (
                    end_time,
                    total_reps,
                    good_reps,
                    faulty_reps,
                    accuracy_pct,
                    summary_json,
                    session_id,
                ),
            )

        cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session_row = cur.fetchone()
        res = dict(session_row) if session_row else summary_data
        if "summary_json" in res:
            res["summary"] = summary_data
        return res

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a complete session and all of its associated reps.

        Args:
            session_id (str): Session identifier.

        Returns:
            dict or None: { 'session': dict, 'reps': list[dict] } or None if not found.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session_row = cur.fetchone()
        if not session_row:
            return None

        session_dict = dict(session_row)
        if session_dict.get("summary_json"):
            try:
                session_dict["summary"] = json.loads(session_dict["summary_json"])
            except Exception:
                session_dict["summary"] = {}

        cur.execute(
            """
            SELECT * FROM reps
            WHERE session_id = ?
            ORDER BY rep_number ASC, rep_id ASC
            """,
            (session_id,),
        )
        reps_list = []
        for r in cur.fetchall():
            rd = dict(r)
            rd["is_correct"] = bool(rd["is_correct"])
            try:
                rd["issues"] = json.loads(rd.get("issues_json") or "[]")
            except Exception:
                rd["issues"] = []
            reps_list.append(rd)

        return {"session": session_dict, "reps": reps_list}

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns list of recent session summaries ordered by start_time DESC.

        Args:
            limit (int): Maximum number of sessions to return.

        Returns:
            list[dict]: List of session dictionaries.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM sessions
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (limit,),
        )
        sessions = []
        for r in cur.fetchall():
            sd = dict(r)
            if sd.get("summary_json"):
                try:
                    sd["summary"] = json.loads(sd["summary_json"])
                except Exception:
                    sd["summary"] = {}
            sessions.append(sd)
        return sessions

    def get_exercise_history(
        self, exercise: Optional[str] = None, days: Optional[int] = 30
    ) -> List[Dict[str, Any]]:
        """
        Returns list of reps joined with session info within the last `days` days.
        If `exercise` is given, filters by exercise.

        Args:
            exercise (str, optional): Exercise name (e.g. 'curl', 'squat', 'press').
            days (int): Number of days back to include. Defaults to 30.

        Returns:
            list[dict]: List of joined rep dictionaries.
        """
        if days is None:
            days = 30
        cutoff_str = (datetime.now() - timedelta(days=days)).isoformat()
        params: List[Any] = [cutoff_str, cutoff_str]
        where_clauses = ["(r.timestamp >= ? OR s.start_time >= ?)"]

        if exercise:
            where_clauses.append("LOWER(r.exercise) = LOWER(?)")
            params.append(exercise)

        query = f"""
            SELECT 
                r.rep_id,
                r.session_id,
                r.exercise,
                r.rep_number,
                r.timestamp,
                r.duration_sec,
                r.min_primary_angle,
                r.max_primary_angle,
                r.secondary_angle,
                r.is_correct,
                r.issues_json,
                r.feedback_cue,
                s.start_time AS session_start_time,
                s.end_time AS session_end_time,
                s.accuracy_pct AS session_accuracy_pct
            FROM reps r
            JOIN sessions s ON r.session_id = s.session_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY r.timestamp DESC, r.rep_id DESC
        """

        cur = self.conn.cursor()
        cur.execute(query, params)
        results = []
        for r in cur.fetchall():
            rd = dict(r)
            rd["is_correct"] = bool(rd["is_correct"])
            try:
                rd["issues"] = json.loads(rd.get("issues_json") or "[]")
            except Exception:
                rd["issues"] = []
            results.append(rd)
        return results

    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Computes aggregate statistics across all workouts:
        Total workouts, total reps, average accuracy, total curl, press, squat reps.

        Returns:
            dict: Aggregated workout statistics.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT 
                COUNT(*) AS total_workouts,
                COALESCE(SUM(total_reps), 0) AS total_reps,
                COALESCE(SUM(good_reps), 0) AS good_reps,
                COALESCE(SUM(faulty_reps), 0) AS faulty_reps,
                COALESCE(AVG(CASE WHEN total_reps > 0 THEN accuracy_pct ELSE NULL END), 0.0) AS avg_accuracy
            FROM sessions
            """
        )
        s_row = cur.fetchone()
        total_workouts = s_row["total_workouts"] if s_row else 0
        total_reps_session = s_row["total_reps"] if s_row else 0
        good_reps_session = s_row["good_reps"] if s_row else 0
        faulty_reps_session = s_row["faulty_reps"] if s_row else 0
        avg_accuracy = (
            round(float(s_row["avg_accuracy"]), 1)
            if s_row and s_row["avg_accuracy"] is not None
            else 0.0
        )

        # Direct verification from reps table
        cur.execute(
            """
            SELECT 
                COUNT(*) AS total_reps,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS good_reps,
                SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) AS faulty_reps
            FROM reps
            """
        )
        r_row = cur.fetchone()
        total_reps_table = (
            r_row["total_reps"] if r_row and r_row["total_reps"] is not None else 0
        )
        good_reps_table = (
            r_row["good_reps"] if r_row and r_row["good_reps"] is not None else 0
        )
        faulty_reps_table = (
            r_row["faulty_reps"] if r_row and r_row["faulty_reps"] is not None else 0
        )

        total_reps = max(total_reps_session, total_reps_table)
        good_reps = max(good_reps_session, good_reps_table)
        faulty_reps = max(faulty_reps_session, faulty_reps_table)

        if avg_accuracy == 0.0 and total_reps > 0:
            avg_accuracy = round(good_reps / total_reps * 100.0, 1)

        # Rep count breakdown per exercise
        cur.execute(
            """
            SELECT LOWER(exercise) AS ex, COUNT(*) AS count
            FROM reps
            GROUP BY LOWER(exercise)
            """
        )
        counts_by_ex = {row["ex"]: row["count"] for row in cur.fetchall()}
        curl_reps = counts_by_ex.get("curl", 0) + counts_by_ex.get("bicep curl", 0)
        press_reps = counts_by_ex.get("press", 0) + counts_by_ex.get("overhead press", 0)
        squat_reps = counts_by_ex.get("squat", 0)

        return {
            "total_workouts": total_workouts,
            "total_sessions": total_workouts,
            "total_reps": total_reps,
            "good_reps": good_reps,
            "faulty_reps": faulty_reps,
            "accuracy_pct": avg_accuracy,
            "avg_accuracy": avg_accuracy,
            "average_accuracy": avg_accuracy,
            "curl_reps": curl_reps,
            "press_reps": press_reps,
            "squat_reps": squat_reps,
            "total_curl_reps": curl_reps,
            "total_press_reps": press_reps,
            "total_squat_reps": squat_reps,
            "exercise_breakdown": {
                "curl": curl_reps,
                "press": press_reps,
                "squat": squat_reps,
            },
        }

    def seed_demo_data_if_empty(self):
        """
        Seeds 3 realistic previous workout sessions (3 days ago, 2 days ago, and yesterday)
        with realistic reps (squats with depth issues, curls with extension issues, presses)
        if the database has 0 sessions.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM sessions")
        count = cur.fetchone()["cnt"]
        if count > 0:
            return  # Already seeded or has existing user data

        now = datetime.now()

        # -------------------------------------------------------------------
        # Session 1: Squat Session (3 days ago)
        # -------------------------------------------------------------------
        s1_start = (now - timedelta(days=3, minutes=30)).isoformat()
        s1_end = (now - timedelta(days=3, minutes=12)).isoformat()
        s1_id = self.start_session(session_id="demo-sq1", start_time=s1_start)

        squat_reps_data = [
            {
                "exercise": "squat",
                "rep_number": 1,
                "duration_sec": 2.4,
                "min_primary_angle": 76.5,
                "max_primary_angle": 172.0,
                "secondary_angle": 82.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Good depth and steady pacing.",
                "timestamp": (now - timedelta(days=3, minutes=28)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 2,
                "duration_sec": 2.3,
                "min_primary_angle": 78.0,
                "max_primary_angle": 170.5,
                "secondary_angle": 84.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Clean rep.",
                "timestamp": (now - timedelta(days=3, minutes=26)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 3,
                "duration_sec": 2.1,
                "min_primary_angle": 102.0,
                "max_primary_angle": 168.0,
                "secondary_angle": 105.0,
                "is_correct": False,
                "issues": ["insufficient_depth"],
                "feedback_cue": "Squat deeper — hips should break parallel.",
                "timestamp": (now - timedelta(days=3, minutes=24)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 4,
                "duration_sec": 2.5,
                "min_primary_angle": 79.0,
                "max_primary_angle": 171.0,
                "secondary_angle": 83.5,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Good depth.",
                "timestamp": (now - timedelta(days=3, minutes=22)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 5,
                "duration_sec": 2.0,
                "min_primary_angle": 98.5,
                "max_primary_angle": 169.0,
                "secondary_angle": 101.0,
                "is_correct": False,
                "issues": ["insufficient_depth"],
                "feedback_cue": "Hip crease remained above knee height.",
                "timestamp": (now - timedelta(days=3, minutes=20)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 6,
                "duration_sec": 2.6,
                "min_primary_angle": 77.0,
                "max_primary_angle": 172.5,
                "secondary_angle": 81.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Excellent form.",
                "timestamp": (now - timedelta(days=3, minutes=18)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 7,
                "duration_sec": 2.7,
                "min_primary_angle": 80.0,
                "max_primary_angle": 170.0,
                "secondary_angle": 85.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Consistent speed.",
                "timestamp": (now - timedelta(days=3, minutes=16)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 8,
                "duration_sec": 2.8,
                "min_primary_angle": 75.0,
                "max_primary_angle": 165.0,
                "secondary_angle": 60.0,
                "is_correct": False,
                "issues": ["knee_valgus", "excessive_forward_lean"],
                "feedback_cue": "Keep chest proud and drive knees out.",
                "timestamp": (now - timedelta(days=3, minutes=15)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 9,
                "duration_sec": 2.8,
                "min_primary_angle": 78.5,
                "max_primary_angle": 171.0,
                "secondary_angle": 82.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Solid recovery.",
                "timestamp": (now - timedelta(days=3, minutes=14)).isoformat(),
            },
            {
                "exercise": "squat",
                "rep_number": 10,
                "duration_sec": 3.1,
                "min_primary_angle": 81.0,
                "max_primary_angle": 169.5,
                "secondary_angle": 84.5,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Finished strong on the last rep.",
                "timestamp": (now - timedelta(days=3, minutes=13)).isoformat(),
            },
        ]
        for rep in squat_reps_data:
            self.log_rep(s1_id, rep)
        self.end_session(s1_id, end_time=s1_end)

        # -------------------------------------------------------------------
        # Session 2: Bicep Curl Session (2 days ago)
        # -------------------------------------------------------------------
        s2_start = (now - timedelta(days=2, minutes=40)).isoformat()
        s2_end = (now - timedelta(days=2, minutes=25)).isoformat()
        s2_id = self.start_session(session_id="demo-cr2", start_time=s2_start)

        curl_reps_data = [
            {
                "exercise": "curl",
                "rep_number": 1,
                "duration_sec": 2.1,
                "min_primary_angle": 36.0,
                "max_primary_angle": 168.0,
                "secondary_angle": 12.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Full range of motion.",
                "timestamp": (now - timedelta(days=2, minutes=38)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 2,
                "duration_sec": 2.0,
                "min_primary_angle": 38.5,
                "max_primary_angle": 165.0,
                "secondary_angle": 14.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Good contraction.",
                "timestamp": (now - timedelta(days=2, minutes=36)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 3,
                "duration_sec": 1.9,
                "min_primary_angle": 42.0,
                "max_primary_angle": 138.0,
                "secondary_angle": 15.0,
                "is_correct": False,
                "issues": ["incomplete_extension"],
                "feedback_cue": "Extend arm fully at the bottom — don't cut it short.",
                "timestamp": (now - timedelta(days=2, minutes=35)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 4,
                "duration_sec": 2.2,
                "min_primary_angle": 35.0,
                "max_primary_angle": 167.0,
                "secondary_angle": 11.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Full stretch achieved.",
                "timestamp": (now - timedelta(days=2, minutes=33)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 5,
                "duration_sec": 1.8,
                "min_primary_angle": 65.0,
                "max_primary_angle": 162.0,
                "secondary_angle": 16.0,
                "is_correct": False,
                "issues": ["incomplete_flexion"],
                "feedback_cue": "Curl higher towards the shoulder.",
                "timestamp": (now - timedelta(days=2, minutes=32)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 6,
                "duration_sec": 2.3,
                "min_primary_angle": 37.0,
                "max_primary_angle": 166.0,
                "secondary_angle": 13.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Clean tempo.",
                "timestamp": (now - timedelta(days=2, minutes=30)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 7,
                "duration_sec": 2.0,
                "min_primary_angle": 40.0,
                "max_primary_angle": 142.0,
                "secondary_angle": 28.0,
                "is_correct": False,
                "issues": ["incomplete_extension", "elbow_drift"],
                "feedback_cue": "Keep elbows pinned to your ribs.",
                "timestamp": (now - timedelta(days=2, minutes=28)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 8,
                "duration_sec": 2.4,
                "min_primary_angle": 39.0,
                "max_primary_angle": 164.0,
                "secondary_angle": 14.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Controlled eccentric phase.",
                "timestamp": (now - timedelta(days=2, minutes=27)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 9,
                "duration_sec": 2.5,
                "min_primary_angle": 38.0,
                "max_primary_angle": 165.5,
                "secondary_angle": 15.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Strong squeeze.",
                "timestamp": (now - timedelta(days=2, minutes=26)).isoformat(),
            },
            {
                "exercise": "curl",
                "rep_number": 10,
                "duration_sec": 2.7,
                "min_primary_angle": 41.0,
                "max_primary_angle": 163.0,
                "secondary_angle": 16.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Great set.",
                "timestamp": (now - timedelta(days=2, minutes=25)).isoformat(),
            },
        ]
        for rep in curl_reps_data:
            self.log_rep(s2_id, rep)
        self.end_session(s2_id, end_time=s2_end)

        # -------------------------------------------------------------------
        # Session 3: Overhead Press Session (1 day ago)
        # -------------------------------------------------------------------
        s3_start = (now - timedelta(days=1, minutes=35)).isoformat()
        s3_end = (now - timedelta(days=1, minutes=20)).isoformat()
        s3_id = self.start_session(session_id="demo-pr3", start_time=s3_start)

        press_reps_data = [
            {
                "exercise": "press",
                "rep_number": 1,
                "duration_sec": 2.5,
                "min_primary_angle": 68.0,
                "max_primary_angle": 174.0,
                "secondary_angle": 175.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Full overhead lockout.",
                "timestamp": (now - timedelta(days=1, minutes=33)).isoformat(),
            },
            {
                "exercise": "press",
                "rep_number": 2,
                "duration_sec": 2.4,
                "min_primary_angle": 70.0,
                "max_primary_angle": 172.0,
                "secondary_angle": 173.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Good vertical bar path.",
                "timestamp": (now - timedelta(days=1, minutes=31)).isoformat(),
            },
            {
                "exercise": "press",
                "rep_number": 3,
                "duration_sec": 2.6,
                "min_primary_angle": 69.0,
                "max_primary_angle": 175.0,
                "secondary_angle": 176.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Strong press.",
                "timestamp": (now - timedelta(days=1, minutes=29)).isoformat(),
            },
            {
                "exercise": "press",
                "rep_number": 4,
                "duration_sec": 2.2,
                "min_primary_angle": 74.0,
                "max_primary_angle": 152.0,
                "secondary_angle": 165.0,
                "is_correct": False,
                "issues": ["incomplete_lockout"],
                "feedback_cue": "Lock your elbows out at the top of the press.",
                "timestamp": (now - timedelta(days=1, minutes=27)).isoformat(),
            },
            {
                "exercise": "press",
                "rep_number": 5,
                "duration_sec": 2.7,
                "min_primary_angle": 68.5,
                "max_primary_angle": 173.0,
                "secondary_angle": 174.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Elbows fully locked.",
                "timestamp": (now - timedelta(days=1, minutes=25)).isoformat(),
            },
            {
                "exercise": "press",
                "rep_number": 6,
                "duration_sec": 2.3,
                "min_primary_angle": 72.0,
                "max_primary_angle": 155.0,
                "secondary_angle": 148.0,
                "is_correct": False,
                "issues": ["incomplete_lockout", "excessive_lumbar_arch"],
                "feedback_cue": "Squeeze glutes and core to avoid overarching lower back.",
                "timestamp": (now - timedelta(days=1, minutes=23)).isoformat(),
            },
            {
                "exercise": "press",
                "rep_number": 7,
                "duration_sec": 2.9,
                "min_primary_angle": 71.0,
                "max_primary_angle": 171.5,
                "secondary_angle": 172.0,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Good torso tightness.",
                "timestamp": (now - timedelta(days=1, minutes=22)).isoformat(),
            },
            {
                "exercise": "press",
                "rep_number": 8,
                "duration_sec": 3.2,
                "min_primary_angle": 70.0,
                "max_primary_angle": 172.0,
                "secondary_angle": 173.5,
                "is_correct": True,
                "issues": [],
                "feedback_cue": "Pushed through fatigue to finish.",
                "timestamp": (now - timedelta(days=1, minutes=21)).isoformat(),
            },
        ]
        for rep in press_reps_data:
            self.log_rep(s3_id, rep)
        self.end_session(s3_id, end_time=s3_end)

    def export_session_json(self, session_id: str) -> str:
        """
        Returns JSON string representation of the session and its reps.

        Args:
            session_id (str): The session to export.

        Returns:
            str: Pretty-printed JSON string.
        """
        session_data = self.get_session(session_id)
        if not session_data:
            return json.dumps({"error": f"Session '{session_id}' not found."}, indent=2)
        return json.dumps(session_data, indent=2)

    def add_session(self, session_data: Dict[str, Any]) -> str:
        """
        Convenience method to save a session dict containing 'session' and 'reps'.

        Args:
            session_data (dict): Dict with 'session' info and 'reps' list.

        Returns:
            str: The session_id.
        """
        s_info = session_data.get("session", {})
        reps_list = session_data.get("reps", [])
        session_id = s_info.get("session_id") or uuid.uuid4().hex[:8]
        start_time = s_info.get("start_time") or datetime.now().isoformat()
        end_time = s_info.get("end_time") or datetime.now().isoformat()

        self.start_session(session_id=session_id, start_time=start_time)
        for rep in reps_list:
            self.log_rep(session_id=session_id, rep_data=rep)
        self.end_session(session_id=session_id, end_time=end_time)
        return session_id

    def get_reps(
        self, exercise: Optional[str] = None, days: Optional[int] = 30
    ) -> List[Dict[str, Any]]:
        """Alias for get_exercise_history."""
        return self.get_exercise_history(exercise=exercise, days=days or 30)

    def get_sessions(self, limit: int = 50, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Alias for get_recent_sessions."""
        return self.get_recent_sessions(limit=limit)

    def get_total_reps(self) -> int:
        """Returns all-time total reps counted."""
        return self.get_overall_stats()["total_reps"]

