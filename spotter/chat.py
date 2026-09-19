"""
spotter/chat.py
---------------
Workout History Chat Assistant for SpotterAI.
Interprets natural-language questions and queries the workout database to provide
grounded, deterministic telemetry answers with optional LLM conversational polishing.
"""

import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from spotter.db import WorkoutDatabase


class WorkoutChatAssistant:
    """
    Conversational assistant for querying historical workout telemetry.

    Capable of calculating:
      - Squat depth trends and multi-session angle progression.
      - Exercise fault rankings and primary issue identification.
      - Session and all-time repetition statistics.
      - Exercise-specific form compliance breakdowns.
    """

    def __init__(self, db: Any, api_key: Optional[str] = None):
        self.db = db
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.provider = "gemini" if (self.api_key and not self.api_key.startswith("sk-")) else "openai" if self.api_key else None

    # ------------------------------------------------------------------
    # Data Access Helpers (Handles real WorkoutDatabase or custom mocks)
    # ------------------------------------------------------------------

    def _fetch_reps(
        self, exercise: Optional[str] = None, days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Safely fetches reps from WorkoutDatabase or mock object."""
        reps = []
        if hasattr(self.db, "get_reps"):
            try:
                reps = self.db.get_reps(exercise=exercise, days=days)
            except TypeError:
                try:
                    reps = self.db.get_reps(exercise=exercise)
                except TypeError:
                    reps = self.db.get_reps()
        elif hasattr(self.db, "get_exercise_history"):
            try:
                reps = self.db.get_exercise_history(exercise=exercise, days=days or 30)
            except TypeError:
                reps = self.db.get_exercise_history()
        elif hasattr(self.db, "reps"):
            reps = list(self.db.reps)
        elif hasattr(self.db, "execute"):
            cursor = self.db.execute("SELECT * FROM reps")
            reps = [dict(r) for r in cursor.fetchall()]

        if exercise and reps:
            reps = [
                r for r in reps
                if str(r.get("exercise", "")).lower() == exercise.lower()
            ]
        return reps

    def _fetch_sessions(self, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Safely fetches sessions from WorkoutDatabase or mock object."""
        if hasattr(self.db, "get_sessions"):
            try:
                return self.db.get_sessions(days=days)
            except TypeError:
                return self.db.get_sessions()
        elif hasattr(self.db, "sessions"):
            return list(self.db.sessions)
        elif hasattr(self.db, "execute"):
            cursor = self.db.execute("SELECT * FROM sessions")
            return [dict(r) for r in cursor.fetchall()]
        return []

    def _fetch_total_reps(self) -> int:
        """Calculates or fetches total all-time reps."""
        if hasattr(self.db, "get_total_reps"):
            return self.db.get_total_reps()
        reps = self._fetch_reps()
        return len(reps)

    # ------------------------------------------------------------------
    # Public Query Interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """
        Analyzes the user's question, retrieves relevant database records,
        and generates a Markdown response.

        Args:
            question (str): User's natural-language query.

        Returns:
            str: Markdown response with metrics, telemetry facts, and coaching advice.
        """
        q_lower = question.lower()

        # 1. Squat depth trend inquiry
        if "squat" in q_lower and any(w in q_lower for w in ["depth", "trend", "change", "improve", "angle"]):
            deterministic_answer = self._handle_squat_depth_trend(q_lower)

        # 2. Exercise fault ranking / most form issues
        elif any(w in q_lower for w in ["most", "worst", "highest", "rank"]) and any(
            w in q_lower for w in ["issue", "fault", "problem", "mistake", "error"]
        ):
            deterministic_answer = self._handle_most_form_issues()

        # 3. Total reps / all-time count
        elif ("how many" in q_lower or "total" in q_lower or "count" in q_lower) and "rep" in q_lower:
            deterministic_answer = self._handle_total_reps()

        # 4. Today / recent workouts stats
        elif any(w in q_lower for w in ["today", "recent", "stats", "history", "last workout", "summary"]):
            deterministic_answer = self._handle_recent_stats(q_lower)

        # 5. Exercise specific form breakdown
        elif any(ex in q_lower for ex in ["curl", "press", "squat"]):
            deterministic_answer = self._handle_exercise_form(q_lower)

        # 6. Fallback general stats
        else:
            deterministic_answer = self._handle_fallback_overview()

        # Optional LLM polish if API key is active
        if self.api_key:
            enhanced = self._enhance_with_llm(question, deterministic_answer)
            if enhanced:
                return enhanced

        return deterministic_answer

    # ------------------------------------------------------------------
    # Deterministic Query Handlers
    # ------------------------------------------------------------------

    def _handle_squat_depth_trend(self, q: str) -> str:
        """Analyzes squat depth trend across sessions / days."""
        reps = self._fetch_reps(exercise="squat", days=7)
        if not reps:
            # Fallback to all squat reps if none in 7 days
            reps = self._fetch_reps(exercise="squat")

        if not reps:
            return "No squat repetitions were found in your workout history."

        # Group by day or session
        grouped: Dict[str, List[float]] = defaultdict(list)
        for r in reps:
            angle = r.get("min_primary_angle")
            if angle is not None:
                ts = r.get("timestamp") or r.get("session_id") or "Session"
                # Extract day name or short date if ISO format
                try:
                    dt = datetime.fromisoformat(str(ts))
                    label = dt.strftime("%A")  # e.g. "Monday", "Wednesday"
                except Exception:
                    label = str(ts)
                grouped[label].append(float(angle))

        if not grouped:
            return (
                "Found squat repetitions, but no joint angle telemetry was recorded for depth analysis."
            )

        daily_avgs = [
            {"label": label, "avg_depth": sum(angles) / len(angles), "reps": len(angles)}
            for label, angles in grouped.items()
        ]

        if len(daily_avgs) >= 2:
            earlier = daily_avgs[0]
            later = daily_avgs[-1]
            diff = earlier["avg_depth"] - later["avg_depth"]

            if diff > 0:
                trend_msg = (
                    f"On {earlier['label']} your average bottom knee angle was {earlier['avg_depth']:.0f}°, "
                    f"and on {later['label']} it improved to {later['avg_depth']:.0f}° - an improvement "
                    f"of {abs(diff):.0f}° towards proper depth!"
                )
            elif diff < 0:
                trend_msg = (
                    f"On {earlier['label']} your average bottom knee angle was {earlier['avg_depth']:.0f}°, "
                    f"while on {later['label']} it averaged {later['avg_depth']:.0f}° (a difference of {abs(diff):.0f}°)."
                )
            else:
                trend_msg = (
                    f"Your squat depth remained steady at an average of {earlier['avg_depth']:.0f}° "
                    f"between {earlier['label']} and {later['label']}."
                )
        else:
            single = daily_avgs[0]
            trend_msg = (
                f"Recorded an average bottom knee angle of {single['avg_depth']:.0f}° "
                f"across {single['reps']} squats on {single['label']}."
            )

        breakdown_lines = [
            f"- **{item['label']}**: {item['avg_depth']:.1f}° avg bottom angle ({item['reps']} reps)"
            for item in daily_avgs
        ]

        return (
            f"### 🏋️ Squat Depth Trend Analysis\n\n"
            f"{trend_msg}\n\n"
            f"**Session Breakdown:**\n"
            + "\n".join(breakdown_lines)
            + "\n\n"
            f"💡 **Coaching Cue:** Target a bottom knee angle of **<=100°** (hip crease below parallel) "
            f"while keeping your chest upright and heels grounded."
        )

    def _handle_most_form_issues(self) -> str:
        """Ranks exercises by form faults and identifies top issue."""
        reps = self._fetch_reps()
        if not reps:
            return "No workout repetitions recorded yet."

        exercise_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "faulty": 0, "issues": defaultdict(int)}
        )

        for r in reps:
            ex = str(r.get("exercise", "unknown")).capitalize()
            exercise_data[ex]["total"] += 1
            if not r.get("is_correct"):
                exercise_data[ex]["faulty"] += 1
                issues_list = r.get("issues", [])
                if isinstance(issues_list, str):
                    issues_list = [issues_list]
                for issue in issues_list:
                    if issue:
                        exercise_data[ex]["issues"][issue] += 1

        # Sort exercises by number of faults descending
        ranked = sorted(
            exercise_data.items(),
            key=lambda item: (item[1]["faulty"], item[1]["faulty"] / item[1]["total"] if item[1]["total"] else 0),
            reverse=True,
        )

        if not ranked or ranked[0][1]["faulty"] == 0:
            return "🎉 Excellent news! Zero form issues have been recorded across all your exercises."

        top_ex, top_stats = ranked[0]
        top_issue = "form deviation"
        if top_stats["issues"]:
            top_issue = max(top_stats["issues"].items(), key=lambda x: x[1])[0]

        lines = []
        for rank, (ex, stats) in enumerate(ranked, start=1):
            fault_pct = (stats["faulty"] / stats["total"] * 100) if stats["total"] else 0
            primary_fault = (
                max(stats["issues"].items(), key=lambda x: x[1])[0]
                if stats["issues"]
                else "None"
            )
            lines.append(
                f"{rank}. **{ex}**: {stats['faulty']} faults / {stats['total']} reps "
                f"({fault_pct:.1f}% fault rate) — *Top issue: {primary_fault}*"
            )

        return (
            f"### ⚠️ Form Issue Ranking\n\n"
            f"**{top_ex}** currently has the most form issues with **{top_stats['faulty']} faults** "
            f"out of {top_stats['total']} reps ({top_stats['faulty'] / top_stats['total'] * 100:.1f}% fault rate). "
            f"The primary culprit is **{top_issue}**.\n\n"
            f"**Full Exercise Breakdown:**\n"
            + "\n".join(lines)
            + "\n\n"
            f"💡 **Recommendation:** Prioritize warm-up drills addressing **{top_issue}** on your next {top_ex.lower()} session."
        )

    def _handle_total_reps(self) -> str:
        """Returns total reps completed with exercise breakdown."""
        reps = self._fetch_reps()
        total_count = len(reps)
        if total_count == 0:
            return "You haven't completed any reps yet. Start a session to log your progress!"

        clean_count = sum(1 for r in reps if r.get("is_correct"))
        fault_count = total_count - clean_count
        overall_acc = (clean_count / total_count * 100) if total_count > 0 else 0

        counts_by_ex: Dict[str, int] = defaultdict(int)
        for r in reps:
            ex = str(r.get("exercise", "exercise")).capitalize()
            counts_by_ex[ex] += 1

        ex_lines = [f"- **{ex}**: {count} reps" for ex, count in sorted(counts_by_ex.items())]

        return (
            f"### 📊 All-Time Repetition Count\n\n"
            f"You have completed **{total_count} total repetitions** across all sessions!\n\n"
            f"- **Clean Reps:** {clean_count} ({overall_acc:.1f}% form accuracy)\n"
            f"- **Reps with Form Faults:** {fault_count}\n\n"
            f"**Exercise Breakdown:**\n"
            + "\n".join(ex_lines)
            + "\n\n"
            f"Keep up the dedication! 🚀"
        )

    def _handle_recent_stats(self, q: str) -> str:
        """Returns stats for today or recent sessions."""
        days = 1 if "today" in q else 7
        sessions = self._fetch_sessions(days=days)
        reps = self._fetch_reps(days=days)

        if not reps:
            # Fallback to all sessions if none found in timeframe
            sessions = self._fetch_sessions()
            reps = self._fetch_reps()

        if not reps:
            return "No recent workouts logged. Complete a session to view your stats!"

        tot = len(reps)
        good = sum(1 for r in reps if r.get("is_correct"))
        acc = (good / tot * 100) if tot > 0 else 0

        ex_breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "good": 0})
        for r in reps:
            ex = str(r.get("exercise", "exercise")).capitalize()
            ex_breakdown[ex]["total"] += 1
            if r.get("is_correct"):
                ex_breakdown[ex]["good"] += 1

        breakdown_lines = [
            f"- **{ex}**: {data['total']} reps ({data['good']}/{data['total']} clean, "
            f"{data['good'] / data['total'] * 100:.0f}% accuracy)"
            for ex, data in sorted(ex_breakdown.items())
        ]

        title = "Today's Workout Stats" if "today" in q else "Recent Workout Stats"
        return (
            f"### 📈 {title}\n\n"
            f"- **Total Reps Completed:** {tot}\n"
            f"- **Clean Reps:** {good}\n"
            f"- **Form Accuracy:** {acc:.1f}%\n\n"
            f"**Exercise Breakdown:**\n"
            + "\n".join(breakdown_lines)
            + "\n\n"
            f"Consistency is key — keep training with mindful technique!"
        )

    def _handle_exercise_form(self, q: str) -> str:
        """Analyzes form for a specific exercise."""
        target_ex = "squat" if "squat" in q else "press" if "press" in q else "curl"
        reps = self._fetch_reps(exercise=target_ex)

        if not reps:
            return f"No {target_ex} repetitions recorded in your workout history."

        tot = len(reps)
        good = sum(1 for r in reps if r.get("is_correct"))
        acc = (good / tot * 100) if tot > 0 else 0

        faults: Dict[str, int] = defaultdict(int)
        for r in reps:
            if not r.get("is_correct"):
                for issue in r.get("issues", []):
                    if issue:
                        faults[issue] += 1

        top_fault_str = "None! Flawless technique."
        if faults:
            top_fault = max(faults.items(), key=lambda x: x[1])
            top_fault_str = f"{top_fault[0]} ({top_fault[1]} times)"

        if target_ex == "curl":
            cue = "Keep elbows locked by your side and fully extend arms at the bottom (>140°) before curling."
        elif target_ex == "squat":
            cue = "Maintain an upright torso and reach full depth (<=100° knee angle) while driving knees outward."
        else:
            cue = "Brace core to protect lower back and lock arms overhead directly above shoulders."

        return (
            f"### 🏋️ {target_ex.capitalize()} Form Analysis\n\n"
            f"- **Total Reps:** {tot}\n"
            f"- **Clean Reps:** {good}\n"
            f"- **Form Accuracy:** {acc:.1f}%\n"
            f"- **Primary Form Issue:** {top_fault_str}\n\n"
            f"💡 **Coaching Tip:** {cue}"
        )

    def _handle_fallback_overview(self) -> str:
        """General overview when question does not match specific intent."""
        tot = self._fetch_total_reps()
        return (
            f"### 🤖 SpotterAI Workout History Assistant\n\n"
            f"You have logged a total of **{tot} reps** in your training journey.\n\n"
            f"Here are a few questions you can ask me:\n"
            f"- *'How has my squat depth changed this week?'*\n"
            f"- *'Which exercise has the most form issues?'*\n"
            f"- *'Show my stats for today'* or *'Show recent workouts'*\n"
            f"- *'How many reps have I done?'*\n"
            f"- *'How is my bicep curl form?'*"
        )

    # ------------------------------------------------------------------
    # LLM Enhancement
    # ------------------------------------------------------------------

    def _enhance_with_llm(self, question: str, telemetry_context: str) -> Optional[str]:
        """Optionally enhances grounded factual response with conversational LLM."""
        prompt = (
            "You are SpotterAI's Workout History Assistant. Answer the athlete's question "
            "conversational, engagingly, and accurately using ONLY the following grounded workout database facts.\n"
            "Never hallucinate numbers, reps, or exercises not in the facts.\n\n"
            f"Athlete Question: {question}\n\n"
            f"Verified Grounding Facts:\n{telemetry_context}\n"
        )
        try:
            if self.provider == "gemini":
                # 1. Direct REST call (fastest, lightweight, zero protobuf conflicts)
                models = [
                    "gemini-flash-latest",
                    "gemini-2.5-flash",
                    "gemini-flash-lite-latest",
                    "gemini-1.5-flash",
                    "gemini-pro",
                ]
                import requests

                for model_name in models:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
                        payload = {"contents": [{"parts": [{"text": prompt}]}]}
                        resp = requests.post(url, json=payload, timeout=12)
                        if resp.status_code == 200:
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text")
                                if text:
                                    return text.strip()
                    except Exception:
                        continue

                # 2. SDK fallback
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=self.api_key)
                    for model_name in models:
                        try:
                            model = genai.GenerativeModel(model_name)
                            res = model.generate_content(prompt)
                            if res and res.text:
                                return res.text.strip()
                        except Exception:
                            continue
                except Exception:
                    pass
            elif self.provider == "openai":
                import openai

                client = openai.OpenAI(api_key=self.api_key)
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return res.choices[0].message.content.strip()
        except Exception:
            return None
        return None
