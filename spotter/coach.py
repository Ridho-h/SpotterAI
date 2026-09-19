"""
spotter/coach.py
----------------
Post-Workout Coaching Agent for SpotterAI.
Performs deterministic rule-based telemetry analysis grounded in exact joint angles
and repetition counts, with optional LLM enhancement via Gemini or OpenAI.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


class IssueDict(dict):
    """
    Dictionary subclass representing a form fault issue.
    Allows dict key access (e.g. issue['count']) and stringifies to its
    formatted human-readable description (e.g. str(issue)).
    """

    def __str__(self) -> str:
        return self.get("description", super().__str__())

    def __repr__(self) -> str:
        return f"IssueDict({super().__repr__()})"


class CoachingAgent:
    """
    Post-workout coaching debrief generator.

    Calculates form scores, grounded positive achievements, recurring fault telemetry,
    and targeted biomechanical cues. Can optionally enhance reports with an LLM.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.api_key = api_key
        self.provider = provider.lower() if provider else "auto"

        # Resolve API keys from environment if not explicitly provided
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if self.provider == "auto":
            if self.api_key:
                if self.api_key.startswith("sk-"):
                    self.provider = "openai"
                else:
                    self.provider = "gemini"
            elif gemini_key:
                self.provider = "gemini"
                self.api_key = gemini_key
            elif openai_key:
                self.provider = "openai"
                self.api_key = openai_key
            else:
                self.provider = None
        elif self.provider == "gemini":
            self.api_key = self.api_key or gemini_key
        elif self.provider == "openai":
            self.api_key = self.api_key or openai_key
        else:
            self.provider = None

    def generate_session_feedback(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes workout session telemetry and produces deterministic feedback
        with optional LLM enhancement.

        Args:
            session_data: Dict containing:
                - 'session': Dict with total_reps, good_reps, faulty_reps, accuracy_pct, start_time
                - 'reps': List of rep dicts with exercise, rep_number, is_correct, issues,
                          min_primary_angle, max_primary_angle, feedback_cue

        Returns:
            dict with keys:
                - 'score': int (0-100)
                - 'markdown_report': str
                - 'strengths': list[str]
                - 'issues': list[dict]
                - 'cues': list[str]
        """
        session_info = session_data.get("session", {})
        reps_list = session_data.get("reps", [])

        # ------------------------------------------------------------------
        # 1. Metric Normalization & Form Score
        # ------------------------------------------------------------------
        total_reps = session_info.get("total_reps", len(reps_list))
        good_reps = session_info.get(
            "good_reps", sum(1 for r in reps_list if r.get("is_correct"))
        )
        faulty_reps = session_info.get("faulty_reps", max(0, total_reps - good_reps))

        if total_reps > 0:
            raw_acc = session_info.get("accuracy_pct", (good_reps / total_reps) * 100.0)
        else:
            raw_acc = 100.0

        form_score = max(0, min(100, int(round(raw_acc))))

        # Group reps by exercise
        exercise_reps: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in reps_list:
            ex = str(r.get("exercise", "exercise")).lower()
            exercise_reps[ex].append(r)

        # ------------------------------------------------------------------
        # 2. Summary
        # ------------------------------------------------------------------
        ex_breakdown = ", ".join(
            f"{len(reps)} {ex}s" for ex, reps in exercise_reps.items()
        )
        if not ex_breakdown:
            ex_breakdown = f"{total_reps} exercises"

        summary = (
            f"Session completed with {total_reps} total reps ({good_reps} clean, "
            f"{faulty_reps} faulty) across {ex_breakdown}, achieving an overall "
            f"form score of {form_score}%."
        )

        # ------------------------------------------------------------------
        # 3. What Went Well (Strengths) - Grounded in exact numbers
        # ------------------------------------------------------------------
        strengths: List[str] = []
        for ex, reps in exercise_reps.items():
            tot_ex = len(reps)
            good_ex = sum(1 for r in reps if r.get("is_correct"))
            acc_ex = (good_ex / tot_ex * 100.0) if tot_ex > 0 else 0.0

            if tot_ex > 0 and good_ex == tot_ex:
                strengths.append(
                    f"Flawless execution on {ex}s: 100% form compliance across all {tot_ex} reps."
                )
            elif good_ex > 0:
                strengths.append(
                    f"Completed {tot_ex} {ex}s with {acc_ex:.0f}% good form compliance "
                    f"({good_ex}/{tot_ex} clean reps)."
                )

            # Biomechanical depth/extension telemetry highlights
            if "squat" in ex:
                clean_squats = [
                    r["min_primary_angle"]
                    for r in reps
                    if r.get("is_correct") and r.get("min_primary_angle") is not None
                ]
                if clean_squats:
                    avg_clean_depth = sum(clean_squats) / len(clean_squats)
                    strengths.append(
                        f"Achieved excellent squat depth on {len(clean_squats)} clean reps "
                        f"(average bottom angle {avg_clean_depth:.0f}° vs target <=100°)."
                    )

            if "curl" in ex:
                clean_curls = [
                    r
                    for r in reps
                    if r.get("is_correct")
                    and (r.get("max_primary_angle") is not None or r.get("min_primary_angle") is not None)
                ]
                if clean_curls:
                    strengths.append(
                        f"Pacing was consistent across {len(clean_curls)} curls with full range of motion."
                    )

        if total_reps > 0 and not strengths:
            strengths.append(
                f"Maintained high training volume by completing {total_reps} total repetitions."
            )

        # ------------------------------------------------------------------
        # 4. Recurring Issues - Exact counts and joint angles
        # ------------------------------------------------------------------
        issues: List[Dict[str, Any]] = []

        for ex, reps in exercise_reps.items():
            tot_ex = len(reps)
            fault_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            for r in reps:
                if not r.get("is_correct"):
                    rep_issues = r.get("issues", [])
                    if isinstance(rep_issues, str):
                        rep_issues = [rep_issues]
                    elif not rep_issues:
                        rep_issues = ["form fault"]

                    for issue_name in rep_issues:
                        norm_name = str(issue_name).strip()
                        if norm_name:
                            fault_buckets[norm_name].append(r)

            for issue_name, faulty_subset in fault_buckets.items():
                count = len(faulty_subset)
                low_name = issue_name.lower()

                # Biomechanical angle analysis
                if "squat" in ex or "depth" in low_name:
                    angles = [
                        r["min_primary_angle"]
                        for r in faulty_subset
                        if r.get("min_primary_angle") is not None
                    ]
                    if angles:
                        avg_angle = sum(angles) / len(angles)
                        desc = (
                            f"{count} of {tot_ex} {ex}s had incomplete depth "
                            f"(average bottom angle {avg_angle:.0f}° vs target <=100°)"
                        )
                    else:
                        avg_angle = None
                        desc = f"{count} of {tot_ex} {ex}s had incomplete depth (target <=100°)"
                    target_str = "<=100°"

                elif "curl" in ex or "extension" in low_name:
                    angles = [
                        r["max_primary_angle"]
                        if r.get("max_primary_angle") is not None
                        else r.get("min_primary_angle")
                        for r in faulty_subset
                        if (r.get("max_primary_angle") is not None or r.get("min_primary_angle") is not None)
                    ]
                    if angles:
                        avg_angle = sum(angles) / len(angles)
                        desc = f"{count} curls missed full extension at bottom (<140°)"
                    else:
                        avg_angle = None
                        desc = f"{count} curls missed full extension at bottom (<140°)"
                    target_str = ">140°"

                else:
                    angles = [
                        r["min_primary_angle"]
                        for r in faulty_subset
                        if r.get("min_primary_angle") is not None
                    ]
                    if angles:
                        avg_angle = sum(angles) / len(angles)
                        desc = (
                            f"{count} of {tot_ex} {ex}s had {issue_name} "
                            f"(average recorded angle {avg_angle:.0f}°)"
                        )
                    else:
                        avg_angle = None
                        desc = f"{count} of {tot_ex} {ex}s had {issue_name}"
                    target_str = None

                issue_dict = IssueDict(
                    {
                        "exercise": ex,
                        "issue": issue_name,
                        "count": count,
                        "total": tot_ex,
                        "avg_angle": avg_angle,
                        "target_angle": target_str,
                        "description": desc,
                    }
                )
                issues.append(issue_dict)

        # ------------------------------------------------------------------
        # 5. Actionable Biomechanical Cues
        # ------------------------------------------------------------------
        cues: List[str] = []
        has_squat_depth_issue = any(
            "depth" in i.get("issue", "").lower() or "depth" in i.get("description", "").lower()
            for i in issues
        )
        has_curl_ext_issue = any(
            "extension" in i.get("issue", "").lower() or "extension" in i.get("description", "").lower()
            for i in issues
        )
        has_valgus_issue = any(
            "valgus" in i.get("issue", "").lower() or "cave" in i.get("issue", "").lower()
            for i in issues
        )
        has_press_issue = any("press" in i.get("exercise", "").lower() for i in issues)

        # Populate prioritized cues
        if has_squat_depth_issue:
            cues.append(
                "Squat Cue: Sit hips down between heels until hip crease dips below knee line (aim for <=100° knee angle)."
            )
        if has_valgus_issue or (has_squat_depth_issue and len(cues) < 2):
            cues.append(
                "Push knees slightly outward over your toes to prevent valgus cave and stabilize your base."
            )
        if has_curl_ext_issue:
            cues.append(
                "Bicep Curl Cue: Fully extend elbows at the bottom of each rep (>140°) before initiating the next contraction."
            )
        if has_press_issue:
            cues.append(
                "Overhead Press Cue: Lock out overhead with biceps aligned with your ears and avoid hyperextending your lumbar spine."
            )

        # Incorporate explicit feedback cues from telemetry if present
        for r in reps_list:
            if not r.get("is_correct") and r.get("feedback_cue"):
                fc = r["feedback_cue"].strip()
                if fc and fc not in cues and len(cues) < 2:
                    cues.append(fc)

        if not cues:
            cues.append(
                "Great consistency! Maintain core tension and controlled eccentric tempo on every repetition."
            )

        # Keep 1-2 targeted cues
        cues = cues[:2]

        # ------------------------------------------------------------------
        # 6. Generate Deterministic Markdown Report
        # ------------------------------------------------------------------
        strengths_md = "\n".join(f"- {s}" for s in strengths) if strengths else "- Consistent effort recorded."
        if issues:
            issues_md = "\n".join(f"- {i['description']}" for i in issues)
        else:
            issues_md = "- None! Outstanding mechanics with zero recorded form deviations."
        cues_md = "\n".join(f"{idx + 1}. {c}" for idx, c in enumerate(cues))

        deterministic_report = (
            f"# 🏋️ Post-Workout Coaching Debrief\n\n"
            f"### 📊 Overall Form Score: **{form_score}/100**\n\n"
            f"**Summary:** {summary}\n\n"
            f"### ✅ What Went Well\n"
            f"{strengths_md}\n\n"
            f"### ⚠️ Recurring Form Issues\n"
            f"{issues_md}\n\n"
            f"### 🎯 Actionable Biomechanical Cues for Next Session\n"
            f"{cues_md}"
        )

        final_report = deterministic_report

        # ------------------------------------------------------------------
        # 7. Optional LLM Enhancement
        # ------------------------------------------------------------------
        if self.api_key and self.provider:
            try:
                llm_prompt = (
                    "You are an elite AI Strength and Conditioning Coach reviewing an athlete's workout telemetry. "
                    "Based ONLY on the verified telemetry facts below (do not hallucinate other exercises or numbers), "
                    "provide a motivating, highly specific coaching debrief formatted in clean Markdown.\n\n"
                    f"Telemetry Facts:\n"
                    f"- Form Score: {form_score}/100\n"
                    f"- Summary: {summary}\n"
                    f"- Verified Strengths:\n{strengths_md}\n"
                    f"- Recurring Issues:\n{issues_md}\n"
                    f"- Actionable Cues:\n{cues_md}\n"
                )
                enhanced_text = self._call_llm(llm_prompt)
                if enhanced_text and len(enhanced_text.strip()) > 50:
                    final_report = enhanced_text.strip()
            except Exception:
                # Clean fallback to deterministic report on any failure or timeout
                final_report = deterministic_report

        return {
            "score": form_score,
            "markdown_report": final_report,
            "strengths": strengths,
            "issues": issues,
            "cues": cues,
        }

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Calls Gemini or OpenAI based on configured provider."""
        if self.provider == "gemini":
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

            # Fallback to SDK
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
            return res.choices[0].message.content

        return None
