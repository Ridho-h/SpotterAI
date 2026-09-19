"""
tests/test_evaluation.py
------------------------
Test suite for SpotterAI evaluation and benchmark suite.
Verifies rep counting F1 >= 0.90, form fault accuracy >= 0.85,
and markdown formatting.
"""

import pytest
from spotter.evaluation import (
    run_benchmark,
    format_benchmark_markdown,
    build_synthetic_dataset,
    generate_squat_rep,
    generate_bicep_curl_rep,
    generate_overhead_press_rep,
)
from spotter.tracker import RepTracker


class TestEvaluationSuite:

    def test_synthetic_dataset_composition(self):
        dataset = build_synthetic_dataset()
        assert len(dataset) >= 55
        exercises = {rep.exercise for rep in dataset}
        assert exercises == {'squat', 'curl', 'press'}

    def test_individual_generators(self):
        clean_squat = generate_squat_rep(min_knee_angle=90.0, knee_valgus=False)
        assert clean_squat.expected_faults == []

        shallow_squat = generate_squat_rep(min_knee_angle=115.0, knee_valgus=False)
        assert shallow_squat.expected_faults == ['incomplete_depth']

        valgus_squat = generate_squat_rep(min_knee_angle=90.0, knee_valgus=True)
        assert valgus_squat.expected_faults == ['knees_caving_in']

        clean_curl = generate_bicep_curl_rep(min_flexion_angle=25.0, return_angle=160.0)
        assert clean_curl.expected_faults == []

        partial_curl = generate_bicep_curl_rep(min_flexion_angle=60.0, return_angle=160.0)
        assert partial_curl.expected_faults == ['incomplete_curl']

        unextended_curl = generate_bicep_curl_rep(min_flexion_angle=25.0, return_angle=120.0)
        assert unextended_curl.expected_faults == ['incomplete_extension']

        clean_press = generate_overhead_press_rep(max_lockout_angle=165.0)
        assert clean_press.expected_faults == []

        incomplete_press = generate_overhead_press_rep(max_lockout_angle=135.0)
        assert incomplete_press.expected_faults == ['incomplete_lockout']

    def test_run_benchmark_metrics(self):
        results = run_benchmark()

        assert 'rep_precision' in results
        assert 'rep_recall' in results
        assert 'rep_f1' in results
        assert 'form_fault_accuracy' in results
        assert 'total_evaluated_reps' in results
        assert 'breakdown_by_exercise' in results

        # Critical accuracy benchmarks required
        assert results['rep_f1'] >= 0.90, f"rep_f1 was {results['rep_f1']:.3f}, expected >= 0.90"
        assert results['form_fault_accuracy'] >= 0.85, (
            f"form_fault_accuracy was {results['form_fault_accuracy']:.3f}, expected >= 0.85"
        )

    def test_format_benchmark_markdown(self):
        results = run_benchmark()
        md = format_benchmark_markdown(results)

        assert isinstance(md, str)
        assert len(md) > 100
        assert "| Metric | Value | Target | Status |" in md
        assert "| Exercise | Tested Reps | Detected Reps |" in md
        assert "incomplete_depth" in md
        assert "knees_caving_in" in md
        assert "incomplete_curl" in md
        assert "incomplete_lockout" in md
