"""
Tests for observability, curriculum learning, auth, and leaderboard (Phase 11-12).
"""

import os
import pytest
import tempfile
from pathlib import Path

from environment.observability import (
    PerformanceTracker,
    MetricsCollector,
    LLMTracer,
)
from environment.curriculum import CurriculumConfig, CurriculumManager
from environment.auth import InputSanitizer, RateLimitTracker
from environment.leaderboard_db import LeaderboardDB


# ── Observability Tests ──────────────────────────────────────────

class TestPerformanceTracker:
    def test_record_step(self):
        tracker = PerformanceTracker("sess-1", "invoice_match")
        tracker.record_step(1, "match_invoice", 3.5, 0.05, False)
        tracker.record_step(2, "submit_report", 5.0, 0.8, True)
        tracker.set_final_score(0.85)

        metrics = tracker.get_episode_metrics()
        assert metrics.total_steps == 2
        assert metrics.final_score == 0.85
        assert metrics.total_latency_ms == 8.5
        assert metrics.avg_step_latency_ms == 4.25

    def test_error_counting(self):
        tracker = PerformanceTracker("sess-2", "itc_audit")
        tracker.record_step(1, "bad_action", 1.0, 0.0, False, error="Invalid")
        tracker.record_step(2, "match_invoice", 2.0, 0.05, False)
        metrics = tracker.get_episode_metrics()
        assert metrics.errors == 1


class TestMetricsCollector:
    def test_record_and_summarize(self):
        collector = MetricsCollector()
        tracker1 = PerformanceTracker("a", "invoice_match")
        tracker1.set_final_score(0.9)
        collector.record_episode(tracker1.get_episode_metrics())

        tracker2 = PerformanceTracker("b", "itc_audit")
        tracker2.set_final_score(0.75)
        collector.record_episode(tracker2.get_episode_metrics())

        summary = collector.get_summary()
        assert summary["total_episodes"] == 2
        assert "invoice_match" in summary["avg_scores_by_task"]
        assert "itc_audit" in summary["avg_scores_by_task"]


class TestLLMTracer:
    def test_trace_call(self):
        tracer = LLMTracer()
        tracer.trace_llm_call("gpt-4", 500, 200, 1500.0)
        tracer.trace_llm_call("gpt-3.5-turbo", 300, 100, 400.0)

        summary = tracer.get_summary()
        assert summary["total_calls"] == 2
        assert summary["total_input_tokens"] == 800
        assert summary["total_output_tokens"] == 300
        assert summary["total_cost_usd"] > 0


# ── Curriculum Tests ─────────────────────────────────────────────

class TestCurriculumManager:
    def test_starts_at_easy(self):
        cm = CurriculumManager()
        assert cm.get_current_difficulty() == "easy"
        assert cm.get_next_task() == "invoice_match"

    def test_advances_on_high_score(self):
        config = CurriculumConfig(easy_threshold=0.8, min_episodes=1)
        cm = CurriculumManager(config)
        result = cm.record_score("invoice_match", 0.9)
        assert result["advanced"] is True
        assert cm.get_current_difficulty() == "medium"

    def test_stays_on_low_score(self):
        config = CurriculumConfig(easy_threshold=0.8, min_episodes=1)
        cm = CurriculumManager(config)
        result = cm.record_score("invoice_match", 0.5)
        assert result["advanced"] is False
        assert cm.get_current_difficulty() == "easy"

    def test_full_progression(self):
        config = CurriculumConfig(
            easy_threshold=0.7, medium_threshold=0.7, hard_threshold=0.7,
            min_episodes=1,
        )
        cm = CurriculumManager(config)
        cm.record_score("invoice_match", 0.8)
        assert cm.get_current_difficulty() == "medium"
        cm.record_score("itc_audit", 0.8)
        assert cm.get_current_difficulty() == "hard"

    def test_mastery(self):
        config = CurriculumConfig(
            easy_threshold=0.5, medium_threshold=0.5, hard_threshold=0.5,
            min_episodes=1,
        )
        cm = CurriculumManager(config)
        cm.record_score("invoice_match", 0.9)
        cm.record_score("itc_audit", 0.9)
        cm.record_score("full_recon", 0.9)
        assert cm.is_mastered() is True

    def test_auto_advance_disabled(self):
        config = CurriculumConfig(auto_advance=False)
        cm = CurriculumManager(config)
        cm.record_score("invoice_match", 0.99)
        assert cm.get_current_difficulty() == "easy"  # Should not advance

    def test_summary(self):
        cm = CurriculumManager()
        cm.record_score("invoice_match", 0.5)
        summary = cm.get_summary()
        assert "current_level" in summary
        assert "score_history" in summary


# ── Auth Tests ───────────────────────────────────────────────────

class TestInputSanitizer:
    def test_valid_action(self):
        action = InputSanitizer.validate_action({"action_type": "match_invoice", "invoice_id": "INV-0001"})
        assert action["action_type"] == "match_invoice"

    def test_invalid_action_type(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_action({"action_type": "hack_system"})

    def test_long_invoice_id_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_action({"action_type": "match_invoice", "invoice_id": "A" * 100})

    def test_valid_task_id(self):
        assert InputSanitizer.validate_task_id("invoice_match") == "invoice_match"

    def test_invalid_task_id(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_task_id("drop_table")


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimitTracker(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.check("1.2.3.4") is True

    def test_blocks_over_limit(self):
        rl = RateLimitTracker(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check("1.2.3.4")
        assert rl.check("1.2.3.4") is False

    def test_different_ips_independent(self):
        rl = RateLimitTracker(max_requests=2, window_seconds=60)
        rl.check("1.1.1.1")
        rl.check("1.1.1.1")
        assert rl.check("2.2.2.2") is True

    def test_remaining_count(self):
        rl = RateLimitTracker(max_requests=10, window_seconds=60)
        rl.check("1.1.1.1")
        assert rl.get_remaining("1.1.1.1") == 9


# ── Leaderboard DB Tests ────────────────────────────────────────

class TestLeaderboardDB:
    def test_add_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = LeaderboardDB(db_path)
            eid = db.add_entry("sess-1", "invoice_match", 0.85, steps=5)
            assert eid >= 1

            top = db.get_top(10)
            assert len(top) == 1
            assert top[0]["score"] == 0.85
        finally:
            os.unlink(db_path)

    def test_ordering(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = LeaderboardDB(db_path)
            db.add_entry("a", "invoice_match", 0.5)
            db.add_entry("b", "invoice_match", 0.9)
            db.add_entry("c", "invoice_match", 0.7)

            top = db.get_top(10)
            assert top[0]["score"] == 0.9
            assert top[1]["score"] == 0.7
            assert top[2]["score"] == 0.5
        finally:
            os.unlink(db_path)

    def test_filter_by_task(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = LeaderboardDB(db_path)
            db.add_entry("a", "invoice_match", 0.8)
            db.add_entry("b", "itc_audit", 0.7)

            match_only = db.get_top(10, task_id="invoice_match")
            assert len(match_only) == 1
            assert match_only[0]["task_id"] == "invoice_match"
        finally:
            os.unlink(db_path)

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = LeaderboardDB(db_path)
            db.add_entry("a", "invoice_match", 0.8)
            db.add_entry("b", "invoice_match", 0.9)

            stats = db.get_stats()
            assert stats["total_entries"] == 2
            assert stats["by_task"]["invoice_match"]["best_score"] == 0.9
        finally:
            os.unlink(db_path)

    def test_breakdown_stored(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = LeaderboardDB(db_path)
            db.add_entry("a", "full_recon", 0.75, breakdown={
                "itc_accuracy": 0.9, "recall_score": 0.6,
                "efficiency_bonus": 0.05, "hallucination_penalty": 0.1,
            })
            entry = db.get_top(1)[0]
            assert entry["itc_accuracy"] == 0.9
            assert entry["hallucination_penalty"] == 0.1
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
