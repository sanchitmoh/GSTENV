"""
API integration tests for the FastAPI server.

Covers: /health, /reset, /step, /state, session isolation, invalid session.
"""

import pytest
from fastapi.testclient import TestClient

from environment.server import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestResetEndpoint:
    def test_reset_returns_observation(self):
        resp = client.post("/reset", json={"task_id": "invoice_match"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["task_id"] == "invoice_match"
        assert len(data["purchase_register"]) > 0

    def test_reset_invalid_task(self):
        resp = client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 400


class TestStepEndpoint:
    def test_step_valid_session(self):
        reset = client.post("/reset", json={"task_id": "invoice_match"})
        session_id = reset.json()["session_id"]

        resp = client.post("/step", json={
            "session_id": session_id,
            "action": {"action_type": "match_invoice", "invoice_id": "INV-0001"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data

    def test_step_invalid_session(self):
        # Use valid UUID format but non-existent session
        resp = client.post("/step", json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "action": {"action_type": "match_invoice", "invoice_id": "INV-0001"},
        })
        assert resp.status_code == 404


class TestStateEndpoint:
    def test_state_returns_cached(self):
        reset = client.post("/reset", json={"task_id": "invoice_match"})
        session_id = reset.json()["session_id"]

        resp = client.get(f"/state/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

    def test_state_invalid_session(self):
        # Use valid UUID format but non-existent session
        resp = client.get("/state/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


class TestSessionIsolation:
    def test_two_sessions_independent(self):
        r1 = client.post("/reset", json={"task_id": "invoice_match"})
        r2 = client.post("/reset", json={"task_id": "itc_audit"})
        s1 = r1.json()["session_id"]
        s2 = r2.json()["session_id"]
        assert s1 != s2

        # Step in session 1
        client.post("/step", json={
            "session_id": s1,
            "action": {"action_type": "match_invoice", "invoice_id": "INV-0001"},
        })

        # Session 2 should be unaffected
        state2 = client.get(f"/state/{s2}").json()
        assert state2["step_number"] == 0


class TestLeaderboard:
    def test_leaderboard_returns_list(self):
        resp = client.get("/leaderboard")
        assert resp.status_code == 200
        assert "entries" in resp.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
