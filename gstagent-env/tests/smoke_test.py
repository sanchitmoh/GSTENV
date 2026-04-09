"""Quick smoke test for all API endpoints.

Requires a running server. Skipped automatically if the server is not reachable.
Run manually: python -m pytest tests/smoke_test.py -v
"""
import pytest
import requests

from environment.config import API_BASE_URL

BASE = API_BASE_URL


def _server_available() -> bool:
    """Check if the API server is reachable."""
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


# Skip entire module if server is not running
pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"API server not reachable at {BASE}",
)


def test_health():
    """1. Health endpoint."""
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_reset_and_step():
    """2-3. Reset + Step endpoints."""
    # Reset
    r = requests.post(f"{BASE}/reset", json={"task_id": "invoice_match"})
    assert r.status_code == 200
    data = r.json()
    sid = data["session_id"]
    assert len(data["purchase_register"]) > 0

    # Step
    r = requests.post(f"{BASE}/step", json={
        "session_id": sid,
        "action": {"action_type": "match_invoice", "invoice_id": "INV-0001"},
    })
    assert r.status_code == 200
    step_data = r.json()
    assert "reward" in step_data


def test_state(reset_session):
    """4. State endpoint."""
    r = requests.get(f"{BASE}/state/{reset_session}")
    assert r.status_code == 200
    assert r.json()["step_number"] >= 0


def test_leaderboard():
    """5. Leaderboard endpoint."""
    r = requests.get(f"{BASE}/leaderboard")
    assert r.status_code == 200
    assert "entries" in r.json()


def test_replay(reset_session):
    """6. Replay endpoint."""
    r = requests.get(f"{BASE}/replay/{reset_session}")
    assert r.status_code == 200
    assert "total_steps" in r.json()


def test_invalid_session():
    """7. Invalid session returns 400 (bad UUID format)."""
    r = requests.post(f"{BASE}/step", json={
        "session_id": "fake-session",
        "action": {"action_type": "match_invoice", "invoice_id": "X"},
    })
    # 400 because "fake-session" fails UUID validation, or 404 if not found
    assert r.status_code in (400, 404)


# ── Fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def reset_session():
    """Create a fresh session for tests that need one."""
    r = requests.post(f"{BASE}/reset", json={"task_id": "invoice_match"})
    assert r.status_code == 200
    return r.json()["session_id"]
