"""Quick smoke test for all API endpoints."""
import requests
import json

from environment.config import API_BASE_URL

BASE = API_BASE_URL

# 1. Health
r = requests.get(f"{BASE}/health")
assert r.status_code == 200
print("1. /health OK:", r.json())

# 2. Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "invoice_match"})
assert r.status_code == 200
data = r.json()
sid = data["session_id"]
print(f"2. /reset OK - session: {sid[:8]}, invoices: {len(data['purchase_register'])}")

# 3. Step
r = requests.post(f"{BASE}/step", json={
    "session_id": sid,
    "action": {"action_type": "match_invoice", "invoice_id": "INV-0001"}
})
assert r.status_code == 200
step_data = r.json()
print(f"3. /step OK - reward: {step_data['reward']}, done: {step_data['done']}")

# 4. State
r = requests.get(f"{BASE}/state/{sid}")
assert r.status_code == 200
print(f"4. /state OK - step: {r.json()['step_number']}")

# 5. Leaderboard
r = requests.get(f"{BASE}/leaderboard")
assert r.status_code == 200
print(f"5. /leaderboard OK - entries: {len(r.json()['entries'])}")

# 6. Replay
r = requests.get(f"{BASE}/replay/{sid}")
assert r.status_code == 200
print(f"6. /replay OK - steps: {r.json()['total_steps']}")

# 7. Invalid session
r = requests.post(f"{BASE}/step", json={
    "session_id": "fake-session",
    "action": {"action_type": "match_invoice", "invoice_id": "X"}
})
assert r.status_code == 404
print(f"7. Invalid session: {r.status_code} (expected 404)")

print()
print("ALL 7 API ENDPOINTS VERIFIED")
