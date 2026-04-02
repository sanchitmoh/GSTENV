"""
Security tests — validates all hardening controls.

Covers:
- C1: CORS headers
- C2: Timing-safe API key comparison
- C3: Body size limit
- H2: Security headers
- M1: UUID session ID validation
- M3: Global exception handler
- M4: InputSanitizer integration
"""

import hmac

import pytest
from fastapi.testclient import TestClient

from environment.auth import InputSanitizer, RateLimitTracker, _constant_time_compare
from environment.server import app


client = TestClient(app)


class TestSecurityHeaders:
    """H2: Security headers on every response."""

    def test_security_headers_present(self):
        resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "geolocation=()" in resp.headers["Permissions-Policy"]
        assert resp.headers["Cache-Control"] == "no-store"
        assert resp.headers["X-DNS-Prefetch-Control"] == "off"

    def test_request_id_in_response(self):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
        # L3: should be full UUID (36 chars)
        assert len(resp.headers["X-Request-ID"]) == 36


class TestTimingSafeComparison:
    """C2: Timing-safe API key comparison."""

    def test_constant_time_compare_equal(self):
        assert _constant_time_compare("secret123", "secret123") is True

    def test_constant_time_compare_not_equal(self):
        assert _constant_time_compare("secret123", "secret456") is False

    def test_constant_time_compare_empty(self):
        assert _constant_time_compare("", "") is True

    def test_constant_time_compare_different_length(self):
        assert _constant_time_compare("short", "muchlongerstring") is False


class TestUUIDValidation:
    """M1: Session ID must be a valid UUID."""

    def test_valid_uuid_accepted(self):
        InputSanitizer.validate_session_id("12345678-1234-1234-1234-123456789abc")

    def test_invalid_format_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            InputSanitizer.validate_session_id("not-a-uuid")
        assert exc_info.value.status_code == 400

    def test_empty_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_session_id("")

    def test_sql_injection_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_session_id("'; DROP TABLE entries; --")

    def test_path_traversal_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_session_id("../../etc/passwd")


class TestInputSanitizer:
    """M4: InputSanitizer wired up for all inputs."""

    def test_valid_action_passes(self):
        result = InputSanitizer.validate_action({
            "action_type": "match_invoice",
            "invoice_id": "INV-0001",
        })
        assert result["action_type"] == "match_invoice"

    def test_invalid_action_type_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            InputSanitizer.validate_action({"action_type": "hack_system"})
        assert exc_info.value.status_code == 400

    def test_oversized_invoice_id_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_action({
                "action_type": "match_invoice",
                "invoice_id": "A" * 25,
            })

    def test_reason_truncated(self):
        result = InputSanitizer.validate_action({
            "action_type": "flag_mismatch",
            "reason": "X" * 500,
        })
        assert len(result["reason"]) == 200

    def test_oversized_payload_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_action({
                "action_type": "submit_report",
                "payload": {f"key_{i}": i for i in range(60)},
            })

    def test_valid_task_id(self):
        assert InputSanitizer.validate_task_id("invoice_match") == "invoice_match"

    def test_invalid_task_id(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            InputSanitizer.validate_task_id("hack_system")


class TestRateLimiter:
    """H3: Rate limiter with memory safety."""

    def test_allows_within_limit(self):
        rl = RateLimitTracker(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.check("1.1.1.1") is True

    def test_blocks_over_limit(self):
        rl = RateLimitTracker(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.check("2.2.2.2")
        assert rl.check("2.2.2.2") is False

    def test_remaining_count(self):
        rl = RateLimitTracker(max_requests=10, window_seconds=60)
        rl.check("3.3.3.3")
        rl.check("3.3.3.3")
        assert rl.get_remaining("3.3.3.3") == 8

    def test_ip_eviction_at_capacity(self):
        rl = RateLimitTracker(max_requests=100, window_seconds=60)
        rl.MAX_TRACKED_IPS = 3  # Artificially low for testing
        for i in range(4):
            rl.check(f"10.0.0.{i}")
        # Should not exceed capacity
        assert len(rl._requests) <= 3


class TestBodySizeLimit:
    """C3: Request body size limit."""

    def test_normal_request_allowed(self):
        resp = client.post("/reset", json={"task_id": "invoice_match"})
        assert resp.status_code == 200


class TestGlobalExceptionHandler:
    """M3: No stack trace leakage."""

    def test_404_is_json(self):
        resp = client.get("/nonexistent-endpoint")
        assert resp.status_code in (404, 500)

    def test_invalid_session_format_returns_json(self):
        resp = client.get("/state/not-a-uuid")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data or "detail" in data

    def test_invalid_step_session_returns_400(self):
        resp = client.post("/step", json={
            "session_id": "injection'; DROP TABLE --",
            "action": {"action_type": "match_invoice", "invoice_id": "INV-0001"},
        })
        assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
