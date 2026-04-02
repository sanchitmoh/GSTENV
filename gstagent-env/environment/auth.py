"""
Authentication and security middleware.

Provides:
- API key authentication with timing-safe comparison (CVE-2026-23996 fix)
- Request validation and sanitization
- Memory-safe rate limiter with IP eviction
"""

from __future__ import annotations

import hmac
import re
import time

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = structlog.get_logger()

# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS: set[str] = set()

# Load keys from centralized config
from environment.config import GST_API_KEY as _env_key
if _env_key:
    VALID_API_KEYS.add(_env_key)


# ── Security Fix C2: Timing-safe API key comparison ─────────────────
def _constant_time_compare(a: str, b: str) -> bool:
    """Compare strings in constant time to prevent timing side-channel attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """
    Verify API key using timing-safe comparison.

    Authentication is always enabled since config.py auto-generates
    an ephemeral key if none is configured.
    """
    # If no keys configured (shouldn't happen — config auto-generates), allow
    if not VALID_API_KEYS:
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header.",
        )

    # Security Fix C2: constant-time comparison prevents timing attacks
    if not any(_constant_time_compare(api_key, k) for k in VALID_API_KEYS):
        logger.warning("auth_failed", key_prefix=api_key[:4] + "..." if len(api_key) >= 4 else "***")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


# ── Regex for UUID validation (M1) ──────────────────────────────────
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class InputSanitizer:
    """
    Validate and sanitize incoming request data.

    Prevents:
    - Excessively long strings (DoS)
    - Invalid characters in IDs
    - Unreasonable numeric values
    - Injection via malformed session_ids
    """

    MAX_STRING_LENGTH = 500
    MAX_INVOICE_ID_LENGTH = 20
    MAX_REASON_LENGTH = 200
    MAX_PAYLOAD_KEYS = 50
    VALID_ACTION_TYPES = {"match_invoice", "flag_mismatch", "compute_itc", "submit_report"}
    VALID_TASK_IDS = {"invoice_match", "itc_audit", "full_recon"}

    @classmethod
    def validate_action(cls, action: dict) -> dict:
        """Validate and sanitize an action dict."""
        action_type = action.get("action_type", "")

        if action_type not in cls.VALID_ACTION_TYPES:
            raise HTTPException(
                400,
                f"Invalid action_type: '{action_type}'. "
                f"Valid: {cls.VALID_ACTION_TYPES}",
            )

        # Validate invoice_id length
        inv_id = action.get("invoice_id", "")
        if inv_id and len(inv_id) > cls.MAX_INVOICE_ID_LENGTH:
            raise HTTPException(400, f"invoice_id too long (max {cls.MAX_INVOICE_ID_LENGTH})")

        # Validate reason length — truncate silently
        reason = action.get("reason", "")
        if reason and len(reason) > cls.MAX_REASON_LENGTH:
            action["reason"] = reason[:cls.MAX_REASON_LENGTH]

        # Validate payload size
        payload = action.get("payload", {})
        if payload and len(payload) > cls.MAX_PAYLOAD_KEYS:
            raise HTTPException(400, f"Payload too large (max {cls.MAX_PAYLOAD_KEYS} keys)")

        return action

    @classmethod
    def validate_task_id(cls, task_id: str) -> str:
        """Validate task_id against allowlist."""
        if task_id not in cls.VALID_TASK_IDS:
            raise HTTPException(
                400,
                f"Invalid task_id: '{task_id}'. Valid: {cls.VALID_TASK_IDS}",
            )
        return task_id

    @classmethod
    def validate_session_id(cls, session_id: str) -> str:
        """Validate session_id is a properly formatted UUID (M1 fix)."""
        if not session_id or not _UUID_PATTERN.match(session_id):
            raise HTTPException(400, "Invalid session_id format. Expected UUID.")
        return session_id


class RateLimitTracker:
    """
    In-memory rate limiter per IP with memory safety (H3 fix).

    Features:
    - Time-windowed request counting
    - Max tracked IPs cap to prevent memory exhaustion
    - LRU eviction when cap is reached
    """

    MAX_TRACKED_IPS = 10_000

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def check(self, client_ip: str) -> bool:
        """Check if request is allowed. Returns True if allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # H3 fix: evict oldest IP if at capacity
        if client_ip not in self._requests and len(self._requests) >= self.MAX_TRACKED_IPS:
            oldest_ip = min(
                self._requests,
                key=lambda ip: self._requests[ip][-1] if self._requests[ip] else 0,
            )
            del self._requests[oldest_ip]

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # Clean old entries within window
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for this IP."""
        now = time.time()
        window_start = now - self.window_seconds
        recent = [t for t in self._requests.get(client_ip, []) if t > window_start]
        return max(0, self.max_requests - len(recent))
