"""
Centralized configuration — single source of truth for all settings.

ALL configurable values are read from environment variables with safe defaults.
No hardcoded URLs, ports, model names, or secrets anywhere else in the codebase.

Security: Validates required vars at import time (fail-fast).
"""

from __future__ import annotations

import os
import secrets
import sys

import structlog

logger = structlog.get_logger()


# ── API / Network ────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:7860")
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")  # nosec B104 — required for Docker
API_PORT: int = int(os.getenv("API_PORT", "7860"))

# ── LLM / Model ─────────────────────────────────────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4")
FAST_MODEL_NAME: str = os.getenv("FAST_MODEL_NAME", "gpt-3.5-turbo")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Data Generation ──────────────────────────────────────────────────
SEED: int = int(os.getenv("DATA_SEED", "42"))

# ── Session Management ───────────────────────────────────────────────
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
SESSION_CLEANUP_INTERVAL: int = int(os.getenv("SESSION_CLEANUP_INTERVAL", "60"))

# ── Timeouts ─────────────────────────────────────────────────────────
RESET_TIMEOUT_SECONDS: int = int(os.getenv("RESET_TIMEOUT_SECONDS", "60"))
STEP_TIMEOUT_SECONDS: int = int(os.getenv("STEP_TIMEOUT_SECONDS", "30"))
INFERENCE_TIMEOUT_SECONDS: int = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "1140"))

# ── Rate Limiting ────────────────────────────────────────────────────
RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# ── Security ─────────────────────────────────────────────────────────
GST_API_KEY: str = os.getenv("GST_API_KEY", "")
ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
MAX_REQUEST_BODY_BYTES: int = int(os.getenv("MAX_REQUEST_BODY_BYTES", "1048576"))  # 1MB

# Auto-generate ephemeral API key if none configured (security fix H1)
if not GST_API_KEY:
    GST_API_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "auth_no_key_configured",
        msg="No GST_API_KEY set. Generated ephemeral key for this session.",
        key_preview=GST_API_KEY[:8] + "...",
    )

# ── Leaderboard ──────────────────────────────────────────────────────
LEADERBOARD_MAX_ENTRIES: int = int(os.getenv("LEADERBOARD_MAX_ENTRIES", "50"))
LEADERBOARD_DB_PATH: str = os.getenv("LEADERBOARD_DB_PATH", "")

# ── Deployment ───────────────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Curriculum ───────────────────────────────────────────────────────
CURRICULUM_EASY_THRESHOLD: float = float(os.getenv("CURRICULUM_EASY_THRESHOLD", "0.8"))
CURRICULUM_MEDIUM_THRESHOLD: float = float(os.getenv("CURRICULUM_MEDIUM_THRESHOLD", "0.75"))
CURRICULUM_HARD_THRESHOLD: float = float(os.getenv("CURRICULUM_HARD_THRESHOLD", "0.7"))


# ── Startup Validation (Security.md §13 — fail-fast) ────────────────
def validate_config() -> list[str]:
    """Validate configuration and return list of warnings."""
    warnings: list[str] = []
    if ALLOWED_ORIGINS == "*":
        warnings.append("CORS: ALLOWED_ORIGINS is wildcard '*'. Set explicit origins in production.")
    if not os.getenv("GST_API_KEY"):
        warnings.append("AUTH: No GST_API_KEY set. Using auto-generated ephemeral key.")
    if not OPENAI_API_KEY:
        warnings.append("LLM: No OPENAI_API_KEY set. LLM-based agents will fail.")
    return warnings


_startup_warnings = validate_config()
for w in _startup_warnings:
    logger.warning("config_validation", warning=w)
