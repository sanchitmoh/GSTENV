"""
FastAPI server for the GST Agent Environment.

Features:
- Session-aware: dict[str, GSTAgentEnv] for concurrent safety (Fix #10)
- Async: run_in_executor for sync env calls (Fix #19)
- Timeouts: configurable per step/reset (Fix #20)
- CORS: configurable allowlist — no wildcard in production (Security C1)
- Security headers: X-Content-Type-Options, X-Frame-Options, etc. (Security H2)
- Body size limit: prevents payload DoS (Security C3)
- Global exception handler: no stack trace leakage (Security M3)
- Input sanitization: InputSanitizer wired into endpoints (Security M4)
- Structured logging: structlog JSON + request_id + redaction (Fix #30, M7)
- Rate limiting: slowapi (configurable req/min)
- Leaderboard + Replay endpoints (Fix #36)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque as _deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from environment.auth import InputSanitizer
from environment.config import (
    ALLOWED_ORIGINS,
    LEADERBOARD_MAX_ENTRIES,
    MAX_REQUEST_BODY_BYTES,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW,
    RESET_TIMEOUT_SECONDS,
    SESSION_CLEANUP_INTERVAL,
    SESSION_TTL_SECONDS,
    STEP_TIMEOUT_SECONDS,
)
from environment.env import GSTAgentEnv
from environment.models import (
    GSTAction,
    GSTObservation,
    ResetRequest,
    StepRequest,
)


# ── Logging with redaction (Fix #30 + Security M7) ──────────────────
def _redact_sensitive(_, __, event_dict):
    """Remove sensitive fields from log output."""
    for key in ("api_key", "token", "password", "openai_key", "authorization", "cookie"):
        if key in event_dict:
            event_dict[key] = "[REDACTED]"
    return event_dict


structlog.configure(
    processors=[
        _redact_sensitive,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()

# ── Session Manager (Fix #10) ────────────────────────────────────────
sessions: dict[str, dict] = {}  # session_id -> {"env": GSTAgentEnv, "created_at": float}

# ── Leaderboard — async-safe in-memory deque (Upgrade #10) ───────────
LEADERBOARD_FILE = Path(__file__).parent.parent / "leaderboard.json"
_leaderboard_lock: asyncio.Lock = asyncio.Lock()
_leaderboard: _deque[dict] = _deque(maxlen=LEADERBOARD_MAX_ENTRIES)


def _load_leaderboard_from_disk() -> None:
    """Sync cold-start load. Called once in lifespan before event loop tasks start."""
    if LEADERBOARD_FILE.exists():
        try:
            with open(LEADERBOARD_FILE) as f:
                entries: list[dict] = json.load(f)
            entries.sort(key=lambda x: x.get("score", 0), reverse=True)
            for entry in entries[:LEADERBOARD_MAX_ENTRIES]:
                _leaderboard.append(entry)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("leaderboard_load_failed", error=str(e))


async def _append_leaderboard(entry: dict) -> None:
    """Thread-safe append + fire-and-forget background disk flush."""
    async with _leaderboard_lock:
        _leaderboard.append(entry)
        snapshot = list(_leaderboard)

    async def _persist() -> None:
        try:
            await asyncio.to_thread(_write_leaderboard_sync, snapshot)
        except Exception as e:
            logger.error("leaderboard_persist_failed", error=str(e))

    asyncio.create_task(_persist())


def _write_leaderboard_sync(entries: list[dict]) -> None:
    """Sync write called in thread pool (no event loop blocking)."""
    sorted_entries = sorted(entries, key=lambda x: x.get("score", 0), reverse=True)
    try:
        with open(LEADERBOARD_FILE, "w") as f:
            json.dump(sorted_entries[:LEADERBOARD_MAX_ENTRIES], f, indent=2)
    except OSError as e:
        logger.error("leaderboard_write_failed", error=str(e))



# ── Rate Limiting ────────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_WINDOW}seconds"],
)


# ── TTL Cleanup ──────────────────────────────────────────────────────
async def cleanup_expired_sessions() -> None:
    """Remove sessions older than TTL."""
    while True:
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL)
        now = time.time()
        expired = [
            sid for sid, data in sessions.items()
            if now - data["created_at"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del sessions[sid]
            logger.info("session_expired", session_id=sid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    _load_leaderboard_from_disk()  # populate in-memory deque at boot
    task = asyncio.create_task(cleanup_expired_sessions())
    logger.info("server_started", ttl_seconds=SESSION_TTL_SECONDS)
    yield
    task.cancel()
    logger.info("server_stopped")


# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="GST Agent Environment",
    description="OpenEnv-compatible RL environment for Indian GST reconciliation",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Security Fix C1: CORS — configurable allowlist ──────────────────
_cors_origins = ALLOWED_ORIGINS.split(",") if ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=ALLOWED_ORIGINS != "*",  # Never credentials+wildcard
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)
app.state.limiter = limiter


# ── Security Fix M3: Global exception handler — no stack trace leak ─
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions — never expose internals."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled_exception",
        error=type(exc).__name__,
        detail=str(exc),
        path=request.url.path,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal error occurred.",
            "request_id": request_id,
        },
    )


# ── Security Fix C3: Body size limit middleware ─────────────────────
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject oversized request bodies to prevent memory exhaustion DoS."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "Request body too large.", "max_bytes": MAX_REQUEST_BODY_BYTES},
        )
    return await call_next(request)


# ── Security Fix H2: Security headers middleware ────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers to every response (OWASP A05)."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-DNS-Prefetch-Control"] = "off"
    # Upgrade #7: CSP — restrict scripts, objects, and frames (XSS + clickjacking)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'none'; "
        "object-src 'none'; "
        "frame-ancestors 'none';"
    )
    return response


# ── Request ID Middleware (Fix #30) ───────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # L3 fix: use full UUID4 instead of truncated 8-char
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    logger.info(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=elapsed,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Root health check — HuggingFace Space readiness probe (GET /)."""
    return {
        "status": "healthy",
        "name": "gstagent-env",
        "description": "OpenEnv-compatible RL environment for Indian GST reconciliation",
        "version": "1.0.0",
        "sessions_active": len(sessions),
        "endpoints": ["/reset", "/step", "/health", "/metadata", "/schema", "/leaderboard"],
    }


@app.get("/health")
async def health():
    """Liveness probe — HuggingFace pings this to verify the Space is alive."""
    return {"status": "healthy", "sessions_active": len(sessions)}


@app.post("/reset")
@limiter.limit(f"{RATE_LIMIT_REQUESTS}/minute")
async def reset(request: Request, body: ResetRequest | None = None):
    """
    Start a new episode. Creates a new session and returns observation with session_id.
    Body is optional — if omitted, defaults to invoice_match task.
    """
    task_id = body.task_id if body else "invoice_match"
    # Security Fix M4: wire up InputSanitizer
    InputSanitizer.validate_task_id(task_id)

    loop = asyncio.get_event_loop()
    env = GSTAgentEnv()

    try:
        # Fix #19 + #20: async with timeout
        obs = await asyncio.wait_for(
            loop.run_in_executor(None, env.reset, task_id),
            timeout=RESET_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        # Security Fix M2: generic error — don't reveal timeout value
        raise HTTPException(504, "Request timed out. Please retry.")
    except ValueError as e:
        raise HTTPException(400, str(e))

    session_id = env.session_id
    sessions[session_id] = {"env": env, "created_at": time.time()}

    logger.info(
        "episode_reset",
        session_id=session_id,
        task_id=task_id,
        invoice_count=len(env.purchase_register),
    )

    return obs.model_dump()


@app.post("/step")
@limiter.limit(f"{RATE_LIMIT_REQUESTS}/minute")
async def step(body: StepRequest, request: Request):
    """
    Advance one step. Requires session_id from /reset response.
    """
    # Security Fix M4: validate session_id and action
    InputSanitizer.validate_session_id(body.session_id)
    InputSanitizer.validate_action(body.action.model_dump())

    session = sessions.get(body.session_id)
    if not session:
        # Security Fix M2: generic — don't confirm session existence
        raise HTTPException(404, "Session not found or expired. Call /reset first.")

    env: GSTAgentEnv = session["env"]
    loop = asyncio.get_event_loop()

    try:
        # Fix #19 + #20: async with timeout
        result = await asyncio.wait_for(
            loop.run_in_executor(None, env.step, body.action),
            timeout=STEP_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Request timed out. Please retry.")

    obs, reward, done, info = result

    # Save to leaderboard on completion (async-safe — Upgrade #10)
    if done and reward > 0:
        await _append_leaderboard({
            "session_id": body.session_id,
            "task_id": env.task_id,
            "score": reward,
            "steps": env._step_number,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    logger.info(
        "step",
        session_id=body.session_id,
        action=body.action.action_type,
        invoice_id=body.action.invoice_id,
        reward=reward,
        done=done,
        step=env._step_number,
    )

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    """Return current observation without advancing. Uses cached obs (Fix #22)."""
    # Security Fix M4: validate session_id
    InputSanitizer.validate_session_id(session_id)

    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired.")

    env: GSTAgentEnv = session["env"]
    return env.state().model_dump()


@app.get("/leaderboard")
async def leaderboard():
    """Return top 10 scores — served from memory (Upgrade #10)."""
    async with _leaderboard_lock:
        top10 = sorted(_leaderboard, key=lambda x: x.get("score", 0), reverse=True)[:10]
    return {"entries": top10}


@app.get("/replay/{session_id}")
async def replay(session_id: str):
    """Return full action history for a session (Fix #36)."""
    # Security Fix M4: validate session_id
    InputSanitizer.validate_session_id(session_id)

    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found or expired.")

    env: GSTAgentEnv = session["env"]
    return {
        "session_id": session_id,
        "task_id": env.task_id,
        "steps": env.replay_log,
        "total_steps": len(env.replay_log),
        "done": env._done,
    }


# ── OpenEnv Runtime Contract Endpoints ─────────────────────────────


@app.get("/metadata")
async def metadata():
    """OpenEnv metadata — returns environment name and description."""
    return {
        "name": "gstagent-env",
        "description": "OpenEnv-compatible RL environment for Indian GST reconciliation",
    }


@app.get("/schema")
async def env_schema():
    """OpenEnv schema — returns action, observation, and state JSON schemas."""
    return {
        "action": GSTAction.model_json_schema(),
        "observation": GSTObservation.model_json_schema(),
        "state": GSTObservation.model_json_schema(),
    }


@app.get("/state")
async def get_state_global():
    """OpenEnv /state — returns current state (latest active session or empty)."""
    if sessions:
        latest = max(sessions.items(), key=lambda x: x[1]["created_at"])
        env: GSTAgentEnv = latest[1]["env"]
        return env.state().model_dump()
    return {"status": "no_active_sessions"}


@app.post("/mcp")
async def mcp_endpoint():
    """OpenEnv MCP — minimal JSON-RPC 2.0 handler."""
    return {
        "jsonrpc": "2.0",
        "result": {
            "name": "gstagent-env",
            "version": "1.0.0",
        },
        "id": None,
    }
