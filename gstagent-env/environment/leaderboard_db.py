"""
SQLite-based persistent leaderboard.

Upgrade from leaderboard.json:
- ACID transactions (no corruption on concurrent writes)
- Indexed queries (fast top-N retrieval)
- Historical data retention
- Per-task and per-model breakdowns
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

from environment.config import LEADERBOARD_DB_PATH

DEFAULT_DB_PATH = Path(LEADERBOARD_DB_PATH) if LEADERBOARD_DB_PATH else Path(__file__).parent.parent / "leaderboard.db"


class LeaderboardDB:
    """
    SQLite-backed leaderboard with per-task scoring.

    Schema:
        entries(id, session_id, task_id, score, steps, model_name,
               itc_accuracy, recall_score, action_correctness,
               efficiency_bonus, hallucination_penalty, timestamp)
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    steps INTEGER DEFAULT 0,
                    model_name TEXT DEFAULT '',
                    itc_accuracy REAL DEFAULT 0,
                    recall_score REAL DEFAULT 0,
                    action_correctness REAL DEFAULT 0,
                    efficiency_bonus REAL DEFAULT 0,
                    hallucination_penalty REAL DEFAULT 0,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_score
                ON entries(score DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task
                ON entries(task_id, score DESC)
            """)

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_entry(
        self,
        session_id: str,
        task_id: str,
        score: float,
        steps: int = 0,
        model_name: str = "",
        breakdown: dict | None = None,
    ) -> int:
        """Add a leaderboard entry. Returns the entry ID."""
        bd = breakdown or {}
        timestamp = datetime.utcnow().isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO entries
                (session_id, task_id, score, steps, model_name,
                 itc_accuracy, recall_score, action_correctness,
                 efficiency_bonus, hallucination_penalty, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    task_id,
                    round(score, 4),
                    steps,
                    model_name,
                    bd.get("itc_accuracy", 0),
                    bd.get("recall_score", 0),
                    bd.get("action_correctness", 0),
                    bd.get("efficiency_bonus", 0),
                    bd.get("hallucination_penalty", 0),
                    timestamp,
                ),
            )
            entry_id = cursor.lastrowid

        logger.info(
            "leaderboard_entry",
            entry_id=entry_id,
            session_id=session_id[:8],
            task_id=task_id,
            score=score,
        )
        return entry_id

    def get_top(self, limit: int = 10, task_id: str | None = None) -> list[dict]:
        """Get top entries, optionally filtered by task."""
        with self._connect() as conn:
            if task_id:
                rows = conn.execute(
                    "SELECT * FROM entries WHERE task_id = ? ORDER BY score DESC LIMIT ?",
                    (task_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entries ORDER BY score DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        return [dict(row) for row in rows]

    def get_by_session(self, session_id: str) -> list[dict]:
        """Get all entries for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entries WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Get aggregate leaderboard statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            avg_score = conn.execute("SELECT AVG(score) FROM entries").fetchone()[0] or 0

            task_stats = {}
            for task_id in ["invoice_match", "itc_audit", "full_recon"]:
                row = conn.execute(
                    """
                    SELECT COUNT(*) as count, AVG(score) as avg, MAX(score) as best
                    FROM entries WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()
                task_stats[task_id] = {
                    "count": row["count"],
                    "avg_score": round(row["avg"] or 0, 4),
                    "best_score": round(row["best"] or 0, 4),
                }

        return {
            "total_entries": total,
            "avg_score": round(avg_score, 4),
            "by_task": task_stats,
        }

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM entries")
