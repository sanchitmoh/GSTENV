"""
Curriculum Learning — auto-advance difficulty based on agent performance.

Tracks scores across episodes and automatically adjusts task difficulty.
Implements the escalation: easy → medium → hard based on score thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    easy_threshold: float = 0.8     # Score needed to advance from easy
    medium_threshold: float = 0.75  # Score needed to advance from medium
    hard_threshold: float = 0.7     # Score needed to "master" hard
    min_episodes: int = 1           # Min episodes at each level before advancing
    auto_advance: bool = True       # Whether to auto-advance


DIFFICULTY_ORDER = ["easy", "medium", "hard"]
TASK_BY_DIFFICULTY = {
    "easy": "invoice_match",
    "medium": "itc_audit",
    "hard": "full_recon",
}


class CurriculumManager:
    """
    Manages difficulty progression based on agent performance.

    Usage:
        curriculum = CurriculumManager()
        task_id = curriculum.get_next_task()  # starts with "invoice_match"
        curriculum.record_score("invoice_match", 0.85)
        task_id = curriculum.get_next_task()  # now returns "itc_audit"
    """

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self.current_level: int = 0  # 0=easy, 1=medium, 2=hard
        self.score_history: dict[str, list[float]] = {
            "invoice_match": [],
            "itc_audit": [],
            "full_recon": [],
        }
        self.episodes_at_level: int = 0
        self.advancement_log: list[dict] = []

    def get_next_task(self) -> str:
        """Get the task ID for the current difficulty level."""
        difficulty = DIFFICULTY_ORDER[min(self.current_level, len(DIFFICULTY_ORDER) - 1)]
        return TASK_BY_DIFFICULTY[difficulty]

    def get_current_difficulty(self) -> str:
        """Get current difficulty label."""
        return DIFFICULTY_ORDER[min(self.current_level, len(DIFFICULTY_ORDER) - 1)]

    def record_score(self, task_id: str, score: float) -> dict:
        """
        Record a score and check for advancement.

        Returns dict with advancement info:
        {"advanced": bool, "from": str, "to": str, "reason": str}
        """
        self.score_history[task_id].append(score)
        self.episodes_at_level += 1

        result = {
            "advanced": False,
            "current_level": self.get_current_difficulty(),
            "score": score,
            "avg_score": self._get_avg_score(task_id),
        }

        if not self.config.auto_advance:
            return result

        if self.episodes_at_level < self.config.min_episodes:
            return result

        # Check advancement thresholds
        avg = self._get_avg_score(task_id)
        threshold = self._get_threshold()

        if avg >= threshold and self.current_level < len(DIFFICULTY_ORDER) - 1:
            old_level = self.get_current_difficulty()
            self.current_level += 1
            new_level = self.get_current_difficulty()
            self.episodes_at_level = 0

            advancement = {
                "from": old_level,
                "to": new_level,
                "trigger_score": score,
                "avg_score": avg,
                "threshold": threshold,
            }
            self.advancement_log.append(advancement)

            logger.info(
                "curriculum_advance",
                from_level=old_level,
                to_level=new_level,
                avg_score=avg,
                threshold=threshold,
            )

            result["advanced"] = True
            result["from"] = old_level
            result["to"] = new_level
            result["current_level"] = new_level
            result["reason"] = f"Avg score {avg:.2f} >= threshold {threshold:.2f}"

        return result

    def _get_avg_score(self, task_id: str) -> float:
        """Get average score for a task."""
        scores = self.score_history.get(task_id, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _get_threshold(self) -> float:
        """Get the advancement threshold for current level."""
        thresholds = [
            self.config.easy_threshold,
            self.config.medium_threshold,
            self.config.hard_threshold,
        ]
        return thresholds[min(self.current_level, len(thresholds) - 1)]

    def is_mastered(self) -> bool:
        """Check if agent has mastered all levels."""
        if self.current_level < len(DIFFICULTY_ORDER) - 1:
            return False
        hard_scores = self.score_history.get("full_recon", [])
        if not hard_scores:
            return False
        return (sum(hard_scores) / len(hard_scores)) >= self.config.hard_threshold

    def get_summary(self) -> dict:
        """Get curriculum learning summary."""
        return {
            "current_level": self.get_current_difficulty(),
            "current_task": self.get_next_task(),
            "episodes_at_level": self.episodes_at_level,
            "score_history": {
                k: [round(s, 4) for s in v]
                for k, v in self.score_history.items()
                if v
            },
            "advancement_log": self.advancement_log,
            "mastered": self.is_mastered(),
        }
