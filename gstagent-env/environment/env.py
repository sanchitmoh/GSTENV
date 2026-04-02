"""
GSTAgentEnv — Core RL environment for GST reconciliation.

Features:
- Session-aware with UUID (Fix #9)
- O(1) invoice lookups via _gstr2b_index (Fix #28)
- Pre-computed ground truth at reset (Fix #29)
- Cached observation to avoid re-serialisation (Fix #22)
- max_steps enforcement (Fix #11)
- Proper error recovery via last_action_error (Fix #12)
- Replay log for action history
- Curriculum learning support (Fix #35)
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path

from environment.graders import grader_full_recon, grader_invoice_match, grader_itc_audit
from environment.models import GSTAction, GSTObservation, GSTReward, Invoice
from environment.rules.gst_rules import (
    check_itc_eligibility,
    compute_total_eligible_itc,
    get_recommended_action,
)
from environment.tasks.task1_invoice_match import get_task1_config
from environment.tasks.task2_itc_audit import get_task2_config
from environment.tasks.task3_full_recon import get_task3_config

DATA_DIR = Path(__file__).parent / "data"

TASK_CONFIGS = {
    "invoice_match": get_task1_config,
    "itc_audit": get_task2_config,
    "full_recon": get_task3_config,
}

VALID_ACTIONS = {"match_invoice", "flag_mismatch", "compute_itc", "submit_report"}


class GSTAgentEnv:
    """
    OpenEnv-compatible RL environment for GST reconciliation.

    Lifecycle: reset(task_id) → step(action) [repeat] → done
    """

    def __init__(self) -> None:
        self.session_id: str = ""
        self.task_id: str = ""
        self.task_config: dict = {}
        self.purchase_register: list[dict] = []
        self.gstr2b_data: list[dict] = []

        # Fix #28: O(1) lookup index
        self._gstr2b_index: dict[str, dict] = {}
        self._invoice_index: dict[str, dict] = {}

        # Fix #29: pre-computed ground truth
        self._ground_truth: dict = {}
        self._ground_truth_itc: float = 0.0

        # Episode state
        self._matches: list[dict] = []
        self._flags: list[dict] = []
        self._itc_results: list[dict] = []
        self._step_number: int = 0
        self._max_steps: int = 20
        self._done: bool = False
        self._last_error: str | None = None

        # Fix #22: cached observation
        self._cached_obs: GSTObservation | None = None

        # Replay log
        self.replay_log: list[dict] = []

        # Scoring history (for curriculum learning)
        self._score_history: dict[str, float] = {}

    def reset(self, task_id: str) -> GSTObservation:
        """
        Reset environment for a new episode.

        Loads task config and data, builds indices, pre-computes ground truth.
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Valid: {list(TASK_CONFIGS.keys())}"
            )

        self.session_id = str(uuid.uuid4())
        self.task_id = task_id
        self.task_config = TASK_CONFIGS[task_id]()
        self._max_steps = self.task_config.get("max_steps", 20)

        # Load data
        data_file = DATA_DIR / self.task_config["data_file"]
        with open(data_file) as f:
            dataset = json.load(f)

        self.purchase_register = dataset["purchase_register"]
        self.gstr2b_data = dataset["gstr2b"]

        # Fix #28: Build O(1) lookup indices at reset
        self._gstr2b_index = {
            inv["invoice_id"]: inv for inv in self.gstr2b_data
        }
        self._invoice_index = {
            inv["invoice_id"]: inv for inv in self.purchase_register
        }

        # Fix #29: Pre-compute ground truth
        self._precompute_ground_truth()

        # Reset episode state
        self._matches = []
        self._flags = []
        self._itc_results = []
        self._step_number = 0
        self._done = False
        self._last_error = None
        self.replay_log = []

        # Build and cache observation
        self._cached_obs = self._build_observation()
        return self._cached_obs

    def step(self, action: GSTAction) -> tuple[GSTObservation, float, bool, dict]:
        """
        Process one agent action.

        Returns: (observation, reward, done, info)
        """
        if self._done:
            return (
                self._cached_obs or self._build_observation(),
                0.0,
                True,
                {"error": "Episode already finished. Call reset() first."},
            )

        # Fix #11: enforce max_steps
        if self._step_number >= self._max_steps:
            self._done = True
            obs = self._build_observation()
            self._cached_obs = obs
            return (obs, 0.0, True, {"error": "Maximum steps exceeded."})

        self._step_number += 1
        reward = 0.0
        info: dict = {}

        # Fix #12: validate action type
        if action.action_type not in VALID_ACTIONS:
            self._last_error = (
                f"Invalid action_type: '{action.action_type}'. "
                f"Valid: {VALID_ACTIONS}"
            )
            obs = self._build_observation()
            self._cached_obs = obs
            self._log_action(action, reward, info)
            return (obs, 0.0, False, {"error": self._last_error})

        self._last_error = None

        # Process action
        if action.action_type == "match_invoice":
            reward, info = self._handle_match(action)
        elif action.action_type == "flag_mismatch":
            reward, info = self._handle_flag(action)
        elif action.action_type == "compute_itc":
            reward, info = self._handle_compute_itc(action)
        elif action.action_type == "submit_report":
            reward, info = self._handle_submit(action)
            self._done = True

        obs = self._build_observation()
        self._cached_obs = obs
        self._log_action(action, reward, info)

        return (obs, reward, self._done, info)

    def state(self) -> GSTObservation:
        """Return current observation without advancing. Uses cache (Fix #22)."""
        if self._cached_obs is not None:
            return self._cached_obs
        return self._build_observation()

    # ── Action Handlers ──────────────────────────────────────────────

    def _handle_match(self, action: GSTAction) -> tuple[float, dict]:
        """Handle match_invoice action. O(1) lookup via index (Fix #28)."""
        inv_id = action.invoice_id
        if not inv_id:
            self._last_error = "match_invoice requires invoice_id"
            return 0.0, {"error": self._last_error}

        if inv_id not in self._invoice_index:
            self._last_error = f"Invoice {inv_id} not found in purchase register"
            return 0.0, {"error": self._last_error}

        # O(1) lookup
        gstr2b_record = self._gstr2b_index.get(inv_id)
        status = "present" if gstr2b_record is not None else "missing"

        match_record = {
            "invoice_id": inv_id,
            "status": status,
            "gstr2b_found": gstr2b_record is not None,
        }

        # Check for amount mismatch
        if gstr2b_record:
            inv = self._invoice_index[inv_id]
            inv_amt = inv.get("taxable_amount", 0)
            gstr_amt = gstr2b_record.get("taxable_amount", 0)
            if inv_amt > 0:
                variance = abs(inv_amt - gstr_amt) / inv_amt
                match_record["amount_variance"] = round(variance, 4)

        self._matches.append(match_record)

        # Intermediate reward: +0.05 for correct match
        truth = self._ground_truth.get("match_status", {}).get(inv_id)
        reward = 0.05 if truth == status else 0.0

        return reward, {"match": match_record}

    def _handle_flag(self, action: GSTAction) -> tuple[float, dict]:
        """Handle flag_mismatch action."""
        inv_id = action.invoice_id
        if not inv_id:
            self._last_error = "flag_mismatch requires invoice_id"
            return 0.0, {"error": self._last_error}

        flag_record = {
            "invoice_id": inv_id,
            "reason": action.reason or "unspecified",
            "status": self._ground_truth.get("eligibility", {}).get(inv_id, "unknown"),
        }
        self._flags.append(flag_record)

        # Intermediate reward for correct flag
        truth_status = self._ground_truth.get("eligibility", {}).get(inv_id)
        reward = 0.05 if truth_status in ("partial", "ineligible") else 0.0

        return reward, {"flag": flag_record}

    def _handle_compute_itc(self, action: GSTAction) -> tuple[float, dict]:
        """Handle compute_itc action — run rules engine."""
        total_itc, details = compute_total_eligible_itc(
            self.purchase_register, self._gstr2b_index
        )
        self._itc_results = details

        return 0.05, {
            "total_itc": total_itc,
            "detail_count": len(details),
        }

    def _handle_submit(self, action: GSTAction) -> tuple[float, dict]:
        """Handle submit_report — run grader and end episode."""
        payload = action.payload or {}

        if self.task_id == "invoice_match":
            return self._grade_invoice_match(payload)
        elif self.task_id == "itc_audit":
            return self._grade_itc_audit(payload)
        elif self.task_id == "full_recon":
            return self._grade_full_recon(payload)
        else:
            return 0.0, {"error": f"No grader for task: {self.task_id}"}

    # ── Grading ──────────────────────────────────────────────────────

    def _grade_invoice_match(self, payload: dict) -> tuple[float, dict]:
        """Grade Task 1 submission."""
        agent_matches = payload.get("matches", {})

        # Build from accumulated matches if payload is empty
        if not agent_matches and self._matches:
            agent_matches = {m["invoice_id"]: m["status"] for m in self._matches}

        ground_truth = self._ground_truth.get("match_status", {})
        score = grader_invoice_match.grade(agent_matches, ground_truth)

        self._score_history[self.task_id] = score
        return score, {"score": score, "grader": "invoice_match"}

    def _grade_itc_audit(self, payload: dict) -> tuple[float, dict]:
        """Grade Task 2 submission."""
        agent_decisions = payload.get("decisions", {})

        # Build from flags if payload is empty
        if not agent_decisions and self._flags:
            agent_decisions = {f["invoice_id"]: f["status"] for f in self._flags}

        ground_truth_elig = self._ground_truth.get("eligibility", {})
        score = grader_itc_audit.grade(agent_decisions, ground_truth_elig)

        self._score_history[self.task_id] = score
        return score, {"score": score, "grader": "itc_audit"}

    def _grade_full_recon(self, payload: dict) -> tuple[float, dict]:
        """Grade Task 3 submission."""
        agent_report = {
            "total_itc": payload.get("total_itc", 0.0),
            "discrepancies": payload.get("discrepancies", []),
        }

        # Build from accumulated data if sparse
        if not agent_report["discrepancies"] and self._flags:
            agent_report["discrepancies"] = self._flags
        if agent_report["total_itc"] == 0.0 and self._itc_results:
            agent_report["total_itc"] = sum(
                r.get("itc_amount", 0) for r in self._itc_results
            )

        all_ids = set(self._invoice_index.keys())
        truth_discrepancies = {}
        for inv_id, status in self._ground_truth.get("eligibility", {}).items():
            if status != "eligible":
                truth_discrepancies[inv_id] = {
                    "status": status,
                    "action": get_recommended_action(status),
                }

        ground_truth = {
            "total_itc": self._ground_truth_itc,
            "discrepancies": truth_discrepancies,
            "all_invoice_ids": all_ids,
        }

        result = grader_full_recon.grade(
            agent_report, ground_truth, self._step_number, self._max_steps
        )

        score = result["total"]
        self._score_history[self.task_id] = score
        return score, {"score": score, "grader": "full_recon", "breakdown": result}

    # ── Internal Helpers ─────────────────────────────────────────────

    def _precompute_ground_truth(self) -> None:
        """Fix #29: pre-compute all ground truth at reset time."""
        # Match status (Task 1)
        match_status = {}
        for inv in self.purchase_register:
            inv_id = inv["invoice_id"]
            match_status[inv_id] = (
                "present" if inv_id in self._gstr2b_index else "missing"
            )

        # Eligibility (Task 2 & 3)
        eligibility = {}
        for inv in self.purchase_register:
            inv_id = inv["invoice_id"]
            gstr2b_rec = self._gstr2b_index.get(inv_id)
            eligibility[inv_id] = check_itc_eligibility(inv, gstr2b_rec)

        # Total ITC (Task 3)
        total_itc, _ = compute_total_eligible_itc(
            self.purchase_register, self._gstr2b_index
        )

        self._ground_truth = {
            "match_status": match_status,
            "eligibility": eligibility,
        }
        self._ground_truth_itc = total_itc

    def _build_observation(self) -> GSTObservation:
        """Build current observation from internal state."""
        matched_ids = {m["invoice_id"] for m in self._matches}
        flagged_ids = {f["invoice_id"] for f in self._flags}
        resolved_ids = matched_ids | flagged_ids
        total_ids = {inv["invoice_id"] for inv in self.purchase_register}
        unresolved = len(total_ids - resolved_ids)

        return GSTObservation(
            session_id=self.session_id,
            task_id=self.task_id,
            purchase_register=[Invoice(**inv) for inv in self.purchase_register],
            gstr2b_data=[Invoice(**inv) for inv in self.gstr2b_data],
            current_matches=deepcopy(self._matches),
            unresolved_count=unresolved,
            step_number=self._step_number,
            max_steps=self._max_steps,
            last_action_error=self._last_error,
        )

    def _log_action(self, action: GSTAction, reward: float, info: dict) -> None:
        """Append action to replay log."""
        self.replay_log.append(
            {
                "step": self._step_number,
                "action_type": action.action_type,
                "invoice_id": action.invoice_id,
                "reason": action.reason,
                "reward": reward,
                "done": self._done,
                "info_keys": list(info.keys()),
            }
        )
