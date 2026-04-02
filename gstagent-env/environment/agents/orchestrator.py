"""
Orchestrator — coordinates the multi-agent GST reconciliation pipeline.

Manages the flow: Matcher → Auditor → Reporter → Validator → Submit.
Each agent receives the accumulated context from previous agents.
The orchestrator maps agent outputs to environment actions.
"""

from __future__ import annotations

import json
import os
import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from environment.agents.auditor import AuditorAgent
from environment.agents.base_agent import AgentMessage
from environment.agents.matcher import MatcherAgent
from environment.agents.reporter import ReporterAgent
from environment.agents.validator import ValidatorAgent
from environment.config import (
    API_BASE_URL,
    FAST_MODEL_NAME,
    MODEL_NAME,
    RESET_TIMEOUT_SECONDS,
    STEP_TIMEOUT_SECONDS,
)


class Orchestrator:
    """
    Multi-agent orchestrator for GST reconciliation.

    Pipeline: Matcher → Auditor → Reporter → Validator → Submit

    Each agent is a specialist:
    - Matcher: fast model (gpt-3.5), identifies present/missing/mismatch
    - Auditor: accurate model (gpt-4), applies GST rules for ITC eligibility
    - Reporter: accurate model (gpt-4), compiles structured report
    - Validator: fast model (gpt-3.5), catches hallucinations before submit
    """

    def __init__(
        self,
        api_base_url: str | None = None,
        matcher_model: str | None = None,
        auditor_model: str | None = None,
        reporter_model: str | None = None,
        validator_model: str | None = None,
    ):
        self.api_base_url = api_base_url or API_BASE_URL
        self.matcher = MatcherAgent(model=matcher_model or FAST_MODEL_NAME)
        self.auditor = AuditorAgent(model=auditor_model or MODEL_NAME)
        self.reporter = ReporterAgent(model=reporter_model or MODEL_NAME)
        self.validator = ValidatorAgent(model=validator_model or FAST_MODEL_NAME)

        self.context: list[AgentMessage] = []
        self.session_id: str = ""
        self.step_count: int = 0
        self.scores: dict[str, float] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _api_reset(self, task_id: str) -> dict:
        resp = requests.post(
            f"{self.api_base_url}/reset",
            json={"task_id": task_id},
            timeout=RESET_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _api_step(self, action: dict) -> dict:
        resp = requests.post(
            f"{self.api_base_url}/step",
            json={"session_id": self.session_id, "action": action},
            timeout=STEP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()

    def run_task(self, task_id: str) -> float:
        """
        Run a complete task through the multi-agent pipeline.

        Returns the final score.
        """
        print(f"\n{'='*60}")
        print(f"  MULTI-AGENT ORCHESTRATOR — Task: {task_id}")
        print(f"{'='*60}")

        # Reset environment
        obs_data = self._api_reset(task_id)
        self.session_id = obs_data.get("session_id", "")
        self.context = []
        self.step_count = 0

        print(f"  Session: {self.session_id[:8]}...")
        print(f"  Invoices: {len(obs_data.get('purchase_register', []))}")
        print(f"  GSTR-2B: {len(obs_data.get('gstr2b_data', []))}")

        # ── Stage 1: Matcher ─────────────────────────────────────
        print(f"\n  📋 Stage 1: Matcher Agent")
        start = time.time()
        matcher_msgs = self.matcher.process(obs_data, self.context)
        self.context.extend(matcher_msgs)

        # Execute match actions
        for msg in matcher_msgs:
            if msg.action and msg.action.get("action_type") in ("match_invoice", "flag_mismatch"):
                result = self._api_step(msg.action)
                self.step_count += 1
                if result.get("done"):
                    score = result.get("info", {}).get("score", result.get("reward", 0))
                    print(f"  ✅ Done early at step {self.step_count}. Score: {score}")
                    return score

        match_time = time.time() - start
        summary = matcher_msgs[-1] if matcher_msgs else None
        if summary:
            print(f"    {summary.content}")
        print(f"    Time: {match_time:.2f}s")

        # ── Stage 2: Auditor ─────────────────────────────────────
        print(f"\n  🔍 Stage 2: Auditor Agent")
        start = time.time()
        auditor_msgs = self.auditor.process(obs_data, self.context)
        self.context.extend(auditor_msgs)

        # Execute ITC computation
        for msg in auditor_msgs:
            if msg.action and msg.action.get("action_type") == "compute_itc":
                result = self._api_step(msg.action)
                self.step_count += 1

        audit_time = time.time() - start
        summary = auditor_msgs[-1] if auditor_msgs else None
        if summary:
            print(f"    {summary.content}")
        print(f"    Time: {audit_time:.2f}s")

        # ── Stage 3: Reporter ────────────────────────────────────
        print(f"\n  📊 Stage 3: Reporter Agent")
        start = time.time()
        reporter_msgs = self.reporter.process(obs_data, self.context)
        self.context.extend(reporter_msgs)
        report_time = time.time() - start
        if reporter_msgs:
            print(f"    {reporter_msgs[0].content}")
        print(f"    Time: {report_time:.2f}s")

        # ── Stage 4: Validator ───────────────────────────────────
        print(f"\n  ✅ Stage 4: Validator Agent")
        start = time.time()
        validator_msgs = self.validator.process(obs_data, self.context)
        self.context.extend(validator_msgs)
        val_time = time.time() - start
        if validator_msgs:
            print(f"    {validator_msgs[0].content}")
        print(f"    Time: {val_time:.2f}s")

        # ── Stage 5: Submit ──────────────────────────────────────
        print(f"\n  🚀 Stage 5: Submitting Report")
        submit_action = None
        for msg in reversed(validator_msgs):
            if msg.action and msg.action.get("action_type") == "submit_report":
                submit_action = msg.action
                break

        if not submit_action:
            # Fallback: use reporter's submission
            for msg in reversed(reporter_msgs):
                if msg.action and msg.action.get("action_type") == "submit_report":
                    submit_action = msg.action
                    break

        if not submit_action:
            print("    ⚠️ No submit action found. Submitting empty report.")
            submit_action = {
                "action_type": "submit_report",
                "payload": {"total_itc": 0.0, "discrepancies": []},
            }

        result = self._api_step(submit_action)
        self.step_count += 1
        score = result.get("info", {}).get("score", result.get("reward", 0))
        breakdown = result.get("info", {}).get("breakdown", {})

        print(f"    Score: {score}")
        if breakdown:
            print(f"    Breakdown: {json.dumps(breakdown, indent=6)}")
        print(f"    Total steps: {self.step_count}")
        print(f"    Total time: {match_time + audit_time + report_time + val_time:.2f}s")

        self.scores[task_id] = score
        return score

    def run_all_tasks(self) -> dict[str, float]:
        """Run all 3 tasks and return scores."""
        print("🏁 Multi-Agent GST Reconciliation — Starting All Tasks")
        print(f"   API: {self.api_base_url}\n")

        for task_id in ["invoice_match", "itc_audit", "full_recon"]:
            try:
                self.run_task(task_id)
            except Exception as e:
                print(f"  ❌ Error in {task_id}: {e}")
                self.scores[task_id] = 0.0

        # Final summary
        print(f"\n{'='*60}")
        print("📊 FINAL SCORES (Multi-Agent)")
        print(f"{'='*60}")
        for task, score in self.scores.items():
            print(f"  {task:20s} : {score:.4f}")
        avg = sum(self.scores.values()) / max(len(self.scores), 1)
        print(f"  {'Average':20s} : {avg:.4f}")

        return self.scores
