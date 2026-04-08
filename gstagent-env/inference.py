"""
Baseline inference script for GST Agent Environment.

Features:
- ReAct pattern: Thought → Action → Observation (Fix #31)
- OpenAI function calling / tool_use schemas (Fix #34)
- Retry with tenacity: 3 attempts, exponential backoff (Fix #24)
- Timeout guard: 1140 seconds (19 min buffer) (Fix #25)
- Runs all 3 tasks and prints scores

Must be at project root (not inside subfolder) — judges check this path.
Must complete in under 20 minutes on a normal laptop.

STDOUT FORMAT (MANDATORY):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

from environment.config import (
    API_BASE_URL,
    INFERENCE_TIMEOUT_SECONDS,
    MODEL_NAME,
    OPENAI_API_KEY,
    RESET_TIMEOUT_SECONDS,
    STEP_TIMEOUT_SECONDS,
)

# HF_TOKEN / API_KEY fallback (mandatory per submission spec)
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or OPENAI_API_KEY
BENCHMARK = "gstagent-env"

# Fix #25: Timeout guard
TIMEOUT_SECONDS = INFERENCE_TIMEOUT_SECONDS


def timeout_handler(signum, frame):
    print("\n⏰ TIMEOUT: 19 minutes exceeded. Exiting.")
    sys.exit(1)


# Set timeout (Unix only; on Windows we use time-based check)
if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

START_TIME = time.time()


def check_timeout():
    """Windows-compatible timeout check."""
    if time.time() - START_TIME > TIMEOUT_SECONDS:
        print("\n⏰ TIMEOUT: 19 minutes exceeded. Exiting.")
        sys.exit(1)


# ── Mandatory STDOUT Logging (OpenEnv submission format) ─────────────


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line per step, immediately after env.step() returns."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line after episode, always emitted (even on exception)."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── OpenAI Function Calling Schemas (Fix #34) ────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "match_invoice",
            "description": "Check if an invoice from purchase register exists in GSTR-2B. Returns match status and any amount variance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "The invoice ID to match, e.g. 'INV-0001'",
                    }
                },
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_mismatch",
            "description": "Flag an invoice that has a discrepancy (missing from GSTR-2B or amount mismatch). Include the reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "The invoice ID to flag",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for flagging: 'missing_from_gstr2b', 'amount_mismatch', etc.",
                    },
                },
                "required": ["invoice_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_itc",
            "description": "Calculate ITC eligibility for all invoices based on GSTR-2B matching and GST rules.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "Submit the final reconciliation report. Include total ITC amount and list of discrepancies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_itc": {
                        "type": "number",
                        "description": "Total eligible ITC amount in INR",
                    },
                    "discrepancies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "invoice_id": {"type": "string"},
                                "status": {"type": "string"},
                                "action": {"type": "string"},
                            },
                        },
                        "description": "List of discrepancies found",
                    },
                    "matches": {
                        "type": "object",
                        "description": "Map of invoice_id to 'present' or 'missing'",
                    },
                    "decisions": {
                        "type": "object",
                        "description": "Map of invoice_id to eligibility status",
                    },
                },
            },
        },
    },
]

# ── API Helpers ──────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def api_reset(task_id: str) -> dict:
    """POST /reset with retry (Fix #24)."""
    resp = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=RESET_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def api_step(session_id: str, action: dict) -> dict:
    """POST /step with retry (Fix #24)."""
    resp = requests.post(
        f"{API_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=STEP_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(messages: list[dict]) -> dict:
    """Call LLM with tools and retry (Fix #24)."""
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,
    )
    return response


# ── ReAct Agent Loop (Fix #31) ───────────────────────────────────────

SYSTEM_PROMPT = """You are a GST reconciliation agent for an Indian MSME.

Your job is to reconcile the company's purchase register with GSTR-2B data from the government portal.

## Available Tools:
1. match_invoice(invoice_id) — Check if an invoice exists in GSTR-2B
2. flag_mismatch(invoice_id, reason) — Flag discrepancies
3. compute_itc() — Calculate ITC eligibility for all invoices
4. submit_report(total_itc, discrepancies, matches, decisions) — Submit final report

## Strategy:
1. First, match each invoice from the purchase register against GSTR-2B
2. Flag any invoices that are missing or have amount mismatches
3. Compute ITC eligibility
4. Submit a complete report with total eligible ITC and all discrepancies

## Rules (Rule 36(4), Section 16(2)):
- If invoice is MISSING from GSTR-2B → ineligible for ITC → action: follow up with supplier
- If amount differs by >20% → ineligible → action: follow up with supplier
- If amount differs by 5-20% → partial ITC → action: claim matched amount only
- If amount matches within 5% → eligible → action: claim full ITC

Be methodical. Process invoices one by one. Do NOT invent invoice IDs that don't exist.
"""


def run_task(task_id: str) -> tuple[float, int, list[float]]:
    """Run a single task with ReAct agent loop.

    Returns: (score, steps_taken, rewards_list)
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    rewards: list[float] = []
    steps_taken = 0

    # Reset environment
    obs_data = api_reset(task_id)
    session_id = obs_data.get("session_id", "")

    invoices = obs_data.get("purchase_register", [])
    gstr2b = obs_data.get("gstr2b_data", [])
    max_steps = obs_data.get("max_steps", 20)

    print(f"Session: {session_id[:8]}...")
    print(f"Invoices: {len(invoices)} | GSTR-2B: {len(gstr2b)} | Max steps: {max_steps}")

    # Emit [START]
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Build context for LLM
    invoice_summary = json.dumps(
        [{"id": i["invoice_id"], "amount": i["taxable_amount"], "supplier": i["supplier_gstin"][:6] + "..."}
         for i in invoices[:15]],  # Limit to avoid token overflow
        indent=2,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Task: {task_id}\n\nPurchase Register ({len(invoices)} invoices):\n{invoice_summary}\n\nGSTR-2B has {len(gstr2b)} entries.\nUnresolved: {obs_data.get('unresolved_count', 0)}\n\nStart reconciliation. Process each invoice.",
        },
    ]

    # Agent loop
    for step_num in range(max_steps):
        check_timeout()

        try:
            response = call_llm(messages)
        except Exception as e:
            print(f"  LLM error at step {step_num + 1}: {e}")
            break

        choice = response.choices[0]

        # Check if LLM wants to use a tool
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"  Step {step_num + 1}: {fn_name}({json.dumps(fn_args)[:80]})")

                # Map to env action
                action = {
                    "action_type": fn_name,
                    "invoice_id": fn_args.get("invoice_id"),
                    "reason": fn_args.get("reason"),
                    "payload": fn_args if fn_name == "submit_report" else None,
                }

                result = api_step(session_id, action)

                reward = result.get("reward", 0)
                done = result.get("done", False)
                info = result.get("info", {})
                error = info.get("error", None)

                steps_taken = step_num + 1
                rewards.append(reward)

                # Emit [STEP]
                action_str = f"{fn_name}({json.dumps(fn_args)[:60]})"
                log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

                # Feed result back to LLM
                messages.append(choice.message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "unresolved": result.get("observation", {}).get("unresolved_count", 0),
                    }),
                })

                if done:
                    score = info.get("score", reward)
                    score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
                    print(f"  ✅ Done! Score: {score}")
                    return score, steps_taken, rewards
        else:
            # LLM responded with text (thinking) — add and continue
            messages.append({"role": "assistant", "content": choice.message.content or ""})
            # Nudge it to act
            messages.append({"role": "user", "content": "Please use one of the available tools to continue."})

    # If we reach here, max steps exhausted without submitting
    print("  ⚠️ Max steps reached. Submitting empty report.")
    action = {
        "action_type": "submit_report",
        "payload": {"total_itc": 0.0, "discrepancies": []},
    }
    result = api_step(session_id, action)
    reward = result.get("reward", 0)
    score = result.get("info", {}).get("score", reward)
    score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
    steps_taken += 1
    rewards.append(reward)
    log_step(step=steps_taken, action="submit_report({})", reward=reward, done=True, error=None)
    print(f"  Score: {score}")
    return score, steps_taken, rewards


# ── Main ─────────────────────────────────────────────────────────────

def main():
    """Run all 3 tasks and print scores."""
    print("🏁 GST Agent Environment — Baseline Inference")
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print()

    if not API_KEY:
        print("⚠️ Warning: No API key set (HF_TOKEN / API_KEY / OPENAI_API_KEY). LLM calls will fail.")

    scores = {}
    for task_id in ["invoice_match", "itc_audit", "full_recon"]:
        task_rewards: list[float] = []
        task_steps = 0
        task_score = 0.0
        task_success = False

        try:
            task_score, task_steps, task_rewards = run_task(task_id)
            task_score = min(max(task_score, 0.0), 1.0)  # clamp to [0, 1]
            task_success = task_score > 0.0
        except Exception as e:
            print(f"  ❌ Error: {e}")
            task_score = 0.0
        finally:
            # [END] always emitted, even on exception
            log_end(success=task_success, steps=task_steps, score=task_score, rewards=task_rewards)

        scores[task_id] = task_score

    # Print final results
    print(f"\n{'='*60}")
    print("📊 Final Scores")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id:20s} : {score:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  {'Average':20s} : {avg:.4f}")
    print()

    elapsed = time.time() - START_TIME
    print(f"⏱️ Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
