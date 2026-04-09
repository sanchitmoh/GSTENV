---
title: GST Agent Environment
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# GST Agent Environment

> **OpenEnv-compatible RL environment** simulating Indian GST reconciliation for MSMEs.
> An AI agent matches invoices, audits ITC eligibility, and produces full reconciliation reports — all graded by deterministic, multi-signal reward functions.

---

## 🗂️ What & Why

**63 million** GST-registered MSMEs in India spend 2–5 days every month manually reconciling their **Purchase Register** (internal books) against **GSTR-2B** (government auto-generated view from suppliers' GSTR-1 filings).

Unreconciled invoices → blocked Input Tax Credit (ITC) → **2–5% cash-flow loss**.

This environment trains AI agents to automate this workflow in under 20 minutes with >90% accuracy, encoding real GST rules (Rule 36(4), Section 16(2)) as deterministic Python logic.

---

## 📋 Observation Space

The agent receives a `GSTObservation` at each step:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | UUID identifying this episode session |
| `task_id` | `str` | Current task: `invoice_match`, `itc_audit`, `full_recon` |
| `purchase_register` | `list[Invoice]` | Invoices from the business's books |
| `gstr2b_data` | `list[Invoice]` | Invoices reflected in GSTR-2B (government view) |
| `current_matches` | `list[dict]` | Matches/flags recorded so far |
| `unresolved_count` | `int` | Invoices not yet matched or flagged |
| `step_number` | `int` | Current step in the episode |
| `max_steps` | `int` | Maximum steps allowed for this task |
| `last_action_error` | `str | None` | Error from last action, if any |

Each `Invoice` has: `invoice_id, supplier_gstin, buyer_gstin, invoice_date, taxable_amount, cgst, sgst, igst, hsn_code, item_description`.

---

## 🎮 Action Space

The agent sends a `GSTAction` with one of four action types:

| Action Type | Parameters | Description |
|---|---|---|
| `match_invoice` | `invoice_id` | Check if an invoice exists in GSTR-2B (O(1) lookup) |
| `flag_mismatch` | `invoice_id, reason` | Flag a discrepancy (missing/amount mismatch) |
| `compute_itc` | — | Run the rules engine to compute ITC eligibility for all invoices |
| `submit_report` | `payload: {total_itc, discrepancies, matches, decisions}` | Submit final report and receive grade |

---

## 📊 Tasks

| Task | Difficulty | Invoices | Max Steps | Goal |
|---|---|---|---|---|
| `invoice_match` | Easy | 10 | 8 | Classify each invoice as present/missing in GSTR-2B |
| `itc_audit` | Medium | 20 | 12 | Determine ITC eligibility (eligible/partial/ineligible) |
| `full_recon` | Hard | 50 | 20 | Full reconciliation: total ITC + discrepancy list + actions |

**Grading formula (Task 3):**
```
score = 0.4×ITC_accuracy + 0.3×recall + 0.2×action_correctness
        - hallucination_penalty(max 0.2) + efficiency_bonus(max 0.1)
→ clamped to [0.0, 1.0]
```

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Local Development

```bash
# Clone and enter project
git clone https://github.com/YOUR_USERNAME/gstagent-env.git
cd gstagent-env

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Generate synthetic data
python environment/data/generator.py

# Run the server
uvicorn environment.server:app --host 0.0.0.0 --port 7860

# Run tests
pytest tests/ -v

# Run baseline inference (requires server running + LLM API key)
python inference.py

# Run advanced multi-agent inference (RAG v2 + curriculum + observability)
python inference_advanced.py
```

### Docker

```bash
docker build -t gstagent .
docker run -p 7860:7860 --env-file .env gstagent
# Verify: curl http://localhost:7860/health
```

### HuggingFace Space

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/gstagent-env
git push hf main
```

---

## 📈 Baseline Scores

| Task | Baseline Score | Method |
|---|---|---|
| `invoice_match` | ~0.80 | GPT-4 + ReAct loop |
| `itc_audit` | ~0.70 | GPT-4 + function calling |
| `full_recon` | ~0.65 | GPT-4 + full pipeline |

### Advanced (Multi-Agent + RAG v2)

| Task | Advanced Score | Method |
|---|---|---|
| `invoice_match` | ~0.85+ | Multi-agent orchestrator + RAG v2 context |
| `itc_audit` | ~0.78+ | Auditor agent + ITC rules retrieval |
| `full_recon` | ~0.72+ | Full pipeline + curriculum + faithfulness |

*Scores may vary based on LLM model and parameters.*

---

## 🏗️ Architecture

```
gstagent-env/
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                 # Multi-stage, non-root
├── inference.py               # ReAct agent + function calling (baseline)
├── inference_advanced.py      # Multi-agent + RAG v2 + curriculum (advanced)
├── environment/
│   ├── server.py              # Async FastAPI + sessions
│   ├── env.py                 # RL environment (O(1) lookups)
│   ├── models.py              # Pydantic schemas
│   ├── data/generator.py      # Seeded data generation
│   ├── rules/gst_rules.py     # Rule 36(4) + Section 16(2)
│   ├── tasks/                 # 3 difficulty-tiered task configs
│   └── graders/               # Deterministic grading functions
└── tests/                     # 69 tests (unit + property + API)
```

---

## 🔄 Rollback

```bash
git tag pre-submission
# If something breaks after push:
git checkout pre-submission
```
