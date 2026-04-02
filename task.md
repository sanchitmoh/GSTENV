# GST Agent Environment — Task Tracker

> Derived from [implementation_plan.md](file:///c:/HACKTHAON/implementation_plan.md) v3 (Merged)

---

## Phase 1 — Day 1: Project Scaffold & Setup

**Gate**: `openenv validate .` recognizes project structure

- [x] Create virtual environment: `python -m venv .venv` *(Fix #1)*
- [x] Create `.env.example` with variables: `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME`, `OPENAI_API_KEY` *(Fix #2)*
- [x] Create `.dockerignore` excluding `.git`, `.venv`, `tests/`, `__pycache__`, `.env`, `*.pyc` *(Fix #3)*
- [x] Create `.pre-commit-config.yaml` with `ruff`, `mypy`, `pytest` hooks *(Fix #4)*
- [x] Create project folder structure:
  - [x] `environment/__init__.py`
  - [x] `environment/data/__init__.py`
  - [x] `environment/rules/__init__.py`
  - [x] `environment/tasks/__init__.py`
  - [x] `environment/graders/__init__.py`
  - [x] `tests/` directory
- [x] Create `openenv.yaml` with name, version, description, tasks, author *(FR-10)*
- [x] Create `requirements.txt`
- [x] Create `requirements-dev.txt`
- [x] Create `.gitignore`
- [ ] Create GitHub repo `gstagent-env`
- [ ] Create HuggingFace Space (Docker SDK)
- [x] Verify: `pip install -r requirements.txt` completes without errors

---

## Phase 2 — Day 2: Data Generation & Rules Engine

**Gate**: JSON files generated; re-running produces identical output

- [x] Create `environment/data/hsn_codes.json` — real HSN subset *(Fix #7)*
- [x] Write `environment/data/generator.py` *(FR-01, FR-02, FR-03, FR-04)*
  - [x] `generate_gstin(state_code)` — valid format with Luhn checksum *(Fix #5)*
  - [x] `generate_invoice()` — realistic amounts, real HSN codes, same fiscal month dates *(Fix #7, #8)*
  - [x] `generate_gstr2b(invoice_list, missing_rate=0.3)` — remove 30%, alter 10% amounts
  - [x] Seed control: `random.seed(42)` + `Faker.seed_instance(42)` *(Fix #6)*
  - [x] Main block: generate and save 3 paired JSON files
- [x] Generate datasets:
  - [x] `environment/data/invoices_easy.json` — 10 invoices
  - [x] `environment/data/invoices_medium.json` — 20 invoices
  - [x] `environment/data/invoices_hard.json` — 50 invoices
- [x] Verify: run generator twice → output deterministic *(Fix #6)*
- [x] Write `environment/rules/gst_rules.py` *(FR-05)*
  - [x] `check_itc_eligibility(invoice, gstr2b_record)`
  - [x] `get_recommended_action(status)`
- [x] Verify: rules return correct status for each case

---

## Phase 3 — Day 3: Pydantic Models & Core Environment

**Gate**: `env.reset("invoice_match")` and `env.step(action)` work without errors ✅

- [x] Write `environment/models.py` *(FR-06, FR-07)*
  - [x] `class Invoice(BaseModel)` — all 10 fields typed
  - [x] `class GSTObservation(BaseModel)` — includes `session_id: str` *(Fix #9)*
  - [x] `class GSTAction(BaseModel)` — 4 action types
  - [x] `class GSTReward(BaseModel)` — total + component scores
  - [x] `class ResetRequest(BaseModel)`
  - [x] `class StepRequest(BaseModel)`
- [x] Write `environment/env.py` *(FR-06, FR-08, FR-09)*
  - [x] `reset(task_id)` — O(1) index build *(Fix #28)*, ground truth *(Fix #29)*
  - [x] `step(action)` — max_steps *(Fix #11)*, error recovery *(Fix #12)*, cache *(Fix #22)*, replay
  - [x] `state()` — returns `_cached_obs` *(Fix #22)*
  - [x] Intermediate reward: +0.05 per correct match
  - [x] Curriculum learning flag *(Fix #35)*
- [x] Verify: reset/step/invalid/max_steps all work ✅

---

## Phase 4 — Day 4: Tasks 1 & 2 + Graders

**Gate**: Both graders deterministic; `hypothesis` property tests pass ✅

- [x] Write `environment/tasks/task1_invoice_match.py` *(FR-10)*
- [x] Write `environment/graders/grader_invoice_match.py` *(Fix #13, #14)*
- [x] Write `environment/tasks/task2_itc_audit.py` *(FR-10)*
- [x] Write `environment/graders/grader_itc_audit.py` *(Fix #14, #15)*
- [x] Write `tests/test_graders.py` — perfect, wrong, empty, determinism
- [x] Write `tests/test_graders_property.py` *(Fix #32)* — hypothesis bounds
- [x] Run: `pytest tests/test_graders.py tests/test_graders_property.py` — ✅ passed

---

## Phase 5 — Day 5: Task 3 + Fixed Composite Reward

**Gate**: Reward always in `[0.0, 1.0]` — verified by hypothesis ✅

- [x] Write `environment/tasks/task3_full_recon.py` *(FR-10)*
- [x] Write `environment/graders/grader_full_recon.py` *(Fix #16, #17, #18)*
  - [x] `itc_accuracy` — proportional decay *(Fix #17)*
  - [x] `hallucination_penalty` — capped at 0.2 *(Fix #16)*
  - [x] `efficiency_bonus` *(Fix #18)*
  - [x] Final formula: `0.4×ITC + 0.3×recall + 0.2×actions - penalty + efficiency`
  - [x] Guard: empty input → 0.0 *(Fix #14)*
- [x] Wire graders into `env.py` `step()` for `submit_report`
- [x] Property test for `grader_full_recon` — ✅ all bounded
- [x] Run: `pytest tests/ -v` — **69 passed** ✅

---

## Phase 6 — Day 6: FastAPI Server + API Testing

**Gate**: All endpoints work; `openenv validate .` passes

- [x] Write `environment/server.py` *(Fix #10, #19-21, #30, #36)*
  - [x] Session manager: `sessions: dict[str, GSTAgentEnv]` *(Fix #10)*
  - [x] TTL cleanup: 30 minutes
  - [x] `POST /reset` — UUID session *(Fix #9)*
  - [x] `POST /step` — session-aware *(Fix #10)*
  - [x] `GET /state/{session_id}` — cached *(Fix #22)*
  - [x] `GET /health` — `{"status": "ok"}`
  - [x] `GET /leaderboard` *(Fix #36)*
  - [x] `GET /replay/{session_id}` *(Fix #36)*
  - [x] Async: `run_in_executor()` *(Fix #19)*
  - [x] Timeouts: 30s/60s *(Fix #20)*
  - [x] CORS *(Fix #21)*
  - [x] Structured logging: `structlog` JSON *(Fix #30)*
  - [x] Rate limiting: `slowapi`
- [x] Create `leaderboard.json`
- [x] Write `tests/test_api.py` — health, reset, step, state, isolation
- [ ] Run locally: `uvicorn environment.server:app --port 7860`
- [ ] Run: `openenv validate .`

---

## Phase 7 — Day 7: Docker + Inference + README + CI/CD

**Gate**: Docker < 200MB; inference completes < 19 min

- [x] Write `Dockerfile` — multi-stage + USER 1000 *(Fix #23)*
- [ ] Verify: `docker build -t gstagent .` completes
- [ ] Verify: `docker images gstagent` shows < 200MB
- [ ] Verify: `docker run` + `/health` returns 200
- [x] Write `inference.py` *(Fix #24, #25, #31, #34)*
  - [x] OpenAI function calling schemas *(Fix #34)*
  - [x] ReAct loop *(Fix #31)*
  - [x] Retry *(Fix #24)*
  - [x] Timeout guard *(Fix #25)*
- [ ] Verify: `python inference.py` runs (needs API key + server running)
- [/] Write `README.md`
- [/] Write `.github/workflows/ci.yml`
- [ ] Run full CI pipeline locally

---

## Phase 8 — Day 8: Deploy & Submit

**Gate**: Live HF Space URL returns 200; 3 non-zero scores

- [ ] Pre-submission checklist
- [ ] `git tag pre-submission` *(Fix #26)*
- [ ] Push to HuggingFace Spaces
- [ ] Verify live Space
- [ ] Submit

---

## Phase 9–12: Advancements (Post-Core)

- [ ] Phase 9: Agent Orchestration (multi-agent, LangGraph)
- [ ] Phase 10: RAG + Knowledge Base
- [ ] Phase 11: Observability + full test suite (>90% coverage)
- [ ] Phase 12: Production features (curriculum, security, CI/CD auto-deploy)

---

## Disqualification Checklist

- [x] Graders do NOT always return 0.5 or 1.0 — verified with tests
- [x] `inference.py` IS at project root
- [ ] HF Space does NOT return 503 — test after deploy
- [ ] `openenv validate .` passes — test after server starts
- [x] All 3 scores between 0.0–1.0, not identical — verified by grader tests
- [ ] README.md has all 6 required sections
