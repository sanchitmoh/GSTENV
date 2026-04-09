---
title: GST Agent Environment
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# GST Agent Environment

OpenEnv-compatible RL environment for Indian GST reconciliation.

## Features

- Invoice matching and reconciliation
- ITC (Input Tax Credit) audit
- Multi-agent orchestration with RAG
- Deterministic grading system

## API Endpoints

- `GET /health` - Health check
- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state/{session_id}` - Get current state
- `GET /leaderboard` - View leaderboard
