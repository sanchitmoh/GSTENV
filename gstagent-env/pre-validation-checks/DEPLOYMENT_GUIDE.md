# 🚀 GST Agent Environment - Complete Deployment Guide

**Last Updated**: April 8, 2026  
**Project**: GST Agent Environment (gstagent-env)

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Docker Deployment](#local-docker-deployment)
3. [HuggingFace Space Deployment](#huggingface-space-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Testing & Validation](#testing--validation)
6. [Troubleshooting](#troubleshooting)
7. [Production Checklist](#production-checklist)

---

## 🔧 Prerequisites

### Required Software

1. **Docker Desktop** (v20.10+)
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - Mac: https://docs.docker.com/desktop/install/mac-install/
   - Linux: https://docs.docker.com/desktop/install/linux-install/

2. **Git** (v2.30+)
   - Download: https://git-scm.com/downloads

3. **Python** (v3.11+) - for local testing
   - Download: https://www.python.org/downloads/

4. **HuggingFace Account**
   - Sign up: https://huggingface.co/join

### Required API Keys

1. **OpenAI API Key** (for LLM agents)
   - Get it: https://platform.openai.com/api-keys
   - Alternative: Google Gemini API key

2. **HuggingFace Token** (for Space deployment)
   - Get it: https://huggingface.co/settings/tokens
   - Permissions needed: `write` access

---

## 🐳 Local Docker Deployment

### Step 1: Prepare Environment Variables

Create a `.env` file in the `gstagent-env` directory:

```bash
cd gstagent-env
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Required
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Optional - adjust as needed
API_BASE_URL=http://localhost:7860
MODEL_NAME=gpt-4
FAST_MODEL_NAME=gpt-3.5-turbo
DATA_SEED=42
```

### Step 2: Build Docker Image

```bash
# From the project root (C:\HACKTHAON)
docker build -t gstagent-env:latest gstagent-env

# Verify the build
docker images gstagent-env
```

**Expected Output**:
```
REPOSITORY     TAG       IMAGE ID       CREATED          SIZE
gstagent-env   latest    331990fc42bf   2 minutes ago    390MB
```

### Step 3: Run Docker Container

#### Option A: Run with Environment File

```bash
docker run -d \
  --name gstagent-server \
  -p 7860:7860 \
  --env-file gstagent-env/.env \
  gstagent-env:latest
```

#### Option B: Run with Inline Environment Variables

```bash
docker run -d \
  --name gstagent-server \
  -p 7860:7860 \
  -e OPENAI_API_KEY=sk-your-key-here \
  -e MODEL_NAME=gpt-4 \
  -e FAST_MODEL_NAME=gpt-3.5-turbo \
  gstagent-env:latest
```

#### Option C: Run with Volume Mount (for development)

```bash
docker run -d \
  --name gstagent-server \
  -p 7860:7860 \
  --env-file gstagent-env/.env \
  -v ${PWD}/gstagent-env:/app \
  gstagent-env:latest
```

### Step 4: Verify Container is Running

```bash
# Check container status
docker ps

# View logs
docker logs gstagent-server

# Follow logs in real-time
docker logs -f gstagent-server
```

**Expected Log Output**:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
```

### Step 5: Test the API

```bash
# Test health endpoint
curl http://localhost:7860/health

# Expected response: {"status":"ok"}

# Test reset endpoint
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "invoice_match"}'
```

### Step 6: Run Inference Script

```bash
# From host machine (not inside container)
cd gstagent-env
python inference.py
```

**Expected Output**:
```
[START] task=invoice_match env=gstagent-env model=gpt-4
[STEP] step=1 action=match_invoice(...) reward=0.05 done=false error=null
...
[END] success=true steps=8 score=0.850 rewards=0.05,0.05,...
```

### Step 7: Stop and Clean Up

```bash
# Stop container
docker stop gstagent-server

# Remove container
docker rm gstagent-server

# Remove image (if needed)
docker rmi gstagent-env:latest
```

---

## 🤗 HuggingFace Space Deployment

### Step 1: Create HuggingFace Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `gstagent-env` (or your preferred name)
   - **License**: `apache-2.0`
   - **SDK**: Select **Docker**
   - **Visibility**: Public or Private
3. Click **Create Space**

### Step 2: Prepare Repository for HF Space

HuggingFace Spaces expect a specific structure. You need to add a `README.md` at the root with YAML frontmatter:

Create `gstagent-env/README_HF.md`:

```markdown
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
- `GET /replay/{session_id}` - Replay episode

## Usage

```python
import requests

# Reset environment
response = requests.post(
    "https://your-space.hf.space/reset",
    json={"task_id": "invoice_match"}
)
session_id = response.json()["session_id"]

# Execute action
response = requests.post(
    "https://your-space.hf.space/step",
    json={
        "session_id": session_id,
        "action": {
            "action_type": "match_invoice",
            "invoice_id": "INV-001"
        }
    }
)
```

## Documentation

See [implementation_plan.md](implementation_plan.md) for architecture details.
```

### Step 3: Configure HuggingFace Space Secrets

1. Go to your Space settings: `https://huggingface.co/spaces/YOUR_USERNAME/gstagent-env/settings`
2. Scroll to **Repository secrets**
3. Add the following secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `OPENAI_API_KEY` | `sk-your-key-here` | OpenAI API key for LLM agents |
| `GST_API_KEY` | `your-secret-key` | Optional: API authentication |
| `ALLOWED_ORIGINS` | `*` | CORS origins (use specific domains in production) |

### Step 4: Push to HuggingFace Space

#### Option A: Using Git (Recommended)

```bash
cd gstagent-env

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/gstagent-env

# Copy README for HF Space
cp README_HF.md README.md

# Commit and push
git add .
git commit -m "Initial deployment to HuggingFace Space"
git push hf main
```

#### Option B: Using HuggingFace CLI

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload files
huggingface-cli upload YOUR_USERNAME/gstagent-env ./gstagent-env --repo-type=space
```

### Step 5: Monitor Deployment

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/gstagent-env`
2. Click on **Logs** tab to monitor build progress
3. Wait for build to complete (5-10 minutes)

**Build Stages**:
```
Building Docker image...
[+] Building 97.8s (20/20) FINISHED
=> [builder 4/4] RUN pip install...
=> [runtime 11/11] RUN chown -R appuser...
=> exporting to image
Build complete!
Starting container...
Application startup complete.
```

### Step 6: Verify Deployment

```bash
# Test your deployed Space
curl https://YOUR_USERNAME-gstagent-env.hf.space/health

# Expected: {"status":"ok"}
```

### Step 7: Update Space Configuration (Optional)

Edit your Space settings at `https://huggingface.co/spaces/YOUR_USERNAME/gstagent-env/settings`:

- **Hardware**: Upgrade to CPU Basic or GPU if needed
- **Sleep time**: Set to "Never" for always-on (paid feature)
- **Visibility**: Change to Private if needed

---

## ⚙️ Environment Configuration

### Required Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | OpenAI API key for LLM agents |
| `API_BASE_URL` | No | `http://localhost:7860` | Base URL for API |
| `MODEL_NAME` | No | `gpt-4` | Primary model for reasoning |
| `FAST_MODEL_NAME` | No | `gpt-3.5-turbo` | Fast model for classification |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_SEED` | `42` | Seed for deterministic data generation |
| `SESSION_TTL_SECONDS` | `1800` | Session timeout (30 minutes) |
| `RESET_TIMEOUT_SECONDS` | `60` | Timeout for /reset endpoint |
| `STEP_TIMEOUT_SECONDS` | `30` | Timeout for /step endpoint |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `GST_API_KEY` | Auto-generated | API authentication key |

### Setting Environment Variables

#### Docker (Local)

```bash
# Using .env file
docker run --env-file .env gstagent-env:latest

# Using -e flag
docker run -e OPENAI_API_KEY=sk-xxx gstagent-env:latest
```

#### HuggingFace Space

1. Go to Space Settings → Repository secrets
2. Add each variable as a secret
3. Restart the Space

#### Docker Compose (Advanced)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  gstagent:
    build: ./gstagent-env
    ports:
      - "7860:7860"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MODEL_NAME=gpt-4
      - FAST_MODEL_NAME=gpt-3.5-turbo
    env_file:
      - ./gstagent-env/.env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up -d
```

---

## 🧪 Testing & Validation

### Pre-Deployment Validation

Run the validation script before deploying:

```bash
cd gstagent-env/pre-validation-checks

# Make executable (Linux/Mac)
chmod +x pre-validation.sh

# Run validation (replace with your Space URL)
./pre-validation.sh https://YOUR_USERNAME-gstagent-env.hf.space
```

**Validation Checks**:
1. ✅ HF Space is live and responds to /reset
2. ✅ Docker build succeeds
3. ✅ openenv validate passes

### Manual Testing

#### 1. Health Check
```bash
curl http://localhost:7860/health
# Expected: {"status":"ok"}
```

#### 2. Reset Endpoint
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "invoice_match"}'
```

#### 3. Step Endpoint
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "SESSION_ID_FROM_RESET",
    "action": {
      "action_type": "match_invoice",
      "invoice_id": "INV-001"
    }
  }'
```

#### 4. Run Full Inference
```bash
cd gstagent-env
python inference.py
```

#### 5. Run Advanced Multi-Agent
```bash
python inference_advanced.py
```

### Automated Testing

```bash
# Run all tests
cd gstagent-env
pytest tests/ -v

# Run specific test suites
pytest tests/test_api.py -v
pytest tests/test_env.py -v
pytest tests/test_agents.py -v

# Run smoke tests
pytest tests/smoke_test.py -v

# Run with coverage
pytest tests/ --cov=environment --cov-report=html
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Docker Build Fails

**Error**: `ERROR: failed to solve: process "/bin/sh -c pip install..."`

**Solution**:
```bash
# Clear Docker cache
docker builder prune -a

# Rebuild without cache
docker build --no-cache -t gstagent-env:latest gstagent-env
```

#### 2. Container Exits Immediately

**Error**: Container stops right after starting

**Solution**:
```bash
# Check logs
docker logs gstagent-server

# Common causes:
# - Missing OPENAI_API_KEY
# - Port 7860 already in use
# - Syntax error in code

# Fix port conflict
docker run -p 8080:7860 gstagent-env:latest
```

#### 3. OpenAI API Key Not Working

**Error**: `openai.error.AuthenticationError: Incorrect API key`

**Solution**:
```bash
# Verify key is set
docker exec gstagent-server env | grep OPENAI_API_KEY

# Re-run with correct key
docker stop gstagent-server
docker rm gstagent-server
docker run -e OPENAI_API_KEY=sk-correct-key-here gstagent-env:latest
```

#### 4. HuggingFace Space Build Fails

**Error**: Build fails on HF Space

**Solution**:
1. Check Logs tab for error details
2. Common issues:
   - Missing `README.md` with YAML frontmatter
   - Dockerfile not at root of repo
   - Missing required files (openenv.yaml, requirements.txt)
3. Test build locally first:
   ```bash
   docker build -t test gstagent-env
   ```

#### 5. Rate Limit Errors

**Error**: `429 Too Many Requests`

**Solution**:
```bash
# Increase rate limits in .env
RATE_LIMIT_REQUESTS=200
RATE_LIMIT_WINDOW=60

# Or disable rate limiting (not recommended for production)
# Comment out rate limiting in server.py
```

#### 6. Memory Issues

**Error**: Container killed due to OOM

**Solution**:
```bash
# Increase Docker memory limit
docker run -m 4g gstagent-env:latest

# Or optimize RAG engine
# Edit environment/config.py
# Reduce cache size, disable graph expansion
```

### Debug Mode

Enable debug logging:

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or run with debug flag
docker run -e LOG_LEVEL=DEBUG gstagent-env:latest
```

### Health Checks

```bash
# Check container health
docker inspect gstagent-server --format='{{.State.Health.Status}}'

# Check API health
curl http://localhost:7860/health

# Check metrics (if enabled)
curl http://localhost:7860/metrics
```

---

## ✅ Production Checklist

### Security

- [ ] Set strong `GST_API_KEY` in environment
- [ ] Configure specific `ALLOWED_ORIGINS` (not `*`)
- [ ] Enable HTTPS/TLS (use reverse proxy)
- [ ] Rotate API keys regularly
- [ ] Review security headers in `server.py`
- [ ] Enable rate limiting
- [ ] Set up monitoring and alerts

### Performance

- [ ] Test with load testing tool (locust, k6)
- [ ] Optimize RAG cache size
- [ ] Enable connection pooling
- [ ] Configure appropriate timeouts
- [ ] Monitor memory usage
- [ ] Set up horizontal scaling (if needed)

### Reliability

- [ ] Configure health checks
- [ ] Set up automatic restarts
- [ ] Implement circuit breakers
- [ ] Add retry logic with exponential backoff
- [ ] Configure session cleanup
- [ ] Set up backup/restore for leaderboard

### Monitoring

- [ ] Enable structured logging
- [ ] Set up log aggregation (ELK, Datadog)
- [ ] Configure metrics collection
- [ ] Set up dashboards (Grafana)
- [ ] Configure alerts for errors/latency
- [ ] Monitor API usage and costs

### Documentation

- [ ] Update README with deployment URL
- [ ] Document API endpoints
- [ ] Add usage examples
- [ ] Create runbook for operations
- [ ] Document troubleshooting steps

---

## 📚 Additional Resources

### Documentation
- [OpenEnv Docs](https://openenv.dev)
- [Docker Docs](https://docs.docker.com)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)
- [FastAPI Docs](https://fastapi.tiangolo.com)

### Tools
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Postman](https://www.postman.com) - API testing
- [k6](https://k6.io) - Load testing
- [Portainer](https://www.portainer.io) - Docker management

### Support
- GitHub Issues: Create issue in your repo
- HuggingFace Community: https://discuss.huggingface.co
- OpenEnv Discord: Check OpenEnv website

---

## 🎯 Quick Reference Commands

### Docker Commands
```bash
# Build
docker build -t gstagent-env:latest gstagent-env

# Run
docker run -d --name gstagent-server -p 7860:7860 --env-file .env gstagent-env:latest

# Logs
docker logs -f gstagent-server

# Stop
docker stop gstagent-server

# Remove
docker rm gstagent-server

# Shell access
docker exec -it gstagent-server bash
```

### HuggingFace Commands
```bash
# Login
huggingface-cli login

# Push to Space
git push hf main

# View logs
# Go to: https://huggingface.co/spaces/YOUR_USERNAME/gstagent-env
```

### Testing Commands
```bash
# Health check
curl http://localhost:7860/health

# Reset
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"invoice_match"}'

# Run inference
python inference.py

# Run tests
pytest tests/ -v
```

---

**Deployment Guide Version**: 1.0  
**Last Updated**: April 8, 2026  
**Maintained By**: GST Agent Team
