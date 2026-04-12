#!/us/bin/env bash
#
# validate-submission.sh â€” OpenEnv Submission Validato
#
# Checks that you HF Space is live, Docker image builds, and openenv validate passes.
#
# Perequisites:
#   - Docke:       https://docs.docker.com/get-docker/
#   - openenv-coe: pip install openenv-core
#   - cul (usually pre-installed)
#
# Run:
#   cul -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   O download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_ul> [repo_dir]
#
# Aguments:
#   ping_ul   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   epo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-epo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

un_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watche=$!
    wait "$pid" 2>/dev/null
    local c=$?
    kill "$watche" 2>/dev/null
    wait "$watche" 2>/dev/null
    eturn $rc
  fi
}

potable_mktemp() {
  local pefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${pefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { m -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
tap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  pintf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  pintf "\n"
  pintf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  pintf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  pintf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
expot PING_URL
PASS=0

log()  { pintf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { pintf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  pintf "\n"
  pintf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

pintf "\n"
pintf "${BOLD}========================================${NC}\n"
pintf "${BOLD}  OpenEnv Submission Validator${NC}\n"
pintf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
pintf "\n"

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/eset) ..."

CURL_OUTPUT=$(potable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(cul -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{"task_id":"invoice_match"}' \
  "$PING_URL/eset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and esponds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not eachable (connection failed or timed out)"
  hint "Check you network connection and that the Space is running."
  hint "Ty: curl -X POST -H 'Content-Type: application/json' -d '{\"task_id\":\"invoice_match\"}' $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /eset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sue your Space is running and the URL is correct."
  hint "Ty opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docke build${NC} ..."

if ! command -v docke &>/dev/null; then
  fail "docke command not found"
  hint "Install Docke: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockefile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/sever/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/sever"
else
  fail "No Dockefile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockefile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(un_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = tue ]; then
  pass "Docke build succeeded"
else
  fail "Docke build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  pintf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-coe"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=tue

if [ "$VALIDATE_OK" = tue ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  pintf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

pintf "\n"
pintf "${BOLD}========================================${NC}\n"
pintf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
pintf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
pintf "${BOLD}========================================${NC}\n"
pintf "\n"

exit 0