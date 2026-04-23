#!/usr/bin/env bash
# =============================================================================
# run_factors_tests.sh — one-shot launcher for the GreenLang Factors test suite
# =============================================================================
# Resolves the CTO-flagged "no reproducible test environment" gap by giving
# every developer (and every CI runner) a single command that gets the full
# `tests/factors/` suite green.
#
# Strategy:
#   1. If `docker` (and `docker compose` / `docker-compose`) is available,
#      bring up the dedicated factors-test stack — Postgres + Redis +
#      pytest runner — and abort on the first container exit so the script
#      surfaces the pytest exit code.
#   2. Otherwise fall back to a local install: detect / activate a venv,
#      install `pip install -e ".[factors-test]"` if pytest isn't found,
#      then run pytest directly.
#
# Usage:
#   bash scripts/run_factors_tests.sh                # full suite
#   bash scripts/run_factors_tests.sh tests/factors/billing -k "credits"
#
# Env vars:
#   GL_FACTORS_FORCE_LOCAL=1   # skip Docker even if available
#   GL_FACTORS_VENV=.venv      # override venv path
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/deployment/docker/docker-compose.factors-test.yml"
VENV_PATH="${GL_FACTORS_VENV:-${REPO_ROOT}/.venv}"

log() { printf "[factors-test] %s\n" "$*" >&2; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

run_with_docker() {
  log "Docker detected — launching reproducible test stack."
  log "Compose file: ${COMPOSE_FILE}"

  # Prefer modern `docker compose`, fall back to `docker-compose`.
  if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
  elif have_cmd docker-compose; then
    DC="docker-compose"
  else
    log "WARN: docker present but compose plugin missing — falling back to local."
    return 99
  fi

  if [ "$#" -gt 0 ]; then
    # Custom pytest args — do a one-shot run.
    log "Running custom pytest invocation: $*"
    $DC -f "${COMPOSE_FILE}" up -d postgres redis
    set +e
    $DC -f "${COMPOSE_FILE}" run --rm factors-test pytest "$@"
    rc=$?
    set -e
    $DC -f "${COMPOSE_FILE}" down -v
    return "${rc}"
  fi

  set +e
  $DC -f "${COMPOSE_FILE}" up --build --abort-on-container-exit --exit-code-from factors-test
  rc=$?
  set -e
  $DC -f "${COMPOSE_FILE}" down -v
  return "${rc}"
}

run_local() {
  log "Running locally (no Docker)."

  # Activate / create venv.
  if [ ! -d "${VENV_PATH}" ]; then
    log "Creating venv at ${VENV_PATH}"
    python3 -m venv "${VENV_PATH}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"

  if ! python -c "import pytest" 2>/dev/null; then
    log "Installing factors-test extras into ${VENV_PATH}"
    pip install --upgrade pip
    pip install -e "${REPO_ROOT}[factors-test]"
  fi

  cd "${REPO_ROOT}"
  if [ "$#" -gt 0 ]; then
    pytest "$@"
  else
    pytest tests/factors -v --maxfail=10 \
      --cov=greenlang.factors --cov-report=term-missing
  fi
}

main() {
  if [ "${GL_FACTORS_FORCE_LOCAL:-0}" = "1" ]; then
    run_local "$@"
    exit $?
  fi

  if have_cmd docker; then
    set +e
    run_with_docker "$@"
    rc=$?
    set -e
    if [ "${rc}" -eq 99 ]; then
      run_local "$@"
      exit $?
    fi
    exit "${rc}"
  fi

  run_local "$@"
}

main "$@"
