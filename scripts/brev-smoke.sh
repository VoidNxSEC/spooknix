#!/usr/bin/env bash

set -euo pipefail

STT_URL="${SPOOKNIX_URL:-http://localhost:8000}"
LLM_URL="${LLM_BASE_URL:-}"
LLM_MODEL_NAME="${LLM_MODEL:-}"
TTS_URL="${TTS_BASE_URL:-${XTTS_BASE_URL:-${CHATTERBOX_BASE_URL:-${F5_TTS_URL:-}}}}"
TTS_HEALTH_URL="${TTS_HEALTH_URL:-}"

say() {
  printf '%s\n' "$*"
}

fail() {
  say "FAIL: $*"
  exit 1
}

warn() {
  say "WARN: $*"
}

pass() {
  say "OK: $*"
}

check_required_env() {
  [ -n "$LLM_URL" ] || fail "LLM_BASE_URL is not set."
  [ -n "$LLM_MODEL_NAME" ] || fail "LLM_MODEL is not set."
  [ -n "$TTS_URL" ] || fail "TTS_BASE_URL (or XTTS_BASE_URL / CHATTERBOX_BASE_URL / F5_TTS_URL) is not set."
  pass "Required environment variables are present."
}

check_stt() {
  curl -fsS "${STT_URL%/}/health" >/dev/null || fail "STT health check failed at ${STT_URL%/}/health"
  pass "STT health check passed at ${STT_URL%/}/health"
}

check_llm() {
  curl -fsS "${LLM_URL%/}/models" >/dev/null || fail "LLM models endpoint failed at ${LLM_URL%/}/models"
  pass "LLM models endpoint passed at ${LLM_URL%/}/models"
}

check_tts() {
  local health_url

  if [ -n "$TTS_HEALTH_URL" ]; then
    health_url="$TTS_HEALTH_URL"
  else
    health_url="${TTS_URL%/}/health"
  fi

  if curl -fsS "$health_url" >/dev/null 2>&1; then
    pass "TTS health check passed at $health_url"
    return
  fi

  warn "TTS health endpoint did not respond at $health_url"
  warn "This is not a hard failure because TTS images are inconsistent here."
  warn "Confirm the worker manually before starting the interview loop."
}

main() {
  say "Spooknix Brev smoke check"
  say "STT: ${STT_URL%/}"
  say "LLM: ${LLM_URL:-<unset>}"
  say "TTS: ${TTS_URL:-<unset>}"
  say ""

  check_required_env
  check_stt
  check_llm
  check_tts

  say ""
  pass "Smoke check complete. You can now run: nix develop --command spooknix interview"
}

main "$@"
