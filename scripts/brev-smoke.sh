#!/usr/bin/env bash
# Spooknix Brev smoke check — preflight before `spooknix interview`.
#
# Delegates static probes (STT, LLM, TTS health, audio devices, CUDA) to
# `spooknix doctor --brev`, then does an end-to-end TTS synthesize call
# because TTS images often respond on /health while still failing on
# real workloads.

set -euo pipefail

STT_URL="${SPOOKNIX_URL:-http://localhost:8000}"
LLM_URL="${LLM_BASE_URL:-}"
LLM_MODEL_NAME="${LLM_MODEL:-}"
TTS_URL="${TTS_BASE_URL:-${XTTS_BASE_URL:-${CHATTERBOX_BASE_URL:-${F5_TTS_URL:-}}}}"
TTS_API_PATH="${TTS_API_PATH:-/tts}"
TTS_VOICE="${TTS_VOICE:-default_voice}"
TTS_LANGUAGE="${TTS_LANGUAGE:-en}"

say()  { printf '%s\n' "$*"; }
fail() { say "FAIL: $*"; exit 1; }
warn() { say "WARN: $*"; }
pass() { say "OK:   $*"; }

check_required_env() {
  [ -n "$LLM_URL" ]        || fail "LLM_BASE_URL is not set."
  [ -n "$LLM_MODEL_NAME" ] || fail "LLM_MODEL is not set."
  [ -n "$TTS_URL" ]        || fail "TTS_BASE_URL (or XTTS_/CHATTERBOX_/F5_TTS_URL) is not set."
  pass "required env vars present"
}

run_doctor() {
  if command -v spooknix >/dev/null 2>&1; then
    say ""
    say "── spooknix doctor --brev ────────────────────────────────"
    spooknix doctor --brev
    say "──────────────────────────────────────────────────────────"
  else
    warn "spooknix CLI not on PATH — skipping doctor table"
  fi
}

tts_synthesize_end_to_end() {
  # Real synthesize round-trip: many TTS images respond on /health but
  # then 500 on the actual /tts endpoint. This catches that.
  local endpoint payload bytes
  endpoint="${TTS_URL%/}${TTS_API_PATH}"
  payload=$(printf '{"text":"hello","voice":"%s","language":"%s"}' "$TTS_VOICE" "$TTS_LANGUAGE")

  bytes=$(curl -fsS -m 30 -X POST -H 'Content-Type: application/json' \
              -d "$payload" -o /tmp/spooknix-tts-probe.wav -w '%{size_download}' \
              "$endpoint" 2>/dev/null) || {
    warn "TTS synthesize failed at $endpoint (image-specific endpoints differ)"
    return
  }

  if [ "${bytes:-0}" -lt 200 ]; then
    warn "TTS returned only $bytes bytes — likely an error payload, not WAV"
    return
  fi

  if head -c 4 /tmp/spooknix-tts-probe.wav | grep -q "RIFF"; then
    pass "TTS synthesize returned $bytes-byte RIFF WAV"
  else
    warn "TTS responded with $bytes bytes but missing RIFF header"
  fi
  rm -f /tmp/spooknix-tts-probe.wav
}

main() {
  say "Spooknix Brev smoke check"
  say "  STT: ${STT_URL%/}"
  say "  LLM: ${LLM_URL:-<unset>}"
  say "  TTS: ${TTS_URL:-<unset>}"

  check_required_env
  run_doctor
  say ""
  say "── end-to-end TTS synthesize ─────────────────────────────"
  tts_synthesize_end_to_end

  say ""
  pass "smoke check complete — try: nix develop --command spooknix interview"
}

main "$@"
