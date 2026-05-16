#!/usr/bin/env bash
# Spooknix Brev one-shot launcher — provisioning + smoke + ready message.
#
# Use when you just spun up a Brev box and want to be in `spooknix interview`
# 60 seconds later, without copy-pasting half the BREV.md.
#
# Idempotent: safe to re-run; will only re-copy .env files if they don't exist,
# and `docker compose up -d` is a no-op when containers are already healthy.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

say()   { printf '\033[1;36m▶\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m!\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m✗\033[0m %s\n' "$*"; exit 1; }

ensure_env_files() {
  if [ ! -f .env ]; then
    say "creating .env from .env.example"
    cp .env.example .env
    ok ".env created — review LLM_MODEL / TTS_VOICE before running interview"
  else
    ok ".env exists"
  fi
  if [ ! -f .env.brev ]; then
    say "creating .env.brev from .env.brev.example"
    cp .env.brev.example .env.brev
    warn ".env.brev placeholders need real LLM_IMAGE / TTS_IMAGE before compose"
    say "edit .env.brev then re-run: bash scripts/brev-launch.sh"
    exit 1
  else
    ok ".env.brev exists"
  fi
}

load_envs() {
  set -a
  # shellcheck disable=SC1091
  source .env
  # shellcheck disable=SC1091
  source .env.brev
  set +a
  ok "env vars loaded"
}

bring_up_stack() {
  say "starting STT + LLM + TTS workers"
  docker compose \
    -f docker-compose.yml \
    -f docker-compose.brev.yml \
    -f docker-compose.workers.yml \
    up -d
  ok "compose up complete"
}

wait_for_stt() {
  say "waiting for STT model load (up to 120s)"
  for _ in $(seq 1 60); do
    if curl -fsS "${SPOOKNIX_URL:-http://localhost:8000}/health" >/dev/null 2>&1; then
      ok "STT /health responding"
      return 0
    fi
    sleep 2
  done
  fail "STT did not become healthy in 120s — check: docker logs spooknix"
}

run_smoke() {
  say "running brev-smoke"
  bash scripts/brev-smoke.sh
}

print_next_steps() {
  cat <<'EOF'

──────────────────────────────────────────────────────────────────
 You're ready. Suggested next commands:

   spooknix interview --persona sarah --scenario behavioral --difficulty hard
   spooknix summarize lecture.mp4 --template lecture
   spooknix doctor --brev                  # re-check anytime

 If you change .env.brev images, re-run this script.
──────────────────────────────────────────────────────────────────
EOF
}

main() {
  say "Spooknix Brev launcher — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  ensure_env_files
  load_envs
  bring_up_stack
  wait_for_stt
  run_smoke
  print_next_steps
}

main "$@"
