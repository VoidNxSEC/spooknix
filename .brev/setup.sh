#!/usr/bin/env bash
# Brev provisioning entry point — runs once when the workspace is created.
#
# This script does the OS-level prep that the brev-launch.sh script
# can't do on its own (installing nix, docker compose plugin, etc.),
# then hands off to brev-launch.sh for the application stack.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '[brev-setup] %s\n' "$*"; }

# ── System deps ─────────────────────────────────────────────────────────────
log "installing system deps (ffmpeg, libsndfile1, jq, curl)"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libopenblas-dev jq curl

# ── Docker compose plugin (Brev images usually ship the engine, not the plugin)
if ! docker compose version >/dev/null 2>&1; then
  log "installing docker compose plugin"
  sudo apt-get install -y docker-compose-plugin
fi

# ── NVIDIA Container Toolkit sanity (Brev usually ships it) ─────────────────
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  log "WARN: docker can't see the GPU — Brev box may be CPU-only or runtime broken"
fi

# ── Nix (optional — only if you want the dev shell on Brev) ─────────────────
if [ "${SPOOKNIX_BREV_INSTALL_NIX:-0}" = "1" ] && ! command -v nix >/dev/null 2>&1; then
  log "installing nix (multi-user)"
  curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install --no-confirm
fi

# ── Hand off to the application launcher ────────────────────────────────────
log "handing off to brev-launch.sh"
bash scripts/brev-launch.sh

log "setup complete"
