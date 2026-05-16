# Spooknix on NVIDIA Brev

The fastest path to validate the full conversational + summarize suite on a
Brev GPU box without accidentally falling back to OpenAI.

## TL;DR (60-second path)

On a fresh Brev box with this repo cloned:

```bash
# 1. Edit .env.brev with your LLM_IMAGE and TTS_IMAGE
cp .env.brev.example .env.brev
$EDITOR .env.brev

# 2. One-shot launch (provisioning + workers up + smoke)
spooknix brev          # or: bash scripts/brev-launch.sh

# 3. Use it
spooknix interview --persona sarah --scenario behavioral --difficulty hard
spooknix summarize lecture.mp4 --template lecture
```

That's it. The launcher handles `.env` creation, `docker compose up`, waits
for the STT model to load, and runs `spooknix doctor --brev` + an
end-to-end TTS synthesize check.

---

## Recommended GPU budget

- `16 GB VRAM`: practical minimum for STT + local LLM + local TTS together
- `24 GB VRAM`: comfortable for stable end-to-end testing
- `6 GB VRAM`: enough for STT-only validation, not the full duplex stack

## Worker topology

Run the suite as 3 separate workers:

1. `STT` on `:8000` — ships in this repo (`spooknix` container)
2. `LLM` on `:8080` exposing OpenAI-compatible `/v1` — your image, your choice
3. `TTS` on `:8001` — your image, your choice (F5-TTS / XTTS / Piper / etc.)

This repo only ships the STT worker. LLM and TTS are expected to be provided
by your Brev runtime or companion containers — `docker-compose.workers.yml`
plus `.env.brev` lets you wire them in without touching code.

## Provisioning script

Brev runs `.brev/setup.sh` automatically when the workspace is created:

- installs `ffmpeg`, `libsndfile1`, `libopenblas-dev`, `jq`, `curl`
- installs the `docker compose` plugin if missing
- verifies the NVIDIA Container Toolkit can see the GPU
- optionally installs Nix (set `SPOOKNIX_BREV_INSTALL_NIX=1`)
- hands off to `scripts/brev-launch.sh`

## Compose layering (Nix vs Brev)

The base `docker-compose.yml` is tuned for NixOS hosts: it uses NVIDIA CDI
(`driver: cdi`) and bind-mounts `/var/lib/ml-models` to share model caches.
Neither exists on a vanilla Brev/Ubuntu box, so the deploy uses three layered
files:

```bash
docker compose \
  -f docker-compose.yml \         # base (NixOS-tuned)
  -f docker-compose.brev.yml \    # override — strips CDI, swaps to runtime: nvidia
  -f docker-compose.workers.yml \ # companion LLM + TTS containers
  up -d
```

`scripts/brev-launch.sh` already wires all three. On a NixOS host running
locally, drop the `-f docker-compose.brev.yml` line and you keep CDI.

The companion file is generic — pick a profile in `.env.brev.example` and copy
to `.env.brev`. The profiles (low / mid / high VRAM) provide concrete
`LLM_IMAGE`, `LLM_START_COMMAND`, `TTS_IMAGE`, `TTS_START_COMMAND` blocks
instead of placeholders.

## Environment

`.env` covers application config; `.env.brev` covers companion images.

Minimum variables for local-first interview mode:

```bash
export LLM_BASE_URL="http://localhost:8080/v1"
export LLM_MODEL="qwen-3.5"
export TTS_BASE_URL="http://localhost:8001"
export TTS_API_PATH="/tts"
export TTS_LANGUAGE="en"
```

Do not set `OPENAI_API_KEY` unless you explicitly want to use OpenAI.

## Diagnostics

`spooknix doctor --brev` prints a single table covering:

- CUDA + VRAM
- STT `/health`
- Audio devices (`sd.query_devices()`)
- ffmpeg presence
- LLM `/v1/models` with latency
- TTS `/health` (with `TTS_HEALTH_URL` override)

`bash scripts/brev-smoke.sh` runs doctor + an actual TTS synthesize POST,
because TTS images often answer `/health` while still failing on real
workloads. The script verifies the response is a valid RIFF WAV.

## Interview loop

```bash
spooknix interview --language en --persona sarah --scenario system_design --difficulty hard
```

Other CLI tools that work the same on Brev as locally:

- `spooknix summarize` — videos, lectures, meetings → markdown with `[mm:ss]` anchors
- `spooknix file` — single-file transcription with `large-v3-turbo`
- `spooknix record --vad-neural --meter` — direct STT from a forwarded mic
- `spooknix interview --list / --show <id> / --diff <a> <b>` — session history

## Tuning tips

- If the candidate gets cut off too early, increase `--silence` to `3.0`–`3.5`,
  or pass `--vad-neural` for Silero VAD (more robust than RMS).
- If the mic is noisy, start with `--threshold 0.03` to `0.05` or use Silero.
- Use headphones on Brev audio passthrough setups to reduce echo and false
  barge-in. PipeWire's AEC handles speaker-into-mic locally; over Brev's
  audio forwarding you don't get that, so headphones are a hard requirement.
- Validate each worker independently with `spooknix doctor --brev` before
  blaming the turn-taking loop.
- For long summaries on small LLMs, lower `--max-tokens` to `2000` so each
  chunk fits comfortably in the context window.
