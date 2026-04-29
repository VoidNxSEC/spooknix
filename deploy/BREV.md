# Spooknix on NVIDIA Brev

This is the fastest path to validate the full conversational suite on a Brev GPU box without accidentally falling back to OpenAI.

## Recommended GPU budget

- `16 GB VRAM`: practical minimum for STT + local LLM + local TTS experiments
- `24 GB VRAM`: much safer for stable end-to-end testing
- `6 GB VRAM`: enough for partial validation only, not the whole full-duplex stack with comfort

## Worker topology

Run the suite as 3 separate workers:

1. `STT` on `:8000`
2. `LLM` on `:8080` exposing an OpenAI-compatible `/v1`
3. `TTS` on `:8001`

This repo only ships the STT worker. LLM and TTS are expected to be provided by your Brev runtime or companion containers.

## Environment bootstrap

```bash
cp .env.example .env
cp .env.brev.example .env.brev
set -a
source .env
source .env.brev
set +a
```

Minimum variables for local-first interview mode:

```bash
export LLM_BASE_URL="http://localhost:8080/v1"
export LLM_MODEL="qwen-3.5"
export TTS_BASE_URL="http://localhost:8001"
export TTS_API_PATH="/tts"
export TTS_LANGUAGE="en"
```

Do not set `OPENAI_API_KEY` unless you explicitly want to use OpenAI.

## Companion compose for LLM and TTS

This repo ships the STT worker directly and a companion compose file for the other 2 workers:

```bash
docker compose -f docker-compose.yml -f docker-compose.workers.yml up -d
```

The companion file is intentionally generic. You must fill these values in `.env.brev`:

- `LLM_IMAGE`
- `LLM_START_COMMAND`
- `TTS_IMAGE`
- `TTS_START_COMMAND`

That keeps the repo portable across different Brev images instead of hardcoding one backend you may not use.

## STT worker

```bash
docker compose up -d spooknix
curl -fsS http://localhost:8000/health
```

## Preflight

```bash
bash scripts/brev-smoke.sh
```

What it checks:

- `LLM_BASE_URL`, `LLM_MODEL`, and TTS base URL presence
- STT `/health`
- LLM `/v1/models`
- best-effort TTS `/health`

## Interview loop

```bash
nix develop --command spooknix interview --language en --silence 2.5 --threshold 0.03
```

If you are using the companion compose file, the full startup flow becomes:

```bash
cp .env.example .env
cp .env.brev.example .env.brev
set -a
source .env
source .env.brev
set +a

docker compose -f docker-compose.yml -f docker-compose.workers.yml up -d
bash scripts/brev-smoke.sh
nix develop --command spooknix interview --language en --silence 2.5 --threshold 0.03
```

## First-run tuning tips

- If the candidate gets cut off too early, increase `--silence` to `3.0` or `3.5`.
- If the mic is noisy, start with `--threshold 0.03` to `0.05`.
- Use headphones on Brev audio passthrough setups to reduce echo and false barge-in.
- Validate each worker independently before blaming the turn-taking loop.
