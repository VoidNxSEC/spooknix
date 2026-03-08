# Spooknix — Privacy-first STT Engine

Transcrição de áudio e vídeo 100% local — nenhum dado sai da máquina.

Baseado em [faster-whisper](https://github.com/SYSTRAN/faster-whisper) com suporte CUDA.

---

## Requisitos

- Python 3.13+
- NVIDIA GPU com 4–6GB VRAM (ou CPU com fallback automático)
- ffmpeg (instalado via `nix develop` ou `apt install ffmpeg`)

---

## Setup

```bash
# Entrar no ambiente Nix (recomendado)
nix develop

# Instalar dependências Python
pip install -r requirements.txt
```

---

## CLI

### `stt info` — Status do sistema

```bash
python -m src.cli info
```

Exibe GPU detectada, VRAM disponível e modelos suportados.

---

### `stt file` — Transcrever arquivo

```bash
python -m src.cli file <audio_path> [opções]
```

**Opções:**

| Flag | Padrão | Descrição |
|---|---|---|
| `-l`, `--language` | `pt` | Código do idioma (`pt`, `en`, `es`, …) |
| `-m`, `--model` | `small` | Tamanho do modelo Whisper |
| `-o`, `--output-dir` | `outputs/` | Diretório de saída |
| `-f`, `--format` | `all` | Formato: `txt`, `srt`, `json`, `all` |

**Exemplos:**

```bash
# Transcrição completa com todos os formatos
python -m src.cli file sample.mp4

# Apenas legenda SRT, modelo medium para maior precisão
python -m src.cli file entrevista.mp3 --model medium --format srt

# Inglês, saída só JSON
python -m src.cli file podcast.m4a --language en --format json
```

**Saída gerada** (com `--format all`):
```
outputs/
├── transcripts/
│   ├── sample.txt    ← texto completo
│   └── sample.json   ← segments com timestamps
└── subtitles/
    └── sample.srt    ← legenda SubRip
```

---

## API HTTP

### Iniciar servidor

```bash
# Local
python -m src.server

# Com variáveis de ambiente
MODEL_SIZE=medium DEVICE=cuda python -m src.server

# Docker
docker compose up
```

**Variáveis de ambiente:**

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_SIZE` | `small` | Modelo Whisper |
| `DEVICE` | auto | `cuda` ou `cpu` |
| `HOST` | `0.0.0.0` | Endereço de bind |
| `PORT` | `8000` | Porta HTTP |

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "small",
  "device": "cuda",
  "uptime_s": 42.1,
  "cuda": true,
  "gpu": "NVIDIA GeForce RTX 3060",
  "vram_gb": 6.0
}
```

---

### `POST /transcribe`

Upload de arquivo via `multipart/form-data`.

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample.mp4" \
  -F "language=pt"
```

**Parâmetros:**

| Campo | Obrigatório | Descrição |
|---|---|---|
| `file` | Sim | Arquivo de áudio/vídeo |
| `language` | Não (padrão `pt`) | Código do idioma |

**Resposta:**

```json
{
  "text": "Texto completo transcrito aqui...",
  "segments": [
    { "start": 0.0, "end": 3.4, "text": "Primeiro segmento." },
    { "start": 3.5, "end": 7.1, "text": "Segundo segmento." }
  ],
  "language": "pt",
  "duration": 120.5
}
```

---

## Modelos disponíveis

| Modelo | VRAM | Velocidade | Precisão |
|---|---|---|---|
| `tiny` | ~1 GB | Muito rápido | Básica |
| `base` | ~1 GB | Rápido | Boa |
| `small` | ~2 GB | Balanceado ← **padrão** | Ótima |
| `medium` | ~5 GB | Lento | Máxima |

---

## Arquitetura

```
src/
├── transcriber.py   ← Motor STT (faster-whisper, VAD, SRT)
├── cli.py           ← Interface de linha de comando (Click + Rich)
├── server.py        ← API HTTP (aiohttp, multipart upload)
└── __init__.py

outputs/
├── transcripts/     ← .txt e .json
└── subtitles/       ← .srt
```

---

## Roadmap

| Sprint | Status | Entregáveis |
|---|---|---|
| Sprint 1 | ✅ Completo | `cli.py`, `server.py`, `README.md` |
| Sprint 2 | Pendente | Word-level timestamps, confidence scores, Rich progress bar |
| Sprint 3 | Pendente | Streaming de microfone em tempo real |
| Sprint 4 | Pendente | Diarização de speakers, MCP integration |
