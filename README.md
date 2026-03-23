# Spooknix — Privacy-first STT Engine

Transcrição de áudio com alta fidelidade, privacy-first, sem nuvem.

Baseado em [faster-whisper](https://github.com/SYSTRAN/faster-whisper) com suporte CUDA via Docker.

---

## Requisitos

- NixOS / Nix com Flakes habilitado
- Docker + NVIDIA Container Toolkit (CDI)
- GPU NVIDIA com 4–6 GB VRAM (CPU funciona, mais lento)

---

## Setup

```bash
# 1. Entrar no ambiente Nix
nix develop

# 2. Instalar dependências Python
poetry install --with gui

# 3. Subir o servidor de inferência (GPU via Docker)
docker compose up -d

# 4. Verificar
spooknix info
curl http://localhost:8000/health
```

---

## CLI

Dentro do `nix develop`, os comandos ficam disponíveis diretamente:

```bash
spooknix --help
```

### `spooknix info` — Status do sistema

```bash
spooknix info
```

Exibe GPU detectada, VRAM disponível e modelos suportados.

---

### `spooknix record` — Gravar do microfone

```bash
spooknix record [opções]
```

Grava até detectar silêncio e envia o áudio ao servidor Docker para transcrição.

| Flag              | Padrão              | Descrição                              |
| ----------------- | ------------------- | -------------------------------------- |
| `-l`, `--language`| `pt`                | Código do idioma (`pt`, `en`, `es`, …) |
| `-s`, `--silence` | `2.0`               | Segundos de silêncio para parar        |
| `-t`, `--threshold`| `0.01`             | RMS mínimo para considerar silêncio    |
| `--clip/--no-clip`| desativado          | Copiar resultado para clipboard        |
| `--max-duration`  | `120.0`             | Teto máximo de gravação (segundos)     |
| `--server`        | `$SPOOKNIX_URL`     | URL do servidor (padrão: localhost:8000)|

**Exemplos:**

```bash
# Gravar em português e copiar para clipboard
spooknix record --clip

# Gravar em inglês
spooknix record --language en --clip

# Servidor remoto
spooknix record --server http://192.168.1.10:8000 --clip

# Parar após 5s de silêncio
spooknix record --silence 5.0 --clip
```

---

### `spooknix file` — Transcrever arquivo

```bash
spooknix file <audio_path> [opções]
```

| Flag                 | Padrão     | Descrição                              |
| -------------------- | ---------- | -------------------------------------- |
| `-l`, `--language`   | `pt`       | Código do idioma (`pt`, `en`, `es`, …) |
| `-m`, `--model`      | `small`    | Tamanho do modelo Whisper              |
| `-o`, `--output-dir` | `outputs/` | Diretório de saída                     |
| `-f`, `--format`     | `all`      | Formato: `txt`, `srt`, `json`, `all`   |

**Exemplos:**

```bash
# Transcrição completa
spooknix file sample.mp4

# Legenda SRT, modelo medium
spooknix file entrevista.mp3 --model medium --format srt

# Inglês, só JSON
spooknix file podcast.m4a --language en --format json
```

**Saída gerada** (com `--format all`):

```
outputs/
├── transcripts/
│   ├── sample.txt
│   └── sample.json
└── subtitles/
    └── sample.srt
```

---

## GUI (Systray)

```bash
spooknix-gui
```

Ícone na bandeja do sistema (Wayland/Hyprland). Clique para gravar, resultado vai ao clipboard.

Atalho de teclado configurado via Home-Manager: `SUPER + R`

---

## Servidor HTTP

### Iniciar

```bash
# Docker (recomendado — GPU via CDI)
docker compose up -d

# Local (sem GPU)
poetry run python -m src.server
```

**Variáveis de ambiente:**

| Variável       | Padrão    | Descrição        |
| -------------- | --------- | ---------------- |
| `MODEL_SIZE`   | `small`   | Modelo Whisper   |
| `DEVICE`       | `cuda`    | `cuda` ou `cpu`  |
| `HOST`         | `0.0.0.0` | Endereço de bind |
| `PORT`         | `8000`    | Porta HTTP       |
| `SPOOKNIX_URL` | —         | URL do servidor (usada pelo CLI) |

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "small",
  "device": "cuda",
  "cuda": true,
  "gpu": "NVIDIA GeForce RTX 3050 6GB"
}
```

### `POST /transcribe`

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample.mp4" \
  -F "language=pt"
```

**Resposta:**

```json
{
  "text": "Texto completo transcrito...",
  "segments": [
    { "start": 0.0, "end": 3.4, "text": "Primeiro segmento." }
  ],
  "language": "pt",
  "duration": 7.7
}
```

---

## Idiomas suportados

O Whisper suporta 99 idiomas. Principais:

| Flag | Idioma     |
| ---- | ---------- |
| `pt` | Português  |
| `en` | Inglês     |
| `es` | Espanhol   |
| `fr` | Francês    |
| `de` | Alemão     |
| `ja` | Japonês    |
| `zh` | Chinês     |

---

## Modelos disponíveis

| Modelo   | VRAM   | Velocidade   | Precisão |
| -------- | ------ | ------------ | -------- |
| `tiny`   | ~1 GB  | Muito rápido | Básica   |
| `base`   | ~1 GB  | Rápido       | Boa      |
| `small`  | ~2 GB  | Balanceado ← **padrão** | Ótima |
| `medium` | ~5 GB  | Lento        | Máxima   |

---

## Arquitetura

```
Cliente (CLI / GUI)
    │  grava WAV localmente (sounddevice, 16kHz mono)
    │  POST multipart/form-data
    ▼
Servidor Docker (GPU)
    │  faster-whisper + CTranslate2 + CUDA
    ▼
Resposta JSON  →  texto + segments + duration
```

```
src/
├── recorder.py      ← Gravação mic (sounddevice, VAD por RMS)
├── transcriber.py   ← Motor STT (faster-whisper, VAD, SRT)
├── cli.py           ← CLI (Click + Rich): info, file, record
├── server.py        ← API HTTP (aiohttp): /health, /transcribe
└── gui.py           ← Systray PyQt6 + RecordThread

nix/
├── modules/nixos/        ← services.spooknix (Docker + NVIDIA)
└── modules/home-manager/ ← programs.spooknix (systemd, Hyprland, Waybar)
```

---

## Testes

```bash
# Suite completa (sem GPU, sem microfone)
pytest

# Com cobertura
pytest-cov
```

19 testes cobrindo recorder e CLI, todos mockados.

---

## Roadmap

| Sprint   | Status      | Entregáveis                                              |
| -------- | ----------- | -------------------------------------------------------- |
| Sprint 1 | ✅ Completo | `cli.py`, `server.py`, API HTTP                          |
| Sprint 2 | ✅ Completo | Progress bar Rich, VAD integrado                         |
| Sprint 3 | ✅ Completo | Gravação por microfone, GUI systray, hotkey SUPER+R      |
| Sprint 4 | Pendente    | Diarização de speakers, MCP integration                  |
