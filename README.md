# Spooknix — Privacy-first STT Engine

Transcrição de áudio com alta fidelidade, privacy-first, sem nuvem.

Baseado em [faster-whisper](https://github.com/SYSTRAN/faster-whisper) com suporte CUDA via Docker.

Para o modo conversacional completo em GPU remota, veja `deploy/BREV.md`. O caminho local-first recomendado usa 3 workers separados: STT, LLM e TTS.

---

## Requisitos

- NixOS / Nix com Flakes habilitado
- Docker + NVIDIA Container Toolkit (CDI)
- GPU NVIDIA com 4–6 GB VRAM (CPU funciona, mais lento)

Observação: esse requisito cobre o STT. Para a suíte conversacional completa com LLM local + TTS local, planeje `16 GB+` de VRAM para um teste realista.

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

### `spooknix doctor` — Self-check do ambiente

```bash
spooknix doctor [--mic]
```

Tabela Rich verificando CUDA, servidor STT, dispositivos de áudio,
`ffmpeg` e llama.cpp local (`http://localhost:8081/v1/models`). Use
`--mic` para incluir 1 s de baseline RMS do microfone.

Quando algo der ruim na CLI, rode `spooknix doctor` primeiro — ele
costuma apontar o problema antes de você ler stacktrace.

---

### `spooknix record` — Gravar do microfone

```bash
spooknix record [opções]
```

Grava até detectar silêncio e envia o áudio ao servidor Docker para transcrição.

| Flag                       | Padrão             | Descrição                                                            |
| -------------------------- | ------------------ | -------------------------------------------------------------------- |
| `-l`, `--language`         | `pt`               | Código do idioma (`pt`, `en`, `es`, …)                               |
| `-s`, `--silence`          | `2.0`              | Segundos de silêncio para parar                                      |
| `-t`, `--threshold`        | `0.01`             | RMS mínimo (ignorado com `--vad-neural`)                             |
| `--vad-neural`             | desativado         | Usar Silero VAD ao invés do threshold RMS                            |
| `--device`                 | default do sistema | Índice do dispositivo de input (veja `spooknix doctor`)              |
| `--meter`                  | desativado         | Widget Rich Live com peak / RMS / LUFS + sparkline                   |
| `--diarize/--no-diarize`   | desativado         | Diarização de speakers via pyannote                                  |
| `--clip/--no-clip`         | desativado         | Copiar resultado para clipboard (`wl-copy`)                          |
| `--stop-word`, `-w`        | `stop`             | Palavra falada que encerra a gravação                                |
| `--max-duration`           | `300.0`            | Teto absoluto de gravação (segundos)                                 |
| `--server`                 | `$SPOOKNIX_URL`    | URL do servidor (padrão: `localhost:8000`)                           |
| `--out`                    | —                  | Salvar transcript em arquivo                                         |
| `-v`, `-vv`                | warning            | `-v` = INFO, `-vv` = DEBUG (RMS por chunk)                           |

**Exemplos:**

```bash
# Gravação com VAD neural + meter visual (recomendado pra mics ruins)
spooknix record --vad-neural --meter --clip

# Forçar um dispositivo específico (após conferir spooknix doctor)
spooknix record --device 3 --vad-neural

# Servidor remoto
spooknix record --server http://192.168.1.10:8000 --clip

# Parar quando você falar "para"
spooknix record --stop-word para --language pt
```

---

### `spooknix file` — Transcrever arquivo

```bash
spooknix file <audio_path> [opções]
```

| Flag                 | Padrão       | Descrição                                                                       |
| -------------------- | ------------ | ------------------------------------------------------------------------------- |
| `-l`, `--language`   | `pt`         | Código do idioma (`pt`, `en`, `es`, …)                                          |
| `-m`, `--model`      | `large-v3`   | `tiny`/`base`/`small`/`medium`/`large-v2`/`large-v3`/`large-v3-turbo`           |
| `-o`, `--output-dir` | `outputs/`   | Diretório de saída                                                              |
| `-f`, `--format`     | `all`        | `txt` / `srt` / `json` / `all`                                                  |
| `-v`, `-vv`          | warning      | `-v` = INFO, `-vv` = DEBUG                                                      |

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

### `spooknix interview` — Simulador de entrevistas (Full-Duplex)

```bash
spooknix interview --persona sarah --scenario behavioral --difficulty hard
```

| Flag             | Padrão          | Descrição                                                     |
| ---------------- | --------------- | ------------------------------------------------------------- |
| `--persona`      | `sarah`         | Nome em `personas/<name>.yaml` (sarah, marcus, priya, …)      |
| `--scenario`     | `system_design` | `behavioral`, `system_design`, `frontend`, `backend`, `ml`, `leadership`, `product` |
| `--difficulty`   | `standard`      | `easy` / `standard` / `hard` (controla duration + addendum)   |
| `--language`     | `en`            | Idioma da sessão                                              |
| `--list`         | —               | Listar histórico de sessões (SQLite)                          |
| `--show <id>`    | —               | Detalhes de uma sessão                                        |
| `--diff <a> <b>` | —               | Comparar duas sessões eixo-a-eixo da rubrica                  |
| `--model`        | —               | Override do modelo LLM                                        |
| `--out`          | —               | Caminho do relatório (default: `outputs/interviews/<ts>-<persona>/feedback.md`) |

Arquitetura:
- **Personas + Scenarios em YAML** com `prompt_addendum` e `rubric_weights` por nível.
- **Orquestrador async** com máquina de 3 estados (`LISTENING`/`PROCESSING`/`SPEAKING`) e barge-in nativo.
- **Sentence chunking**: o LLM streama tokens, o orchestrator cospe frase a frase pro TTS — delay inicial em ms.
- **PipeWire** cuida de AEC e supressão de ruído de hardware; o pipeline Python pré-trata só o que chega limpo.
- **LLM**: OpenAI API-compatible. Se `OPENAI_API_KEY` ausente, sonda `http://localhost:8081/v1/models` (llama.cpp) automaticamente.
- **Persistência**: cada sessão grava transcript + rubric JSON em SQLite (`~/.local/share/spooknix/sessions.db`).
- **Rubric estruturada** (5 eixos × score 0-5) com parser tolerante a JSON fenceado.

---

### `spooknix summarize` — Resumo de vídeo/aula/reunião

```bash
spooknix summarize lecture.mp4 --template lecture --language en
spooknix summarize meeting.m4a --diarize --template meeting
```

| Flag          | Padrão    | Descrição                                                |
| ------------- | --------- | -------------------------------------------------------- |
| `--template`  | `summary` | `summary` / `lecture` / `meeting` / `notes` / `study_guide` |
| `--language`  | `pt`      | Idioma                                                   |
| `--model`     | `large-v3`| Modelo Whisper                                           |
| `--format`    | `md`      | `md` / `json` / `srt-summary`                            |
| `--diarize`   | desativado| Identificar speakers antes de resumir                    |
| `--max-tokens`| `3000`    | Orçamento por chunk no LLM                               |
| `--out`       | —         | Default: `outputs/summaries/<stem>.<fmt>`                |

Por dentro: transcribe → chunking via `tiktoken` (`o200k_base`) preservando intervalos `[mm:ss → mm:ss]` → cada chunk roda no LLM → stitching final pelo template Jinja2. Os bullets contêm links `[12:34](lecture.mp4#t=754)` clicáveis em players locais e renderizadores Markdown.

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

| Variável             | Padrão                    | Descrição                                                              |
| -------------------- | ------------------------- | ---------------------------------------------------------------------- |
| `MODEL_SIZE`         | `large-v3`                | Modelo Whisper (incluindo `large-v3-turbo`)                            |
| `DEVICE`             | `cuda`                    | `cuda` ou `cpu`                                                        |
| `COMPUTE_TYPE`       | derivado                  | `int8`, `float16`, `int8_float16` (auto pelo modelo+device se vazio)   |
| `ENABLE_DIARIZATION` | `false`                   | Habilita endpoint diarize (precisa pyannote-audio + `HF_TOKEN`)        |
| `HOST`               | `0.0.0.0`                 | Endereço de bind                                                       |
| `PORT`               | `8000`                    | Porta HTTP                                                             |
| `SPOOKNIX_URL`       | —                         | URL do servidor STT (usada pelo CLI)                                   |
| `LLAMACPP_URL`       | `http://localhost:8081`   | URL do `llama-server` (probe `/v1/models`)                             |
| `LLAMACPP_MODEL`     | —                         | Força um modelo específico no fallback                                 |
| `TTS_BASE_URL`       | `http://localhost:8001`   | URL do worker TTS (F5-TTS/XTTS/Piper)                                 |
| `TTS_TIMEOUT_S`      | `30.0`                    | Timeout por request ao TTS                                             |

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

| Modelo             | VRAM                      | Velocidade                      | Precisão                    |
| ------------------ | ------------------------- | ------------------------------- | --------------------------- |
| `tiny`             | ~1 GB                     | Muito rápido                    | Básica                      |
| `base`             | ~1 GB                     | Rápido                          | Boa                         |
| `small`            | ~2 GB                     | Balanceado                      | Ótima                       |
| `medium`           | ~5 GB                     | Lento                           | Alta                        |
| `large-v2`         | ~3 GB (`int8_float16`)    | Lento                           | Máxima — geração anterior   |
| `large-v3`         | ~3 GB (`int8_float16`)    | Lento ← **padrão**              | Máxima                      |
| `large-v3-turbo`   | ~3 GB (`int8_float16`)    | ~8× mais rápido que v3          | Alta (decoder reduzido)     |

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
├── recorder.py        ← Gravação mic (sounddevice, VAD RMS ou Silero neural)
├── vad_silero.py      ← Wrapper Silero VAD com hangover sticky
├── audio_pipeline.py  ← High-pass → denoise → pre-emphasis → LUFS → clip
├── denoise.py         ← DeepFilterNet (df-3) — 16/48 kHz resample na borda
├── loudness.py        ← Loudness EBU R128 (pyloudnorm) com fallback RMS
├── audio_meter.py     ← Widget Rich Live (peak/RMS/LUFS + sparkline)
├── media.py           ← extract_audio via ffmpeg (mp4/mkv/m4a → wav 16 kHz)
├── transcriber.py     ← Motor STT (faster-whisper, VAD compartilhado, SRT)
├── server.py          ← API HTTP (aiohttp): /health, /transcribe, /metrics, /ws/stream
├── orchestrator.py    ← Máquina de estados full-duplex (LISTEN/PROCESS/SPEAK)
├── personas.py        ← Loader de personas/*.yaml
├── scenarios.py       ← Loader de scenarios/*.yaml com difficulty levels
├── sessions_db.py     ← SQLite persistente (~/.local/share/spooknix/sessions.db)
├── rubric.py          ← 5-axis Rubric + parser tolerante (JSON cru ou fenceado)
├── llm_client.py      ← Cliente OpenAI-compatible (OpenAI / llama.cpp / vLLM)
├── llm_fallback.py    ← Probe llama.cpp local em :8081 quando sem API key
├── tts_client.py      ← Cliente para F5-TTS / XTTS / Piper (porta 8001)
├── summarizer.py      ← Chunking via tiktoken preservando [mm:ss] anchors
├── timestamp_links.py ← Geração de links clicáveis [mm:ss](video#t=N)
├── diarizer.py        ← pyannote 3.1 + boundary-aware speaker split
├── doctor.py          ← Self-check (CUDA, STT, áudio, ffmpeg, llama.cpp)
├── types.py           ← Dataclasses canônicas (Segment, SessionRecord)
├── cli.py             ← CLI completa (info, doctor, file, record, stream, interview, summarize)
├── mcp_tool.py        ← MCP server (5 tools: health/transcribe/diarize/doctor/summarize)
└── gui.py             ← Systray PyQt6 + RecordThread
```

---

```bash
# Suite completa (sem GPU, sem microfone)
pytest

# Com cobertura
pytest-cov
```

**~195 testes** cobrindo recorder, audio pipeline, VAD, loudness, media, doctor, personas, scenarios, sessions_db, rubric parser, LLM fallback, summarizer chunking, diarização boundary-aware, MCP tools e TTS client — todos mockados, sem necessidade de GPU/mic.

---

## Roadmap

| Sprint                     | Status      | Entregáveis                                                                                                |
| -------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------- |
| Sprint 1                   | ✅ Completo | `cli.py`, `server.py`, API HTTP                                                                            |
| Sprint 2                   | ✅ Completo | Progress bar Rich, VAD integrado                                                                           |
| Sprint 3                   | ✅ Completo | Gravação por microfone, GUI systray, hotkey SUPER+R                                                        |
| Sprint 4                   | ✅ Completo | Diarização de speakers, MCP integration, Suite Conversacional Full-Duplex (Orquestrador + F5-TTS + PipeWire) |
| Sprint 5 — Reliability     | ✅ Completo | `has_spoken` guard (fix capturas vazias), VAD consistente arquivo↔stream, `spooknix doctor`, `src/types.py` |
| Sprint 6 — Audio quality   | ✅ Completo | Silero VAD, DeepFilterNet (df-3), pyloudnorm (EBU R128), audio meter, `large-v3-turbo`, `extract_audio`     |
| Sprint 7 — Interview suite | ✅ Completo | 3 personas + 7 scenarios YAML, SQLite history, rubric estruturada, llama.cpp fallback, `--list/--show/--diff` |
| Sprint 8 — Summarize       | ✅ Completo | `spooknix summarize`, tiktoken chunking, 5 Jinja2 templates, `[mm:ss](video#t=N)` links, diarização boundary-aware |
