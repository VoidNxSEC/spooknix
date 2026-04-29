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

---

## Suite Conversacional (Simulador Full-Duplex)

O Spooknix conta com um **Orquestrador de Entrevistas** que simula diálogos *Full-Duplex* (você e a IA podem se interromper mutuamente), focado no treino prático para vagas técnicas.

### Arquitetura em 5 Camadas

A arquitetura do simulador foi desenhada para isolar o estado da conversa do processamento pesado:

1. **Personas & Cenários (Camadas 1 e 2):** Injeção dinâmica de perfis (ex: Recrutadora Americana Sênior) e cenários (System Design, Behavioral).
2. **Orquestrador Async (Camada 3):** O cérebro local. Uma máquina de 3 estados (`LISTENING`, `PROCESSING`, `SPEAKING`).
    - *Barge-in Nativo:* Usa o microfone para interromper o *playback* e o raciocínio da IA em tempo real.
    - *Sentence Chunking:* Não espera o LLM gerar todo o texto. Envia frase a frase para o TTS, reduzindo o delay inicial a milissegundos.
3. **PipeWire (O Hub Acústico):** O Spooknix não inventa a roda com filtros de ruído no Python. Delega o AEC (Acoustic Echo Cancellation) e supressão de ruído ao servidor de áudio do sistema operacional, pegando sinais cruzos do microfone (fone/iPhone) e cuspindo PCM limpo no alto-falante.
4. **Workers GPU (Camada 4):** Serviços "burros" acessados via rede local.
    - **STT:** `faster-whisper` (Docker local via Porta 8000).
    - **LLM:** OpenAI API-compatible (pode apontar para um Ollama ou vLLM).
    - **TTS:** Integrado ao **F5-TTS** (suporta *Zero-Shot Voice Cloning* e baixíssima latência), operando na porta 8001.
5. **Reflexão (Camada 5):** Ao final do `Ctrl+C`, o histórico é enviado ao LLM com um *System Prompt* de avaliação (Evaluator) que gera um relatório Markdown (`outputs/interviews/session.md`) sobre a performance, vocabulário e inglês do candidato.

**Como rodar:**

```bash
spooknix interview --language en --silence 2.0 --threshold 0.03
```
*Dica:* Ajuste o `--threshold` de acordo com a captação de fundo do seu microfone.


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
| Sprint 4 | ✅ Completo | Diarização de speakers, MCP integration, Suite Conversacional Full-Duplex (Orquestrador + F5-TTS + PipeWire) |
