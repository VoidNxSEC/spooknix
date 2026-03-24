# Spooknix — PoC Direction

> Arquivo de direcionamento interno. Estado ao encerrar o ciclo de desenvolvimento ativo.

---

## O que é este PoC

Pipeline completo de **voz → texto → clipboard** rodando 100% local,
privacy-first, sem dependência de nuvem, com latência aceitável para uso diário no desktop.

**Hipótese validada:** `faster-whisper large-v3` (int8_float16) + `sounddevice` via Docker CDI
é suficiente — e altamente preciso — para transcrição conversacional num laptop com GPU discreta.

---

## Estado ao encerrar (2026-03-23)

### Implementado e funcional

| Componente | Arquivo | Observação |
|---|---|---|
| Motor STT | `src/transcriber.py` | faster-whisper large-v3, VAD, word timestamps, beam_size=5, best_of=5 |
| Gravação mic | `src/recorder.py` | silence detection, stop_check_fn (stop word), WAV int16 16kHz |
| CLI | `src/cli.py` | `info`, `file`, `record --clip --stop-word --language` |
| Servidor HTTP | `src/server.py` | aiohttp `/health` + `/transcribe`; aceita model_size e diarize por request |
| Diarização | `src/diarizer.py` | pyannote-audio, lazy import, opt-in via `--diarize` |
| GUI systray | `src/gui.py` | PyQt6, fade, drop zone, botão Gravar (blink vermelho), RecordThread |
| Secrets | `secrets/secrets.yaml` | SOPS + age encriptado; HF_TOKEN exportado automaticamente no nix develop |
| Nix Flake | `flake.nix` | Poetry, portaudio, wl-clipboard, sops, age; sops-nix input |
| NixOS module | `nix/modules/nixos/` | Docker + NVIDIA CDI |
| HM module | `nix/modules/home-manager/` | systemd user, SUPER+S/R, Waybar |

### Infraestrutura

| Item | Detalhe |
|---|---|
| Modelo padrão | `large-v3` com `int8_float16` (~3GB VRAM) |
| GPU | RTX 3050 6GB Laptop — 6.1GB total |
| Cache de modelos | `/var/lib/ml-models/huggingface/hub` montado no container |
| Secrets | SOPS + age: `secrets/age.key` (gitignored) + `secrets/secrets.yaml` (commitado encriptado) |
| DNS Docker build | `network: host` no build para resolver apt no NixOS |

### Testes

- 30 testes unitários passando (`pytest`)
- Zero dependências reais: GPU, mic e servidor todos mockados
- Pre-commit hook ativo — bloqueia commit se testes falharem

---

## Decisões tomadas

### Arquitetura cliente-servidor

CLI e GUI gravam localmente (sounddevice, sem GPU) e enviam WAV via HTTP multipart
ao servidor Docker. O servidor carrega o modelo e faz inferência com CUDA.
Elimina a necessidade de torch/CUDA no ambiente do cliente.

### large-v3 com int8_float16

- Qualidade máxima para transcrição conversacional
- `int8_float16`: mesma qualidade do float16, metade do VRAM (~3GB vs ~6GB)
- Sobra VRAM para diarização pyannote (~2GB) quando ativada

### Stop word por polling periódico

A cada 2s, os últimos 3s de áudio são enviados ao servidor para transcrição rápida.
Se a stop word aparece no resultado, `stop_event` é setado. Sem wake word,
sem modelo dedicado de keyword spotting — reutiliza a infraestrutura existente.

### SOPS + age para secrets

- Chave privada age em `secrets/age.key` (gitignored, gerada por máquina)
- `secrets/secrets.yaml` encriptado, commitado no git — seguro commitar
- `nix develop` decripta automaticamente e exporta `HF_TOKEN`
- `docker compose up` herda `HF_TOKEN` do ambiente

### Poetry em vez de venv manual

`pyproject.toml` + Poetry. Shell aliases no nix develop expõem `spooknix` e
`spooknix-gui` diretamente sem prefixo `poetry run`.

---

## Critérios de sucesso — todos validados ✅

- `spooknix record --clip` funciona no nix develop shell
- `spooknix record --language en --clip` transcreve inglês com alta precisão
- Stop word "stop" encerra a gravação quando falado claramente
- Latência < 15s para falas de até 30s (large-v3, CUDA, int8_float16)
- Nenhum dado sai da máquina (sem chamadas de rede no caminho quente)
- GUI systray com botão Gravar funcional

---

## Backlog (hold)

| Item | Sprint previsto |
|---|---|
| Diarização ativa (HF token pyannote) | Sprint 5 |
| MCP tool expondo transcrição ao Claude | Sprint 5 |
| Streaming WebSocket (transcrição parcial em tempo real) | Sprint 6 |
| GUI: histórico de sessão, seleção de idioma, indicador de modelo | Sprint 5 |
| `language_probability` e `avg_logprob` no retorno do CLI | Sprint 2 (pendente) |
| **GUI: erro de numpy** — `spooknix-gui` acusa falta de numpy mesmo instalado; provável conflito entre o Python do wrapper nix (`guiPkg`) e o venv do Poetry. TODO: investigar `PYTHONPATH` no wrapper ou mover GUI para rodar via `poetry run` | Sprint 5 |

---

## Setup em nova máquina

```bash
# 1. Entrar no ambiente
git clone <repo> && cd spooknix
nix develop

# 2. Gerar chave age local e adicionar pubkey ao .sops.yaml
age-keygen -o secrets/age.key
# copiar a pubkey impressa → adicionar em .sops.yaml → sops updatekeys secrets/secrets.yaml

# 3. Subir servidor (modelo já em /var/lib/ml-models ou baixa automaticamente)
docker compose up -d

# 4. Testar
spooknix record --language en --clip
```
