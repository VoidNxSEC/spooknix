# Spooknix — Runbook & Operations Guide

Este documento detalha os procedimentos operacionais para subir, testar e debugar a **Suite Conversacional Full-Duplex** do Spooknix.

## 1. Topologia de Deploy (Workers)

A arquitetura exige 3 *Workers* rodando independentemente para garantir a reatividade do orquestrador local.

### Worker 1: STT (Speech-to-Text)
Responsável por transcrever o áudio do candidato.
- **Porta:** `8000` (Padrão)
- **Comando:** `docker compose up -d spooknix` (Utiliza CDI da NVIDIA nativo do NixOS).
- **Check de Saúde:** `curl http://localhost:8000/health`

### Worker 2: LLM (Raciocínio)
Responsável por gerar a resposta da Persona.
- **Provedor:** Pode ser um container local (vLLM, Ollama) ou API remota (OpenAI, Groq).
- **Variáveis Mapeadas:** `LLM_BASE_URL` e `LLM_API_KEY` (ou `OPENAI_API_KEY`).
- **Compose auxiliar:** `docker-compose.workers.yml` aceita `LLM_IMAGE` e `LLM_START_COMMAND`.

### Worker 3: TTS (XTTS-v2 / Chatterbox / F5-TTS)
Sintetiza a voz clonada da Persona em tempo real.
- **Porta:** `8001` (Padrão sugerido)
- **Comando:** *(Depende da sua imagem TTS local, ex: `docker run -p 8001:8001 xtts-api`)*.
- **Variável Mapeada:** `TTS_BASE_URL=http://localhost:8001`
- **Compatibilidade legada:** `XTTS_BASE_URL`, `CHATTERBOX_BASE_URL` e `F5_TTS_URL` também são aceitas como fallback.
- **Compose auxiliar:** `docker-compose.workers.yml` aceita `TTS_IMAGE` e `TTS_START_COMMAND`.

---

## 2. Configuração do Ambiente

Todos os segredos devem estar preferencialmente injetados via SOPS/Age ou exportados no shell.

```bash
# 1. Entre no ambiente Nix hermético
nix develop

# 2. Exporte as variáveis dos Workers (setup local recomendado)
export LLM_BASE_URL="http://localhost:8080/v1"
export LLM_MODEL="qwen-3.5"
export TTS_BASE_URL="http://localhost:8001"
export TTS_API_PATH="/tts"
export TTS_LANGUAGE="en"
```

Se você quiser usar OpenAI em vez de backend local, aí sim defina `OPENAI_API_KEY`.

Para um bootstrap mais rápido em Brev, use `deploy/BREV.md` e `scripts/brev-smoke.sh`.

---

## 3. Operação da Entrevista (Orquestrador)

O Orquestrador gerencia a entrada/saída do PipeWire e a máquina de estados.

```bash
# Iniciar simulação padrão
spooknix interview

# Iniciar com ajuste de VAD (para microfones mais ruidosos)
spooknix interview --threshold 0.05 --silence 1.5

# Forçar outro modelo localmente
spooknix interview --model llama-3
```

**Comportamento Esperado:**
1. O terminal exibirá `[Ouvindo]`.
2. Ao falar, a barra de transcrição aparecerá indicando o envio ao Worker 1.
3. A resposta começará a ser impressa token a token (Worker 2).
4. O áudio começará a tocar via PipeWire logo na primeira pontuação (Worker 3).
5. Se você falar por cima da IA, ela será interrompida imediatamente (Barge-in).
6. Pressione `Ctrl+C` para encerrar e aguarde a geração do relatório Markdown.

---

## 4. Troubleshooting & Tuning

### Problema: A IA me corta antes de eu terminar de falar
- **Causa:** O tempo de silêncio para considerar o fim do turno está muito baixo.
- **Solução:** Aumente a flag `-s` ou `--silence` (ex: `spooknix interview -s 3.0`).

### Problema: O VAD dispara sozinho / Falsos Positivos
- **Causa:** O microfone capta ruído de fundo (teclado, respiração) acima do limiar RMS atual.
- **Solução:** Aumente o *Threshold* (ex: `--threshold 0.04`).
- **Solução Arquitetural:** Habilite o módulo de Noise Suppression do PipeWire no seu NixOS (`services.pipewire.extraConfig`).

### Problema: A voz da IA entra em loop (Acoustic Echo)
- **Causa:** O microfone está captando o áudio que sai do alto-falante.
- **Solução (Software):** Certifique-se de que o *Acoustic Echo Cancellation* (AEC) nativo do PipeWire está ativo.
- **Solução (Hardware):** Use fones de ouvido.

### Problema: Latência muito alta antes da IA começar a falar
- **Causa:** O *chunk* da primeira frase está muito longo, ou o Worker F5-TTS não está segurando o modelo na VRAM.
- **Solução:** Verifique o uso de GPU do F5-TTS (`nvidia-smi`). Certifique-se de que o *Sentence Chunking* no Orquestrador (`src/orchestrator.py`) está quebrando nas vírgulas ou pontos adequadamente para o idioma escolhido.

### Problema: Relatório não é gerado ao pressionar Ctrl+C
- **Causa:** Conversa curta demais (menos de 10 palavras transcritas) ou falha de conexão com o LLM ao solicitar o `evaluator.md`.
- **Solução:** Verifique os logs de erro ou garanta que a conversa inicial teve trocas suficientes para uma avaliação robusta.
