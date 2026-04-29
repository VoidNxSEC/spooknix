# Plano de Implementação: Spooknix Interview Simulator

Este documento detalha a estratégia arquitetural e os passos de implementação para a nova funcionalidade de Simulação de Entrevistas em Tempo Real no projeto Spooknix.

## Visão Geral

O objetivo é criar uma experiência interativa ("turn-based") onde o usuário simula uma entrevista profissional. O sistema ouvirá a fala do usuário (usando STT via streaming com VAD), interpretará a naturalidade e as pausas, e enviará o texto transcrito para um LLM (Agente Entrevistador), que responderá em tempo real. Ao final da sessão, um relatório completo e recursivo avaliará idioma, coerência técnica e confiança.

A arquitetura será híbrida e flexível: o STT pode rodar localmente ou em cloud (ex: NVIDIA Inception/Brev), e o motor LLM será configurável para usar APIs externas (OpenAI, Groq) ou instâncias privadas/locais (Ollama, vLLM).

## Arquitetura Proposta

O Spooknix opera com uma arquitetura modularizada que separa perfeitamente a inferência pesada (STT/LLM) da interface do usuário (CLI). Para o Simulador de Entrevistas, o fluxo de dados em tempo real ("turn-based") será orquestrado pelo `src/cli.py` conectando dois motores independentes:

### 1. Motor STT (Spooknix Backend Local)
O STT atual (`src/server.py` e `src/transcriber.py`) continua processando o áudio via WebSockets (`/ws/stream`).
- O usuário fala no microfone $\rightarrow$ `cli.py` captura com `sounddevice`.
- O VAD (baseado em RMS e janela de silêncio configurável) determina o fim do turno.
- O áudio vai para o Docker do Spooknix que devolve a transcrição exata usando `faster-whisper`.

### 2. Motor LLM (Entrevistador Local via `llama.cpp`)
Em vez de engessar o Spooknix com bibliotecas pesadas de geração de texto, usamos o módulo `src/llm_client.py` (baseado no protocolo universal da OpenAI).
Isso permite plugar o **seu container Docker atual rodando `llama.cpp` com o modelo Qwen 3.5 (6GB)** de forma nativa e sem latência de rede.

**Fluxo de Comunicação Local:**
`[Spooknix CLI] --(JSON/HTTP)--> [llama.cpp Docker Server (ex: localhost:8080)]`

O `llama.cpp` expõe um endpoint `/v1/chat/completions` compatível com o padrão de mercado. O `cli.py` envia a transcrição do seu turno, e o Qwen 3.5 responde em tempo real (streaming de tokens) atuando como o entrevistador.

### 3. Pipeline de Avaliação e Documentação
Enquanto a entrevista ocorre, o histórico completo (suas falas transcritas pelo Whisper + respostas do Qwen) é mantido em memória pela classe `InterviewSession`.
- Ao encerrar (`Ctrl+C`), todo esse histórico é envelopado no template `evaluator.md` e enviado ao Qwen.
- O modelo gera o relatório de *Feedback Completo* (Idioma, Coerência Técnica, Confiança) que é salvo em `outputs/interviews/session.md`.

## Como Configurar para o Ambiente Atual (llama.cpp + Qwen)

Como você já tem o servidor `llama.cpp` rodando, a integração no Spooknix se dá apenas configurando as variáveis de ambiente antes de rodar o comando:

```bash
# Aponta o cliente para o seu container do llama.cpp local
export LLM_BASE_URL="http://localhost:8080/v1" # Ajuste a porta se necessário
export LLM_API_KEY="sk-no-key"                 # llama.cpp não exige chave real
export LLM_MODEL="qwen-3.5"                    # Ou o nome que o llama.cpp espera

# Inicia o simulador
spooknix interview
```

Esta arquitetura é a **melhor forma de encaixar a ideia** pois garante que o Spooknix continue focado no STT (sua especialidade), enquanto aproveita 100% do poder da sua GPU local rodando o Qwen para a inferência de texto, mantendo tudo *Privacy-First* e offline. Se no futuro você quiser usar a Brev.dev ou NVIDIA NIM, basta mudar a variável `LLM_BASE_URL`.

## Fases de Implementação

### Fase 1: Fundação LLM e Templates
1. Criar módulo `src/llm_client.py` com suporte a streaming e troca de backend.
2. Definir a estrutura do diretório `templates/` com prompts para o *Interviewer* e o *Evaluator*.
3. Atualizar configurações (`.env`, argumentos CLI) para injetar chaves/URLs do LLM.

### Fase 2: Aprimoramento do VAD e Turnos
1. Modificar a lógica do WebSocket no `server.py` e o cliente `cli.py` para introduzir o conceito de "Fim de Turno" (End of Turn) baseado em detecção de silêncio configurável (ex: 2.5 segundos de silêncio contínuo = fim do turno).
2. Garantir que hesitações transcritas pelo `faster-whisper` não interrompam o turno prematuramente.

### Fase 3: Orquestração do Simulador (CLI)
1. Criar o comando `spooknix interview` no `cli.py`.
2. Implementar o loop principal:
   - Grava áudio $\rightarrow$ Transcreve $\rightarrow$ Detecta fim do turno.
   - Envia texto para `llm_client.py`.
   - Imprime a resposta do LLM em streaming no console.
   - Reinicia a gravação para o próximo turno do usuário.

### Fase 4: Documentação e Feedback
1. Ao encerrar o loop da entrevista, formatar todo o histórico acumulado.
2. Enviar o histórico para o LLM usando o template de "Feedback Completo".
3. Salvar o Markdown gerado (ex: `outputs/interviews/session_YYYY-MM-DD.md`).

## Riscos e Mitigações
- **Latência:** A combinação de STT em streaming + LLM pode introduzir atrasos perceptíveis. *Mitigação:* Usar STT local pequeno (ex: `small.en` se o foco for inglês) ou LLMs super-rápidos via Groq/NVIDIA na cloud para as respostas da entrevista, reservando modelos densos/locais para a geração do relatório final.
- **Interrupções Acidentais:** O VAD cortando o usuário no meio do pensamento. *Mitigação:* Ajustar o tempo limite de silêncio (`silence_threshold`) para um valor tolerante durante entrevistas (ex: 3-4 segundos), e permitir que o usuário force a passagem do turno (ex: pressionando `Enter`).