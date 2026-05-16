# Spooknix Technical TODO — Sprints 3 & 4

## 📋 Sprint 3: Audio Quality & TTS Professionalization

- [ ] **tts_client**:
  - [ ] Implementar logger profissional (logging.getLogger).
  - [ ] Criar suíte de testes unitários (`tests/test_tts_client.py`).
  - [ ] Adicionar suporte a cache de áudio para evitar regeração de frases idênticas.
- [ ] **Audio Pipeline Enhancements**:
  - [ ] Integrar `deepfilternet` para denoise em tempo real.
  - [ ] Validar LUFS normalization em streams longos.

## 📋 Sprint 4: Infraestrutura & Server Stability

- [ ] **server.py**:
  - [ ] Garantir Thread-safety no gerenciamento de sessões e modelos.
  - [ ] Integrar `media.py` para processamento assíncrono de arquivos enviados via API.
  - [ ] Implementar health-check endpoint para monitoramento de VRAM.
- [ ] **LLM Backend Migration**:
  - [ ] Reescrever loader do Ollama para `llamacpp` (suporte a GGUF nativo).
  - [ ] Adicionar suporte a `brev.dev` (NVIDIA) para instâncias remotas dinâmicas.

## 📚 Documentação & Governança

- [ ] **README.md**: Refatorar para incluir arquitetura técnica, guias de Nix e exemplos de CLI.
- [ ] **Roadmap**: Definir próximos passos pós-MVP (v1.0).
- [ ] **MCP Tools**: Finalizar ferramentas de integração com o ecossistema Agentic.

## 🧪 Qualidade & Testes

- [ ] Aumentar cobertura de testes em `src/cli.py`.
- [ ] Implementar testes de integração ponta-a-ponta (E2E) simulando uma entrevista completa.
