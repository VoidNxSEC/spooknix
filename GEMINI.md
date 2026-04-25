# Convenções de Desenvolvimento - Spooknix

Este documento define as diretrizes de desenvolvimento para o projeto Spooknix.

## 1. Ambiente de Desenvolvimento
- **Nix:** O uso de `nix develop --command` é obrigatório para garantir a reprodutibilidade do ambiente. Recomenda-se a criação de aliases locais no seu `shell.nix` ou `bashrc` para simplificar:
  ```bash
  alias spook='nix develop --command spooknix'
  ```

## 2. CLI e UX (User Experience)
- **Design de Saída:**
  - Utilize `rich` para feedback visual em terminais interativos.
  - Para saída de texto (transcrições, logs de dados), use `out_console` (ou `print` condicional) para garantir que o conteúdo seja "pipe-friendly".
  - Verifique `console.is_terminal` antes de aplicar *markup* (cores/negrito/panels) para evitar poluição em redirecionamentos (`>`) ou pipes (`|`).
- **Flags de CLI:**
  - Comandos que geram saída de dados devem suportar `--out <caminho>` para salvar transcrições em arquivos (`.txt`, `.md`).
  - Sempre forneça ajuda clara em `--help` para novas opções.

## 3. Gestão de Código (Git)
- **Commits:**
  - Utilize o padrão *Conventional Commits* (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`).
  - Mantenha mensagens concisas focadas no "porquê" da mudança.
- **Fluxo:**
  - Valide sempre `git status` e `git diff` antes de realizar o `commit`.
  - O `push` é manual e explícito; não realize *push* automático sem revisão.

## 5. Design de Funcionalidades Efêmeras
- **Objetivo:** Ferramentas de gravação e transcrição devem ser focadas em conveniência e utilidade imediata.
- **Processamento de Dados:** Ao utilizar os comandos de gravação ou streaming, o processamento deve evitar a propagação de artefatos (linhas quebradas, erros de ASR, caracteres de controle não tratados) para a saída final (web/stdout).
- **Limpeza:** A saída final deve ser preferencialmente limpa e estruturada, pronta para consumo humano ou integração direta via *pipe*, tratando o fluxo como temporário e focado no resultado final.

