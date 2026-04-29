# flake.nix
{
  description = "STT Pipeline - Privacy-first Speech-to-Text";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    sops-nix.url = "github:Mic92/sops-nix";
    sops-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      sops-nix,
    }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      # ── GUI package (thin client — sem torch/ML) ──────────────────────────
      guiPkg = pkgs.python313.withPackages (ps: [
        ps.pyqt6
        ps.pyqt6-sip
        ps.requests
        ps.numpy
        ps.sounddevice
        ps.scipy
        ps.click
        ps.openai
      ]);

      spooknixGui = pkgs.writeShellApplication {
        name = "spooknix-gui";
        runtimeInputs = [
          guiPkg
          pkgs.portaudio
          pkgs.wl-clipboard
        ];
        text = ''
          export PYTHONPATH="${self}''${PYTHONPATH:+:$PYTHONPATH}"
          exec python -m src.gui "$@"
        '';
      };

      poetryWrappedCommand =
        name: command:
        pkgs.writeShellApplication {
          inherit name;
          runtimeInputs = [ pkgs.poetry ];
          text = ''
            exec poetry run ${command} "$@"
          '';
        };

      spooknixCmd = poetryWrappedCommand "spooknix" "python -m src.cli";
      spooknixGuiCmd = poetryWrappedCommand "spooknix-gui" "python -m src.gui";
      pytestCmd = poetryWrappedCommand "pytest" "pytest";
      pytestCovCmd = pkgs.writeShellApplication {
        name = "pytest-cov";
        runtimeInputs = [ pkgs.poetry ];
        text = ''
          exec poetry run pytest --cov=src --cov-report=term-missing "$@"
        '';
      };

    in
    {
      # ── Dev shell ─────────────────────────────────────────────────────────
      devShells.${system}.default = pkgs.mkShell {
        name = "stt-pipeline";

        packages = with pkgs; [
          # Python + gerenciador de pacotes
          python313
          python313Packages.click
          poetry
          spooknixCmd
          spooknixGuiCmd
          pytestCmd
          pytestCovCmd

          # CUDA
          cudaPackages.cudatoolkit
          cudaPackages.cudnn

          # Áudio
          ffmpeg
          portaudio # backend C do sounddevice
          wl-clipboard # wl-copy para clipboard Wayland

          # Signal processing (scipy system libs)
          blas
          lapack

          # Utils
          just

          # Secrets
          sops
          age

        ];

        shellHook = ''
          echo ""
          echo "🎤 STT Pipeline - Ambiente de Desenvolvimento"
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo ""

          # Verificar NVIDIA
          if command -v nvidia-smi &> /dev/null; then
            echo "✅ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
            echo "✅ VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
          else
            echo "⚠️  nvidia-smi não encontrado"
          fi
          echo ""

          # Instalar dependências se necessário
          if [ ! -d ".venv" ]; then
            echo "📦 Instalando dependências com Poetry..."
            poetry install --with gui --with dev
          fi

          echo "🐍 Python: $(poetry run python --version)"
          echo "📁 Projeto: $(pwd)"
          echo ""
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo "  spooknix info"
          echo "  spooknix record --clip"
          echo "  spooknix record --language en --clip"
          echo "  spooknix file <audio> --format srt"
          echo "  spooknix-gui"
          echo ""
          echo "  pytest          → testes sem GPU/mic"
          echo "  pytest-cov      → com cobertura"
          echo ""

          # ── Secrets (SOPS + age) ───────────────────────────────────────────
          export SOPS_AGE_KEY_FILE="$PWD/secrets/age.key"
          if [ -f "$SOPS_AGE_KEY_FILE" ] && [ -f "$PWD/secrets/secrets.yaml" ]; then
            export HF_TOKEN
            HF_TOKEN=$(sops -d --extract '["hf_token"]' "$PWD/secrets/secrets.yaml" 2>/dev/null || echo "")
            if [ -n "$HF_TOKEN" ]; then
              echo "🔑 Secrets: HF_TOKEN ✓ (via SOPS)"
            else
              echo "⚠️  Secrets: HF_TOKEN vazio (verifique secrets/age.key)"
            fi

            export OPENAI_API_KEY
            OPENAI_API_KEY=$(sops -d --extract '["openai_api_key"]' "$PWD/secrets/secrets.yaml" 2>/dev/null || echo "")
            if [ -n "$OPENAI_API_KEY" ]; then
              echo "🔑 Secrets: OPENAI_API_KEY ✓ (via SOPS)"
            elif [ -n "$LLM_BASE_URL" ]; then
              echo "🧠 LLM local configurado via LLM_BASE_URL=$LLM_BASE_URL"
            else
              echo "ℹ️  LLM: defina LLM_BASE_URL para backend local ou OPENAI_API_KEY para OpenAI"
            fi
          else
            echo "⚠️  Secrets: secrets/age.key não encontrado"
            echo "   Gere com: age-keygen -o secrets/age.key"
          fi
          echo ""

        '';

        # Variáveis de ambiente para CUDA, áudio e libs C++ (numpy/torch via pip)
        LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.portaudio}/lib";

        # Redirecionar cache do HuggingFace para home do usuário
        HF_HOME = "$HOME/.cache/huggingface";
        HUGGINGFACE_HUB_CACHE = "$HOME/.cache/huggingface/hub";

        # Mantém o ambiente previsível para wrappers e para `nix develop --command`
        POETRY_VIRTUALENVS_IN_PROJECT = "true";
      };

      # ── Packages ──────────────────────────────────────────────────────────
      packages.${system} = {
        default = spooknixGui;
        gui = spooknixGui;
      };

      # ── NixOS module (backend container) ─────────────────────────────────
      nixosModules.default = import ./nix/modules/nixos/default.nix;
      nixosModules.spooknix = import ./nix/modules/nixos/default.nix;

      # ── Home-Manager module (systray GUI) ─────────────────────────────────
      homeManagerModules.default = import ./nix/modules/home-manager/default.nix;
      homeManagerModules.spooknix = import ./nix/modules/home-manager/default.nix;
    };
}
