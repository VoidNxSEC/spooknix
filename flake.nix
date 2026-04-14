# flake.nix
{
  description = "STT Pipeline - Privacy-first Speech-to-Text";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    sops-nix.url = "github:Mic92/sops-nix";
    sops-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    { self, nixpkgs, sops-nix }:
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
      ]);

      spooknixGui = pkgs.writeShellApplication {
        name = "spooknix-gui";
        runtimeInputs = [ guiPkg pkgs.portaudio pkgs.wl-clipboard ];
        text = ''
          export PYTHONPATH="${self}''${PYTHONPATH:+:$PYTHONPATH}"
          exec python -m src.gui "$@"
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

          # CUDA
          cudaPackages.cudatoolkit
          cudaPackages.cudnn

          # Áudio
          ffmpeg
          portaudio    # backend C do sounddevice
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
          if [ ! -f "poetry.lock" ]; then
            echo "📦 Instalando dependências com Poetry..."
            poetry install --with gui
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
          else
            echo "⚠️  Secrets: secrets/age.key não encontrado"
            echo "   Gere com: age-keygen -o secrets/age.key"
          fi
          echo ""

          # Aliases de conveniência — delegam para o venv do poetry
          alias spooknix="poetry run spooknix"
          alias spooknix-gui="poetry run spooknix-gui"
          alias pytest="poetry run pytest"
          alias pytest-cov="poetry run pytest --cov=src --cov-report=term-missing"
        '';

        # Variáveis de ambiente para CUDA, áudio e libs C++ (numpy/torch via pip)
        LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.portaudio}/lib";

        # Redirecionar cache do HuggingFace para home do usuário
        HF_HOME = "$HOME/.cache/huggingface";
        HUGGINGFACE_HUB_CACHE = "$HOME/.cache/huggingface/hub";

        # Poetry não usa venv no path do projeto por padrão no NixOS
        POETRY_VIRTUALENVS_IN_PROJECT = "false";
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
