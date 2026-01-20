# flake.nix
{
  description = "STT Pipeline - Privacy-first Speech-to-Text";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "stt-pipeline";

        packages = with pkgs; [
          # Python
          python313
          python313Packages.pip
          python313Packages.virtualenv

          # CUDA
          cudaPackages.cudatoolkit
          cudaPackages.cudnn

          # Áudio
          ffmpeg

          # Utils
          just
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

          # Criar venv se não existir
          if [ ! -d ".venv" ]; then
            echo "📦 Criando ambiente virtual Python..."
            python -m venv .venv
          fi

          source .venv/bin/activate

          echo "🐍 Python: $(python --version)"
          echo "📁 Projeto: $(pwd)"
          echo ""
          echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
          echo "Próximo passo: pip install -r requirements.txt"
          echo ""
        '';

        # Variáveis de ambiente para CUDA
        LD_LIBRARY_PATH = "${pkgs.cudaPackages.cudatoolkit}/lib";
      };
    };
}
