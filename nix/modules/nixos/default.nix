# nix/modules/nixos/default.nix — NixOS module para o backend Spooknix
#
# Ativa o container Docker (oci-container) com faster-whisper + CUDA.
# O backend pesado (torch, CTranslate2) fica isolado no container;
# a GUI roda como user service separado (veja home-manager module).
#
# Uso em configuration.nix:
#   services.spooknix.enable = true;
#   services.spooknix.model  = "small";   # tiny | base | small | medium | large-v3
#   services.spooknix.device = "cuda";    # cuda | cpu
{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.spooknix;
in
{
  # ── Options ────────────────────────────────────────────────────────────────
  options.services.spooknix = {
    enable = lib.mkEnableOption "Spooknix STT backend (Docker container)";

    image = lib.mkOption {
      type = lib.types.str;
      default = "spooknix:latest";
      description = "Imagem Docker do backend Spooknix.";
    };

    model = lib.mkOption {
      type = lib.types.enum [
        "tiny"
        "base"
        "small"
        "medium"
        "large-v2"
        "large-v3"
      ];
      default = "large-v3";
      description = "Tamanho do modelo Whisper.";
    };

    device = lib.mkOption {
      type = lib.types.enum [
        "cuda"
        "cpu"
      ];
      default = "cuda";
      description = "Dispositivo de inferência (cuda requer GPU NVIDIA).";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Porta HTTP do servidor STT.";
    };

    outputsDir = lib.mkOption {
      type = lib.types.str;
      # default = "/var/lib/spooknix/outputs";
      default = "/home/kernelcore/master/spooknix/outputs";
      description = "Diretório host para outputs persistentes (transcrições, legendas).";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Abrir porta no firewall local.";
    };
  };

  # ── Config ─────────────────────────────────────────────────────────────────
  config = lib.mkIf cfg.enable {
    # Docker deve estar habilitado
    virtualisation.docker.enable = lib.mkDefault true;

    # Container declarativo via oci-containers
    virtualisation.oci-containers.backend = lib.mkDefault "docker";
    virtualisation.oci-containers.containers.spooknix = {
      image = cfg.image;
      autoStart = true;

      environment = {
        MODEL_SIZE = cfg.model;
        DEVICE = cfg.device;
        PORT = toString cfg.port;
        HOST = "0.0.0.0";
      };

      ports = [ "${toString cfg.port}:${toString cfg.port}" ];

      volumes = [
        "${cfg.outputsDir}:/app/outputs"
      ];

      # NVIDIA GPU passthrough (CDI)
      extraOptions = lib.optionals (cfg.device == "cuda") [
        "--gpus"
        "all"
        "--runtime"
        "nvidia"
      ];
    };

    # Garantir que o diretório de outputs exista
    systemd.tmpfiles.rules = [
      "d ${cfg.outputsDir}               0755 kernelcore kernelcore -"
      "d ${cfg.outputsDir}/transcripts   0755 kernelcore kernelcore -"
      "d ${cfg.outputsDir}/subtitles     0755 kernelcore kernelcore -"
    ];

    # Firewall opcional
    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}
