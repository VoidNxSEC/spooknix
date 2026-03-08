# nix/modules/home-manager/default.nix — Home-Manager module para o GUI Spooknix
#
# Ativa o systray como user service, opcionalmente com integração Hyprland e Waybar.
#
# Uso no home.nix do usuário:
#   programs.spooknix = {
#     enable = true;
#     hyprland.enable = true;
#     waybar.enable   = true;
#   };
{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.programs.spooknix;

  # Pacote PyQt6 mínimo para a GUI
  guiPkg = pkgs.python3.withPackages (ps: [
    ps.pyqt6
    ps.pyqt6-sip
    ps.requests
  ]);

  # Script wrapper que aponta PYTHONPATH para o source
  guiBin = pkgs.writeShellApplication {
    name = "spooknix-gui";
    runtimeInputs = [ guiPkg ];
    text = ''
      export SPOOKNIX_URL="${cfg.serverUrl}"
      exec python -c "import sys; sys.path.insert(0, '${cfg.sourcePath}'); from src.gui import main; main()"
    '';
  };
in
{
  # ── Options ──────────────────────────────────────────────────────────────
  options.programs.spooknix = {
    enable = mkEnableOption "Spooknix systray GUI (user service)";

    serverUrl = mkOption {
      type = types.str;
      default = "http://localhost:8000";
      description = "URL base do backend Spooknix.";
    };

    sourcePath = mkOption {
      type = types.str;
      default = "/home/kernelcore/arch/spooknix";
      description = "Caminho para o código-fonte do Spooknix (necessário para PYTHONPATH).";
    };

    autostart = mkOption {
      type = types.bool;
      default = true;
      description = "Iniciar automaticamente com a sessão gráfica.";
    };

    hyprland = {
      enable = mkEnableOption "Integração Hyprland (windowrules + keybind)";

      keybind = mkOption {
        type = types.str;
        default = "SUPER, S";
        description = "Tecla de atalho para mostrar/ocultar a janela Spooknix (formato Hyprland).";
        example = "SUPER, S";
      };

      windowRules = mkOption {
        type = types.listOf types.str;
        default = [
          "float on, match:class ^(spooknix)$"
          "size 380 480, match:class ^(spooknix)$"
          "center on, match:class ^(spooknix)$"
          "animation slide, match:class ^(spooknix)$"
          "rounding 12, match:class ^(spooknix)$"
        ];
        description = "Regras de janela Hyprland para o Spooknix.";
      };
    };

    waybar = {
      enable = mkEnableOption "Widget Waybar com status do servidor STT";

      interval = mkOption {
        type = types.int;
        default = 10;
        description = "Intervalo de polling do /health em segundos.";
      };
    };
  };

  # ── Config ───────────────────────────────────────────────────────────────
  config = mkIf cfg.enable {

    # ── User service (systemd) ──────────────────────────────────────────────
    systemd.user.services.spooknix-gui = mkIf cfg.autostart {
      Unit = {
        Description = "Spooknix STT — systray GUI";
        After = [ "graphical-session.target" ];
        PartOf = [ "graphical-session.target" ];
      };
      Service = {
        ExecStart = "${guiBin}/bin/spooknix-gui";
        Restart = "on-failure";
        RestartSec = 5;
        Environment = [
          "SPOOKNIX_URL=${cfg.serverUrl}"
          "QT_QPA_PLATFORM=xcb"
        ];
      };
      Install = {
        WantedBy = [ "graphical-session.target" ];
      };
    };

    # ── XDG Desktop Entry ──────────────────────────────────────────────────
    xdg.desktopEntries.spooknix = {
      name = "Spooknix";
      comment = "Privacy-first Speech-to-Text";
      exec = "${guiBin}/bin/spooknix-gui";
      icon = "audio-input-microphone";
      categories = [ "Utility" "AudioVideo" ];
      startupNotify = false;
    };

    # ── Hyprland ───────────────────────────────────────────────────────────
    wayland.windowManager.hyprland.settings = mkIf cfg.hyprland.enable {
      bind = [
        "${cfg.hyprland.keybind}, exec, ${guiBin}/bin/spooknix-gui"
      ];
      windowrule = cfg.hyprland.windowRules;
    };

    # ── Waybar ─────────────────────────────────────────────────────────────
    programs.waybar.settings.mainBar = mkIf cfg.waybar.enable {
      "custom/spooknix" = {
        format = "{}";
        exec = pkgs.writeShellScript "spooknix-waybar" ''
          URL="${cfg.serverUrl}/health"
          RESP=$(${pkgs.curl}/bin/curl -sf --max-time 2 "$URL" 2>/dev/null || echo '{}')

          if echo "$RESP" | ${pkgs.gnugrep}/bin/grep -q '"status":"ok"'; then
            MODEL=$(echo "$RESP" | ${pkgs.gnused}/bin/sed 's/.*"model":"\([^"]*\)".*/\1/')
            DEVICE=$(echo "$RESP" | ${pkgs.gnused}/bin/sed 's/.*"device":"\([^"]*\)".*/\1/')
            echo "{\"text\": \"🎙 $MODEL\", \"class\": \"active\", \"tooltip\": \"Spooknix online · $DEVICE\"}"
          else
            echo '{"text": "🎙", "class": "inactive", "tooltip": "Spooknix offline"}'
          fi
        '';
        return-type = "json";
        interval = cfg.waybar.interval;
        on-click = "${guiBin}/bin/spooknix-gui";
      };
    };

    programs.waybar.style = mkIf cfg.waybar.enable ''
      #custom-spooknix {
        padding: 0 10px;
        margin: 0 4px;
        border-radius: 8px;
        background: rgba(30, 30, 46, 0.8);
        color: #cdd6f4;
        font-size: 13px;
      }

      #custom-spooknix.active {
        background: rgba(137, 180, 250, 0.2);
        color: #89b4fa;
        border: 1px solid rgba(137, 180, 250, 0.3);
      }

      #custom-spooknix.inactive {
        background: rgba(243, 139, 168, 0.1);
        color: #585b70;
      }
    '';
  };
}
