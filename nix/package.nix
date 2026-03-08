# nix/package.nix — pacote do spooknix-gui (thin client PyQt6, sem ML)
{
  lib,
  python3,
  makeWrapper,
  writeShellApplication,
}:

let
  pyEnv = python3.withPackages (ps: [
    ps.pyqt6
    ps.pyqt6-sip
    ps.requests
  ]);

  guiScript = writeShellApplication {
    name = "spooknix-gui";
    runtimeInputs = [ pyEnv ];
    text = ''
      export PYTHONPATH="@src@''${PYTHONPATH:+:$PYTHONPATH}"
      exec python -m src.gui "$@"
    '';
  };
in
  guiScript
