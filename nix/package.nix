# nix/package.nix — pacote do spooknix-gui (thin client PyQt6, sem ML)
{
  lib,
  python313,
  makeWrapper,
  writeShellApplication,
}:

let
  pyEnv = python313.withPackages (ps: [
    ps.pyqt6
    ps.pyqt6-sip
    ps.requests
    ps.numpy
    ps.sounddevice
    ps.scipy
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
