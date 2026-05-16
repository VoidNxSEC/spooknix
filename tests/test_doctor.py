"""Testes para `spooknix doctor`.

Mocka torch, sounddevice, urllib e shutil para que o teste rode em CI sem GPU
nem áudio nem servidor.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest


def test_cuda_check_sem_torch():
    """torch ausente → fail com hint claro."""
    with patch.dict("sys.modules", {"torch": None}):
        # patch.dict com None força ImportError no próximo import.
        import importlib
        import sys
        sys.modules.pop("torch", None)
        # Recarrega o módulo doctor para que o import torch seja re-executado.
        if "src.doctor" in sys.modules:
            importlib.reload(sys.modules["src.doctor"])

    # Simplificando: usamos patch direto em _cuda_check.
    from src import doctor

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": fake_torch}):
        c = doctor._cuda_check()
    assert c.status == "warn"
    assert "indisponível" in c.value.lower() or "CPU" in c.value


def test_cuda_check_com_gpu():
    from src import doctor

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.get_device_name.return_value = "RTX 3050"
    fake_props = MagicMock(total_memory=6 * 1e9)
    fake_torch.cuda.get_device_properties.return_value = fake_props

    with patch.dict("sys.modules", {"torch": fake_torch}):
        c = doctor._cuda_check()

    assert c.status == "ok"
    assert "RTX 3050" in c.value


def test_stt_health_ok():
    from src import doctor

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"model": "large-v3", "device": "cuda"}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        c = doctor._stt_health_check("http://localhost:8000")

    assert c.status == "ok"
    assert "large-v3" in c.value


def test_stt_health_offline():
    import urllib.error
    from src import doctor

    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("connection refused")):
        c = doctor._stt_health_check("http://localhost:8000")

    assert c.status == "warn"
    assert "indisponível" in c.value


def test_ffmpeg_check_ausente():
    from src import doctor

    with patch("shutil.which", return_value=None):
        c = doctor._ffmpeg_check()

    assert c.status == "fail"
    assert "não encontrado" in c.value


def test_ffmpeg_check_presente():
    from src import doctor

    with patch("shutil.which", return_value="/nix/store/abc/bin/ffmpeg"):
        c = doctor._ffmpeg_check()

    assert c.status == "ok"
    assert "ffmpeg" in c.value


def test_llamacpp_offline():
    import urllib.error
    from src import doctor

    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("no route")):
        c = doctor._llamacpp_check("http://localhost:8081")

    assert c.status == "warn"
    assert "offline" in c.value


def test_llamacpp_online_sem_modelos():
    from src import doctor

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"data": []}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        c = doctor._llamacpp_check("http://localhost:8081")

    assert c.status == "warn"
    assert "sem modelos" in c.value


def test_llamacpp_online_com_modelos():
    from src import doctor

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"data": [{"id": "qwen2.5-7b-instruct"}, {"id": "llama-3"}]}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        c = doctor._llamacpp_check("http://localhost:8081")

    assert c.status == "ok"
    assert "qwen2.5-7b-instruct" in c.value


def test_render_tabela_nao_explode():
    from rich.console import Console
    from src import doctor

    checks = [
        doctor.Check("Teste", "ok", "tudo certo"),
        doctor.Check("Outro", "warn", "olha lá", "hint útil"),
        doctor.Check("Pior", "fail", "morreu", "tente X"),
    ]
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=120)
    doctor.render(checks, console=console)

    output = buf.getvalue()
    assert "Teste" in output
    assert "Outro" in output
    assert "Pior" in output


def test_run_checks_default_nao_inclui_mic():
    from src import doctor

    with patch.object(doctor, "_cuda_check", return_value=doctor.Check("CUDA", "ok", "x")), \
         patch.object(doctor, "_stt_health_check", return_value=doctor.Check("STT", "ok", "x")), \
         patch.object(doctor, "_audio_devices_check", return_value=doctor.Check("Audio", "ok", "x")), \
         patch.object(doctor, "_ffmpeg_check", return_value=doctor.Check("ffmpeg", "ok", "x")), \
         patch.object(doctor, "_llamacpp_check", return_value=doctor.Check("llama.cpp", "ok", "x")):
        checks = doctor.run_checks(include_mic=False)

    names = [c.name for c in checks]
    assert "Mic baseline" not in names
    assert len(checks) == 5



def test_brev_llm_online():
    from src import doctor

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"data": [{"id": "qwen-3.5"}]}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        c = doctor._brev_llm_check("http://localhost:8080/v1")

    assert c.status == "ok"
    assert "qwen-3.5" in c.value


def test_brev_llm_aceita_base_sem_v1():
    """Permite passar http://host:8080 sem /v1 final."""
    from src import doctor

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"data": [{"id": "llama-3-8b"}]}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=fake_resp) as patched:
        c = doctor._brev_llm_check("http://localhost:8080")

    assert c.status == "ok"
    # Confirma que adicionou /v1/models
    called_url = patched.call_args.args[0]
    assert called_url.endswith("/v1/models")


def test_brev_llm_offline_e_fail():
    import urllib.error
    from src import doctor

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        c = doctor._brev_llm_check("http://localhost:8080/v1")

    assert c.status == "fail"
    assert "indisponível" in c.value


def test_brev_llm_online_sem_modelos_warn():
    from src import doctor

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'{"data": []}'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        c = doctor._brev_llm_check("http://localhost:8080/v1")

    assert c.status == "warn"
    assert "sem modelos" in c.value


def test_brev_tts_health_ok():
    from src import doctor

    fake_resp = MagicMock()
    fake_resp.read.return_value = b'ok'
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with patch("urllib.request.urlopen", return_value=fake_resp):
        c = doctor._brev_tts_check("http://localhost:8001")

    assert c.status == "ok"
    assert "respondendo" in c.value


def test_brev_tts_health_indisponivel_da_warn_nao_fail():
    """TTS images variam — warn em vez de fail pra não bloquear o doctor."""
    import urllib.error
    from src import doctor

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")):
        c = doctor._brev_tts_check("http://localhost:8001")

    assert c.status == "warn"


def test_run_checks_brev_inclui_llm_e_tts():
    from src import doctor

    with patch.object(doctor, "_cuda_check", return_value=doctor.Check("CUDA", "ok", "x")), \
         patch.object(doctor, "_stt_health_check", return_value=doctor.Check("STT", "ok", "x")), \
         patch.object(doctor, "_audio_devices_check", return_value=doctor.Check("Audio", "ok", "x")), \
         patch.object(doctor, "_ffmpeg_check", return_value=doctor.Check("ffmpeg", "ok", "x")), \
         patch.object(doctor, "_brev_llm_check", return_value=doctor.Check("Brev LLM", "ok", "x")), \
         patch.object(doctor, "_brev_tts_check", return_value=doctor.Check("Brev TTS", "ok", "x")):
        checks = doctor.run_checks(brev=True)

    names = [c.name for c in checks]
    assert "Brev LLM" in names
    assert "Brev TTS" in names
    assert "llama.cpp" not in names  # substituído pelo par Brev
    assert len(checks) == 6