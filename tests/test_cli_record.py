"""Testes de integração para o comando `spooknix record` (src/cli.py).

O comando grava do microfone e envia o WAV via HTTP para o servidor.
Todas as dependências externas (recorder, urllib, subprocess) são mockadas.
Nenhum áudio real, GPU ou servidor é necessário.
"""

from __future__ import annotations

import json
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from click.testing import CliRunner

from src.cli import cli


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_url_response(payload: dict, status: int = 200) -> MagicMock:
    """Cria um mock de resposta HTTP compatível com urllib.request.urlopen."""
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_wav(tmp_path: Path) -> str:
    """WAV temporário válido (100 amostras de silêncio)."""
    path = tmp_path / "test.wav"
    samples = np.zeros(100, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(samples.tobytes())
    return str(path)


@pytest.fixture()
def health_response():
    return {"status": "ok", "model": "small", "device": "cuda", "cuda": True}


@pytest.fixture()
def transcribe_response():
    return {"text": "Olá, mundo! Isso é um teste.", "language": "pt", "duration": 1.5}


# ── Testes principais ─────────────────────────────────────────────────────────


def test_record_basico(fake_wav, health_response, transcribe_response):
    """Fluxo completo: health check → grava → POST → exibe resultado."""
    runner = CliRunner()

    health_resp = _make_url_response(health_response)
    transcribe_resp = _make_url_response(transcribe_response)

    with patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("urllib.request.urlopen", side_effect=[health_resp, transcribe_resp]), \
         patch("os.unlink"):

        result = runner.invoke(cli, ["record", "--language", "pt"])

    assert result.exit_code == 0, result.output
    assert "Olá, mundo! Isso é um teste." in result.output


def test_record_servidor_fora_do_ar(fake_wav):
    """Se o servidor não responde no health check, exibe erro e sai sem gravar."""
    import urllib.error
    runner = CliRunner()

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("recusado")), \
         patch("src.recorder.record_until_silence") as mock_rec:

        result = runner.invoke(cli, ["record"])

    assert result.exit_code == 0, result.output
    assert "Servidor não disponível" in result.output
    mock_rec.assert_not_called()


def test_record_com_clip_chama_wl_copy(fake_wav, health_response, transcribe_response):
    """--clip chama `wl-copy` com o texto transcrito."""
    runner = CliRunner()

    health_resp = _make_url_response(health_response)
    transcribe_resp = _make_url_response(transcribe_response)

    with patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("urllib.request.urlopen", side_effect=[health_resp, transcribe_resp]), \
         patch("subprocess.run") as mock_run, \
         patch("os.unlink"):

        result = runner.invoke(cli, ["record", "--clip"])

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(
        ["wl-copy", transcribe_response["text"]],
        check=True,
        timeout=5,
    )
    assert "Copiado para o clipboard" in result.output


def test_record_clip_sem_wl_copy(fake_wav, health_response, transcribe_response):
    """Se wl-copy não existir, exibe aviso sem falhar."""
    runner = CliRunner()

    health_resp = _make_url_response(health_response)
    transcribe_resp = _make_url_response(transcribe_response)

    with patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("urllib.request.urlopen", side_effect=[health_resp, transcribe_resp]), \
         patch("subprocess.run", side_effect=FileNotFoundError), \
         patch("os.unlink"):

        result = runner.invoke(cli, ["record", "--clip"])

    assert result.exit_code == 0, result.output
    assert "wl-copy não encontrado" in result.output


def test_record_sem_texto_nao_chama_wl_copy(fake_wav, health_response):
    """Se a transcrição for vazia, wl-copy NÃO deve ser chamado."""
    runner = CliRunner()
    empty_result = {"text": "", "language": "pt", "duration": 0.5}

    health_resp = _make_url_response(health_response)
    transcribe_resp = _make_url_response(empty_result)

    with patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("urllib.request.urlopen", side_effect=[health_resp, transcribe_resp]), \
         patch("subprocess.run") as mock_run, \
         patch("os.unlink"):

        result = runner.invoke(cli, ["record", "--clip"])

    assert result.exit_code == 0, result.output
    mock_run.assert_not_called()


def test_record_recording_error_exibe_mensagem(health_response):
    """RecordingError exibe mensagem de erro e termina sem crash."""
    from src.recorder import RecordingError

    runner = CliRunner()
    health_resp = _make_url_response(health_response)

    with patch("urllib.request.urlopen", return_value=health_resp), \
         patch("src.recorder.record_until_silence",
               side_effect=RecordingError("microfone não encontrado")):

        result = runner.invoke(cli, ["record"])

    assert result.exit_code == 0, result.output
    assert "microfone não encontrado" in result.output


def test_record_apaga_arquivo_temporario(fake_wav, health_response, transcribe_response):
    """O arquivo WAV temporário é deletado após a transcrição."""
    runner = CliRunner()

    health_resp = _make_url_response(health_response)
    transcribe_resp = _make_url_response(transcribe_response)

    with patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("urllib.request.urlopen", side_effect=[health_resp, transcribe_resp]), \
         patch("os.unlink") as mock_unlink:

        result = runner.invoke(cli, ["record"])

    assert result.exit_code == 0, result.output
    mock_unlink.assert_called_once_with(fake_wav)


def test_record_apaga_tmp_mesmo_em_erro_de_post(fake_wav, health_response):
    """O WAV temporário é deletado mesmo quando o POST ao servidor falha."""
    import urllib.error
    runner = CliRunner()

    health_resp = _make_url_response(health_response)

    with patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("urllib.request.urlopen",
               side_effect=[health_resp, urllib.error.URLError("timeout")]), \
         patch("os.unlink") as mock_unlink:

        result = runner.invoke(cli, ["record"])

    mock_unlink.assert_called_once_with(fake_wav)


def test_record_language_enviada_no_form(fake_wav, health_response, transcribe_response):
    """A opção --language é incluída no body multipart enviado ao servidor."""
    runner = CliRunner()

    health_resp = _make_url_response(health_response)
    transcribe_resp = _make_url_response(transcribe_response)

    captured_request: list = []
    call_count = 0

    def fake_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return health_resp
        if hasattr(req, "data") and req.data is not None:
            captured_request.append(req.data)
        return transcribe_resp

    with patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("urllib.request.urlopen", side_effect=fake_urlopen), \
         patch("os.unlink"):

        result = runner.invoke(cli, ["record", "--language", "en"])

    assert result.exit_code == 0, result.output
    assert len(captured_request) == 1
    assert b"en" in captured_request[0]
