"""Testes de integração para o comando `stt record` (src/cli.py).

Todas as dependências pesadas (modelo, recorder, torch) são mockadas.
Nenhum áudio real ou GPU é necessário.
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from click.testing import CliRunner

from src.cli import cli


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_wav(tmp_path: Path) -> str:
    """WAV temporário válido (100 amostras de silêncio) para usar como retorno do recorder."""
    path = tmp_path / "test.wav"
    samples = np.zeros(100, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(samples.tobytes())
    return str(path)


@pytest.fixture()
def mock_model():
    return MagicMock(name="WhisperModel")


@pytest.fixture()
def transcribe_result():
    return {
        "text": "Olá, mundo! Isso é um teste.",
        "language": "pt",
        "duration": 1.5,
        "segments": [],
    }


# ── Testes principais ─────────────────────────────────────────────────────────


def test_record_basico(fake_wav, mock_model, transcribe_result):
    """Fluxo completo: grava → transcreve → exibe resultado no terminal."""
    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", return_value=transcribe_result):

        result = runner.invoke(cli, ["record", "--language", "pt"])

    assert result.exit_code == 0, result.output
    assert "Olá, mundo! Isso é um teste." in result.output


def test_record_com_clip_chama_wl_copy(fake_wav, mock_model, transcribe_result):
    """--clip chama `wl-copy` com o texto transcrito."""
    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", return_value=transcribe_result), \
         patch("subprocess.run") as mock_run:

        result = runner.invoke(cli, ["record", "--clip"])

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(
        ["wl-copy", transcribe_result["text"]],
        check=True,
        timeout=5,
    )
    assert "Copiado para o clipboard" in result.output


def test_record_clip_sem_wl_copy(fake_wav, mock_model, transcribe_result):
    """Se wl-copy não existir, exibe aviso sem falhar."""
    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", return_value=transcribe_result), \
         patch("subprocess.run", side_effect=FileNotFoundError):

        result = runner.invoke(cli, ["record", "--clip"])

    assert result.exit_code == 0, result.output
    assert "wl-copy não encontrado" in result.output


def test_record_sem_texto_nao_chama_wl_copy(fake_wav, mock_model):
    """Se a transcrição for vazia, wl-copy NÃO deve ser chamado."""
    runner = CliRunner()
    empty_result = {"text": "", "language": "pt", "duration": 0.5, "segments": []}

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", return_value=empty_result), \
         patch("subprocess.run") as mock_run:

        result = runner.invoke(cli, ["record", "--clip"])

    assert result.exit_code == 0, result.output
    mock_run.assert_not_called()


def test_record_recording_error_exibe_mensagem(mock_model):
    """RecordingError exibe mensagem de erro e termina sem crash."""
    from src.recorder import RecordingError

    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence",
               side_effect=RecordingError("microfone não encontrado")):

        result = runner.invoke(cli, ["record"])

    assert result.exit_code == 0, result.output
    assert "microfone não encontrado" in result.output


def test_record_apaga_arquivo_temporario(fake_wav, mock_model, transcribe_result):
    """O arquivo WAV temporário é deletado após a transcrição."""
    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", return_value=transcribe_result), \
         patch("os.unlink") as mock_unlink:

        result = runner.invoke(cli, ["record"])

    assert result.exit_code == 0, result.output
    mock_unlink.assert_called_once_with(fake_wav)


def test_record_apaga_tmp_mesmo_em_erro(fake_wav, mock_model):
    """O arquivo WAV temporário é deletado mesmo quando a transcrição falha."""
    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", side_effect=RuntimeError("falha")), \
         patch("os.unlink") as mock_unlink:

        result = runner.invoke(cli, ["record"])

    # O unlink deve ter sido chamado no bloco finally
    mock_unlink.assert_called_once_with(fake_wav)


def test_record_modelo_tiny(fake_wav, mock_model, transcribe_result):
    """--model tiny passa o tamanho correto para get_model."""
    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model) as mock_get, \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", return_value=transcribe_result):

        result = runner.invoke(cli, ["record", "--model", "tiny"])

    assert result.exit_code == 0, result.output
    mock_get.assert_called_once_with("tiny", "cpu")


def test_record_language_option(fake_wav, mock_model, transcribe_result):
    """--language é passado corretamente para transcribe_file."""
    runner = CliRunner()

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.transcriber.get_model", return_value=mock_model), \
         patch("src.recorder.record_until_silence", return_value=fake_wav), \
         patch("src.transcriber.transcribe_file", return_value=transcribe_result) as mock_tf:

        runner.invoke(cli, ["record", "--language", "en"])

    # language deve ser "en"
    _, kwargs = mock_tf.call_args
    assert kwargs.get("language") == "en" or mock_tf.call_args[0][2] == "en"
