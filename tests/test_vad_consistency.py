"""Garante que transcribe_file e transcribe_stream usam o mesmo VAD config.

Antes da refatoração, transcribe_file passava {threshold: 0.4, min_silence_duration_ms: 500}
e transcribe_stream usava o default do faster-whisper (~0.5). Resultado: comportamento
divergente entre o `file` e o `stream` da CLI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_vad_params_constant_existe():
    from src.transcriber import _VAD_PARAMS

    assert "threshold" in _VAD_PARAMS
    assert "min_silence_duration_ms" in _VAD_PARAMS
    assert _VAD_PARAMS["threshold"] == 0.4
    assert _VAD_PARAMS["min_silence_duration_ms"] == 500


def test_transcribe_file_passa_vad_params():
    from src import transcriber

    mock_model = MagicMock()
    mock_info = MagicMock(duration=1.0, language="pt", language_probability=1.0)
    mock_model.transcribe.return_value = ([], mock_info)

    transcriber.transcribe_file(mock_model, "fake.wav", language="pt")

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs.get("vad_filter") is True
    assert call_kwargs.get("vad_parameters") is transcriber._VAD_PARAMS


def test_transcribe_stream_passa_vad_params():
    from src import transcriber

    mock_model = MagicMock()
    mock_info = MagicMock()
    mock_model.transcribe.return_value = ([], mock_info)

    audio = np.zeros(16_000, dtype=np.float32)
    # Consome o generator
    list(transcriber.transcribe_stream(mock_model, audio, language="pt"))

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs.get("vad_filter") is True
    assert call_kwargs.get("vad_parameters") is transcriber._VAD_PARAMS


def test_vad_params_compartilhado_por_referencia():
    """Ambas as funções devem referenciar o MESMO dict — não cópias."""
    from src import transcriber

    mock_model_file = MagicMock()
    mock_model_file.transcribe.return_value = (
        [], MagicMock(duration=1.0, language="pt", language_probability=1.0)
    )
    transcriber.transcribe_file(mock_model_file, "fake.wav", language="pt")
    params_file = mock_model_file.transcribe.call_args.kwargs["vad_parameters"]

    mock_model_stream = MagicMock()
    mock_model_stream.transcribe.return_value = ([], MagicMock())
    list(transcriber.transcribe_stream(
        mock_model_stream, np.zeros(16_000, dtype=np.float32), language="pt"
    ))
    params_stream = mock_model_stream.transcribe.call_args.kwargs["vad_parameters"]

    assert params_file is params_stream
