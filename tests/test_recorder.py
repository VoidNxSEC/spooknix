"""Testes unitários para src/recorder.py.

sounddevice é mockado — nenhum dispositivo de áudio real é necessário.
"""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

BLOCKSIZE = 1_600
SAMPLE_RATE = 16_000


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_sd(speech_chunks: int = 5, silence_chunks: int = 30):
    """Cria mock de sounddevice que injeta dados sintéticos no callback."""

    # Exceções fake que substituem sd.CallbackStop e sd.PortAudioError
    class FakeCallbackStop(BaseException):
        pass

    class FakePortAudioError(Exception):
        pass

    class MockInputStream:
        def __init__(self, *args, callback=None, **kwargs):
            self._cb = callback

        def __enter__(self):
            # Fala: chunks com RMS alto (0.5 >> 0.01 threshold)
            for _ in range(speech_chunks):
                data = np.ones((BLOCKSIZE, 1), dtype=np.float32) * 0.5
                try:
                    self._cb(data, BLOCKSIZE, {}, None)
                except FakeCallbackStop:
                    return self

            # Silêncio: chunks com RMS 0
            for _ in range(silence_chunks):
                data = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
                try:
                    self._cb(data, BLOCKSIZE, {}, None)
                except FakeCallbackStop:
                    return self

            return self

        def __exit__(self, *args):
            return False

    mock_sd = MagicMock()
    mock_sd.InputStream = MockInputStream
    mock_sd.CallbackStop = FakeCallbackStop
    mock_sd.PortAudioError = FakePortAudioError
    return mock_sd, FakePortAudioError


# ── _save_wav ─────────────────────────────────────────────────────────────────


def test_save_wav_formato_correto():
    """`_save_wav` produz WAV mono 16kHz int16 válido."""
    from src.recorder import _save_wav

    chunk = np.sin(np.linspace(0, 2 * np.pi, SAMPLE_RATE, dtype=np.float32)) * 0.8
    path = _save_wav([chunk], SAMPLE_RATE)
    try:
        with wave.open(path) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # int16 = 2 bytes
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnframes() == SAMPLE_RATE
    finally:
        Path(path).unlink(missing_ok=True)


def test_save_wav_clamp_overflow():
    """float32 fora de [-1, 1] é clamped antes de converter para int16."""
    from src.recorder import _save_wav

    chunk = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    path = _save_wav([chunk], SAMPLE_RATE)
    try:
        with wave.open(path) as wf:
            raw = wf.readframes(3)
        samples = np.frombuffer(raw, dtype=np.int16)
        assert samples[0] == 32767   # 2.0 → clamped
        assert samples[1] == -32768  # -3.0 → clamped
        assert -32767 <= samples[2] <= 32767
    finally:
        Path(path).unlink(missing_ok=True)


def test_save_wav_concatena_chunks():
    """`_save_wav` concatena múltiplos chunks corretamente."""
    from src.recorder import _save_wav

    c1 = np.ones(100, dtype=np.float32) * 0.1
    c2 = np.ones(200, dtype=np.float32) * 0.2
    path = _save_wav([c1, c2], SAMPLE_RATE)
    try:
        with wave.open(path) as wf:
            assert wf.getnframes() == 300
    finally:
        Path(path).unlink(missing_ok=True)


# ── record_until_silence ──────────────────────────────────────────────────────


def test_record_until_silence_retorna_wav():
    """`record_until_silence` retorna path de WAV válido em fluxo normal."""
    from src.recorder import record_until_silence

    mock_sd, _ = _make_mock_sd(speech_chunks=5, silence_chunks=30)
    with patch("src.recorder.sd", mock_sd):
        path = record_until_silence(silence_duration=2.0, max_duration=60.0)

    assert path.endswith(".wav")
    try:
        with wave.open(path) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE
    finally:
        Path(path).unlink(missing_ok=True)


def test_record_until_silence_para_no_silencio():
    """Gravação para quando detecta N chunks consecutivos de silêncio."""
    from src.recorder import record_until_silence

    mock_sd, _ = _make_mock_sd(speech_chunks=3, silence_chunks=40)
    with patch("src.recorder.sd", mock_sd):
        path = record_until_silence(silence_duration=2.0)

    # Deve ter gravado (speech + pelo menos silence_chunks_needed)
    with wave.open(path) as wf:
        total_frames = wf.getnframes()
    Path(path).unlink(missing_ok=True)

    # 3 speech + ~20 silence chunks (para silence_duration=2.0)
    # 23 chunks × 1600 frames = 36800 frames mínimo
    assert total_frames >= 3 * BLOCKSIZE


def test_record_until_silence_para_na_duracao_maxima():
    """Gravação para quando atinge max_duration mesmo sem silêncio."""
    from src.recorder import record_until_silence

    # Só fala, sem silêncio → deve parar por max_duration
    mock_sd, _ = _make_mock_sd(speech_chunks=200, silence_chunks=0)
    with patch("src.recorder.sd", mock_sd):
        # max_duration curtíssimo → max_chunks = 0 → para no 1º chunk
        path = record_until_silence(silence_duration=2.0, max_duration=0.05)

    with wave.open(path) as wf:
        assert wf.getnframes() > 0
    Path(path).unlink(missing_ok=True)


def test_record_until_silence_levanta_sem_audio():
    """Levanta `RecordingError` se nenhum áudio for capturado."""
    from src.recorder import record_until_silence, RecordingError

    class NoOpStream:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    mock_sd = MagicMock()
    mock_sd.InputStream = NoOpStream
    mock_sd.CallbackStop = BaseException
    mock_sd.PortAudioError = OSError

    with patch("src.recorder.sd", mock_sd), \
         patch("src.recorder.threading") as mock_threading:
        mock_event = MagicMock()
        mock_event.wait.return_value = None
        mock_event.is_set.return_value = True  # callback nunca chamado
        mock_threading.Event.return_value = mock_event

        with pytest.raises(RecordingError, match="Nenhum áudio"):
            record_until_silence()


def test_record_until_silence_levanta_portaudio_error():
    """Converte `sd.PortAudioError` em `RecordingError`."""
    from src.recorder import record_until_silence, RecordingError

    mock_sd, FakePortAudioError = _make_mock_sd()

    class FailingStream:
        def __init__(self, *a, **kw): pass
        def __enter__(self): raise FakePortAudioError("no device")
        def __exit__(self, *a): return False

    mock_sd.InputStream = FailingStream

    with patch("src.recorder.sd", mock_sd):
        with pytest.raises(RecordingError, match="Erro no dispositivo"):
            record_until_silence()


# ── record_fixed_duration ─────────────────────────────────────────────────────


def test_record_fixed_duration_produz_wav():
    """`record_fixed_duration` grava por N segundos e retorna WAV válido."""
    from src.recorder import record_fixed_duration

    mock_sd, _ = _make_mock_sd(speech_chunks=100, silence_chunks=0)
    with patch("src.recorder.sd", mock_sd):
        path = record_fixed_duration(duration=0.1)

    with wave.open(path) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
    Path(path).unlink(missing_ok=True)


def test_record_fixed_duration_levanta_portaudio_error():
    """Converte `sd.PortAudioError` em `RecordingError` em gravação fixa."""
    from src.recorder import record_fixed_duration, RecordingError

    mock_sd, FakePortAudioError = _make_mock_sd()

    class FailingStream:
        def __init__(self, *a, **kw): pass
        def __enter__(self): raise FakePortAudioError("no device")
        def __exit__(self, *a): return False

    mock_sd.InputStream = FailingStream

    with patch("src.recorder.sd", mock_sd):
        with pytest.raises(RecordingError, match="Erro no dispositivo"):
            record_fixed_duration(duration=1.0)
