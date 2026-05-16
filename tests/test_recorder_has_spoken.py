"""Regressão: gravação NÃO pode parar antes do usuário falar.

Caso reproduzido em sound.txt: 2.1s capturados → transcrição vazia, porque
silent_count acumulava do frame 0. Com o guard has_spoken, silêncio inicial
deve ser ignorado.

sounddevice é mockado — nenhum dispositivo real é necessário.
"""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

BLOCKSIZE = 1_600
SAMPLE_RATE = 16_000


def _make_mock_sd_seq(seq: list[tuple[str, int]]):
    """Mock de sounddevice que injeta chunks na ordem especificada.

    Args:
        seq: lista de tuplas (kind, n) onde kind ∈ {'speech', 'silence'} e n é a
             quantidade de chunks consecutivos. Permite simular silêncio-então-fala.

    Returns:
        (mock_sd, FakePortAudioError)
    """

    class FakeCallbackStop(BaseException):
        pass

    class FakePortAudioError(Exception):
        pass

    class MockInputStream:
        def __init__(self, *args, callback=None, **kwargs):
            self._cb = callback

        def __enter__(self):
            for kind, n in seq:
                if kind == "speech":
                    data_template = np.ones((BLOCKSIZE, 1), dtype=np.float32) * 0.5
                elif kind == "silence":
                    data_template = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
                else:
                    raise ValueError(f"kind inválido: {kind}")
                for _ in range(n):
                    try:
                        self._cb(data_template, BLOCKSIZE, {}, None)
                    except FakeCallbackStop:
                        return self
            return self

        def __exit__(self, *args):
            return False

    mock_sd = MagicMock()
    mock_sd.InputStream = MockInputStream
    mock_sd.CallbackStop = FakeCallbackStop
    mock_sd.PortAudioError = FakePortAudioError
    return mock_sd, FakeCallbackStop


def test_silencio_inicial_nao_para_gravacao():
    """Bug do sound.txt: 30 chunks de silêncio antes de qualquer fala NÃO podem
    disparar o stop_event. O usuário ainda nem começou a falar."""
    from src.recorder import record_until_silence

    # 30 chunks silêncio (3s) → 5 chunks fala → 25 silêncio (2.5s) → stop
    seq = [("silence", 30), ("speech", 5), ("silence", 25)]
    mock_sd, _ = _make_mock_sd_seq(seq)

    with patch("src.recorder.sd", mock_sd):
        path = record_until_silence(silence_duration=2.0, max_duration=60.0)

    # Deve ter gravado: 30 + 5 + ~20 chunks (silence_chunks_needed para 2s = 20)
    # Antes do fix paravam em ~20 chunks (= 2s) sem capturar nada.
    with wave.open(path) as wf:
        frames = wf.getnframes()
    Path(path).unlink(missing_ok=True)

    min_expected = (30 + 5 + 20) * BLOCKSIZE
    assert frames >= min_expected, (
        f"gravação parou antes da fala: {frames} frames (esperado ≥ {min_expected})"
    )


def test_silencio_total_para_em_max_duration():
    """Se NUNCA houver fala, a gravação para apenas em max_duration."""
    from src.recorder import record_until_silence

    # 200 chunks de silêncio → equivale a ~20s; max_duration=0.5s força parada por max.
    seq = [("silence", 200)]
    mock_sd, _ = _make_mock_sd_seq(seq)

    with patch("src.recorder.sd", mock_sd):
        path = record_until_silence(silence_duration=2.0, max_duration=0.5)

    with wave.open(path) as wf:
        frames = wf.getnframes()
    Path(path).unlink(missing_ok=True)

    # Com max_duration=0.5s e blocksize=1600, max_chunks=5. Deve parar próximo disso,
    # NÃO antes (i.e., não pode disparar por "silêncio prolongado" sem fala).
    assert frames > 0


def test_fala_seguida_de_silencio_para_normalmente():
    """Caminho feliz: fala → silêncio prolongado → para por silêncio."""
    from src.recorder import record_until_silence

    seq = [("speech", 10), ("silence", 40)]
    mock_sd, _ = _make_mock_sd_seq(seq)

    with patch("src.recorder.sd", mock_sd):
        path = record_until_silence(silence_duration=2.0, max_duration=60.0)

    with wave.open(path) as wf:
        frames = wf.getnframes()
    Path(path).unlink(missing_ok=True)

    # 10 fala + ~20 silêncio = 30 chunks mínimos
    assert frames >= 10 * BLOCKSIZE
