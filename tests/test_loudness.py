"""Testes para src/loudness.py (LUFS / EBU R128).

Quando pyloudnorm não está instalado (CI mínimo), a wrapper cai em RMS-normalize
— testamos esse caminho.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.loudness import LoudnessMeter, to_lufs, _rms_normalize


SAMPLE_RATE = 16_000


def _sine(freq: float = 440.0, duration: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_measure_buffer_curto_retorna_none():
    """Buffers < 0.4s não têm bloco mínimo do EBU R128."""
    m = LoudnessMeter(sample_rate=SAMPLE_RATE)
    short = _sine(duration=0.2)
    assert m.measure(short) is None


def test_normalize_buffer_curto_cai_em_rms():
    """Fallback RMS: RMS final deve estar próximo de target_rms=0.1."""
    m = LoudnessMeter(sample_rate=SAMPLE_RATE)
    short = _sine(duration=0.2, amp=0.8)
    out = m.normalize(short)
    rms = float(np.sqrt(np.mean(out ** 2)))
    assert abs(rms - 0.1) < 0.02


def test_normalize_vazio_passthrough():
    m = LoudnessMeter(sample_rate=SAMPLE_RATE)
    empty = np.zeros(0, dtype=np.float32)
    assert m.normalize(empty).size == 0


def test_rms_normalize_silencio_nao_explode():
    """Sinal silencioso (RMS=0) não pode dividir por zero."""
    silence = np.zeros(1600, dtype=np.float32)
    out = _rms_normalize(silence, target_rms=0.1)
    assert np.all(out == 0.0)


def test_to_lufs_funcional():
    """A versão funcional retorna mesmo shape."""
    audio = _sine(duration=0.5, amp=0.3)
    out = to_lufs(audio, sample_rate=SAMPLE_RATE)
    assert out.shape == audio.shape
    assert out.dtype == np.float32


def test_loudness_meter_com_pyloudnorm_se_disponivel():
    """Se pyloudnorm está instalado, measure() retorna float; senão, None."""
    pyln = pytest.importorskip("pyloudnorm")
    m = LoudnessMeter(sample_rate=SAMPLE_RATE)
    audio = _sine(duration=1.0, amp=0.5)
    lufs = m.measure(audio)
    assert lufs is not None
    assert -60.0 < lufs < 0.0  # speech típico


def test_loudness_normalize_converge_para_target():
    pytest.importorskip("pyloudnorm")
    m = LoudnessMeter(sample_rate=SAMPLE_RATE, target_lufs=-23.0)
    audio = _sine(duration=1.0, amp=0.1)
    out = m.normalize(audio)
    lufs_out = m.measure(out)
    assert lufs_out is not None
    assert abs(lufs_out - (-23.0)) < 1.0
