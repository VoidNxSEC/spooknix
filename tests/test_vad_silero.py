"""Testes para src/vad_silero.py — sem baixar o modelo real.

Mockamos `silero_vad.load_silero_vad` para retornar um callable previsível.
Os testes validam threshold, warm-up e hangover sticky.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def fake_silero(monkeypatch):
    """Instala um fake `silero_vad` e `torch` no sys.modules."""
    fake_module = types.ModuleType("silero_vad")
    fake_module.load_silero_vad = MagicMock()
    monkeypatch.setitem(sys.modules, "silero_vad", fake_module)

    # torch é necessário para .from_numpy
    pytest.importorskip("torch")
    return fake_module


def _model_factory(probs: list[float]):
    """Cria fake model que retorna probs sucessivos via .item()."""
    it = iter(probs * 100)  # repete o suficiente
    fake_model = MagicMock()

    def _call(tensor, sr):
        result = MagicMock()
        result.item.return_value = next(it)
        return result

    fake_model.side_effect = _call
    return fake_model


def test_warmup_descarta_primeiros_chunks(fake_silero):
    """Os primeiros 6 chunks (WARMUP_CHUNKS) sempre retornam False."""
    fake_silero.load_silero_vad.return_value = _model_factory([0.9])
    from src.vad_silero import SileroVAD, WARMUP_CHUNKS, WINDOW_SAMPLES

    vad = SileroVAD(threshold=0.5, hangover=0)
    audio = np.ones(WINDOW_SAMPLES, dtype=np.float32)
    for i in range(WARMUP_CHUNKS):
        assert vad.is_speech(audio) is False
    # Próximo chunk já vale
    assert vad.is_speech(audio) is True


def test_threshold_separa_speech_silencio(fake_silero):
    """Probabilidade > threshold → speech; menor → silencio."""
    fake_silero.load_silero_vad.return_value = _model_factory([0.1])
    from src.vad_silero import SileroVAD, WARMUP_CHUNKS, WINDOW_SAMPLES

    vad = SileroVAD(threshold=0.5, hangover=0)
    audio = np.ones(WINDOW_SAMPLES, dtype=np.float32)
    # Pula warm-up
    for _ in range(WARMUP_CHUNKS):
        vad.is_speech(audio)
    # Agora chunks "reais" — prob=0.1 < 0.5 = não speech
    assert vad.is_speech(audio) is False


def test_hangover_mantem_speech_em_silencio_curto(fake_silero):
    """Após detectar speech, hangover mantém True por N chunks de silêncio."""
    from src.vad_silero import SileroVAD, WARMUP_CHUNKS, WINDOW_SAMPLES
    # Warm-up consome WARMUP_CHUNKS probs; depois: speech, silêncios.
    probs = [0.0] * WARMUP_CHUNKS + [0.9] + [0.1] * 10
    fake_silero.load_silero_vad.return_value = _model_factory(probs)

    vad = SileroVAD(threshold=0.5, hangover=2)
    audio = np.ones(WINDOW_SAMPLES, dtype=np.float32)
    # Warm-up
    for _ in range(WARMUP_CHUNKS):
        vad.is_speech(audio)
    # Chunk de speech (prob=0.9)
    assert vad.is_speech(audio) is True
    # Próximos 2 silêncios: sustentados pelo hangover
    assert vad.is_speech(audio) is True
    assert vad.is_speech(audio) is True
    # 3º silêncio: hangover esgotou
    assert vad.is_speech(audio) is False


def test_reset_zera_estado(fake_silero):
    fake_silero.load_silero_vad.return_value = _model_factory([0.9])
    from src.vad_silero import SileroVAD, WARMUP_CHUNKS, WINDOW_SAMPLES

    vad = SileroVAD()
    audio = np.ones(WINDOW_SAMPLES, dtype=np.float32)
    for _ in range(WARMUP_CHUNKS + 2):
        vad.is_speech(audio)
    assert vad._warm == WARMUP_CHUNKS
    vad.reset()
    assert vad._warm == 0
    assert vad._silent_streak == 0
    assert vad._last_decision is False


def test_sample_rate_invalido_explode():
    from src.vad_silero import SileroVAD
    with pytest.raises(ValueError):
        SileroVAD(sample_rate=44_100)


def test_import_error_se_silero_ausente():
    """Se silero_vad não está instalado, levanta ImportError com hint."""
    from src import vad_silero
    vad = vad_silero.SileroVAD()
    with patch.dict(sys.modules, {"silero_vad": None}):
        sys.modules.pop("silero_vad", None)
        with pytest.raises(ImportError, match="audio-quality"):
            vad._load()
