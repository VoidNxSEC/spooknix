"""Testes para src/audio_meter.py — não roda Rich.Live de verdade.

Validamos apenas que .feed() atualiza as métricas corretamente e que
.render() retorna um Panel sem explodir.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rich.panel import Panel

from src.audio_meter import AudioMeter


def test_feed_silencio_da_db_baixissimo():
    m = AudioMeter(sample_rate=16_000)
    m.feed(np.zeros(1_600, dtype=np.float32))
    assert m.peak_db < -100.0
    assert m.rms_db < -100.0


def test_feed_sinal_alto_da_db_alto():
    m = AudioMeter(sample_rate=16_000)
    m.feed(np.ones(1_600, dtype=np.float32) * 0.5)
    assert m.peak_db > -7.0  # -6 dBFS ≈ 0.5
    assert m.peak_db < -5.0


def test_feed_chunk_vazio_nao_explode():
    m = AudioMeter(sample_rate=16_000)
    m.feed(np.zeros(0, dtype=np.float32))
    # Estado inicial preservado
    assert m.peak_db == pytest.approx(-120.0)


def test_render_retorna_panel_valido():
    m = AudioMeter(sample_rate=16_000)
    m.feed(np.ones(1_600, dtype=np.float32) * 0.3)
    panel = m.render()
    assert isinstance(panel, Panel)


def test_history_acumula_e_satura():
    m = AudioMeter(sample_rate=16_000)
    for _ in range(100):
        m.feed(np.ones(160, dtype=np.float32) * 0.1)
    # _HISTORY_LEN = 80; deve saturar
    assert len(m._history) == 80


def test_lufs_acumula_apos_1s():
    """LUFS só dispara quando o buffer interno cruza ~1s (pyloudnorm precisa ≥0.4s)."""
    pytest.importorskip("pyloudnorm")
    m = AudioMeter(sample_rate=16_000)
    # 10 chunks de 100ms = 1s
    for _ in range(10):
        m.feed(np.ones(1_600, dtype=np.float32) * 0.3)
    assert m.lufs is not None
    assert -60.0 < m.lufs < 0.0
