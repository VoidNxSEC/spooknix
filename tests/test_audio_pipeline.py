"""Testes para src/audio_pipeline.py — sem GPU, sem mic."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio_pipeline import AudioPipeline, PipelineConfig


# ── Helpers ─────────────────────────────────────────────────────────────────

def _sine(freq: float = 440.0, duration: float = 0.1, sr: int = 16_000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(n: int = 1600) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ── PipelineConfig defaults ──────────────────────────────────────────────────

def test_default_config():
    cfg = PipelineConfig()
    assert cfg.normalize is True
    assert cfg.high_pass is True
    assert cfg.high_pass_cutoff_hz == 80.0
    assert cfg.clip_ceiling == 0.99
    assert cfg.target_rms == 0.1


# ── AudioPipeline construction ───────────────────────────────────────────────

def test_highpass_coefficients_computed_on_init():
    p = AudioPipeline()
    assert p._b is not None
    assert p._a is not None


def test_no_highpass_skips_design():
    p = AudioPipeline(PipelineConfig(high_pass=False))
    assert p._b is None
    assert p._a is None


# ── process() ────────────────────────────────────────────────────────────────

def test_process_returns_float32():
    p = AudioPipeline()
    out = p.process(_sine())
    assert out.dtype == np.float32


def test_process_normalizes_rms():
    p = AudioPipeline(PipelineConfig(normalize=True, target_rms=0.1, high_pass=False, clip_ceiling=1.0))
    chunk = (_sine() * 0.5)  # RMS ~0.35
    out = p.process(chunk)
    rms = float(np.sqrt(np.mean(out ** 2)))
    assert abs(rms - 0.1) < 0.02


def test_process_clips_ceiling():
    cfg = PipelineConfig(normalize=False, high_pass=False, clip_ceiling=0.5)
    p = AudioPipeline(cfg)
    chunk = np.ones(1600, dtype=np.float32)  # all 1.0
    out = p.process(chunk)
    assert float(out.max()) <= 0.5 + 1e-6


def test_process_silence_no_nan():
    """Silêncio (RMS ≈ 0) não deve produzir NaN ou Inf."""
    p = AudioPipeline()
    out = p.process(_silence())
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


def test_process_does_not_mutate_input():
    p = AudioPipeline(PipelineConfig(high_pass=False))
    original = _sine().copy()
    chunk = original.copy()
    p.process(chunk)
    np.testing.assert_array_equal(chunk, original)


# ── process_buffer() ─────────────────────────────────────────────────────────

def test_process_buffer_empty_returns_zeros():
    p = AudioPipeline()
    out = p.process_buffer([])
    assert out.size == 0
    assert out.dtype == np.float32


def test_process_buffer_concatenates_correctly():
    p = AudioPipeline(PipelineConfig(normalize=False, high_pass=False, clip_ceiling=1.0))
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.3, 0.4], dtype=np.float32)
    out = p.process_buffer([a, b])
    assert out.shape == (4,)


def test_process_buffer_same_length_as_concatenated():
    p = AudioPipeline()
    chunks = [_sine() for _ in range(5)]
    out = p.process_buffer(chunks)
    expected_len = sum(len(c) for c in chunks)
    assert len(out) == expected_len


def test_process_buffer_returns_float32():
    p = AudioPipeline()
    out = p.process_buffer([_sine(), _sine()])
    assert out.dtype == np.float32


# ── Disable stages ────────────────────────────────────────────────────────────

def test_all_stages_disabled_passthrough():
    cfg = PipelineConfig(normalize=False, high_pass=False, clip_ceiling=1.0)
    p = AudioPipeline(cfg)
    chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = p.process(chunk)
    np.testing.assert_allclose(out, chunk, atol=1e-6)


def test_high_pass_attenuates_low_freq():
    """O filtro high-pass deve atenuar componentes abaixo de 80 Hz."""
    p = AudioPipeline(PipelineConfig(normalize=False, high_pass=True, clip_ceiling=1.0))
    low = _sine(freq=20.0, duration=0.5)   # 20 Hz — abaixo do corte
    high = _sine(freq=1000.0, duration=0.5)  # 1 kHz — acima do corte

    out_low = p.process(low)
    out_high = p.process(high)

    rms_low = float(np.sqrt(np.mean(out_low ** 2)))
    rms_high = float(np.sqrt(np.mean(out_high ** 2)))
    # Componente de baixa frequência deve ser significativamente atenuado
    assert rms_low < rms_high * 0.1
