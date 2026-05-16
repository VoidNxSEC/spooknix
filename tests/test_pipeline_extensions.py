"""Testes para as extensões de PipelineConfig (Sprint 1).

Cobrem pre_emphasis, lufs_normalize fallback, denoise mockado e a ordem
fixa do pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio_pipeline import AudioPipeline, PipelineConfig


SAMPLE_RATE = 16_000


def _sine(freq: float = 440.0, duration: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_default_config_extends_have_safe_defaults():
    """Novos campos default-off — comportamento histórico preservado."""
    cfg = PipelineConfig()
    assert cfg.denoise is False
    assert cfg.lufs_normalize is False
    assert cfg.pre_emphasis is False
    assert cfg.target_lufs == -23.0
    assert cfg.pre_emphasis_coef == 0.97


def test_pre_emphasis_aplica_diferenca_de_primeira_ordem():
    cfg = PipelineConfig(
        normalize=False, high_pass=False, clip_ceiling=1.0,
        pre_emphasis=True, pre_emphasis_coef=0.97,
    )
    p = AudioPipeline(cfg)
    audio = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    out = p.process(audio)
    # y[0] = x[0]; y[i] = x[i] - 0.97*x[i-1]
    expected = np.array([1.0, 0.03, 0.03, 0.03], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_pre_emphasis_buffer_1_amostra_passthrough():
    cfg = PipelineConfig(
        normalize=False, high_pass=False, clip_ceiling=1.0,
        pre_emphasis=True,
    )
    p = AudioPipeline(cfg)
    out = p.process(np.array([0.7], dtype=np.float32))
    np.testing.assert_allclose(out, [0.7], atol=1e-6)


def test_lufs_normalize_substitui_rms_quando_ativo():
    """Quando lufs_normalize=True, o caminho RMS deve ser pulado."""
    cfg = PipelineConfig(
        normalize=True, lufs_normalize=True, high_pass=False, clip_ceiling=1.0,
    )
    p = AudioPipeline(cfg)
    audio = _sine(amp=0.3)
    out = p.process(audio)
    assert out.dtype == np.float32
    # Não explode mesmo sem pyloudnorm (cai em RMS-normalize via fallback)


def test_denoise_chama_module_lazy():
    """Quando denoise=True, _get_denoiser é invocado uma vez e cached."""
    cfg = PipelineConfig(denoise=True, normalize=False, high_pass=False, clip_ceiling=1.0)
    p = AudioPipeline(cfg)

    fake_denoiser = MagicMock()
    audio_in = _sine()
    fake_denoiser.denoise.return_value = audio_in  # pass-through
    p._denoiser = fake_denoiser

    p.process(audio_in.copy())
    p.process(audio_in.copy())
    assert fake_denoiser.denoise.call_count == 2


def test_pipeline_order_pre_emphasis_apos_denoise():
    """Ordem: high-pass → denoise → pre-emphasis → LUFS → clip.

    Verifica chamando denoise e pre-emphasis com um sinal conhecido.
    """
    cfg = PipelineConfig(
        normalize=False, high_pass=False, clip_ceiling=1.0,
        denoise=True, pre_emphasis=True, pre_emphasis_coef=0.5,
    )
    p = AudioPipeline(cfg)

    # denoise = retorna input vezes 2
    fake = MagicMock()
    fake.denoise.side_effect = lambda x: (x * 2.0).astype(np.float32)
    p._denoiser = fake

    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = p.process(audio)
    # após denoise: [0.2, 0.4, 0.6]
    # após pre-emphasis (coef=0.5): [0.2, 0.4 - 0.1 = 0.3, 0.6 - 0.2 = 0.4]
    np.testing.assert_allclose(out, [0.2, 0.3, 0.4], atol=1e-6)


def test_backward_compat_default_pipeline_normaliza_rms_para_0_1():
    """Defaults antigos preservados: normalize=True ainda alcança RMS≈0.1."""
    p = AudioPipeline()
    out = p.process(_sine(amp=0.5))
    rms = float(np.sqrt(np.mean(out ** 2)))
    assert abs(rms - 0.1) < 0.02
