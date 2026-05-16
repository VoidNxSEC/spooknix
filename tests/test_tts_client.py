"""Testes para src/tts_client.py — sem servidor TTS real, tudo mockado."""

from __future__ import annotations

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.tts_client import LocalTTSClient


def _wav_bytes(samples: np.ndarray, samplerate: int = 24_000) -> bytes:
    """Helper: gera bytes de um WAV PCM int16 mono."""
    buf = io.BytesIO()
    int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def test_init_usa_envs_em_ordem(monkeypatch):
    monkeypatch.delenv("TTS_BASE_URL", raising=False)
    monkeypatch.setenv("F5_TTS_URL", "http://f5:9000")
    c = LocalTTSClient()
    assert c.base_url == "http://f5:9000"


def test_init_explicit_overrides_envs(monkeypatch):
    monkeypatch.setenv("TTS_BASE_URL", "http://env:8001")
    c = LocalTTSClient(base_url="http://explicit:7777")
    assert c.base_url == "http://explicit:7777"


def test_init_strip_trailing_slash():
    c = LocalTTSClient(base_url="http://x:8001/")
    assert c.base_url == "http://x:8001"


def test_init_timeout_default():
    c = LocalTTSClient()
    assert c.timeout_s == 30.0


def test_init_timeout_via_env(monkeypatch):
    monkeypatch.setenv("TTS_TIMEOUT_S", "5")
    c = LocalTTSClient()
    assert c.timeout_s == 5.0


def test_decode_wav_bytes_vazios():
    c = LocalTTSClient()
    audio, sr = c.decode_wav(b"")
    assert audio.size == 0
    assert sr == 24000


def test_decode_wav_bytes_invalidos():
    c = LocalTTSClient()
    audio, sr = c.decode_wav(b"not a wav")
    assert audio.size == 0
    assert sr == 24000


def test_decode_wav_roundtrip():
    c = LocalTTSClient()
    sine = (np.sin(np.linspace(0, 2 * np.pi, 24_000, dtype=np.float32)) * 0.5).astype(np.float32)
    payload = _wav_bytes(sine, samplerate=24_000)
    audio, sr = c.decode_wav(payload)
    assert sr == 24_000
    assert audio.dtype == np.float32
    assert audio.size == 24_000
    # Sinal preservado dentro de tolerância do int16 round-trip
    np.testing.assert_allclose(audio, sine, atol=1e-3)


@pytest.mark.asyncio
async def test_synthesize_200_retorna_bytes():
    c = LocalTTSClient(base_url="http://tts:8001")

    resp_mock = MagicMock()
    resp_mock.status = 200
    resp_mock.read = AsyncMock(return_value=b"WAVdata")

    cm_resp = MagicMock()
    cm_resp.__aenter__ = AsyncMock(return_value=resp_mock)
    cm_resp.__aexit__ = AsyncMock(return_value=False)

    session_mock = MagicMock()
    session_mock.post = MagicMock(return_value=cm_resp)
    session_mock.__aenter__ = AsyncMock(return_value=session_mock)
    session_mock.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=session_mock):
        out = await c.synthesize("hello", voice="alloy")

    assert out == b"WAVdata"


@pytest.mark.asyncio
async def test_synthesize_status_ruim_loga_e_retorna_vazio(caplog):
    import logging
    c = LocalTTSClient(base_url="http://tts:8001")

    resp_mock = MagicMock()
    resp_mock.status = 500
    resp_mock.text = AsyncMock(return_value="server boom")

    cm_resp = MagicMock()
    cm_resp.__aenter__ = AsyncMock(return_value=resp_mock)
    cm_resp.__aexit__ = AsyncMock(return_value=False)

    session_mock = MagicMock()
    session_mock.post = MagicMock(return_value=cm_resp)
    session_mock.__aenter__ = AsyncMock(return_value=session_mock)
    session_mock.__aexit__ = AsyncMock(return_value=False)

    with caplog.at_level(logging.WARNING, logger="src.tts_client"):
        with patch("aiohttp.ClientSession", return_value=session_mock):
            out = await c.synthesize("hello")

    assert out == b""
    assert any("tts.bad_status" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_synthesize_conexao_falha_retorna_vazio(caplog):
    import aiohttp
    import logging

    c = LocalTTSClient(base_url="http://tts-offline:8001")

    session_mock = MagicMock()
    session_mock.post = MagicMock(side_effect=aiohttp.ClientConnectorError(
        connection_key=MagicMock(), os_error=OSError("refused")
    ))
    session_mock.__aenter__ = AsyncMock(return_value=session_mock)
    session_mock.__aexit__ = AsyncMock(return_value=False)

    with caplog.at_level(logging.ERROR, logger="src.tts_client"):
        with patch("aiohttp.ClientSession", return_value=session_mock):
            out = await c.synthesize("hello")

    assert out == b""
    assert any("tts.connection_error" in r.message for r in caplog.records)
