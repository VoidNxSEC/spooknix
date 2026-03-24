"""Testes para WebSocket streaming e métricas do servidor — sem GPU/mic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from aiohttp.test_utils import AioHTTPTestCase

from src.server import create_app, StreamSession
from src.audio_pipeline import AudioPipeline, PipelineConfig


# ── StreamSession ─────────────────────────────────────────────────────────────

class TestStreamSession:
    def _session(self, window_s: float = 3.0) -> StreamSession:
        pipeline = AudioPipeline(PipelineConfig(normalize=False, high_pass=False, clip_ceiling=1.0))
        return StreamSession(window_s=window_s, pipeline=pipeline)

    def test_push_adds_chunk(self):
        s = self._session()
        data = np.zeros(1600, dtype=np.float32)
        s.push(data.tobytes())
        assert len(s.chunks) == 1

    def test_should_flush_false_before_window(self):
        s = self._session(window_s=3.0)  # 3s * 16000 = 48000 samples
        data = np.zeros(1600, dtype=np.float32)
        s.push(data.tobytes())  # só 1600 amostras
        assert not s.should_flush()

    def test_should_flush_true_at_window(self):
        s = self._session(window_s=0.1)  # 1600 samples
        data = np.zeros(1600, dtype=np.float32)
        s.push(data.tobytes())
        assert s.should_flush()

    def test_push_decodes_float32_little_endian(self):
        s = self._session()
        arr = np.array([0.5, -0.5, 0.25], dtype="<f4")
        s.push(arr.tobytes())
        np.testing.assert_allclose(s.chunks[0], arr, atol=1e-6)

    def test_flush_clears_buffer(self):
        s = self._session(window_s=0.01)
        data = np.zeros(1600, dtype=np.float32)
        s.push(data.tobytes())

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock(duration=0.1))

        with patch("src.server.transcribe_stream", return_value=iter([])):
            s.flush(mock_model, "pt")

        assert s.chunks == []

    def test_flush_empty_buffer_returns_empty(self):
        s = self._session()
        mock_model = MagicMock()
        result = s.flush(mock_model, "pt")
        assert result == []


# ── HTTP endpoints (sem GPU) ──────────────────────────────────────────────────

class TestHealthEndpoint(AioHTTPTestCase):
    async def get_application(self):
        return create_app()

    async def test_health_returns_ok(self):
        resp = await self.client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert "device" in data

    async def test_health_has_uptime(self):
        resp = await self.client.get("/health")
        data = await resp.json()
        assert "uptime_s" in data
        assert data["uptime_s"] >= 0


class TestMetricsEndpoint(AioHTTPTestCase):
    async def get_application(self):
        return create_app()

    async def test_metrics_returns_200(self):
        resp = await self.client.get("/metrics")
        assert resp.status == 200

    async def test_metrics_content_type(self):
        resp = await self.client.get("/metrics")
        assert "text/plain" in resp.headers.get("Content-Type", "")

    async def test_metrics_contains_prometheus_names(self):
        resp = await self.client.get("/metrics")
        body = await resp.text()
        assert "spooknix_chunks_total" in body
        assert "spooknix_latency_ms" in body
        assert "spooknix_confidence" in body


class TestTranscribeEndpoint(AioHTTPTestCase):
    async def get_application(self):
        return create_app()

    async def test_transcribe_missing_file_returns_400(self):
        data = {"language": "pt"}
        resp = await self.client.post("/transcribe", data=data)
        assert resp.status == 400
        body = await resp.json()
        assert "error" in body


# ── WebSocket /ws/stream — handshake ─────────────────────────────────────────

class TestStreamWebSocket(AioHTTPTestCase):
    async def get_application(self):
        return create_app()

    async def test_ws_session_start_message(self):
        """Ao conectar, servidor envia session_start."""
        mock_model = MagicMock()
        with patch("src.server.get_loaded_model", return_value=mock_model):
            async with self.client.ws_connect("/ws/stream?language=pt&window=2.0") as ws:
                msg = await ws.receive_json()
                assert msg["type"] == "session_start"
                assert msg["window_s"] == 2.0
                await ws.close()

    async def test_ws_ping_pong(self):
        """cmd ping → resposta pong."""
        mock_model = MagicMock()
        with patch("src.server.get_loaded_model", return_value=mock_model):
            async with self.client.ws_connect("/ws/stream") as ws:
                await ws.receive_json()  # session_start
                await ws.send_json({"cmd": "ping"})
                msg = await ws.receive_json()
                assert msg["type"] == "pong"
                await ws.close()

    async def test_ws_stop_cmd_closes(self):
        """cmd stop encerra a conexão graciosamente."""
        mock_model = MagicMock()
        with patch("src.server.get_loaded_model", return_value=mock_model):
            async with self.client.ws_connect("/ws/stream") as ws:
                await ws.receive_json()  # session_start
                await ws.send_json({"cmd": "stop"})
                # Servidor deve fechar — receber CLOSE frame
                from aiohttp import WSMsgType
                msg = await ws.receive()
                assert msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED)

    async def test_ws_flush_empty_buffer_no_response(self):
        """flush com buffer vazio não deve enviar final."""
        mock_model = MagicMock()
        with patch("src.server.get_loaded_model", return_value=mock_model):
            async with self.client.ws_connect("/ws/stream") as ws:
                await ws.receive_json()  # session_start
                await ws.send_json({"cmd": "flush"})
                # Nenhuma resposta extra (buffer vazio)
                await ws.send_json({"cmd": "ping"})
                msg = await ws.receive_json()
                assert msg["type"] == "pong"
                await ws.close()

    async def test_ws_binary_chunks_accepted(self):
        """Chunks binários são aceitos sem erro."""
        mock_model = MagicMock()
        with patch("src.server.get_loaded_model", return_value=mock_model):
            async with self.client.ws_connect("/ws/stream?window=100.0") as ws:
                await ws.receive_json()  # session_start
                silence = np.zeros(1600, dtype=np.float32).tobytes()
                await ws.send_bytes(silence)
                # Sem flush ainda (window=100s) — servidor aceita e não responde
                await ws.send_json({"cmd": "ping"})
                msg = await ws.receive_json()
                assert msg["type"] == "pong"
                await ws.close()
