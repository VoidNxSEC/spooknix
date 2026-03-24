# src/server.py
"""Servidor HTTP/WebSocket do Spooknix — Privacy-first STT Engine.

Endpoints:
  GET  /health       → status do servidor, modelo e GPU
  POST /transcribe   → transcrição de arquivo (multipart/form-data)
  GET  /ws/stream    → WebSocket streaming STT
  GET  /metrics      → métricas Prometheus

Variáveis de ambiente:
  MODEL_SIZE          : tiny|base|small|medium|large-v2|large-v3  (padrão: large-v3)
  DEVICE              : cuda | cpu                                 (padrão: auto)
  COMPUTE_TYPE        : int8|float16|int8_float16                  (padrão: derivado do modelo)
  ENABLE_DIARIZATION  : true | false                               (padrão: false)
  HF_TOKEN            : token HuggingFace (necessário p/ pyannote) (padrão: vazio)
  HOST                : endereço de bind                           (padrão: 0.0.0.0)
  PORT                : porta HTTP                                 (padrão: 8000)
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import web
import aiohttp

from .transcriber import get_model, transcribe_file, transcribe_stream, _compute_type
from .audio_pipeline import AudioPipeline, PipelineConfig
from . import metrics as m

# ── Configuração via variáveis de ambiente ─────────────────────────────────
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") or None
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Estado global do modelo (singleton, lazy) ──────────────────────────────
_model = None
_model_name: str = MODEL_SIZE
_server_start: float = time.time()


def get_loaded_model(size: str | None = None):
    """Carrega o modelo na primeira chamada; recarrega se `size` diferir do atual."""
    global _model, _model_name
    target = size or MODEL_SIZE
    if _model is None or target != _model_name:
        _model = get_model(target, DEVICE, COMPUTE_TYPE)
        _model_name = target
    return _model


# ── Handlers ───────────────────────────────────────────────────────────────

async def health(request: web.Request) -> web.Response:
    """GET /health — status do servidor."""
    cuda = torch.cuda.is_available()
    data = {
        "status": "ok",
        "model": _model_name,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE or _compute_type(_model_name, DEVICE),
        "diarization": ENABLE_DIARIZATION,
        "uptime_s": round(time.time() - _server_start, 1),
        "cuda": cuda,
    }
    if cuda:
        data["gpu"] = torch.cuda.get_device_name(0)
        data["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    return web.json_response(data)


async def transcribe(request: web.Request) -> web.Response:
    """POST /transcribe — transcrição de arquivo de áudio/vídeo.

    Parâmetros (multipart/form-data):
      file        : arquivo de áudio/vídeo (obrigatório)
      language    : código do idioma, padrão 'pt' (opcional)
      model_size  : override do modelo por request (opcional)
      diarize     : 'true' para diarização de speakers (opcional)
    """
    try:
        reader = await request.multipart()
    except (AssertionError, Exception):
        return web.json_response(
            {"error": "Content-Type multipart/form-data obrigatório"}, status=400
        )

    audio_bytes = None
    filename = "audio"
    language = "pt"
    model_size_override: str | None = None
    diarize = False

    async for part in reader:
        if part.name == "file":
            filename = part.filename or "audio"
            audio_bytes = await part.read()
        elif part.name == "language":
            language = (await part.read()).decode()
        elif part.name == "model_size":
            model_size_override = (await part.read()).decode().strip() or None
        elif part.name == "diarize":
            diarize = (await part.read()).decode().strip().lower() == "true"

    if audio_bytes is None:
        return web.json_response(
            {"error": "campo 'file' é obrigatório"}, status=400
        )

    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = get_loaded_model(model_size_override)
        result = transcribe_file(model, tmp_path, language=language)
        result["model"] = _model_name
        result["diarized"] = False

        if diarize:
            if not ENABLE_DIARIZATION:
                return web.json_response(
                    {"error": "Diarização não habilitada no servidor (ENABLE_DIARIZATION=false)"},
                    status=400,
                )
            try:
                from .diarizer import diarize as run_diarize, assign_speakers
                diarization = run_diarize(tmp_path)
                result["segments"] = assign_speakers(result["segments"], diarization)
                result["diarized"] = True
            except ImportError:
                return web.json_response(
                    {"error": "pyannote-audio não instalado — instale com: poetry install --with diarization"},
                    status=500,
                )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return web.json_response(result)


# ── WebSocket streaming ────────────────────────────────────────────────────

class StreamSession:
    """Acumula chunks float32 do cliente e faz flush para transcrição."""

    def __init__(self, window_s: float, pipeline: AudioPipeline, sample_rate: int = 16_000):
        self.chunks: list[np.ndarray] = []
        self.window_samples: int = int(window_s * sample_rate)
        self.pipeline = pipeline

    def push(self, raw_bytes: bytes) -> None:
        """Converte bytes float32 little-endian para ndarray e acumula."""
        arr = np.frombuffer(raw_bytes, dtype="<f4").copy()
        self.chunks.append(arr)

    def should_flush(self) -> bool:
        total = sum(len(c) for c in self.chunks)
        return total >= self.window_samples

    def flush(self, model, language: str) -> list[dict]:
        """Processa buffer acumulado e retorna lista de segmentos."""
        if not self.chunks:
            return []
        audio = self.pipeline.process_buffer(self.chunks)
        self.chunks = []
        return list(transcribe_stream(model, audio, language=language))


async def stream_ws(request: web.Request) -> web.WebSocketResponse:
    """GET /ws/stream — WebSocket streaming STT.

    Query params:
      language : código do idioma (padrão: pt)
      window   : janela de flush em segundos (padrão: 3.0)

    Protocolo cliente→servidor:
      frames binários : float32 little-endian, BLOCKSIZE=1600 amostras (6400 bytes)
      frames text JSON: {"cmd": "stop"} | {"cmd": "flush"} | {"cmd": "ping"}

    Protocolo servidor→cliente:
      {"type": "session_start", "model": ..., "device": ..., "window_s": ...}
      {"type": "partial",  "text": " palavra", "confidence": 0.97, "t": 1.34}
      {"type": "final",    "text": "...", "segments": [...], "latency_ms": 312.4}
      {"type": "pong"}
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    language = request.rel_url.query.get("language", "pt")
    window_s = float(request.rel_url.query.get("window", "3.0"))

    model = get_loaded_model()
    session = StreamSession(window_s=window_s, pipeline=AudioPipeline())

    m.sessions_total.inc()
    m.active_sessions.inc()

    await ws.send_json({
        "type": "session_start",
        "model": _model_name,
        "device": DEVICE,
        "window_s": window_s,
    })

    loop = asyncio.get_running_loop()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                session.push(msg.data)
                m.chunks_total.inc({"type": "received"})

                if session.should_flush():
                    await _do_flush(ws, session, model, language, loop)

            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                action = cmd.get("cmd")
                if action == "stop":
                    break
                elif action == "flush":
                    await _do_flush(ws, session, model, language, loop)
                elif action == "ping":
                    await ws.send_json({"type": "pong"})

            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        m.active_sessions.dec()

    return ws


async def _do_flush(
    ws: web.WebSocketResponse,
    session: StreamSession,
    model,
    language: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Executa flush do buffer em executor e envia partials + final ao cliente."""
    t0 = time.perf_counter()
    segments = await loop.run_in_executor(
        None, lambda: session.flush(model, language)
    )
    latency = (time.perf_counter() - t0) * 1000
    m.latency_ms.observe(latency)

    if not segments:
        return

    # Envia partial por palavra
    for seg in segments:
        for word in seg.get("words", []):
            if ws.closed:
                return
            await ws.send_json({
                "type": "partial",
                "text": word["word"],
                "confidence": word.get("probability", 0.0),
                "t": word["start"],
            })
        word_count = len(seg.get("words", []))
        if word_count:
            m.words_total.inc(n=word_count)

    # Atualiza confiança média do último segmento
    last_conf = segments[-1].get("avg_confidence", 0.0)
    m.confidence.set(last_conf)

    # Envia final da janela
    full_text = " ".join(s["text"] for s in segments if s["text"])
    await ws.send_json({
        "type": "final",
        "text": full_text,
        "segments": segments,
        "latency_ms": round(latency, 1),
    })

    m.chunks_total.inc({"type": "flushed"})


async def metrics_handler(request: web.Request) -> web.Response:
    """GET /metrics — métricas no formato Prometheus text."""
    body = m.render_prometheus()
    return web.Response(
        body=body,
        headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
    )


# ── App factory ────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application(client_max_size=500 * 1024 * 1024)  # 500 MB
    app.router.add_get("/health", health)
    app.router.add_post("/transcribe", transcribe)
    app.router.add_get("/ws/stream", stream_ws)
    app.router.add_get("/metrics", metrics_handler)
    return app


if __name__ == "__main__":
    ct = COMPUTE_TYPE or _compute_type(MODEL_SIZE, DEVICE)
    print(f"🎙️  Spooknix STT Server")
    print(f"📦  Modelo: {MODEL_SIZE} | Dispositivo: {DEVICE} | compute_type: {ct}")
    print(f"🔊  Diarização: {'habilitada' if ENABLE_DIARIZATION else 'desabilitada'}")
    print(f"🚀  Pré-carregando modelo…")
    get_loaded_model()
    print(f"✅  Pronto. Aguardando em http://{HOST}:{PORT}")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)
