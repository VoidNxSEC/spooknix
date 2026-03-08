# src/server.py
"""Servidor HTTP do Spooknix — Privacy-first STT Engine.

Endpoints:
  GET  /health      → status do servidor, modelo e GPU
  POST /transcribe  → transcrição de arquivo (multipart/form-data)

Variáveis de ambiente:
  MODEL_SIZE   : tiny | base | small | medium  (padrão: small)
  DEVICE       : cuda | cpu                    (padrão: auto)
  HOST         : endereço de bind              (padrão: 0.0.0.0)
  PORT         : porta HTTP                    (padrão: 8000)
"""

import json
import os
import tempfile
import time
from pathlib import Path

import torch
from aiohttp import web

from .transcriber import get_model, transcribe_file

# ── Configuração via variáveis de ambiente ─────────────────────────────────
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Estado global do modelo (singleton, lazy) ──────────────────────────────
_model = None
_server_start: float = time.time()


def get_loaded_model():
    """Carrega o modelo na primeira chamada e reutiliza nas seguintes."""
    global _model
    if _model is None:
        _model = get_model(MODEL_SIZE, DEVICE)
    return _model


# ── Handlers ───────────────────────────────────────────────────────────────

async def health(request: web.Request) -> web.Response:
    """GET /health — status do servidor."""
    cuda = torch.cuda.is_available()
    data = {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
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
      file      : arquivo de áudio/vídeo (obrigatório)
      language  : código do idioma, padrão 'pt' (opcional)

    Retorna JSON: { text, segments, language, duration }
    """
    reader = await request.multipart()

    audio_bytes = None
    filename = "audio"
    language = "pt"

    async for part in reader:
        if part.name == "file":
            filename = part.filename or "audio"
            audio_bytes = await part.read()
        elif part.name == "language":
            language = (await part.read()).decode()

    if audio_bytes is None:
        return web.json_response(
            {"error": "campo 'file' é obrigatório"}, status=400
        )

    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = get_loaded_model()
        result = transcribe_file(model, tmp_path, language=language)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return web.json_response(result)


# ── App factory ────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application(client_max_size=500 * 1024 * 1024)  # 500 MB
    app.router.add_get("/health", health)
    app.router.add_post("/transcribe", transcribe)
    return app


if __name__ == "__main__":
    print(f"🎙️  Spooknix STT Server")
    print(f"📦  Modelo: {MODEL_SIZE} | Dispositivo: {DEVICE}")
    print(f"🚀  Pré-carregando modelo…")
    get_loaded_model()
    print(f"✅  Pronto. Aguardando em http://{HOST}:{PORT}")
    web.run_app(create_app(), host=HOST, port=PORT, print=None)
