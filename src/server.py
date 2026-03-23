# src/server.py
"""Servidor HTTP do Spooknix — Privacy-first STT Engine.

Endpoints:
  GET  /health      → status do servidor, modelo e GPU
  POST /transcribe  → transcrição de arquivo (multipart/form-data)

Variáveis de ambiente:
  MODEL_SIZE          : tiny|base|small|medium|large-v2|large-v3  (padrão: large-v3)
  DEVICE              : cuda | cpu                                 (padrão: auto)
  COMPUTE_TYPE        : int8|float16|int8_float16                  (padrão: derivado do modelo)
  ENABLE_DIARIZATION  : true | false                               (padrão: false)
  HF_TOKEN            : token HuggingFace (necessário p/ pyannote) (padrão: vazio)
  HOST                : endereço de bind                           (padrão: 0.0.0.0)
  PORT                : porta HTTP                                 (padrão: 8000)
"""

import json
import os
import tempfile
import time
from pathlib import Path

import torch
from aiohttp import web

from .transcriber import get_model, transcribe_file, _compute_type

# ── Configuração via variáveis de ambiente ─────────────────────────────────
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE") or None  # None → derivado automaticamente
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

    Retorna JSON: { text, segments, language, duration, model, diarized }
    """
    reader = await request.multipart()

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


# ── App factory ────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application(client_max_size=500 * 1024 * 1024)  # 500 MB
    app.router.add_get("/health", health)
    app.router.add_post("/transcribe", transcribe)
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
