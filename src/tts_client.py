"""
Cliente assíncrono 100% local para o Worker TTS (ex: XTTS-v2, F5-TTS, Piper).
Não possui dependências de nuvem ou pacotes de terceiros como OpenAI.
"""

import io
import os
import wave

import aiohttp
import numpy as np


class LocalTTSClient:
    """Cliente para interagir diretamente com o container TTS local via REST."""

    def __init__(self, base_url: str | None = None):
        """
        Inicializa o cliente local.
        Por padrão, procura o serviço na porta 8001 da própria máquina.
        """
        resolved_base_url = (
            base_url
            or os.getenv("TTS_BASE_URL")
            or os.getenv("XTTS_BASE_URL")
            or os.getenv("CHATTERBOX_BASE_URL")
            or os.getenv("F5_TTS_URL")
            or "http://localhost:8001"
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.api_path = os.getenv("TTS_API_PATH", "/tts")
        self.default_voice = os.getenv("TTS_VOICE", "default_voice")
        self.default_language = os.getenv("TTS_LANGUAGE", "en")

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """
        Envia a string de texto para o container local de TTS.
        O endpoint exato (/tts, /generate, /api/tts) depende da imagem docker usada.
        """
        # Payload adaptável para as imagens TTS open-source mais comuns (XTTS/Coqui/Piper)
        payload = {
            "text": text,
            "voice": voice or self.default_voice,
            "language": self.default_language,
        }

        try:
            async with aiohttp.ClientSession() as session:
                # O endpoint comum em wrappers locais (ajuste conforme o seu container suba)
                endpoint = f"{self.base_url}{self.api_path}"

                async with session.post(endpoint, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.read() # Espera os bytes do WAV (PCM)
                    else:
                        err = await resp.text()
                        print(f"\n[TTS Local Error] Erro {resp.status} do container TTS: {err}")
                        return b""
        except Exception as e:
            print(f"\n[TTS Local Error] Container offline ou erro de conexão em {self.base_url}: {e}")
            return b""

    def decode_wav(self, wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Converte WAV bytes brutos em ndarray float32 e extrai a sample rate."""
        if not wav_bytes:
            return np.array([], dtype=np.float32), 24000

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                samplerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                # Converte int16 para float32 (formato esperado pelo sounddevice / PipeWire)
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                return audio_float32, samplerate
        except Exception as e:
            print(f"[TTS Local Error] Falha ao fazer parse do WAV: {e}")
            return np.array([], dtype=np.float32), 24000
