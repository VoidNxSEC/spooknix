# src/recorder.py
"""Gravação de microfone para o Spooknix.

Captura áudio via sounddevice (PortAudio), para automaticamente após silêncio
detectado por RMS, e salva um WAV 16kHz mono em arquivo temporário.

O caller é responsável por deletar o arquivo temporário após o uso.
"""

from __future__ import annotations

import tempfile
import threading
import wave
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # Whisper espera 16 kHz
BLOCKSIZE = 1_600     # 100ms por chunk
_STOP_WINDOW_S = 3.0  # segundos de áudio enviados para o stop_check_fn


class RecordingError(RuntimeError):
    """Erro durante a gravação do microfone."""


def record_until_silence(
    silence_duration: float = 2.0,
    silence_threshold: float = 0.01,
    max_duration: float = 360.0,
    samplerate: int = SAMPLE_RATE,
    stop_check_fn: Callable[[bytes], bool] | None = None,
    stop_check_interval: float = 2.0,
) -> str:
    """Grava do microfone até detectar silêncio.

    Args:
        silence_duration: Segundos de silêncio contínuo para parar a gravação.
        silence_threshold: Nível RMS abaixo do qual o áudio é considerado silêncio.
        max_duration: Duração máxima absoluta em segundos.
        samplerate: Taxa de amostragem (padrão 16000 Hz para Whisper).
        stop_check_fn: Função opcional chamada a cada `stop_check_interval` segundos
            com os últimos _STOP_WINDOW_S segundos de áudio como bytes WAV int16.
            Retorna True para parar a gravação imediatamente (ex: keyword "stop").
        stop_check_interval: Intervalo em segundos entre chamadas de stop_check_fn.

    Returns:
        Caminho do arquivo WAV temporário (int16, mono, 16kHz).

    Raises:
        RecordingError: Se nenhum áudio for capturado ou se ocorrer erro no dispositivo.
    """
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    # Quantidade de chunks consecutivos de silêncio para parar
    silent_chunks_needed = int(silence_duration * samplerate / BLOCKSIZE)
    max_chunks = int(max_duration * samplerate / BLOCKSIZE)
    window_chunks = int(_STOP_WINDOW_S * samplerate / BLOCKSIZE)
    silent_count = 0

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal silent_count
        if stop_event.is_set():
            raise sd.CallbackStop()

        chunk = indata[:, 0].copy()  # mono
        chunks.append(chunk)

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < silence_threshold:
            silent_count += 1
        else:
            silent_count = 0

        # Parar por silêncio (mas só depois de ter gravado algo)
        if len(chunks) > silent_chunks_needed and silent_count >= silent_chunks_needed:
            stop_event.set()
            raise sd.CallbackStop()

        # Parar por duração máxima
        if len(chunks) >= max_chunks:
            stop_event.set()
            raise sd.CallbackStop()

    def _stop_checker() -> None:
        """Thread que checa periodicamente o stop_check_fn."""
        while not stop_event.wait(timeout=stop_check_interval):
            if not chunks:
                continue
            window = chunks[-window_chunks:]
            wav_bytes = _chunks_to_wav_bytes(window, samplerate)
            try:
                if stop_check_fn(wav_bytes):
                    stop_event.set()
                    return
            except Exception:
                pass  # falha silenciosa — não interrompe a gravação

    checker_thread: threading.Thread | None = None
    if stop_check_fn is not None:
        checker_thread = threading.Thread(target=_stop_checker, daemon=True)
        checker_thread.start()

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=callback,
        ):
            stop_event.wait(timeout=max_duration + 5)
    except sd.PortAudioError as exc:
        raise RecordingError(f"Erro no dispositivo de áudio: {exc}") from exc
    finally:
        stop_event.set()  # garante que o checker_thread encerra

    if not chunks:
        raise RecordingError("Nenhum áudio foi capturado.")

    return _save_wav(chunks, samplerate)


def record_fixed_duration(duration: float, samplerate: int = SAMPLE_RATE) -> str:
    """Grava por um período fixo de tempo.

    Args:
        duration: Duração em segundos.
        samplerate: Taxa de amostragem.

    Returns:
        Caminho do arquivo WAV temporário (int16, mono, 16kHz).

    Raises:
        RecordingError: Se nenhum áudio for capturado ou se ocorrer erro no dispositivo.
    """
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()
    max_chunks = int(duration * samplerate / BLOCKSIZE)

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if stop_event.is_set():
            raise sd.CallbackStop()
        chunks.append(indata[:, 0].copy())
        if len(chunks) >= max_chunks:
            stop_event.set()
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=callback,
        ):
            stop_event.wait(timeout=duration + 5)
    except sd.PortAudioError as exc:
        raise RecordingError(f"Erro no dispositivo de áudio: {exc}") from exc

    if not chunks:
        raise RecordingError("Nenhum áudio foi capturado.")

    return _save_wav(chunks, samplerate)


def _chunks_to_wav_bytes(chunks: list[np.ndarray], samplerate: int) -> bytes:
    """Converte chunks float32 para bytes WAV int16 (in-memory)."""
    import io
    audio = np.concatenate(chunks)
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _save_wav(chunks: list[np.ndarray], samplerate: int) -> str:
    """Salva chunks float32 como WAV int16 mono em arquivo temporário."""
    audio = np.concatenate(chunks)
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

    return tmp.name
