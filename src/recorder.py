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

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # Whisper espera 16 kHz
BLOCKSIZE = 1_600     # 100ms por chunk


class RecordingError(RuntimeError):
    """Erro durante a gravação do microfone."""


def record_until_silence(
    silence_duration: float = 2.0,
    silence_threshold: float = 0.01,
    max_duration: float = 120.0,
    samplerate: int = SAMPLE_RATE,
) -> str:
    """Grava do microfone até detectar silêncio.

    Args:
        silence_duration: Segundos de silêncio contínuo para parar a gravação.
        silence_threshold: Nível RMS abaixo do qual o áudio é considerado silêncio.
        max_duration: Duração máxima absoluta em segundos.
        samplerate: Taxa de amostragem (padrão 16000 Hz para Whisper).

    Returns:
        Caminho do arquivo WAV temporário (int16, mono, 16kHz).

    Raises:
        RecordingError: Se nenhum áudio for capturado ou se ocorrer erro no dispositivo.
    """
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()
    error_holder: list[Exception] = []

    # Quantidade de chunks consecutivos de silêncio para parar
    silent_chunks_needed = int(silence_duration * samplerate / BLOCKSIZE)
    max_chunks = int(max_duration * samplerate / BLOCKSIZE)
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


def _save_wav(chunks: list[np.ndarray], samplerate: int) -> str:
    """Salva chunks float32 como WAV int16 mono em arquivo temporário."""
    audio = np.concatenate(chunks)
    # float32 → int16
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())

    return tmp.name
