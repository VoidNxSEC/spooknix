# src/recorder.py
"""Gravação de microfone para o Spooknix.

Captura áudio via sounddevice (PortAudio), para automaticamente após silêncio
detectado por RMS, e salva um WAV 16kHz mono em arquivo temporário.

O caller é responsável por deletar o arquivo temporário após o uso.

Observabilidade:
  - Logs em `src.recorder` (INFO no fluxo normal, DEBUG por chunk, WARNING em
    overflow do PortAudio).
  - Ative com `SPOOKNIX_LOG_LEVEL=DEBUG` ou via `--verbose` na CLI.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import wave
from typing import Callable

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # Whisper espera 16 kHz
BLOCKSIZE = 1_600     # 100ms por chunk
_STOP_WINDOW_S = 3.0  # segundos de áudio enviados para o stop_check_fn


class RecordingError(RuntimeError):
    """Erro durante a gravação do microfone."""


def _describe_status(status) -> str:
    """Stringifica `CallbackFlags` do sounddevice para um log legível."""
    if status is None:
        return ""
    s = str(status).strip()
    return s or repr(status)


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
    chunks_lock = threading.Lock()
    stop_event = threading.Event()
    overflow_count = [0]  # mutable container — alterado da thread do PortAudio

    silent_chunks_needed = int(silence_duration * samplerate / BLOCKSIZE)
    max_chunks = int(max_duration * samplerate / BLOCKSIZE)
    window_chunks = int(_STOP_WINDOW_S * samplerate / BLOCKSIZE)
    silent_count = 0
    stop_reason = ["unknown"]
    started_at = time.monotonic()

    log.info(
        "recording.start sr=%d blocksize=%d threshold=%.4f silence_s=%.2f max_s=%.1f stop_word=%s",
        samplerate, BLOCKSIZE, silence_threshold, silence_duration, max_duration,
        "yes" if stop_check_fn else "no",
    )

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal silent_count
        if status:
            overflow_count[0] += 1
            log.warning(
                "audio.status flags=%s overflow_count=%d (chunk perdido pelo PortAudio)",
                _describe_status(status), overflow_count[0],
            )

        if stop_event.is_set():
            raise sd.CallbackStop()

        chunk = indata[:, 0].copy()
        with chunks_lock:
            chunks.append(chunk)
            n = len(chunks)

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        is_speech = rms >= silence_threshold

        if is_speech:
            silent_count = 0
        else:
            silent_count += 1

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "chunk #%d rms=%.4f speech=%s silent=%d/%d",
                n, rms, is_speech, silent_count, silent_chunks_needed,
            )

        if n > silent_chunks_needed and silent_count >= silent_chunks_needed:
            stop_reason[0] = "silence"
            log.info(
                "recording.stop reason=silence chunks=%d silent_run=%d",
                n, silent_count,
            )
            stop_event.set()
            raise sd.CallbackStop()

        if n >= max_chunks:
            stop_reason[0] = "max_duration"
            log.info("recording.stop reason=max_duration chunks=%d", n)
            stop_event.set()
            raise sd.CallbackStop()

    def _stop_checker() -> None:
        """Thread que checa periodicamente o stop_check_fn."""
        log.debug("stop_checker.start interval=%.1fs", stop_check_interval)
        while not stop_event.wait(timeout=stop_check_interval):
            with chunks_lock:
                if not chunks:
                    continue
                window = list(chunks[-window_chunks:])  # cópia rasa sob o lock
            wav_bytes = _chunks_to_wav_bytes(window, samplerate)
            try:
                hit = stop_check_fn(wav_bytes)
            except Exception as exc:
                log.warning("stop_checker.error %s", exc)
                continue
            if hit:
                stop_reason[0] = "stop_word"
                log.info("recording.stop reason=stop_word")
                stop_event.set()
                return
        log.debug("stop_checker.exit")

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
        log.error("recording.portaudio_error %s", exc)
        raise RecordingError(f"Erro no dispositivo de áudio: {exc}") from exc
    finally:
        stop_event.set()  # garante que o checker_thread encerra

    elapsed = time.monotonic() - started_at
    with chunks_lock:
        n = len(chunks)

    if not chunks:
        log.error("recording.empty elapsed=%.2fs overflow=%d", elapsed, overflow_count[0])
        raise RecordingError("Nenhum áudio foi capturado.")

    log.info(
        "recording.done reason=%s chunks=%d duration=%.2fs overflow=%d",
        stop_reason[0], n, elapsed, overflow_count[0],
    )

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
    overflow_count = [0]
    started_at = time.monotonic()

    log.info("recording.fixed.start duration=%.2fs samplerate=%d", duration, samplerate)

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            overflow_count[0] += 1
            log.warning("audio.status flags=%s overflow_count=%d",
                        _describe_status(status), overflow_count[0])
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
        log.error("recording.fixed.portaudio_error %s", exc)
        raise RecordingError(f"Erro no dispositivo de áudio: {exc}") from exc

    if not chunks:
        log.error("recording.fixed.empty elapsed=%.2fs", time.monotonic() - started_at)
        raise RecordingError("Nenhum áudio foi capturado.")

    log.info(
        "recording.fixed.done chunks=%d duration=%.2fs overflow=%d",
        len(chunks), time.monotonic() - started_at, overflow_count[0],
    )

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
