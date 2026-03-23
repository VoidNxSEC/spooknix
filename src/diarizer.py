# src/diarizer.py
"""Diarização de speakers via pyannote-audio (feature opcional).

Requer:
  poetry install --with diarization
  HF_TOKEN=<token> no ambiente  (aceitar termos em hf.co/pyannote/speaker-diarization-3.1)

Importação lazy — o servidor não falha se pyannote não estiver instalado.
"""

from __future__ import annotations

import os


def diarize(audio_path: str) -> list[dict]:
    """Roda pipeline pyannote e retorna [{start, end, speaker}].

    Args:
        audio_path: Caminho para o arquivo WAV/MP3/etc.

    Returns:
        Lista de segmentos com speaker label.

    Raises:
        ImportError: pyannote-audio não instalado.
        RuntimeError: HF_TOKEN ausente ou pipeline falhou.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyannote-audio não instalado. "
            "Execute: poetry install --with diarization"
        ) from exc

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN não configurado. "
            "Defina HF_TOKEN com seu token HuggingFace e aceite os termos em "
            "hf.co/pyannote/speaker-diarization-3.1"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    # Mover para GPU se disponível
    try:
        import torch
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
    except Exception:
        pass

    diarization = pipeline(audio_path)

    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    return segments


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Adiciona campo 'speaker' a cada segmento de transcrição.

    A atribuição é feita por maior sobreposição temporal entre o segmento
    de transcrição e os turnos de diarização.

    Args:
        segments: Segmentos de transcrição [{start, end, text, ...}].
        diarization: Resultado de diarize() [{start, end, speaker}].

    Returns:
        Segmentos com campo 'speaker' adicionado.
    """
    result = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        best_speaker = "Unknown"
        best_overlap = 0.0

        for d in diarization:
            overlap = min(seg_end, d["end"]) - max(seg_start, d["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        result.append({**seg, "speaker": best_speaker})

    return result
