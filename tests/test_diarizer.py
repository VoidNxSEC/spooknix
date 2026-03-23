"""Testes para src/diarizer.py.

Não requer pyannote-audio instalado — a importação lazy é mockada.
"""

from __future__ import annotations

import pytest


# ── assign_speakers ────────────────────────────────────────────────────────


from src.diarizer import assign_speakers


def _seg(start: float, end: float, text: str = "") -> dict:
    return {"start": start, "end": end, "text": text}


def _turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


def test_assign_speakers_exact_match():
    """Segmento coincide exatamente com um turno."""
    segs = [_seg(0.0, 2.0, "Olá mundo")]
    diarz = [_turn(0.0, 2.0, "Speaker-A")]
    result = assign_speakers(segs, diarz)
    assert result[0]["speaker"] == "Speaker-A"


def test_assign_speakers_partial_overlap():
    """Maior sobreposição vence."""
    segs = [_seg(1.0, 4.0, "Texto")]
    diarz = [
        _turn(0.0, 2.0, "Speaker-A"),  # sobreposição = 1s
        _turn(2.0, 5.0, "Speaker-B"),  # sobreposição = 2s
    ]
    result = assign_speakers(segs, diarz)
    assert result[0]["speaker"] == "Speaker-B"


def test_assign_speakers_no_overlap():
    """Sem sobreposição mantém 'Unknown'."""
    segs = [_seg(5.0, 6.0, "Texto")]
    diarz = [_turn(0.0, 4.0, "Speaker-A")]
    result = assign_speakers(segs, diarz)
    assert result[0]["speaker"] == "Unknown"


def test_assign_speakers_multiple_segments():
    """Múltiplos segmentos são atribuídos corretamente."""
    segs = [
        _seg(0.0, 2.0, "Oi"),
        _seg(2.5, 5.0, "Tudo bem?"),
        _seg(5.5, 8.0, "Sim, obrigado."),
    ]
    diarz = [
        _turn(0.0, 3.0, "Speaker-A"),
        _turn(3.0, 9.0, "Speaker-B"),
    ]
    result = assign_speakers(segs, diarz)
    assert result[0]["speaker"] == "Speaker-A"
    assert result[1]["speaker"] == "Speaker-B"
    assert result[2]["speaker"] == "Speaker-B"


def test_assign_speakers_preserves_original_fields():
    """Campos originais dos segmentos são preservados."""
    words = [{"start": 0.0, "end": 1.0, "word": "Oi", "probability": 0.99}]
    segs = [{"start": 0.0, "end": 1.0, "text": "Oi", "words": words}]
    diarz = [_turn(0.0, 1.0, "Speaker-A")]
    result = assign_speakers(segs, diarz)
    assert result[0]["words"] == words
    assert result[0]["text"] == "Oi"


def test_assign_speakers_empty_diarization():
    """Diarização vazia → todos os segmentos ficam 'Unknown'."""
    segs = [_seg(0.0, 2.0, "Texto"), _seg(2.0, 4.0, "Mais texto")]
    result = assign_speakers(segs, [])
    assert all(s["speaker"] == "Unknown" for s in result)


def test_assign_speakers_empty_segments():
    """Segmentos vazios → retorna lista vazia."""
    diarz = [_turn(0.0, 5.0, "Speaker-A")]
    result = assign_speakers([], diarz)
    assert result == []


# ── diarize() — importação lazy ────────────────────────────────────────────


def test_diarize_raises_import_error_without_pyannote():
    """diarize() levanta ImportError quando pyannote não está instalado."""
    import sys
    from unittest.mock import patch

    # Remover pyannote do cache de módulos, se presente
    mods_to_block = [k for k in sys.modules if k.startswith("pyannote")]
    saved = {k: sys.modules.pop(k) for k in mods_to_block}

    try:
        with patch.dict("sys.modules", {"pyannote.audio": None}):
            from src.diarizer import diarize
            with pytest.raises(ImportError, match="pyannote-audio"):
                diarize("/tmp/fake.wav")
    finally:
        sys.modules.update(saved)


def test_diarize_raises_runtime_error_without_hf_token(monkeypatch):
    """diarize() levanta RuntimeError quando HF_TOKEN está ausente."""
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("HF_TOKEN", "")

    fake_pyannote = MagicMock()
    with patch.dict("sys.modules", {"pyannote.audio": fake_pyannote}):
        from importlib import reload
        import src.diarizer as diarizer_mod
        reload(diarizer_mod)
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            diarizer_mod.diarize("/tmp/fake.wav")
