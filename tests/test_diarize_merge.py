"""Merge boundary-aware: segmentos do Whisper são partidos onde o speaker muda."""

from __future__ import annotations

from src.diarizer import assign_speakers


def _seg(start, end, text, words=None):
    d = {"start": start, "end": end, "text": text}
    if words is not None:
        d["words"] = words
    return d


def test_merge_simples_sem_split():
    segments = [_seg(0, 5, "hello world")]
    diar = [{"start": 0, "end": 5, "speaker": "SPK_00"}]
    out = assign_speakers(segments, diar)
    assert len(out) == 1
    assert out[0]["speaker"] == "SPK_00"


def test_segmento_que_cruza_speakers_e_quebrado():
    segments = [_seg(0.0, 10.0, "first half second half")]
    diar = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "speaker": "B"},
    ]
    out = assign_speakers(segments, diar, split_at_boundaries=True)
    assert len(out) == 2
    assert out[0]["speaker"] == "A"
    assert out[1]["speaker"] == "B"
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 5.0
    assert out[1]["start"] == 5.0
    assert out[1]["end"] == 10.0


def test_split_usa_word_timestamps_quando_disponivel():
    words = [
        {"start": 0.0, "end": 2.0, "word": "Alice "},
        {"start": 2.0, "end": 4.0, "word": "speaks "},
        {"start": 6.0, "end": 8.0, "word": "Bob "},
        {"start": 8.0, "end": 9.0, "word": "replies"},
    ]
    segments = [_seg(0.0, 9.0, "Alice speaks Bob replies", words=words)]
    diar = [
        {"start": 0.0, "end": 5.0, "speaker": "ALICE"},
        {"start": 5.0, "end": 9.0, "speaker": "BOB"},
    ]
    out = assign_speakers(segments, diar, split_at_boundaries=True)
    assert len(out) == 2
    assert "Alice" in out[0]["text"]
    assert "Bob" in out[1]["text"]


def test_split_disabled_keeps_one_segment():
    segments = [_seg(0.0, 10.0, "mixed turn")]
    diar = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "speaker": "B"},
    ]
    out = assign_speakers(segments, diar, split_at_boundaries=False)
    assert len(out) == 1
    # Maior overlap: empate em 5s — fica com o primeiro encontrado
    assert out[0]["speaker"] == "A"


def test_split_sem_fronteiras_internas_passa_direto():
    segments = [_seg(0.0, 5.0, "single speaker")]
    diar = [{"start": 0.0, "end": 5.0, "speaker": "SOLO"}]
    out = assign_speakers(segments, diar, split_at_boundaries=True)
    assert len(out) == 1
    assert out[0]["speaker"] == "SOLO"
