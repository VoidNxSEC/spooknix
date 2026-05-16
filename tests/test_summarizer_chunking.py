"""Chunking de segmentos com orçamento de tokens."""

from __future__ import annotations

import pytest

from src.summarizer import Chunk, chunk_segments, stitch


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def test_chunking_respeita_max_tokens():
    segs = [_seg(i * 10.0, (i + 1) * 10.0, "hello world " * 20) for i in range(20)]
    chunks = chunk_segments(segs, max_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text  # nenhum chunk vazio


def test_chunk_prefixa_mm_ss_em_cada_linha():
    segs = [_seg(60.0, 90.0, "first"), _seg(90.0, 120.0, "second")]
    chunks = chunk_segments(segs, max_tokens=1000)
    assert len(chunks) == 1
    text = chunks[0].text
    assert "[01:00 → 01:30]" in text
    assert "first" in text
    assert "[01:30 → 02:00]" in text
    assert "second" in text


def test_chunk_com_source_uri_gera_link():
    segs = [_seg(120.0, 130.0, "thing")]
    chunks = chunk_segments(segs, max_tokens=1000, source_uri="vid.mp4")
    assert "[02:00](vid.mp4#t=120)" in chunks[0].text


def test_chunk_range_preserva_intervalo():
    segs = [_seg(0, 5, "a"), _seg(5, 10, "b"), _seg(10, 15, "c")]
    chunks = chunk_segments(segs, max_tokens=1000)
    assert chunks[0].start == 0.0
    assert chunks[0].end == 15.0


def test_chunk_segmento_vazio_ignorado():
    segs = [_seg(0, 1, ""), _seg(1, 2, "real text"), _seg(2, 3, "   ")]
    chunks = chunk_segments(segs, max_tokens=1000)
    assert len(chunks) == 1
    assert "real text" in chunks[0].text


def test_stitch_separa_com_hr():
    assert "---" in stitch(["a", "b"])
    assert stitch([" ", ""]) == ""
    assert stitch(["solo"]) == "solo"
