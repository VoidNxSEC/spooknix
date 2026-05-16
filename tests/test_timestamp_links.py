"""Formatação mm:ss e links de âncora temporal."""

from __future__ import annotations

import pytest

from src.timestamp_links import format_mmss, seconds_to_link


def test_format_mmss_zero():
    assert format_mmss(0) == "00:00"


def test_format_mmss_menos_de_uma_hora():
    assert format_mmss(754) == "12:34"
    assert format_mmss(59.9) == "00:59"


def test_format_mmss_acima_de_uma_hora():
    assert format_mmss(3661) == "1:01:01"
    assert format_mmss(7200) == "2:00:00"


def test_format_mmss_negativo_clamp():
    assert format_mmss(-5) == "00:00"


def test_seconds_to_link_com_uri():
    assert seconds_to_link(754, "lecture.mp4") == "[12:34](lecture.mp4#t=754)"


def test_seconds_to_link_sem_uri_so_label():
    assert seconds_to_link(60, None) == "[01:00]"
    assert seconds_to_link(60, "") == "[01:00]"
