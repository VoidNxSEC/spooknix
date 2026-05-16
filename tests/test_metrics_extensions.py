"""Métricas novas do Sprint 9: interview/summarize/llm/tts."""

from __future__ import annotations

from src import metrics
from src.metrics import render_prometheus


def test_interviews_total_e_render():
    metrics.interviews_total = type(metrics.interviews_total)()  # zera
    metrics.interviews_total.inc({"persona": "sarah", "scenario": "behavioral", "difficulty": "hard"})
    metrics.interviews_total.inc({"persona": "sarah", "scenario": "behavioral", "difficulty": "hard"})
    out = render_prometheus()
    assert "spooknix_interviews_total" in out
    assert 'persona="sarah"' in out
    assert 'scenario="behavioral"' in out
    assert 'difficulty="hard"' in out


def test_interview_duration_histograma():
    metrics.interview_duration_seconds = type(metrics.interview_duration_seconds)(
        metrics.interview_duration_seconds._buckets
    )
    metrics.interview_duration_seconds.observe(45.0)
    metrics.interview_duration_seconds.observe(800.0)
    out = render_prometheus()
    assert "spooknix_interview_duration_seconds_bucket" in out
    assert "spooknix_interview_duration_seconds_count 2" in out


def test_summaries_total_por_template():
    metrics.summaries_total = type(metrics.summaries_total)()
    metrics.summaries_total.inc({"template": "lecture"})
    metrics.summaries_total.inc({"template": "lecture"})
    metrics.summaries_total.inc({"template": "meeting"})
    out = render_prometheus()
    assert 'spooknix_summaries_total{template="lecture"} 2' in out
    assert 'spooknix_summaries_total{template="meeting"} 1' in out


def test_summary_chunks_total_aceita_n():
    metrics.summary_chunks_total = type(metrics.summary_chunks_total)()
    metrics.summary_chunks_total.inc(n=12)
    out = render_prometheus()
    assert "spooknix_summary_chunks_total 12" in out


def test_llm_turn_latency_render():
    metrics.llm_turn_latency_ms = type(metrics.llm_turn_latency_ms)(
        metrics.llm_turn_latency_ms._buckets
    )
    metrics.llm_turn_latency_ms.observe(300.0)
    out = render_prometheus()
    assert "spooknix_llm_turn_latency_ms_bucket" in out
    assert "spooknix_llm_turn_latency_ms_count 1" in out


def test_tts_synthesize_latency_render():
    metrics.tts_synthesize_latency_ms = type(metrics.tts_synthesize_latency_ms)(
        metrics.tts_synthesize_latency_ms._buckets
    )
    metrics.tts_synthesize_latency_ms.observe(180.0)
    out = render_prometheus()
    assert "spooknix_tts_synthesize_latency_ms_bucket" in out
    assert "spooknix_tts_synthesize_latency_ms_count 1" in out
