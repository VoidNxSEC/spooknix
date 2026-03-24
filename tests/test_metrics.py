"""Testes para src/metrics.py — sem dependências externas."""

from __future__ import annotations

import threading

import pytest

import src.metrics as metrics
from src.metrics import (
    _Counter,
    _Gauge,
    _Histogram,
    render_prometheus,
)


# ── _Counter ─────────────────────────────────────────────────────────────────

def test_counter_starts_at_zero():
    c = _Counter()
    assert c.snapshot() == {}


def test_counter_inc_no_labels():
    c = _Counter()
    c.inc()
    snap = c.snapshot()
    assert snap[()] == 1


def test_counter_inc_with_labels():
    c = _Counter()
    c.inc({"type": "received"})
    c.inc({"type": "received"})
    c.inc({"type": "flushed"})
    snap = c.snapshot()
    assert snap[(("type", "received"),)] == 2
    assert snap[(("type", "flushed"),)] == 1


def test_counter_thread_safe():
    c = _Counter()
    threads = [threading.Thread(target=lambda: c.inc()) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.snapshot()[()] == 100


# ── _Gauge ────────────────────────────────────────────────────────────────────

def test_gauge_starts_at_zero():
    g = _Gauge()
    assert g.get() == 0.0


def test_gauge_set():
    g = _Gauge()
    g.set(0.95)
    assert abs(g.get() - 0.95) < 1e-9


def test_gauge_inc_dec():
    g = _Gauge()
    g.inc(3)
    g.dec(1)
    assert abs(g.get() - 2.0) < 1e-9


# ── _Histogram ────────────────────────────────────────────────────────────────

def test_histogram_observe_below_first_bucket():
    h = _Histogram((50, 100, 200))
    h.observe(30)
    snap = h.snapshot()
    assert snap["counts"] == [1, 1, 1]
    assert snap["inf"] == 1
    assert snap["count"] == 1


def test_histogram_observe_between_buckets():
    h = _Histogram((50, 100, 200))
    h.observe(75)
    snap = h.snapshot()
    # 75 <= 100 and 75 <= 200, but 75 > 50
    assert snap["counts"][0] == 0  # 75 > 50
    assert snap["counts"][1] == 1  # 75 <= 100
    assert snap["counts"][2] == 1  # 75 <= 200


def test_histogram_observe_above_all_buckets():
    h = _Histogram((50, 100, 200))
    h.observe(9999)
    snap = h.snapshot()
    assert snap["counts"] == [0, 0, 0]
    assert snap["inf"] == 1


def test_histogram_sum_and_count():
    h = _Histogram((100, 200))
    h.observe(50)
    h.observe(150)
    snap = h.snapshot()
    assert snap["count"] == 2
    assert abs(snap["sum"] - 200.0) < 1e-9


# ── render_prometheus ─────────────────────────────────────────────────────────

def test_render_prometheus_contains_all_metric_names():
    text = render_prometheus()
    for name in [
        "spooknix_chunks_total",
        "spooknix_latency_ms",
        "spooknix_confidence",
        "spooknix_sessions_total",
        "spooknix_active_sessions",
        "spooknix_words_total",
    ]:
        assert name in text, f"Metric '{name}' not found in output"


def test_render_prometheus_has_help_and_type():
    text = render_prometheus()
    assert "# HELP" in text
    assert "# TYPE" in text


def test_render_prometheus_histogram_buckets():
    text = render_prometheus()
    assert 'le="50"' in text
    assert 'le="+Inf"' in text


def test_render_prometheus_ends_with_newline():
    text = render_prometheus()
    assert text.endswith("\n")


def test_render_prometheus_after_observe():
    # Usar instâncias locais para não poluir o estado global
    h = _Histogram((50, 100))
    h.observe(80)
    snap = h.snapshot()
    # 80 > 50, 80 <= 100
    assert snap["counts"][0] == 0
    assert snap["counts"][1] == 1
