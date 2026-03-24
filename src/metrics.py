# src/metrics.py
"""In-memory thread-safe metrics store.

Expõe métricas no formato OpenMetrics/Prometheus via /metrics.
Renderização manual — sem dependência de prometheus-client.

Métricas:
  spooknix_chunks_total{type}       Counter
  spooknix_latency_ms               Histogram (buckets: 50,100,200,500,1000,2000,5000)
  spooknix_confidence               Gauge (média de confiança do último segmento)
  spooknix_sessions_total           Counter
  spooknix_active_sessions          Gauge (sessões WebSocket abertas no momento)
  spooknix_words_total              Counter
"""

from __future__ import annotations

import threading
from collections import defaultdict

_LATENCY_BUCKETS: tuple[float, ...] = (50, 100, 200, 500, 1000, 2000, 5000)


class _Counter:
    def __init__(self) -> None:
        self._v: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def inc(self, labels: dict | None = None, n: int = 1) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._v[key] += n

    def snapshot(self) -> dict[tuple, int]:
        with self._lock:
            return dict(self._v)


class _Histogram:
    """Histograma com buckets cumulativos no formato Prometheus."""

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self._buckets = tuple(sorted(buckets))
        # _counts[i] = nº de observações com valor <= _buckets[i]
        self._counts = [0] * len(self._buckets)
        self._inf = 0
        self._sum = 0.0
        self._total = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._total += 1
            self._inf += 1
            for i, b in enumerate(self._buckets):
                if value <= b:
                    self._counts[i] += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "buckets": list(self._buckets),
                "counts": list(self._counts),
                "inf": self._inf,
                "sum": self._sum,
                "count": self._total,
            }


class _Gauge:
    def __init__(self) -> None:
        self._v = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._v = value

    def inc(self, n: float = 1.0) -> None:
        with self._lock:
            self._v += n

    def dec(self, n: float = 1.0) -> None:
        with self._lock:
            self._v -= n

    def get(self) -> float:
        with self._lock:
            return self._v


# ── Singleton metrics ────────────────────────────────────────────────────────

chunks_total = _Counter()
latency_ms = _Histogram(_LATENCY_BUCKETS)
confidence = _Gauge()
sessions_total = _Counter()
active_sessions = _Gauge()
words_total = _Counter()


# ── Prometheus text format renderer ─────────────────────────────────────────

def _labels_str(labels: tuple) -> str:
    if not labels:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + parts + "}"


def render_prometheus() -> str:
    lines: list[str] = []

    # chunks_total
    lines.append("# HELP spooknix_chunks_total Total audio chunks processed")
    lines.append("# TYPE spooknix_chunks_total counter")
    snap = chunks_total.snapshot()
    if snap:
        for labels, v in snap.items():
            lines.append(f"spooknix_chunks_total{_labels_str(labels)} {v}")
    else:
        lines.append("spooknix_chunks_total 0")

    # latency_ms histogram
    lines.append("# HELP spooknix_latency_ms Transcription latency in milliseconds")
    lines.append("# TYPE spooknix_latency_ms histogram")
    snap_h = latency_ms.snapshot()
    for b, c in zip(snap_h["buckets"], snap_h["counts"]):
        lines.append(f'spooknix_latency_ms_bucket{{le="{b}"}} {c}')
    lines.append(f'spooknix_latency_ms_bucket{{le="+Inf"}} {snap_h["inf"]}')
    lines.append(f"spooknix_latency_ms_sum {snap_h['sum']}")
    lines.append(f"spooknix_latency_ms_count {snap_h['count']}")

    # confidence gauge
    lines.append("# HELP spooknix_confidence Average confidence of last transcribed segment")
    lines.append("# TYPE spooknix_confidence gauge")
    lines.append(f"spooknix_confidence {confidence.get()}")

    # sessions_total
    lines.append("# HELP spooknix_sessions_total Total WebSocket sessions opened")
    lines.append("# TYPE spooknix_sessions_total counter")
    snap = sessions_total.snapshot()
    if snap:
        for labels, v in snap.items():
            lines.append(f"spooknix_sessions_total{_labels_str(labels)} {v}")
    else:
        lines.append("spooknix_sessions_total 0")

    # active_sessions gauge
    lines.append("# HELP spooknix_active_sessions Current open WebSocket sessions")
    lines.append("# TYPE spooknix_active_sessions gauge")
    lines.append(f"spooknix_active_sessions {active_sessions.get()}")

    # words_total
    lines.append("# HELP spooknix_words_total Total words transcribed")
    lines.append("# TYPE spooknix_words_total counter")
    snap = words_total.snapshot()
    if snap:
        for labels, v in snap.items():
            lines.append(f"spooknix_words_total{_labels_str(labels)} {v}")
    else:
        lines.append("spooknix_words_total 0")

    lines.append("")
    return "\n".join(lines)
