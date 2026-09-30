"""Prometheus metrics, served on a dedicated port (not through the ingress).

The previous setup exposed /metrics on the public API port, leaking route
names, traffic volumes and error rates to anyone who could reach the ingress.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, start_http_server

    LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens consumed", ["operation"])
    GUARDRAIL_BLOCKS = Counter("rag_guardrail_blocks_total", "Answers blocked by guardrails", ["rule"])
    QUOTA_REJECTIONS = Counter("rag_quota_rejections_total", "Requests rejected by quotas", ["kind"])
    DOCUMENT_EVENTS = Counter("rag_document_events_total", "Document lifecycle events", ["event"])
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False

    class _Noop:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            return None

    LLM_TOKENS = GUARDRAIL_BLOCKS = QUOTA_REJECTIONS = DOCUMENT_EVENTS = _Noop()

_started = False


def start_metrics_server(port: int) -> None:
    """Start the metrics listener once per process (no-op when port == 0)."""
    global _started
    if _started or port <= 0 or not _AVAILABLE:
        return
    start_http_server(port)
    _started = True
    logger.info("Metrics server listening", extra={"port": port})


def instrument_app(app) -> None:
    """Record HTTP metrics into the default registry without exposing a route."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:  # pragma: no cover
        return
    Instrumentator(should_group_status_codes=False, excluded_handlers=["/healthz/.*"]).instrument(app)
