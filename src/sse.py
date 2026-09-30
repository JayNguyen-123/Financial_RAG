"""Server-Sent Events framing.

The original endpoint advertised ``text/event-stream`` but yielded raw tokens,
which is not valid SSE (no ``data:`` framing, and any newline inside a token
breaks the event). Every event here is a single JSON-encoded ``data:`` line.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable proxy buffering (nginx) so tokens reach the client immediately.
    "X-Accel-Buffering": "no",
}


def sse_event(event: str, data: Mapping[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
