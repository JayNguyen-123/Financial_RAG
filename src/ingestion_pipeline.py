"""GPT-4o vision summaries of table/figure crops, optimised for dense retrieval.

Changes vs. the original:
* The client is built lazily (no API client at import time) with explicit
  timeout and retry settings.
* Crops are summarised concurrently (bounded by SUMMARY_CONCURRENCY) instead
  of one blocking call at a time.
* The prompt asks for a faithful Markdown transcription plus a retrieval
  summary, which both improves recall and gives the answer model exact figures
  to cite even when the image cannot be re-read.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from typing import List, Sequence

from config.settings import get_settings

SUMMARY_INSTRUCTIONS = (
    "You are given a cropped region from a scanned financial document: a data table or a chart.\n"
    "Produce two sections:\n"
    "1. TRANSCRIPTION - For tables: a faithful Markdown table with every row, column header, "
    "unit, currency and footnote marker exactly as printed. For charts: chart type, title, axes, "
    "series names, and every labelled data point.\n"
    "2. SUMMARY - A dense paragraph naming the entity, statement type, reporting periods, key "
    "line items and notable values/outliers, so the content can be matched by semantic search.\n"
    "Rules: copy numbers exactly; do not calculate, infer or fill in values that are not visible; "
    "write [illegible] for unreadable cells."
)


@lru_cache(maxsize=1)
def _vision_llm():
    from src.clients import build_chat_llm

    s = get_settings()
    return build_chat_llm(s, model=s.SUMMARY_MODEL, max_tokens=s.SUMMARY_MAX_TOKENS)


def _message(png_bytes: bytes):
    from langchain_core.messages import HumanMessage

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return [
        HumanMessage(
            content=[
                {"type": "text", "text": SUMMARY_INSTRUCTIONS},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ]
        )
    ]


def generate_table_summary(png_bytes: bytes) -> str:
    result = _vision_llm().invoke(_message(png_bytes), config={"run_name": "table_summary"})
    return result.content if isinstance(result.content, str) else str(result.content)


def summarize_images(images: Sequence[bytes]) -> tuple[List[str], int]:
    """Summarise crops concurrently. Returns (summaries, total_llm_tokens)."""
    if not images:
        return [], 0
    results = _vision_llm().batch(
        [_message(img) for img in images],
        config={"run_name": "table_summary", "max_concurrency": get_settings().SUMMARY_CONCURRENCY},
    )
    texts = [r.content if isinstance(r.content, str) else str(r.content) for r in results]
    tokens = sum(int((getattr(r, "usage_metadata", None) or {}).get("total_tokens", 0)) for r in results)
    return texts, tokens
