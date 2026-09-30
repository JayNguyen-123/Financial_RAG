"""Narrative text extraction for scanned pages (MD&A, notes, footnotes).

Tables and figures are already indexed as image crops, so they are masked
(painted white) before OCR. That avoids indexing the same numbers twice as
garbled OCR text, which would compete with the clean vision transcription.

Chunks never cross page boundaries so every text hit cites an exact page.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from config.settings import Settings

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    tokens: int = 0  # LLM tokens spent (vision mode)


# ── Pure helpers ────────────────────────────────────────────────────────
def mask_regions(page_image, boxes: Sequence[Box]):
    """Return a copy of the page with the given boxes painted white."""
    from PIL import ImageDraw

    img = page_image.copy()
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], fill="white")
    return img


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_ocr_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x0c", "\n\n")  # form feed = page/column break
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # re-join words hyphenated across lines
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Join soft-wrapped lines inside a paragraph; keep blank-line paragraph breaks.
    text = re.sub(r"(?<=[^\n])\n(?=[^\n])", " ", text)
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware greedy chunking with character overlap."""
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    units: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        # Split long paragraphs on sentence boundaries, then hard-wrap.
        sentences = re.split(r"(?<=[.!?;])\s+", para)
        for s in sentences:
            while len(s) > size:
                cut = s.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                units.append(s[:cut].strip())
                s = s[cut:].strip()
            if s:
                units.append(s)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current = f"{tail} {unit}".strip() if tail and len(tail) + 1 + len(unit) <= size else unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks


# ── Extractors ──────────────────────────────────────────────────────────
TextExtractor = Callable[[object, Sequence[Box], int], PageText]


class TesseractExtractor:
    def __init__(self, lang: str = "eng"):
        import pytesseract

        self._tess = pytesseract
        self.lang = lang

    def __call__(self, page_image, masked: Sequence[Box], page_number: int) -> PageText:
        img = mask_regions(page_image, masked).convert("L")
        raw = self._tess.image_to_string(img, lang=self.lang, config="--oem 1 --psm 3")
        return PageText(page_number, clean_ocr_text(raw))


VISION_TEXT_PROMPT = (
    "This is a scanned page from a financial filing. Tables and charts have been blanked out. "
    "Transcribe ALL remaining text exactly, in reading order, preserving headings and paragraph "
    "breaks. Include footnotes and page headers/footers. Do not summarise, interpret or describe "
    "images. Write [illegible] for unreadable words. If there is no text, reply with nothing."
)


class VisionExtractor:
    MAX_SIDE = 2000

    def __init__(self, settings: Settings):
        from src.clients import build_chat_llm

        self.llm = build_chat_llm(settings, model=settings.SUMMARY_MODEL, max_tokens=4000)

    def __call__(self, page_image, masked: Sequence[Box], page_number: int) -> PageText:
        from langchain_core.messages import HumanMessage

        img = mask_regions(page_image, masked)
        img.thumbnail((self.MAX_SIDE, self.MAX_SIDE))
        buf = io.BytesIO()
        img.convert("L").save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        msg = HumanMessage(content=[
            {"type": "text", "text": VISION_TEXT_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
        ])
        result = self.llm.invoke([msg], config={"run_name": "page_text_transcription"})
        usage = getattr(result, "usage_metadata", None) or {}
        text = result.content if isinstance(result.content, str) else str(result.content)
        return PageText(page_number, clean_ocr_text(text), int(usage.get("total_tokens", 0)))


def build_text_extractor(settings: Settings) -> TextExtractor | None:
    if settings.TEXT_EXTRACTION == "tesseract":
        return TesseractExtractor(settings.OCR_LANG)
    if settings.TEXT_EXTRACTION == "vision":
        return VisionExtractor(settings)
    return None


def chunk_pages(pages: Sequence[PageText], settings: Settings) -> list[tuple[int, int, str]]:
    """-> [(page_number, chunk_index, text)] skipping near-empty pages."""
    out: list[tuple[int, int, str]] = []
    for page in pages:
        if len(page.text) < settings.MIN_PAGE_TEXT_CHARS:
            continue
        for idx, chunk in enumerate(chunk_text(page.text, settings.TEXT_CHUNK_SIZE, settings.TEXT_CHUNK_OVERLAP)):
            out.append((page.page_number, idx, chunk))
    return out
