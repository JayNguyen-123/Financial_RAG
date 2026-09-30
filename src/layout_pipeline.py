"""PDF -> page images -> Detectron2 layout detection -> table/figure crops.

Fixes vs. the original:
* Pages are rendered one at a time. ``convert_from_path(pdf)`` rendered every
  page at 200 DPI into memory at once (~25 MB/page), so a 300-page filing
  could OOM-kill the worker.
* A page cap (MAX_PDF_PAGES) bounds cost and runtime for hostile/huge uploads.
* Overlapping duplicate detections (common with Faster R-CNN) are suppressed,
  and tiny boxes are dropped.
* LayoutParser expects RGB input (per its docs); the original converted to BGR.
* Crops are returned as raw PNG bytes; base64 is produced only where needed.
* Model config/weights and device are configurable so weights can be baked
  into the image instead of downloaded from Dropbox at runtime.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from config.settings import Settings

logger = logging.getLogger(__name__)

TARGET_TYPES = ("Table", "Figure")
LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
PADDING_PX = 12
MIN_BOX_AREA_RATIO = 0.002  # drop boxes smaller than 0.2% of the page
IOU_DEDUP_THRESHOLD = 0.6
CONTAINMENT_THRESHOLD = 0.9

Box = Tuple[int, int, int, int]


@dataclass
class ParseResult:
    pages_total: int = 0
    pages_processed: int = 0
    elements: List[Dict[str, Any]] = field(default_factory=list)
    page_texts: List[Any] = field(default_factory=list)  # List[PageText]
    llm_tokens: int = 0


# ── Pure geometry helpers (unit-tested without heavy deps) ──────────────
def box_area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a: Box, b: Box) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return box_area((x1, y1, x2, y2))


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def dedupe_boxes(
    detections: Sequence[Tuple[Box, float, str]],
    iou_threshold: float = IOU_DEDUP_THRESHOLD,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
) -> List[Tuple[Box, float, str]]:
    """Greedy NMS by score; also drops boxes almost fully inside a kept box."""
    kept: List[Tuple[Box, float, str]] = []
    for det in sorted(detections, key=lambda d: d[1], reverse=True):
        box = det[0]
        area = box_area(box) or 1
        redundant = any(
            iou(box, k[0]) >= iou_threshold or intersection(box, k[0]) / area >= containment_threshold
            for k in kept
        )
        if not redundant:
            kept.append(det)
    # Reading order: top-to-bottom, then left-to-right.
    return sorted(kept, key=lambda d: (d[0][1], d[0][0]))


def pad_box(box: Box, width: int, height: int, padding: int = PADDING_PX) -> Box:
    x1, y1, x2, y2 = box
    return (max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding))


# ── Parser ──────────────────────────────────────────────────────────────
class FinancialLayoutParser:
    def __init__(self, settings: Settings):
        import layoutparser as lp

        self.settings = settings
        extra_config = [
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", settings.LAYOUT_SCORE_THRESHOLD,
            "MODEL.DEVICE", settings.LAYOUT_DEVICE,
        ]
        kwargs: Dict[str, Any] = {
            "config_path": settings.LAYOUT_MODEL_CONFIG,
            "label_map": LABEL_MAP,
            "extra_config": extra_config,
        }
        if settings.LAYOUT_MODEL_WEIGHTS:
            kwargs["model_path"] = settings.LAYOUT_MODEL_WEIGHTS
        logger.info("Loading layout model", extra={"config": settings.LAYOUT_MODEL_CONFIG})
        self.model = lp.Detectron2LayoutModel(**kwargs)

    def page_count(self, pdf_path: str) -> int:
        from pdf2image import pdfinfo_from_path

        return int(pdfinfo_from_path(pdf_path)["Pages"])

    def render_page(self, pdf_path: str, page_number: int):
        from pdf2image import convert_from_path

        pages = convert_from_path(
            pdf_path, dpi=self.settings.PDF_RENDER_DPI, first_page=page_number, last_page=page_number
        )
        return pages[0].convert("RGB")

    def detect(self, page_image) -> List[Tuple[Box, float, str]]:
        import numpy as np

        layout = self.model.detect(np.asarray(page_image))
        dets: List[Tuple[Box, float, str]] = []
        for block in layout:
            if block.type not in TARGET_TYPES:
                continue
            x1, y1, x2, y2 = (int(round(c)) for c in block.coordinates)
            dets.append(((x1, y1, x2, y2), float(block.score or 0.0), block.type))
        return dets

    def process_pdf(self, pdf_path: str, text_extractor: Any = None) -> ParseResult:
        """Detect and crop tables/figures; optionally extract narrative text per page.

        ``text_extractor(page_image, masked_boxes, page_number) -> PageText`` runs
        on the same rendered page (no second render), with table/figure regions
        masked so their numbers are not indexed twice.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

        total = self.page_count(pdf_path)
        limit = min(total, self.settings.MAX_PDF_PAGES)
        if total > limit:
            logger.warning("PDF truncated to page limit", extra={"pages": total, "limit": limit})

        result = ParseResult(pages_total=total, pages_processed=limit)
        for page_number in range(1, limit + 1):
            page = self.render_page(pdf_path, page_number)
            width, height = page.size
            min_area = MIN_BOX_AREA_RATIO * width * height
            kept = dedupe_boxes([d for d in self.detect(page) if box_area(d[0]) >= min_area])
            padded_boxes = []
            for block_index, (box, score, label) in enumerate(kept):
                padded = pad_box(box, width, height)
                padded_boxes.append(padded)
                crop = page.crop(padded)
                buf = io.BytesIO()
                crop.save(buf, format="PNG", optimize=True)
                result.elements.append(
                    {
                        "page_number": page_number,
                        "block_index": block_index,
                        "type": label.lower(),
                        "bbox": list(box),
                        "score": score,
                        "dpi": self.settings.PDF_RENDER_DPI,
                        "png_bytes": buf.getvalue(),
                    }
                )
            if text_extractor is not None:
                page_text = text_extractor(page, padded_boxes, page_number)
                result.page_texts.append(page_text)
                result.llm_tokens += page_text.tokens
            page.close()

        logger.info(
            "Layout extraction complete",
            extra={"pages": limit, "elements": len(result.elements), "text_pages": len(result.page_texts)},
        )
        return result
