from src.layout_pipeline import box_area, dedupe_boxes, iou, pad_box


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_dedupe_keeps_highest_score_and_drops_contained():
    dets = [
        ((0, 0, 100, 100), 0.80, "Table"),
        ((2, 2, 98, 98), 0.95, "Table"),       # near-duplicate, higher score -> kept
        ((10, 10, 50, 50), 0.90, "Figure"),    # fully contained -> dropped
        ((0, 200, 100, 300), 0.85, "Figure"),  # separate -> kept
    ]
    kept = dedupe_boxes(dets)
    assert [d[0] for d in kept] == [(2, 2, 98, 98), (0, 200, 100, 300)]


def test_pad_box_clamps_to_page():
    assert pad_box((5, 5, 95, 95), width=100, height=100, padding=12) == (0, 0, 100, 100)
    assert box_area((0, 0, 0, 10)) == 0


def test_process_pdf_crops_masks_and_extracts_text(tmp_path):
    from PIL import Image

    from config.settings import get_settings
    from src.layout_pipeline import FinancialLayoutParser
    from src.text_extraction import PageText

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    parser = object.__new__(FinancialLayoutParser)  # skip Detectron2 model loading
    parser.settings = get_settings().model_copy(update={"MAX_PDF_PAGES": 2})
    parser.page_count = lambda path: 3
    parser.render_page = lambda path, n: Image.new("RGB", (1000, 1000), "white")
    parser.detect = lambda page: [((100, 100, 600, 400), 0.95, "Table"), ((110, 110, 590, 390), 0.80, "Table"),
                                  ((0, 0, 5, 5), 0.99, "Figure")]  # duplicate + tiny box
    seen = []

    def extractor(page, boxes, page_number):
        seen.append((page_number, boxes))
        return PageText(page_number, f"narrative {page_number}", tokens=7)

    result = parser.process_pdf(str(pdf), extractor)
    assert result.pages_total == 3 and result.pages_processed == 2          # page cap applied
    assert [(e["page_number"], e["block_index"]) for e in result.elements] == [(1, 0), (2, 0)]
    assert result.elements[0]["png_bytes"].startswith(b"\x89PNG")
    assert seen[0] == (1, [(88, 88, 612, 412)])                             # padded mask passed to OCR
    assert result.llm_tokens == 14 and [p.text for p in result.page_texts] == ["narrative 1", "narrative 2"]
