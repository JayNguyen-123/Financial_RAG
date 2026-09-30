from PIL import Image

from config.settings import get_settings
from src.text_extraction import PageText, chunk_pages, chunk_text, clean_ocr_text, mask_regions


def test_clean_ocr_text_rejoins_hyphens_and_soft_wraps():
    raw = "Revenue in-\ncreased  12%\nyear over year.\n\n\n\nNote 4:\x0cLeases"
    assert clean_ocr_text(raw) == "Revenue increased 12% year over year.\n\nNote 4:\n\nLeases"


def test_chunk_text_respects_size_and_overlap():
    para = " ".join(f"Sentence number {i} about operating margin." for i in range(200))
    chunks = chunk_text(para, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # consecutive chunks share some text (overlap)
    assert any(chunks[i][-20:].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_chunk_text_keeps_small_paragraphs_together():
    assert chunk_text("A.\n\nB.", size=100, overlap=10) == ["A.\n\nB."]
    assert chunk_text("", 100, 10) == []


def test_mask_regions_whites_out_boxes():
    img = Image.new("RGB", (100, 100), "black")
    masked = mask_regions(img, [(10, 10, 50, 50)])
    assert masked.getpixel((20, 20)) == (255, 255, 255)
    assert masked.getpixel((80, 80)) == (0, 0, 0)
    assert img.getpixel((20, 20)) == (0, 0, 0)  # original untouched


def test_chunk_pages_skips_empty_pages_and_keeps_page_numbers():
    s = get_settings()
    pages = [PageText(1, "x" * 5), PageText(2, "Management discussion. " * 100)]
    out = chunk_pages(pages, s)
    assert out and all(p == 2 for p, _, _ in out)
    assert [i for _, i, _ in out] == list(range(len(out)))


def test_tesseract_extractor_masks_tables_end_to_end():
    import shutil

    import pytest

    pytest.importorskip("pytesseract")
    if not shutil.which("tesseract"):
        pytest.skip("tesseract binary not installed")
    from PIL import ImageDraw, ImageFont

    from src.text_extraction import TesseractExtractor

    img = Image.new("RGB", (1600, 600), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    draw.text((50, 50), "Operating income increased significantly.", fill="black", font=font)
    draw.text((50, 350), "SECRETTABLEVALUE 999", fill="black", font=font)
    page = TesseractExtractor("eng")(img, [(0, 300, 1600, 450)], page_number=3)
    assert page.page_number == 3
    assert "Operating income" in page.text
    assert "SECRETTABLEVALUE" not in page.text  # masked region is not OCR'd
