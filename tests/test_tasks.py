from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src import tasks
from src.documents import DocumentManifest, DocumentRegistry
from src.layout_pipeline import ParseResult
from src.storage import image_key, original_key
from src.text_extraction import PageText
from tests.fakes import FakeVectorStore

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
DOC = "c" * 32


def _element(page=1, idx=0):
    return {"page_number": page, "block_index": idx, "type": "table", "bbox": [1, 2, 3, 4],
            "score": 0.9, "dpi": 200, "png_bytes": PNG}


@pytest.fixture
def env(blob_store):
    registry = DocumentRegistry(blob_store)
    vs = FakeVectorStore()
    parser = MagicMock()
    blob_store.put(original_key("acme", DOC), b"%PDF-1.4 test")
    registry.save(DocumentManifest(document_id=DOC, tenant="acme", filename="q3.pdf", owner="alice",
                                   acl=["tenant:all", "user:alice"], sha256="c" * 64, size_bytes=12))
    patches = [
        patch.object(tasks, "get_blob_store", return_value=blob_store),
        patch.object(tasks, "get_registry", return_value=registry),
        patch.object(tasks, "get_vectorstore", return_value=vs),
        patch.object(tasks, "get_layout_parser", return_value=parser),
        patch.object(tasks, "get_text_extractor", return_value=None),
        patch.object(tasks, "get_sync_redis", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()
    yield SimpleNamespace(store=blob_store, registry=registry, vs=vs, parser=parser)
    for p in patches:
        p.stop()


def test_metadata_is_pinecone_safe():
    m = SimpleNamespace(document_id=DOC, tenant="acme", acl=["tenant:all"], filename="f.pdf", sha256="s")
    for meta in (tasks.crop_metadata(_element(), "id1", m, "images/acme/id1.png"),
                 tasks.text_metadata("id2", 3, 0, m)):
        for value in meta.values():
            assert isinstance(value, (str, int, float, bool)) or (
                isinstance(value, list) and all(isinstance(v, str) for v in value))


def test_ids():
    assert tasks.crop_element_id(DOC, 3, 1) == f"{DOC}-p0003-b001"
    assert tasks.is_crop_id(tasks.crop_element_id(DOC, 3, 1))
    assert not tasks.is_crop_id(tasks.text_chunk_id(DOC, 3, 1))


def test_rejects_untrusted_ids(env):
    res = tasks.ingest_document.apply(args=("../etc", DOC))
    assert res.failed()


def test_success_indexes_crops_and_text(env):
    env.parser.process_pdf.return_value = ParseResult(
        pages_total=2, pages_processed=2, elements=[_element(1, 0)],
        page_texts=[PageText(2, "Management discussion and analysis. " * 20)],
    )
    with patch("src.ingestion_pipeline.summarize_images", return_value=(["| a | b |"], 50)):
        result = tasks.ingest_document.apply(args=("acme", DOC)).get()

    assert result["status"] == "indexed" and result["crops_indexed"] == 1 and result["text_chunks_indexed"] >= 1
    docs, ids, namespace = env.vs.added[0]
    assert namespace == "tenant-acme" and ids[0] == f"{DOC}-p0001-b000"
    assert docs[-1].metadata["content_type"] == "text" and docs[-1].metadata["acl"] == ["tenant:all", "user:alice"]
    assert env.store.get(image_key("acme", ids[0])) == PNG
    m = env.registry.get("acme", DOC)
    assert m.status == "indexed" and m.crop_ids == [ids[0]] and len(m.text_chunk_ids) == result["text_chunks_indexed"]


def test_reindex_removes_stale_vectors(env):
    m = env.registry.get("acme", DOC)
    m.crop_ids = [f"{DOC}-p0001-b000", f"{DOC}-p0009-b000"]
    m.status = "indexed"
    env.registry.save(m)
    env.parser.process_pdf.return_value = ParseResult(pages_processed=1, elements=[_element(1, 0)])
    with patch("src.ingestion_pipeline.summarize_images", return_value=(["summary"], 10)):
        result = tasks.ingest_document.apply(args=("acme", DOC)).get()
    assert result["stale_removed"] == 1
    assert ([f"{DOC}-p0009-b000"], "tenant-acme") in env.vs.deleted


def test_terminal_failure_marks_manifest_failed(env):
    env.parser.process_pdf.side_effect = ValueError("corrupt pdf")
    res = tasks.ingest_document.apply(args=("acme", DOC))
    assert res.failed()
    m = env.registry.get("acme", DOC)
    assert m.status == "failed" and m.error == "ValueError"


def test_deleted_manifest_is_a_noop(env):
    env.registry.delete_manifest("acme", DOC)
    assert tasks.ingest_document.apply(args=("acme", DOC)).get()["status"] == "cancelled"


def test_transient_error_classification():
    assert tasks.is_transient(ConnectionError())
    assert not tasks.is_transient(ValueError())
