from src.documents import DocumentManifest, DocumentRegistry, purge_document
from src.storage import image_key, original_key
from tests.fakes import FakeVectorStore


def _manifest(**kw):
    base = dict(document_id="a" * 32, tenant="acme", filename="q3.pdf", owner="alice",
                acl=["tenant:all", "user:alice"], sha256="a" * 64, size_bytes=10)
    return DocumentManifest(**{**base, **kw})


def test_registry_roundtrip_and_tenant_scoping(blob_store):
    reg = DocumentRegistry(blob_store)
    reg.save(_manifest())
    reg.save(_manifest(document_id="b" * 32, tenant="other"))
    got = reg.get("acme", "a" * 32)
    assert got.filename == "q3.pdf" and got.status == "queued"
    assert [m.document_id for m in reg.list("acme")] == ["a" * 32]
    assert reg.get("acme", "b" * 32) is None


def test_public_view_hides_ids():
    view = _manifest(crop_ids=["x"], text_chunk_ids=["y", "z"]).public_view()
    assert "crop_ids" not in view and view["crops_indexed"] == 1 and view["text_chunks_indexed"] == 2


def test_purge_removes_everything(blob_store):
    reg = DocumentRegistry(blob_store)
    m = _manifest(crop_ids=["a" * 32 + "-p0001-b000"], text_chunk_ids=["a" * 32 + "-p0001-t000"], status="indexed")
    reg.save(m)
    blob_store.put(image_key("acme", m.crop_ids[0]), b"png")
    blob_store.put(original_key("acme", m.document_id), b"%PDF")
    vs = FakeVectorStore()
    purge_document(m, blob_store, vs, "tenant-acme", reg)
    assert vs.deleted == [(m.vector_ids, "tenant-acme")]
    assert blob_store.get(image_key("acme", m.crop_ids[0])) is None
    assert blob_store.get(original_key("acme", m.document_id)) is None
    assert reg.get("acme", m.document_id) is None


def test_manifest_tolerates_unknown_fields():
    import json

    raw = json.dumps({**_manifest().__dict__, "future_field": 1}).encode()
    assert DocumentManifest.from_json(raw).document_id == "a" * 32
