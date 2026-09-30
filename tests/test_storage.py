import pytest

from src.storage import LocalBlobStore, image_key, manifest_key, original_key, validate_key


def test_put_get_list_delete_roundtrip(blob_store):
    blob_store.put(image_key("acme", "d1-p0001-b000"), b"png")
    blob_store.put(manifest_key("acme", "d1"), b"{}")
    blob_store.put(manifest_key("other", "d2"), b"{}")
    assert blob_store.get(image_key("acme", "d1-p0001-b000")) == b"png"
    assert blob_store.get("images/acme/missing.png") is None
    assert list(blob_store.list("manifests/acme/")) == ["manifests/acme/d1.json"]
    blob_store.delete([manifest_key("acme", "d1"), "manifests/acme/never-existed.json"])
    assert list(blob_store.list("manifests/acme/")) == []


def test_put_file_and_download(blob_store, tmp_path):
    src = tmp_path / "in.pdf"
    src.write_bytes(b"%PDF-1.4 data")
    blob_store.put_file(original_key("acme", "d1"), str(src))
    out = tmp_path / "out.pdf"
    assert blob_store.download_to(original_key("acme", "d1"), str(out))
    assert out.read_bytes() == b"%PDF-1.4 data"
    assert not blob_store.download_to(original_key("acme", "nope"), str(out))


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/../../b", "/abs", "a//b", "", "a b"])
def test_rejects_unsafe_keys(bad):
    with pytest.raises(ValueError):
        validate_key(bad)


def test_no_temp_files_left_behind(blob_store):
    blob_store.put("images/acme/x.png", b"1")
    assert list(blob_store.list("images/")) == ["images/acme/x.png"]
    assert blob_store.healthy()


def test_s3_backend_with_moto():
    moto = pytest.importorskip("moto")
    import boto3

    from src.storage import S3BlobStore

    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="rag")
        store = S3BlobStore("rag", prefix="prod", client=client)
        store.put("manifests/acme/d1.json", b"{}")
        assert store.get("manifests/acme/d1.json") == b"{}"
        assert store.get("manifests/acme/none.json") is None
        assert list(store.list("manifests/acme/")) == ["manifests/acme/d1.json"]
        store.delete(["manifests/acme/d1.json"])
        assert list(store.list("manifests/")) == []
        assert store.healthy()


def test_local_store_confines_to_root(tmp_path):
    store = LocalBlobStore(str(tmp_path / "root"))
    with pytest.raises(ValueError):
        store.put("../escape.txt", b"x")
