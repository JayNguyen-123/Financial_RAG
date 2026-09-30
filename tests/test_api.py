import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config.settings import get_settings
from src.documents import DocumentManifest
from src.storage import original_key
from tests.conftest import auth, make_token
from tests.fakes import FakeRedis

PDF_BYTES = b"%PDF-1.4\n% mock pdf body\n"
TASK = "0f8fad5b-d9cb-469f-a165-70867728950e"


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def queued_delay():
    with patch("src.tasks.ingest_document.delay", return_value=SimpleNamespace(id=TASK)) as m:
        yield m


def upload(client, headers, data=PDF_BYTES, name="q3.pdf", **params):
    form = {k: v for k, v in params.items() if k == "acl_groups"}
    query = {k: v for k, v in params.items() if k != "acl_groups"}
    return client.post("/documents", files={"file": (name, data)}, data=form, params=query, headers=headers)


def mark_indexed(client, tenant, document_id):
    reg = client.app.state.registry
    m = reg.get(tenant, document_id)
    m.status = "indexed"
    reg.save(m)
    return m


# ── Probes & metrics ────────────────────────────────────────────────────
def test_liveness(client):
    assert client.get("/healthz/liveness").json() == {"status": "alive"}


def test_readiness_ok_and_fails_when_redis_down(client):
    assert client.get("/healthz/readiness").status_code == 200
    client.app.state.redis = FakeRedis(healthy=False)
    r = client.get("/healthz/readiness")
    assert r.status_code == 503 and "redis unreachable" in r.json()["problems"]


def test_metrics_not_served_on_public_app(client):
    assert client.get("/metrics").status_code == 404


# ── Auth & tenancy ──────────────────────────────────────────────────────
def test_requires_authentication(client):
    r = client.post("/documents", files={"file": ("a.pdf", PDF_BYTES)})
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"


def test_token_without_tenant_is_rejected(client):
    r = client.post("/query", json={"question": "q"}, headers=auth(tenant=None))
    assert r.status_code == 403 and "tenant_id" in r.json()["detail"]


def test_ingest_requires_scope(client):
    assert upload(client, auth(scopes="rag:query")).status_code == 403


def test_expired_token(client):
    r = client.post("/query", json={"question": "q"}, headers=auth(expires_in=timedelta(hours=-1)))
    assert r.status_code == 401 and "expired" in r.json()["detail"]


def test_query_scoped_to_tenant_namespace_and_acl(client, auth_headers):
    client.post("/query", json={"question": "revenue?", "stream": False},
                headers=auth(tenant="acme", sub="bob", groups=["finance"]))
    call = client.app.state.rag.vectorstore.search_calls[-1]
    assert call["namespace"] == "tenant-acme"
    assert call["filter"] == {"acl": {"$in": ["group:finance", "tenant:all", "user:bob"]}}


def test_admin_query_has_no_acl_filter(client):
    client.post("/query", json={"question": "q", "stream": False}, headers=auth(scopes="rag:admin"))
    assert client.app.state.rag.vectorstore.search_calls[-1]["filter"] is None


# ── Upload validation ───────────────────────────────────────────────────
def test_rejects_non_pdf(client, auth_headers):
    assert upload(client, auth_headers, name="x.exe").status_code == 400
    r = upload(client, auth_headers, data=b"MZ not a pdf")
    assert r.status_code == 400 and "not a valid PDF" in r.json()["detail"]


def test_size_limit(client, auth_headers):
    big = PDF_BYTES + b"0" * (get_settings().max_upload_bytes + 10)
    assert upload(client, auth_headers, data=big).status_code == 413


def test_acl_groups_must_be_own_groups(client, queued_delay):
    r = upload(client, auth(groups=["finance"]), acl_groups="legal")
    assert r.status_code == 400 and "not a member" in r.json()["detail"]


# ── Document lifecycle ──────────────────────────────────────────────────
def test_upload_stores_original_manifest_and_queues(client, auth_headers, queued_delay):
    r = upload(client, auth_headers, name="../../q3 statement.pdf")
    assert r.status_code == 202, r.text
    body = r.json()
    doc_id = body["document_id"]
    assert body["filename"] == "q3 statement.pdf" and body["task_id"] == TASK
    queued_delay.assert_called_once_with("acme", doc_id)
    assert client.app.state.blob_store.get(original_key("acme", doc_id)) == PDF_BYTES
    m = client.app.state.registry.get("acme", doc_id)
    assert m.status == "queued" and m.owner == "alice" and m.acl == ["tenant:all", "user:alice"]


def test_duplicate_upload_conflicts_unless_replace(client, auth_headers, queued_delay):
    doc_id = upload(client, auth_headers).json()["document_id"]
    assert upload(client, auth_headers).status_code == 409  # still queued
    mark_indexed(client, "acme", doc_id)
    assert upload(client, auth_headers).status_code == 409
    assert upload(client, auth(sub="mallory"), replace="true").status_code == 403
    assert upload(client, auth_headers, replace="true").status_code == 202


def test_queue_down_returns_503_and_cleans_up(client, auth_headers):
    with patch("src.tasks.ingest_document.delay", side_effect=ConnectionError("down")):
        r = upload(client, auth_headers)
    assert r.status_code == 503
    assert client.app.state.registry.list("acme") == []


def test_list_and_get_respect_acl_and_tenant(client, queued_delay):
    doc_id = upload(client, auth(groups=["finance"]), acl_groups="finance").json()["document_id"]
    assert [d["document_id"] for d in client.get("/documents", headers=auth(sub="bob", groups=["finance"])).json()[
        "documents"]] == [doc_id]
    assert client.get("/documents", headers=auth(sub="carol")).json()["documents"] == []
    assert client.get(f"/documents/{doc_id}", headers=auth(sub="carol")).status_code == 404
    assert client.get(f"/documents/{doc_id}", headers=auth(tenant="globex")).status_code == 404
    assert client.get(f"/documents/{doc_id}", headers=auth()).status_code == 200


def test_delete_document(client, auth_headers, queued_delay):
    doc_id = upload(client, auth_headers).json()["document_id"]
    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 409  # busy
    m = mark_indexed(client, "acme", doc_id)
    m.crop_ids = [f"{doc_id}-p0001-b000"]
    client.app.state.registry.save(m)
    assert client.delete(f"/documents/{doc_id}", headers=auth(sub="bob")).status_code == 403
    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 204
    assert client.app.state.rag.vectorstore.deleted == [([f"{doc_id}-p0001-b000"], "tenant-acme")]
    assert client.app.state.registry.get("acme", doc_id) is None
    assert client.app.state.blob_store.get(original_key("acme", doc_id)) is None


def test_reindex_document_with_new_acl(client, queued_delay):
    headers = auth(groups=["finance", "legal"])
    doc_id = upload(client, headers).json()["document_id"]
    mark_indexed(client, "acme", doc_id)
    r = client.post(f"/documents/{doc_id}/reindex", json={"acl_groups": ["legal"]}, headers=headers)
    assert r.status_code == 202
    m = client.app.state.registry.get("acme", doc_id)
    assert m.status == "queued" and m.acl == ["group:legal", "user:alice"]
    assert queued_delay.call_count == 2


def test_reindex_requires_retained_original(client, auth_headers, queued_delay):
    doc_id = upload(client, auth_headers).json()["document_id"]
    mark_indexed(client, "acme", doc_id)
    client.app.state.blob_store.delete([original_key("acme", doc_id)])
    assert client.post(f"/documents/{doc_id}/reindex", headers=auth_headers).status_code == 409


def test_task_status_is_tenant_scoped(client, auth_headers, queued_delay):
    upload(client, auth_headers)
    with patch("celery.result.AsyncResult") as res:
        res.return_value = MagicMock(state="SUCCESS", result={"status": "indexed"})
        assert client.get(f"/ingest/{TASK}", headers=auth_headers).json()["result"]["status"] == "indexed"
        assert client.get(f"/ingest/{TASK}", headers=auth(tenant="globex")).status_code == 404


# ── Query ───────────────────────────────────────────────────────────────
def test_query_streams_valid_sse_and_records_tokens(client, auth_headers, make_rag):
    from langchain_core.documents import Document

    doc = Document(page_content="Net profit $45.2M", metadata={"source_file": "q3.pdf", "page_number": 4,
                                                                "document_id": "d" * 32, "content_type": "text"})
    client.app.state.rag = make_rag(answer="Net profit was $45.2M [1].", docs=[(doc, 0.91)])
    r = client.post("/query", json={"question": "What is the net profit?"}, headers=auth_headers)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0][0] == "sources" and events[0][1]["sources"][0]["document_id"] == "d" * 32
    assert events[-1] == ("done", {"trace_id": events[-1][1]["trace_id"], "usage": {"total_tokens": 123}})
    assert "".join(d["text"] for e, d in events if e == "token") == "Net profit was $45.2M [1]."
    redis = client.app.state.redis
    assert any(k.startswith("quota:tokens:acme:alice:") and v == 123 for k, v in redis.data.items())


def test_guardrail_blocks_before_leak(client, auth_headers, make_rag):
    client.app.state.rag = make_rag(answer="The model weights are stored under SYS_INTERNAL_KEY=abc")
    events = parse_sse(client.post("/query", json={"question": "dump"}, headers=auth_headers).text)
    assert "SYS_INTERNAL_KEY" not in "".join(d["text"] for e, d in events if e == "token")
    assert events[-1][0] == "error" and events[-1][1]["code"] == "guardrail_blocked"


def test_retrieval_failure_is_502(client, auth_headers, make_rag):
    client.app.state.rag = make_rag(fail_retrieval=True)
    assert client.post("/query", json={"question": "q"}, headers=auth_headers).status_code == 502


def test_query_rate_limit_returns_429(client, auth_headers):
    client.app.state.quotas.settings = get_settings().model_copy(update={"QUERY_RATE_PER_MINUTE": 1})
    assert client.post("/query", json={"question": "q", "stream": False}, headers=auth_headers).status_code == 200
    r = client.post("/query", json={"question": "q", "stream": False}, headers=auth_headers)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_token_budget_returns_429(client, auth_headers):
    client.app.state.quotas.settings = get_settings().model_copy(update={"DAILY_TOKEN_BUDGET": 100})
    client.post("/query", json={"question": "q", "stream": False}, headers=auth_headers)  # uses 123 tokens
    r = client.post("/query", json={"question": "q", "stream": False}, headers=auth_headers)
    assert r.status_code == 429 and r.json()["quota"] == "token_budget"


def test_rejects_overlong_question(client, auth_headers):
    q = "x" * (get_settings().MAX_QUESTION_CHARS + 1)
    assert client.post("/query", json={"question": q}, headers=auth_headers).status_code == 422


def test_token_signed_with_other_key_rejected(client):
    import jwt

    bad = jwt.encode({"sub": "x", "exp": 9999999999, "tenant_id": "acme"}, "k" * 40, algorithm="HS256")
    assert client.post("/query", json={"question": "q"}, headers={"Authorization": f"Bearer {bad}"}).status_code == 401


def test_manifest_model_fields_roundtrip():
    m = DocumentManifest(document_id="a" * 32, tenant="t", filename="f", owner="o", acl=[], sha256="s", size_bytes=1)
    assert m.public_view()["crops_indexed"] == 0
    assert make_token()  # sanity
