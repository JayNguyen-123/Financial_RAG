"""Shared fixtures. No network: OpenAI, Pinecone, Redis and S3 are all faked."""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta

import pytest

_TMP = tempfile.mkdtemp(prefix="rag-tests-")
os.environ.update(
    {
        "ENV": "test",
        "LOG_JSON": "false",
        "OPENAI_API_KEY": "sk-test",
        "PINECONE_API_KEY": "pc-test",
        "JWT_SECRET": "unit-test-secret-that-is-long-enough-1234567890",
        "JWT_INGEST_SCOPE": "rag:ingest",
        "LANGCHAIN_TRACING_V2": "false",
        "STORAGE_BACKEND": "local",
        "STORAGE_PATH": os.path.join(_TMP, "store"),
        "SCRATCH_DIR": os.path.join(_TMP, "scratch"),
        "MAX_UPLOAD_MB": "1",
        "REDIS_URL": "redis://localhost:6399/0",
        "METRICS_PORT": "0",
        "REQUIRE_TENANT": "true",
        "QUERY_RATE_PER_MINUTE": "1000",
        "INGEST_RATE_PER_HOUR": "1000",
        "DAILY_TOKEN_BUDGET": "0",
        "TEXT_EXTRACTION": "none",
    }
)
os.makedirs(os.environ["SCRATCH_DIR"], exist_ok=True)

import jwt  # noqa: E402

from config.settings import get_settings  # noqa: E402

get_settings.cache_clear()


def make_token(
    scopes: str | None = "rag:ingest",
    expires_in: timedelta = timedelta(hours=1),
    tenant: str | None = "acme",
    sub: str = "alice",
    groups: list[str] | None = None,
    **extra,
) -> str:
    s = get_settings()
    payload = {"sub": sub, "exp": datetime.now(UTC) + expires_in, **extra}
    if scopes is not None:
        payload["scope"] = scopes
    if tenant is not None:
        payload[s.TENANT_CLAIM] = tenant
    if groups is not None:
        payload[s.GROUPS_CLAIM] = groups
    if s.JWT_ISSUER:
        payload["iss"] = s.JWT_ISSUER
    if s.JWT_AUDIENCE:
        payload["aud"] = s.JWT_AUDIENCE
    return jwt.encode(payload, s.JWT_SECRET.get_secret_value(), algorithm=s.JWT_ALGORITHM)


def auth(**kwargs) -> dict:
    return {"Authorization": f"Bearer {make_token(**kwargs)}"}


@pytest.fixture
def blob_store(tmp_path):
    from src.storage import LocalBlobStore

    return LocalBlobStore(str(tmp_path / "blobs"))


@pytest.fixture
def make_rag(blob_store):
    from src.rag import RAGService
    from tests.fakes import FakeLLM, FakeVectorStore

    def _make(answer="Net profit was $45.2M [1].", docs=None, fail_retrieval=False):
        return RAGService(
            settings=get_settings(),
            vectorstore=FakeVectorStore(docs, fail=fail_retrieval),
            blob_store=blob_store,
            llm=FakeLLM(answer),
        )

    return _make


@pytest.fixture
def client(make_rag, blob_store):
    from fastapi.testclient import TestClient

    from src.documents import DocumentRegistry
    from src.main import app
    from src.quotas import QuotaService
    from tests.fakes import FakeRedis

    redis = FakeRedis()
    app.state.blob_store = blob_store
    app.state.registry = DocumentRegistry(blob_store)
    app.state.rag = make_rag()
    app.state.redis = redis
    app.state.quotas = QuotaService(redis, get_settings())
    with TestClient(app) as c:
        yield c
    for attr in ("blob_store", "registry", "rag", "redis", "quotas"):
        setattr(app.state, attr, None)


@pytest.fixture
def auth_headers():
    return auth()
