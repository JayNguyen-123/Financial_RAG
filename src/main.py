"""FastAPI gateway: tenant-isolated document lifecycle + streaming multimodal RAG."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import tempfile
import unicodedata
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from src.documents import BUSY_STATUSES, DocumentManifest, DocumentRegistry, purge_document
from src.guardrails import GuardrailViolation, StreamingGuard
from src.logging_config import configure_logging, request_id_ctx
from src.metrics import (
    DOCUMENT_EVENTS,
    GUARDRAIL_BLOCKS,
    LLM_TOKENS,
    QUOTA_REJECTIONS,
    instrument_app,
    start_metrics_server,
)
from src.quotas import QuotaBackendUnavailable, QuotaExceeded, QuotaService
from src.rag import RAGService
from src.security import get_ingest_principal, get_principal
from src.sse import SSE_HEADERS, sse_event
from src.storage import BlobStore, build_blob_store, original_key
from src.tenancy import Principal, TenancyError, build_acl, namespace_for

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
CHUNK_SIZE = 1024 * 1024
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
TASK_ID_RE = re.compile(r"^[0-9a-fA-F\-]{36}$")
DOC_ID_PATTERN = r"^[0-9a-f]{32}$"
TASK_TENANT_TTL = 7 * 86400


# ────────────────────────────────────────────────────────────────────────
# Lifespan: build clients once, after config is validated, not at import.
# Anything pre-set on app.state (tests) is left alone.
# ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)
    settings.export_langsmith_env()
    os.makedirs(settings.SCRATCH_DIR, exist_ok=True)
    start_metrics_server(settings.METRICS_PORT)

    state = app.state
    if getattr(state, "blob_store", None) is None:
        state.blob_store = build_blob_store(settings)
    if getattr(state, "registry", None) is None:
        state.registry = DocumentRegistry(state.blob_store)
    if getattr(state, "rag", None) is None:
        state.rag = RAGService.from_settings(settings, blob_store=state.blob_store)
    if getattr(state, "redis", None) is None:
        import redis.asyncio as aioredis

        state.redis = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
    if getattr(state, "quotas", None) is None:
        state.quotas = QuotaService(state.redis, settings)

    logger.info("API started", extra={"environment": settings.ENVIRONMENT, "storage": settings.STORAGE_BACKEND})
    try:
        yield
    finally:
        redis_client = getattr(state, "redis", None)
        if redis_client is not None and hasattr(redis_client, "aclose"):
            await redis_client.aclose()


app = FastAPI(title="Multimodal Financial RAG Core", version="1.2.0", lifespan=lifespan)
instrument_app(app)  # metrics are recorded here but served on METRICS_PORT, not on this app


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    rid = incoming if REQUEST_ID_RE.match(incoming) else uuid4().hex
    token = request_id_ctx.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(QuotaExceeded)
async def _quota_handler(_: Request, exc: QuotaExceeded):
    QUOTA_REJECTIONS.labels(exc.kind).inc()
    return JSONResponse(
        status_code=429,
        content={"detail": exc.message, "quota": exc.kind},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(QuotaBackendUnavailable)
async def _quota_backend_handler(_: Request, __: QuotaBackendUnavailable):
    return JSONResponse(status_code=503, content={"detail": "Quota service unavailable. Retry later."})


# ── Dependencies (overridable in tests) ─────────────────────────────────
def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag


def get_blob_store_dep(request: Request) -> BlobStore:
    return request.app.state.blob_store


def get_registry_dep(request: Request) -> DocumentRegistry:
    return request.app.state.registry


def get_quotas(request: Request) -> QuotaService:
    return request.app.state.quotas


# ────────────────────────────────────────────────────────────────────────
# Schemas
# ────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=20000)
    stream: bool = True


class IngestResponse(BaseModel):
    status: str
    document_id: str
    task_id: str
    filename: str


class ReindexRequest(BaseModel):
    acl_groups: List[str] | None = Field(default=None, description="Replace the document ACL; omit to keep it")


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str
    result: Dict[str, Any] | None = None
    error: str | None = None


# ────────────────────────────────────────────────────────────────────────
# Upload helpers
# ────────────────────────────────────────────────────────────────────────
def sanitize_filename(name: str | None) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    base = unicodedata.normalize("NFKC", base)
    base = "".join(ch for ch in base if ch.isprintable())
    return base[:255] or "upload.pdf"


def _parse_groups(raw: str | None) -> List[str]:
    return [g for g in re.split(r"[,\s]+", raw or "") if g]


async def _spool_upload(file: UploadFile, scratch_dir: str, max_bytes: int) -> tuple[str, str, int]:
    """Stream upload to node-local scratch, enforcing size and PDF signature.

    Returns (path, sha256_hex, size). Caller must delete the path.
    """
    fd, path = tempfile.mkstemp(dir=scratch_dir, suffix=".pdf")
    fh = os.fdopen(fd, "wb")
    digest = hashlib.sha256()
    written = 0
    try:
        first = True
        while chunk := await file.read(CHUNK_SIZE):
            if first:
                # The PDF spec allows the header anywhere in the first 1024 bytes.
                if PDF_MAGIC not in chunk[:1024]:
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, "File content is not a valid PDF.")
                first = False
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"File exceeds maximum upload size of {max_bytes // (1024 * 1024)} MB.",
                )
            digest.update(chunk)
            await run_in_threadpool(fh.write, chunk)
        if written == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    except BaseException:
        fh.close()
        _silent_remove(path)
        raise
    fh.close()
    return path, digest.hexdigest(), written


def _silent_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


async def _enqueue(request: Request, manifest: DocumentManifest) -> str:
    from src.tasks import ingest_document

    task = ingest_document.delay(manifest.tenant, manifest.document_id)
    try:
        await request.app.state.redis.set(f"task-tenant:{task.id}", manifest.tenant, ex=TASK_TENANT_TTL)
    except Exception:
        logger.warning("Could not record task tenant mapping", exc_info=True)
    return task.id


def _visible_or_404(m: DocumentManifest | None, principal: Principal) -> DocumentManifest:
    # 404 (not 403) for invisible documents so ids cannot be probed.
    if m is None or not principal.can_see(m.acl, m.owner):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return m


# ────────────────────────────────────────────────────────────────────────
# Documents
# ────────────────────────────────────────────────────────────────────────
@app.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=IngestResponse)
@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED, response_model=IngestResponse, include_in_schema=False)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    acl_groups: str | None = Form(default=None, description="Comma-separated groups; empty = whole tenant"),
    replace: bool = Query(default=False, description="Re-index if this exact file already exists"),
    principal: Principal = Depends(get_ingest_principal),
    settings: Settings = Depends(get_settings),
    store: BlobStore = Depends(get_blob_store_dep),
    registry: DocumentRegistry = Depends(get_registry_dep),
    quotas: QuotaService = Depends(get_quotas),
):
    """Upload a scanned PDF and queue it for ingestion into the caller's tenant."""
    await quotas.check_ingest(principal.tenant, principal.sub)

    filename = sanitize_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only .pdf uploads are supported.")
    try:
        acl = build_acl(principal, _parse_groups(acl_groups), settings)
    except TenancyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    path, sha256, size = await _spool_upload(file, settings.SCRATCH_DIR, settings.max_upload_bytes)
    try:
        document_id = sha256[:32]  # content-addressed: same file => same document
        existing = await run_in_threadpool(registry.get, principal.tenant, document_id)
        if existing is not None:
            if existing.status in BUSY_STATUSES:
                raise HTTPException(status.HTTP_409_CONFLICT, f"Document {document_id} is already being processed.")
            if not replace:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"Document {document_id} already exists. Use ?replace=true to re-index it.",
                )
            if not principal.can_manage(existing.owner):
                raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the owner or an admin can replace this document.")

        await run_in_threadpool(store.put_file, original_key(principal.tenant, document_id), path, "application/pdf")
    finally:
        _silent_remove(path)

    manifest = existing or DocumentManifest(
        document_id=document_id, tenant=principal.tenant, filename=filename, owner=principal.sub,
        acl=acl, sha256=sha256, size_bytes=size,
    )
    manifest.filename, manifest.acl, manifest.status, manifest.error = filename, acl, "queued", None
    manifest.original_retained = True
    await run_in_threadpool(registry.save, manifest)

    try:
        manifest.task_id = await _enqueue(request, manifest)
    except Exception as exc:
        logger.exception("Failed to enqueue ingestion task")
        if existing is None:
            await run_in_threadpool(store.delete, [original_key(principal.tenant, document_id)])
            await run_in_threadpool(registry.delete_manifest, principal.tenant, document_id)
        else:
            manifest.status, manifest.error = "failed", "QueueUnavailable"
            await run_in_threadpool(registry.save, manifest)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Ingestion queue unavailable. Retry later.") from exc
    await run_in_threadpool(registry.save, manifest)

    DOCUMENT_EVENTS.labels("queued").inc()
    logger.info("Ingestion queued", extra={"document_id": document_id, "tenant": principal.tenant,
                                           "task_id": manifest.task_id, "bytes": size, "sub": principal.sub})
    return IngestResponse(status="queued", document_id=document_id, task_id=manifest.task_id, filename=filename)


@app.get("/documents")
async def list_documents(
    principal: Principal = Depends(get_principal),
    registry: DocumentRegistry = Depends(get_registry_dep),
):
    docs = await run_in_threadpool(registry.list, principal.tenant)
    return {"documents": [m.public_view() for m in docs if principal.can_see(m.acl, m.owner)]}


@app.get("/documents/{document_id}")
async def get_document(
    document_id: str = Path(..., pattern=DOC_ID_PATTERN),
    principal: Principal = Depends(get_principal),
    registry: DocumentRegistry = Depends(get_registry_dep),
):
    m = _visible_or_404(await run_in_threadpool(registry.get, principal.tenant, document_id), principal)
    return m.public_view()


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str = Path(..., pattern=DOC_ID_PATTERN),
    principal: Principal = Depends(get_ingest_principal),
    settings: Settings = Depends(get_settings),
    store: BlobStore = Depends(get_blob_store_dep),
    registry: DocumentRegistry = Depends(get_registry_dep),
    rag: RAGService = Depends(get_rag_service),
):
    m = _visible_or_404(await run_in_threadpool(registry.get, principal.tenant, document_id), principal)
    if not principal.can_manage(m.owner):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the owner or an admin can delete this document.")
    if m.status in BUSY_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is being processed; retry when it finishes.")
    try:
        await run_in_threadpool(
            purge_document, m, store, rag.vectorstore, namespace_for(principal.tenant, settings), registry
        )
    except Exception as exc:
        logger.exception("Document purge failed", extra={"document_id": document_id})
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Deletion failed; it is safe to retry.") from exc
    DOCUMENT_EVENTS.labels("deleted").inc()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/documents/{document_id}/reindex", status_code=status.HTTP_202_ACCEPTED, response_model=IngestResponse)
async def reindex_document(
    request: Request,
    body: ReindexRequest | None = None,
    document_id: str = Path(..., pattern=DOC_ID_PATTERN),
    principal: Principal = Depends(get_ingest_principal),
    settings: Settings = Depends(get_settings),
    store: BlobStore = Depends(get_blob_store_dep),
    registry: DocumentRegistry = Depends(get_registry_dep),
    quotas: QuotaService = Depends(get_quotas),
):
    """Re-run ingestion from the retained original (e.g. after a model/prompt upgrade or ACL change)."""
    m = _visible_or_404(await run_in_threadpool(registry.get, principal.tenant, document_id), principal)
    if not principal.can_manage(m.owner):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the owner or an admin can re-index this document.")
    if m.status in BUSY_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is already being processed.")
    if not await run_in_threadpool(store.exists, original_key(principal.tenant, document_id)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Original PDF was not retained; upload it again.")
    await quotas.check_ingest(principal.tenant, principal.sub)

    if body is not None and body.acl_groups is not None:
        try:
            m.acl = build_acl(principal, body.acl_groups, settings)
        except TenancyError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    m.status, m.error = "queued", None
    await run_in_threadpool(registry.save, m)
    try:
        m.task_id = await _enqueue(request, m)
    except Exception as exc:
        m.status, m.error = "failed", "QueueUnavailable"
        await run_in_threadpool(registry.save, m)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Ingestion queue unavailable. Retry later.") from exc
    await run_in_threadpool(registry.save, m)
    DOCUMENT_EVENTS.labels("reindex_queued").inc()
    return IngestResponse(status="queued", document_id=document_id, task_id=m.task_id, filename=m.filename)


@app.get("/ingest/{task_id}", response_model=TaskStatusResponse)
async def ingestion_status(request: Request, task_id: str, principal: Principal = Depends(get_principal)):
    if not TASK_ID_RE.match(task_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Malformed task id.")
    owner_tenant = await request.app.state.redis.get(f"task-tenant:{task_id}")
    if isinstance(owner_tenant, bytes):
        owner_tenant = owner_tenant.decode()
    if owner_tenant != principal.tenant:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found.")

    from celery.result import AsyncResult

    from src.tasks import celery_app

    res = AsyncResult(task_id, app=celery_app)
    state = await run_in_threadpool(lambda: res.state)
    body = TaskStatusResponse(task_id=task_id, state=state)
    if state == "SUCCESS":
        body.result = res.result if isinstance(res.result, dict) else {"value": str(res.result)}
    elif state == "FAILURE":
        body.error = type(res.result).__name__ if res.result is not None else "Unknown error"
    return body


# ────────────────────────────────────────────────────────────────────────
# Query
# ────────────────────────────────────────────────────────────────────────
async def stream_answer(
    rag: RAGService, messages: List[Any], sources: List[Any], trace_id: str,
    quotas: QuotaService, principal: Principal,
) -> AsyncIterator[str]:
    yield sse_event("sources", {"sources": [s.to_dict() for s in sources]})
    guard = StreamingGuard(policy=rag.policy)
    usage: Dict[str, int] = {}
    try:
        async for token in rag.stream_tokens(messages, trace_id, usage):
            safe = guard.feed(token)
            if safe:
                yield sse_event("token", {"text": safe})
        tail = guard.flush()
        if tail:
            yield sse_event("token", {"text": tail})
        yield sse_event("done", {"trace_id": trace_id, "usage": usage})
    except GuardrailViolation as gv:
        GUARDRAIL_BLOCKS.labels(gv.rule).inc()
        logger.warning("Guardrail blocked streamed answer", extra={"rule": gv.rule, "trace_id": trace_id})
        yield sse_event("error", {"code": "guardrail_blocked", "rule": gv.rule, "message": gv.message})
    except asyncio.CancelledError:
        logger.info("Client disconnected mid-stream", extra={"trace_id": trace_id})
        raise
    except Exception:
        logger.exception("Generation failed mid-stream", extra={"trace_id": trace_id})
        yield sse_event("error", {"code": "generation_failed", "message": "Answer generation failed."})
    finally:
        # Usage arrives in the final chunk; if the stream was cut short, estimate
        # from emitted text (~4 chars/token) so aborted streams still count.
        tokens = usage.get("total_tokens") or len(guard.text) // 4
        LLM_TOKENS.labels("query").inc(tokens)
        await asyncio.shield(quotas.record_tokens(principal.tenant, principal.sub, tokens))


@app.post("/query")
async def execute_query(
    payload: QueryRequest,
    principal: Principal = Depends(get_principal),
    rag: RAGService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
    quotas: QuotaService = Depends(get_quotas),
):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Question must not be blank.")
    if len(question) > settings.MAX_QUESTION_CHARS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Question exceeds {settings.MAX_QUESTION_CHARS} characters.",
        )
    await quotas.check_query(principal.tenant, principal.sub)
    trace_id = request_id_ctx.get()

    if not payload.stream:
        try:
            result = await rag.answer(question, principal, trace_id)
        except GuardrailViolation as gv:
            GUARDRAIL_BLOCKS.labels(gv.rule).inc()
            return JSONResponse(status_code=403, content={"detail": gv.message, "rule": gv.rule})
        tokens = result["usage"]["total_tokens"]
        LLM_TOKENS.labels("query").inc(tokens)
        await quotas.record_tokens(principal.tenant, principal.sub, tokens)
        return result

    # Retrieval runs *before* the response starts so failures map to real HTTP
    # status codes instead of a broken 200 stream.
    try:
        hits = await rag.retrieve(question, principal)
        messages, sources = await rag.build_messages(question, hits)
    except Exception as exc:
        logger.exception("Retrieval failed", extra={"trace_id": trace_id})
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Retrieval backend unavailable.") from exc

    return StreamingResponse(
        stream_answer(rag, messages, sources, trace_id, quotas, principal),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ────────────────────────────────────────────────────────────────────────
# Kubernetes probes
# ────────────────────────────────────────────────────────────────────────
@app.get("/healthz/liveness", include_in_schema=False)
async def liveness_probe():
    return {"status": "alive"}


def _is_writable_dir(path: str) -> bool:
    return os.path.isdir(path) and os.access(path, os.W_OK)


@app.get("/healthz/readiness", include_in_schema=False)
async def readiness_probe(request: Request, settings: Settings = Depends(get_settings)):
    problems: List[str] = []
    if not await run_in_threadpool(_is_writable_dir, settings.SCRATCH_DIR):
        problems.append("scratch dir not writable")
    if not await run_in_threadpool(request.app.state.blob_store.healthy):
        problems.append("blob storage unavailable")
    try:
        await asyncio.wait_for(request.app.state.redis.ping(), timeout=2)
    except Exception:
        problems.append("redis unreachable")
    # Pinecone/OpenAI are deliberately NOT probed: an external outage should
    # surface as 502s, not pull every pod out of the Service at once.
    if problems:
        logger.warning("Readiness failed", extra={"problems": problems})
        return JSONResponse(status_code=503, content={"status": "unready", "problems": problems})
    return {"status": "ready"}
