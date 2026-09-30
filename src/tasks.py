"""Celery worker: out-of-process multimodal ingestion pipeline.

Pipeline per document (tenant-scoped, idempotent, re-runnable):
  1. load manifest (ACL/owner) -> status=processing
  2. download original from blob storage to node-local scratch
  3. per page: detect tables/figures, crop them, OCR the remaining narrative text
  4. GPT-4o summaries of crops; chunk page text
  5. store crops, upsert vectors into the tenant namespace (deterministic ids)
  6. delete vectors/crops from a previous index run that no longer exist
  7. record LLM token usage against the uploader's daily budget
  8. manifest -> status=indexed (or failed)

Reliability properties kept from v1.1: failures raise (Celery records FAILURE),
transient OpenAI/network errors retry with backoff, acks_late + prefetch=1,
per-process model cache, and a worker only ever receives opaque ids.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import after_setup_logger, worker_process_init

from config.settings import get_settings
from src.logging_config import configure_logging
from src.tenancy import TENANT_RE, namespace_for

logger = logging.getLogger(__name__)
settings = get_settings()

ID_KEY = "doc_id"
DOCUMENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")

celery_app = Celery("rag_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    result_expires=settings.CELERY_RESULT_EXPIRES,
    broker_transport_options={"visibility_timeout": settings.CELERY_TASK_TIME_LIMIT + 300},
    broker_connection_retry_on_startup=True,
    # Queue name is referenced by the KEDA ScaledObject (listName).
    task_default_queue="ingestion",
)


class IngestionError(Exception):
    """Terminal (non-retryable) ingestion failure."""


@after_setup_logger.connect
def _setup_logging(**_: Any) -> None:
    configure_logging(settings.LOG_LEVEL, settings.LOG_JSON)


@worker_process_init.connect
def _init_worker(**_: Any) -> None:
    settings.export_langsmith_env()


# ── Per-process lazily built resources ──────────────────────────────────
@lru_cache(maxsize=1)
def get_layout_parser():
    from src.layout_pipeline import FinancialLayoutParser

    return FinancialLayoutParser(settings)


@lru_cache(maxsize=1)
def get_text_extractor():
    from src.text_extraction import build_text_extractor

    return build_text_extractor(settings)


@lru_cache(maxsize=1)
def get_vectorstore():
    from src.clients import build_vectorstore

    return build_vectorstore(settings)


@lru_cache(maxsize=1)
def get_blob_store():
    from src.storage import build_blob_store

    return build_blob_store(settings)


@lru_cache(maxsize=1)
def get_registry():
    from src.documents import DocumentRegistry

    return DocumentRegistry(get_blob_store())


@lru_cache(maxsize=1)
def get_sync_redis():
    import redis

    return redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2)


# ── Pure helpers (unit-tested) ──────────────────────────────────────────
def crop_element_id(document_id: str, page_number: int, block_index: int) -> str:
    return f"{document_id}-p{page_number:04d}-b{block_index:03d}"


def is_crop_id(element_id: str) -> bool:
    return element_id.rsplit("-", 1)[-1].startswith("b")


def text_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    return f"{document_id}-p{page_number:04d}-t{chunk_index:03d}"


def crop_metadata(element: dict[str, Any], element_id: str, manifest: Any, img_key: str) -> dict[str, Any]:
    """Pinecone-safe metadata: scalars and lists of strings only."""
    x1, y1, x2, y2 = element["bbox"]
    return {
        ID_KEY: element_id,
        "image_key": img_key,
        "document_id": manifest.document_id,
        "tenant": manifest.tenant,
        "acl": list(manifest.acl),
        "content_type": f"{element['type']}_crop",
        "source_file": manifest.filename,
        "file_sha256": manifest.sha256,
        "page_number": int(element["page_number"]),
        "bbox": f"{x1},{y1},{x2},{y2}",
        "render_dpi": int(element.get("dpi", settings.PDF_RENDER_DPI)),
        "detection_score": float(element.get("score", 0.0)),
    }


def text_metadata(chunk_id: str, page_number: int, chunk_index: int, manifest: Any) -> dict[str, Any]:
    return {
        ID_KEY: chunk_id,
        "document_id": manifest.document_id,
        "tenant": manifest.tenant,
        "acl": list(manifest.acl),
        "content_type": "text",
        "source_file": manifest.filename,
        "file_sha256": manifest.sha256,
        "page_number": int(page_number),
        "chunk_index": int(chunk_index),
    }


def is_transient(exc: BaseException) -> bool:
    try:
        import openai

        if isinstance(
            exc,
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError),
        ):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, (ConnectionError, TimeoutError))


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Failed to delete scratch file", extra={"path": path}, exc_info=True)


def _delete_vectors_and_crops(tenant: str, ids: list[str]) -> None:
    from src.storage import image_key

    if not ids:
        return
    try:
        get_vectorstore().delete(ids=ids, namespace=namespace_for(tenant, settings))
    except Exception:  # pragma: no cover
        logger.warning("Failed to delete vectors", exc_info=True)
    crop_keys = [image_key(tenant, i) for i in ids if is_crop_id(i)]
    try:
        get_blob_store().delete(crop_keys)
    except Exception:  # pragma: no cover
        logger.warning("Failed to delete crops", exc_info=True)


# ── Task ────────────────────────────────────────────────────────────────
@celery_app.task(bind=True, name="src.tasks.ingest_document", max_retries=5)
def ingest_document(self, tenant: str, document_id: str) -> dict[str, Any]:
    if not TENANT_RE.match(tenant or "") or not DOCUMENT_ID_RE.match(document_id or ""):
        raise IngestionError("Invalid tenant or document id")

    from langchain_core.documents import Document

    from src.documents import utcnow
    from src.ingestion_pipeline import summarize_images
    from src.quotas import record_tokens_sync
    from src.storage import image_key, original_key
    from src.text_extraction import chunk_pages

    log_ctx = {"task_id": self.request.id, "tenant": tenant, "document_id": document_id}
    registry, store = get_registry(), get_blob_store()
    manifest = registry.get(tenant, document_id)
    if manifest is None:
        logger.info("Manifest gone before processing (deleted); skipping", extra=log_ctx)
        return {"status": "cancelled", "document_id": document_id}

    namespace = namespace_for(tenant, settings)
    previous_ids = set(manifest.vector_ids)
    manifest.status, manifest.task_id, manifest.error = "processing", self.request.id, None
    registry.save(manifest)

    os.makedirs(settings.SCRATCH_DIR, exist_ok=True)
    scratch = os.path.join(settings.SCRATCH_DIR, f"{document_id}-{self.request.id}.pdf")
    new_ids: list[str] = []
    try:
        if not store.download_to(original_key(tenant, document_id), scratch):
            raise IngestionError("Original PDF not found in storage")
        logger.info("Ingestion started", extra=log_ctx)

        parsed = get_layout_parser().process_pdf(scratch, get_text_extractor())
        summaries, summary_tokens = summarize_images([e["png_bytes"] for e in parsed.elements])

        docs: list[Any] = []
        crop_ids: list[str] = []
        images: list[tuple[str, bytes]] = []
        for element, summary in zip(parsed.elements, summaries, strict=True):
            if not summary or not summary.strip():
                logger.warning("Empty summary; skipping crop", extra={**log_ctx, "page": element["page_number"]})
                continue
            eid = crop_element_id(document_id, int(element["page_number"]), int(element["block_index"]))
            key = image_key(tenant, eid)
            crop_ids.append(eid)
            images.append((key, element["png_bytes"]))
            docs.append(Document(page_content=summary, metadata=crop_metadata(element, eid, manifest, key)))

        text_ids: list[str] = []
        for page_number, chunk_index, chunk in chunk_pages(parsed.page_texts, settings):
            tid = text_chunk_id(document_id, page_number, chunk_index)
            text_ids.append(tid)
            docs.append(Document(page_content=chunk, metadata=text_metadata(tid, page_number, chunk_index, manifest)))

        all_ids = crop_ids + text_ids
        new_ids = [i for i in all_ids if i not in previous_ids]
        for key, png in images:
            store.put(key, png, "image/png")
        if docs:
            # Deterministic ids => idempotent across retries and re-indexing.
            get_vectorstore().add_documents(docs, ids=all_ids, namespace=namespace)

        stale = sorted(previous_ids - set(all_ids))
        _delete_vectors_and_crops(tenant, stale)

        record_tokens_sync(get_sync_redis(), settings, tenant, manifest.owner,
                           summary_tokens + parsed.llm_tokens)

        # The document may have been deleted while we were working.
        if registry.get(tenant, document_id) is None:
            _delete_vectors_and_crops(tenant, all_ids)
            logger.info("Document deleted during processing; cleaned up", extra=log_ctx)
            return {"status": "cancelled", "document_id": document_id}

        manifest.crop_ids, manifest.text_chunk_ids = crop_ids, text_ids
        manifest.pages_processed = parsed.pages_processed
        manifest.status, manifest.indexed_at = "indexed", utcnow()
        if not settings.RETAIN_ORIGINALS:
            store.delete([original_key(tenant, document_id)])
            manifest.original_retained = False
        registry.save(manifest)

        result = {
            "status": "indexed",
            "document_id": document_id,
            "crops_indexed": len(crop_ids),
            "text_chunks_indexed": len(text_ids),
            "stale_removed": len(stale),
            "pages_processed": parsed.pages_processed,
        }
        logger.info("Ingestion finished", extra={**log_ctx, **result})
        return result

    except SoftTimeLimitExceeded:
        logger.error("Ingestion exceeded soft time limit", extra=log_ctx)
        _fail(manifest, "TimeLimitExceeded", new_ids)
        raise
    except Exception as exc:
        if is_transient(exc) and self.request.retries < self.max_retries:
            logger.warning("Transient failure; retrying", extra={**log_ctx, "error": repr(exc)})
            raise self.retry(exc=exc, countdown=min(600, 2 ** (self.request.retries + 2))) from exc
        logger.exception("Ingestion failed", extra=log_ctx)
        _fail(manifest, type(exc).__name__, new_ids)
        raise
    finally:
        _remove_quietly(scratch)


def _fail(manifest: Any, error: str, new_ids: list[str]) -> None:
    """Mark failed and remove only vectors created by this run (earlier index stays intact)."""
    _delete_vectors_and_crops(manifest.tenant, new_ids)
    try:
        if get_registry().get(manifest.tenant, manifest.document_id) is not None:
            manifest.status, manifest.error = "failed", error
            get_registry().save(manifest)
    except Exception:  # pragma: no cover
        logger.warning("Failed to update manifest status", exc_info=True)
