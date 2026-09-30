"""Document registry (manifests in blob storage) and lifecycle helpers.

A manifest is the source of truth for what was indexed for a document: its
ACL, owner, status and the exact vector/element ids. That makes deletion and
re-indexing exact (Pinecone serverless cannot delete by metadata filter), and
lets re-indexing remove vectors that no longer exist in the new run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from src.storage import BlobStore, image_key, manifest_key, manifest_prefix, original_key

logger = logging.getLogger(__name__)

Status = Literal["queued", "processing", "indexed", "failed"]
BUSY_STATUSES = {"queued", "processing"}


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class DocumentManifest:
    document_id: str
    tenant: str
    filename: str
    owner: str
    acl: list[str]
    sha256: str
    size_bytes: int
    status: Status = "queued"
    task_id: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    indexed_at: str | None = None
    pages_processed: int = 0
    crop_ids: list[str] = field(default_factory=list)
    text_chunk_ids: list[str] = field(default_factory=list)
    original_retained: bool = True
    error: str | None = None

    @property
    def vector_ids(self) -> list[str]:
        return [*self.crop_ids, *self.text_chunk_ids]

    def public_view(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("crop_ids")
        data.pop("text_chunk_ids")
        data["crops_indexed"] = len(self.crop_ids)
        data["text_chunks_indexed"] = len(self.text_chunk_ids)
        return data

    @classmethod
    def from_json(cls, raw: bytes) -> DocumentManifest:
        data = json.loads(raw)
        known = {f for f in cls.__dataclass_fields__}  # tolerate fields added later
        return cls(**{k: v for k, v in data.items() if k in known})


class DocumentRegistry:
    def __init__(self, store: BlobStore):
        self.store = store

    def save(self, m: DocumentManifest) -> None:
        m.updated_at = utcnow()
        self.store.put(manifest_key(m.tenant, m.document_id), json.dumps(asdict(m)).encode(), "application/json")

    def get(self, tenant: str, document_id: str) -> DocumentManifest | None:
        raw = self.store.get(manifest_key(tenant, document_id))
        return DocumentManifest.from_json(raw) if raw else None

    def list(self, tenant: str) -> list[DocumentManifest]:
        out = []
        for key in self.store.list(manifest_prefix(tenant)):
            if key.endswith(".json"):
                raw = self.store.get(key)
                if raw:
                    out.append(DocumentManifest.from_json(raw))
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    def delete_manifest(self, tenant: str, document_id: str) -> None:
        self.store.delete([manifest_key(tenant, document_id)])


def purge_document(m: DocumentManifest, store: BlobStore, vectorstore: Any, namespace: str,
                   registry: DocumentRegistry) -> None:
    """Remove every artefact of a document: vectors, crops, original, manifest.

    Order matters: vectors first (so nothing retrievable points at missing
    crops), manifest last (so a failed purge can be retried from it).
    """
    ids = m.vector_ids
    for i in range(0, len(ids), 1000):  # Pinecone delete-by-id batch limit
        vectorstore.delete(ids=ids[i:i + 1000], namespace=namespace)
    store.delete([image_key(m.tenant, cid) for cid in m.crop_ids])
    store.delete([original_key(m.tenant, m.document_id)])
    registry.delete_manifest(m.tenant, m.document_id)
    logger.info("Document purged", extra={"document_id": m.document_id, "tenant": m.tenant, "vectors": len(ids)})
