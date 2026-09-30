"""Centralised, validated application configuration.

All values come from environment variables (or a local ``.env`` file in
development). Secrets are wrapped in ``SecretStr`` so they never appear in
logs or reprs. Use :func:`get_settings` instead of instantiating directly so
the object is built lazily (tests can set env vars first) and cached.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_WEAK_SECRET_MARKERS = ("dev-", "change-me", "changeme", "secret", "xxxx")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── 1. Operational environment ──────────────────────────────────────
    ENVIRONMENT: Literal["development", "test", "staging", "production"] = Field(
        default="production", validation_alias="ENV"
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON: bool = True

    # ── 2. Broker / vector store ────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    PINECONE_API_KEY: SecretStr
    PINECONE_INDEX_NAME: str = "financial-scans"
    # Each tenant gets its own Pinecone namespace: f"{PREFIX}{tenant_id}".
    PINECONE_NAMESPACE_PREFIX: str = "tenant-"

    # ── 3. Models & telemetry ───────────────────────────────────────────
    OPENAI_API_KEY: SecretStr
    OPENAI_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    OPENAI_MAX_RETRIES: int = Field(default=3, ge=0, le=10)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GENERATION_MODEL: str = "gpt-4o"
    SUMMARY_MODEL: str = "gpt-4o"
    SUMMARY_MAX_TOKENS: int = Field(default=1500, gt=0)
    GENERATION_MAX_TOKENS: int = Field(default=1500, gt=0)

    LANGCHAIN_API_KEY: SecretStr | None = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "financial-multimodal-rag"

    # ── 4. Retrieval ────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = Field(default=4, ge=1, le=20)
    MAX_QUESTION_CHARS: int = Field(default=2000, ge=10)

    # ── 5. Security ─────────────────────────────────────────────────────
    JWT_SECRET: SecretStr
    # Only HMAC algorithms are valid with a shared secret; "none" is never allowed.
    JWT_ALGORITHM: Literal["HS256", "HS384", "HS512"] = "HS256"
    JWT_AUDIENCE: str | None = None
    JWT_ISSUER: str | None = None
    JWT_LEEWAY_SECONDS: int = Field(default=30, ge=0, le=300)
    # If set, callers of /ingest must carry this value in the token's
    # "scope" (space-separated) or "scopes"/"roles" (list) claim.
    JWT_INGEST_SCOPE: str | None = "rag:ingest"
    # Holders may see/manage every document in their tenant and share to any group.
    JWT_ADMIN_SCOPE: str = "rag:admin"

    # ── 5b. Multi-tenancy & document ACLs ───────────────────────────────
    TENANT_CLAIM: str = "tenant_id"
    GROUPS_CLAIM: str = "groups"
    # When False, tokens without a tenant claim fall into DEFAULT_TENANT
    # (single-tenant deployments). Keep True for SaaS / multi-customer use.
    REQUIRE_TENANT: bool = True
    DEFAULT_TENANT: str = "default"
    MAX_ACL_GROUPS: int = Field(default=20, ge=1, le=100)

    # ── 5c. Per-user quotas (Redis-backed) ──────────────────────────────
    QUOTA_ENABLED: bool = True
    # If Redis is unavailable, allow traffic (True) or reject with 503 (False).
    QUOTA_FAIL_OPEN: bool = True
    QUERY_RATE_PER_MINUTE: int = Field(default=30, ge=0)       # 0 = unlimited
    INGEST_RATE_PER_HOUR: int = Field(default=60, ge=0)        # 0 = unlimited
    DAILY_TOKEN_BUDGET: int = Field(default=2_000_000, ge=0)   # per user per UTC day, 0 = unlimited

    # ── 5d. Observability ───────────────────────────────────────────────
    # Prometheus metrics are served on this separate port (never via the
    # public ingress). 0 disables the listener (tests).
    METRICS_PORT: int = Field(default=9000, ge=0, le=65535)

    # ── 6. Storage & ingestion limits ───────────────────────────────────
    # "s3" works with AWS S3, GCS (S3 interoperability / HMAC keys), MinIO,
    # Cloudflare R2, etc. "local" needs a ReadWriteMany volume when API and
    # workers run on different nodes.
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    STORAGE_PATH: str = "./data/store"          # root for the local backend
    S3_BUCKET: str | None = None
    S3_PREFIX: str = ""
    S3_ENDPOINT_URL: str | None = None          # e.g. https://storage.googleapis.com
    S3_REGION: str | None = None
    # Node-local scratch space for in-flight uploads/renders (never shared).
    SCRATCH_DIR: str = "/tmp/rag-scratch"  # noqa: S108 - per-pod emptyDir, never shared
    # Keep the original PDF so documents can be re-indexed later.
    RETAIN_ORIGINALS: bool = True
    MAX_UPLOAD_MB: int = Field(default=50, ge=1, le=1024)
    MAX_PDF_PAGES: int = Field(default=300, ge=1)
    PDF_RENDER_DPI: int = Field(default=200, ge=72, le=400)

    # ── 7. Layout model ─────────────────────────────────────────────────
    LAYOUT_MODEL_CONFIG: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    # Optional local weights path; bake into the image to avoid runtime downloads.
    LAYOUT_MODEL_WEIGHTS: str | None = None
    LAYOUT_SCORE_THRESHOLD: float = Field(default=0.78, ge=0.0, le=1.0)
    LAYOUT_DEVICE: Literal["cpu", "cuda"] = "cpu"
    SUMMARY_CONCURRENCY: int = Field(default=4, ge=1, le=32)

    # ── 7b. Narrative text extraction ───────────────────────────────────
    # tesseract: local OCR (cheap).  vision: GPT-4o page transcription (more
    # accurate on poor scans, costs tokens).  none: tables/figures only.
    TEXT_EXTRACTION: Literal["none", "tesseract", "vision"] = "tesseract"
    OCR_LANG: str = "eng"
    TEXT_CHUNK_SIZE: int = Field(default=1200, ge=200, le=8000)
    TEXT_CHUNK_OVERLAP: int = Field(default=150, ge=0, le=2000)
    MIN_PAGE_TEXT_CHARS: int = Field(default=40, ge=0)

    # ── 8. Celery ───────────────────────────────────────────────────────
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=1800, ge=60)
    CELERY_TASK_TIME_LIMIT: int = Field(default=2100, ge=60)
    CELERY_RESULT_EXPIRES: int = Field(default=86400, ge=60)

    # ── Validators ──────────────────────────────────────────────────────
    @field_validator(
        "JWT_AUDIENCE", "JWT_ISSUER", "JWT_INGEST_SCOPE", "LANGCHAIN_API_KEY",
        "LAYOUT_MODEL_WEIGHTS", "S3_BUCKET", "S3_ENDPOINT_URL", "S3_REGION",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v):
        # docker-compose / Helm render unset values as "", which must mean "not set".
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("LANGCHAIN_TRACING_V2", "REQUIRE_TENANT", "QUOTA_ENABLED", "QUOTA_FAIL_OPEN",
                     "RETAIN_ORIGINALS", "LOG_JSON", mode="before")
    @classmethod
    def _parse_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return v

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.CELERY_TASK_TIME_LIMIT <= self.CELERY_TASK_SOFT_TIME_LIMIT:
            raise ValueError("CELERY_TASK_TIME_LIMIT must exceed CELERY_TASK_SOFT_TIME_LIMIT")
        if self.STORAGE_BACKEND == "s3" and not self.S3_BUCKET:
            raise ValueError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        if self.TEXT_CHUNK_OVERLAP >= self.TEXT_CHUNK_SIZE:
            raise ValueError("TEXT_CHUNK_OVERLAP must be smaller than TEXT_CHUNK_SIZE")
        if self.LANGCHAIN_TRACING_V2 and self.LANGCHAIN_API_KEY is None:
            raise ValueError("LANGCHAIN_TRACING_V2=true requires LANGCHAIN_API_KEY")
        if self.ENVIRONMENT in {"staging", "production"}:
            secret = self.JWT_SECRET.get_secret_value()
            if len(secret) < 32 or any(m in secret.lower() for m in _WEAK_SECRET_MARKERS):
                raise ValueError(
                    "JWT_SECRET must be >= 32 chars of high-entropy data in staging/production "
                    "(generate with: python -c 'import secrets; print(secrets.token_urlsafe(48))')"
                )
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    def export_langsmith_env(self) -> None:
        """LangChain reads tracing config from os.environ, not from this object.

        Values loaded from ``.env`` are not exported automatically, so push them
        into the process environment once at startup.
        """
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if self.LANGCHAIN_TRACING_V2 else "false"
        os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
        if self.LANGCHAIN_API_KEY is not None:
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
