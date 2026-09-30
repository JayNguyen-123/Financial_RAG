"""Blob storage abstraction for originals, image crops and document manifests.

Two backends:
* ``LocalBlobStore``  - filesystem (dev, or a ReadWriteMany volume)
* ``S3BlobStore``     - any S3-compatible object store (AWS S3, GCS via the
                        S3 interoperability API, MinIO, R2). Credentials come
                        from the default AWS chain (IRSA / Workload Identity
                        env vars / instance profile) - never from app config.

Moving to object storage removes the need for an RWX volume shared by API
and worker pods: the API uploads the original, the worker downloads it.

Key layout (all keys are relative, "/"-separated):
    originals/{tenant}/{document_id}.pdf
    images/{tenant}/{element_id}.png
    manifests/{tenant}/{document_id}.json
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from config.settings import Settings

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-/]{0,1023}$")


def validate_key(key: str) -> str:
    if not _KEY_RE.match(key) or ".." in key.split("/") or "//" in key:
        raise ValueError(f"Invalid storage key: {key!r}")
    return key


def original_key(tenant: str, document_id: str) -> str:
    return f"originals/{tenant}/{document_id}.pdf"


def image_key(tenant: str, element_id: str) -> str:
    return f"images/{tenant}/{element_id}.png"


def manifest_key(tenant: str, document_id: str) -> str:
    return f"manifests/{tenant}/{document_id}.json"


def manifest_prefix(tenant: str) -> str:
    return f"manifests/{tenant}/"


class BlobStore(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def put_file(self, key: str, path: str, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    def download_to(self, key: str, path: str) -> bool: ...

    @abstractmethod
    def delete(self, keys: Iterable[str]) -> None: ...

    @abstractmethod
    def list(self, prefix: str) -> Iterator[str]: ...

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def mget(self, keys: Iterable[str]) -> list[bytes | None]:
        return [self.get(k) for k in keys]

    def healthy(self) -> bool:
        """Cheap liveness check used by the readiness probe."""
        try:
            next(iter(self.list("__healthcheck__/")), None)
            return True
        except Exception:
            return False


class LocalBlobStore(BlobStore):
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        path = os.path.abspath(os.path.join(self.root, validate_key(key)))
        if not path.startswith(self.root + os.sep):
            raise ValueError(f"Key escapes storage root: {key!r}")
        return path

    def _atomic_write(self, key: str, writer) -> None:
        dest = self._path(key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                writer(fh)
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def put(self, key, data, content_type="application/octet-stream"):
        self._atomic_write(key, lambda fh: fh.write(data))

    def put_file(self, key, path, content_type="application/octet-stream"):
        def _copy(fh):
            with open(path, "rb") as src:
                shutil.copyfileobj(src, fh, 1 << 20)

        self._atomic_write(key, _copy)

    def get(self, key):
        try:
            with open(self._path(key), "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    def download_to(self, key, path):
        src = self._path(key)
        if not os.path.exists(src):
            return False
        shutil.copyfile(src, path)
        return True

    def delete(self, keys):
        for key in keys:
            try:
                os.remove(self._path(key))
            except FileNotFoundError:
                pass

    def list(self, prefix):
        base = self._path(prefix.rstrip("/") or ".") if prefix.strip("/") else self.root
        if not os.path.isdir(base):
            return
        for dirpath, _, files in os.walk(base):
            for name in sorted(files):
                if name.startswith(".tmp-"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), self.root).replace(os.sep, "/")
                if rel.startswith(prefix):
                    yield rel

    def healthy(self):
        return os.path.isdir(self.root) and os.access(self.root, os.W_OK)


class S3BlobStore(BlobStore):
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None,
                 region: str | None = None, client=None):
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region,
                config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, connect_timeout=5, read_timeout=60),
            )
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""

    def _k(self, key: str) -> str:
        return self.prefix + validate_key(key)

    def put(self, key, data, content_type="application/octet-stream"):
        self.client.put_object(Bucket=self.bucket, Key=self._k(key), Body=data, ContentType=content_type,
                               ServerSideEncryption="AES256")

    def put_file(self, key, path, content_type="application/octet-stream"):
        # upload_file does multipart uploads for large files automatically.
        self.client.upload_file(path, self.bucket, self._k(key),
                                ExtraArgs={"ContentType": content_type, "ServerSideEncryption": "AES256"})

    def get(self, key):
        from botocore.exceptions import ClientError

        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._k(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        return obj["Body"].read()

    def download_to(self, key, path):
        from botocore.exceptions import ClientError

        try:
            self.client.download_file(self.bucket, self._k(key), path)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise

    def delete(self, keys):
        batch = [{"Key": self._k(k)} for k in keys]
        for i in range(0, len(batch), 1000):  # S3 DeleteObjects limit
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch[i:i + 1000], "Quiet": True})

    def list(self, prefix):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"][len(self.prefix):]

    def healthy(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


def build_blob_store(settings: Settings) -> BlobStore:
    if settings.STORAGE_BACKEND == "s3":
        return S3BlobStore(settings.S3_BUCKET or "", settings.S3_PREFIX, settings.S3_ENDPOINT_URL, settings.S3_REGION)
    return LocalBlobStore(settings.STORAGE_PATH)
