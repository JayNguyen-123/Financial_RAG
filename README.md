# Multimodal Financial RAG Platform

A multi-tenant, multimodal RAG service for scanned financial PDFs. For each page it:

- finds tables and charts with a layout model, transcribes and summarizes them with GPT-4o vision, and indexes the results;
- runs OCR on the remaining narrative text and indexes it.

At query time the answer model sees the retrieved text, the table summaries and the original crops. Answers include page citations.

See **[REVIEW.md](REVIEW.md)** for the code review, the changes in v1.1/v1.2, and migration notes.

## Architecture

```
POST /documents ─► size/magic checks ─► sha256 = document_id ─► originals/{tenant}/{id}.pdf (blob store)
   (JWT: tenant, groups, scopes)        manifest (owner, ACL, status) ─► Redis queue "ingestion"
                                                                                │  (KEDA scales workers on depth)
                                                                                ▼
                                                   Celery worker (per page, bounded memory)
                                                   ├─ Detectron2: Table/Figure boxes → PNG crops → blob store
                                                   ├─ Tesseract OCR of the page with those boxes masked
                                                   ├─ GPT-4o: Markdown transcription + summary per crop
                                                   └─ Pinecone upsert → namespace tenant-{tenant}, metadata.acl
POST /query ─► quota check ─► Pinecone(namespace=tenant, filter acl ∈ caller principals)
            ─► crops + text ─► GPT-4o ─► SSE: sources → token* → done | error ─► token usage → daily budget
```

| Component | Image target | Notes |
|---|---|---|
| API | `api` | FastAPI; port 8000 (via ingress), port 9000 metrics (internal only) |
| Worker | `worker` | CPU torch, Detectron2 (pinned commit), Tesseract, poppler |
| Broker | `redis:7.4` | `noeviction`, AOF; also stores quota counters |
| Blob store | S3-compatible bucket, or a local/RWX volume | originals, crops, manifests |

## Prerequisites

- **Pinecone:** a serverless index with **dimension 1536** and the **cosine** metric. Namespaces are created automatically, one per tenant.
- **OpenAI:** a key with GPT-4o access.
- **JWTs** signed with `JWT_SECRET` (HS256/384/512) and containing:

| Claim | Required | Purpose |
|---|---|---|
| `sub` | yes | User id (document ownership, quotas) |
| `exp` | yes | Expiry |
| `tenant_id` | yes (unless `REQUIRE_TENANT=false`) | Tenant isolation |
| `groups` | no | Group-scoped document visibility |
| `scope` | no | `rag:ingest` to upload or manage, `rag:admin` for tenant-wide admin |

## Local development

The Docker build installs from hash-locked requirement files, so generate them once (this needs network access):

```bash
pip install uv && ./scripts/lock.sh
cp .env.example .env     # fill in keys
docker compose --env-file .env up -d --build
# S3 backend via MinIO instead of local disk:
STORAGE_BACKEND=s3 docker compose --env-file .env --profile s3 up -d --build
```

Mint a dev token:

```bash
export TOKEN=$(python -c "import jwt,time,os; print(jwt.encode({'sub':'alice','tenant_id':'acme','groups':['finance'],'scope':'rag:ingest','exp':int(time.time())+3600}, os.environ['JWT_SECRET'], algorithm='HS256'))")
```

## API

| Method & path | Scope | Description |
|---|---|---|
| `POST /documents` (multipart `file`, optional `acl_groups`, `?replace=true`) | `rag:ingest` | Upload and queue a PDF. Returns 202 with `document_id` and `task_id`. A duplicate upload returns 409. |
| `GET /documents` | any | Documents visible to the caller |
| `GET /documents/{id}` | any | Status (`queued` / `processing` / `indexed` / `failed`), counts, ACL |
| `DELETE /documents/{id}` | owner or admin | Removes vectors, crops, the original and the manifest |
| `POST /documents/{id}/reindex` (optional `{"acl_groups": [...]}`) | owner or admin | Re-runs ingestion from the retained original |
| `GET /ingest/{task_id}` | same tenant | Celery task state |
| `POST /query` `{"question": "...", "stream": true}` | any | SSE stream, or JSON when `stream` is false |

```bash
curl -H "Authorization: Bearer $TOKEN" -F file=@10-Q.pdf -F acl_groups=finance localhost:8000/documents
curl -H "Authorization: Bearer $TOKEN" localhost:8000/documents
curl -N -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"question":"What was Q3 operating income?"}' localhost:8000/query
```

**Errors:** 401 means the token is bad. 403 means no tenant or missing scope. 404 means the document isn't found or isn't visible. 409 means a duplicate upload or a document that is busy. 413 means the file is too large. 429 means a quota was hit (the response has `Retry-After`). 502 means retrieval failed.

### SSE events (`POST /query`)

| event | data |
|---|---|
| `sources` | `{"sources":[{"index","doc_id","document_id","source_file","page_number","content_type","score"}]}` |
| `token` | `{"text": "..."}` (repeated) |
| `done` | `{"trace_id": "...", "usage": {"total_tokens": N}}` |
| `error` | `{"code": "guardrail_blocked" \| "generation_failed", "message": "..."}` |

## Access model

- **Tenants** are isolated at the storage level. Each tenant has its own Pinecone namespace and blob prefix.
- **Document visibility:** with no `acl_groups`, the whole tenant can see a document. With groups, only members of those groups, plus the uploader, can see it. Uploaders can only grant groups they belong to; admins can grant any group.
- **Admins** (`rag:admin`) see and manage every document in their own tenant, and no other tenant's.

## Quotas

Limits are per user within a tenant and stored in Redis. `QUERY_RATE_PER_MINUTE` defaults to 30, `INGEST_RATE_PER_HOUR` to 60, and `DAILY_TOKEN_BUDGET` to 2M tokens per UTC day. The budget counts query tokens plus the ingestion vision and OCR tokens charged to the uploader. `QUOTA_FAIL_OPEN` decides whether traffic is allowed or rejected with 503 when Redis is down.

## Tests

```bash
pip install --require-hashes --no-deps -r requirements/dev.lock
ruff check . && pytest
```

The tests use fakes for OpenAI, Pinecone and Redis, and moto for S3. The Tesseract test runs when the binary is installed.

## Kubernetes (Helm)

1. Create the Secret out-of-band with the keys `OPENAI_API_KEY`, `PINECONE_API_KEY`, `JWT_SECRET`, and optionally `LANGCHAIN_API_KEY`.
2. Create a bucket and grant the chart's ServiceAccount read/write access through workload identity (IRSA or GKE Workload Identity; see `serviceAccount.annotations`).
3. Install [KEDA](https://keda.sh), or set `worker.autoscaling.keda.enabled=false`.

```bash
helm upgrade --install rag ./helm/multimodal-rag \
  --set secrets.existingSecret=rag-secrets \
  --set storage.s3.bucket=my-rag-bucket \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=arn:aws:iam::123:role/rag \
  --set image.api.repository=ghcr.io/<org>/<repo>/api       --set image.api.tag=<sha> \
  --set image.worker.repository=ghcr.io/<org>/<repo>/worker --set image.worker.tag=<sha> \
  --set ingress.hosts[0].host=rag.example.com
```

The chart creates the API Deployment (with HPA, PDB and a NetworkPolicy), the worker Deployment (with a KEDA ScaledObject and PDB), a Redis StatefulSet and NetworkPolicy, an Ingress (with upload size and SSE settings), a ServiceMonitor on the internal `metrics` port, and a Grafana dashboard. With `storage.backend=local` it also creates an RWX PVC.

## CI/CD

The `test` job runs ruff, pytest and pip-audit on the lock files, installing with hash checks. The `helm` job runs lint and kubeconform against both storage modes. `eval` is the LangSmith accuracy gate: it needs at least 90% and runs against `EVAL_TENANT`. `build` builds the `api` and `worker` images and pushes them on `main`. `notify` sends a Slack alert on any failure.

## Dependencies

See [requirements/README.md](requirements/README.md). Edit the `*.in` files, run `./scripts/lock.sh`, and commit the `*.lock` files.
