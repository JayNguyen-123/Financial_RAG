# Dependencies

| File | Edited by | Used by |
|---|---|---|
| `api.in`, `worker.in`, `dev.in` | humans (direct deps, exact pins) | `scripts/lock.sh` |
| `api.lock`, `worker.lock`, `dev.lock` | `scripts/lock.sh` (never by hand) | Dockerfile, CI (`pip install --require-hashes`) |

The lock files contain every transitive dependency with its sha256 hashes, so
a compromised or re-published package on the index fails the install instead
of silently entering the image.

```bash
pip install uv
./scripts/lock.sh        # regenerate after changing any .in file
git add requirements/*.lock
```

Detectron2 is not on PyPI. It is built in the worker image from a pinned commit
(`DETECTRON2_REF` in the Dockerfile, default = the `v0.6` tag commit
`d1e04565d3bec8719335b88be9e9b961bf3ec464`). It is installed with `--no-deps`;
its runtime dependencies are listed in `worker.in`.
