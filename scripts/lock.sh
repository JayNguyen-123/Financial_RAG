#!/usr/bin/env bash
# Generate fully-resolved, hash-pinned lock files from requirements/*.in.
# Run after changing any .in file and commit the resulting *.lock files.
#   pip install uv  &&  ./scripts/lock.sh
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(--generate-hashes --python-version 3.11 --python-platform x86_64-manylinux_2_28
        --index-strategy unsafe-best-match --no-header --quiet)

for name in api dev worker; do
  echo "Locking requirements/${name}.in -> requirements/${name}.lock"
  uv pip compile "requirements/${name}.in" "${COMMON[@]}" -o "requirements/${name}.lock"
done
echo "Done. Commit requirements/*.lock."
