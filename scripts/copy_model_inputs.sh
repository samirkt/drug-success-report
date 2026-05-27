#!/usr/bin/env bash
# Copy the four parquet inputs the model expects into a flat directory.
#
# Usage:
#   ./scripts/copy_model_inputs.sh                # dest defaults to ./model_inputs
#   ./scripts/copy_model_inputs.sh /path/to/dest
#
# Output layout:
#   <dest>/
#   ├── candidate_detail.parquet
#   ├── trial_detail.parquet
#   └── features/
#       ├── fingerprints.parquet
#       └── molformer_embeddings.parquet

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEST="${1:-./model_inputs}"

SOURCES=(
    "outputs/candidate_detail.parquet"
    "outputs/trial_detail.parquet"
    "outputs/features/fingerprints.parquet"
    "outputs/features/molformer_embeddings.parquet"
)
for src in "${SOURCES[@]}"; do
    if [[ ! -f "$src" ]]; then
        echo "error: missing source file: $src" >&2
        exit 1
    fi
done

mkdir -p "$DEST/features"
cp outputs/candidate_detail.parquet "$DEST/"
cp outputs/trial_detail.parquet "$DEST/"
cp outputs/features/fingerprints.parquet "$DEST/features/"
cp outputs/features/molformer_embeddings.parquet "$DEST/features/"

echo "Copied 4 parquet files into $DEST/"
