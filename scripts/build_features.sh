#!/usr/bin/env bash
# build_features.sh — Run both featurization scripts against a pipeline run.
#
# Usage:
#   ./scripts/build_features.sh                                  # uses defaults below
#   ./scripts/build_features.sh path/to/candidate_detail.parquet path/to/features/
#
# MolFormer embeddings need transformers<5 (the IBM model's remote code
# uses APIs removed in transformers 5.x). We override per-run with
# `uv run --with "transformers<5"` so pyproject.toml stays untouched.
# Fingerprints work with the project's pinned versions.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CANDIDATES="${1:-outputs/candidate_detail.parquet}"
FEATURES_DIR="${2:-outputs/features}"

if [[ ! -f "$CANDIDATES" ]]; then
    echo "error: candidates parquet not found at: $CANDIDATES" >&2
    exit 1
fi

mkdir -p "$FEATURES_DIR"

echo "==> ECFP4 + MACCS fingerprints"
uv run python scripts/build_fingerprints.py \
    --candidates "$CANDIDATES" \
    --output "$FEATURES_DIR/fingerprints.parquet"

echo ""
echo "==> MolFormer-XL embeddings (transformers<5 override)"
uv run --with "transformers<5" python scripts/build_molformer_embeddings.py \
    --candidates "$CANDIDATES" \
    --output "$FEATURES_DIR/molformer_embeddings.parquet"

echo ""
echo "Done. Outputs in $FEATURES_DIR/"
