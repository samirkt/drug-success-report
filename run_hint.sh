#!/bin/bash
# Run HINT inference on a HINT-format CSV.
#   usage: ./run_hint.sh <input.csv> <output.csv>
# Both args are required. Paths are resolved against the *current* working
# directory before we cd into the HINT repo (so relative paths from the
# drug-success-report repo still work).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <input.csv> <output.csv>" >&2
  exit 1
fi

INPUT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUTPUT_DIR="$(cd "$(dirname "$2")" && pwd)"
OUTPUT="${OUTPUT_DIR}/$(basename "$2")"

HINT_ROOT="/Users/samirtownsley/Documents/projects/hint_standalone/repo"
cd "$HINT_ROOT" && uv run python run_hint_on_dataset.py --input "$INPUT" --output "$OUTPUT"
