#!/usr/bin/env bash
# Extract the model/ track into a standalone zip that can initialize a new repo.
# Produces ./drug-success-model.zip in the current working directory.
#
# Run from the repo root:
#   ./scripts/make_model_bundle.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BUNDLE_NAME="drug-success-model"
STAGE_PARENT="$(mktemp -d)"
STAGE="$STAGE_PARENT/$BUNDLE_NAME"
mkdir -p "$STAGE/model/_vendor" \
         "$STAGE/scripts" \
         "$STAGE/tests" \
         "$STAGE/notebooks" \
         "$STAGE/outputs/features"

echo "[1/8] Staging into $STAGE"

# ── 1. Copy model/ verbatim (minus pycache) ─────────────────────────────────
rsync -a --exclude '__pycache__' --exclude '*.pyc' model/ "$STAGE/model/"

# ── 2. Vendor admet_columns.py ───────────────────────────────────────────────
echo "[2/8] Vendoring src/pipeline/admet/admet_columns.py"
{
  echo "# Vendored from drug-success-report src/pipeline/admet/admet_columns.py"
  echo "# at extraction time. Regenerate via that repo's scripts/dump_admet_columns.py"
  echo "# when admet_ai upgrades."
  cat src/pipeline/admet/admet_columns.py
} > "$STAGE/model/_vendor/admet_columns.py"
: > "$STAGE/model/_vendor/__init__.py"

# ── 3. Patch model/features/admet.py to drop the sys.path hack ──────────────
echo "[3/8] Patching model/features/admet.py"
python3 - "$STAGE/model/features/admet.py" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()

# Drop the sys.path block (the "Locate ..." comment through the import line).
pattern = re.compile(
    r"\n# Locate the canonical ADMET column list under src/pipeline/admet/\.\n"
    r"_PROJECT_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n"
    r"_SRC = _PROJECT_ROOT / \"src\"\n"
    r"if str\(_SRC\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\(_SRC\)\)\n"
    r"\n"
    r"from pipeline\.admet\.admet_columns import ADMET_COLUMNS, field_name  # noqa: E402\n"
)
new = pattern.sub(
    "\nfrom model._vendor.admet_columns import ADMET_COLUMNS, field_name\n",
    src,
)
if new == src:
    raise SystemExit("admet.py patch did not match — source layout changed")

# Drop now-unused imports if nothing else references them.
if "sys." not in new.replace("import sys\n", ""):
    new = new.replace("import sys\n", "", 1)
if "Path(" not in new.replace("from pathlib import Path\n", ""):
    new = new.replace("from pathlib import Path\n", "", 1)

path.write_text(new)
PY

# ── 4. Copy model-track scripts ─────────────────────────────────────────────
echo "[4/8] Copying model-track scripts"
cp scripts/build_consolidated_report.py \
   scripts/build_hint_dataset.py \
   scripts/feature_class_audit.py "$STAGE/scripts/"

# ── 5. Copy pure-model tests (skip test_model_data.py — couples to pipeline) ─
echo "[5/8] Copying tests and stripping the sys.path REPO_ROOT hack"
for t in test_baselines.py test_killer_figure.py test_nn_similarity.py; do
  python3 - "src/tests/$t" "$STAGE/tests/$t" <<'PY'
import re
import sys
from pathlib import Path

src_path, dst_path = map(Path, sys.argv[1:3])
src = src_path.read_text()

# Strip the REPO_ROOT sys.path block — packaging makes it unnecessary.
pattern = re.compile(
    r"\nREPO_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\]\n"
    r"if str\(REPO_ROOT\) not in sys\.path:\n"
    r"    sys\.path\.insert\(0, str\(REPO_ROOT\)\)\n"
)
new = pattern.sub("\n", src)
if new == src:
    raise SystemExit(f"REPO_ROOT block not found in {src_path}")

# Drop now-unused `import sys` if no other references remain.
if "sys." not in new.replace("import sys\n", ""):
    new = new.replace("import sys\n", "", 1)

dst_path.write_text(new)
PY
done

# ── 6. Copy notebooks and bundled parquet inputs ────────────────────────────
echo "[6/8] Copying notebooks and parquet inputs (~363 MB)"
cp explore_feature_set.ipynb inspect_modeling_data.ipynb "$STAGE/notebooks/"
cp outputs/candidate_detail.parquet outputs/trial_detail.parquet "$STAGE/outputs/"
cp outputs/features/fingerprints.parquet \
   outputs/features/molformer_embeddings.parquet "$STAGE/outputs/features/"

# ── 7. Copy entry points and write new top-level files ──────────────────────
echo "[7/8] Writing pyproject.toml, README.md, CLAUDE.md, .gitignore"
cp run_modeling.py train_model.sh run_hint.sh "$STAGE/"

cat > "$STAGE/pyproject.toml" <<'TOML'
[project]
name = "drug-success-model"
version = "0.1.0"
description = "ML modeling track extracted from drug-success-report"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.3.3",
    "pyarrow>=24.0.0",
    "numpy>=2.0.0",
    "rdkit>=2024.3.1",
    "scikit-learn>=1.4.0",
    "xgboost>=2.0.0",
    "torch>=2.2.0",
    "transformers>=4.40,<5",
    "matplotlib>=3.10.8",
    "plotly>=6.0.0",
    "tqdm>=4.67.1",
    "tabulate>=0.9.0",
    "openpyxl>=3.1.5",
    "pdfplumber>=0.11.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["model"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "ipykernel>=7.2.0",
]
TOML

cat > "$STAGE/README.md" <<'MD'
# drug-success-model

ML modeling track extracted from `drug-success-report`. Trains phase-transition
success classifiers on top of pipeline-produced parquet inputs and benchmarks
against HINT.

## Quickstart

```bash
uv sync
uv run python -m model train --time-split-year 2019 --output ./model_runs/t2019
```

`train_model.sh` runs the full sequence (train + RFE + ablation + baselines +
HINT inference + consolidated PDF report).

## Input contract

The CLI expects four parquets under `outputs/` (bundled in this zip):

| Path | Purpose |
|---|---|
| `outputs/candidate_detail.parquet` | Candidate-level features and labels |
| `outputs/trial_detail.parquet` | Per-trial rows |
| `outputs/features/fingerprints.parquet` | ECFP4 fingerprints |
| `outputs/features/molformer_embeddings.parquet` | MolFormer embeddings |

These are produced by the upstream `drug-success-report` pipeline.

## HINT inference

`run_hint.sh` hardcodes a sibling-repo path:

```
HINT_ROOT="/Users/samirtownsley/Documents/projects/hint_standalone/repo"
```

Edit it for your machine. HINT itself is not vendored.

## Vendored module

`model/_vendor/admet_columns.py` is a verbatim copy of
`src/pipeline/admet/admet_columns.py` from the source repo. Regenerate it via
that repo's `scripts/dump_admet_columns.py` when admet_ai upgrades.

## Tests

```bash
uv run pytest tests/ -q
```

Three test modules ship: `test_baselines`, `test_killer_figure`,
`test_nn_similarity`. The pipeline-coupled `test_model_data.py` was left in the
source repo.
MD

cat > "$STAGE/CLAUDE.md" <<'MD'
# CLAUDE.md

## What this is
ML modeling track extracted from `drug-success-report`. Consumes pipeline-produced
parquets under `outputs/` and trains phase-transition success classifiers.

## Layout
- `model/cli.py` — `python -m model {train,rfe,ablate,baselines}` entry point
- `model/{train,rfe,ablate,evaluate,report,killer_figure,hint_format}.py` — top-level commands
- `model/features/` — feature groups (fingerprints, embeddings, admet, pathway, targets, moa, ...)
- `model/models/` — `logreg`, `xgb`
- `model/baselines/` — stratum / tanimoto / killer-figure / target-only baselines
- `model/_vendor/admet_columns.py` — vendored from upstream pipeline; do not edit by hand
- `scripts/build_consolidated_report.py` — PDF assembly across runs
- `scripts/build_hint_dataset.py` — formats CSV for HINT inference
- `scripts/feature_class_audit.py` — feature-engineering audit

## Commands
```bash
uv sync
uv run python -m model train --time-split-year 2019 --output ./model_runs/t2019
./train_model.sh                              # full run + report
./run_hint.sh <input.csv> <output.csv>        # external HINT inference
uv run pytest tests/ -q
```

## Working rules
- Don't import from any `pipeline.*` namespace — there is no upstream package here. Use `model._vendor` if you need a constant that originated in the pipeline.
- `model/config.py` resolves the four parquet inputs relative to repo root; keep them under `outputs/` or pass explicit `--candidate-detail` / `--trial-detail` / etc.
- `run_hint.sh` has a hardcoded sibling-repo path. Edit it per machine.
MD

cat > "$STAGE/.gitignore" <<'GI'
.venv/
__pycache__/
*.pyc
.pytest_cache/
model_runs/
.DS_Store
GI

# ── 8. Zip ──────────────────────────────────────────────────────────────────
echo "[8/8] Zipping"
OUT="$REPO_ROOT/${BUNDLE_NAME}.zip"
rm -f "$OUT"
( cd "$STAGE_PARENT" && zip -rq "$OUT" "$BUNDLE_NAME" )

echo ""
echo "Done: $OUT"
du -h "$OUT"
rm -rf "$STAGE_PARENT"
