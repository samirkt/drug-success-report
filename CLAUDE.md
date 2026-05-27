# CLAUDE.md

Durable project context for Claude Code. A deeper architecture deep-dive lives at `src/CLAUDE.md` — read it before touching anything under `src/pipeline/`.

## What the repo does

Automated pipeline for reconstructing drug-development trajectories from public clinical trial data (AACT, DrugBank, openFDA, ChEMBL, OpenTargets) and computing phase-transition success rates stratified by drug modality and disease area. A second track (`model/`) trains ML models on the resulting candidate/trial tables and benchmarks against HINT.

## Tech stack

- Python >= 3.11, managed with `uv` (lockfile: `uv.lock`); virtualenv at `.venv/`
- Packaging: `hatchling`, packages = `src/pipeline`, `src/utils`
- Core deps: `pandas`, `pyarrow`, `rdkit`, `chembl_structure_pipeline`, `admet-ai`, `transformers`, `anthropic`, `psycopg`, `httpx`, `pdfplumber`, `plotly`
- Modeling extras: `torch`, `scikit-learn`, `xgboost`
- Tests: `pytest`, custom runner at `src/run_tests.sh`
- External services: Anthropic API (LLM), AACT Postgres, openFDA HTTP, local Ollama for FDA-adjudication LLM
- HINT inference lives in a separate repo at `/Users/samirtownsley/Documents/projects/hint_standalone/repo` and is invoked via `run_hint.sh`

## Important commands

```bash
# Run the trajectory pipeline (from src/)
cd src && ./execute.sh
# or directly:
uv run python src/run_pipeline.py --source aact --keyword peptide --output ./outputs --formats html excel

# Tests (from src/)
cd src && ./run_tests.sh            # compact
./run_tests.sh -v                   # verbose
./run_tests.sh --include aggregation
./run_tests.sh --include pipeline   # integration test, skipped by default

# Modeling — full training + RFE + ablation + baselines + HINT + report
./train_model.sh
# or invoke directly
uv run python -m model train --time-split-year 2019 --output ./model_runs/t2019
uv run python -m model rfe --time-split-year 2019 --output ./model_runs/rfe_t2019
uv run python -m model ablate --mode loo --time-split-year 2019 --output ./model_runs/loo_t2019
uv run python -m model baselines --time-split-year 2019 --output ./model_runs/baselines

# HINT trial-level inference
./run_hint.sh ./model_runs/t2019/trial/hint_test.csv ./model_runs/t2019/trial/hint_results.csv

# Consolidated PDF report across all run artifacts
uv run python scripts/build_consolidated_report.py --run-root ./model_runs --train-name t2019 \
  --rfe-name rfe_t2019 --ablate-name loo_t2019 --baselines-name baselines \
  --hint-metrics ./model_runs/t2019/trial/hint_results.csv \
  --output ./outputs/consolidated_report.pdf
```

## Important files & directories

- `src/run_pipeline.py` — trajectory pipeline entrypoint
- `src/pipeline/` — staged pipeline (ingestion → clustering → classification → adjudication → aggregation → reporting). Single source of truth for inter-stage types is `pipeline/models.py`
- `src/pipeline/stages/prompts/*.txt` — LLM prompts; edit these, not the Python
- `src/utils/` — shared helpers (`prompt_runner.py`, `tiered_router.py`, `ndc_lookup.py`, etc.)
- `src/CLAUDE.md` — detailed pipeline architecture, env vars, non-obvious patterns. READ FIRST when editing `src/`
- `src/tests/` + `src/run_tests.sh` — test suite and runner
- `model/` — ML training package (`python -m model {train,rfe,ablate,baselines}`); features under `model/features/`, models under `model/models/`, baselines under `model/baselines/`
- `run_modeling.py` — thin wrapper for `python -m model …`
- `scripts/` — one-off builders & audits (fingerprints, MolFormer embeddings, ChEMBL/OpenTargets snapshots, HINT dataset, consolidated report). Also `scripts/make_model_bundle.sh` — produces `drug-success-model.zip`, a standalone seed for the extracted model-track repo (vendors `pipeline/admet/admet_columns.py`, bundles the four parquet inputs)
- `paper/` — methods/results write-ups in Markdown
- `outputs/`, `model_runs/`, `docs/` — generated artifacts (gitignored by pattern)
- Top-level data files (`drugbank_approvals*.csv`, `chembl_targets.sqlite`, `opentargets_snapshot.sqlite`, `src/fda.db`, `src/icd10_cache.sqlite`) — local caches/snapshots, not committed
- `.fda_cache/` — openFDA HTTP cache

## Environment variables

| Variable | Required for | Purpose |
|---|---|---|
| `CLAUDE_API_KEY_UCD` | LLM stages | Anthropic API key |
| `AACT_DB_USER` | `--source aact` | AACT Postgres username |
| `AACT_DB_PASSWORD` | `--source aact` | AACT Postgres password |

## Inferred conventions

- Use `uv run` rather than activating the venv directly
- Pipeline stages exchange typed dataclasses defined in `src/pipeline/models.py` — add fields there first
- LLM prompt changes go in `src/pipeline/stages/prompts/*.txt`, not Python
- All phase-transition / LOA rate math lives in `src/pipeline/stages/aggregation.py` — do not re-implement inline
- Bash scripts use `set -euo pipefail`
- Generated artifacts go under `outputs/`, `model_runs/`, `docs/`; raw data and `*.csv`/`*.xlsx`/`*.pdf`/`*.pkl`/`*.ipynb` are gitignored

## Working rules for Claude Code

- Before editing anything under `src/pipeline/`, read `src/CLAUDE.md` — it documents non-obvious patterns (tiered LLM routing, cohort aggregation method, swappable FDA adjudicator, M×N trial expansion, etc.)
- Don't bypass `aggregation.py` for rate math; reuse `transition_rate_from_records`
- Don't commit data caches: `*.csv`, `*.xlsx`, `*.pdf`, `*.pkl`, `*.db`/`_*.db`, `*.ipynb`, `docs/*` are gitignored — but `notebooks/` and some explicitly-added notebooks are checked in. Confirm before adding new large binaries
- HINT inference requires a sibling repo at `/Users/samirtownsley/Documents/projects/hint_standalone/repo`; `run_hint.sh` `cd`s into it
- Local Ollama (`qwen2.5:14b-instruct`) is the default FDA-adjudication LLM; respect `OLLAMA_NUM_PARALLEL` when raising `--fda-adjudication-workers`
- TODO: no top-level lint/format config detected (no ruff/black settings in `pyproject.toml`); follow existing file style
- Model-track ↔ pipeline coupling: the only code-level crossing is `model/features/admet.py` importing `pipeline.admet.admet_columns` via a `sys.path` hack. If you add a second cross-import, the extraction bundler (`scripts/make_model_bundle.sh`) will need a new vendor step
