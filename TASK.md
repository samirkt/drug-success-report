# TASK.md

## Current goal
Extract the `model/` track into a standalone zip that can initialize a new repo.

## Status
Done. `drug-success-model.zip` (295 MB) built at repo root, verified end-to-end (static checks clean, 30+ modules import cleanly, 34/34 tests pass).

## What changed
- Built a reproducible bundling driver `scripts/make_model_bundle.sh` (rerun anytime to regenerate the zip)
- Generated `drug-success-model.zip` containing a ready-to-init `drug-success-model/` repo:
  - `model/` copied verbatim, with `features/admet.py` patched to drop the `sys.path → src/pipeline.admet.admet_columns` hack
  - `model/_vendor/admet_columns.py` — vendored from `src/pipeline/admet/admet_columns.py` (104 entries, 52 percentile siblings)
  - `scripts/` — model-track only: `build_consolidated_report.py`, `build_hint_dataset.py`, `feature_class_audit.py`
  - `tests/` — `test_baselines.py`, `test_killer_figure.py`, `test_nn_similarity.py` (REPO_ROOT `sys.path` block stripped during bundling)
  - `notebooks/` — `explore_feature_set.ipynb`, `inspect_modeling_data.ipynb`
  - `outputs/` — the four pipeline-produced parquets the model reads (`candidate_detail.parquet` 21M, `trial_detail.parquet` 80M, `features/fingerprints.parquet` 6.1M, `features/molformer_embeddings.parquet` 256M)
  - `run_modeling.py`, `train_model.sh`, `run_hint.sh` (unchanged)
  - New `pyproject.toml` (drops `psycopg`, `lxml`, `chembl_structure_pipeline`, `anthropic`, `httpx`, `requests`, `fuzzywuzzy`, `python-levenshtein`, `admet-ai`; keeps `pdfplumber` for `build_consolidated_report.py`)
  - New `README.md`, `CLAUDE.md`, `.gitignore`
- CLAUDE.md updated: noted `scripts/make_model_bundle.sh` under `scripts/` and added the one-and-only cross-import (`model/features/admet.py` → `pipeline.admet.admet_columns`) to working rules
- Earlier in this session: created root `CLAUDE.md` and `TASK.md` (now superseded by this version)

## Files touched
- `scripts/make_model_bundle.sh` (new, ~270 lines, executable)
- `CLAUDE.md` (two small additions; see above)
- `TASK.md` (this file)
- `drug-success-model.zip` (new artifact at repo root — should NOT be committed)
- No edits to anything under `model/`, `src/`, or `scripts/build_*`. The bundling script only reads from those locations and writes to a staging tempdir.

## Commands run
```bash
./scripts/make_model_bundle.sh                # build the zip
# verification (one-off, no script):
unzip -q drug-success-model.zip -d "$VERIFY"
grep -rnE '^\s*(from|import)\s+(pipeline|utils|src)\b' model/ scripts/ tests/   # → clean
grep -rn 'sys.path' model/                                                       # → clean
cd "$VERIFY/drug-success-model" && \
  /Users/samirtownsley/Documents/projects/drug-success-report/.venv/bin/python -m pytest tests/ -q
# → 34 passed in 1.91s
```

## Remaining work
- None for this task. Optional follow-ups (not started, not required):
  - Delete `model/`, `run_modeling.py`, `train_model.sh`, `run_hint.sh`, the three model-track `scripts/`, and the three model-track `src/tests/` files from this repo once the extracted repo is confirmed working in production. Hold off until then.
  - Update `src/CLAUDE.md` to remove model references after the deletion above.
  - `run_hint.sh` in the bundle still hardcodes `/Users/samirtownsley/Documents/projects/hint_standalone/repo`. Each new clone needs to edit it; README in the bundle calls this out.
  - The bundle is large (295 MB) because of `molformer_embeddings.parquet` (256 MB). If GitHub LFS or a smaller fixture is preferable, swap the parquet step in `make_model_bundle.sh`.

## Notes / gotchas
- `src/tests/test_model_data.py` was deliberately NOT bundled — it imports `pipeline.models.TrialStatus` and `pipeline.stages.aggregation.FunnelAggregationStage`. Pulling it in would have required vendoring large chunks of the pipeline. It stays in the source repo; the three pure model tests (`test_baselines`, `test_killer_figure`, `test_nn_similarity`) move cleanly.
- `src/tests/conftest.py` (336 lines, all pipeline fixtures) was NOT bundled — none of the three moved tests reference its fixtures by name; verified via grep.
- The bundler's `python3 -c` patch for `model/features/admet.py` matches the exact 7-line `sys.path` block. If that block is reformatted in the source, the bundler will fail loudly with `"admet.py patch did not match — source layout changed"` rather than silently producing a broken bundle. Same defense for the `REPO_ROOT` test block.
- During verification, `PYTHONPATH=…` was insufficient because the source-repo CWD shadowed the bundled `model/` package. Must `cd` into the unzipped bundle before running Python.
- The model's data contract is the four parquets at the paths declared in `model/config.py`; if those paths change in the source repo, update both `model/config.py` AND the bundler's `cp` step.
- `drug-success-model.zip` is at repo root and matches the gitignore pattern (no specific entry, but it's a generated artifact — don't `git add` it).
