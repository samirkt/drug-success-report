# TASK.md

## Current goal
Pull OpenTargets genetic evidence + tractability into the candidate parquet so the modeling track has full target-disease + druggability features.

## Status
Done. Snapshot builder, Candidate dataclass, OT enrichment, parquet writer, CSV writer, tests, and downloader all extended. Tests: 23/23 OT enrichment + 29/29 snapshot writer + 836/837 full suite (one pre-existing pipeline_sampling failure, unrelated).

Follow-up (2026-05-26): the CSV writer `write_candidate_detail` was missing the five new OT fields that the parquet writer already emitted — `candidate_detail.csv` had `opentargets_*` stop at `opentargets_indication_max_phase`. Added the five missing headers + row serializers so the CSV mirrors the parquet. List fields pipe-joined; scalars stringified (empty string when `None`).

## What changed

### Schema
- New OT snapshot table `name_opentargets_target_disease_evidence(chembl_id, ensembl_id, efo_id, genetic_score)` — long-form OT `genetic_association` scores joined onto each drug's targets at snapshot-build time. Indexed by `(chembl_id, efo_id)`.
- Three new columns on `name_opentargets_drug`: `tractability_modalities` (pipe-joined coarse modality buckets, e.g. "SM|AB"), `tractability_labels` (full OT assessment label list), `loeuf_min` (REAL — gnomAD LOEUF min across the drug's targets).
- Five new fields on `pipeline.models.Candidate` and on `outputs/candidate_detail.parquet`:
  - `opentargets_tractability_modalities` (`list[str]`)
  - `opentargets_tractability_labels` (`list[str]`)
  - `opentargets_loeuf_min` (`Optional[float]`)
  - `opentargets_genetic_score` (`Optional[float]`) — max across the drug's targets at the matched indication EFO
  - `opentargets_genetic_score_max_any_indication` (`Optional[float]`) — same but across any EFO; target-quality fallback when indication match misses

### Code
- `scripts/build_opentargets_snapshot.py`:
  - `load_targets_table` now also extracts `tractability` and `geneticConstraint`/`constraint` per Ensembl. Leniently resolves OT's release-to-release column-name drift (LOEUF score key tried in order: `oe_ci_upper`, `oeCiUpper`, `oeUpper`, `oe_upper`, `upperBin`, `upper_bin`, `score`).
  - New `load_associations_table` reads `associationByDatatypeDirect/` filtered to `datatypeId == 'genetic_association'` and only ensembls drugs actually target (full OT table is ~10M rows; this keeps the SQLite snapshot lean).
  - `aggregate_drug_level` unions tractability labels/modalities across the drug's targets and takes the min LOEUF (lower = more constrained gene = more disease-relevance).
  - `write_snapshot` emits the new columns/tables and creates `idx_ot_evidence_chembl_efo`.
- `src/pipeline/models.py`: five new Candidate fields (listed above) inserted after `opentargets_indication_max_phase`.
- `src/pipeline/enrichment/opentargets.py`:
  - Detects new columns/table via `PRAGMA table_info` and a `try/except sqlite3.DatabaseError` around the evidence-table SELECT — older snapshots still load with the new features all `None`/`[]` and coverage at zero.
  - Pre-aggregates evidence to `{(chembl, efo): max_score}` and `{chembl: max_score}` so the hot loop is two dict lookups.
  - Indication matching now also resolves the matched EFO from `name_opentargets_indications.indication_efo_id` (previously dropped); reuses it for the genetic-score join. Phase-resolution behavior is unchanged.
  - New ledger coverage keys: `opentargets_tractability`, `opentargets_loeuf`, `opentargets_genetic_score`, `opentargets_genetic_score_any`.
- `src/pipeline/stages/reporting/_writer.py`: five new columns in `write_candidate_parquet`. List columns stay native lists; scalars stay native floats. Null defaults round-trip as NaN under pyarrow Float64 inference (test uses `pd.isna`). Follow-up: also added the same five columns to `write_candidate_detail` (the CSV sibling) so `candidate_detail.csv` matches the parquet — list fields pipe-joined, scalars stringified with `""` for `None`.
- `scripts/download_opentargets.py`: added `associationByDatatypeDirect` to `_DATASETS` with `--skip` support; updated arg help + module docstring.
- `scripts/copy_model_inputs.sh` (new): small driver that copies the four parquets into `<dest>/{candidate_detail,trial_detail}.parquet` and `<dest>/features/{fingerprints,molformer_embeddings}.parquet`. Defaults to `./model_inputs`. Pre-checks each source exists.

### Tests
- `src/tests/test_opentargets_enrichment.py`:
  - Fixture `_make_ot_snapshot` now accepts optional new drug-row columns (7-tuples still work — trailing slots default to `None`), optional `evidence_rows`, and an `include_new_columns=False` switch for backward-compat coverage.
  - `ot_snapshot` fixture extended with tractability + LOEUF + evidence rows for Semaglutide, Insulin (two ChEMBL IDs), Pembrolizumab.
  - New `TestTractabilityLoeufGenetics` (8 cases): tractability attach + union-across-ChEMBL-rows, LOEUF min-aggregation, genetic-score matched/no-match/indication-matched-but-no-evidence cases, any-indication fallback, coverage assertions.
  - New `TestBackwardCompatibility::test_old_schema_still_loads`: older snapshot without new columns/table still loads, new features default to empty/`None`, ledger records `(0, N)` coverage.
- `src/tests/test_snapshot_writer.py`:
  - `_enriched_candidate` populates the five new fields with non-default values.
  - New `test_opentargets_tractability_genetics_roundtrip` + `test_opentargets_genetics_null_when_unset` for parquet round-trip.

## Files touched
- `scripts/build_opentargets_snapshot.py` (extended)
- `scripts/download_opentargets.py` (added associations dataset + `--skip`)
- `scripts/copy_model_inputs.sh` (new, ~30 lines, executable)
- `src/pipeline/models.py` (5 new fields)
- `src/pipeline/enrichment/opentargets.py` (lookups + run loop + coverage)
- `src/pipeline/stages/reporting/_writer.py` (5 new parquet columns; follow-up: same 5 columns added to `write_candidate_detail` CSV writer)
- `src/tests/test_opentargets_enrichment.py` (fixture params + 9 new tests)
- `src/tests/test_snapshot_writer.py` (new field defaults + 2 new tests)
- `TASK.md` (this file; supersedes the previous extraction-task version)
- No edits to `model/` — the new columns sit in `candidate_detail.parquet` but no feature group consumes them yet.

## Commands run
```bash
cd src && ./run_tests.sh --include opentargets_enrichment,snapshot_writer -v   # 52/52 pass
cd src && ./run_tests.sh                                                        # 836/837 (one unrelated pipeline_sampling identity-check failure)
uv run python -c "import ast; ast.parse(open('scripts/build_opentargets_snapshot.py').read()); print('OK')"
uv run python scripts/build_opentargets_snapshot.py --help
uv run python scripts/download_opentargets.py --help
bash -n scripts/copy_model_inputs.sh

# Follow-up (2026-05-26)
uv run python -c "import ast; ast.parse(open('src/pipeline/stages/reporting/_writer.py').read()); print('OK')"
```

## Remaining work
- **To activate end-to-end**, regenerate the OT snapshot then re-run the pipeline:
  ```bash
  uv run python scripts/download_opentargets.py --release 25.03 --dest data/opentargets
  uv run python scripts/build_opentargets_snapshot.py \
      --opentargets-dir data/opentargets/25.03 \
      --chembl-snapshot chembl_targets.sqlite \
      --out opentargets_snapshot.sqlite
  cd src && ./execute.sh    # regenerates outputs/candidate_detail.parquet
  ```
- Optional (not done): add `model/features/genetics.py` so training actually consumes the new columns. Pattern would mirror `model/features/targets.py` (TopKMultiLabel for `opentargets_tractability_modalities`) plus a small NumericGroup for `opentargets_loeuf_min` + `opentargets_genetic_score*`. Until added, the columns ride along in the parquet but no `FeatureGroup` reads them.
- Pre-existing failure in `src/tests/test_pipeline_sampling.py::test_bypass_when_population_le_sample_size`: asserts `out_cands is cands` (identity), but the implementation at `pipeline.py:612` always constructs a fresh `CandidateTable(candidates=sampled)` whenever `max_candidates is not None`. Either fix the implementation to return the original when no truncation happens, or relax the assertion to equality. Out of scope for this task.

## Notes / gotchas
- **Three sibling writers must stay in lockstep when adding Candidate fields**: `write_candidate_detail` (CSV), `write_candidate_parquet` (parquet), and `write_trial_detail`/`write_trial_parquet` (trial-level pair) — all in `src/pipeline/stages/reporting/_writer.py`. The original tractability+genetics task added the five new fields to the parquet writer but missed the CSV writer; that's what the 2026-05-26 follow-up fixed. When extending `Candidate`, grep `_writer.py` for the previously-added neighboring field and add the new field in every place it appears.
- The OT `associationByDatatypeDirect/` dataset is several GB at 25.x — by far the biggest of the five subdirs the downloader now pulls. `--skip associationByDatatypeDirect` is supported on the downloader, and the snapshot builder gracefully emits an empty evidence table if the dir is absent (genetic-score columns will be `None` for all candidates).
- LOEUF score key drifts across OT releases. The snapshot builder tries `oe_ci_upper`, `oeCiUpper`, `oeUpper`, `oe_upper`, `upperBin`, `upper_bin`, `score` in order. If a future release uses none of those, LOEUF comes back `None` rather than crashing.
- Indication matching is name-keyed and case-insensitive against `name_opentargets_indications.indication_name`. The matched-EFO genetic-score join inherits the same match — if the candidate's `indication` doesn't text-match what OT carries (and `mesh_indication` doesn't either), the matched-EFO score stays `None` even when target-level genetic evidence exists. `opentargets_genetic_score_max_any_indication` is the safety valve for the "this gene has genetic backing somewhere" signal.
- Genetic-score aggregation rule is **max across the drug's targets** (best-evidence). Mean would penalize promiscuous drugs; max preserves the strongest signal. If this changes, update both the snapshot writer's evidence-table population and the enrichment's `genetic_by_efo` / `genetic_any` rollup so they stay consistent.
- Backward compatibility: the enrichment uses `PRAGMA table_info` to detect tractability/LOEUF columns and catches `sqlite3.DatabaseError` around the evidence table. An old snapshot built before this work still drives the pipeline — the new features just stay default. So you can deploy the code change before rebuilding the snapshot.
- Bundle parity: `scripts/make_model_bundle.sh` already copies `candidate_detail.parquet` — no bundle changes needed, just rebuild the parquet (step 1 of "Remaining work") before running the bundler.
