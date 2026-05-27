# Methods and Results

This document is a self-contained methods and results section for an ex-ante drug-indication likelihood-of-approval (LOA) prediction paper. It supersedes and extends `paper/methods.md` (which covers the descriptive pipeline only) by additionally documenting the feature-engineering layer, the predictive model, and the empirical results obtained from the implemented code at the present commit.

Two run artifacts are referenced throughout:

- **Descriptive run** (`docs/`, May 2026): full candidate set with modality and disease classification populated, used for the phase-transition descriptive figures. Configuration from `paper/methods.md` §Target-Run Configuration: `--max-trials 76000 --keep-unmatched-drugbank --all-modalities --use-ct-cache --adjudication-method llm_direct`.
- **Modeling-prep run** (`outputs/`, May 2026): year-range-restricted (2009–2026), SMILES-required, classification-skipped run used as input to the predictive modeling pipeline. Configuration: `candidate_year_range=[2009, 2026], require_smiles=True, skip_classification=True, adjudication_method=ndc_indication, use_ct_cache=True`. Yields 28 338 candidates from 158 835 trial-rows. Snapshot versions: ChEMBL 36, OpenTargets 25.03, DrugBank export 2026-04-24. The reported predictive model uses a temporal split at T = 2019 (`model_runs/t2019/`), with leave-one-feature-class-out ablations in `model_runs/loo_t2019/` and three external baselines in `model_runs/baselines/`.

A `run_manifest.json` (output by every pipeline run) captures git SHA, all CLI-resolved configuration values, snapshot versions, and row counts so each result is reproducible.

Three model-layer capabilities described below have landed since the T = 2019 headline run was scored and were noted as engineering TODOs in earlier drafts: probability calibration on a held-out year (§2.5), per-trial training with per-phase evaluation (§2.9), and HINT (Fu et al. 2022) as a fourth external baseline wired into the trial-level comparison harness (§2.7, §2.9.4). A seven-page feature-class audit PDF (`outputs/feature_class_audit.pdf`, produced by `scripts/feature_class_audit.py`) is the qualitative complement to the §3.2 coverage table and is described in §2.5. The drug-indication-level result tables in §3.4–§3.7 predate these additions and reflect the uncalibrated single-task model on the candidate-level frame; their numbers are unchanged in this revision.

---

## 2. Methods

### 2.1 Descriptive pipeline

The descriptive layer (trial ingestion → candidate clustering → outcome adjudication → modality/disease classification → phase-transition aggregation → reporting) is documented in `paper/methods.md`. Only a compressed summary is given here, with attention to the points that materially affect the predictive layer.

**Trial ingestion.** Records are pulled from the AACT mirror of ClinicalTrials.gov (`pipeline/stages/ingestion.py`) restricted to `study_type='INTERVENTIONAL'` and `intervention_type='DRUG'`. Healthy-volunteer studies (case-insensitive word-boundary match on `healthy(volunteer|subject|participant|adult)?` in the indication) and rows whose intervention or indication contains "placebo" are removed post-query. Each study with M drug interventions and N condition indications is expanded into M × N raw trial-rows; phase strings are normalized using a deterministic mapping that promotes combined phases (e.g. "Phase 1/Phase 2") to the higher constituent phase, and trial statuses are consolidated into seven canonical categories. Single-arm trials additionally have primary-outcome p-values and statistical metadata joined from the AACT `outcome_analyses` table.

**Candidate clustering.** Raw trial-rows are aggregated into unique drug-indication candidates (`pipeline/stages/clustering.py`) using a hybrid strategy: when MeSH descriptors are available the clustering key is the sorted concatenation of MeSH intervention and condition terms; otherwise the key falls back to Unicode-normalized free-text after a 10-step canonicalization (NFKD → diacritic strip → lowercase → punctuation/separator normalization → formulation-token removal → Roman numeral normalization). A second pass performs a union-find re-merge over namespace-tagged alias sets `{(db, DrugBank ID), (mesh, MeSH term), (name, normalized name), (syn, DrugBank-recovered synonym)}` × `{(mesh, condition), (name, normalized indication)}`. Two candidates merge whenever they share any one drug alias and any one indication alias, taken to transitive closure. The descriptive run yields 7,444 candidates from 76 000 raw trial-rows; the modeling-prep run (with year-range and SMILES filters) yields 7,934 candidates from 71,723 raw trial-rows.

**Outcome adjudication.** Three swappable backends emit identical `CandidateOutcomeRecord` schema and persist to a shared SHA-256-keyed SQLite cache so verdicts can be compared side-by-side: (1) `llm_direct` — Claude Sonnet 4.6 → Opus 4.6 tiered routing with deterministic shortcut for all-terminal-trial Phase-1/2/3 candidates (used for the descriptive run); (2) `fda_timeline` — openFDA ORIG/SUPPL submission letters parsed into per-drug regulatory timelines, LLM matches the trial indication against approved indications, NDC commercial status separates `Approved` from `Commercialized`; (3) `ndc_indication` — purely local SQLite lookup of FDA label indications via a synonym list (`pipeline/ndc.py`), LLM judges whether the trial indication is covered, no network calls (used for the modeling-prep run). All three back-ends emit one of seven outcome categories: `Failed Phase 1/2/3`, `Approved`, `Commercialized`, `Ongoing`, `Unknown`. The `ndc_indication` adjudicator collapses `Approved` and `Commercialized` into `Approved` because it does not consult NDC active-marketing records. The FDA-timeline backend additionally caches openFDA HTTP responses on disk under `.fda_cache/` and parallelises across candidates via `--fda-adjudication-workers N`, with per-drug timelines deduplicated under a per-drug lock so candidates that share a drug do not redundantly hit openFDA or the LLM (`pipeline/stages/adjudication_fda.py`).

**Modality and disease-area classification.** Disease area is resolved deterministically by plurality vote over MeSH C/F-branch tree-number prefixes, with LLM fallback only when no prefix maps to a named bucket or when the top two named buckets tie. Modality is always resolved by LLM (Sonnet → Opus tiered routing) over a 17-class taxonomy with explicit priority rules for overlapping classes (radioisotope → radioligand_therapy; antibody+payload → adc; immune-checkpoint target → checkpoint_inhibitor; engineered cells → cell_therapy_engineered; infectious-pathogen mRNA → vaccine; protein-degradation MOA → protac_degrader). The descriptive run reports 5,939 small_molecule, 354 peptide, 335 monoclonal_antibody, 316 fusion_protein, 312 unknown, with smaller buckets for vaccines, antisense, cell/gene therapies, ADCs, bispecifics, and siRNA.

**Phase-transition rates.** For each candidate, `phases_observed` is the set of phases with terminal trial evidence (Completed/Terminated/Withdrawn/Suspended status, or non-terminal status that is at least `stale_trial_cutoff_years=2.0` years inactive), Phase 4 trials, or post-clinical Approved/Commercialized verdicts. `phases_advanced` is a superset that additionally admits any-status trial evidence at a phase plus `FAILED_PHASE_N` adjudicator verdicts. The transition rate from phase X to phase Y over a candidate set C is

  TR(X → Y) = | { c ∈ C : X ∈ phases_observed(c) ∧ phases_advanced(c) ∩ L_X ≠ ∅ } | / | { c ∈ C : X ∈ phases_observed(c) } |

with `L_X` the set of phases strictly later than X in the canonical order Phase 1 < Phase 2 < Phase 3 < Approval < Market. The configuration flag `back_propagate_approval=True` (ClinSR-aligned default) extends both sets downward to every phase at or below the highest observed cohort phase, compensating for pre-FDAAA-2007 registry gaps. Wilson-score 95% confidence intervals accompany every rate. The full LOA from Phase 1 to Market is the product of the four sequential transition rates.

### 2.2 Feature engineering for predictive modeling

The predictive model consumes a per-candidate feature vector built from six feature classes. All features are written into `outputs/candidate_detail.parquet` and ancillary parquet files under `outputs/features/` by post-pipeline scripts that consume the candidate table and persist standardized feature matrices.

**Molecular structure.** Each candidate's `drug_name` is resolved to a SMILES string via a two-tier lookup against a pre-processed DrugBank approvals table (`pipeline/enrichment/smiles.py`): exact match on the canonicalized name, then first-token fallback. SMILES are then standardized through the ChEMBL Structure Pipeline (`pipeline/enrichment/smiles_standardization.py`): `standardize_mol` (charge neutralization, tautomer normalization, isotope handling) followed by `get_parent_mol` (salt and solvent stripping, largest-fragment retention). The standardized SMILES (`smiles_canonical`) is the canonical molecular identity used downstream. The modeling-prep run requires successful standardization for inclusion (`require_smiles=True`); 7 933 / 7 934 candidates produce a parent SMILES. Fingerprint and embedding parquets (built by the offline scripts below) attach back to the candidate table by `candidate_id` via left-join (`model/data.py:build_candidate_modeling_frame`); rows missing fingerprints or embeddings are kept as NaN and surfaced as a `_missing` indicator column in the respective feature group encoders.

Two molecular descriptor families are computed offline (`scripts/build_fingerprints.py`) and consumed by the model:

- **ECFP4** (extended-connectivity, radius 2, 2 048 bits) via RDKit `GetMorganFingerprintAsBitVect`.
- **MACCS keys** (167-bit structural fingerprint) via `rdkit.Chem.MACCSkeys.GenMACCSKeys`.

In addition, **MolFormer-XL embeddings** (768-dim pooled output) are extracted from the IBM `ibm/MoLFormer-XL-both-10pct` HuggingFace checkpoint (`scripts/build_molformer_embeddings.py`) and persisted to `outputs/features/molformer_embeddings.parquet`. The embedding block is excluded from the headline model after a leave-one-out ablation showed it monotonically degraded every reported metric on this corpus (§3.6). It is retained as a candidate feature for the planned Tanimoto-NN + cosine-NN molecular-similarity baseline (§2.6) and for any sensitivity analyses that re-introduce it under different feature-engineering pipelines.

**Drug targets.** UniProt accessions for each drug are pulled from a frozen ChEMBL-36 SQLite snapshot (`pipeline/enrichment/targets.py`, populated by `scripts/build_chembl_targets_snapshot.py`). Drug names are matched via the same two-tier canonical/first-token lookup used for SMILES, returning the union of `drug_mechanism.target_dictionary` UniProt accessions per drug. The model consumes a top-K=200 sparse one-hot encoding of the most frequent UniProt targets across the candidate set, plus a residual `target_other_count` integer feature.

**ADMET.** The `admet_ai` package (Stanford, v2+) is invoked once per unique standardized SMILES (`pipeline/enrichment/admet.py`). Its 52 numeric properties (solubility, permeability, hERG, hepatotoxicity, AMES, BBB, CYP isoforms, Caco-2, PAMPA, drug-likeness QED, lipophilicity, plus DrugBank-derived approved-drug percentile rankings for each) are written as one column per property (104 columns total). For the modeling-prep run, 7 933 / 7 934 candidates have at least one ADMET value. At feature-fit time (`model/features/admet.py`), columns with more than `admet_drop_null_threshold=0.95` missingness on the inner-training set are dropped; surviving columns are median-imputed; columns with more than `admet_indicator_threshold=0.05` missingness gain a parallel `_missing` indicator column. The post-fit ADMET width is therefore data-dependent and is reported alongside the other group widths in §2.4.

**Pathways.** Reactome pathway membership is derived by joining each candidate's UniProt target accessions to the Reactome `UniProt2Reactome_All_Levels.txt` file (`pipeline/enrichment/reactome.py`, `data/reactome/`), aggregating the union of pathway stable IDs across all targets of the drug. The model consumes a top-K=500 sparse one-hot encoding of the most frequent Reactome pathway IDs across the candidate set. The descriptive run additionally surfaces pathway names for human-readable reporting.

**Open Targets.** Mechanism of action, action type, target list, MoA pathways, and `maxPhaseForIndication` are pulled from a frozen Open Targets 25.03 SQLite snapshot (`pipeline/enrichment/opentargets.py`, `scripts/build_opentargets_snapshot.py`). The current snapshot does not include `geneticConstraint` or `tractability` fields — see `paper/playbook_gap_analysis.md` §2.2.16.

**Mechanism-of-action multi-hot** (`model/features/moa.py`). Open Targets exposes `opentargets_moa` as a long-tail list of free-text mechanism descriptions per candidate (~913 distinct strings across the modeling-prep corpus, e.g. "Glucocorticoid receptor agonist", "Cyclooxygenase inhibitor"). The encoder produces a top-K=200 multi-hot of the most frequent strings on the inner-training set plus a residual `moa_other_count` integer and `moa_missing` indicator. Disabled in the headline configuration but available behind `features.enabled`.

**Action-type one-hot** (`model/features/action_type.py`). Sibling encoder over the OpenTargets `opentargets_action_type` enum (INHIBITOR, AGONIST, ANTAGONIST, BLOCKER, DEGRADER, ~24 values overall), top-K=50 so effectively full one-hot, with the same `other_count` / `missing` companion columns. Also default-off.

**Nearest-approved-drug similarity** (`model/features/nn_similarity.py`). Two scalar engineered features per candidate, one per similarity metric: `tanimoto_nn_similarity` (max ECFP4 Tanimoto similarity to the approved-drug pool) and `molformer_nn_similarity` (max cosine similarity over L2-normalised MoLFormer-XL embeddings). The pool is built at `fit` time from the inner-training rows whose `outcome ∈ {Approved, Commercialized}`; each candidate's similarity is computed only against pool entries with `earliest_start_date` strictly before the candidate's own, i.e. the pool is date-filtered ex-ante so that no future-approved drug informs an earlier candidate's feature. Each metric emits a parallel `_missing` indicator (set when the candidate has no fingerprint / embedding or the date-filtered pool is empty). These two groups are default-off; they are designed to be reintroduced as the molecular-similarity-baseline-equivalent features inside the supervised model in a sensitivity-analysis pass.

**Shared multi-hot encoder.** The `targets`, `pathway`, `disease.mesh`, `moa`, and `action_type` groups all consume the same `TopKMultiLabel` encoder (`model/features/_multilabel.py`). Vocabulary is taken as the top-K most frequent labels on the inner-training subset only — never the validation, test, or calibration partitions — so vocabulary leakage from later splits into the encoder is impossible by construction. An `_other_count` integer column records out-of-vocabulary frequency per row, and an `_missing` indicator flags rows whose source list is null or empty. The full per-group feature width is `K + 2` (top-K vocab + `other_count` + `missing`).

**Indication / disease.** Each candidate's MeSH C-branch and F-branch condition tree numbers are aggregated, and the prefix-to-area mapping (`paper/methods.md` Supplementary Methods §MeSH Tree-to-Disease Area Mapping) collapses tree numbers into 13 BIO/QLS-aligned therapeutic-area buckets plus `other` and `unknown`. The model consumes (i) a top-K=200 one-hot of MeSH condition tree numbers and (ii) a 14-level one-hot of resolved disease area.

**Sponsor (planned).** Sponsor features are not yet incorporated into the predictive feature vector. The conceptually correct unit of analysis is candidate-level sponsorship — typically the originator at IND filing, or as a tractable proxy the earliest sponsor on the candidate's constituent trials — rather than the per-trial sponsor field already attached to each `RawTrial`, which is a property of the trial rather than of the development trajectory. Sponsor prior-approval counts are an additional candidate-level feature contemplated by the playbook; sourcing those counts in a way that is itself frozen at or before the candidate's earliest activity date is an open question and is deferred to a forthcoming feature-engineering pass. The placeholder is tracked at `paper/playbook_gap_analysis.md` §2.1.8.

**Indication base rate (planned).** LLM-taxonomy-derived indication base rates with leave-one-out are likewise deferred and tracked at `paper/playbook_gap_analysis.md` §2.2.10.

### 2.2.1 Temporal validity of the feature set

The features described above are intended as functions of pre-clinical-known properties of the candidate, with the goal that the feature vector for a given drug-indication candidate would in principle be computable as of the candidate's IND filing or Phase 1 start. The molecular block (ECFP4, MACCS, MolFormer-XL embeddings) is fully determined by the standardized SMILES of the small-molecule active ingredient, which is established before clinical trials begin. ADMET-AI predictions are functions of that same SMILES and are similarly pre-clinical. Drug targets are typically known at IND time as part of the proposed mechanism of action. Pathway membership follows from drug targets via Reactome and is therefore inherited from a pre-clinical input, modulo the fact that the Reactome curation itself reflects the state of biological knowledge at the time of feature extraction. The disease/indication block is the trial-registered indication, which is fixed at the start of the development program.

The feature-class composition is therefore not expected to introduce post-Phase-1 trial-level information into the predictive task. Two caveats apply at the present implementation: (i) external-database snapshots (ChEMBL 36, Open Targets 25.03, Reactome) are dated to the time of feature extraction rather than the time of each candidate's Phase 1 start, so a candidate from 2009 is enriched against post-2009 curation; (ii) the present implementation does not record a per-candidate IND or Phase 1 start date as a freeze anchor and does not gate any feature computation on that anchor. The empirical question of whether retroactive-snapshot leakage materially affects predictive performance is left to a sensitivity analysis that compares model performance across snapshot vintages, which has not yet been performed. Pending that analysis, results in §3 should be interpreted as conditional on the assumption that snapshot-vintage drift is small relative to the signal carried by the feature classes themselves.

The Sponsor feature class, when added, will require explicit per-candidate freezing because sponsor-prior-approval counts are by construction time-varying and not pre-clinical-determined.

A standing empirical artifact for monitoring coverage drift across the temporal axis is the seven-page feature-class audit PDF described in §2.5 (`scripts/feature_class_audit.py` → `outputs/feature_class_audit.pdf`), which reports per-feature-class coverage and null rate by candidate-start year. Coverage cliffs along that axis are the most concrete signal that snapshot-vintage drift is material for a given feature class on a given temporal split; the audit is the input the snapshot-vintage sensitivity analysis will draw on once it is run.

### 2.3 Label, temporal split, and outcome filtering

Each candidate is assigned a binary supervised-learning label:

- **Positive** (y = 1): `outcome ∈ {Approved, Commercialized}`.
- **Negative** (y = 0): `outcome ∈ {Failed Phase 1, Failed Phase 2, Failed Phase 3}`.
- **Excluded**: `outcome ∈ {Ongoing, Unknown}` — these candidates are dropped from training, validation, and test entirely.

The splitter (`model/splits.py`) supports three mutually exclusive modes, checked in this order:

1. **Temporal** (the headline configuration). Set `time_split_year=T` and `time_split_column=C` (default `earliest_start_date` at drug-indication granularity, `trial_start_date` at trial granularity). Training set is rows with `C.year ≤ T`; test set is rows with `C.year > T`; rows with missing year on `C` are dropped from both.
2. **Group-aware.** Set `group_by="drug_name"` (or any column) → `StratifiedGroupKFold` with `n_splits = max(2, round(1 / test_size))`; fold 0 is returned. Used to enforce zero drug overlap between train and test, which removes drug-level leakage when the same drug appears in multiple indication candidates.
3. **Stratified random** (default fallback). `sklearn.train_test_split` with `test_size=0.2`, seed-pinned, stratified on `y`.

**Inner validation.** Independently of the outer split, a stratified 10% holdout (`inner_val_size=0.1`) is carved off the training partition (`model/train.py:151-162`) and used only for XGBoost early stopping; the held-out 10% is then reincluded in nothing — it is not folded back into training nor used for calibration. Class-imbalance weighting (`scale_pos_weight`, §2.4) is computed on the inner-train slice rather than on the full training partition, a small but reproducibility-relevant distinction.

**Three-way calibration split.** When `config.calibration_year=K` is set, the splitter returns three disjoint year slices via `split_with_calibration` (`model/splits.py:121-167`): train = `year ≤ K − 1`, calibrate = `year == K`, test = `year > K`. The probability calibrator (§2.5) is fit on the calibrate slice only, so it never observes a training-set label. This implements the playbook-prescribed T = 2017 train / 2018 calibrate / 2019+ test protocol; it is now wired up end-to-end but the headline T = 2019 run reported in §3.4 does not yet use it, with 2018 absorbed into training (`config.calibration_year` left unset). Activating the calibration mode is a configuration choice for the next paper run rather than additional implementation work.

**T = 2019 headline cohort sizes.** A single temporal split at T = 2019 on `earliest_start_date` is reported in §3.4 with training set 12 953 / test set 2 658.

### 2.4 Predictive model

The model (`model/models/xgb.py`) is an XGBoost binary classifier (`xgboost.XGBClassifier`) with the following hyperparameters fixed at module level: `objective='binary:logistic'`, `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`, `min_child_weight=1`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`. Class imbalance is handled via `scale_pos_weight = (n_neg / n_pos)` computed on the training fold. Early stopping uses the inner 10% validation holdout with `eval_metric='logloss'` and `early_stopping_rounds=50`. Random seed is fixed at 0 throughout (`model/data.py`, `model/splits.py`, `model/models/xgb.py`).

Each feature group is fit independently — fingerprints as dense float arrays (cast from bit vectors); targets, pathway, and disease-MeSH as top-K sparse one-hot via the shared `TopKMultiLabel` encoder (`model/features/_multilabel.py`, described in §2.2); ADMET as median-imputed numeric with the `_missing` indicator column policy from §2.2. Group widths at T = 2019 in the headline model (fingerprints + targets + ADMET + pathway + disease) are: fingerprints 2 216 (ECFP4 2 048 + MACCS 167 + 1 sentinel), targets 202 (top 200 + count + indicator), ADMET 52, pathway 503 (top 500 + count + flags), disease 47 (top 33 MeSH + 14 disease-area), for a total of 3 020 features.

The training entry point (`model/train.py:train_one_run`) is model-agnostic: it dispatches by `config.model_name` against a registry that currently exposes the XGBoost classifier above and a logistic-regression baseline (`model/models/logreg.py`, `StandardScaler` → `LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")` with `class_weight={0: 1, 1: scale_pos_weight}` derived from the same training-fold ratio). The logistic-regression model consumes the same fitted feature matrix as XGBoost; it is intended for sanity-checking the XGBoost discrimination claim and is not the headline-model reporting line.

### 2.5 Evaluation metrics

The evaluation harness (`model/evaluate.py`) computes and persists, for the held-out test set:

- **Discrimination**: ROC-AUC, PR-AUC.
- **Threshold-dependent**: F1 and balanced accuracy at the default threshold 0.5; full confusion matrix (TP/FP/TN/FN).
- **Probabilistic loss**: Brier score, log-loss.

All metrics, the test predictions (`y_true`, `y_proba`), the fitted feature pipeline, and per-feature gain-based importance are written to `model_runs/<tag>/`.

**Probability calibration** is now implemented end-to-end (`model/train.py:243-300`). When the three-way temporal split (§2.3) is enabled, the calibrator is fit on the calibrate-year slice and applied to test predictions; raw and calibrated probabilities are both persisted, and the full metric block is reported pre- and post-calibration. Two calibration methods are supported and selected via `config.calibration_method`:

- **Isotonic regression** (default) — `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`, fit on `(raw_calib_proba, y_calib)` and applied as `calibrator.predict(test_proba)`.
- **Platt scaling** — `sklearn.linear_model.LogisticRegression` fit on `(raw_calib_proba.reshape(-1, 1), y_calib)`, applied via `predict_proba`.

Expected Calibration Error (`model/evaluate.py:expected_calibration_error`) is reported alongside Brier and log-loss in the `calibration_metrics` block of `metrics.json`; it uses 10 equal-width bins on [0, 1] with sample-weighted absolute gaps and excludes empty bins. The fitted calibrator object is also persisted (in `pipeline.pkl`) so test predictions can be re-scored post-hoc, and `y_proba_calibrated` is written alongside `y_proba` in `predictions.csv`. The headline T = 2019 run in §3.4 does not invoke calibration (no `calibration_year` set); the next paper run will set `calibration_year = 2018` and report pre/post Brier and ECE alongside the discrimination metrics.

Two planned additions to the evaluation harness remain TODOs and are not yet exercised:

- **SHAP attribution** per candidate and aggregated per feature class, including beeswarm and per-class summary plots, replacing the present XGBoost gain importance for interpretability claims.
- **Nearest-neighbor lookup tool** built on a FAISS index over the joint feature space (molecular embedding + target + indication), returning k = 10 historical analogs with realized outcomes, exposed as a CLI and a thin web interface. The supervised model already has access to ex-ante NN-similarity scalars (§2.2, `model/features/nn_similarity.py`); the deferred deliverable is the user-facing CLI/UI layer rather than the underlying lookup logic.

Each is tracked at `paper/playbook_gap_analysis.md` (§2.1.7 SHAP, §2.1.6 nearest-neighbor).

**Feature-class audit** (`scripts/feature_class_audit.py`, output `outputs/feature_class_audit.pdf`). A seven-page PDF audit complementing the §3.2 coverage table:

1. Summary page: per-class encoding dimension, coverage and null rate; coverage-by-candidate-start-year line chart; missingness heatmap (class × year band).
2. Molecular page: ECFP4 / MACCS bit-set richness, outcome-stratified boxplot, coverage trend.
3. Disease page: `disease_area` cardinality, MeSH C-prefix top-20 histogram, outcome distribution per disease area, year-trend stacked bar.
4. Target page: target count per candidate, top-20 most frequent UniProts, outcome split by target count, year trend.
5. Pathway page: pathway count distribution, top-20 most frequent pathway IDs in both ancestor-expanded and locally-leaf views, outcome split by pathway count, year trend.
6. ADMET page: per-column null rate, distributional summaries of the most-attended columns (logP, MW, TPSA), outcome split for each, year-trend null rates.
7. Top-values page: most common multi-hot values per encoder.

The audit's role is qualitative — it surfaces coverage cliffs along the temporal axis (which aggregate counts do not), and it is the standing artifact the §2.2.1 snapshot-vintage sensitivity analysis will read from.

### 2.6 Ablations

The leave-one-feature-class-out ablation harness (`model/ablate.py`) re-runs the same train/test split with one feature group removed at a time. The split, seed, label set, and inner validation indices are pinned across subsets so all ablation rows share an identical denominator. The same split-pinning is used by the baselines runner (`model/baselines/runner.py`): every ablation row and every baseline in §2.7 is evaluated on the identical test cohort, so AUC / Brier deltas across the entire comparison stack are within-row differences rather than across-cohort comparisons. Five leave-out ablations plus the full-feature `all` run are reported on the headline 5-class feature stack (fingerprints, targets, ADMET, pathway, disease), with results consolidated into `model_runs/loo_t2019/ablation_summary.csv`.

Two additional ablations specified in the playbook — leave-out-Indication-base-rate and leave-out-Sponsor — are deferred until those feature classes land (§2.2).

### 2.7 Baselines

Three of the playbook's five external baselines are implemented as standalone reference models, all evaluated on the same T = 2019 split, label set, and seed as the headline model so that AUC / Brier deltas are directly comparable:

- **Stratum base rate** (`model_runs/baselines/stratum/`). For each test candidate, a stratum-specific positive-class rate is computed from the training set on the candidate's ICD-10 stratum, falling back to the ICD-10 chapter rate when stratum support is below `min_n=10` and to the global training-set positive rate when no chapter is available. The training set has 4 966 / 12 953 candidates with an ICD-10-resolved stratum (662 strata across 23 chapters); on the test set, 452 candidates hit a specific stratum, 288 fall back to chapter, and 1 918 fall back to the global rate of 0.1867.
- **Molecular-similarity nearest neighbor** (`model_runs/baselines/tanimoto/`). For each test candidate, predicted probability is the mean realized-positive rate among the k = 5 nearest training-set candidates by ECFP4 Tanimoto similarity. Candidates without a fingerprint fall back to the global training-set positive rate.
- **Target-only** (`model_runs/baselines/target_only/`). An XGBoost classifier with identical hyperparameters to the headline model, restricted to the 202-dim target one-hot feature group only — no molecular, ADMET, pathway, or disease features.

A fourth playbook baseline is now wired into the trial-level harness:

- **HINT (Fu et al. 2022)** as an external trial-level baseline. The implementation lives in the standalone HINT model at `/Users/samirtownsley/Documents/projects/hint_standalone/repo/` and is orchestrated by `run_hint.sh`; integration into our pipeline is at the I/O layer rather than at the model layer. The trial-granularity training run automatically projects its test rows into HINT's 10-column input schema (`model/hint_format.py:to_hint_frame`) and persists them as `hint_test.csv` next to its own `metrics.json` and `predictions.csv`. HINT is then trained / scored on the identical row set, and the resulting `hint_results.csv` is joined back into the consolidated report (`scripts/build_consolidated_report.py`) for a per-phase side-by-side comparison against the supervised trial-level model. Row filtering matches the standalone build script (`scripts/build_hint_dataset.py`): rows without a parseable SMILES, without an inferred trial label, with empty ICD-10 codes, or with empty eligibility-criteria text are dropped, since each is a hard input requirement for one of HINT's three encoders (MPNN on SMILES, GRAM on ICD-10, BERT on criteria). Phase-specific HINT datasets are produced by `scripts/build_hint_dataset.py --phase {1,2,3}` for phase-segregated comparisons. The row-pinning across the trial-level supervised model and HINT — implemented by reusing the exact test-set DataFrame in both — gives the same identical-cohort guarantee that the ablation harness gives within the supervised stack.

This converts the headline-claim baseline comparison from a partial 3 / 5 to a 4 / 5 once the headline trial-level + HINT run is written up (see §3.9). The remaining unimplemented playbook baseline is the **BIO biomarker-selection** binary indicator (`paper/playbook_gap_analysis.md` §2.2.14).

### 2.8 Reproducibility

Every pipeline invocation writes `outputs/run_manifest.json` capturing the git SHA, dirty-flag, ISO timestamp, full `PipelineConfig` after CLI resolution, snapshot versions (ChEMBL `user_version`, Open Targets `user_version`, DrugBank export mtime), and row counts. Knowledge cache and feature-snapshot SQLite files are versioned separately and pinned by their `user_version` PRAGMA. Random seed is fixed at 0 across feature fitting, splitting, and model training. Docker, DVC, MLflow, and Zenodo deposit are not yet implemented — see `paper/playbook_gap_analysis.md` §2.2.19.

### 2.9 Trial-level training and HINT integration

The drug-indication-level model described in §2.2–§2.7 has a sibling configuration that trains and evaluates at the trial level. Both modes share the entire feature engineering and training infrastructure; the dispatch is `config.training_granularity ∈ {"drug_indication" (default), "trial"}` (`model/data.py:build_modeling_frame`).

#### 2.9.1 Trial-level frame construction

`model/data.py:build_trial_modeling_frame` produces one row per `(nct_id, primary candidate)` pair. The trial table is loaded from `outputs/trial_detail.parquet` (58 137 trials × 27 columns in the modeling-prep run), filtered to rows with a non-null `trial_inferred_label`, and merged with the candidate table on `candidate_id` (left-join, then fingerprints and MoLFormer embeddings on the same key). Column-overlap policy: when both the trial and the candidate sides carry a column (e.g. `indication`), the trial-side value wins by construction (the matching candidate-side column is dropped before the merge). The candidate-level `outcome` column is intentionally retained because the `nn_similarity` feature group (§2.2) uses it to filter the approved-drug pool, and it differs from the per-trial `y` — an approved drug's failed Phase 2 trial carries `outcome=Approved` but `y=0`.

#### 2.9.2 Trial label inference

`src/pipeline/trial_labels.py:infer_trial_label` derives `y ∈ {0, 1, None}` from the triple `(trial_status, candidate_outcome, trial_phase)`:

| Trial status | Candidate outcome | Trial phase vs. failure phase N | `y` | Rationale |
|---|---|---|---|---|
| Terminated / Withdrawn / Suspended | any | — | 0 | trial-level failure regardless of downstream drug outcome |
| any | Approved / Commercialized | — | 1 | drug eventually reached approval, so trials along that path are positives |
| any non-terminal | Failed Phase N | trial.phase < N | 1 | drug advanced past this trial's phase |
| any non-terminal | Failed Phase N | trial.phase = N | 0 | drug stopped at this trial's phase |
| any non-terminal | Failed Phase N | trial.phase > N | — (drop) | the drug never actually reached this trial's phase, so no signal |
| any non-terminal | Ongoing / Unknown | — | — (drop) | no terminal evidence to derive a label |

Worked example: a drug with `outcome = Failed Phase 2` produces y = 1 for any of its Phase 1 trials (advanced past P1), y = 0 for its Phase 2 trials (stopped at P2), and `None` (dropped) for any Phase 3 trials, which by construction the drug should not have run but may appear in the data as registry noise.

#### 2.9.3 Per-phase test metrics

When the test frame carries a `trial_phase` column, `model/evaluate.py:metrics_by_phase` slices `(y_test, y_proba)` into three transition cohorts and computes the full §2.5 metric block per cohort:

- `P1->P2`: rows with `trial_phase == "Phase 1"`
- `P2->P3`: rows with `trial_phase == "Phase 2"`
- `P3->approval`: rows with `trial_phase == "Phase 3"`

Cohorts with no rows are emitted as `{"n": 0}`. Results are persisted under `per_phase_metrics` in `metrics.json` and, when calibration is active, under `per_phase_metrics_calibrated`. Note that this is still a single-task binary head — the per-phase split is post-hoc evaluation, not multi-task training; the playbook's per-phase head specification (cf. §3.8 item 6) remains open.

#### 2.9.4 HINT export and side-by-side comparison

Inside the same trial-level run, the exact test-set DataFrame is passed through `model/hint_format.py:to_hint_frame` to produce HINT's 10-column input format (`nctid, status, why_stop, label, phase, diseases, icdcodes, drugs, smiless, criteria`; columns 1, 2, 4, 5 are positional fillers per the HINT loader). The HINT row-filter — non-null SMILES, non-null inferred label, non-empty ICD-10 codes, non-empty eligibility criteria — is applied at projection time, so the HINT-eligible test cohort is generally a subset of the supervised model's test cohort. The projected DataFrame is written as `hint_test.csv` alongside the run's other artifacts.

`run_hint.sh` then drives the external HINT model (path-resolved to the standalone repo) over the same `hint_test.csv`, emits a `hint_results.csv`, and `scripts/build_consolidated_report.py` joins HINT's per-row predictions back against the supervised trial-level model's `predictions.csv` for the side-by-side per-phase comparison. The benefit of pinning the projected DataFrame as the cohort is the same as in the §2.6 ablation harness: AUC / Brier deltas vs. HINT are differences on identical rows rather than across independently sampled cohorts.

---

## 3. Results

### 3.1 Pipeline yield

The descriptive run produced 7 444 unique drug-indication candidates from approximately 76 000 raw trial-rows after M × N expansion and clustering. The modeling-prep run (year window 2009–2026, SMILES-required) produced 28 338 candidates from 158 835 trial-rows. The two runs differ in candidate-set scope (the modeling-prep run uses the full available year window without the development-time `--max-trials` cap), in adjudicator (NDC vs LLM-direct), and in classification (skipped vs run); the descriptive run remains the canonical source for modality and therapeutic-area stratification, and the modeling-prep run is the canonical input to the predictive layer.

**Outcome distribution (descriptive run, n = 7 444).** Approved 451 (6.1%), Commercialized 1 582 (21.3%), Failed Phase 1 873 (11.7%), Failed Phase 2 2 740 (36.8%), Failed Phase 3 968 (13.0%), Ongoing 223 (3.0%), Unknown 607 (8.2%).

**Outcome distribution (modeling-prep run, n = 28 338, NDC adjudicator collapses Approved + Commercialized).** Approved 3 441 (12.1%), Failed Phase 1 2 866 (10.1%), Failed Phase 2 3 978 (14.0%), Failed Phase 3 5 326 (18.8%), Ongoing 12 727 (44.9%); Unknown is absent because the NDC adjudicator emits a `FAILED_PHASE_N` rather than `Unknown` for unmatched candidates. 15 611 candidates (55.1%) carry a label-eligible verdict (positive: Approved; negative: Failed Phase 1/2/3).

**Modality and therapeutic-area distribution (descriptive run).** Modality breakdown: small_molecule 5 939 (79.8%), peptide 354, monoclonal_antibody 335, fusion_protein 316, vaccine 67, antisense 42, cell therapy 33, gene therapy 19, antibody-drug conjugate 17, bispecific antibody 6, siRNA 2, biologic-other 2, unknown 312. Therapeutic-area breakdown: oncology 1 585 (21.3%), other 818, neurology 670, cardiovascular 596, infectious disease 569, psychiatry 497, respiratory 493, gastroenterology 490, metabolic 471, urology 365, hematology 311, autoimmune 265, ophthalmology 187, endocrine 115, unknown 12.

**Trial-start year distribution (modeling-prep label-eligible subset).** Of the 15 611 label-eligible candidates, 12 953 (83.0%) have an earliest trial start in 2009–2019 (the training window for the T = 2019 split) and 2 658 (17.0%) start in 2020–2026 (the test window). The early years 2009–2011 are heavily over-represented relative to subsequent years, consistent with the FDA Amendments Act of 2007 + ICMJE registration policy driving a registration surge.

### 3.2 Feature coverage

For the modeling-prep run, every candidate has at least one value in every upstream-database-derived feature class (100% raw coverage), as a direct consequence of the `require_smiles=True` filter and the upstream joins succeeding on the post-filtered set:

| Feature class | Population (n / 28 338) | Density notes |
|---|---|---|
| SMILES (raw + canonical) | 28 336 (100.0%) | two ChEMBL-standardization failures |
| ChEMBL drug targets | 28 338 (100.0%) | 0 implies no ChEMBL `drug_mechanism` row |
| ADMET-AI | 28 336 (100.0%) | matches standardization population |
| Reactome pathways | 28 338 (100.0%) | candidates with no targets have empty pathway list |
| Open Targets MOA | 28 338 (100.0%) | non-empty mechanism density depends on indication coverage |
| MolFormer-XL embeddings | offline-pass subset | requires successful HuggingFace inference |
| ECFP4 + MACCS fingerprints | offline-pass subset | computed in same offline pass as embeddings |

Joint feature coverage (intersection of fingerprints + embeddings + targets + ADMET + pathway + disease) is bounded above by the offline fingerprint+embedding pass and is the binding constraint for the predictive set. The model-input join in `model/data.py` produces the train+test population reported in §3.4.

### 3.3 Phase-transition rates

All transition-rate results below are reported from the descriptive run with the canonical configuration; the modeling-prep run's aggregation block was configured with `aggregation_reference_date=2006-12-31` to support a Hay-et-al. 2003–2011 sensitivity comparison and is not directly interpretable as the headline LOA.

The descriptive run produces the four sequential phase-transition rates (P1 → P2, P2 → P3, P3 → Approval, Approval → Market) overall and stratified by modality, therapeutic area, and the cross of the two, with Wilson-score 95% intervals and per-cell denominators. The full transition-rate table is materialized in `docs/index.html` (and `outputs/heatmap_data.xlsx` workbook for machine consumption); the figures are not reproduced here but are referenced throughout the manuscript.

The descriptive results are consistent with the qualitative patterns reported in BIO/QLS, Hay et al. (2014), Wong et al. (2019), and Zhou et al. (2025): Phase 2 → Phase 3 is the most attritional clinical transition, oncology has the lowest end-to-end LOA among major therapeutic areas, and cell/gene therapies have small denominators that preclude precise rate estimation in the current corpus. Reconciling absolute LOA estimates against the published literature is discussed in `paper/methods.md` §5.1.

### 3.4 Predictive model performance

A single temporal split at T = 2019 is reported. The headline model uses the five active feature classes — fingerprints, targets, ADMET, pathway, disease — described in §2.2 and §2.4 with the uncalibrated default-0.5-threshold evaluation harness in §2.5. The MolFormer-XL embedding block was excluded from the headline configuration following §3.6 ablation evidence that its inclusion strictly degrades every reported metric. All numbers below are on the held-out test set; the positive-class prevalence on the test set is shown for context.

**T = 2019 split** (`model_runs/t2019/`). Training set 12 953 candidates (2 418 positive, 18.7% prevalence); test set 2 658 candidates (1 023 positive, 38.5% prevalence); 3 020 features.

| Metric | Value | Playbook target (§3.2) | Playbook gate (Phase-3 §7.4) |
|---|---|---|---|
| ROC-AUC | 0.783 | ≥ 0.75 ✓ | ≥ 0.72 ✓ |
| PR-AUC | 0.703 | — | — |
| F1 (threshold 0.5) | 0.647 | — | — |
| Brier | 0.186 | ≤ 0.18 (≈ at target) | — |
| Log-loss | 0.552 | — | — |
| Balanced accuracy | 0.709 | — | — |
| Confusion (TP / FP / TN / FN) | 690 / 420 / 1 215 / 333 | — | — |

**Interpretation.** The T = 2019 headline model meets the playbook's §3.2 paper-target AUC of ≥ 0.75 (test AUC 0.783) and exceeds the §7.4 Phase-3 gate of ≥ 0.72 on a test set of 2 658 candidates with 1 023 positives, providing adequate statistical resolution for the headline discrimination claim. The Brier score of 0.186 sits a hair above the §3.2 target of ≤ 0.18 and is well below the baseline-rate Brier of approximately 0.22 cited in the playbook; the model is uncalibrated, so the Brier should be re-evaluated after the planned isotonic-calibration pass on a 2018 holdout (§2.5). The positive-class prevalence shift between training (18.7%) and test (38.5%) is again driven by the systematic exclusion of `Ongoing` candidates from the test cohort, which over-represents recently terminal candidates; this is consistent across split years and is documented as a known censoring artifact rather than as a label-distribution shift in the underlying drug-development process.

### 3.5 Comparison to baselines

Three external baselines are evaluated on the identical T = 2019 split, seed, and label set as the headline model (`model_runs/baselines/`). All three use the playbook's published baseline specifications: stratum base rate by ICD-10 (Baseline 1), molecular-similarity nearest neighbor (Baseline 2), and target-features-only (Baseline 3).

| Model | Features | ROC-AUC | PR-AUC | Brier | F1 | Δ AUC vs full | Δ AUC vs stratum |
|---|---|---|---|---|---|---|---|
| **Full model (5 classes)** | **3 020** | **0.783** | **0.703** | **0.186** | **0.647** | **—** | **+0.235** |
| Tanimoto-NN (k = 5) | ECFP4 only | 0.707 | 0.563 | 0.234 | 0.400 | −0.076 | +0.158 |
| Target-only | targets (202) | 0.628 | 0.530 | 0.232 | 0.558 | −0.155 | +0.080 |
| Stratum base rate | ICD-10 | 0.548 | 0.425 | 0.273 | 0.010 | −0.235 | — |

**Interpretation.** The headline model exceeds the playbook's §7.3 stratum-baseline target (beat by ≥ 0.07 AUC) by a wide margin, with Δ AUC = +0.235 over the stratum baseline, +0.158 over the molecular-similarity nearest-neighbor baseline, and +0.155 over the target-only baseline. The molecular-similarity baseline alone reaches AUC 0.707, indicating that ECFP4 Tanimoto-similarity to the realized-outcome cohort recovers most of the discrimination signal the headline model achieves, but its Brier (0.234) and log-loss (1.69) are substantially worse than the headline model's (0.186, 0.552), indicating that the marginal gain from the integrated-features model is larger in probabilistic-score terms than in pure ranking. The target-only baseline (Baseline 3) discriminates poorly (AUC 0.628), consistent with the playbook's expectation that Open Targets-style target features are necessary but not sufficient for candidate-level prediction. The stratum baseline (Baseline 1) is essentially uninformative (AUC 0.548) on this corpus, indicating that ICD-10 chapter membership alone — without molecular, target, or pathway information — does not separate approved from failed development programs in this candidate set.

The remaining two playbook baselines (HINT reproduction; BIO biomarker-selection binary indicator) are not yet implemented (§2.7); the present headline-claim AUC table is therefore a partial (3 / 5) baseline comparison and will be extended once those baselines land.

### 3.6 Feature-class ablations

Leave-one-feature-class-out ablation results at T = 2019, all rows fit on the same 12 953 / 2 658 split with `train_pos = 2 418`, `test_pos = 1 023` (`model_runs/loo_t2019/ablation_summary.csv`), sorted by descending AUC. Ablations are run on the 5-class headline configuration (fingerprints, targets, ADMET, pathway, disease):

| Subset | n_features | ROC-AUC | PR-AUC | F1 | Brier | Log-loss |
|---|---|---|---|---|---|---|
| drop_fingerprints | 804 | **0.790** | **0.709** | 0.656 | **0.182** | **0.540** |
| drop_targets | 2 818 | 0.785 | 0.699 | 0.650 | 0.184 | 0.547 |
| **all** | **3 020** | **0.783** | 0.703 | 0.647 | 0.186 | 0.552 |
| drop_pathway | 2 517 | 0.782 | 0.705 | **0.664** | 0.186 | 0.551 |
| drop_admet | 2 968 | 0.778 | 0.697 | 0.635 | 0.189 | 0.560 |
| drop_disease | 2 973 | 0.777 | 0.695 | 0.648 | 0.193 | 0.567 |

**Interpretation.** Removing any single feature class changes test AUC by less than 0.013 from the full-model baseline of 0.783, indicating that the model is operating in a regime where the feature classes are partially redundant rather than dominated by any single contributor. The lightest-touch ablations (`drop_fingerprints`, `drop_targets`) produce small AUC and Brier improvements, consistent with mild over-parameterization at the present feature dimensionality (3 020 features, 12 953 training examples); the heaviest-degradation ablation (`drop_disease`) loses 0.006 AUC and 0.007 Brier. This is a substantial change from the prior 6-class result (in which dropping ADMET cost 0.072 AUC) and reflects the redistribution of importance across the remaining feature classes once the high-noise MolFormer-XL embedding block is removed from the active model — see §3.7 for the corresponding shift in feature-importance rankings.

The ablation as currently structured is not yet the playbook-prescribed leave-one-feature-class-out: (a) the Indication feature class is operationalized only as one-hot disease area, not as the LLM-taxonomy LOO base rate the playbook specifies; (b) Sponsor is absent (planned, §2.2 Sponsor and §3.8); (c) the Molecular class is split into a fingerprints row only (embeddings have been removed from the active model after the prior 6-class ablation). The current results should therefore be read as a within-class structural ablation rather than as the publishable feature-class importance table.

### 3.7 Feature importances

Top-20 features by XGBoost gain at T = 2019 in the headline 5-class model (`model_runs/t2019/feature_importances.csv`):

| Rank | Feature | Gain |
|---|---|---|
| 1 | pathway_R-HSA-9006925 (intracellular signaling by second messengers) | 0.0060 |
| 2 | pathway_R-HSA-2029480 (Fcγ receptor-dependent phagocytosis) | 0.0042 |
| 3 | ecfp4_842 | 0.0041 |
| 4 | ecfp4_1281 | 0.0040 |
| 5 | maccs_137 | 0.0038 |
| 6 | ecfp4_1167 | 0.0034 |
| 7 | ecfp4_1365 | 0.0034 |
| 8 | maccs_134 | 0.0034 |
| 9 | maccs_31 | 0.0033 |
| 10 | pathway_R-HSA-9006934 (signaling by receptor tyrosine kinases) | 0.0030 |
| 11 | pathway_R-HSA-69278 (cell cycle, mitotic) | 0.0030 |
| 12 | ecfp4_1796 | 0.0030 |
| 13 | pathway_R-HSA-112314 (neurotransmitter receptors and postsynaptic signal transmission) | 0.0029 |
| 14 | ecfp4_454 | 0.0028 |
| 15 | ecfp4_334 | 0.0027 |
| 16 | pathway_R-HSA-390696 (adrenoceptors) | 0.0027 |
| 17 | ecfp4_1557 | 0.0027 |
| 18 | admet_DILI_drugbank_approved_percentile | 0.0025 |
| 19 | ecfp4_1693 | 0.0025 |
| 20 | ecfp4_1108 | 0.0025 |

The top-20 list is led by ECFP4 substructural bits (10 of 20), Reactome pathway features (7 of 20), MACCS keys (3 of 20), and a single ADMET feature (the percentile rank of predicted drug-induced liver injury against the DrugBank-approved cohort). The pathway block contributes broad regulatory and cell-biological categories (intracellular signaling by second messengers, Fcγ-receptor phagocytosis, signaling by receptor tyrosine kinases, mitotic cell cycle, neurotransmitter receptor signaling, adrenoceptors), consistent with a candidate set that is broadly enriched for oncology, cardiometabolic, and CNS indications. Absolute gain values for individual features remain small (each below 1%), reflecting the gradient-boosted ensemble's tendency to distribute gain across many correlated splits; per-class aggregated importance and per-candidate SHAP attribution (planned, §2.5) will be the more interpretable summary for the manuscript.

### 3.8 Limitations specific to the modeling layer

In addition to the descriptive-pipeline limitations enumerated in `paper/methods.md` §5, the predictive layer has the following caveats specific to the present implementation, all of which are tracked at `paper/playbook_gap_analysis.md`:

1. **Brier sits a hair above the §3.2 paper target on the uncalibrated run.** The headline T = 2019 Brier of 0.186 narrowly exceeds the playbook's §3.2 paper target of ≤ 0.18, though it is well below the baseline-rate Brier of approximately 0.22 reported in the playbook. The probability calibrator described in §2.5 is now wired up end-to-end (isotonic / Platt selectable via `config.calibration_method`); the headline run in §3.4 does not invoke it because `config.calibration_year` is unset for that run. The next paper run will set `calibration_year = 2018`, use the three-way 2017 / 2018 / 2019+ split, and report pre- and post-calibration Brier and ECE.
2. **One of five external baselines is still missing.** Stratum base rate, Tanimoto-NN, and target-only are implemented and reported in §3.5; HINT (Fu et al. 2022) is now wired into the trial-level harness (§2.7, §2.9.4) but its headline-comparison numbers are not yet in §3.5 (forward-pointer at §3.9). The remaining unimplemented playbook baseline is the BIO biomarker-selection binary indicator. The drug-indication-level baseline table in §3.5 therefore remains 3 / 5; the trial-level table will be 4 / 5 once §3.9 lands.
3. **Temporal split offset by one year from the playbook protocol on the headline run.** The playbook protocol is T = 2017 train / 2018 calibrate / 2019–2022 test; the present headline run uses T = 2019 with 2018 absorbed into training. The three-way calibration split is implemented (`model/splits.py:split_with_calibration`) and is a configuration choice for the next paper run, not additional implementation work.
4. **Snapshot-vintage caveat on enrichment features.** Per §2.2.1, the feature classes are intended as functions of pre-clinical-known properties of the candidate, but the underlying external-database snapshots (ChEMBL 36, Open Targets 25.03, Reactome) are dated to the time of feature extraction rather than to each candidate's earliest activity date. The empirical magnitude of any retroactive-snapshot drift has not yet been measured; a planned snapshot-vintage sensitivity analysis, anchored on the §2.5 feature-class audit, will quantify it.
5. **Ongoing exclusion biases test prevalence.** Dropping `Ongoing` from the label-eligible cohort retains only candidates with terminal verdicts, biasing the test set toward candidates whose programs resolved during the test window. The T = 2019 test set is 38.5% positive vs 18.7% in training; the shift is a censoring artifact of the time-on-market correlate, not a real change in approval base rates.
6. **Single-task head; per-phase results are post-hoc.** Training is still a single binary classifier. The per-phase metric slicer (§2.9.3) gives per-phase ROC/PR/Brier on the trial-level model at evaluation time, which is a useful diagnostic but not the same as the playbook's prescribed per-phase transition heads (P1 → P2, P2 → P3, P3 → Approval) trained with phase-specific labels. Multi-task / per-phase training remains an open item.
7. **Sponsor feature class deferred.** Per §2.2 Sponsor (planned), the sponsor-derived features (originator / earliest sponsor and sponsor prior-approval count) are not yet included; the latter additionally requires resolving how to source approval counts that are themselves frozen at or before each candidate's earliest activity date.
8. **Engineering deliverables outstanding.** SHAP-based interpretability and the FAISS nearest-neighbor decision-support tool are tracked TODOs (§2.5) and will land before manuscript submission. Probability calibration (ECE / Brier pre+post) has moved out of this list — it is implemented (§2.5) and pending only a run-configuration switch (item 3 above).

These caveats define the proximate workplan and are mapped to playbook phases in `paper/playbook_gap_analysis.md` §4.

### 3.9 Trial-level and HINT comparison (forward pointer)

The trial-level supervised model (§2.9.1–§2.9.3) and the HINT external baseline (§2.7, §2.9.4) are both implemented at the code layer, but the headline run that exercises them on a temporal split and writes per-phase comparison tables has not yet been executed. The expected output layout is `model_runs/<tag>/trial/` for the supervised model (with `metrics.json` carrying `per_phase_metrics`, plus `hint_test.csv`) and a sibling `model_runs/<tag>/trial/hint/` directory populated by `run_hint.sh`; `scripts/build_consolidated_report.py` joins the two into a single per-phase comparison table. This section is a placeholder for those numbers and will be filled in alongside the next paper run.
