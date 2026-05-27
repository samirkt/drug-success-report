# LOA Playbook Gap Analysis & Workplan to Submission

## Scope

Evaluation of the implemented pipeline (`src/`) and modeling code (`model/` + `run_modeling.py` + `train_model.sh` + `model_runs/`) against the **Ex-Ante LOA Prediction Framework Playbook** (root-level PDF, 17 pp.).

The playbook positions this work as the *first reproducible, calibrated, ex-ante drug-indication-candidate-level LOA prediction model*, beating (a) stratum base rates, (b) HINT, and (c) molecular-similarity-only — with calibrated probabilities and per-feature attribution. Target venue Cell Patterns / Nat. Commun. Headline gates: test AUC ≥ 0.72, ECE ≤ 0.06, decision-curve dominance over baselines on threshold range 0.10–0.40.

This document inventories the gap between current state and that target, and lays out what remains.

---

## 1. Implementation Inventory (current state)

### 1.1 Data pipeline (`src/`) — what exists

| Playbook component | Status | Where |
|---|---|---|
| Trial registry (AACT) | Full | `pipeline/stages/ingestion.py`, AACT cache pickling |
| CT.gov API v2 ingestion | Missing — AACT mirror only | — |
| Drug identity + SMILES (DrugBank) | Full | `pipeline/enrichment/smiles.py` |
| ChEMBL Structure Pipeline standardization | Full | `pipeline/enrichment/smiles_standardization.py` |
| RDKit ECFP4 + MACCS fingerprints | Computed offline | `scripts/build_fingerprints.py` -> `fingerprints.parquet` (4,236 rows) |
| MolFormer-XL embeddings | Computed offline | `scripts/build_molformer_embeddings.py` -> `molformer_embeddings.parquet` (4,236 rows, 768-dim) |
| Max-Tanimoto / max-cosine to nearest-approved (LOO) | **Not computed** | — |
| ChEMBL targets (drug -> UniProt) | Full | `pipeline/enrichment/targets.py` (snapshot SQLite) |
| Open Targets — mechanism, action, maxPhase, pathways, target list | Partial | `pipeline/enrichment/opentargets.py` |
| Open Targets — genetic-evidence score, tractability flag, prior approvals at target | **Not extracted** | — |
| Reactome pathway membership | Full | `pipeline/enrichment/reactome.py` |
| Reactome "pathway has any approved drug" flag | Missing | — |
| ADMET-AI predictions (52 props + percentiles) | Full | `pipeline/enrichment/admet.py` |
| MeSH disease tagging | Full | AACT `mesh_condition_terms` |
| EFO / MONDO disease ontology bridge | Missing | — |
| Indication base rate (LLM taxonomy, LOO) | Missing | — |
| LLM modality + disease-area classification | Full | `pipeline/stages/classification.py` |
| Approval ground truth (adjudication) | Full, 3 swappable backends | `adjudication.py`, `adjudication_fda.py`, `adjudication_ndc.py` (~7,460 resolved per latest run; playbook cites 11,099) |
| Phase-1-start-date feature freeze | **No `phase1_start_date` on `Candidate`**; enrichments use current snapshots | — |
| `as_of` cutoff for FDA adjudication | Implemented | `PipelineConfig.fda_adjudication_as_of` |
| Sponsor feature engineering (prior approvals, total trials, biotech vs pharma) | **Entirely missing** — raw `sponsor` string only | — |
| Coverage report per feature class × modality × outcome | Missing | — |

### 1.2 Modeling (`model/`) — what exists

| Playbook component | Status | Where |
|---|---|---|
| XGBoost classifier | Implemented | `model/models/xgb.py` |
| Hyperparameter tuning on validation set | Fixed defaults; early stopping on inner 10% holdout, no grid search | `model/models/xgb.py` |
| Isotonic calibration on validation set | **Missing** | — |
| Temporal split (playbook: ≤2017 train / 2018 val / 2019–2022 test) | **Wrong cut** — split is `earliest_start_date.year ≤ 2020` train / >2020 test in `train_model.sh` | `model/splits.py` |
| Per-phase transition heads (P1->P2, P2->P3, P3->Approval) | Missing | — |
| Full-funnel LOA head | Single binary head only — collapses Approved/Commercialized vs Failed_P1/P2/P3 | `model/config.py` |
| Indication base rate as anchor feature | Missing | — |
| Feature freezing at Phase 1 start | Missing | — |
| AUC, Brier, log-loss, F1, PR-AUC | Implemented | `model/evaluate.py` |
| ECE, MCE, reliability diagrams | Missing | — |
| Decision-curve analysis (threshold 0.05–0.50) | Missing | — |
| Baseline 1: stratum base rate (modality × TA) | Missing | — |
| Baseline 2: Tanimoto-NN + MolFormer-cosine only | Missing | — |
| Baseline 3: target-only | Missing | — |
| Baseline 4: HINT | Missing | — |
| Baseline 5: BIO biomarker-selection indicator | Missing | — |
| Leave-one-feature-class-out ablation | Group LOO harness in place | `model/ablate.py` |
| SHAP per-candidate + per-feature-class beeswarm | Missing — XGBoost gain importance only | `model/evaluate.py` |
| Sensitivity analyses (small-mol only / exclude onc / exclude pre-2010) | Missing | — |
| FAISS nearest-neighbor index + CLI / Streamlit | Missing | — |
| Case studies (Lecanemab, Donanemab, Sotorasib, T-DXd, ...; aducanumab, BACE inhibitors) | Missing | — |
| Reproducibility (Docker, DVC, MLflow, Zenodo, pinned deps) | All missing — file-based only | — |

---

## 2. Concerns by severity

### 2.1 Critical (challenges publishability)

These are showstoppers — the paper's central claim *"calibrated, leakage-free, beats baselines"* fails without each of them.

1. **No phase-1-start feature freeze.**
   The playbook (§7.1, Risks §6) names temporal leakage as the single biggest failure mode for this class of paper. Today, target / ADMET / OpenTargets enrichments resolve at *current* snapshot time and join onto candidates whose Phase 1 started years earlier. A reviewer who notices a target with prior approvals enriched after the candidate's Phase 1 start will reject the paper outright.
   - Required: `Candidate.phase1_start_date` populated from earliest Phase 1 trial; every feature must respect that date or be flagged unfreezable.
   - Open Targets and ChEMBL must use snapshots dated ≤ freeze year.
   - Sponsor "prior approvals" must count only approvals before freeze date.

2. **No isotonic calibration.**
   The playbook positions calibration as the ML novelty over Hay/Wong/ClinSR. ECE ≤ 0.06 is a Phase-3 gate. Without calibrated probabilities the decision-curve analysis (the killer-figure decision-support claim) cannot run. Trivial to bolt on — `sklearn.calibration.CalibratedClassifierCV(method="isotonic", cv="prefit")` against the 2018 holdout.

3. **No baselines implemented (0 of 5).**
   Per playbook §2.3 / §7.3 the contribution "stands or falls" on outperforming stratum base rates, HINT, and molecular-similarity-only. With *zero* baselines, there is currently no way to produce the headline AUC table. The stratum base rate (Baseline 1) and Tanimoto-NN (Baseline 2) are the two reviewers will demand and are the ones a reviewer can compute themselves to falsify the paper's claim — they must be in.

4. **Wrong temporal split.**
   Current `train_model.sh` uses `≤2020 train / >2020 test`. Playbook prescribes `≤2017 train / 2018 calibration / 2019–2022 test`. The 2020 split (a) leaks COVID-era trials into training, (b) undersizes the test window, and (c) leaves no held-out validation set for calibration / hyperparameter tuning.

5. **No molecular-similarity features.**
   Tanimoto-NN (max ECFP4 Tanimoto to training-approved drug, leave-one-out by drug) and MolFormer max-cosine are required both as model features (Molecular feature class) *and* as Baseline 2. The fingerprints + embeddings already exist in parquet — the missing piece is the LOO similarity computation, which is a single FAISS or matrix-multiply pass over training-approved candidates per test candidate.

6. **No FAISS nearest-neighbor lookup ("killer figure").**
   §3.1 Part C explicitly calls this out as the deliverable reviewers and readers remember. *"Candidates like this one historically reached approval X% of the time, vs Y% stratum base rate, because Z."* This is the decision-support artifact in §8.3 (CLI + minimal Streamlit). The paper has a publishable model without it but loses its strongest narrative hook and one of the four pillars of §2.3 ("calibrated probabilities with per-feature attribution and case studies").

7. **No SHAP attribution.**
   The "which feature class matters" SHAP table is the most-cited result of this kind of paper (§7.4). XGBoost gain importance is not a substitute — it cannot do per-candidate attribution and is not reviewer-credible for non-tree readers.

8. **No sponsor features.**
   Sponsor is one of the six required feature classes. Without it, the LOO-class ablation has only five rows and the §3.1 Part B claim of "six feature classes integrated" is literally false. AACT already has sponsor strings; SEC EDGAR is needed only if biotech-vs-pharma classification needs market-cap thresholds — a curated list (top-20 large pharma names) is acceptable as a v1.

### 2.2 Major (big gap, can be addressed within the 12-week window)

These are publishable as limitations only with effort, but the playbook expects them.

9. **No multi-task / per-phase heads.**
   Playbook §3.1 Part B and §4.2 prescribe per-phase transition heads (P1->P2, P2->P3, P3->Approval) plus a full-funnel LOA head. Current single binary head collapses everything into approved-vs-failed and discards the per-phase structure that the funnel aggregation in `pipeline/stages/aggregation.py` already computes. Addressable as a multi-output XGBoost or as four independent heads sharing features.

10. **No indication base rate feature.**
    LLM-taxonomy-derived indication base rates with LOO are a feature-class on their own (Indication, §3.1) and are an essential anchor — without them the model cannot exceed the Stratum baseline's ceiling. The pipeline already produces `disease_area` and `funnel_results` per disease; deriving an LOO base-rate feature from those is roughly a half-day join.

11. **Calibration metrics suite (ECE, MCE, reliability diagram).**
    Required to report the Phase-3 gate. Once isotonic is in, these are one-page additions in `model/evaluate.py`.

12. **Decision-curve analysis (threshold 0.05–0.50).**
    Phase-3 gate. Standard `dcurves` package or 30-line implementation.

13. **HINT baseline reproduction.**
    Playbook (§7.3) says "if not reproducible, report the attempt and cite their published numbers." Acceptable to ship as cited reference, but at minimum the attempt and failure mode must be documented.

14. **BIO biomarker-selection indicator + Target-only baseline + Stratum base rate baseline.**
    Three out of the five baselines that are computable from existing data alone — no excuse to omit. Stratum base rate is one SQL group-by.

15. **Sensitivity analyses (small-mol-only / exclude oncology / exclude pre-2010).**
    Three re-runs of the existing harness with a `--filter` flag. ~1 day total.

16. **Open Targets — genetic evidence score + tractability flag.**
    Single strongest external signal in the literature (Nelson 2015, King 2019 cited in §10.3). Currently the OpenTargets snapshot pulls mechanism/action/maxPhase/pathways/targets only — the `geneticConstraint` / `tractability` GraphQL fields are not extracted.

17. **Reactome "pathway has any approved drug" flag.**
    A single join from drug-target-pathway × DrugBank approved-drug list. Required for the Pathway feature class to have predictive content beyond raw membership IDs.

18. **Case studies (5–7 approvals + 3–5 failures).**
    Playbook §7.5–7.6 names the molecules. Each requires (a) freezing features at IND/Phase 1, (b) running prediction, (c) writing a 200-word retrospective. ~1 week given infrastructure. Reviewers remember case studies, not AUC numbers (§3.1 Part C).

19. **Reproducibility stack (Docker + DVC + MLflow + Zenodo).**
    Phase-5 deliverable. Without it, the open-source-release pillar of §3.1 Part D is unmet. None of these are research risks but all are submission blockers — Cell Patterns and Nat. Commun. require code/data DOIs.

20. **Coverage report per feature class × modality × outcome.**
    Phase-1 gate: ≥60% SMILES coverage for small molecules AND ≥80% target coverage across all modalities. Currently 4,236 of 7,460 candidates (57%) have full molecular features — already below the 60% gate for small-molecule SMILES coverage if biologics aren't excluded first. Must be measured and either reported as-is or improved (more aggressive DrugBank synonym matching) before Phase 2.

### 2.3 Minor

21. **No EFO/MONDO ontology bridge.** MeSH alone is workable; the bridge becomes important only if cross-mapping to Open Targets disease IDs (which uses EFO). Currently the OT enrichment relies on indication-string matches — a documented limitation.
22. **No live CT.gov API v2 fallback.** AACT mirror is sufficient for retrospective work; only matters for the §9.4 long-term extension.
23. **No Streamlit UI.** CLI is acceptable for the §8.3 deliverable; Streamlit is "minimal" per the playbook.
24. **Aducanumab and BACE-inhibitor failure-class case studies** require manual curation that is not pipeline-automatable; account for this in scope.
25. **AACT data volume.** Playbook cites 11,099 candidates; current resolved set is ~7,460. Worth confirming the year-range / sponsor-filter combination matches the cited number, or revising the figure in the manuscript.

---

## 3. Methodology missteps (separate from gaps)

These are *implemented but wrong* and need correction, not just addition.

- **Train/test split year.** `train_model.sh` hard-codes `--time-split-year 2020`. Change to a three-way temporal split with 2018 carved out as calibration set per §7.2.
- **Single-task collapsing of phase outcomes.** Current label maps `{Approved, Commercialized}` vs `{Failed_P1, Failed_P2, Failed_P3}` and **drops Ongoing and Unknown**. Dropping Ongoing biases toward older candidates (which have had more time to fail or approve), inflating apparent base rates and leaking time-on-market into the label. Either include Ongoing as a censored class with survival-style handling, or restrict the test window to candidates whose Phase 1 start was >5 years before `as_of` so the censoring bias is bounded and documented.
- **Inner-validation early stopping.** Current 10% inner split overlaps thematically with the 2018 calibration set the playbook wants. Consolidate: use 2018 as both early-stopping validation and calibration set. This also frees up 2019–2022 as a clean test set.
- **Joint coverage of 4,236 / 7,460 (57%).** Below the 60% Phase-1 gate. Options: (a) tighten DrugBank synonym matching to lift SMILES coverage, (b) split into small-molecule (with SMILES) and biologics (target/pathway-only) tracks per playbook §4.2, and report stratified results.
- **No leave-one-out for similarity / base-rate features.** When implemented, both Tanimoto-NN and indication-base-rate features must be computed LOO at the *drug* level (not candidate level — multiple candidates share a drug) to avoid trivial leakage of approved-drug status into similarity scores.

---

## 4. Workplan to finish (mapped to playbook phases)

Phases below assume the fixes above are sequenced; gates are the playbook's own.

### Phase 1 (Weeks 1–2): Coverage audit + freeze infrastructure
- Add `phase1_start_date` to `Candidate` (derive from earliest Phase 1 trial in `trial_table`).
- Extend `PipelineConfig` with per-feature freeze policy; refactor each enrichment to accept and respect a freeze date or declare itself static.
- Build coverage report: SMILES, targets, ADMET, OT genetic evidence, Reactome — each per modality × outcome. Validate ≥60% SMILES (small mols) and ≥80% target (all modalities) gate.
- Decide biologics handling: separate track or fold-in with NaN imputation (playbook §4.2 prescribes asymmetric handling).
- **Gate:** coverage report passes thresholds OR scope revised.

### Phase 2 (Weeks 3–4): Feature extraction completion
- Compute max-Tanimoto + max-MolFormer-cosine to nearest training-approved drug, LOO at drug level. Write to `candidate_detail.parquet`.
- Extend OpenTargets enrichment with `geneticConstraint`, `tractability`, prior-approvals-at-target.
- Add Reactome "approved drug on pathway" flag.
- Build sponsor feature table: `prior_approvals_count`, `prior_trials_count`, `is_large_pharma` (curated top-20 list).
- Compute indication base rate from LLM disease-area taxonomy, LOO.
- **Gate:** feature matrix with ≤10% missing per column.

### Phase 3 (Weeks 5–6): Model training + baselines
- Switch `model/splits.py` to three-way temporal split: ≤2017 train, 2018 calibrate, 2019–2022 test.
- Add isotonic calibration via `CalibratedClassifierCV(method="isotonic", cv="prefit")`.
- Add per-phase heads (4 separate XGBoost classifiers: P1->P2, P2->P3, P3->Approval, full-funnel) sharing features.
- Implement five baselines as separate `Model` subclasses in `model/models/`: stratum rate, Tanimoto-NN, target-only, HINT (or cited fallback), BIO biomarker indicator.
- Add ECE / MCE / reliability diagram / decision-curve analysis to `model/evaluate.py`.
- Add SHAP per-candidate + per-feature-class aggregation; produce beeswarm.
- **Gate:** test AUC ≥ 0.72 AND ECE ≤ 0.06 AND decision-curve dominance on 0.10–0.40.

### Phase 4 (Weeks 7–8): Decision-support tool + case studies + figures
- Build FAISS index on (molecular embedding ⊕ target one-hot ⊕ indication MeSH one-hot) with realized outcomes.
- Wrap as CLI: input SMILES + UniProt + MeSH -> top-10 neighbors with outcomes + LOA estimate. Optional Streamlit shell.
- Run case studies on Lecanemab, Donanemab, Semaglutide-MASH, T-DXd, Sotorasib, Voretigene, Obecabtagene + 3–5 failures (aducanumab, BACE inhibitors). Features frozen at IND / Phase 1 start.
- Generate playbook-required figures: stratified-LOA funnel, AUC/Brier/calibration curves, SHAP beeswarm, case-study predictions vs realized, neighbor-tool mock.
- **Gate:** figures final; case studies written.

### Phase 5 (Weeks 9–12): Ablations, sensitivity, manuscript
- Run six leave-one-feature-class-out ablations (–Molecular / –Target / –Pathway / –ADMET / –Indication / –Sponsor) reporting ΔAUC / ΔBrier / ΔECE.
- Run three sensitivity analyses (small-mol only / exclude oncology / exclude pre-2010).
- Add reproducibility stack: pinned `pyproject.toml` lockfile already exists — add Dockerfile, MLflow tracking, DVC for large parquets, Zenodo deposit.
- Draft 6,000–8,000-word manuscript; arXiv preprint at Week 10.
- **Gate:** submission to Cell Patterns (or chosen venue).

---

## 5. Top recommendations (if I had to pick three to do first)

1. **Phase-1 feature freeze + correct temporal split.** This is the leakage fix the entire paper depends on. Everything else can be retrofitted; leakage cannot — it invalidates already-computed metrics.
2. **Isotonic calibration + ECE/decision-curve metrics.** Single-day work that unlocks Phase-3 gate evaluation and the calibration-novelty narrative.
3. **Stratum-rate + Tanimoto-NN baselines.** Cheapest two of five baselines, both required for the headline claim. Without them there is no AUC table to put in the paper.

Critical items 4–8 above are next priority.

---

## 6. Out of scope for this evaluation

- Independent verification of the playbook's prior-art claims (HINT, inClinico, Zhou 2025).
- Re-running the existing pipeline to verify the 7,460 vs 11,099 candidate-count discrepancy.
- Manuscript draft itself (Phase 5 deliverable).
