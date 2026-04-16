# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running & Testing

All commands should be run from `src/`.

```bash
# Run the pipeline (from project root)
python src/run_pipeline.py --source aact --keyword peptide --output ./docs --formats html excel

# Run all tests
./run_tests.sh

# Run tests for a specific component
./run_tests.sh --include aggregation
./run_tests.sh --include ingestion,clustering

# Verbose output (show every test name)
./run_tests.sh -v

# Show only failing modules
./run_tests.sh -f
```

`run_tests.sh` skips `test_pipeline.py` by default (integration test). Pass `--include pipeline` to run it explicitly.

## Architecture

### Data flow

Each stage accepts and returns typed dataclasses from `pipeline/models.py`:

```
TrialIngestionStage           -> TrialTable
CandidateClusteringStage      -> CandidateTable
AttributeClassificationStage  -+  (parallel via ThreadPoolExecutor)
OutcomeAdjudicationStage      -+  -> AttributeTable, OutcomeTable
FunnelAggregationStage        -> FunnelResults
ReportingStage                -> ReportOutput
```

The orchestrator (`pipeline/pipeline.py`) holds `PipelineConfig` and wires all stages. Stages 3a and 3b share a single `KnowledgeCache` instance and run concurrently -- the cache uses per-thread SQLite connections in WAL mode.

### Key design decisions

- **`pipeline/models.py`** is the single source of truth for all inter-stage types. Add new fields here before touching any stage.
- **`pipeline/knowledge_cache.py`** -- SQLite-backed LLM result cache keyed by SHA-256 of `drug_name|indication` (classification) or `drug_name|indication|highest_phase` (adjudication). Avoids redundant API calls on re-runs.
- **LLM prompt editing**: `classification.py` and `adjudication.py` load prompts from `pipeline/stages/prompts/*.txt` via `utils/prompt_runner.py`. To change LLM behavior, edit the `.txt` prompt files, not the Python.
- **Clustering strategies** (`pipeline/stages/clustering.py`) follow a Strategy pattern -- `StringMatchStrategy`, `HybridStrategy` (preferred, uses MeSH terms with text fallback), and `EmbeddingsStrategy` (not yet implemented) all implement `ClusteringStrategy.cluster()`.
- **`pipeline/drugbank_norm.py`** -- builds two deduplicated lookup DataFrames from `drugbank_approvals.csv`: exact match by `query_name` and normalized match by `query_norm`. Clustering uses these for DrugBank ID assignment, then re-deduplicates candidates by `(drugbank_id, mesh_indication)`.

### Non-obvious patterns

- **Tiered LLM routing** (`utils/tiered_router.py`): Classification and adjudication run candidates through Sonnet first, then escalate low-confidence or unknown results to Opus. This is controlled by `escalation_predicate` functions, not config. The `CostLedger` tracks per-stage, per-tier API costs.
- **Forward-looking cohort aggregation**: Funnel rates follow the BIO/QLS cohort method with an asymmetric cohort/advancement split. `phases_observed` (denominator) requires terminal trial evidence at a clinical phase — a COMPLETED/TERMINATED/WITHDRAWN/SUSPENDED trial, or a Phase 4 trial / APPROVED / COMMERCIALIZED event for the Approval cohort (COMMERCIALIZED for Market). `phases_advanced` (numerator) is a superset that additionally admits (a) any-status trial at a phase (a started-but-active Phase 2 trial is enough to evidence P1→P2 advancement) and (b) FAILED_PHASE_N adjudicator verdicts (credits advancement to Phase N but not cohort membership, since an LLM verdict without terminal trial data would otherwise pad the next denominator with uncorroborated failures). A candidate counts as a success for N→N+1 iff `from_phase` is in `phases_observed` AND `phases_advanced` intersects any strictly later cohort. Approval is not back-propagated to earlier clinical phases when those trials were not observed.
- **Single source of truth for transition rates**: All phase-success and LOA math lives in `pipeline/stages/aggregation.py`. The module-level `transition_rate_from_records` is the one function to call; `FunnelAggregationStage._transition_rate` is a thin wrapper. Reporting components either consume `FunnelResults` from the aggregation stage or flow flat records through `reporting/_compute.compute_transition_rate` (an alias of the aggregation function). Time-period slices now build records via `FunnelAggregationStage._join` (plumbed through `ReportContext.trial_table`), so per-period rates use the identical cohort definition as overall rates. Do not re-implement rate logic inline.
- **All-baseline bars**: Every rate / LOA bar chart carries an `All` bar driven by `funnel_results.overall` (or the per-period overall) as a population baseline. This is visible in LOA by disease area, LOA by modality, Oncology vs Non-Oncology, the time-period transition chart, and the time-period disease comparison chart.
- **Peptide-only optimization**: When `peptide_only_report=True`, the pipeline classifies all candidates first, filters to peptides, then adjudicates only the filtered set. Non-peptides skip the expensive adjudication LLM calls entirely.
- **Deterministic failure shortcut**: Adjudication skips LLM calls when all of a candidate's trials are TERMINATED/WITHDRAWN -- returns FAILED_PHASE_X directly.
- **Swappable adjudication**: Two adjudicators produce the same `OutcomeTable` shape and are selected via `PipelineConfig.adjudication_method` (CLI: `--adjudication-method`). (a) `"fda_timeline"` (default) — `stages/adjudication_fda.py` + `fda/` package. Reconstructs each drug's indication-level FDA approval timeline from openFDA (ORIG submissions + SUPPL new-indication letters), LLM-matches the trial indication, and checks NDC commercial status. LLM extract/match calls go through a swappable LLM backend (`AnthropicJSONClient` or `OpenAICompatJSONClient`), which shares the same `KnowledgeCache` (table: `llm_json_cache`) and `CostLedger` as the rest of the pipeline. FDA HTTP responses are cached on disk under `fda_cache_dir`. (b) `"llm_direct"` — the original `OutcomeAdjudicationStage` which asks the LLM directly per candidate, backed by the regulatory-index shortcut and the adjudication_cache table.
- **Swappable FDA-adjudication LLM backend**: The FDA-timeline adjudicator's LLM is itself swappable via `--fda-llm-backend {openai_compat,anthropic}` (default `openai_compat`). The default points at local Ollama (`http://localhost:11434/v1`) serving `qwen2.5:32b-instruct` — sized for a 36GB unified-memory Mac; install once with `ollama pull qwen2.5:32b-instruct`. Override with `--fda-llm-base-url` / `--fda-llm-model` to target any OpenAI-compatible endpoint (vLLM, llama.cpp, Together, Groq, etc.) or pass `--fda-llm-backend anthropic` for Sonnet on this stage. Scope: this flag affects **FDA-timeline adjudication only** — `llm_direct` adjudication and classification continue to use Anthropic unchanged. Local models record token volume at $0 in the cost ledger (unknown model strings silently price at zero in `CostLedger`).
- **M x N trial expansion**: Ingestion produces one row per (study, drug, condition) tuple. A trial with 3 drugs and 2 conditions yields 6 RawTrial rows. Clustering re-groups them.
- **P-value inconsistency flagging**: Reporting flags candidates where a statistically significant result (p <= 0.05) appears at a failed phase, or a non-significant result at Phase 3 but the candidate is Approved/Commercialized.

### Environment variables

| Variable | Required for | Purpose |
|---|---|---|
| `CLAUDE_API_KEY_UCD` | LLM stages | Anthropic API key used by `utils/prompt_runner.py` |
| `AACT_DB_USER` | `--source aact` | AACT PostgreSQL username |
| `AACT_DB_PASSWORD` | `--source aact` | AACT PostgreSQL password |

### Test fixtures

`tests/conftest.py` provides shared fixtures for every stage's input/output types. Individual test files should use these fixtures rather than constructing model objects inline.

### Implementation status

All pipeline stages (ingestion, clustering, classification, adjudication, aggregation, reporting) are fully implemented. The REST API ingestion source (`--source api`) and `EmbeddingsStrategy` for clustering are not yet implemented and raise `NotImplementedError`. Config flags `use_literature_lookup` and `use_regulatory_data` are accepted but not yet integrated.
