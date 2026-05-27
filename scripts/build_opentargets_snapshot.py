"""Build a name-keyed OpenTargets snapshot for the enrichment pipeline.

Reads OpenTargets Platform Parquet datasets (downloaded manually — see
the "Manual download" section in the enrichment plan) and writes a slim
indexed SQLite that `OpenTargetsEnrichment` can open read-only and look
up by normalized drug name. Three tables:

    name_opentargets_drug(query_norm, chembl_id, drug_name, moa_text,
                          action_type, target_symbols, target_ensembl,
                          pathways, tractability_modalities,
                          tractability_labels, loeuf_min)
    name_opentargets_indications(chembl_id, indication_efo_id,
                                 indication_name, indication_max_phase)
    name_opentargets_target_disease_evidence(
        chembl_id, ensembl_id, efo_id, genetic_score)

The `tractability_*` and `loeuf_min` columns aggregate across each
drug's targets (union of category labels; min LOEUF across targets —
lower = more constrained). The `target_disease_evidence` table stores
OpenTargets `associationByDatatypeDirect` `genetic_association` scores
joined onto the drug's targets, so the enrichment can fetch a per-
(drug, indication) genetic score without re-reading the full ~10M-row
association dataset at runtime.

`query_norm` is computed with the same `canonicalize_drug_name`
normalizer the rest of the pipeline uses, so OT lookups compose with the
ChEMBL targets and ChEMBL SMILES fallback: all three stages key on
`canonicalize_drug_name(Candidate.drug_name_raw)`.

This script is **not** part of the pipeline — it runs once per OT
release.

Inputs
------

    --opentargets-dir  Root of the manually-downloaded OT Parquet.
                      Expected subdirectories (any of these layouts is
                      tolerated — the builder picks the first that
                      resolves):
                          molecule/                         (OT 22.x+)
                          drug/                             (alt naming)
                          mechanismOfAction/
                          targets/                          (OT 22.x+)
                          target/                           (alt naming)

    --chembl-snapshot  Path to the ChEMBL targets SQLite snapshot built
                      by scripts/build_chembl_targets_snapshot.py — used
                      purely to derive `query_norm` <-> `chembl_id`
                      mappings so OT stays name-keyed.

Output
------

    --out              SQLite file path. PRAGMA user_version is stamped
                      with the OT release number (pass --release or let
                      the script parse it from --opentargets-dir's
                      basename, e.g. ".../opentargets/25.03").

Download hint (as of 2026)
-----------------------------

    ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/<release>/output/etl/parquet/

    Minimal sync:
        rsync -rvz rsync.ebi.ac.uk::pub/databases/opentargets/platform/25.03/output/etl/parquet/molecule/ data/opentargets/25.03/molecule/
        rsync -rvz rsync.ebi.ac.uk::pub/databases/opentargets/platform/25.03/output/etl/parquet/mechanismOfAction/ data/opentargets/25.03/mechanismOfAction/
        rsync -rvz rsync.ebi.ac.uk::pub/databases/opentargets/platform/25.03/output/etl/parquet/targets/ data/opentargets/25.03/targets/
        rsync -rvz rsync.ebi.ac.uk::pub/databases/opentargets/platform/25.03/output/etl/parquet/associationByDatatypeDirect/ data/opentargets/25.03/associationByDatatypeDirect/

    The associationByDatatypeDirect subdir is optional — if absent the
    `name_opentargets_target_disease_evidence` table is left empty and
    the enrichment falls back to skipping the genetic-score feature.
    Tractability + LOEUF are pulled from the `targets/` parquet and need
    no extra download.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path

try:
    import pyarrow.dataset as pads  # type: ignore
    import pyarrow.parquet as pq    # type: ignore
except ImportError:  # pragma: no cover - import-time guidance
    print(
        "error: build_opentargets_snapshot requires pyarrow. "
        "Install with `pip install pyarrow`.",
        file=sys.stderr,
    )
    raise

# Import the shared drug-name normalizer from the package under src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.drugbank_norm import canonicalize_drug_name  # noqa: E402

logger = logging.getLogger("build_opentargets_snapshot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOLECULE_CANDIDATE_DIRS = (
    "drug_molecule", "molecule", "drug", "molecules", "drug/molecule",
)
_MOA_CANDIDATE_DIRS = (
    "drug_mechanism_of_action", "mechanismOfAction", "mechanismsOfAction",
    "moa",
)
_INDICATION_CANDIDATE_DIRS = (
    "drug_indication", "indication", "indications",
)
_TARGETS_CANDIDATE_DIRS = ("target", "targets")
_ASSOCIATIONS_CANDIDATE_DIRS = (
    "associationByDatatypeDirect",
    "association_by_datatype_direct",
    "associationByDatatypeIndirect",
    "association_by_datatype_indirect",
)

# Aggregation caps to keep snapshot lean and enrichment output readable.
# A drug with dozens of MoAs or hundreds of pathways points at a noisy
# aggregation; we want the top-N, not the long tail.
_MOA_CAP = 3
_ACTION_TYPE_CAP = 3
_PATHWAY_CAP = 20
_TARGET_SYMBOL_CAP = 50
_TRACTABILITY_LABEL_CAP = 30

# Only this datatype is treated as "genetic evidence" — OT exposes
# several association datatypes (literature, somatic_mutation, etc.)
# but `genetic_association` is the one downstream callers want.
_GENETIC_DATATYPE = "genetic_association"


def parse_release_number(opentargets_dir: Path) -> int:
    """Best-effort parse of the OT release from its dir name (e.g. 25.03 -> 2503)."""
    stem = opentargets_dir.name
    m = re.search(r"(\d+)[._-]?(\d+)?", stem)
    if not m:
        return 0
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) else 0
    # Stamp as major*100 + minor so "25.03" -> 2503, sortable and unique
    # enough for log-line identification.
    return major * 100 + minor


def resolve_subdir(root: Path, candidates: tuple[str, ...]) -> Path:
    """Return the first candidate subdirectory of ``root`` that exists."""
    for name in candidates:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        f"None of the expected OT subdirectories exist under {root}: "
        f"{', '.join(candidates)}"
    )


def read_parquet_columns(path: Path, columns: list[str]) -> "pads.Dataset":
    """Open a Parquet dataset with the requested column subset.

    OT schemas shift across releases — columns listed here must exist in
    the dataset, otherwise pyarrow raises. The caller is responsible for
    catching that and retrying with a narrower set.
    """
    return pads.dataset(str(path), format="parquet").to_table(columns=columns)


def _first_present(available: set[str], *names: str) -> str | None:
    """Return the first name in ``names`` that is in ``available``."""
    for n in names:
        if n in available:
            return n
    return None


# ---------------------------------------------------------------------------
# Load steps
# ---------------------------------------------------------------------------

def load_chembl_name_index(chembl_snapshot: Path) -> dict[str, list[str]]:
    """Return {query_norm: [chembl_id, ...]} derived from the ChEMBL snapshot.

    Each query_norm can resolve to multiple ChEMBL IDs (e.g. "insulin"
    hits several molregnos), matching the behavior of TargetsEnrichment
    so callers see a consistent union across name-keyed enrichments.
    """
    conn = sqlite3.connect(f"file:{chembl_snapshot}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT DISTINCT query_norm, chembl_id FROM name_targets "
            "WHERE query_norm IS NOT NULL AND query_norm <> ''"
        )
        out: dict[str, list[str]] = {}
        for query_norm, chembl_id in cur:
            if not query_norm or not chembl_id:
                continue
            out.setdefault(str(query_norm), []).append(str(chembl_id))
    finally:
        conn.close()
    logger.info(
        "Loaded %d query_norm -> chembl_id mappings from ChEMBL snapshot",
        len(out),
    )
    return out


def load_molecule_table(molecule_dir: Path):
    """Return a pyarrow.Table with (id, name, indications) columns.

    Tolerates minor column-naming drift across OT releases. Raises if
    none of the expected aliases resolve.
    """
    schema = pq.ParquetFile(
        next(iter(molecule_dir.glob("*.parquet")))
    ).schema_arrow
    available = {f.name for f in schema}

    id_col = _first_present(available, "id", "chemblId", "chembl_id")
    name_col = _first_present(available, "name", "prefName", "pref_name")
    ind_col = _first_present(available, "indications", "indication")
    if not (id_col and name_col):
        raise RuntimeError(
            f"molecule dataset is missing id/name columns; saw {sorted(available)}"
        )

    cols = [id_col, name_col]
    if ind_col:
        cols.append(ind_col)
    tbl = pads.dataset(str(molecule_dir), format="parquet").to_table(columns=cols)
    tbl = tbl.rename_columns(
        [
            "id",
            "name",
            *(["indications"] if ind_col else []),
        ]
    )
    logger.info(
        "Loaded molecule table: %d rows (columns: %s -> id, name%s)",
        tbl.num_rows, cols, (", indications" if ind_col else ""),
    )
    return tbl


def load_moa_table(moa_dir: Path):
    """Return a pyarrow.Table with (chembl_ids, action_type, mechanism_of_action, targets).

    OpenTargets `mechanismOfAction` records are often per-(drug, MoA).
    The `chemblIds` column is a list; we keep rows as-is and expand
    later when we aggregate per drug.
    """
    schema = pq.ParquetFile(
        next(iter(moa_dir.glob("*.parquet")))
    ).schema_arrow
    available = {f.name for f in schema}

    chembl_col = _first_present(
        available, "chemblIds", "chembl_ids", "drugIds"
    )
    action_col = _first_present(available, "actionType", "action_type")
    moa_col = _first_present(
        available, "mechanismOfAction", "mechanism_of_action"
    )
    targets_col = _first_present(available, "targets", "targetIds")
    if not chembl_col:
        raise RuntimeError(
            f"mechanismOfAction dataset lacks chemblIds/drugIds; saw "
            f"{sorted(available)}"
        )

    cols = [c for c in (chembl_col, action_col, moa_col, targets_col) if c]
    tbl = pads.dataset(str(moa_dir), format="parquet").to_table(columns=cols)
    rename = {
        chembl_col: "chembl_ids",
        action_col: "action_type" if action_col else None,
        moa_col: "mechanism_of_action" if moa_col else None,
        targets_col: "targets" if targets_col else None,
    }
    new_names = [rename[c] for c in cols]
    tbl = tbl.rename_columns(new_names)
    logger.info(
        "Loaded mechanismOfAction table: %d rows (columns: %s)",
        tbl.num_rows, new_names,
    )
    return tbl


def _extract_loeuf(raw_constraint) -> float | None:
    """Pull LOEUF from an OT target `constraint` / `geneticConstraint` list.

    OT exposes a list of structs with `constraintType` in {"syn", "mis",
    "lof"}. LOEUF is the upper bound of the observed/expected LoF ratio
    on the "lof" row — lower values = more constraint = more disease-
    relevance. Different releases label the score field differently:
    `oe_ci_upper`, `oeUpper`, `upperBin`, or a generic `score`. We try
    those names in order, falling back to None if no row resolves.
    """
    if not raw_constraint:
        return None
    for entry in raw_constraint:
        if not isinstance(entry, dict):
            continue
        ctype = (
            entry.get("constraintType")
            or entry.get("constraint_type")
            or entry.get("type")
            or ""
        )
        if str(ctype).lower() != "lof":
            continue
        for key in (
            "oe_ci_upper", "oeCiUpper", "oeUpper", "oe_upper",
            "upperBin", "upper_bin", "score",
        ):
            v = entry.get(key)
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _extract_tractability(raw_tractability) -> tuple[list[str], list[str]]:
    """Return (modalities, labels) from an OT target `tractability` list.

    OT stores tractability as a list of structs, one per (modality, label)
    bucket with a boolean `value`. We keep only the rows where the bucket
    is true and surface both the modality (e.g. "SM", "AB", "PR", "OC",
    "OM") and the human-readable label (e.g. "Clinical_Precedence_sm").
    Deduplication preserves first-seen order; the caller caps the list.
    """
    if not raw_tractability:
        return [], []
    modalities: list[str] = []
    labels: list[str] = []
    seen_mod: set[str] = set()
    seen_lab: set[str] = set()
    for entry in raw_tractability:
        if not isinstance(entry, dict):
            continue
        # OT 22.x+ uses (modality, id, value); some releases use `label`
        # instead of `id`. Older releases store boolean buckets at the
        # top-level (no value column) so a present row implies True.
        val = entry.get("value")
        if val is False:
            continue
        modality = (
            entry.get("modality")
            or entry.get("category")
            or ""
        )
        label = (
            entry.get("id")
            or entry.get("label")
            or entry.get("name")
            or ""
        )
        if modality and modality not in seen_mod:
            seen_mod.add(modality)
            modalities.append(str(modality))
        if label and label not in seen_lab:
            seen_lab.add(label)
            labels.append(str(label))
    return modalities, labels


def load_targets_table(targets_dir: Path) -> dict[str, dict]:
    """Return {ensembl_id: {approvedSymbol, pathways, tractability_modalities,
    tractability_labels, loeuf}}.

    Pulls pathways (Reactome IDs/names), tractability (per-modality
    druggability buckets), and gnomAD-derived LoF constraint (LOEUF)
    from the OT `targets/` parquet in a single read. Columns are
    detected leniently to tolerate the column-name drift OT introduces
    between releases.
    """
    schema = pq.ParquetFile(
        next(iter(targets_dir.glob("*.parquet")))
    ).schema_arrow
    available = {f.name for f in schema}

    id_col = _first_present(available, "id", "ensemblId", "ensembl_id")
    symbol_col = _first_present(
        available, "approvedSymbol", "approved_symbol", "symbol"
    )
    pathways_col = _first_present(available, "pathways", "reactome")
    tractability_col = _first_present(
        available, "tractability", "tractabilityAssessments",
    )
    constraint_col = _first_present(
        available, "constraint", "geneticConstraint", "constraints",
    )
    if not id_col:
        raise RuntimeError(
            f"targets dataset lacks an id/ensemblId column; saw {sorted(available)}"
        )
    cols = [
        c for c in (id_col, symbol_col, pathways_col, tractability_col, constraint_col)
        if c
    ]
    tbl = pads.dataset(str(targets_dir), format="parquet").to_table(columns=cols)
    out: dict[str, dict] = {}
    rows = tbl.to_pylist()
    for row in rows:
        ensembl = row.get(id_col)
        if not ensembl:
            continue
        pathways = []
        raw_pathways = row.get(pathways_col) if pathways_col else None
        if raw_pathways:
            for p in raw_pathways:
                if isinstance(p, dict):
                    name = p.get("pathway") or p.get("name")
                    if name:
                        pathways.append(name)
                elif isinstance(p, str):
                    pathways.append(p)
        modalities, labels = _extract_tractability(
            row.get(tractability_col) if tractability_col else None
        )
        loeuf = _extract_loeuf(
            row.get(constraint_col) if constraint_col else None
        )
        out[str(ensembl)] = {
            "approvedSymbol": row.get(symbol_col) if symbol_col else None,
            "pathways": pathways,
            "tractability_modalities": modalities,
            "tractability_labels": labels,
            "loeuf": loeuf,
        }
    n_tract = sum(1 for v in out.values() if v["tractability_modalities"])
    n_loeuf = sum(1 for v in out.values() if v["loeuf"] is not None)
    logger.info(
        "Loaded targets table: %d ensembl ids (pathways=%s, "
        "tractability=%d, loeuf=%d)",
        len(out), bool(pathways_col), n_tract, n_loeuf,
    )
    return out


def load_associations_table(
    associations_dir: Path,
    keep_ensembls: set[str],
) -> dict[str, list[tuple[str, float]]]:
    """Return {ensembl_id: [(efo_id, genetic_score), ...]} for ensembls in ``keep_ensembls``.

    The full OT association dataset is ~10M rows of (target × disease ×
    datatype × score); we filter to the `genetic_association` datatype
    and only retain rows for ensembls that some drug actually targets.
    That keeps the join lean — the snapshot is sized to fit a single
    SQLite file rather than mirroring the whole OT release.
    """
    schema = pq.ParquetFile(
        next(iter(associations_dir.glob("*.parquet")))
    ).schema_arrow
    available = {f.name for f in schema}

    target_col = _first_present(
        available, "targetId", "target_id", "ensemblId",
    )
    disease_col = _first_present(
        available, "diseaseId", "disease_id", "efoId",
    )
    datatype_col = _first_present(
        available, "datatypeId", "datatype_id", "datatype",
    )
    score_col = _first_present(
        available, "score", "associationScore",
    )
    if not (target_col and disease_col and score_col):
        raise RuntimeError(
            f"associations dataset is missing required columns; saw {sorted(available)}"
        )
    cols = [c for c in (target_col, disease_col, datatype_col, score_col) if c]

    tbl = pads.dataset(str(associations_dir), format="parquet").to_table(columns=cols)
    out: dict[str, list[tuple[str, float]]] = {}
    n_seen = 0
    n_kept = 0
    for row in tbl.to_pylist():
        n_seen += 1
        if datatype_col:
            dt = row.get(datatype_col)
            if dt and str(dt) != _GENETIC_DATATYPE:
                continue
        ensembl = row.get(target_col)
        if not ensembl or str(ensembl) not in keep_ensembls:
            continue
        efo = row.get(disease_col)
        score = row.get(score_col)
        if not efo or score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        out.setdefault(str(ensembl), []).append((str(efo), score_f))
        n_kept += 1
    logger.info(
        "Loaded associations table: %d/%d rows kept after "
        "datatype=%s + target-ensembl filter (covers %d ensembls)",
        n_kept, n_seen, _GENETIC_DATATYPE, len(out),
    )
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _dedup_cap(items: list[str], cap: int) -> list[str]:
    """Preserve insertion order, dedup, keep the first ``cap`` items."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def aggregate_drug_level(
    molecule_tbl,
    moa_tbl,
    targets_map: dict[str, dict],
) -> dict[str, dict]:
    """Return {chembl_id: {name, moa_text, action_type, target_symbols,
    target_ensembl, pathways}} aggregated across all MoA rows for that drug."""
    # Start from the molecule table so every drug gets a row, even if
    # it has no MoA.
    agg: dict[str, dict] = {}
    for row in molecule_tbl.to_pylist():
        chembl_id = row.get("id")
        if not chembl_id:
            continue
        agg[str(chembl_id)] = {
            "name": row.get("name"),
            "moa_list": [],
            "action_list": [],
            "target_ensembl": [],
        }

    for row in moa_tbl.to_pylist():
        chembl_ids = row.get("chembl_ids") or []
        if not chembl_ids:
            continue
        moa_text = row.get("mechanism_of_action")
        action_type = row.get("action_type")
        targets = row.get("targets") or []
        for cid in chembl_ids:
            bucket = agg.get(str(cid))
            if bucket is None:
                # MoA points at a ChEMBL ID we don't have a molecule row
                # for — create a minimal bucket so the data still lands.
                bucket = {
                    "name": None,
                    "moa_list": [],
                    "action_list": [],
                    "target_ensembl": [],
                }
                agg[str(cid)] = bucket
            if moa_text:
                bucket["moa_list"].append(moa_text)
            if action_type:
                bucket["action_list"].append(action_type)
            for t in targets:
                if t:
                    bucket["target_ensembl"].append(str(t))

    # Resolve Ensembl IDs to approved symbols + union their pathways +
    # aggregate tractability buckets / LOEUF across the drug's targets.
    # Tractability: union of (modalities, labels) across all targets.
    # LOEUF: min across targets (lower = more constrained gene).
    for bucket in agg.values():
        ensembls = _dedup_cap(bucket.pop("target_ensembl"), _TARGET_SYMBOL_CAP)
        symbols: list[str] = []
        pathways: list[str] = []
        modalities: list[str] = []
        labels: list[str] = []
        loeuf_vals: list[float] = []
        for ensembl in ensembls:
            info = targets_map.get(ensembl)
            if not info:
                continue
            if info.get("approvedSymbol"):
                symbols.append(str(info["approvedSymbol"]))
            for p in info.get("pathways") or []:
                pathways.append(p)
            for m in info.get("tractability_modalities") or []:
                modalities.append(str(m))
            for lbl in info.get("tractability_labels") or []:
                labels.append(str(lbl))
            loeuf = info.get("loeuf")
            if loeuf is not None:
                loeuf_vals.append(float(loeuf))
        bucket["target_ensembl"] = ensembls
        bucket["target_symbols"] = _dedup_cap(symbols, _TARGET_SYMBOL_CAP)
        bucket["pathways"] = _dedup_cap(pathways, _PATHWAY_CAP)
        bucket["tractability_modalities"] = _dedup_cap(modalities, _TRACTABILITY_LABEL_CAP)
        bucket["tractability_labels"] = _dedup_cap(labels, _TRACTABILITY_LABEL_CAP)
        bucket["loeuf_min"] = min(loeuf_vals) if loeuf_vals else None
        bucket["moa_text"] = "|".join(_dedup_cap(bucket.pop("moa_list"), _MOA_CAP))
        bucket["action_type"] = "|".join(
            _dedup_cap(bucket.pop("action_list"), _ACTION_TYPE_CAP)
        )
    return agg


def _flatten_indication_list(
    chembl_id: str, indications,
) -> list[tuple[str, str | None, str | None, int | None]]:
    """Expand an `indications` list (nested struct or plain) into rows.

    Handles both the OT 23.x/24.x in-molecule shape (list[struct]) and
    the 25.x+ standalone drug_indication shape (list[struct] with
    `disease`, `efoName`, `maxPhaseForIndication`). Returns one tuple
    per (drug, indication) pair: (chembl_id, efo_id, efo_name, max_phase).
    """
    rows: list[tuple[str, str | None, str | None, int | None]] = []
    # OT sometimes nests indications under "rows": [ ... ].
    if isinstance(indications, dict) and "rows" in indications:
        indications = indications["rows"] or []
    indications = indications or []
    for ind in indications:
        if not isinstance(ind, dict):
            continue
        efo_id = (
            ind.get("disease")
            or ind.get("efoId")
            or ind.get("efo_id")
        )
        efo_name = (
            ind.get("efoName")
            or ind.get("disease_name")
            or ind.get("name")
        )
        max_phase = (
            ind.get("maxPhaseForIndication")
            or ind.get("max_phase_for_indication")
            or ind.get("maxPhase")
        )
        if max_phase is None and not efo_id and not efo_name:
            continue
        rows.append(
            (
                str(chembl_id),
                str(efo_id) if efo_id else None,
                str(efo_name) if efo_name else None,
                int(max_phase) if max_phase is not None else None,
            )
        )
    return rows


def flatten_indications(molecule_tbl) -> list[tuple[str, str | None, str | None, int | None]]:
    """Yield (chembl_id, efo_id, efo_name, max_phase) from the molecule table.

    Only used for OT releases where indications are nested inside the
    molecule dataset (23.x/24.x). In 25.x+ indications moved to their
    own dataset — see `load_indication_table`.
    """
    if "indications" not in molecule_tbl.column_names:
        return []
    rows: list[tuple[str, str | None, str | None, int | None]] = []
    for row in molecule_tbl.to_pylist():
        chembl_id = row.get("id")
        if not chembl_id:
            continue
        rows.extend(_flatten_indication_list(str(chembl_id), row.get("indications")))
    return rows


def load_indication_table(
    indication_dir: Path,
) -> list[tuple[str, str | None, str | None, int | None]]:
    """Read the standalone OT 25.x+ drug_indication dataset.

    Schema is {id: chembl_id, indications: list<struct<disease, efoName,
    maxPhaseForIndication>>}. Returns the same flat (chembl_id, efo_id,
    efo_name, max_phase) rows as `flatten_indications`.
    """
    schema = pq.ParquetFile(
        next(iter(indication_dir.glob("*.parquet")))
    ).schema_arrow
    available = {f.name for f in schema}
    id_col = _first_present(available, "id", "chemblId", "chembl_id")
    ind_col = _first_present(available, "indications", "indication")
    if not (id_col and ind_col):
        raise RuntimeError(
            f"drug_indication dataset is missing id/indications columns; "
            f"saw {sorted(available)}"
        )
    tbl = pads.dataset(str(indication_dir), format="parquet").to_table(
        columns=[id_col, ind_col]
    )
    rows: list[tuple[str, str | None, str | None, int | None]] = []
    for raw in tbl.to_pylist():
        chembl_id = raw.get(id_col)
        if not chembl_id:
            continue
        rows.extend(
            _flatten_indication_list(str(chembl_id), raw.get(ind_col))
        )
    logger.info(
        "Loaded drug_indication table: %d (drug, indication) rows",
        len(rows),
    )
    return rows


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_snapshot(
    out_path: Path,
    drug_agg: dict[str, dict],
    indications: list[tuple[str, str | None, str | None, int | None]],
    name_index: dict[str, list[str]],
    associations: dict[str, list[tuple[str, float]]],
    release: int,
) -> None:
    """Write the slim SQLite snapshot with all tables and the release stamp.

    ``associations`` maps {ensembl_id: [(efo_id, genetic_score), ...]} —
    pre-filtered to the genetic_association datatype and the ensembls
    that some drug actually targets. May be empty if the OT release
    didn't ship an associations dir.
    """
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(out_path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(
            """
            CREATE TABLE name_opentargets_drug (
                query_norm                TEXT NOT NULL,
                chembl_id                 TEXT NOT NULL,
                drug_name                 TEXT,
                moa_text                  TEXT,
                action_type               TEXT,
                target_symbols            TEXT,
                target_ensembl            TEXT,
                pathways                  TEXT,
                tractability_modalities   TEXT,
                tractability_labels       TEXT,
                loeuf_min                 REAL,
                PRIMARY KEY (query_norm, chembl_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE name_opentargets_indications (
                chembl_id               TEXT NOT NULL,
                indication_efo_id       TEXT,
                indication_name         TEXT,
                indication_max_phase    INTEGER,
                PRIMARY KEY (chembl_id, indication_efo_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE name_opentargets_target_disease_evidence (
                chembl_id      TEXT NOT NULL,
                ensembl_id     TEXT NOT NULL,
                efo_id         TEXT NOT NULL,
                genetic_score  REAL NOT NULL,
                PRIMARY KEY (chembl_id, ensembl_id, efo_id)
            )
            """
        )

        # Invert the name index so we can look up query_norms by chembl_id.
        by_chembl: dict[str, list[str]] = {}
        for qn, cids in name_index.items():
            for cid in cids:
                by_chembl.setdefault(str(cid), []).append(qn)

        # Also allow a fallback by canonicalized drug_name when we have no
        # ChEMBL mapping (i.e. the drug isn't in our ChEMBL targets
        # snapshot). That still lets the enrichment hit by exact-name.
        drug_rows: list[tuple] = []
        seen: set[tuple[str, str]] = set()
        for chembl_id, bucket in drug_agg.items():
            query_norms = list(by_chembl.get(chembl_id, []))
            name = bucket.get("name")
            if not query_norms and name:
                qn = canonicalize_drug_name(name)
                if qn:
                    query_norms = [qn]
            if not query_norms:
                continue
            for qn in query_norms:
                if (qn, chembl_id) in seen:
                    continue
                seen.add((qn, chembl_id))
                drug_rows.append(
                    (
                        qn,
                        chembl_id,
                        name,
                        bucket.get("moa_text") or None,
                        bucket.get("action_type") or None,
                        "|".join(bucket.get("target_symbols") or []) or None,
                        "|".join(bucket.get("target_ensembl") or []) or None,
                        "|".join(bucket.get("pathways") or []) or None,
                        "|".join(bucket.get("tractability_modalities") or []) or None,
                        "|".join(bucket.get("tractability_labels") or []) or None,
                        bucket.get("loeuf_min"),
                    )
                )

        if drug_rows:
            conn.executemany(
                "INSERT INTO name_opentargets_drug VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                drug_rows,
            )

        # Indications: keep only rows for drugs that landed in name_opentargets_drug.
        retained_chembl_ids = {row[1] for row in drug_rows}
        ind_rows = [r for r in indications if r[0] in retained_chembl_ids]
        if ind_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO name_opentargets_indications VALUES "
                "(?, ?, ?, ?)",
                ind_rows,
            )

        # Target-disease evidence: cross each drug's targets with the
        # pre-filtered associations to produce (chembl_id, ensembl_id,
        # efo_id, genetic_score) rows. Kept in long form so the
        # enrichment can aggregate (max across targets) at lookup time.
        evidence_rows: list[tuple[str, str, str, float]] = []
        if associations:
            for chembl_id, bucket in drug_agg.items():
                if chembl_id not in retained_chembl_ids:
                    continue
                for ensembl in bucket.get("target_ensembl") or []:
                    for efo, score in associations.get(str(ensembl), ()):
                        evidence_rows.append(
                            (str(chembl_id), str(ensembl), str(efo), float(score))
                        )
        if evidence_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO "
                "name_opentargets_target_disease_evidence VALUES "
                "(?, ?, ?, ?)",
                evidence_rows,
            )

        conn.execute(
            "CREATE INDEX idx_ot_query_norm ON name_opentargets_drug(query_norm)"
        )
        conn.execute(
            "CREATE INDEX idx_ot_ind_chembl ON name_opentargets_indications(chembl_id)"
        )
        conn.execute(
            "CREATE INDEX idx_ot_evidence_chembl_efo "
            "ON name_opentargets_target_disease_evidence(chembl_id, efo_id)"
        )
        conn.execute(f"PRAGMA user_version = {release}")
        conn.commit()

        n_drugs = conn.execute(
            "SELECT COUNT(DISTINCT chembl_id) FROM name_opentargets_drug"
        ).fetchone()[0]
        n_norms = conn.execute(
            "SELECT COUNT(DISTINCT query_norm) FROM name_opentargets_drug"
        ).fetchone()[0]
        n_inds = conn.execute(
            "SELECT COUNT(*) FROM name_opentargets_indications"
        ).fetchone()[0]
        n_ev = conn.execute(
            "SELECT COUNT(*) FROM name_opentargets_target_disease_evidence"
        ).fetchone()[0]
        n_tract = conn.execute(
            "SELECT COUNT(*) FROM name_opentargets_drug "
            "WHERE tractability_modalities IS NOT NULL"
        ).fetchone()[0]
        n_loeuf = conn.execute(
            "SELECT COUNT(*) FROM name_opentargets_drug WHERE loeuf_min IS NOT NULL"
        ).fetchone()[0]
        logger.info(
            "Wrote OT snapshot: %d drugs | %d query_norms | %d indications | "
            "%d genetic-evidence rows | tractability=%d | loeuf=%d (release=%d)",
            n_drugs, n_norms, n_inds, n_ev, n_tract, n_loeuf, release,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_snapshot(
    opentargets_dir: Path,
    chembl_snapshot: Path,
    out_path: Path,
    release: int,
) -> None:
    molecule_dir = resolve_subdir(opentargets_dir, _MOLECULE_CANDIDATE_DIRS)
    moa_dir = resolve_subdir(opentargets_dir, _MOA_CANDIDATE_DIRS)
    targets_dir = resolve_subdir(opentargets_dir, _TARGETS_CANDIDATE_DIRS)
    logger.info("molecule dir:           %s", molecule_dir)
    logger.info("mechanismOfAction dir:  %s", moa_dir)
    logger.info("targets dir:            %s", targets_dir)

    # Indications live in their own dataset from OT 25.x onward; earlier
    # releases nest them inside molecule. Try the dedicated dir first,
    # fall back to the molecule-nested shape.
    indication_dir: Path | None = None
    for name in _INDICATION_CANDIDATE_DIRS:
        candidate = opentargets_dir / name
        if candidate.exists() and candidate.is_dir():
            indication_dir = candidate
            break
    if indication_dir is not None:
        logger.info("drug_indication dir:    %s", indication_dir)
    else:
        logger.info(
            "drug_indication dir:    (none — falling back to "
            "molecule.indications)"
        )

    # associationByDatatypeDirect is optional — if missing the
    # target-disease evidence table is left empty and the enrichment
    # falls back to skipping the genetic-score feature.
    associations_dir: Path | None = None
    for name in _ASSOCIATIONS_CANDIDATE_DIRS:
        candidate = opentargets_dir / name
        if candidate.exists() and candidate.is_dir():
            associations_dir = candidate
            break
    if associations_dir is not None:
        logger.info("associations dir:       %s", associations_dir)
    else:
        logger.info(
            "associations dir:       (none — genetic-evidence table will be empty; "
            "sync associationByDatatypeDirect/ to populate)"
        )

    name_index = load_chembl_name_index(chembl_snapshot)
    molecule_tbl = load_molecule_table(molecule_dir)
    moa_tbl = load_moa_table(moa_dir)
    targets_map = load_targets_table(targets_dir)

    drug_agg = aggregate_drug_level(molecule_tbl, moa_tbl, targets_map)
    if indication_dir is not None:
        indications = load_indication_table(indication_dir)
    else:
        indications = flatten_indications(molecule_tbl)

    # Filter the assocations dataset down to the ensembl set that any
    # drug in our aggregation actually targets — the full table is too
    # big to keep in memory at OT 25.x scale.
    drug_target_ensembls: set[str] = set()
    for bucket in drug_agg.values():
        for ensembl in bucket.get("target_ensembl") or []:
            drug_target_ensembls.add(str(ensembl))
    if associations_dir is not None and drug_target_ensembls:
        associations = load_associations_table(associations_dir, drug_target_ensembls)
    else:
        associations = {}

    write_snapshot(out_path, drug_agg, indications, name_index, associations, release)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build a slim name-keyed OpenTargets snapshot "
            "(drug-level aggregates + per-indication max phases)."
        )
    )
    parser.add_argument(
        "--opentargets-dir",
        type=Path,
        required=True,
        help=(
            "Root of manually-downloaded OT Parquet (containing "
            "molecule/, mechanismOfAction/, and targets/ subdirs)."
        ),
    )
    parser.add_argument(
        "--chembl-snapshot",
        type=Path,
        required=True,
        help=(
            "Path to the ChEMBL targets SQLite snapshot "
            "(scripts/build_chembl_targets_snapshot.py output) — used "
            "to derive query_norm -> chembl_id mappings."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output SQLite path for the slim OT snapshot.",
    )
    parser.add_argument(
        "--release",
        type=int,
        default=None,
        help="OT release number to stamp via PRAGMA user_version. "
             "Defaults to a best-effort parse of --opentargets-dir's name.",
    )
    args = parser.parse_args(argv)

    if not args.opentargets_dir.exists():
        parser.error(f"--opentargets-dir not found: {args.opentargets_dir}")
    if not args.chembl_snapshot.exists():
        parser.error(f"--chembl-snapshot not found: {args.chembl_snapshot}")

    release = args.release if args.release is not None else parse_release_number(
        args.opentargets_dir
    )

    build_snapshot(
        args.opentargets_dir,
        args.chembl_snapshot,
        args.out,
        release,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
