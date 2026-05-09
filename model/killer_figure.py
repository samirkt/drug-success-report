"""Killer-figure NN analog lookup (LOA Playbook §3.1 Part C).

For any drug candidate, the retriever returns the k=10 nearest neighbors
in the joint molecule × target × indication space, computes the
historical approval rate among them, compares it against the stratum
base rate, and emits a structured "because Z" decomposition.

Joint similarity is a weighted average of three component scores in
[0,1]:
  - molecule: MolFormer cosine when both query and pool row have
    embeddings; ECFP4 Tanimoto fallback when only fingerprints are
    present.
  - target: Jaccard over UniProt accession sets.
  - indication: most-specific available signal, MeSH tree-prefix Jaccard
    > ICD-10 3-char-prefix Jaccard > disease_area exact match.

Components missing for either side of a (query, pool) pair are dropped
and the remaining weights re-normalized.

Pool is `df` filtered to candidates with `earliest_start_date` strictly
earlier than the query's start date. Post-retrieval the neighbors are
deduped by `drug_name` (best similarity per drug) before truncation to
`k`.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .baselines.stratum import StratumBaseline, _first_code
from .features.nn_similarity import _has_emb, _has_fp, _to_bitvect, _to_ordinal

logger = logging.getLogger(__name__)


_ECFP4_BITS = 2048
_EMB_DIM = 768
_DEFAULT_K = 10
_DEFAULT_MIN_NEIGHBORS = 5
_DEFAULT_WEIGHTS: dict[str, float] = {
    "molecule": 1.0 / 3.0,
    "target": 1.0 / 3.0,
    "indication": 1.0 / 3.0,
}
_TOP_DECOMP = 3  # how many entries to surface per shared-attribute list


# ---------------------------------------------------------------------------
# Component-similarity helpers (pure functions)
# ---------------------------------------------------------------------------


def _coerce_to_set(arr) -> set[str]:
    if arr is None:
        return set()
    if isinstance(arr, float) and np.isnan(arr):
        return set()
    try:
        return {str(t) for t in arr if t is not None and str(t).strip()}
    except TypeError:
        return set()


def _icd10_prefix_set(arr) -> set[str]:
    return {c[:3] for c in _coerce_to_set(arr)}


def _icd10_chapter(arr) -> Optional[str]:
    s = _coerce_to_set(arr)
    if not s:
        return None
    # Pick the lexicographically smallest first letter — deterministic and
    # matches StratumBaseline's behavior of taking the first code.
    return sorted(s)[0][:1]


def _mesh_prefix_set(arr) -> set[str]:
    """MeSH tree-number 3-char prefixes (e.g. 'C04.557.580' -> 'C04')."""
    return {c[:3] for c in _coerce_to_set(arr)}


def _disease_area_or_none(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def _jaccard(a: set, b: set) -> Optional[float]:
    """Jaccard, or None when either side is empty.

    "Empty on one side" is treated as missing signal — not zero overlap —
    so the component drops out of joint_sim via weight re-normalization
    rather than dragging down candidates that simply lack annotation.
    """
    if not a or not b:
        return None
    return float(len(a & b) / len(a | b))


# ---------------------------------------------------------------------------
# Pool entry + report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _PoolEntry:
    candidate_id: str
    drug_name: Optional[str]
    outcome: str
    y: int
    earliest_start_date: Optional[date]
    earliest_start_date_ord: int
    targets: set
    icd10_prefixes: set
    icd10_chapter: Optional[str]
    mesh_prefixes: set
    disease_area: Optional[str]
    has_fp: bool
    has_emb: bool
    fp_bv: object  # rdkit ExplicitBitVect or None
    emb_unit: Optional[np.ndarray]


@dataclass
class KillerFigureNeighbor:
    candidate_id: str
    drug_name: Optional[str]
    outcome: str
    y: int
    earliest_start_date: Optional[str]
    joint_sim: float
    mol_sim: Optional[float]
    target_sim: Optional[float]
    indication_sim: Optional[float]
    n_components_used: int
    mol_metric: Optional[str]  # "molformer_cosine" | "tanimoto" | None


@dataclass
class KillerFigureReport:
    candidate_id: Optional[str]
    drug_name: Optional[str]
    query_start_date: Optional[str]
    approval_rate_neighbors: float
    stratum_base_rate: float
    pool_base_rate: float
    n_approved: int
    n_failed: int
    n_neighbors: int
    mean_sim_by_space: dict
    shared_targets: list
    shared_icd10_chapters: list
    shared_disease_areas: list
    neighbors: list
    insufficient_prior_art: bool
    n_components_query: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@dataclass
class KillerFigureRetriever:
    """k-NN retriever in joint molecule × target × indication space.

    Use `fit(df, y)` once, then `retrieve(query_row)` per candidate.
    """

    weights: Mapping[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    k: int = _DEFAULT_K
    min_neighbors: int = _DEFAULT_MIN_NEIGHBORS

    def __post_init__(self) -> None:
        for c in ("molecule", "target", "indication"):
            if c not in self.weights:
                raise ValueError(f"weights missing component {c!r}")
        self._entries: list[_PoolEntry] = []
        self._dates_ord: np.ndarray = np.empty(0, dtype=np.int64)
        self._sort_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._emb_matrix: Optional[np.ndarray] = None
        self._has_emb_mask: np.ndarray = np.empty(0, dtype=bool)
        self._has_fp_mask: np.ndarray = np.empty(0, dtype=bool)
        self._fp_bvs: list = []
        self._stratum: Optional[StratumBaseline] = None
        self._global_pool_rate: float = 0.0
        self._n_fit: int = 0

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> None:
        if len(df) != len(y):
            raise ValueError("df and y length mismatch")
        self._n_fit = len(df)

        entries: list[_PoolEntry] = []
        for i, row in enumerate(df.itertuples(index=False)):
            d = row._asdict()
            ordn = _to_ordinal(d.get("earliest_start_date"))
            if ordn is None:
                continue  # no date → can't temporally place this row, skip
            ecfp4 = d.get("ecfp4")
            embedding = d.get("embedding")
            has_fp = _has_fp(ecfp4)
            has_emb = _has_emb(embedding)
            emb_unit: Optional[np.ndarray] = None
            if has_emb:
                arr = np.asarray(embedding, dtype=np.float32)
                n = float(np.linalg.norm(arr))
                if n > 0:
                    emb_unit = arr / n
                else:
                    has_emb = False  # zero-vector embedding is unusable

            entries.append(
                _PoolEntry(
                    candidate_id=str(d.get("candidate_id")),
                    drug_name=(str(d["drug_name"]) if d.get("drug_name") else None),
                    outcome=str(d.get("outcome", "")),
                    y=int(y[i]),
                    earliest_start_date=d.get("earliest_start_date"),
                    earliest_start_date_ord=ordn,
                    targets=_coerce_to_set(d.get("drug_targets")),
                    icd10_prefixes=_icd10_prefix_set(d.get("icd10_codes")),
                    icd10_chapter=_icd10_chapter(d.get("icd10_codes")),
                    mesh_prefixes=_mesh_prefix_set(d.get("mesh_condition_tree_numbers")),
                    disease_area=_disease_area_or_none(d.get("disease_area")),
                    has_fp=has_fp,
                    has_emb=has_emb,
                    fp_bv=_to_bitvect(ecfp4) if has_fp else None,
                    emb_unit=emb_unit,
                )
            )

        # Sort by date ascending so date cutoff is a single searchsorted.
        entries.sort(key=lambda e: e.earliest_start_date_ord)
        self._entries = entries
        self._dates_ord = np.asarray(
            [e.earliest_start_date_ord for e in entries], dtype=np.int64
        )
        self._has_emb_mask = np.asarray([e.has_emb for e in entries], dtype=bool)
        self._has_fp_mask = np.asarray([e.has_fp for e in entries], dtype=bool)
        self._fp_bvs = [e.fp_bv for e in entries]

        if self._has_emb_mask.any():
            self._emb_matrix = np.vstack(
                [e.emb_unit for e in entries if e.has_emb]
            ).astype(np.float32)
        else:
            self._emb_matrix = None

        # Pool base rate among all labeled rows (used as one of three comparators).
        self._global_pool_rate = (
            float(np.mean([e.y for e in entries])) if entries else 0.0
        )

        # Fit a stratum baseline on the same df + y for the Y% comparator.
        self._stratum = StratumBaseline()
        self._stratum.fit(df, y)

        logger.info(
            "killer_figure: fit on %d rows; pool=%d (%d w/ fp, %d w/ emb); "
            "global pool rate=%.4f",
            self._n_fit,
            len(entries),
            int(self._has_fp_mask.sum()),
            int(self._has_emb_mask.sum()),
            self._global_pool_rate,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _stratum_rate_for(self, query_row: Mapping) -> float:
        if self._stratum is None:
            return 0.0
        df = pd.DataFrame([{"icd10_codes": query_row.get("icd10_codes")}])
        rate = float(self._stratum.predict_proba(df)[0])
        return rate

    def _pool_base_rate(self, cutoff: int) -> float:
        if cutoff <= 0 or not self._entries:
            return 0.0
        ys = [self._entries[i].y for i in range(cutoff)]
        return float(np.mean(ys)) if ys else 0.0

    def _component_sim_target(
        self, query_targets: set, entry: _PoolEntry
    ) -> Optional[float]:
        return _jaccard(query_targets, entry.targets)

    def _component_sim_indication(
        self,
        query_mesh: set,
        query_icd_pref: set,
        query_disease: Optional[str],
        entry: _PoolEntry,
    ) -> Optional[float]:
        if query_mesh and entry.mesh_prefixes:
            return _jaccard(query_mesh, entry.mesh_prefixes)
        if query_icd_pref and entry.icd10_prefixes:
            return _jaccard(query_icd_pref, entry.icd10_prefixes)
        if query_disease and entry.disease_area:
            return 1.0 if query_disease == entry.disease_area else 0.0
        return None

    def _mol_sims(
        self,
        cutoff: int,
        q_has_emb: bool,
        q_emb_unit: Optional[np.ndarray],
        q_has_fp: bool,
        q_fp_bv,
    ) -> tuple[np.ndarray, list[Optional[str]]]:
        """Return (mol_sims_array, mol_metrics) of length `cutoff`.

        Each entry in mol_sims is NaN when the molecule component is
        missing for that pool row given the query. mol_metrics tracks
        which metric was used per row so the report can surface it.
        """
        n = cutoff
        sims = np.full(n, np.nan, dtype=np.float64)
        metrics: list[Optional[str]] = [None] * n

        if n == 0:
            return sims, metrics

        emb_mask_sub = self._has_emb_mask[:n]
        fp_mask_sub = self._has_fp_mask[:n]

        # Pair (1) — both have embedding → MolFormer cosine, vectorized.
        if q_has_emb and self._emb_matrix is not None and emb_mask_sub.any():
            # _emb_matrix indexes into the *embedding-only* subspace, in the
            # same global order as self._entries. Compute its row indices
            # within the subset.
            emb_global_idx = np.flatnonzero(self._has_emb_mask)
            in_window = emb_global_idx < n
            sub_global_idx = emb_global_idx[in_window]  # global pool indices
            sub_mat = self._emb_matrix[in_window]
            cos = sub_mat @ q_emb_unit  # (M_sub,)
            sims[sub_global_idx] = cos
            for gi in sub_global_idx:
                metrics[int(gi)] = "molformer_cosine"

        # Pair (2) — Tanimoto fallback for pool rows that did NOT get a
        # cosine score AND have a fingerprint AND query has a fingerprint.
        if q_has_fp:
            need_fallback = np.isnan(sims) & fp_mask_sub
            if need_fallback.any():
                from rdkit.DataStructs import BulkTanimotoSimilarity

                fb_idx = np.flatnonzero(need_fallback)
                bvs = [self._fp_bvs[int(i)] for i in fb_idx]
                tani = np.asarray(
                    BulkTanimotoSimilarity(q_fp_bv, bvs), dtype=np.float64
                )
                sims[fb_idx] = tani
                for gi in fb_idx:
                    metrics[int(gi)] = "tanimoto"

        return sims, metrics

    # ------------------------------------------------------------------
    # Public retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query_row: Mapping) -> KillerFigureReport:
        """Return a structured killer-figure report for `query_row`.

        `query_row` can be a dict, a pd.Series, or any mapping-shaped
        object exposing the candidate_detail columns.
        """
        # ---- Materialize query payloads -------------------------------
        get = (
            query_row.get if hasattr(query_row, "get") else lambda k, d=None: query_row[k]
            if k in query_row
            else d
        )
        q_id = get("candidate_id")
        q_drug = get("drug_name")
        q_date = get("earliest_start_date")
        q_date_ord = _to_ordinal(q_date)
        q_ecfp4 = get("ecfp4")
        q_emb = get("embedding")
        q_has_fp = _has_fp(q_ecfp4)
        q_fp_bv = _to_bitvect(q_ecfp4) if q_has_fp else None
        q_emb_unit: Optional[np.ndarray] = None
        q_has_emb = _has_emb(q_emb)
        if q_has_emb:
            arr = np.asarray(q_emb, dtype=np.float32)
            n_norm = float(np.linalg.norm(arr))
            if n_norm > 0:
                q_emb_unit = arr / n_norm
            else:
                q_has_emb = False

        q_targets = _coerce_to_set(get("drug_targets"))
        q_icd_pref = _icd10_prefix_set(get("icd10_codes"))
        q_mesh = _mesh_prefix_set(get("mesh_condition_tree_numbers"))
        q_disease = _disease_area_or_none(get("disease_area"))

        n_comp_query = sum(
            [
                bool(q_has_emb or q_has_fp),
                bool(q_targets),
                bool(q_mesh or q_icd_pref or q_disease),
            ]
        )

        stratum_rate = self._stratum_rate_for({"icd10_codes": get("icd10_codes")})

        # ---- Date cutoff ---------------------------------------------
        if q_date_ord is None:
            cutoff = 0
        else:
            cutoff = int(np.searchsorted(self._dates_ord, q_date_ord, side="left"))

        pool_base = self._pool_base_rate(cutoff)

        # Insufficient-prior-art floor counts unique drug names in window.
        unique_drugs = {
            self._entries[i].drug_name
            for i in range(cutoff)
            if self._entries[i].drug_name
        }
        if cutoff < self.min_neighbors or len(unique_drugs) < self.min_neighbors:
            return KillerFigureReport(
                candidate_id=str(q_id) if q_id is not None else None,
                drug_name=str(q_drug) if q_drug else None,
                query_start_date=_date_str(q_date),
                approval_rate_neighbors=stratum_rate,
                stratum_base_rate=stratum_rate,
                pool_base_rate=pool_base,
                n_approved=0,
                n_failed=0,
                n_neighbors=0,
                mean_sim_by_space={"molecule": None, "target": None, "indication": None},
                shared_targets=[],
                shared_icd10_chapters=[],
                shared_disease_areas=[],
                neighbors=[],
                insufficient_prior_art=True,
                n_components_query=n_comp_query,
            )

        # ---- Per-component similarities over the cutoff window -------
        mol_sims, mol_metrics = self._mol_sims(
            cutoff, q_has_emb, q_emb_unit, q_has_fp, q_fp_bv
        )
        target_sims = np.full(cutoff, np.nan, dtype=np.float64)
        ind_sims = np.full(cutoff, np.nan, dtype=np.float64)
        for i in range(cutoff):
            entry = self._entries[i]
            ts = self._component_sim_target(q_targets, entry)
            if ts is not None:
                target_sims[i] = ts
            ins = self._component_sim_indication(
                q_mesh, q_icd_pref, q_disease, entry
            )
            if ins is not None:
                ind_sims[i] = ins

        # ---- Joint sim with on-the-fly weight re-normalization -------
        w_mol = float(self.weights["molecule"])
        w_tgt = float(self.weights["target"])
        w_ind = float(self.weights["indication"])

        mol_present = ~np.isnan(mol_sims)
        tgt_present = ~np.isnan(target_sims)
        ind_present = ~np.isnan(ind_sims)
        weight_total = (
            mol_present * w_mol + tgt_present * w_tgt + ind_present * w_ind
        )
        contrib = (
            np.where(mol_present, mol_sims, 0.0) * w_mol
            + np.where(tgt_present, target_sims, 0.0) * w_tgt
            + np.where(ind_present, ind_sims, 0.0) * w_ind
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            joint_sims = np.where(weight_total > 0, contrib / weight_total, -np.inf)
        n_components_used = (
            mol_present.astype(np.int8)
            + tgt_present.astype(np.int8)
            + ind_present.astype(np.int8)
        )

        # ---- Drug-level dedup + top-k --------------------------------
        # Order by joint_sim desc; for each unique drug, keep its best row.
        order = np.argsort(-joint_sims, kind="stable")
        seen_drugs: set = set()
        chosen_idx: list[int] = []
        # Also exclude self-match if the query has a candidate_id present in
        # the pool (defensive — temporal cutoff is strict, so this is rare).
        for gi in order:
            gi_int = int(gi)
            if not np.isfinite(joint_sims[gi_int]):
                break
            entry = self._entries[gi_int]
            if q_id is not None and entry.candidate_id == str(q_id):
                continue
            key = entry.drug_name or f"__cid_{entry.candidate_id}"
            if key in seen_drugs:
                continue
            seen_drugs.add(key)
            chosen_idx.append(gi_int)
            if len(chosen_idx) >= self.k:
                break

        # If after dedup we still don't have enough unique drugs, treat
        # like insufficient prior art (rare but possible if the pre-cutoff
        # window is mostly one drug).
        if len(chosen_idx) < self.min_neighbors:
            return KillerFigureReport(
                candidate_id=str(q_id) if q_id is not None else None,
                drug_name=str(q_drug) if q_drug else None,
                query_start_date=_date_str(q_date),
                approval_rate_neighbors=stratum_rate,
                stratum_base_rate=stratum_rate,
                pool_base_rate=pool_base,
                n_approved=0,
                n_failed=0,
                n_neighbors=len(chosen_idx),
                mean_sim_by_space={"molecule": None, "target": None, "indication": None},
                shared_targets=[],
                shared_icd10_chapters=[],
                shared_disease_areas=[],
                neighbors=[],
                insufficient_prior_art=True,
                n_components_query=n_comp_query,
            )

        # ---- Build neighbor records + decomposition ------------------
        neighbors: list[KillerFigureNeighbor] = []
        for gi in chosen_idx:
            entry = self._entries[gi]
            mol_s = float(mol_sims[gi]) if mol_present[gi] else None
            tgt_s = float(target_sims[gi]) if tgt_present[gi] else None
            ind_s = float(ind_sims[gi]) if ind_present[gi] else None
            neighbors.append(
                KillerFigureNeighbor(
                    candidate_id=entry.candidate_id,
                    drug_name=entry.drug_name,
                    outcome=entry.outcome,
                    y=entry.y,
                    earliest_start_date=_date_str(entry.earliest_start_date),
                    joint_sim=float(joint_sims[gi]),
                    mol_sim=mol_s,
                    target_sim=tgt_s,
                    indication_sim=ind_s,
                    n_components_used=int(n_components_used[gi]),
                    mol_metric=mol_metrics[gi],
                )
            )

        ys = np.asarray([nb.y for nb in neighbors], dtype=np.int8)
        approval_rate = float(np.mean(ys))
        n_approved = int(ys.sum())
        n_failed = int(len(ys) - n_approved)

        # Per-space mean sim (across neighbors that have that component).
        def _mean_or_none(values: list[Optional[float]]) -> Optional[float]:
            xs = [v for v in values if v is not None]
            return float(np.mean(xs)) if xs else None

        mean_by_space = {
            "molecule": _mean_or_none([nb.mol_sim for nb in neighbors]),
            "target": _mean_or_none([nb.target_sim for nb in neighbors]),
            "indication": _mean_or_none([nb.indication_sim for nb in neighbors]),
        }

        # Shared-attribute decomposition: only count attributes the query
        # shares with the neighbor (intersection counts), not just neighbor
        # frequencies. This makes "because Z" actually attributable.
        target_counter: Counter = Counter()
        chap_counter: Counter = Counter()
        area_counter: Counter = Counter()
        for gi in chosen_idx:
            entry = self._entries[gi]
            for t in q_targets & entry.targets:
                target_counter[t] += 1
            if entry.icd10_chapter and entry.icd10_chapter in {
                c[:1] for c in q_icd_pref
            }:
                chap_counter[entry.icd10_chapter] += 1
            if entry.disease_area and q_disease and entry.disease_area == q_disease:
                area_counter[entry.disease_area] += 1

        shared_targets = [
            {"uniprot": k, "n": v}
            for k, v in target_counter.most_common(_TOP_DECOMP)
        ]
        shared_chapters = [
            {"chapter": k, "n": v}
            for k, v in chap_counter.most_common(_TOP_DECOMP)
        ]
        shared_areas = [
            {"area": k, "n": v}
            for k, v in area_counter.most_common(_TOP_DECOMP)
        ]

        return KillerFigureReport(
            candidate_id=str(q_id) if q_id is not None else None,
            drug_name=str(q_drug) if q_drug else None,
            query_start_date=_date_str(q_date),
            approval_rate_neighbors=approval_rate,
            stratum_base_rate=stratum_rate,
            pool_base_rate=pool_base,
            n_approved=n_approved,
            n_failed=n_failed,
            n_neighbors=len(neighbors),
            mean_sim_by_space=mean_by_space,
            shared_targets=shared_targets,
            shared_icd10_chapters=shared_chapters,
            shared_disease_areas=shared_areas,
            neighbors=[asdict(nb) for nb in neighbors],
            insufficient_prior_art=False,
            n_components_query=n_comp_query,
        )

    # ------------------------------------------------------------------
    def metadata(self) -> dict:
        return {
            "k": self.k,
            "min_neighbors": self.min_neighbors,
            "weights": dict(self.weights),
            "n_fit_rows": self._n_fit,
            "n_pool_rows": len(self._entries),
            "n_pool_with_fp": int(self._has_fp_mask.sum()) if len(self._entries) else 0,
            "n_pool_with_emb": int(self._has_emb_mask.sum()) if len(self._entries) else 0,
            "global_pool_rate": self._global_pool_rate,
        }


# ---------------------------------------------------------------------------
# Ad-hoc query builder + text formatter
# ---------------------------------------------------------------------------


def smiles_to_ecfp4(smiles: str) -> Optional[np.ndarray]:
    """Render a SMILES into a 2048-bit ECFP4 (radius=2) numpy array."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=_ECFP4_BITS)
    arr = np.zeros(_ECFP4_BITS, dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr


def build_adhoc_query(
    *,
    smiles: Optional[str] = None,
    targets: Optional[list[str]] = None,
    icd10: Optional[list[str]] = None,
    mesh_tree_numbers: Optional[list[str]] = None,
    disease_area: Optional[str] = None,
    start_date: Optional[date] = None,
    candidate_id: str = "<adhoc>",
    drug_name: Optional[str] = None,
) -> dict:
    """Construct a query_row dict for an ad-hoc (no-pool-row) lookup.

    Caller must provide at least one component (SMILES, targets, or
    indication codes/area). MolFormer embedding is not generated for
    ad-hoc queries; molecular similarity falls back to ECFP4 Tanimoto
    when SMILES is provided.
    """
    if start_date is None:
        start_date = date.today()
        logger.warning(
            "build_adhoc_query: no start_date provided; defaulting to %s. "
            "All retrieved neighbors will have earliest_start_date < this date.",
            start_date,
        )
    ecfp4: Optional[np.ndarray] = None
    if smiles:
        ecfp4 = smiles_to_ecfp4(smiles)
        if ecfp4 is None:
            logger.warning(
                "build_adhoc_query: SMILES %r failed to parse; molecular "
                "similarity will be missing for this query.",
                smiles,
            )

    return {
        "candidate_id": candidate_id,
        "drug_name": drug_name,
        "outcome": None,
        "earliest_start_date": start_date,
        "ecfp4": ecfp4,
        "embedding": None,
        "drug_targets": np.asarray(list(targets or []), dtype=object),
        "icd10_codes": np.asarray(list(icd10 or []), dtype=object),
        "mesh_condition_tree_numbers": np.asarray(
            list(mesh_tree_numbers or []), dtype=object
        ),
        "disease_area": disease_area,
    }


def format_text_summary(report: KillerFigureReport) -> str:
    """One-shot pretty rendering of the killer-figure report.

    Floats are rounded to 2 decimals — raw precision is retained in
    `report.to_dict()` for downstream JSON consumers.
    """

    def _pct(x: Optional[float]) -> str:
        return "—" if x is None else f"{100.0 * x:.0f}%"

    lines: list[str] = []
    head = report.drug_name or report.candidate_id or "<query>"
    lines.append(f"Killer-figure report for {head}")
    if report.query_start_date:
        lines.append(f"  Query start date: {report.query_start_date}")
    if report.insufficient_prior_art:
        lines.append("  ! Insufficient prior art — falling back to stratum rate.")
        lines.append(f"  Stratum base rate: {_pct(report.stratum_base_rate)}")
        return "\n".join(lines)

    delta = report.approval_rate_neighbors - report.stratum_base_rate
    lines.append(
        f"  k={report.n_neighbors} nearest neighbors → approval {_pct(report.approval_rate_neighbors)} "
        f"({report.n_approved}/{report.n_neighbors} approved, {report.n_failed} failed)"
    )
    lines.append(
        f"  Stratum base rate: {_pct(report.stratum_base_rate)} "
        f"(Δ {('+' if delta >= 0 else '')}{100*delta:.0f}pp)"
    )
    lines.append(f"  Pool base rate (pre-cutoff): {_pct(report.pool_base_rate)}")

    sims = report.mean_sim_by_space
    sim_parts = [
        f"{name}={sims[name]:.2f}"
        for name in ("molecule", "target", "indication")
        if sims.get(name) is not None
    ]
    if sim_parts:
        lines.append("  Mean similarity by space: " + ", ".join(sim_parts))

    if report.shared_targets:
        lines.append(
            "  Shared targets: "
            + ", ".join(f"{t['uniprot']} ({t['n']}/{report.n_neighbors})"
                        for t in report.shared_targets)
        )
    if report.shared_icd10_chapters:
        lines.append(
            "  Shared ICD-10 chapters: "
            + ", ".join(f"{t['chapter']} ({t['n']}/{report.n_neighbors})"
                        for t in report.shared_icd10_chapters)
        )
    if report.shared_disease_areas:
        lines.append(
            "  Shared disease areas: "
            + ", ".join(f"{t['area']} ({t['n']}/{report.n_neighbors})"
                        for t in report.shared_disease_areas)
        )

    lines.append("  Neighbors:")
    for nb in report.neighbors:
        name = nb.get("drug_name") or nb.get("candidate_id")
        date_s = nb.get("earliest_start_date") or "—"
        lines.append(
            f"    - {name:30s}  {nb['outcome']:>20s}  "
            f"sim={nb['joint_sim']:.2f}  start={date_s}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal date helpers
# ---------------------------------------------------------------------------


def _date_str(d) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, float) and np.isnan(d):
        return None
    if hasattr(d, "isoformat"):
        try:
            return d.isoformat()[:10]
        except Exception:
            return None
    try:
        return pd.Timestamp(d).date().isoformat()
    except Exception:
        return None
