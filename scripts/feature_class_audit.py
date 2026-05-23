"""One-shot feature-class audit for the LOA modeling dataset.

Lean PDF (default ``outputs/feature_class_audit.pdf``) — 9 pages:

  1. Summary           - per-representation table (sub-rows for each class
                         showing enc_dim + coverage), coverage by year,
                         missingness heatmap
  2-6. Per class       - molecular | disease | target | pathway | admet
                         text stats + richness distribution + outcome split
                         + coverage by year
  7. Top values        - most common values for one-hot / multi-hot columns
                         (pathway shown both ancestor-expanded and locally-leaf)
  8. Frequency tail    - target / pathway / disease ID rank-frequency
  9. PCA               - molecule embedding + disease/pathway multi-hot proxy

Run: ``uv run python scripts/feature_class_audit.py``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import data as data_mod  # noqa: E402
from model.config import LabelConfig, ModelingConfig  # noqa: E402
from model.features._multilabel import TopKMultiLabel  # noqa: E402
from model.features.admet import AdmetGroup, _percentile_columns  # noqa: E402
from model.features.disease import DiseaseGroup  # noqa: E402
from model.features.pathway import PathwayGroup  # noqa: E402

REACTOME_HIERARCHY_PATH = PROJECT_ROOT / "data" / "reactome" / "pathway_hierarchy.parquet"

logger = logging.getLogger("feature_class_audit")

CLASSES: tuple[str, ...] = ("molecular", "disease", "target", "pathway", "admet")
CLASS_COLOR = {
    "molecular": "#1f77b4", "disease": "#ff7f0e", "target": "#2ca02c",
    "pathway":   "#9467bd", "admet":   "#d62728",
}
OUT_COLOR = {0: "#d62728", 1: "#2ca02c"}
ADMET_COLS = _percentile_columns()


# ---------------------------------------------------------------------------
# Load + presence + scalars
# ---------------------------------------------------------------------------

def load_full_frame(args: argparse.Namespace) -> pd.DataFrame:
    cfg = ModelingConfig(
        candidate_detail_path=Path(args.candidates),
        fingerprints_path=Path(args.fingerprints),
        embeddings_path=Path(args.embeddings),
        label=LabelConfig(),
    )
    df = data_mod.build_modeling_frame(cfg)
    df["_year"] = pd.to_datetime(df.get("earliest_start_date"), errors="coerce").dt.year
    return df


def _len(s: pd.Series) -> pd.Series:
    return s.apply(lambda v: 0 if (v is None or (isinstance(v, float) and np.isnan(v))) else len(v))


def _mesh_prefixes_series(s: pd.Series) -> pd.Series:
    return s.apply(
        lambda toks: sorted({t.split(".")[0] for t in toks if t})
        if (toks is not None and not (isinstance(toks, float) and np.isnan(toks)))
        else []
    )


def _leaves_series(s: pd.Series, children_of: dict[str, list[str]]) -> pd.Series:
    def _leaves(pids):
        if pids is None or (isinstance(pids, float) and np.isnan(pids)):
            return []
        s_ = set(pids)
        return [p for p in pids if not (set(children_of.get(p, [])) & s_)]
    return s.apply(_leaves)


def compute_subrep_table(
    df: pd.DataFrame,
    hierarchy: tuple[dict[str, str], dict[str, list[str]]] | None,
) -> list[dict]:
    """Per-representation rows: class, repr, enc_dim, note, coverage stats."""
    y = df["y"]
    rows: list[dict] = []

    def add(cls: str, name: str, present: pd.Series, enc_dim: int, note: str = "") -> None:
        rows.append({
            "class": cls, "repr": name, "enc_dim": enc_dim, "note": note,
            "cov":  100 * present.mean(),
            "cov0": 100 * present[y == 0].mean(),
            "cov1": 100 * present[y == 1].mean(),
        })

    has = lambda c: df[c].notna() if c in df.columns else pd.Series(False, index=df.index)

    # ---- molecular
    has_emb, has_fp = has("embedding"), has("ecfp4")
    add("molecular", "embedding",   has_emb,            768 + 1, "MolFormer 768d + missing")
    add("molecular", "fingerprint", has_fp,             2048 + 167 + 1, "ECFP4 2048 + MACCS 167 + missing")
    add("molecular", "similarity",  has_fp | has_emb,   4, "tanimoto_nn + molformer_nn (each: sim + miss)")

    # ---- disease
    if "mesh_condition_tree_numbers" in df.columns:
        mesh_pref = _mesh_prefixes_series(df["mesh_condition_tree_numbers"])
        has_mesh = mesh_pref.apply(len) > 0
        enc = TopKMultiLabel(top_k=200, prefix="mesh"); enc.fit(mesh_pref)
        add("disease", "mesh",  has_mesh, len(enc.feature_names()),
            f"top-level prefix multi-hot; vocab={len(enc.vocab)}")
    else:
        add("disease", "mesh", pd.Series(False, index=df.index), 0, "column absent")
    if "icd10_codes" in df.columns:
        has_icd = df["icd10_codes"].apply(
            lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)) and len(v) > 0
        )
        enc = TopKMultiLabel(top_k=200, prefix="icd10"); enc.fit(df["icd10_codes"])
        full_vocab = df["icd10_codes"].explode().dropna().astype(str).nunique()
        add("disease", "icd10", has_icd, len(enc.feature_names()),
            f"top-200 multi-hot (not in production); full vocab={full_vocab:,}")
    else:
        add("disease", "icd10", pd.Series(False, index=df.index), 0, "column absent")

    # ---- target
    if "drug_targets" in df.columns:
        has_tgt = df["drug_targets"].apply(
            lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)) and len(v) > 0
        )
        enc = TopKMultiLabel(top_k=200, prefix="target"); enc.fit(df["drug_targets"])
        full_vocab = df["drug_targets"].explode().dropna().astype(str).nunique()
        add("target", "multi_hot", has_tgt, len(enc.feature_names()),
            f"top-200 UniProt; full vocab={full_vocab:,}")
    else:
        add("target", "multi_hot", pd.Series(False, index=df.index), 0, "column absent")

    # ---- pathway
    if "reactome_pathway_ids" in df.columns:
        has_pw = df["reactome_pathway_ids"].apply(
            lambda v: v is not None and not (isinstance(v, float) and np.isnan(v)) and len(v) > 0
        )
        enc_all = TopKMultiLabel(top_k=500, prefix="pathway"); enc_all.fit(df["reactome_pathway_ids"])
        dim_all = len(enc_all.feature_names()) + (1 if "reactome_n_pathways" in df.columns else 0)
        full_vocab_all = df["reactome_pathway_ids"].explode().dropna().astype(str).nunique()
        add("pathway", "multi_hot_all",  has_pw, dim_all,
            f"any depth (ancestor-expanded); top-500 + n_pathways; vocab={full_vocab_all:,}")
        if hierarchy:
            _, children_of = hierarchy
            leaf_series = _leaves_series(df["reactome_pathway_ids"], children_of)
            has_leaf = leaf_series.apply(len) > 0
            enc_leaf = TopKMultiLabel(top_k=500, prefix="pathway_leaf"); enc_leaf.fit(leaf_series)
            full_vocab_leaf = leaf_series.explode().dropna().astype(str).nunique()
            add("pathway", "multi_hot_leaf", has_leaf, len(enc_leaf.feature_names()),
                f"locally-leaf only; top-500; leaf vocab={full_vocab_leaf:,}")
        else:
            add("pathway", "multi_hot_leaf", pd.Series(False, index=df.index), 0,
                "hierarchy parquet missing")
    else:
        add("pathway", "multi_hot_all", pd.Series(False, index=df.index), 0, "column absent")
        add("pathway", "multi_hot_leaf", pd.Series(False, index=df.index), 0, "column absent")

    # ---- admet
    admet_grp = AdmetGroup()
    if admet_grp.is_available(df):
        admet_grp.fit(df)
        present_cols = [c for c in ADMET_COLS if c in df.columns]
        has_admet = df[present_cols].notna().any(axis=1) if present_cols else pd.Series(False, index=df.index)
        add("admet", "admet_ai", has_admet, len(admet_grp.feature_names()),
            f"admet_ai percentiles; {len(admet_grp._kept_cols)}/{len(present_cols)} cols kept, "
            f"{len(admet_grp._indicator_cols)} indicators")
    else:
        add("admet", "admet_ai", pd.Series(False, index=df.index), 0, "columns absent")

    return rows


def load_reactome_hierarchy() -> tuple[dict[str, str], dict[str, list[str]]] | None:
    """Return (id->name, id->children) maps; None if hierarchy parquet absent."""
    if not REACTOME_HIERARCHY_PATH.exists():
        return None
    h = pd.read_parquet(REACTOME_HIERARCHY_PATH).set_index("pathway_id")
    id_to_name = h["pathway_name"].to_dict()
    children_of = {
        pid: (list(c) if c is not None and not (isinstance(c, float) and np.isnan(c)) else [])
        for pid, c in h["child_ids"].items()
    }
    return id_to_name, children_of


def compute_presence(df: pd.DataFrame) -> pd.DataFrame:
    has = lambda c: df[c].notna() if c in df.columns else pd.Series(False, index=df.index)
    has_list = lambda c: (_len(df[c]) > 0) if c in df.columns else pd.Series(False, index=df.index)
    admet_cols = [c for c in ADMET_COLS if c in df.columns]
    return pd.DataFrame({
        "molecular": has("ecfp4") | has("embedding") | has("smiles_canonical") | has("smiles"),
        "disease":   has("disease_area") | has_list("mesh_condition_tree_numbers"),
        "target":    has_list("drug_targets"),
        "pathway":   has_list("reactome_pathway_ids"),
        "admet":     df[admet_cols].notna().any(axis=1) if admet_cols else pd.Series(False, index=df.index),
    }, index=df.index)


# ---------------------------------------------------------------------------
# Page 1 — Summary
# ---------------------------------------------------------------------------

def write_summary_page(pdf: PdfPages, df: pd.DataFrame, presence: pd.DataFrame, raw_n: int,
                       subreps: list[dict]) -> None:
    n = len(df); n_pos = int(df["y"].sum()); n_neg = n - n_pos
    year = df["_year"]

    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.5, height_ratios=[2.3, 1.1, 1.4])
    fig.suptitle("Feature Class Audit — Summary", fontsize=14, y=0.995)

    # 1. Per-representation table (class -> sub-rows)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    lines = [
        f"Source: outputs/candidate_detail.parquet           Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        f"Rows in source:   {raw_n:,}    Labeled:   {n:,}    "
        f"Approved: {n_pos:,} ({100*n_pos/max(n,1):.1f}%)    Failed: {n_neg:,} ({100*n_neg/max(n,1):.1f}%)",
        f"Year range: {year.min():.0f} - {year.max():.0f}",
        "",
        f"  {'class':<10}{'representation':<18}{'enc_dim':>9}{'cov':>8}{'fail':>8}{'appr':>8}  note",
        "  " + "-" * 100,
    ]
    last_cls: str | None = None
    for r in subreps:
        cls_lbl = r["class"] if r["class"] != last_cls else ""
        last_cls = r["class"]
        lines.append(
            f"  {cls_lbl:<10}{r['repr']:<18}{r['enc_dim']:>9,}"
            f"{r['cov']:>7.1f}%{r['cov0']:>7.1f}%{r['cov1']:>7.1f}%  {r['note']}"
        )
    lines += ["",
              "  Reps are alternative encodings of the same class (not summed). "
              "icd10 and pathway/multi_hot_leaf are not currently",
              "  enabled in the production model — listed here for comparison."]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5)

    # 2. Coverage by year (all 5 classes — by class, not sub-rep)
    ax = fig.add_subplot(gs[1])
    yrs = sorted(year.dropna().unique())
    n_per = year.value_counts().reindex(yrs).fillna(0)
    yrs = [y_ for y_ in yrs if n_per.loc[y_] >= 10]
    for c in CLASSES:
        pcts = [100 * presence[c][year == y_].mean() for y_ in yrs]
        ax.plot(yrs, pcts, marker="o", lw=1.5, color=CLASS_COLOR[c], label=c)
    ax.set_ylim(0, 105); ax.set_xlabel("year of earliest_start_date"); ax.set_ylabel("coverage (%)")
    ax.set_title("Coverage by year (years with >=10 rows)"); ax.legend(ncols=5, fontsize=8, loc="lower right")

    # 2. Coverage by year (all 5 classes)
    ax = fig.add_subplot(gs[1])
    yrs = sorted(year.dropna().unique())
    n_per = year.value_counts().reindex(yrs).fillna(0)
    yrs = [y_ for y_ in yrs if n_per.loc[y_] >= 10]
    for c in CLASSES:
        pcts = [100 * presence[c][year == y_].mean() for y_ in yrs]
        ax.plot(yrs, pcts, marker="o", lw=1.5, color=CLASS_COLOR[c], label=c)
    ax.set_ylim(0, 105); ax.set_xlabel("year of earliest_start_date"); ax.set_ylabel("coverage (%)")
    ax.set_title("Coverage by year (years with >=10 rows)"); ax.legend(ncols=5, fontsize=8, loc="lower right")

    # 3. Missingness heatmap (500 rows sample)
    ax = fig.add_subplot(gs[2])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(presence), size=min(500, len(presence)), replace=False)
    sub = presence.iloc[idx].copy()
    sub["_miss"] = (~sub[list(CLASSES)]).sum(axis=1)
    sub = sub.sort_values("_miss")
    M = (~sub[list(CLASSES)]).astype(int).values
    im = ax.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_yticks([]); ax.set_ylabel("500 candidates (sorted by # missing classes)")
    ax.set_title("Missingness heatmap (black = missing)")
    fig.colorbar(im, ax=ax, shrink=0.6, ticks=[0, 1], label="missing")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Pages 2-6 — One page per class (4-panel)
# ---------------------------------------------------------------------------

def _coverage_test(present: pd.Series, y: pd.Series) -> str:
    tab = pd.crosstab(present.astype(int), y)
    if tab.shape == (2, 2):
        chi2, p, _, _ = stats.chi2_contingency(tab)
        return f"; chi^2={chi2:.1f}, p={p:.1e}"
    return ""


def _class_richness(df: pd.DataFrame, cls: str) -> tuple[pd.Series, str]:
    """Return (per-row richness scalar, x-axis label) for the class."""
    if cls == "target":
        return _len(df["drug_targets"]) if "drug_targets" in df.columns else pd.Series(0, index=df.index), "drug_targets per row"
    if cls == "pathway":
        return df["reactome_n_pathways"].fillna(0).astype(float) if "reactome_n_pathways" in df.columns else pd.Series(0, index=df.index), "reactome_n_pathways"
    if cls == "disease":
        return _len(df["mesh_condition_tree_numbers"]) if "mesh_condition_tree_numbers" in df.columns else pd.Series(0, index=df.index), "MeSH codes per row"
    if cls == "molecular":
        if "ecfp4" in df.columns:
            return df["ecfp4"].apply(
                lambda v: float(np.mean(v)) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan
            ), "ECFP4 bit density"
        return pd.Series(np.nan, index=df.index), "ECFP4 bit density"
    if cls == "admet":
        cols = [c for c in ADMET_COLS if c in df.columns]
        return (df[cols].notna().sum(axis=1) if cols else pd.Series(0, index=df.index)), "# ADMET columns populated"
    raise ValueError(cls)


def _class_header_text(df: pd.DataFrame, cls: str, presence: pd.Series, y: pd.Series) -> str:
    cov = 100 * presence.mean()
    cov0 = 100 * presence[y == 0].mean(); cov1 = 100 * presence[y == 1].mean()
    test = _coverage_test(presence, y)
    if cls == "target":
        col, sub = "drug_targets", df["drug_targets"]
        vocab = sub.explode().dropna().astype(str).nunique() if "drug_targets" in df.columns else 0
        return (f"List column: `{col}`   vocab={vocab:,} UniProt IDs\n"
                f"Coverage: {cov:.1f}%   (failed={cov0:.1f}%, approved={cov1:.1f}%{test})")
    if cls == "pathway":
        col, sub = "reactome_pathway_ids", df["reactome_pathway_ids"]
        vocab = sub.explode().dropna().astype(str).nunique() if col in df.columns else 0
        n_path = df["reactome_n_pathways"].dropna()
        med = n_path.median() if len(n_path) else 0
        return (f"List column: `{col}`   vocab={vocab:,} Reactome IDs   median n_pathways={med:.0f}\n"
                f"Coverage: {cov:.1f}%   (failed={cov0:.1f}%, approved={cov1:.1f}%{test})")
    if cls == "disease":
        n_area = df["disease_area"].nunique() if "disease_area" in df.columns else 0
        n_pref = (df["mesh_condition_tree_numbers"].explode().dropna()
                  .apply(lambda s: s.split(".")[0]).nunique()
                  if "mesh_condition_tree_numbers" in df.columns else 0)
        return (f"Columns: `disease_area` ({n_area} vals)  +  `mesh_condition_tree_numbers` ({n_pref} top-level prefixes)\n"
                f"Coverage: {cov:.1f}%   (failed={cov0:.1f}%, approved={cov1:.1f}%{test})")
    if cls == "molecular":
        nfp = int(df["ecfp4"].notna().sum()) if "ecfp4" in df.columns else 0
        nem = int(df["embedding"].notna().sum()) if "embedding" in df.columns else 0
        return (f"Columns: smiles_canonical, ecfp4 (2048-bit), maccs (167-bit), embedding (768d MolFormer)\n"
                f"SMILES coverage: {cov:.1f}%   (failed={cov0:.1f}%, approved={cov1:.1f}%{test})\n"
                f"With fingerprint: {nfp:,} rows ({100*nfp/len(df):.1f}%)   "
                f"With embedding: {nem:,} rows ({100*nem/len(df):.1f}%)")
    if cls == "admet":
        cols = [c for c in ADMET_COLS if c in df.columns]
        med_null = pd.Series({c: df[c].isna().mean() for c in cols}).median() * 100 if cols else 0
        return (f"{len(cols)}/{len(ADMET_COLS)} percentile columns present   median per-col null rate = {med_null:.1f}%\n"
                f"Any-ADMET coverage: {cov:.1f}%   (failed={cov0:.1f}%, approved={cov1:.1f}%{test})")
    return ""


def write_class_page(pdf: PdfPages, cls: str, df: pd.DataFrame, presence: pd.DataFrame) -> None:
    y = df["y"]; year = df["_year"]; pres = presence[cls]
    richness, xlabel = _class_richness(df, cls)
    rich_present = richness[pres] if cls != "admet" else richness  # admet richness is always defined
    if cls == "molecular":
        rich_present = richness.dropna()

    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.3, height_ratios=[0.6, 1.1, 1.1])
    fig.suptitle(f"{cls.upper()}", fontsize=14, y=0.995)

    # Header text spanning both columns
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(0.0, 1.0, _class_header_text(df, cls, pres, y),
            family="monospace", fontsize=10, va="top", ha="left")

    # Distribution (overall)
    ax = fig.add_subplot(gs[1, 0])
    if cls == "admet":
        cols = [c for c in ADMET_COLS if c in df.columns]
        null_rates = pd.Series({c: df[c].isna().mean() for c in cols}).sort_values()
        names = [c.replace("admet_", "").replace("_drugbank_approved_percentile", "") for c in null_rates.index]
        colors_ = ["#d62728" if r > 0.95 else CLASS_COLOR["admet"] for r in null_rates.values]
        ax.barh(range(len(null_rates)), null_rates.values * 100, color=colors_)
        ax.set_yticks(range(len(null_rates))); ax.set_yticklabels(names, fontsize=4)
        ax.axvline(95, color="grey", ls=":", lw=0.5)
        ax.set_xlim(0, 100); ax.set_xlabel("null rate (%)"); ax.invert_yaxis()
        ax.set_title("Per-column null rate (red = dropped at fit)")
    else:
        vals = rich_present
        if vals.size:
            upper = max(1, np.quantile(vals, 0.99))
            if cls == "molecular":
                bins = np.linspace(vals.min(), vals.max(), 40)
            else:
                upper = int(upper) or 1
                bins = np.arange(1, upper + 2) if vals.dtype.kind in "iu" else np.linspace(0, upper, 30)
                vals = vals.clip(upper=upper)
            ax.hist(vals, bins=bins, color=CLASS_COLOR[cls], edgecolor="white")
            ax.set_xlabel(xlabel + ("" if cls == "molecular" else f"  (clip p99={int(upper)})"))
        ax.set_ylabel("rows"); ax.set_title(f"Distribution of {xlabel}")

    # Outcome-stratified
    ax = fig.add_subplot(gs[1, 1])
    if cls == "admet":
        cols = [c for c in ADMET_COLS if c in df.columns]
        scored = []
        for c in cols:
            s = df[c].dropna()
            if len(s) < 50: continue
            y_s = y.loc[s.index]
            if (y_s == 0).sum() < 20 or (y_s == 1).sum() < 20: continue
            try:
                _, p = stats.mannwhitneyu(s[y_s == 0], s[y_s == 1], alternative="two-sided")
            except ValueError:
                p = 1.0
            scored.append((c, p, s[y_s == 1].mean() - s[y_s == 0].mean()))
        scored.sort(key=lambda r: r[1])
        names = [r[0].replace("admet_", "").replace("_drugbank_approved_percentile", "") for r in scored[:15]]
        diffs = [r[2] for r in scored[:15]]
        pvals = [r[1] for r in scored[:15]]
        colors_ = [OUT_COLOR[1] if d > 0 else OUT_COLOR[0] for d in diffs]
        ax.barh(range(len(diffs))[::-1], diffs, color=colors_)
        ax.set_yticks(range(len(diffs))[::-1])
        ax.set_yticklabels([f"{n}  p={p:.0e}" for n, p in zip(names, pvals)], fontsize=6)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("approved - failed mean percentile")
        ax.set_title("Top-15 ADMET cols by Mann-Whitney p")
    else:
        v0 = rich_present[y.loc[rich_present.index] == 0]
        v1 = rich_present[y.loc[rich_present.index] == 1]
        if v0.size and v1.size:
            if cls == "molecular":
                bins = np.linspace(min(v0.min(), v1.min()), max(v0.max(), v1.max()), 40)
            else:
                upper = max(1, int(np.quantile(rich_present, 0.99))) or 1
                bins = np.arange(1, upper + 2)
                v0 = v0.clip(upper=upper); v1 = v1.clip(upper=upper)
            ax.hist([v0, v1], bins=bins, density=True,
                    color=[OUT_COLOR[0], OUT_COLOR[1]], alpha=0.7, label=["Failed", "Approved"])
            ax.legend(fontsize=8)
            try:
                _, p = stats.mannwhitneyu(v0, v1, alternative="two-sided")
                ax.set_title(f"By outcome  (Mann-Whitney p={p:.1e})")
            except ValueError:
                ax.set_title("By outcome")
        ax.set_xlabel(xlabel); ax.set_ylabel("density")

    # Coverage by year (spans both columns)
    ax = fig.add_subplot(gs[2, :])
    yrs = sorted(year.dropna().unique())
    n_per = year.value_counts().reindex(yrs).fillna(0)
    yrs = [y_ for y_ in yrs if n_per.loc[y_] >= 10]
    pcts = [100 * pres[year == y_].mean() for y_ in yrs]
    ax.plot(yrs, pcts, marker="o", color=CLASS_COLOR[cls], lw=1.5)
    ax2 = ax.twinx()
    ax2.bar(yrs, [n_per.loc[y_] for y_ in yrs], color="grey", alpha=0.2, width=0.8)
    ax2.set_ylabel("rows per year", color="grey")
    ax.set_ylim(0, 105); ax.set_ylabel("coverage (%)"); ax.set_xlabel("year")
    ax.set_title(f"{cls} coverage by year")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Top-values page — most common values for one-hot / multi-hot features
# Pathway shows both ancestor (any depth) and locally-leaf views.
# ---------------------------------------------------------------------------

def _explode_top(series: pd.Series, n: int) -> pd.Series:
    """Top-n most common tokens from a list-typed column."""
    return series.explode().dropna().astype(str).value_counts().head(n)


def _locally_leaf_top(df: pd.DataFrame, children_of: dict[str, list[str]], n: int) -> pd.Series:
    """For each candidate, keep only pathways whose children aren't in its set; then top-n."""
    def _leaves(pids):
        if pids is None or (isinstance(pids, float) and np.isnan(pids)):
            return []
        s = set(pids)
        return [p for p in pids if not (set(children_of.get(p, [])) & s)]
    leaves = df["reactome_pathway_ids"].apply(_leaves)
    return leaves.explode().dropna().astype(str).value_counts().head(n)


def _hbar(ax: plt.Axes, counts: pd.Series, color: str, title: str,
          label_fn=None) -> None:
    if not len(counts):
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title, fontsize=9); ax.axis("off"); return
    labels = [label_fn(k) if label_fn else str(k) for k in counts.index]
    ax.barh(labels[::-1], counts.values[::-1], color=color)
    ax.set_xlabel("# candidates")
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)


def write_top_values_page(pdf: PdfPages, df: pd.DataFrame,
                          hierarchy: tuple[dict[str, str], dict[str, list[str]]] | None,
                          n: int = 15) -> None:
    id_to_name, children_of = (hierarchy if hierarchy else ({}, {}))

    def _path_label(pid: str) -> str:
        name = id_to_name.get(pid, "")
        return f"{pid}  {name[:40]}" if name else pid

    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.65, wspace=0.55)
    fig.suptitle(f"Top-{n} values for one-hot / multi-hot feature columns",
                 fontsize=13, y=0.995)

    # disease_area (low-card categorical)
    ax = fig.add_subplot(gs[0, 0])
    if "disease_area" in df.columns:
        _hbar(ax, df["disease_area"].value_counts().head(n), CLASS_COLOR["disease"],
              "disease_area")
    else:
        ax.set_title("disease_area missing"); ax.axis("off")

    # MeSH top-level prefixes (disease multi-hot)
    ax = fig.add_subplot(gs[0, 1])
    if "mesh_condition_tree_numbers" in df.columns:
        prefixes = df["mesh_condition_tree_numbers"].apply(
            lambda toks: sorted({t.split(".")[0] for t in toks if t}) if toks is not None
                         and not (isinstance(toks, float) and np.isnan(toks)) else []
        )
        _hbar(ax, _explode_top(prefixes, n), CLASS_COLOR["disease"],
              "MeSH top-level prefix")
    else:
        ax.set_title("mesh_condition_tree_numbers missing"); ax.axis("off")

    # drug_targets (UniProt multi-hot)
    ax = fig.add_subplot(gs[1, 0])
    if "drug_targets" in df.columns:
        _hbar(ax, _explode_top(df["drug_targets"], n), CLASS_COLOR["target"],
              "drug_targets (UniProt)")
    else:
        ax.set_title("drug_targets missing"); ax.axis("off")

    # Reactome pathway — any depth (ancestor-expanded)
    ax = fig.add_subplot(gs[1, 1])
    if "reactome_pathway_ids" in df.columns:
        _hbar(ax, _explode_top(df["reactome_pathway_ids"], n), CLASS_COLOR["pathway"],
              "Reactome pathway — any depth (ancestor)", label_fn=_path_label)
    else:
        ax.set_title("reactome_pathway_ids missing"); ax.axis("off")

    # Reactome pathway — locally leaf only
    ax = fig.add_subplot(gs[2, 0])
    if "reactome_pathway_ids" in df.columns and children_of:
        _hbar(ax, _locally_leaf_top(df, children_of, n), CLASS_COLOR["pathway"],
              "Reactome pathway — locally leaf only", label_fn=_path_label)
    elif "reactome_pathway_ids" in df.columns:
        ax.text(0.5, 0.5, "hierarchy parquet not available\n(data/reactome/pathway_hierarchy.parquet)",
                ha="center", va="center", fontsize=8)
        ax.set_title("Reactome pathway — locally leaf"); ax.axis("off")
    else:
        ax.set_title("reactome_pathway_ids missing"); ax.axis("off")

    # Legend / explanatory text (bottom-right)
    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ax.text(0.0, 0.95,
            "Ancestor vs leaf (Reactome):\n\n"
            "  ANCESTOR (any depth) explodes\n"
            "  the ancestry-expanded list, so a\n"
            "  candidate annotated only with\n"
            "  'Signaling by GPCR' (R-HSA-372790)\n"
            "  also contributes to its parents\n"
            "  ('Signal transduction', etc.).\n"
            "  Top entries here are broad\n"
            "  umbrellas.\n\n"
            "  LOCALLY LEAF keeps only the\n"
            "  most specific pathway in each\n"
            "  candidate's set (no child of\n"
            "  the pathway is also annotated).\n"
            "  Top entries here are the actual\n"
            "  biological processes captured.",
            family="monospace", fontsize=8, va="top", ha="left")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Frequency long-tail (3 panels)
# ---------------------------------------------------------------------------

def write_long_tail_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    panels = [
        ("target",  "drug_targets",                CLASS_COLOR["target"]),
        ("pathway", "reactome_pathway_ids",        CLASS_COLOR["pathway"]),
        ("disease", "mesh_condition_tree_numbers", CLASS_COLOR["disease"]),
    ]
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.5)
    fig.suptitle("Frequency long-tail (rank vs # candidates)", fontsize=13, y=0.99)
    for i, (label, col, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if col not in df.columns:
            ax.set_title(f"{label}: {col} missing"); continue
        counts = df[col].explode().dropna().astype(str).value_counts()
        if not len(counts):
            ax.text(0.5, 0.5, "no data", ha="center", va="center"); continue
        ax.plot(np.arange(1, len(counts) + 1), counts.values, color=color)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("rank (log)"); ax.set_ylabel("# candidates (log)")
        for j, (name, c) in enumerate(counts.head(5).items()):
            ax.annotate(f"{j+1}. {name} ({c})", xy=(j + 1, c),
                        xytext=(5, 0), textcoords="offset points", fontsize=7, va="center")
        ax.set_title(f"{label}  vocab={len(counts):,}, "
                     f"top-50 captures {counts.head(50).sum()/counts.sum()*100:.0f}% of mentions",
                     fontsize=10)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Page 8 — PCA (3 panels)
# ---------------------------------------------------------------------------

def _pca_scatter(ax: plt.Axes, X: np.ndarray, y: np.ndarray, title: str, seed: int) -> None:
    if X.shape[0] < 5 or X.shape[1] < 2:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center"); ax.set_title(title); return
    Z = PCA(n_components=2, random_state=seed).fit_transform(X)
    for cv, color, lbl in [(0, OUT_COLOR[0], "Failed"), (1, OUT_COLOR[1], "Approved")]:
        m = y == cv
        ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.35, c=color, label=lbl, linewidths=0)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, markerscale=2)


def write_pca_page(pdf: PdfPages, df: pd.DataFrame, seed: int, sample: int) -> None:
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.5)
    fig.suptitle("PCA — molecule (true embedding) + disease/pathway (multi-hot proxy)",
                 fontsize=13, y=0.99)

    ax = fig.add_subplot(gs[0])
    if "embedding" in df.columns and df["embedding"].notna().any():
        sub = df[df["embedding"].notna()]
        idx = rng.choice(len(sub), size=min(sample, len(sub)), replace=False)
        sub = sub.iloc[idx]
        X = np.vstack(sub["embedding"].values).astype(np.float32)
        _pca_scatter(ax, X, sub["y"].values, "Molecule — MolFormer 768d", seed)
    else:
        ax.text(0.5, 0.5, "no molecule embeddings present", ha="center", va="center")

    for k, (grp, title) in enumerate([
        (DiseaseGroup(), "Disease — multi-hot proxy (disease_area + MeSH prefix)"),
        (PathwayGroup(), "Pathway — multi-hot proxy (top-500 Reactome IDs + n_pathways)"),
    ]):
        ax = fig.add_subplot(gs[1 + k])
        try:
            if grp.is_available(df):
                grp.fit(df)
                X = grp.transform(df).astype(np.float32)
                idx = rng.choice(len(df), size=min(sample, len(df)), replace=False)
                _pca_scatter(ax, X[idx], df["y"].values[idx], title, seed)
            else:
                ax.text(0.5, 0.5, "columns absent", ha="center", va="center")
        except Exception as e:
            logger.exception("%s PCA failed", title)
            ax.text(0.5, 0.5, f"failed: {e}", ha="center", va="center")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = ModelingConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "feature_class_audit.pdf")
    p.add_argument("--candidates", type=Path, default=d.candidate_detail_path)
    p.add_argument("--fingerprints", type=Path, default=d.fingerprints_path)
    p.add_argument("--embeddings", type=Path, default=d.embeddings_path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pca-sample", type=int, default=5000)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    raw_n = len(pd.read_parquet(args.candidates, columns=["candidate_id"]))
    df = load_full_frame(args)
    logger.info("loaded %d labeled rows", len(df))
    presence = compute_presence(df)
    hierarchy = load_reactome_hierarchy()
    if hierarchy is None:
        logger.warning("reactome hierarchy parquet missing — pathway leaf rows will be empty")
    subreps = compute_subrep_table(df, hierarchy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        write_summary_page(pdf, df, presence, raw_n, subreps)
        for cls in CLASSES:
            write_class_page(pdf, cls, df, presence)
        write_top_values_page(pdf, df, hierarchy)
        write_long_tail_page(pdf, df)
        write_pca_page(pdf, df, seed=args.seed, sample=args.pca_sample)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
