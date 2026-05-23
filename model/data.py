"""Load + join + label the modeling frame.

Joins `candidate_detail.parquet` (left) with `fingerprints.parquet` and
`molformer_embeddings.parquet` on `candidate_id`. Rows missing fingerprints
or embeddings are kept (NaN); per-group encoders handle the missing case
explicitly via a `_missing` indicator column. Applies the binary label
policy from `LabelConfig`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LabelConfig, ModelingConfig

logger = logging.getLogger(__name__)


def load_candidate_detail(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "candidate_id" not in df.columns:
        raise ValueError(f"{path} has no candidate_id column")
    return df


def load_trial_detail(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("candidate_id", "nct_id", "trial_inferred_label"):
        if col not in df.columns:
            raise ValueError(f"{path} has no {col!r} column")
    return df


def load_fingerprints(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("fingerprints parquet not found at %s — fingerprint coverage will be 0", path)
        return pd.DataFrame(columns=["candidate_id", "ecfp4", "maccs"])
    df = pd.read_parquet(path)
    return df[["candidate_id", "ecfp4", "maccs"]]


def load_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("embeddings parquet not found at %s — embedding coverage will be 0", path)
        return pd.DataFrame(columns=["candidate_id", "embedding"])
    df = pd.read_parquet(path)
    return df[["candidate_id", "embedding"]]


def apply_label(df: pd.DataFrame, label: LabelConfig) -> pd.DataFrame:
    """Filter rows by `outcome` and attach a binary `y` column.

    Drops rows whose outcome is in `exclude_outcomes`, then maps the
    remaining `outcome` values to {0, 1} via `positive` / `negative`. Any
    outcome not appearing in either list is dropped with a warning.
    """
    if "outcome" not in df.columns:
        raise ValueError("candidate_detail has no `outcome` column")

    n_before = len(df)
    df = df[~df["outcome"].isin(label.exclude_outcomes)].copy()
    pos = set(label.positive)
    neg = set(label.negative)
    df = df[df["outcome"].isin(pos | neg)].copy()
    df["y"] = df["outcome"].isin(pos).astype(np.int8)
    n_pos = int(df["y"].sum())
    n_neg = len(df) - n_pos
    logger.info(
        "label: filtered %d -> %d rows; positive=%d (%.1f%%) negative=%d",
        n_before,
        len(df),
        n_pos,
        100.0 * n_pos / max(len(df), 1),
        n_neg,
    )
    return df


def build_modeling_frame(config: ModelingConfig) -> pd.DataFrame:
    """Dispatch to the candidate- or trial-level builder.

    Routed by `config.training_granularity`:
      - "drug_indication" (default) → :func:`build_candidate_modeling_frame`
      - "trial" → :func:`build_trial_modeling_frame`
    """
    granularity = (config.training_granularity or "drug_indication").lower()
    if granularity == "trial":
        return build_trial_modeling_frame(config)
    if granularity == "drug_indication":
        return build_candidate_modeling_frame(config)
    raise ValueError(
        f"unknown training_granularity={config.training_granularity!r}; "
        "expected 'drug_indication' or 'trial'"
    )


def build_candidate_modeling_frame(config: ModelingConfig) -> pd.DataFrame:
    """Join candidate_detail with fingerprints & embeddings; apply label.

    Returns a DataFrame containing `candidate_id`, `y`, all candidate_detail
    columns, plus `ecfp4`, `maccs`, `embedding` (any of which may be NaN
    for candidates without a SMILES string).
    """
    cand = load_candidate_detail(config.candidate_detail_path)
    fps = load_fingerprints(config.fingerprints_path)
    embs = load_embeddings(config.embeddings_path)

    df = cand.merge(fps, on="candidate_id", how="left")
    df = df.merge(embs, on="candidate_id", how="left")

    n_with_fp = df["ecfp4"].notna().sum() if "ecfp4" in df.columns else 0
    n_with_emb = df["embedding"].notna().sum() if "embedding" in df.columns else 0
    logger.info(
        "joined: %d candidates; fingerprints=%d (%.1f%%) embeddings=%d (%.1f%%)",
        len(df),
        n_with_fp,
        100.0 * n_with_fp / max(len(df), 1),
        n_with_emb,
        100.0 * n_with_emb / max(len(df), 1),
    )

    df = apply_label(df, config.label)
    return df.reset_index(drop=True)


def build_trial_modeling_frame(config: ModelingConfig) -> pd.DataFrame:
    """One row per NCT × primary-candidate; y = `trial_inferred_label`.

    Drops trials with a null inferred label (ongoing / phase-not-reached).
    Then joins the candidate feature columns from `candidate_detail`
    (everything except `outcome` and the trial-frame's own candidate_* /
    trial_* fields) plus fingerprints + embeddings on `candidate_id`,
    so existing feature groups operate without modification.
    """
    trials = load_trial_detail(config.trial_detail_path)
    n_before = len(trials)
    trials = trials[trials["trial_inferred_label"].notna()].copy()
    trials["y"] = trials["trial_inferred_label"].astype(np.int8)
    n_pos = int(trials["y"].sum())
    n_neg = len(trials) - n_pos
    logger.info(
        "trial label: filtered %d -> %d trials; positive=%d (%.1f%%) negative=%d",
        n_before,
        len(trials),
        n_pos,
        100.0 * n_pos / max(len(trials), 1),
        n_neg,
    )

    cand = load_candidate_detail(config.candidate_detail_path)
    # Resolve column overlap: the trial frame already carries
    # `candidate_drug`, `candidate_indication`, etc. — prefer those, drop
    # the matching candidate-side columns so `merge` doesn't suffix.
    # `outcome` (candidate-level) is intentionally retained — the
    # `nn_similarity` feature group uses it to filter the approved-NN pool,
    # and it differs from the trial-level `y` (an Approved drug's phase-2
    # failure trial is y=0 even though outcome=Approved).
    overlap = (set(trials.columns) & set(cand.columns)) - {"candidate_id"}
    cand = cand.drop(columns=list(overlap), errors="ignore")

    df = trials.merge(cand, on="candidate_id", how="left")
    df = df.merge(load_fingerprints(config.fingerprints_path), on="candidate_id", how="left")
    df = df.merge(load_embeddings(config.embeddings_path), on="candidate_id", how="left")

    n_with_fp = df["ecfp4"].notna().sum() if "ecfp4" in df.columns else 0
    n_with_emb = df["embedding"].notna().sum() if "embedding" in df.columns else 0
    logger.info(
        "joined trial frame: %d trials across %d candidates; fingerprints=%d (%.1f%%) embeddings=%d (%.1f%%)",
        len(df),
        df["candidate_id"].nunique(),
        n_with_fp,
        100.0 * n_with_fp / max(len(df), 1),
        n_with_emb,
        100.0 * n_with_emb / max(len(df), 1),
    )
    return df.reset_index(drop=True)
