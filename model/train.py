"""Single training run — assemble feature matrix, fit model, evaluate.

`train_one_run(config)` is reused by `ablate.py` per subset. Fixing the
seed makes the train/test split identical across subsets so ablation
deltas are directly comparable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import data as data_mod
from . import evaluate, splits
from .config import ModelingConfig
from .features import FEATURE_GROUPS, FeatureGroup, build_group
from .models import build_model

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    config: ModelingConfig
    groups: tuple[str, ...]
    feature_names: list[str]
    n_features: int
    n_train: int
    n_test: int
    train_pos: int
    test_pos: int
    metrics: dict
    feature_importances: Optional[np.ndarray] = None
    test_predictions: Optional[pd.DataFrame] = None  # candidate_id, y_true, y_proba [, y_proba_calibrated]
    fitted_groups: list[FeatureGroup] = field(default_factory=list)
    fitted_model: object = None  # ModelProtocol
    calibrator: object = None  # IsotonicRegression / LogisticRegression / None
    calibration_metrics: dict = field(default_factory=dict)
    n_calib: int = 0
    calib_pos: int = 0
    # Per-phase test-set metrics for trial-granularity runs. Empty dict
    # for drug-indication runs (no `trial_phase` column).
    per_phase_metrics: dict = field(default_factory=dict)
    per_phase_metrics_calibrated: dict = field(default_factory=dict)


def _instantiate_groups(config: ModelingConfig) -> list[FeatureGroup]:
    """Build group instances honouring config.feature_config knobs."""
    fc = config.features
    out: list[FeatureGroup] = []
    for name in fc.enabled:
        if name == "targets":
            out.append(build_group(name, top_k=fc.top_k_targets))
        elif name == "pathway":
            out.append(build_group(name, top_k=fc.top_k_pathways))
        elif name == "disease":
            out.append(build_group(name, top_k_mesh=fc.top_k_mesh))
        elif name == "moa":
            out.append(build_group(name, top_k=fc.top_k_moa))
        elif name == "admet":
            out.append(
                build_group(
                    name,
                    drop_null_threshold=fc.admet_drop_null_threshold,
                    indicator_threshold=fc.admet_indicator_threshold,
                )
            )
        else:
            out.append(build_group(name))
    return out


def _check_groups_available(groups: list[FeatureGroup], df: pd.DataFrame) -> None:
    for g in groups:
        if not g.is_available(df):
            logger.warning(
                "feature group %r reports unavailable on the joined frame — output will be empty",
                g.name,
            )


def _stack(matrices: list[np.ndarray]) -> np.ndarray:
    nonempty = [m for m in matrices if m is not None and m.shape[1] > 0]
    if not nonempty:
        raise ValueError("no enabled feature group produced any columns")
    return np.hstack(nonempty)


def train_one_run(
    config: ModelingConfig,
    *,
    df: Optional[pd.DataFrame] = None,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    calib_idx: Optional[np.ndarray] = None,
) -> RunResult:
    """Run one full train/eval cycle.

    `df`, `train_idx`, `test_idx`, `calib_idx` are optional escape hatches
    that the ablation / RFE harnesses use to fix the split across subsets.
    When omitted, data is loaded fresh and split per `config`.

    Calibration: when `config.calibration_year` is set (and `calib_idx` is
    not explicitly passed), a three-way temporal split is taken so the
    calibrator can be fit on a held-out year disjoint from train/val/test.
    """
    if not config.features.enabled:
        raise ValueError("config.features.enabled is empty — nothing to train on")
    for name in config.features.enabled:
        if name not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group {name!r}")

    if df is None:
        df = data_mod.build_modeling_frame(config)
    if train_idx is None or test_idx is None:
        if config.calibration_year is not None:
            train_idx, calib_idx, test_idx = splits.split_with_calibration(
                df,
                calibration_year=config.calibration_year,
                time_split_column=config.time_split_column,
            )
        else:
            train_idx, test_idx = splits.split(
                df,
                test_size=config.test_size,
                seed=config.seed,
                group_by=config.group_by,
                time_split_column=config.time_split_column,
                time_split_year=config.time_split_year,
            )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train = train_df["y"].values.astype(int)
    y_test = test_df["y"].values.astype(int)
    calib_df = (
        df.iloc[calib_idx].reset_index(drop=True)
        if calib_idx is not None and len(calib_idx) > 0
        else None
    )
    y_calib = calib_df["y"].values.astype(int) if calib_df is not None else None

    # Inner-val slice (off training set) for early stopping.
    from sklearn.model_selection import train_test_split

    inner_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=config.inner_val_size,
        stratify=y_train,
        random_state=config.seed,
    )
    inner_df = train_df.iloc[inner_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    y_inner = y_train[inner_idx]
    y_val = y_train[val_idx]

    # Fit groups on inner-train (so val + test stay held out).
    groups = _instantiate_groups(config)
    _check_groups_available(groups, inner_df)
    feat_train: list[np.ndarray] = []
    feat_val: list[np.ndarray] = []
    feat_test: list[np.ndarray] = []
    feat_calib: list[np.ndarray] = []
    feature_names: list[str] = []
    group_widths: list[tuple[str, int]] = []
    for g in groups:
        g.fit(inner_df)
        m_train = g.transform(inner_df)
        m_val = g.transform(val_df)
        m_test = g.transform(test_df)
        feat_train.append(m_train)
        feat_val.append(m_val)
        feat_test.append(m_test)
        if calib_df is not None:
            feat_calib.append(g.transform(calib_df))
        names = g.feature_names()
        feature_names.extend(names)
        group_widths.append((g.name, m_train.shape[1]))
        logger.info("group %s: n_features=%d", g.name, m_train.shape[1])

    X_train = _stack(feat_train)
    X_val = _stack(feat_val)
    X_test = _stack(feat_test)
    X_calib = _stack(feat_calib) if calib_df is not None else None
    n_features = X_train.shape[1]
    logger.info(
        "assembled feature matrix: train=%s val=%s test=%s%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
        f" calib={X_calib.shape}" if X_calib is not None else "",
    )

    # Build + fit model.
    n_pos = int(y_inner.sum())
    n_neg = len(y_inner) - n_pos
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = build_model(
        config.model_name,
        scale_pos_weight=spw,
        random_state=config.seed,
        **config.model_kwargs,
    )
    model.fit(X_train, y_inner, X_val=X_val, y_val=y_val)

    y_proba = model.predict_proba(X_test)[:, 1]
    m = evaluate.metrics(y_test, y_proba)

    # Per-phase test metrics for trial-granularity runs. Dispatched on
    # column presence so this stays a no-op for drug-indication runs.
    per_phase_metrics: dict = {}
    if "trial_phase" in test_df.columns:
        per_phase_metrics = evaluate.metrics_by_phase(
            y_test, y_proba, test_df["trial_phase"].values
        )
        for label, mp in per_phase_metrics.items():
            logger.info(
                "phase %s: n=%d pos=%d roc_auc=%.4f pr_auc=%.4f",
                label,
                mp.get("n", 0),
                mp.get("n_pos", 0),
                mp.get("roc_auc", float("nan")),
                mp.get("pr_auc", float("nan")),
            )

    # Optional probability calibration on the held-out year slice.
    calibrator = None
    y_proba_calibrated = None
    calibration_metrics: dict = {}
    per_phase_metrics_calibrated: dict = {}
    if X_calib is not None and y_calib is not None and len(y_calib) > 0:
        raw_calib = model.predict_proba(X_calib)[:, 1]
        method = (config.calibration_method or "isotonic").lower()
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_calib, y_calib)
            y_proba_calibrated = calibrator.predict(y_proba)
        elif method == "sigmoid":
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression()
            calibrator.fit(raw_calib.reshape(-1, 1), y_calib)
            y_proba_calibrated = calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        else:
            raise ValueError(f"unknown calibration_method: {config.calibration_method!r}")
        y_proba_calibrated = np.clip(np.asarray(y_proba_calibrated), 0.0, 1.0)
        m_cal = evaluate.metrics(y_test, y_proba_calibrated)
        if "trial_phase" in test_df.columns:
            per_phase_metrics_calibrated = evaluate.metrics_by_phase(
                y_test, y_proba_calibrated, test_df["trial_phase"].values
            )
        else:
            per_phase_metrics_calibrated = {}
        ece_pre = evaluate.expected_calibration_error(y_test, y_proba)
        ece_post = evaluate.expected_calibration_error(y_test, y_proba_calibrated)
        calibration_metrics = {
            "method": method,
            "calibration_year": config.calibration_year,
            "n_calib": int(len(y_calib)),
            "calib_pos": int(y_calib.sum()),
            "pre": {
                "brier": m["brier"],
                "log_loss": m["log_loss"],
                "ece": ece_pre,
            },
            "post": {
                "roc_auc": m_cal["roc_auc"],
                "pr_auc": m_cal["pr_auc"],
                "f1": m_cal["f1"],
                "brier": m_cal["brier"],
                "log_loss": m_cal["log_loss"],
                "ece": ece_post,
                "balanced_accuracy": m_cal["balanced_accuracy"],
                "tp": m_cal["tp"], "fp": m_cal["fp"], "tn": m_cal["tn"], "fn": m_cal["fn"],
            },
        }
        logger.info(
            "calibration (%s): n_calib=%d  Brier %.4f→%.4f  logloss %.4f→%.4f  ECE %.4f→%.4f",
            method, len(y_calib),
            m["brier"], m_cal["brier"],
            m["log_loss"], m_cal["log_loss"],
            ece_pre, ece_post,
        )

    # Predictions table for downstream inspection.
    pred_cols: dict = {}
    if "nct_id" in test_df.columns:
        pred_cols["nct_id"] = test_df["nct_id"].values
    pred_cols["candidate_id"] = test_df["candidate_id"].values
    pred_cols["y_true"] = y_test
    pred_cols["y_proba"] = y_proba
    if y_proba_calibrated is not None:
        pred_cols["y_proba_calibrated"] = y_proba_calibrated
    pred_df = pd.DataFrame(pred_cols)

    # Annotate feature importances by group.
    fi = model.feature_importances()

    return RunResult(
        config=config,
        groups=tuple(g.name for g in groups),
        feature_names=feature_names,
        n_features=n_features,
        n_train=len(train_df),
        n_test=len(test_df),
        train_pos=int(y_train.sum()),
        test_pos=int(y_test.sum()),
        metrics={**m, "group_widths": dict(group_widths)},
        feature_importances=fi,
        test_predictions=pred_df,
        fitted_groups=groups,
        fitted_model=model,
        calibrator=calibrator,
        calibration_metrics=calibration_metrics,
        n_calib=int(len(y_calib)) if y_calib is not None else 0,
        calib_pos=int(y_calib.sum()) if y_calib is not None else 0,
        per_phase_metrics=per_phase_metrics,
        per_phase_metrics_calibrated=per_phase_metrics_calibrated,
    )


def to_summary_row(name: str, result: RunResult) -> dict:
    """Flatten a RunResult into one row for the ablation summary CSV."""
    m = result.metrics
    return {
        "subset_name": name,
        "groups": ",".join(result.groups),
        "n_train": result.n_train,
        "n_test": result.n_test,
        "train_pos": result.train_pos,
        "test_pos": result.test_pos,
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "f1": m.get("f1"),
        "brier": m.get("brier"),
        "log_loss": m.get("log_loss"),
        "balanced_accuracy": m.get("balanced_accuracy"),
        "n_features": result.n_features,
        "tp": m.get("tp"),
        "fp": m.get("fp"),
        "tn": m.get("tn"),
        "fn": m.get("fn"),
    }
