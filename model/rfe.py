"""Group-level Recursive Feature Elimination.

Iteratively trains the full pipeline, ranks feature *groups* by summed
gain-importance across each group's column slice, and drops the lowest-
ranked group. Records the chosen metric at each iteration so the caller
can pick the optimum.

Single-split mode: uses the same train/test split as the production
config (honors `time_split_year` / `calibration_year`).

CV mode: stratified K-fold over the training portion. Calibration is
disabled inside folds (no per-fold calibration-year guarantee); the final
retrain at the optimum re-enables calibration if requested by the base
config.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import data as data_mod
from . import splits
from .artifacts import save_run
from .config import ModelingConfig
from .train import RunResult, train_one_run

logger = logging.getLogger(__name__)


VALID_METRICS = ("roc_auc", "pr_auc", "f1", "brier", "log_loss")
# Metrics where lower is better.
MINIMIZE = {"brier", "log_loss"}


@dataclass
class RFEConfig:
    base: ModelingConfig
    metric: str = "roc_auc"
    cv: int = 0          # 0 → single split using base config; ≥2 → stratified K-fold on train portion
    step: int = 1        # groups removed per iteration
    seed: int = 0


@dataclass
class RFEStep:
    iteration: int
    groups_remaining: tuple[str, ...]
    group_dropped: Optional[str]
    dropped_importance: float
    metric_value: float
    metric_value_std: float = 0.0
    n_features: int = 0


@dataclass
class RFESummary:
    history: list[RFEStep] = field(default_factory=list)
    optimal_groups: tuple[str, ...] = ()
    optimal_iteration: int = -1
    optimal_metric: float = float("nan")
    metric_name: str = "roc_auc"
    cv: int = 0


def _aggregate_group_importance(result: RunResult) -> dict[str, float]:
    """Sum the model's feature importances across each group's column slice."""
    fi = result.feature_importances
    if fi is None or len(fi) == 0:
        return {g: 0.0 for g in result.groups}
    widths = result.metrics.get("group_widths", {})
    totals: dict[str, float] = {}
    offset = 0
    for g in result.groups:
        w = int(widths.get(g, 0))
        if w <= 0:
            totals[g] = 0.0
            continue
        block = fi[offset:offset + w]
        totals[g] = float(np.asarray(block).sum())
        offset += w
    return totals


def _score(result: RunResult, metric: str) -> float:
    return float(result.metrics.get(metric, float("nan")))


def _better(a: float, b: float, metric: str) -> bool:
    """Is `a` better than `b` under the chosen metric?"""
    if np.isnan(b):
        return True
    if metric in MINIMIZE:
        return a < b
    return a > b


def _config_for_subset(
    base: ModelingConfig, enabled: tuple[str, ...], *, disable_calibration: bool = False,
) -> ModelingConfig:
    sub_features = replace(base.features, enabled=enabled)
    cfg = replace(base, features=sub_features)
    if disable_calibration:
        cfg = replace(cfg, calibration_year=None)
    return cfg


def _evaluate_subset_single(
    base: ModelingConfig,
    enabled: tuple[str, ...],
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    calib_idx: Optional[np.ndarray],
    metric: str,
) -> tuple[float, float, RunResult]:
    cfg = _config_for_subset(base, enabled)
    result = train_one_run(
        cfg, df=df, train_idx=train_idx, test_idx=test_idx, calib_idx=calib_idx,
    )
    return _score(result, metric), 0.0, result


def _evaluate_subset_cv(
    base: ModelingConfig,
    enabled: tuple[str, ...],
    df: pd.DataFrame,
    train_idx_all: np.ndarray,
    cv: int,
    seed: int,
    metric: str,
) -> tuple[float, float, RunResult]:
    from sklearn.model_selection import StratifiedKFold

    cfg = _config_for_subset(base, enabled, disable_calibration=True)
    y_all = df["y"].values.astype(int)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores: list[float] = []
    last_result: Optional[RunResult] = None
    for fold_i, (fold_train_local, fold_test_local) in enumerate(
        skf.split(np.zeros(len(train_idx_all)), y_all[train_idx_all])
    ):
        fold_train_idx = train_idx_all[fold_train_local]
        fold_test_idx = train_idx_all[fold_test_local]
        result = train_one_run(
            cfg, df=df, train_idx=fold_train_idx, test_idx=fold_test_idx,
        )
        scores.append(_score(result, metric))
        last_result = result
        logger.info(
            "  fold %d/%d: %s=%.4f", fold_i + 1, cv, metric, scores[-1],
        )
    scores_arr = np.asarray(scores, dtype=float)
    assert last_result is not None
    return float(scores_arr.mean()), float(scores_arr.std(ddof=0)), last_result


def run_group_rfe(cfg: RFEConfig) -> tuple[RFESummary, Optional[RunResult]]:
    """Run group-level RFE and return (summary, final_retrain_result).

    final_retrain_result is `train_one_run` invoked one more time at
    `optimal_groups` using the original pinned split (with calibration
    honored from the base config) — this is what gets saved as the
    canonical optimal model.
    """
    if cfg.metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}; got {cfg.metric!r}")
    base = cfg.base
    enabled = tuple(base.features.enabled)
    if len(enabled) < 2:
        raise ValueError(
            f"need at least 2 enabled groups to run RFE; got {enabled}"
        )

    df = data_mod.build_modeling_frame(base)
    calib_idx: Optional[np.ndarray] = None
    if base.calibration_year is not None:
        train_idx, calib_idx, test_idx = splits.split_with_calibration(
            df,
            calibration_year=base.calibration_year,
            time_split_column=base.time_split_column,
        )
    else:
        train_idx, test_idx = splits.split(
            df,
            test_size=base.test_size,
            seed=base.seed,
            group_by=base.group_by,
            time_split_column=base.time_split_column,
            time_split_year=base.time_split_year,
        )

    summary = RFESummary(metric_name=cfg.metric, cv=cfg.cv)
    remaining = tuple(enabled)
    iteration = 0
    while True:
        iteration += 1
        logger.info(
            "RFE iter %d: %d groups remaining = %s",
            iteration, len(remaining), list(remaining),
        )
        if cfg.cv >= 2:
            score, score_std, last_result = _evaluate_subset_cv(
                base, remaining, df, train_idx, cfg.cv, cfg.seed, cfg.metric,
            )
        else:
            score, score_std, last_result = _evaluate_subset_single(
                base, remaining, df, train_idx, test_idx, calib_idx, cfg.metric,
            )
        logger.info("RFE iter %d: %s = %.4f ± %.4f", iteration, cfg.metric, score, score_std)
        importances = _aggregate_group_importance(last_result)
        # Sort ascending → lowest importance first
        ranked = sorted(remaining, key=lambda g: importances.get(g, 0.0))

        if len(remaining) <= 1:
            summary.history.append(RFEStep(
                iteration=iteration,
                groups_remaining=remaining,
                group_dropped=None,
                dropped_importance=float("nan"),
                metric_value=score,
                metric_value_std=score_std,
                n_features=last_result.n_features,
            ))
            break

        n_drop = max(1, min(cfg.step, len(remaining) - 1))
        to_drop = ranked[:n_drop]
        # For the history row we record the *first* dropped group's importance,
        # which is the lowest in this iteration.
        dropped_imp = float(importances.get(to_drop[0], 0.0))
        summary.history.append(RFEStep(
            iteration=iteration,
            groups_remaining=remaining,
            group_dropped=",".join(to_drop),
            dropped_importance=dropped_imp,
            metric_value=score,
            metric_value_std=score_std,
            n_features=last_result.n_features,
        ))
        remaining = tuple(g for g in remaining if g not in to_drop)

    # Pick optimal step. For CV: use mean - std (penalize variance) as the
    # ranking score. For single-split: just the metric.
    def rank_score(step: RFEStep) -> float:
        v = step.metric_value
        s = step.metric_value_std
        if cfg.metric in MINIMIZE:
            return v + s  # smaller is better → add std to penalize variance
        return v - s

    best_step = None
    for step in summary.history:
        if best_step is None:
            best_step = step
            continue
        if cfg.metric in MINIMIZE:
            if rank_score(step) < rank_score(best_step):
                best_step = step
        else:
            if rank_score(step) > rank_score(best_step):
                best_step = step
    assert best_step is not None
    summary.optimal_groups = best_step.groups_remaining
    summary.optimal_iteration = best_step.iteration
    summary.optimal_metric = best_step.metric_value

    # Final retrain on the original pinned split, honoring calibration.
    final_cfg = _config_for_subset(base, summary.optimal_groups)
    final_result = train_one_run(
        final_cfg, df=df, train_idx=train_idx, test_idx=test_idx, calib_idx=calib_idx,
    )

    logger.info(
        "RFE optimal: iteration=%d groups=%s metric(%s)=%.4f",
        summary.optimal_iteration,
        list(summary.optimal_groups),
        cfg.metric,
        summary.optimal_metric,
    )
    return summary, final_result


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_rfe(
    summary: RFESummary,
    final_result: Optional[RunResult],
    output_dir: Path,
) -> Path:
    """Persist RFE artifacts: history CSV, summary JSON, final retrain artifacts, report PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame([asdict(s) for s in summary.history])
    history_df.to_csv(output_dir / "rfe_history.csv", index=False)

    (output_dir / "rfe_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, default=_json_default)
    )

    if final_result is not None:
        # Save the optimal retrain alongside (its own metrics.json + report.pdf go under final/).
        save_run(final_result, output_dir / "final", write_report=False)
        # Top-level report.pdf bundles the optimal retrain's metrics + curves + the RFE pages.
        from .report import write_run_report
        try:
            write_run_report(final_result, output_dir / "report.pdf", rfe_summary=summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RFE report.pdf generation failed: %s", exc)

    logger.info("RFE artifacts saved to %s", output_dir)
    return output_dir
