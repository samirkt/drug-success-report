"""Orchestration: load data, split once, run all baselines, save artifacts.

Mirrors `model/ablate.py:run_ablation` so the same train/test split is
shared across baselines and the resulting `ablation_summary.csv` is
auto-discovered by `model/report.py`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import data as data_mod
from .. import evaluate, splits
from ..artifacts import save_run
from ..config import ModelingConfig
from ..train import RunResult, to_summary_row
from . import ALL_BASELINES, build_baseline
from .target_only import TARGET_ONLY_GROUPS, run_target_only

logger = logging.getLogger(__name__)


@dataclass
class BaselinesConfig:
    base: ModelingConfig = field(default_factory=ModelingConfig)
    baselines: tuple[str, ...] = ALL_BASELINES


@dataclass
class BaselineRunInfo:
    name: str
    metrics: dict
    n_features: int
    summary_row: dict


@dataclass
class BaselinesResult:
    runs: dict[str, BaselineRunInfo] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _save_lookup_baseline(
    *,
    name: str,
    output_dir: Path,
    config: ModelingConfig,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_proba: np.ndarray,
    metrics_dict: dict,
    baseline_state: dict,
) -> None:
    """Persist a non-XGBoost baseline (stratum, tanimoto) in the same shape
    as `model/artifacts.save_run` so report.py can render it identically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "groups": [name],
        "n_features": 1,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_pos": int(y_train.sum()),
        "test_pos": int(y_test.sum()),
        "metrics": metrics_dict,
        "config": config,
        "baseline": {"name": name, "state": baseline_state},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    pd.DataFrame(
        {
            "candidate_id": test_df["candidate_id"].values,
            "y_true": y_test,
            "y_proba": y_proba,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)


def _summary_row(
    *,
    name: str,
    groups: str,
    n_features: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    metrics_dict: dict,
) -> dict:
    return {
        "subset_name": name,
        "groups": groups,
        "n_features": n_features,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_pos": int(y_train.sum()),
        "test_pos": int(y_test.sum()),
        "roc_auc": metrics_dict.get("roc_auc"),
        "pr_auc": metrics_dict.get("pr_auc"),
        "f1": metrics_dict.get("f1"),
        "brier": metrics_dict.get("brier"),
        "log_loss": metrics_dict.get("log_loss"),
        "balanced_accuracy": metrics_dict.get("balanced_accuracy"),
        "tp": metrics_dict.get("tp"),
        "fp": metrics_dict.get("fp"),
        "tn": metrics_dict.get("tn"),
        "fn": metrics_dict.get("fn"),
    }


def run_baselines(config: BaselinesConfig) -> BaselinesResult:
    base = config.base
    df = data_mod.build_modeling_frame(base)
    train_idx, test_idx = splits.split(
        df,
        test_size=base.test_size,
        seed=base.seed,
        group_by=base.group_by,
        time_split_column=base.time_split_column,
        time_split_year=base.time_split_year,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train = train_df["y"].values.astype(int)
    y_test = test_df["y"].values.astype(int)

    output_root = Path(base.output_dir) if base.output_dir else None
    runs: dict[str, BaselineRunInfo] = {}
    rows: list[dict] = []

    for name in config.baselines:
        logger.info("--- baseline %s ---", name)
        if name == "target_only":
            result = run_target_only(
                base, df=df, train_idx=train_idx, test_idx=test_idx
            )
            if output_root is not None:
                save_run(result, output_root / name)
            row = to_summary_row(name, result)
            rows.append(row)
            runs[name] = BaselineRunInfo(
                name=name,
                metrics=result.metrics,
                n_features=result.n_features,
                summary_row=row,
            )
        else:
            bl = build_baseline(name)
            bl.fit(train_df, y_train)
            y_proba = bl.predict_proba(test_df)
            m = evaluate.metrics(y_test, y_proba)
            if output_root is not None:
                _save_lookup_baseline(
                    name=name,
                    output_dir=output_root / name,
                    config=base,
                    test_df=test_df,
                    y_train=y_train,
                    y_test=y_test,
                    y_proba=y_proba,
                    metrics_dict=m,
                    baseline_state=bl.metadata(),
                )
                # Optional hook for baselines that emit extra artifacts.
                if hasattr(bl, "write_artifacts"):
                    bl.write_artifacts(output_root / name)
            row = _summary_row(
                name=name,
                groups=name,
                n_features=1,
                y_train=y_train,
                y_test=y_test,
                metrics_dict=m,
            )
            rows.append(row)
            runs[name] = BaselineRunInfo(
                name=name, metrics=m, n_features=1, summary_row=row
            )

    summary = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_root / "ablation_summary.csv", index=False)

    return BaselinesResult(runs=runs, summary=summary)
