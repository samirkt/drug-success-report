"""Ablation harness — drop-one-out, single-group, all, custom subsets.

Reuses `train_one_run` per subset and pins the train/test split (same df
+ split indices) so subsets are directly comparable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from . import data as data_mod
from . import splits
from .artifacts import save_run
from .config import AblationConfig, FeatureConfig
from .train import RunResult, to_summary_row, train_one_run

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    runs: dict[str, RunResult] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _build_subsets(config: AblationConfig) -> dict[str, tuple[str, ...]]:
    enabled = tuple(config.base.features.enabled)
    if config.mode == "all":
        return {"all": enabled}
    if config.mode == "loo":
        out = {"all": enabled}
        for drop in enabled:
            name = f"drop_{drop}"
            out[name] = tuple(g for g in enabled if g != drop)
        return out
    if config.mode == "single":
        return {f"only_{g}": (g,) for g in enabled}
    if config.mode == "custom":
        if not config.custom_subsets:
            raise ValueError("mode='custom' requires non-empty custom_subsets")
        return {k: tuple(v) for k, v in config.custom_subsets.items()}
    raise ValueError(f"unknown ablation mode: {config.mode!r}")


def run_ablation(config: AblationConfig) -> AblationResult:
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

    subsets = _build_subsets(config)
    logger.info("ablation: mode=%s subsets=%d", config.mode, len(subsets))

    runs: dict[str, RunResult] = {}
    rows: list[dict] = []
    output_root = Path(base.output_dir) if base.output_dir else None

    for name, groups in subsets.items():
        logger.info("--- ablation subset %s: groups=%s ---", name, list(groups))
        sub_features = replace(base.features, enabled=groups)
        sub_cfg = replace(base, features=sub_features)
        result = train_one_run(sub_cfg, df=df, train_idx=train_idx, test_idx=test_idx)
        runs[name] = result
        rows.append(to_summary_row(name, result))
        if output_root is not None:
            save_run(result, output_root / "runs" / name)

    summary = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_root / "ablation_summary.csv", index=False)

    return AblationResult(runs=runs, summary=summary)
