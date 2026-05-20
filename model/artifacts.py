"""Persist a RunResult to disk: metrics.json, predictions.csv,
feature_importances.csv, pipeline.pkl.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .train import RunResult

logger = logging.getLogger(__name__)


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


def save_run(result: RunResult, output_dir: Path, *, write_report: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "groups": list(result.groups),
        "n_features": result.n_features,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "n_calib": result.n_calib,
        "train_pos": result.train_pos,
        "test_pos": result.test_pos,
        "calib_pos": result.calib_pos,
        "metrics": result.metrics,
        "calibration": result.calibration_metrics or None,
        "config": result.config,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )

    if result.test_predictions is not None:
        result.test_predictions.to_csv(output_dir / "predictions.csv", index=False)

    if result.feature_importances is not None and len(result.feature_names) == len(result.feature_importances):
        fi_df = pd.DataFrame({
            "feature": result.feature_names,
            "importance": result.feature_importances,
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(output_dir / "feature_importances.csv", index=False)

    pipeline_path = output_dir / "pipeline.pkl"
    with pipeline_path.open("wb") as f:
        pickle.dump(
            {
                "fitted_groups": result.fitted_groups,
                "fitted_model": result.fitted_model,
                "calibrator": result.calibrator,
                "feature_names": result.feature_names,
                "label_config": result.config.label,
            },
            f,
        )

    if write_report:
        try:
            from .report import write_run_report
            write_run_report(result, output_dir / "report.pdf")
        except Exception as exc:  # noqa: BLE001
            logger.warning("report.pdf generation failed: %s", exc)

    logger.info("artifacts saved to %s", output_dir)
    return output_dir
