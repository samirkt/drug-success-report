"""Assemble a single consolidated PDF from per-subcommand model artifacts.

Reads from `model_runs/<train|rfe|ablate|baselines>` (paths configurable) and
the HINT runner's per-phase metrics CSV (produced by `run_hint.sh` next to the
input dataset) and emits one PDF covering:

  1. Title / run-section manifest
  2. All-features model: ROC + PRC (side-by-side, landscape)
  3. All-features model: F1 vs threshold + operating-points table
  4. RFE: knee plot + history table
  5. Ablation: per-group ΔAUC bar chart + full summary table
  6. Trial-level model vs HINT: per-phase metrics comparison table

Missing sections are noted in the manifest and rendered as skip-notice pages
(per-section is optional). Run via:

    uv run python scripts/build_consolidated_report.py \\
      --run-root ./model_runs \\
      --train-name t2019 \\
      --rfe-name rfe_t2019 \\
      --ablate-name loo_t2019 \\
      --baselines-name baselines \\
      --hint-metrics ./model_runs/t2019/trial/hint_results.csv \\
      --output ./outputs/consolidated_report.pdf

Optional alternative for HINT row-level predictions (if available):
    --hint-predictions /path/to/hint_predictions.csv  (joins on nct_id; renders ROC+PRC)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

# Make the project root importable so `model.*` resolves when this script is
# invoked directly via `uv run python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from model.report import (  # noqa: E402
    _add_caption,
    _curve_page_roc,
    _curve_page_prc,
    _dual_curves_page,
    _f1_threshold_page,
    _new_landscape,
    _new_portrait,
    _rfe_pages,
    _write_table_page,
    _write_text_page,
)
from model.rfe import RFEStep, RFESummary  # noqa: E402

logger = logging.getLogger("consolidated_report")


# ─────────────────────────────────────────── loaders ────


def _load_run(run_dir: Path) -> SimpleNamespace:
    """Load metrics.json + predictions.csv + feature_importances.csv into a
    duck-typed RunResult that the report.py helpers can consume.
    """
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "predictions.csv"
    fi_path = run_dir / "feature_importances.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    with open(metrics_path) as fh:
        meta = json.load(fh)

    preds_df = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        feature_names = fi_df["feature"].tolist()
        feature_importances = fi_df["importance"].to_numpy()
    else:
        feature_names = []
        feature_importances = np.array([])

    return SimpleNamespace(
        groups=list(meta.get("groups", [])),
        n_features=int(meta.get("n_features", 0)),
        n_train=int(meta.get("n_train", 0)),
        n_test=int(meta.get("n_test", 0)),
        n_calib=int(meta.get("n_calib", 0)),
        train_pos=int(meta.get("train_pos", 0)),
        test_pos=int(meta.get("test_pos", 0)),
        calib_pos=int(meta.get("calib_pos", 0)),
        metrics=dict(meta.get("metrics", {})),
        calibration_metrics=meta.get("calibration"),
        config=meta.get("config", {}),
        test_predictions=preds_df,
        feature_names=feature_names,
        feature_importances=feature_importances,
    )


def _load_rfe_summary(rfe_dir: Path) -> RFESummary:
    summary_path = rfe_dir / "rfe_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"rfe_summary.json not found in {rfe_dir}")
    with open(summary_path) as fh:
        d = json.load(fh)
    history = [
        RFEStep(
            iteration=int(s["iteration"]),
            groups_remaining=tuple(s["groups_remaining"]),
            group_dropped=s.get("group_dropped"),
            dropped_importance=float(s.get("dropped_importance") or float("nan")),
            metric_value=float(s["metric_value"]),
            metric_value_std=float(s.get("metric_value_std", 0.0)),
            n_features=int(s.get("n_features", 0)),
        )
        for s in d.get("history", [])
    ]
    return RFESummary(
        history=history,
        optimal_groups=tuple(d.get("optimal_groups", ())),
        optimal_iteration=int(d.get("optimal_iteration", -1)),
        optimal_metric=float(d.get("optimal_metric", float("nan"))),
        metric_name=str(d.get("metric_name", "roc_auc")),
        cv=int(d.get("cv", 0)),
    )


def _load_hint_predictions(path: Path) -> pd.DataFrame:
    """Load HINT predictions and normalise to columns: nct_id, y_proba_hint, y_true_hint."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    nctid_col = cols.get("nctid") or cols.get("nct_id") or cols.get("trial_id")
    if nctid_col is None:
        raise ValueError(
            f"HINT predictions CSV at {path} must include nctid / nct_id / trial_id; "
            f"got columns {list(df.columns)}"
        )
    proba_candidates = (
        "y_pred_proba", "y_proba", "y_score", "score", "prediction", "pred",
        "probability", "hint_pred", "predicted_proba",
    )
    proba_col = next((cols[c] for c in proba_candidates if c in cols), None)
    if proba_col is None:
        raise ValueError(
            f"HINT predictions CSV at {path} must include a probability column "
            f"(one of {proba_candidates}); got columns {list(df.columns)}"
        )
    label_col = cols.get("y_true") or cols.get("label")
    out = pd.DataFrame({
        "nct_id": df[nctid_col].astype(str),
        "y_proba_hint": df[proba_col].astype(float),
    })
    if label_col is not None:
        out["y_true_hint"] = df[label_col].astype(int)
    return out


def _normalize_phase(v) -> Optional[str]:
    """Map heterogeneous phase strings ('Phase 3', '3', 'III', 'iii', …) → 'I'/'II'/'III'."""
    s = str(v).strip().lower().replace("phase", "").strip()
    return {"1": "I", "i": "I", "2": "II", "ii": "II", "3": "III", "iii": "III"}.get(s)


def _load_hint_metrics(path: Path) -> pd.DataFrame:
    """Load HINT's per-phase metrics CSV.

    Expected schema (written by hint_standalone/repo/run_hint_on_dataset.py):
      index=phase ∈ {'I','II','III'}; columns ⊇ {n, pos_rate, ROC-AUC, PR-AUC, F1, Precision, Recall, Accuracy}.
    Returns the frame with phase as a regular column.
    """
    df = pd.read_csv(path)
    # `to_csv(index=True)` from `metrics_df.set_index('phase')` writes 'phase' as the first column.
    if "phase" not in df.columns and df.columns[0].lower() in {"phase", "unnamed: 0"}:
        df = df.rename(columns={df.columns[0]: "phase"})
    if "phase" not in df.columns:
        raise ValueError(
            f"HINT metrics CSV at {path} must include a 'phase' column; got {list(df.columns)}"
        )
    return df


def _per_phase_model_metrics(
    trial_preds: pd.DataFrame,
    hint_input: pd.DataFrame,
) -> pd.DataFrame:
    """Compute our trial-model's per-phase metrics on the inner-join of nct_ids
    in `trial/predictions.csv` and the HINT input dataset (so both sides are
    evaluated on approximately the same trial subset)."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, accuracy_score,
    )
    if "nct_id" not in trial_preds.columns:
        raise ValueError("trial predictions.csv missing nct_id column")
    if "nctid" in hint_input.columns:
        right = hint_input.rename(columns={"nctid": "nct_id"})
    else:
        right = hint_input
    if "nct_id" not in right.columns:
        raise ValueError("hint_test.csv missing nctid/nct_id column")
    right = right[["nct_id", "phase"]].copy()
    right["phase"] = right["phase"].map(_normalize_phase)
    merged = trial_preds.merge(right, on="nct_id", how="inner")
    merged = merged.dropna(subset=["phase"])

    rows = []
    for phase, sub in merged.groupby("phase"):
        y = sub["y_true"].values.astype(int)
        p = sub["y_proba"].values.astype(float)
        pred = (p >= 0.5).astype(int)
        rows.append({
            "phase": phase,
            "n": int(len(sub)),
            "pos_rate": float(y.mean()) if len(y) else float("nan"),
            "ROC-AUC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "PR-AUC": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "F1": float(f1_score(y, pred)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "Accuracy": float(accuracy_score(y, pred)),
        })
    # Order by HINT's canonical phase order if possible.
    order = {"I": 0, "II": 1, "III": 2}
    rows.sort(key=lambda r: order.get(r["phase"], 99))
    return pd.DataFrame(rows)


def _build_hint_metrics_comparison(
    hint_metrics_df: pd.DataFrame,
    model_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Stack HINT's metrics next to our model's metrics in a long table:
    rows = (phase × {our model, HINT}); columns = metric names.
    """
    metric_cols = ["n", "pos_rate", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy"]
    keep_hint = [c for c in metric_cols if c in hint_metrics_df.columns]
    hint_long = hint_metrics_df[["phase"] + keep_hint].assign(model="HINT")
    keep_ours = [c for c in metric_cols if c in model_metrics_df.columns]
    ours_long = model_metrics_df[["phase"] + keep_ours].assign(model="our model")
    combined = pd.concat([ours_long, hint_long], ignore_index=True)
    # Interleave by phase for at-a-glance comparison.
    order = {"I": 0, "II": 1, "III": 2}
    combined["_phase_order"] = combined["phase"].map(order).fillna(99)
    combined["_model_order"] = combined["model"].map({"our model": 0, "HINT": 1})
    combined = combined.sort_values(["_phase_order", "_model_order"]).drop(
        columns=["_phase_order", "_model_order"]
    ).reset_index(drop=True)
    cols = ["phase", "model"] + [c for c in metric_cols if c in combined.columns]
    return combined[cols]


# ─────────────────────────────────────── page builders ────


@dataclass
class SectionStatus:
    name: str
    available: bool
    note: str = ""


@dataclass
class Manifest:
    sections: list[SectionStatus] = field(default_factory=list)

    def add(self, name: str, available: bool, note: str = "") -> None:
        self.sections.append(SectionStatus(name=name, available=available, note=note))


def _title_page(pdf: PdfPages, train_result: Optional[SimpleNamespace], manifest: Manifest) -> None:
    lines = ["Consolidated model report", "", "Sections:"]
    for s in manifest.sections:
        mark = "[x]" if s.available else "[ ]"
        suffix = f"  — {s.note}" if s.note else ""
        lines.append(f"  {mark} {s.name}{suffix}")
    lines.append("")
    if train_result is not None:
        cfg = train_result.config if isinstance(train_result.config, dict) else {}
        lines.append("Headline run (drug-indication, all features):")
        lines.append(f"  time_split_year     : {cfg.get('time_split_year')}")
        lines.append(f"  calibration_year    : {cfg.get('calibration_year')}")
        lines.append(f"  n_train / n_test    : {train_result.n_train} / {train_result.n_test}")
        m = train_result.metrics
        lines.append(
            f"  ROC-AUC / PR-AUC / F1: {m.get('roc_auc', float('nan')):.4f}"
            f" / {m.get('pr_auc', float('nan')):.4f}"
            f" / {m.get('f1', float('nan')):.4f}"
        )
    fig = _new_portrait()
    ax = fig.add_axes([0.07, 0.05, 0.86, 0.88])
    ax.set_axis_off()
    ax.set_title("Consolidated model report", fontsize=14, loc="left", pad=12)
    ax.text(0.0, 1.0, "\n".join(lines[2:]), family="monospace", fontsize=9, va="top", ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def _curves_page_combined(pdf: PdfPages, result: SimpleNamespace) -> None:
    """ROC + PRC side-by-side on a single landscape page (all-features model)."""
    from sklearn.metrics import (
        roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    )

    preds = result.test_predictions
    if preds is None or preds.empty:
        _write_text_page(pdf, "ROC + PRC", "(no test predictions)", caption="")
        return
    y_true = preds["y_true"].values.astype(int)
    y_proba = preds["y_proba"].values.astype(float)
    base_rate = float(y_true.mean()) if len(y_true) else 0.0

    fig = _new_landscape()
    _add_caption(
        fig,
        "All-features model — ROC and Precision-Recall curves on the test set. "
        f"Base rate = {base_rate:.3f} (dashed horizontal line on the PRC panel).",
    )
    gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.07, right=0.97, top=0.85, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"raw  AUC={roc_auc_score(y_true, y_proba):.4f}")
    if "y_proba_calibrated" in preds.columns and preds["y_proba_calibrated"].notna().any():
        y_cal = preds["y_proba_calibrated"].values.astype(float)
        fpr_c, tpr_c, _ = roc_curve(y_true, y_cal)
        ax.plot(fpr_c, tpr_c, linestyle="--", label=f"calibrated  AUC={roc_auc_score(y_true, y_cal):.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC", fontsize=11, pad=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(recall, precision, label=f"raw  AP={average_precision_score(y_true, y_proba):.4f}")
    if "y_proba_calibrated" in preds.columns and preds["y_proba_calibrated"].notna().any():
        y_cal = preds["y_proba_calibrated"].values.astype(float)
        p_c, r_c, _ = precision_recall_curve(y_true, y_cal)
        ax2.plot(r_c, p_c, linestyle="--", label=f"calibrated  AP={average_precision_score(y_true, y_cal):.4f}")
    ax2.axhline(base_rate, color="grey", linestyle=":", linewidth=0.8, label=f"base rate={base_rate:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall", fontsize=11, pad=6)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.01)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _ablation_bar_page(pdf: PdfPages, summary_df: pd.DataFrame, *, source: str) -> None:
    """Bar chart of ΔROC-AUC and ΔPR-AUC vs the `all` baseline."""
    df = summary_df.copy()
    if "subset_name" not in df.columns:
        _write_text_page(pdf, "Ablation", f"(unexpected schema in {source}: {list(df.columns)})")
        return
    all_row = df[df["subset_name"] == "all"]
    if all_row.empty:
        _write_text_page(pdf, "Ablation", f"(no `all` baseline row in {source})")
        return
    base_roc = float(all_row["roc_auc"].iloc[0])
    base_pr = float(all_row["pr_auc"].iloc[0])

    df = df[df["subset_name"] != "all"].copy()
    df["d_roc"] = df["roc_auc"].astype(float) - base_roc
    df["d_pr"] = df["pr_auc"].astype(float) - base_pr
    df["display"] = df["subset_name"].str.replace("^drop_", "− ", regex=True)
    df = df.sort_values("d_roc")  # most-negative (most-important when removed) first

    fig = _new_landscape()
    _add_caption(
        fig,
        f"Leave-one-group-out ablation. Bars show ΔAUC vs the all-features baseline "
        f"(ROC-AUC={base_roc:.4f}, PR-AUC={base_pr:.4f}). Negative bars = removing that group "
        f"hurt performance (i.e., the group was contributing). Source: {source}.",
    )
    gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.13, right=0.97, top=0.85, bottom=0.10)

    y = np.arange(len(df))
    colors_roc = ["tab:red" if v < 0 else "tab:green" for v in df["d_roc"]]
    colors_pr = ["tab:red" if v < 0 else "tab:green" for v in df["d_pr"]]

    ax = fig.add_subplot(gs[0, 0])
    ax.barh(y, df["d_roc"].values, color=colors_roc)
    ax.set_yticks(y); ax.set_yticklabels(df["display"].tolist(), fontsize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"Δ ROC-AUC (vs all={base_roc:.4f})")
    ax.set_title("ROC-AUC ablation", fontsize=11, pad=6)
    ax.grid(axis="x", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(y, df["d_pr"].values, color=colors_pr)
    ax2.set_yticks(y); ax2.set_yticklabels(df["display"].tolist(), fontsize=8)
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(f"Δ PR-AUC (vs all={base_pr:.4f})")
    ax2.set_title("PR-AUC ablation", fontsize=11, pad=6)
    ax2.grid(axis="x", alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _skip_page(pdf: PdfPages, section: str, reason: str) -> None:
    _write_text_page(
        pdf,
        f"{section} — skipped",
        f"{reason}",
        caption=f"This section was skipped because the source data wasn't available.",
    )


# ──────────────────────────────────────────── HINT helpers ────


def _build_hint_result(
    trial_preds: pd.DataFrame,
    hint_preds: pd.DataFrame,
) -> tuple[SimpleNamespace, SimpleNamespace, str]:
    """Inner-join trial-model predictions with HINT predictions on nct_id and
    return two duck-typed RunResults (model_side, hint_side) plus a coverage note.
    """
    if "nct_id" not in trial_preds.columns:
        raise ValueError("trial predictions.csv missing nct_id column")
    merged = trial_preds.merge(hint_preds, on="nct_id", how="inner")
    n_hint = len(hint_preds)
    n_trial = len(trial_preds)
    n_join = len(merged)

    # If HINT carried its own labels, flag mismatches but defer to trial y_true.
    label_note = ""
    if "y_true_hint" in merged.columns:
        mismatch = int((merged["y_true_hint"] != merged["y_true"]).sum())
        if mismatch:
            label_note = f" Label mismatch on {mismatch}/{n_join} rows (using trial-model labels)."

    model_side = SimpleNamespace(
        test_predictions=merged[["y_true", "y_proba"]].copy(),
    )
    hint_side = SimpleNamespace(
        test_predictions=merged.rename(columns={"y_proba_hint": "y_proba"})[["y_true", "y_proba"]].copy(),
    )
    note = (
        f"Joined on nct_id: HINT scored {n_join} of {n_trial} trial-model rows "
        f"({n_hint} HINT rows in total).{label_note}"
    )
    return model_side, hint_side, note


# ──────────────────────────────────────────── orchestration ────


def build_report(
    output_path: Path,
    train_dir: Optional[Path],
    rfe_dir: Optional[Path],
    ablate_dir: Optional[Path],
    baselines_dir: Optional[Path],
    trial_dir: Optional[Path],
    hint_predictions: Optional[Path] = None,
    hint_metrics: Optional[Path] = None,
    hint_input_csv: Optional[Path] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = Manifest()

    # Load the headline (drug-indication) run.
    train_result: Optional[SimpleNamespace] = None
    if train_dir is not None and train_dir.exists():
        try:
            train_result = _load_run(train_dir)
            manifest.add("All-features model: ROC + PRC", True, str(train_dir))
            manifest.add("All-features model: F1 + operating points", True, str(train_dir))
        except Exception as exc:  # noqa: BLE001
            manifest.add("All-features model: ROC + PRC", False, f"load failed: {exc}")
            manifest.add("All-features model: F1 + operating points", False, f"load failed: {exc}")
    else:
        manifest.add("All-features model: ROC + PRC", False, f"not found: {train_dir}")
        manifest.add("All-features model: F1 + operating points", False, f"not found: {train_dir}")

    # RFE
    rfe_summary: Optional[RFESummary] = None
    if rfe_dir is not None and rfe_dir.exists():
        try:
            rfe_summary = _load_rfe_summary(rfe_dir)
            manifest.add("RFE", True, str(rfe_dir))
        except Exception as exc:  # noqa: BLE001
            manifest.add("RFE", False, f"load failed: {exc}")
    else:
        manifest.add("RFE", False, f"not found: {rfe_dir}")

    # Ablation
    ablate_df: Optional[pd.DataFrame] = None
    if ablate_dir is not None and (ablate_dir / "ablation_summary.csv").exists():
        ablate_df = pd.read_csv(ablate_dir / "ablation_summary.csv")
        manifest.add("Ablation", True, str(ablate_dir / "ablation_summary.csv"))
    else:
        manifest.add("Ablation", False, f"not found: {ablate_dir}/ablation_summary.csv")

    # Baselines (informational; not its own section, but useful in trial vs HINT context)
    if baselines_dir is not None and (baselines_dir / "ablation_summary.csv").exists():
        baselines_df: Optional[pd.DataFrame] = pd.read_csv(baselines_dir / "ablation_summary.csv")
    else:
        baselines_df = None

    # Trial-level vs HINT
    trial_result: Optional[SimpleNamespace] = None
    hint_pair: Optional[tuple[SimpleNamespace, SimpleNamespace, str]] = None
    hint_metrics_pair: Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]] = None
    if trial_dir is not None and trial_dir.exists():
        try:
            trial_result = _load_run(trial_dir)
        except Exception as exc:  # noqa: BLE001
            manifest.add("Trial-level vs HINT", False, f"trial run load failed: {exc}")
    # Prefer the per-phase metrics path (matches the HINT runner's output) when supplied;
    # fall back to row-level predictions if the user passed those instead.
    if trial_result is not None and hint_metrics is not None and hint_metrics.exists():
        try:
            hint_metrics_df = _load_hint_metrics(hint_metrics)
            # Locate hint_test.csv (the input HINT scored) for the nct_id → phase mapping.
            hint_input_path = hint_input_csv if hint_input_csv else (trial_dir / "hint_test.csv")
            if not hint_input_path.exists():
                raise FileNotFoundError(
                    f"need {hint_input_path} for the nct_id → phase mapping when --hint-metrics is supplied"
                )
            hint_input_df = pd.read_csv(hint_input_path, usecols=["nctid", "phase"])
            ours_df = _per_phase_model_metrics(trial_result.test_predictions, hint_input_df)
            combined_df = _build_hint_metrics_comparison(hint_metrics_df, ours_df)
            note = (
                f"HINT per-phase metrics from {hint_metrics}; our model's per-phase metrics "
                f"computed on the inner-join of trial/predictions.csv ↔ {hint_input_path.name} (n by phase shown below)."
            )
            hint_metrics_pair = (hint_metrics_df, ours_df, combined_df, note)
            manifest.add("Trial-level vs HINT", True, str(hint_metrics))
        except Exception as exc:  # noqa: BLE001
            manifest.add("Trial-level vs HINT", False, f"HINT metrics load failed: {exc}")
    elif trial_result is not None and hint_predictions is not None and hint_predictions.exists():
        try:
            hint_df = _load_hint_predictions(hint_predictions)
            hint_pair = _build_hint_result(trial_result.test_predictions, hint_df)
            manifest.add("Trial-level vs HINT", True, str(hint_predictions))
        except Exception as exc:  # noqa: BLE001
            manifest.add("Trial-level vs HINT", False, f"HINT alignment failed: {exc}")
    elif trial_result is not None:
        manifest.add("Trial-level vs HINT", False, "--hint-metrics / --hint-predictions not supplied")
    else:
        manifest.add("Trial-level vs HINT", False, f"trial run dir not found: {trial_dir}")

    # ─── Render PDF ───
    with PdfPages(output_path) as pdf:
        _title_page(pdf, train_result, manifest)

        # All-features (drug indication)
        if train_result is not None:
            _curves_page_combined(pdf, train_result)
            chosen_thr = float(train_result.metrics.get("threshold", 0.5))
            _f1_threshold_page(pdf, train_result, chosen_threshold=chosen_thr)
        else:
            _skip_page(pdf, "All-features model — ROC + PRC", f"train dir missing: {train_dir}")
            _skip_page(pdf, "All-features model — F1 + operating points", f"train dir missing: {train_dir}")

        # RFE
        if rfe_summary is not None:
            _rfe_pages(pdf, rfe_summary)
        else:
            _skip_page(
                pdf, "RFE",
                f"RFE artifacts not found at {rfe_dir}. "
                f"Run: `uv run python -m model rfe --time-split-year 2019 --output {rfe_dir}`",
            )

        # Ablation
        if ablate_df is not None:
            _ablation_bar_page(pdf, ablate_df, source=str(ablate_dir / "ablation_summary.csv"))
            display_cols = [c for c in ["subset_name", "n_features", "roc_auc", "pr_auc", "f1", "brier"] if c in ablate_df.columns]
            _write_table_page(
                pdf, ablate_df[display_cols].sort_values("roc_auc", ascending=False).reset_index(drop=True),
                title="Ablation summary",
                caption="Per-subset metrics from the LOO ablation. Sorted by ROC-AUC (descending).",
            )
        else:
            _skip_page(
                pdf, "Ablation",
                f"Ablation summary not found at {ablate_dir}. "
                f"Run: `uv run python -m model ablate --mode loo --time-split-year 2019 --output {ablate_dir}`",
            )

        # Baselines (compact summary; not a top-level section unless present)
        if baselines_df is not None:
            display_cols = [c for c in ["subset_name", "n_features", "roc_auc", "pr_auc", "f1", "brier"] if c in baselines_df.columns]
            _write_table_page(
                pdf, baselines_df[display_cols].sort_values("roc_auc", ascending=False).reset_index(drop=True),
                title="Baselines summary",
                caption="Per-baseline metrics on the same test split as the full model.",
            )

        # Trial-level vs HINT
        if hint_metrics_pair is not None:
            _hint_df, _ours_df, combined_df, note = hint_metrics_pair
            _write_table_page(
                pdf, combined_df,
                title="Trial-level model vs HINT — per-phase metrics",
                caption=note,
            )
        elif hint_pair is not None:
            model_side, hint_side, note = hint_pair
            _dual_curves_page(
                pdf, model_side, hint_side,
                label_a="our model", label_b="HINT",
                title="Trial-level model vs HINT",
                caption=note,
            )
            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            def _row(name, side):
                y = side.test_predictions["y_true"].values.astype(int)
                p = side.test_predictions["y_proba"].values.astype(float)
                pred = (p >= 0.5).astype(int)
                return {
                    "model": name,
                    "n": len(y),
                    "n_pos": int(y.sum()),
                    "roc_auc": float(roc_auc_score(y, p)),
                    "pr_auc": float(average_precision_score(y, p)),
                    "f1@0.5": float(f1_score(y, pred)),
                }
            comp_df = pd.DataFrame([_row("our model", model_side), _row("HINT", hint_side)])
            _write_table_page(
                pdf, comp_df,
                title="Trial-level vs HINT — scalar metrics on the joined test set",
                caption="Both models evaluated on the inner-join of nct_id between trial-model predictions and HINT predictions.",
            )
        elif trial_result is not None:
            _skip_page(
                pdf, "Trial-level vs HINT",
                "No --hint-metrics (or --hint-predictions) provided. Run "
                "`./run_hint.sh model_runs/t2019/trial/hint_test.csv` and pass the resulting "
                "metrics CSV via --hint-metrics.",
            )
        else:
            _skip_page(
                pdf, "Trial-level vs HINT",
                f"Trial-level run dir not found at {trial_dir}. "
                f"Re-run training; `model train` writes both drug_indication/ and trial/ subdirs.",
            )

    logger.info("consolidated report written to %s", output_path)
    return output_path


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, default=Path("./model_runs"), help="Root containing per-subcommand run dirs.")
    ap.add_argument("--train-name", default="t2019", help="Name of the train run directory under --run-root.")
    ap.add_argument("--rfe-name", default="rfe_t2019", help="Name of the RFE run directory under --run-root.")
    ap.add_argument("--ablate-name", default="loo_t2019", help="Name of the ablation run directory under --run-root.")
    ap.add_argument("--baselines-name", default="baselines", help="Name of the baselines run directory under --run-root.")
    ap.add_argument("--hint-metrics", type=Path, default=None,
                    help="Per-phase metrics CSV produced by the HINT runner (default output: next to the input hint_test.csv).")
    ap.add_argument("--hint-predictions", type=Path, default=None,
                    help="Optional: row-level HINT predictions CSV (nctid + probability). Used only if --hint-metrics is not supplied.")
    ap.add_argument("--hint-input-csv", type=Path, default=None,
                    help="Override path to the HINT input dataset (defaults to <trial_dir>/hint_test.csv) for the nct_id→phase mapping.")
    ap.add_argument("--output", "-o", type=Path, default=Path("./outputs/consolidated_report.pdf"), help="Output PDF path.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_root: Path = args.run_root
    train_root = run_root / args.train_name
    # `model train` writes drug_indication/ and trial/ subdirs when granularity is "drug_indication"
    # (the default), so use those when present; otherwise fall back to the train dir itself.
    train_dir = train_root / "drug_indication"
    if not train_dir.exists():
        train_dir = train_root
    trial_dir = train_root / "trial"

    out = build_report(
        output_path=args.output,
        train_dir=train_dir if train_dir.exists() else None,
        rfe_dir=run_root / args.rfe_name,
        ablate_dir=run_root / args.ablate_name,
        baselines_dir=run_root / args.baselines_name,
        trial_dir=trial_dir if trial_dir.exists() else None,
        hint_predictions=args.hint_predictions,
        hint_metrics=args.hint_metrics,
        hint_input_csv=args.hint_input_csv,
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
