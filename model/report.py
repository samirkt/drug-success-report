"""Per-run PDF report.

Bundles the existing scalar outputs (run summary, group widths, metrics)
together with the new ROC / PRC / reliability / feature-importance figures
into a single document. Used by `train`, `ablate`, `baselines`, and `rfe`.

Helpers are lifted from `scripts/analyze_feature_distributions.py` so the
model package stays self-contained (no cross-script imports).
"""

from __future__ import annotations

import logging
import textwrap
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from . import evaluate  # noqa: E402

if TYPE_CHECKING:
    from .train import RunResult

logger = logging.getLogger(__name__)


# ─────────────────────────────── PDF helpers (lifted from scripts) ────


def _add_caption(fig: plt.Figure, caption: str) -> None:
    wrapped = "\n".join(textwrap.wrap(caption, width=110))
    fig.suptitle(wrapped, fontsize=9, y=0.995, ha="center", va="top")
    fig.subplots_adjust(top=0.88)


def _new_landscape() -> plt.Figure:
    return plt.figure(figsize=(11, 8.5))


def _new_portrait() -> plt.Figure:
    return plt.figure(figsize=(8.5, 11))


def _write_text_page(
    pdf: PdfPages,
    title: str,
    body: str,
    *,
    caption: Optional[str] = None,
) -> None:
    fig = _new_portrait()
    if caption:
        _add_caption(fig, caption)
    ax = fig.add_axes([0.06, 0.04, 0.9, 0.86])
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    ax.text(0.0, 1.0, body, family="monospace", fontsize=8, va="top", ha="left", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _write_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    *,
    title: str,
    caption: str,
    max_rows_per_page: int = 38,
    float_fmt: str = "{:.4f}",
) -> None:
    if df.empty:
        _write_text_page(pdf, title, "(no rows)", caption=caption)
        return
    chunks = [df.iloc[i:i + max_rows_per_page] for i in range(0, len(df), max_rows_per_page)]
    n_chunks = len(chunks)
    for j, chunk in enumerate(chunks, start=1):
        formatters = {
            c: float_fmt.format
            for c in chunk.columns
            if pd.api.types.is_float_dtype(chunk[c])
        }
        body = chunk.to_string(index=False, formatters=formatters, max_colwidth=40)
        page_title = title if n_chunks == 1 else f"{title}  ({j}/{n_chunks})"
        _write_text_page(pdf, page_title, body, caption=caption)


# ───────────────────────────────────────────────── page builders ────


def _format_config(result: "RunResult") -> str:
    cfg = result.config
    cfg_d = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    lines = []
    keep = (
        "candidate_detail_path",
        "fingerprints_path",
        "embeddings_path",
        "model_name",
        "seed",
        "test_size",
        "group_by",
        "time_split_column",
        "time_split_year",
        "calibration_year",
        "calibration_method",
    )
    for k in keep:
        if k in cfg_d and cfg_d[k] not in (None, ""):
            lines.append(f"  {k:>26s} : {cfg_d[k]}")
    label = cfg_d.get("label", {})
    features = cfg_d.get("features", {})
    if label:
        lines.append(f"  {'positive_outcomes':>26s} : {label.get('positive')}")
        lines.append(f"  {'negative_outcomes':>26s} : {label.get('negative')}")
        lines.append(f"  {'exclude_outcomes':>26s} : {label.get('exclude_outcomes')}")
    if features:
        lines.append(f"  {'enabled_groups':>26s} : {list(features.get('enabled', ()))}")
        lines.append(f"  {'top_k_targets':>26s} : {features.get('top_k_targets')}")
        lines.append(f"  {'top_k_pathways':>26s} : {features.get('top_k_pathways')}")
        lines.append(f"  {'top_k_mesh':>26s} : {features.get('top_k_mesh')}")
    return "\n".join(lines)


def _summary_body(result: "RunResult") -> str:
    n_train, n_test, n_calib = result.n_train, result.n_test, result.n_calib
    train_pos = result.train_pos
    test_pos = result.test_pos
    calib_pos = result.calib_pos
    tr_rate = 100.0 * train_pos / max(n_train, 1)
    te_rate = 100.0 * test_pos / max(n_test, 1)
    lines = [
        f"groups (in order):  {', '.join(result.groups)}",
        f"n_features:         {result.n_features}",
        "",
        f"n_train:  {n_train:>7}   positives: {train_pos:>5} ({tr_rate:5.2f}%)",
        f"n_test:   {n_test:>7}   positives: {test_pos:>5} ({te_rate:5.2f}%)",
    ]
    if n_calib > 0:
        ca_rate = 100.0 * calib_pos / max(n_calib, 1)
        lines.append(f"n_calib:  {n_calib:>7}   positives: {calib_pos:>5} ({ca_rate:5.2f}%)")
    group_widths = result.metrics.get("group_widths", {})
    if group_widths:
        lines.append("")
        lines.append("group widths:")
        for g, w in group_widths.items():
            lines.append(f"  {g:>16s} : {w}")
    lines.append("")
    lines.append("config:")
    lines.append(_format_config(result))
    return "\n".join(lines)


def _metrics_table(result: "RunResult") -> pd.DataFrame:
    m = result.metrics
    rows = [
        ("roc_auc",            m.get("roc_auc")),
        ("pr_auc",             m.get("pr_auc")),
        ("f1",                 m.get("f1")),
        ("brier",              m.get("brier")),
        ("log_loss",           m.get("log_loss")),
        ("balanced_accuracy",  m.get("balanced_accuracy")),
        ("threshold",          m.get("threshold")),
        ("tp",                 m.get("tp")),
        ("fp",                 m.get("fp")),
        ("tn",                 m.get("tn")),
        ("fn",                 m.get("fn")),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value"])
    cal = result.calibration_metrics
    if cal:
        post = cal.get("post", {})
        pre = cal.get("pre", {})
        pre_col = []
        post_col = []
        for name, _ in rows:
            if name in post:
                post_col.append(post.get(name))
            else:
                post_col.append(None)
            if name in pre:
                pre_col.append(pre.get(name))
            else:
                pre_col.append(None)
        df = df.rename(columns={"value": "raw"})
        df["raw_extra"] = pre_col
        df["calibrated"] = post_col
        # Merge pre into raw if raw is None (so ece shows up under raw too)
        df["raw"] = df.apply(
            lambda r: r["raw_extra"] if r["raw"] is None else r["raw"], axis=1,
        )
        df = df.drop(columns=["raw_extra"])
    return df


def _curve_page_roc(pdf: PdfPages, result: "RunResult") -> None:
    from sklearn.metrics import roc_curve, roc_auc_score

    preds = result.test_predictions
    if preds is None or preds.empty:
        _write_text_page(pdf, "ROC curve", "(no test predictions)", caption="")
        return
    y_true = preds["y_true"].values.astype(int)
    y_proba = preds["y_proba"].values.astype(float)
    fig = _new_landscape()
    _add_caption(
        fig,
        "ROC curve on the test set. AUC is the area under this curve; the diagonal is random performance. "
        "When calibration is enabled, the calibrated probabilities are overlaid (rank preserved, so AUC matches).",
    )
    ax = fig.add_subplot(111)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"raw  AUC={roc_auc_score(y_true, y_proba):.4f}")
    if "y_proba_calibrated" in preds.columns and preds["y_proba_calibrated"].notna().any():
        y_cal = preds["y_proba_calibrated"].values.astype(float)
        fpr_c, tpr_c, _ = roc_curve(y_true, y_cal)
        ax.plot(fpr_c, tpr_c, linestyle="--", label=f"calibrated  AUC={roc_auc_score(y_true, y_cal):.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — test set", fontsize=11, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _curve_page_prc(pdf: PdfPages, result: "RunResult") -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score

    preds = result.test_predictions
    if preds is None or preds.empty:
        _write_text_page(pdf, "Precision-Recall curve", "(no test predictions)", caption="")
        return
    y_true = preds["y_true"].values.astype(int)
    y_proba = preds["y_proba"].values.astype(float)
    fig = _new_landscape()
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    _add_caption(
        fig,
        f"Precision-Recall curve on the test set. Base-rate precision (the dashed horizontal line at {base_rate:.3f}) "
        "is what a constant predictor would achieve; average precision (AP) is the area under this curve.",
    )
    ax = fig.add_subplot(111)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall, precision, label=f"raw  AP={average_precision_score(y_true, y_proba):.4f}")
    if "y_proba_calibrated" in preds.columns and preds["y_proba_calibrated"].notna().any():
        y_cal = preds["y_proba_calibrated"].values.astype(float)
        p_c, r_c, _ = precision_recall_curve(y_true, y_cal)
        ax.plot(r_c, p_c, linestyle="--", label=f"calibrated  AP={average_precision_score(y_true, y_cal):.4f}")
    ax.axhline(base_rate, color="grey", linestyle=":", linewidth=0.8, label=f"base rate={base_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — test set", fontsize=11, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _reliability_page(pdf: PdfPages, result: "RunResult", n_bins: int = 10) -> None:
    preds = result.test_predictions
    if preds is None or preds.empty:
        return
    if "y_proba_calibrated" not in preds.columns or not preds["y_proba_calibrated"].notna().any():
        return
    y_true = preds["y_true"].values.astype(int)
    y_raw = preds["y_proba"].values.astype(float)
    y_cal = preds["y_proba_calibrated"].values.astype(float)
    rc_raw = evaluate.reliability_curve(y_true, y_raw, n_bins=n_bins)
    rc_cal = evaluate.reliability_curve(y_true, y_cal, n_bins=n_bins)
    ece_raw = evaluate.expected_calibration_error(y_true, y_raw, n_bins=n_bins)
    ece_cal = evaluate.expected_calibration_error(y_true, y_cal, n_bins=n_bins)

    fig = _new_landscape()
    cal = result.calibration_metrics
    method = cal.get("method", "?")
    n_calib = cal.get("n_calib", 0)
    _add_caption(
        fig,
        f"Reliability diagram on the test set. Each point = one bin; the y=x diagonal is perfect calibration. "
        f"Calibrator: {method} fit on n={n_calib} held-out rows. "
        f"ECE raw → calibrated: {ece_raw:.4f} → {ece_cal:.4f}.",
    )
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, label="ideal")
    valid_raw = rc_raw["bin_count"] > 0
    valid_cal = rc_cal["bin_count"] > 0
    ax.plot(rc_raw["mean_pred"][valid_raw], rc_raw["frac_pos"][valid_raw], marker="o", label="raw")
    ax.plot(rc_cal["mean_pred"][valid_cal], rc_cal["frac_pos"][valid_cal], marker="s", label="calibrated")
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed positive rate (per bin)")
    ax.set_title("Reliability diagram — test set", fontsize=11, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _group_membership(result: "RunResult") -> list[str]:
    """Return parallel array (one entry per feature column) → group name."""
    out: list[str] = []
    widths = result.metrics.get("group_widths", {})
    # Preserve the canonical order from result.groups
    for g in result.groups:
        w = int(widths.get(g, 0))
        out.extend([g] * w)
    # Pad/truncate to len(feature_names) defensively
    if len(out) < len(result.feature_names):
        out.extend(["?"] * (len(result.feature_names) - len(out)))
    return out[: len(result.feature_names)]


def _feature_importance_page(pdf: PdfPages, result: "RunResult", top_n: int = 30) -> None:
    fi = result.feature_importances
    names = result.feature_names
    if fi is None or len(fi) == 0 or len(names) != len(fi):
        return
    membership = _group_membership(result)
    df = pd.DataFrame({"feature": names, "group": membership, "importance": fi})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    fig = _new_landscape()
    _add_caption(
        fig,
        f"Top-{top_n} features by XGBoost gain importance, color-coded by feature group. "
        "Full ranking is in feature_importances.csv.",
    )
    ax = fig.add_subplot(111)
    groups_unique = list(dict.fromkeys(df["group"].tolist()))
    cmap = plt.get_cmap("tab10")
    color_map = {g: cmap(i % 10) for i, g in enumerate(groups_unique)}
    colors = [color_map[g] for g in df["group"]]
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df["importance"].values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].tolist(), fontsize=7)
    ax.set_xlabel("importance (XGBoost gain)")
    ax.set_title(f"Top-{top_n} feature importances", fontsize=11, pad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[g]) for g in groups_unique]
    ax.legend(handles, groups_unique, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _f1_threshold_page(
    pdf: PdfPages,
    result: "RunResult",
    *,
    chosen_threshold: float = 0.5,
    extra_thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6),
) -> None:
    """F1/precision/recall sweep over thresholds + operating-points table on one page."""
    from sklearn.metrics import precision_recall_curve

    preds = result.test_predictions
    if preds is None or preds.empty:
        _write_text_page(pdf, "F1 vs threshold", "(no test predictions)", caption="")
        return
    y_true = preds["y_true"].values.astype(int)
    y_proba = preds["y_proba"].values.astype(float)

    precision, recall, thr = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns precision/recall arrays one longer than thr.
    p_at_thr = precision[:-1]
    r_at_thr = recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_at_thr = np.where(
            (p_at_thr + r_at_thr) > 0,
            2 * p_at_thr * r_at_thr / (p_at_thr + r_at_thr),
            0.0,
        )

    best_idx = int(np.nanargmax(f1_at_thr)) if len(f1_at_thr) else 0
    best_thr = float(thr[best_idx]) if len(thr) else float("nan")
    best_f1 = float(f1_at_thr[best_idx]) if len(f1_at_thr) else float("nan")

    fig = _new_landscape()
    _add_caption(
        fig,
        "Precision/recall/F1 as a function of decision threshold. The vertical red line marks the chosen operating "
        f"threshold ({chosen_threshold:.2f}); the green dashed line marks the F1-maximizing threshold ({best_thr:.3f} → F1={best_f1:.3f}). "
        "The table below lists confusion-matrix counts and PR/F1 at a few selected thresholds.",
    )
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.45, left=0.07, right=0.97, top=0.85, bottom=0.06)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(thr, p_at_thr, label="precision", color="tab:blue")
    ax.plot(thr, r_at_thr, label="recall", color="tab:orange")
    ax.plot(thr, f1_at_thr, label="F1", color="tab:green", linewidth=2.0)
    ax.axvline(chosen_threshold, color="red", linestyle="--", linewidth=0.9, label=f"chosen={chosen_threshold:.2f}")
    ax.axvline(best_thr, color="tab:green", linestyle=":", linewidth=0.9, label=f"argmax F1={best_thr:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / recall / F1 vs threshold — test set", fontsize=11, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", fontsize=8, ncols=2)
    ax.grid(alpha=0.25)

    # Operating-points table
    thresholds = sorted(set(list(extra_thresholds) + [chosen_threshold, best_thr]))
    rows = []
    n_pos_total = int(y_true.sum())
    n_neg_total = int(len(y_true) - n_pos_total)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(n_pos_total, 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        marker = ""
        if abs(t - chosen_threshold) < 1e-9:
            marker = " (chosen)"
        elif abs(t - best_thr) < 1e-9:
            marker = " (argmax F1)"
        rows.append({
            "threshold": f"{t:.3f}{marker}",
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": p, "recall": r, "F1": f1,
        })
    table_df = pd.DataFrame(rows)
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.set_axis_off()
    formatters = {c: "{:.4f}".format for c in table_df.columns if pd.api.types.is_float_dtype(table_df[c])}
    body = table_df.to_string(index=False, formatters=formatters)
    ax_tbl.text(
        0.0, 1.0, body, family="monospace", fontsize=8.5, va="top", ha="left",
    )
    ax_tbl.set_title(
        f"Operating points  (positives={n_pos_total}, negatives={n_neg_total})",
        fontsize=10, loc="left", pad=4,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _dual_curves_page(
    pdf: PdfPages,
    result_a: "RunResult",
    result_b: "RunResult",
    *,
    label_a: str,
    label_b: str,
    title: str,
    caption: str | None = None,
) -> None:
    """Side-by-side ROC + PRC comparing two models on (assumed-aligned) test sets."""
    from sklearn.metrics import (
        roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    )

    def _arrays(result):
        preds = result.test_predictions
        if preds is None or preds.empty:
            return None, None
        return (
            preds["y_true"].values.astype(int),
            preds["y_proba"].values.astype(float),
        )

    yA, pA = _arrays(result_a)
    yB, pB = _arrays(result_b)
    if yA is None or yB is None:
        _write_text_page(pdf, title, "(missing predictions for one of the two models)", caption=caption or "")
        return

    fig = _new_landscape()
    if caption:
        _add_caption(fig, caption)
    gs = fig.add_gridspec(1, 2, wspace=0.25, left=0.07, right=0.97, top=0.85, bottom=0.08)

    # ROC
    ax = fig.add_subplot(gs[0, 0])
    for y, p, name in ((yA, pA, label_a), (yB, pB, label_b)):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(y, p):.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC", fontsize=11, pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)

    # PRC
    ax2 = fig.add_subplot(gs[0, 1])
    base_rate = float(np.mean(np.concatenate([yA, yB])))
    for y, p, name in ((yA, pA, label_a), (yB, pB, label_b)):
        precision, recall, _ = precision_recall_curve(y, p)
        ax2.plot(recall, precision, label=f"{name}  AP={average_precision_score(y, p):.4f}")
    ax2.axhline(base_rate, color="grey", linestyle=":", linewidth=0.8, label=f"base rate≈{base_rate:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall", fontsize=11, pad=6)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.01)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle("" if caption else title, fontsize=12, y=0.97)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _rfe_pages(pdf: PdfPages, rfe_summary) -> None:
    if rfe_summary is None:
        return
    history = getattr(rfe_summary, "history", None)
    if not history:
        return

    rows = []
    for step in history:
        rows.append({
            "iteration": step.iteration,
            "n_groups": len(step.groups_remaining),
            "groups_remaining": ", ".join(step.groups_remaining),
            "group_dropped": step.group_dropped or "—",
            "dropped_importance": step.dropped_importance,
            "metric": step.metric_value,
            "metric_std": step.metric_value_std,
        })
    df = pd.DataFrame(rows)

    # Knee plot — metric vs n_groups
    fig = _new_landscape()
    optimal_n = len(rfe_summary.optimal_groups)
    _add_caption(
        fig,
        f"RFE curve: chosen metric vs number of remaining groups. Optimal subset (n={optimal_n}) "
        f"= {{{', '.join(rfe_summary.optimal_groups)}}}; metric = {rfe_summary.optimal_metric:.4f}.",
    )
    ax = fig.add_subplot(111)
    ax.plot(df["n_groups"], df["metric"], marker="o", label="metric")
    if (df["metric_std"] > 0).any():
        ax.fill_between(
            df["n_groups"],
            df["metric"] - df["metric_std"],
            df["metric"] + df["metric_std"],
            alpha=0.15, label="±1σ across folds",
        )
    ax.axvline(optimal_n, color="red", linestyle="--", linewidth=0.8, label=f"optimal n={optimal_n}")
    ax.set_xlabel("# groups remaining")
    ax.set_ylabel("metric")
    ax.set_title("Group-level RFE", fontsize=11, pad=8)
    ax.set_xticks(sorted(df["n_groups"].unique()))
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # History table
    table_df = df[["iteration", "n_groups", "group_dropped", "dropped_importance", "metric", "metric_std"]]
    _write_table_page(
        pdf, table_df,
        title="RFE history (each row = one iteration)",
        caption="`group_dropped` is the lowest-importance group removed at that step; `metric` is measured AFTER the drop.",
    )

    # Remaining groups per iteration (separate page for the long names)
    _write_table_page(
        pdf, df[["iteration", "n_groups", "groups_remaining"]],
        title="RFE remaining groups per iteration",
        caption="Snapshot of the feature-group set evaluated at each iteration.",
    )


# ───────────────────────────────────── public API ────


def write_run_report(
    result: "RunResult",
    out_path: Path,
    *,
    rfe_summary=None,
) -> Path:
    """Build the per-run PDF report at `out_path` and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        _write_text_page(
            pdf,
            "Training run — summary",
            _summary_body(result),
            caption=(
                "Snapshot of the dataset, split, and feature configuration used for this run. "
                "All downstream metrics in this report are computed on the test slice."
            ),
        )
        _write_table_page(
            pdf, _metrics_table(result),
            title="Scalar metrics",
            caption=(
                "Per-metric values on the test set. When calibration is enabled, the `calibrated` "
                "column shows post-calibration scores; ranking metrics (ROC-AUC, PR-AUC) are unchanged."
            ),
        )
        _curve_page_roc(pdf, result)
        _curve_page_prc(pdf, result)
        _reliability_page(pdf, result)
        _feature_importance_page(pdf, result)
        _rfe_pages(pdf, rfe_summary)

    logger.info("report PDF saved to %s", out_path)
    return out_path
