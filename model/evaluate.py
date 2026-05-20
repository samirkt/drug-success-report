"""Metrics for binary classification."""

from __future__ import annotations

import numpy as np


def reliability_curve(y_true, y_proba, n_bins: int = 10) -> dict:
    """Equal-width reliability bins on [0, 1].

    Returns a dict with `bin_edges`, `bin_centers`, `bin_count`, `mean_pred`,
    `frac_pos`. Empty bins have NaN for mean_pred / frac_pos.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.clip(np.asarray(y_proba).astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Right-closed bin assignment with explicit handling for the upper edge.
    bin_idx = np.digitize(y_proba, edges[1:-1], right=False)
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_pred = np.full(n_bins, np.nan, dtype=np.float64)
    frac_pos = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        counts[b] = c
        if c > 0:
            mean_pred[b] = float(y_proba[mask].mean())
            frac_pos[b] = float(y_true[mask].mean())
    return {
        "bin_edges": edges,
        "bin_centers": centers,
        "bin_count": counts,
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
    }


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Sample-weighted mean absolute gap between mean predicted prob and
    observed positive rate per bin. Empty bins are excluded.
    """
    rc = reliability_curve(y_true, y_proba, n_bins=n_bins)
    counts = rc["bin_count"]
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    gaps = np.abs(rc["mean_pred"] - rc["frac_pos"])
    valid = counts > 0
    if not valid.any():
        return float("nan")
    return float(np.sum(counts[valid] * gaps[valid]) / total)


def metrics(y_true, y_proba, *, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-7, 1 - 1e-7))) if len(set(y_true)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return out
