"""
Outcome validation module for the peptide research pipeline.

Evaluates pipeline predictions against a manually-labeled benchmark using
binary one-vs-rest classification metrics per outcome class.

Cascade rule: a prediction (or ground truth) of "commercialized" also counts
as positive for the "approved" binary task. The inverse is not true.

Usage (programmatic):
    from eval.validate import load_benchmark, load_predictions_csv, evaluate, print_report

    benchmark, skipped = load_benchmark("eval/outcome_benchmark.csv")
    predictions = load_predictions_csv("eval/candidate_pairs.csv")
    report = evaluate(predictions, benchmark, skipped)
    print_report(report)

Usage (pipeline objects):
    from eval.validate import predictions_from_pipeline, evaluate, print_report

    preds = predictions_from_pipeline(result.candidate_table, result.outcome_table)
    benchmark, skipped = load_benchmark("eval/outcome_benchmark.csv")
    report = evaluate(preds, benchmark, skipped)
    print_report(report)
"""

import csv
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = ("investigational", "approved", "commercialized")

# Maps CandidateOutcome.value strings to evaluation buckets.
OUTCOME_TO_CLASS: dict[str, str] = {
    "Commercialized": "commercialized",
    "Approved": "approved",
    "Failed Phase 1": "investigational",
    "Failed Phase 2": "investigational",
    "Failed Phase 3": "investigational",
    "Ongoing": "investigational",
    "Unknown": "investigational",
}

# Benchmark label pairs that are unambiguous enough to evaluate.
# Rows with "unknown" or "ambiguous" in either field are skipped.
_SKIP_LABELS = {"unknown", "ambiguous"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float  # binary (one-vs-rest) accuracy for this class
    support: int     # number of true positives in ground truth (TP + FN)


@dataclass
class ValidationReport:
    class_metrics: dict[str, ClassMetrics]  # keyed by CLASSES bucket
    overall_accuracy: float                  # 3-class exact-match accuracy
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_accuracy: float
    matched: int    # benchmark rows that had a matching prediction
    skipped: int    # benchmark rows with unknown/ambiguous labels (excluded)
    unmatched: int  # benchmark rows with no matching prediction


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase and collapse whitespace — used for all matching keys."""
    return re.sub(r"\s+", " ", s.lower().strip())


def _is_positive(bucket: str, class_: str) -> bool:
    """
    Returns True if `bucket` counts as a positive example for `class_`'s
    binary task, applying the commercialized-implies-approved cascade rule.
    """
    if class_ == "investigational":
        return bucket == "investigational"
    if class_ == "approved":
        return bucket in {"approved", "commercialized"}
    if class_ == "commercialized":
        return bucket == "commercialized"
    raise ValueError(f"Unknown class: {class_!r}")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _class_metrics(pairs: list[tuple[str, str]], class_: str) -> ClassMetrics:
    """Compute binary one-vs-rest metrics for a single class over all pairs."""
    tp = fp = fn = tn = 0
    for true_bucket, pred_bucket in pairs:
        t = _is_positive(true_bucket, class_)
        p = _is_positive(pred_bucket, class_)
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    support = tp + fn
    return ClassMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        support=support,
    )


# ---------------------------------------------------------------------------
# Public API — data loading
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> tuple[dict[tuple[str, str], str], int]:
    """
    Read the benchmark CSV and return:
        ({(drug_norm, indication_norm): bucket}, skipped_count)

    Rows where either commercialization_label or approval_label is
    "unknown" or "ambiguous" are excluded (counted in skipped_count).

    Bucket priority:
        commercialization_label == "positive"           → "commercialized"
        approval_label == "positive" (comm not positive) → "approved"
        both negative                                    → "investigational"
    """
    benchmark: dict[tuple[str, str], str] = {}
    skipped = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            comm = row.get("commercialization_label", "").strip().lower()
            appr = row.get("approval_label", "").strip().lower()

            if comm in _SKIP_LABELS or appr in _SKIP_LABELS:
                skipped += 1
                continue

            if comm == "positive":
                bucket = "commercialized"
            elif appr == "positive":
                bucket = "approved"
            else:
                bucket = "investigational"

            key = (_norm(row["drug_name"]), _norm(row["indication"]))
            benchmark[key] = bucket

    return benchmark, skipped


def load_benchmark_binary_labels(path: str) -> tuple[dict[tuple[str, str], dict[str, str]], int]:
    """
    Load benchmark rows with raw binary labels for approval/commercialization.
    Rows with unknown/ambiguous labels in either field are skipped.
    Returns:
        ({(drug_norm, indication_norm): row_info}, skipped_count)
    """
    rows: dict[tuple[str, str], dict[str, str]] = {}
    skipped = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            comm = row.get("commercialization_label", "").strip().lower()
            appr = row.get("approval_label", "").strip().lower()

            if comm in _SKIP_LABELS or appr in _SKIP_LABELS:
                skipped += 1
                continue

            key = (_norm(row["drug_name"]), _norm(row["indication"]))
            rows[key] = {
                "drug_name": row["drug_name"],
                "indication": row["indication"],
                "approval_label": appr,
                "commercialization_label": comm,
            }

    return rows, skipped


def load_predictions_csv(path: str) -> dict[tuple[str, str], str]:
    """
    Read a predictions CSV with drug_name, indication, outcome columns.
    Returns {(drug_norm, indication_norm): bucket} using OUTCOME_TO_CLASS.
    Rows with an unrecognized outcome are mapped to "investigational".
    """
    predictions: dict[tuple[str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            outcome = row.get("outcome", "").strip()
            bucket = OUTCOME_TO_CLASS.get(outcome, "investigational")
            key = (_norm(row["drug_name"]), _norm(row["indication"]))
            predictions[key] = bucket
    return predictions


def predictions_from_pipeline(candidate_table, outcome_table) -> dict[tuple[str, str], str]:
    """
    Build a predictions dict from live pipeline objects.
    Duck-typed — does not import from pipeline/ to preserve module isolation.

    candidate_table:  has .candidates list, each with .candidate_id, .drug_name, .indication
    outcome_table:    has .outcomes dict[candidate_id → record], each record has .outcome.value
    """
    id_to_candidate = {c.candidate_id: c for c in candidate_table.candidates}
    predictions: dict[tuple[str, str], str] = {}
    for cand_id, record in outcome_table.outcomes.items():
        cand = id_to_candidate.get(cand_id)
        if cand is None:
            continue
        outcome_value = record.outcome.value if hasattr(record.outcome, "value") else str(record.outcome)
        bucket = OUTCOME_TO_CLASS.get(outcome_value, "investigational")
        key = (_norm(cand.drug_name), _norm(cand.indication))
        predictions[key] = bucket
    return predictions


# ---------------------------------------------------------------------------
# Public API — evaluation
# ---------------------------------------------------------------------------

def evaluate(
    predictions: dict[tuple[str, str], str],
    benchmark: dict[tuple[str, str], str],
    skipped: int,
) -> ValidationReport:
    """
    Compute per-class binary metrics and overall metrics.

    Only benchmark rows that have a matching prediction are evaluated.
    Benchmark rows with no matching prediction are counted as unmatched
    (not penalized in metric calculations).
    """
    pairs: list[tuple[str, str]] = []  # (true_bucket, pred_bucket)
    unmatched = 0

    for key, true_bucket in benchmark.items():
        if key not in predictions:
            unmatched += 1
            continue
        pairs.append((true_bucket, predictions[key]))

    matched = len(pairs)

    class_metrics: dict[str, ClassMetrics] = {}
    for class_ in CLASSES:
        class_metrics[class_] = _class_metrics(pairs, class_)

    # 3-class exact-match accuracy
    correct = sum(1 for t, p in pairs if t == p)
    overall_accuracy = _safe_div(correct, matched)

    # Macro averages over per-class metrics
    macro_precision = _safe_div(sum(m.precision for m in class_metrics.values()), len(CLASSES))
    macro_recall = _safe_div(sum(m.recall for m in class_metrics.values()), len(CLASSES))
    macro_f1 = _safe_div(sum(m.f1 for m in class_metrics.values()), len(CLASSES))
    macro_accuracy = _safe_div(sum(m.accuracy for m in class_metrics.values()), len(CLASSES))

    return ValidationReport(
        class_metrics=class_metrics,
        overall_accuracy=overall_accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        macro_accuracy=macro_accuracy,
        matched=matched,
        skipped=skipped,
        unmatched=unmatched,
    )


# ---------------------------------------------------------------------------
# Public API — display
# ---------------------------------------------------------------------------

_W = 70
_COL = 18


def _prediction_binary_label(bucket: str, task: str) -> str:
    """Map a 3-class bucket to a binary label for the given task."""
    if task == "approval":
        return "positive" if bucket in {"approved", "commercialized"} else "negative"
    if task == "commercialization":
        return "positive" if bucket == "commercialized" else "negative"
    raise ValueError(f"Unknown task: {task!r}")


def print_error_cases(
    predictions: dict[tuple[str, str], str],
    benchmark_rows: dict[tuple[str, str], dict[str, str]],
) -> None:
    """
    Print false positives and false negatives for approval and commercialization.
    Uses the same matching keys and approved<-commercialized cascade as evaluation.
    """
    for task in ("approval", "commercialization"):
        label_field = f"{task}_label"
        fps: list[dict[str, str]] = []
        fns: list[dict[str, str]] = []
        matched = 0

        for key, row in benchmark_rows.items():
            pred_bucket = predictions.get(key)
            if pred_bucket is None:
                continue
            matched += 1

            pred = _prediction_binary_label(pred_bucket, task)
            actual = row[label_field]
            entry = {
                "drug_name": row["drug_name"],
                "indication": row["indication"],
                "predicted": pred,
                "actual": actual,
            }

            if pred == "positive" and actual == "negative":
                fps.append(entry)
            elif pred == "negative" and actual == "positive":
                fns.append(entry)

        title = task.capitalize()
        print(f"{title} Error Cases ({matched} matched rows)")
        print(f"  False Positives ({len(fps)})")
        if not fps:
            print("    (none)")
        else:
            for e in fps:
                print(
                    f"    - Drug: {e['drug_name']} | Indication: {e['indication']} | "
                    f"Predicted: {e['predicted']} | Actual: {e['actual']}"
                )

        print(f"  False Negatives ({len(fns)})")
        if not fns:
            print("    (none)")
        else:
            for e in fns:
                print(
                    f"    - Drug: {e['drug_name']} | Indication: {e['indication']} | "
                    f"Predicted: {e['predicted']} | Actual: {e['actual']}"
                )
        print()


def print_report(report: ValidationReport) -> None:
    """Print a formatted classification metrics table to stdout."""
    hdr = (
        f"  Outcome Validation  —  {report.matched} matched  |  "
        f"{report.skipped} skipped  |  {report.unmatched} unmatched"
    )
    note = "  (binary one-vs-rest per class; commercialized \u2286 approved)"

    print("\n" + "═" * _W)
    print(hdr)
    print(note)
    print("═" * _W)
    print()

    header = f"  {'Class':<{_COL}}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Bin.Acc':>8}  {'Support':>7}"
    sep = "  " + "─" * (_W - 2)
    print(header)
    print(sep)

    for class_ in CLASSES:
        m = report.class_metrics[class_]
        print(
            f"  {class_:<{_COL}}  {m.precision:>9.3f}  {m.recall:>7.3f}"
            f"  {m.f1:>7.3f}  {m.accuracy:>8.3f}  {m.support:>7}"
        )

    print(sep)
    print(
        f"  {'macro avg':<{_COL}}  {report.macro_precision:>9.3f}  {report.macro_recall:>7.3f}"
        f"  {report.macro_f1:>7.3f}  {report.macro_accuracy:>8.3f}  {'—':>7}"
    )
    print(
        f"  {'3-class accuracy':<{_COL}}  {'':>9}  {'':>7}"
        f"  {'':>7}  {report.overall_accuracy:>8.3f}  {report.matched:>7}"
    )

    print()
    print("═" * _W + "\n")
