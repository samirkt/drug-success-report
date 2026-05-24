"""argparse entrypoint: `python -m model train` / `python -m model ablate`."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_HINT_SCRIPT = PROJECT_ROOT / "run_hint.sh"

from .ablate import run_ablation
from .artifacts import save_run
from .baselines import ALL_BASELINES
from .baselines.runner import BaselinesConfig, run_baselines
from .config import (
    ALL_FEATURE_GROUPS,
    AblationConfig,
    FeatureConfig,
    LabelConfig,
    ModelingConfig,
    DEFAULT_CANDIDATE_DETAIL,
    DEFAULT_EMBEDDINGS,
    DEFAULT_FINGERPRINTS,
    DEFAULT_TRIAL_DETAIL,
)
from .train import train_one_run


def _parse_csv(s: str | None) -> tuple[str, ...]:
    if not s:
        return ()
    return tuple(p.strip() for p in s.split(",") if p.strip())


def _parse_kwargs(s: str | None) -> dict:
    if not s:
        return {}
    out: dict = {}
    for kv in s.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Coerce simple types.
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            else:
                out[k] = v
    return out


_DEFAULT_TIME_SPLIT_COLUMN = "earliest_start_date"
_TRIAL_TIME_SPLIT_COLUMN = "trial_start_date"


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--candidate-detail", type=Path, default=DEFAULT_CANDIDATE_DETAIL)
    p.add_argument("--trial-detail", type=Path, default=DEFAULT_TRIAL_DETAIL)
    p.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--group-by",
        default=None,
        help="Column to keep disjoint between train/test (e.g. drug_name). "
             "Ignored when --time-split-year is set.",
    )
    p.add_argument(
        "--time-split-year",
        type=int,
        default=None,
        help="Temporal split: train on rows whose --time-split-column year "
             "is <= this, test on rows with year > this. Disables --test-size "
             "and --group-by when set.",
    )
    p.add_argument(
        "--time-split-column",
        default=_DEFAULT_TIME_SPLIT_COLUMN,
        help=(
            f"Column used for the temporal split (default: {_DEFAULT_TIME_SPLIT_COLUMN} "
            f"for drug-indication mode; auto-switched to {_TRIAL_TIME_SPLIT_COLUMN} for "
            f"trial mode unless explicitly overridden)."
        ),
    )
    p.add_argument(
        "--calibration-year",
        type=int,
        default=None,
        help="If set, fits a probability calibrator on rows in this year and uses "
             "a three-way time slice: train <= Y-1, calibrate == Y, test > Y. "
             "Takes precedence over --time-split-year.",
    )
    p.add_argument(
        "--calibration-method",
        default="isotonic",
        choices=("isotonic", "sigmoid"),
        help="Calibration method when --calibration-year is set (default: isotonic).",
    )
    p.add_argument(
        "--groups",
        default=",".join(ALL_FEATURE_GROUPS),
        help=f"Comma-separated feature groups. Available: {','.join(ALL_FEATURE_GROUPS)}",
    )
    p.add_argument("--top-k-targets", type=int, default=200)
    p.add_argument("--top-k-pathways", type=int, default=500)
    p.add_argument("--top-k-mesh", type=int, default=200)
    p.add_argument("--top-k-moa", type=int, default=200)
    p.add_argument(
        "--label-positive",
        default="Approved,Commercialized",
        help="Outcomes mapping to class 1.",
    )
    p.add_argument(
        "--label-negative",
        default="Failed Phase 1,Failed Phase 2,Failed Phase 3",
        help="Outcomes mapping to class 0.",
    )
    p.add_argument(
        "--exclude-outcomes",
        default="Unknown,Ongoing",
        help="Outcomes to drop entirely.",
    )
    p.add_argument("--model", default="xgb", help="Model registry name (default: xgb).")
    p.add_argument(
        "--model-kwargs",
        default=None,
        help="Comma-separated k=v overrides forwarded to the model constructor.",
    )
    p.add_argument("--output", type=Path, required=True)


def _build_config(
    args: argparse.Namespace,
    *,
    granularity: str = "drug_indication",
) -> ModelingConfig:
    label = LabelConfig(
        positive=_parse_csv(args.label_positive),
        negative=_parse_csv(args.label_negative),
        exclude_outcomes=_parse_csv(args.exclude_outcomes),
    )
    enabled = _parse_csv(args.groups)
    for g in enabled:
        if g not in ALL_FEATURE_GROUPS:
            raise SystemExit(
                f"unknown feature group {g!r}; choose from {ALL_FEATURE_GROUPS}"
            )
    features = FeatureConfig(
        enabled=enabled,
        top_k_targets=args.top_k_targets,
        top_k_pathways=args.top_k_pathways,
        top_k_mesh=args.top_k_mesh,
        top_k_moa=args.top_k_moa,
    )
    time_split_year = args.time_split_year
    if args.calibration_year is not None and time_split_year is not None:
        import warnings as _w
        _w.warn(
            "--calibration-year takes precedence over --time-split-year; the latter will be ignored",
            stacklevel=2,
        )
        time_split_year = None

    # Trial-mode defaults: the candidate-level columns the drug_indication
    # defaults assume (`earliest_start_date`, no group_by) don't make sense
    # on the trial frame. Auto-switch unless the user explicitly overrode.
    time_split_column = args.time_split_column
    group_by = args.group_by
    if granularity == "trial":
        if time_split_column == _DEFAULT_TIME_SPLIT_COLUMN:
            time_split_column = _TRIAL_TIME_SPLIT_COLUMN
        if group_by is None:
            group_by = "candidate_id"

    return ModelingConfig(
        candidate_detail_path=args.candidate_detail,
        trial_detail_path=args.trial_detail,
        fingerprints_path=args.fingerprints,
        embeddings_path=args.embeddings,
        label=label,
        features=features,
        training_granularity=granularity,
        model_name=args.model,
        model_kwargs=_parse_kwargs(args.model_kwargs),
        test_size=args.test_size,
        seed=args.seed,
        group_by=group_by,
        time_split_column=time_split_column,
        time_split_year=time_split_year,
        calibration_year=args.calibration_year,
        calibration_method=args.calibration_method,
        output_dir=args.output,
    )


def _invoke_run_hint(csv_path: Path, *, prefix: str = "") -> None:
    """Run `run_hint.sh <input.csv> <output.csv>` and stream its stdout/stderr through.

    Output lands at `<input_dir>/hint_results.csv` to match the path the
    consolidated report script picks up via `--hint-metrics`.

    Failures (missing script, nonzero exit) are logged as warnings, not
    raised — the trainer run is already complete and the CSV is on disk,
    so the user can replay HINT manually.
    """
    if not RUN_HINT_SCRIPT.exists():
        logger.warning(
            "run_hint.sh not found at %s — skipping HINT eval (CSV is at %s)",
            RUN_HINT_SCRIPT,
            csv_path,
        )
        return
    out_path = csv_path.parent / "hint_results.csv"
    print(f"\n{prefix}invoking HINT: {RUN_HINT_SCRIPT} {csv_path} {out_path}")
    # Flush so our prints land before the subprocess's stdout when piped.
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run([str(RUN_HINT_SCRIPT), str(csv_path), str(out_path)], check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "run_hint.sh exited with code %d — HINT eval failed (CSV is at %s)",
            exc.returncode,
            csv_path,
        )
    except OSError as exc:
        logger.warning(
            "run_hint.sh could not be executed (%s) — HINT eval skipped (CSV is at %s)",
            exc,
            csv_path,
        )


def _cmd_train(args: argparse.Namespace) -> None:
    if args.training_granularity == "both":
        granularities = ("drug_indication", "trial")
    else:
        granularities = (args.training_granularity,)

    out_root = Path(args.output)
    multi = len(granularities) > 1
    for g in granularities:
        config = _build_config(args, granularity=g)
        out_dir = out_root / g if multi else out_root
        config = replace(config, output_dir=out_dir)
        result = train_one_run(config)
        save_run(result, out_dir)
        prefix = f"[{g}] " if multi else ""
        print(
            f"\n{prefix}ROC-AUC={result.metrics['roc_auc']:.4f} "
            f"PR-AUC={result.metrics['pr_auc']:.4f} "
            f"F1={result.metrics['f1']:.4f} "
            f"Brier={result.metrics['brier']:.4f}"
        )
        for label, mp in (result.per_phase_metrics or {}).items():
            n = mp.get("n", 0)
            if n == 0:
                print(f"{prefix}  {label}: n=0 (no test rows at this phase)")
                continue
            print(
                f"{prefix}  {label}: n={n} pos={mp.get('n_pos', 0)} "
                f"ROC-AUC={mp.get('roc_auc', float('nan')):.4f} "
                f"PR-AUC={mp.get('pr_auc', float('nan')):.4f} "
                f"F1={mp.get('f1', float('nan')):.4f} "
                f"Brier={mp.get('brier', float('nan')):.4f}"
            )

        # Trial mode side-step: hand the same test rows to HINT.
        if g == "trial" and result.hint_test_df is not None:
            hint_csv = (out_dir / "hint_test.csv").resolve()
            if args.skip_hint:
                print(f"{prefix}HINT eval skipped (--skip-hint); CSV at {hint_csv}")
            elif hint_csv.exists():
                _invoke_run_hint(hint_csv, prefix=prefix)


def _cmd_baselines(args: argparse.Namespace) -> None:
    config = _build_config(args)
    selected = _parse_csv(args.baselines)
    if not selected:
        selected = ALL_BASELINES
    for name in selected:
        if name not in ALL_BASELINES:
            raise SystemExit(
                f"unknown baseline {name!r}; choose from {ALL_BASELINES}"
            )
    bl_cfg = BaselinesConfig(base=config, baselines=tuple(selected))
    result = run_baselines(bl_cfg)
    print(result.summary.to_string(index=False))


def _cmd_killer_figure(args: argparse.Namespace) -> None:
    """`model killer-figure` — single-candidate NN analog lookup."""
    from . import data as data_mod
    from .killer_figure import (
        KillerFigureRetriever,
        build_adhoc_query,
        format_text_summary,
    )
    from .baselines.killer_figure import _json_default

    config = _build_config(args)
    df = data_mod.build_modeling_frame(config)

    if args.candidate_id and args.smiles:
        raise SystemExit("--candidate-id and --smiles are mutually exclusive")
    if not args.candidate_id and not (args.smiles or args.targets or args.icd10):
        raise SystemExit(
            "killer-figure: must provide either --candidate-id or at least one of "
            "--smiles / --targets / --icd10 for an ad-hoc query"
        )

    if args.candidate_id:
        match = df[df["candidate_id"] == args.candidate_id]
        if match.empty:
            raise SystemExit(
                f"--candidate-id {args.candidate_id!r} not found in modeling frame "
                f"({len(df)} rows). Check the id or relax the label filter."
            )
        query_row = match.iloc[0].to_dict()
        # Pool = entire labeled df; date cutoff strictly excludes the query.
        pool_df = df
    else:
        targets = _parse_csv(args.targets)
        icd10 = _parse_csv(args.icd10)
        mesh = _parse_csv(args.mesh)
        start_date = _parse_iso_date(args.start_date) if args.start_date else date.today()
        query_row = build_adhoc_query(
            smiles=args.smiles,
            targets=list(targets) if targets else None,
            icd10=list(icd10) if icd10 else None,
            mesh_tree_numbers=list(mesh) if mesh else None,
            disease_area=args.disease_area,
            start_date=start_date,
        )
        pool_df = df

    retriever = KillerFigureRetriever(k=args.k, min_neighbors=args.min_neighbors)
    retriever.fit(pool_df, pool_df["y"].values.astype(int))
    report = retriever.retrieve(query_row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2, default=_json_default))
    print(format_text_summary(report))
    print(f"\nReport written to {out_path}")


def _parse_iso_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--start-date {s!r}: {exc}")


def _cmd_ablate(args: argparse.Namespace) -> None:
    config = _build_config(args)
    custom: dict[str, tuple[str, ...]] = {}
    for entry in args.subset or []:
        if "=" not in entry:
            raise SystemExit(f"--subset expects name=g1,g2,... ; got {entry!r}")
        name, groups = entry.split("=", 1)
        custom[name.strip()] = _parse_csv(groups)
    abl = AblationConfig(base=config, mode=args.mode, custom_subsets=custom)
    result = run_ablation(abl)
    print(result.summary.to_string(index=False))


def _cmd_rfe(args: argparse.Namespace) -> None:
    from .rfe import RFEConfig, run_group_rfe, save_rfe

    config = _build_config(args)
    rfe_cfg = RFEConfig(
        base=config,
        metric=args.rfe_metric,
        cv=args.rfe_cv,
        step=args.rfe_step,
        seed=args.seed,
    )
    summary, final_result = run_group_rfe(rfe_cfg)
    save_rfe(summary, final_result, Path(args.output))

    rows = [
        {
            "iter": s.iteration,
            "n_groups": len(s.groups_remaining),
            "groups_remaining": ", ".join(s.groups_remaining),
            "group_dropped": s.group_dropped or "—",
            "dropped_importance": round(s.dropped_importance, 6),
            args.rfe_metric: round(s.metric_value, 4),
            f"{args.rfe_metric}_std": round(s.metric_value_std, 4),
        }
        for s in summary.history
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"\noptimal subset (iter={summary.optimal_iteration}): "
        f"{list(summary.optimal_groups)}  {args.rfe_metric}={summary.optimal_metric:.4f}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a single model and write artifacts.")
    _add_common_args(p_train)
    p_train.add_argument(
        "--training-granularity",
        default="both",
        choices=("drug_indication", "trial", "both"),
        help=(
            "drug_indication = one row per candidate, y from candidate.outcome "
            "(the original mode). trial = one row per NCT, y from "
            "trial_inferred_label (drops trials with a null label). both = run "
            "each in turn and write artifacts under per-mode subdirs. "
            "Default: both."
        ),
    )
    p_train.add_argument(
        "--skip-hint",
        action="store_true",
        help=(
            "Trial mode only. Still write hint_test.csv but do not invoke "
            "run_hint.sh on it. Useful for quick iterations where you only "
            "want to inspect our model's metrics."
        ),
    )
    p_train.set_defaults(func=_cmd_train)

    p_abl = sub.add_parser("ablate", help="Run ablation across feature subsets.")
    _add_common_args(p_abl)
    p_abl.add_argument(
        "--mode",
        default="loo",
        choices=("all", "loo", "single", "custom"),
        help="Ablation mode (default: loo).",
    )
    p_abl.add_argument(
        "--subset",
        action="append",
        help="For --mode=custom: name=g1,g2,... ; pass repeatedly.",
    )
    p_abl.set_defaults(func=_cmd_ablate)

    p_rfe = sub.add_parser(
        "rfe",
        help="Group-level recursive feature elimination across enabled feature groups.",
    )
    _add_common_args(p_rfe)
    p_rfe.add_argument(
        "--rfe-metric",
        default="roc_auc",
        choices=("roc_auc", "pr_auc", "f1", "brier", "log_loss"),
        help="Metric used to score each RFE iteration (default: roc_auc).",
    )
    p_rfe.add_argument(
        "--rfe-cv",
        type=int,
        default=0,
        help="0 = single train/test split using the base config; "
             ">=2 = stratified K-fold over the train portion.",
    )
    p_rfe.add_argument(
        "--rfe-step",
        type=int,
        default=1,
        help="Number of groups removed per iteration (default: 1).",
    )
    p_rfe.set_defaults(func=_cmd_rfe)

    p_bl = sub.add_parser(
        "baselines",
        help="Run baseline models for 1:1 comparison against the full model.",
    )
    _add_common_args(p_bl)
    p_bl.add_argument(
        "--baselines",
        default=",".join(ALL_BASELINES),
        help=f"Comma-separated baselines to run. Available: {','.join(ALL_BASELINES)}",
    )
    p_bl.set_defaults(func=_cmd_baselines)

    p_kf = sub.add_parser(
        "killer-figure",
        help="NN analog lookup: 'candidates like this one reached approval X% vs Y% stratum, because Z'.",
    )
    _add_common_args(p_kf)
    p_kf.add_argument("--k", type=int, default=10, help="Number of neighbors (default: 10).")
    p_kf.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum unique-drug neighbors required; below this falls back to stratum rate.",
    )
    p_kf.add_argument(
        "--candidate-id",
        default=None,
        help="Look up an existing candidate row by ID. Mutually exclusive with --smiles/--targets/--icd10.",
    )
    p_kf.add_argument(
        "--smiles", default=None, help="Ad-hoc query: candidate SMILES string."
    )
    p_kf.add_argument(
        "--targets",
        default=None,
        help="Ad-hoc query: comma-separated UniProt accessions (e.g. 'P51681,P52333').",
    )
    p_kf.add_argument(
        "--icd10",
        default=None,
        help="Ad-hoc query: comma-separated ICD-10 codes (e.g. 'C71.9,C71.0').",
    )
    p_kf.add_argument(
        "--mesh",
        default=None,
        help="Ad-hoc query: comma-separated MeSH tree numbers (e.g. 'C04.557,C04.588').",
    )
    p_kf.add_argument(
        "--disease-area",
        default=None,
        help="Ad-hoc query: free-text disease area string matching the project taxonomy.",
    )
    p_kf.add_argument(
        "--start-date",
        default=None,
        help="Ad-hoc query: ISO date YYYY-MM-DD; pool restricted to candidates with start < this. Defaults to today.",
    )
    p_kf.set_defaults(func=_cmd_killer_figure)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
