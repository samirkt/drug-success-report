"""argparse entrypoint: `python -m model train` / `python -m model ablate`."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .ablate import run_ablation
from .artifacts import save_run
from .config import (
    ALL_FEATURE_GROUPS,
    AblationConfig,
    FeatureConfig,
    LabelConfig,
    ModelingConfig,
    DEFAULT_CANDIDATE_DETAIL,
    DEFAULT_EMBEDDINGS,
    DEFAULT_FINGERPRINTS,
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


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--candidate-detail", type=Path, default=DEFAULT_CANDIDATE_DETAIL)
    p.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--group-by",
        default=None,
        help="Column to keep disjoint between train/test (e.g. drug_name).",
    )
    p.add_argument(
        "--groups",
        default=",".join(ALL_FEATURE_GROUPS),
        help=f"Comma-separated feature groups. Available: {','.join(ALL_FEATURE_GROUPS)}",
    )
    p.add_argument("--top-k-targets", type=int, default=200)
    p.add_argument("--top-k-pathways", type=int, default=500)
    p.add_argument("--top-k-mesh", type=int, default=200)
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


def _build_config(args: argparse.Namespace) -> ModelingConfig:
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
    )
    return ModelingConfig(
        candidate_detail_path=args.candidate_detail,
        fingerprints_path=args.fingerprints,
        embeddings_path=args.embeddings,
        label=label,
        features=features,
        model_name=args.model,
        model_kwargs=_parse_kwargs(args.model_kwargs),
        test_size=args.test_size,
        seed=args.seed,
        group_by=args.group_by,
        output_dir=args.output,
    )


def _cmd_train(args: argparse.Namespace) -> None:
    config = _build_config(args)
    result = train_one_run(config)
    save_run(result, config.output_dir)
    print(
        f"\nROC-AUC={result.metrics['roc_auc']:.4f} "
        f"PR-AUC={result.metrics['pr_auc']:.4f} "
        f"F1={result.metrics['f1']:.4f} "
        f"Brier={result.metrics['brier']:.4f}"
    )


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a single model and write artifacts.")
    _add_common_args(p_train)
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
