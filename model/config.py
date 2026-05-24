"""Dataclass configs for the modeling pipeline.

Single source of truth for paths, label policy, feature toggles, model
choice, and ablation knobs. Every entrypoint (`train`, `ablate`) builds a
`ModelingConfig` from CLI args and threads it through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_DETAIL = PROJECT_ROOT / "outputs" / "candidate_detail.parquet"
DEFAULT_TRIAL_DETAIL = PROJECT_ROOT / "outputs" / "trial_detail.parquet"
DEFAULT_FINGERPRINTS = PROJECT_ROOT / "outputs" / "features" / "fingerprints.parquet"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "outputs" / "features" / "molformer_embeddings.parquet"

ALL_FEATURE_GROUPS: tuple[str, ...] = (
    #"tanimoto_nn",
    #"molformer_nn",
    "embeddings",
    "targets",
    "admet",
    "pathway",
    "disease",
    #"action_type",
    #"moa",
)


@dataclass
class LabelConfig:
    """Binary label derived from `candidate_detail.outcome`.

    Default: positive = {Approved, Commercialized}, negative = the three
    failure phases. Rows with outcome in `exclude_outcomes` are dropped.
    """

    positive: tuple[str, ...] = ("Approved", "Commercialized")
    negative: tuple[str, ...] = ("Failed Phase 1", "Failed Phase 2", "Failed Phase 3")
    exclude_outcomes: tuple[str, ...] = ("Unknown", "Ongoing")


@dataclass
class FeatureConfig:
    """Per-group toggles + per-group hyperparams (top-K cardinalities)."""

    enabled: tuple[str, ...] = ALL_FEATURE_GROUPS
    top_k_targets: int = 200
    top_k_pathways: int = 500
    top_k_mesh: int = 200
    top_k_moa: int = 200
    admet_drop_null_threshold: float = 0.95
    admet_indicator_threshold: float = 0.05


@dataclass
class ModelingConfig:
    """Top-level config for a single training run."""

    candidate_detail_path: Path = DEFAULT_CANDIDATE_DETAIL
    trial_detail_path: Path = DEFAULT_TRIAL_DETAIL
    fingerprints_path: Path = DEFAULT_FINGERPRINTS
    embeddings_path: Path = DEFAULT_EMBEDDINGS

    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # "drug_indication" → one row per candidate, y from candidate.outcome.
    # "trial" → one row per NCT, y from trial_detail.trial_inferred_label.
    training_granularity: str = "drug_indication"

    model_name: str = "xgb"
    model_kwargs: dict = field(default_factory=dict)

    test_size: float = 0.2
    inner_val_size: float = 0.1
    seed: int = 0
    group_by: Optional[str] = None  # None or "drug_name"

    # Temporal split: train on rows whose `time_split_column` year is
    # <= time_split_year, test on rows whose year is > it. When
    # `time_split_year` is set, `test_size` and `group_by` are ignored.
    time_split_column: str = "earliest_start_date"
    time_split_year: Optional[int] = None

    # When set, carves out a calibration slice as a three-way time slice:
    # train = year <= calibration_year-1, calibrate = year == calibration_year,
    # test = year > calibration_year. The calibrator is fit on the calibrate
    # slice after the base model is trained on the train slice.
    calibration_year: Optional[int] = None
    calibration_method: str = "isotonic"  # "isotonic" | "sigmoid"

    output_dir: Optional[Path] = None


@dataclass
class KillerFigureConfig:
    """Hyperparameters for the killer-figure NN analog lookup.

    `weights` is the per-component contribution to `joint_sim`; on a
    query where a component is missing for either side, weights are
    re-normalized over the components that *are* present. `min_neighbors`
    is the unique-drug count below which retrieval falls back to the
    stratum rate and flags `insufficient_prior_art`.
    """

    k: int = 10
    weights: tuple[tuple[str, float], ...] = (
        ("molecule", 1.0 / 3.0),
        ("target", 1.0 / 3.0),
        ("indication", 1.0 / 3.0),
    )
    min_neighbors: int = 5


@dataclass
class AblationConfig:
    """Ablation harness config — wraps a base ModelingConfig.

    `mode` controls which subsets of `features.enabled` get evaluated.
    `custom_subsets` is consulted only when mode == "custom".
    """

    base: ModelingConfig = field(default_factory=ModelingConfig)
    mode: str = "loo"  # "all" | "loo" | "single" | "custom"
    custom_subsets: dict[str, tuple[str, ...]] = field(default_factory=dict)
