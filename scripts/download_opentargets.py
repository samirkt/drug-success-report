"""Download the OpenTargets Parquet datasets needed for the OT enrichment.

Pulls five subdirectories from EBI's public mirror into a local
directory, then confirms the expected Parquet files are present. Use
rsync when available (fast + resumable); fall back to curl over HTTPS
only if rsync is missing. This is **not** part of the pipeline — it
runs once per OT release.

The `associationByDatatypeDirect/` subdir is the largest of the five
(several GB at OT 25.x — it carries every target × disease × datatype
association). It is required for the genetic-evidence column on the
candidate parquet; if you don't need that feature, pass
``--skip associationByDatatypeDirect`` and the snapshot builder will
emit an empty target-disease evidence table.

Usage:

    python scripts/download_opentargets.py --release 25.03 \
        --dest data/opentargets

    # Then build the slim snapshot:
    python scripts/build_opentargets_snapshot.py \
        --opentargets-dir data/opentargets/25.03 \
        --chembl-snapshot data/chembl_targets.sqlite \
        --out data/opentargets_snapshot.sqlite

The script tries both the current dataset naming ("molecule",
"mechanismOfAction", "targets") and the older aliases ("drug",
"mechanismsOfAction", "target") — OpenTargets has renamed these between
releases. If none of the aliases resolve, the script prints the
directory tree it could see so you can pass the right name manually
with --override-path.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("download_opentargets")

# Rsync + HTTPS base URLs. As of OT 25.x the on-disk layout dropped the
# `etl/parquet/` intermediate prefix and datasets live directly under
# `output/`. Older releases (24.x and earlier) used
# `output/etl/parquet/<dataset>/`; we try the new layout first and fall
# back to the old one per-subdir, so one script works across releases.
_RSYNC_BASE_NEW = "rsync.ebi.ac.uk::pub/databases/opentargets/platform/{release}/output"
_RSYNC_BASE_OLD = "rsync.ebi.ac.uk::pub/databases/opentargets/platform/{release}/output/etl/parquet"
_HTTPS_BASE_NEW = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{release}/output"
_HTTPS_BASE_OLD = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{release}/output/etl/parquet"

# Per-dataset name aliases. OT renamed these between releases (25.x
# prefixed drug datasets with `drug_` and pluralized / unpluralized a
# few); we try each alias in order and stop at the first one that
# resolves non-empty. The aliases mirror the name-resolution logic in
# scripts/build_opentargets_snapshot.py.
_DATASETS: list[tuple[str, list[str]]] = [
    # (local dir name we want on disk, [source aliases to try in order])
    # New 25.x names first, older names as fallbacks.
    ("molecule",          ["drug_molecule", "molecule", "drug", "molecules"]),
    ("mechanismOfAction", ["drug_mechanism_of_action", "mechanismOfAction",
                           "mechanismsOfAction", "moa"]),
    ("indication",        ["drug_indication", "indication", "indications"]),
    ("targets",           ["target", "targets"]),
    # Target × disease genetic-association scores. Big (~GB) but the
    # only source for the genetic-evidence feature. Drop via `--skip
    # associationByDatatypeDirect` if you don't need it.
    ("associationByDatatypeDirect",
                          ["association_by_datatype_direct",
                           "associationByDatatypeDirect",
                           "associationByDatatypeIndirect",
                           "association_by_datatype_indirect"]),
]


def _check_rsync_subdir_exists(rsync_base: str, subdir: str) -> bool:
    """Return True when `rsync --list-only` can see the remote subdir."""
    url = rsync_base + f"/{subdir}/"
    try:
        proc = subprocess.run(
            ["rsync", "--list-only", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _rsync_pull(rsync_base: str, subdir: str, dest: Path) -> None:
    """Rsync a single subdir into ``dest`` (overwrites existing files)."""
    dest.mkdir(parents=True, exist_ok=True)
    url = rsync_base + f"/{subdir}/"
    logger.info("  rsync %s -> %s", url, dest)
    cmd = ["rsync", "-rvz", "--progress", "--delete", url, str(dest) + "/"]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rsync failed for {subdir} (exit {proc.returncode}). Re-run the "
            "command shown above in a terminal to see the error."
        )


def _curl_list_https(https_base: str, subdir: str) -> list[str]:
    """List Parquet file basenames at the HTTPS mirror. Returns [] on failure."""
    if not shutil.which("curl"):
        return []
    url = https_base + f"/{subdir}/"
    try:
        proc = subprocess.run(
            ["curl", "-sSL", "--fail", url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    # Scrape <a href="xxx.parquet"> out of the directory listing. The
    # EBI mirror serves a plain HTML directory index.
    names = re.findall(r'href="([^"]+\.parquet)"', proc.stdout.decode("utf-8", "replace"))
    return names


def _curl_pull(https_base: str, subdir: str, dest: Path) -> None:
    """Download every Parquet file in ``subdir`` via HTTPS into ``dest``."""
    names = _curl_list_https(https_base, subdir)
    if not names:
        raise RuntimeError(
            f"HTTPS fallback could not list {subdir} at {https_base}. "
            "Install rsync (brew install rsync) or check the release number "
            "and try again."
        )
    dest.mkdir(parents=True, exist_ok=True)
    base = https_base + f"/{subdir}/"
    logger.info(
        "  curl (HTTPS fallback): %d Parquet files -> %s", len(names), dest
    )
    for name in names:
        url = base + name
        out = dest / name
        cmd = ["curl", "-fSL", "-o", str(out), url]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"curl failed for {url} (exit {proc.returncode})."
            )


def _resolve_alias(
    release: str, aliases: list[str]
) -> tuple[str, str, str] | None:
    """Return (resolved_alias, rsync_base, https_base) or None.

    Tries the new (25.x) layout first, then the legacy etl/parquet layout.
    """
    rsync_new = _RSYNC_BASE_NEW.format(release=release)
    rsync_old = _RSYNC_BASE_OLD.format(release=release)
    https_new = _HTTPS_BASE_NEW.format(release=release)
    https_old = _HTTPS_BASE_OLD.format(release=release)

    for rsync_base, https_base in ((rsync_new, https_new), (rsync_old, https_old)):
        for name in aliases:
            if _check_rsync_subdir_exists(rsync_base, name):
                return name, rsync_base, https_base
        # Rsync check failed for this base — try HTTPS listing.
        for name in aliases:
            if _curl_list_https(https_base, name):
                return name, rsync_base, https_base
    return None


def download(
    release: str,
    dest: Path,
    dry_run: bool = False,
    skip: set[str] | None = None,
) -> None:
    """Download all OT datasets for a given release.

    Output layout:
        dest/<release>/molecule/
        dest/<release>/mechanismOfAction/
        dest/<release>/indication/
        dest/<release>/targets/
        dest/<release>/associationByDatatypeDirect/
    """
    release_dir = dest / release
    skip = skip or set()
    use_rsync = shutil.which("rsync") is not None
    if not use_rsync and not shutil.which("curl"):
        raise RuntimeError(
            "Neither rsync nor curl is available — install one of them "
            "(brew install rsync) and re-run."
        )
    if use_rsync:
        logger.info("Downloader: rsync (preferred)")
    else:
        logger.info("Downloader: curl (HTTPS fallback — slower)")

    total_gb_before = _dir_size_gb(release_dir)
    for local_name, aliases in _DATASETS:
        if local_name in skip:
            logger.info("Skipping %s (--skip)", local_name)
            continue
        logger.info("Resolving %s (aliases: %s)", local_name, aliases)
        resolution = _resolve_alias(release, aliases)
        if resolution is None:
            raise RuntimeError(
                f"None of the aliases {aliases} exist at release {release}. "
                f"Check {_HTTPS_BASE_NEW.format(release=release)}/ in a browser "
                "to see what names are actually published and rerun with "
                f"--override-path {local_name}=<actual_name>."
            )
        resolved, rsync_base, https_base = resolution
        logger.info("  resolved -> %s (base: %s)", resolved, https_base)
        target = release_dir / local_name
        if dry_run:
            logger.info("  [dry-run] would pull into %s", target)
            continue
        if use_rsync:
            _rsync_pull(rsync_base, resolved, target)
        else:
            _curl_pull(https_base, resolved, target)
        _sanity_check_parquet(target, local_name)

    total_gb_after = _dir_size_gb(release_dir)
    logger.info(
        "Done. %s grew by %.1f GB (final size: %.1f GB). Next step:",
        release_dir,
        total_gb_after - total_gb_before,
        total_gb_after,
    )
    logger.info(
        "  python scripts/build_opentargets_snapshot.py "
        "--opentargets-dir %s --chembl-snapshot <path> --out <path>",
        release_dir,
    )


def _dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 ** 3)


def _sanity_check_parquet(path: Path, label: str) -> None:
    """Ensure at least one .parquet file landed in ``path``."""
    if not path.exists() or not any(path.glob("*.parquet")):
        raise RuntimeError(
            f"{label}: no .parquet files found in {path} after download. "
            "The transfer may have failed mid-way; rerun the script."
        )


def _apply_overrides(
    overrides: list[str],
) -> None:
    """Apply `--override-path LOCAL=REMOTE` pairs to _DATASETS in place."""
    if not overrides:
        return
    by_local = {name: aliases for name, aliases in _DATASETS}
    for pair in overrides:
        if "=" not in pair:
            raise ValueError(
                f"--override-path expects LOCAL=REMOTE, got: {pair!r}"
            )
        local, remote = pair.split("=", 1)
        if local not in by_local:
            raise ValueError(
                f"--override-path local name must be one of "
                f"{list(by_local)}; got {local!r}"
            )
        by_local[local] = [remote]
    # Rewrite _DATASETS preserving the canonical ordering.
    _DATASETS[:] = [(name, by_local[name]) for name, _ in _DATASETS]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Download the OpenTargets Parquet datasets (molecule, "
            "mechanismOfAction, indication, targets, "
            "associationByDatatypeDirect) for a single release."
        )
    )
    parser.add_argument(
        "--release",
        required=True,
        help=(
            "OT release number, e.g. '25.03'. Browse "
            "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/ "
            "to see what's currently published."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/opentargets"),
        help=(
            "Parent directory that will contain <release>/molecule/, "
            "<release>/mechanismOfAction/, <release>/indication/, "
            "<release>/targets/, and "
            "<release>/associationByDatatypeDirect/. "
            "Default: data/opentargets"
        ),
    )
    parser.add_argument(
        "--override-path",
        action="append",
        default=[],
        metavar="LOCAL=REMOTE",
        help=(
            "Override a remote dataset name when OT renames it. E.g. "
            "--override-path molecule=drug. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve remote names and report sizes without downloading.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="LOCAL",
        help=(
            "Skip a local dataset name. Repeatable. Useful for the big "
            "associationByDatatypeDirect dataset when the genetic-evidence "
            "feature isn't needed."
        ),
    )
    args = parser.parse_args(argv)

    _apply_overrides(args.override_path)
    try:
        download(
            args.release,
            args.dest,
            dry_run=args.dry_run,
            skip=set(args.skip),
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
