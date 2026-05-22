"""Reactome pathway hierarchy — parent/child DAG with depth and roots.

Builds a derived table from `ReactomePathways.txt` and
`ReactomePathwaysRelation.txt` (both downloaded into the same directory
that holds `UniProt2Reactome_All_Levels.txt`).

The resulting `pathway_hierarchy.parquet` has one row per Homo sapiens
pathway with:

    pathway_id            str
    pathway_name          str
    parent_ids            list[str]   (direct parents)
    child_ids             list[str]   (direct children)
    depth_from_top        int         (shortest path to any root)
    is_leaf_global        bool
    top_level_pathway_ids list[str]   (root ancestors; DAG -> can be many)
    n_descendants         int

Reactome's pathway relation is a DAG, not a strict tree — a child can
have multiple parents. We therefore record `top_level_pathway_ids` as a
set rather than a single string.

CLI:

    cd src
    python -m pipeline.enrichment.reactome_hierarchy \
        --data-dir ../data/reactome
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_HSA_PREFIX = "R-HSA-"
_HIERARCHY_FILENAME = "pathway_hierarchy.parquet"
_PATHWAYS_FILENAME = "ReactomePathways.txt"
_RELATIONS_FILENAME = "ReactomePathwaysRelation.txt"


def _read_pathways(path: Path) -> dict[str, str]:
    """Parse `ReactomePathways.txt`, return {pathway_id: pathway_name}
    for Homo sapiens rows only."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[2] != "Homo sapiens":
                continue
            pid, pname = parts[0].strip(), parts[1].strip()
            if pid.startswith(_HSA_PREFIX) and pid not in out:
                out[pid] = pname
    return out


def _read_relations(path: Path) -> list[tuple[str, str]]:
    """Parse `ReactomePathwaysRelation.txt`, return HSA-only edges."""
    edges: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            parent, child = parts[0].strip(), parts[1].strip()
            if parent.startswith(_HSA_PREFIX) and child.startswith(_HSA_PREFIX):
                edges.append((parent, child))
    return edges


def _build_dataframe(
    pathways: dict[str, str],
    edges: list[tuple[str, str]],
) -> pd.DataFrame:
    parents: dict[str, set[str]] = {pid: set() for pid in pathways}
    children: dict[str, set[str]] = {pid: set() for pid in pathways}
    for parent, child in edges:
        if parent in pathways and child in pathways:
            children[parent].add(child)
            parents[child].add(parent)

    roots = sorted(pid for pid, ps in parents.items() if not ps)

    # depth_from_top: BFS downward from all roots simultaneously.
    depth: dict[str, int] = {r: 0 for r in roots}
    queue: deque[str] = deque(roots)
    while queue:
        u = queue.popleft()
        for c in children[u]:
            d = depth[u] + 1
            if c not in depth or d < depth[c]:
                depth[c] = d
                queue.append(c)
    # Any pathway unreachable from a root is malformed; assign -1 so it
    # surfaces in audits rather than silently becoming 0.
    for pid in pathways:
        depth.setdefault(pid, -1)

    # top_level_pathway_ids: walk up parents until we reach a root.
    def _roots_of(pid: str) -> list[str]:
        seen: set[str] = set()
        stack = [pid]
        out: set[str] = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if not parents[n]:
                out.add(n)
            else:
                stack.extend(parents[n])
        return sorted(out)

    top_levels = {pid: _roots_of(pid) for pid in pathways}

    # n_descendants: transitive closure size downward.
    def _n_desc(pid: str) -> int:
        seen: set[str] = set()
        stack = list(children[pid])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(children[n])
        return len(seen)

    rows: list[dict[str, Any]] = []
    for pid in sorted(pathways):
        rows.append({
            "pathway_id": pid,
            "pathway_name": pathways[pid],
            "parent_ids": sorted(parents[pid]),
            "child_ids": sorted(children[pid]),
            "depth_from_top": int(depth[pid]),
            "is_leaf_global": len(children[pid]) == 0,
            "top_level_pathway_ids": top_levels[pid],
            "n_descendants": _n_desc(pid),
        })
    return pd.DataFrame(rows)


def _is_stale(parquet: Path, sources: list[Path]) -> bool:
    if not parquet.exists():
        return True
    pq_mtime = parquet.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > pq_mtime for s in sources)


def build_hierarchy(data_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Build (or load cached) Homo sapiens pathway hierarchy table.

    Rebuilds when the parquet is missing, older than either source file,
    or `force=True`.
    """
    data_dir = Path(data_dir)
    parquet = data_dir / _HIERARCHY_FILENAME
    pathways_file = data_dir / _PATHWAYS_FILENAME
    relations_file = data_dir / _RELATIONS_FILENAME

    if not pathways_file.exists():
        raise FileNotFoundError(f"missing {pathways_file}")
    if not relations_file.exists():
        raise FileNotFoundError(
            f"missing {relations_file} — run data/reactome/download.sh"
        )

    if not force and not _is_stale(parquet, [pathways_file, relations_file]):
        logger.info("reactome_hierarchy: loading cached %s", parquet)
        return pd.read_parquet(parquet)

    logger.info("reactome_hierarchy: rebuilding from source files")
    pathways = _read_pathways(pathways_file)
    edges = _read_relations(relations_file)
    df = _build_dataframe(pathways, edges)
    df.to_parquet(parquet, index=False)
    logger.info(
        "reactome_hierarchy: wrote %d rows -> %s (roots=%d, leaves=%d)",
        len(df),
        parquet,
        int((df["depth_from_top"] == 0).sum()),
        int(df["is_leaf_global"].sum()),
    )
    return df


def load_hierarchy(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Return `{pathway_id: {depth, parents, children, top_level_ids,
    is_leaf, name}}` lookup for downstream consumers."""
    df = build_hierarchy(Path(data_dir))
    out: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        out[row.pathway_id] = {
            "name": row.pathway_name,
            "depth": int(row.depth_from_top),
            "parents": list(row.parent_ids),
            "children": list(row.child_ids),
            "top_level_ids": list(row.top_level_pathway_ids),
            "is_leaf": bool(row.is_leaf_global),
        }
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing ReactomePathways.txt and "
        "ReactomePathwaysRelation.txt (the same dir as the UniProt file).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the cached parquet is up to date.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_hierarchy(args.data_dir, force=args.force)
    print(df.head())
    print(f"rows={len(df)} roots={(df['depth_from_top'] == 0).sum()} "
          f"leaves={df['is_leaf_global'].sum()}")


if __name__ == "__main__":
    _cli()


__all__ = ["build_hierarchy", "load_hierarchy"]
