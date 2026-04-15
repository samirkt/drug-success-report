"""
Build compact derivative CSVs from the full DrugBank XML.

Source:  docs/data/full database.xml  (1.8 GB, license-gated — do NOT commit)
Outputs: data/drugbank_synonyms.csv
         data/drugbank_products.csv

Parsing strategy
----------------
The XML is too large to load into memory. We use an iterparse-based streaming
scan that (a) only touches each <drug> element once, (b) clears the element
plus its preceding siblings after emitting, and (c) uses the namespaced tag
prefix `{http://www.drugbank.ca}`.

Outputs
-------
data/drugbank_synonyms.csv
    drugbank_id, synonym_norm, kind
    kind ∈ {primary_name, synonym, brand, international_brand, product_name}

data/drugbank_products.csv
    One row per DrugBank drug (not per product). We collapse the <products>
    block into a single summary row capturing the earliest US-approved
    marketing date and the associated FDA application number.

    drugbank_id, drug_name_norm, primary_name, appl_no, approved,
    approval_date, first_marketed_date, us_country_hit, groups

Normalization uses the existing `canonicalize_drug_name()` helper from
`pipeline.drugbank_norm` so downstream lookups compare apples-to-apples
against the candidate names already normalized by that module.

License
-------
The DrugBank full XML export is license-gated. The derived CSVs likely
inherit those restrictions. Keep them out of version control (see
`.gitignore`) and regenerate locally on demand.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Iterator
from xml.etree.ElementTree import Element, iterparse

# Allow importing the canonicalizer when the script is invoked directly.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent
sys.path.insert(0, str(_SRC))

from pipeline.drugbank_norm import canonicalize_drug_name  # noqa: E402

NS = "http://www.drugbank.ca"
NS_PREFIX = f"{{{NS}}}"

DRUG_TAG = f"{NS_PREFIX}drug"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("build_drugbank_derivatives")


def _q(tag: str) -> str:
    """Qualify a local tag name with the DrugBank namespace."""
    return f"{NS_PREFIX}{tag}"


def _text(elem: Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _primary_drugbank_id(drug: Element) -> str | None:
    for db_id in drug.findall(_q("drugbank-id")):
        if db_id.get("primary") == "true":
            return _text(db_id)
    # Fallback: first drugbank-id child.
    first = drug.find(_q("drugbank-id"))
    return _text(first) if first is not None else None


def _groups(drug: Element) -> list[str]:
    groups_elem = drug.find(_q("groups"))
    if groups_elem is None:
        return []
    return [_text(g) for g in groups_elem.findall(_q("group")) if _text(g)]


def _synonyms(drug: Element) -> Iterator[tuple[str, str]]:
    """Yield (synonym_norm, kind) pairs for the drug.

    Depth of uniqueness is enforced by the caller via a per-drug set.
    """
    primary = _text(drug.find(_q("name")))
    if primary:
        yield canonicalize_drug_name(primary), "primary_name"

    syns = drug.find(_q("synonyms"))
    if syns is not None:
        for s in syns.findall(_q("synonym")):
            t = _text(s)
            if t:
                yield canonicalize_drug_name(t), "synonym"

    int_brands = drug.find(_q("international-brands"))
    if int_brands is not None:
        for b in int_brands.findall(_q("international-brand")):
            name = _text(b.find(_q("name")))
            if name:
                yield canonicalize_drug_name(name), "international_brand"

    products = drug.find(_q("products"))
    if products is not None:
        for p in products.findall(_q("product")):
            name = _text(p.find(_q("name")))
            if name:
                yield canonicalize_drug_name(name), "product_name"


def _products_summary(drug: Element) -> dict:
    """Collapse the <products> block into a single summary row.

    We pick the earliest US-approved marketing date as `approval_date` and
    the associated FDA application number. `first_marketed_date` is the
    earliest marketing date globally (any country, approved or not).
    """
    summary = {
        "appl_no": "",
        "approved": False,
        "approval_date": "",
        "first_marketed_date": "",
        "us_country_hit": False,
    }
    products = drug.find(_q("products"))
    if products is None:
        return summary

    earliest_us_approved: str | None = None
    earliest_us_approved_appl: str = ""
    earliest_any: str | None = None
    any_approved = False

    for p in products.findall(_q("product")):
        started = _text(p.find(_q("started-marketing-on")))
        country = _text(p.find(_q("country")))
        approved_txt = _text(p.find(_q("approved"))).lower()
        approved = approved_txt == "true"
        appl_no = _text(p.find(_q("fda-application-number")))

        if approved:
            any_approved = True

        if started:
            if earliest_any is None or started < earliest_any:
                earliest_any = started
            if approved and country == "US":
                summary["us_country_hit"] = True
                if earliest_us_approved is None or started < earliest_us_approved:
                    earliest_us_approved = started
                    earliest_us_approved_appl = appl_no

    summary["approved"] = any_approved
    summary["approval_date"] = earliest_us_approved or ""
    summary["first_marketed_date"] = earliest_any or ""
    summary["appl_no"] = earliest_us_approved_appl
    return summary


def parse(
    xml_path: Path,
    synonyms_out: Path,
    products_out: Path,
    progress_every: int = 2000,
) -> tuple[int, int, int]:
    """Stream through the XML once, emitting both derivative CSVs.

    Returns (drugs_seen, synonym_rows, products_rows).
    """
    synonyms_out.parent.mkdir(parents=True, exist_ok=True)
    products_out.parent.mkdir(parents=True, exist_ok=True)

    drugs = 0
    syn_rows = 0
    prod_rows = 0
    t0 = time.time()

    with (
        synonyms_out.open("w", newline="", encoding="utf-8") as syn_f,
        products_out.open("w", newline="", encoding="utf-8") as prod_f,
    ):
        syn_writer = csv.writer(syn_f)
        prod_writer = csv.writer(prod_f)
        syn_writer.writerow(["drugbank_id", "synonym_norm", "kind"])
        prod_writer.writerow([
            "drugbank_id", "drug_name_norm", "primary_name",
            "appl_no", "approved", "approval_date", "first_marketed_date",
            "us_country_hit", "groups",
        ])

        # iterparse yields (event, elem). We watch for end-events on the
        # top-level <drug> element — nested <drug> sub-elements may exist
        # inside references, so we must gate on depth. ElementTree does not
        # expose depth directly; guard by ensuring the parent is the root.
        context = iterparse(str(xml_path), events=("start", "end"))
        _, root = next(context)  # consume root start event

        depth = 1  # root seen
        for event, elem in context:
            if event == "start":
                if elem.tag == DRUG_TAG:
                    depth += 1
                continue

            # end
            if elem.tag == DRUG_TAG:
                depth -= 1
                # Only process the top-level drug (depth 1 == root only).
                if depth == 1:
                    db_id = _primary_drugbank_id(elem)
                    primary = _text(elem.find(_q("name")))
                    if db_id:
                        # Emit deduplicated synonyms for this drug.
                        seen: set[tuple[str, str]] = set()
                        for norm, kind in _synonyms(elem):
                            if not norm:
                                continue
                            key = (norm, kind)
                            if key in seen:
                                continue
                            seen.add(key)
                            syn_writer.writerow([db_id, norm, kind])
                            syn_rows += 1

                        summary = _products_summary(elem)
                        prod_writer.writerow([
                            db_id,
                            canonicalize_drug_name(primary),
                            primary,
                            summary["appl_no"],
                            "true" if summary["approved"] else "false",
                            summary["approval_date"],
                            summary["first_marketed_date"],
                            "true" if summary["us_country_hit"] else "false",
                            "|".join(_groups(elem)),
                        ])
                        prod_rows += 1
                        drugs += 1

                    # Release memory for this drug and all preceding siblings.
                    elem.clear()
                    # Drop earlier siblings from the root so the parse tree
                    # does not balloon. This is the standard iterparse trick.
                    while len(root) > 1:
                        del root[0]

                    if drugs % progress_every == 0:
                        elapsed = time.time() - t0
                        logger.info(
                            "  parsed %s drugs (syn_rows=%s, prod_rows=%s) in %.1fs",
                            f"{drugs:,}", f"{syn_rows:,}", f"{prod_rows:,}", elapsed,
                        )

    return drugs, syn_rows, prod_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--xml",
        default="docs/data/full database.xml",
        help="Path to DrugBank full XML (default: docs/data/full database.xml)",
    )
    parser.add_argument(
        "--synonyms-out",
        default="data/drugbank_synonyms.csv",
        help="Output path for synonyms CSV (default: data/drugbank_synonyms.csv)",
    )
    parser.add_argument(
        "--products-out",
        default="data/drugbank_products.csv",
        help="Output path for products CSV (default: data/drugbank_products.csv)",
    )
    args = parser.parse_args(argv)

    xml_path = Path(args.xml)
    if not xml_path.exists():
        logger.error("DrugBank XML not found at %s", xml_path)
        return 2

    logger.info("Parsing %s (%.1f GB)", xml_path, xml_path.stat().st_size / 1e9)
    t0 = time.time()
    drugs, syn_rows, prod_rows = parse(
        xml_path=xml_path,
        synonyms_out=Path(args.synonyms_out),
        products_out=Path(args.products_out),
    )
    elapsed = time.time() - t0
    logger.info(
        "Done: %s drugs -> %s synonym rows, %s product rows in %.1fs",
        f"{drugs:,}", f"{syn_rows:,}", f"{prod_rows:,}", elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
