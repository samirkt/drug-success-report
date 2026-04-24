#!/usr/bin/env python3
"""
Streams DrugBank XML and writes a minimized CSV for fast pandas loading.

Emits lookup rows for:
  - Canonical drug name
  - Synonyms (<synonyms>/<synonym>)
  - International brand names (<international-brands>/<international-brand>/<name>)
  - Product names (<products>/<product>/<name>)
  - DrugBank accession / IDs (primary + secondary <drugbank-id>)
  - External identifiers (<external-identifiers>/<external-identifier>)

Adds:
  - Approval status via <groups><group>approved|investigational|withdrawn|...</group></groups>
  - is_approved flag (1/0)
  - Indications / approved conditions via <indication> and/or <indications><indication>...</indication></indications>

Also:
  - Only computes aa_length from the drug's OWN <sequences><sequence> blobs (not targets/etc).
  - Only outputs aa_length when a credible AA sequence is present (conservative parsing).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from lxml import etree as ET  # faster
    USING_LXML = True
except Exception:
    import xml.etree.ElementTree as ET
    USING_LXML = False


# Bumped whenever the emitted CSV schema changes. Downstream consumers
# (drugbank_norm.load_drugbank_lookup, the SMILES enrichment) log a warning
# when they encounter a CSV missing columns they expect — but never fail,
# so older caches keep working through the minimizer re-run cycle.
MINIMIZER_SCHEMA_VERSION = 2


# ----------------------------
# helpers
# ----------------------------

_WS = re.compile(r"\s+")
_KEEP_ID = re.compile(r"[^a-z0-9:]+")


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def norm(s: str) -> str:
    # for names/synonyms/brands/products
    return _WS.sub(" ", s.strip().lower())


def norm_id(s: str) -> str:
    # for IDs: lowercase, remove whitespace, keep only [a-z0-9:] (preserve "chebi:123")
    s = _WS.sub("", s.strip().lower())
    return _KEEP_ID.sub("", s)


_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")  # include common ambiguous/rare codes
_SEQ_STOP_TOKENS = {"SEQUENCE"}  # avoid counting common non-seq tokens if they appear uppercase
_NUCLEIC_BASES = set("ACGTUN")


def _clean_aa_sequence_from_text(text: str, *, min_len: int = 5) -> Optional[str]:
    if not text:
        return None

    aa_chunks: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue

        for tok in line.split():
            tok = re.sub(r"[^A-Za-z]", "", tok)
            if not tok:
                continue
            if not tok.isupper():
                continue
            if tok in _SEQ_STOP_TOKENS:
                continue
            if all(c in _VALID_AA for c in tok):
                aa_chunks.append(tok)

    seq = "".join(aa_chunks)
    return seq if len(seq) >= min_len else None


def _looks_nucleotide_sequence(seq: str) -> bool:
    """
    Conservative filter to avoid treating oligonucleotide sequences as amino acids.
    If a sequence only uses DNA/RNA base letters, treat it as nucleic-acid (not AA).
    """
    if not seq:
        return False
    return set(seq).issubset(_NUCLEIC_BASES)


def aa_sequence_from_drug_sequence_text(text: str) -> Optional[str]:
    seq = _clean_aa_sequence_from_text(text)
    if not seq:
        return None
    if _looks_nucleotide_sequence(seq):
        return None
    return seq


def dedupe_keep_order(values: List[str], normalizer=norm) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        k = normalizer(v)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


# ----------------------------
# extraction
# ----------------------------

def extract_all_drugbank_ids(drug_elem) -> Tuple[Optional[str], List[str]]:
    primary = None
    all_ids: List[str] = []
    for c in list(drug_elem):
        if strip_ns(c.tag) == "drugbank-id":
            t = (c.text or "").strip()
            if not t:
                continue
            all_ids.append(t)
            if c.attrib.get("primary", "").lower() == "true":
                primary = t

    all_ids = dedupe_keep_order(all_ids, normalizer=norm_id)
    return primary, all_ids


def extract_name(drug_elem) -> Optional[str]:
    for c in list(drug_elem):
        if strip_ns(c.tag) == "name":
            t = (c.text or "").strip()
            if t:
                return t
    return None


def extract_synonyms(drug_elem) -> List[str]:
    out: List[str] = []
    for c in list(drug_elem):
        if strip_ns(c.tag) == "synonyms":
            for s in list(c):
                if strip_ns(s.tag) == "synonym":
                    t = (s.text or "").strip()
                    if t:
                        out.append(t)
    return dedupe_keep_order(out, normalizer=norm)


def extract_international_brand_names(drug_elem) -> List[str]:
    out: List[str] = []
    for c in list(drug_elem):
        if strip_ns(c.tag) != "international-brands":
            continue
        for ib in list(c):
            if strip_ns(ib.tag) != "international-brand":
                continue
            for node in list(ib):
                if strip_ns(node.tag) == "name":
                    t = (node.text or "").strip()
                    if t:
                        out.append(t)
        break
    return dedupe_keep_order(out, normalizer=norm)


def extract_product_names(drug_elem) -> List[str]:
    out: List[str] = []
    for c in list(drug_elem):
        if strip_ns(c.tag) != "products":
            continue
        for prod in list(c):
            if strip_ns(prod.tag) != "product":
                continue
            for node in list(prod):
                if strip_ns(node.tag) == "name":
                    t = (node.text or "").strip()
                    if t:
                        out.append(t)
        break
    return dedupe_keep_order(out, normalizer=norm)


def extract_external_ids(drug_elem) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    for c in list(drug_elem):
        if strip_ns(c.tag) != "external-identifiers":
            continue
        for ext in list(c):
            if strip_ns(ext.tag) != "external-identifier":
                continue

            resource = None
            ident = None
            for node in list(ext):
                ln = strip_ns(node.tag)
                if ln == "resource":
                    r = (node.text or "").strip()
                    if r:
                        resource = r
                elif ln == "identifier":
                    i = (node.text or "").strip()
                    if i:
                        ident = i

            if resource and ident:
                pairs.append((resource, ident))

    seen = set()
    out: List[Tuple[str, str]] = []
    for r, i in pairs:
        k = f"{norm(r)}:{norm_id(i)}"
        if k in seen:
            continue
        seen.add(k)
        out.append((r, i))
    return out


def extract_drug_sequence_texts(drug_elem) -> List[str]:
    seq_texts: List[str] = []
    for c in list(drug_elem):
        if strip_ns(c.tag) != "sequences":
            continue
        for s in list(c):
            if strip_ns(s.tag) != "sequence":
                continue
            t = (s.text or "").strip()
            if t:
                seq_texts.append(t)
        break
    return seq_texts


def extract_approval_groups(drug_elem) -> List[str]:
    """
    DrugBank approval / lifecycle is usually in:
      <groups>
        <group>approved</group>
        <group>investigational</group>
        ...
      </groups>
    Returns lowercased group strings in original order (deduped).
    """
    out: List[str] = []
    for c in list(drug_elem):
        if strip_ns(c.tag) != "groups":
            continue
        for g in list(c):
            if strip_ns(g.tag) != "group":
                continue
            t = (g.text or "").strip()
            if t:
                out.append(t.lower())
        break
    # keep order, dedupe by exact lowercase
    seen = set()
    deduped: List[str] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        deduped.append(v)
    return deduped


def extract_indications(drug_elem) -> List[str]:
    """
    DrugBank commonly uses a single free-text field:
      <indication>...</indication>

    Some variants (or other XML sources) may include:
      <indications>
        <indication>...</indication>
        ...
      </indications>

    This function supports both and returns a deduped list.
    """
    out: List[str] = []

    # Common: direct child <indication>
    for c in list(drug_elem):
        ln = strip_ns(c.tag)
        if ln == "indication":
            t = (c.text or "").strip()
            if t:
                out.append(_WS.sub(" ", t))
        elif ln == "indications":
            # less common: container of multiple <indication>
            for node in list(c):
                if strip_ns(node.tag) == "indication":
                    t = (node.text or "").strip()
                    if t:
                        out.append(_WS.sub(" ", t))

    # Deduplicate by normalized text (case/whitespace)
    return dedupe_keep_order(out, normalizer=norm)


def infer_modality(drug_elem, aa_len: Optional[int]) -> str:
    if aa_len is not None:
        return "inferred_protein_or_peptide"

    t = drug_elem.attrib.get("type", "").lower()
    if "biotech" in t or "biologic" in t or "biological" in t:
        return "inferred_biologic_non_protein"

    return "inferred_small_molecule_or_other"


def extract_classyfire_classification(drug_elem) -> dict:
    out = {
        "kingdom": "",
        "superclass": "",
        "class": "",
        "subclass": "",
        "direct_parent": "",
        "alt_parents": [],
        "substituents": [],
    }

    for c in list(drug_elem):
        if strip_ns(c.tag) != "classification":
            continue

        for node in list(c):
            ln = strip_ns(node.tag)
            t = (node.text or "").strip()
            if not t:
                continue

            if ln in ("kingdom", "superclass", "class", "subclass", "direct-parent"):
                key = ln.replace("-", "_")
                out[key] = t
            elif ln == "alternative-parent":
                out["alt_parents"].append(t)
            elif ln == "substituent":
                out["substituents"].append(t)

        break

    out["alt_parents"] = dedupe_keep_order(out["alt_parents"], normalizer=norm)
    out["substituents"] = dedupe_keep_order(out["substituents"], normalizer=norm)
    return out


def peptide_like_from_classyfire(cf: dict) -> bool:
    hay = " | ".join([
        cf.get("kingdom", ""),
        cf.get("superclass", ""),
        cf.get("class", ""),
        cf.get("subclass", ""),
        cf.get("direct_parent", ""),
        " | ".join(cf.get("alt_parents", [])),
        " | ".join(cf.get("substituents", [])),
    ]).lower()

    return ("peptide" in hay) or ("polypeptide" in hay) or ("oligopeptide" in hay)


def search_drug_cat(drug_elem) -> bool:
    for c in list(drug_elem):
        if strip_ns(c.tag) != "categories":
            continue
        for cat_wrapper in list(c):
            if strip_ns(cat_wrapper.tag) != "category":
                continue
            for inner in list(cat_wrapper):
                if strip_ns(inner.tag) == "category":
                    t = (inner.text or "").strip().lower()
                    if t == "peptides":
                        return True
    return False

_CALC_PROP_KINDS = {
    "SMILES": "smiles",
    "InChI": "inchi",
    "logP": "logp",
    "Molecular Weight": "molecular_weight",
}


def extract_calculated_properties(drug_elem) -> dict:
    """Read selected entries from <calculated-properties>.

    Returns a dict with keys `smiles`, `inchi`, `logp`, `molecular_weight`
    (any missing keys are empty strings). DrugBank also exposes the same
    concepts under <experimental-properties>; we prefer the calculated
    values for consistency and only fall back to experimental when a
    calculated value is absent.
    """
    out = {v: "" for v in _CALC_PROP_KINDS.values()}

    def _read_properties(container, *, overwrite: bool) -> None:
        for prop in list(container):
            if strip_ns(prop.tag) != "property":
                continue
            kind = ""
            value = ""
            for node in list(prop):
                ln = strip_ns(node.tag)
                if ln == "kind":
                    kind = (node.text or "").strip()
                elif ln == "value":
                    value = (node.text or "").strip()
            key = _CALC_PROP_KINDS.get(kind)
            if not key or not value:
                continue
            if overwrite or not out[key]:
                out[key] = value

    for c in list(drug_elem):
        if strip_ns(c.tag) == "calculated-properties":
            _read_properties(c, overwrite=True)
    for c in list(drug_elem):
        if strip_ns(c.tag) == "experimental-properties":
            _read_properties(c, overwrite=False)

    return out


def extract_reported_modality(drug_elem) -> str:
    """
    Prefer DrugBank-reported drug type/modality from the <drug> element.

    In DrugBank XML this is typically an attribute on <drug>, e.g.:
      <drug type="biotech" ...>
      <drug type="small molecule" ...>

    Returns a normalized token (e.g., "biotech", "small_molecule"),
    or "" if not present.
    """
    t = (drug_elem.attrib.get("type", "") or "").strip()

    # Defensive fallback in case a variant export stores it as a child node
    if not t:
        for c in list(drug_elem):
            if strip_ns(c.tag) == "type":
                t = (c.text or "").strip()
                if t:
                    break

    if not t:
        return ""

    # normalize to stable token
    t = _WS.sub(" ", t).strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t

# Helper function for examining XML [DON'T DELETE]
def _print_structure(elem, depth=0):
    indent = "  " * depth
    print(f"{indent}{strip_ns(elem.tag)}: {elem.text.strip() if elem.text else ''}")

    for child in list(elem):
        if strip_ns(child.tag) != "drug-interaction":
            _print_structure(child, depth + 1)


# ----------------------------
# main streaming loop
# ----------------------------

def stream_drugbank(xml_path: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "drug_id",
            "query_name",
            "query_norm",
            "query_kind",     # canonical | synonym | brand | product | drugbank_id | external_id | external_scoped
            "modality",
            "aa_sequence",
            "aa_length",

            "approval_groups",   # pipe-separated, e.g. approved|withdrawn
            "is_approved",       # 1/0
            "indications",       # pipe-separated free-text indications

            "cf_superclass",
            "cf_class",
            "cf_subclass",
            "cf_direct_parent",
            "cf_alt_parents",
            "cf_substituents",
            "peptide_like_cf",
            "peptide_drug_cat",

            # Calculated chemical properties (schema v2). Sourced from
            # DrugBank <calculated-properties>, falling back to
            # <experimental-properties> when calculated is absent.
            "smiles",
            "inchi",
            "logp",
            "molecular_weight",
        ])

        context = ET.iterparse(str(xml_path), events=("end",))
        for _, elem in context:
            if strip_ns(elem.tag) != "drug":
                continue

            primary_id, all_db_ids = extract_all_drugbank_ids(elem)
            drug_id = primary_id or (all_db_ids[0] if all_db_ids else None)
            name = extract_name(elem)
            if not drug_id or not name:
                elem.clear()
                continue

            # sequences
            seq_texts = extract_drug_sequence_texts(elem)
            aa_len: Optional[int] = None
            aa_sequences: List[str] = []
            for blob in seq_texts:
                aa_seq = aa_sequence_from_drug_sequence_text(blob)
                if aa_seq is None:
                    continue
                aa_sequences.append(aa_seq)
                aa_len = (aa_len or 0) + len(aa_seq)
            aa_sequence_str = "|".join(aa_sequences)

            modality_reported = extract_reported_modality(elem)
            #modality = modality_reported or infer_modality(elem, aa_len)
            modality = modality_reported or "unknown"

            # NEW: approval + indications
            approval_groups = extract_approval_groups(elem)          # list[str], lowercase
            is_approved = "1" if ("approved" in approval_groups) else "0"
            indications = extract_indications(elem)                  # list[str]

            approval_groups_str = "|".join(approval_groups)
            indications_str = "|".join(indications)

            # existing annotations
            cf = extract_classyfire_classification(elem)
            peptide_like_cf = peptide_like_from_classyfire(cf)
            peptide_drug_cat = search_drug_cat(elem)

            calc_props = extract_calculated_properties(elem)

            def write_token(qname: str, qnorm: str, kind: str) -> None:
                writer.writerow([
                    drug_id,
                    qname,
                    qnorm,
                    kind,
                    modality,
                    aa_sequence_str,
                    aa_len if aa_len is not None else "",

                    approval_groups_str,
                    is_approved,
                    indications_str,

                    cf.get("superclass", ""),
                    cf.get("class", ""),
                    cf.get("subclass", ""),
                    cf.get("direct_parent", ""),
                    "|".join(cf.get("alt_parents", [])),
                    "|".join(cf.get("substituents", [])),
                    "1" if peptide_like_cf else "0",
                    "1" if peptide_drug_cat else "0",

                    calc_props["smiles"],
                    calc_props["inchi"],
                    calc_props["logp"],
                    calc_props["molecular_weight"],
                ])

            # canonical
            write_token(name, norm(name), "canonical")

            # synonyms
            for syn in extract_synonyms(elem):
                write_token(syn, norm(syn), "synonym")

            # international brands
            for brand in extract_international_brand_names(elem):
                write_token(brand, norm(brand), "brand")

            # product names
            for prod_name in extract_product_names(elem):
                write_token(prod_name, norm(prod_name), "product")

            # DrugBank IDs
            for dbid in all_db_ids:
                write_token(dbid, norm_id(dbid), "drugbank_id")

            # external IDs
            for resource, ident in extract_external_ids(elem):
                write_token(ident, norm_id(ident), "external_id")
                scoped = f"{resource}:{ident}"
                write_token(scoped, f"{norm(resource)}:{norm_id(ident)}", "external_scoped")

            # free memory
            elem.clear()
            if USING_LXML:
                while elem.getprevious() is not None:
                    del elem.getparent()[0]


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="~/Documents/projects/amp_research/databases/drugbank/full database.xml", type=Path)
    ap.add_argument("--out", default="~/Documents/projects/amp_research/databases/drugbank/minimized_test.csv", type=Path)
    args = ap.parse_args()
    print(f"Streaming DrugBank XML from {args.xml} and writing minimized CSV to {args.out}...")

    stream_drugbank(args.xml, args.out)


if __name__ == "__main__":
    main()
