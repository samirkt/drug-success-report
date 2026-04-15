"""Unit tests for the DrugBank XML streaming parser script."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from scripts.build_drugbank_derivatives import parse  # noqa: E402


_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca" version="5.1" exported-on="2026-01-01">
<drug type="biotech" created="2005-06-13" updated="2025-11-04">
  <drugbank-id primary="true">DB00001</drugbank-id>
  <drugbank-id>BTD00024</drugbank-id>
  <name>Lepirudin</name>
  <groups>
    <group>approved</group>
    <group>withdrawn</group>
  </groups>
  <synonyms>
    <synonym language="english" coder="inn">Lepirudin</synonym>
    <synonym language="english">Hirudin variant-1</synonym>
    <synonym language="english">R-hirudin</synonym>
  </synonyms>
  <international-brands>
    <international-brand>
      <name>Refludan</name>
      <company>Bayer</company>
    </international-brand>
  </international-brands>
  <products>
    <product>
      <name>Refludan</name>
      <started-marketing-on>1998-03-06</started-marketing-on>
      <fda-application-number>NDA020807</fda-application-number>
      <approved>true</approved>
      <country>US</country>
    </product>
    <product>
      <name>Refludan</name>
      <started-marketing-on>2000-01-31</started-marketing-on>
      <approved>true</approved>
      <country>Canada</country>
    </product>
  </products>
</drug>
<drug type="small molecule" created="2006-01-01" updated="2025-01-01">
  <drugbank-id primary="true">DB00002</drugbank-id>
  <name>CompoundX</name>
  <groups>
    <group>investigational</group>
  </groups>
  <synonyms>
    <synonym>Codename-123</synonym>
  </synonyms>
  <international-brands/>
  <products/>
</drug>
</drugbank>
"""


def test_parse_writes_expected_rows(tmp_path: Path):
    xml_path = tmp_path / "mini.xml"
    xml_path.write_text(_SAMPLE_XML, encoding="utf-8")
    syn_out = tmp_path / "synonyms.csv"
    prod_out = tmp_path / "products.csv"

    drugs, syn_rows, prod_rows = parse(xml_path, syn_out, prod_out, progress_every=1000)

    assert drugs == 2
    assert syn_rows >= 5  # at least primary + 3 synonyms + 1 international brand for DB00001
    assert prod_rows == 2

    syn_data = list(csv.DictReader(syn_out.open()))
    prod_data = list(csv.DictReader(prod_out.open()))

    # DB00001: primary name, synonyms, international brand, product_name
    lepirudin_rows = [r for r in syn_data if r["drugbank_id"] == "DB00001"]
    lepirudin_kinds = {r["kind"] for r in lepirudin_rows}
    assert "primary_name" in lepirudin_kinds
    assert "synonym" in lepirudin_kinds
    assert "international_brand" in lepirudin_kinds
    assert "product_name" in lepirudin_kinds
    # "Hirudin variant-1" normalizes to something like "hirudin variant 1"
    norms = {r["synonym_norm"] for r in lepirudin_rows}
    assert any("lepirudin" in n for n in norms)
    assert any("refludan" in n for n in norms)

    # Products summary for DB00001: earliest US-approved = 1998-03-06, appl=NDA020807
    db1_prod = next(r for r in prod_data if r["drugbank_id"] == "DB00001")
    assert db1_prod["approval_date"] == "1998-03-06"
    assert db1_prod["appl_no"] == "NDA020807"
    assert db1_prod["approved"] == "true"
    assert db1_prod["us_country_hit"] == "true"
    # first_marketed_date is min across all countries
    assert db1_prod["first_marketed_date"] == "1998-03-06"
    assert "approved" in db1_prod["groups"]
    assert "withdrawn" in db1_prod["groups"]

    # DB00002 has no products — summary should be empty.
    db2_prod = next(r for r in prod_data if r["drugbank_id"] == "DB00002")
    assert db2_prod["approval_date"] == ""
    assert db2_prod["appl_no"] == ""
    assert db2_prod["approved"] == "false"
    assert db2_prod["us_country_hit"] == "false"

    # DB00002 should still have synonyms (primary + codename).
    db2_syns = [r for r in syn_data if r["drugbank_id"] == "DB00002"]
    assert len(db2_syns) >= 2
    db2_kinds = {r["kind"] for r in db2_syns}
    assert "primary_name" in db2_kinds
    assert "synonym" in db2_kinds
