"""I/O layer: serialize ReportOutput to HTML, CSV, Excel, and xlsx data export."""

from __future__ import annotations

import base64
import json
import os
from typing import Optional

import numpy as np

from ...models import (
    AttributeTable,
    CandidateOutcomeRecord,
    CandidateTable,
    OutcomeTable,
    ReportOutput,
    TrialTable,
)
from . import _compute


def write_report(
    report: ReportOutput,
    output_path: str,
    formats: list[str],
    export_data: dict,
    sections: list[dict] | None = None,
    exec_summary: str = "",
    introduction: str = "",
) -> None:
    """Serialize report tables and figures to output_path."""
    os.makedirs(output_path, exist_ok=True)

    # Save all PNG figures to output dir
    #for fig_name, fig_bytes in report.figures.items():
    #    if fig_bytes and not fig_name.startswith("disease_spider_"):
    #        with open(os.path.join(output_path, f"{fig_name}.png"), "wb") as f:
    #            f.write(fig_bytes)

    write_csv(report, output_path)
    write_data_xlsx(export_data, output_path)

    if "html" in formats:
        write_html(report, output_path, sections=sections,
                   exec_summary=exec_summary, introduction=introduction)

    if "excel" in formats:
        write_excel(report, output_path)


def write_csv(report: ReportOutput, output_path: str) -> None:
    import csv
    rows = report.tables.get("candidate_summary", [])
    if not rows:
        return
    csv_path = os.path.join(output_path, "candidate_summary.csv")
    headers = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_detail(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
    outcome_table: OutcomeTable,
    output_path: str,
) -> None:
    """Write one CSV row per candidate with every field joined in — for debugging.

    Sibling of `write_trial_detail` but candidate-centric: no per-trial
    expansion. Dumps every `Candidate` dataclass field (including the
    enrichment fields — SMILES, drug targets, ICD-10), plus the joined
    `CandidateAttributes` and `CandidateOutcomeRecord` (modality / disease
    area / outcome / approval dates / LLM reasoning strings). List-valued
    fields are pipe-joined to match the `trial_detail.csv` convention.
    """
    import csv

    headers = [
        # Candidate dataclass fields
        "candidate_id",
        "drug_name",
        "drug_name_raw",
        "indication",
        "highest_phase",
        "trial_count",
        "trial_ids",
        "sponsors",
        "earliest_start_date",
        "latest_completion_date",
        "drugbank_id",
        "mesh_drug",
        "mesh_indication",
        "mesh_condition_tree_numbers",
        "single_arm_p_values_count",
        # Enrichment fields (schema v2+)
        "smiles",
        "drug_targets",
        "target_names",
        "icd10_code",
        "icd10_description",
        "opentargets_moa",
        "opentargets_action_type",
        "opentargets_targets",
        "opentargets_pathways",
        "opentargets_indication_max_phase",
        # CandidateAttributes
        "modality",
        "disease_area",
        "modality_confidence",
        "disease_confidence",
        "modality_reasoning",
        # CandidateOutcomeRecord
        "outcome",
        "outcome_confidence",
        "outcome_reasoning",
        "outcome_evidence_sources",
        "approval_date",
        "commercialization_date",
    ]

    csv_path = os.path.join(output_path, "candidate_detail.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for c in candidate_table.candidates:
            attrs = attribute_table.attributes.get(c.candidate_id)
            out = outcome_table.outcomes.get(c.candidate_id)

            writer.writerow({
                "candidate_id": c.candidate_id,
                "drug_name": c.drug_name,
                "drug_name_raw": c.drug_name_raw,
                "indication": c.indication,
                "highest_phase": c.highest_phase.value,
                "trial_count": len(c.trial_ids),
                "trial_ids": "|".join(c.trial_ids),
                "sponsors": "|".join(c.sponsors),
                "earliest_start_date": (
                    c.earliest_start_date.isoformat() if c.earliest_start_date else ""
                ),
                "latest_completion_date": (
                    c.latest_completion_date.isoformat()
                    if c.latest_completion_date else ""
                ),
                "drugbank_id": c.drugbank_id or "",
                "mesh_drug": c.mesh_drug or "",
                "mesh_indication": c.mesh_indication or "",
                "mesh_condition_tree_numbers": "|".join(c.mesh_condition_tree_numbers),
                "single_arm_p_values_count": len(c.single_arm_p_values),
                "smiles": c.smiles or "",
                "drug_targets": "|".join(c.drug_targets),
                "target_names": "|".join(c.target_names),
                "icd10_code": c.icd10_code or "",
                "icd10_description": c.icd10_description or "",
                "opentargets_moa": c.opentargets_moa or "",
                "opentargets_action_type": c.opentargets_action_type or "",
                "opentargets_targets": "|".join(c.opentargets_targets),
                "opentargets_pathways": "|".join(c.opentargets_pathways),
                "opentargets_indication_max_phase": (
                    str(c.opentargets_indication_max_phase)
                    if c.opentargets_indication_max_phase is not None else ""
                ),
                "modality": attrs.drug_modality if attrs else "",
                "disease_area": attrs.disease_area if attrs else "",
                "modality_confidence": attrs.modality_confidence if attrs else "",
                "disease_confidence": attrs.disease_confidence if attrs else "",
                "modality_reasoning": attrs.reasoning if attrs else "",
                "outcome": out.outcome.value if out else "",
                "outcome_confidence": out.confidence if out else "",
                "outcome_reasoning": out.reasoning if out else "",
                "outcome_evidence_sources": (
                    "|".join(out.evidence_sources) if out else ""
                ),
                "approval_date": (
                    out.approval_date.isoformat()
                    if (out and out.approval_date) else ""
                ),
                "commercialization_date": (
                    out.commercialization_date.isoformat()
                    if (out and out.commercialization_date) else ""
                ),
            })


def write_trial_detail(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
    outcome_table: OutcomeTable,
    trial_table: TrialTable | None,
    output_path: str,
) -> None:
    """Write a (candidate × trial) CSV for debugging classification and clustering.

    One row per constituent trial. Each row shows both the candidate-level
    canonical drug/indication and classification tags and the trial-level
    raw AACT intervention/indication strings plus MeSH terms, so that a
    reader can vet whether the clustering merged the right trials and whether
    the modality/disease-area classification is consistent with the trial
    evidence.
    """
    import csv

    if trial_table is None:
        return

    trial_index = {t.nct_id: t for t in trial_table.trials}

    headers = [
        "candidate_id",
        "candidate_drug",
        "candidate_drug_raw",
        "candidate_indication",
        "candidate_modality",
        "candidate_disease_area",
        "candidate_outcome",
        "candidate_highest_phase",
        "candidate_drugbank_id",
        "candidate_mesh_drug",
        "candidate_mesh_indication",
        "nct_id",
        "trial_intervention",
        "trial_indication",
        "trial_phase",
        "trial_status",
        "trial_mesh_intervention_terms",
        "trial_mesh_condition_terms",
        "trial_mesh_condition_tree_numbers",
        "trial_start_date",
        "trial_completion_date",
        "trial_is_single_arm",
        "trial_sponsor",
        "trial_title",
    ]

    csv_path = os.path.join(output_path, "trial_detail.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for c in candidate_table.candidates:
            attrs = attribute_table.attributes.get(c.candidate_id)
            out = outcome_table.outcomes.get(c.candidate_id)

            cand_cols = {
                "candidate_id": c.candidate_id,
                "candidate_drug": c.drug_name,
                "candidate_drug_raw": c.drug_name_raw,
                "candidate_indication": c.indication,
                "candidate_modality": attrs.drug_modality if attrs else "",
                "candidate_disease_area": attrs.disease_area if attrs else "",
                "candidate_outcome": out.outcome.value if out else "",
                "candidate_highest_phase": c.highest_phase.value,
                "candidate_drugbank_id": c.drugbank_id or "",
                "candidate_mesh_drug": c.mesh_drug or "",
                "candidate_mesh_indication": c.mesh_indication or "",
            }

            if not c.trial_ids:
                writer.writerow({**cand_cols, **{h: "" for h in headers if h not in cand_cols}})
                continue

            for nct in c.trial_ids:
                t = trial_index.get(nct)
                if t is None:
                    writer.writerow({
                        **cand_cols,
                        "nct_id": nct,
                        **{h: "" for h in headers
                           if h not in cand_cols and h != "nct_id"},
                    })
                    continue

                writer.writerow({
                    **cand_cols,
                    "nct_id": t.nct_id,
                    "trial_intervention": t.intervention,
                    "trial_indication": t.indication,
                    "trial_phase": t.phase.value,
                    "trial_status": t.status.value,
                    "trial_mesh_intervention_terms": "|".join(t.mesh_intervention_terms),
                    "trial_mesh_condition_terms": "|".join(t.mesh_condition_terms),
                    "trial_mesh_condition_tree_numbers": "|".join(t.mesh_condition_tree_numbers),
                    "trial_start_date": t.start_date.isoformat() if t.start_date else "",
                    "trial_completion_date": t.completion_date.isoformat() if t.completion_date else "",
                    "trial_is_single_arm": "true" if t.is_single_arm else "false",
                    "trial_sponsor": t.sponsor,
                    "trial_title": t.title,
                })


def write_html(
    report: ReportOutput,
    output_path: str,
    sections: list[dict] | None = None,
    exec_summary: str = "",
    introduction: str = "",
) -> None:
    from datetime import date

    from . import _html_template as tmpl

    if sections is not None:
        _write_html_narrative(report, output_path, sections, exec_summary, introduction)
    else:
        _write_html_legacy(report, output_path)

    # Always generate the standalone interactive heatmaps file
    #_write_interactive_heatmaps(report, output_path)


def _write_html_narrative(
    report: ReportOutput,
    output_path: str,
    sections: list[dict],
    exec_summary: str,
    introduction: str,
) -> None:
    """Render a BIO/QLS-style narrative report with interleaved text and figures."""
    from datetime import date

    from . import _html_template as tmpl

    parts = [tmpl.html_head("Clinical Development Success Rates")]

    # Title page
    parts.append(tmpl.title_page(
        "Clinical Development Success Rates and Contributing Factors",
        "Automated Pipeline Analysis",
        "Generated " + date.today().strftime("%B %Y"),
    ))

    # Executive summary
    if exec_summary:
        parts.append(tmpl.executive_summary_block(exec_summary))

    # Table of contents
    parts.append(_build_toc(sections))

    # Introduction
    if introduction:
        parts.append('<h2>Introduction</h2>\n')
        paragraphs = introduction.strip().split("\n\n")
        parts.append('<div class="narrative">\n')
        for p in paragraphs:
            p = p.strip()
            if p:
                parts.append(f'<p>{p}</p>\n')
        parts.append('</div>\n')

    # -- Render sections grouped into parts --
    PART_MAP: dict[str, tuple[str | None, int]] = {
        "funnel":            (None, 1),
        "disease_breakdown": (None, 1),
        "heatmaps":          (None, 1),
        "bubble_heatmaps":   (None, 1),
        "loa":               (None, 1),
        "oncology":          (None, 1),
        "timeline":          ("Part 2. Drug Development Timelines", 2),
        "time_period":       ("Part 3. Temporal and Structural Analysis", 3),
        "modality_trend":    (None, 3),
        "sponsor":           (None, 3),
        "spider":            ("Part 1. Phase Transition Success and Likelihood of Approval", 1),
        "candidate_summary": (None, 5),
    }

    fig_counter = [0]
    table_counter = [0]
    current_part = 0

    for section in sections:
        key = section["key"]
        part_heading, part_num = PART_MAP.get(key, (None, 0))

        # Emit part heading when entering a new part
        if part_num > current_part and part_heading:
            parts.append(f'<h2>{part_heading}</h2>\n')
            current_part = part_num

        sec_figures = section.get("figures", {})
        sec_tables = section.get("tables", {})
        sec_html_figures = section.get("html_figures", {})

        # Appendix: candidate summary
        if key == "candidate_summary":
            parts.append('<div class="appendix">\n')
            parts.append('<h2>Appendix: Candidate Summary</h2>\n')
            if "candidate_summary" in sec_tables:
                table_counter[0] += 1
                parts.append(tmpl.render_table(
                    sec_tables["candidate_summary"],
                    "Individual candidate development summary.",
                    table_counter[0],
                ))
            parts.append('</div>\n')
            continue

        title = section.get("title", key.replace("_", " ").title())
        narrative = section.get("narrative", "")

        if narrative:
            parts.append(f'<h3>{title}</h3>\n')
            parts.append(tmpl.render_narrative(
                narrative,
                sec_figures,
                sec_tables,
                sec_html_figures,
                fig_counter,
                table_counter,
            ))
        else:
            # No narrative -- dump figures and tables directly
            parts.append(f'<h3>{title}</h3>\n')
            for fig_key, fig_bytes in sec_figures.items():
                fig_counter[0] += 1
                caption = fig_key.replace("_", " ").title()
                parts.append(tmpl.render_figure(fig_bytes, caption, fig_counter[0]))
            for html_key, html_content in sec_html_figures.items():
                fig_counter[0] += 1
                caption = html_key.replace("_", " ").title()
                parts.append(tmpl.render_html_figure(html_content, caption, fig_counter[0]))
            for tbl_key, tbl_rows in sec_tables.items():
                table_counter[0] += 1
                caption = tbl_key.replace("_", " ").title()
                parts.append(tmpl.render_table(tbl_rows, caption, table_counter[0]))

    parts.append('</body>\n</html>')

    html_path = os.path.join(output_path, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def _build_toc(sections: list[dict]) -> str:
    """Build a table-of-contents block from the sections list."""
    PART_LABELS: dict[str, str] = {
        "spider":          "Part 1. Phase Transition Success and Likelihood of Approval",
        "timeline":        "Part 2. Drug Development Timelines",
        "time_period":     "Part 3. Temporal and Structural Analysis",
    }
    PART_KEYS = set(PART_LABELS.keys())

    items: list[str] = []
    for section in sections:
        key = section["key"]
        if key in PART_KEYS:
            items.append(f'  <li class="part">{PART_LABELS[key]}</li>')
        if key == "candidate_summary":
            items.append('  <li class="part">Appendix</li>')
            items.append('  <li class="section">Candidate Summary</li>')
        else:
            title = section.get("title", key.replace("_", " ").title())
            items.append(f'  <li class="section">{title}</li>')

    return (
        '<div class="toc">\n'
        '  <h2>Table of Contents</h2>\n'
        '  <ul>\n'
        + "\n".join(items) + "\n"
        '  </ul>\n'
        '</div>\n'
    )


def _write_html_legacy(report: ReportOutput, output_path: str) -> None:
    """Original flat HTML dump -- used when no sections are provided."""
    parts = ['<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>']
    parts.append("<h1>Peptide Pipeline Report</h1>")
    parts.append(f"<p>{report.summary_text}</p>")

    figure_order = [
        "disease_breakdown", "modality_heatmap", "disease_heatmap",
        "loa_by_disease", "loa_by_modality", "timeline_by_disease", "oncology_comparison",
        "time_period_transitions", "time_period_comparison",
        "modality_proportion", "sponsor_concentration",
    ]
    figure_order.append("overall_spider")
    figure_order.extend(
        sorted(k for k in report.figures if k.startswith("disease_spider_"))
    )
    figure_order.append("modality_breakdown")
    for name in figure_order:
        if name not in report.figures:
            continue
        b64 = base64.b64encode(report.figures[name]).decode()
        parts.append(f"<h2>{name}</h2><img src='data:image/png;base64,{b64}' />")

    for name in ["modality_bubble_heatmap", "disease_bubble_heatmap", "modality_proportion"]:
        if name in report.html_figures:
            pretty = name.replace("_", " ").title()
            parts.append(f"<h2>{pretty}</h2>")
            parts.append(f"<div>{report.html_figures[name]}</div>")

    non_summary_tables = [k for k in report.tables if k != "candidate_summary"]
    table_order = non_summary_tables + (
        ["candidate_summary"] if "candidate_summary" in report.tables else []
    )
    for name in table_order:
        rows = report.tables[name]
        parts.append(f"<h2>{name}</h2>")
        if rows:
            headers = list(rows[0].keys())
            parts.append("<table border='1'><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
            for row in rows:
                parts.append("<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>")
            parts.append("</table>")

    parts.append("</body></html>")
    html_path = os.path.join(output_path, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def _write_interactive_heatmaps(report: ReportOutput, output_path: str) -> None:
    """Write standalone interactive heatmaps HTML file."""
    if not report.html_figures:
        return
    interactive_parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        '<title>Interactive Bubble Heatmaps</title></head><body>',
    ]
    for name in ["modality_bubble_heatmap", "disease_bubble_heatmap", "modality_proportion"]:
        if name in report.html_figures:
            pretty = name.replace("_", " ").title()
            interactive_parts.append(f"<h2>{pretty}</h2>")
            interactive_parts.append(f"<div>{report.html_figures[name]}</div>")
    interactive_parts.append("</body></html>")
    interactive_path = os.path.join(output_path, "interactive_heatmaps.html")
    with open(interactive_path, "w", encoding="utf-8") as f:
        f.write("\n".join(interactive_parts))


def write_excel(report: ReportOutput, output_path: str) -> None:
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    for name, rows in report.tables.items():
        ws = wb.create_sheet(title=name[:31])
        if rows:
            headers = list(rows[0].keys())
            ws.append(headers)
            for row in rows:
                ws.append([row.get(h, "") for h in headers])

    xlsx_path = os.path.join(output_path, "report.xlsx")
    wb.save(xlsx_path)


def write_data_xlsx(export_data: dict, output_path: str) -> None:
    """Export heatmap + LOA + timeline data to structured xlsx."""
    import openpyxl

    if not export_data:
        return

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    headers = [
        "Row Label",
        "P1\u2192P2 Rate", "P1\u2192P2 n", "P1\u2192P2 denom", "P1\u2192P2 CI Low", "P1\u2192P2 CI High",
        "P2\u2192P3 Rate", "P2\u2192P3 n", "P2\u2192P3 denom", "P2\u2192P3 CI Low", "P2\u2192P3 CI High",
        "P3\u2192Appr Rate", "P3\u2192Appr n", "P3\u2192Appr denom", "P3\u2192Appr CI Low", "P3\u2192Appr CI High",
        "Appr\u2192Mkt Rate", "Appr\u2192Mkt n", "Appr\u2192Mkt denom", "Appr\u2192Mkt CI Low", "Appr\u2192Mkt CI High",
    ]

    def _write_grid_sheet(ws, labels, rates, nums, denoms, ci_bounds):
        ws.append(headers)
        for i, label in enumerate(labels):
            row = [label]
            for j in range(4):
                r = rates[i, j]
                rate_val = r if not np.isnan(r) else None
                lo, hi = ci_bounds[i][j]
                row.extend([rate_val, int(nums[i, j]), int(denoms[i, j]),
                            lo if rate_val is not None else None,
                            hi if rate_val is not None else None])
            ws.append(row)

    # Heatmap sheets
    if "heatmap_modality" in export_data:
        mod_labels, _, mod_rates, mod_denoms, mod_nums, mod_cis = export_data["heatmap_modality"]
        ws = wb.create_sheet(title="Modality Heatmap")
        _write_grid_sheet(ws, mod_labels, mod_rates, mod_nums, mod_denoms, mod_cis)

    if "heatmap_disease" in export_data:
        da_labels, _, da_rates, da_denoms, da_nums, da_cis = export_data["heatmap_disease"]
        ws = wb.create_sheet(title="Disease Heatmap")
        _write_grid_sheet(ws, da_labels, da_rates, da_nums, da_denoms, da_cis)

    # Modality Bubble views per disease area
    bio_qls_funnel = export_data.get("heatmap_modality_bubble_funnel")
    if bio_qls_funnel:
        for da in sorted(bio_qls_funnel.by_disease_area.keys()):
            sheet_name = f"Modality - {da}"[:31]
            labels, _, rates, denoms, nums, cis = _compute.heatmap_grid(
                bio_qls_funnel, disease_area=da
            )
            if labels:
                ws = wb.create_sheet(title=sheet_name)
                _write_grid_sheet(ws, labels, rates, nums, denoms, cis)

    # Disease Bubble views per modality
    bio_qls_funnel_da = export_data.get("heatmap_disease_bubble_funnel")
    if bio_qls_funnel_da:
        for mod in sorted(bio_qls_funnel_da.by_modality.keys()):
            sheet_name = f"Disease - {mod}"[:31]
            labels, _, rates, denoms, nums, cis = _compute.disease_heatmap_grid(
                bio_qls_funnel_da, modality=mod
            )
            if labels:
                ws = wb.create_sheet(title=sheet_name)
                _write_grid_sheet(ws, labels, rates, nums, denoms, cis)

    # LOA sheets
    def _write_table_sheet(title, rows):
        if rows:
            ws = wb.create_sheet(title=title[:31])
            ws.append(list(rows[0].keys()))
            for row in rows:
                ws.append(list(row.values()))

    _write_table_sheet("LOA by Disease", export_data.get("loa_by_disease"))
    _write_table_sheet("LOA by Modality", export_data.get("loa_by_modality"))
    _write_table_sheet("Oncology vs Non-Oncology", export_data.get("oncology_comparison"))

    # Development Timelines
    timelines = export_data.get("timelines")
    if timelines:
        ws = wb.create_sheet(title="Development Timelines")
        ws.append(["Disease Area", "P1\u2192P2 (yr)", "P2\u2192P3 (yr)", "P3\u2192Appr (yr)",
                   "Appr\u2192Mkt (yr)", "Total P1\u2192Appr (yr)"])
        for name, fs in sorted(timelines.items()):
            durs = [t.avg_duration_years for t in fs.transitions]
            clinical_total = sum(d or 0 for d in durs[:3])
            ws.append([name] + list(durs) + [clinical_total if any(d is not None for d in durs[:3]) else None])

    _write_table_sheet("Time Period LOA", export_data.get("time_period_loa"))
    _write_table_sheet("Modality Proportion", export_data.get("modality_proportion"))
    _write_table_sheet("Sponsor Concentration", export_data.get("sponsor_concentration"))

    xlsx_path = os.path.join(output_path, "heatmap_data.xlsx")
    wb.save(xlsx_path)


# ---------------------------------------------------------------------------
# Analytical-snapshot outputs (Parquet + JSON manifest).
#
# Sibling of the CSV writers above, but typed for downstream analytical
# tooling: list-valued fields stay as `list[str]`, dates stay as `date`,
# bools stay as bool. The CSV writers keep their pipe-joined / ISO-string
# layout for human review and the existing HTML report.
# ---------------------------------------------------------------------------


def _split_pipe_joined(value: Optional[str]) -> list[str]:
    """Split a `" | "`-joined string into a list, preserving empties as `[]`.

    `Candidate.opentargets_moa` and `Candidate.opentargets_action_type`
    are stored as pipe-joined strings inside the OT enrichment but each
    represents one concept holding multiple values. The Parquet snapshot
    rehydrates them into native lists so downstream consumers can
    `.explode()` / `.len()` uniformly across all enrichment columns.
    """
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def _adjudication_lookup(
    candidate, cache,
) -> tuple[Optional[CandidateOutcomeRecord], Optional[CandidateOutcomeRecord]]:
    """Return (llm_direct, fda_timeline) records from the shared cache.

    Mirrors `_candidate_summary._lookup_cached_outcomes` so the snapshot
    surfaces both methods side-by-side. Either side may be `None` when
    that adjudicator hasn't been run against this candidate's
    drug/indication/phase.
    """
    if cache is None:
        return None, None
    from ...knowledge_cache import KnowledgeCache
    key = KnowledgeCache.make_adjudication_key(
        candidate.drug_name,
        candidate.indication,
        candidate.highest_phase.value,
    )
    return cache.get_outcome(key, candidate.candidate_id), cache.get_fda_outcome(
        key, candidate.candidate_id
    )


def _record_columns(record: Optional[CandidateOutcomeRecord]) -> dict:
    """Flatten a CandidateOutcomeRecord into snapshot columns (or empty defaults)."""
    if record is None:
        return {
            "outcome": None,
            "confidence": None,
            "reasoning": None,
            "evidence_sources": [],
            "approval_date": None,
            "commercialization_date": None,
        }
    return {
        "outcome": record.outcome.value,
        "confidence": record.confidence,
        "reasoning": record.reasoning,
        "evidence_sources": list(record.evidence_sources),
        "approval_date": record.approval_date,
        "commercialization_date": record.commercialization_date,
    }


def write_candidate_parquet(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
    outcome_table: OutcomeTable,
    output_path: str,
    *,
    cache=None,
    adjudication_method: str = "fda_timeline",
) -> None:
    """Write `candidate_detail.parquet` — analytical snapshot, native types.

    One row per candidate. List columns (drug_targets, opentargets_*,
    trial_ids, sponsors, mesh_*, evidence_sources) are stored as native
    `list[str]`. Dates as `date`. Both adjudicator outcomes
    (`llm_direct_*`, `fda_timeline_*`) are surfaced when present in the
    KnowledgeCache; the active-method record from `outcome_table`
    overrides the cache for its own column block.
    """
    import pandas as pd

    rows: list[dict] = []
    for c in candidate_table.candidates:
        attrs = attribute_table.attributes.get(c.candidate_id)
        active = outcome_table.outcomes.get(c.candidate_id)
        llm_cached, fda_cached = _adjudication_lookup(c, cache)

        # Active-run record wins for its own method's column block; the
        # other method falls back to whatever the cache has (or None).
        if adjudication_method == "llm_direct" and active is not None:
            llm_record, fda_record = active, fda_cached
        elif adjudication_method == "fda_timeline" and active is not None:
            llm_record, fda_record = llm_cached, active
        else:
            llm_record, fda_record = llm_cached, fda_cached

        llm_cols = _record_columns(llm_record)
        fda_cols = _record_columns(fda_record)
        if llm_cols["outcome"] is None or fda_cols["outcome"] is None:
            outcomes_agree: bool | None = None
        else:
            outcomes_agree = llm_cols["outcome"] == fda_cols["outcome"]

        active_cols = _record_columns(active)

        rows.append({
            # Identity / clustering
            "candidate_id": c.candidate_id,
            "drug_name": c.drug_name,
            "drug_name_raw": c.drug_name_raw,
            "indication": c.indication,
            "highest_phase": c.highest_phase.value,
            "trial_count": len(c.trial_ids),
            "trial_ids": list(c.trial_ids),
            "sponsors": list(c.sponsors),
            "earliest_start_date": c.earliest_start_date,
            "latest_completion_date": c.latest_completion_date,
            # Cross-references
            "drugbank_id": c.drugbank_id,
            "mesh_drug": c.mesh_drug,
            "mesh_indication": c.mesh_indication,
            "mesh_condition_tree_numbers": list(c.mesh_condition_tree_numbers),
            # SMILES / ChEMBL
            "smiles": c.smiles,
            "drug_targets": list(c.drug_targets),
            "target_names": list(c.target_names),
            # ICD-10
            "icd10_code": c.icd10_code,
            "icd10_description": c.icd10_description,
            # OpenTargets
            "opentargets_moa": _split_pipe_joined(c.opentargets_moa),
            "opentargets_action_type": _split_pipe_joined(c.opentargets_action_type),
            "opentargets_targets": list(c.opentargets_targets),
            "opentargets_pathways": list(c.opentargets_pathways),
            "opentargets_indication_max_phase": c.opentargets_indication_max_phase,
            # Classification
            "modality": attrs.drug_modality if attrs else None,
            "disease_area": attrs.disease_area if attrs else None,
            "modality_confidence": attrs.modality_confidence if attrs else None,
            "disease_confidence": attrs.disease_confidence if attrs else None,
            "modality_reasoning": attrs.reasoning if attrs else None,
            # Active-run outcome
            "outcome": active_cols["outcome"],
            "outcome_confidence": active_cols["confidence"],
            "outcome_reasoning": active_cols["reasoning"],
            "outcome_evidence_sources": active_cols["evidence_sources"],
            "approval_date": active_cols["approval_date"],
            "commercialization_date": active_cols["commercialization_date"],
            # llm_direct adjudicator
            "llm_direct_outcome": llm_cols["outcome"],
            "llm_direct_confidence": llm_cols["confidence"],
            "llm_direct_reasoning": llm_cols["reasoning"],
            "llm_direct_evidence_sources": llm_cols["evidence_sources"],
            "llm_direct_approval_date": llm_cols["approval_date"],
            "llm_direct_commercialization_date": llm_cols["commercialization_date"],
            # fda_timeline adjudicator
            "fda_timeline_outcome": fda_cols["outcome"],
            "fda_timeline_confidence": fda_cols["confidence"],
            "fda_timeline_reasoning": fda_cols["reasoning"],
            "fda_timeline_evidence_sources": fda_cols["evidence_sources"],
            "fda_timeline_approval_date": fda_cols["approval_date"],
            "fda_timeline_commercialization_date": fda_cols["commercialization_date"],
            "outcomes_agree": outcomes_agree,
        })

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(output_path, "candidate_detail.parquet")
    df.to_parquet(parquet_path, index=False)


def write_trial_parquet(
    candidate_table: CandidateTable,
    attribute_table: AttributeTable,
    outcome_table: OutcomeTable,
    trial_table: TrialTable | None,
    output_path: str,
) -> None:
    """Write `trial_detail.parquet` — same scope as trial_detail.csv, native types."""
    import pandas as pd

    if trial_table is None:
        return

    trial_index = {t.nct_id: t for t in trial_table.trials}

    rows: list[dict] = []
    for c in candidate_table.candidates:
        attrs = attribute_table.attributes.get(c.candidate_id)
        out = outcome_table.outcomes.get(c.candidate_id)

        cand_cols = {
            "candidate_id": c.candidate_id,
            "candidate_drug": c.drug_name,
            "candidate_drug_raw": c.drug_name_raw,
            "candidate_indication": c.indication,
            "candidate_modality": attrs.drug_modality if attrs else None,
            "candidate_disease_area": attrs.disease_area if attrs else None,
            "candidate_outcome": out.outcome.value if out else None,
            "candidate_highest_phase": c.highest_phase.value,
            "candidate_drugbank_id": c.drugbank_id,
            "candidate_mesh_drug": c.mesh_drug,
            "candidate_mesh_indication": c.mesh_indication,
        }

        if not c.trial_ids:
            rows.append({
                **cand_cols,
                "nct_id": None,
                "trial_intervention": None,
                "trial_indication": None,
                "trial_phase": None,
                "trial_status": None,
                "trial_mesh_intervention_terms": [],
                "trial_mesh_condition_terms": [],
                "trial_mesh_condition_tree_numbers": [],
                "trial_start_date": None,
                "trial_completion_date": None,
                "trial_is_single_arm": None,
                "trial_sponsor": None,
                "trial_title": None,
            })
            continue

        for nct in c.trial_ids:
            t = trial_index.get(nct)
            if t is None:
                rows.append({
                    **cand_cols,
                    "nct_id": nct,
                    "trial_intervention": None,
                    "trial_indication": None,
                    "trial_phase": None,
                    "trial_status": None,
                    "trial_mesh_intervention_terms": [],
                    "trial_mesh_condition_terms": [],
                    "trial_mesh_condition_tree_numbers": [],
                    "trial_start_date": None,
                    "trial_completion_date": None,
                    "trial_is_single_arm": None,
                    "trial_sponsor": None,
                    "trial_title": None,
                })
                continue

            rows.append({
                **cand_cols,
                "nct_id": t.nct_id,
                "trial_intervention": t.intervention,
                "trial_indication": t.indication,
                "trial_phase": t.phase.value,
                "trial_status": t.status.value,
                "trial_mesh_intervention_terms": list(t.mesh_intervention_terms),
                "trial_mesh_condition_terms": list(t.mesh_condition_terms),
                "trial_mesh_condition_tree_numbers": list(t.mesh_condition_tree_numbers),
                "trial_start_date": t.start_date,
                "trial_completion_date": t.completion_date,
                "trial_is_single_arm": bool(t.is_single_arm),
                "trial_sponsor": t.sponsor,
                "trial_title": t.title,
            })

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(output_path, "trial_detail.parquet")
    df.to_parquet(parquet_path, index=False)


def write_run_manifest(output_path: str, payload: dict) -> None:
    """Write `run_manifest.json` — pipeline config + snapshot versions for the run.

    Pure I/O. The orchestrator is responsible for assembling `payload`
    (timestamp, git SHA, config, snapshot versions, row counts).
    """
    manifest_path = os.path.join(output_path, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
