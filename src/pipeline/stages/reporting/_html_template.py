"""HTML template and styling for BIO/QLS-style narrative reports."""

from __future__ import annotations

import base64
import re

REPORT_CSS = """
    :root {
        --primary: #1B3A5C;
        --accent: #4472C4;
        --text: #333;
        --light-bg: #f8f9fa;
        --border: #dee2e6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: "Georgia", "Times New Roman", serif;
        font-size: 11pt;
        line-height: 1.6;
        color: var(--text);
        max-width: 900px;
        margin: 0 auto;
        padding: 40px 60px;
        background: #fff;
    }
    /* Title page */
    .title-page {
        text-align: center;
        padding: 80px 0 60px 0;
        border-bottom: 3px solid var(--primary);
        margin-bottom: 40px;
        page-break-after: always;
    }
    .title-page h1 {
        font-size: 28pt;
        color: var(--primary);
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 12px;
    }
    .title-page .subtitle {
        font-size: 14pt;
        color: var(--accent);
        margin-top: 10px;
    }
    .title-page .date {
        font-size: 11pt;
        color: #666;
        margin-top: 20px;
    }
    /* Executive Summary */
    .executive-summary {
        background: var(--light-bg);
        border-left: 4px solid var(--primary);
        padding: 24px 30px;
        margin: 30px 0;
        page-break-inside: avoid;
    }
    .executive-summary h2 {
        font-size: 16pt;
        color: var(--primary);
        margin-bottom: 14px;
        border: none;
    }
    .executive-summary .subtitle {
        font-size: 11pt;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 10px;
    }
    .executive-summary ul {
        margin-left: 20px;
    }
    .executive-summary li {
        margin-bottom: 8px;
        line-height: 1.5;
    }
    /* Section headings */
    h2 {
        font-size: 16pt;
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
        padding-bottom: 6px;
        margin-top: 40px;
        margin-bottom: 16px;
    }
    h3 {
        font-size: 13pt;
        color: var(--primary);
        margin-top: 30px;
        margin-bottom: 12px;
    }
    /* Narrative text */
    .narrative {
        text-align: justify;
        margin-bottom: 16px;
    }
    .narrative p {
        margin-bottom: 12px;
    }
    /* Figures */
    figure {
        margin: 24px 0;
        text-align: center;
        page-break-inside: avoid;
    }
    figure img {
        max-width: 100%;
        height: auto;
        border: 1px solid var(--border);
        border-radius: 2px;
    }
    figcaption {
        font-size: 9pt;
        color: #666;
        font-style: italic;
        margin-top: 8px;
        text-align: center;
    }
    /* Tables */
    .table-container {
        margin: 24px 0;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    .table-caption {
        font-size: 9pt;
        color: #666;
        font-style: italic;
        margin-bottom: 6px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 9pt;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }
    th {
        background: var(--primary);
        color: white;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
        white-space: nowrap;
    }
    td {
        padding: 6px 12px;
        border-bottom: 1px solid var(--border);
    }
    tr:nth-child(even) td {
        background: var(--light-bg);
    }
    tr:last-child td {
        font-weight: 600;
        border-top: 2px solid var(--primary);
    }
    /* Interactive figures */
    .interactive-figure {
        margin: 24px 0;
    }
    /* Appendix */
    .appendix {
        margin-top: 60px;
        border-top: 3px solid var(--primary);
        padding-top: 20px;
    }
    .appendix h2 {
        border-bottom: none;
    }
    /* Table of contents */
    .toc {
        margin: 30px 0;
        page-break-after: always;
    }
    .toc h2 {
        border-bottom: 2px solid var(--primary);
    }
    .toc ul {
        list-style: none;
        margin: 0;
        padding: 0;
    }
    .toc li {
        padding: 4px 0;
        border-bottom: 1px dotted var(--border);
    }
    .toc li.part {
        font-weight: 700;
        color: var(--primary);
        margin-top: 12px;
        border-bottom: none;
        font-size: 11pt;
    }
    .toc li.section {
        padding-left: 20px;
        font-size: 10pt;
    }
    /* Print */
    @media print {
        body { padding: 20px 40px; }
        .title-page { page-break-after: always; }
        h2 { page-break-before: auto; }
        figure { page-break-inside: avoid; }
    }
"""


def html_head(title: str) -> str:
    """Return <!DOCTYPE> through <body> opening."""
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="utf-8">\n'
        f'  <title>{title}</title>\n'
        f'  <style>{REPORT_CSS}</style>\n'
        '</head>\n'
        '<body>\n'
    )


def title_page(title: str, subtitle: str, date_str: str) -> str:
    """Render the title page."""
    return (
        '<div class="title-page">\n'
        f'  <h1>{title}</h1>\n'
        f'  <div class="subtitle">{subtitle}</div>\n'
        f'  <div class="date">{date_str}</div>\n'
        '</div>\n'
    )


def executive_summary_block(summary_bullets: str) -> str:
    """Render the executive summary box."""
    # Convert bullet lines to HTML list
    bullets = []
    for line in summary_bullets.strip().split("\n"):
        line = line.strip()
        if line.startswith("* "):
            line = line[2:]
        if line:
            bullets.append(f"    <li>{line}</li>")

    return (
        '<div class="executive-summary">\n'
        '  <h2>Executive Summary</h2>\n'
        '  <div class="subtitle">Key Takeaways</div>\n'
        '  <ul>\n'
        + "\n".join(bullets) + "\n"
        '  </ul>\n'
        '</div>\n'
    )


def render_figure(img_bytes: bytes, caption: str, fig_num: int) -> str:
    """Render a figure with base64-encoded image and caption."""
    b64 = base64.b64encode(img_bytes).decode()
    return (
        '<figure>\n'
        f'  <img src="data:image/png;base64,{b64}" alt="{caption}" />\n'
        f'  <figcaption>Figure {fig_num}: {caption}</figcaption>\n'
        '</figure>\n'
    )


def render_table(rows: list[dict], caption: str, table_num: int) -> str:
    """Render a styled HTML table."""
    if not rows:
        return ""
    headers = list(rows[0].keys())

    parts = ['<div class="table-container">']
    parts.append(f'  <div class="table-caption">Table {table_num}: {caption}</div>')
    parts.append("  <table>")
    parts.append("    <tr>" + "".join(f"<th>{_format_header(h)}</th>" for h in headers) + "</tr>")
    for row in rows:
        cells = []
        for h in headers:
            val = row.get(h, "")
            cells.append(f"<td>{_format_cell(val)}</td>")
        parts.append("    <tr>" + "".join(cells) + "</tr>")
    parts.append("  </table>")
    parts.append("</div>")
    return "\n".join(parts) + "\n"


def render_html_figure(html_content: str, caption: str, fig_num: int) -> str:
    """Render an interactive (Plotly) figure with caption."""
    return (
        '<div class="interactive-figure">\n'
        f'  <figcaption>Figure {fig_num}: {caption}</figcaption>\n'
        f'  <div>{html_content}</div>\n'
        '</div>\n'
    )


def render_narrative(text: str, figures: dict[str, bytes],
                     tables: dict[str, list[dict]],
                     html_figures: dict[str, str],
                     fig_counter: list[int],
                     table_counter: list[int]) -> str:
    """Render narrative text, resolving {{figure:key:caption}} and {{table:key:caption}} refs."""
    # Split on template references
    pattern = r'\{\{(figure|table|html_figure):([^:}]+):([^}]*)\}\}'

    parts = []
    last_end = 0
    for match in re.finditer(pattern, text):
        # Add text before this match
        before_text = text[last_end:match.start()].strip()
        if before_text:
            paragraphs = before_text.split("\n\n")
            for p in paragraphs:
                p = p.strip()
                if p:
                    parts.append(f'<p class="narrative">{p}</p>\n')

        ref_type = match.group(1)
        key = match.group(2)
        caption = match.group(3)

        if ref_type == "figure" and key in figures:
            fig_counter[0] += 1
            parts.append(render_figure(figures[key], caption, fig_counter[0]))
        elif ref_type == "table" and key in tables:
            table_counter[0] += 1
            parts.append(render_table(tables[key], caption, table_counter[0]))
        elif ref_type == "html_figure" and key in html_figures:
            fig_counter[0] += 1
            parts.append(render_html_figure(html_figures[key], caption, fig_counter[0]))

        last_end = match.end()

    # Remaining text after last match
    remaining = text[last_end:].strip()
    if remaining:
        paragraphs = remaining.split("\n\n")
        for p in paragraphs:
            p = p.strip()
            if p:
                parts.append(f'<p class="narrative">{p}</p>\n')

    return "".join(parts)


def _format_header(h: str) -> str:
    """Format a table header: replace underscores, title case."""
    return h.replace("_", " ").title()


def _format_cell(val) -> str:
    """Format a table cell value."""
    if val is None:
        return "\u2014"
    if isinstance(val, float):
        if 0 <= val <= 1:
            return f"{val * 100:.1f}%"
        return f"{val:.2f}"
    return str(val)
