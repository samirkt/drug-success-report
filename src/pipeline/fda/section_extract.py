"""Pre-LLM narrowing of label and approval-letter text.

The LLM extraction prompt only cares about the indication clause, but the
inputs we feed it carry a lot of unrelated content:

  - Approval letter PDFs (15-30k chars) include letterhead, reference
    blocks, REMS / pediatric / postmarketing sections, and signatures.
  - Label `indications_and_usage` text from openFDA already starts narrow,
    but commonly contains a "Limitations of Use" subsection that the LLM
    is supposed to ignore.

These helpers shrink the input to just the relevant text BEFORE it hits
the LLM. Both helpers are conservative — if the heuristic can't find a
clean target, the original text is returned untouched so the LLM still
has a chance to extract from the full source.
"""

from __future__ import annotations

import re

# Phrases that mark indication-clause text in approval letters. Matches
# the common opening verbs ("provides for", "is indicated for", "approve
# your", "treatment of", "in combination with", "new indication").
_LETTER_ANCHORS = re.compile(
    r"\b("
    r"provides? for"
    r"|indicat(?:ed|ion)"
    r"|approv(?:ed|es) (?:your|the|this)"
    r"|new indication"
    r"|treatment of"
    r"|in combination with"
    r")\b",
    re.IGNORECASE,
)

# Sections that always appear AFTER the indication clause in FDA letters.
# Truncate scanning here to avoid pulling boilerplate windows.
_LETTER_TAIL = re.compile(
    r"\b("
    r"REMS"
    r"|Risk Evaluation and Mitigation"
    r"|Pediatric Research Equity Act"
    r"|Postmarketing (?:Requirements|Commitments)"
    r"|Required Pediatric Assessments"
    r"|Sincerely,"
    r")\b",
    re.IGNORECASE,
)

# SPL section headers that appear AFTER the indications-and-usage clause.
# Truncating at the first such header drops dosing, contraindications,
# warnings, and limitations boilerplate — none of which the extract LLM
# should consider, all of which inflate prompt tokens. The extract prompt
# already says to ignore limitations, but stripping at the input layer is
# more robust than relying on the model and saves tokens regardless.
_LABEL_TAIL = re.compile(
    r"\b("
    r"Limitations? of Use|Limitations? of Usage"
    r"|Important Limitations"
    r"|Dosage and Administration"
    r"|Dosage Forms? and Strengths?"
    r"|Contraindications"
    r"|Warnings? and Precautions"
    r"|Adverse Reactions"
    r")\b",
    re.IGNORECASE,
)


def extract_letter_indication_clause(
    text: str,
    *,
    window_chars: int = 500,
    max_total_chars: int = 2500,
    min_input_chars: int = 4000,
) -> str:
    """Heuristically narrow approval-letter text to the indication clause.

    Strategy: stop scanning at the first post-indication boilerplate
    section, find anchor phrases (e.g. "provides for", "is indicated for"),
    take a +/-window around each match, merge overlapping windows.

    Falls back to the original text if:
      - input is shorter than `min_input_chars` (already short enough),
      - no anchors matched (would risk losing the indication entirely),
      - the merged extraction is somehow longer than `max_total_chars` or
        90% of the original (no real win).
    """
    if not text or len(text) < min_input_chars:
        return text

    boundary = _LETTER_TAIL.search(text)
    scan_text = text[: boundary.start()] if boundary else text

    matches = list(_LETTER_ANCHORS.finditer(scan_text))
    if not matches:
        return text

    windows: list[list[int]] = []
    for m in matches:
        start = max(0, m.start() - window_chars)
        end = min(len(scan_text), m.end() + window_chars)
        windows.append([start, end])

    windows.sort()
    merged: list[list[int]] = [windows[0]]
    for start, end in windows[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    parts = [scan_text[s:e].strip() for s, e in merged]
    extracted = "\n[...]\n".join(p for p in parts if p)

    if not extracted or len(extracted) > max_total_chars or len(extracted) >= len(text) * 0.9:
        return text
    return extracted


def narrow_label_indications(text: str) -> str:
    """Strip the 'Limitations of Use' subsection from label text.

    SPL `indications_and_usage` sections typically have the structure:

        1 INDICATIONS AND USAGE
        Drug X is indicated for ...
        Limitations of Use
        Drug X is NOT indicated for ...

    The LLM prompt already says to ignore limitations, but the model
    sometimes drifts and includes them. Stripping at the input layer is
    more robust and saves tokens.

    Falls back to the original text if no header is found, or if dropping
    the tail would leave nothing meaningful.
    """
    if not text:
        return text

    match = _LABEL_TAIL.search(text)
    if not match:
        return text

    head = text[: match.start()].rstrip()
    if len(head) < 50:
        return text
    return head
