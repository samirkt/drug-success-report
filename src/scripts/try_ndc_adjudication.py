"""Quick interactive smoke test for the NDC adjudicator.

Prereqs: local fda.db built (set FDA_DB env or default ./fda.db), and an
LLM backend reachable at FDA_LLM_BASE_URL (defaults to local Ollama).

Edit TEST_CASES below and run:  python scripts/try_ndc_adjudication.py
"""
from pipeline.ndc import NDCAdjudicator
from pipeline.fda import OpenAICompatJSONClient

TEST_CASES = [
    (["pembrolizumab", "Keytruda"], "metastatic NSCLC"),
    (["pembrolizumab", "Keytruda"], "Alzheimer disease"),
    (["aspirin"], "fever"),
]

adj = NDCAdjudicator(OpenAICompatJSONClient(
    base_url="http://localhost:11434/v1", model="qwen2.5:14b-instruct"))
for synonyms, indication in TEST_CASES:
    v = adj.adjudicate(synonyms, indication)
    print(f"{indication!r:40s} -> approved={v.approved} ({v.confidence:.2f}) "
          f"matched={v.matched_synonym}/{v.matched_indication!r} | {v.reasoning}")
