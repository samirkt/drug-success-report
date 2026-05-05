"""Quick interactive smoke test + accuracy benchmark for the NDC adjudicator.

Prereqs: local fda.db built (set FDA_DB env or default ./fda.db), and an
LLM backend reachable at FDA_LLM_BASE_URL (defaults to local Ollama).

Edit TEST_CASES / MODEL below and run:  python scripts/try_ndc_adjudication.py
"""
import time
from pipeline.ndc import NDCAdjudicator
from pipeline.fda import OpenAICompatJSONClient

MODEL = "qwen2.5:7b-instruct"

# (synonyms, indication, expected_approved, note)
TEST_CASES = [
    # --- Oncology, on-label ---
    (["pembrolizumab", "Keytruda"], "melanoma", True, "anti-PD-1, melanoma"),
    (["pembrolizumab", "Keytruda"], "metastatic non-small cell lung cancer", True, "NSCLC"),
    (["pembrolizumab", "Keytruda"], "Hodgkin lymphoma", True, "classical HL"),
    (["pembrolizumab", "Keytruda"], "head and neck squamous cell carcinoma", True, "HNSCC"),
    (["nivolumab", "Opdivo"], "melanoma", True, "anti-PD-1"),
    (["nivolumab", "Opdivo"], "non-small cell lung cancer", True, "NSCLC"),
    (["ipilimumab", "Yervoy"], "melanoma", True, "anti-CTLA-4"),
    (["trastuzumab", "Herceptin"], "HER2-positive breast cancer", True, "HER2+ breast"),
    (["imatinib", "Gleevec"], "chronic myeloid leukemia", True, "CML"),
    (["imatinib", "Gleevec"], "gastrointestinal stromal tumor", True, "GIST"),
    (["rituximab", "Rituxan"], "non-Hodgkin lymphoma", True, "anti-CD20, NHL"),
    (["erlotinib", "Tarceva"], "non-small cell lung cancer", True, "EGFR inhibitor"),
    (["osimertinib", "Tagrisso"], "EGFR-mutated non-small cell lung cancer", True, "EGFR T790M"),
    (["bevacizumab", "Avastin"], "metastatic colorectal cancer", True, "VEGF, mCRC"),
    (["ibrutinib", "Imbruvica"], "chronic lymphocytic leukemia", True, "BTK, CLL"),
    (["sorafenib", "Nexavar"], "hepatocellular carcinoma", True, "HCC"),
    (["sunitinib", "Sutent"], "renal cell carcinoma", True, "RCC"),
    (["lenalidomide", "Revlimid"], "multiple myeloma", True, "IMiD, MM"),

    # --- Cardio / diabetes, on-label ---
    (["metformin", "Glucophage"], "type 2 diabetes mellitus", True, "first-line T2D"),
    (["semaglutide", "Ozempic"], "type 2 diabetes mellitus", True, "GLP-1"),
    (["liraglutide", "Victoza"], "type 2 diabetes mellitus", True, "GLP-1"),
    (["empagliflozin", "Jardiance"], "type 2 diabetes mellitus", True, "SGLT2"),
    (["atorvastatin", "Lipitor"], "primary hypercholesterolemia", True, "statin"),
    (["rosuvastatin", "Crestor"], "primary hypercholesterolemia", True, "statin"),
    (["apixaban", "Eliquis"], "nonvalvular atrial fibrillation", True, "Xa inhibitor, AF"),
    (["clopidogrel", "Plavix"], "acute coronary syndrome", True, "ADP inhibitor"),
    (["warfarin", "Coumadin"], "atrial fibrillation", True, "VKA, AF prophylaxis"),

    # --- Pain / inflammation, on-label ---
    (["aspirin"], "pain", True, "OTC analgesic"),
    (["ibuprofen", "Advil", "Motrin"], "pain", True, "NSAID"),
    (["ibuprofen", "Advil"], "fever", True, "NSAID antipyretic"),
    (["acetaminophen", "Tylenol"], "pain", True, "OTC analgesic"),
    (["acetaminophen", "Tylenol"], "fever", True, "OTC antipyretic"),
    (["celecoxib", "Celebrex"], "osteoarthritis", True, "COX-2, OA"),
    (["naproxen", "Aleve"], "pain", True, "NSAID"),

    # --- Mental health, on-label ---
    (["sertraline", "Zoloft"], "major depressive disorder", True, "SSRI"),
    (["fluoxetine", "Prozac"], "major depressive disorder", True, "SSRI"),
    (["duloxetine", "Cymbalta"], "major depressive disorder", True, "SNRI"),
    (["escitalopram", "Lexapro"], "major depressive disorder", True, "SSRI"),
    (["aripiprazole", "Abilify"], "schizophrenia", True, "atypical antipsychotic"),

    # --- Other on-label ---
    (["albuterol", "ProAir"], "asthma", True, "SABA"),
    (["montelukast", "Singulair"], "asthma", True, "leukotriene antagonist"),
    (["omeprazole", "Prilosec"], "gastroesophageal reflux disease", True, "PPI, GERD"),
    (["sildenafil", "Viagra"], "erectile dysfunction", True, "PDE5"),
    (["tadalafil", "Cialis"], "erectile dysfunction", True, "PDE5"),
    (["methotrexate"], "rheumatoid arthritis", True, "DMARD"),
    (["adalimumab", "Humira"], "rheumatoid arthritis", True, "anti-TNF"),
    (["adalimumab", "Humira"], "Crohn disease", True, "anti-TNF"),
    (["etanercept", "Enbrel"], "rheumatoid arthritis", True, "anti-TNF"),
    (["amoxicillin"], "acute otitis media", True, "antibiotic, AOM"),

    # --- Wrong indication: real drug, not approved for this disease ---
    (["pembrolizumab", "Keytruda"], "Alzheimer disease", False, "cancer drug, no AD"),
    (["pembrolizumab", "Keytruda"], "type 2 diabetes", False, "cancer drug, no T2D"),
    (["pembrolizumab", "Keytruda"], "depression", False, "cancer drug, no MDD"),
    (["aspirin"], "type 2 diabetes", False, "OTC analgesic, no T2D"),
    (["aspirin"], "Alzheimer disease", False, "no AD indication"),
    (["metformin", "Glucophage"], "breast cancer", False, "investigational only"),
    (["sildenafil", "Viagra"], "Alzheimer disease", False, "PDE5, no AD"),
    (["atorvastatin", "Lipitor"], "depression", False, "statin, no MDD"),
    (["trastuzumab", "Herceptin"], "non-small cell lung cancer", False, "HER2+ breast/gastric only"),
    (["semaglutide", "Ozempic"], "Alzheimer disease", False, "investigational only"),
    (["methotrexate"], "major depressive disorder", False, "no MDD indication"),
    (["ibuprofen"], "Alzheimer disease", False, "NSAID, no AD"),

    # --- Fake / unknown drugs (deterministic short-circuit) ---
    (["neverapproved-xyz"], "metastatic NSCLC", False, "fake drug"),
    (["imaginarytinib"], "chronic myeloid leukemia", False, "fake drug"),
    (["fakedrugmab"], "melanoma", False, "fake drug"),
    (["totallymadeupname-001"], "depression", False, "fake drug"),
]

adj = NDCAdjudicator(OpenAICompatJSONClient(
    base_url="http://localhost:11434/v1", model=MODEL))
print(f"Model: {MODEL}  |  {len(TEST_CASES)} cases\n" + "-" * 100)

correct = fp = fn = 0
total_time = 0.0
for synonyms, indication, expected, note in TEST_CASES:
    t0 = time.monotonic()
    v = adj.adjudicate(synonyms, indication)
    dt = time.monotonic() - t0
    total_time += dt
    ok = v.approved == expected
    correct += ok
    fp += (v.approved and not expected)
    fn += (not v.approved and expected)
    mark = "✓" if ok else "✗"
    print(f"{mark} [{dt:5.1f}s] exp={expected!s:5s} got={v.approved!s:5s} | "
          f"{synonyms[0]:20s} / {indication[:40]:40s} | {note}")

n = len(TEST_CASES)
print("\n" + "-" * 100)
print(f"Accuracy: {correct}/{n} = {100*correct/n:.0f}%  |  "
      f"FP: {fp} (wrongly approved)  |  FN: {fn} (missed approval)")
print(f"Time: avg {total_time/n:.1f}s/call  |  total {total_time:.0f}s")
