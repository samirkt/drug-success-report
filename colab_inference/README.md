# NDC adjudication on Colab GPU

Self-contained payload to offload the per-candidate LLM call in
`pipeline.ndc.NDCAdjudicator.match_indication` from local Ollama to a
Colab GPU runtime.

## Flow

```
LOCAL                                  COLAB                          LOCAL
-----                                  -----                          -----
export_ndc_work.py                                                    import_ndc_verdicts.py
  - clusters + bulk fda.db lookup     run_inference.ipynb              - read verdicts.jsonl
  - writes work_units.jsonl             - load HF model                - cache.put_ndc_outcome(...)
  - writes clustered.pkl                - greedy decode each unit       - next pipeline run is a
  -zip->                                - writes verdicts.jsonl  ->     100% cache hit, no LLM
```

## Step 1 -- Local export

From the project root:

```bash
python -m scripts.export_ndc_work \
    --source aact --keyword peptide \
    --max-candidates 100 --sample-seed 42 \
    --year-range 2000-2006 \
    --output /tmp/work_units.jsonl \
    --candidates-pickle /tmp/clustered.pkl
```

Skip-already-cached for incremental fill-in across waves:

```bash
python -m scripts.export_ndc_work ... --skip-cached
```

## Step 2 -- Bundle + upload

```bash
zip -j /tmp/colab_bundle.zip /tmp/work_units.jsonl
zip -r /tmp/colab_bundle.zip colab_inference/
```

Upload `colab_bundle.zip` to a Colab session (Runtime > Change runtime
type > GPU; T4 is fine for 7B/14B, A100 for 32B+).

## Step 3 -- Run inference in Colab

In a Colab cell:

```python
!unzip -o /content/colab_bundle.zip -d /content/
%cd /content/colab_inference
!pip install -q -r requirements.txt
!python run_inference.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --work /content/work_units.jsonl \
    --out  /content/verdicts.jsonl
```

Then download `/content/verdicts.jsonl`.

Use `run_inference.ipynb` for a click-through version of the same flow.

## Step 4 -- Local import

```bash
python -m scripts.import_ndc_verdicts \
    --verdicts /tmp/verdicts.jsonl \
    --candidates /tmp/clustered.pkl \
    --cache knowledge_cache.db
```

The next normal pipeline run with `--adjudication-method ndc_indication`
against the same candidate set will hit the cache for every imported
verdict -- zero LLM calls, no Ollama required.

## Notes

- Failed candidates (parse error, OOM, generation error) are omitted
  from `verdicts.jsonl`. The local pipeline will fall back to a normal
  LLM call for those on the next run.
- `temperature=0` + `do_sample=False` makes generation deterministic
  given a fixed model, weights and tokenizer.
- This payload has **no imports from the parent project** so it can be
  zipped and shipped independently.
