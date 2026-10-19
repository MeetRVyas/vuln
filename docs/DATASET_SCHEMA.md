# Dataset schema

The actual labeled vulnerability dataset's on-disk format wasn't decided when this was
built. Nothing in `vulnrem.evaluation`, `vulnrem.finetuning`, or the CLI's
`dataset-split`/`finetune-data-prep`/`eval-run` commands touches the raw format directly
-- they all consume `list[CanonicalRecord]` (`vulnrem/evaluation/schemas.py`), produced
by a `DatasetAdapter` (`vulnrem/evaluation/dataset_adapter/base.py`).

## The canonical schema

```python
class CanonicalRecord(BaseModel):
    record_id: str            # stable, unique identifier for this example
    language: Language        # "python" (only language implemented in this build)
    cwe_id: str                # e.g. "CWE-89"
    vulnerable_code: str        # the vulnerable snippet/function/file
    fixed_code: str | None      # the ground-truth patched version, if the dataset has one
    description: str            # free text, optional
    source_dataset: str          # e.g. "bigvul", "custom" -- just provenance, not used for logic
```

`fixed_code` is optional at the schema level (a pure-detection dataset without ground-truth
fixes is still usable for the evaluation harness's *detection* recall), but is **required**
for anything that builds Patcher fine-tuning examples
(`vulnrem.finetuning.data_prep.build_patcher_training_example` raises
`DatasetError` if it's missing).

## If your dataset is already close to this shape: use the generic adapter

`GenericJsonLinesAdapter` (`vulnrem/evaluation/dataset_adapter/generic_json.py`) reads one
JSON object per line and accepts a `field_map` for whatever key names your file actually
uses:

```python
from vulnrem.evaluation.dataset_adapter.registry import build_dataset_adapter

adapter = build_dataset_adapter(
    "generic_jsonl",
    path="data/my_dataset.jsonl",
    field_map={
        "record_id": "id",
        "language": "lang",
        "cwe_id": "cwe",
        "vulnerable_code": "func_before",
        "fixed_code": "func_after",
    },
    source_dataset_name="my_dataset",
)
records = adapter.load()
```

By default, a record that fails to parse (unknown language, a missing required field) is
skipped with a warning log rather than aborting the whole load; pass `strict=True` to the
adapter to raise on the first bad record instead.

## If your dataset's format is genuinely different: write a new adapter

Implement `DatasetAdapter.load() -> list[CanonicalRecord]` (see
`GenericJsonLinesAdapter` as a template for the pattern -- read the raw format, validate,
skip-or-raise on malformed rows) and register it:

```python
# vulnrem/evaluation/dataset_adapter/registry.py
DATASET_ADAPTER_REGISTRY["my_format"] = lambda **kwargs: MyAdapter(**kwargs)
```

`bigvul`, `cvefixes`, `crossvul`, `diversevul`, and `primevul` are already registered as
named placeholders (`evaluation/dataset_adapter/not_implemented.py`) that raise a clear
error until a real adapter replaces them -- this is where to add one of those specific
public datasets.

## What happens after loading

1. **Split** (`vulnrem.evaluation.splitter.stratified_split`): a deterministic,
   `(language, cwe_id)`-stratified, non-overlapping split into a fine-tuning set and a
   held-out evaluation set, saved as a checksummed `SplitManifest` JSON file. Re-running
   the split against a dataset whose checksum has changed is caught by
   `verify_no_leakage` rather than silently producing a different split.
2. **Fine-tuning data prep** (`vulnrem.finetuning.data_prep.write_training_jsonl`): only
   for the train split, builds one Patcher instruction-tuning example per record by
   calling the *real* `pipeline.prompts.patcher.build_request` function -- so the model is
   trained on exactly the prompt shape it's served with.
3. **Evaluation** (`vulnrem.evaluation.harness.EvaluationHarness`): only for the eval
   split, runs each record's `vulnerable_code` through the real, production
   `FileCheckRunner` and derives detection/remediation outcomes from the same
   `FileCheckReport` the dashboard would show a user.

All three CLI commands (`vulnrem dataset-split`, `vulnrem finetune-data-prep`,
`vulnrem eval-run`) wire this up end to end -- see `vulnrem/cli/main.py`.
