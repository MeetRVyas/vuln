"""Builds fine-tuning training examples by calling the *same*
prompt-builder functions the pipeline uses at serving time
(:mod:`vulnrem.pipeline.prompts`), then appending the ground-truth
assistant completion. This guarantees the fine-tuned adapter is trained on
exactly the prompt shape ``generate_structured`` will actually send it in
production -- there is no second, drifting copy of the prompt template.

Only the Patcher example-builder is fully implemented. A Test-Writer
training example additionally needs a *confirmed dynamic reproduction*
(a working PoC script) per the Reproducer's contract
(:mod:`vulnrem.pipeline.prompts.reproducer`), which a generic
vulnerable/fixed-code dataset does not provide -- see
:func:`build_test_writer_training_example` for the extension point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vulnrem.core.enums import Severity
from vulnrem.core.errors import DatasetError
from vulnrem.evaluation.schemas import CanonicalRecord
from vulnrem.languages.base import CWESpec, LocalizationContext
from vulnrem.pipeline.prompts import patcher as patcher_prompt
from vulnrem.pipeline.prompts.schemas import PatcherOutput
from vulnrem.pipeline.state import Finding

logger = logging.getLogger(__name__)


def build_patcher_training_example(record: CanonicalRecord, cwe_spec: CWESpec | None = None) -> dict:
    if not record.fixed_code:
        raise DatasetError(
            f"Record {record.record_id!r} has no fixed_code -- the Patcher needs a ground-truth "
            "patch to train against."
        )

    lines = record.vulnerable_code.splitlines() or [""]
    # The generic dataset schema doesn't carry a line-level location for the
    # vulnerability, so the whole given snippet is treated as the
    # localization context -- a reasonable simplification given most public
    # vuln-fix datasets are already function-scoped (see docs/DATASET_SCHEMA.md).
    finding = Finding(
        finding_id=record.record_id,
        language=record.language,
        file_path=f"{record.record_id}.py",
        line_start=1,
        line_end=len(lines),
        cwe_id=record.cwe_id,
        severity=Severity.MEDIUM,
        tool_findings=[],
    )
    context = LocalizationContext(
        primary_snippet=record.vulnerable_code,
        primary_start_line=1,
        primary_end_line=len(lines),
        enclosing_symbol_name=None,
        related_snippets=[],
    )

    request = patcher_prompt.build_request(
        finding=finding,
        original_source=record.vulnerable_code,
        localization_context=context,
        cwe_spec=cwe_spec,
        attempt_history=[],
    )

    assistant_output = PatcherOutput(
        patched_source=record.fixed_code,
        explanation=f"Fixes {record.cwe_id}"
        + (f" ({record.description})" if record.description else "")
        + " using the dataset's ground-truth patch.",
    )

    messages = [m.model_dump() for m in request.messages]
    messages.append({"role": "assistant", "content": assistant_output.model_dump_json()})
    return {"record_id": record.record_id, "cwe_id": record.cwe_id, "messages": messages}


def build_test_writer_training_example(record: CanonicalRecord, reproduction_poc_script: str) -> dict:
    """Requires ``reproduction_poc_script`` -- a working PoC for this
    record's vulnerability -- to be supplied by the caller (e.g. a
    dataset-specific augmentation step, or a curated subset the Reproducer
    was run against once and confirmed). This is the honest boundary: a
    Test-Writer adapter cannot be trained from vulnerable/fixed code pairs
    alone."""
    raise NotImplementedError(
        "Test-Writer training-example construction needs a confirmed PoC per record, which the "
        "generic canonical schema does not provide. Populate `reproduction_poc_script` (e.g. from "
        "a curated subset run through vulnrem.pipeline.nodes.reproducer.ReproducerNode once) before "
        "implementing this the same way build_patcher_training_example calls the real prompt "
        "builder."
    )


def write_training_jsonl(
    records: list[CanonicalRecord], output_path: Path | str, cwe_lookup: dict[str, CWESpec]
) -> int:
    """Writes Patcher training examples for every record with a
    ``fixed_code``, skipping (with a warning) any that don't. Returns the
    number of examples written."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with target.open("w", encoding="utf-8") as fh:
        for record in records:
            if not record.fixed_code:
                logger.warning("training_record_skipped_no_fixed_code", extra={"record_id": record.record_id})
                continue
            example = build_patcher_training_example(record, cwe_lookup.get(record.cwe_id))
            fh.write(json.dumps(example) + "\n")
            written += 1
    return written
