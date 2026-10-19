from __future__ import annotations

import json

import pytest

from vulnrem.core.enums import Language
from vulnrem.core.errors import DatasetError
from vulnrem.evaluation.schemas import CanonicalRecord
from vulnrem.finetuning.data_prep import (
    build_patcher_training_example,
    build_test_writer_training_example,
    write_training_jsonl,
)
from vulnrem.languages.base import CWESpec


def _record(**overrides) -> CanonicalRecord:
    defaults = dict(
        record_id="r1",
        language=Language.PYTHON,
        cwe_id="CWE-89",
        vulnerable_code="def f(x):\n    return 'SELECT * FROM t WHERE x=' + x\n",
        fixed_code="def f(x):\n    return conn.execute('SELECT * FROM t WHERE x=?', (x,))\n",
        description="sql injection",
    )
    defaults.update(overrides)
    return CanonicalRecord(**defaults)


def test_build_patcher_example_reuses_real_prompt_and_appends_target():
    record = _record()
    cwe_spec = CWESpec(id="CWE-89", name="SQL Injection", description="desc")

    example = build_patcher_training_example(record, cwe_spec)

    assert example["record_id"] == "r1"
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]

    assistant_payload = json.loads(example["messages"][-1]["content"])
    assert assistant_payload["patched_source"] == record.fixed_code

    # the user message really is the same content the real Patcher prompt builds
    assert record.vulnerable_code in example["messages"][1]["content"]
    assert "CWE-89" in example["messages"][1]["content"]


def test_build_patcher_example_raises_without_fixed_code():
    record = _record(fixed_code=None)
    with pytest.raises(DatasetError):
        build_patcher_training_example(record)


def test_test_writer_example_is_a_documented_not_yet_implemented_extension_point():
    with pytest.raises(NotImplementedError):
        build_test_writer_training_example(_record(), reproduction_poc_script="whatever")


def test_write_training_jsonl_skips_records_without_fixed_code(tmp_path):
    records = [_record(record_id="r1"), _record(record_id="r2", fixed_code=None)]
    output = tmp_path / "patcher_train.jsonl"

    count = write_training_jsonl(records, output, cwe_lookup={})

    assert count == 1
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["record_id"] == "r1"
