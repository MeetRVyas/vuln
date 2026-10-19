from __future__ import annotations

import json

import pytest

from vulnrem.core.enums import Language
from vulnrem.core.errors import DatasetError, DatasetLeakageError, DatasetSchemaError
from vulnrem.evaluation.dataset_adapter.generic_json import GenericJsonLinesAdapter
from vulnrem.evaluation.dataset_adapter.not_implemented import NotImplementedDatasetAdapter
from vulnrem.evaluation.dataset_adapter.registry import build_dataset_adapter
from vulnrem.evaluation.schemas import CanonicalRecord
from vulnrem.evaluation.splitter import (
    compute_dataset_checksum,
    load_manifest,
    save_manifest,
    select,
    stratified_split,
    verify_no_leakage,
)


def _write_jsonl(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "dataset.jsonl"
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(path)


def test_generic_adapter_parses_well_formed_records(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [
            {
                "record_id": "r1",
                "language": "python",
                "cwe_id": "CWE-89",
                "vulnerable_code": "bad code",
                "fixed_code": "good code",
                "description": "sqli",
            }
        ],
    )
    records = GenericJsonLinesAdapter(path, source_dataset_name="custom").load()
    assert len(records) == 1
    assert records[0].language == Language.PYTHON
    assert records[0].source_dataset == "custom"


def test_generic_adapter_supports_field_map(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [{"id": "r1", "lang": "python", "cwe": "CWE-89", "func_before": "bad code"}],
    )
    adapter = GenericJsonLinesAdapter(
        path,
        field_map={"record_id": "id", "language": "lang", "cwe_id": "cwe", "vulnerable_code": "func_before"},
    )
    records = adapter.load()
    assert records[0].record_id == "r1"
    assert records[0].vulnerable_code == "bad code"


def test_generic_adapter_skips_bad_records_by_default(tmp_path):
    path = _write_jsonl(
        tmp_path,
        [
            {"record_id": "r1", "language": "python", "cwe_id": "CWE-89", "vulnerable_code": "x"},
            {"record_id": "r2", "language": "klingon", "cwe_id": "CWE-89", "vulnerable_code": "y"},
        ],
    )
    records = GenericJsonLinesAdapter(path).load()
    assert len(records) == 1
    assert records[0].record_id == "r1"


def test_generic_adapter_strict_mode_raises(tmp_path):
    path = _write_jsonl(
        tmp_path, [{"record_id": "r1", "language": "klingon", "cwe_id": "CWE-89", "vulnerable_code": "y"}]
    )
    with pytest.raises(DatasetSchemaError):
        GenericJsonLinesAdapter(path, strict=True).load()


def test_registry_builds_generic_adapter(tmp_path):
    path = _write_jsonl(
        tmp_path, [{"record_id": "r1", "language": "python", "cwe_id": "CWE-89", "vulnerable_code": "x"}]
    )
    adapter = build_dataset_adapter("generic_jsonl", path=path)
    assert len(adapter.load()) == 1


def test_registry_named_public_datasets_are_documented_stubs():
    adapter = build_dataset_adapter("bigvul")
    assert isinstance(adapter, NotImplementedDatasetAdapter)
    with pytest.raises(DatasetError):
        adapter.load()


def _record(record_id: str, language: Language, cwe_id: str) -> CanonicalRecord:
    return CanonicalRecord(record_id=record_id, language=language, cwe_id=cwe_id, vulnerable_code="x")


def test_stratified_split_has_no_overlap_and_respects_minimum():
    records = [_record(f"r{i}", Language.PYTHON, "CWE-89") for i in range(10)] + [
        _record(f"s{i}", Language.PYTHON, "CWE-78") for i in range(3)
    ]
    manifest = stratified_split(records, held_out_fraction=0.2, random_seed=1, min_cases_per_stratum=1)

    assert set(manifest.train_record_ids) & set(manifest.eval_record_ids) == set()
    assert len(manifest.train_record_ids) + len(manifest.eval_record_ids) == 13
    assert manifest.strata_counts["python:CWE-89"]["eval"] >= 1
    assert manifest.strata_counts["python:CWE-78"]["eval"] >= 1


def test_stratified_split_is_deterministic_given_same_seed():
    records = [_record(f"r{i}", Language.PYTHON, "CWE-89") for i in range(20)]
    m1 = stratified_split(records, held_out_fraction=0.25, random_seed=42)
    m2 = stratified_split(records, held_out_fraction=0.25, random_seed=42)
    assert sorted(m1.eval_record_ids) == sorted(m2.eval_record_ids)


def test_stratified_split_single_record_stratum_goes_entirely_to_eval():
    records = [_record("only", Language.PYTHON, "CWE-798")]
    manifest = stratified_split(records, held_out_fraction=0.2, random_seed=1, min_cases_per_stratum=1)
    assert manifest.eval_record_ids == ["only"]
    assert manifest.train_record_ids == []


def test_manifest_roundtrip_and_leakage_check(tmp_path):
    records = [_record(f"r{i}", Language.PYTHON, "CWE-89") for i in range(6)]
    manifest = stratified_split(records, held_out_fraction=0.34, random_seed=7)
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    reloaded = load_manifest(path)
    assert reloaded == manifest

    verify_no_leakage(records, reloaded)  # should not raise

    tampered = records + [_record("extra", Language.PYTHON, "CWE-89")]
    with pytest.raises(DatasetLeakageError):
        verify_no_leakage(tampered, reloaded)


def test_select_filters_by_ids():
    records = [_record(f"r{i}", Language.PYTHON, "CWE-89") for i in range(5)]
    subset = select(records, ["r1", "r3"])
    assert {r.record_id for r in subset} == {"r1", "r3"}


def test_checksum_is_order_independent():
    a = [_record("r1", Language.PYTHON, "CWE-89"), _record("r2", Language.PYTHON, "CWE-89")]
    b = list(reversed(a))
    assert compute_dataset_checksum(a) == compute_dataset_checksum(b)
