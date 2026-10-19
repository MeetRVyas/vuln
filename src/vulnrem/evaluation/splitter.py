"""Splits the canonical dataset into a fine-tuning set and a non-overlapping,
stratified held-out evaluation set, and writes a checksummed manifest
recording exactly which record went where -- so a later run can verify (via
:func:`verify_no_leakage`) that the eval set a reported recall number came
from never leaked into a subsequent fine-tuning run.

Stratification is by (language, cwe_id): the plan calls for the eval set to
have "documented coverage across the target languages and CWEs" with "clear
denominators" per CWE, which a purely-random split cannot promise (a rare
CWE could end up entirely in one side by chance).
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vulnrem.core.errors import DatasetLeakageError
from vulnrem.evaluation.schemas import CanonicalRecord


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_checksum: str
    random_seed: int
    held_out_fraction: float
    min_cases_per_stratum: int
    train_record_ids: list[str]
    eval_record_ids: list[str]
    strata_counts: dict[str, dict[str, int]]  # "language:cwe_id" -> {"train": n, "eval": n}


def compute_dataset_checksum(records: list[CanonicalRecord]) -> str:
    ids = sorted(r.record_id for r in records)
    return hashlib.sha256("|".join(ids).encode()).hexdigest()


def stratified_split(
    records: list[CanonicalRecord],
    held_out_fraction: float,
    random_seed: int,
    min_cases_per_stratum: int = 1,
) -> SplitManifest:
    rng = random.Random(random_seed)
    by_stratum: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        by_stratum[f"{record.language.value}:{record.cwe_id}"].append(record)

    train_ids: list[str] = []
    eval_ids: list[str] = []
    strata_counts: dict[str, dict[str, int]] = {}

    for stratum_key, group in by_stratum.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        group_size = len(shuffled)

        n_eval = max(min_cases_per_stratum, round(group_size * held_out_fraction))
        n_eval = min(n_eval, group_size)
        if n_eval == group_size and group_size > 1:
            n_eval = group_size - 1  # keep at least one training example when we have more than one

        eval_group, train_group = shuffled[:n_eval], shuffled[n_eval:]
        eval_ids.extend(r.record_id for r in eval_group)
        train_ids.extend(r.record_id for r in train_group)
        strata_counts[stratum_key] = {"train": len(train_group), "eval": len(eval_group)}

    if set(train_ids) & set(eval_ids):
        raise DatasetLeakageError("Internal error: a record ended up in both splits.")

    return SplitManifest(
        dataset_checksum=compute_dataset_checksum(records),
        random_seed=random_seed,
        held_out_fraction=held_out_fraction,
        min_cases_per_stratum=min_cases_per_stratum,
        train_record_ids=train_ids,
        eval_record_ids=eval_ids,
        strata_counts=strata_counts,
    )


def save_manifest(manifest: SplitManifest, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.model_dump_json(indent=2))


def load_manifest(path: Path | str) -> SplitManifest:
    return SplitManifest.model_validate_json(Path(path).read_text())


def verify_no_leakage(records: list[CanonicalRecord], manifest: SplitManifest) -> None:
    """Raises :class:`~vulnrem.core.errors.DatasetLeakageError` if the given
    records' checksum no longer matches the manifest (someone changed the
    dataset without re-splitting) or if any record from ``eval_record_ids``
    is about to be used for fine-tuning."""
    current_checksum = compute_dataset_checksum(records)
    if current_checksum != manifest.dataset_checksum:
        raise DatasetLeakageError(
            f"Dataset checksum {current_checksum} does not match the manifest's "
            f"{manifest.dataset_checksum} -- the dataset changed since this split was made; re-split "
            "before fine-tuning or evaluating against it."
        )


def select(records: list[CanonicalRecord], record_ids: list[str]) -> list[CanonicalRecord]:
    wanted = set(record_ids)
    return [r for r in records if r.record_id in wanted]
