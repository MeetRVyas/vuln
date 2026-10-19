"""Merges the raw output of however many SAST tools ran for a language into
one de-duplicated list of :class:`~vulnrem.pipeline.state.Finding`.

Merge semantics (a union, not an intersection -- corroboration from a
second tool raises confidence but a finding reported by only one tool is
still kept):

1. Group raw findings by file.
2. Within a file, cluster findings whose line ranges overlap or sit within
   ``line_proximity`` lines of each other -- this is what lets Opengrep and
   Bandit's two different reports of "the same" SQL-injection line become
   one finding instead of two.
3. Within a cluster, drop exact repeats (same tool + rule + line range).
4. The cluster's CWE is whichever non-null CWE id is most common among its
   members (ties broken by first-seen order); its severity is the highest
   severity among its members.

This is deliberately simple, deterministic, deduplication logic -- no
custom static analysis is implemented here, per the project plan's
"don't reinvent infrastructure that already exists" principle.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from vulnrem.core.enums import Language, Severity
from vulnrem.pipeline.state import Finding
from vulnrem.sast.schemas import RawFinding

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def aggregate_findings(
    raw_findings: list[RawFinding], language: Language, line_proximity: int = 2
) -> list[Finding]:
    by_file: dict[str, list[RawFinding]] = defaultdict(list)
    for raw in raw_findings:
        by_file[raw.file_path].append(raw)

    findings: list[Finding] = []
    for file_path, items in by_file.items():
        for cluster in _cluster_by_proximity(items, line_proximity):
            unique = _dedup_exact(cluster)
            line_start = min(f.line_start for f in unique)
            line_end = max(f.line_end for f in unique)
            cwe_id = _pick_dominant_cwe(unique)
            severity = max((f.severity for f in unique), key=lambda s: _SEVERITY_RANK[s])
            findings.append(
                Finding(
                    finding_id=Finding.make_id(file_path, line_start, cwe_id),
                    language=language,
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    cwe_id=cwe_id,
                    severity=severity,
                    tool_findings=unique,
                )
            )
    return findings


def _cluster_by_proximity(items: list[RawFinding], proximity: int) -> list[list[RawFinding]]:
    clusters: list[list[RawFinding]] = []
    for item in sorted(items, key=lambda f: (f.line_start, f.line_end)):
        for cluster in clusters:
            if any(_near(existing, item, proximity) for existing in cluster):
                cluster.append(item)
                break
        else:
            clusters.append([item])
    return clusters


def _near(a: RawFinding, b: RawFinding, proximity: int) -> bool:
    return a.line_start <= b.line_end + proximity and b.line_start <= a.line_end + proximity


def _dedup_exact(items: list[RawFinding]) -> list[RawFinding]:
    seen: set[tuple] = set()
    unique: list[RawFinding] = []
    for item in items:
        key = (item.tool, item.rule_id, item.line_start, item.line_end)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _pick_dominant_cwe(items: list[RawFinding]) -> str | None:
    cwe_ids = [f.cwe_id for f in items if f.cwe_id]
    if not cwe_ids:
        return None
    counts = Counter(cwe_ids)
    max_count = max(counts.values())
    for cwe in cwe_ids:  # first-seen order for deterministic tie-break
        if counts[cwe] == max_count:
            return cwe
    return cwe_ids[0]  # unreachable, satisfies type checkers
