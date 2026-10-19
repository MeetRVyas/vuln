from __future__ import annotations

import json
from pathlib import Path

from vulnrem.evaluation.metrics import (
    EvalCaseResult,
    compute_breakdown_by_cwe,
    compute_breakdown_by_language,
    compute_metrics,
)
from vulnrem.evaluation.splitter import SplitManifest


def build_full_report(
    results: list[EvalCaseResult], manifest: SplitManifest
) -> dict:
    overall = compute_metrics(results)
    by_cwe = compute_breakdown_by_cwe(results)
    by_language = compute_breakdown_by_language(results)
    return {
        "dataset_checksum": manifest.dataset_checksum,
        "random_seed": manifest.random_seed,
        "overall": overall.model_dump(),
        "by_cwe": {cwe: m.model_dump() for cwe, m in by_cwe.items()},
        "by_language": {lang.value: m.model_dump() for lang, m in by_language.items()},
        "cases": [r.model_dump() for r in results],
    }


def save_json_report(results: list[EvalCaseResult], manifest: SplitManifest, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_full_report(results, manifest), indent=2, default=str))


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def build_markdown_report(results: list[EvalCaseResult], manifest: SplitManifest) -> str:
    overall = compute_metrics(results)
    by_cwe = compute_breakdown_by_cwe(results)
    by_language = compute_breakdown_by_language(results)

    lines = [
        "# Vulnerability Remediation -- Evaluation Report",
        "",
        f"- Dataset checksum: `{manifest.dataset_checksum[:16]}`",
        f"- Random seed: {manifest.random_seed}",
        f"- Held-out fraction target: {manifest.held_out_fraction:.0%}",
        f"- Total eval cases: {overall.total_eval_cases}",
        "",
        "## Overall",
        "",
        f"- SAST detection rate: {_fmt_pct(overall.sast_detection_rate)}",
        f"- **Detection recall**: {_fmt_pct(overall.detection_recall)}",
        f"- **Remediation recall**: {_fmt_pct(overall.remediation_recall)}",
        f"- Weighted overall score: {overall.weighted_overall_score:.3f}",
        "",
        "## By CWE",
        "",
        "| CWE | n | Detection recall | Remediation recall |",
        "|---|---|---|---|",
    ]
    for cwe, m in sorted(by_cwe.items()):
        lines.append(
            f"| {cwe} | {m.total_eval_cases} | {_fmt_pct(m.detection_recall)} | "
            f"{_fmt_pct(m.remediation_recall)} |"
        )

    lines += ["", "## By language", "", "| Language | n | Detection recall | Remediation recall |", "|---|---|---|---|"]
    for language, m in sorted(by_language.items(), key=lambda kv: kv[0].value):
        lines.append(
            f"| {language.value} | {m.total_eval_cases} | {_fmt_pct(m.detection_recall)} | "
            f"{_fmt_pct(m.remediation_recall)} |"
        )

    return "\n".join(lines) + "\n"


def save_markdown_report(results: list[EvalCaseResult], manifest: SplitManifest, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_markdown_report(results, manifest))
