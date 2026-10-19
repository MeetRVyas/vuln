"""Command-line entrypoint (`vulnrem ...`, installed via the
`[project.scripts]` entry point in pyproject.toml).

Kept intentionally thin: every subcommand just wires together already-tested
library functions (:mod:`vulnrem.pipeline.runner`,
:mod:`vulnrem.evaluation.*`, :mod:`vulnrem.config.hardware`) -- there is no
CLI-only logic worth unit testing separately from those modules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vulnrem.config.hardware import resolve_hardware_profile
from vulnrem.config.loader import load_config
from vulnrem.core.enums import Language
from vulnrem.evaluation.dataset_adapter.registry import build_dataset_adapter
from vulnrem.evaluation.harness import EvaluationHarness
from vulnrem.evaluation.report import save_json_report, save_markdown_report
from vulnrem.evaluation.splitter import load_manifest, save_manifest, select, stratified_split
from vulnrem.finetuning.data_prep import write_training_jsonl
from vulnrem.observability.logging_config import configure_logging
from vulnrem.pipeline.runner import build_file_check_runner


def _cmd_scan(args: argparse.Namespace) -> None:
    app_config = load_config(env=args.env)
    configure_logging(app_config.logging)

    path = Path(args.file)
    runner = build_file_check_runner(app_config, Language(args.language))
    report = runner.check_file(str(path), path.read_text())

    if args.json:
        print(report.model_dump_json(indent=2))
        return

    print(f"{report.file_path}: {len(report.findings)} finding(s)")
    for result in report.findings:
        state = result.final_state
        print(
            f"  [{result.finding.cwe_id or 'unknown-cwe'}] line {result.finding.line_start}: "
            f"{state.status.value} -- {state.status_detail or ''}"
        )


def _cmd_hardware_profile(args: argparse.Namespace) -> None:
    app_config = load_config(env=args.env)
    profile = resolve_hardware_profile(app_config.hardware)
    print(json.dumps(profile.model_dump(), indent=2))


def _cmd_dataset_split(args: argparse.Namespace) -> None:
    adapter = build_dataset_adapter(args.format, path=args.dataset)
    records = adapter.load()
    manifest = stratified_split(
        records,
        held_out_fraction=args.held_out_fraction,
        random_seed=args.seed,
        min_cases_per_stratum=args.min_cases_per_stratum,
    )
    save_manifest(manifest, args.manifest_out)
    print(
        f"Wrote manifest to {args.manifest_out}: {len(manifest.train_record_ids)} train / "
        f"{len(manifest.eval_record_ids)} eval, {len(manifest.strata_counts)} strata."
    )


def _cmd_finetune_data_prep(args: argparse.Namespace) -> None:
    adapter = build_dataset_adapter(args.format, path=args.dataset)
    records = adapter.load()
    manifest = load_manifest(args.manifest)
    train_records = select(records, manifest.train_record_ids)

    count = write_training_jsonl(train_records, args.output, cwe_lookup={})
    print(f"Wrote {count} Patcher training examples to {args.output}")


def _cmd_eval_run(args: argparse.Namespace) -> None:
    adapter = build_dataset_adapter(args.format, path=args.dataset)
    records = adapter.load()
    manifest = load_manifest(args.manifest)
    eval_records = select(records, manifest.eval_record_ids)

    app_config = load_config(env=args.env)
    configure_logging(app_config.logging)

    harness = EvaluationHarness(runner_factory=lambda language: build_file_check_runner(app_config, language))
    results = harness.run(eval_records)

    save_json_report(results, manifest, args.output_json)
    save_markdown_report(results, manifest, args.output_markdown)
    print(f"Wrote {args.output_json} and {args.output_markdown}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vulnrem")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Check a single file.")
    scan.add_argument("file", help="Path to the source file to check.")
    scan.add_argument("--language", default="python")
    scan.add_argument("--env", default="dev")
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=_cmd_scan)

    hw = subparsers.add_parser("hardware-profile", help="Print the resolved hardware profile.")
    hw.add_argument("--env", default="dev")
    hw.set_defaults(func=_cmd_hardware_profile)

    split = subparsers.add_parser("dataset-split", help="Create a stratified train/eval split manifest.")
    split.add_argument("dataset")
    split.add_argument("--format", default="generic_jsonl")
    split.add_argument("--manifest-out", required=True)
    split.add_argument("--held-out-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--min-cases-per-stratum", type=int, default=1)
    split.set_defaults(func=_cmd_dataset_split)

    prep = subparsers.add_parser("finetune-data-prep", help="Build Patcher fine-tuning JSONL from the train split.")
    prep.add_argument("dataset")
    prep.add_argument("--format", default="generic_jsonl")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--output", required=True)
    prep.set_defaults(func=_cmd_finetune_data_prep)

    eval_run = subparsers.add_parser("eval-run", help="Run the evaluation harness over the eval split.")
    eval_run.add_argument("dataset")
    eval_run.add_argument("--format", default="generic_jsonl")
    eval_run.add_argument("--manifest", required=True)
    eval_run.add_argument("--env", default="dev")
    eval_run.add_argument("--output-json", required=True)
    eval_run.add_argument("--output-markdown", required=True)
    eval_run.set_defaults(func=_cmd_eval_run)

    return parser


def app(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    app()
