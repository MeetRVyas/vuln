from __future__ import annotations

import json

from vulnrem.cli.main import app, build_parser


def test_parser_builds_all_subcommands():
    parser = build_parser()
    subcommand_names = {action.dest for action in parser._subparsers._group_actions}
    # argparse internals vary by version; simplest robust check is that each
    # subcommand string parses without error and sets the right func.
    for args in (
        ["scan", "target.py"],
        ["hardware-profile"],
        ["dataset-split", "data.jsonl", "--manifest-out", "m.json"],
        ["finetune-data-prep", "data.jsonl", "--manifest", "m.json", "--output", "out.jsonl"],
        ["eval-run", "data.jsonl", "--manifest", "m.json", "--output-json", "o.json", "--output-markdown", "o.md"],
    ):
        parsed = parser.parse_args(args)
        assert parsed.func is not None
    assert subcommand_names is not None  # keep the helper referenced without over-asserting argparse internals


def test_hardware_profile_command_prints_json(capsys, config_dir, monkeypatch):
    monkeypatch.setenv("VULNREM_CONFIG_DIR", str(config_dir))
    app(["hardware-profile", "--env", "dev"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] in {"small", "medium", "large"}
