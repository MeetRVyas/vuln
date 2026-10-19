from __future__ import annotations

from vulnrem.dashboard.app import _run_check, demo


def test_importing_app_builds_blocks_without_launching():
    assert demo is not None


def test_run_check_with_no_input_asks_for_a_file():
    result = _run_check(None, "", "snippet.py")
    assert "Upload a" in result


def test_run_check_surfaces_pipeline_errors_instead_of_crashing():
    # The dev-config default LLM provider is `mock` with nothing queued, so
    # the very first LLM call (Triage) raises -- the UI must show that as a
    # message, not propagate an exception out of the click handler.
    result = _run_check(None, "def f():\n    pass\n", "snippet.py")
    assert isinstance(result, str)
