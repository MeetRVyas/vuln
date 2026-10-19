"""Proves the whole architecture wires together correctly: real SAST
(Bandit) finds a real SQL injection in the fixture file, and a fully
scripted (but genuinely-executed) Triage -> Reproducer -> Localizer ->
Test-Writer -> Patcher -> Verifier -> Judge run reaches ACCEPTED end to end,
using the real subprocess Sandbox (so every "canned" script below is
actually executed, actually parsed, actually asserted on) and only the LLM
calls themselves mocked.
"""

from __future__ import annotations

import shutil

import pytest

from vulnrem.config.loader import load_config
from vulnrem.config.schemas import RoleModelMap
from vulnrem.core.enums import AgentRole, FindingStatus, Language, RetryCategory
from vulnrem.llm.providers.mock import MockLLMProvider
from vulnrem.pipeline.prompts.reproducer import POC_CONFIRMED_MARKER
from vulnrem.pipeline.prompts.schemas import PatcherOutput, ReproducerOutput, TestWriterOutput
from vulnrem.pipeline.runner import build_file_check_runner
from vulnrem.pipeline.state import JudgeVerdict, TestCase
from vulnrem.sandbox.subprocess_sandbox import SubprocessSandbox
from vulnrem.tracking.in_memory import InMemoryTracker

pytestmark = pytest.mark.integration

_POC_PAYLOAD = "' OR '1'='1"

_POC_SCRIPT = f"""from target import get_connection, find_user_by_name

conn = get_connection()
rows = find_user_by_name(conn, "{_POC_PAYLOAD}")
if len(rows) > 1:
    print("{POC_CONFIRMED_MARKER}")
else:
    print("VULN_NOT_CONFIRMED")
"""

_POC_TEST = TestCase(
    name="test_poc_sql_injection_blocked",
    kind="poc",
    source=f'''def test_poc_sql_injection_blocked():
    from target import get_connection, find_user_by_name
    conn = get_connection()
    rows = find_user_by_name(conn, "{_POC_PAYLOAD}")
    assert len(rows) <= 1
''',
    expected_result_on_original="fail",
)

_REGRESSION_TEST = TestCase(
    name="test_regression_find_existing_user",
    kind="regression",
    source='''def test_regression_find_existing_user():
    from target import get_connection, find_user_by_name
    conn = get_connection()
    rows = find_user_by_name(conn, "alice")
    assert len(rows) == 1
    assert rows[0][1] == "alice"
''',
    expected_result_on_original="pass",
)

_EDGE_CASE_TEST = TestCase(
    name="test_edge_case_nonexistent_username_returns_no_rows",
    kind="edge_case",
    source='''def test_edge_case_nonexistent_username_returns_no_rows():
    from target import get_connection, find_user_by_name
    conn = get_connection()
    rows = find_user_by_name(conn, "nonexistent_user_xyz")
    assert len(rows) == 0
''',
    expected_result_on_original="pass",
)

_PATCHED_SOURCE = '''"""Tiny in-memory 'user store' used only to give the SQL injection below a
runnable, testable context."""

import sqlite3


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, is_admin INTEGER)")
    conn.execute("INSERT INTO users (username, is_admin) VALUES ('alice', 0)")
    conn.execute("INSERT INTO users (username, is_admin) VALUES ('admin', 1)")
    conn.commit()
    return conn


def find_user_by_name(conn: sqlite3.Connection, username: str):
    query = "SELECT id, username, is_admin FROM users WHERE username = ?"
    cursor = conn.execute(query, (username,))
    return cursor.fetchall()
'''


@pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not installed")
def test_sql_injection_finding_is_fixed_and_accepted(config_dir, sql_injection_source):
    app_config = load_config(env="dev", config_dir=config_dir)

    mock_llm = MockLLMProvider(models=RoleModelMap(default="mock-model"))
    mock_llm.enqueue_structured(
        AgentRole.TRIAGE,
        {
            "is_true_positive": True,
            "confidence": 0.9,
            "rationale": "Untrusted `username` is string-concatenated directly into a SQL query.",
        },
    )
    mock_llm.enqueue_structured(
        AgentRole.REPRODUCER,
        ReproducerOutput(
            poc_script=_POC_SCRIPT,
            explanation="A classic ' OR '1'='1 tautology returns every row instead of one.",
        ),
    )
    mock_llm.enqueue_structured(
        AgentRole.TEST_WRITER,
        TestWriterOutput(test_cases=[_POC_TEST, _REGRESSION_TEST, _EDGE_CASE_TEST]),
    )
    mock_llm.enqueue_structured(
        AgentRole.PATCHER,
        PatcherOutput(
            patched_source=_PATCHED_SOURCE,
            explanation="Switched to a parameterized query so untrusted input is never part of the SQL text.",
        ),
    )
    mock_llm.enqueue_structured(
        AgentRole.JUDGE,
        JudgeVerdict(
            decision="accept",
            category=RetryCategory.NONE,
            confidence=0.95,
            rationale="Both oracles pass and the fix is the standard parameterized-query idiom.",
        ),
    )

    tracker = InMemoryTracker()
    runner = build_file_check_runner(
        app_config=app_config,
        language=Language.PYTHON,
        config_dir=str(config_dir),
        llm=mock_llm,
        sandbox=SubprocessSandbox(),
        tracker=tracker,
    )

    report = runner.check_file("sql_injection.py", sql_injection_source)

    assert len(report.findings) == 1
    result = report.findings[0]
    assert result.finding.cwe_id == "CWE-89"

    final_state = result.final_state
    assert final_state.status == FindingStatus.ACCEPTED, final_state.status_detail

    assert final_state.triage_confirmed is True
    assert final_state.reproduction is not None
    assert final_state.reproduction.confirmed is True

    battery = final_state.test_battery
    assert battery is not None
    assert battery.discarded_case_names == []
    assert {tc.name for tc in battery.test_cases} == {
        "test_poc_sql_injection_blocked",
        "test_regression_find_existing_user",
        "test_edge_case_nonexistent_username_returns_no_rows",
    }

    report_v = final_state.verifier_report
    assert report_v is not None
    assert report_v.functional.passed is True
    assert report_v.security.passed is True
    assert report_v.security.poc_no_longer_triggers.passed is True
    assert report_v.security.sast_rescan_clean.passed is True

    assert final_state.judge_verdict is not None
    assert final_state.judge_verdict.decision == "accept"
    assert len(final_state.attempt_history) == 1  # accepted on the first attempt

    assert len(tracker.finding_runs) == 1
    tracked = tracker.finding_runs[0]
    assert tracked.remediated is True
    assert tracked.cwe_id == "CWE-89"
    assert tracked.llm_provider == "mock"


_BROKEN_PATCH_SOURCE = '''"""Deliberately-broken first attempt: blocks the injection by breaking all lookups."""

import sqlite3


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, is_admin INTEGER)")
    conn.execute("INSERT INTO users (username, is_admin) VALUES ('alice', 0)")
    conn.execute("INSERT INTO users (username, is_admin) VALUES ('admin', 1)")
    conn.commit()
    return conn


def find_user_by_name(conn: sqlite3.Connection, username: str):
    return []
'''


@pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not installed")
def test_retry_loop_recovers_after_a_regression_failure(config_dir, sql_injection_source):
    """Proves the judge -> patcher loop-back edge and attempt_history
    accumulation actually work: the first patch is functionally broken
    (Judge rejects it as a regression_failure), the second patch is the
    correct fix (Judge accepts), and both attempts show up in history."""
    app_config = load_config(env="dev", config_dir=config_dir)

    mock_llm = MockLLMProvider(models=RoleModelMap(default="mock-model"))
    mock_llm.enqueue_structured(
        AgentRole.TRIAGE,
        {"is_true_positive": True, "confidence": 0.9, "rationale": "String-concatenated SQL."},
    )
    mock_llm.enqueue_structured(
        AgentRole.REPRODUCER,
        ReproducerOutput(poc_script=_POC_SCRIPT, explanation="Tautology returns every row."),
    )
    mock_llm.enqueue_structured(
        AgentRole.TEST_WRITER,
        TestWriterOutput(test_cases=[_POC_TEST, _REGRESSION_TEST, _EDGE_CASE_TEST]),
    )
    # Attempt 1: broken patch.
    mock_llm.enqueue_structured(
        AgentRole.PATCHER,
        PatcherOutput(patched_source=_BROKEN_PATCH_SOURCE, explanation="Block everything (wrong)."),
    )
    mock_llm.enqueue_structured(
        AgentRole.JUDGE,
        JudgeVerdict(
            decision="retry",
            category=RetryCategory.REGRESSION_FAILURE,
            confidence=0.85,
            rationale="This breaks legitimate lookups entirely -- always returns no rows.",
        ),
    )
    # Attempt 2: correct patch.
    mock_llm.enqueue_structured(
        AgentRole.PATCHER,
        PatcherOutput(patched_source=_PATCHED_SOURCE, explanation="Parameterized query."),
    )
    mock_llm.enqueue_structured(
        AgentRole.JUDGE,
        JudgeVerdict(
            decision="accept",
            category=RetryCategory.NONE,
            confidence=0.95,
            rationale="Both oracles pass now.",
        ),
    )

    runner = build_file_check_runner(
        app_config=app_config,
        language=Language.PYTHON,
        config_dir=str(config_dir),
        llm=mock_llm,
        sandbox=SubprocessSandbox(),
    )

    report = runner.check_file("sql_injection.py", sql_injection_source)
    final_state = report.findings[0].final_state

    assert final_state.status == FindingStatus.ACCEPTED, final_state.status_detail
    assert len(final_state.attempt_history) == 2

    first, second = final_state.attempt_history
    assert first.judge_verdict.decision == "retry"
    assert first.judge_verdict.category == RetryCategory.REGRESSION_FAILURE
    assert first.verifier_report.functional.passed is False

    assert second.judge_verdict.decision == "accept"
    assert second.verifier_report.functional.passed is True
    assert second.verifier_report.security.passed is True

    # Only one Triage/Reproducer/Test-Writer call across the whole run --
    # the retry loop re-enters at the Patcher, not further upstream.
    assert len(mock_llm.call_log) == 3 + 2 * 2  # triage(1) + reproducer(1) + test_writer(1) + (patcher+judge)*2

