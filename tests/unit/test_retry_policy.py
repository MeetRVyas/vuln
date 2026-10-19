from __future__ import annotations

import pytest

from vulnrem.config.schemas import RetryConfig
from vulnrem.core.enums import FindingStatus, RetryCategory
from vulnrem.pipeline.retry_policy import RetryPolicy
from vulnrem.pipeline.state import AttemptRecord, JudgeVerdict


def _record(n: int, decision: str, category: RetryCategory) -> AttemptRecord:
    return AttemptRecord(
        attempt_number=n,
        judge_verdict=JudgeVerdict(decision=decision, category=category, confidence=0.7, rationale="r"),
    )


@pytest.fixture
def policy() -> RetryPolicy:
    return RetryPolicy(RetryConfig(max_normal_retries=2, max_escalation_attempts=1))


def test_accept_is_terminal_accepted(policy):
    history = [_record(1, "accept", RetryCategory.NONE)]
    decision = policy.decide(history)
    assert decision.action == "terminal"
    assert decision.terminal_status == FindingStatus.ACCEPTED


def test_multi_file_scope_is_immediately_unresolved_no_retry(policy):
    history = [_record(1, "retry", RetryCategory.MULTI_FILE_SCOPE_NEEDED)]
    decision = policy.decide(history)
    assert decision.action == "terminal"
    assert decision.terminal_status == FindingStatus.UNRESOLVED_ESCALATION_UNAVAILABLE


def test_normal_category_retries_within_budget(policy):
    history = [_record(1, "retry", RetryCategory.REGRESSION_FAILURE)]
    decision = policy.decide(history)
    assert decision.action == "retry"


def test_normal_category_exhausts_budget(policy):
    # policy allows max_normal_retries=2 -> 3rd normal-category verdict exhausts it
    history = [
        _record(1, "retry", RetryCategory.SYNTAX_OR_PLUMBING),
        _record(2, "retry", RetryCategory.REGRESSION_FAILURE),
        _record(3, "retry", RetryCategory.SUSPICIOUS_OR_DEGENERATE),
    ]
    decision = policy.decide(history)
    assert decision.action == "terminal"
    assert decision.terminal_status == FindingStatus.UNRESOLVED_RETRIES_EXHAUSTED


def test_vulnerability_still_present_degrades_to_retry_within_escalation_budget(policy):
    history = [_record(1, "retry", RetryCategory.VULNERABILITY_STILL_PRESENT)]
    decision = policy.decide(history)
    assert decision.action == "retry"
    assert "Debugger-in-the-loop is not implemented" in decision.note


def test_vulnerability_still_present_exhausts_escalation_budget(policy):
    # max_escalation_attempts=1 -> 2nd such verdict exhausts it
    history = [
        _record(1, "retry", RetryCategory.VULNERABILITY_STILL_PRESENT),
        _record(2, "retry", RetryCategory.VULNERABILITY_STILL_PRESENT),
    ]
    decision = policy.decide(history)
    assert decision.action == "terminal"
    assert decision.terminal_status == FindingStatus.UNRESOLVED_ESCALATION_UNAVAILABLE


def test_escalation_and_normal_budgets_are_independent(policy):
    # one escalation-category verdict should not consume the normal budget
    history = [
        _record(1, "retry", RetryCategory.VULNERABILITY_STILL_PRESENT),
        _record(2, "retry", RetryCategory.REGRESSION_FAILURE),
    ]
    decision = policy.decide(history)
    assert decision.action == "retry"  # normal budget (2) not yet exhausted by a single entry


def test_missing_judge_verdict_is_treated_as_cheap_plumbing_retry(policy):
    history = [AttemptRecord(attempt_number=1, error="sandbox crashed")]
    decision = policy.decide(history)
    assert decision.action == "retry"


def test_empty_history_raises():
    policy = RetryPolicy(RetryConfig())
    with pytest.raises(ValueError):
        policy.decide([])
