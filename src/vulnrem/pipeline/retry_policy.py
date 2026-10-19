"""Categorized retry policy.

Reads the Judge's category for the *last* attempt plus how many previous
attempts fell into each budget, and decides: accept, retry (loop back to
the Patcher), or give up (with a specific terminal status explaining why).
Nothing here is a graph node itself -- it's the pure function the
conditional edge after the Judge node calls (see
:mod:`vulnrem.pipeline.graph`), which keeps the branching logic unit-testable
without needing to run any part of the graph.

Escalation categories and how this build handles them, since neither the
Debugger nor the repo-scale agent is implemented (both are Ambitious-tier,
out of this Strong-tier MVP's scope -- see docs/ARCHITECTURE.md):

* ``vulnerability_still_present`` would normally hand off to a Debugger
  that instruments and re-runs the failure to produce a root-cause trace.
  Without one, we gracefully degrade: retry the Patcher anyway, using the
  Judge's own rationale as the only extra signal, under a smaller, separate
  attempt budget (``retry.max_escalation_attempts``) so this degraded path
  can't quietly consume the whole normal retry budget.
* ``multi_file_scope_needed`` would hand off to a repo-scale agent that can
  edit more than one file. A single-file Patcher retrying the same file
  cannot fix this by construction, so there is no degraded substitute --
  this is unresolved immediately, not after burning retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vulnrem.config.schemas import RetryConfig
from vulnrem.core.enums import FindingStatus, RetryCategory
from vulnrem.pipeline.state import AttemptRecord

RouteAction = Literal["retry", "terminal"]


@dataclass(frozen=True)
class RetryDecision:
    action: RouteAction
    terminal_status: FindingStatus | None = None
    note: str = ""


_NORMAL_BUDGET_CATEGORIES = frozenset(
    {
        RetryCategory.SYNTAX_OR_PLUMBING,
        RetryCategory.REGRESSION_FAILURE,
        RetryCategory.SUSPICIOUS_OR_DEGENERATE,
    }
)


class RetryPolicy:
    def __init__(self, config: RetryConfig):
        self._config = config

    def decide(self, attempt_history: list[AttemptRecord]) -> RetryDecision:
        if not attempt_history:
            raise ValueError("decide() must be called after at least one Judge verdict exists")

        latest = attempt_history[-1]
        verdict = latest.judge_verdict
        if verdict is None:
            # The attempt errored out before reaching the Judge (e.g. sandbox
            # crash) -- treat as a cheap plumbing retry rather than losing
            # the whole finding to an infrastructure hiccup.
            return self._decide_normal_retry(attempt_history, RetryCategory.SYNTAX_OR_PLUMBING)

        if verdict.decision == "accept":
            return RetryDecision(action="terminal", terminal_status=FindingStatus.ACCEPTED)

        if verdict.category == RetryCategory.MULTI_FILE_SCOPE_NEEDED:
            return RetryDecision(
                action="terminal",
                terminal_status=FindingStatus.UNRESOLVED_ESCALATION_UNAVAILABLE,
                note=(
                    "Judge determined a correct fix requires changes outside this file. "
                    "Repo-scale remediation is not implemented in this build (single-file "
                    "scope only) -- see docs/ARCHITECTURE.md for the extension point."
                ),
            )

        if verdict.category == RetryCategory.VULNERABILITY_STILL_PRESENT:
            escalation_count = self._count(attempt_history, {RetryCategory.VULNERABILITY_STILL_PRESENT})
            if escalation_count > self._config.max_escalation_attempts:
                return RetryDecision(
                    action="terminal",
                    terminal_status=FindingStatus.UNRESOLVED_ESCALATION_UNAVAILABLE,
                    note=(
                        f"Debugger-assisted escalation budget "
                        f"({self._config.max_escalation_attempts}) exhausted. Debugger-in-the-loop "
                        "is not implemented in this build; attempts were retried using only the "
                        "Judge's rationale as a degraded substitute."
                    ),
                )
            return RetryDecision(
                action="retry",
                note=(
                    "Debugger-in-the-loop is not implemented in this build -- retrying with the "
                    "Judge's rationale as the only additional signal (degraded escalation)."
                ),
            )

        return self._decide_normal_retry(attempt_history, verdict.category)

    def _decide_normal_retry(
        self, attempt_history: list[AttemptRecord], category: RetryCategory
    ) -> RetryDecision:
        normal_count = self._count(attempt_history, _NORMAL_BUDGET_CATEGORIES)
        if normal_count > self._config.max_normal_retries:
            return RetryDecision(
                action="terminal",
                terminal_status=FindingStatus.UNRESOLVED_RETRIES_EXHAUSTED,
                note=f"Normal retry budget ({self._config.max_normal_retries}) exhausted "
                f"(last category: {category.value}).",
            )
        return RetryDecision(action="retry")

    @staticmethod
    def _count(attempt_history: list[AttemptRecord], categories: frozenset[RetryCategory]) -> int:
        return sum(
            1
            for record in attempt_history
            if record.judge_verdict is not None and record.judge_verdict.category in categories
        )
