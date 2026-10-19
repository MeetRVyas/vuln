"""The per-finding remediation graph.

One file can produce several findings (SAST + aggregation runs once,
outside this graph -- see :mod:`vulnrem.pipeline.runner`); each finding gets
its own independent run of this compiled graph.

Topology::

    START -> triage -> [false positive?] -> finalize -> END
                     -> reproducer -> [not confirmed?] -> finalize -> END
                                    -> localizer -> test_writer -> patcher
                                       -> verifier -> judge -> [retry?] -> patcher (loop)
                                                              -> finalize -> END

Patcher and Test-Writer run sequentially here (Test-Writer's tests only
depend on the *original* source and the Reproducer's confirmed PoC, never
on a specific patch, so they have no real data dependency on each other --
LangGraph can run them as true parallel branches with a fan-in at
``verifier`` instead; sequential execution was chosen for this build for
simplicity and deterministic step-by-step logs, see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from vulnrem.core.enums import FindingStatus
from vulnrem.languages.base import CWESpec, LanguageProfile
from vulnrem.llm.base import LLMProvider
from vulnrem.oracles.regression.base import RegressionChecker
from vulnrem.oracles.sanitizer import SanitizerChecker
from vulnrem.oracles.security import SecurityRescanChecker
from vulnrem.pipeline.nodes.judge import JudgeNode
from vulnrem.pipeline.nodes.localizer import LocalizerNode
from vulnrem.pipeline.nodes.patcher import PatcherNode
from vulnrem.pipeline.nodes.reproducer import ReproducerNode
from vulnrem.pipeline.nodes.test_writer import TestWriterNode
from vulnrem.pipeline.nodes.triage import TriageNode
from vulnrem.pipeline.nodes.verifier import VerifierNode
from vulnrem.pipeline.retry_policy import RetryPolicy
from vulnrem.pipeline.state import PipelineState
from vulnrem.sandbox.base import Sandbox
from vulnrem.search.base import RepoSearchTool
from vulnrem.search.single_file import SingleFileSearchTool


@dataclass
class PipelineDependencies:
    llm: LLMProvider
    sandbox: Sandbox
    language_profile: LanguageProfile
    regression_checker: RegressionChecker
    security_rescan_checker: SecurityRescanChecker
    sanitizer_checker: SanitizerChecker
    retry_policy: RetryPolicy
    sandbox_timeout_s: float = 20.0
    sandbox_memory_limit_mb: int = 512
    sandbox_cpu_limit_seconds: float = 15.0
    test_writer_max_self_validation_attempts: int = 2
    search_tool_factory: type[RepoSearchTool] = SingleFileSearchTool
    cwe_lookup: dict[str, CWESpec] = field(default_factory=dict)


def _route_after_triage(state: PipelineState) -> str:
    return "reproducer" if state.triage_confirmed else "finalize"


def _route_after_reproducer(state: PipelineState) -> str:
    return "localizer" if (state.reproduction is not None and state.reproduction.confirmed) else "finalize"


def _make_route_after_judge(retry_policy: RetryPolicy):
    def _route(state: PipelineState) -> str:
        decision = retry_policy.decide(state.attempt_history)
        return "patcher" if decision.action == "retry" else "finalize"

    return _route


def _make_finalize_node(retry_policy: RetryPolicy):
    def _finalize(state: PipelineState) -> dict:
        if state.triage_confirmed is False:
            return {
                "status": FindingStatus.TRIAGED_FALSE_POSITIVE,
                "status_detail": f"Triage: {state.triage_rationale}",
            }
        if state.reproduction is not None and not state.reproduction.confirmed:
            return {
                "status": FindingStatus.TRIAGED_FALSE_POSITIVE,
                "status_detail": f"Dynamic reproduction did not confirm the finding: "
                f"{state.reproduction.explanation}",
            }
        if state.attempt_history:
            decision = retry_policy.decide(state.attempt_history)
            if decision.terminal_status is not None:
                detail = decision.note
                if not detail and state.judge_verdict is not None:
                    detail = state.judge_verdict.rationale
                return {"status": decision.terminal_status, "status_detail": detail}
        return {
            "status": FindingStatus.ERROR,
            "status_detail": "Reached finalize without a determinable terminal status "
            "(this indicates a bug in the graph's routing logic, not a pipeline outcome).",
        }

    return _finalize


def build_pipeline_graph(deps: PipelineDependencies) -> CompiledStateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("triage", TriageNode(deps.llm, deps.cwe_lookup))
    graph.add_node(
        "reproducer",
        ReproducerNode(
            llm=deps.llm,
            sandbox=deps.sandbox,
            language_profile=deps.language_profile,
            cwe_lookup=deps.cwe_lookup,
            timeout_s=deps.sandbox_timeout_s,
            memory_limit_mb=deps.sandbox_memory_limit_mb,
            cpu_limit_seconds=deps.sandbox_cpu_limit_seconds,
        ),
    )
    graph.add_node("localizer", LocalizerNode(deps.language_profile, deps.search_tool_factory))
    graph.add_node(
        "test_writer",
        TestWriterNode(
            llm=deps.llm,
            sandbox=deps.sandbox,
            language_profile=deps.language_profile,
            timeout_s=deps.sandbox_timeout_s,
            memory_limit_mb=deps.sandbox_memory_limit_mb,
            cpu_limit_seconds=deps.sandbox_cpu_limit_seconds,
            max_self_validation_attempts=deps.test_writer_max_self_validation_attempts,
        ),
    )
    graph.add_node("patcher", PatcherNode(deps.llm, deps.cwe_lookup))
    graph.add_node(
        "verifier",
        VerifierNode(
            sandbox=deps.sandbox,
            language_profile=deps.language_profile,
            regression_checker=deps.regression_checker,
            security_rescan_checker=deps.security_rescan_checker,
            sanitizer_checker=deps.sanitizer_checker,
            timeout_s=deps.sandbox_timeout_s,
            memory_limit_mb=deps.sandbox_memory_limit_mb,
            cpu_limit_seconds=deps.sandbox_cpu_limit_seconds,
        ),
    )
    graph.add_node("judge", JudgeNode(deps.llm))
    graph.add_node("finalize", _make_finalize_node(deps.retry_policy))

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", _route_after_triage, {"reproducer": "reproducer", "finalize": "finalize"})
    graph.add_conditional_edges(
        "reproducer", _route_after_reproducer, {"localizer": "localizer", "finalize": "finalize"}
    )
    graph.add_edge("localizer", "test_writer")
    graph.add_edge("test_writer", "patcher")
    graph.add_edge("patcher", "verifier")
    graph.add_edge("verifier", "judge")
    graph.add_conditional_edges(
        "judge", _make_route_after_judge(deps.retry_policy), {"patcher": "patcher", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()
