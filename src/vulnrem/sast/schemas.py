from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vulnrem.core.enums import Severity


class RawFinding(BaseModel):
    """One finding as reported by a single SAST tool, before aggregation.
    See :mod:`vulnrem.pipeline.nodes.aggregator` for how multiple
    ``RawFinding``s (possibly from different tools, possibly overlapping)
    become one de-duplicated ``Finding`` that enters triage."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    rule_id: str
    cwe_id: str | None = None
    file_path: str
    line_start: int
    line_end: int
    message: str
    severity: Severity
    confidence: str | None = None
