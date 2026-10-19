from vulnrem.core.enums import Language, Severity
from vulnrem.pipeline.prompts.reproducer import build_request
from vulnrem.pipeline.state import Finding


def test_reproducer_receives_original_source():
    finding = Finding(
        finding_id="sample",
        language=Language.PYTHON,
        file_path="snippet.py",
        line_start=2,
        line_end=2,
        cwe_id="CWE-89",
        severity=Severity.MEDIUM,
        tool_findings=[],
    )
    source = "def find_user(conn, name):\n    return conn.execute(name)\n"

    request = build_request(finding, "target.py", source, None)

    assert "`target.py`" in request.messages[1].content
    assert source in request.messages[1].content
    assert request.max_output_tokens == 4096
