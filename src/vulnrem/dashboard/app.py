"""Gradio dashboard. Deploy target: Hugging Face Spaces (see
docs/SETUP_DEPLOYMENT.md) -- this file is the Space's entrypoint.

Everything here is presentation: accepting a file or pasted snippet,
calling :class:`~vulnrem.dashboard.service.PipelineService`, and rendering
the result. No pipeline logic lives in this file.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from vulnrem.core.enums import Language
from vulnrem.dashboard.formatting import format_report_markdown
from vulnrem.dashboard.service import PipelineService

_service = PipelineService(env=os.environ.get("VULNREM_ENV", "dev"))

_EXAMPLE_VULNERABLE_SNIPPET = '''def find_user_by_name(conn, username):
    query = "SELECT id, username FROM users WHERE username = '" + username + "'"
    return conn.execute(query).fetchall()
'''


def _run_check(uploaded_file, pasted_code: str, filename: str) -> str:
    if uploaded_file is not None:
        path = Path(uploaded_file.name)
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            return "That file doesn't look like text -- only single Python source files are supported."
        name = path.name
    elif pasted_code.strip():
        source = pasted_code
        name = (filename or "snippet.py").strip() or "snippet.py"
    else:
        return "Upload a `.py` file or paste some code first."

    try:
        report = _service.check_source(name, source, Language.PYTHON)
    except Exception as exc:  # noqa: BLE001 - surface the error in the UI rather than crashing the Space
        return f"**Something went wrong while checking this file:**\n\n```\n{exc}\n```"

    return format_report_markdown(report)


with gr.Blocks(title="Vulnerability Remediation") as demo:
    gr.Markdown(
        "# Vulnerability Remediation Checker\n"
        "Upload a single Python file, or paste a snippet, to scan it for known vulnerability "
        "classes (SQL injection, command injection, path traversal, hard-coded credentials) and "
        "attempt an automatic, verified fix."
    )
    with gr.Row():
        file_input = gr.File(label="Upload a .py file", file_types=[".py"])
    with gr.Row():
        with gr.Column(scale=3):
            code_input = gr.Code(label="...or paste code here", language="python", value=_EXAMPLE_VULNERABLE_SNIPPET)
        with gr.Column(scale=1):
            filename_input = gr.Textbox(label="Filename (for pasted code)", value="snippet.py")
    run_button = gr.Button("Check for vulnerabilities", variant="primary")
    output = gr.Markdown(label="Result")

    run_button.click(fn=_run_check, inputs=[file_input, code_input, filename_input], outputs=output)


if __name__ == "__main__":
    demo.launch()
