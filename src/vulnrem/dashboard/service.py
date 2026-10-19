"""The seam between the presentation layer and the core pipeline.

Nothing under :mod:`vulnrem.dashboard` besides this module imports
:mod:`vulnrem.pipeline` directly -- swapping Gradio for something else later
means rewriting :mod:`vulnrem.dashboard.app`, not this class.
"""

from __future__ import annotations

from vulnrem.config.loader import load_config
from vulnrem.core.enums import Language
from vulnrem.pipeline.runner import FileCheckReport, FileCheckRunner, build_file_check_runner


class PipelineService:
    def __init__(self, env: str = "dev"):
        self._app_config = load_config(env=env)
        self._runners: dict[Language, FileCheckRunner] = {}

    def _runner_for(self, language: Language) -> FileCheckRunner:
        if language not in self._runners:
            self._runners[language] = build_file_check_runner(self._app_config, language)
        return self._runners[language]

    def check_source(self, file_name: str, source: str, language: Language = Language.PYTHON) -> FileCheckReport:
        return self._runner_for(language).check_file(file_name, source)
