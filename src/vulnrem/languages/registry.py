"""Registry mapping :class:`~vulnrem.core.enums.Language` to a concrete
:class:`~vulnrem.languages.base.LanguageProfile` class.

To add C or Java support: implement a new ``LanguageProfile`` subclass,
add one entry to ``LANGUAGE_PROFILE_REGISTRY``, and drop a
``configs/languages/<name>.yaml``. Nothing in ``vulnrem.pipeline`` needs to
change -- nodes only ever call ``get_language_profile(language)``.
"""

from __future__ import annotations

from pathlib import Path

from vulnrem.core.enums import Language
from vulnrem.core.errors import ConfigError
from vulnrem.languages.base import LanguageProfile, load_language_spec
from vulnrem.languages.python_profile import PythonLanguageProfile

LANGUAGE_PROFILE_REGISTRY: dict[Language, type[LanguageProfile]] = {
    Language.PYTHON: PythonLanguageProfile,
    # Language.C: CLanguageProfile,       # not implemented in this build
    # Language.JAVA: JavaLanguageProfile,  # not implemented in this build
}


def get_language_profile(
    language: Language, config_dir: Path | str | None = None
) -> LanguageProfile:
    profile_cls = LANGUAGE_PROFILE_REGISTRY.get(language)
    if profile_cls is None:
        raise ConfigError(
            f"No LanguageProfile registered for {language.value!r}. "
            f"Registered: {[lang.value for lang in LANGUAGE_PROFILE_REGISTRY]}"
        )
    spec = load_language_spec(language, config_dir=config_dir)
    return profile_cls(spec)
