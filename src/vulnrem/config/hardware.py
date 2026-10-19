"""Resolves which :class:`~vulnrem.config.schemas.HardwareProfile` to use.

Order of precedence:

1. ``hardware.override_profile`` in config, if set -- always wins.
2. Best-effort auto-detection: query the first GPU's total VRAM via
   ``nvidia-smi`` and pick the profile whose ``[min_vram_gb, max_vram_gb)``
   bucket contains it.
3. ``hardware.fallback_profile`` if detection fails for any reason (no
   ``nvidia-smi`` on PATH, no GPU, parse error, timeout) -- this is the path
   taken in this sandbox and on any CPU-only dev machine, and it is not
   treated as an error.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from vulnrem.config.schemas import HardwareConfig, HardwareProfile

logger = logging.getLogger(__name__)


def _detect_vram_gb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        return float(first_line.strip()) / 1024.0  # MiB -> GiB
    except ValueError:
        return None


def _bucket_for_vram(vram_gb: float, profiles: dict[str, HardwareProfile]) -> str | None:
    for name, profile in profiles.items():
        upper = profile.max_vram_gb if profile.max_vram_gb is not None else float("inf")
        if profile.min_vram_gb <= vram_gb < upper:
            return name
    return None


def resolve_hardware_profile(config: HardwareConfig) -> HardwareProfile:
    if config.override_profile:
        logger.info("hardware_profile_selected", extra={"profile": config.override_profile, "source": "override"})
        return config.profiles[config.override_profile]

    vram_gb = _detect_vram_gb()
    if vram_gb is not None:
        bucket = _bucket_for_vram(vram_gb, config.profiles)
        if bucket is not None:
            logger.info(
                "hardware_profile_selected",
                extra={"profile": bucket, "source": "nvidia-smi", "detected_vram_gb": round(vram_gb, 1)},
            )
            return config.profiles[bucket]
        logger.warning(
            "hardware_profile_no_bucket_match",
            extra={"detected_vram_gb": round(vram_gb, 1), "falling_back_to": config.fallback_profile},
        )
    else:
        logger.info(
            "hardware_profile_auto_detect_unavailable",
            extra={"falling_back_to": config.fallback_profile},
        )

    return config.profiles[config.fallback_profile]
