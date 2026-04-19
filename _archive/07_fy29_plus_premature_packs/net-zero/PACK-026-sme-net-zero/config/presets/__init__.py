# -*- coding: utf-8 -*-
"""
PACK-026 SME Net Zero Pack - Configuration Presets
====================================================

6 SME-specific YAML configuration presets covering the full diversity of
small and medium enterprise types: micro businesses, small businesses,
medium businesses, and three sector-specific presets for services,
manufacturing, and retail SMEs.

Size-Based Presets (3):
    1. micro_business.yaml       -- Micro (1-9 employees, <EUR 2M), BRONZE
    2. small_business.yaml       -- Small (10-49 employees, EUR 2-10M), SILVER
    3. medium_business.yaml      -- Medium (50-249 employees, EUR 10-50M), GOLD

Sector-Specific Presets (3):
    4. service_sme.yaml          -- Services/Tech/Consulting (S3-dominant)
    5. manufacturing_sme.yaml    -- Manufacturing/Production (S1+S2-heavy)
    6. retail_sme.yaml           -- Retail/Shops (S3-dominant, packaging focus)

All presets include:
    - SME data quality tier (BRONZE/SILVER/GOLD)
    - Simplified scope configuration
    - Pre-set targets (30-50% by 2030)
    - Quick wins focus
    - Grant preferences
    - Certification pathway
    - Accounting software preferences
    - Budget constraints

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__pack_id__ = "PACK-026"
__pack_name__ = "SME Net Zero Pack"

import copy
import os as _os
from pathlib import Path
from typing import Any, Dict, List, Optional

_PRESET_DIR = _os.path.dirname(_os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

AVAILABLE_PRESETS: Dict[str, str] = {
    "micro_business": _os.path.join(_PRESET_DIR, "micro_business.yaml"),
    "small_business": _os.path.join(_PRESET_DIR, "small_business.yaml"),
    "medium_business": _os.path.join(_PRESET_DIR, "medium_business.yaml"),
    "service_sme": _os.path.join(_PRESET_DIR, "service_sme.yaml"),
    "manufacturing_sme": _os.path.join(_PRESET_DIR, "manufacturing_sme.yaml"),
    "retail_sme": _os.path.join(_PRESET_DIR, "retail_sme.yaml"),
}

SME_SIZE_PRESET_MAP: Dict[str, str] = {
    "MICRO": "micro_business",
    "SMALL": "small_business",
    "MEDIUM": "medium_business",
}

SECTOR_PRESET_MAP: Dict[str, str] = {
    "SERVICES": "service_sme",
    "TECHNOLOGY": "service_sme",
    "CONSULTING": "service_sme",
    "PROFESSIONAL_SERVICES": "service_sme",
    "MANUFACTURING": "manufacturing_sme",
    "PRODUCTION": "manufacturing_sme",
    "RETAIL": "retail_sme",
    "ECOMMERCE": "retail_sme",
    "SHOPS": "retail_sme",
}

DEFAULT_PRESET = "small_business"

# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _ensure_yaml():
    """Import and return the ``yaml`` module (PyYAML or ruamel.yaml)."""
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load presets. "
            "Install it with: pip install pyyaml"
        )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    Lists in *override* replace those in *base* (no append).
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_preset_path(preset_name: str) -> str:
    """Return the absolute file path for the given preset name.

    Args:
        preset_name: One of the keys in ``AVAILABLE_PRESETS``.

    Returns:
        Absolute path to the YAML preset file.

    Raises:
        KeyError: If *preset_name* is not a recognised preset.
    """
    if preset_name not in AVAILABLE_PRESETS:
        raise KeyError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {sorted(AVAILABLE_PRESETS.keys())}"
        )
    return AVAILABLE_PRESETS[preset_name]


def get_preset_for_sme_size(sme_size: str) -> str:
    """Return the recommended preset name for a given SME size.

    Args:
        sme_size: One of MICRO, SMALL, MEDIUM.

    Returns:
        Preset name string (key into ``AVAILABLE_PRESETS``).

    Raises:
        KeyError: If *sme_size* is not mapped.
    """
    size_upper = sme_size.upper()
    if size_upper not in SME_SIZE_PRESET_MAP:
        raise KeyError(
            f"Unknown SME size '{sme_size}'. "
            f"Available: {sorted(SME_SIZE_PRESET_MAP.keys())}"
        )
    return SME_SIZE_PRESET_MAP[size_upper]


def get_preset_for_sector(sector: str) -> str:
    """Return the recommended sector-specific preset name.

    Args:
        sector: One of the keys in ``SECTOR_PRESET_MAP`` (case-insensitive).

    Returns:
        Preset name string.

    Raises:
        KeyError: If *sector* is not mapped.
    """
    sector_upper = sector.upper()
    if sector_upper not in SECTOR_PRESET_MAP:
        raise KeyError(
            f"Unknown sector '{sector}'. "
            f"Available: {sorted(SECTOR_PRESET_MAP.keys())}"
        )
    return SECTOR_PRESET_MAP[sector_upper]


def get_best_preset(
    sme_size: str,
    sector: Optional[str] = None,
) -> str:
    """Return the best preset for a given SME size and optional sector.

    If a sector-specific preset exists, it takes precedence over the
    size-based preset. Otherwise falls back to the size-based preset.

    Args:
        sme_size: SME size classification (MICRO, SMALL, MEDIUM).
        sector: Optional sector string for sector-specific override.

    Returns:
        Preset name string.
    """
    # Try sector-specific first
    if sector:
        sector_upper = sector.upper()
        if sector_upper in SECTOR_PRESET_MAP:
            return SECTOR_PRESET_MAP[sector_upper]

    # Fall back to size-based
    size_upper = sme_size.upper()
    if size_upper in SME_SIZE_PRESET_MAP:
        return SME_SIZE_PRESET_MAP[size_upper]

    return DEFAULT_PRESET


def load_preset(preset_name: str) -> Dict[str, Any]:
    """Load and parse a preset YAML file, returning a Python dict.

    Args:
        preset_name: One of the keys in ``AVAILABLE_PRESETS``.

    Returns:
        Parsed YAML configuration as a nested dict.

    Raises:
        KeyError: If *preset_name* is not recognised.
        FileNotFoundError: If the YAML file is missing on disk.
        yaml.YAMLError: If the YAML is malformed.
    """
    yaml = _ensure_yaml()
    path = get_preset_path(preset_name)
    if not _os.path.isfile(path):
        raise FileNotFoundError(f"Preset file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_preset_with_overrides(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a preset and deep-merge caller-supplied overrides on top.

    This is the primary entry point for engines that need to customise
    a preset at runtime (e.g. organisation-specific parameters).

    Args:
        preset_name: Base preset to load.
        overrides: Optional dict of values that override the preset.
            Nested dicts are merged recursively; other types replace.

    Returns:
        Merged configuration dict.
    """
    config = load_preset(preset_name)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def load_preset_for_sme(
    sme_size: str,
    sector: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience: resolve SME size/sector to preset, load, and apply overrides.

    Args:
        sme_size: SME size classification (e.g. ``"SMALL"``).
        sector: Optional sector for sector-specific preset selection.
        overrides: Optional runtime overrides.

    Returns:
        Merged configuration dict.
    """
    preset_name = get_best_preset(sme_size, sector)
    return load_preset_with_overrides(preset_name, overrides)


def load_all_presets() -> Dict[str, Dict[str, Any]]:
    """Load every available preset and return them keyed by name.

    Returns:
        ``{preset_name: config_dict, ...}`` for all 6 presets.
    """
    return {name: load_preset(name) for name in AVAILABLE_PRESETS}


def list_presets() -> List[str]:
    """Return a sorted list of all available preset names."""
    return sorted(AVAILABLE_PRESETS.keys())


def validate_preset(preset_name: str) -> Dict[str, Any]:
    """Load a preset and perform basic structural validation.

    Checks that required top-level sections exist and SME-specific
    fields are properly configured.

    Args:
        preset_name: Preset to validate.

    Returns:
        Dict with ``valid`` (bool), ``errors`` (list), ``warnings`` (list).
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        config = load_preset(preset_name)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}

    # Required top-level sections
    required_sections = [
        "organization",
        "data_quality",
        "boundary",
        "scope",
        "target",
        "reduction",
        "grant",
        "certification",
        "reporting",
        "performance",
        "audit_trail",
        "verification",
        "scorecard",
    ]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    # Validate SME-specific fields
    org = config.get("organization", {})
    sme_size = org.get("sme_size")
    if sme_size not in ("MICRO", "SMALL", "MEDIUM"):
        errors.append(
            f"Invalid sme_size: '{sme_size}'. Must be MICRO, SMALL, or MEDIUM."
        )

    # Validate data quality tier
    dq = config.get("data_quality", {})
    tier = dq.get("tier")
    if tier not in ("BRONZE", "SILVER", "GOLD"):
        errors.append(
            f"Invalid data quality tier: '{tier}'. Must be BRONZE, SILVER, or GOLD."
        )

    # Validate target
    target = config.get("target", {})
    if target.get("pathway_type") != "ACA":
        warnings.append(
            "SME Net Zero Pack requires ACA pathway. "
            f"Current: {target.get('pathway_type')}"
        )
    reduction_pct = target.get("near_term_reduction_pct", 0)
    if reduction_pct < 30.0:
        warnings.append(
            f"Near-term reduction target ({reduction_pct}%) is below recommended "
            f"minimum of 30% for SMEs."
        )

    # Validate budget
    reduction = config.get("reduction", {})
    max_budget = reduction.get("budget_range_max_eur", 0)
    if max_budget > 200_000:
        warnings.append(
            f"Budget range max (EUR {max_budget:,.0f}) is unusually high for an SME."
        )

    # Validate performance constraints
    perf = config.get("performance", {})
    mem = perf.get("memory_limit_mb", 0)
    if mem > 2048:
        warnings.append(
            f"Memory limit ({mem}MB) exceeds SME maximum of 2048MB."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_all_presets() -> Dict[str, Dict[str, Any]]:
    """Validate every preset and return results keyed by name.

    Returns:
        ``{preset_name: validation_result, ...}``
    """
    return {name: validate_preset(name) for name in AVAILABLE_PRESETS}


def get_preset_summary(preset_name: str) -> Dict[str, Any]:
    """Get a summary of a preset without loading the full config.

    Args:
        preset_name: Preset to summarize.

    Returns:
        Dict with key parameters: sme_size, sector, data_tier, target_pct,
        budget_range, certification, verification_level.
    """
    config = load_preset(preset_name)
    org = config.get("organization", {})
    dq = config.get("data_quality", {})
    target = config.get("target", {})
    reduction = config.get("reduction", {})
    cert = config.get("certification", {})
    verify = config.get("verification", {})

    return {
        "preset_name": preset_name,
        "sme_size": org.get("sme_size"),
        "sector": org.get("sector"),
        "data_quality_tier": dq.get("tier"),
        "near_term_reduction_pct": target.get("near_term_reduction_pct"),
        "target_year": target.get("near_term_target_year"),
        "budget_range_eur": f"{reduction.get('budget_range_min_eur', 0):,.0f}-{reduction.get('budget_range_max_eur', 0):,.0f}",
        "max_actions": reduction.get("max_actions"),
        "primary_certification": cert.get("primary_pathway"),
        "verification_level": verify.get("level"),
    }


def get_all_preset_summaries() -> Dict[str, Dict[str, Any]]:
    """Get summaries of all presets for comparison display.

    Returns:
        ``{preset_name: summary_dict, ...}``
    """
    return {name: get_preset_summary(name) for name in AVAILABLE_PRESETS}


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "AVAILABLE_PRESETS",
    "SME_SIZE_PRESET_MAP",
    "SECTOR_PRESET_MAP",
    "DEFAULT_PRESET",
    # Path helpers
    "get_preset_path",
    "get_preset_for_sme_size",
    "get_preset_for_sector",
    "get_best_preset",
    # Loaders
    "load_preset",
    "load_preset_with_overrides",
    "load_preset_for_sme",
    "load_all_presets",
    "list_presets",
    # Validation
    "validate_preset",
    "validate_all_presets",
    # Summaries
    "get_preset_summary",
    "get_all_preset_summaries",
]
