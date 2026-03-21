# -*- coding: utf-8 -*-
"""
PACK-025 Race to Zero Pack - Configuration Presets
===================================================

8 actor-type and sector-specific YAML configuration presets covering
the full diversity of Race to Zero campaign participants: large
corporates, financial institutions, cities/municipalities, regions/states,
SMEs, high emitters (heavy industry/energy/mining), service sector
organizations, and general manufacturing.

Presets:
    1. corporate_commitment.yaml    -- Large Corporate (>1000 employees)
    2. financial_institution.yaml   -- Bank/Insurance/Asset Manager (GFANZ)
    3. city_municipality.yaml       -- City/Municipality (C40/ICLEI)
    4. region_state.yaml            -- Region/State/Province (Under2)
    5. sme_business.yaml            -- SME (<250 employees, simplified)
    6. high_emitter.yaml            -- Heavy Industry/Energy/Mining (SDA)
    7. service_sector.yaml          -- Professional/Technology Services (ACA)
    8. manufacturing_sector.yaml    -- General Manufacturing (SDA/ACA)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-025 Race to Zero Pack
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__pack_id__ = "PACK-025"
__pack_name__ = "Race to Zero Pack"

import copy
import os as _os
from pathlib import Path
from typing import Any, Dict, List, Optional

_PRESET_DIR = _os.path.dirname(_os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_commitment": _os.path.join(_PRESET_DIR, "corporate_commitment.yaml"),
    "financial_institution": _os.path.join(_PRESET_DIR, "financial_institution.yaml"),
    "city_municipality": _os.path.join(_PRESET_DIR, "city_municipality.yaml"),
    "region_state": _os.path.join(_PRESET_DIR, "region_state.yaml"),
    "sme_business": _os.path.join(_PRESET_DIR, "sme_business.yaml"),
    "high_emitter": _os.path.join(_PRESET_DIR, "high_emitter.yaml"),
    "service_sector": _os.path.join(_PRESET_DIR, "service_sector.yaml"),
    "manufacturing_sector": _os.path.join(_PRESET_DIR, "manufacturing_sector.yaml"),
}

ACTOR_TYPE_PRESET_MAP: Dict[str, str] = {
    "CORPORATE": "corporate_commitment",
    "FINANCIAL_INSTITUTION": "financial_institution",
    "CITY": "city_municipality",
    "REGION": "region_state",
    "SME": "sme_business",
    "HEAVY_INDUSTRY": "high_emitter",
    "SERVICES": "service_sector",
    "MANUFACTURING": "manufacturing_sector",
}

SECTOR_PRESET_MAP: Dict[str, str] = {
    "MULTI_SECTOR": "corporate_commitment",
    "FINANCIAL_SERVICES": "financial_institution",
    "PUBLIC_SECTOR_CITY": "city_municipality",
    "PUBLIC_SECTOR_REGION": "region_state",
    "SME": "sme_business",
    "HEAVY_INDUSTRY": "high_emitter",
    "STEEL": "high_emitter",
    "CEMENT": "high_emitter",
    "CHEMICALS": "high_emitter",
    "OIL_GAS": "high_emitter",
    "MINING": "high_emitter",
    "SERVICES": "service_sector",
    "TECHNOLOGY": "service_sector",
    "PROFESSIONAL_SERVICES": "service_sector",
    "MANUFACTURING": "manufacturing_sector",
    "AUTOMOTIVE": "manufacturing_sector",
    "ELECTRONICS": "manufacturing_sector",
    "FOOD_BEVERAGE": "manufacturing_sector",
}

DEFAULT_PRESET = "corporate_commitment"

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


def get_preset_for_actor_type(actor_type: str) -> str:
    """Return the recommended preset name for a given actor type.

    Args:
        actor_type: One of CORPORATE, FINANCIAL_INSTITUTION, CITY,
            REGION, SME, HEAVY_INDUSTRY, SERVICES, MANUFACTURING.

    Returns:
        Preset name string (key into ``AVAILABLE_PRESETS``).

    Raises:
        KeyError: If *actor_type* is not mapped.
    """
    actor_upper = actor_type.upper()
    if actor_upper not in ACTOR_TYPE_PRESET_MAP:
        raise KeyError(
            f"Unknown actor type '{actor_type}'. "
            f"Available: {sorted(ACTOR_TYPE_PRESET_MAP.keys())}"
        )
    return ACTOR_TYPE_PRESET_MAP[actor_upper]


def get_preset_for_sector(sector: str) -> str:
    """Return the recommended preset name for a given sector.

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


def load_preset_for_actor(
    actor_type: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience: resolve actor type to preset, load, and apply overrides.

    Args:
        actor_type: Actor type string (e.g. ``"CORPORATE"``).
        overrides: Optional runtime overrides.

    Returns:
        Merged configuration dict.
    """
    preset_name = get_preset_for_actor_type(actor_type)
    return load_preset_with_overrides(preset_name, overrides)


def load_all_presets() -> Dict[str, Dict[str, Any]]:
    """Load every available preset and return them keyed by name.

    Returns:
        ``{preset_name: config_dict, ...}`` for all 8 presets.
    """
    return {name: load_preset(name) for name in AVAILABLE_PRESETS}


def list_presets() -> List[str]:
    """Return a sorted list of all available preset names."""
    return sorted(AVAILABLE_PRESETS.keys())


def validate_preset(preset_name: str) -> Dict[str, Any]:
    """Load a preset and perform basic structural validation.

    Checks that required top-level sections exist.

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

    required_sections = [
        "actor_type",
        "campaign",
        "pledge",
        "starting_line",
        "interim_target",
        "action_plan",
        "hleg",
        "reporting",
        "progress",
        "readiness",
        "performance",
        "audit_trail",
    ]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    recommended_sections = [
        "credibility_weights",
        "verification",
        "offset_strategy",
        "partnership",
    ]
    for section in recommended_sections:
        if section not in config:
            warnings.append(f"Missing recommended section: '{section}'")

    # Validate campaign fields
    campaign = config.get("campaign", {})
    if campaign.get("net_zero_target_year") != 2050:
        warnings.append("net_zero_target_year should be 2050 for Race to Zero")
    if campaign.get("interim_target_year") != 2030:
        warnings.append("interim_target_year should be 2030 for Race to Zero")

    # Validate starting line pillars
    sl = config.get("starting_line", {})
    if sl.get("enabled") and "pillars" not in sl:
        errors.append("Starting line enabled but no pillars defined")

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


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "__version__",
    "__pack_id__",
    "__pack_name__",
    "AVAILABLE_PRESETS",
    "ACTOR_TYPE_PRESET_MAP",
    "SECTOR_PRESET_MAP",
    "DEFAULT_PRESET",
    # Path helpers
    "get_preset_path",
    "get_preset_for_actor_type",
    "get_preset_for_sector",
    # Loaders
    "load_preset",
    "load_preset_with_overrides",
    "load_preset_for_actor",
    "load_all_presets",
    "list_presets",
    # Validation
    "validate_preset",
    "validate_all_presets",
]
