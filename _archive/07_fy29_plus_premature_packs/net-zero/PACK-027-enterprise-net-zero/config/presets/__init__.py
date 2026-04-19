# -*- coding: utf-8 -*-
"""
PACK-027 Enterprise Net Zero Pack - Configuration Presets
===========================================================

8 enterprise sector-specific YAML configuration presets covering the full
diversity of large enterprise sectors: manufacturing, financial services,
technology, energy/utilities, retail/consumer goods, healthcare/pharma,
transport/logistics, and agriculture/food.

Sector-Specific Presets (8):
    1. manufacturing.yaml          -- Heavy industry, process emissions, SDA, ETS/CBAM
    2. financial_services.yaml     -- PCAF financed emissions, Cat 15, FINZ targets
    3. technology.yaml             -- Data center S2, hardware S3, RE100, avoided emissions
    4. energy_utilities.yaml       -- High S1, SDA mandatory, stranded assets, renewables
    5. retail_consumer.yaml        -- Supply chain Cat 1 + use-phase Cat 11, FLAG
    6. healthcare.yaml             -- Labs + procurement, anesthetic gases, cold chain
    7. transport_logistics.yaml    -- SDA transport, fleet electrification, SAF
    8. agriculture_food.yaml       -- FLAG mandatory, land use, farm-to-fork

All presets include:
    - Enterprise organization profile and sector classification
    - Multi-entity consolidation approach (financial/operational/equity)
    - Financial-grade data quality targets (+/-3% accuracy)
    - All 15 Scope 3 categories with sector-prioritized methods
    - SBTi Corporate Standard targets (ACA/SDA/FLAG/MIXED)
    - Monte Carlo scenario modeling (10,000 runs)
    - Internal carbon pricing ($50-$200/tCO2e)
    - Supply chain mapping and engagement (Tier 1-5)
    - External assurance readiness (limited -> reasonable)
    - Financial integration (carbon-adjusted P&L, CBAM, Taxonomy)
    - Full reporting suite (10 templates, 7+ regulatory frameworks)
    - Enterprise performance tuning (16GB+, 16+ threads)
    - 7-year audit trail with SHA-256 provenance
    - 12-dimension maturity scorecard

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__pack_id__ = "PACK-027"
__pack_name__ = "Enterprise Net Zero Pack"

import copy
import os as _os
from pathlib import Path
from typing import Any, Dict, List, Optional

_PRESET_DIR = _os.path.dirname(_os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

AVAILABLE_PRESETS: Dict[str, str] = {
    "manufacturing": _os.path.join(_PRESET_DIR, "manufacturing.yaml"),
    "financial_services": _os.path.join(_PRESET_DIR, "financial_services.yaml"),
    "technology": _os.path.join(_PRESET_DIR, "technology.yaml"),
    "energy_utilities": _os.path.join(_PRESET_DIR, "energy_utilities.yaml"),
    "retail_consumer": _os.path.join(_PRESET_DIR, "retail_consumer.yaml"),
    "healthcare": _os.path.join(_PRESET_DIR, "healthcare.yaml"),
    "transport_logistics": _os.path.join(_PRESET_DIR, "transport_logistics.yaml"),
    "agriculture_food": _os.path.join(_PRESET_DIR, "agriculture_food.yaml"),
}

SECTOR_PRESET_MAP: Dict[str, str] = {
    "MANUFACTURING": "manufacturing",
    "MINING_METALS": "manufacturing",
    "CHEMICALS": "manufacturing",
    "CONSTRUCTION": "manufacturing",
    "AUTOMOTIVE": "manufacturing",
    "AEROSPACE_DEFENSE": "manufacturing",
    "ENERGY_UTILITIES": "energy_utilities",
    "FINANCIAL_SERVICES": "financial_services",
    "TECHNOLOGY": "technology",
    "TELECOMMUNICATIONS": "technology",
    "MEDIA_ENTERTAINMENT": "technology",
    "PROFESSIONAL_SERVICES": "technology",
    "CONSUMER_GOODS": "retail_consumer",
    "REAL_ESTATE": "retail_consumer",
    "HOSPITALITY_LEISURE": "retail_consumer",
    "FOOD_BEVERAGE": "agriculture_food",
    "AGRICULTURE": "agriculture_food",
    "TRANSPORT_LOGISTICS": "transport_logistics",
    "HEALTHCARE_PHARMA": "healthcare",
    "EDUCATION": "technology",
    "PUBLIC_SECTOR": "manufacturing",
    "OTHER": "manufacturing",
}

DEFAULT_PRESET = "manufacturing"

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


def get_best_preset(sector: str) -> str:
    """Return the best preset for a given enterprise sector.

    Args:
        sector: Enterprise sector classification string.

    Returns:
        Preset name string.
    """
    sector_upper = sector.upper()
    if sector_upper in SECTOR_PRESET_MAP:
        return SECTOR_PRESET_MAP[sector_upper]
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


def load_preset_for_sector(
    sector: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience: resolve sector to preset, load, and apply overrides.

    Args:
        sector: Enterprise sector classification (e.g. ``"MANUFACTURING"``).
        overrides: Optional runtime overrides.

    Returns:
        Merged configuration dict.
    """
    preset_name = get_best_preset(sector)
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

    Checks that required top-level sections exist and enterprise-specific
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
        "consolidation",
        "data_quality",
        "scope",
        "target",
        "scenarios",
        "carbon_pricing",
        "scope4",
        "supply_chain",
        "assurance",
        "financial_integration",
        "reporting",
        "performance",
        "audit_trail",
        "scorecard",
    ]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    # Validate sector
    org = config.get("organization", {})
    sector = org.get("sector")
    valid_sectors = [
        "MANUFACTURING", "ENERGY_UTILITIES", "FINANCIAL_SERVICES",
        "TECHNOLOGY", "CONSUMER_GOODS", "TRANSPORT_LOGISTICS",
        "REAL_ESTATE", "HEALTHCARE_PHARMA", "MINING_METALS",
        "CHEMICALS", "TELECOMMUNICATIONS", "AUTOMOTIVE",
        "FOOD_BEVERAGE", "AGRICULTURE", "CONSTRUCTION",
        "AEROSPACE_DEFENSE", "HOSPITALITY_LEISURE",
        "PROFESSIONAL_SERVICES", "OTHER",
    ]
    if sector not in valid_sectors:
        errors.append(
            f"Invalid sector: '{sector}'. Must be one of: {valid_sectors}"
        )

    # Validate all 15 Scope 3 categories
    scope = config.get("scope", {})
    s3_cats = scope.get("scope3_categories", [])
    if len(s3_cats) < 15:
        warnings.append(
            f"Only {len(s3_cats)} Scope 3 categories configured. "
            f"Enterprise pack should include all 15."
        )

    # Validate consolidation approach
    consol = config.get("consolidation", {})
    approach = consol.get("approach")
    if approach not in ("FINANCIAL_CONTROL", "OPERATIONAL_CONTROL", "EQUITY_SHARE"):
        errors.append(
            f"Invalid consolidation approach: '{approach}'. "
            f"Must be FINANCIAL_CONTROL, OPERATIONAL_CONTROL, or EQUITY_SHARE."
        )

    # Validate SBTi pathway
    target = config.get("target", {})
    pathway = target.get("sbti_pathway")
    if pathway not in ("ACA_15C", "ACA_WB2C", "SDA", "FLAG", "MIXED"):
        errors.append(
            f"Invalid SBTi pathway: '{pathway}'. "
            f"Must be ACA_15C, ACA_WB2C, SDA, FLAG, or MIXED."
        )

    # Validate data quality
    dq = config.get("data_quality", {})
    accuracy = dq.get("accuracy_target_pct", 0)
    if accuracy > 10.0:
        warnings.append(
            f"Accuracy target (+/-{accuracy}%) is above recommended enterprise "
            f"threshold of +/-3%."
        )

    # Validate assurance
    assurance = config.get("assurance", {})
    assurance_level = assurance.get("level")
    if assurance_level not in ("LIMITED", "REASONABLE"):
        errors.append(
            f"Invalid assurance level: '{assurance_level}'. "
            f"Must be LIMITED or REASONABLE."
        )

    # Validate performance for enterprise scale
    perf = config.get("performance", {})
    mem = perf.get("memory_limit_mb", 0)
    if mem < 4096:
        warnings.append(
            f"Memory limit ({mem}MB) is below enterprise minimum of 4096MB."
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
        Dict with key parameters: sector, consolidation, data_quality,
        sbti_pathway, carbon_price, assurance_level, etc.
    """
    config = load_preset(preset_name)
    org = config.get("organization", {})
    consol = config.get("consolidation", {})
    dq = config.get("data_quality", {})
    target = config.get("target", {})
    cp = config.get("carbon_pricing", {})
    assurance = config.get("assurance", {})
    scenarios = config.get("scenarios", {})
    supply = config.get("supply_chain", {})

    return {
        "preset_name": preset_name,
        "sector": org.get("sector"),
        "consolidation_approach": consol.get("approach"),
        "entity_count": consol.get("entity_count"),
        "data_quality_target": dq.get("accuracy_target_pct"),
        "sbti_pathway": target.get("sbti_pathway"),
        "ambition_level": target.get("ambition_level"),
        "near_term_scope1_2_pct": target.get("near_term_scope1_2_reduction_pct"),
        "carbon_price_usd": cp.get("price_usd_per_tco2e"),
        "assurance_level": assurance.get("level"),
        "monte_carlo_runs": scenarios.get("monte_carlo_runs"),
        "supply_chain_tier_depth": supply.get("tier_depth"),
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
    "SECTOR_PRESET_MAP",
    "DEFAULT_PRESET",
    # Path helpers
    "get_preset_path",
    "get_preset_for_sector",
    "get_best_preset",
    # Loaders
    "load_preset",
    "load_preset_with_overrides",
    "load_preset_for_sector",
    "load_all_presets",
    "list_presets",
    # Validation
    "validate_preset",
    "validate_all_presets",
    # Summaries
    "get_preset_summary",
    "get_all_preset_summaries",
]
