"""
Policy defaults for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines the default values used by the Policy Engine when
request context is missing or incomplete. These defaults are applied
in LENIENT mode with warnings, and cause errors in STRICT mode.

Key Design Principles:
    - Industry-standard defaults aligned with GHG Protocol
    - Configurable per-organization overrides
    - Complete documentation of default sources
    - Deterministic default resolution

Example:
    >>> from gl_normalizer_core.policy.defaults import (
    ...     DEFAULT_GWP_VERSION,
    ...     DEFAULT_BASIS,
    ...     get_org_defaults,
    ... )
    >>> print(DEFAULT_GWP_VERSION)
    'AR5'
"""

from typing import Any, Dict, Optional
from pathlib import Path
import json

import structlog

from gl_normalizer_core.policy.models import (
    PolicyDefaults,
    ReferenceConditions,
    ComplianceProfile,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Global Default Constants
# =============================================================================

DEFAULT_GWP_VERSION: str = "AR5"
"""
Default Global Warming Potential Assessment Report version.

AR5 (Fifth Assessment Report, 2014) is the most widely used version
as of 2024, aligned with GHG Protocol and most regulatory frameworks.
AR6 (2021) is increasingly being adopted but not yet universal.

Reference:
    IPCC Fifth Assessment Report (AR5), 2014
    https://www.ipcc.ch/assessment-report/ar5/
"""

DEFAULT_BASIS: str = "LHV"
"""
Default energy basis for fuel conversions.

LHV (Lower Heating Value) is the standard for GHG Protocol and most
regulatory frameworks. HHV (Higher Heating Value) is used in some
North American contexts.

Reference:
    GHG Protocol Corporate Standard, Chapter 7
    https://ghgprotocol.org/corporate-standard
"""

DEFAULT_TEMPERATURE_REF: float = 15.0  # degrees C
"""
Default reference temperature for gas volume conversions.

15 degrees Celsius (288.15 K, 59 degrees F) is the standard reference
temperature for natural gas measurement per ISO 13443 and GPA 2145.

Reference:
    ISO 13443:1996 Natural gas - Standard reference conditions
    GPA 2145 Table of Physical Properties of Hydrocarbons
"""

DEFAULT_PRESSURE_REF: float = 101.325  # kPa_abs
"""
Default reference pressure for gas volume conversions.

101.325 kPa (1 atm, 14.696 psia) is the standard atmospheric pressure
used as reference for natural gas measurement.

Reference:
    ISO 13443:1996 Natural gas - Standard reference conditions
"""

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.8
"""
Default minimum confidence score for entity resolution.

0.8 (80%) is the threshold below which entity matches are flagged
for human review. This balances automation with data quality.
"""

DEFAULT_PRECISION_DIGITS: int = 6
"""
Default number of significant digits for output values.

6 significant digits provides sufficient precision for most
sustainability calculations while avoiding false precision.
"""


# =============================================================================
# Reference Conditions Presets
# =============================================================================

REFERENCE_CONDITIONS_ISO: ReferenceConditions = ReferenceConditions(
    temperature_c=15.0,
    pressure_kpa=101.325,
)
"""
ISO 13443 standard reference conditions (15C, 101.325 kPa).

Used for natural gas measurement in most international contexts.
"""

REFERENCE_CONDITIONS_US: ReferenceConditions = ReferenceConditions(
    temperature_c=15.5556,  # 60 degrees F
    pressure_kpa=101.325,
)
"""
US standard reference conditions (60F, 14.696 psia).

Used for natural gas measurement in North American contexts.
"""

REFERENCE_CONDITIONS_NTP: ReferenceConditions = ReferenceConditions(
    temperature_c=20.0,
    pressure_kpa=101.325,
)
"""
Normal Temperature and Pressure (NTP) conditions (20C, 101.325 kPa).

Used in some scientific and laboratory contexts.
"""

REFERENCE_CONDITIONS_STP: ReferenceConditions = ReferenceConditions(
    temperature_c=0.0,
    pressure_kpa=101.325,
)
"""
Standard Temperature and Pressure (STP) conditions (0C, 101.325 kPa).

IUPAC standard; used in some European contexts for gas measurement.
"""

# Mapping of preset names to conditions
REFERENCE_CONDITIONS_PRESETS: Dict[str, ReferenceConditions] = {
    "ISO": REFERENCE_CONDITIONS_ISO,
    "US": REFERENCE_CONDITIONS_US,
    "NTP": REFERENCE_CONDITIONS_NTP,
    "STP": REFERENCE_CONDITIONS_STP,
}


# =============================================================================
# Compliance Profile Defaults
# =============================================================================

PROFILE_DEFAULTS: Dict[ComplianceProfile, Dict[str, Any]] = {
    ComplianceProfile.GHG_PROTOCOL: {
        "gwp_version": "AR5",
        "basis": "LHV",
        "reference_conditions": "ISO",
        "confidence_threshold": 0.8,
        "allow_deprecated_factors": False,
    },
    ComplianceProfile.EU_CSRD: {
        "gwp_version": "AR5",
        "basis": "LHV",
        "reference_conditions": "ISO",
        "confidence_threshold": 0.85,
        "allow_deprecated_factors": False,
    },
    ComplianceProfile.IFRS_S2: {
        "gwp_version": "AR5",
        "basis": "LHV",
        "reference_conditions": "ISO",
        "confidence_threshold": 0.85,
        "allow_deprecated_factors": False,
    },
    ComplianceProfile.EU_TAXONOMY: {
        "gwp_version": "AR5",
        "basis": "LHV",
        "reference_conditions": "ISO",
        "confidence_threshold": 0.9,
        "allow_deprecated_factors": False,
    },
    ComplianceProfile.INDIA_BRSR: {
        "gwp_version": "AR5",
        "basis": "LHV",
        "reference_conditions": "ISO",
        "confidence_threshold": 0.8,
        "allow_deprecated_factors": True,  # More lenient for emerging market
    },
    ComplianceProfile.CALIFORNIA_SB253: {
        "gwp_version": "AR5",
        "basis": "LHV",
        "reference_conditions": "US",  # US reference conditions
        "confidence_threshold": 0.85,
        "allow_deprecated_factors": False,
    },
    ComplianceProfile.US_SEC: {
        "gwp_version": "AR5",
        "basis": "LHV",
        "reference_conditions": "US",  # US reference conditions
        "confidence_threshold": 0.85,
        "allow_deprecated_factors": False,
    },
}
"""
Default values by compliance profile.

Each profile specifies the defaults that are mandated or recommended
by the corresponding regulatory framework.
"""


# =============================================================================
# Organization-Specific Defaults
# =============================================================================

# Cache for loaded organization defaults
_org_defaults_cache: Dict[str, PolicyDefaults] = {}


def get_system_defaults() -> PolicyDefaults:
    """
    Get the system-wide default policy values.

    Returns:
        PolicyDefaults with system defaults.

    Example:
        >>> defaults = get_system_defaults()
        >>> print(defaults.gwp_version)
        'AR5'
    """
    return PolicyDefaults(
        gwp_version=DEFAULT_GWP_VERSION,
        basis=DEFAULT_BASIS,
        reference_conditions=ReferenceConditions(
            temperature_c=DEFAULT_TEMPERATURE_REF,
            pressure_kpa=DEFAULT_PRESSURE_REF,
        ),
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        precision_digits=DEFAULT_PRECISION_DIGITS,
        allow_deprecated_factors=False,
        require_unit_validation=True,
    )


def get_profile_defaults(profile: ComplianceProfile) -> PolicyDefaults:
    """
    Get default values for a specific compliance profile.

    Args:
        profile: The compliance profile.

    Returns:
        PolicyDefaults with profile-specific defaults.

    Example:
        >>> defaults = get_profile_defaults(ComplianceProfile.GHG_PROTOCOL)
        >>> print(defaults.gwp_version)
        'AR5'
    """
    profile_config = PROFILE_DEFAULTS.get(profile, {})

    # Resolve reference conditions preset
    ref_conditions_key = profile_config.get("reference_conditions", "ISO")
    reference_conditions = REFERENCE_CONDITIONS_PRESETS.get(
        ref_conditions_key,
        REFERENCE_CONDITIONS_ISO,
    )

    return PolicyDefaults(
        gwp_version=profile_config.get("gwp_version", DEFAULT_GWP_VERSION),
        basis=profile_config.get("basis", DEFAULT_BASIS),
        reference_conditions=reference_conditions,
        confidence_threshold=profile_config.get(
            "confidence_threshold",
            DEFAULT_CONFIDENCE_THRESHOLD,
        ),
        precision_digits=DEFAULT_PRECISION_DIGITS,
        allow_deprecated_factors=profile_config.get(
            "allow_deprecated_factors",
            False,
        ),
        require_unit_validation=True,
    )


def get_org_defaults(
    org_id: str,
    config_path: Optional[Path] = None,
) -> Optional[PolicyDefaults]:
    """
    Load organization-specific default values from configuration.

    Defaults are loaded from a JSON file at:
    {config_path}/orgs/{org_id}/policy_defaults.json

    Args:
        org_id: Organization identifier.
        config_path: Base path for configuration files.

    Returns:
        PolicyDefaults if org config exists, None otherwise.

    Example:
        >>> defaults = get_org_defaults("org-acme")
        >>> if defaults:
        ...     print(defaults.gwp_version)
    """
    # Check cache first
    if org_id in _org_defaults_cache:
        logger.debug("Returning cached org defaults", org_id=org_id)
        return _org_defaults_cache[org_id]

    # Determine config path
    if config_path is None:
        # Default to environment variable or current directory
        import os
        config_path = Path(os.environ.get("GLNORM_CONFIG_PATH", "."))

    # Build path to org config
    org_config_file = config_path / "orgs" / org_id / "policy_defaults.json"

    if not org_config_file.exists():
        logger.debug(
            "No org defaults file found",
            org_id=org_id,
            path=str(org_config_file),
        )
        return None

    try:
        with open(org_config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Build reference conditions if specified
        ref_conditions = None
        if "reference_conditions" in config_data:
            ref_data = config_data["reference_conditions"]
            if isinstance(ref_data, str):
                # It's a preset name
                ref_conditions = REFERENCE_CONDITIONS_PRESETS.get(
                    ref_data,
                    REFERENCE_CONDITIONS_ISO,
                )
            else:
                # It's inline conditions
                ref_conditions = ReferenceConditions(
                    temperature_c=ref_data.get("temperature_c", DEFAULT_TEMPERATURE_REF),
                    pressure_kpa=ref_data.get("pressure_kpa", DEFAULT_PRESSURE_REF),
                )
        else:
            ref_conditions = ReferenceConditions(
                temperature_c=DEFAULT_TEMPERATURE_REF,
                pressure_kpa=DEFAULT_PRESSURE_REF,
            )

        defaults = PolicyDefaults(
            gwp_version=config_data.get("gwp_version", DEFAULT_GWP_VERSION),
            basis=config_data.get("basis", DEFAULT_BASIS),
            reference_conditions=ref_conditions,
            confidence_threshold=config_data.get(
                "confidence_threshold",
                DEFAULT_CONFIDENCE_THRESHOLD,
            ),
            precision_digits=config_data.get(
                "precision_digits",
                DEFAULT_PRECISION_DIGITS,
            ),
            allow_deprecated_factors=config_data.get(
                "allow_deprecated_factors",
                False,
            ),
            require_unit_validation=config_data.get(
                "require_unit_validation",
                True,
            ),
        )

        # Cache the result
        _org_defaults_cache[org_id] = defaults

        logger.info(
            "Loaded org defaults",
            org_id=org_id,
            gwp_version=defaults.gwp_version,
            basis=defaults.basis,
        )

        return defaults

    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse org defaults JSON",
            org_id=org_id,
            error=str(e),
        )
        return None
    except Exception as e:
        logger.error(
            "Failed to load org defaults",
            org_id=org_id,
            error=str(e),
            exc_info=True,
        )
        return None


def clear_org_defaults_cache(org_id: Optional[str] = None) -> None:
    """
    Clear the organization defaults cache.

    Args:
        org_id: Specific org to clear, or None to clear all.

    Example:
        >>> clear_org_defaults_cache("org-acme")  # Clear one org
        >>> clear_org_defaults_cache()  # Clear all
    """
    global _org_defaults_cache
    if org_id is None:
        _org_defaults_cache = {}
        logger.debug("Cleared all org defaults cache")
    elif org_id in _org_defaults_cache:
        del _org_defaults_cache[org_id]
        logger.debug("Cleared org defaults cache", org_id=org_id)


def merge_defaults(
    base: PolicyDefaults,
    override: PolicyDefaults,
) -> PolicyDefaults:
    """
    Merge two PolicyDefaults, with override taking precedence.

    Args:
        base: Base defaults.
        override: Override defaults (takes precedence).

    Returns:
        Merged PolicyDefaults.

    Example:
        >>> base = get_system_defaults()
        >>> override = PolicyDefaults(gwp_version="AR6")
        >>> merged = merge_defaults(base, override)
        >>> print(merged.gwp_version)
        'AR6'
    """
    return PolicyDefaults(
        gwp_version=override.gwp_version or base.gwp_version,
        basis=override.basis or base.basis,
        reference_conditions=override.reference_conditions or base.reference_conditions,
        confidence_threshold=override.confidence_threshold or base.confidence_threshold,
        precision_digits=override.precision_digits or base.precision_digits,
        allow_deprecated_factors=(
            override.allow_deprecated_factors
            if override.allow_deprecated_factors is not None
            else base.allow_deprecated_factors
        ),
        require_unit_validation=(
            override.require_unit_validation
            if override.require_unit_validation is not None
            else base.require_unit_validation
        ),
    )


# =============================================================================
# GWP Values by Assessment Report
# =============================================================================

GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR5": {
        "CO2": 1.0,
        "CH4": 28.0,
        "N2O": 265.0,
        "HFC-134a": 1300.0,
        "SF6": 23500.0,
        "NF3": 16100.0,
    },
    "AR6": {
        "CO2": 1.0,
        "CH4": 27.9,  # Slightly revised
        "N2O": 273.0,
        "HFC-134a": 1526.0,
        "SF6": 25200.0,
        "NF3": 17400.0,
    },
    "AR4": {
        "CO2": 1.0,
        "CH4": 25.0,
        "N2O": 298.0,
        "HFC-134a": 1430.0,
        "SF6": 22800.0,
        "NF3": 17200.0,
    },
}
"""
GWP values by IPCC Assessment Report version.

These values are used for converting non-CO2 greenhouse gases
to CO2 equivalent (CO2e).

Reference:
    IPCC Assessment Reports AR4, AR5, AR6
"""


def get_gwp_value(gas: str, gwp_version: str = DEFAULT_GWP_VERSION) -> Optional[float]:
    """
    Get the GWP value for a specific gas and AR version.

    Args:
        gas: Gas identifier (e.g., "CH4", "N2O").
        gwp_version: GWP Assessment Report version.

    Returns:
        GWP value or None if not found.

    Example:
        >>> gwp = get_gwp_value("CH4", "AR5")
        >>> print(gwp)
        28.0
    """
    version_values = GWP_VALUES.get(gwp_version, {})
    return version_values.get(gas.upper())


__all__ = [
    # Constants
    "DEFAULT_GWP_VERSION",
    "DEFAULT_BASIS",
    "DEFAULT_TEMPERATURE_REF",
    "DEFAULT_PRESSURE_REF",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_PRECISION_DIGITS",
    # Reference conditions
    "REFERENCE_CONDITIONS_ISO",
    "REFERENCE_CONDITIONS_US",
    "REFERENCE_CONDITIONS_NTP",
    "REFERENCE_CONDITIONS_STP",
    "REFERENCE_CONDITIONS_PRESETS",
    # Profile defaults
    "PROFILE_DEFAULTS",
    # Functions
    "get_system_defaults",
    "get_profile_defaults",
    "get_org_defaults",
    "clear_org_defaults_cache",
    "merge_defaults",
    "get_gwp_value",
    "GWP_VALUES",
]
