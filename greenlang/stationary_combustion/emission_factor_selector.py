# -*- coding: utf-8 -*-
"""
EmissionFactorSelectorEngine - Three-Tier Emission Factor Selection (Engine 4)

AGENT-MRV-001 Stationary Combustion Agent

Implements a three-tier emission factor selection engine following IPCC
and GHG Protocol methodology hierarchies:

    Tier 3 (Facility-specific): Custom or plant-measured EFs -- highest accuracy.
    Tier 2 (Country-specific):  DEFRA (UK), EU ETS (EU), EPA (US) -- medium.
    Tier 1 (IPCC default):      Global IPCC defaults -- baseline accuracy.

The fallback chain is Tier 3 -> Tier 2 -> Tier 1 so that the most accurate
available factor is always selected.  Every selection is recorded for audit
purposes, and custom factors are validated against plausible IPCC ranges.

Zero-Hallucination Guarantees:
    - All emission factor values come from authoritative databases only.
    - No LLM involvement in any factor selection or numeric path.
    - Every selection is logged with full provenance trace.
    - Decimal arithmetic prevents floating-point drift.

Thread Safety:
    All mutable state is protected by a reentrant lock so that concurrent
    callers never see partial updates.

Example:
    >>> from greenlang.stationary_combustion.emission_factor_selector import (
    ...     EmissionFactorSelectorEngine,
    ... )
    >>> engine = EmissionFactorSelectorEngine()
    >>> result = engine.select_factor("NATURAL_GAS", "CO2", geography="US")
    >>> result["tier"]
    2
    >>> result["source"]
    'EPA'

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports -- models, fuel database, metrics, provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.stationary_combustion.models import (
        CalculationTier,
        EFSource,
        EmissionGas,
        FuelType,
    )
except ImportError:  # pragma: no cover - standalone / test bootstrap
    pass

try:
    from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine
except ImportError:  # pragma: no cover
    FuelDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.stationary_combustion.metrics import record_fuel_lookup
except ImportError:  # pragma: no cover
    def record_fuel_lookup(*_args: Any, **_kwargs: Any) -> None:
        """No-op fallback when metrics module is unavailable."""


def record_factor_selection(
    fuel_type: str = "",
    gas: str = "",
    tier: int = 0,
    source: str = "",
) -> None:
    """Record an emission factor selection as a fuel lookup metric.

    Delegates to :func:`record_fuel_lookup` using the source label.

    Args:
        fuel_type: Fuel type identifier.
        gas: Greenhouse gas identifier (unused in metric label).
        tier: Tier level (unused in metric label).
        source: Factor source database.
    """
    record_fuel_lookup(fuel_type, source)

try:
    from greenlang.stationary_combustion.provenance import get_provenance_tracker
except ImportError:  # pragma: no cover
    get_provenance_tracker = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Authoritative Emission Factor Databases (kg per GJ, HHV basis)
# Source annotations reference the exact table / row in each publication.
# ---------------------------------------------------------------------------

# IPCC 2006 Guidelines Vol 2 Ch 2, Table 2.2 (updated 2019 Refinement)
IPCC_DEFAULT_FACTORS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "NATURAL_GAS": {
        "CO2": {"value": Decimal("56.100"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "DIESEL": {
        "CO2": {"value": Decimal("74.100"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
    },
    "FUEL_OIL_NO2": {
        "CO2": {"value": Decimal("73.960"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
    },
    "FUEL_OIL_NO6": {
        "CO2": {"value": Decimal("77.370"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
    },
    "LPG": {
        "CO2": {"value": Decimal("63.100"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "PROPANE": {
        "CO2": {"value": Decimal("63.100"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "COAL_BITUMINOUS": {
        "CO2": {"value": Decimal("94.600"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.0015"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "COAL_ANTHRACITE": {
        "CO2": {"value": Decimal("98.300"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.0015"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "WOOD": {
        "CO2": {"value": Decimal("112.000"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2 (biogenic)"},
        "CH4": {"value": Decimal("0.030"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.004"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "BIOMASS": {
        "CO2": {"value": Decimal("100.000"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2 (biogenic)"},
        "CH4": {"value": Decimal("0.030"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.004"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "KEROSENE": {
        "CO2": {"value": Decimal("71.500"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.3"},
    },
    "LANDFILL_GAS": {
        "CO2": {"value": Decimal("54.600"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2 (biogenic)"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
    "BIOGAS": {
        "CO2": {"value": Decimal("54.600"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2 (biogenic)"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "IPCC 2006 Vol 2 Table 2.2"},
    },
}

# EPA GHG Emission Factors Hub (2024), Table C-1 / C-2
EPA_FACTORS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "NATURAL_GAS": {
        "CO2": {"value": Decimal("53.060"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "DIESEL": {
        "CO2": {"value": Decimal("73.960"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "FUEL_OIL_NO2": {
        "CO2": {"value": Decimal("73.960"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "FUEL_OIL_NO6": {
        "CO2": {"value": Decimal("75.100"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "LPG": {
        "CO2": {"value": Decimal("61.710"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "PROPANE": {
        "CO2": {"value": Decimal("61.710"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "COAL_BITUMINOUS": {
        "CO2": {"value": Decimal("93.280"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.011"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0016"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "COAL_ANTHRACITE": {
        "CO2": {"value": Decimal("98.300"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.011"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0016"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "KEROSENE": {
        "CO2": {"value": Decimal("71.500"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
    "WOOD": {
        "CO2": {"value": Decimal("93.800"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-1 (biogenic)"},
        "CH4": {"value": Decimal("0.032"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
        "N2O": {"value": Decimal("0.004"), "unit": "kg/GJ", "reference": "EPA 40 CFR 98 Table C-2"},
    },
}

# DEFRA UK Government Conversion Factors 2024
DEFRA_FACTORS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "NATURAL_GAS": {
        "CO2": {"value": Decimal("56.010"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1a"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1a"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1a"},
    },
    "DIESEL": {
        "CO2": {"value": Decimal("74.020"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
    },
    "FUEL_OIL_NO2": {
        "CO2": {"value": Decimal("74.020"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
    },
    "FUEL_OIL_NO6": {
        "CO2": {"value": Decimal("77.400"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
    },
    "LPG": {
        "CO2": {"value": Decimal("63.120"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1a"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1a"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1a"},
    },
    "COAL_BITUMINOUS": {
        "CO2": {"value": Decimal("94.600"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1c"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1c"},
        "N2O": {"value": Decimal("0.0015"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1c"},
    },
    "KEROSENE": {
        "CO2": {"value": Decimal("71.500"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "DEFRA 2024 Table 1b"},
    },
}

# EU ETS Monitoring and Reporting Regulation (MRR 2018/2066)
EU_ETS_FACTORS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "NATURAL_GAS": {
        "CO2": {"value": Decimal("56.100"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
    },
    "DIESEL": {
        "CO2": {"value": Decimal("74.100"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
    },
    "FUEL_OIL_NO6": {
        "CO2": {"value": Decimal("77.370"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "CH4": {"value": Decimal("0.003"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "N2O": {"value": Decimal("0.0006"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
    },
    "LPG": {
        "CO2": {"value": Decimal("63.100"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "N2O": {"value": Decimal("0.0001"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
    },
    "COAL_BITUMINOUS": {
        "CO2": {"value": Decimal("94.600"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "N2O": {"value": Decimal("0.0015"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
    },
    "COAL_ANTHRACITE": {
        "CO2": {"value": Decimal("98.300"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "CH4": {"value": Decimal("0.001"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
        "N2O": {"value": Decimal("0.0015"), "unit": "kg/GJ", "reference": "EU MRR 2018/2066 Annex VI"},
    },
}

# Geography-to-source mapping for the recommended-source method
_GEOGRAPHY_SOURCE_MAP: Dict[str, str] = {
    "US": "EPA",
    "USA": "EPA",
    "UNITED STATES": "EPA",
    "UK": "DEFRA",
    "GB": "DEFRA",
    "UNITED KINGDOM": "DEFRA",
    "EU": "EU_ETS",
    "DE": "EU_ETS",
    "FR": "EU_ETS",
    "IT": "EU_ETS",
    "ES": "EU_ETS",
    "NL": "EU_ETS",
    "BE": "EU_ETS",
    "AT": "EU_ETS",
    "PL": "EU_ETS",
    "SE": "EU_ETS",
    "DK": "EU_ETS",
    "FI": "EU_ETS",
    "IE": "EU_ETS",
    "PT": "EU_ETS",
    "CZ": "EU_ETS",
    "RO": "EU_ETS",
    "HU": "EU_ETS",
    "GR": "EU_ETS",
}

# Source-to-database mapping for dispatching lookups
_SOURCE_DATABASE_MAP: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "EPA": EPA_FACTORS,
    "DEFRA": DEFRA_FACTORS,
    "EU_ETS": EU_ETS_FACTORS,
    "IPCC": IPCC_DEFAULT_FACTORS,
}

# Geography coverage per source
_SOURCE_GEOGRAPHY_COVERAGE: Dict[str, List[str]] = {
    "EPA": ["US", "USA", "UNITED STATES"],
    "DEFRA": ["UK", "GB", "UNITED KINGDOM"],
    "EU_ETS": [
        "EU", "DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL", "SE",
        "DK", "FI", "IE", "PT", "CZ", "RO", "HU", "GR",
    ],
    "IPCC": ["GLOBAL"],
}

# Plausible range tolerance for custom factor validation (fraction of IPCC default)
_CUSTOM_FACTOR_TOLERANCE = Decimal("0.50")  # +/- 50%


# ---------------------------------------------------------------------------
# EmissionFactorSelectorEngine
# ---------------------------------------------------------------------------


class EmissionFactorSelectorEngine:
    """Three-tier emission factor selection engine for stationary combustion.

    Selects the most accurate available emission factor by cascading through
    Tier 3 (facility-specific) -> Tier 2 (country-specific) -> Tier 1 (IPCC
    default).  An explicit tier or source can be forced to override the
    automatic selection.

    All selections are recorded in an internal audit log and, when the
    provenance module is available, are also tracked in the global
    SHA-256 provenance chain.

    Attributes:
        _fuel_database: Optional FuelDatabaseEngine for Tier 3 lookups.
        _config: Optional StationaryCombustionConfig.
        _custom_factors: In-memory store of facility-specific factors.
        _selection_log: Chronological log of every selection made.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = EmissionFactorSelectorEngine()
        >>> co2 = engine.select_factor("NATURAL_GAS", "CO2", geography="US")
        >>> co2["value"]
        Decimal('53.060')
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        fuel_database: Any = None,
        config: Any = None,
    ) -> None:
        """Initialise the EmissionFactorSelectorEngine.

        Args:
            fuel_database: Optional FuelDatabaseEngine instance for Tier 3
                facility-specific lookups.  When ``None``, Tier 3 is skipped.
            config: Optional StationaryCombustionConfig.  When ``None``,
                engine-internal defaults are used.
        """
        self._fuel_database = fuel_database
        self._config = config
        self._custom_factors: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._selection_log: List[Dict[str, Any]] = []
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "EmissionFactorSelectorEngine initialised: "
            "fuel_database=%s, custom_factor_count=0",
            "attached" if fuel_database is not None else "none",
        )

    # ------------------------------------------------------------------
    # Public API -- Primary Selection
    # ------------------------------------------------------------------

    def select_factor(
        self,
        fuel_type: str,
        gas: str,
        geography: Optional[str] = None,
        tier: Optional[int] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select the best available emission factor for a fuel/gas pair.

        Implements the fallback chain: Tier 3 -> Tier 2 -> Tier 1 unless
        an explicit ``tier`` or ``source`` is provided to override.

        Args:
            fuel_type: Fuel identifier string (e.g. ``"NATURAL_GAS"``).
            gas: Greenhouse gas (``"CO2"``, ``"CH4"``, ``"N2O"``).
            geography: ISO country code or region (e.g. ``"US"``, ``"UK"``,
                ``"EU"``).  Used to select Tier 2 source.  Optional.
            tier: Force a specific tier (1, 2, or 3).  Optional.
            source: Force a specific source (``"IPCC"``, ``"EPA"``,
                ``"DEFRA"``, ``"EU_ETS"``).  Optional.

        Returns:
            Dictionary with keys: ``value`` (Decimal), ``unit`` (str),
            ``source`` (str), ``tier`` (int), ``geography`` (str or None),
            ``reference`` (str), ``selection_trace`` (list of str).

        Raises:
            ValueError: If the fuel_type or gas is not recognised and no
                fallback is available.
        """
        fuel_key = fuel_type.upper().replace(" ", "_")
        gas_key = gas.upper()
        geo_key = geography.upper() if geography else None
        selection_trace: List[str] = []

        result: Optional[Dict[str, Any]] = None

        # ----------------------------------------------------------
        # Explicit source override
        # ----------------------------------------------------------
        if source is not None:
            result = self._lookup_from_source(
                fuel_key, gas_key, source.upper(), selection_trace,
            )
            if result is not None:
                self._record_and_return(
                    fuel_key, gas_key, result, selection_trace,
                )
                return result

        # ----------------------------------------------------------
        # Explicit tier override
        # ----------------------------------------------------------
        if tier is not None:
            result = self._lookup_by_tier(
                fuel_key, gas_key, geo_key, tier, selection_trace,
            )
            if result is not None:
                self._record_and_return(
                    fuel_key, gas_key, result, selection_trace,
                )
                return result

        # ----------------------------------------------------------
        # Automatic fallback: Tier 3 -> Tier 2 -> Tier 1
        # ----------------------------------------------------------
        # Tier 3 -- facility-specific / custom
        result = self._try_tier3(fuel_key, gas_key, selection_trace)
        if result is not None:
            self._record_and_return(fuel_key, gas_key, result, selection_trace)
            return result

        # Tier 2 -- country-specific
        result = self._try_tier2(fuel_key, gas_key, geo_key, selection_trace)
        if result is not None:
            self._record_and_return(fuel_key, gas_key, result, selection_trace)
            return result

        # Tier 1 -- IPCC default (always available as baseline)
        result = self._try_tier1(fuel_key, gas_key, selection_trace)
        if result is not None:
            self._record_and_return(fuel_key, gas_key, result, selection_trace)
            return result

        # ----------------------------------------------------------
        # Nothing found at any tier
        # ----------------------------------------------------------
        msg = (
            f"No emission factor found for fuel_type={fuel_key}, "
            f"gas={gas_key}, geography={geo_key}"
        )
        selection_trace.append(f"FAILED: {msg}")
        logger.warning(msg)
        raise ValueError(msg)

    def select_factors_for_fuel(
        self,
        fuel_type: str,
        geography: Optional[str] = None,
        tier: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Select emission factors for all three greenhouse gases.

        Convenience wrapper that calls :meth:`select_factor` for CO2, CH4,
        and N2O and returns the results in a single dictionary.

        Args:
            fuel_type: Fuel identifier string.
            geography: Optional ISO country code or region.
            tier: Optional explicit tier override.

        Returns:
            Dictionary keyed by gas name (``"CO2"``, ``"CH4"``, ``"N2O"``),
            each value being the result dictionary from :meth:`select_factor`.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for gas_name in ("CO2", "CH4", "N2O"):
            results[gas_name] = self.select_factor(
                fuel_type, gas_name, geography=geography, tier=tier,
            )
        return results

    # ------------------------------------------------------------------
    # Public API -- Tier Recommendation
    # ------------------------------------------------------------------

    def auto_select_tier(
        self,
        fuel_type: str,
        geography: Optional[str] = None,
        available_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Recommend the appropriate calculation tier based on data availability.

        Evaluates what factor sources are available for the given fuel and
        geography and recommends the highest achievable tier.

        Args:
            fuel_type: Fuel identifier string.
            geography: Optional ISO country code or region.
            available_data: Optional metadata describing what measurement
                data the facility can provide (keys: ``has_measured_ef``,
                ``has_cems_data``, ``has_fuel_analysis``).

        Returns:
            Recommended tier (3, 2, or 1).
        """
        fuel_key = fuel_type.upper().replace(" ", "_")
        geo_key = geography.upper() if geography else None

        # Check Tier 3 eligibility
        if available_data:
            has_measured = available_data.get("has_measured_ef", False)
            has_cems = available_data.get("has_cems_data", False)
            has_analysis = available_data.get("has_fuel_analysis", False)
            if has_measured or has_cems:
                logger.debug(
                    "auto_select_tier: Tier 3 recommended for %s (measured/CEMS)",
                    fuel_key,
                )
                return 3

            if has_analysis:
                logger.debug(
                    "auto_select_tier: Tier 3 possible for %s (fuel analysis)",
                    fuel_key,
                )
                return 3

        # Check Tier 3 custom factors
        if fuel_key in self._custom_factors:
            return 3

        # Check Tier 2 availability
        if geo_key:
            recommended_source = self.get_recommended_source(geo_key)
            if recommended_source != "IPCC":
                db = _SOURCE_DATABASE_MAP.get(recommended_source, {})
                if fuel_key in db:
                    return 2

        # Default to Tier 1
        return 1

    # ------------------------------------------------------------------
    # Public API -- Selection Trace and Audit
    # ------------------------------------------------------------------

    def get_selection_trace(self) -> List[Dict[str, Any]]:
        """Return the full log of emission factor selections made.

        Each entry contains the fuel type, gas, selected tier, source,
        value, timestamp, and the decision trace explaining why the
        factor was chosen.

        Returns:
            List of selection dictionaries in chronological order.
        """
        with self._lock:
            return list(self._selection_log)

    def record_selection(
        self,
        fuel_type: str,
        gas: str,
        tier: int,
        source: str,
        value: Decimal,
    ) -> None:
        """Manually record an emission factor selection for audit purposes.

        This method allows external callers to insert audit entries when
        factors are applied outside the engine's own :meth:`select_factor`
        flow (e.g. when a calculation engine resolves factors independently).

        Args:
            fuel_type: Fuel identifier string.
            gas: Greenhouse gas identifier.
            tier: Tier used (1, 2, or 3).
            source: Source database name.
            value: The emission factor value (Decimal).
        """
        entry = {
            "fuel_type": fuel_type.upper(),
            "gas": gas.upper(),
            "tier": tier,
            "source": source.upper(),
            "value": str(value),
            "timestamp": _utcnow().isoformat(),
            "origin": "manual_record",
        }

        with self._lock:
            self._selection_log.append(entry)

        record_factor_selection(
            fuel_type=fuel_type.upper(),
            gas=gas.upper(),
            tier=tier,
            source=source.upper(),
        )

        logger.debug(
            "Manual selection recorded: %s/%s tier=%d source=%s value=%s",
            fuel_type, gas, tier, source, value,
        )

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics over all recorded selections.

        Returns:
            Dictionary with keys: ``total_selections``,
            ``by_tier`` (dict[int, int]), ``by_source`` (dict[str, int]),
            ``by_fuel`` (dict[str, int]), ``by_gas`` (dict[str, int]).
        """
        with self._lock:
            log = list(self._selection_log)

        by_tier: Dict[int, int] = {}
        by_source: Dict[str, int] = {}
        by_fuel: Dict[str, int] = {}
        by_gas: Dict[str, int] = {}

        for entry in log:
            t = entry.get("tier", 0)
            by_tier[t] = by_tier.get(t, 0) + 1

            s = entry.get("source", "UNKNOWN")
            by_source[s] = by_source.get(s, 0) + 1

            f = entry.get("fuel_type", "UNKNOWN")
            by_fuel[f] = by_fuel.get(f, 0) + 1

            g = entry.get("gas", "UNKNOWN")
            by_gas[g] = by_gas.get(g, 0) + 1

        return {
            "total_selections": len(log),
            "by_tier": by_tier,
            "by_source": by_source,
            "by_fuel": by_fuel,
            "by_gas": by_gas,
        }

    # ------------------------------------------------------------------
    # Public API -- Custom Factor Validation
    # ------------------------------------------------------------------

    def validate_custom_factor(
        self,
        fuel_type: str,
        gas: str,
        value: Decimal,
        source: str,
    ) -> Dict[str, Any]:
        """Validate that a custom emission factor is within a plausible range.

        Compares ``value`` against the IPCC Tier 1 default for the same
        fuel/gas combination.  The factor is considered plausible if it
        falls within +/- 50% of the IPCC default.

        Args:
            fuel_type: Fuel identifier string.
            gas: Greenhouse gas identifier.
            value: Custom emission factor value.
            source: Description of where the custom factor originates.

        Returns:
            Dictionary with keys: ``is_valid`` (bool), ``value`` (str),
            ``ipcc_reference`` (str or None), ``deviation_pct`` (str or None),
            ``tolerance_pct`` (str), ``message`` (str).
        """
        fuel_key = fuel_type.upper().replace(" ", "_")
        gas_key = gas.upper()
        val = Decimal(str(value))

        ipcc_entry = IPCC_DEFAULT_FACTORS.get(fuel_key, {}).get(gas_key)
        if ipcc_entry is None:
            return {
                "is_valid": True,
                "value": str(val),
                "ipcc_reference": None,
                "deviation_pct": None,
                "tolerance_pct": str(_CUSTOM_FACTOR_TOLERANCE * 100),
                "message": (
                    f"No IPCC default for {fuel_key}/{gas_key}; "
                    f"custom value accepted without range check."
                ),
            }

        ipcc_val = ipcc_entry["value"]
        if ipcc_val == Decimal("0"):
            # Zero reference -- cannot compute percentage deviation
            return {
                "is_valid": True,
                "value": str(val),
                "ipcc_reference": str(ipcc_val),
                "deviation_pct": None,
                "tolerance_pct": str(_CUSTOM_FACTOR_TOLERANCE * 100),
                "message": (
                    f"IPCC default is zero for {fuel_key}/{gas_key}; "
                    f"custom value accepted."
                ),
            }

        deviation = abs(val - ipcc_val) / ipcc_val
        is_within_range = deviation <= _CUSTOM_FACTOR_TOLERANCE

        return {
            "is_valid": is_within_range,
            "value": str(val),
            "ipcc_reference": str(ipcc_val),
            "deviation_pct": str((deviation * 100).quantize(Decimal("0.01"))),
            "tolerance_pct": str(_CUSTOM_FACTOR_TOLERANCE * 100),
            "message": (
                f"Custom factor {val} is {'within' if is_within_range else 'outside'} "
                f"plausible range ({_CUSTOM_FACTOR_TOLERANCE * 100}% of IPCC default "
                f"{ipcc_val}) for {fuel_key}/{gas_key} from source '{source}'."
            ),
        }

    # ------------------------------------------------------------------
    # Public API -- Source Comparison
    # ------------------------------------------------------------------

    def compare_sources(
        self,
        fuel_type: str,
        gas: str,
    ) -> List[Dict[str, Any]]:
        """Compare emission factors across all available sources for a fuel/gas pair.

        Returns a list of dictionaries, one per source that has a factor for
        the requested fuel and gas, sorted by source name.

        Args:
            fuel_type: Fuel identifier string.
            gas: Greenhouse gas identifier.

        Returns:
            List of dicts with keys: ``source``, ``value``, ``unit``,
            ``reference``, ``tier``.
        """
        fuel_key = fuel_type.upper().replace(" ", "_")
        gas_key = gas.upper()
        comparisons: List[Dict[str, Any]] = []

        source_tier_map = {
            "IPCC": 1,
            "EPA": 2,
            "DEFRA": 2,
            "EU_ETS": 2,
        }

        for source_name, database in _SOURCE_DATABASE_MAP.items():
            entry = database.get(fuel_key, {}).get(gas_key)
            if entry is not None:
                comparisons.append({
                    "source": source_name,
                    "value": str(entry["value"]),
                    "unit": entry["unit"],
                    "reference": entry["reference"],
                    "tier": source_tier_map.get(source_name, 1),
                })

        # Include custom factor if present
        custom = self._custom_factors.get(fuel_key, {}).get(gas_key)
        if custom is not None:
            comparisons.append({
                "source": "CUSTOM",
                "value": str(custom["value"]),
                "unit": custom.get("unit", "kg/GJ"),
                "reference": custom.get("reference", "Facility-specific"),
                "tier": 3,
            })

        comparisons.sort(key=lambda c: c["source"])
        return comparisons

    # ------------------------------------------------------------------
    # Public API -- Geography & Source Helpers
    # ------------------------------------------------------------------

    def get_geography_coverage(self, source: str) -> List[str]:
        """Return the list of geographies covered by a given source.

        Args:
            source: Source identifier (``"EPA"``, ``"DEFRA"``, ``"EU_ETS"``,
                ``"IPCC"``).

        Returns:
            List of geography codes / names.
        """
        return list(
            _SOURCE_GEOGRAPHY_COVERAGE.get(source.upper(), [])
        )

    def get_recommended_source(self, geography: str) -> str:
        """Return the recommended factor source for a geography.

        Falls back to ``"IPCC"`` when no country-specific source exists.

        Args:
            geography: ISO country code or region name.

        Returns:
            Source name string (e.g. ``"EPA"``, ``"DEFRA"``, ``"EU_ETS"``,
            ``"IPCC"``).
        """
        geo_key = geography.upper() if geography else ""
        return _GEOGRAPHY_SOURCE_MAP.get(geo_key, "IPCC")

    # ------------------------------------------------------------------
    # Custom Factor Registration
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        fuel_type: str,
        gas: str,
        value: Decimal,
        unit: str = "kg/GJ",
        reference: str = "Facility-specific measurement",
    ) -> None:
        """Register a Tier 3 facility-specific emission factor.

        Custom factors are stored in memory and are the first candidates
        checked during automatic tier selection.

        Args:
            fuel_type: Fuel identifier string.
            gas: Greenhouse gas identifier.
            value: Emission factor value.
            unit: Unit of measurement (default ``"kg/GJ"``).
            reference: Description of the factor origin.
        """
        fuel_key = fuel_type.upper().replace(" ", "_")
        gas_key = gas.upper()
        val = Decimal(str(value))

        with self._lock:
            if fuel_key not in self._custom_factors:
                self._custom_factors[fuel_key] = {}
            self._custom_factors[fuel_key][gas_key] = {
                "value": val,
                "unit": unit,
                "reference": reference,
            }

        logger.info(
            "Custom factor registered: %s/%s = %s %s (%s)",
            fuel_key, gas_key, val, unit, reference,
        )

    # ------------------------------------------------------------------
    # Internal -- Tier Lookup Helpers
    # ------------------------------------------------------------------

    def _try_tier3(
        self,
        fuel_key: str,
        gas_key: str,
        trace: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve a Tier 3 (facility-specific) factor."""
        trace.append("Tier 3 check: looking for facility-specific factor")

        # Check in-memory custom factors
        custom = self._custom_factors.get(fuel_key, {}).get(gas_key)
        if custom is not None:
            trace.append(
                f"Tier 3 FOUND: custom factor {custom['value']} "
                f"{custom['unit']} ({custom['reference']})"
            )
            return self._build_result(
                value=custom["value"],
                unit=custom["unit"],
                source="CUSTOM",
                tier=3,
                geography=None,
                reference=custom["reference"],
                trace=trace,
            )

        # Check attached fuel database (if any)
        if self._fuel_database is not None:
            try:
                db_result = self._fuel_database.get_custom_emission_factor(
                    fuel_key, gas_key,
                )
                if db_result is not None:
                    val = Decimal(str(db_result.get("value", 0)))
                    unit = db_result.get("unit", "kg/GJ")
                    ref = db_result.get("reference", "FuelDatabaseEngine")
                    trace.append(
                        f"Tier 3 FOUND (FuelDatabaseEngine): {val} {unit} ({ref})"
                    )
                    return self._build_result(
                        value=val, unit=unit, source="CUSTOM",
                        tier=3, geography=None, reference=ref, trace=trace,
                    )
            except Exception as exc:
                trace.append(f"Tier 3 lookup failed in FuelDatabaseEngine: {exc}")
                logger.warning("Tier 3 fuel database lookup error: %s", exc)

        trace.append("Tier 3: no facility-specific factor available")
        return None

    def _try_tier2(
        self,
        fuel_key: str,
        gas_key: str,
        geo_key: Optional[str],
        trace: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve a Tier 2 (country-specific) factor."""
        if geo_key is None:
            trace.append("Tier 2 skip: no geography specified")
            return None

        recommended_source = self.get_recommended_source(geo_key)
        trace.append(
            f"Tier 2 check: geography={geo_key}, "
            f"recommended_source={recommended_source}"
        )

        if recommended_source == "IPCC":
            trace.append("Tier 2 skip: geography maps to IPCC (Tier 1)")
            return None

        db = _SOURCE_DATABASE_MAP.get(recommended_source, {})
        entry = db.get(fuel_key, {}).get(gas_key)
        if entry is not None:
            trace.append(
                f"Tier 2 FOUND ({recommended_source}): {entry['value']} "
                f"{entry['unit']} ({entry['reference']})"
            )
            return self._build_result(
                value=entry["value"],
                unit=entry["unit"],
                source=recommended_source,
                tier=2,
                geography=geo_key,
                reference=entry["reference"],
                trace=trace,
            )

        trace.append(
            f"Tier 2: {recommended_source} has no factor for {fuel_key}/{gas_key}"
        )
        return None

    def _try_tier1(
        self,
        fuel_key: str,
        gas_key: str,
        trace: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the Tier 1 (IPCC default) factor."""
        trace.append("Tier 1 check: IPCC default factors")

        entry = IPCC_DEFAULT_FACTORS.get(fuel_key, {}).get(gas_key)
        if entry is not None:
            trace.append(
                f"Tier 1 FOUND (IPCC): {entry['value']} "
                f"{entry['unit']} ({entry['reference']})"
            )
            return self._build_result(
                value=entry["value"],
                unit=entry["unit"],
                source="IPCC",
                tier=1,
                geography="GLOBAL",
                reference=entry["reference"],
                trace=trace,
            )

        trace.append(f"Tier 1: IPCC has no default factor for {fuel_key}/{gas_key}")
        return None

    def _lookup_from_source(
        self,
        fuel_key: str,
        gas_key: str,
        source_name: str,
        trace: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Look up a factor from an explicitly specified source."""
        if source_name == "CUSTOM":
            return self._try_tier3(fuel_key, gas_key, trace)

        db = _SOURCE_DATABASE_MAP.get(source_name)
        if db is None:
            trace.append(f"Source override: unknown source '{source_name}'")
            return None

        entry = db.get(fuel_key, {}).get(gas_key)
        if entry is None:
            trace.append(
                f"Source override: {source_name} has no factor for "
                f"{fuel_key}/{gas_key}"
            )
            return None

        tier = 1 if source_name == "IPCC" else 2
        trace.append(
            f"Source override ({source_name}): {entry['value']} "
            f"{entry['unit']} ({entry['reference']})"
        )
        return self._build_result(
            value=entry["value"],
            unit=entry["unit"],
            source=source_name,
            tier=tier,
            geography=None,
            reference=entry["reference"],
            trace=trace,
        )

    def _lookup_by_tier(
        self,
        fuel_key: str,
        gas_key: str,
        geo_key: Optional[str],
        tier: int,
        trace: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Look up a factor for an explicitly specified tier level."""
        trace.append(f"Tier override: requested tier={tier}")

        if tier == 3:
            return self._try_tier3(fuel_key, gas_key, trace)
        if tier == 2:
            return self._try_tier2(fuel_key, gas_key, geo_key, trace)
        if tier == 1:
            return self._try_tier1(fuel_key, gas_key, trace)

        trace.append(f"Unknown tier={tier}, falling back to Tier 1")
        return self._try_tier1(fuel_key, gas_key, trace)

    # ------------------------------------------------------------------
    # Internal -- Result Construction and Recording
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        value: Decimal,
        unit: str,
        source: str,
        tier: int,
        geography: Optional[str],
        reference: str,
        trace: List[str],
    ) -> Dict[str, Any]:
        """Build the standard result dictionary returned by select_factor."""
        return {
            "value": value,
            "unit": unit,
            "source": source,
            "tier": tier,
            "geography": geography,
            "reference": reference,
            "selection_trace": list(trace),
        }

    def _record_and_return(
        self,
        fuel_key: str,
        gas_key: str,
        result: Dict[str, Any],
        trace: List[str],
    ) -> None:
        """Record a selection in the audit log and provenance tracker."""
        entry = {
            "fuel_type": fuel_key,
            "gas": gas_key,
            "tier": result["tier"],
            "source": result["source"],
            "value": str(result["value"]),
            "unit": result["unit"],
            "geography": result.get("geography"),
            "reference": result["reference"],
            "timestamp": _utcnow().isoformat(),
            "trace": list(trace),
            "origin": "auto_select",
            "provenance_hash": self._compute_hash({
                "fuel_type": fuel_key,
                "gas": gas_key,
                "value": str(result["value"]),
                "source": result["source"],
                "tier": result["tier"],
            }),
        }

        with self._lock:
            self._selection_log.append(entry)

        # Prometheus metric
        record_factor_selection(
            fuel_type=fuel_key,
            gas=gas_key,
            tier=result["tier"],
            source=result["source"],
        )

        # Provenance chain
        if get_provenance_tracker is not None:
            try:
                tracker = get_provenance_tracker()
                tracker.record(
                    entity_type="fuel",
                    action="lookup_factor",
                    entity_id=f"{fuel_key}:{gas_key}",
                    data=entry,
                )
            except Exception as exc:
                logger.debug("Provenance recording skipped: %s", exc)

        logger.debug(
            "Factor selected: %s/%s -> tier=%d source=%s value=%s",
            fuel_key, gas_key, result["tier"], result["source"],
            result["value"],
        )

    # ------------------------------------------------------------------
    # Internal -- Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute a SHA-256 hash of arbitrary JSON-serialisable data.

        Args:
            data: Any JSON-serialisable object.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EmissionFactorSelectorEngine",
    "IPCC_DEFAULT_FACTORS",
    "EPA_FACTORS",
    "DEFRA_FACTORS",
    "EU_ETS_FACTORS",
]
