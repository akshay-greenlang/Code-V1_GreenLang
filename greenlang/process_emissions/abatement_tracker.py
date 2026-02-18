# -*- coding: utf-8 -*-
"""
AbatementTrackerEngine - Abatement Technology Tracking (Engine 4 of 6)

AGENT-MRV-004: Process Emissions Agent

Tracks abatement technologies and their destruction/removal efficiencies
for industrial process emissions. Supports technology registration,
performance degradation modelling, combined efficiency calculation for
series-connected abatement trains, cost analysis, regulatory minimum
verification, and monitoring frequency management.

Abatement Technology Database (defaults):

    **Nitric Acid N2O:**
    - NSCR (Non-Selective Catalytic Reduction): 80-90%
    - SCR (Selective Catalytic Reduction): 70-85%
    - Extended absorption: 10-30%

    **Adipic Acid N2O:**
    - Thermal destruction: 95-99%
    - Catalytic reduction: 90-98%

    **Aluminum PFC:**
    - CWPB (Continuous Work Practice Best): reduces anode effects by 90%
    - Point-feed prebake optimization: 70-90%

    **Semiconductor PFC:**
    - Point-of-use (POU) abatement: 90-99.9% per gas
    - Remote plasma clean (RPC): 85-95%

    **SF6 Recovery:**
    - Recovery and recycling: 90-99%
    - Leak detection and repair: 50-80%

    **Carbon Capture:**
    - Post-combustion capture: 85-95%
    - Oxy-fuel combustion: 90-99%

    **Scrubbing:**
    - Wet scrubbing: 70-95%
    - Dry scrubbing: 60-90%

Combined Efficiency Formula (technologies in series):
    combined_efficiency = 1 - product(1 - eff_i) for each technology

Age-Adjusted Efficiency:
    eff_adjusted = eff_base * (1 - degradation_rate * age_years)
    Degradation rates: 0.5-2% per year depending on technology

Zero-Hallucination Guarantees:
    - All efficiency calculations use deterministic Python Decimal arithmetic.
    - No LLM involvement in any numeric path.
    - Every result carries a SHA-256 provenance hash.
    - Technology parameters sourced from IPCC 2006, EPA, EU ETS MRR databases.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.process_emissions.abatement_tracker import (
    ...     AbatementTrackerEngine,
    ... )
    >>> engine = AbatementTrackerEngine()
    >>> tech_id = engine.register_technology({
    ...     "name": "NSCR Catalyst",
    ...     "technology_type": "NSCR",
    ...     "process_type": "NITRIC_ACID",
    ...     "target_gas": "N2O",
    ...     "destruction_efficiency": 0.85,
    ...     "installation_date": "2022-01-15",
    ... })
    >>> eff = engine.get_abatement_efficiency(tech_id)
    >>> print(eff)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["AbatementTrackerEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.process_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_material_operation as _record_material_operation,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    _record_material_operation = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _decimal_clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    """Clamp a Decimal value to [low, high].

    Args:
        value: Value to clamp.
        low: Minimum.
        high: Maximum.

    Returns:
        Clamped Decimal.
    """
    return max(low, min(high, value))


# ===========================================================================
# Enumerations
# ===========================================================================


class AbatementTechnologyType(str, Enum):
    """Supported abatement technology types.

    NSCR: Non-Selective Catalytic Reduction (nitric acid N2O).
    SCR: Selective Catalytic Reduction (nitric acid N2O).
    EXTENDED_ABSORPTION: Extended absorption column (nitric acid N2O).
    THERMAL_DESTRUCTION: Thermal decomposition (adipic acid N2O).
    CATALYTIC_REDUCTION: Catalytic N2O reduction (adipic acid).
    CWPB: Continuous Work Practice Best (aluminum PFC anode effects).
    POINT_FEED_PREBAKE: Point-feed prebake optimization (aluminum PFC).
    POU_ABATEMENT: Point-of-use abatement (semiconductor PFC/NF3).
    REMOTE_PLASMA_CLEAN: Remote plasma clean (semiconductor).
    SF6_RECOVERY: SF6 recovery and recycling.
    SF6_LEAK_REPAIR: SF6 leak detection and repair.
    POST_COMBUSTION_CAPTURE: Post-combustion carbon capture.
    OXY_FUEL_CAPTURE: Oxy-fuel combustion capture.
    WET_SCRUBBING: Wet scrubber for acid gas / particulate removal.
    DRY_SCRUBBING: Dry sorbent injection / baghouse scrubbing.
    FLARE: Enclosed or open flare for combustible gases.
    BIOFILTER: Biofiltration for VOC / low-concentration streams.
    """

    NSCR = "NSCR"
    SCR = "SCR"
    EXTENDED_ABSORPTION = "EXTENDED_ABSORPTION"
    THERMAL_DESTRUCTION = "THERMAL_DESTRUCTION"
    CATALYTIC_REDUCTION = "CATALYTIC_REDUCTION"
    CWPB = "CWPB"
    POINT_FEED_PREBAKE = "POINT_FEED_PREBAKE"
    POU_ABATEMENT = "POU_ABATEMENT"
    REMOTE_PLASMA_CLEAN = "REMOTE_PLASMA_CLEAN"
    SF6_RECOVERY = "SF6_RECOVERY"
    SF6_LEAK_REPAIR = "SF6_LEAK_REPAIR"
    POST_COMBUSTION_CAPTURE = "POST_COMBUSTION_CAPTURE"
    OXY_FUEL_CAPTURE = "OXY_FUEL_CAPTURE"
    WET_SCRUBBING = "WET_SCRUBBING"
    DRY_SCRUBBING = "DRY_SCRUBBING"
    FLARE = "FLARE"
    BIOFILTER = "BIOFILTER"


class MaintenanceStatus(str, Enum):
    """Maintenance status for an abatement technology installation.

    OPERATIONAL: Fully operational, within specification.
    DEGRADED: Operational but below optimal performance.
    MAINTENANCE_DUE: Scheduled maintenance overdue.
    UNDER_MAINTENANCE: Currently offline for maintenance.
    FAILED: Non-operational / bypassed.
    DECOMMISSIONED: Permanently removed from service.
    """

    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    MAINTENANCE_DUE = "MAINTENANCE_DUE"
    UNDER_MAINTENANCE = "UNDER_MAINTENANCE"
    FAILED = "FAILED"
    DECOMMISSIONED = "DECOMMISSIONED"


class VerificationStatus(str, Enum):
    """Third-party verification status.

    VERIFIED: Third-party verified within validity period.
    UNVERIFIED: Not yet verified by a third party.
    EXPIRED: Verification period has lapsed.
    PENDING: Verification in progress.
    """

    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"


class MonitoringFrequency(str, Enum):
    """Required monitoring frequency for abatement systems.

    CONTINUOUS: Continuous Emissions Monitoring System (CEMS).
    DAILY: Daily spot check or automated sampling.
    WEEKLY: Weekly testing / inspection.
    MONTHLY: Monthly performance testing.
    QUARTERLY: Quarterly stack testing or audits.
    ANNUAL: Annual performance verification.
    """

    CONTINUOUS = "CONTINUOUS"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


# ===========================================================================
# Default Technology Database
# ===========================================================================

# Each entry: (default_efficiency_low, default_efficiency_high,
#              degradation_rate_per_year, typical_capex_usd_per_tco2e,
#              typical_opex_usd_per_tco2e_yr, default_monitoring_frequency)

_TECHNOLOGY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    AbatementTechnologyType.NSCR.value: {
        "efficiency_low": Decimal("0.80"),
        "efficiency_high": Decimal("0.90"),
        "default_efficiency": Decimal("0.85"),
        "degradation_rate": Decimal("0.010"),
        "typical_capex_per_tco2e": Decimal("5.00"),
        "typical_opex_per_tco2e_yr": Decimal("1.50"),
        "default_monitoring": MonitoringFrequency.MONTHLY.value,
        "applicable_gases": frozenset({"N2O"}),
        "applicable_processes": frozenset({"NITRIC_ACID"}),
        "description": "Non-Selective Catalytic Reduction for N2O from nitric acid production",
    },
    AbatementTechnologyType.SCR.value: {
        "efficiency_low": Decimal("0.70"),
        "efficiency_high": Decimal("0.85"),
        "default_efficiency": Decimal("0.78"),
        "degradation_rate": Decimal("0.012"),
        "typical_capex_per_tco2e": Decimal("4.00"),
        "typical_opex_per_tco2e_yr": Decimal("1.80"),
        "default_monitoring": MonitoringFrequency.MONTHLY.value,
        "applicable_gases": frozenset({"N2O"}),
        "applicable_processes": frozenset({"NITRIC_ACID"}),
        "description": "Selective Catalytic Reduction for N2O from nitric acid production",
    },
    AbatementTechnologyType.EXTENDED_ABSORPTION.value: {
        "efficiency_low": Decimal("0.10"),
        "efficiency_high": Decimal("0.30"),
        "default_efficiency": Decimal("0.20"),
        "degradation_rate": Decimal("0.005"),
        "typical_capex_per_tco2e": Decimal("2.00"),
        "typical_opex_per_tco2e_yr": Decimal("0.50"),
        "default_monitoring": MonitoringFrequency.QUARTERLY.value,
        "applicable_gases": frozenset({"N2O"}),
        "applicable_processes": frozenset({"NITRIC_ACID"}),
        "description": "Extended absorption column for N2O reduction in nitric acid",
    },
    AbatementTechnologyType.THERMAL_DESTRUCTION.value: {
        "efficiency_low": Decimal("0.95"),
        "efficiency_high": Decimal("0.99"),
        "default_efficiency": Decimal("0.975"),
        "degradation_rate": Decimal("0.008"),
        "typical_capex_per_tco2e": Decimal("3.00"),
        "typical_opex_per_tco2e_yr": Decimal("2.00"),
        "default_monitoring": MonitoringFrequency.MONTHLY.value,
        "applicable_gases": frozenset({"N2O"}),
        "applicable_processes": frozenset({"ADIPIC_ACID"}),
        "description": "Thermal destruction of N2O from adipic acid production",
    },
    AbatementTechnologyType.CATALYTIC_REDUCTION.value: {
        "efficiency_low": Decimal("0.90"),
        "efficiency_high": Decimal("0.98"),
        "default_efficiency": Decimal("0.94"),
        "degradation_rate": Decimal("0.010"),
        "typical_capex_per_tco2e": Decimal("4.50"),
        "typical_opex_per_tco2e_yr": Decimal("1.20"),
        "default_monitoring": MonitoringFrequency.MONTHLY.value,
        "applicable_gases": frozenset({"N2O"}),
        "applicable_processes": frozenset({"ADIPIC_ACID", "NITRIC_ACID"}),
        "description": "Catalytic reduction of N2O for chemical process emissions",
    },
    AbatementTechnologyType.CWPB.value: {
        "efficiency_low": Decimal("0.85"),
        "efficiency_high": Decimal("0.95"),
        "default_efficiency": Decimal("0.90"),
        "degradation_rate": Decimal("0.005"),
        "typical_capex_per_tco2e": Decimal("8.00"),
        "typical_opex_per_tco2e_yr": Decimal("0.80"),
        "default_monitoring": MonitoringFrequency.CONTINUOUS.value,
        "applicable_gases": frozenset({"CF4", "C2F6"}),
        "applicable_processes": frozenset({"ALUMINUM_PREBAKE", "ALUMINUM_SODERBERG"}),
        "description": "Continuous Work Practice Best for reducing anode effects in aluminum smelting",
    },
    AbatementTechnologyType.POINT_FEED_PREBAKE.value: {
        "efficiency_low": Decimal("0.70"),
        "efficiency_high": Decimal("0.90"),
        "default_efficiency": Decimal("0.80"),
        "degradation_rate": Decimal("0.005"),
        "typical_capex_per_tco2e": Decimal("6.00"),
        "typical_opex_per_tco2e_yr": Decimal("0.60"),
        "default_monitoring": MonitoringFrequency.CONTINUOUS.value,
        "applicable_gases": frozenset({"CF4", "C2F6"}),
        "applicable_processes": frozenset({"ALUMINUM_PREBAKE"}),
        "description": "Point-feed prebake cell optimization for PFC reduction",
    },
    AbatementTechnologyType.POU_ABATEMENT.value: {
        "efficiency_low": Decimal("0.90"),
        "efficiency_high": Decimal("0.999"),
        "default_efficiency": Decimal("0.95"),
        "degradation_rate": Decimal("0.015"),
        "typical_capex_per_tco2e": Decimal("15.00"),
        "typical_opex_per_tco2e_yr": Decimal("5.00"),
        "default_monitoring": MonitoringFrequency.WEEKLY.value,
        "applicable_gases": frozenset({"CF4", "C2F6", "SF6", "NF3", "CHF3", "C3F8", "C4F8"}),
        "applicable_processes": frozenset({"SEMICONDUCTOR"}),
        "description": "Point-of-use thermal/plasma abatement for semiconductor fab PFC/HFC gases",
    },
    AbatementTechnologyType.REMOTE_PLASMA_CLEAN.value: {
        "efficiency_low": Decimal("0.85"),
        "efficiency_high": Decimal("0.95"),
        "default_efficiency": Decimal("0.90"),
        "degradation_rate": Decimal("0.010"),
        "typical_capex_per_tco2e": Decimal("12.00"),
        "typical_opex_per_tco2e_yr": Decimal("3.50"),
        "default_monitoring": MonitoringFrequency.WEEKLY.value,
        "applicable_gases": frozenset({"CF4", "C2F6", "NF3", "SF6"}),
        "applicable_processes": frozenset({"SEMICONDUCTOR"}),
        "description": "Remote plasma clean chamber clean replacement for semiconductor fabs",
    },
    AbatementTechnologyType.SF6_RECOVERY.value: {
        "efficiency_low": Decimal("0.90"),
        "efficiency_high": Decimal("0.99"),
        "default_efficiency": Decimal("0.95"),
        "degradation_rate": Decimal("0.008"),
        "typical_capex_per_tco2e": Decimal("3.00"),
        "typical_opex_per_tco2e_yr": Decimal("0.80"),
        "default_monitoring": MonitoringFrequency.MONTHLY.value,
        "applicable_gases": frozenset({"SF6"}),
        "applicable_processes": frozenset({
            "MAGNESIUM", "ALUMINUM_PREBAKE", "ALUMINUM_SODERBERG",
            "SEMICONDUCTOR", "ELECTRICAL_EQUIPMENT",
        }),
        "description": "SF6 gas recovery, purification, and recycling system",
    },
    AbatementTechnologyType.SF6_LEAK_REPAIR.value: {
        "efficiency_low": Decimal("0.50"),
        "efficiency_high": Decimal("0.80"),
        "default_efficiency": Decimal("0.65"),
        "degradation_rate": Decimal("0.005"),
        "typical_capex_per_tco2e": Decimal("1.00"),
        "typical_opex_per_tco2e_yr": Decimal("0.40"),
        "default_monitoring": MonitoringFrequency.QUARTERLY.value,
        "applicable_gases": frozenset({"SF6"}),
        "applicable_processes": frozenset({
            "MAGNESIUM", "ELECTRICAL_EQUIPMENT",
        }),
        "description": "SF6 leak detection and repair programme (LDAR)",
    },
    AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value: {
        "efficiency_low": Decimal("0.85"),
        "efficiency_high": Decimal("0.95"),
        "default_efficiency": Decimal("0.90"),
        "degradation_rate": Decimal("0.008"),
        "typical_capex_per_tco2e": Decimal("50.00"),
        "typical_opex_per_tco2e_yr": Decimal("25.00"),
        "default_monitoring": MonitoringFrequency.CONTINUOUS.value,
        "applicable_gases": frozenset({"CO2"}),
        "applicable_processes": frozenset({
            "CEMENT", "LIME", "IRON_STEEL_BF_BOF", "IRON_STEEL_EAF",
            "IRON_STEEL_DRI", "AMMONIA", "HYDROGEN",
        }),
        "description": "Post-combustion CO2 capture using amine solvent or membrane",
    },
    AbatementTechnologyType.OXY_FUEL_CAPTURE.value: {
        "efficiency_low": Decimal("0.90"),
        "efficiency_high": Decimal("0.99"),
        "default_efficiency": Decimal("0.95"),
        "degradation_rate": Decimal("0.005"),
        "typical_capex_per_tco2e": Decimal("60.00"),
        "typical_opex_per_tco2e_yr": Decimal("20.00"),
        "default_monitoring": MonitoringFrequency.CONTINUOUS.value,
        "applicable_gases": frozenset({"CO2"}),
        "applicable_processes": frozenset({
            "CEMENT", "LIME", "IRON_STEEL_BF_BOF", "GLASS",
        }),
        "description": "Oxy-fuel combustion CO2 capture with near-pure O2 input",
    },
    AbatementTechnologyType.WET_SCRUBBING.value: {
        "efficiency_low": Decimal("0.70"),
        "efficiency_high": Decimal("0.95"),
        "default_efficiency": Decimal("0.85"),
        "degradation_rate": Decimal("0.012"),
        "typical_capex_per_tco2e": Decimal("2.50"),
        "typical_opex_per_tco2e_yr": Decimal("1.00"),
        "default_monitoring": MonitoringFrequency.MONTHLY.value,
        "applicable_gases": frozenset({"CO2", "SO2", "HCl", "HF"}),
        "applicable_processes": frozenset({
            "CEMENT", "LIME", "GLASS", "CERAMICS", "SODA_ASH",
            "PHOSPHORIC_ACID", "PULP_PAPER",
        }),
        "description": "Wet scrubber for acid gas and particulate removal from process exhaust",
    },
    AbatementTechnologyType.DRY_SCRUBBING.value: {
        "efficiency_low": Decimal("0.60"),
        "efficiency_high": Decimal("0.90"),
        "default_efficiency": Decimal("0.75"),
        "degradation_rate": Decimal("0.015"),
        "typical_capex_per_tco2e": Decimal("3.00"),
        "typical_opex_per_tco2e_yr": Decimal("1.50"),
        "default_monitoring": MonitoringFrequency.MONTHLY.value,
        "applicable_gases": frozenset({"CO2", "SO2", "HCl", "HF"}),
        "applicable_processes": frozenset({
            "CEMENT", "LIME", "GLASS", "CERAMICS",
        }),
        "description": "Dry sorbent injection and baghouse for process gas cleanup",
    },
    AbatementTechnologyType.FLARE.value: {
        "efficiency_low": Decimal("0.95"),
        "efficiency_high": Decimal("0.995"),
        "default_efficiency": Decimal("0.98"),
        "degradation_rate": Decimal("0.005"),
        "typical_capex_per_tco2e": Decimal("1.00"),
        "typical_opex_per_tco2e_yr": Decimal("0.30"),
        "default_monitoring": MonitoringFrequency.CONTINUOUS.value,
        "applicable_gases": frozenset({"CH4", "VOC"}),
        "applicable_processes": frozenset({
            "PETROCHEMICAL", "HYDROGEN", "AMMONIA",
        }),
        "description": "Enclosed or open flare for combustible waste gas destruction",
    },
    AbatementTechnologyType.BIOFILTER.value: {
        "efficiency_low": Decimal("0.50"),
        "efficiency_high": Decimal("0.90"),
        "default_efficiency": Decimal("0.70"),
        "degradation_rate": Decimal("0.020"),
        "typical_capex_per_tco2e": Decimal("2.00"),
        "typical_opex_per_tco2e_yr": Decimal("0.80"),
        "default_monitoring": MonitoringFrequency.WEEKLY.value,
        "applicable_gases": frozenset({"CH4", "VOC", "H2S"}),
        "applicable_processes": frozenset({
            "PULP_PAPER", "FOOD_BEVERAGE",
        }),
        "description": "Biofiltration for low-concentration organic/methane streams",
    },
}

# ---------------------------------------------------------------------------
# Technology applicability matrix: process -> applicable technology types
# ---------------------------------------------------------------------------

_PROCESS_TECHNOLOGY_MATRIX: Dict[str, List[str]] = {
    "CEMENT": [
        AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value,
        AbatementTechnologyType.OXY_FUEL_CAPTURE.value,
        AbatementTechnologyType.WET_SCRUBBING.value,
        AbatementTechnologyType.DRY_SCRUBBING.value,
    ],
    "LIME": [
        AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value,
        AbatementTechnologyType.OXY_FUEL_CAPTURE.value,
        AbatementTechnologyType.WET_SCRUBBING.value,
        AbatementTechnologyType.DRY_SCRUBBING.value,
    ],
    "GLASS": [
        AbatementTechnologyType.OXY_FUEL_CAPTURE.value,
        AbatementTechnologyType.WET_SCRUBBING.value,
        AbatementTechnologyType.DRY_SCRUBBING.value,
    ],
    "CERAMICS": [
        AbatementTechnologyType.WET_SCRUBBING.value,
        AbatementTechnologyType.DRY_SCRUBBING.value,
    ],
    "SODA_ASH": [
        AbatementTechnologyType.WET_SCRUBBING.value,
    ],
    "NITRIC_ACID": [
        AbatementTechnologyType.NSCR.value,
        AbatementTechnologyType.SCR.value,
        AbatementTechnologyType.EXTENDED_ABSORPTION.value,
        AbatementTechnologyType.CATALYTIC_REDUCTION.value,
    ],
    "ADIPIC_ACID": [
        AbatementTechnologyType.THERMAL_DESTRUCTION.value,
        AbatementTechnologyType.CATALYTIC_REDUCTION.value,
    ],
    "AMMONIA": [
        AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value,
        AbatementTechnologyType.FLARE.value,
    ],
    "HYDROGEN": [
        AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value,
        AbatementTechnologyType.FLARE.value,
    ],
    "IRON_STEEL_BF_BOF": [
        AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value,
        AbatementTechnologyType.OXY_FUEL_CAPTURE.value,
    ],
    "IRON_STEEL_EAF": [
        AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value,
    ],
    "IRON_STEEL_DRI": [
        AbatementTechnologyType.POST_COMBUSTION_CAPTURE.value,
    ],
    "ALUMINUM_PREBAKE": [
        AbatementTechnologyType.CWPB.value,
        AbatementTechnologyType.POINT_FEED_PREBAKE.value,
        AbatementTechnologyType.SF6_RECOVERY.value,
    ],
    "ALUMINUM_SODERBERG": [
        AbatementTechnologyType.CWPB.value,
        AbatementTechnologyType.SF6_RECOVERY.value,
    ],
    "MAGNESIUM": [
        AbatementTechnologyType.SF6_RECOVERY.value,
        AbatementTechnologyType.SF6_LEAK_REPAIR.value,
    ],
    "SEMICONDUCTOR": [
        AbatementTechnologyType.POU_ABATEMENT.value,
        AbatementTechnologyType.REMOTE_PLASMA_CLEAN.value,
        AbatementTechnologyType.SF6_RECOVERY.value,
    ],
    "ELECTRICAL_EQUIPMENT": [
        AbatementTechnologyType.SF6_RECOVERY.value,
        AbatementTechnologyType.SF6_LEAK_REPAIR.value,
    ],
    "PETROCHEMICAL": [
        AbatementTechnologyType.FLARE.value,
        AbatementTechnologyType.WET_SCRUBBING.value,
    ],
    "PHOSPHORIC_ACID": [
        AbatementTechnologyType.WET_SCRUBBING.value,
    ],
    "PULP_PAPER": [
        AbatementTechnologyType.WET_SCRUBBING.value,
        AbatementTechnologyType.BIOFILTER.value,
    ],
    "FOOD_BEVERAGE": [
        AbatementTechnologyType.BIOFILTER.value,
    ],
}

# ---------------------------------------------------------------------------
# Regulatory minimum efficiency requirements per framework
# ---------------------------------------------------------------------------

_REGULATORY_MINIMUMS: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    # Framework -> ProcessType -> MinimumEfficiency
    "EU_ETS_MRR": {
        "NITRIC_ACID": {"N2O": Decimal("0.70")},
        "ADIPIC_ACID": {"N2O": Decimal("0.90")},
        "ALUMINUM_PREBAKE": {"CF4": Decimal("0.80")},
        "ALUMINUM_SODERBERG": {"CF4": Decimal("0.70")},
    },
    "EPA_PART_98": {
        "NITRIC_ACID": {"N2O": Decimal("0.60")},
        "ADIPIC_ACID": {"N2O": Decimal("0.85")},
        "SEMICONDUCTOR": {"CF4": Decimal("0.80"), "C2F6": Decimal("0.80")},
    },
    "GHG_PROTOCOL": {
        "NITRIC_ACID": {"N2O": Decimal("0.50")},
        "ADIPIC_ACID": {"N2O": Decimal("0.80")},
    },
    "ISO_14064": {
        # ISO requires documented efficiency; no prescriptive minimum
    },
    "CSRD_ESRS_E1": {
        "NITRIC_ACID": {"N2O": Decimal("0.60")},
        "ADIPIC_ACID": {"N2O": Decimal("0.85")},
    },
    "UK_SECR": {
        # UK SECR defers to best available techniques
    },
}

# ---------------------------------------------------------------------------
# Monitoring frequency requirements per framework
# ---------------------------------------------------------------------------

_MONITORING_REQUIREMENTS: Dict[str, Dict[str, str]] = {
    "EU_ETS_MRR": {
        "NITRIC_ACID": MonitoringFrequency.CONTINUOUS.value,
        "ADIPIC_ACID": MonitoringFrequency.CONTINUOUS.value,
        "ALUMINUM_PREBAKE": MonitoringFrequency.CONTINUOUS.value,
        "ALUMINUM_SODERBERG": MonitoringFrequency.CONTINUOUS.value,
        "SEMICONDUCTOR": MonitoringFrequency.MONTHLY.value,
        "DEFAULT": MonitoringFrequency.MONTHLY.value,
    },
    "EPA_PART_98": {
        "NITRIC_ACID": MonitoringFrequency.CONTINUOUS.value,
        "ADIPIC_ACID": MonitoringFrequency.CONTINUOUS.value,
        "SEMICONDUCTOR": MonitoringFrequency.QUARTERLY.value,
        "DEFAULT": MonitoringFrequency.QUARTERLY.value,
    },
    "GHG_PROTOCOL": {
        "DEFAULT": MonitoringFrequency.ANNUAL.value,
    },
    "ISO_14064": {
        "DEFAULT": MonitoringFrequency.ANNUAL.value,
    },
    "CSRD_ESRS_E1": {
        "DEFAULT": MonitoringFrequency.ANNUAL.value,
    },
    "UK_SECR": {
        "DEFAULT": MonitoringFrequency.ANNUAL.value,
    },
}

# Decimal precision constant
_PRECISION_8 = Decimal("0.00000001")
_PRECISION_4 = Decimal("0.0001")
_PRECISION_2 = Decimal("0.01")

_ZERO = Decimal("0")
_ONE = Decimal("1")


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class AbatementTechnology:
    """Registered abatement technology installation.

    Attributes:
        technology_id: Unique identifier.
        name: Human-readable name.
        technology_type: Type key (see AbatementTechnologyType).
        process_type: Industrial process this is installed on.
        target_gas: Primary gas being abated.
        base_efficiency: Nameplate destruction / removal efficiency [0-1].
        current_efficiency: Age- and maintenance-adjusted efficiency.
        installation_date: ISO date of installation.
        age_years: Years since installation.
        degradation_rate: Annual performance loss as fraction.
        maintenance_status: Current maintenance status.
        verification_status: Third-party verification status.
        verification_expiry: ISO date when verification expires.
        capex_usd: Capital expenditure (USD).
        annual_opex_usd: Annual operating expenditure (USD).
        annual_abated_tco2e: Annual tonnes CO2e abated.
        cost_per_tco2e: Cost per tonne CO2e avoided (annualized).
        monitoring_frequency: Required monitoring frequency.
        last_inspection_date: ISO date of last inspection.
        next_inspection_date: ISO date of next scheduled inspection.
        metadata: Additional metadata.
        provenance_hash: SHA-256 hash.
        created_at: ISO timestamp of registration.
        updated_at: ISO timestamp of last update.
    """

    technology_id: str
    name: str
    technology_type: str
    process_type: str
    target_gas: str
    base_efficiency: Decimal
    current_efficiency: Decimal
    installation_date: str
    age_years: Decimal
    degradation_rate: Decimal
    maintenance_status: str
    verification_status: str
    verification_expiry: Optional[str]
    capex_usd: Decimal
    annual_opex_usd: Decimal
    annual_abated_tco2e: Decimal
    cost_per_tco2e: Decimal
    monitoring_frequency: str
    last_inspection_date: Optional[str]
    next_inspection_date: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary with string Decimals."""
        return {
            "technology_id": self.technology_id,
            "name": self.name,
            "technology_type": self.technology_type,
            "process_type": self.process_type,
            "target_gas": self.target_gas,
            "base_efficiency": str(self.base_efficiency),
            "current_efficiency": str(self.current_efficiency),
            "installation_date": self.installation_date,
            "age_years": str(self.age_years),
            "degradation_rate": str(self.degradation_rate),
            "maintenance_status": self.maintenance_status,
            "verification_status": self.verification_status,
            "verification_expiry": self.verification_expiry,
            "capex_usd": str(self.capex_usd),
            "annual_opex_usd": str(self.annual_opex_usd),
            "annual_abated_tco2e": str(self.annual_abated_tco2e),
            "cost_per_tco2e": str(self.cost_per_tco2e),
            "monitoring_frequency": self.monitoring_frequency,
            "last_inspection_date": self.last_inspection_date,
            "next_inspection_date": self.next_inspection_date,
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class AbatementPerformance:
    """Performance snapshot for an abatement installation at a point in time.

    Attributes:
        record_id: Unique record identifier.
        technology_id: Abatement technology identifier.
        measurement_date: ISO date of measurement.
        measured_efficiency: Efficiency measured during testing.
        expected_efficiency: Age-adjusted expected efficiency.
        deviation_pct: Percentage deviation from expected.
        is_within_spec: True if deviation is acceptable.
        notes: Operator notes.
        provenance_hash: SHA-256 hash.
    """

    record_id: str
    technology_id: str
    measurement_date: str
    measured_efficiency: Decimal
    expected_efficiency: Decimal
    deviation_pct: Decimal
    is_within_spec: bool
    notes: str
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary."""
        return {
            "record_id": self.record_id,
            "technology_id": self.technology_id,
            "measurement_date": self.measurement_date,
            "measured_efficiency": str(self.measured_efficiency),
            "expected_efficiency": str(self.expected_efficiency),
            "deviation_pct": str(self.deviation_pct),
            "is_within_spec": self.is_within_spec,
            "notes": self.notes,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class AbatementCostAnalysis:
    """Cost analysis for an abatement technology or train.

    Attributes:
        analysis_id: Unique analysis identifier.
        technology_ids: List of technology IDs analysed.
        total_capex_usd: Total capital expenditure.
        total_annual_opex_usd: Total annual operating cost.
        total_annual_abated_tco2e: Total annual abated emissions.
        levelized_cost_per_tco2e: Levelized cost per tonne CO2e.
        payback_years: Simple payback in years (capex / annual_savings).
        npv_usd: Net present value at discount rate.
        discount_rate: Discount rate used for NPV.
        analysis_period_years: Analysis horizon.
        carbon_price_usd_per_tco2e: Assumed carbon price.
        provenance_hash: SHA-256 hash.
        timestamp: ISO timestamp.
    """

    analysis_id: str
    technology_ids: List[str]
    total_capex_usd: Decimal
    total_annual_opex_usd: Decimal
    total_annual_abated_tco2e: Decimal
    levelized_cost_per_tco2e: Decimal
    payback_years: Decimal
    npv_usd: Decimal
    discount_rate: Decimal
    analysis_period_years: int
    carbon_price_usd_per_tco2e: Decimal
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "technology_ids": self.technology_ids,
            "total_capex_usd": str(self.total_capex_usd),
            "total_annual_opex_usd": str(self.total_annual_opex_usd),
            "total_annual_abated_tco2e": str(self.total_annual_abated_tco2e),
            "levelized_cost_per_tco2e": str(self.levelized_cost_per_tco2e),
            "payback_years": str(self.payback_years),
            "npv_usd": str(self.npv_usd),
            "discount_rate": str(self.discount_rate),
            "analysis_period_years": self.analysis_period_years,
            "carbon_price_usd_per_tco2e": str(self.carbon_price_usd_per_tco2e),
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


# ===========================================================================
# AbatementTrackerEngine
# ===========================================================================


class AbatementTrackerEngine:
    """Abatement technology tracking engine for process emissions.

    Provides an in-memory registry for abatement technologies with
    performance tracking, age-based efficiency degradation, combined
    train efficiency computation, cost analysis, and regulatory minimum
    compliance verification.

    Zero-Hallucination:
        All calculations are deterministic Decimal arithmetic. No LLM
        is involved in any numeric path.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Example:
        >>> engine = AbatementTrackerEngine()
        >>> tid = engine.register_technology({
        ...     "name": "NSCR Unit A",
        ...     "technology_type": "NSCR",
        ...     "process_type": "NITRIC_ACID",
        ...     "target_gas": "N2O",
        ...     "destruction_efficiency": 0.85,
        ...     "installation_date": "2022-01-15",
        ... })
        >>> eff = engine.get_abatement_efficiency(tid)
        >>> combined = engine.calculate_combined_efficiency([tid])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AbatementTrackerEngine.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - max_technologies: Maximum registered technologies (default 10000).
                - max_performance_records: Max per-technology records (default 50000).
                - default_spec_tolerance_pct: Acceptable deviation from
                  expected efficiency (default 10).
                - default_verification_validity_years: Years a verification
                  remains valid (default 3).
        """
        self._config = config or {}
        self._max_technologies: int = self._config.get("max_technologies", 10_000)
        self._max_perf_records: int = self._config.get("max_performance_records", 50_000)
        self._spec_tolerance_pct: Decimal = _to_decimal(
            self._config.get("default_spec_tolerance_pct", 10)
        )
        self._verification_validity_years: int = self._config.get(
            "default_verification_validity_years", 3
        )

        self._technologies: Dict[str, AbatementTechnology] = {}
        self._performance_history: Dict[str, List[AbatementPerformance]] = {}
        self._lock = threading.RLock()

        logger.info(
            "AbatementTrackerEngine initialized: %d default technology types, "
            "max_technologies=%d, spec_tolerance=%.1f%%",
            len(_TECHNOLOGY_DEFAULTS),
            self._max_technologies,
            self._spec_tolerance_pct,
        )

    # ------------------------------------------------------------------
    # Public API: Technology Registration
    # ------------------------------------------------------------------

    def register_technology(self, registration: Dict[str, Any]) -> str:
        """Register an abatement technology installation.

        Args:
            registration: Technology registration data. Required keys:
                - technology_type (str): One of AbatementTechnologyType values.
                - process_type (str): Industrial process type.
                - target_gas (str): Primary gas being abated.
                Optional keys:
                - technology_id (str): Custom ID (auto-generated if absent).
                - name (str): Human-readable name.
                - destruction_efficiency (float): Base efficiency [0-1].
                - installation_date (str): ISO date (defaults to today).
                - degradation_rate (float): Annual degradation fraction.
                - capex_usd (float): Capital cost.
                - annual_opex_usd (float): Annual operating cost.
                - annual_abated_tco2e (float): Annual abated tonnes CO2e.
                - monitoring_frequency (str): Monitoring frequency.
                - verification_status (str): VERIFIED / UNVERIFIED / etc.
                - verification_expiry (str): ISO date.
                - metadata (dict): Additional metadata.

        Returns:
            Technology ID string.

        Raises:
            ValueError: If required fields are missing, type is unknown,
                or maximum capacity is reached.
        """
        with self._lock:
            if len(self._technologies) >= self._max_technologies:
                raise ValueError(
                    f"Maximum technology registrations reached ({self._max_technologies})"
                )

            tech_type = registration.get("technology_type", "").upper().strip()
            if tech_type not in _TECHNOLOGY_DEFAULTS:
                valid_types = sorted(_TECHNOLOGY_DEFAULTS.keys())
                raise ValueError(
                    f"Unknown technology_type '{tech_type}'. "
                    f"Valid types: {valid_types}"
                )

            process_type = registration.get("process_type", "").upper().strip()
            if not process_type:
                raise ValueError("process_type is required")

            target_gas = registration.get("target_gas", "").upper().strip()
            if not target_gas:
                raise ValueError("target_gas is required")

            defaults = _TECHNOLOGY_DEFAULTS[tech_type]

            tech_id = registration.get("technology_id") or f"abate_{uuid4().hex[:12]}"
            if tech_id in self._technologies:
                raise ValueError(f"Technology ID already exists: '{tech_id}'")

            # Base efficiency
            if "destruction_efficiency" in registration:
                base_eff = _to_decimal(registration["destruction_efficiency"])
                base_eff = _decimal_clamp(base_eff, _ZERO, _ONE)
            else:
                base_eff = defaults["default_efficiency"]

            # Installation date and age
            install_date_str = registration.get(
                "installation_date", _utcnow().date().isoformat()
            )
            age_years = self._calculate_age_years(install_date_str)

            # Degradation rate
            if "degradation_rate" in registration:
                deg_rate = _to_decimal(registration["degradation_rate"])
                deg_rate = _decimal_clamp(deg_rate, _ZERO, Decimal("0.10"))
            else:
                deg_rate = defaults["degradation_rate"]

            # Current efficiency (age-adjusted)
            current_eff = self._age_adjusted_efficiency(base_eff, deg_rate, age_years)

            # Cost parameters
            capex = _to_decimal(registration.get("capex_usd", 0))
            annual_opex = _to_decimal(registration.get("annual_opex_usd", 0))
            annual_abated = _to_decimal(registration.get("annual_abated_tco2e", 0))

            cost_per_tco2e = _ZERO
            if annual_abated > _ZERO:
                annualized_capex = capex / Decimal("10")  # 10-year amortization
                cost_per_tco2e = (
                    (annualized_capex + annual_opex) / annual_abated
                ).quantize(_PRECISION_2, rounding=ROUND_HALF_UP)

            # Monitoring frequency
            monitoring = registration.get(
                "monitoring_frequency",
                defaults["default_monitoring"],
            ).upper().strip()

            # Verification
            ver_status = registration.get(
                "verification_status", VerificationStatus.UNVERIFIED.value
            ).upper().strip()
            ver_expiry = registration.get("verification_expiry")

            now = _utcnow()

            name = registration.get("name", f"{tech_type} ({process_type})")

            provenance_data = {
                "technology_id": tech_id,
                "technology_type": tech_type,
                "process_type": process_type,
                "target_gas": target_gas,
                "base_efficiency": str(base_eff),
                "installation_date": install_date_str,
            }
            provenance_hash = self._compute_hash("register_technology", provenance_data)

            technology = AbatementTechnology(
                technology_id=tech_id,
                name=name,
                technology_type=tech_type,
                process_type=process_type,
                target_gas=target_gas,
                base_efficiency=base_eff,
                current_efficiency=current_eff,
                installation_date=install_date_str,
                age_years=age_years,
                degradation_rate=deg_rate,
                maintenance_status=MaintenanceStatus.OPERATIONAL.value,
                verification_status=ver_status,
                verification_expiry=ver_expiry,
                capex_usd=capex,
                annual_opex_usd=annual_opex,
                annual_abated_tco2e=annual_abated,
                cost_per_tco2e=cost_per_tco2e,
                monitoring_frequency=monitoring,
                last_inspection_date=None,
                next_inspection_date=None,
                metadata=registration.get("metadata", {}),
                provenance_hash=provenance_hash,
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
            )

            self._technologies[tech_id] = technology
            self._performance_history[tech_id] = []

            logger.info(
                "Abatement technology registered: %s (type=%s, process=%s, "
                "gas=%s, base_eff=%.3f, current_eff=%.3f, age=%.1f yr)",
                tech_id, tech_type, process_type, target_gas,
                base_eff, current_eff, age_years,
            )

            self._record_metrics("register", tech_type)
            return tech_id

    # ------------------------------------------------------------------
    # Public API: Efficiency Retrieval
    # ------------------------------------------------------------------

    def get_abatement_efficiency(
        self,
        technology_id: str,
        as_of_date: Optional[str] = None,
    ) -> Decimal:
        """Get current or projected abatement efficiency.

        Returns the age-adjusted and maintenance-adjusted efficiency for
        a registered technology. If as_of_date is provided, projects the
        efficiency to that future date.

        Args:
            technology_id: Registered technology identifier.
            as_of_date: Optional ISO date for projected efficiency.
                Defaults to today.

        Returns:
            Efficiency as Decimal in [0, 1].

        Raises:
            ValueError: If technology_id is not found.
        """
        with self._lock:
            tech = self._get_technology(technology_id)

            if as_of_date:
                age = self._calculate_age_years(
                    tech.installation_date, as_of_date
                )
            else:
                age = self._calculate_age_years(tech.installation_date)

            base_eff = self._age_adjusted_efficiency(
                tech.base_efficiency, tech.degradation_rate, age
            )

            # Apply maintenance adjustment
            adjusted = self._maintenance_adjusted_efficiency(
                base_eff, tech.maintenance_status
            )

            return adjusted.quantize(_PRECISION_4, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Public API: Combined Efficiency
    # ------------------------------------------------------------------

    def calculate_combined_efficiency(
        self,
        technology_ids: List[str],
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate combined efficiency for technologies in series.

        For multiple abatement systems in series, the combined efficiency
        is: combined = 1 - product(1 - eff_i) for each technology.

        Args:
            technology_ids: Ordered list of technology IDs in the series.
            as_of_date: Optional ISO date for projected efficiency.

        Returns:
            Dictionary with combined_efficiency, individual efficiencies,
            and provenance hash.

        Raises:
            ValueError: If technology_ids is empty or any ID not found.
        """
        if not technology_ids:
            raise ValueError("technology_ids must not be empty")

        t_start = time.monotonic()

        individual: Dict[str, Decimal] = {}
        product_complement = _ONE  # product of (1 - eff_i)

        with self._lock:
            for tid in technology_ids:
                eff = self.get_abatement_efficiency(tid, as_of_date)
                individual[tid] = eff
                complement = _ONE - eff
                product_complement = product_complement * complement

        combined = (_ONE - product_complement).quantize(
            _PRECISION_4, rounding=ROUND_HALF_UP
        )

        elapsed_ms = (time.monotonic() - t_start) * 1000

        provenance_data = {
            "technology_ids": technology_ids,
            "individual_efficiencies": {k: str(v) for k, v in individual.items()},
            "combined_efficiency": str(combined),
        }
        provenance_hash = self._compute_hash("combined_efficiency", provenance_data)

        logger.debug(
            "Combined efficiency for %d technologies: %.4f (%.1fms)",
            len(technology_ids), combined, elapsed_ms,
        )

        return {
            "combined_efficiency": combined,
            "individual_efficiencies": individual,
            "technology_count": len(technology_ids),
            "technology_ids": technology_ids,
            "as_of_date": as_of_date or _utcnow().date().isoformat(),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 3),
        }

    # ------------------------------------------------------------------
    # Public API: Applicable Technologies
    # ------------------------------------------------------------------

    def get_applicable_technologies(
        self,
        process_type: str,
        target_gas: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get technologies applicable to a given process and gas.

        Args:
            process_type: Industrial process type (e.g. CEMENT, NITRIC_ACID).
            target_gas: Optional gas filter (e.g. N2O, CF4, CO2).

        Returns:
            List of technology type descriptors with efficiency ranges,
            costs, and applicability notes.
        """
        process_key = process_type.upper().strip()
        applicable_types = _PROCESS_TECHNOLOGY_MATRIX.get(process_key, [])

        results: List[Dict[str, Any]] = []

        for tech_type in applicable_types:
            defaults = _TECHNOLOGY_DEFAULTS.get(tech_type, {})
            if not defaults:
                continue

            # Gas filter
            applicable_gases: FrozenSet[str] = defaults.get("applicable_gases", frozenset())
            if target_gas:
                gas_upper = target_gas.upper().strip()
                if gas_upper not in applicable_gases:
                    continue

            results.append({
                "technology_type": tech_type,
                "description": defaults.get("description", ""),
                "efficiency_low": str(defaults["efficiency_low"]),
                "efficiency_high": str(defaults["efficiency_high"]),
                "default_efficiency": str(defaults["default_efficiency"]),
                "degradation_rate_per_year": str(defaults["degradation_rate"]),
                "typical_capex_per_tco2e": str(defaults["typical_capex_per_tco2e"]),
                "typical_opex_per_tco2e_yr": str(defaults["typical_opex_per_tco2e_yr"]),
                "default_monitoring_frequency": defaults["default_monitoring"],
                "applicable_gases": sorted(applicable_gases),
            })

        logger.debug(
            "Applicable technologies for %s (gas=%s): %d found",
            process_key, target_gas, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Performance Tracking
    # ------------------------------------------------------------------

    def track_performance(
        self,
        technology_id: str,
        measured_efficiency: float,
        measurement_date: Optional[str] = None,
        notes: str = "",
    ) -> AbatementPerformance:
        """Record a performance measurement for a technology.

        Compares measured efficiency against the age-adjusted expected
        efficiency and flags deviations that exceed the spec tolerance.

        Args:
            technology_id: Technology identifier.
            measured_efficiency: Measured destruction efficiency [0-1].
            measurement_date: ISO date of measurement. Defaults to today.
            notes: Operator notes.

        Returns:
            AbatementPerformance record.

        Raises:
            ValueError: If technology_id not found or efficiency out of range.
        """
        measured = _to_decimal(measured_efficiency)
        measured = _decimal_clamp(measured, _ZERO, _ONE)

        if measurement_date is None:
            measurement_date = _utcnow().date().isoformat()

        with self._lock:
            tech = self._get_technology(technology_id)

            expected = self.get_abatement_efficiency(
                technology_id, measurement_date
            )

            deviation_pct = _ZERO
            if expected > _ZERO:
                deviation_pct = (
                    (measured - expected) / expected * Decimal("100")
                ).quantize(_PRECISION_2, rounding=ROUND_HALF_UP)

            is_within_spec = abs(deviation_pct) <= self._spec_tolerance_pct

            record_id = f"perf_{uuid4().hex[:12]}"
            provenance_data = {
                "technology_id": technology_id,
                "measured": str(measured),
                "expected": str(expected),
                "deviation_pct": str(deviation_pct),
                "date": measurement_date,
            }
            provenance_hash = self._compute_hash("track_performance", provenance_data)

            record = AbatementPerformance(
                record_id=record_id,
                technology_id=technology_id,
                measurement_date=measurement_date,
                measured_efficiency=measured,
                expected_efficiency=expected,
                deviation_pct=deviation_pct,
                is_within_spec=is_within_spec,
                notes=notes,
                provenance_hash=provenance_hash,
            )

            # Store history
            hist = self._performance_history.get(technology_id, [])
            if len(hist) >= self._max_perf_records:
                hist.pop(0)
            hist.append(record)
            self._performance_history[technology_id] = hist

            # Update technology record
            tech.current_efficiency = measured
            tech.last_inspection_date = measurement_date
            tech.updated_at = _utcnow().isoformat()

            # Auto-detect degraded status
            if not is_within_spec and deviation_pct < _ZERO:
                if tech.maintenance_status == MaintenanceStatus.OPERATIONAL.value:
                    tech.maintenance_status = MaintenanceStatus.DEGRADED.value
                    logger.warning(
                        "Technology %s degraded: measured=%.4f expected=%.4f "
                        "(deviation=%.2f%%)",
                        technology_id, measured, expected, deviation_pct,
                    )

            logger.info(
                "Performance recorded: %s measured=%.4f expected=%.4f "
                "deviation=%.2f%% within_spec=%s",
                technology_id, measured, expected, deviation_pct, is_within_spec,
            )

            return record

    def get_performance_history(
        self,
        technology_id: str,
    ) -> List[AbatementPerformance]:
        """Get performance measurement history for a technology.

        Args:
            technology_id: Technology identifier.

        Returns:
            List of AbatementPerformance records sorted by date.

        Raises:
            ValueError: If technology_id not found.
        """
        with self._lock:
            self._get_technology(technology_id)  # validate exists
            history = self._performance_history.get(technology_id, [])
            return list(history)

    # ------------------------------------------------------------------
    # Public API: Cost Analysis
    # ------------------------------------------------------------------

    def get_cost_analysis(
        self,
        technology_ids: List[str],
        carbon_price_usd_per_tco2e: float = 50.0,
        discount_rate: float = 0.08,
        analysis_period_years: int = 10,
    ) -> AbatementCostAnalysis:
        """Compute cost analysis for a set of abatement technologies.

        Calculates levelized cost, payback period, and NPV for the
        specified technologies aggregated together.

        Args:
            technology_ids: List of technology IDs to analyse.
            carbon_price_usd_per_tco2e: Assumed carbon price for
                savings calculation (default 50 USD/tCO2e).
            discount_rate: Annual discount rate for NPV (default 8%).
            analysis_period_years: Analysis horizon in years (default 10).

        Returns:
            AbatementCostAnalysis with full financial breakdown.

        Raises:
            ValueError: If technology_ids is empty or any ID not found.
        """
        if not technology_ids:
            raise ValueError("technology_ids must not be empty")

        t_start = time.monotonic()

        carbon_price = _to_decimal(carbon_price_usd_per_tco2e)
        disc_rate = _to_decimal(discount_rate)
        period = max(1, analysis_period_years)

        total_capex = _ZERO
        total_opex = _ZERO
        total_abated = _ZERO

        with self._lock:
            for tid in technology_ids:
                tech = self._get_technology(tid)
                total_capex += tech.capex_usd
                total_opex += tech.annual_opex_usd
                total_abated += tech.annual_abated_tco2e

        # Levelized cost per tCO2e
        levelized = _ZERO
        if total_abated > _ZERO:
            annualized_capex = total_capex / _to_decimal(period)
            levelized = (
                (annualized_capex + total_opex) / total_abated
            ).quantize(_PRECISION_2, rounding=ROUND_HALF_UP)

        # Annual savings from carbon price
        annual_savings = (total_abated * carbon_price - total_opex)

        # Simple payback
        payback = _ZERO
        if annual_savings > _ZERO:
            payback = (total_capex / annual_savings).quantize(
                _PRECISION_2, rounding=ROUND_HALF_UP
            )

        # NPV calculation
        npv = -total_capex
        for year in range(1, period + 1):
            discount_factor = _ONE / (_ONE + disc_rate) ** year
            annual_cash = annual_savings
            npv += (annual_cash * _to_decimal(discount_factor)).quantize(
                _PRECISION_2, rounding=ROUND_HALF_UP
            )

        npv = npv.quantize(_PRECISION_2, rounding=ROUND_HALF_UP)

        provenance_data = {
            "technology_ids": technology_ids,
            "total_capex": str(total_capex),
            "total_opex": str(total_opex),
            "total_abated": str(total_abated),
            "carbon_price": str(carbon_price),
            "npv": str(npv),
        }
        provenance_hash = self._compute_hash("cost_analysis", provenance_data)

        elapsed_ms = (time.monotonic() - t_start) * 1000

        result = AbatementCostAnalysis(
            analysis_id=f"cost_{uuid4().hex[:12]}",
            technology_ids=list(technology_ids),
            total_capex_usd=total_capex,
            total_annual_opex_usd=total_opex,
            total_annual_abated_tco2e=total_abated,
            levelized_cost_per_tco2e=levelized,
            payback_years=payback,
            npv_usd=npv,
            discount_rate=disc_rate,
            analysis_period_years=period,
            carbon_price_usd_per_tco2e=carbon_price,
            provenance_hash=provenance_hash,
            timestamp=_utcnow().isoformat(),
        )

        logger.info(
            "Cost analysis for %d technologies: levelized=%.2f USD/tCO2e, "
            "payback=%.1f yr, NPV=%.0f USD (%.1fms)",
            len(technology_ids), levelized, payback, npv, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Regulatory Minimum Check
    # ------------------------------------------------------------------

    def check_regulatory_minimum(
        self,
        technology_id: str,
        framework: str,
    ) -> Dict[str, Any]:
        """Check if a technology meets regulatory minimum requirements.

        Args:
            technology_id: Technology identifier.
            framework: Regulatory framework (e.g. EU_ETS_MRR, EPA_PART_98).

        Returns:
            Dictionary with pass/fail status, required minimum, current
            efficiency, gap, and recommendations.

        Raises:
            ValueError: If technology_id not found or framework unknown.
        """
        framework_key = framework.upper().strip()
        if framework_key not in _REGULATORY_MINIMUMS:
            raise ValueError(
                f"Unknown framework '{framework_key}'. Valid: "
                f"{sorted(_REGULATORY_MINIMUMS.keys())}"
            )

        with self._lock:
            tech = self._get_technology(technology_id)
            current_eff = self.get_abatement_efficiency(technology_id)

            minimums = _REGULATORY_MINIMUMS[framework_key]
            process_mins = minimums.get(tech.process_type, {})
            required_min = process_mins.get(tech.target_gas)

            if required_min is None:
                return {
                    "technology_id": technology_id,
                    "framework": framework_key,
                    "status": "NOT_APPLICABLE",
                    "message": (
                        f"No minimum efficiency requirement for "
                        f"{tech.process_type}/{tech.target_gas} under {framework_key}"
                    ),
                    "current_efficiency": str(current_eff),
                    "required_minimum": None,
                    "gap": None,
                    "recommendations": [],
                }

            is_compliant = current_eff >= required_min
            gap = (required_min - current_eff) if not is_compliant else _ZERO

            recommendations: List[str] = []
            if not is_compliant:
                recommendations.append(
                    f"Current efficiency ({current_eff}) is below the {framework_key} "
                    f"minimum ({required_min}) by {gap.quantize(_PRECISION_4)}. "
                    f"Consider upgrading or replacing the abatement technology."
                )
                if tech.maintenance_status != MaintenanceStatus.OPERATIONAL.value:
                    recommendations.append(
                        f"Technology is in {tech.maintenance_status} status. "
                        f"Restoring to operational may improve efficiency."
                    )
                if tech.age_years > Decimal("5"):
                    recommendations.append(
                        f"Technology is {tech.age_years} years old. "
                        f"Consider catalyst replacement or system overhaul."
                    )

            status = "COMPLIANT" if is_compliant else "NON_COMPLIANT"

            logger.info(
                "Regulatory check: %s under %s -> %s (current=%.4f, min=%.4f)",
                technology_id, framework_key, status, current_eff, required_min,
            )

            return {
                "technology_id": technology_id,
                "framework": framework_key,
                "process_type": tech.process_type,
                "target_gas": tech.target_gas,
                "status": status,
                "current_efficiency": str(current_eff),
                "required_minimum": str(required_min),
                "gap": str(gap.quantize(_PRECISION_4, rounding=ROUND_HALF_UP)),
                "recommendations": recommendations,
                "provenance_hash": self._compute_hash(
                    "regulatory_check",
                    {
                        "technology_id": technology_id,
                        "framework": framework_key,
                        "status": status,
                    },
                ),
            }

    # ------------------------------------------------------------------
    # Public API: Monitoring Requirements
    # ------------------------------------------------------------------

    def get_monitoring_requirements(
        self,
        process_type: str,
        framework: str,
    ) -> Dict[str, Any]:
        """Get monitoring frequency requirements for a process/framework.

        Args:
            process_type: Industrial process type.
            framework: Regulatory framework.

        Returns:
            Dictionary with required monitoring frequency, description,
            and recommendations.

        Raises:
            ValueError: If framework is not recognized.
        """
        framework_key = framework.upper().strip()
        process_key = process_type.upper().strip()

        if framework_key not in _MONITORING_REQUIREMENTS:
            raise ValueError(
                f"Unknown framework '{framework_key}'. Valid: "
                f"{sorted(_MONITORING_REQUIREMENTS.keys())}"
            )

        fw_reqs = _MONITORING_REQUIREMENTS[framework_key]
        required_freq = fw_reqs.get(process_key, fw_reqs.get("DEFAULT", "ANNUAL"))

        # Build descriptions
        freq_descriptions = {
            MonitoringFrequency.CONTINUOUS.value: (
                "Continuous Emissions Monitoring System (CEMS) required. "
                "Real-time data acquisition and automated reporting."
            ),
            MonitoringFrequency.DAILY.value: (
                "Daily spot checks or automated sampling required."
            ),
            MonitoringFrequency.WEEKLY.value: (
                "Weekly performance testing and equipment inspection."
            ),
            MonitoringFrequency.MONTHLY.value: (
                "Monthly stack testing or performance verification."
            ),
            MonitoringFrequency.QUARTERLY.value: (
                "Quarterly performance testing or compliance audit."
            ),
            MonitoringFrequency.ANNUAL.value: (
                "Annual performance verification and calibration."
            ),
        }

        return {
            "process_type": process_key,
            "framework": framework_key,
            "required_frequency": required_freq,
            "description": freq_descriptions.get(required_freq, ""),
            "recommendations": [
                f"Ensure monitoring equipment is calibrated per {framework_key} requirements.",
                f"Maintain records for at least 5 years (EPA) or 10 years (EU ETS).",
                "Document any monitoring gaps and apply approved missing data procedures.",
            ],
        }

    # ------------------------------------------------------------------
    # Public API: Technology Management
    # ------------------------------------------------------------------

    def get_technology(self, technology_id: str) -> Dict[str, Any]:
        """Retrieve a registered technology record.

        Args:
            technology_id: Technology identifier.

        Returns:
            Technology record as dictionary.

        Raises:
            ValueError: If technology_id not found.
        """
        with self._lock:
            tech = self._get_technology(technology_id)
            return tech.to_dict()

    def list_technologies(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List registered technologies with optional filters.

        Args:
            filters: Optional filter dictionary. Supported keys:
                - process_type (str): Filter by process type.
                - target_gas (str): Filter by target gas.
                - technology_type (str): Filter by technology type.
                - maintenance_status (str): Filter by maintenance status.
                - verification_status (str): Filter by verification status.
                - min_efficiency (float): Minimum current efficiency.

        Returns:
            List of matching technology dictionaries.
        """
        filters = filters or {}
        with self._lock:
            results: List[Dict[str, Any]] = []

            f_process = filters.get("process_type", "").upper().strip() or None
            f_gas = filters.get("target_gas", "").upper().strip() or None
            f_type = filters.get("technology_type", "").upper().strip() or None
            f_maint = filters.get("maintenance_status", "").upper().strip() or None
            f_ver = filters.get("verification_status", "").upper().strip() or None
            f_min_eff = filters.get("min_efficiency")

            for tech in self._technologies.values():
                if f_process and tech.process_type != f_process:
                    continue
                if f_gas and tech.target_gas != f_gas:
                    continue
                if f_type and tech.technology_type != f_type:
                    continue
                if f_maint and tech.maintenance_status != f_maint:
                    continue
                if f_ver and tech.verification_status != f_ver:
                    continue
                if f_min_eff is not None:
                    if tech.current_efficiency < _to_decimal(f_min_eff):
                        continue
                results.append(tech.to_dict())

            return results

    def update_maintenance_status(
        self,
        technology_id: str,
        new_status: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Update the maintenance status of a technology.

        Args:
            technology_id: Technology identifier.
            new_status: New MaintenanceStatus value.
            notes: Optional notes about the status change.

        Returns:
            Updated technology record.

        Raises:
            ValueError: If technology_id not found or status invalid.
        """
        status_upper = new_status.upper().strip()
        valid_statuses = {s.value for s in MaintenanceStatus}
        if status_upper not in valid_statuses:
            raise ValueError(
                f"Invalid maintenance status '{status_upper}'. "
                f"Valid: {sorted(valid_statuses)}"
            )

        with self._lock:
            tech = self._get_technology(technology_id)
            old_status = tech.maintenance_status
            tech.maintenance_status = status_upper
            tech.updated_at = _utcnow().isoformat()

            # Recalculate current efficiency with new status
            age = self._calculate_age_years(tech.installation_date)
            base = self._age_adjusted_efficiency(
                tech.base_efficiency, tech.degradation_rate, age
            )
            tech.current_efficiency = self._maintenance_adjusted_efficiency(
                base, status_upper
            )

            logger.info(
                "Maintenance status updated: %s %s -> %s (notes: %s)",
                technology_id, old_status, status_upper, notes,
            )

            return tech.to_dict()

    def update_verification_status(
        self,
        technology_id: str,
        new_status: str,
        expiry_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update the verification status of a technology.

        Args:
            technology_id: Technology identifier.
            new_status: New VerificationStatus value.
            expiry_date: Optional ISO date when verification expires.

        Returns:
            Updated technology record.

        Raises:
            ValueError: If technology_id not found or status invalid.
        """
        status_upper = new_status.upper().strip()
        valid_statuses = {s.value for s in VerificationStatus}
        if status_upper not in valid_statuses:
            raise ValueError(
                f"Invalid verification status '{status_upper}'. "
                f"Valid: {sorted(valid_statuses)}"
            )

        with self._lock:
            tech = self._get_technology(technology_id)
            tech.verification_status = status_upper
            tech.verification_expiry = expiry_date
            tech.updated_at = _utcnow().isoformat()

            logger.info(
                "Verification status updated: %s -> %s (expiry: %s)",
                technology_id, status_upper, expiry_date,
            )

            return tech.to_dict()

    # ------------------------------------------------------------------
    # Public API: Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics for the abatement tracker.

        Returns:
            Dictionary with counts, efficiency summaries, and status
            breakdown.
        """
        with self._lock:
            total = len(self._technologies)
            if total == 0:
                return {
                    "total_technologies": 0,
                    "by_type": {},
                    "by_process": {},
                    "by_maintenance_status": {},
                    "by_verification_status": {},
                    "avg_current_efficiency": "0",
                    "total_performance_records": 0,
                }

            by_type: Dict[str, int] = {}
            by_process: Dict[str, int] = {}
            by_maint: Dict[str, int] = {}
            by_ver: Dict[str, int] = {}
            eff_sum = _ZERO

            for tech in self._technologies.values():
                by_type[tech.technology_type] = by_type.get(tech.technology_type, 0) + 1
                by_process[tech.process_type] = by_process.get(tech.process_type, 0) + 1
                by_maint[tech.maintenance_status] = by_maint.get(tech.maintenance_status, 0) + 1
                by_ver[tech.verification_status] = by_ver.get(tech.verification_status, 0) + 1
                eff_sum += tech.current_efficiency

            avg_eff = (eff_sum / _to_decimal(total)).quantize(
                _PRECISION_4, rounding=ROUND_HALF_UP
            )

            total_perf = sum(
                len(h) for h in self._performance_history.values()
            )

            return {
                "total_technologies": total,
                "by_type": by_type,
                "by_process": by_process,
                "by_maintenance_status": by_maint,
                "by_verification_status": by_ver,
                "avg_current_efficiency": str(avg_eff),
                "total_performance_records": total_perf,
            }

    # ==================================================================
    # Internal: Efficiency Calculations
    # ==================================================================

    def _age_adjusted_efficiency(
        self,
        base_efficiency: Decimal,
        degradation_rate: Decimal,
        age_years: Decimal,
    ) -> Decimal:
        """Calculate age-adjusted efficiency.

        Formula: eff = base * (1 - degradation_rate * age_years)
        Clamped to [0.01 * base, base] to prevent negative values
        while allowing significant degradation.

        Args:
            base_efficiency: Nameplate efficiency.
            degradation_rate: Annual degradation as fraction.
            age_years: Years since installation.

        Returns:
            Age-adjusted efficiency as Decimal in [0, 1].
        """
        adjustment_factor = _ONE - (degradation_rate * age_years)
        minimum_factor = Decimal("0.01")  # At least 1% of base
        adjustment_factor = max(minimum_factor, min(_ONE, adjustment_factor))
        result = base_efficiency * adjustment_factor
        return _decimal_clamp(result, _ZERO, _ONE)

    def _maintenance_adjusted_efficiency(
        self,
        base_efficiency: Decimal,
        maintenance_status: str,
    ) -> Decimal:
        """Apply maintenance status adjustment to efficiency.

        Args:
            base_efficiency: Age-adjusted efficiency before maintenance factor.
            maintenance_status: Current maintenance status.

        Returns:
            Maintenance-adjusted efficiency.
        """
        # Maintenance multipliers
        multipliers = {
            MaintenanceStatus.OPERATIONAL.value: _ONE,
            MaintenanceStatus.DEGRADED.value: Decimal("0.90"),
            MaintenanceStatus.MAINTENANCE_DUE.value: Decimal("0.95"),
            MaintenanceStatus.UNDER_MAINTENANCE.value: _ZERO,
            MaintenanceStatus.FAILED.value: _ZERO,
            MaintenanceStatus.DECOMMISSIONED.value: _ZERO,
        }

        multiplier = multipliers.get(maintenance_status, _ONE)
        return _decimal_clamp(base_efficiency * multiplier, _ZERO, _ONE)

    def _calculate_age_years(
        self,
        installation_date_str: str,
        as_of_date_str: Optional[str] = None,
    ) -> Decimal:
        """Calculate age in years from installation date.

        Args:
            installation_date_str: ISO date string of installation.
            as_of_date_str: Optional ISO date string for the reference
                date. Defaults to today.

        Returns:
            Age in years as Decimal, minimum 0.
        """
        try:
            install = date.fromisoformat(installation_date_str)
        except (ValueError, TypeError):
            return _ZERO

        if as_of_date_str:
            try:
                ref = date.fromisoformat(as_of_date_str)
            except (ValueError, TypeError):
                ref = _utcnow().date()
        else:
            ref = _utcnow().date()

        delta_days = (ref - install).days
        if delta_days < 0:
            return _ZERO

        age = Decimal(str(delta_days)) / Decimal("365.25")
        return age.quantize(_PRECISION_2, rounding=ROUND_HALF_UP)

    # ==================================================================
    # Internal: Lookup
    # ==================================================================

    def _get_technology(self, technology_id: str) -> AbatementTechnology:
        """Get a technology record or raise ValueError.

        Args:
            technology_id: Technology identifier.

        Returns:
            AbatementTechnology instance.

        Raises:
            ValueError: If not found.
        """
        tech = self._technologies.get(technology_id)
        if tech is None:
            raise ValueError(f"Technology not found: '{technology_id}'")
        return tech

    # ==================================================================
    # Internal: Provenance and Metrics
    # ==================================================================

    def _compute_hash(self, operation: str, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            operation: Operation name.
            data: Data dictionary.

        Returns:
            Hexadecimal SHA-256 hash.
        """
        hash_input = json.dumps(
            {"operation": operation, "data": data, "timestamp": _utcnow().isoformat()},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def _record_metrics(self, action: str, technology_type: str) -> None:
        """Record Prometheus metrics if available.

        Args:
            action: Action label.
            technology_type: Technology type label.
        """
        if _METRICS_AVAILABLE and _record_material_operation is not None:
            try:
                _record_material_operation(action, technology_type)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
