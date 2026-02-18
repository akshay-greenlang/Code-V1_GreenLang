# -*- coding: utf-8 -*-
"""
FugitiveSourceDatabaseEngine - Source Types, Emission Factors, Gas Compositions (Engine 1 of 7)

AGENT-MRV-005: Fugitive Emissions Agent

Provides the authoritative reference data repository for all fugitive emission
source types, EPA component-level emission factors, screening range factors,
correlation equation coefficients, coal mine methane factors, wastewater
treatment factors, pneumatic device emission rates, natural gas compositions,
and Global Warming Potential (GWP) values across IPCC assessment reports.

This engine is the single source of truth for numeric constants used by
the EmissionCalculatorEngine (Engine 2). By centralizing all emission
factors and reference data in one module, we guarantee that every
calculation in the pipeline uses identical, auditable, peer-reviewed values.

Built-In Reference Data:
    - 25 fugitive emission source types with full metadata
    - EPA component-level average emission factors (11 component/service combos)
    - Screening range leak/no-leak emission factors per EPA Protocol
    - Correlation equation coefficients (log(kg/hr) = a + b*log(ppmv))
    - Coal mine methane emission factors by coal rank (4 ranks)
    - Wastewater treatment factors (Bo, MCF by treatment type)
    - Pneumatic device emission rates (high-bleed, low-bleed, intermittent)
    - Default natural gas composition (CH4, C2H6, CO2, N2)
    - GWP values for CH4, CO2, N2O across AR4/AR5/AR6/AR6_20YR

Zero-Hallucination Guarantees:
    - All emission factors are hard-coded from published regulatory sources.
    - All lookups are deterministic dictionary access.
    - No LLM involvement in any data retrieval path.
    - Every query result carries a SHA-256 provenance hash.

Thread Safety:
    All reference data is immutable after initialization. The mutable
    custom factor registry is protected by a reentrant lock.

Example:
    >>> from greenlang.fugitive_emissions.fugitive_source_database import (
    ...     FugitiveSourceDatabaseEngine,
    ... )
    >>> db = FugitiveSourceDatabaseEngine()
    >>> info = db.get_source_info("EQUIPMENT_LEAK")
    >>> ef = db.get_component_ef("valve", "gas")
    >>> gwp = db.get_gwp("CH4", "AR6")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["FugitiveSourceDatabaseEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.fugitive_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.fugitive_emissions.metrics import (
        record_component_operation as _record_db_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_db_operation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


# ===========================================================================
# Enumerations
# ===========================================================================


class SourceCategory(str, Enum):
    """Top-level fugitive emission source categories.

    EQUIPMENT_LEAK: Leaks from valves, pumps, compressors, connectors, etc.
    COAL_MINE: Methane emissions from coal mining operations.
    WASTEWATER: Emissions from wastewater treatment processes.
    PNEUMATIC_DEVICE: Natural gas emissions from pneumatic controllers.
    TANK_STORAGE: Evaporative losses from storage tanks.
    DIRECT_MEASUREMENT: Source quantified via direct measurement.
    """

    EQUIPMENT_LEAK = "EQUIPMENT_LEAK"
    COAL_MINE = "COAL_MINE"
    WASTEWATER = "WASTEWATER"
    PNEUMATIC_DEVICE = "PNEUMATIC_DEVICE"
    TANK_STORAGE = "TANK_STORAGE"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class CalculationMethod(str, Enum):
    """Fugitive emission calculation methods.

    AVERAGE_EMISSION_FACTOR: EPA average component emission factor method.
    SCREENING_RANGES: EPA screening value ranges method (leak/no-leak).
    CORRELATION_EQUATION: EPA correlation equation method (ppmv -> kg/hr).
    ENGINEERING_ESTIMATE: Engineering calculations for specific equipment.
    DIRECT_MEASUREMENT: Hi-Flow sampler, bagging, or calibrated instrument.
    """

    AVERAGE_EMISSION_FACTOR = "AVERAGE_EMISSION_FACTOR"
    SCREENING_RANGES = "SCREENING_RANGES"
    CORRELATION_EQUATION = "CORRELATION_EQUATION"
    ENGINEERING_ESTIMATE = "ENGINEERING_ESTIMATE"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"


class GWPSource(str, Enum):
    """IPCC Assessment Report editions for GWP values.

    AR4: Fourth Assessment Report (2007).
    AR5: Fifth Assessment Report (2014).
    AR6: Sixth Assessment Report (2021), 100-year horizon.
    AR6_20YR: Sixth Assessment Report (2021), 20-year horizon.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class CoalRank(str, Enum):
    """Coal rank classifications for methane emission factors.

    ANTHRACITE: High-rank coal with low volatile matter.
    BITUMINOUS: Medium-rank coal, most common for energy.
    SUBBITUMINOUS: Lower-rank coal with higher moisture.
    LIGNITE: Lowest-rank coal (brown coal).
    """

    ANTHRACITE = "ANTHRACITE"
    BITUMINOUS = "BITUMINOUS"
    SUBBITUMINOUS = "SUBBITUMINOUS"
    LIGNITE = "LIGNITE"


class WastewaterTreatmentType(str, Enum):
    """Wastewater treatment system types for MCF selection.

    UNTREATED_DISCHARGE: No treatment, direct discharge.
    AEROBIC_WELL_MANAGED: Well-managed aerobic treatment (activated sludge).
    AEROBIC_POORLY_MANAGED: Poorly managed aerobic treatment.
    ANAEROBIC_REACTOR: Controlled anaerobic reactor (covered digester).
    ANAEROBIC_LAGOON_DEEP: Deep anaerobic lagoon (> 2m).
    ANAEROBIC_LAGOON_SHALLOW: Shallow anaerobic lagoon (< 2m).
    FACULTATIVE_LAGOON: Facultative (mixed aerobic/anaerobic) lagoon.
    SEPTIC_SYSTEM: Septic tank system.
    LATRINE_DRY: Dry climate latrine.
    LATRINE_WET: Wet climate or groundwater-connected latrine.
    """

    UNTREATED_DISCHARGE = "UNTREATED_DISCHARGE"
    AEROBIC_WELL_MANAGED = "AEROBIC_WELL_MANAGED"
    AEROBIC_POORLY_MANAGED = "AEROBIC_POORLY_MANAGED"
    ANAEROBIC_REACTOR = "ANAEROBIC_REACTOR"
    ANAEROBIC_LAGOON_DEEP = "ANAEROBIC_LAGOON_DEEP"
    ANAEROBIC_LAGOON_SHALLOW = "ANAEROBIC_LAGOON_SHALLOW"
    FACULTATIVE_LAGOON = "FACULTATIVE_LAGOON"
    SEPTIC_SYSTEM = "SEPTIC_SYSTEM"
    LATRINE_DRY = "LATRINE_DRY"
    LATRINE_WET = "LATRINE_WET"


# ===========================================================================
# Reference Data: 25 Source Types
# ===========================================================================

#: Complete metadata for all 25 fugitive emission source types.
#: Each entry includes description, applicable calculation methods,
#: primary gas species, regulatory references, and data requirements.
SOURCE_TYPES: Dict[str, Dict[str, Any]] = {
    "EQUIPMENT_LEAK_VALVE_GAS": {
        "name": "Equipment Leak - Valve (Gas Service)",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from valves in gas or vapor service, "
                       "including gate, globe, ball, butterfly, plug, and needle valves.",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "CORRELATION_EQUATION", "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017", "40 CFR 60 Subpart VVa"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_VALVE_LL": {
        "name": "Equipment Leak - Valve (Light Liquid Service)",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from valves in light liquid service "
                       "(vapor pressure > 0.3 kPa at 20C).",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "CORRELATION_EQUATION", "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017", "40 CFR 60 Subpart VVa"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_VALVE_HL": {
        "name": "Equipment Leak - Valve (Heavy Liquid Service)",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from valves in heavy liquid service "
                       "(vapor pressure <= 0.3 kPa at 20C).",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_PUMP_LL": {
        "name": "Equipment Leak - Pump Seal (Light Liquid Service)",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from pump seals in light liquid service.",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "CORRELATION_EQUATION", "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017", "40 CFR 60 Subpart VVa"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_PUMP_HL": {
        "name": "Equipment Leak - Pump Seal (Heavy Liquid Service)",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from pump seals in heavy liquid service.",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_COMPRESSOR": {
        "name": "Equipment Leak - Compressor Seal",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from compressor shaft seals "
                       "(reciprocating, rotary, centrifugal).",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "CORRELATION_EQUATION", "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017", "40 CFR 60 Subpart KKK"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_PRD": {
        "name": "Equipment Leak - Pressure Relief Device",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from pressure relief valves and "
                       "rupture discs in gas service.",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_CONNECTOR": {
        "name": "Equipment Leak - Connector",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from flanged, threaded, and other "
                       "pipe connections.",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "CORRELATION_EQUATION", "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017", "40 CFR 60 Subpart VVa"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_OEL": {
        "name": "Equipment Leak - Open-Ended Line",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from open-ended lines, sampling "
                       "connections, and drain lines.",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_SAMPLING": {
        "name": "Equipment Leak - Sampling Connection",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from sampling connections and "
                       "analytical instrument connections.",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "EQUIPMENT_LEAK_FLANGE": {
        "name": "Equipment Leak - Flange",
        "category": "EQUIPMENT_LEAK",
        "description": "Fugitive emissions from flanged connections "
                       "(distinguished from connectors in some protocols).",
        "primary_gas": "CH4",
        "applicable_methods": [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "CORRELATION_EQUATION", "DIRECT_MEASUREMENT",
        ],
        "regulatory_refs": ["EPA 453/R-95-017"],
        "data_requirements": ["component_count", "operating_hours", "gas_fraction"],
    },
    "COAL_MINE_UNDERGROUND": {
        "name": "Coal Mine Methane - Underground Mining",
        "category": "COAL_MINE",
        "description": "Methane released during underground coal mining from "
                       "ventilation air, degasification systems, and mine face.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"],
        "regulatory_refs": ["IPCC 2006 GL Vol.2 Ch.4", "EPA Subpart FF"],
        "data_requirements": ["coal_production_tonnes", "coal_rank", "recovery_fraction"],
    },
    "COAL_MINE_SURFACE": {
        "name": "Coal Mine Methane - Surface Mining",
        "category": "COAL_MINE",
        "description": "Methane released during surface (open-cut) coal mining "
                       "from exposed coal seams.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"],
        "regulatory_refs": ["IPCC 2006 GL Vol.2 Ch.4"],
        "data_requirements": ["coal_production_tonnes", "coal_rank", "recovery_fraction"],
    },
    "COAL_MINE_POST_MINING": {
        "name": "Coal Mine Methane - Post-Mining Activities",
        "category": "COAL_MINE",
        "description": "Methane released during coal handling, processing, "
                       "and transport after extraction.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE"],
        "regulatory_refs": ["IPCC 2006 GL Vol.2 Ch.4"],
        "data_requirements": ["coal_production_tonnes", "coal_rank"],
    },
    "WASTEWATER_INDUSTRIAL": {
        "name": "Wastewater Treatment - Industrial",
        "category": "WASTEWATER",
        "description": "CH4 and N2O emissions from industrial wastewater "
                       "treatment processes.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"],
        "regulatory_refs": ["IPCC 2006 GL Vol.5 Ch.6", "EPA Subpart II"],
        "data_requirements": [
            "bod_load_kg", "treatment_type", "recovery_fraction",
            "nitrogen_load_kg",
        ],
    },
    "WASTEWATER_MUNICIPAL": {
        "name": "Wastewater Treatment - Municipal",
        "category": "WASTEWATER",
        "description": "CH4 and N2O emissions from municipal wastewater "
                       "treatment plants.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"],
        "regulatory_refs": ["IPCC 2006 GL Vol.5 Ch.6", "EPA Subpart HH"],
        "data_requirements": [
            "bod_load_kg", "treatment_type", "recovery_fraction",
            "nitrogen_load_kg",
        ],
    },
    "WASTEWATER_SLUDGE": {
        "name": "Wastewater Treatment - Sludge Handling",
        "category": "WASTEWATER",
        "description": "CH4 emissions from anaerobic decomposition of "
                       "wastewater sludge during handling and disposal.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE"],
        "regulatory_refs": ["IPCC 2006 GL Vol.5 Ch.6"],
        "data_requirements": ["sludge_mass_kg", "treatment_type"],
    },
    "PNEUMATIC_HIGH_BLEED": {
        "name": "Pneumatic Device - High-Bleed Controller",
        "category": "PNEUMATIC_DEVICE",
        "description": "Continuous high-bleed natural gas pneumatic controllers "
                       "with emission rate > 6 scfh.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"],
        "regulatory_refs": ["EPA Subpart W Table W-1A", "40 CFR 60 Subpart OOOOa"],
        "data_requirements": ["device_count", "operating_hours"],
    },
    "PNEUMATIC_LOW_BLEED": {
        "name": "Pneumatic Device - Low-Bleed Controller",
        "category": "PNEUMATIC_DEVICE",
        "description": "Continuous low-bleed natural gas pneumatic controllers "
                       "with emission rate <= 6 scfh.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"],
        "regulatory_refs": ["EPA Subpart W Table W-1A", "40 CFR 60 Subpart OOOOa"],
        "data_requirements": ["device_count", "operating_hours"],
    },
    "PNEUMATIC_INTERMITTENT": {
        "name": "Pneumatic Device - Intermittent-Vent Controller",
        "category": "PNEUMATIC_DEVICE",
        "description": "Intermittent-vent pneumatic controllers that vent "
                       "natural gas only on actuation.",
        "primary_gas": "CH4",
        "applicable_methods": ["ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"],
        "regulatory_refs": ["EPA Subpart W Table W-1A"],
        "data_requirements": ["device_count", "operating_hours", "actuation_rate"],
    },
    "TANK_FIXED_ROOF": {
        "name": "Tank Storage Loss - Fixed Roof",
        "category": "TANK_STORAGE",
        "description": "Breathing and working losses from vertical and "
                       "horizontal fixed-roof storage tanks per AP-42 Ch 7.",
        "primary_gas": "VOC",
        "applicable_methods": ["ENGINEERING_ESTIMATE"],
        "regulatory_refs": ["EPA AP-42 Chapter 7 Section 7.1"],
        "data_requirements": [
            "tank_diameter_ft", "tank_height_ft", "vapor_pressure_psia",
            "molecular_weight", "annual_throughput_gal",
        ],
    },
    "TANK_FLOATING_ROOF": {
        "name": "Tank Storage Loss - Floating Roof",
        "category": "TANK_STORAGE",
        "description": "Rim seal, deck fitting, and deck seam losses from "
                       "external and internal floating-roof tanks per AP-42 Ch 7.",
        "primary_gas": "VOC",
        "applicable_methods": ["ENGINEERING_ESTIMATE"],
        "regulatory_refs": ["EPA AP-42 Chapter 7 Section 7.1"],
        "data_requirements": [
            "tank_diameter_ft", "rim_seal_type", "fitting_counts",
            "vapor_pressure_psia", "molecular_weight",
        ],
    },
    "TANK_PRESSURIZED": {
        "name": "Tank Storage Loss - Pressurized",
        "category": "TANK_STORAGE",
        "description": "Pressurized storage vessels (spheres, bullets) with "
                       "negligible evaporative losses under normal operation.",
        "primary_gas": "VOC",
        "applicable_methods": ["ENGINEERING_ESTIMATE"],
        "regulatory_refs": ["EPA AP-42 Chapter 7"],
        "data_requirements": ["tank_type"],
    },
    "DIRECT_MEASUREMENT_HIFLOW": {
        "name": "Direct Measurement - Hi-Flow Sampler",
        "category": "DIRECT_MEASUREMENT",
        "description": "Fugitive emissions quantified using a calibrated "
                       "Hi-Flow sampler instrument.",
        "primary_gas": "CH4",
        "applicable_methods": ["DIRECT_MEASUREMENT"],
        "regulatory_refs": ["EPA OTM-33A"],
        "data_requirements": ["measured_rate_kg_hr", "measurement_duration_hr"],
    },
    "DIRECT_MEASUREMENT_BAGGING": {
        "name": "Direct Measurement - Bagging",
        "category": "DIRECT_MEASUREMENT",
        "description": "Fugitive emissions quantified using enclosure (bagging) "
                       "technique for individual components.",
        "primary_gas": "CH4",
        "applicable_methods": ["DIRECT_MEASUREMENT"],
        "regulatory_refs": ["EPA Method 21"],
        "data_requirements": ["measured_rate_kg_hr", "measurement_duration_hr"],
    },
}


# ===========================================================================
# Reference Data: EPA Component Emission Factors
# ===========================================================================

#: EPA average emission factors by component type and service type.
#: Source: EPA Protocol for Equipment Leak Emission Estimates (EPA-453/R-95-017).
#: Units: kg total organic compound (TOC) per hour per component.
#:
#: Key format: (component_type, service_type) -> Decimal(kg/hr)
#:
#: Valve-Gas:       0.00597 kg/hr
#: Valve-LL:        0.00403 kg/hr
#: Valve-HL:        0.00023 kg/hr
#: Pump-LL:         0.01140 kg/hr
#: Pump-HL:         0.00862 kg/hr
#: Compressor-Gas:  0.22800 kg/hr
#: PRD-Gas:         0.10400 kg/hr
#: Connector-All:   0.00183 kg/hr
#: OEL-All:         0.00170 kg/hr
#: Sampling-All:    0.01500 kg/hr
#: Flange-All:      0.00083 kg/hr
COMPONENT_EMISSION_FACTORS: Dict[Tuple[str, str], Decimal] = {
    ("valve", "gas"):           Decimal("0.00597"),
    ("valve", "light_liquid"):  Decimal("0.00403"),
    ("valve", "heavy_liquid"):  Decimal("0.00023"),
    ("pump", "light_liquid"):   Decimal("0.01140"),
    ("pump", "heavy_liquid"):   Decimal("0.00862"),
    ("compressor", "gas"):      Decimal("0.22800"),
    ("pressure_relief", "gas"): Decimal("0.10400"),
    ("connector", "gas"):       Decimal("0.00183"),
    ("connector", "light_liquid"):  Decimal("0.00183"),
    ("connector", "heavy_liquid"):  Decimal("0.00183"),
    ("open_ended_line", "gas"):     Decimal("0.00170"),
    ("open_ended_line", "light_liquid"): Decimal("0.00170"),
    ("sampling", "gas"):        Decimal("0.01500"),
    ("sampling", "light_liquid"): Decimal("0.01500"),
    ("flange", "gas"):          Decimal("0.00083"),
    ("flange", "light_liquid"): Decimal("0.00083"),
    ("flange", "heavy_liquid"): Decimal("0.00083"),
}


# ===========================================================================
# Reference Data: Screening Range Emission Factors
# ===========================================================================

#: EPA screening range emission factors for the leak/no-leak method.
#: Source: EPA-453/R-95-017 Table 2-4 and 2-9.
#: Units: kg TOC per hour per component.
#:
#: For each (component, service) pair, provides:
#:   - leak_ef: Factor applied when screening value >= threshold (default 10000 ppmv)
#:   - no_leak_ef: Factor applied when screening value < threshold
#:
#: The threshold is configurable but defaults to 10,000 ppmv per EPA guidance.
SCREENING_RANGE_FACTORS: Dict[Tuple[str, str], Dict[str, Decimal]] = {
    ("valve", "gas"): {
        "leak_ef": Decimal("0.02680"),
        "no_leak_ef": Decimal("0.00006"),
    },
    ("valve", "light_liquid"): {
        "leak_ef": Decimal("0.01090"),
        "no_leak_ef": Decimal("0.00014"),
    },
    ("valve", "heavy_liquid"): {
        "leak_ef": Decimal("0.00230"),
        "no_leak_ef": Decimal("0.00003"),
    },
    ("pump", "light_liquid"): {
        "leak_ef": Decimal("0.11400"),
        "no_leak_ef": Decimal("0.00210"),
    },
    ("pump", "heavy_liquid"): {
        "leak_ef": Decimal("0.07200"),
        "no_leak_ef": Decimal("0.00050"),
    },
    ("compressor", "gas"): {
        "leak_ef": Decimal("0.50700"),
        "no_leak_ef": Decimal("0.02570"),
    },
    ("pressure_relief", "gas"): {
        "leak_ef": Decimal("1.69100"),
        "no_leak_ef": Decimal("0.01950"),
    },
    ("connector", "gas"): {
        "leak_ef": Decimal("0.01130"),
        "no_leak_ef": Decimal("0.00020"),
    },
    ("connector", "light_liquid"): {
        "leak_ef": Decimal("0.01130"),
        "no_leak_ef": Decimal("0.00020"),
    },
    ("connector", "heavy_liquid"): {
        "leak_ef": Decimal("0.01130"),
        "no_leak_ef": Decimal("0.00020"),
    },
    ("open_ended_line", "gas"): {
        "leak_ef": Decimal("0.01700"),
        "no_leak_ef": Decimal("0.00020"),
    },
    ("open_ended_line", "light_liquid"): {
        "leak_ef": Decimal("0.01700"),
        "no_leak_ef": Decimal("0.00020"),
    },
    ("sampling", "gas"): {
        "leak_ef": Decimal("0.15000"),
        "no_leak_ef": Decimal("0.00150"),
    },
    ("flange", "gas"): {
        "leak_ef": Decimal("0.00830"),
        "no_leak_ef": Decimal("0.00005"),
    },
    ("flange", "light_liquid"): {
        "leak_ef": Decimal("0.00830"),
        "no_leak_ef": Decimal("0.00005"),
    },
}


# ===========================================================================
# Reference Data: Correlation Equation Coefficients
# ===========================================================================

#: EPA correlation equation coefficients for leak rate estimation.
#: Source: EPA-453/R-95-017 Table 2-8.
#:
#: Equation: log10(kg/hr) = a + b * log10(ppmv)
#:
#: Each entry provides coefficients (a, b) and the valid screening
#: value range (min_ppmv, max_ppmv) over which the correlation is valid.
#: A default_zero_ef is provided for screening values of 0 ppmv (below
#: the minimum detection limit).
CORRELATION_COEFFICIENTS: Dict[Tuple[str, str], Dict[str, Decimal]] = {
    ("valve", "gas"): {
        "a": Decimal("-6.36040"),
        "b": Decimal("0.79690"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000020"),
    },
    ("valve", "light_liquid"): {
        "a": Decimal("-6.29400"),
        "b": Decimal("0.74000"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000014"),
    },
    ("pump", "light_liquid"): {
        "a": Decimal("-5.03500"),
        "b": Decimal("0.61000"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000210"),
    },
    ("compressor", "gas"): {
        "a": Decimal("-4.43640"),
        "b": Decimal("0.56220"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.002570"),
    },
    ("connector", "gas"): {
        "a": Decimal("-6.78900"),
        "b": Decimal("0.73500"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000010"),
    },
    ("connector", "light_liquid"): {
        "a": Decimal("-6.78900"),
        "b": Decimal("0.73500"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000010"),
    },
    ("flange", "gas"): {
        "a": Decimal("-6.52600"),
        "b": Decimal("0.68000"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000005"),
    },
    ("flange", "light_liquid"): {
        "a": Decimal("-6.52600"),
        "b": Decimal("0.68000"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000005"),
    },
    ("open_ended_line", "gas"): {
        "a": Decimal("-6.25100"),
        "b": Decimal("0.70700"),
        "min_ppmv": Decimal("1"),
        "max_ppmv": Decimal("1000000"),
        "default_zero_ef": Decimal("0.000020"),
    },
}


# ===========================================================================
# Reference Data: Coal Mine Methane Emission Factors
# ===========================================================================

#: Coal mine methane emission factors by coal rank.
#: Source: IPCC 2006 Guidelines Vol. 2 Ch. 4, Table 4.1.4.
#: Units: m3 CH4 per tonne of coal mined (in situ basis).
#:
#: These factors represent the average methane content released per tonne
#: of coal extracted. Actual emissions depend on mining method, depth,
#: and methane recovery systems.
COAL_METHANE_FACTORS: Dict[str, Dict[str, Any]] = {
    "ANTHRACITE": {
        "ef_m3_per_tonne": Decimal("18"),
        "description": "High-rank anthracite coal",
        "uncertainty_pct": Decimal("50"),
        "source": "IPCC 2006 GL Vol.2 Ch.4 Table 4.1.4",
    },
    "BITUMINOUS": {
        "ef_m3_per_tonne": Decimal("10"),
        "description": "Medium-rank bituminous coal",
        "uncertainty_pct": Decimal("50"),
        "source": "IPCC 2006 GL Vol.2 Ch.4 Table 4.1.4",
    },
    "SUBBITUMINOUS": {
        "ef_m3_per_tonne": Decimal("3"),
        "description": "Lower-rank sub-bituminous coal",
        "uncertainty_pct": Decimal("75"),
        "source": "IPCC 2006 GL Vol.2 Ch.4 Table 4.1.4",
    },
    "LIGNITE": {
        "ef_m3_per_tonne": Decimal("1"),
        "description": "Lowest-rank lignite (brown coal)",
        "uncertainty_pct": Decimal("75"),
        "source": "IPCC 2006 GL Vol.2 Ch.4 Table 4.1.4",
    },
}

#: CH4 density at standard conditions (kg/m3) for coal mine calculations.
#: At 0C, 101.325 kPa: 0.7168 kg/m3. At 15C, 101.325 kPa: 0.6785 kg/m3.
#: We use 20C reference: 0.6682 kg/m3 per IPCC.
CH4_DENSITY_KG_PER_M3 = Decimal("0.6682")

#: Post-mining emission factor as fraction of mining factor.
#: Source: IPCC 2006 Guidelines default for Tier 1.
POST_MINING_FRACTION = Decimal("0.33")


# ===========================================================================
# Reference Data: Wastewater Treatment Factors
# ===========================================================================

#: Maximum methane producing capacity (Bo) for wastewater.
#: Source: IPCC 2006 Guidelines Vol. 5 Ch. 6.
#: Units: kg CH4 per kg BOD removed.
WASTEWATER_BO_KG_CH4_PER_KG_BOD = Decimal("0.25")

#: Methane Correction Factors (MCF) by treatment system type.
#: Source: IPCC 2006 Guidelines Vol. 5 Ch. 6, Table 6.3.
#: MCF represents the fraction of maximum methane potential realized
#: by each treatment system (0 = fully aerobic, 1 = fully anaerobic).
WASTEWATER_MCF: Dict[str, Dict[str, Any]] = {
    "UNTREATED_DISCHARGE": {
        "mcf": Decimal("0.1"),
        "description": "Untreated discharge to sea, river, or lake",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "AEROBIC_WELL_MANAGED": {
        "mcf": Decimal("0.0"),
        "description": "Well-managed centralized aerobic treatment (activated sludge)",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "AEROBIC_POORLY_MANAGED": {
        "mcf": Decimal("0.3"),
        "description": "Poorly managed or overloaded aerobic treatment",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "ANAEROBIC_REACTOR": {
        "mcf": Decimal("0.8"),
        "description": "Anaerobic reactor (covered digester without CH4 recovery)",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "ANAEROBIC_LAGOON_DEEP": {
        "mcf": Decimal("0.8"),
        "description": "Deep anaerobic lagoon (depth > 2 meters)",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "ANAEROBIC_LAGOON_SHALLOW": {
        "mcf": Decimal("0.2"),
        "description": "Shallow anaerobic lagoon (depth < 2 meters)",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "FACULTATIVE_LAGOON": {
        "mcf": Decimal("0.2"),
        "description": "Facultative lagoon (mixed aerobic/anaerobic)",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "SEPTIC_SYSTEM": {
        "mcf": Decimal("0.5"),
        "description": "Septic tank system",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "LATRINE_DRY": {
        "mcf": Decimal("0.1"),
        "description": "Dry climate latrine",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
    "LATRINE_WET": {
        "mcf": Decimal("0.7"),
        "description": "Wet climate or groundwater-connected latrine",
        "source": "IPCC 2006 Vol.5 Ch.6 Table 6.3",
    },
}

#: N2O emission factor for wastewater nitrogen.
#: Source: IPCC 2006 Guidelines Vol. 5 Ch. 6.
#: Units: kg N2O-N per kg nitrogen discharged.
WASTEWATER_N2O_EF_KG_PER_KG_N = Decimal("0.005")

#: Molecular weight ratio N2O/N for converting N2O-N to N2O.
#: N2O MW = 44.013, N MW = 14.007, ratio = 44.013/28.014 = 1.5714.
N2O_N_RATIO = Decimal("1.5714")


# ===========================================================================
# Reference Data: Pneumatic Device Emission Rates
# ===========================================================================

#: Default emission rates for gas-driven pneumatic devices.
#: Source: EPA Subpart W Table W-1A; CCAC Technical Guidance.
#:
#: high_bleed: 37.8 m3/day whole gas per device (EPA default = 37.3 scfh equiv.)
#: low_bleed: 1.39 scfh per device
#: intermittent: 13.5 scfh per device (EPA average actuation-weighted)
#: zero_bleed: 0 (instrument air or electric actuators)
#:
#: For unit consistency with the calculator engine, we store rates in
#: both m3/day and scfh.
PNEUMATIC_RATES: Dict[str, Dict[str, Decimal]] = {
    "high_bleed": {
        "rate_m3_per_day": Decimal("37.8"),
        "rate_scfh": Decimal("55.62"),
        "description": "Continuous high-bleed controller (> 6 scfh)",
        "source": "EPA Subpart W Table W-1A",
    },
    "low_bleed": {
        "rate_m3_per_day": Decimal("0.9440"),
        "rate_scfh": Decimal("1.39"),
        "description": "Continuous low-bleed controller (<= 6 scfh)",
        "source": "EPA Subpart W Table W-1A",
    },
    "intermittent": {
        "rate_m3_per_day": Decimal("9.166"),
        "rate_scfh": Decimal("13.50"),
        "description": "Intermittent-vent controller (EPA actuation-weighted avg)",
        "source": "EPA Subpart W Table W-1A",
    },
    "zero_bleed": {
        "rate_m3_per_day": Decimal("0"),
        "rate_scfh": Decimal("0"),
        "description": "Zero-bleed (instrument air or electric actuator)",
        "source": "N/A - no process gas emissions",
    },
}


# ===========================================================================
# Reference Data: Natural Gas Composition
# ===========================================================================

#: Default natural gas composition (mole fraction / volume fraction).
#: Source: Typical pipeline-quality natural gas.
#: CH4: 95%, C2H6: 2.5%, CO2: 1%, N2: 1.5%
#:
#: The gas_fraction for individual species is used to convert total
#: hydrocarbon (TOC) emission factors to species-specific rates.
DEFAULT_GAS_COMPOSITION: Dict[str, Dict[str, Any]] = {
    "CH4": {
        "mole_fraction": Decimal("0.950"),
        "molecular_weight": Decimal("16.043"),
        "description": "Methane",
    },
    "C2H6": {
        "mole_fraction": Decimal("0.025"),
        "molecular_weight": Decimal("30.070"),
        "description": "Ethane",
    },
    "CO2": {
        "mole_fraction": Decimal("0.010"),
        "molecular_weight": Decimal("44.009"),
        "description": "Carbon dioxide",
    },
    "N2": {
        "mole_fraction": Decimal("0.015"),
        "molecular_weight": Decimal("28.014"),
        "description": "Nitrogen",
    },
}

#: Weight-fraction computation helpers.
#: The weight fraction of each species = (mole_frac * MW) / sum(mole_frac * MW).
def _compute_weight_fractions(
    composition: Dict[str, Dict[str, Any]],
) -> Dict[str, Decimal]:
    """Compute weight fractions from mole fractions and molecular weights.

    Args:
        composition: Gas composition dictionary with mole_fraction and
            molecular_weight per species.

    Returns:
        Dictionary mapping species name to weight fraction (Decimal).
    """
    total_mw = Decimal("0")
    species_mw: Dict[str, Decimal] = {}
    for species, data in composition.items():
        mf = _D(data["mole_fraction"])
        mw = _D(data["molecular_weight"])
        contrib = mf * mw
        species_mw[species] = contrib
        total_mw += contrib

    if total_mw == Decimal("0"):
        return {s: Decimal("0") for s in composition}

    return {
        species: (contrib / total_mw).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        for species, contrib in species_mw.items()
    }


#: Pre-computed default weight fractions.
DEFAULT_WEIGHT_FRACTIONS: Dict[str, Decimal] = _compute_weight_fractions(
    DEFAULT_GAS_COMPOSITION
)


# ===========================================================================
# Reference Data: Global Warming Potentials
# ===========================================================================

#: GWP values for greenhouse gases across IPCC assessment reports.
#: Source: IPCC AR4 (2007), AR5 (2014), AR6 (2021).
#:
#: AR4:      CH4=25, N2O=298, CO2=1
#: AR5:      CH4=28, N2O=265, CO2=1
#: AR6:      CH4=27.9, N2O=273, CO2=1 (100-year horizon)
#: AR6_20YR: CH4=81.2, N2O=273, CO2=1 (20-year horizon)
GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": Decimal("1"),
        "CH4": Decimal("27.9"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("81.2"),
        "N2O": Decimal("273"),
    },
}


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class CustomEmissionFactor:
    """User-defined custom emission factor entry.

    Attributes:
        factor_id: Unique identifier for the custom factor.
        component_type: Component type this factor applies to.
        service_type: Service type this factor applies to.
        ef_kg_per_hr: Emission factor in kg TOC per hour.
        source: Source reference for the factor.
        description: Optional description.
        valid_from: ISO date from which this factor is valid.
        valid_to: ISO date until which this factor is valid.
        created_at: Record creation timestamp.
        provenance_hash: SHA-256 audit trail hash.
    """

    factor_id: str
    component_type: str
    service_type: str
    ef_kg_per_hr: Decimal
    source: str = ""
    description: str = ""
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    created_at: str = ""
    provenance_hash: str = ""


# ===========================================================================
# FugitiveSourceDatabaseEngine
# ===========================================================================


class FugitiveSourceDatabaseEngine:
    """Authoritative reference data repository for all fugitive emission
    source types, emission factors, gas compositions, and GWP values.

    This engine serves as the single source of truth for numeric constants
    used by the EmissionCalculatorEngine. All built-in data is immutable
    and derived from published regulatory sources (EPA, IPCC, CCAC).

    Users can register custom emission factors for site-specific data,
    which take precedence over built-in factors when queried.

    All lookups are deterministic and carry SHA-256 provenance hashes
    for complete audit trail transparency.

    Attributes:
        config: Configuration dictionary.

    Example:
        >>> db = FugitiveSourceDatabaseEngine()
        >>> info = db.get_source_info("EQUIPMENT_LEAK_VALVE_GAS")
        >>> ef = db.get_component_ef("valve", "gas")
        >>> gwp = db.get_gwp("CH4", "AR6")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the FugitiveSourceDatabaseEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - default_gwp_source (str): Default GWP source (AR4/AR5/AR6/AR6_20YR).
                - default_gas_composition (dict): Override default gas composition.
        """
        self._config = config or {}
        self._lock = threading.RLock()

        # Custom emission factor registry (mutable, lock-protected)
        self._custom_factors: Dict[str, CustomEmissionFactor] = {}

        # Custom gas composition override
        self._custom_composition: Optional[Dict[str, Dict[str, Any]]] = (
            self._config.get("default_gas_composition")
        )

        # Statistics
        self._total_queries: int = 0
        self._total_custom_factors: int = 0

        # Default GWP source
        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6"
        )

        logger.info(
            "FugitiveSourceDatabaseEngine initialized: "
            "%d source types, %d component EFs, %d screening ranges, "
            "%d correlation eqs, %d coal ranks, %d wastewater types, "
            "%d pneumatic types, default_gwp=%s",
            len(SOURCE_TYPES),
            len(COMPONENT_EMISSION_FACTORS),
            len(SCREENING_RANGE_FACTORS),
            len(CORRELATION_COEFFICIENTS),
            len(COAL_METHANE_FACTORS),
            len(WASTEWATER_MCF),
            len(PNEUMATIC_RATES),
            self._default_gwp_source,
        )

    # ------------------------------------------------------------------
    # Source Type Queries
    # ------------------------------------------------------------------

    def get_source_info(
        self,
        source_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get complete metadata for a fugitive emission source type.

        Args:
            source_type: Source type identifier (e.g., "EQUIPMENT_LEAK_VALVE_GAS").

        Returns:
            Dictionary with source type metadata and provenance hash,
            or None if the source type is not found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        info = SOURCE_TYPES.get(source_type)
        if info is None:
            logger.warning("Source type not found: %s", source_type)
            return None

        result = {
            "source_type": source_type,
            **info,
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_source_info(%s) in %.3fms",
            source_type, elapsed_ms,
        )
        return result

    def list_sources(
        self,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all source types with optional category filter.

        Args:
            category: Optional SourceCategory value to filter by
                (e.g., "EQUIPMENT_LEAK", "COAL_MINE").

        Returns:
            Dictionary with sources list, total count, and provenance hash.
        """
        t0 = time.monotonic()
        self._increment_queries()

        sources: List[Dict[str, Any]] = []
        for source_type, info in SOURCE_TYPES.items():
            if category is not None and info["category"] != category:
                continue
            sources.append({
                "source_type": source_type,
                "name": info["name"],
                "category": info["category"],
                "primary_gas": info["primary_gas"],
                "applicable_methods": info["applicable_methods"],
            })

        result = {
            "sources": sources,
            "total": len(sources),
            "category_filter": category,
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "list_sources(category=%s): %d results in %.3fms",
            category, len(sources), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Component Emission Factor Queries
    # ------------------------------------------------------------------

    def get_component_ef(
        self,
        component_type: str,
        service_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the EPA average emission factor for a component/service pair.

        Checks custom factors first (if any are registered), then falls
        back to built-in EPA factors.

        Args:
            component_type: Component type (e.g., "valve", "pump", "compressor").
            service_type: Service type (e.g., "gas", "light_liquid", "heavy_liquid").

        Returns:
            Dictionary with emission factor details and provenance hash,
            or None if no factor is found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        # Check custom factors first
        custom = self._get_custom_factor(component_type, service_type)
        if custom is not None:
            result = {
                "component_type": component_type,
                "service_type": service_type,
                "ef_kg_per_hr": str(custom.ef_kg_per_hr),
                "ef_decimal": custom.ef_kg_per_hr,
                "source": custom.source or "CUSTOM",
                "is_custom": True,
                "factor_id": custom.factor_id,
            }
            result["provenance_hash"] = _compute_hash({
                k: v for k, v in result.items() if k != "ef_decimal"
            })
            logger.debug(
                "get_component_ef(%s, %s): custom factor %s = %s kg/hr",
                component_type, service_type,
                custom.factor_id, custom.ef_kg_per_hr,
            )
            return result

        # Built-in EPA factors
        key = (component_type.lower(), service_type.lower())
        ef = COMPONENT_EMISSION_FACTORS.get(key)
        if ef is None:
            logger.warning(
                "No component EF found for (%s, %s)",
                component_type, service_type,
            )
            return None

        result = {
            "component_type": component_type,
            "service_type": service_type,
            "ef_kg_per_hr": str(ef),
            "ef_decimal": ef,
            "source": "EPA-453/R-95-017",
            "is_custom": False,
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items() if k != "ef_decimal"
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_component_ef(%s, %s): %s kg/hr in %.3fms",
            component_type, service_type, ef, elapsed_ms,
        )
        return result

    def list_components(self) -> Dict[str, Any]:
        """List all available component/service emission factor pairs.

        Returns:
            Dictionary with list of all component EF entries and metadata.
        """
        t0 = time.monotonic()
        self._increment_queries()

        entries: List[Dict[str, Any]] = []
        for (comp, svc), ef in sorted(COMPONENT_EMISSION_FACTORS.items()):
            entries.append({
                "component_type": comp,
                "service_type": svc,
                "ef_kg_per_hr": str(ef),
            })

        # Append custom factors
        with self._lock:
            for cf in self._custom_factors.values():
                entries.append({
                    "component_type": cf.component_type,
                    "service_type": cf.service_type,
                    "ef_kg_per_hr": str(cf.ef_kg_per_hr),
                    "source": cf.source or "CUSTOM",
                    "is_custom": True,
                    "factor_id": cf.factor_id,
                })

        result = {
            "components": entries,
            "total_builtin": len(COMPONENT_EMISSION_FACTORS),
            "total_custom": self._total_custom_factors,
            "total": len(entries),
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "list_components: %d entries in %.3fms",
            len(entries), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Screening Range Factor Queries
    # ------------------------------------------------------------------

    def get_screening_factor(
        self,
        component_type: str,
        service_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get screening range leak/no-leak emission factors.

        Args:
            component_type: Component type (e.g., "valve", "pump").
            service_type: Service type (e.g., "gas", "light_liquid").

        Returns:
            Dictionary with leak_ef, no_leak_ef, and provenance hash,
            or None if no screening factors are found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        key = (component_type.lower(), service_type.lower())
        factors = SCREENING_RANGE_FACTORS.get(key)
        if factors is None:
            logger.warning(
                "No screening range factors for (%s, %s)",
                component_type, service_type,
            )
            return None

        result = {
            "component_type": component_type,
            "service_type": service_type,
            "leak_ef_kg_per_hr": str(factors["leak_ef"]),
            "no_leak_ef_kg_per_hr": str(factors["no_leak_ef"]),
            "leak_ef_decimal": factors["leak_ef"],
            "no_leak_ef_decimal": factors["no_leak_ef"],
            "source": "EPA-453/R-95-017 Table 2-4/2-9",
            "threshold_ppmv": 10000,
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items()
            if k not in ("leak_ef_decimal", "no_leak_ef_decimal")
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_screening_factor(%s, %s): leak=%s, no_leak=%s in %.3fms",
            component_type, service_type,
            factors["leak_ef"], factors["no_leak_ef"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Correlation Equation Queries
    # ------------------------------------------------------------------

    def get_correlation_coefficients(
        self,
        component_type: str,
        service_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get correlation equation coefficients for a component/service pair.

        Equation: log10(kg/hr) = a + b * log10(ppmv)

        Args:
            component_type: Component type.
            service_type: Service type.

        Returns:
            Dictionary with coefficients a, b, valid range, and provenance hash,
            or None if no coefficients are found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        key = (component_type.lower(), service_type.lower())
        coeffs = CORRELATION_COEFFICIENTS.get(key)
        if coeffs is None:
            logger.warning(
                "No correlation coefficients for (%s, %s)",
                component_type, service_type,
            )
            return None

        result = {
            "component_type": component_type,
            "service_type": service_type,
            "a": str(coeffs["a"]),
            "b": str(coeffs["b"]),
            "a_decimal": coeffs["a"],
            "b_decimal": coeffs["b"],
            "min_ppmv": str(coeffs["min_ppmv"]),
            "max_ppmv": str(coeffs["max_ppmv"]),
            "default_zero_ef_kg_per_hr": str(coeffs["default_zero_ef"]),
            "default_zero_ef_decimal": coeffs["default_zero_ef"],
            "equation": "log10(kg/hr) = a + b * log10(ppmv)",
            "source": "EPA-453/R-95-017 Table 2-8",
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items()
            if not k.endswith("_decimal")
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_correlation_coefficients(%s, %s): a=%s, b=%s in %.3fms",
            component_type, service_type,
            coeffs["a"], coeffs["b"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Coal Mine Methane Factor Queries
    # ------------------------------------------------------------------

    def get_coal_methane_factor(
        self,
        coal_rank: str,
    ) -> Optional[Dict[str, Any]]:
        """Get coal mine methane emission factor by coal rank.

        Args:
            coal_rank: Coal rank (ANTHRACITE, BITUMINOUS, SUBBITUMINOUS, LIGNITE).

        Returns:
            Dictionary with emission factor, unit, and provenance hash,
            or None if the coal rank is not found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        rank_upper = coal_rank.upper()
        factor_data = COAL_METHANE_FACTORS.get(rank_upper)
        if factor_data is None:
            logger.warning("Unknown coal rank: %s", coal_rank)
            return None

        result = {
            "coal_rank": rank_upper,
            "ef_m3_per_tonne": str(factor_data["ef_m3_per_tonne"]),
            "ef_m3_per_tonne_decimal": factor_data["ef_m3_per_tonne"],
            "ch4_density_kg_per_m3": str(CH4_DENSITY_KG_PER_M3),
            "ef_kg_per_tonne": str(
                (factor_data["ef_m3_per_tonne"] * CH4_DENSITY_KG_PER_M3).quantize(
                    _PRECISION, rounding=ROUND_HALF_UP
                )
            ),
            "ef_kg_per_tonne_decimal": (
                factor_data["ef_m3_per_tonne"] * CH4_DENSITY_KG_PER_M3
            ).quantize(_PRECISION, rounding=ROUND_HALF_UP),
            "post_mining_fraction": str(POST_MINING_FRACTION),
            "uncertainty_pct": str(factor_data["uncertainty_pct"]),
            "description": factor_data["description"],
            "source": factor_data["source"],
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items()
            if not k.endswith("_decimal")
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_coal_methane_factor(%s): %s m3/t, %s kg/t in %.3fms",
            rank_upper, factor_data["ef_m3_per_tonne"],
            result["ef_kg_per_tonne"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Wastewater Factor Queries
    # ------------------------------------------------------------------

    def get_wastewater_factor(
        self,
        treatment_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get wastewater treatment emission factors for a treatment system type.

        Returns the maximum methane producing capacity (Bo), the methane
        correction factor (MCF), and the N2O emission factor for nitrogen.

        Args:
            treatment_type: Treatment system type (e.g., "ANAEROBIC_REACTOR",
                "AEROBIC_WELL_MANAGED").

        Returns:
            Dictionary with Bo, MCF, N2O factor, and provenance hash,
            or None if the treatment type is not found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        type_upper = treatment_type.upper()
        mcf_data = WASTEWATER_MCF.get(type_upper)
        if mcf_data is None:
            logger.warning("Unknown wastewater treatment type: %s", treatment_type)
            return None

        result = {
            "treatment_type": type_upper,
            "bo_kg_ch4_per_kg_bod": str(WASTEWATER_BO_KG_CH4_PER_KG_BOD),
            "bo_decimal": WASTEWATER_BO_KG_CH4_PER_KG_BOD,
            "mcf": str(mcf_data["mcf"]),
            "mcf_decimal": mcf_data["mcf"],
            "n2o_ef_kg_per_kg_n": str(WASTEWATER_N2O_EF_KG_PER_KG_N),
            "n2o_ef_decimal": WASTEWATER_N2O_EF_KG_PER_KG_N,
            "n2o_n_ratio": str(N2O_N_RATIO),
            "description": mcf_data["description"],
            "source": mcf_data["source"],
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items()
            if not k.endswith("_decimal")
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_wastewater_factor(%s): Bo=%s, MCF=%s in %.3fms",
            type_upper, WASTEWATER_BO_KG_CH4_PER_KG_BOD,
            mcf_data["mcf"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Pneumatic Rate Queries
    # ------------------------------------------------------------------

    def get_pneumatic_rate(
        self,
        device_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get emission rate for a pneumatic device type.

        Args:
            device_type: Device type (high_bleed, low_bleed, intermittent, zero_bleed).

        Returns:
            Dictionary with emission rates in m3/day and scfh,
            plus provenance hash, or None if device type not found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        type_lower = device_type.lower()
        rate_data = PNEUMATIC_RATES.get(type_lower)
        if rate_data is None:
            logger.warning("Unknown pneumatic device type: %s", device_type)
            return None

        result = {
            "device_type": type_lower,
            "rate_m3_per_day": str(rate_data["rate_m3_per_day"]),
            "rate_m3_per_day_decimal": rate_data["rate_m3_per_day"],
            "rate_scfh": str(rate_data["rate_scfh"]),
            "rate_scfh_decimal": rate_data["rate_scfh"],
            "description": rate_data["description"],
            "source": rate_data["source"],
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items()
            if not k.endswith("_decimal")
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_pneumatic_rate(%s): %s m3/day in %.3fms",
            type_lower, rate_data["rate_m3_per_day"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Gas Composition Queries
    # ------------------------------------------------------------------

    def get_gas_composition(
        self,
        custom_composition: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Get natural gas composition with mole and weight fractions.

        Uses custom composition if provided, then engine-level override,
        then default pipeline-quality gas composition.

        Args:
            custom_composition: Optional custom composition dictionary with
                species -> {mole_fraction, molecular_weight}.

        Returns:
            Dictionary with species details, weight fractions, and provenance hash.
        """
        t0 = time.monotonic()
        self._increment_queries()

        # Priority: argument > engine config > module default
        composition = (
            custom_composition
            or self._custom_composition
            or DEFAULT_GAS_COMPOSITION
        )

        # Compute weight fractions
        weight_fractions = _compute_weight_fractions(composition)

        species_details: List[Dict[str, Any]] = []
        for species, data in composition.items():
            species_details.append({
                "species": species,
                "mole_fraction": str(data["mole_fraction"]),
                "molecular_weight": str(data["molecular_weight"]),
                "weight_fraction": str(weight_fractions.get(species, Decimal("0"))),
                "description": data.get("description", species),
            })

        is_custom = composition is not DEFAULT_GAS_COMPOSITION
        result = {
            "species": species_details,
            "total_species": len(species_details),
            "is_custom": is_custom,
            "ch4_mole_fraction": str(
                composition.get("CH4", {}).get("mole_fraction", Decimal("0"))
            ),
            "ch4_weight_fraction": str(weight_fractions.get("CH4", Decimal("0"))),
        }
        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_gas_composition: %d species, custom=%s in %.3fms",
            len(species_details), is_custom, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # GWP Queries
    # ------------------------------------------------------------------

    def get_gwp(
        self,
        gas: str,
        gwp_source: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the Global Warming Potential for a greenhouse gas.

        Args:
            gas: Gas species (CO2, CH4, N2O).
            gwp_source: IPCC assessment report (AR4, AR5, AR6, AR6_20YR).
                Defaults to the engine's configured default.

        Returns:
            Dictionary with GWP value and provenance hash,
            or None if gas or source is not found.
        """
        t0 = time.monotonic()
        self._increment_queries()

        source = (gwp_source or self._default_gwp_source).upper()
        gas_upper = gas.upper()

        gwp_table = GWP_VALUES.get(source)
        if gwp_table is None:
            logger.warning("Unknown GWP source: %s", source)
            return None

        gwp_value = gwp_table.get(gas_upper)
        if gwp_value is None:
            logger.warning("No GWP for gas %s in %s", gas_upper, source)
            return None

        result = {
            "gas": gas_upper,
            "gwp_source": source,
            "gwp_value": str(gwp_value),
            "gwp_decimal": gwp_value,
            "horizon": "20-year" if source == "AR6_20YR" else "100-year",
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items()
            if not k.endswith("_decimal")
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_gwp(%s, %s): %s in %.3fms",
            gas_upper, source, gwp_value, elapsed_ms,
        )
        return result

    def get_all_gwps(
        self,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get all GWP values for a given assessment report.

        Args:
            gwp_source: IPCC assessment report. Defaults to engine default.

        Returns:
            Dictionary with all gas GWP values and provenance hash.
        """
        t0 = time.monotonic()
        self._increment_queries()

        source = (gwp_source or self._default_gwp_source).upper()
        gwp_table = GWP_VALUES.get(source)
        if gwp_table is None:
            return {
                "error": f"Unknown GWP source: {source}",
                "available_sources": list(GWP_VALUES.keys()),
            }

        result = {
            "gwp_source": source,
            "horizon": "20-year" if source == "AR6_20YR" else "100-year",
            "values": {
                gas: str(gwp) for gas, gwp in gwp_table.items()
            },
            "values_decimal": dict(gwp_table),
        }
        result["provenance_hash"] = _compute_hash({
            k: v for k, v in result.items()
            if k != "values_decimal"
        })

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug(
            "get_all_gwps(%s): %d gases in %.3fms",
            source, len(gwp_table), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Custom Emission Factor Management
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a custom emission factor for a component/service pair.

        Custom factors take precedence over built-in EPA factors when
        queried via get_component_ef().

        Args:
            data: Dictionary with:
                - component_type (str): Component type.
                - service_type (str): Service type.
                - ef_kg_per_hr (float/str/Decimal): Emission factor.
                - source (str, optional): Source reference.
                - description (str, optional): Description.
                - valid_from (str, optional): ISO date.
                - valid_to (str, optional): ISO date.

        Returns:
            Dictionary with factor_id and provenance hash.

        Raises:
            ValueError: If required fields are missing or ef is negative.
        """
        component_type = data.get("component_type", "")
        service_type = data.get("service_type", "")
        ef_raw = data.get("ef_kg_per_hr")

        if not component_type:
            raise ValueError("component_type is required")
        if not service_type:
            raise ValueError("service_type is required")
        if ef_raw is None:
            raise ValueError("ef_kg_per_hr is required")

        ef = _D(ef_raw)
        if ef < Decimal("0"):
            raise ValueError(
                f"ef_kg_per_hr must be >= 0, got {ef}"
            )

        factor_id = f"cef_{uuid4().hex[:12]}"
        now_iso = _utcnow().isoformat()

        record = CustomEmissionFactor(
            factor_id=factor_id,
            component_type=component_type.lower(),
            service_type=service_type.lower(),
            ef_kg_per_hr=ef,
            source=data.get("source", ""),
            description=data.get("description", ""),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            created_at=now_iso,
        )

        record.provenance_hash = _compute_hash({
            "factor_id": factor_id,
            "component_type": record.component_type,
            "service_type": record.service_type,
            "ef_kg_per_hr": str(ef),
            "created_at": now_iso,
        })

        with self._lock:
            self._custom_factors[factor_id] = record
            self._total_custom_factors += 1

        logger.info(
            "Registered custom EF %s: (%s, %s) = %s kg/hr",
            factor_id, record.component_type,
            record.service_type, ef,
        )

        return {
            "factor_id": factor_id,
            "component_type": record.component_type,
            "service_type": record.service_type,
            "ef_kg_per_hr": str(ef),
            "source": record.source,
            "provenance_hash": record.provenance_hash,
        }

    def remove_custom_factor(
        self,
        factor_id: str,
    ) -> bool:
        """Remove a custom emission factor by ID.

        Args:
            factor_id: Factor identifier to remove.

        Returns:
            True if the factor was found and removed, False otherwise.
        """
        with self._lock:
            if factor_id in self._custom_factors:
                del self._custom_factors[factor_id]
                logger.info("Removed custom EF: %s", factor_id)
                return True
        logger.warning("Custom EF not found for removal: %s", factor_id)
        return False

    def list_custom_factors(self) -> Dict[str, Any]:
        """List all registered custom emission factors.

        Returns:
            Dictionary with custom factors list and provenance hash.
        """
        self._increment_queries()

        with self._lock:
            factors = []
            for cf in self._custom_factors.values():
                factors.append({
                    "factor_id": cf.factor_id,
                    "component_type": cf.component_type,
                    "service_type": cf.service_type,
                    "ef_kg_per_hr": str(cf.ef_kg_per_hr),
                    "source": cf.source,
                    "description": cf.description,
                    "valid_from": cf.valid_from,
                    "valid_to": cf.valid_to,
                    "created_at": cf.created_at,
                    "provenance_hash": cf.provenance_hash,
                })

        result = {
            "custom_factors": factors,
            "total": len(factors),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Utility: Weight Fraction Lookup
    # ------------------------------------------------------------------

    def get_weight_fraction(
        self,
        species: str,
        custom_composition: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Decimal:
        """Get the weight fraction for a specific gas species.

        Convenience method for the EmissionCalculatorEngine to convert
        TOC emission factors to species-specific rates.

        Args:
            species: Gas species (e.g., "CH4").
            custom_composition: Optional custom gas composition.

        Returns:
            Weight fraction as Decimal. Returns 0 if species not in composition.
        """
        composition = (
            custom_composition
            or self._custom_composition
            or DEFAULT_GAS_COMPOSITION
        )
        weight_fractions = _compute_weight_fractions(composition)
        return weight_fractions.get(species, Decimal("0"))

    def get_mole_fraction(
        self,
        species: str,
        custom_composition: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Decimal:
        """Get the mole fraction for a specific gas species.

        Args:
            species: Gas species (e.g., "CH4").
            custom_composition: Optional custom gas composition.

        Returns:
            Mole fraction as Decimal. Returns 0 if species not in composition.
        """
        composition = (
            custom_composition
            or self._custom_composition
            or DEFAULT_GAS_COMPOSITION
        )
        species_data = composition.get(species)
        if species_data is None:
            return Decimal("0")
        return _D(species_data["mole_fraction"])

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine query and registry statistics.

        Returns:
            Dictionary with query counts, registry sizes, and metadata.
        """
        with self._lock:
            return {
                "total_queries": self._total_queries,
                "total_custom_factors": self._total_custom_factors,
                "builtin_source_types": len(SOURCE_TYPES),
                "builtin_component_efs": len(COMPONENT_EMISSION_FACTORS),
                "builtin_screening_ranges": len(SCREENING_RANGE_FACTORS),
                "builtin_correlation_eqs": len(CORRELATION_COEFFICIENTS),
                "builtin_coal_ranks": len(COAL_METHANE_FACTORS),
                "builtin_wastewater_types": len(WASTEWATER_MCF),
                "builtin_pneumatic_types": len(PNEUMATIC_RATES),
                "gwp_sources": list(GWP_VALUES.keys()),
                "default_gwp_source": self._default_gwp_source,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _increment_queries(self) -> None:
        """Thread-safe increment of query counter."""
        with self._lock:
            self._total_queries += 1

    def _get_custom_factor(
        self,
        component_type: str,
        service_type: str,
    ) -> Optional[CustomEmissionFactor]:
        """Find the most recently registered custom factor for a component/service pair.

        Args:
            component_type: Component type to match.
            service_type: Service type to match.

        Returns:
            CustomEmissionFactor or None.
        """
        comp_lower = component_type.lower()
        svc_lower = service_type.lower()

        with self._lock:
            matching: Optional[CustomEmissionFactor] = None
            for cf in self._custom_factors.values():
                if (cf.component_type == comp_lower
                        and cf.service_type == svc_lower):
                    # Pick the most recently created
                    if matching is None or cf.created_at > matching.created_at:
                        matching = cf
            return matching
