# -*- coding: utf-8 -*-
"""
FuelBasedCalculatorEngine - Fuel Consumption Emissions Calculator (Engine 3 of 7)

AGENT-MRV-017: Upstream Transportation & Distribution Agent

Core calculation engine implementing the fuel-based method for GHG Protocol
Scope 3 Category 4 emissions. This method uses actual fuel consumption data
to calculate emissions, providing higher accuracy than distance-based or
spend-based approaches when fuel records are available.

Core Formula:
    emissions = fuel_consumed x fuel_emission_factor

Scope Breakdown:
    TTW = fuel x TTW_EF   (Tank-to-Wheel: direct combustion)
    WTT = fuel x WTT_EF   (Well-to-Tank: upstream fuel lifecycle)
    WTW = TTW + WTT        (Well-to-Wheel: complete lifecycle)

Supported Fuels (15):
    Liquid  - Diesel, Petrol, Jet Kerosene, HFO, VLSFO, MGO, Biodiesel B20,
              Biodiesel B100, HVO, Methanol
    Gaseous - CNG, LNG (marine), LNG (road)
    Other   - Electricity (grid-dependent), Hydrogen (production-dependent)

All calculations use Python Decimal arithmetic for zero-hallucination
determinism. Every result includes a SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Stateless per-calculation. Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.upstream_transportation.fuel_based_calculator import (
    ...     FuelBasedCalculatorEngine,
    ... )
    >>> engine = FuelBasedCalculatorEngine()
    >>> result = engine.calculate(FuelConsumptionInput(
    ...     record_id="FC-001",
    ...     fuel_type=TransportFuelType.DIESEL,
    ...     fuel_consumed_litres=Decimal("500"),
    ...     mode=TransportMode.ROAD,
    ... ))
    >>> result.emissions_kgco2e
    Decimal('1650.00')

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-017 Upstream Transportation (GL-MRV-S3-004)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["FuelBasedCalculatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.upstream_transportation.models import (
        TransportFuelType,
        TransportMode,
        EFScope,
        GWPSource,
        EmissionGas,
        AllocationMethod,
        CalculationMethod,
        LadenState,
        DQIScore,
        FuelConsumptionInput,
        AllocationConfig,
        LegResult,
        FUEL_EMISSION_FACTORS,
        GWP_VALUES,
        UNCERTAINTY_RANGES,
        DQI_SCORE_VALUES,
        calculate_provenance_hash,
        get_gwp,
        calculate_co2e,
        get_dqi_composite_score,
        get_dqi_quality_tier,
        VERSION,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.warning(
        "greenlang.upstream_transportation.models not available; "
        "using inline fallback constants"
    )

try:
    from greenlang.upstream_transportation.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.upstream_transportation.provenance import (
        get_provenance_tracker,
        ProvenanceStage,
        hash_fuel_calculation,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False

try:
    from greenlang.upstream_transportation.metrics import (
        UpstreamTransportationMetrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID: str = "GL-MRV-S3-004"
AGENT_COMPONENT: str = "AGENT-MRV-017"
ENGINE_NAME: str = "FuelBasedCalculatorEngine"
ENGINE_NUMBER: int = 3
_VERSION: str = "1.0.0"

# Decimal precision: 8 decimal places, half-up rounding
_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")

# ---------------------------------------------------------------------------
# Fuel Emission Factor Table (kgCO2e per unit)
# Authoritative lookup -- all values from IPCC 2006 / DEFRA 2023 / GLEC v3.0
# ---------------------------------------------------------------------------

# Each entry: (ttw, wtt, wtw, unit)
# unit is either "litre", "kg", or "kWh"

_FUEL_EF_TABLE: Dict[str, Dict[str, Any]] = {
    "diesel": {
        "ttw": Decimal("2.68"),
        "wtt": Decimal("0.62"),
        "wtw": Decimal("3.30"),
        "unit": "litre",
        "density_kg_per_litre": Decimal("0.832"),
        "ncv_mj_per_kg": Decimal("42.6"),
        "biogenic_fraction": Decimal("0"),
    },
    "petrol": {
        "ttw": Decimal("2.31"),
        "wtt": Decimal("0.58"),
        "wtw": Decimal("2.89"),
        "unit": "litre",
        "density_kg_per_litre": Decimal("0.745"),
        "ncv_mj_per_kg": Decimal("43.0"),
        "biogenic_fraction": Decimal("0"),
    },
    "jet_kerosene": {
        "ttw": Decimal("2.54"),
        "wtt": Decimal("0.55"),
        "wtw": Decimal("3.09"),
        "unit": "litre",
        "density_kg_per_litre": Decimal("0.800"),
        "ncv_mj_per_kg": Decimal("43.2"),
        "biogenic_fraction": Decimal("0"),
    },
    "hfo": {
        "ttw": Decimal("3.114"),
        "wtt": Decimal("0.50"),
        "wtw": Decimal("3.614"),
        "unit": "kg",
        "density_kg_per_litre": Decimal("0.991"),
        "ncv_mj_per_kg": Decimal("40.2"),
        "biogenic_fraction": Decimal("0"),
    },
    "vlsfo": {
        "ttw": Decimal("3.151"),
        "wtt": Decimal("0.52"),
        "wtw": Decimal("3.671"),
        "unit": "kg",
        "density_kg_per_litre": Decimal("0.950"),
        "ncv_mj_per_kg": Decimal("41.0"),
        "biogenic_fraction": Decimal("0"),
    },
    "mgo": {
        "ttw": Decimal("3.206"),
        "wtt": Decimal("0.60"),
        "wtw": Decimal("3.806"),
        "unit": "kg",
        "density_kg_per_litre": Decimal("0.855"),
        "ncv_mj_per_kg": Decimal("42.7"),
        "biogenic_fraction": Decimal("0"),
    },
    "lng_marine": {
        "ttw": Decimal("2.75"),
        "wtt": Decimal("0.65"),
        "wtw": Decimal("3.40"),
        "unit": "kg",
        "density_kg_per_litre": Decimal("0.450"),
        "ncv_mj_per_kg": Decimal("48.6"),
        "biogenic_fraction": Decimal("0"),
        "methane_slip_included": True,
    },
    "cng": {
        "ttw": Decimal("2.54"),
        "wtt": Decimal("0.45"),
        "wtw": Decimal("2.99"),
        "unit": "kg",
        "density_kg_per_litre": Decimal("0.200"),
        "ncv_mj_per_kg": Decimal("48.0"),
        "biogenic_fraction": Decimal("0"),
    },
    "lng_road": {
        "ttw": Decimal("2.56"),
        "wtt": Decimal("0.60"),
        "wtw": Decimal("3.16"),
        "unit": "kg",
        "density_kg_per_litre": Decimal("0.450"),
        "ncv_mj_per_kg": Decimal("48.6"),
        "biogenic_fraction": Decimal("0"),
    },
    "electricity": {
        "ttw": Decimal("0.0"),
        "wtt_global": Decimal("0.475"),
        "wtt_us": Decimal("0.417"),
        "wtt_eu": Decimal("0.295"),
        "wtt_china": Decimal("0.581"),
        "wtt_uk": Decimal("0.233"),
        "wtt_india": Decimal("0.708"),
        "wtt_japan": Decimal("0.457"),
        "wtt_brazil": Decimal("0.075"),
        "wtt_france": Decimal("0.052"),
        "wtt_germany": Decimal("0.338"),
        "wtt_canada": Decimal("0.120"),
        "wtt_australia": Decimal("0.656"),
        "unit": "kWh",
        "biogenic_fraction": Decimal("0"),
    },
    "hydrogen": {
        "ttw": Decimal("0.0"),
        "wtt_grey": Decimal("10.80"),
        "wtt_blue": Decimal("3.50"),
        "wtt_green": Decimal("1.50"),
        "wtt_default": Decimal("10.80"),
        "unit": "kg",
        "biogenic_fraction": Decimal("0"),
    },
    "biodiesel_b20": {
        "ttw": Decimal("2.14"),
        "wtt": Decimal("0.35"),
        "wtw": Decimal("2.49"),
        "unit": "litre",
        "density_kg_per_litre": Decimal("0.838"),
        "ncv_mj_per_kg": Decimal("41.8"),
        "biogenic_fraction": Decimal("0.20"),
        "blend_base": "diesel",
        "blend_bio": "biodiesel_b100",
        "blend_ratio": Decimal("0.20"),
    },
    "biodiesel_b100": {
        "ttw": Decimal("0.0"),
        "wtt": Decimal("0.85"),
        "wtw": Decimal("0.85"),
        "unit": "litre",
        "density_kg_per_litre": Decimal("0.880"),
        "ncv_mj_per_kg": Decimal("37.0"),
        "biogenic_fraction": Decimal("1.0"),
        "biogenic_co2_per_litre": Decimal("2.50"),
    },
    "hvo": {
        "ttw": Decimal("0.0"),
        "wtt": Decimal("0.50"),
        "wtw": Decimal("0.50"),
        "unit": "litre",
        "density_kg_per_litre": Decimal("0.780"),
        "ncv_mj_per_kg": Decimal("44.0"),
        "biogenic_fraction": Decimal("1.0"),
        "biogenic_co2_per_litre": Decimal("2.46"),
    },
    "methanol": {
        "ttw": Decimal("1.375"),
        "wtt": Decimal("0.40"),
        "wtw": Decimal("1.775"),
        "unit": "litre",
        "density_kg_per_litre": Decimal("0.792"),
        "ncv_mj_per_kg": Decimal("19.9"),
        "biogenic_fraction": Decimal("0"),
    },
}

# ---------------------------------------------------------------------------
# Fuel density table (kg per litre)
# ---------------------------------------------------------------------------

_FUEL_DENSITY: Dict[str, Decimal] = {
    "diesel": Decimal("0.832"),
    "petrol": Decimal("0.745"),
    "jet_kerosene": Decimal("0.800"),
    "hfo": Decimal("0.991"),
    "vlsfo": Decimal("0.950"),
    "mgo": Decimal("0.855"),
    "lng_marine": Decimal("0.450"),
    "lng_road": Decimal("0.450"),
    "cng": Decimal("0.200"),
    "biodiesel_b20": Decimal("0.838"),
    "biodiesel_b100": Decimal("0.880"),
    "hvo": Decimal("0.780"),
    "methanol": Decimal("0.792"),
}

# ---------------------------------------------------------------------------
# Fuel heating values (MJ/kg, NCV basis)
# ---------------------------------------------------------------------------

_FUEL_HEATING_VALUES: Dict[str, Dict[str, Decimal]] = {
    "diesel": {"ncv": Decimal("42.6"), "hhv": Decimal("45.4")},
    "petrol": {"ncv": Decimal("43.0"), "hhv": Decimal("46.4")},
    "jet_kerosene": {"ncv": Decimal("43.2"), "hhv": Decimal("46.2")},
    "hfo": {"ncv": Decimal("40.2"), "hhv": Decimal("42.5")},
    "vlsfo": {"ncv": Decimal("41.0"), "hhv": Decimal("43.3")},
    "mgo": {"ncv": Decimal("42.7"), "hhv": Decimal("45.5")},
    "lng_marine": {"ncv": Decimal("48.6"), "hhv": Decimal("55.5")},
    "lng_road": {"ncv": Decimal("48.6"), "hhv": Decimal("55.5")},
    "cng": {"ncv": Decimal("48.0"), "hhv": Decimal("55.0")},
    "biodiesel_b20": {"ncv": Decimal("41.8"), "hhv": Decimal("44.6")},
    "biodiesel_b100": {"ncv": Decimal("37.0"), "hhv": Decimal("39.8")},
    "hvo": {"ncv": Decimal("44.0"), "hhv": Decimal("47.1")},
    "methanol": {"ncv": Decimal("19.9"), "hhv": Decimal("22.7")},
}

# ---------------------------------------------------------------------------
# Gas split ratios (fraction of total CO2e by gas)
# Source: IPCC 2006 Vol 2 Ch 3, DEFRA 2023 conversion factors
# ---------------------------------------------------------------------------

_GAS_SPLIT_RATIOS: Dict[str, Dict[str, Decimal]] = {
    "diesel": {
        "co2": Decimal("0.993"),
        "ch4": Decimal("0.001"),
        "n2o": Decimal("0.006"),
    },
    "petrol": {
        "co2": Decimal("0.990"),
        "ch4": Decimal("0.003"),
        "n2o": Decimal("0.007"),
    },
    "jet_kerosene": {
        "co2": Decimal("0.994"),
        "ch4": Decimal("0.001"),
        "n2o": Decimal("0.005"),
    },
    "hfo": {
        "co2": Decimal("0.995"),
        "ch4": Decimal("0.001"),
        "n2o": Decimal("0.004"),
    },
    "vlsfo": {
        "co2": Decimal("0.995"),
        "ch4": Decimal("0.001"),
        "n2o": Decimal("0.004"),
    },
    "mgo": {
        "co2": Decimal("0.994"),
        "ch4": Decimal("0.001"),
        "n2o": Decimal("0.005"),
    },
    "lng_marine": {
        "co2": Decimal("0.920"),
        "ch4": Decimal("0.075"),
        "n2o": Decimal("0.005"),
    },
    "lng_road": {
        "co2": Decimal("0.940"),
        "ch4": Decimal("0.055"),
        "n2o": Decimal("0.005"),
    },
    "cng": {
        "co2": Decimal("0.945"),
        "ch4": Decimal("0.050"),
        "n2o": Decimal("0.005"),
    },
    "biodiesel_b20": {
        "co2": Decimal("0.993"),
        "ch4": Decimal("0.001"),
        "n2o": Decimal("0.006"),
    },
    "biodiesel_b100": {
        "co2": Decimal("0.990"),
        "ch4": Decimal("0.002"),
        "n2o": Decimal("0.008"),
    },
    "hvo": {
        "co2": Decimal("0.990"),
        "ch4": Decimal("0.002"),
        "n2o": Decimal("0.008"),
    },
    "methanol": {
        "co2": Decimal("0.992"),
        "ch4": Decimal("0.003"),
        "n2o": Decimal("0.005"),
    },
    "electricity": {
        "co2": Decimal("0.980"),
        "ch4": Decimal("0.010"),
        "n2o": Decimal("0.010"),
    },
    "hydrogen": {
        "co2": Decimal("0.970"),
        "ch4": Decimal("0.020"),
        "n2o": Decimal("0.010"),
    },
}

# ---------------------------------------------------------------------------
# Methane slip factors by engine type (kg CH4 per kg fuel)
# Source: IMO Fourth GHG Study 2020, ICCT 2022
# ---------------------------------------------------------------------------

_METHANE_SLIP_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "lng_marine": {
        "low_pressure_2_stroke": Decimal("0.0019"),
        "high_pressure_2_stroke": Decimal("0.0003"),
        "medium_speed_4_stroke": Decimal("0.0364"),
        "steam_turbine": Decimal("0.0001"),
        "default": Decimal("0.0180"),
    },
    "lng_road": {
        "spark_ignition": Decimal("0.0110"),
        "hpdi": Decimal("0.0020"),
        "default": Decimal("0.0065"),
    },
    "cng": {
        "spark_ignition": Decimal("0.0130"),
        "stoichiometric": Decimal("0.0055"),
        "default": Decimal("0.0090"),
    },
}

# ---------------------------------------------------------------------------
# Unit conversion factors
# ---------------------------------------------------------------------------

_LITRES_PER_US_GALLON = Decimal("3.78541")
_LITRES_PER_IMP_GALLON = Decimal("4.54609")
_LITRES_PER_CUBIC_METRE = Decimal("1000")
_KWH_PER_THERM = Decimal("29.3071")
_MJ_PER_KWH = Decimal("3.6")
_BTU_PER_KWH = Decimal("3412.14")
_KG_PER_METRIC_TON = Decimal("1000")

# ---------------------------------------------------------------------------
# Default fuel efficiency estimates (litres per 100 km)
# Used by estimate_fuel_from_distance fallback
# ---------------------------------------------------------------------------

_DEFAULT_FUEL_EFFICIENCY: Dict[str, Dict[str, Decimal]] = {
    "road": {
        "lcv_diesel": Decimal("12.0"),
        "lcv_petrol": Decimal("14.5"),
        "hgv_rigid_7_5_17t": Decimal("22.0"),
        "hgv_rigid_17t_plus": Decimal("28.0"),
        "hgv_artic_33t_plus": Decimal("32.0"),
        "hgv_artic_3_5_33t": Decimal("27.0"),
        "hgv_cng": Decimal("38.0"),
        "hgv_lng": Decimal("36.0"),
        "default": Decimal("30.0"),
    },
    "rail": {
        "diesel": Decimal("4.5"),
        "default": Decimal("4.5"),
    },
    "maritime": {
        "container_feeder": Decimal("3500"),
        "container_panamax": Decimal("8500"),
        "container_ulcv": Decimal("14000"),
        "bulk_handysize": Decimal("2800"),
        "bulk_capesize": Decimal("5200"),
        "tanker_product": Decimal("3200"),
        "default": Decimal("5000"),
    },
    "air": {
        "narrowbody_freighter": Decimal("3200"),
        "widebody_freighter": Decimal("8500"),
        "belly_freight": Decimal("5500"),
        "default": Decimal("5500"),
    },
}

# ---------------------------------------------------------------------------
# Fuel type key normalisation map
# ---------------------------------------------------------------------------

_FUEL_TYPE_KEY_MAP: Dict[str, str] = {
    "diesel": "diesel",
    "petrol": "petrol",
    "gasoline": "petrol",
    "jet_fuel": "jet_kerosene",
    "jet_kerosene": "jet_kerosene",
    "kerosene": "jet_kerosene",
    "heavy_fuel_oil": "hfo",
    "hfo": "hfo",
    "vlsfo": "vlsfo",
    "very_low_sulphur_fuel_oil": "vlsfo",
    "marine_gas_oil": "mgo",
    "mgo": "mgo",
    "marine_diesel_oil": "mgo",
    "lng": "lng_marine",
    "lng_marine": "lng_marine",
    "lng_road": "lng_road",
    "cng": "cng",
    "compressed_natural_gas": "cng",
    "electricity": "electricity",
    "electric": "electricity",
    "hydrogen": "hydrogen",
    "h2": "hydrogen",
    "biodiesel": "biodiesel_b100",
    "biodiesel_b20": "biodiesel_b20",
    "biodiesel_b100": "biodiesel_b100",
    "fame": "biodiesel_b100",
    "hvo": "hvo",
    "hydrotreated_vegetable_oil": "hvo",
    "methanol": "methanol",
    "sustainable_aviation_fuel": "jet_kerosene",
    "saf": "jet_kerosene",
}


# ============================================================================
# HELPER: deterministic SHA-256 hashing
# ============================================================================

def _decimal_default(obj: Any) -> Any:
    """JSON encoder default for Decimal and datetime."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def _hash_dict(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary with deterministic serialisation."""
    serialised = json.dumps(data, sort_keys=True, default=_decimal_default)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places with ROUND_HALF_UP."""
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _safe_decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely, returning 0 on failure."""
    if value is None:
        return _ZERO
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        logger.warning("Could not convert %r to Decimal; defaulting to 0", value)
        return _ZERO


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class FuelCalculationResult:
    """
    Result from a single fuel-based emissions calculation.

    All emissions values are in kgCO2e unless otherwise noted.

    Attributes:
        record_id: Source record identifier.
        fuel_type: Normalised fuel type key.
        fuel_quantity: Quantity consumed (in the fuel's native unit).
        fuel_unit: Native unit for this fuel (litre, kg, kWh).
        ttw_kgco2e: Tank-to-Wheel emissions.
        wtt_kgco2e: Well-to-Tank emissions.
        wtw_kgco2e: Well-to-Wheel total emissions.
        co2_kg: CO2 component of total.
        ch4_kg: CH4 component of total.
        n2o_kg: N2O component of total.
        biogenic_co2_kg: Biogenic CO2 (reported separately per GHG Protocol).
        methane_slip_kg: Methane slip for LNG/CNG engines.
        allocation_pct: Allocation percentage applied (0-100).
        allocated_kgco2e: Emissions after allocation.
        ef_source: Emission factor source description.
        ef_scope: Scope applied (TTW / WTT / WTW).
        data_quality_score: Data quality indicator (1-5).
        provenance_hash: SHA-256 audit hash.
        processing_time_ms: Calculation duration in milliseconds.
        warnings: List of validation warnings.
        calculation_timestamp: UTC timestamp of calculation.
    """

    record_id: str = ""
    fuel_type: str = ""
    fuel_quantity: Decimal = _ZERO
    fuel_unit: str = ""
    ttw_kgco2e: Decimal = _ZERO
    wtt_kgco2e: Decimal = _ZERO
    wtw_kgco2e: Decimal = _ZERO
    co2_kg: Decimal = _ZERO
    ch4_kg: Decimal = _ZERO
    n2o_kg: Decimal = _ZERO
    biogenic_co2_kg: Decimal = _ZERO
    methane_slip_kg: Decimal = _ZERO
    allocation_pct: Decimal = _HUNDRED
    allocated_kgco2e: Decimal = _ZERO
    ef_source: str = "IPCC 2006 / DEFRA 2023 / GLEC v3.0"
    ef_scope: str = "WTW"
    data_quality_score: Decimal = Decimal("2")
    provenance_hash: str = ""
    processing_time_ms: float = 0.0
    warnings: List[str] = dc_field(default_factory=list)
    calculation_timestamp: str = dc_field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with string Decimals for JSON safety."""
        return {
            "record_id": self.record_id,
            "fuel_type": self.fuel_type,
            "fuel_quantity": str(self.fuel_quantity),
            "fuel_unit": self.fuel_unit,
            "ttw_kgco2e": str(self.ttw_kgco2e),
            "wtt_kgco2e": str(self.wtt_kgco2e),
            "wtw_kgco2e": str(self.wtw_kgco2e),
            "co2_kg": str(self.co2_kg),
            "ch4_kg": str(self.ch4_kg),
            "n2o_kg": str(self.n2o_kg),
            "biogenic_co2_kg": str(self.biogenic_co2_kg),
            "methane_slip_kg": str(self.methane_slip_kg),
            "allocation_pct": str(self.allocation_pct),
            "allocated_kgco2e": str(self.allocated_kgco2e),
            "ef_source": self.ef_source,
            "ef_scope": self.ef_scope,
            "data_quality_score": str(self.data_quality_score),
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class BlendCalculationResult:
    """
    Result from a fuel blend calculation.

    Attributes:
        base_fuel: Base fossil fuel key.
        bio_fuel: Biofuel component key.
        blend_ratio: Biofuel fraction (0-1).
        total_quantity: Total fuel quantity in native unit.
        fossil_quantity: Fossil fuel component quantity.
        bio_quantity: Biofuel component quantity.
        fossil_emissions_kgco2e: Emissions from fossil component.
        bio_emissions_kgco2e: Emissions from biofuel component (WTT only).
        total_kgco2e: Total reportable emissions.
        biogenic_co2_kg: Biogenic CO2 from biofuel combustion.
        blend_saving_pct: Percentage saving vs 100% fossil.
        provenance_hash: SHA-256 audit hash.
    """

    base_fuel: str = ""
    bio_fuel: str = ""
    blend_ratio: Decimal = _ZERO
    total_quantity: Decimal = _ZERO
    fossil_quantity: Decimal = _ZERO
    bio_quantity: Decimal = _ZERO
    fossil_emissions_kgco2e: Decimal = _ZERO
    bio_emissions_kgco2e: Decimal = _ZERO
    total_kgco2e: Decimal = _ZERO
    biogenic_co2_kg: Decimal = _ZERO
    blend_saving_pct: Decimal = _ZERO
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_fuel": self.base_fuel,
            "bio_fuel": self.bio_fuel,
            "blend_ratio": str(self.blend_ratio),
            "total_quantity": str(self.total_quantity),
            "fossil_quantity": str(self.fossil_quantity),
            "bio_quantity": str(self.bio_quantity),
            "fossil_emissions_kgco2e": str(self.fossil_emissions_kgco2e),
            "bio_emissions_kgco2e": str(self.bio_emissions_kgco2e),
            "total_kgco2e": str(self.total_kgco2e),
            "biogenic_co2_kg": str(self.biogenic_co2_kg),
            "blend_saving_pct": str(self.blend_saving_pct),
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CarrierEmissionsResult:
    """
    Result from carrier-level emissions aggregation.

    Attributes:
        carrier_name: Carrier / operator name.
        total_kgco2e: Total emissions across all fuel records.
        total_tco2e: Total emissions in metric tonnes CO2e.
        by_fuel_type: Breakdown by fuel type.
        record_count: Number of fuel records processed.
        provenance_hash: SHA-256 audit hash.
    """

    carrier_name: str = ""
    total_kgco2e: Decimal = _ZERO
    total_tco2e: Decimal = _ZERO
    by_fuel_type: Dict[str, Decimal] = dc_field(default_factory=dict)
    record_count: int = 0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "carrier_name": self.carrier_name,
            "total_kgco2e": str(self.total_kgco2e),
            "total_tco2e": str(self.total_tco2e),
            "by_fuel_type": {k: str(v) for k, v in self.by_fuel_type.items()},
            "record_count": self.record_count,
            "provenance_hash": self.provenance_hash,
        }


# ============================================================================
# ENGINE CLASS
# ============================================================================


class FuelBasedCalculatorEngine:
    """
    Engine 3 of 7: Fuel-Based Emissions Calculator.

    Calculates GHG emissions from actual fuel consumption data for upstream
    transportation and distribution (GHG Protocol Scope 3 Category 4).

    Core formula:
        emissions = fuel_consumed x fuel_emission_factor

    Where:
        TTW = fuel x TTW_EF   (direct combustion)
        WTT = fuel x WTT_EF   (upstream fuel lifecycle)
        WTW = TTW + WTT        (complete Well-to-Wheel lifecycle)

    Supports 15 fuel types, unit conversions between litres/kg/gallons/m3/
    BTU/kWh/therm, biofuel blending with biogenic CO2 accounting, LNG/CNG
    methane slip calculation, electricity grid-specific EFs, hydrogen
    production pathway EFs, allocation to cargo, per-tkm intensity,
    batch processing, and aggregation.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Example:
        >>> engine = FuelBasedCalculatorEngine()
        >>> result = engine.calculate_wtw("diesel", Decimal("100"), "litre")
        >>> print(result)  # Decimal('330.00000000')

    Attributes:
        _calculations_count: Running count of calculations performed.
        _total_emissions: Running total of emissions calculated (kgCO2e).
        _lock: Thread-safety lock for mutable state.
    """

    def __init__(self) -> None:
        """Initialise FuelBasedCalculatorEngine with zero counters."""
        self._calculations_count: int = 0
        self._total_emissions: Decimal = _ZERO
        self._lock = threading.RLock()
        logger.info(
            "FuelBasedCalculatorEngine initialised (agent=%s, version=%s)",
            AGENT_ID,
            _VERSION,
        )

    # -----------------------------------------------------------------------
    # PROPERTY: engine statistics
    # -----------------------------------------------------------------------

    @property
    def calculations_count(self) -> int:
        """Return total number of calculations performed."""
        with self._lock:
            return self._calculations_count

    @property
    def total_emissions_kgco2e(self) -> Decimal:
        """Return running total of emissions calculated in kgCO2e."""
        with self._lock:
            return self._total_emissions

    def reset_counters(self) -> None:
        """Reset running counters to zero (testing utility)."""
        with self._lock:
            self._calculations_count = 0
            self._total_emissions = _ZERO

    # -----------------------------------------------------------------------
    # METHOD 1: calculate (primary entry point)
    # -----------------------------------------------------------------------

    def calculate(self, fuel_input: Any) -> FuelCalculationResult:
        """
        Calculate emissions from a fuel consumption record.

        This is the primary entry point for fuel-based calculations. Accepts
        either a FuelConsumptionInput model or a plain dictionary.

        Args:
            fuel_input: FuelConsumptionInput instance or dict with keys:
                - fuel_type (str): Fuel type identifier.
                - fuel_consumed_litres (Decimal, optional): Litres consumed.
                - fuel_consumed_kg (Decimal, optional): Kilograms consumed.
                - electricity_consumed_kwh (Decimal, optional): kWh consumed.
                - mode (str, optional): Transport mode.
                - allocation_percentage (Decimal, optional): 0-100.
                - ef_scope (str, optional): TTW / WTT / WTW.
                - record_id (str, optional): Record identifier.
                - carrier_name (str, optional): Carrier name.

        Returns:
            FuelCalculationResult with full emissions breakdown.

        Raises:
            ValueError: If no valid fuel quantity is provided.
            KeyError: If fuel_type is not recognised.
        """
        start_time = time.monotonic()

        # Normalise input to dict
        if hasattr(fuel_input, "model_dump"):
            data = fuel_input.model_dump()
        elif hasattr(fuel_input, "__dict__") and not isinstance(fuel_input, dict):
            data = fuel_input.__dict__
        elif isinstance(fuel_input, dict):
            data = fuel_input
        else:
            raise TypeError(
                f"fuel_input must be FuelConsumptionInput or dict, got {type(fuel_input)}"
            )

        # Extract fields
        raw_fuel_type = str(data.get("fuel_type", "")).lower().strip()
        fuel_key = self._normalise_fuel_key(raw_fuel_type)
        record_id = str(data.get("record_id", f"fc_{uuid4().hex[:12]}"))
        scope_str = str(data.get("ef_scope", "wtw")).lower().strip()
        alloc_pct = _safe_decimal(data.get("allocation_percentage", _HUNDRED))
        mode = str(data.get("mode", "road")).lower().strip()
        grid_region = str(data.get("grid_region", "global")).lower().strip()
        h2_pathway = str(data.get("hydrogen_pathway", "default")).lower().strip()
        engine_type = str(data.get("engine_type", "default")).lower().strip()

        # Determine quantity and unit
        quantity, unit = self._resolve_quantity(data, fuel_key)

        # Validate
        warnings = self.validate_fuel_input(data)

        # Calculate TTW, WTT, WTW
        ttw = self._calculate_ttw_internal(fuel_key, quantity, unit, grid_region, h2_pathway)
        wtt = self._calculate_wtt_internal(fuel_key, quantity, unit, grid_region, h2_pathway)
        wtw = _quantize(ttw + wtt)

        # Select scope result
        if scope_str == "ttw":
            primary_emissions = ttw
            ef_scope_label = "TTW"
        elif scope_str == "wtt":
            primary_emissions = wtt
            ef_scope_label = "WTT"
        else:
            primary_emissions = wtw
            ef_scope_label = "WTW"

        # Gas split
        gas_split = self.split_by_gas(primary_emissions, fuel_key)
        co2_kg = gas_split.get("co2", _ZERO)
        ch4_kg = gas_split.get("ch4", _ZERO)
        n2o_kg = gas_split.get("n2o", _ZERO)

        # Biogenic CO2
        biogenic_co2 = self.calculate_biogenic_co2(fuel_key, quantity)

        # Methane slip
        methane_slip = self.calculate_methane_slip(fuel_key, engine_type)
        methane_slip_total = _quantize(methane_slip * quantity) if fuel_key in _METHANE_SLIP_FACTORS else _ZERO

        # Allocation
        allocated = _quantize(primary_emissions * alloc_pct / _HUNDRED)

        # Data quality
        has_actual = quantity > _ZERO
        fuel_known = fuel_key in _FUEL_EF_TABLE
        dq_score = self.get_data_quality_score(has_actual, fuel_known)

        # Provenance hash
        provenance_data = {
            "record_id": record_id,
            "fuel_type": fuel_key,
            "quantity": str(quantity),
            "unit": unit,
            "ttw": str(ttw),
            "wtt": str(wtt),
            "wtw": str(wtw),
            "scope": ef_scope_label,
            "allocation_pct": str(alloc_pct),
            "allocated": str(allocated),
            "engine": ENGINE_NAME,
            "agent": AGENT_ID,
            "version": _VERSION,
        }
        provenance_hash = _hash_dict(provenance_data)

        processing_time = (time.monotonic() - start_time) * 1000.0

        # Update counters
        with self._lock:
            self._calculations_count += 1
            self._total_emissions += allocated

        result = FuelCalculationResult(
            record_id=record_id,
            fuel_type=fuel_key,
            fuel_quantity=quantity,
            fuel_unit=unit,
            ttw_kgco2e=ttw,
            wtt_kgco2e=wtt,
            wtw_kgco2e=wtw,
            co2_kg=co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            biogenic_co2_kg=biogenic_co2,
            methane_slip_kg=methane_slip_total,
            allocation_pct=alloc_pct,
            allocated_kgco2e=allocated,
            ef_source="IPCC 2006 / DEFRA 2023 / GLEC v3.0",
            ef_scope=ef_scope_label,
            data_quality_score=dq_score,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time,
            warnings=warnings,
        )

        logger.info(
            "Fuel-based calculation complete: record=%s fuel=%s qty=%s%s "
            "ttw=%s wtt=%s wtw=%s allocated=%s scope=%s time=%.2fms",
            record_id, fuel_key, quantity, unit,
            ttw, wtt, wtw, allocated, ef_scope_label, processing_time,
        )

        return result

    # -----------------------------------------------------------------------
    # METHOD 2: calculate_ttw
    # -----------------------------------------------------------------------

    def calculate_ttw(
        self,
        fuel_type: str,
        quantity: Union[Decimal, float, int, str],
        unit: str = "litre",
        grid_region: str = "global",
        h2_pathway: str = "default",
    ) -> Decimal:
        """
        Calculate Tank-to-Wheel (direct combustion) emissions.

        TTW represents the emissions released during fuel combustion in
        the vehicle engine. For electricity and hydrogen, TTW is zero.

        Args:
            fuel_type: Fuel type identifier (e.g. "diesel", "hfo", "lng_marine").
            quantity: Fuel quantity consumed.
            unit: Quantity unit ("litre", "kg", "kWh", "gallon_us", etc.).
            grid_region: Grid region for electricity (default "global").
            h2_pathway: Hydrogen production pathway (default "default").

        Returns:
            TTW emissions in kgCO2e.

        Raises:
            KeyError: If fuel_type is not recognised.
            ValueError: If quantity is negative.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        qty = _safe_decimal(quantity)
        self._validate_quantity(qty)
        converted = self._convert_to_native_unit(qty, unit, fuel_key)
        return self._calculate_ttw_internal(fuel_key, converted, self._native_unit(fuel_key), grid_region, h2_pathway)

    # -----------------------------------------------------------------------
    # METHOD 3: calculate_wtt
    # -----------------------------------------------------------------------

    def calculate_wtt(
        self,
        fuel_type: str,
        quantity: Union[Decimal, float, int, str],
        unit: str = "litre",
        grid_region: str = "global",
        h2_pathway: str = "default",
    ) -> Decimal:
        """
        Calculate Well-to-Tank (upstream fuel lifecycle) emissions.

        WTT covers extraction, refining, transport, and distribution of the
        fuel before it reaches the vehicle tank.

        Args:
            fuel_type: Fuel type identifier.
            quantity: Fuel quantity consumed.
            unit: Quantity unit.
            grid_region: Grid region for electricity.
            h2_pathway: Hydrogen production pathway.

        Returns:
            WTT emissions in kgCO2e.

        Raises:
            KeyError: If fuel_type is not recognised.
            ValueError: If quantity is negative.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        qty = _safe_decimal(quantity)
        self._validate_quantity(qty)
        converted = self._convert_to_native_unit(qty, unit, fuel_key)
        return self._calculate_wtt_internal(fuel_key, converted, self._native_unit(fuel_key), grid_region, h2_pathway)

    # -----------------------------------------------------------------------
    # METHOD 4: calculate_wtw
    # -----------------------------------------------------------------------

    def calculate_wtw(
        self,
        fuel_type: str,
        quantity: Union[Decimal, float, int, str],
        unit: str = "litre",
        grid_region: str = "global",
        h2_pathway: str = "default",
    ) -> Decimal:
        """
        Calculate Well-to-Wheel (complete lifecycle) emissions.

        WTW = TTW + WTT. This is the most complete accounting method and
        the default for GLEC Framework and ISO 14083.

        Args:
            fuel_type: Fuel type identifier.
            quantity: Fuel quantity consumed.
            unit: Quantity unit.
            grid_region: Grid region for electricity.
            h2_pathway: Hydrogen production pathway.

        Returns:
            WTW emissions in kgCO2e.

        Raises:
            KeyError: If fuel_type is not recognised.
            ValueError: If quantity is negative.
        """
        ttw = self.calculate_ttw(fuel_type, quantity, unit, grid_region, h2_pathway)
        wtt = self.calculate_wtt(fuel_type, quantity, unit, grid_region, h2_pathway)
        return _quantize(ttw + wtt)

    # -----------------------------------------------------------------------
    # METHOD 5: calculate_with_blend
    # -----------------------------------------------------------------------

    def calculate_with_blend(
        self,
        base_fuel: str,
        bio_fuel: str,
        blend_ratio: Union[Decimal, float],
        quantity: Union[Decimal, float, int, str],
        unit: str = "litre",
    ) -> BlendCalculationResult:
        """
        Calculate emissions for a fuel blend (e.g. B20 diesel/biodiesel).

        Splits the total fuel volume into fossil and biofuel fractions,
        calculates emissions separately, and reports biogenic CO2 from
        the biofuel component.

        Args:
            base_fuel: Fossil fuel key (e.g. "diesel").
            bio_fuel: Biofuel key (e.g. "biodiesel_b100", "hvo").
            blend_ratio: Biofuel fraction (0.0 to 1.0).
            quantity: Total fuel quantity.
            unit: Quantity unit.

        Returns:
            BlendCalculationResult with fossil/bio breakdown.

        Raises:
            ValueError: If blend_ratio not in [0, 1].
            KeyError: If fuel types not recognised.
        """
        base_key = self._normalise_fuel_key(str(base_fuel).lower().strip())
        bio_key = self._normalise_fuel_key(str(bio_fuel).lower().strip())
        ratio = _safe_decimal(blend_ratio)
        total_qty = _safe_decimal(quantity)

        if ratio < _ZERO or ratio > _ONE:
            raise ValueError(f"blend_ratio must be 0.0-1.0, got {ratio}")

        self._validate_quantity(total_qty)
        total_native = self._convert_to_native_unit(total_qty, unit, base_key)

        bio_qty = _quantize(total_native * ratio)
        fossil_qty = _quantize(total_native - bio_qty)

        base_unit = self._native_unit(base_key)
        bio_unit = self._native_unit(bio_key)

        # Fossil emissions (full WTW)
        fossil_ttw = self._calculate_ttw_internal(base_key, fossil_qty, base_unit)
        fossil_wtt = self._calculate_wtt_internal(base_key, fossil_qty, base_unit)
        fossil_total = _quantize(fossil_ttw + fossil_wtt)

        # Bio emissions (only WTT for biogenic fuels; TTW is biogenic CO2)
        bio_ttw = self._calculate_ttw_internal(bio_key, bio_qty, bio_unit)
        bio_wtt = self._calculate_wtt_internal(bio_key, bio_qty, bio_unit)
        bio_total = _quantize(bio_ttw + bio_wtt)

        total_emissions = _quantize(fossil_total + bio_total)

        # Biogenic CO2
        biogenic_co2 = self.calculate_biogenic_co2(bio_key, bio_qty)

        # Saving vs 100% fossil
        fossil_only_total = self.calculate_wtw(base_key, total_native, base_unit)
        saving_pct = _ZERO
        if fossil_only_total > _ZERO:
            saving_pct = _quantize(
                (_ONE - total_emissions / fossil_only_total) * _HUNDRED
            )

        provenance_data = {
            "base_fuel": base_key,
            "bio_fuel": bio_key,
            "blend_ratio": str(ratio),
            "total_quantity": str(total_native),
            "fossil_emissions": str(fossil_total),
            "bio_emissions": str(bio_total),
            "total_emissions": str(total_emissions),
            "engine": ENGINE_NAME,
        }

        return BlendCalculationResult(
            base_fuel=base_key,
            bio_fuel=bio_key,
            blend_ratio=ratio,
            total_quantity=total_native,
            fossil_quantity=fossil_qty,
            bio_quantity=bio_qty,
            fossil_emissions_kgco2e=fossil_total,
            bio_emissions_kgco2e=bio_total,
            total_kgco2e=total_emissions,
            biogenic_co2_kg=biogenic_co2,
            blend_saving_pct=saving_pct,
            provenance_hash=_hash_dict(provenance_data),
        )

    # -----------------------------------------------------------------------
    # METHOD 6: convert_fuel_units
    # -----------------------------------------------------------------------

    def convert_fuel_units(
        self,
        value: Union[Decimal, float, int, str],
        from_unit: str,
        to_unit: str,
        fuel_type: str,
    ) -> Decimal:
        """
        Convert fuel quantity between units.

        Supports: litres, kg, gallon_us, gallon_imp, m3, BTU, kWh, therm,
        metric_ton.

        Uses fuel-specific density for volumetric/mass conversions.

        Args:
            value: Quantity to convert.
            from_unit: Source unit.
            to_unit: Target unit.
            fuel_type: Fuel type (needed for density-based conversions).

        Returns:
            Converted quantity as Decimal.

        Raises:
            ValueError: If units are not supported or conversion is impossible.
        """
        val = _safe_decimal(value)
        if from_unit == to_unit:
            return val

        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())

        # Step 1: convert from_unit -> litres (for volume) or kg (for mass)
        val_in_litres = self._to_litres(val, from_unit, fuel_key)

        # Step 2: convert litres -> to_unit
        return self._from_litres(val_in_litres, to_unit, fuel_key)

    # -----------------------------------------------------------------------
    # METHOD 7: get_fuel_density
    # -----------------------------------------------------------------------

    def get_fuel_density(self, fuel_type: str) -> Decimal:
        """
        Get fuel density in kg per litre.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Density in kg/litre.

        Raises:
            KeyError: If fuel_type has no density data.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        density = _FUEL_DENSITY.get(fuel_key)
        if density is None:
            raise KeyError(
                f"No density data for fuel type '{fuel_key}'. "
                f"Available: {sorted(_FUEL_DENSITY.keys())}"
            )
        return density

    # -----------------------------------------------------------------------
    # METHOD 8: get_fuel_heating_value
    # -----------------------------------------------------------------------

    def get_fuel_heating_value(
        self,
        fuel_type: str,
        basis: str = "ncv",
    ) -> Decimal:
        """
        Get fuel heating value in MJ/kg.

        Args:
            fuel_type: Fuel type identifier.
            basis: "ncv" (Net Calorific Value) or "hhv" (Higher Heating Value).

        Returns:
            Heating value in MJ/kg.

        Raises:
            KeyError: If fuel_type or basis not found.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        hv = _FUEL_HEATING_VALUES.get(fuel_key)
        if hv is None:
            raise KeyError(
                f"No heating value data for fuel type '{fuel_key}'. "
                f"Available: {sorted(_FUEL_HEATING_VALUES.keys())}"
            )
        basis_lower = basis.lower().strip()
        if basis_lower not in hv:
            raise KeyError(
                f"Heating value basis '{basis}' not available for '{fuel_key}'. "
                f"Available: {sorted(hv.keys())}"
            )
        return hv[basis_lower]

    # -----------------------------------------------------------------------
    # METHOD 9: split_by_gas
    # -----------------------------------------------------------------------

    def split_by_gas(
        self,
        total_kgco2e: Union[Decimal, float, int],
        fuel_type: str,
    ) -> Dict[str, Decimal]:
        """
        Split total CO2e emissions into individual gases.

        Uses fuel-specific gas split ratios from IPCC 2006 and DEFRA 2023.

        Args:
            total_kgco2e: Total emissions in kgCO2e.
            fuel_type: Fuel type identifier.

        Returns:
            Dictionary with keys "co2", "ch4", "n2o" (all in kgCO2e).
        """
        total = _safe_decimal(total_kgco2e)
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        ratios = _GAS_SPLIT_RATIOS.get(fuel_key, {
            "co2": Decimal("0.995"),
            "ch4": Decimal("0.002"),
            "n2o": Decimal("0.003"),
        })
        return {
            "co2": _quantize(total * ratios["co2"]),
            "ch4": _quantize(total * ratios["ch4"]),
            "n2o": _quantize(total * ratios["n2o"]),
        }

    # -----------------------------------------------------------------------
    # METHOD 10: calculate_biogenic_fraction
    # -----------------------------------------------------------------------

    def calculate_biogenic_fraction(self, fuel_type: str) -> Decimal:
        """
        Calculate the biogenic fraction of a fuel (0.0 to 1.0).

        Biogenic fuels (B100, HVO) have fraction 1.0; blends like B20 have
        the fraction of the biofuel component; fossil fuels have 0.0.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Biogenic fraction (0.0 to 1.0).
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        ef_data = _FUEL_EF_TABLE.get(fuel_key, {})
        return _safe_decimal(ef_data.get("biogenic_fraction", _ZERO))

    # -----------------------------------------------------------------------
    # METHOD 11: calculate_biogenic_co2
    # -----------------------------------------------------------------------

    def calculate_biogenic_co2(
        self,
        fuel_type: str,
        quantity: Union[Decimal, float, int, str],
    ) -> Decimal:
        """
        Calculate biogenic CO2 from fuel combustion.

        Per GHG Protocol, biogenic CO2 is reported separately and not
        included in Scope 1/2/3 totals. This method calculates the
        biogenic CO2 released during combustion of biofuels.

        Args:
            fuel_type: Fuel type identifier.
            quantity: Fuel quantity in native unit.

        Returns:
            Biogenic CO2 in kg.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        qty = _safe_decimal(quantity)
        ef_data = _FUEL_EF_TABLE.get(fuel_key, {})

        biogenic_fraction = _safe_decimal(ef_data.get("biogenic_fraction", _ZERO))
        if biogenic_fraction <= _ZERO:
            return _ZERO

        biogenic_ef = _safe_decimal(ef_data.get("biogenic_co2_per_litre", _ZERO))
        if biogenic_ef > _ZERO:
            return _quantize(qty * biogenic_ef)

        # Fallback: estimate biogenic CO2 from a typical diesel combustion EF
        # scaled by the biogenic fraction
        diesel_ttw = Decimal("2.68")
        return _quantize(qty * diesel_ttw * biogenic_fraction)

    # -----------------------------------------------------------------------
    # METHOD 12: calculate_methane_slip
    # -----------------------------------------------------------------------

    def calculate_methane_slip(
        self,
        fuel_type: str,
        engine_type: str = "default",
    ) -> Decimal:
        """
        Calculate methane slip factor for LNG/CNG engines.

        Methane slip occurs when unburned methane escapes through the
        engine exhaust. Significant for LNG marine and CNG road vehicles.

        Args:
            fuel_type: Fuel type identifier ("lng_marine", "lng_road", "cng").
            engine_type: Engine technology type (e.g. "low_pressure_2_stroke",
                "hpdi", "spark_ignition"). Defaults to "default".

        Returns:
            Methane slip factor in kg CH4 per kg fuel.
            Returns Decimal("0") for non-gas fuels.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        slip_table = _METHANE_SLIP_FACTORS.get(fuel_key)
        if slip_table is None:
            return _ZERO
        engine_key = engine_type.lower().strip()
        return slip_table.get(engine_key, slip_table.get("default", _ZERO))

    # -----------------------------------------------------------------------
    # METHOD 13: allocate_to_cargo
    # -----------------------------------------------------------------------

    def allocate_to_cargo(
        self,
        total_emissions: Union[Decimal, float, int],
        allocation: Any,
    ) -> Decimal:
        """
        Allocate shared-vehicle emissions to this shipment's cargo.

        Supports AllocationConfig model or dict with allocation fields.
        Falls back to mass-based allocation if method-specific data
        is unavailable.

        Args:
            total_emissions: Total vehicle/vessel emissions in kgCO2e.
            allocation: AllocationConfig or dict with:
                - allocation_method: "mass", "volume", "teu", "revenue", etc.
                - shipment_mass_tonnes / total_capacity_tonnes (for mass).
                - shipment_volume_m3 / total_capacity_m3 (for volume).
                - Other method-specific shipment/total pairs.

        Returns:
            Allocated emissions in kgCO2e.

        Raises:
            ValueError: If required allocation data is missing.
        """
        total = _safe_decimal(total_emissions)

        if hasattr(allocation, "model_dump"):
            data = allocation.model_dump()
        elif isinstance(allocation, dict):
            data = allocation
        else:
            data = {}

        method = str(data.get("allocation_method", "mass")).lower().strip()

        share = self._compute_allocation_share(method, data)
        return _quantize(total * share)

    # -----------------------------------------------------------------------
    # METHOD 14: calculate_per_tkm
    # -----------------------------------------------------------------------

    def calculate_per_tkm(
        self,
        total_emissions: Union[Decimal, float, int],
        mass_tonnes: Union[Decimal, float, int],
        distance_km: Union[Decimal, float, int],
    ) -> Decimal:
        """
        Calculate emissions intensity per tonne-kilometre.

        Args:
            total_emissions: Total emissions in kgCO2e.
            mass_tonnes: Cargo mass in metric tonnes.
            distance_km: Transport distance in kilometres.

        Returns:
            Emissions intensity in kgCO2e per tonne-km.

        Raises:
            ValueError: If mass or distance is zero or negative.
        """
        total = _safe_decimal(total_emissions)
        mass = _safe_decimal(mass_tonnes)
        dist = _safe_decimal(distance_km)

        if mass <= _ZERO:
            raise ValueError(f"mass_tonnes must be > 0, got {mass}")
        if dist <= _ZERO:
            raise ValueError(f"distance_km must be > 0, got {dist}")

        tkm = _quantize(mass * dist)
        return _quantize(total / tkm)

    # -----------------------------------------------------------------------
    # METHOD 15: calculate_carrier_emissions
    # -----------------------------------------------------------------------

    def calculate_carrier_emissions(
        self,
        carrier_fuel_data: List[Any],
        carrier_name: str = "",
    ) -> CarrierEmissionsResult:
        """
        Calculate total emissions for a carrier from multiple fuel records.

        Aggregates emissions across all fuel types for a single carrier.

        Args:
            carrier_fuel_data: List of FuelConsumptionInput or dicts.
            carrier_name: Carrier / operator name.

        Returns:
            CarrierEmissionsResult with total and per-fuel breakdown.
        """
        total_kgco2e = _ZERO
        by_fuel: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        count = 0

        for record in carrier_fuel_data:
            result = self.calculate(record)
            total_kgco2e += result.allocated_kgco2e
            by_fuel[result.fuel_type] = _quantize(
                by_fuel[result.fuel_type] + result.allocated_kgco2e
            )
            count += 1

        total_kgco2e = _quantize(total_kgco2e)
        total_tco2e = _quantize(total_kgco2e / _THOUSAND)

        provenance_data = {
            "carrier_name": carrier_name,
            "total_kgco2e": str(total_kgco2e),
            "record_count": count,
            "by_fuel": {k: str(v) for k, v in by_fuel.items()},
            "engine": ENGINE_NAME,
        }

        return CarrierEmissionsResult(
            carrier_name=carrier_name,
            total_kgco2e=total_kgco2e,
            total_tco2e=total_tco2e,
            by_fuel_type=dict(by_fuel),
            record_count=count,
            provenance_hash=_hash_dict(provenance_data),
        )

    # -----------------------------------------------------------------------
    # METHOD 16: batch_calculate
    # -----------------------------------------------------------------------

    def batch_calculate(
        self,
        inputs: List[Any],
    ) -> List[FuelCalculationResult]:
        """
        Calculate emissions for a batch of fuel consumption records.

        Processes records sequentially with error isolation -- a failure
        in one record does not prevent processing of subsequent records.

        Args:
            inputs: List of FuelConsumptionInput or dicts.

        Returns:
            List of FuelCalculationResult (one per input).
            Failed records will have warnings populated and zero emissions.
        """
        results: List[FuelCalculationResult] = []
        for idx, record in enumerate(inputs):
            try:
                result = self.calculate(record)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch item %d failed: %s", idx, str(exc), exc_info=True
                )
                error_result = FuelCalculationResult(
                    record_id=str(getattr(record, "record_id", f"batch_{idx}")),
                    warnings=[f"Calculation failed: {str(exc)}"],
                )
                results.append(error_result)
        return results

    # -----------------------------------------------------------------------
    # METHOD 17: aggregate_by_fuel_type
    # -----------------------------------------------------------------------

    def aggregate_by_fuel_type(
        self,
        results: List[FuelCalculationResult],
    ) -> Dict[str, Decimal]:
        """
        Aggregate calculation results by fuel type.

        Args:
            results: List of FuelCalculationResult from batch_calculate.

        Returns:
            Dictionary mapping fuel_type to total kgCO2e.
        """
        aggregation: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        for r in results:
            aggregation[r.fuel_type] = _quantize(
                aggregation[r.fuel_type] + r.allocated_kgco2e
            )
        return dict(aggregation)

    # -----------------------------------------------------------------------
    # METHOD 18: get_data_quality_score
    # -----------------------------------------------------------------------

    def get_data_quality_score(
        self,
        has_actual_fuel_data: bool,
        fuel_type_known: bool,
    ) -> Decimal:
        """
        Assess data quality for fuel-based calculation.

        Per ISO 14083, DQI scores range from 1 (very good) to 5 (very poor).
        Fuel-based method with actual fuel data and known fuel type scores
        best (1 = very good).

        Args:
            has_actual_fuel_data: True if actual fuel consumption data exists.
            fuel_type_known: True if fuel type is specifically identified.

        Returns:
            DQI score as Decimal (1-5).
        """
        if has_actual_fuel_data and fuel_type_known:
            return Decimal("1")  # Very good: primary fuel data
        if has_actual_fuel_data and not fuel_type_known:
            return Decimal("2")  # Good: fuel data but generic type
        if not has_actual_fuel_data and fuel_type_known:
            return Decimal("3")  # Fair: estimated quantity, known type
        return Decimal("4")  # Poor: both estimated

    # -----------------------------------------------------------------------
    # METHOD 19: validate_fuel_input
    # -----------------------------------------------------------------------

    def validate_fuel_input(self, fuel_input: Any) -> List[str]:
        """
        Validate fuel input and return list of warnings.

        Non-fatal validation: returns warnings but does not raise exceptions.
        Checks for common data quality issues.

        Args:
            fuel_input: FuelConsumptionInput or dict.

        Returns:
            List of warning strings (empty if no issues).
        """
        if hasattr(fuel_input, "model_dump"):
            data = fuel_input.model_dump()
        elif isinstance(fuel_input, dict):
            data = fuel_input
        else:
            return ["Input is not a recognised type"]

        warnings: List[str] = []

        # Check fuel type
        raw_ft = str(data.get("fuel_type", "")).lower().strip()
        if not raw_ft:
            warnings.append("fuel_type is empty or missing")
        elif raw_ft not in _FUEL_TYPE_KEY_MAP and raw_ft not in _FUEL_EF_TABLE:
            warnings.append(f"fuel_type '{raw_ft}' is not in standard lookup table")

        # Check at least one quantity
        litres = _safe_decimal(data.get("fuel_consumed_litres"))
        kg = _safe_decimal(data.get("fuel_consumed_kg"))
        kwh = _safe_decimal(data.get("electricity_consumed_kwh"))
        if litres <= _ZERO and kg <= _ZERO and kwh <= _ZERO:
            warnings.append("No fuel quantity provided (litres, kg, or kWh)")

        # Unreasonably large quantities
        if litres > Decimal("1000000"):
            warnings.append(
                f"fuel_consumed_litres={litres} is unusually large (>1M litres)"
            )
        if kg > Decimal("1000000"):
            warnings.append(
                f"fuel_consumed_kg={kg} is unusually large (>1M kg)"
            )
        if kwh > Decimal("10000000"):
            warnings.append(
                f"electricity_consumed_kwh={kwh} is unusually large (>10M kWh)"
            )

        # Allocation out of range
        alloc = _safe_decimal(data.get("allocation_percentage", _HUNDRED))
        if alloc < _ZERO or alloc > _HUNDRED:
            warnings.append(
                f"allocation_percentage={alloc} is outside 0-100 range"
            )

        # Missing mode
        if not data.get("mode"):
            warnings.append("transport mode is missing (defaulting to road)")

        return warnings

    # -----------------------------------------------------------------------
    # METHOD 20: estimate_fuel_from_distance
    # -----------------------------------------------------------------------

    def estimate_fuel_from_distance(
        self,
        mode: str,
        vehicle_type: str = "default",
        distance_km: Union[Decimal, float, int, str] = 0,
    ) -> Decimal:
        """
        Estimate fuel consumption from distance using default efficiencies.

        This is a fallback method when only distance data is available.
        Uses average fuel efficiency values per vehicle/vessel type.

        Args:
            mode: Transport mode ("road", "rail", "maritime", "air").
            vehicle_type: Vehicle sub-type or "default".
            distance_km: Transport distance in kilometres.

        Returns:
            Estimated fuel consumption in litres (road/rail/air) or kg (maritime).
        """
        dist = _safe_decimal(distance_km)
        if dist <= _ZERO:
            return _ZERO

        mode_key = mode.lower().strip()
        veh_key = vehicle_type.lower().strip()

        efficiencies = _DEFAULT_FUEL_EFFICIENCY.get(mode_key, {})
        eff = efficiencies.get(veh_key, efficiencies.get("default", Decimal("30.0")))

        if mode_key == "maritime":
            # Maritime efficiency is in kg per voyage-day at ~14 knots
            # Approximate: fuel_kg = eff * (distance / (14 * 24 * 1.852))
            hours = dist / (Decimal("14") * Decimal("1.852"))
            days = hours / Decimal("24")
            return _quantize(eff * days)

        # Road/Rail/Air: efficiency is litres per 100 km
        return _quantize(eff * dist / _HUNDRED)

    # -----------------------------------------------------------------------
    # METHOD 21: get_supported_fuel_types
    # -----------------------------------------------------------------------

    def get_supported_fuel_types(self) -> List[str]:
        """
        Return list of all supported fuel type keys.

        Returns:
            Sorted list of canonical fuel type keys.
        """
        return sorted(_FUEL_EF_TABLE.keys())

    # -----------------------------------------------------------------------
    # METHOD 22: get_fuel_ef
    # -----------------------------------------------------------------------

    def get_fuel_ef(
        self,
        fuel_type: str,
        scope: str = "wtw",
    ) -> Decimal:
        """
        Get emission factor for a fuel type and scope.

        Args:
            fuel_type: Fuel type identifier.
            scope: "ttw", "wtt", or "wtw".

        Returns:
            Emission factor in kgCO2e per native unit.

        Raises:
            KeyError: If fuel_type not found.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        ef_data = _FUEL_EF_TABLE.get(fuel_key)
        if ef_data is None:
            raise KeyError(f"Unknown fuel type: '{fuel_key}'")
        scope_lower = scope.lower().strip()
        if scope_lower in ef_data:
            return _safe_decimal(ef_data[scope_lower])
        raise KeyError(
            f"Scope '{scope}' not available for fuel '{fuel_key}'. "
            f"Use grid_region/h2_pathway for electricity/hydrogen."
        )

    # -----------------------------------------------------------------------
    # METHOD 23: get_fuel_ef_all_scopes
    # -----------------------------------------------------------------------

    def get_fuel_ef_all_scopes(
        self,
        fuel_type: str,
    ) -> Dict[str, Decimal]:
        """
        Get emission factors for all scopes for a fuel type.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Dict with keys "ttw", "wtt", "wtw" (each in kgCO2e/unit).

        Raises:
            KeyError: If fuel_type not found.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        ef_data = _FUEL_EF_TABLE.get(fuel_key)
        if ef_data is None:
            raise KeyError(f"Unknown fuel type: '{fuel_key}'")
        ttw = _safe_decimal(ef_data.get("ttw", _ZERO))
        wtt = _safe_decimal(ef_data.get("wtt", _ZERO))
        wtw = _safe_decimal(ef_data.get("wtw", _ZERO))
        if wtw == _ZERO and (ttw > _ZERO or wtt > _ZERO):
            wtw = _quantize(ttw + wtt)
        return {"ttw": ttw, "wtt": wtt, "wtw": wtw}

    # -----------------------------------------------------------------------
    # METHOD 24: calculate_energy_content
    # -----------------------------------------------------------------------

    def calculate_energy_content(
        self,
        fuel_type: str,
        quantity: Union[Decimal, float, int, str],
        unit: str = "litre",
    ) -> Decimal:
        """
        Calculate energy content in MJ for a given fuel quantity.

        Uses NCV (Net Calorific Value) basis per IPCC convention.

        Args:
            fuel_type: Fuel type identifier.
            quantity: Fuel quantity.
            unit: Quantity unit.

        Returns:
            Energy content in MJ.
        """
        fuel_key = self._normalise_fuel_key(str(fuel_type).lower().strip())
        qty = _safe_decimal(quantity)

        # Convert to kg
        if unit.lower() in ("litre", "litres", "l"):
            density = self.get_fuel_density(fuel_key)
            qty_kg = _quantize(qty * density)
        elif unit.lower() in ("kg", "kilogram", "kilograms"):
            qty_kg = qty
        elif unit.lower() in ("kwh",):
            return _quantize(qty * _MJ_PER_KWH)
        else:
            # Convert through litres first
            qty_litres = self._to_litres(qty, unit, fuel_key)
            density = self.get_fuel_density(fuel_key)
            qty_kg = _quantize(qty_litres * density)

        ncv = self.get_fuel_heating_value(fuel_key, "ncv")
        return _quantize(qty_kg * ncv)

    # -----------------------------------------------------------------------
    # METHOD 25: summary
    # -----------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Return engine summary with configuration and statistics.

        Returns:
            Dict with engine metadata and runtime statistics.
        """
        with self._lock:
            return {
                "engine_name": ENGINE_NAME,
                "engine_number": ENGINE_NUMBER,
                "agent_id": AGENT_ID,
                "agent_component": AGENT_COMPONENT,
                "version": _VERSION,
                "supported_fuels": self.get_supported_fuel_types(),
                "supported_fuel_count": len(_FUEL_EF_TABLE),
                "calculations_count": self._calculations_count,
                "total_emissions_kgco2e": str(self._total_emissions),
                "total_emissions_tco2e": str(
                    _quantize(self._total_emissions / _THOUSAND)
                ),
            }

    # ======================================================================
    # INTERNAL METHODS
    # ======================================================================

    def _normalise_fuel_key(self, raw: str) -> str:
        """
        Normalise a raw fuel type string to canonical key.

        Args:
            raw: Raw fuel type string.

        Returns:
            Canonical fuel key.

        Raises:
            KeyError: If fuel type cannot be resolved.
        """
        if raw in _FUEL_EF_TABLE:
            return raw
        mapped = _FUEL_TYPE_KEY_MAP.get(raw)
        if mapped is not None:
            return mapped
        # Try stripping enum prefix (e.g. "TransportFuelType.DIESEL")
        if "." in raw:
            suffix = raw.split(".")[-1].lower()
            mapped = _FUEL_TYPE_KEY_MAP.get(suffix)
            if mapped:
                return mapped
        raise KeyError(
            f"Unknown fuel type: '{raw}'. Available: "
            f"{sorted(_FUEL_EF_TABLE.keys())}"
        )

    def _native_unit(self, fuel_key: str) -> str:
        """
        Get the native measurement unit for a fuel.

        Args:
            fuel_key: Canonical fuel key.

        Returns:
            "litre", "kg", or "kWh".
        """
        ef_data = _FUEL_EF_TABLE.get(fuel_key, {})
        return str(ef_data.get("unit", "litre"))

    def _resolve_quantity(
        self,
        data: Dict[str, Any],
        fuel_key: str,
    ) -> Tuple[Decimal, str]:
        """
        Resolve the fuel quantity and unit from input data.

        Priority: fuel_consumed_litres > fuel_consumed_kg > electricity_consumed_kwh.
        Converts to the fuel's native unit if needed.

        Args:
            data: Input data dictionary.
            fuel_key: Canonical fuel key.

        Returns:
            Tuple of (quantity, unit) in native fuel unit.

        Raises:
            ValueError: If no valid quantity is found.
        """
        native_unit = self._native_unit(fuel_key)

        litres = _safe_decimal(data.get("fuel_consumed_litres"))
        kg = _safe_decimal(data.get("fuel_consumed_kg"))
        kwh = _safe_decimal(data.get("electricity_consumed_kwh"))

        if fuel_key == "electricity":
            if kwh > _ZERO:
                return kwh, "kWh"
            if litres > _ZERO or kg > _ZERO:
                raise ValueError(
                    "Electricity fuel type requires electricity_consumed_kwh"
                )
            raise ValueError("No electricity quantity provided")

        if fuel_key == "hydrogen":
            if kg > _ZERO:
                return kg, "kg"
            if litres > _ZERO:
                logger.warning(
                    "Hydrogen typically measured in kg; converting litres via density"
                )
                density = _FUEL_DENSITY.get(fuel_key, Decimal("0.070"))
                return _quantize(litres * density), "kg"
            raise ValueError("No hydrogen quantity provided (use fuel_consumed_kg)")

        if native_unit == "kg":
            if kg > _ZERO:
                return kg, "kg"
            if litres > _ZERO:
                density = self.get_fuel_density(fuel_key)
                return _quantize(litres * density), "kg"
            raise ValueError(f"No quantity provided for {fuel_key} (need kg or litres)")

        # native_unit == "litre"
        if litres > _ZERO:
            return litres, "litre"
        if kg > _ZERO:
            density = self.get_fuel_density(fuel_key)
            if density > _ZERO:
                return _quantize(kg / density), "litre"
            raise ValueError(f"Cannot convert kg to litres: no density for {fuel_key}")

        raise ValueError(
            f"No valid fuel quantity for {fuel_key} "
            f"(litres={litres}, kg={kg}, kwh={kwh})"
        )

    def _validate_quantity(self, quantity: Decimal) -> None:
        """
        Validate that a fuel quantity is non-negative.

        Args:
            quantity: Fuel quantity.

        Raises:
            ValueError: If quantity is negative.
        """
        if quantity < _ZERO:
            raise ValueError(f"Fuel quantity must be >= 0, got {quantity}")

    def _calculate_ttw_internal(
        self,
        fuel_key: str,
        quantity: Decimal,
        unit: str,
        grid_region: str = "global",
        h2_pathway: str = "default",
    ) -> Decimal:
        """
        Internal TTW calculation.

        Args:
            fuel_key: Canonical fuel key.
            quantity: Fuel quantity in native unit.
            unit: Native unit.
            grid_region: Grid region for electricity.
            h2_pathway: Hydrogen production pathway.

        Returns:
            TTW emissions in kgCO2e.
        """
        ef_data = _FUEL_EF_TABLE.get(fuel_key)
        if ef_data is None:
            raise KeyError(f"No emission factors for fuel type: {fuel_key}")

        # Electricity and hydrogen have zero TTW
        if fuel_key in ("electricity", "hydrogen"):
            return _ZERO

        ttw_ef = _safe_decimal(ef_data.get("ttw", _ZERO))
        return _quantize(quantity * ttw_ef)

    def _calculate_wtt_internal(
        self,
        fuel_key: str,
        quantity: Decimal,
        unit: str,
        grid_region: str = "global",
        h2_pathway: str = "default",
    ) -> Decimal:
        """
        Internal WTT calculation.

        Handles special cases for electricity (grid-dependent) and
        hydrogen (production-pathway-dependent).

        Args:
            fuel_key: Canonical fuel key.
            quantity: Fuel quantity in native unit.
            unit: Native unit.
            grid_region: Grid region for electricity.
            h2_pathway: Hydrogen production pathway.

        Returns:
            WTT emissions in kgCO2e.
        """
        ef_data = _FUEL_EF_TABLE.get(fuel_key)
        if ef_data is None:
            raise KeyError(f"No emission factors for fuel type: {fuel_key}")

        if fuel_key == "electricity":
            region_key = f"wtt_{grid_region.lower().strip()}"
            wtt_ef = _safe_decimal(
                ef_data.get(region_key, ef_data.get("wtt_global", Decimal("0.475")))
            )
            return _quantize(quantity * wtt_ef)

        if fuel_key == "hydrogen":
            pathway_key = f"wtt_{h2_pathway.lower().strip()}"
            wtt_ef = _safe_decimal(
                ef_data.get(pathway_key, ef_data.get("wtt_default", Decimal("10.80")))
            )
            return _quantize(quantity * wtt_ef)

        wtt_ef = _safe_decimal(ef_data.get("wtt", _ZERO))
        return _quantize(quantity * wtt_ef)

    def _convert_to_native_unit(
        self,
        value: Decimal,
        from_unit: str,
        fuel_key: str,
    ) -> Decimal:
        """
        Convert a value from any supported unit to the fuel's native unit.

        Args:
            value: Quantity to convert.
            from_unit: Source unit.
            fuel_key: Canonical fuel key.

        Returns:
            Converted quantity in native unit.
        """
        native = self._native_unit(fuel_key)
        from_lower = from_unit.lower().strip()

        # Already native
        if from_lower in ("litre", "litres", "l") and native == "litre":
            return value
        if from_lower in ("kg", "kilogram", "kilograms") and native == "kg":
            return value
        if from_lower in ("kwh",) and native == "kWh":
            return value

        # Convert through litres as intermediate
        if native == "litre":
            return self._to_litres(value, from_lower, fuel_key)
        elif native == "kg":
            litres = self._to_litres(value, from_lower, fuel_key)
            density = self.get_fuel_density(fuel_key)
            return _quantize(litres * density)
        elif native == "kWh":
            if from_lower in ("kwh",):
                return value
            if from_lower in ("therm", "therms"):
                return _quantize(value * _KWH_PER_THERM)
            if from_lower in ("btu",):
                return _quantize(value / _BTU_PER_KWH)
            if from_lower in ("mj",):
                return _quantize(value / _MJ_PER_KWH)
            raise ValueError(f"Cannot convert '{from_unit}' to kWh")
        else:
            raise ValueError(f"Unknown native unit '{native}' for fuel '{fuel_key}'")

    def _to_litres(
        self,
        value: Decimal,
        from_unit: str,
        fuel_key: str,
    ) -> Decimal:
        """
        Convert a value to litres.

        Args:
            value: Quantity to convert.
            from_unit: Source unit (lowercase, stripped).
            fuel_key: Canonical fuel key.

        Returns:
            Value in litres.
        """
        u = from_unit.lower().strip()

        if u in ("litre", "litres", "l", "liter", "liters"):
            return value
        if u in ("gallon_us", "gallons_us", "us_gallon", "gal_us"):
            return _quantize(value * _LITRES_PER_US_GALLON)
        if u in ("gallon_imp", "gallons_imp", "imperial_gallon", "gal_imp"):
            return _quantize(value * _LITRES_PER_IMP_GALLON)
        if u in ("m3", "cubic_metre", "cubic_meter"):
            return _quantize(value * _LITRES_PER_CUBIC_METRE)
        if u in ("kg", "kilogram", "kilograms"):
            density = self.get_fuel_density(fuel_key)
            if density <= _ZERO:
                raise ValueError(f"Cannot convert kg to litres: density is 0 for {fuel_key}")
            return _quantize(value / density)
        if u in ("metric_ton", "tonne", "tonnes", "metric_tons"):
            density = self.get_fuel_density(fuel_key)
            if density <= _ZERO:
                raise ValueError(f"Cannot convert tonnes to litres for {fuel_key}")
            kg_val = _quantize(value * _KG_PER_METRIC_TON)
            return _quantize(kg_val / density)
        if u in ("btu",):
            kwh = _quantize(value / _BTU_PER_KWH)
            return self._kwh_to_litres(kwh, fuel_key)
        if u in ("kwh",):
            return self._kwh_to_litres(value, fuel_key)
        if u in ("therm", "therms"):
            kwh = _quantize(value * _KWH_PER_THERM)
            return self._kwh_to_litres(kwh, fuel_key)
        if u in ("mj",):
            kwh = _quantize(value / _MJ_PER_KWH)
            return self._kwh_to_litres(kwh, fuel_key)

        raise ValueError(
            f"Unsupported unit '{from_unit}'. Supported: litre, kg, gallon_us, "
            f"gallon_imp, m3, metric_ton, BTU, kWh, therm, MJ"
        )

    def _from_litres(
        self,
        litres: Decimal,
        to_unit: str,
        fuel_key: str,
    ) -> Decimal:
        """
        Convert a value from litres to target unit.

        Args:
            litres: Quantity in litres.
            to_unit: Target unit.
            fuel_key: Canonical fuel key.

        Returns:
            Converted value.
        """
        u = to_unit.lower().strip()

        if u in ("litre", "litres", "l", "liter", "liters"):
            return litres
        if u in ("gallon_us", "gallons_us", "us_gallon", "gal_us"):
            return _quantize(litres / _LITRES_PER_US_GALLON)
        if u in ("gallon_imp", "gallons_imp", "imperial_gallon", "gal_imp"):
            return _quantize(litres / _LITRES_PER_IMP_GALLON)
        if u in ("m3", "cubic_metre", "cubic_meter"):
            return _quantize(litres / _LITRES_PER_CUBIC_METRE)
        if u in ("kg", "kilogram", "kilograms"):
            density = self.get_fuel_density(fuel_key)
            return _quantize(litres * density)
        if u in ("metric_ton", "tonne", "tonnes", "metric_tons"):
            density = self.get_fuel_density(fuel_key)
            return _quantize(litres * density / _KG_PER_METRIC_TON)
        if u in ("kwh",):
            return self._litres_to_kwh(litres, fuel_key)
        if u in ("btu",):
            kwh = self._litres_to_kwh(litres, fuel_key)
            return _quantize(kwh * _BTU_PER_KWH)
        if u in ("therm", "therms"):
            kwh = self._litres_to_kwh(litres, fuel_key)
            return _quantize(kwh / _KWH_PER_THERM)
        if u in ("mj",):
            kwh = self._litres_to_kwh(litres, fuel_key)
            return _quantize(kwh * _MJ_PER_KWH)

        raise ValueError(
            f"Unsupported target unit '{to_unit}'. Supported: litre, kg, "
            f"gallon_us, gallon_imp, m3, metric_ton, BTU, kWh, therm, MJ"
        )

    def _kwh_to_litres(self, kwh: Decimal, fuel_key: str) -> Decimal:
        """
        Convert kWh energy to litres via heating value and density.

        energy_MJ = kWh * 3.6
        mass_kg = energy_MJ / NCV_MJ_per_kg
        litres = mass_kg / density_kg_per_litre

        Args:
            kwh: Energy in kWh.
            fuel_key: Canonical fuel key.

        Returns:
            Equivalent volume in litres.
        """
        hv_data = _FUEL_HEATING_VALUES.get(fuel_key)
        if hv_data is None:
            raise ValueError(f"Cannot convert kWh to litres: no heating value for {fuel_key}")
        ncv = hv_data["ncv"]
        density = self.get_fuel_density(fuel_key)
        if ncv <= _ZERO or density <= _ZERO:
            raise ValueError(f"Invalid NCV or density for {fuel_key}")
        energy_mj = _quantize(kwh * _MJ_PER_KWH)
        mass_kg = _quantize(energy_mj / ncv)
        return _quantize(mass_kg / density)

    def _litres_to_kwh(self, litres: Decimal, fuel_key: str) -> Decimal:
        """
        Convert litres to kWh energy via density and heating value.

        mass_kg = litres * density
        energy_MJ = mass_kg * NCV
        kWh = energy_MJ / 3.6

        Args:
            litres: Volume in litres.
            fuel_key: Canonical fuel key.

        Returns:
            Energy in kWh.
        """
        hv_data = _FUEL_HEATING_VALUES.get(fuel_key)
        if hv_data is None:
            raise ValueError(f"Cannot convert litres to kWh: no heating value for {fuel_key}")
        ncv = hv_data["ncv"]
        density = self.get_fuel_density(fuel_key)
        mass_kg = _quantize(litres * density)
        energy_mj = _quantize(mass_kg * ncv)
        return _quantize(energy_mj / _MJ_PER_KWH)

    def _compute_allocation_share(
        self,
        method: str,
        data: Dict[str, Any],
    ) -> Decimal:
        """
        Compute allocation share (0.0 to 1.0) based on method.

        Args:
            method: Allocation method string.
            data: Dictionary with shipment and total capacity values.

        Returns:
            Allocation share as Decimal.

        Raises:
            ValueError: If required capacity data is missing.
        """
        if method == "mass":
            ship = _safe_decimal(data.get("shipment_mass_tonnes"))
            total = _safe_decimal(data.get("total_capacity_tonnes"))
        elif method == "volume":
            ship = _safe_decimal(data.get("shipment_volume_m3"))
            total = _safe_decimal(data.get("total_capacity_m3"))
        elif method == "teu":
            ship = _safe_decimal(data.get("shipment_teu"))
            total = _safe_decimal(data.get("total_teu"))
        elif method == "revenue":
            ship = _safe_decimal(data.get("shipment_revenue"))
            total = _safe_decimal(data.get("total_revenue"))
        elif method == "pallet_positions":
            ship = _safe_decimal(data.get("shipment_pallet_positions"))
            total = _safe_decimal(data.get("total_pallet_positions"))
        elif method == "chargeable_weight":
            ship = _safe_decimal(data.get("shipment_chargeable_weight_kg"))
            total = _safe_decimal(data.get("total_chargeable_weight_kg"))
        elif method == "floor_area":
            ship = _safe_decimal(data.get("shipment_floor_area_m2"))
            total = _safe_decimal(data.get("total_floor_area_m2"))
        else:
            logger.warning(
                "Unknown allocation method '%s'; defaulting to mass", method
            )
            ship = _safe_decimal(data.get("shipment_mass_tonnes"))
            total = _safe_decimal(data.get("total_capacity_tonnes"))

        if total <= _ZERO:
            logger.warning(
                "Total capacity is zero for %s allocation; returning 1.0", method
            )
            return _ONE

        if ship <= _ZERO:
            raise ValueError(
                f"Shipment quantity is zero or missing for {method} allocation"
            )

        share = _quantize(ship / total)
        if share > _ONE:
            logger.warning(
                "Allocation share %.4f > 1.0 for %s method; capping at 1.0",
                share, method,
            )
            return _ONE
        return share


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================


_engine_instance: Optional[FuelBasedCalculatorEngine] = None
_engine_lock = threading.RLock()


def get_fuel_calculator() -> FuelBasedCalculatorEngine:
    """
    Get singleton FuelBasedCalculatorEngine instance.

    Thread-safe lazy initialisation.

    Returns:
        FuelBasedCalculatorEngine singleton.
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = FuelBasedCalculatorEngine()
    return _engine_instance


def reset_fuel_calculator() -> None:
    """
    Reset the singleton engine instance (testing utility).
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None


def calculate_fuel_emissions(fuel_input: Any) -> FuelCalculationResult:
    """
    Convenience function: calculate emissions from fuel consumption.

    Args:
        fuel_input: FuelConsumptionInput or dict.

    Returns:
        FuelCalculationResult.
    """
    return get_fuel_calculator().calculate(fuel_input)


def batch_calculate_fuel_emissions(
    inputs: List[Any],
) -> List[FuelCalculationResult]:
    """
    Convenience function: batch calculate emissions.

    Args:
        inputs: List of FuelConsumptionInput or dicts.

    Returns:
        List of FuelCalculationResult.
    """
    return get_fuel_calculator().batch_calculate(inputs)
