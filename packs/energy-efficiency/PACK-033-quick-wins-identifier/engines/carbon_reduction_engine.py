# -*- coding: utf-8 -*-
"""
CarbonReductionEngine - PACK-033 Quick Wins Identifier Engine 4
================================================================

Calculates tCO2e reductions from energy savings quick wins per the GHG
Protocol Corporate Standard, with full Scope 1 / Scope 2 / Scope 3
attribution, dual location-based and market-based Scope 2 methods, and
SBTi alignment assessment against Well-Below-2C, 1.5C, and Net-Zero
ambition levels.

Calculation Methodology:
    Electricity CO2e (location-based):
        co2e_kg = electricity_savings_kwh * grid_emission_factor_kg_per_kwh

    Electricity CO2e (market-based):
        co2e_kg = electricity_savings_kwh * residual_mix_factor_kg_per_kwh
        (or contractual instrument factor where applicable)

    Natural Gas CO2e:
        co2e_kg = gas_savings_therms * gas_emission_factor_kg_per_therm

    Other Fuel CO2e:
        co2e_kg = fuel_savings_units * fuel_emission_factor_kg_per_unit

    Grid Decarbonization Adjustment:
        adjusted_factor(t) = base_factor * (1 - decarbonization_rate) ^ t

    Cumulative Reduction:
        cumulative = sum(annual_co2e(t) for t in range(1, years + 1))

    SBTi Alignment:
        required_pct = annual_rate * (target_year - base_year) * 100
        achieved_pct = (base_emissions - current_emissions + reductions)
                       / base_emissions * 100
        on_track = achieved_pct >= required_pct

Regulatory References:
    - GHG Protocol Corporate Standard (WRI/WBCSD, 2015 rev.)
    - GHG Protocol Scope 2 Guidance (WRI, 2015)
    - GHG Protocol Corporate Value Chain (Scope 3) Standard
    - SBTi Corporate Net-Zero Standard v1.1 (October 2023)
    - SBTi Target Validation Protocol v3.0
    - EPA eGRID 2023 (US grid emission factors)
    - DEFRA 2024 Government Conversion Factors (UK)
    - IEA Emission Factors 2024 (international grids)
    - ISO 14064-1:2018 - GHG quantification and reporting

Zero-Hallucination:
    - All emission factors sourced from EPA eGRID 2023, DEFRA 2024, IEA 2024
    - SBTi reduction rates from published SBTi standards
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-033 Quick Wins Identifier
Engine:  4 of 8
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EmissionScope(str, Enum):
    """GHG Protocol emission scope classification.

    SCOPE_1: Direct emissions from owned or controlled sources.
    SCOPE_2: Indirect emissions from purchased electricity, steam, heat, cooling.
    SCOPE_3: All other indirect emissions in the value chain.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class CalculationMethod(str, Enum):
    """Scope 2 calculation method per GHG Protocol Scope 2 Guidance.

    LOCATION_BASED: Reflects average grid emission intensity.
    MARKET_BASED: Reflects contractual instruments and residual mix.
    """
    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"


class EmissionFactorType(str, Enum):
    """Type of emission factor applied.

    AVERAGE: Average grid or fuel emission factor.
    MARGINAL: Marginal grid emission factor (short-run or long-run).
    RESIDUAL_MIX: Residual mix factor for market-based Scope 2.
    """
    AVERAGE = "average"
    MARGINAL = "marginal"
    RESIDUAL_MIX = "residual_mix"


class FuelType(str, Enum):
    """Common fuel types for Scope 1 combustion calculations.

    Emission factors sourced from EPA AP-42, DEFRA 2024, IPCC 2006.
    """
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    DIESEL = "diesel"
    COAL = "coal"
    BIOMASS = "biomass"
    BIOGAS = "biogas"


class GridRegion(str, Enum):
    """Grid region for electricity emission factors.

    US regions from EPA eGRID 2023 NERC sub-regions.
    EU/international from IEA 2024 and DEFRA 2024.
    """
    US_NATIONAL = "us_national"
    US_NERC_NPCC = "us_nerc_npcc"
    US_NERC_RFC = "us_nerc_rfc"
    US_NERC_SERC = "us_nerc_serc"
    US_NERC_TRE = "us_nerc_tre"
    US_NERC_WECC = "us_nerc_wecc"
    US_NERC_MRO = "us_nerc_mro"
    EU_AVERAGE = "eu_average"
    UK = "uk"
    DE = "de"
    FR = "fr"
    IT = "it"
    ES = "es"
    NL = "nl"
    SE = "se"
    NO = "no"
    DK = "dk"
    FI = "fi"
    PL = "pl"
    AT = "at"
    BE = "be"
    IE = "ie"
    PT = "pt"
    CZ = "cz"
    AU = "au"
    JP = "jp"
    CN = "cn"
    IN = "in_"
    BR = "br"
    CA = "ca"
    KR = "kr"
    ZA = "za"
    MX = "mx"
    CUSTOM = "custom"


class SBTiAmbition(str, Enum):
    """SBTi target ambition level.

    WELL_BELOW_2C: Well-below 2 degrees Celsius pathway.
    ONE_POINT_FIVE_C: 1.5 degrees Celsius pathway (SBTi minimum for new targets).
    NET_ZERO: Net-Zero standard (4.2% near-term + 90% long-term reduction).
    """
    WELL_BELOW_2C = "well_below_2c"
    ONE_POINT_FIVE_C = "one_point_five_c"
    NET_ZERO = "net_zero"


class ProjectionMethod(str, Enum):
    """Method for projecting future emission reductions.

    LINEAR: Fixed annual reduction applied each year.
    COMPOUND: Compound reduction with grid decarbonization adjustment.
    SDA_PATHWAY: Sectoral Decarbonization Approach pathway.
    """
    LINEAR = "linear"
    COMPOUND = "compound"
    SDA_PATHWAY = "sda_pathway"


# ---------------------------------------------------------------------------
# Constants -- Emission Factor Database
# ---------------------------------------------------------------------------


# Grid electricity emission factors by region (kgCO2e per kWh).
# Sources: EPA eGRID 2023, DEFRA 2024, IEA Emission Factors 2024.
GRID_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    GridRegion.US_NATIONAL.value: {
        "factor": Decimal("0.3856"),
        "source": "EPA eGRID 2023",
        "year": 2023,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.US_NERC_NPCC.value: {
        "factor": Decimal("0.2280"),
        "source": "EPA eGRID 2023 (NPCC)",
        "year": 2023,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.US_NERC_RFC.value: {
        "factor": Decimal("0.4138"),
        "source": "EPA eGRID 2023 (RFC)",
        "year": 2023,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.US_NERC_SERC.value: {
        "factor": Decimal("0.3844"),
        "source": "EPA eGRID 2023 (SERC)",
        "year": 2023,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.US_NERC_TRE.value: {
        "factor": Decimal("0.3628"),
        "source": "EPA eGRID 2023 (TRE/ERCOT)",
        "year": 2023,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.US_NERC_WECC.value: {
        "factor": Decimal("0.2946"),
        "source": "EPA eGRID 2023 (WECC)",
        "year": 2023,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.US_NERC_MRO.value: {
        "factor": Decimal("0.4435"),
        "source": "EPA eGRID 2023 (MRO)",
        "year": 2023,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.EU_AVERAGE.value: {
        "factor": Decimal("0.2560"),
        "source": "IEA 2024 (EU-27 average)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.UK.value: {
        "factor": Decimal("0.2072"),
        "source": "DEFRA 2024 (UK grid average)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.DE.value: {
        "factor": Decimal("0.3500"),
        "source": "IEA 2024 (Germany)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.FR.value: {
        "factor": Decimal("0.0520"),
        "source": "IEA 2024 (France - nuclear dominated)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.IT.value: {
        "factor": Decimal("0.3150"),
        "source": "IEA 2024 (Italy)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.ES.value: {
        "factor": Decimal("0.1880"),
        "source": "IEA 2024 (Spain)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.NL.value: {
        "factor": Decimal("0.3280"),
        "source": "IEA 2024 (Netherlands)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.SE.value: {
        "factor": Decimal("0.0130"),
        "source": "IEA 2024 (Sweden - hydro/nuclear)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.NO.value: {
        "factor": Decimal("0.0080"),
        "source": "IEA 2024 (Norway - hydropower)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.DK.value: {
        "factor": Decimal("0.1360"),
        "source": "IEA 2024 (Denmark - high wind)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.FI.value: {
        "factor": Decimal("0.0820"),
        "source": "IEA 2024 (Finland)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.PL.value: {
        "factor": Decimal("0.6630"),
        "source": "IEA 2024 (Poland - coal heavy)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.AT.value: {
        "factor": Decimal("0.0930"),
        "source": "IEA 2024 (Austria)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.BE.value: {
        "factor": Decimal("0.1530"),
        "source": "IEA 2024 (Belgium)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.IE.value: {
        "factor": Decimal("0.2960"),
        "source": "IEA 2024 (Ireland)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.PT.value: {
        "factor": Decimal("0.1740"),
        "source": "IEA 2024 (Portugal)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.CZ.value: {
        "factor": Decimal("0.4280"),
        "source": "IEA 2024 (Czech Republic)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.AU.value: {
        "factor": Decimal("0.6560"),
        "source": "IEA 2024 (Australia)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.JP.value: {
        "factor": Decimal("0.4570"),
        "source": "IEA 2024 (Japan)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.CN.value: {
        "factor": Decimal("0.5810"),
        "source": "IEA 2024 (China)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.IN.value: {
        "factor": Decimal("0.7080"),
        "source": "IEA 2024 (India)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.BR.value: {
        "factor": Decimal("0.0740"),
        "source": "IEA 2024 (Brazil - hydro dominated)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.CA.value: {
        "factor": Decimal("0.1200"),
        "source": "IEA 2024 (Canada)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.KR.value: {
        "factor": Decimal("0.4590"),
        "source": "IEA 2024 (South Korea)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.ZA.value: {
        "factor": Decimal("0.9280"),
        "source": "IEA 2024 (South Africa - coal dominated)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
    GridRegion.MX.value: {
        "factor": Decimal("0.4310"),
        "source": "IEA 2024 (Mexico)",
        "year": 2024,
        "unit": "kgCO2e/kWh",
    },
}

# Residual mix factors for market-based Scope 2 (kgCO2e per kWh).
# Used when no contractual instrument (REC/GO) is available.
RESIDUAL_MIX_FACTORS: Dict[str, Decimal] = {
    GridRegion.US_NATIONAL.value: Decimal("0.4120"),
    GridRegion.US_NERC_NPCC.value: Decimal("0.2510"),
    GridRegion.US_NERC_RFC.value: Decimal("0.4380"),
    GridRegion.US_NERC_SERC.value: Decimal("0.4090"),
    GridRegion.US_NERC_TRE.value: Decimal("0.3870"),
    GridRegion.US_NERC_WECC.value: Decimal("0.3180"),
    GridRegion.US_NERC_MRO.value: Decimal("0.4710"),
    GridRegion.EU_AVERAGE.value: Decimal("0.3720"),
    GridRegion.UK.value: Decimal("0.3120"),
    GridRegion.DE.value: Decimal("0.4690"),
    GridRegion.FR.value: Decimal("0.0580"),
    GridRegion.IT.value: Decimal("0.4570"),
    GridRegion.ES.value: Decimal("0.2960"),
    GridRegion.NL.value: Decimal("0.4450"),
    GridRegion.SE.value: Decimal("0.0190"),
    GridRegion.NO.value: Decimal("0.0110"),
    GridRegion.DK.value: Decimal("0.2080"),
    GridRegion.FI.value: Decimal("0.1200"),
    GridRegion.PL.value: Decimal("0.7240"),
    GridRegion.AT.value: Decimal("0.1320"),
    GridRegion.BE.value: Decimal("0.2210"),
    GridRegion.IE.value: Decimal("0.3880"),
    GridRegion.PT.value: Decimal("0.2480"),
    GridRegion.CZ.value: Decimal("0.5310"),
    GridRegion.AU.value: Decimal("0.7020"),
    GridRegion.JP.value: Decimal("0.4870"),
    GridRegion.CN.value: Decimal("0.6150"),
    GridRegion.IN.value: Decimal("0.7390"),
    GridRegion.BR.value: Decimal("0.0950"),
    GridRegion.CA.value: Decimal("0.1540"),
    GridRegion.KR.value: Decimal("0.4820"),
    GridRegion.ZA.value: Decimal("0.9510"),
    GridRegion.MX.value: Decimal("0.4580"),
}

# Fuel combustion emission factors (kgCO2e per unit).
# Natural gas: kgCO2e per therm (1 therm = 100,000 BTU = 29.3 kWh).
# Liquid fuels: kgCO2e per gallon (US gallon = 3.785 litres).
# Coal: kgCO2e per short ton (907.2 kg).
# Biomass / biogas: biogenic CO2 reported separately per GHG Protocol.
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    FuelType.NATURAL_GAS.value: {
        "factor": Decimal("5.3000"),
        "unit": "kgCO2e/therm",
        "source": "EPA GHG Emission Factors Hub 2023",
        "year": 2023,
    },
    FuelType.PROPANE.value: {
        "factor": Decimal("5.7200"),
        "unit": "kgCO2e/gallon",
        "source": "EPA GHG Emission Factors Hub 2023",
        "year": 2023,
    },
    FuelType.FUEL_OIL_2.value: {
        "factor": Decimal("10.1600"),
        "unit": "kgCO2e/gallon",
        "source": "EPA GHG Emission Factors Hub 2023",
        "year": 2023,
    },
    FuelType.FUEL_OIL_6.value: {
        "factor": Decimal("11.2700"),
        "unit": "kgCO2e/gallon",
        "source": "EPA GHG Emission Factors Hub 2023",
        "year": 2023,
    },
    FuelType.DIESEL.value: {
        "factor": Decimal("10.2100"),
        "unit": "kgCO2e/gallon",
        "source": "EPA GHG Emission Factors Hub 2023",
        "year": 2023,
    },
    FuelType.COAL.value: {
        "factor": Decimal("2268.0000"),
        "unit": "kgCO2e/short_ton",
        "source": "EPA GHG Emission Factors Hub 2023",
        "year": 2023,
    },
    FuelType.BIOMASS.value: {
        "factor": Decimal("0.0000"),
        "unit": "kgCO2e/kWh",
        "source": "GHG Protocol (biogenic CO2 reported separately)",
        "year": 2023,
    },
    FuelType.BIOGAS.value: {
        "factor": Decimal("0.0000"),
        "unit": "kgCO2e/kWh",
        "source": "GHG Protocol (biogenic CO2 reported separately)",
        "year": 2023,
    },
}

# SBTi target required annual reduction rates.
# Sources: SBTi Corporate Net-Zero Standard v1.1, SBTi Criteria v5.1.
SBTI_TARGETS: Dict[str, Dict[str, Any]] = {
    SBTiAmbition.WELL_BELOW_2C.value: {
        "annual_rate": Decimal("0.025"),
        "description": "Minimum 2.5% annual linear reduction (Scope 1+2)",
        "minimum_coverage_pct": Decimal("95"),
        "timeframe_years": 10,
    },
    SBTiAmbition.ONE_POINT_FIVE_C.value: {
        "annual_rate": Decimal("0.042"),
        "description": "Minimum 4.2% annual linear reduction (Scope 1+2)",
        "minimum_coverage_pct": Decimal("95"),
        "timeframe_years": 10,
    },
    SBTiAmbition.NET_ZERO.value: {
        "annual_rate": Decimal("0.042"),
        "description": "4.2% near-term + 90% long-term absolute reduction",
        "minimum_coverage_pct": Decimal("95"),
        "timeframe_years": 10,
        "long_term_reduction_pct": Decimal("90"),
        "long_term_timeframe_years": 30,
    },
}

# Default grid decarbonization rate (2% per year global average trend).
DEFAULT_GRID_DECARBONIZATION_RATE: Decimal = Decimal("0.02")

# Default projection horizon (years).
DEFAULT_PROJECTION_YEARS: int = 10


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class EmissionFactor(BaseModel):
    """Emission factor with full provenance metadata.

    Attributes:
        region: Grid region for electricity factors.
        fuel_type: Fuel type for combustion factors (None for electricity).
        factor_value: Emission factor value (kgCO2e per unit).
        factor_type: Type of factor (average, marginal, residual_mix).
        source: Published data source.
        year: Reference year for the factor.
        unit: Unit of measurement (e.g. kgCO2e/kWh, kgCO2e/therm).
    """
    region: GridRegion = Field(
        default=GridRegion.US_NATIONAL,
        description="Grid region"
    )
    fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Fuel type (None for electricity)"
    )
    factor_value: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Emission factor (kgCO2e per unit)"
    )
    factor_type: EmissionFactorType = Field(
        default=EmissionFactorType.AVERAGE,
        description="Factor type"
    )
    source: str = Field(
        default="",
        description="Published data source"
    )
    year: int = Field(
        default=2024, ge=2000, le=2030,
        description="Reference year"
    )
    unit: str = Field(
        default="kgCO2e/kWh",
        description="Unit of measurement"
    )


class EnergyReduction(BaseModel):
    """Energy reduction data for a single quick-win measure.

    All energy savings are expressed in their native units:
    electricity in kWh, gas in therms, other fuels in their standard unit.

    Attributes:
        measure_id: Unique measure identifier.
        name: Measure name / description.
        electricity_savings_kwh: Annual electricity savings (kWh).
        gas_savings_therms: Annual natural gas savings (therms).
        other_fuel_savings: Annual other fuel savings (in fuel unit).
        other_fuel_type: Fuel type for other fuel savings.
        scope: GHG Protocol emission scope.
    """
    measure_id: str = Field(
        default_factory=_new_uuid,
        description="Unique measure identifier"
    )
    name: str = Field(
        default="",
        max_length=500,
        description="Measure name"
    )
    electricity_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual electricity savings (kWh)"
    )
    gas_savings_therms: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual natural gas savings (therms)"
    )
    other_fuel_savings: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual other fuel savings (fuel-specific units)"
    )
    other_fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Fuel type for other fuel savings"
    )
    scope: EmissionScope = Field(
        default=EmissionScope.SCOPE_2,
        description="GHG Protocol emission scope"
    )

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: EmissionScope) -> EmissionScope:
        """Validate scope is a recognised GHG Protocol scope."""
        valid = {s for s in EmissionScope}
        if v not in valid:
            raise ValueError(
                f"Unknown scope '{v}'. Must be one of: {sorted(s.value for s in valid)}"
            )
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class AnnualProjection(BaseModel):
    """Projected CO2e reduction for a single future year.

    Attributes:
        year: Calendar year of projection.
        projected_co2e_tonnes: Projected annual reduction (tCO2e).
        cumulative_co2e_tonnes: Cumulative reduction through this year (tCO2e).
        target_co2e_tonnes: SBTi target reduction for this year (tCO2e), if set.
    """
    year: int = Field(default=0, description="Calendar year")
    projected_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Projected annual reduction (tCO2e)"
    )
    cumulative_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Cumulative reduction through this year (tCO2e)"
    )
    target_co2e_tonnes: Optional[Decimal] = Field(
        default=None,
        description="SBTi target for this year (tCO2e)"
    )


class SBTiAssessment(BaseModel):
    """SBTi alignment assessment result.

    Evaluates whether the portfolio of quick-win carbon reductions
    contributes meaningfully to SBTi-aligned decarbonization targets.

    Attributes:
        base_year: SBTi base year.
        target_year: SBTi target year.
        base_year_emissions: Total base year emissions (tCO2e).
        required_reduction_pct: Required reduction percentage by target year.
        achieved_reduction_pct: Achieved or projected reduction percentage.
        on_track: Whether reductions are on track vs SBTi pathway.
        ambition_level: SBTi ambition level assessed against.
        gap_tonnes: Gap between required and achieved reductions (tCO2e).
        notes: Assessment notes and caveats.
    """
    base_year: int = Field(default=2020, description="SBTi base year")
    target_year: int = Field(default=2030, description="SBTi target year")
    base_year_emissions: Decimal = Field(
        default=Decimal("0"),
        description="Base year emissions (tCO2e)"
    )
    required_reduction_pct: Decimal = Field(
        default=Decimal("0"),
        description="Required reduction (%)"
    )
    achieved_reduction_pct: Decimal = Field(
        default=Decimal("0"),
        description="Achieved reduction (%)"
    )
    on_track: bool = Field(
        default=False,
        description="On track vs SBTi pathway"
    )
    ambition_level: SBTiAmbition = Field(
        default=SBTiAmbition.ONE_POINT_FIVE_C,
        description="SBTi ambition level"
    )
    gap_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Gap (tCO2e, positive = shortfall)"
    )
    notes: str = Field(
        default="",
        description="Assessment notes"
    )


class CarbonReductionResult(BaseModel):
    """Carbon reduction result for a single measure.

    Attributes:
        measure_id: Measure identifier.
        name: Measure name.
        scope: GHG Protocol emission scope.
        calculation_method: Location-based or market-based.
        annual_co2e_kg: Annual CO2e reduction (kg).
        annual_co2e_tonnes: Annual CO2e reduction (tonnes = kg / 1000).
        cumulative_co2e_tonnes: Cumulative CO2e over projection period (tonnes).
        projection_years: Number of years projected.
        emission_factor_used: Emission factor applied with provenance.
        methodology_notes: Calculation methodology description.
        calculated_at: Calculation timestamp (UTC).
        provenance_hash: SHA-256 audit hash.
    """
    measure_id: str = Field(default="", description="Measure identifier")
    name: str = Field(default="", description="Measure name")
    scope: EmissionScope = Field(
        default=EmissionScope.SCOPE_2,
        description="Emission scope"
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.LOCATION_BASED,
        description="Scope 2 calculation method"
    )
    annual_co2e_kg: Decimal = Field(
        default=Decimal("0"),
        description="Annual CO2e reduction (kg)"
    )
    annual_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Annual CO2e reduction (tonnes)"
    )
    cumulative_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Cumulative CO2e over projection period (tonnes)"
    )
    projection_years: int = Field(
        default=DEFAULT_PROJECTION_YEARS,
        description="Projection horizon (years)"
    )
    emission_factor_used: EmissionFactor = Field(
        default_factory=EmissionFactor,
        description="Emission factor applied"
    )
    methodology_notes: str = Field(
        default="",
        description="Calculation methodology"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Calculation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )


class PortfolioReduction(BaseModel):
    """Aggregated carbon reduction for an entire quick-wins portfolio.

    Attributes:
        portfolio_id: Unique portfolio identifier.
        measures: List of per-measure carbon reduction results.
        total_annual_co2e_tonnes: Total annual reduction (tCO2e).
        total_cumulative_co2e_tonnes: Total cumulative reduction (tCO2e).
        scope_1_reduction: Scope 1 annual reduction (tCO2e).
        scope_2_reduction: Scope 2 annual reduction (tCO2e).
        scope_3_reduction: Scope 3 annual reduction (tCO2e).
        location_based_total: Location-based total annual (tCO2e).
        market_based_total: Market-based total annual (tCO2e).
        sbti_assessment: SBTi alignment assessment.
        calculated_at: Calculation timestamp (UTC).
        provenance_hash: SHA-256 audit hash.
    """
    portfolio_id: str = Field(
        default_factory=_new_uuid,
        description="Portfolio identifier"
    )
    measures: List[CarbonReductionResult] = Field(
        default_factory=list,
        description="Per-measure results"
    )
    total_annual_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Total annual reduction (tCO2e)"
    )
    total_cumulative_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        description="Total cumulative reduction (tCO2e)"
    )
    scope_1_reduction: Decimal = Field(
        default=Decimal("0"),
        description="Scope 1 annual reduction (tCO2e)"
    )
    scope_2_reduction: Decimal = Field(
        default=Decimal("0"),
        description="Scope 2 annual reduction (tCO2e)"
    )
    scope_3_reduction: Decimal = Field(
        default=Decimal("0"),
        description="Scope 3 annual reduction (tCO2e)"
    )
    location_based_total: Decimal = Field(
        default=Decimal("0"),
        description="Location-based total (tCO2e)"
    )
    market_based_total: Decimal = Field(
        default=Decimal("0"),
        description="Market-based total (tCO2e)"
    )
    sbti_assessment: SBTiAssessment = Field(
        default_factory=SBTiAssessment,
        description="SBTi alignment assessment"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Calculation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CarbonReductionEngine:
    """Carbon reduction calculation engine for quick-win measures.

    Calculates tCO2e reductions from energy savings quick wins per the GHG
    Protocol Corporate Standard.  Supports Scope 1 (fuel combustion),
    Scope 2 (purchased electricity via location-based and market-based
    methods), and Scope 3 attribution.  Provides SBTi alignment assessment
    and multi-year projections with grid decarbonization adjustment.

    Usage::

        engine = CarbonReductionEngine()
        result = engine.calculate_reduction(
            reduction=EnergyReduction(
                name="LED retrofit",
                electricity_savings_kwh=Decimal("50000"),
            ),
            region=GridRegion.UK,
        )
        print(f"Annual CO2e: {result.annual_co2e_tonnes} tCO2e")

    Portfolio-level::

        portfolio = engine.calculate_portfolio(
            reductions=[measure_1, measure_2, measure_3],
            region=GridRegion.US_NATIONAL,
        )
        print(f"Total: {portfolio.total_annual_co2e_tonnes} tCO2e")
        print(f"SBTi on track: {portfolio.sbti_assessment.on_track}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise CarbonReductionEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - grid_decarbonization_rate (Decimal): annual grid
                  decarbonization rate (default 0.02 = 2%).
                - default_projection_years (int): projection horizon
                  (default 10 years).
                - custom_grid_factors (dict): override grid emission factors.
                - custom_fuel_factors (dict): override fuel emission factors.
        """
        self.config = config or {}
        self._decarb_rate = _decimal(
            self.config.get(
                "grid_decarbonization_rate", DEFAULT_GRID_DECARBONIZATION_RATE
            )
        )
        self._default_years = int(
            self.config.get("default_projection_years", DEFAULT_PROJECTION_YEARS)
        )
        # Allow custom factor overrides
        self._grid_factors: Dict[str, Dict[str, Any]] = dict(GRID_EMISSION_FACTORS)
        if "custom_grid_factors" in self.config:
            self._grid_factors.update(self.config["custom_grid_factors"])

        self._fuel_factors: Dict[str, Dict[str, Any]] = dict(FUEL_EMISSION_FACTORS)
        if "custom_fuel_factors" in self.config:
            self._fuel_factors.update(self.config["custom_fuel_factors"])

        logger.info(
            "CarbonReductionEngine v%s initialised (decarb_rate=%.3f, years=%d)",
            self.engine_version,
            float(self._decarb_rate),
            self._default_years,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_reduction(
        self,
        reduction: EnergyReduction,
        region: GridRegion,
        method: CalculationMethod = CalculationMethod.LOCATION_BASED,
        projection_years: int = 10,
        custom_factor: Optional[EmissionFactor] = None,
    ) -> CarbonReductionResult:
        """Calculate CO2e reduction for a single energy-saving measure.

        Applies the appropriate emission factor based on region, scope,
        and calculation method (location-based vs market-based).
        Projects cumulative reductions over the specified horizon,
        adjusting for grid decarbonization.

        Args:
            reduction: Energy reduction data for the measure.
            region: Grid region for electricity emission factors.
            method: Scope 2 calculation method.
            projection_years: Number of years to project (default 10).
            custom_factor: Optional custom emission factor override.

        Returns:
            CarbonReductionResult with annual and cumulative tCO2e.

        Raises:
            ValueError: If region is CUSTOM and no custom_factor provided.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating reduction: measure=%s, region=%s, method=%s",
            reduction.name, region.value, method.value,
        )

        if region == GridRegion.CUSTOM and custom_factor is None:
            raise ValueError(
                "CUSTOM region requires a custom_factor to be provided."
            )

        # Step 1: Resolve emission factor for electricity
        elec_factor = self._resolve_electricity_factor(
            region, method, custom_factor
        )

        # Step 2: Resolve emission factor for natural gas
        gas_factor = self._resolve_fuel_factor(FuelType.NATURAL_GAS)

        # Step 3: Resolve emission factor for other fuels
        other_factor = self._resolve_fuel_factor(reduction.other_fuel_type)

        # Step 4: Calculate annual CO2e (kg)
        elec_co2e_kg = reduction.electricity_savings_kwh * elec_factor.factor_value
        gas_co2e_kg = reduction.gas_savings_therms * gas_factor.factor_value
        other_co2e_kg = reduction.other_fuel_savings * other_factor.factor_value

        annual_co2e_kg = elec_co2e_kg + gas_co2e_kg + other_co2e_kg
        annual_co2e_tonnes = annual_co2e_kg / Decimal("1000")

        # Step 5: Select the primary factor for provenance
        primary_factor = self._select_primary_factor(
            elec_factor, gas_factor, other_factor,
            reduction, region, method,
        )

        # Step 6: Project cumulative reductions with grid decarbonization
        cumulative_tonnes = self._project_cumulative(
            annual_co2e_kg=annual_co2e_kg,
            elec_fraction=_safe_divide(elec_co2e_kg, annual_co2e_kg),
            projection_years=projection_years,
        )

        # Step 7: Build methodology notes
        notes = self._build_methodology_notes(
            reduction, region, method, elec_factor, gas_factor, other_factor,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.debug(
            "Reduction calculation complete in %.3f ms", elapsed_ms,
        )

        result = CarbonReductionResult(
            measure_id=reduction.measure_id,
            name=reduction.name,
            scope=reduction.scope,
            calculation_method=method,
            annual_co2e_kg=_round_val(annual_co2e_kg, 4),
            annual_co2e_tonnes=_round_val(annual_co2e_tonnes, 6),
            cumulative_co2e_tonnes=_round_val(cumulative_tonnes, 6),
            projection_years=projection_years,
            emission_factor_used=primary_factor,
            methodology_notes=notes,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Reduction result: measure=%s, annual=%.4f tCO2e, "
            "cumulative=%.4f tCO2e, hash=%s",
            reduction.name,
            float(annual_co2e_tonnes),
            float(cumulative_tonnes),
            result.provenance_hash[:16],
        )
        return result

    def calculate_portfolio(
        self,
        reductions: List[EnergyReduction],
        region: GridRegion,
        method: CalculationMethod = CalculationMethod.LOCATION_BASED,
        projection_years: int = 10,
        base_year_emissions: Optional[Decimal] = None,
    ) -> PortfolioReduction:
        """Calculate aggregated carbon reductions for a portfolio of measures.

        Processes each measure individually, then aggregates by scope and
        method.  Optionally performs SBTi alignment assessment when
        base_year_emissions is provided.

        Args:
            reductions: List of energy reduction measures.
            region: Grid region for electricity factors.
            method: Scope 2 calculation method.
            projection_years: Projection horizon (years).
            base_year_emissions: Total base-year emissions for SBTi (tCO2e).

        Returns:
            PortfolioReduction with aggregated results and SBTi assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Portfolio calculation: %d measures, region=%s, method=%s",
            len(reductions), region.value, method.value,
        )

        # Step 1: Calculate each measure
        measure_results: List[CarbonReductionResult] = []
        for reduction in reductions:
            result = self.calculate_reduction(
                reduction=reduction,
                region=region,
                method=method,
                projection_years=projection_years,
            )
            measure_results.append(result)

        # Step 2: Also calculate market-based for location-based (and vice versa)
        alt_method = (
            CalculationMethod.MARKET_BASED
            if method == CalculationMethod.LOCATION_BASED
            else CalculationMethod.LOCATION_BASED
        )
        alt_results: List[CarbonReductionResult] = []
        for reduction in reductions:
            alt_result = self.calculate_reduction(
                reduction=reduction,
                region=region,
                method=alt_method,
                projection_years=projection_years,
            )
            alt_results.append(alt_result)

        # Step 3: Aggregate by scope
        scope_1 = sum(
            (mr.annual_co2e_tonnes for mr in measure_results
             if mr.scope == EmissionScope.SCOPE_1),
            Decimal("0"),
        )
        scope_2 = sum(
            (mr.annual_co2e_tonnes for mr in measure_results
             if mr.scope == EmissionScope.SCOPE_2),
            Decimal("0"),
        )
        scope_3 = sum(
            (mr.annual_co2e_tonnes for mr in measure_results
             if mr.scope == EmissionScope.SCOPE_3),
            Decimal("0"),
        )

        total_annual = sum(
            (mr.annual_co2e_tonnes for mr in measure_results),
            Decimal("0"),
        )
        total_cumulative = sum(
            (mr.cumulative_co2e_tonnes for mr in measure_results),
            Decimal("0"),
        )

        # Step 4: Location-based vs market-based totals
        if method == CalculationMethod.LOCATION_BASED:
            location_total = total_annual
            market_total = sum(
                (ar.annual_co2e_tonnes for ar in alt_results),
                Decimal("0"),
            )
        else:
            market_total = total_annual
            location_total = sum(
                (ar.annual_co2e_tonnes for ar in alt_results),
                Decimal("0"),
            )

        # Step 5: SBTi assessment (if base year emissions provided)
        sbti = SBTiAssessment()
        if base_year_emissions is not None and base_year_emissions > Decimal("0"):
            current_year = _utcnow().year
            sbti = self.assess_sbti_alignment(
                base_year_emissions=base_year_emissions,
                current_emissions=base_year_emissions - total_annual,
                reductions=measure_results,
                base_year=current_year - 1,
                target_year=current_year + projection_years - 1,
                ambition=SBTiAmbition.ONE_POINT_FIVE_C,
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        portfolio = PortfolioReduction(
            measures=measure_results,
            total_annual_co2e_tonnes=_round_val(total_annual, 6),
            total_cumulative_co2e_tonnes=_round_val(total_cumulative, 6),
            scope_1_reduction=_round_val(scope_1, 6),
            scope_2_reduction=_round_val(scope_2, 6),
            scope_3_reduction=_round_val(scope_3, 6),
            location_based_total=_round_val(location_total, 6),
            market_based_total=_round_val(market_total, 6),
            sbti_assessment=sbti,
        )
        portfolio.provenance_hash = _compute_hash(portfolio)

        logger.info(
            "Portfolio complete: %d measures, total=%.4f tCO2e/yr, "
            "S1=%.4f S2=%.4f S3=%.4f, hash=%s (%.1f ms)",
            len(measure_results),
            float(total_annual),
            float(scope_1), float(scope_2), float(scope_3),
            portfolio.provenance_hash[:16],
            elapsed_ms,
        )
        return portfolio

    def assess_sbti_alignment(
        self,
        base_year_emissions: Decimal,
        current_emissions: Decimal,
        reductions: List[CarbonReductionResult],
        base_year: int,
        target_year: int,
        ambition: SBTiAmbition = SBTiAmbition.ONE_POINT_FIVE_C,
    ) -> SBTiAssessment:
        """Assess alignment of reductions against SBTi targets.

        Evaluates whether the combined energy quick-win reductions
        are on track to meet the required SBTi decarbonization pathway.

        Args:
            base_year_emissions: Total base year emissions (tCO2e).
            current_emissions: Current total emissions (tCO2e).
            reductions: List of calculated carbon reduction results.
            base_year: SBTi base year.
            target_year: SBTi target year.
            ambition: SBTi ambition level.

        Returns:
            SBTiAssessment with on_track status and gap analysis.
        """
        logger.info(
            "SBTi assessment: base=%d, target=%d, ambition=%s, "
            "base_emissions=%.2f tCO2e",
            base_year, target_year, ambition.value,
            float(base_year_emissions),
        )

        if base_year_emissions <= Decimal("0"):
            return SBTiAssessment(
                base_year=base_year,
                target_year=target_year,
                base_year_emissions=Decimal("0"),
                notes="Cannot assess: base year emissions must be positive.",
            )

        target_config = SBTI_TARGETS.get(ambition.value, {})
        annual_rate = _decimal(target_config.get("annual_rate", Decimal("0.042")))
        years_elapsed = max(target_year - base_year, 1)

        # Required reduction = linear rate * years
        required_reduction_pct = annual_rate * _decimal(years_elapsed) * Decimal("100")
        required_reduction_pct = min(required_reduction_pct, Decimal("100"))

        # For net-zero, also check long-term 90% requirement
        if ambition == SBTiAmbition.NET_ZERO:
            long_term_pct = _decimal(
                target_config.get("long_term_reduction_pct", Decimal("90"))
            )
            long_term_years = int(
                target_config.get("long_term_timeframe_years", 30)
            )
            if years_elapsed >= long_term_years:
                required_reduction_pct = max(required_reduction_pct, long_term_pct)

        required_reduction_tonnes = (
            base_year_emissions * required_reduction_pct / Decimal("100")
        )

        # Achieved reduction
        total_reduction = sum(
            (r.annual_co2e_tonnes for r in reductions),
            Decimal("0"),
        )
        actual_reduction = base_year_emissions - current_emissions
        combined_reduction = actual_reduction + total_reduction

        achieved_pct = _safe_pct(combined_reduction, base_year_emissions)

        # Gap analysis
        gap = required_reduction_tonnes - combined_reduction
        on_track = achieved_pct >= required_reduction_pct

        # Build notes
        notes_parts: List[str] = [
            f"SBTi {ambition.value} assessment:",
            f"Required: {_round_val(required_reduction_pct, 1)}% reduction "
            f"({_round_val(required_reduction_tonnes, 2)} tCO2e) "
            f"over {years_elapsed} years.",
            f"Achieved: {_round_val(achieved_pct, 1)}% "
            f"({_round_val(combined_reduction, 2)} tCO2e).",
        ]
        if on_track:
            notes_parts.append("Status: ON TRACK - reductions meet or exceed target.")
        else:
            notes_parts.append(
                f"Status: OFF TRACK - gap of {_round_val(gap, 2)} tCO2e "
                f"({_round_val(_safe_pct(gap, base_year_emissions), 1)}% "
                f"of base year)."
            )
        if ambition == SBTiAmbition.NET_ZERO:
            notes_parts.append(
                "Note: Net-Zero standard requires 90% absolute reduction "
                "by long-term target year."
            )

        return SBTiAssessment(
            base_year=base_year,
            target_year=target_year,
            base_year_emissions=_round_val(base_year_emissions, 2),
            required_reduction_pct=_round_val(required_reduction_pct, 2),
            achieved_reduction_pct=_round_val(achieved_pct, 2),
            on_track=on_track,
            ambition_level=ambition,
            gap_tonnes=_round_val(max(gap, Decimal("0")), 2),
            notes=" ".join(notes_parts),
        )

    def project_reductions(
        self,
        annual_reduction: Decimal,
        years: int,
        method: ProjectionMethod = ProjectionMethod.LINEAR,
        grid_decarbonization_rate: Decimal = DEFAULT_GRID_DECARBONIZATION_RATE,
    ) -> List[AnnualProjection]:
        """Project emission reductions over multiple years.

        Supports linear (constant annual reduction), compound (with grid
        decarbonization adjustment), and SDA pathway methods.

        Args:
            annual_reduction: First-year annual reduction (tCO2e).
            years: Number of years to project.
            method: Projection method.
            grid_decarbonization_rate: Annual grid decarbonization rate.

        Returns:
            List of AnnualProjection, one per year.
        """
        logger.info(
            "Projecting reductions: annual=%.4f tCO2e, years=%d, method=%s",
            float(annual_reduction), years, method.value,
        )

        current_year = _utcnow().year
        projections: List[AnnualProjection] = []
        cumulative = Decimal("0")

        for yr_offset in range(1, years + 1):
            if method == ProjectionMethod.LINEAR:
                # Constant annual reduction each year
                year_reduction = annual_reduction

            elif method == ProjectionMethod.COMPOUND:
                # Adjust for grid decarbonization: emission factor decreases
                # so the same kWh savings yields fewer avoided tCO2e over time
                decarb_multiplier = self._apply_grid_decarbonization(
                    Decimal("1"), yr_offset, grid_decarbonization_rate,
                )
                year_reduction = annual_reduction * decarb_multiplier

            elif method == ProjectionMethod.SDA_PATHWAY:
                # SDA: reduction accelerates as sector pathway steepens
                # Simplified model: base * (1 + 0.01 * year_offset)
                sda_multiplier = Decimal("1") + Decimal("0.01") * _decimal(yr_offset)
                decarb = self._apply_grid_decarbonization(
                    Decimal("1"), yr_offset, grid_decarbonization_rate,
                )
                year_reduction = annual_reduction * decarb * sda_multiplier
            else:
                year_reduction = annual_reduction

            cumulative += year_reduction

            projections.append(AnnualProjection(
                year=current_year + yr_offset,
                projected_co2e_tonnes=_round_val(year_reduction, 6),
                cumulative_co2e_tonnes=_round_val(cumulative, 6),
            ))

        return projections

    def get_emission_factor(
        self,
        region: GridRegion,
        fuel_type: Optional[FuelType] = None,
        factor_type: EmissionFactorType = EmissionFactorType.AVERAGE,
    ) -> EmissionFactor:
        """Retrieve an emission factor with full provenance metadata.

        Args:
            region: Grid region.
            fuel_type: Fuel type (None for electricity grid factor).
            factor_type: Type of factor (average, marginal, residual_mix).

        Returns:
            EmissionFactor with value, source, year, and unit.

        Raises:
            ValueError: If region or fuel_type is not found in database.
        """
        if fuel_type is not None:
            return self._resolve_fuel_factor(fuel_type)

        if factor_type == EmissionFactorType.RESIDUAL_MIX:
            return self._resolve_electricity_factor(
                region, CalculationMethod.MARKET_BASED, None,
            )

        return self._resolve_electricity_factor(
            region, CalculationMethod.LOCATION_BASED, None,
        )

    def compare_methods(
        self,
        reduction: EnergyReduction,
        region: GridRegion,
    ) -> Dict[str, Any]:
        """Compare location-based and market-based CO2e reductions.

        Calculates the same measure under both GHG Protocol Scope 2
        methods and returns a comparison dictionary.

        Args:
            reduction: Energy reduction data.
            region: Grid region.

        Returns:
            Dictionary with location_based, market_based results and delta.
        """
        logger.info(
            "Method comparison: measure=%s, region=%s",
            reduction.name, region.value,
        )

        location_result = self.calculate_reduction(
            reduction=reduction,
            region=region,
            method=CalculationMethod.LOCATION_BASED,
        )
        market_result = self.calculate_reduction(
            reduction=reduction,
            region=region,
            method=CalculationMethod.MARKET_BASED,
        )

        delta_kg = market_result.annual_co2e_kg - location_result.annual_co2e_kg
        delta_tonnes = market_result.annual_co2e_tonnes - location_result.annual_co2e_tonnes
        delta_pct = _safe_pct(
            abs(delta_tonnes),
            location_result.annual_co2e_tonnes,
        )

        comparison: Dict[str, Any] = {
            "measure_id": reduction.measure_id,
            "measure_name": reduction.name,
            "region": region.value,
            "location_based": {
                "annual_co2e_kg": str(location_result.annual_co2e_kg),
                "annual_co2e_tonnes": str(location_result.annual_co2e_tonnes),
                "cumulative_co2e_tonnes": str(location_result.cumulative_co2e_tonnes),
                "factor_value": str(location_result.emission_factor_used.factor_value),
                "factor_source": location_result.emission_factor_used.source,
            },
            "market_based": {
                "annual_co2e_kg": str(market_result.annual_co2e_kg),
                "annual_co2e_tonnes": str(market_result.annual_co2e_tonnes),
                "cumulative_co2e_tonnes": str(market_result.cumulative_co2e_tonnes),
                "factor_value": str(market_result.emission_factor_used.factor_value),
                "factor_source": market_result.emission_factor_used.source,
            },
            "delta": {
                "annual_co2e_kg": str(_round_val(delta_kg, 4)),
                "annual_co2e_tonnes": str(_round_val(delta_tonnes, 6)),
                "percentage_difference": str(_round_val(delta_pct, 2)),
                "higher_method": (
                    "market_based" if delta_tonnes > Decimal("0")
                    else "location_based"
                ),
            },
            "provenance_hash": _compute_hash({
                "location": location_result.provenance_hash,
                "market": market_result.provenance_hash,
            }),
        }

        logger.info(
            "Method comparison complete: location=%.4f, market=%.4f, "
            "delta=%.4f tCO2e (%.1f%%)",
            float(location_result.annual_co2e_tonnes),
            float(market_result.annual_co2e_tonnes),
            float(delta_tonnes),
            float(delta_pct),
        )
        return comparison

    # ------------------------------------------------------------------ #
    # Internal: Factor Resolution                                         #
    # ------------------------------------------------------------------ #

    def _resolve_electricity_factor(
        self,
        region: GridRegion,
        method: CalculationMethod,
        custom_factor: Optional[EmissionFactor],
    ) -> EmissionFactor:
        """Resolve the electricity emission factor for a region and method.

        For location-based: uses average grid factor.
        For market-based: uses residual mix factor.
        For custom: uses the provided custom factor.

        Args:
            region: Grid region.
            method: Calculation method.
            custom_factor: Optional custom factor override.

        Returns:
            EmissionFactor with resolved value and provenance.
        """
        if custom_factor is not None:
            return custom_factor

        if method == CalculationMethod.MARKET_BASED:
            residual = RESIDUAL_MIX_FACTORS.get(region.value)
            if residual is not None:
                grid_info = self._grid_factors.get(region.value, {})
                return EmissionFactor(
                    region=region,
                    fuel_type=None,
                    factor_value=residual,
                    factor_type=EmissionFactorType.RESIDUAL_MIX,
                    source=f"Residual mix - {grid_info.get('source', 'IEA 2024')}",
                    year=grid_info.get("year", 2024),
                    unit="kgCO2e/kWh",
                )

        # Location-based (default) or fallback
        grid_info = self._grid_factors.get(region.value)
        if grid_info is None:
            logger.warning(
                "Grid factor not found for region '%s', "
                "falling back to US_NATIONAL.",
                region.value,
            )
            grid_info = self._grid_factors[GridRegion.US_NATIONAL.value]

        return EmissionFactor(
            region=region,
            fuel_type=None,
            factor_value=_decimal(grid_info["factor"]),
            factor_type=EmissionFactorType.AVERAGE,
            source=grid_info.get("source", ""),
            year=grid_info.get("year", 2024),
            unit=grid_info.get("unit", "kgCO2e/kWh"),
        )

    def _resolve_fuel_factor(
        self,
        fuel_type: Optional[FuelType],
    ) -> EmissionFactor:
        """Resolve emission factor for a fuel type.

        Args:
            fuel_type: Fuel type, or None (returns zero factor).

        Returns:
            EmissionFactor with resolved value and provenance.
        """
        if fuel_type is None:
            return EmissionFactor(
                factor_value=Decimal("0"),
                source="N/A (no fuel type specified)",
                unit="kgCO2e/unit",
            )

        fuel_info = self._fuel_factors.get(fuel_type.value)
        if fuel_info is None:
            logger.warning(
                "Fuel factor not found for type '%s', returning zero.",
                fuel_type.value,
            )
            return EmissionFactor(
                fuel_type=fuel_type,
                factor_value=Decimal("0"),
                source="Unknown fuel type",
                unit="kgCO2e/unit",
            )

        return EmissionFactor(
            fuel_type=fuel_type,
            factor_value=_decimal(fuel_info["factor"]),
            factor_type=EmissionFactorType.AVERAGE,
            source=fuel_info.get("source", ""),
            year=fuel_info.get("year", 2023),
            unit=fuel_info.get("unit", "kgCO2e/unit"),
        )

    # ------------------------------------------------------------------ #
    # Internal: Grid Decarbonization                                      #
    # ------------------------------------------------------------------ #

    def _apply_grid_decarbonization(
        self,
        factor: Decimal,
        year_offset: int,
        rate: Decimal,
    ) -> Decimal:
        """Apply grid decarbonization adjustment to an emission factor.

        Models the trend of decreasing grid carbon intensity over time
        as renewable energy penetration increases.

        Formula: adjusted = factor * (1 - rate) ^ year_offset

        Args:
            factor: Base emission factor or multiplier.
            year_offset: Number of years from base year.
            rate: Annual decarbonization rate (e.g. 0.02 = 2%).

        Returns:
            Adjusted factor after decarbonization.
        """
        if rate <= Decimal("0") or year_offset <= 0:
            return factor

        decay = (Decimal("1") - rate) ** _decimal(year_offset)
        return factor * decay

    # ------------------------------------------------------------------ #
    # Internal: Projection                                                #
    # ------------------------------------------------------------------ #

    def _project_cumulative(
        self,
        annual_co2e_kg: Decimal,
        elec_fraction: Decimal,
        projection_years: int,
    ) -> Decimal:
        """Project cumulative CO2e reduction over multiple years.

        Electricity-related reductions are adjusted for grid
        decarbonization; fuel-related reductions remain constant.

        Args:
            annual_co2e_kg: Annual CO2e reduction (kg).
            elec_fraction: Fraction of CO2e from electricity.
            projection_years: Number of years.

        Returns:
            Cumulative CO2e reduction (tonnes).
        """
        cumulative_kg = Decimal("0")
        elec_portion = annual_co2e_kg * elec_fraction
        fuel_portion = annual_co2e_kg * (Decimal("1") - elec_fraction)

        for yr in range(1, projection_years + 1):
            # Electricity portion decays with grid decarbonization
            adjusted_elec = elec_portion * self._apply_grid_decarbonization(
                Decimal("1"), yr, self._decarb_rate,
            )
            # Fuel portion stays constant (combustion factors are stable)
            year_total = adjusted_elec + fuel_portion
            cumulative_kg += year_total

        return cumulative_kg / Decimal("1000")

    # ------------------------------------------------------------------ #
    # Internal: Factor Selection                                          #
    # ------------------------------------------------------------------ #

    def _select_primary_factor(
        self,
        elec_factor: EmissionFactor,
        gas_factor: EmissionFactor,
        other_factor: EmissionFactor,
        reduction: EnergyReduction,
        region: GridRegion,
        method: CalculationMethod,
    ) -> EmissionFactor:
        """Select the primary emission factor for provenance reporting.

        Chooses the factor contributing the largest CO2e share as the
        primary factor attached to the result.

        Args:
            elec_factor: Electricity emission factor.
            gas_factor: Gas emission factor.
            other_factor: Other fuel emission factor.
            reduction: Energy reduction data.
            region: Grid region.
            method: Calculation method.

        Returns:
            The dominant EmissionFactor.
        """
        elec_co2e = reduction.electricity_savings_kwh * elec_factor.factor_value
        gas_co2e = reduction.gas_savings_therms * gas_factor.factor_value
        other_co2e = reduction.other_fuel_savings * other_factor.factor_value

        max_co2e = max(elec_co2e, gas_co2e, other_co2e)

        if max_co2e == gas_co2e and gas_co2e > Decimal("0"):
            return gas_factor
        if max_co2e == other_co2e and other_co2e > Decimal("0"):
            return other_factor
        # Default to electricity factor
        return elec_factor

    # ------------------------------------------------------------------ #
    # Internal: Methodology Notes                                         #
    # ------------------------------------------------------------------ #

    def _build_methodology_notes(
        self,
        reduction: EnergyReduction,
        region: GridRegion,
        method: CalculationMethod,
        elec_factor: EmissionFactor,
        gas_factor: EmissionFactor,
        other_factor: EmissionFactor,
    ) -> str:
        """Build detailed methodology notes for audit trail.

        Args:
            reduction: Energy reduction data.
            region: Grid region.
            method: Calculation method.
            elec_factor: Electricity factor used.
            gas_factor: Gas factor used.
            other_factor: Other fuel factor used.

        Returns:
            Multi-line methodology description.
        """
        parts: List[str] = [
            f"GHG Protocol {reduction.scope.value} calculation.",
            f"Region: {region.value}.",
            f"Method: {method.value}.",
        ]

        if reduction.electricity_savings_kwh > Decimal("0"):
            parts.append(
                f"Electricity: {reduction.electricity_savings_kwh} kWh * "
                f"{elec_factor.factor_value} {elec_factor.unit} "
                f"(source: {elec_factor.source}, year: {elec_factor.year})."
            )

        if reduction.gas_savings_therms > Decimal("0"):
            parts.append(
                f"Natural gas: {reduction.gas_savings_therms} therms * "
                f"{gas_factor.factor_value} {gas_factor.unit} "
                f"(source: {gas_factor.source}, year: {gas_factor.year})."
            )

        if (
            reduction.other_fuel_savings > Decimal("0")
            and reduction.other_fuel_type is not None
        ):
            parts.append(
                f"Other fuel ({reduction.other_fuel_type.value}): "
                f"{reduction.other_fuel_savings} units * "
                f"{other_factor.factor_value} {other_factor.unit} "
                f"(source: {other_factor.source}, year: {other_factor.year})."
            )

        parts.append(
            f"Grid decarbonization rate: {self._decarb_rate} per year "
            f"applied to electricity portion of projections."
        )

        return " ".join(parts)
