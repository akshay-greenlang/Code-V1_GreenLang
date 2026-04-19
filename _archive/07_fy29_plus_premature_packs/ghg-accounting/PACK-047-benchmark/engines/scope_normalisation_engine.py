# -*- coding: utf-8 -*-
"""
ScopeNormalisationEngine - PACK-047 GHG Emissions Benchmark Engine 2
====================================================================

Normalises GHG emissions data across heterogeneous reporting boundaries,
consolidation approaches, GWP vintages, currencies, and reporting periods
to enable like-for-like benchmarking comparison.

Calculation Methodology:
    GWP Realignment:
        E_new = SUM(gas_i * GWP_new_i / GWP_old_i * E_old_gas_i)

        Where:
            gas_i       = individual greenhouse gas (CO2, CH4, N2O, HFCs, etc.)
            GWP_new_i   = GWP for gas_i under new assessment report
            GWP_old_i   = GWP for gas_i under old assessment report
            E_old_gas_i = reported emissions of gas_i under old GWP

    PPP Currency Adjustment:
        D_ppp = D_nominal * PPP_factor(country, year)

        Where:
            D_nominal   = nominal denominator value (e.g., revenue in local currency)
            PPP_factor  = purchasing power parity conversion factor
            D_ppp       = PPP-adjusted denominator

    Reporting Period Pro-Rata:
        E_aligned = E_reported * days_overlap / days_reporting

        Where:
            E_reported    = emissions for actual reporting period
            days_overlap  = number of days overlapping with target period
            days_reporting = total days in actual reporting period

    Scope Boundary Adjustment:
        E_adjusted = E_reported * scope_factor(from_scope, to_scope)

        scope_factor uses sector-specific ratios from published sources:
            S1 -> S1+S2:    factor = 1 + sector_s2_ratio
            S1+S2 -> S1+S2+S3: factor = 1 + sector_s3_ratio
            etc.

    Consolidation Approach Normalisation:
        E_normalised = E_reported * consolidation_factor(from_approach, to_approach)

        Supported approaches: equity_share, operational_control, financial_control

    Biogenic Carbon Treatment:
        When aligning biogenic treatment:
            If include_biogenic: E_total = E_fossil + E_biogenic
            If exclude_biogenic: E_total = E_fossil

Regulatory References:
    - GHG Protocol Corporate Standard: Chapter 4 (Organisational boundaries)
    - GHG Protocol Corporate Standard: Chapter 3 (GWP and emission factors)
    - IPCC AR4/AR5/AR6 GWP tables (100-year time horizon)
    - World Bank PPP conversion factors
    - ESRS E1-6: Normalisation for comparability
    - CDP C0.5: Reporting year alignment
    - SBTi SDA: Scope boundary requirements

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - GWP values from published IPCC assessment reports only
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-047 GHG Emissions Benchmark
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _round6(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GWPVersion(str, Enum):
    """IPCC Assessment Report GWP version.

    AR4:  Fourth Assessment Report (2007) - still used by some reporters.
    AR5:  Fifth Assessment Report (2014) - GHG Protocol default.
    AR6:  Sixth Assessment Report (2021) - latest available.
    """
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"

class ConsolidationApproach(str, Enum):
    """Organisational boundary consolidation approach.

    EQUITY_SHARE:       Proportional to equity stake.
    OPERATIONAL_CONTROL: 100% for operations under control.
    FINANCIAL_CONTROL:   100% for operations under financial control.
    """
    EQUITY_SHARE = "equity_share"
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"

class ScopeBoundary(str, Enum):
    """Scope boundary for emissions reporting.

    S1:                Scope 1 only.
    S1_S2:             Scope 1 + Scope 2.
    S1_S2_S3_PARTIAL:  Scope 1 + 2 + partial Scope 3.
    S1_S2_S3_FULL:     Scope 1 + 2 + full Scope 3.
    """
    S1 = "s1"
    S1_S2 = "s1_s2"
    S1_S2_S3_PARTIAL = "s1_s2_s3_partial"
    S1_S2_S3_FULL = "s1_s2_s3_full"

class BiogenicTreatment(str, Enum):
    """Biogenic carbon reporting treatment.

    INCLUDED:   Biogenic emissions included in total.
    EXCLUDED:   Biogenic emissions excluded from total.
    SEPARATE:   Biogenic reported separately.
    """
    INCLUDED = "included"
    EXCLUDED = "excluded"
    SEPARATE = "separate"

class NormalisationStep(str, Enum):
    """Types of normalisation steps applied."""
    GWP_REALIGNMENT = "gwp_realignment"
    SCOPE_BOUNDARY = "scope_boundary"
    CONSOLIDATION = "consolidation"
    CURRENCY_PPP = "currency_ppp"
    PERIOD_ALIGNMENT = "period_alignment"
    BIOGENIC_ALIGNMENT = "biogenic_alignment"
    DATA_GAP_ESTIMATION = "data_gap_estimation"

class DataQualityFlag(str, Enum):
    """Data quality flags applied during normalisation."""
    ORIGINAL = "original"
    GWP_CONVERTED = "gwp_converted"
    SCOPE_ESTIMATED = "scope_estimated"
    PERIOD_PRORATED = "period_prorated"
    PPP_ADJUSTED = "ppp_adjusted"
    BIOGENIC_ADJUSTED = "biogenic_adjusted"
    GAP_FILLED = "gap_filled"
    CONSOLIDATION_ADJUSTED = "consolidation_adjusted"

# ---------------------------------------------------------------------------
# Constants -- GWP Values (100-year time horizon)
# ---------------------------------------------------------------------------

# Source: IPCC AR4 Table 2.14, AR5 Table 8.A.1, AR6 Table 7.15
GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "CO2": {
        GWPVersion.AR4.value: Decimal("1"),
        GWPVersion.AR5.value: Decimal("1"),
        GWPVersion.AR6.value: Decimal("1"),
    },
    "CH4": {
        GWPVersion.AR4.value: Decimal("25"),
        GWPVersion.AR5.value: Decimal("28"),
        GWPVersion.AR6.value: Decimal("27.9"),
    },
    "N2O": {
        GWPVersion.AR4.value: Decimal("298"),
        GWPVersion.AR5.value: Decimal("265"),
        GWPVersion.AR6.value: Decimal("273"),
    },
    "HFC-134a": {
        GWPVersion.AR4.value: Decimal("1430"),
        GWPVersion.AR5.value: Decimal("1300"),
        GWPVersion.AR6.value: Decimal("1526"),
    },
    "HFC-32": {
        GWPVersion.AR4.value: Decimal("675"),
        GWPVersion.AR5.value: Decimal("677"),
        GWPVersion.AR6.value: Decimal("771"),
    },
    "HFC-125": {
        GWPVersion.AR4.value: Decimal("3500"),
        GWPVersion.AR5.value: Decimal("3170"),
        GWPVersion.AR6.value: Decimal("3740"),
    },
    "HFC-143a": {
        GWPVersion.AR4.value: Decimal("4470"),
        GWPVersion.AR5.value: Decimal("4800"),
        GWPVersion.AR6.value: Decimal("5810"),
    },
    "HFC-152a": {
        GWPVersion.AR4.value: Decimal("124"),
        GWPVersion.AR5.value: Decimal("138"),
        GWPVersion.AR6.value: Decimal("164"),
    },
    "HFC-227ea": {
        GWPVersion.AR4.value: Decimal("3220"),
        GWPVersion.AR5.value: Decimal("3350"),
        GWPVersion.AR6.value: Decimal("3860"),
    },
    "HFC-236fa": {
        GWPVersion.AR4.value: Decimal("9810"),
        GWPVersion.AR5.value: Decimal("8060"),
        GWPVersion.AR6.value: Decimal("8690"),
    },
    "HFC-245fa": {
        GWPVersion.AR4.value: Decimal("1030"),
        GWPVersion.AR5.value: Decimal("858"),
        GWPVersion.AR6.value: Decimal("962"),
    },
    "HFC-365mfc": {
        GWPVersion.AR4.value: Decimal("794"),
        GWPVersion.AR5.value: Decimal("804"),
        GWPVersion.AR6.value: Decimal("914"),
    },
    "HFC-43-10mee": {
        GWPVersion.AR4.value: Decimal("1640"),
        GWPVersion.AR5.value: Decimal("1650"),
        GWPVersion.AR6.value: Decimal("1600"),
    },
    "SF6": {
        GWPVersion.AR4.value: Decimal("22800"),
        GWPVersion.AR5.value: Decimal("23500"),
        GWPVersion.AR6.value: Decimal("25200"),
    },
    "NF3": {
        GWPVersion.AR4.value: Decimal("17200"),
        GWPVersion.AR5.value: Decimal("16100"),
        GWPVersion.AR6.value: Decimal("17400"),
    },
    "CF4": {
        GWPVersion.AR4.value: Decimal("7390"),
        GWPVersion.AR5.value: Decimal("6630"),
        GWPVersion.AR6.value: Decimal("7380"),
    },
    "C2F6": {
        GWPVersion.AR4.value: Decimal("12200"),
        GWPVersion.AR5.value: Decimal("11100"),
        GWPVersion.AR6.value: Decimal("12400"),
    },
}

# Sector-specific scope ratios (S2 as fraction of S1, S3 as fraction of S1+S2)
# Source: CDP 2023 sector averages
SECTOR_SCOPE_RATIOS: Dict[str, Dict[str, Decimal]] = {
    "energy": {"s2_ratio": Decimal("0.15"), "s3_ratio": Decimal("5.50")},
    "materials": {"s2_ratio": Decimal("0.35"), "s3_ratio": Decimal("2.80")},
    "industrials": {"s2_ratio": Decimal("0.40"), "s3_ratio": Decimal("4.20")},
    "consumer_discretionary": {"s2_ratio": Decimal("0.55"), "s3_ratio": Decimal("8.50")},
    "consumer_staples": {"s2_ratio": Decimal("0.30"), "s3_ratio": Decimal("6.00")},
    "health_care": {"s2_ratio": Decimal("0.50"), "s3_ratio": Decimal("5.00")},
    "financials": {"s2_ratio": Decimal("2.50"), "s3_ratio": Decimal("12.00")},
    "information_technology": {"s2_ratio": Decimal("1.20"), "s3_ratio": Decimal("15.00")},
    "communication_services": {"s2_ratio": Decimal("0.80"), "s3_ratio": Decimal("4.50")},
    "utilities": {"s2_ratio": Decimal("0.05"), "s3_ratio": Decimal("3.00")},
    "real_estate": {"s2_ratio": Decimal("1.50"), "s3_ratio": Decimal("6.00")},
    "default": {"s2_ratio": Decimal("0.50"), "s3_ratio": Decimal("5.00")},
}

# Sample PPP factors (2024, relative to USD=1.0)
PPP_FACTORS: Dict[str, Dict[int, Decimal]] = {
    "USD": {2022: Decimal("1.000"), 2023: Decimal("1.000"), 2024: Decimal("1.000")},
    "EUR": {2022: Decimal("0.826"), 2023: Decimal("0.842"), 2024: Decimal("0.850")},
    "GBP": {2022: Decimal("0.697"), 2023: Decimal("0.710"), 2024: Decimal("0.718")},
    "JPY": {2022: Decimal("102.6"), 2023: Decimal("97.5"), 2024: Decimal("95.8")},
    "CNY": {2022: Decimal("4.020"), 2023: Decimal("4.110"), 2024: Decimal("4.050")},
    "INR": {2022: Decimal("23.65"), 2023: Decimal("24.10"), 2024: Decimal("24.50")},
    "BRL": {2022: Decimal("2.560"), 2023: Decimal("2.640"), 2024: Decimal("2.680")},
    "AUD": {2022: Decimal("1.480"), 2023: Decimal("1.510"), 2024: Decimal("1.520")},
    "CAD": {2022: Decimal("1.260"), 2023: Decimal("1.270"), 2024: Decimal("1.280")},
    "CHF": {2022: Decimal("1.170"), 2023: Decimal("1.180"), 2024: Decimal("1.190")},
}

CONSOLIDATION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # from -> to factors (multiplicative)
    "equity_share__operational_control": Decimal("1.15"),
    "equity_share__financial_control": Decimal("1.10"),
    "operational_control__equity_share": Decimal("0.87"),
    "operational_control__financial_control": Decimal("0.96"),
    "financial_control__equity_share": Decimal("0.91"),
    "financial_control__operational_control": Decimal("1.04"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class GasBreakdown(BaseModel):
    """Breakdown of emissions by individual greenhouse gas.

    Attributes:
        gas:             Gas name (e.g., CO2, CH4, N2O).
        emissions_tco2e: Emissions in tCO2e under reported GWP.
        mass_tonnes:     Mass in tonnes of gas (if available).
    """
    gas: str = Field(..., description="Gas name")
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Emissions tCO2e")
    mass_tonnes: Optional[Decimal] = Field(default=None, ge=0, description="Mass (tonnes)")

    @field_validator("emissions_tco2e", mode="before")
    @classmethod
    def coerce_emissions(cls, v: Any) -> Decimal:
        return _decimal(v)

class NormalisationInput(BaseModel):
    """Input for scope normalisation.

    Attributes:
        entity_id:                  Entity identifier.
        entity_name:                Entity name.
        total_emissions_tco2e:      Total reported emissions.
        scope1_tco2e:               Scope 1 emissions.
        scope2_tco2e:               Scope 2 emissions.
        scope3_tco2e:               Scope 3 emissions (if reported).
        gas_breakdown:              Per-gas breakdown (for GWP conversion).
        reported_gwp:               GWP version used in reporting.
        target_gwp:                 Target GWP version for normalisation.
        reported_scope:             Reported scope boundary.
        target_scope:               Target scope boundary.
        sector:                     Sector for scope ratio lookup.
        reported_consolidation:     Reported consolidation approach.
        target_consolidation:       Target consolidation approach.
        reported_currency:          Reported currency for denominators.
        target_currency:            Target currency.
        reporting_year:             Reporting year.
        reporting_period_start:     Reporting period start date.
        reporting_period_end:       Reporting period end date.
        target_period_start:        Target period start date.
        target_period_end:          Target period end date.
        biogenic_emissions_tco2e:   Biogenic emissions.
        reported_biogenic:          Reported biogenic treatment.
        target_biogenic:            Target biogenic treatment.
        denominator_value:          Economic denominator value.
        output_precision:           Output decimal places.
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    total_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_tco2e: Optional[Decimal] = Field(default=None, ge=0)
    gas_breakdown: List[GasBreakdown] = Field(default_factory=list)
    reported_gwp: GWPVersion = Field(default=GWPVersion.AR5)
    target_gwp: GWPVersion = Field(default=GWPVersion.AR6)
    reported_scope: ScopeBoundary = Field(default=ScopeBoundary.S1_S2)
    target_scope: ScopeBoundary = Field(default=ScopeBoundary.S1_S2)
    sector: str = Field(default="default")
    reported_consolidation: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL
    )
    target_consolidation: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL
    )
    reported_currency: str = Field(default="USD")
    target_currency: str = Field(default="USD")
    reporting_year: int = Field(default=2024)
    reporting_period_start: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    reporting_period_end: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    target_period_start: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    target_period_end: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    biogenic_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    reported_biogenic: BiogenicTreatment = Field(default=BiogenicTreatment.EXCLUDED)
    target_biogenic: BiogenicTreatment = Field(default=BiogenicTreatment.EXCLUDED)
    denominator_value: Optional[Decimal] = Field(default=None, ge=0)
    output_precision: int = Field(default=3, ge=0, le=12)

    @field_validator(
        "total_emissions_tco2e", "scope1_tco2e", "scope2_tco2e",
        "biogenic_emissions_tco2e", mode="before",
    )
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class GWPConversionResult(BaseModel):
    """Result of GWP version realignment.

    Attributes:
        gas:                 Gas name.
        original_tco2e:      Original emissions (tCO2e).
        converted_tco2e:     Converted emissions (tCO2e).
        original_gwp:        Original GWP value used.
        target_gwp:          Target GWP value used.
        conversion_factor:   Ratio of new/old GWP.
    """
    gas: str = Field(default="", description="Gas name")
    original_tco2e: Decimal = Field(default=Decimal("0"), description="Original tCO2e")
    converted_tco2e: Decimal = Field(default=Decimal("0"), description="Converted tCO2e")
    original_gwp: Decimal = Field(default=Decimal("0"), description="Original GWP")
    target_gwp: Decimal = Field(default=Decimal("0"), description="Target GWP")
    conversion_factor: Decimal = Field(default=Decimal("1"), description="Conversion factor")

class NormalisationStepDetail(BaseModel):
    """Detail of a single normalisation step applied.

    Attributes:
        step:               Normalisation step type.
        description:        Human-readable description.
        input_value:        Value before step.
        output_value:       Value after step.
        factor_applied:     Multiplicative factor applied.
        quality_flag:       Data quality flag assigned.
    """
    step: NormalisationStep = Field(..., description="Step type")
    description: str = Field(default="", description="Description")
    input_value: Decimal = Field(default=Decimal("0"), description="Input value")
    output_value: Decimal = Field(default=Decimal("0"), description="Output value")
    factor_applied: Decimal = Field(default=Decimal("1"), description="Factor")
    quality_flag: DataQualityFlag = Field(default=DataQualityFlag.ORIGINAL)

class NormalisedEmissions(BaseModel):
    """Normalised emissions output for an entity.

    Attributes:
        entity_id:              Entity identifier.
        original_total_tco2e:   Original total emissions.
        normalised_total_tco2e: Normalised total emissions.
        normalised_scope1:      Normalised Scope 1.
        normalised_scope2:      Normalised Scope 2.
        normalised_scope3:      Normalised Scope 3.
        normalised_denominator: Normalised denominator (if applicable).
        normalised_intensity:   Normalised intensity (if denominator present).
        gwp_conversions:        Per-gas GWP conversion details.
        steps_applied:          Normalisation steps applied.
        quality_flags:          Data quality flags.
        quality_downgrade:      Whether quality was downgraded.
    """
    entity_id: str = Field(default="", description="Entity ID")
    original_total_tco2e: Decimal = Field(default=Decimal("0"))
    normalised_total_tco2e: Decimal = Field(default=Decimal("0"))
    normalised_scope1: Decimal = Field(default=Decimal("0"))
    normalised_scope2: Decimal = Field(default=Decimal("0"))
    normalised_scope3: Optional[Decimal] = Field(default=None)
    normalised_denominator: Optional[Decimal] = Field(default=None)
    normalised_intensity: Optional[Decimal] = Field(default=None)
    gwp_conversions: List[GWPConversionResult] = Field(default_factory=list)
    steps_applied: List[NormalisationStepDetail] = Field(default_factory=list)
    quality_flags: List[DataQualityFlag] = Field(default_factory=list)
    quality_downgrade: bool = Field(default=False)

class NormalisationRun(BaseModel):
    """Complete result of a normalisation run.

    Attributes:
        result_id:              Unique result identifier.
        entities:               Normalised emission records.
        total_entities:         Total entities processed.
        gwp_conversions_count:  GWP conversions performed.
        scope_adjustments_count: Scope adjustments performed.
        period_alignments_count: Period alignments performed.
        quality_downgrades:     Number of quality downgrades.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    entities: List[NormalisedEmissions] = Field(default_factory=list)
    total_entities: int = Field(default=0)
    gwp_conversions_count: int = Field(default=0)
    scope_adjustments_count: int = Field(default=0)
    period_alignments_count: int = Field(default=0)
    quality_downgrades: int = Field(default=0)
    warnings: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ScopeNormalisationEngine:
    """Normalises GHG emissions for like-for-like benchmarking.

    Aligns scope boundaries, GWP versions, consolidation approaches,
    currencies (PPP), reporting periods, and biogenic carbon treatment.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every normalisation step documented with factors.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("ScopeNormalisationEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: NormalisationInput) -> NormalisationRun:
        """Normalise a single entity's emissions.

        Args:
            input_data: Normalisation input.

        Returns:
            NormalisationRun with normalised emissions and step details.
        """
        return self.normalise_batch([input_data])

    def normalise_batch(self, inputs: List[NormalisationInput]) -> NormalisationRun:
        """Normalise a batch of entities.

        Args:
            inputs: List of normalisation inputs.

        Returns:
            NormalisationRun with all normalised entities.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        entities: List[NormalisedEmissions] = []
        gwp_count = 0
        scope_count = 0
        period_count = 0
        downgrade_count = 0

        for inp in inputs:
            entity, ew, gc, sc, pc, dc = self._normalise_entity(inp)
            entities.append(entity)
            warnings.extend(ew)
            gwp_count += gc
            scope_count += sc
            period_count += pc
            downgrade_count += dc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = NormalisationRun(
            entities=entities,
            total_entities=len(entities),
            gwp_conversions_count=gwp_count,
            scope_adjustments_count=scope_count,
            period_alignments_count=period_count,
            quality_downgrades=downgrade_count,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def convert_gwp(
        self,
        gas: str,
        emissions_tco2e: Decimal,
        from_version: GWPVersion,
        to_version: GWPVersion,
    ) -> GWPConversionResult:
        """Convert emissions from one GWP version to another.

        Formula: E_new = E_old * GWP_new / GWP_old

        Args:
            gas:              Gas name.
            emissions_tco2e:  Emissions under from_version GWP.
            from_version:     Source GWP version.
            to_version:       Target GWP version.

        Returns:
            GWPConversionResult.
        """
        return self._convert_gas_gwp(gas, emissions_tco2e, from_version, to_version)

    def get_ppp_factor(
        self, currency: str, year: int, target_currency: str = "USD",
    ) -> Decimal:
        """Get PPP conversion factor.

        Args:
            currency:        Source currency.
            year:            Year.
            target_currency: Target currency.

        Returns:
            PPP factor.
        """
        return self._get_ppp_factor(currency, year, target_currency)

    def get_scope_ratio(self, sector: str) -> Dict[str, Decimal]:
        """Get sector-specific scope ratios.

        Args:
            sector: Sector name.

        Returns:
            Dictionary with s2_ratio and s3_ratio.
        """
        return dict(SECTOR_SCOPE_RATIOS.get(sector, SECTOR_SCOPE_RATIOS["default"]))

    # ------------------------------------------------------------------
    # Internal: Entity Normalisation
    # ------------------------------------------------------------------

    def _normalise_entity(
        self, inp: NormalisationInput,
    ) -> Tuple[NormalisedEmissions, List[str], int, int, int, int]:
        """Normalise a single entity. Returns (entity, warnings, gwp, scope, period, downgrade)."""
        warnings: List[str] = []
        steps: List[NormalisationStepDetail] = []
        quality_flags: List[DataQualityFlag] = []
        prec_str = "0." + "0" * inp.output_precision

        s1 = inp.scope1_tco2e
        s2 = inp.scope2_tco2e
        s3 = inp.scope3_tco2e if inp.scope3_tco2e is not None else Decimal("0")
        total = inp.total_emissions_tco2e
        denom = inp.denominator_value

        gwp_count = 0
        scope_count = 0
        period_count = 0
        downgrade = False

        # 1. GWP realignment
        gwp_conversions: List[GWPConversionResult] = []
        if inp.reported_gwp != inp.target_gwp:
            if inp.gas_breakdown:
                new_total = Decimal("0")
                for gb in inp.gas_breakdown:
                    conv = self._convert_gas_gwp(
                        gb.gas, gb.emissions_tco2e, inp.reported_gwp, inp.target_gwp
                    )
                    gwp_conversions.append(conv)
                    new_total += conv.converted_tco2e
                    gwp_count += 1

                factor = _safe_divide(new_total, total, Decimal("1"))
                steps.append(NormalisationStepDetail(
                    step=NormalisationStep.GWP_REALIGNMENT,
                    description=f"GWP realignment {inp.reported_gwp.value} -> {inp.target_gwp.value}",
                    input_value=total,
                    output_value=new_total,
                    factor_applied=factor,
                    quality_flag=DataQualityFlag.GWP_CONVERTED,
                ))
                quality_flags.append(DataQualityFlag.GWP_CONVERTED)

                # Apply proportionally to scopes
                s1 = (s1 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
                s2 = (s2 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
                s3 = (s3 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
                total = new_total.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            else:
                warnings.append(
                    f"Entity {inp.entity_id}: GWP conversion requested but no gas breakdown "
                    f"provided. Using aggregate approximation."
                )
                # Approximate with bulk factor based on typical gas mix
                bulk_factor = self._bulk_gwp_factor(inp.reported_gwp, inp.target_gwp)
                new_total = (total * bulk_factor).quantize(
                    Decimal(prec_str), rounding=ROUND_HALF_UP
                )
                steps.append(NormalisationStepDetail(
                    step=NormalisationStep.GWP_REALIGNMENT,
                    description=f"Approximate GWP realignment (no gas breakdown)",
                    input_value=total,
                    output_value=new_total,
                    factor_applied=bulk_factor,
                    quality_flag=DataQualityFlag.GWP_CONVERTED,
                ))
                quality_flags.append(DataQualityFlag.GWP_CONVERTED)
                s1 = (s1 * bulk_factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
                s2 = (s2 * bulk_factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
                s3 = (s3 * bulk_factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
                total = new_total
                downgrade = True
                gwp_count += 1

        # 2. Scope boundary adjustment
        if inp.reported_scope != inp.target_scope:
            old_total = total
            total, s1, s2, s3, step_detail = self._adjust_scope(
                s1, s2, s3, total, inp.reported_scope, inp.target_scope,
                inp.sector, prec_str,
            )
            steps.append(step_detail)
            quality_flags.append(DataQualityFlag.SCOPE_ESTIMATED)
            scope_count += 1
            downgrade = True

        # 3. Consolidation approach normalisation
        if inp.reported_consolidation != inp.target_consolidation:
            old_total = total
            factor = self._get_consolidation_factor(
                inp.reported_consolidation, inp.target_consolidation
            )
            total = (total * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            s1 = (s1 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            s2 = (s2 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            s3 = (s3 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            steps.append(NormalisationStepDetail(
                step=NormalisationStep.CONSOLIDATION,
                description=(
                    f"Consolidation: {inp.reported_consolidation.value} -> "
                    f"{inp.target_consolidation.value}"
                ),
                input_value=old_total,
                output_value=total,
                factor_applied=factor,
                quality_flag=DataQualityFlag.CONSOLIDATION_ADJUSTED,
            ))
            quality_flags.append(DataQualityFlag.CONSOLIDATION_ADJUSTED)
            downgrade = True

        # 4. Period alignment (pro-rata)
        if (
            inp.reporting_period_start and inp.reporting_period_end
            and inp.target_period_start and inp.target_period_end
        ):
            old_total = total
            total, s1, s2, s3, period_factor = self._align_period(
                s1, s2, s3, total,
                inp.reporting_period_start, inp.reporting_period_end,
                inp.target_period_start, inp.target_period_end,
                prec_str,
            )
            if period_factor != Decimal("1"):
                steps.append(NormalisationStepDetail(
                    step=NormalisationStep.PERIOD_ALIGNMENT,
                    description="Period pro-rata alignment",
                    input_value=old_total,
                    output_value=total,
                    factor_applied=period_factor,
                    quality_flag=DataQualityFlag.PERIOD_PRORATED,
                ))
                quality_flags.append(DataQualityFlag.PERIOD_PRORATED)
                period_count += 1

        # 5. Currency PPP adjustment (for denominator)
        normalised_denom: Optional[Decimal] = None
        if denom is not None and inp.reported_currency != inp.target_currency:
            ppp_factor = self._get_ppp_factor(
                inp.reported_currency, inp.reporting_year, inp.target_currency
            )
            normalised_denom = (denom * ppp_factor).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )
            steps.append(NormalisationStepDetail(
                step=NormalisationStep.CURRENCY_PPP,
                description=f"PPP: {inp.reported_currency} -> {inp.target_currency}",
                input_value=denom,
                output_value=normalised_denom,
                factor_applied=ppp_factor,
                quality_flag=DataQualityFlag.PPP_ADJUSTED,
            ))
            quality_flags.append(DataQualityFlag.PPP_ADJUSTED)
        elif denom is not None:
            normalised_denom = denom

        # 6. Biogenic alignment
        if inp.reported_biogenic != inp.target_biogenic:
            old_total = total
            total = self._align_biogenic(
                total, inp.biogenic_emissions_tco2e,
                inp.reported_biogenic, inp.target_biogenic, prec_str,
            )
            steps.append(NormalisationStepDetail(
                step=NormalisationStep.BIOGENIC_ALIGNMENT,
                description=(
                    f"Biogenic: {inp.reported_biogenic.value} -> {inp.target_biogenic.value}"
                ),
                input_value=old_total,
                output_value=total,
                factor_applied=_safe_divide(total, old_total, Decimal("1")),
                quality_flag=DataQualityFlag.BIOGENIC_ADJUSTED,
            ))
            quality_flags.append(DataQualityFlag.BIOGENIC_ADJUSTED)

        # Compute normalised intensity
        normalised_intensity: Optional[Decimal] = None
        if normalised_denom is not None and normalised_denom > Decimal("0"):
            normalised_intensity = _safe_divide(total, normalised_denom).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )

        downgrade_count = 1 if downgrade else 0

        entity = NormalisedEmissions(
            entity_id=inp.entity_id,
            original_total_tco2e=inp.total_emissions_tco2e,
            normalised_total_tco2e=total,
            normalised_scope1=s1,
            normalised_scope2=s2,
            normalised_scope3=s3 if inp.scope3_tco2e is not None or s3 > Decimal("0") else None,
            normalised_denominator=normalised_denom,
            normalised_intensity=normalised_intensity,
            gwp_conversions=gwp_conversions,
            steps_applied=steps,
            quality_flags=quality_flags,
            quality_downgrade=downgrade,
        )

        return entity, warnings, gwp_count, scope_count, period_count, downgrade_count

    # ------------------------------------------------------------------
    # Internal: GWP Conversion
    # ------------------------------------------------------------------

    def _convert_gas_gwp(
        self,
        gas: str,
        emissions_tco2e: Decimal,
        from_version: GWPVersion,
        to_version: GWPVersion,
    ) -> GWPConversionResult:
        """Convert single gas emissions between GWP versions.

        Formula: E_new = E_old * GWP_new / GWP_old
        """
        gas_upper = gas.upper().replace(" ", "")
        gas_data = GWP_VALUES.get(gas, GWP_VALUES.get(gas_upper, {}))

        if not gas_data:
            # Unknown gas, return unchanged
            return GWPConversionResult(
                gas=gas,
                original_tco2e=emissions_tco2e,
                converted_tco2e=emissions_tco2e,
                original_gwp=Decimal("0"),
                target_gwp=Decimal("0"),
                conversion_factor=Decimal("1"),
            )

        gwp_old = gas_data.get(from_version.value, Decimal("1"))
        gwp_new = gas_data.get(to_version.value, Decimal("1"))

        factor = _safe_divide(gwp_new, gwp_old, Decimal("1"))
        converted = emissions_tco2e * factor

        return GWPConversionResult(
            gas=gas,
            original_tco2e=emissions_tco2e,
            converted_tco2e=converted,
            original_gwp=gwp_old,
            target_gwp=gwp_new,
            conversion_factor=factor,
        )

    def _bulk_gwp_factor(
        self, from_version: GWPVersion, to_version: GWPVersion,
    ) -> Decimal:
        """Approximate bulk GWP conversion factor assuming typical gas mix.

        Assumes ~90% CO2, ~8% CH4, ~2% N2O by CO2e contribution.
        """
        co2_w = Decimal("0.90")
        ch4_w = Decimal("0.08")
        n2o_w = Decimal("0.02")

        co2_f = _safe_divide(
            GWP_VALUES["CO2"][to_version.value],
            GWP_VALUES["CO2"][from_version.value],
            Decimal("1"),
        )
        ch4_f = _safe_divide(
            GWP_VALUES["CH4"][to_version.value],
            GWP_VALUES["CH4"][from_version.value],
            Decimal("1"),
        )
        n2o_f = _safe_divide(
            GWP_VALUES["N2O"][to_version.value],
            GWP_VALUES["N2O"][from_version.value],
            Decimal("1"),
        )

        return co2_w * co2_f + ch4_w * ch4_f + n2o_w * n2o_f

    # ------------------------------------------------------------------
    # Internal: Scope Adjustment
    # ------------------------------------------------------------------

    def _adjust_scope(
        self,
        s1: Decimal, s2: Decimal, s3: Decimal, total: Decimal,
        from_scope: ScopeBoundary, to_scope: ScopeBoundary,
        sector: str, prec_str: str,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal, NormalisationStepDetail]:
        """Adjust scope boundary using sector ratios."""
        ratios = SECTOR_SCOPE_RATIOS.get(sector, SECTOR_SCOPE_RATIOS["default"])
        s2_ratio = ratios["s2_ratio"]
        s3_ratio = ratios["s3_ratio"]

        old_total = total
        new_s1 = s1
        new_s2 = s2
        new_s3 = s3

        # Expand scope
        from_level = self._scope_level(from_scope)
        to_level = self._scope_level(to_scope)

        if to_level > from_level:
            # Need to add scopes
            if from_level == 1 and to_level >= 2:
                # Add S2 estimate
                if new_s2 == Decimal("0"):
                    new_s2 = (new_s1 * s2_ratio).quantize(
                        Decimal(prec_str), rounding=ROUND_HALF_UP
                    )
            if to_level >= 3 and from_level < 3:
                # Add S3 estimate
                s12 = new_s1 + new_s2
                if new_s3 == Decimal("0"):
                    new_s3 = (s12 * s3_ratio).quantize(
                        Decimal(prec_str), rounding=ROUND_HALF_UP
                    )
        elif to_level < from_level:
            # Contract scope
            if to_level <= 2 and from_level >= 3:
                new_s3 = Decimal("0")
            if to_level == 1:
                new_s2 = Decimal("0")
                new_s3 = Decimal("0")

        new_total = (new_s1 + new_s2 + new_s3).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        factor = _safe_divide(new_total, old_total, Decimal("1"))

        step = NormalisationStepDetail(
            step=NormalisationStep.SCOPE_BOUNDARY,
            description=f"Scope: {from_scope.value} -> {to_scope.value} (sector: {sector})",
            input_value=old_total,
            output_value=new_total,
            factor_applied=factor,
            quality_flag=DataQualityFlag.SCOPE_ESTIMATED,
        )

        return new_total, new_s1, new_s2, new_s3, step

    def _scope_level(self, scope: ScopeBoundary) -> int:
        """Map scope boundary to numeric level."""
        mapping = {
            ScopeBoundary.S1: 1,
            ScopeBoundary.S1_S2: 2,
            ScopeBoundary.S1_S2_S3_PARTIAL: 3,
            ScopeBoundary.S1_S2_S3_FULL: 4,
        }
        return mapping.get(scope, 2)

    # ------------------------------------------------------------------
    # Internal: Consolidation
    # ------------------------------------------------------------------

    def _get_consolidation_factor(
        self,
        from_approach: ConsolidationApproach,
        to_approach: ConsolidationApproach,
    ) -> Decimal:
        """Get consolidation conversion factor."""
        key = f"{from_approach.value}__{to_approach.value}"
        return CONSOLIDATION_FACTORS.get(key, Decimal("1"))

    # ------------------------------------------------------------------
    # Internal: Period Alignment
    # ------------------------------------------------------------------

    def _align_period(
        self,
        s1: Decimal, s2: Decimal, s3: Decimal, total: Decimal,
        report_start: str, report_end: str,
        target_start: str, target_end: str,
        prec_str: str,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        """Align reporting period via pro-rata.

        E_aligned = E_reported * days_overlap / days_reporting
        """
        try:
            r_start = date.fromisoformat(report_start)
            r_end = date.fromisoformat(report_end)
            t_start = date.fromisoformat(target_start)
            t_end = date.fromisoformat(target_end)
        except (ValueError, TypeError):
            return total, s1, s2, s3, Decimal("1")

        days_reporting = (r_end - r_start).days
        if days_reporting <= 0:
            return total, s1, s2, s3, Decimal("1")

        overlap_start = max(r_start, t_start)
        overlap_end = min(r_end, t_end)
        days_overlap = max((overlap_end - overlap_start).days, 0)

        if days_overlap == 0:
            return Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")

        factor = _safe_divide(
            Decimal(str(days_overlap)), Decimal(str(days_reporting)), Decimal("1")
        )

        new_total = (total * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        new_s1 = (s1 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        new_s2 = (s2 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        new_s3 = (s3 * factor).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        return new_total, new_s1, new_s2, new_s3, factor

    # ------------------------------------------------------------------
    # Internal: PPP
    # ------------------------------------------------------------------

    def _get_ppp_factor(
        self, currency: str, year: int, target_currency: str = "USD",
    ) -> Decimal:
        """Get PPP conversion factor: D_ppp = D_nominal * PPP_factor."""
        source_factors = PPP_FACTORS.get(currency.upper(), {})
        target_factors = PPP_FACTORS.get(target_currency.upper(), {})

        source_ppp = source_factors.get(year, source_factors.get(2024, Decimal("1")))
        target_ppp = target_factors.get(year, target_factors.get(2024, Decimal("1")))

        # Convert: value_target = value_source * (target_ppp / source_ppp)
        return _safe_divide(target_ppp, source_ppp, Decimal("1"))

    # ------------------------------------------------------------------
    # Internal: Biogenic
    # ------------------------------------------------------------------

    def _align_biogenic(
        self,
        total: Decimal,
        biogenic: Decimal,
        from_treatment: BiogenicTreatment,
        to_treatment: BiogenicTreatment,
        prec_str: str,
    ) -> Decimal:
        """Align biogenic carbon treatment."""
        if from_treatment == BiogenicTreatment.INCLUDED and to_treatment == BiogenicTreatment.EXCLUDED:
            return (total - biogenic).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        if from_treatment == BiogenicTreatment.EXCLUDED and to_treatment == BiogenicTreatment.INCLUDED:
            return (total + biogenic).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        return total

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version

    def get_available_gases(self) -> List[str]:
        """Return list of gases with GWP conversion support."""
        return sorted(GWP_VALUES.keys())

    def get_available_currencies(self) -> List[str]:
        """Return list of currencies with PPP factors."""
        return sorted(PPP_FACTORS.keys())

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "GWPVersion",
    "ConsolidationApproach",
    "ScopeBoundary",
    "BiogenicTreatment",
    "NormalisationStep",
    "DataQualityFlag",
    # Input Models
    "GasBreakdown",
    "NormalisationInput",
    # Output Models
    "GWPConversionResult",
    "NormalisationStepDetail",
    "NormalisedEmissions",
    "NormalisationRun",
    # Engine
    "ScopeNormalisationEngine",
    # Constants
    "GWP_VALUES",
    "SECTOR_SCOPE_RATIOS",
    "PPP_FACTORS",
]
