# -*- coding: utf-8 -*-
"""
SDAPathwayEngine - PACK-022 Net Zero Acceleration Engine 2
=============================================================

SBTi Sectoral Decarbonization Approach (SDA) for 12 sectors with
intensity convergence pathways, activity growth modeling, and
IEA Net Zero benchmark alignment checking.

The SDA methodology requires that a company's emission intensity
converges towards the sector-specific benchmark by the target year.
The convergence formula ensures all companies within a sector reach
a common intensity level by 2050 (or sector-specific date), regardless
of their starting point.

This engine covers 12 SDA sectors:
    1. Power generation (tCO2e/MWh -> 0.014 by 2050)
    2. Cement (tCO2e/tonne -> 0.119 by 2050)
    3. Steel (tCO2e/tonne -> 0.156 by 2050)
    4. Aluminium (tCO2e/tonne -> 1.31 by 2050)
    5. Pulp & paper (tCO2e/tonne -> 0.175 by 2050)
    6. Transport (road) (gCO2e/pkm -> 5.3 by 2050)
    7. Buildings (commercial) (kgCO2e/m2 -> 3.1 by 2050)
    8. Buildings (residential) (kgCO2e/m2 -> 2.3 by 2050)
    9. Chemicals (tCO2e/tonne)
   10. Aviation (gCO2e/RPK)
   11. Shipping (gCO2e/tkm)
   12. Food & beverage (tCO2e/tonne)

Calculation Methodology:
    SDA convergence formula:
        I(t) = I_sector(t) + (I_company(base) - I_sector(base))
               * ((I_sector(target) - I_sector(t))
               / (I_sector(target) - I_sector(base)))

    Simplified linear convergence (starter approach):
        I(t) = I_company(base) + (I_sector(target) - I_company(base))
               * ((t - base_year) / (target_year - base_year))

    Absolute emissions from intensity:
        E(t) = I(t) * Activity(t)

    Activity growth:
        Activity(t) = Activity(base) * (1 + growth_rate)^(t - base_year)

    ACA comparison:
        E_aca(t) = E(base) * (1 - aca_rate * (t - base_year))

Regulatory References:
    - SBTi Sectoral Decarbonization Approach (SDA) Methodology
    - SBTi Corporate Net-Zero Standard v1.2 (2023)
    - SBTi Sector Guidance: Power, Cement, Steel, Aluminium, etc.
    - IEA Net Zero by 2050 Roadmap (2021) - Sector benchmarks
    - IEA Energy Technology Perspectives (2023)
    - IPCC AR6 WG3 (2022) - Sectoral mitigation pathways

Zero-Hallucination:
    - All convergence calculations use deterministic Decimal arithmetic
    - Sector benchmarks are hard-coded from SBTi/IEA publications
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-022 Net Zero Acceleration
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class SDASector(str, Enum):
    """SBTi SDA sector classification.

    Each sector has its own intensity convergence benchmark.
    """
    POWER_GENERATION = "power_generation"
    CEMENT = "cement"
    STEEL = "steel"
    ALUMINIUM = "aluminium"
    PULP_PAPER = "pulp_paper"
    TRANSPORT_ROAD = "transport_road"
    BUILDINGS_COMMERCIAL = "buildings_commercial"
    BUILDINGS_RESIDENTIAL = "buildings_residential"
    CHEMICALS = "chemicals"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    FOOD_BEVERAGE = "food_beverage"

class IntensityUnit(str, Enum):
    """Intensity metric units by sector."""
    TCO2E_PER_MWH = "tCO2e/MWh"
    TCO2E_PER_TONNE_CEMENT = "tCO2e/t_cement"
    TCO2E_PER_TONNE_STEEL = "tCO2e/t_steel"
    TCO2E_PER_TONNE_ALUMINIUM = "tCO2e/t_aluminium"
    TCO2E_PER_TONNE_PRODUCT = "tCO2e/t_product"
    GCO2E_PER_PKM = "gCO2e/pkm"
    KGCO2E_PER_M2 = "kgCO2e/m2"
    GCO2E_PER_RPK = "gCO2e/RPK"
    GCO2E_PER_TKM = "gCO2e/tkm"

class ActivityMetric(str, Enum):
    """Activity metric types for each sector."""
    MWH_GENERATED = "mwh_generated"
    TONNES_PRODUCED = "tonnes_produced"
    PASSENGER_KM = "passenger_km"
    SQUARE_METERS = "square_meters"
    REVENUE_PASSENGER_KM = "revenue_passenger_km"
    TONNE_KM = "tonne_km"

class PathwayStatus(str, Enum):
    """SDA pathway alignment status."""
    ALIGNED = "aligned"
    ABOVE_PATHWAY = "above_pathway"
    BELOW_PATHWAY = "below_pathway"
    NOT_APPLICABLE = "not_applicable"

# ---------------------------------------------------------------------------
# Constants -- Sector Convergence Benchmarks
# ---------------------------------------------------------------------------

# SDA sector convergence benchmarks.
# Source: SBTi SDA Tool, IEA NZE Scenario (2021), IEA ETP (2023).
# Each sector has: intensity_unit, activity_metric, and benchmark
# values at 2020, 2025, 2030, 2035, 2040, 2045, 2050.
SECTOR_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    SDASector.POWER_GENERATION: {
        "name": "Power Generation",
        "intensity_unit": IntensityUnit.TCO2E_PER_MWH.value,
        "activity_metric": ActivityMetric.MWH_GENERATED.value,
        "benchmarks": {
            2020: Decimal("0.376"),
            2025: Decimal("0.253"),
            2030: Decimal("0.138"),
            2035: Decimal("0.082"),
            2040: Decimal("0.046"),
            2045: Decimal("0.025"),
            2050: Decimal("0.014"),
        },
    },
    SDASector.CEMENT: {
        "name": "Cement",
        "intensity_unit": IntensityUnit.TCO2E_PER_TONNE_CEMENT.value,
        "activity_metric": ActivityMetric.TONNES_PRODUCED.value,
        "benchmarks": {
            2020: Decimal("0.610"),
            2025: Decimal("0.512"),
            2030: Decimal("0.416"),
            2035: Decimal("0.322"),
            2040: Decimal("0.232"),
            2045: Decimal("0.168"),
            2050: Decimal("0.119"),
        },
    },
    SDASector.STEEL: {
        "name": "Steel",
        "intensity_unit": IntensityUnit.TCO2E_PER_TONNE_STEEL.value,
        "activity_metric": ActivityMetric.TONNES_PRODUCED.value,
        "benchmarks": {
            2020: Decimal("1.890"),
            2025: Decimal("1.510"),
            2030: Decimal("1.140"),
            2035: Decimal("0.810"),
            2040: Decimal("0.530"),
            2045: Decimal("0.320"),
            2050: Decimal("0.156"),
        },
    },
    SDASector.ALUMINIUM: {
        "name": "Aluminium",
        "intensity_unit": IntensityUnit.TCO2E_PER_TONNE_ALUMINIUM.value,
        "activity_metric": ActivityMetric.TONNES_PRODUCED.value,
        "benchmarks": {
            2020: Decimal("8.60"),
            2025: Decimal("6.75"),
            2030: Decimal("5.10"),
            2035: Decimal("3.85"),
            2040: Decimal("2.80"),
            2045: Decimal("1.95"),
            2050: Decimal("1.31"),
        },
    },
    SDASector.PULP_PAPER: {
        "name": "Pulp & Paper",
        "intensity_unit": IntensityUnit.TCO2E_PER_TONNE_PRODUCT.value,
        "activity_metric": ActivityMetric.TONNES_PRODUCED.value,
        "benchmarks": {
            2020: Decimal("0.560"),
            2025: Decimal("0.470"),
            2030: Decimal("0.385"),
            2035: Decimal("0.310"),
            2040: Decimal("0.245"),
            2045: Decimal("0.200"),
            2050: Decimal("0.175"),
        },
    },
    SDASector.TRANSPORT_ROAD: {
        "name": "Transport (Road)",
        "intensity_unit": IntensityUnit.GCO2E_PER_PKM.value,
        "activity_metric": ActivityMetric.PASSENGER_KM.value,
        "benchmarks": {
            2020: Decimal("98.0"),
            2025: Decimal("72.5"),
            2030: Decimal("49.0"),
            2035: Decimal("32.0"),
            2040: Decimal("19.5"),
            2045: Decimal("10.8"),
            2050: Decimal("5.3"),
        },
    },
    SDASector.BUILDINGS_COMMERCIAL: {
        "name": "Buildings (Commercial)",
        "intensity_unit": IntensityUnit.KGCO2E_PER_M2.value,
        "activity_metric": ActivityMetric.SQUARE_METERS.value,
        "benchmarks": {
            2020: Decimal("38.0"),
            2025: Decimal("27.5"),
            2030: Decimal("18.5"),
            2035: Decimal("12.8"),
            2040: Decimal("8.2"),
            2045: Decimal("5.1"),
            2050: Decimal("3.1"),
        },
    },
    SDASector.BUILDINGS_RESIDENTIAL: {
        "name": "Buildings (Residential)",
        "intensity_unit": IntensityUnit.KGCO2E_PER_M2.value,
        "activity_metric": ActivityMetric.SQUARE_METERS.value,
        "benchmarks": {
            2020: Decimal("28.0"),
            2025: Decimal("20.8"),
            2030: Decimal("14.5"),
            2035: Decimal("10.2"),
            2040: Decimal("6.5"),
            2045: Decimal("3.9"),
            2050: Decimal("2.3"),
        },
    },
    SDASector.CHEMICALS: {
        "name": "Chemicals",
        "intensity_unit": IntensityUnit.TCO2E_PER_TONNE_PRODUCT.value,
        "activity_metric": ActivityMetric.TONNES_PRODUCED.value,
        "benchmarks": {
            2020: Decimal("0.850"),
            2025: Decimal("0.710"),
            2030: Decimal("0.575"),
            2035: Decimal("0.450"),
            2040: Decimal("0.340"),
            2045: Decimal("0.245"),
            2050: Decimal("0.170"),
        },
    },
    SDASector.AVIATION: {
        "name": "Aviation",
        "intensity_unit": IntensityUnit.GCO2E_PER_RPK.value,
        "activity_metric": ActivityMetric.REVENUE_PASSENGER_KM.value,
        "benchmarks": {
            2020: Decimal("90.0"),
            2025: Decimal("76.0"),
            2030: Decimal("61.0"),
            2035: Decimal("46.0"),
            2040: Decimal("33.0"),
            2045: Decimal("22.0"),
            2050: Decimal("13.0"),
        },
    },
    SDASector.SHIPPING: {
        "name": "Shipping",
        "intensity_unit": IntensityUnit.GCO2E_PER_TKM.value,
        "activity_metric": ActivityMetric.TONNE_KM.value,
        "benchmarks": {
            2020: Decimal("7.10"),
            2025: Decimal("5.85"),
            2030: Decimal("4.60"),
            2035: Decimal("3.45"),
            2040: Decimal("2.40"),
            2045: Decimal("1.55"),
            2050: Decimal("0.85"),
        },
    },
    SDASector.FOOD_BEVERAGE: {
        "name": "Food & Beverage",
        "intensity_unit": IntensityUnit.TCO2E_PER_TONNE_PRODUCT.value,
        "activity_metric": ActivityMetric.TONNES_PRODUCED.value,
        "benchmarks": {
            2020: Decimal("0.480"),
            2025: Decimal("0.400"),
            2030: Decimal("0.325"),
            2035: Decimal("0.255"),
            2040: Decimal("0.195"),
            2045: Decimal("0.150"),
            2050: Decimal("0.115"),
        },
    },
}

# ACA reference rate for comparison (SBTi 1.5C).
ACA_ANNUAL_RATE: Decimal = Decimal("0.042")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class SDAInput(BaseModel):
    """Input data for SDA pathway calculation.

    Attributes:
        entity_name: Reporting entity name.
        sector: SDA sector classification.
        base_year: Base year.
        target_year: Target convergence year (default 2050).
        base_year_intensity: Company's emission intensity in base year.
        base_year_activity: Activity level in base year (units depend on sector).
        base_year_emissions_tco2e: Absolute emissions in base year.
        activity_growth_rate_pct: Annual activity growth rate (%).
        projection_interval_years: Year interval for trajectory.
        near_term_target_year: Near-term milestone year (default 2030).
        include_aca_comparison: Whether to include ACA comparison.
        include_iea_alignment: Whether to include IEA NZE alignment check.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    sector: SDASector = Field(..., description="SDA sector")
    base_year: int = Field(..., ge=2015, le=2030, description="Base year")
    target_year: int = Field(
        default=2050, ge=2030, le=2070, description="Target year"
    )
    base_year_intensity: Decimal = Field(
        ..., gt=Decimal("0"), description="Base year emission intensity"
    )
    base_year_activity: Decimal = Field(
        ..., gt=Decimal("0"), description="Base year activity level"
    )
    base_year_emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Base year absolute emissions"
    )
    activity_growth_rate_pct: Decimal = Field(
        default=Decimal("2.0"), ge=Decimal("-5"), le=Decimal("15"),
        description="Annual activity growth (%)",
    )
    projection_interval_years: int = Field(
        default=5, ge=1, le=10, description="Projection interval"
    )
    near_term_target_year: int = Field(
        default=2030, ge=2025, le=2040, description="Near-term target year"
    )
    include_aca_comparison: bool = Field(
        default=True, description="Include ACA comparison"
    )
    include_iea_alignment: bool = Field(
        default=True, description="Include IEA NZE alignment check"
    )

    @field_validator("target_year")
    @classmethod
    def validate_target(cls, v: int, info: Any) -> int:
        """Validate target year after base year."""
        base = info.data.get("base_year", 2015)
        if v <= base:
            raise ValueError(f"target_year ({v}) must be after base_year ({base})")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class IntensityPoint(BaseModel):
    """A single point on the intensity trajectory.

    Attributes:
        year: Projection year.
        company_intensity: Company's projected intensity at this year.
        sector_benchmark: Sector benchmark intensity at this year.
        gap_to_benchmark: Company intensity - sector benchmark.
        convergence_pct: How far company has converged (0-100).
        status: Whether company is aligned/above/below pathway.
    """
    year: int = Field(default=0)
    company_intensity: Decimal = Field(default=Decimal("0"))
    sector_benchmark: Decimal = Field(default=Decimal("0"))
    gap_to_benchmark: Decimal = Field(default=Decimal("0"))
    convergence_pct: Decimal = Field(default=Decimal("0"))
    status: str = Field(default=PathwayStatus.NOT_APPLICABLE.value)

class AbsolutePoint(BaseModel):
    """Absolute emissions at a projection year.

    Attributes:
        year: Projection year.
        activity_level: Projected activity level.
        intensity: Projected intensity.
        absolute_emissions_tco2e: Intensity * Activity.
        reduction_from_base_pct: Reduction from base year (%).
    """
    year: int = Field(default=0)
    activity_level: Decimal = Field(default=Decimal("0"))
    intensity: Decimal = Field(default=Decimal("0"))
    absolute_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_base_pct: Decimal = Field(default=Decimal("0"))

class ACAComparisonPoint(BaseModel):
    """ACA pathway comparison at a projection year.

    Attributes:
        year: Projection year.
        sda_emissions_tco2e: SDA pathway emissions.
        aca_emissions_tco2e: ACA pathway emissions.
        delta_tco2e: SDA - ACA (positive = SDA is higher).
        sda_more_ambitious: Whether SDA gives lower emissions.
    """
    year: int = Field(default=0)
    sda_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    aca_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    delta_tco2e: Decimal = Field(default=Decimal("0"))
    sda_more_ambitious: bool = Field(default=False)

class IEAAlignmentCheck(BaseModel):
    """IEA Net Zero benchmark alignment assessment.

    Attributes:
        year: Reference year checked.
        company_intensity: Company intensity.
        iea_benchmark: IEA NZE benchmark.
        aligned: Whether company is at or below IEA benchmark.
        gap_pct: Gap as percentage above benchmark.
    """
    year: int = Field(default=0)
    company_intensity: Decimal = Field(default=Decimal("0"))
    iea_benchmark: Decimal = Field(default=Decimal("0"))
    aligned: bool = Field(default=False)
    gap_pct: Decimal = Field(default=Decimal("0"))

class SDAResult(BaseModel):
    """Complete SDA pathway calculation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        sector: SDA sector.
        sector_name: Human-readable sector name.
        intensity_unit: Intensity metric unit.
        activity_metric: Activity metric type.
        base_year: Base year.
        target_year: Target year.
        base_year_intensity: Starting intensity.
        target_year_intensity: Required target intensity.
        annual_convergence_rate_pct: Annual intensity reduction rate.
        intensity_trajectory: Year-by-year intensity convergence.
        absolute_trajectory: Year-by-year absolute emissions.
        activity_projections: Activity level projections.
        aca_comparison: ACA vs SDA comparison (if included).
        iea_alignment_checks: IEA NZE alignment checks (if included).
        total_cumulative_abatement_tco2e: Total abatement vs baseline.
        near_term_intensity: Intensity at near-term target year.
        near_term_reduction_pct: Intensity reduction at near-term year.
        pathway_status: Overall alignment status.
        recommendations: Improvement recommendations.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    sector: str = Field(default="")
    sector_name: str = Field(default="")
    intensity_unit: str = Field(default="")
    activity_metric: str = Field(default="")
    base_year: int = Field(default=0)
    target_year: int = Field(default=0)
    base_year_intensity: Decimal = Field(default=Decimal("0"))
    target_year_intensity: Decimal = Field(default=Decimal("0"))
    annual_convergence_rate_pct: Decimal = Field(default=Decimal("0"))
    intensity_trajectory: List[IntensityPoint] = Field(default_factory=list)
    absolute_trajectory: List[AbsolutePoint] = Field(default_factory=list)
    activity_projections: Dict[int, Decimal] = Field(default_factory=dict)
    aca_comparison: List[ACAComparisonPoint] = Field(default_factory=list)
    iea_alignment_checks: List[IEAAlignmentCheck] = Field(default_factory=list)
    total_cumulative_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    near_term_intensity: Decimal = Field(default=Decimal("0"))
    near_term_reduction_pct: Decimal = Field(default=Decimal("0"))
    pathway_status: str = Field(default=PathwayStatus.NOT_APPLICABLE.value)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SDAPathwayEngine:
    """SBTi Sectoral Decarbonization Approach pathway engine.

    Calculates intensity convergence pathways for 12 sectors,
    projects absolute emissions from intensity * activity, and
    compares SDA with ACA pathways.

    All calculations use Decimal arithmetic.  No LLM in any path.

    Usage::

        engine = SDAPathwayEngine()
        result = engine.calculate(sda_input)
        for pt in result.intensity_trajectory:
            print(f"{pt.year}: {pt.company_intensity} {result.intensity_unit}")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: SDAInput) -> SDAResult:
        """Run complete SDA pathway calculation.

        Args:
            data: Validated SDA input.

        Returns:
            SDAResult with intensity/absolute trajectories and comparisons.
        """
        t0 = time.perf_counter()
        logger.info(
            "SDA pathway: entity=%s, sector=%s, base=%d, target=%d",
            data.entity_name, data.sector.value, data.base_year,
            data.target_year,
        )

        sector_data = SECTOR_BENCHMARKS.get(data.sector)
        if sector_data is None:
            raise ValueError(f"Unknown SDA sector: {data.sector}")

        benchmarks = sector_data["benchmarks"]
        projection_years = self._build_projection_years(
            data.base_year, data.target_year, data.projection_interval_years
        )

        # Step 1: Get target intensity (sector benchmark at target year)
        target_intensity = self._interpolate_benchmark(
            benchmarks, data.target_year
        )

        # Step 2: Calculate convergence rate
        total_years = _decimal(data.target_year - data.base_year)
        intensity_reduction = data.base_year_intensity - target_intensity
        annual_convergence_rate = _safe_divide(
            _safe_pct(intensity_reduction, data.base_year_intensity),
            total_years,
        )

        # Step 3: Build intensity trajectory
        intensity_trajectory = self._build_intensity_trajectory(
            data, benchmarks, projection_years, target_intensity
        )

        # Step 4: Project activity levels
        activity_projections = self._project_activity(
            data, projection_years
        )

        # Step 5: Build absolute trajectory
        absolute_trajectory = self._build_absolute_trajectory(
            data, intensity_trajectory, activity_projections
        )

        # Step 6: ACA comparison
        aca_comparison: List[ACAComparisonPoint] = []
        if data.include_aca_comparison:
            aca_comparison = self._build_aca_comparison(
                data, absolute_trajectory
            )

        # Step 7: IEA alignment checks
        iea_checks: List[IEAAlignmentCheck] = []
        if data.include_iea_alignment:
            iea_checks = self._check_iea_alignment(
                data, intensity_trajectory, benchmarks
            )

        # Step 8: Cumulative abatement
        cumulative_abatement = self._compute_cumulative_abatement(
            data, absolute_trajectory
        )

        # Step 9: Near-term metrics
        near_term_intensity = self._get_intensity_at_year(
            intensity_trajectory, data.near_term_target_year
        )
        near_term_reduction = _safe_pct(
            data.base_year_intensity - near_term_intensity,
            data.base_year_intensity,
        )

        # Step 10: Overall pathway status
        pathway_status = self._assess_pathway_status(
            data, intensity_trajectory
        )

        # Step 11: Recommendations
        recommendations = self._generate_recommendations(
            data, intensity_trajectory, aca_comparison, pathway_status
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        # Convert activity projections to int keys for serialization
        activity_proj_output: Dict[int, Decimal] = {}
        for year, val in activity_projections.items():
            activity_proj_output[year] = _round_val(val)

        result = SDAResult(
            entity_name=data.entity_name,
            sector=data.sector.value,
            sector_name=sector_data["name"],
            intensity_unit=sector_data["intensity_unit"],
            activity_metric=sector_data["activity_metric"],
            base_year=data.base_year,
            target_year=data.target_year,
            base_year_intensity=_round_val(data.base_year_intensity),
            target_year_intensity=_round_val(target_intensity),
            annual_convergence_rate_pct=_round_val(annual_convergence_rate, 3),
            intensity_trajectory=intensity_trajectory,
            absolute_trajectory=absolute_trajectory,
            activity_projections=activity_proj_output,
            aca_comparison=aca_comparison,
            iea_alignment_checks=iea_checks,
            total_cumulative_abatement_tco2e=_round_val(cumulative_abatement),
            near_term_intensity=_round_val(near_term_intensity),
            near_term_reduction_pct=_round_val(near_term_reduction, 2),
            pathway_status=pathway_status,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "SDA complete: sector=%s, target_intensity=%s, "
            "convergence_rate=%.3f%%/yr, status=%s",
            data.sector.value, str(target_intensity),
            float(annual_convergence_rate), pathway_status,
        )
        return result

    # ------------------------------------------------------------------ #
    # Projection Years                                                    #
    # ------------------------------------------------------------------ #

    def _build_projection_years(
        self, base_year: int, target_year: int, interval: int
    ) -> List[int]:
        """Build sorted list of projection years."""
        years = set()
        y = base_year
        while y <= target_year:
            years.add(y)
            y += interval
        years.add(target_year)
        for ref in [2025, 2030, 2035, 2040, 2045, 2050]:
            if base_year <= ref <= target_year:
                years.add(ref)
        return sorted(years)

    # ------------------------------------------------------------------ #
    # Benchmark Interpolation                                             #
    # ------------------------------------------------------------------ #

    def _interpolate_benchmark(
        self,
        benchmarks: Dict[int, Decimal],
        year: int,
    ) -> Decimal:
        """Interpolate sector benchmark for a given year.

        Uses linear interpolation between the two nearest benchmark years.

        Args:
            benchmarks: Year-to-intensity benchmark map.
            year: Target year.

        Returns:
            Interpolated benchmark intensity.
        """
        years_sorted = sorted(benchmarks.keys())

        # Exact match
        if year in benchmarks:
            return benchmarks[year]

        # Clamp to range
        if year <= years_sorted[0]:
            return benchmarks[years_sorted[0]]
        if year >= years_sorted[-1]:
            return benchmarks[years_sorted[-1]]

        # Find bracketing years
        lower_year = years_sorted[0]
        upper_year = years_sorted[-1]
        for i, y in enumerate(years_sorted):
            if y <= year:
                lower_year = y
            if y > year:
                upper_year = y
                break

        lower_val = benchmarks[lower_year]
        upper_val = benchmarks[upper_year]
        span = _decimal(upper_year - lower_year)
        elapsed = _decimal(year - lower_year)

        interpolated = lower_val + (upper_val - lower_val) * _safe_divide(elapsed, span)
        return interpolated

    # ------------------------------------------------------------------ #
    # Intensity Trajectory                                                #
    # ------------------------------------------------------------------ #

    def _build_intensity_trajectory(
        self,
        data: SDAInput,
        benchmarks: Dict[int, Decimal],
        projection_years: List[int],
        target_intensity: Decimal,
    ) -> List[IntensityPoint]:
        """Build the SDA intensity convergence trajectory.

        Uses linear convergence from company base intensity to sector
        target intensity.

        Args:
            data: SDA input.
            benchmarks: Sector benchmarks.
            projection_years: Years to project.
            target_intensity: Final target intensity.

        Returns:
            List of IntensityPoint entries.
        """
        trajectory: List[IntensityPoint] = []
        total_years = _decimal(max(data.target_year - data.base_year, 1))
        company_base = data.base_year_intensity

        for year in projection_years:
            elapsed = _decimal(year - data.base_year)
            fraction = _safe_divide(elapsed, total_years)

            # Linear convergence
            company_intensity = company_base + (
                target_intensity - company_base
            ) * fraction
            company_intensity = max(Decimal("0"), company_intensity)

            # Sector benchmark at this year
            sector_bm = self._interpolate_benchmark(benchmarks, year)

            gap = company_intensity - sector_bm

            # Convergence percentage
            total_gap = company_base - target_intensity
            converged = company_base - company_intensity
            convergence_pct = _safe_pct(converged, total_gap) if total_gap > Decimal("0") else Decimal("100")
            convergence_pct = min(convergence_pct, Decimal("100"))

            # Status
            tolerance = sector_bm * Decimal("0.05")  # 5% tolerance
            if company_intensity <= sector_bm + tolerance:
                status = PathwayStatus.ALIGNED.value
            elif company_intensity > sector_bm:
                status = PathwayStatus.ABOVE_PATHWAY.value
            else:
                status = PathwayStatus.BELOW_PATHWAY.value

            trajectory.append(IntensityPoint(
                year=year,
                company_intensity=_round_val(company_intensity),
                sector_benchmark=_round_val(sector_bm),
                gap_to_benchmark=_round_val(gap),
                convergence_pct=_round_val(convergence_pct, 2),
                status=status,
            ))

        return trajectory

    # ------------------------------------------------------------------ #
    # Activity Projection                                                 #
    # ------------------------------------------------------------------ #

    def _project_activity(
        self,
        data: SDAInput,
        projection_years: List[int],
    ) -> Dict[int, Decimal]:
        """Project activity levels using compound growth.

        Activity(t) = Activity(base) * (1 + rate)^(t - base)

        Args:
            data: SDA input with growth rate.
            projection_years: Years to project.

        Returns:
            Dict mapping year to projected activity level.
        """
        projections: Dict[int, Decimal] = {}
        growth_rate = data.activity_growth_rate_pct / Decimal("100")

        for year in projection_years:
            elapsed = year - data.base_year
            growth_factor = (Decimal("1") + growth_rate) ** elapsed
            projections[year] = _round_val(
                data.base_year_activity * growth_factor
            )

        return projections

    # ------------------------------------------------------------------ #
    # Absolute Trajectory                                                 #
    # ------------------------------------------------------------------ #

    def _build_absolute_trajectory(
        self,
        data: SDAInput,
        intensity_trajectory: List[IntensityPoint],
        activity_projections: Dict[int, Decimal],
    ) -> List[AbsolutePoint]:
        """Build absolute emissions trajectory from intensity * activity.

        Args:
            data: SDA input.
            intensity_trajectory: Intensity convergence points.
            activity_projections: Activity level projections.

        Returns:
            List of AbsolutePoint entries.
        """
        trajectory: List[AbsolutePoint] = []

        for ip in intensity_trajectory:
            activity = activity_projections.get(
                ip.year, data.base_year_activity
            )
            absolute = ip.company_intensity * activity

            # Handle unit conversion for sectors with g or kg units
            # For gCO2e and kgCO2e, convert to tCO2e
            sector_data = SECTOR_BENCHMARKS.get(data.sector, {})
            unit = sector_data.get("intensity_unit", "")

            if "gCO2e" in unit:
                absolute = absolute / Decimal("1000000")  # g to t
            elif "kgCO2e" in unit:
                absolute = absolute / Decimal("1000")  # kg to t

            reduction_pct = _safe_pct(
                data.base_year_emissions_tco2e - absolute,
                data.base_year_emissions_tco2e,
            )

            trajectory.append(AbsolutePoint(
                year=ip.year,
                activity_level=_round_val(activity),
                intensity=ip.company_intensity,
                absolute_emissions_tco2e=_round_val(absolute),
                reduction_from_base_pct=_round_val(reduction_pct, 2),
            ))

        return trajectory

    # ------------------------------------------------------------------ #
    # ACA Comparison                                                      #
    # ------------------------------------------------------------------ #

    def _build_aca_comparison(
        self,
        data: SDAInput,
        absolute_trajectory: List[AbsolutePoint],
    ) -> List[ACAComparisonPoint]:
        """Build ACA vs SDA comparison points.

        ACA: E(t) = E(base) * (1 - 0.042 * (t - base))

        Args:
            data: SDA input.
            absolute_trajectory: SDA absolute trajectory.

        Returns:
            List of ACAComparisonPoint entries.
        """
        comparison: List[ACAComparisonPoint] = []

        for ap in absolute_trajectory:
            elapsed = _decimal(ap.year - data.base_year)
            reduction_factor = max(
                Decimal("0"),
                Decimal("1") - ACA_ANNUAL_RATE * elapsed / Decimal("100")
            )
            aca_emissions = data.base_year_emissions_tco2e * reduction_factor

            delta = ap.absolute_emissions_tco2e - aca_emissions
            sda_more_ambitious = ap.absolute_emissions_tco2e < aca_emissions

            comparison.append(ACAComparisonPoint(
                year=ap.year,
                sda_emissions_tco2e=ap.absolute_emissions_tco2e,
                aca_emissions_tco2e=_round_val(aca_emissions),
                delta_tco2e=_round_val(delta),
                sda_more_ambitious=sda_more_ambitious,
            ))

        return comparison

    # ------------------------------------------------------------------ #
    # IEA Alignment                                                       #
    # ------------------------------------------------------------------ #

    def _check_iea_alignment(
        self,
        data: SDAInput,
        trajectory: List[IntensityPoint],
        benchmarks: Dict[int, Decimal],
    ) -> List[IEAAlignmentCheck]:
        """Check alignment with IEA NZE benchmarks at key years.

        Args:
            data: SDA input.
            trajectory: Intensity trajectory.
            benchmarks: Sector benchmarks (used as IEA proxy).

        Returns:
            List of IEAAlignmentCheck entries.
        """
        checks: List[IEAAlignmentCheck] = []
        check_years = [2030, 2040, 2050]

        for year in check_years:
            if year < data.base_year or year > data.target_year:
                continue

            company_int = self._get_intensity_at_year(trajectory, year)
            iea_bm = self._interpolate_benchmark(benchmarks, year)

            aligned = company_int <= iea_bm
            gap_pct = Decimal("0")
            if not aligned and iea_bm > Decimal("0"):
                gap_pct = _safe_pct(company_int - iea_bm, iea_bm)

            checks.append(IEAAlignmentCheck(
                year=year,
                company_intensity=_round_val(company_int),
                iea_benchmark=_round_val(iea_bm),
                aligned=aligned,
                gap_pct=_round_val(gap_pct, 2),
            ))

        return checks

    # ------------------------------------------------------------------ #
    # Cumulative Abatement                                                #
    # ------------------------------------------------------------------ #

    def _compute_cumulative_abatement(
        self,
        data: SDAInput,
        trajectory: List[AbsolutePoint],
    ) -> Decimal:
        """Compute total cumulative abatement vs fixed-base baseline.

        Args:
            data: SDA input.
            trajectory: Absolute trajectory.

        Returns:
            Total cumulative abatement in tCO2e.
        """
        cumulative = Decimal("0")
        prev_year = data.base_year

        for ap in trajectory:
            if ap.year == data.base_year:
                prev_year = ap.year
                continue
            years_gap = _decimal(ap.year - prev_year)
            abated_per_year = data.base_year_emissions_tco2e - ap.absolute_emissions_tco2e
            cumulative += abated_per_year * years_gap
            prev_year = ap.year

        return max(Decimal("0"), cumulative)

    # ------------------------------------------------------------------ #
    # Pathway Status                                                      #
    # ------------------------------------------------------------------ #

    def _assess_pathway_status(
        self,
        data: SDAInput,
        trajectory: List[IntensityPoint],
    ) -> str:
        """Assess overall pathway alignment status.

        Args:
            data: SDA input.
            trajectory: Intensity trajectory.

        Returns:
            PathwayStatus string value.
        """
        if not trajectory:
            return PathwayStatus.NOT_APPLICABLE.value

        # Check the target year point
        target_point = None
        for pt in trajectory:
            if pt.year == data.target_year:
                target_point = pt
                break

        if target_point is None:
            target_point = trajectory[-1]

        return target_point.status

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_intensity_at_year(
        self,
        trajectory: List[IntensityPoint],
        year: int,
    ) -> Decimal:
        """Get company intensity at a specific year from trajectory.

        Args:
            trajectory: Intensity trajectory.
            year: Target year.

        Returns:
            Company intensity at the year.
        """
        for pt in trajectory:
            if pt.year == year:
                return pt.company_intensity

        # Interpolate from nearest
        if not trajectory:
            return Decimal("0")

        closest = min(trajectory, key=lambda p: abs(p.year - year))
        return closest.company_intensity

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: SDAInput,
        trajectory: List[IntensityPoint],
        aca_comparison: List[ACAComparisonPoint],
        pathway_status: str,
    ) -> List[str]:
        """Generate sector-specific recommendations.

        Args:
            data: SDA input.
            trajectory: Intensity trajectory.
            aca_comparison: ACA comparison points.
            pathway_status: Current pathway status.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if pathway_status == PathwayStatus.ABOVE_PATHWAY.value:
            recs.append(
                "Company intensity is above the sector benchmark pathway. "
                "Accelerate emission intensity reduction to achieve convergence."
            )

        if pathway_status == PathwayStatus.ALIGNED.value:
            recs.append(
                "Company intensity is aligned with the sector benchmark. "
                "Maintain current trajectory and monitor for deviations."
            )

        # Check if ACA is more ambitious
        if aca_comparison:
            aca_more_count = sum(
                1 for c in aca_comparison if not c.sda_more_ambitious
            )
            if aca_more_count > len(aca_comparison) / 2:
                recs.append(
                    "The ACA pathway yields lower emissions than SDA for this "
                    "entity. Consider using ACA for more ambitious targeting."
                )

        # Activity growth concern
        if data.activity_growth_rate_pct > Decimal("3.0"):
            recs.append(
                f"High activity growth ({data.activity_growth_rate_pct}%/yr) "
                "may offset intensity improvements. Absolute emissions could "
                "increase despite falling intensity. Consider absolute "
                "emission caps alongside intensity targets."
            )

        # Sector-specific guidance
        sector_recs = {
            SDASector.POWER_GENERATION: (
                "Prioritize renewable energy transition and phase out "
                "fossil fuel generation capacity."
            ),
            SDASector.CEMENT: (
                "Explore clinker substitution, alternative fuels, and "
                "carbon capture to reduce process emissions."
            ),
            SDASector.STEEL: (
                "Evaluate electric arc furnace (EAF) transition and "
                "green hydrogen-based direct reduced iron (H2-DRI)."
            ),
            SDASector.ALUMINIUM: (
                "Focus on renewable-powered smelting and inert anode "
                "technology to eliminate process CO2."
            ),
            SDASector.TRANSPORT_ROAD: (
                "Accelerate fleet electrification and modal shift to "
                "rail and public transport."
            ),
            SDASector.BUILDINGS_COMMERCIAL: (
                "Invest in building envelope upgrades, heat pump "
                "installation, and on-site renewables."
            ),
            SDASector.BUILDINGS_RESIDENTIAL: (
                "Prioritize deep retrofit programs, heat pump rollout, "
                "and net-zero new builds."
            ),
            SDASector.AVIATION: (
                "Invest in sustainable aviation fuel (SAF), fleet "
                "renewal, and operational efficiency."
            ),
            SDASector.SHIPPING: (
                "Explore ammonia/methanol propulsion, wind-assisted "
                "propulsion, and slow steaming optimization."
            ),
        }

        sector_rec = sector_recs.get(data.sector)
        if sector_rec:
            recs.append(sector_rec)

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_sector_info(self, sector: SDASector) -> Dict[str, Any]:
        """Get sector information and benchmarks.

        Args:
            sector: SDA sector.

        Returns:
            Dict with sector name, unit, and benchmark values.

        Raises:
            ValueError: If sector is not found.
        """
        data = SECTOR_BENCHMARKS.get(sector)
        if data is None:
            raise ValueError(f"Unknown sector: {sector}")
        return {
            "name": data["name"],
            "intensity_unit": data["intensity_unit"],
            "activity_metric": data["activity_metric"],
            "benchmarks": {
                str(k): str(v) for k, v in data["benchmarks"].items()
            },
        }

    def get_supported_sectors(self) -> List[Dict[str, str]]:
        """List all supported SDA sectors.

        Returns:
            List of dicts with sector value and name.
        """
        return [
            {
                "sector": sector.value,
                "name": SECTOR_BENCHMARKS[sector]["name"],
                "intensity_unit": SECTOR_BENCHMARKS[sector]["intensity_unit"],
            }
            for sector in SDASector
            if sector in SECTOR_BENCHMARKS
        ]

    def get_summary(self, result: SDAResult) -> Dict[str, Any]:
        """Generate concise summary of SDA result.

        Args:
            result: SDAResult to summarize.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "entity_name": result.entity_name,
            "sector": result.sector,
            "sector_name": result.sector_name,
            "base_year_intensity": str(result.base_year_intensity),
            "target_year_intensity": str(result.target_year_intensity),
            "convergence_rate_pct_yr": str(result.annual_convergence_rate_pct),
            "near_term_reduction_pct": str(result.near_term_reduction_pct),
            "pathway_status": result.pathway_status,
            "cumulative_abatement_tco2e": str(
                result.total_cumulative_abatement_tco2e
            ),
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary
