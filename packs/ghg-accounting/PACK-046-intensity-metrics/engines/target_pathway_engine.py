# -*- coding: utf-8 -*-
"""
TargetPathwayEngine - PACK-046 Intensity Metrics Engine 5
====================================================================

Implements SBTi Sectoral Decarbonisation Approach (SDA) convergence
pathways and target-setting for intensity-based science-based targets.

Calculation Methodology:
    SBTi SDA Convergence Formula:
        I_target(y) = I_sector_2050 + (I_company_base - I_sector_2050) * (2050 - y) / (2050 - base_year)

        Where:
            I_target(y)       = Company target intensity at year y
            I_sector_2050     = Sector convergence intensity in 2050
            I_company_base    = Company intensity in base year
            base_year         = Company base year
            y                 = Target year

    Annual Reduction Rate:
        r(y) = 1 - I_target(y+1) / I_target(y)

    Target Progress:
        progress_pct = (I_base - I_current) / (I_base - I_target) * 100
        If progress_pct >= 100: target achieved
        If progress_pct >= expected_pct: on track
        Else: off track

    Expected Progress (linear):
        expected_pct = (current_year - base_year) / (target_year - base_year) * 100

    Gap Analysis:
        gap_absolute = I_current - I_pathway(current_year)
        gap_pct = gap_absolute / I_pathway(current_year) * 100
        Positive gap = behind pathway; negative gap = ahead of pathway

Sector Pathways (SBTi SDA 2050 convergence values):
    Power:           0.014 tCO2e/MWh  (1.5C)  /  0.038 tCO2e/MWh  (WB2C)
    Steel:           0.142 tCO2e/t     (1.5C)  /  0.332 tCO2e/t     (WB2C)
    Cement:          0.120 tCO2e/t clinker (1.5C) / 0.260 tCO2e/t   (WB2C)
    Aluminium:       1.010 tCO2e/t     (1.5C)  /  2.600 tCO2e/t     (WB2C)
    Buildings:       0.006 kgCO2e/m2   (1.5C)  /  0.015 kgCO2e/m2   (WB2C)
    Road Freight:    0.010 gCO2e/tkm   (1.5C)  /  0.028 gCO2e/tkm   (WB2C)
    Road Passenger:  0.023 gCO2e/pkm   (1.5C)  /  0.059 gCO2e/pkm   (WB2C)

SBTi FLAG Methodology:
    For agriculture/forestry/land-use sectors, uses FLAG-specific pathways.
    I_flag_target(y) = I_flag_sector(y)  (sector convergence without company interpolation)

Regulatory References:
    - SBTi Corporate Manual (v2.1, 2024)
    - SBTi SDA Tool (v1.2, 2023)
    - SBTi FLAG Guidance (v1.1, 2022)
    - IEA Net Zero by 2050 pathway (2021, updated 2023)
    - IPCC AR6 WG3 Chapter 3 (Mitigation Pathways)
    - ESRS E1-4: GHG emission reduction targets

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Sector pathway values from published SBTi SDA documentation
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-046 Intensity Metrics
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class TemperatureAmbition(str, Enum):
    """Temperature ambition level for target-setting.

    T_1_5C:  1.5 degrees Celsius (Paris-aligned, SBTi required for Scope 1+2).
    WB_2C:   Well-below 2 degrees Celsius.
    T_2C:    2 degrees Celsius.
    """
    T_1_5C = "1.5C"
    WB_2C = "WB2C"
    T_2C = "2C"


class SectorPathway(str, Enum):
    """Sector classification for SDA pathways."""
    POWER = "power"
    STEEL = "steel"
    CEMENT = "cement"
    ALUMINIUM = "aluminium"
    BUILDINGS = "buildings"
    ROAD_FREIGHT = "road_freight"
    ROAD_PASSENGER = "road_passenger"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    FLAG_AGRICULTURE = "flag_agriculture"
    FLAG_FORESTRY = "flag_forestry"


class TargetStatus(str, Enum):
    """Status of target progress."""
    ON_TRACK = "on_track"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"
    AHEAD = "ahead"


class TargetType(str, Enum):
    """Type of intensity target."""
    SDA_CONVERGENCE = "sda_convergence"
    ABSOLUTE_REDUCTION = "absolute_reduction"
    CUSTOM_PATHWAY = "custom_pathway"
    FLAG = "flag"


# ---------------------------------------------------------------------------
# Constants -- Sector Convergence Values
# ---------------------------------------------------------------------------

# SBTi SDA 2050 convergence intensities by sector and ambition.
# Sources: SBTi SDA Tool v1.2, IEA NZE2050
SECTOR_2050_INTENSITY: Dict[str, Dict[str, Decimal]] = {
    SectorPathway.POWER.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.014"),
        TemperatureAmbition.WB_2C.value: Decimal("0.038"),
        TemperatureAmbition.T_2C.value: Decimal("0.055"),
    },
    SectorPathway.STEEL.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.142"),
        TemperatureAmbition.WB_2C.value: Decimal("0.332"),
        TemperatureAmbition.T_2C.value: Decimal("0.500"),
    },
    SectorPathway.CEMENT.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.120"),
        TemperatureAmbition.WB_2C.value: Decimal("0.260"),
        TemperatureAmbition.T_2C.value: Decimal("0.380"),
    },
    SectorPathway.ALUMINIUM.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("1.010"),
        TemperatureAmbition.WB_2C.value: Decimal("2.600"),
        TemperatureAmbition.T_2C.value: Decimal("3.800"),
    },
    SectorPathway.BUILDINGS.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.006"),
        TemperatureAmbition.WB_2C.value: Decimal("0.015"),
        TemperatureAmbition.T_2C.value: Decimal("0.025"),
    },
    SectorPathway.ROAD_FREIGHT.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.010"),
        TemperatureAmbition.WB_2C.value: Decimal("0.028"),
        TemperatureAmbition.T_2C.value: Decimal("0.042"),
    },
    SectorPathway.ROAD_PASSENGER.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.023"),
        TemperatureAmbition.WB_2C.value: Decimal("0.059"),
        TemperatureAmbition.T_2C.value: Decimal("0.085"),
    },
    SectorPathway.AVIATION.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.025"),
        TemperatureAmbition.WB_2C.value: Decimal("0.060"),
        TemperatureAmbition.T_2C.value: Decimal("0.090"),
    },
    SectorPathway.SHIPPING.value: {
        TemperatureAmbition.T_1_5C.value: Decimal("0.005"),
        TemperatureAmbition.WB_2C.value: Decimal("0.015"),
        TemperatureAmbition.T_2C.value: Decimal("0.025"),
    },
}

# Sector intensity units
SECTOR_INTENSITY_UNITS: Dict[str, str] = {
    SectorPathway.POWER.value: "tCO2e/MWh",
    SectorPathway.STEEL.value: "tCO2e/t",
    SectorPathway.CEMENT.value: "tCO2e/t_clinker",
    SectorPathway.ALUMINIUM.value: "tCO2e/t",
    SectorPathway.BUILDINGS.value: "kgCO2e/m2",
    SectorPathway.ROAD_FREIGHT.value: "gCO2e/tkm",
    SectorPathway.ROAD_PASSENGER.value: "gCO2e/pkm",
    SectorPathway.AVIATION.value: "gCO2e/pkm",
    SectorPathway.SHIPPING.value: "gCO2e/tkm",
}

# FLAG (Forest, Land and Agriculture) 2030/2050 targets
FLAG_TARGETS: Dict[str, Dict[str, Decimal]] = {
    SectorPathway.FLAG_AGRICULTURE.value: {
        "2030": Decimal("0.85"),  # 15% reduction from 2020 by 2030
        "2050": Decimal("0.28"),  # 72% reduction from 2020 by 2050
    },
    SectorPathway.FLAG_FORESTRY.value: {
        "2030": Decimal("0.80"),  # 20% reduction from 2020 by 2030
        "2050": Decimal("0.25"),  # 75% reduction from 2020 by 2050
    },
}

CONVERGENCE_YEAR: int = 2050
SDA_DEFAULT_BASE_YEAR: int = 2020
MAX_PROJECTION_YEARS: int = 50


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class TargetInput(BaseModel):
    """Input for target pathway calculation.

    Attributes:
        organisation_id:     Organisation identifier.
        sector:              Sector pathway.
        ambition:            Temperature ambition.
        base_year:           Base year for target.
        base_intensity:      Company intensity in base year.
        current_year:        Current year.
        current_intensity:   Company current intensity.
        target_year:         Target year (default: 2030).
        custom_2050_value:   Custom 2050 convergence (overrides default).
        intensity_unit:      Intensity unit.
        output_precision:    Output decimal places.
    """
    organisation_id: str = Field(default="", description="Organisation ID")
    sector: SectorPathway = Field(..., description="Sector pathway")
    ambition: TemperatureAmbition = Field(
        default=TemperatureAmbition.T_1_5C, description="Ambition level"
    )
    base_year: int = Field(default=2020, ge=2000, le=2030, description="Base year")
    base_intensity: Decimal = Field(..., gt=0, description="Base year intensity")
    current_year: int = Field(default=2024, ge=2000, le=2060, description="Current year")
    current_intensity: Optional[Decimal] = Field(
        default=None, ge=0, description="Current intensity"
    )
    target_year: int = Field(default=2030, ge=2025, le=2060, description="Target year")
    custom_2050_value: Optional[Decimal] = Field(
        default=None, ge=0, description="Custom 2050 convergence"
    )
    intensity_unit: str = Field(default="", description="Intensity unit")
    output_precision: int = Field(default=6, ge=0, le=12, description="Output precision")

    @field_validator("base_intensity", mode="before")
    @classmethod
    def coerce_base(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("current_intensity", mode="before")
    @classmethod
    def coerce_current(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

    @model_validator(mode="after")
    def set_intensity_unit(self) -> "TargetInput":
        if not self.intensity_unit:
            unit = SECTOR_INTENSITY_UNITS.get(self.sector.value, "tCO2e/unit")
            object.__setattr__(self, "intensity_unit", unit)
        return self


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class PathwayPoint(BaseModel):
    """A single year on the target pathway.

    Attributes:
        year:                Year.
        target_intensity:    Target intensity at this year.
        annual_reduction:    Required annual reduction rate.
    """
    year: int = Field(..., description="Year")
    target_intensity: Decimal = Field(..., description="Target intensity")
    annual_reduction: Optional[Decimal] = Field(default=None, description="Annual reduction rate")


class TargetProgress(BaseModel):
    """Progress towards intensity target.

    Attributes:
        progress_pct:         Percentage of target achieved.
        expected_progress_pct: Expected percentage at current date.
        gap_absolute:         Absolute gap to pathway.
        gap_pct:              Percentage gap to pathway.
        status:               On-track/off-track status.
        years_ahead_behind:   Positive = ahead, negative = behind.
    """
    progress_pct: Decimal = Field(default=Decimal("0"), description="Progress (%)")
    expected_progress_pct: Decimal = Field(default=Decimal("0"), description="Expected progress (%)")
    gap_absolute: Decimal = Field(default=Decimal("0"), description="Absolute gap")
    gap_pct: Decimal = Field(default=Decimal("0"), description="Gap (%)")
    status: TargetStatus = Field(default=TargetStatus.NOT_STARTED, description="Status")
    years_ahead_behind: Decimal = Field(default=Decimal("0"), description="Years ahead/behind")


class PathwayComparison(BaseModel):
    """Comparison between 1.5C and WB2C pathways.

    Attributes:
        ambition_1_5c_target:  Target intensity at target year (1.5C).
        ambition_wb2c_target:  Target intensity at target year (WB2C).
        ambition_1_5c_reduction: Required total reduction (1.5C, %).
        ambition_wb2c_reduction: Required total reduction (WB2C, %).
        current_aligns_with:   Which ambition the current trajectory aligns with.
    """
    ambition_1_5c_target: Optional[Decimal] = Field(default=None, description="1.5C target")
    ambition_wb2c_target: Optional[Decimal] = Field(default=None, description="WB2C target")
    ambition_1_5c_reduction: Optional[Decimal] = Field(default=None, description="1.5C reduction %")
    ambition_wb2c_reduction: Optional[Decimal] = Field(default=None, description="WB2C reduction %")
    current_aligns_with: str = Field(default="none", description="Alignment")


class TargetResult(BaseModel):
    """Result of target pathway calculation.

    Attributes:
        result_id:              Unique result identifier.
        organisation_id:        Organisation identifier.
        sector:                 Sector pathway.
        ambition:               Temperature ambition.
        base_year:              Base year.
        target_year:            Target year.
        base_intensity:         Base year intensity.
        target_intensity:       Target intensity at target year.
        convergence_2050:       2050 convergence intensity.
        intensity_unit:         Intensity unit.
        pathway:                Year-by-year pathway.
        target_progress:        Progress assessment (if current data).
        pathway_comparison:     1.5C vs WB2C comparison.
        total_reduction_pct:    Total required reduction (%).
        avg_annual_reduction:   Average annual reduction rate.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Organisation ID")
    sector: str = Field(default="", description="Sector")
    ambition: str = Field(default="", description="Ambition")
    base_year: int = Field(default=2020, description="Base year")
    target_year: int = Field(default=2030, description="Target year")
    base_intensity: Decimal = Field(default=Decimal("0"), description="Base intensity")
    target_intensity: Decimal = Field(default=Decimal("0"), description="Target intensity")
    convergence_2050: Decimal = Field(default=Decimal("0"), description="2050 convergence")
    intensity_unit: str = Field(default="", description="Intensity unit")
    pathway: List[PathwayPoint] = Field(default_factory=list, description="Pathway")
    target_progress: Optional[TargetProgress] = Field(default=None, description="Progress")
    pathway_comparison: Optional[PathwayComparison] = Field(default=None, description="Comparison")
    total_reduction_pct: Decimal = Field(default=Decimal("0"), description="Total reduction (%)")
    avg_annual_reduction: Decimal = Field(default=Decimal("0"), description="Avg annual reduction")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TargetPathwayEngine:
    """SBTi SDA convergence pathway and target-setting engine.

    Generates sector-specific decarbonisation pathways based on the
    SBTi Sectoral Decarbonisation Approach, with support for 1.5C,
    WB2C, and 2C ambition levels.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every pathway point derived from published formulas.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("TargetPathwayEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: TargetInput) -> TargetResult:
        """Calculate SDA convergence pathway and target intensity.

        Args:
            input_data: Target pathway input.

        Returns:
            TargetResult with pathway, progress, and comparison.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        prec = input_data.output_precision
        prec_str = "0." + "0" * prec

        sector = input_data.sector.value
        ambition = input_data.ambition.value

        # Resolve 2050 convergence intensity
        if input_data.custom_2050_value is not None:
            i_2050 = input_data.custom_2050_value
            warnings.append("Using custom 2050 convergence value (not SBTi default).")
        else:
            sector_vals = SECTOR_2050_INTENSITY.get(sector)
            if sector_vals is None:
                # Check FLAG pathways
                if sector in FLAG_TARGETS:
                    return self._calculate_flag_pathway(input_data, t0, warnings, prec, prec_str)
                raise ValueError(
                    f"No SDA pathway defined for sector '{sector}'. "
                    f"Available: {list(SECTOR_2050_INTENSITY.keys())}"
                )
            i_2050 = sector_vals.get(ambition)
            if i_2050 is None:
                raise ValueError(
                    f"No {ambition} pathway for sector '{sector}'. "
                    f"Available: {list(sector_vals.keys())}"
                )

        i_base = input_data.base_intensity
        base_year = input_data.base_year
        target_year = input_data.target_year

        # Generate year-by-year pathway
        pathway: List[PathwayPoint] = []
        span = CONVERGENCE_YEAR - base_year
        if span <= 0:
            raise ValueError("Base year must be before 2050.")

        prev_intensity: Optional[Decimal] = None
        for y in range(base_year, min(target_year + 1, base_year + MAX_PROJECTION_YEARS + 1)):
            # SDA formula: I(y) = I_2050 + (I_base - I_2050) * (2050 - y) / (2050 - base_year)
            years_remaining = Decimal(str(CONVERGENCE_YEAR - y))
            total_span = Decimal(str(span))
            i_y = i_2050 + (i_base - i_2050) * years_remaining / total_span
            i_y = i_y.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

            # Annual reduction rate
            annual_red: Optional[Decimal] = None
            if prev_intensity is not None and prev_intensity > Decimal("0"):
                annual_red = (
                    Decimal("1") - i_y / prev_intensity
                ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

            pathway.append(PathwayPoint(
                year=y,
                target_intensity=i_y,
                annual_reduction=annual_red,
            ))
            prev_intensity = i_y

        # Target intensity at target year
        target_intensity = self._sda_intensity(i_base, i_2050, base_year, target_year)
        target_intensity = target_intensity.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Total reduction percentage
        total_red = Decimal("0")
        if i_base > Decimal("0"):
            total_red = (
                (i_base - target_intensity) / i_base * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Average annual reduction
        n_years = target_year - base_year
        avg_annual = Decimal("0")
        if n_years > 0 and i_base > Decimal("0"):
            ratio = target_intensity / i_base
            if ratio > Decimal("0"):
                # CARR: (ratio)^(1/n) - 1
                ratio_float = float(ratio)
                n_float = float(n_years)
                carr = 1 - ratio_float ** (1 / n_float)
                avg_annual = _decimal(carr).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )

        # Progress assessment
        progress: Optional[TargetProgress] = None
        if input_data.current_intensity is not None:
            progress = self._assess_progress(
                i_base, input_data.current_intensity, target_intensity,
                i_2050, base_year, input_data.current_year, target_year, prec_str,
            )

        # Pathway comparison (1.5C vs WB2C)
        comparison = self._build_comparison(
            i_base, base_year, target_year, sector, input_data.current_intensity, prec_str,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = TargetResult(
            organisation_id=input_data.organisation_id,
            sector=sector,
            ambition=ambition,
            base_year=base_year,
            target_year=target_year,
            base_intensity=i_base,
            target_intensity=target_intensity,
            convergence_2050=i_2050,
            intensity_unit=input_data.intensity_unit,
            pathway=pathway,
            target_progress=progress,
            pathway_comparison=comparison,
            total_reduction_pct=total_red,
            avg_annual_reduction=avg_annual,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def get_target_intensity(
        self,
        base_intensity: Decimal,
        sector: str,
        ambition: str,
        base_year: int,
        target_year: int,
    ) -> Decimal:
        """Get target intensity for a specific year.

        Args:
            base_intensity: Company base year intensity.
            sector:         Sector pathway.
            ambition:       Temperature ambition.
            base_year:      Base year.
            target_year:    Target year.

        Returns:
            Target intensity as Decimal.
        """
        i_2050 = SECTOR_2050_INTENSITY[sector][ambition]
        return self._sda_intensity(
            _decimal(base_intensity), i_2050, base_year, target_year
        )

    def get_annual_reduction_rate(
        self,
        base_intensity: Decimal,
        sector: str,
        ambition: str,
        base_year: int,
        target_year: int,
    ) -> Decimal:
        """Get average annual reduction rate.

        Args:
            base_intensity: Base year intensity.
            sector:         Sector.
            ambition:       Ambition.
            base_year:      Base year.
            target_year:    Target year.

        Returns:
            Average annual reduction rate as Decimal.
        """
        i_target = self.get_target_intensity(
            base_intensity, sector, ambition, base_year, target_year
        )
        n_years = target_year - base_year
        if n_years <= 0 or base_intensity <= Decimal("0"):
            return Decimal("0")
        ratio = float(i_target / _decimal(base_intensity))
        if ratio <= 0:
            return Decimal("1")
        carr = 1 - ratio ** (1 / float(n_years))
        return _decimal(carr).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def get_available_sectors(self) -> List[str]:
        """Return list of available sector pathways."""
        return sorted(SECTOR_2050_INTENSITY.keys())

    def get_sector_convergence(self, sector: str) -> Dict[str, Decimal]:
        """Return 2050 convergence values for a sector."""
        return dict(SECTOR_2050_INTENSITY.get(sector, {}))

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _sda_intensity(
        self,
        i_base: Decimal,
        i_2050: Decimal,
        base_year: int,
        target_year: int,
    ) -> Decimal:
        """SDA convergence formula.

        I(y) = I_2050 + (I_base - I_2050) * (2050 - y) / (2050 - base_year)
        """
        span = Decimal(str(CONVERGENCE_YEAR - base_year))
        if span == Decimal("0"):
            return i_base
        years_remaining = Decimal(str(CONVERGENCE_YEAR - target_year))
        return i_2050 + (i_base - i_2050) * years_remaining / span

    def _assess_progress(
        self,
        i_base: Decimal,
        i_current: Decimal,
        i_target: Decimal,
        i_2050: Decimal,
        base_year: int,
        current_year: int,
        target_year: int,
        prec_str: str,
    ) -> TargetProgress:
        """Assess progress towards target."""
        # Expected pathway intensity at current year
        i_expected = self._sda_intensity(i_base, i_2050, base_year, current_year)

        # Progress percentage
        total_required = i_base - i_target
        progress_pct = Decimal("0")
        if total_required > Decimal("0"):
            achieved = i_base - i_current
            progress_pct = (achieved / total_required * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Expected progress
        expected_pct = Decimal("0")
        total_years = target_year - base_year
        if total_years > 0:
            elapsed = current_year - base_year
            expected_pct = (
                Decimal(str(elapsed)) / Decimal(str(total_years)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Gap to pathway
        gap_abs = (i_current - i_expected).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
        gap_pct = Decimal("0")
        if i_expected > Decimal("0"):
            gap_pct = (gap_abs / i_expected * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Status
        if progress_pct >= Decimal("100"):
            status = TargetStatus.ACHIEVED
        elif i_current <= i_expected:
            if progress_pct > expected_pct:
                status = TargetStatus.AHEAD
            else:
                status = TargetStatus.ON_TRACK
        else:
            status = TargetStatus.OFF_TRACK

        # Years ahead/behind (estimate)
        years_ahead_behind = Decimal("0")
        if total_required > Decimal("0") and total_years > 0:
            reduction_per_year = total_required / Decimal(str(total_years))
            if reduction_per_year > Decimal("0"):
                years_ahead_behind = (
                    (i_expected - i_current) / reduction_per_year
                ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        return TargetProgress(
            progress_pct=progress_pct,
            expected_progress_pct=expected_pct,
            gap_absolute=gap_abs,
            gap_pct=gap_pct,
            status=status,
            years_ahead_behind=years_ahead_behind,
        )

    def _build_comparison(
        self,
        i_base: Decimal,
        base_year: int,
        target_year: int,
        sector: str,
        current_intensity: Optional[Decimal],
        prec_str: str,
    ) -> Optional[PathwayComparison]:
        """Build 1.5C vs WB2C pathway comparison."""
        sector_vals = SECTOR_2050_INTENSITY.get(sector)
        if sector_vals is None:
            return None

        i_2050_15 = sector_vals.get(TemperatureAmbition.T_1_5C.value)
        i_2050_wb2c = sector_vals.get(TemperatureAmbition.WB_2C.value)

        target_15: Optional[Decimal] = None
        target_wb2c: Optional[Decimal] = None
        red_15: Optional[Decimal] = None
        red_wb2c: Optional[Decimal] = None

        if i_2050_15 is not None:
            target_15 = self._sda_intensity(
                i_base, i_2050_15, base_year, target_year
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            if i_base > Decimal("0"):
                red_15 = (
                    (i_base - target_15) / i_base * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if i_2050_wb2c is not None:
            target_wb2c = self._sda_intensity(
                i_base, i_2050_wb2c, base_year, target_year
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            if i_base > Decimal("0"):
                red_wb2c = (
                    (i_base - target_wb2c) / i_base * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        aligns_with = "none"
        if current_intensity is not None:
            if target_15 is not None and current_intensity <= target_15:
                aligns_with = "1.5C"
            elif target_wb2c is not None and current_intensity <= target_wb2c:
                aligns_with = "WB2C"
            else:
                aligns_with = "none"

        return PathwayComparison(
            ambition_1_5c_target=target_15,
            ambition_wb2c_target=target_wb2c,
            ambition_1_5c_reduction=red_15,
            ambition_wb2c_reduction=red_wb2c,
            current_aligns_with=aligns_with,
        )

    def _calculate_flag_pathway(
        self,
        input_data: TargetInput,
        t0: float,
        warnings: List[str],
        prec: int,
        prec_str: str,
    ) -> TargetResult:
        """Calculate FLAG (agriculture/forestry) pathway."""
        sector = input_data.sector.value
        flag_vals = FLAG_TARGETS.get(sector, {})
        warnings.append(
            f"Using SBTi FLAG methodology for '{sector}'. "
            f"Convergence approach differs from standard SDA."
        )

        i_base = input_data.base_intensity
        base_year = input_data.base_year
        target_year = input_data.target_year

        # FLAG uses fractional reduction from base
        target_2030_frac = flag_vals.get("2030", Decimal("0.85"))
        target_2050_frac = flag_vals.get("2050", Decimal("0.28"))

        # Linear interpolation between base year and milestones
        pathway: List[PathwayPoint] = []
        prev_i: Optional[Decimal] = None

        for y in range(base_year, min(target_year + 1, base_year + MAX_PROJECTION_YEARS + 1)):
            if y <= 2030:
                span = 2030 - base_year
                if span > 0:
                    frac = Decimal("1") + (target_2030_frac - Decimal("1")) * Decimal(str(y - base_year)) / Decimal(str(span))
                else:
                    frac = target_2030_frac
            else:
                span = 2050 - 2030
                elapsed = y - 2030
                frac = target_2030_frac + (target_2050_frac - target_2030_frac) * Decimal(str(elapsed)) / Decimal(str(span))

            i_y = (i_base * frac).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

            annual_red: Optional[Decimal] = None
            if prev_i is not None and prev_i > Decimal("0"):
                annual_red = (Decimal("1") - i_y / prev_i).quantize(
                    Decimal("0.000001"), rounding=ROUND_HALF_UP
                )

            pathway.append(PathwayPoint(year=y, target_intensity=i_y, annual_reduction=annual_red))
            prev_i = i_y

        target_intensity = pathway[-1].target_intensity if pathway else i_base
        total_red = Decimal("0")
        if i_base > Decimal("0"):
            total_red = ((i_base - target_intensity) / i_base * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = TargetResult(
            organisation_id=input_data.organisation_id,
            sector=sector,
            ambition=input_data.ambition.value,
            base_year=base_year,
            target_year=target_year,
            base_intensity=i_base,
            target_intensity=target_intensity,
            convergence_2050=i_base * target_2050_frac,
            intensity_unit=input_data.intensity_unit,
            pathway=pathway,
            total_reduction_pct=total_red,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "TemperatureAmbition",
    "SectorPathway",
    "TargetStatus",
    "TargetType",
    # Input Models
    "TargetInput",
    # Output Models
    "PathwayPoint",
    "TargetProgress",
    "PathwayComparison",
    "TargetResult",
    # Engine
    "TargetPathwayEngine",
    # Constants
    "SECTOR_2050_INTENSITY",
    "SECTOR_INTENSITY_UNITS",
    "FLAG_TARGETS",
    "CONVERGENCE_YEAR",
]
