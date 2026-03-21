# -*- coding: utf-8 -*-
"""
SDA Target Setting Workflow
=================================

4-phase workflow for calculating Sectoral Decarbonisation Approach (SDA)
targets within PACK-022 Net-Zero Acceleration Pack.  The workflow
classifies the company into an SDA sector, calculates sector benchmark
convergence pathways from IEA NZE data, sets company-specific intensity
targets converging to the sector benchmark by 2050, and validates
against SBTi SDA criteria.

Phases:
    1. SectorClassification   -- Classify company into SDA sector(s),
                                  determine activity metrics
    2. BenchmarkCalculation   -- Calculate sector benchmark convergence
                                  pathway, IEA NZE alignment
    3. TargetSetting          -- Set company intensity targets converging
                                  to sector benchmark by 2050
    4. Validation             -- Validate against SBTi SDA criteria,
                                  compare with ACA alternative

Regulatory references:
    - SBTi Sectoral Decarbonisation Approach (SDA) Methodology
    - SBTi Net-Zero Standard v1.2 (2024)
    - IEA Net Zero by 2050 Roadmap (2021/2023 update)
    - GHG Protocol Corporate Standard

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class SDASector(str, Enum):
    """SDA-eligible sectors with IEA benchmarks."""

    POWER_GENERATION = "power_generation"
    CEMENT = "cement"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    PULP_AND_PAPER = "pulp_and_paper"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    REAL_ESTATE = "real_estate"
    CHEMICALS = "chemicals"
    AUTOMOTIVE = "automotive"


# =============================================================================
# SDA REFERENCE DATA (Zero-Hallucination, from IEA NZE / SBTi SDA)
# =============================================================================

# Sector benchmark intensity pathways (tCO2e per activity unit)
# Values from IEA NZE 2050 Scenario, used by SBTi SDA methodology
SDA_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "activity_metric": "MWh",
        "unit": "tCO2e/MWh",
        "base_2020": 0.460,
        "value_2025": 0.338,
        "value_2030": 0.138,
        "value_2035": 0.058,
        "value_2040": 0.018,
        "value_2045": 0.005,
        "value_2050": 0.000,
    },
    "cement": {
        "activity_metric": "tonne_cement",
        "unit": "tCO2e/t_cement",
        "base_2020": 0.610,
        "value_2025": 0.540,
        "value_2030": 0.469,
        "value_2035": 0.370,
        "value_2040": 0.270,
        "value_2045": 0.200,
        "value_2050": 0.143,
    },
    "steel": {
        "activity_metric": "tonne_steel",
        "unit": "tCO2e/t_steel",
        "base_2020": 1.400,
        "value_2025": 1.200,
        "value_2030": 1.013,
        "value_2035": 0.700,
        "value_2040": 0.400,
        "value_2045": 0.200,
        "value_2050": 0.050,
    },
    "aluminum": {
        "activity_metric": "tonne_aluminum",
        "unit": "tCO2e/t_aluminum",
        "base_2020": 6.800,
        "value_2025": 5.700,
        "value_2030": 4.600,
        "value_2035": 3.500,
        "value_2040": 2.500,
        "value_2045": 1.800,
        "value_2050": 1.100,
    },
    "pulp_and_paper": {
        "activity_metric": "tonne_product",
        "unit": "tCO2e/t_product",
        "base_2020": 0.360,
        "value_2025": 0.280,
        "value_2030": 0.210,
        "value_2035": 0.150,
        "value_2040": 0.100,
        "value_2045": 0.065,
        "value_2050": 0.040,
    },
    "aviation": {
        "activity_metric": "revenue_tonne_km",
        "unit": "gCO2e/RTK",
        "base_2020": 820.0,
        "value_2025": 720.0,
        "value_2030": 600.0,
        "value_2035": 440.0,
        "value_2040": 300.0,
        "value_2045": 180.0,
        "value_2050": 80.0,
    },
    "shipping": {
        "activity_metric": "tonne_nautical_mile",
        "unit": "gCO2e/t-nm",
        "base_2020": 8.50,
        "value_2025": 7.50,
        "value_2030": 6.20,
        "value_2035": 4.80,
        "value_2040": 3.50,
        "value_2045": 2.20,
        "value_2050": 1.00,
    },
    "real_estate": {
        "activity_metric": "square_meter",
        "unit": "kgCO2e/sqm",
        "base_2020": 35.0,
        "value_2025": 28.0,
        "value_2030": 22.0,
        "value_2035": 15.0,
        "value_2040": 10.0,
        "value_2045": 5.5,
        "value_2050": 2.5,
    },
    "chemicals": {
        "activity_metric": "tonne_product",
        "unit": "tCO2e/t_product",
        "base_2020": 1.100,
        "value_2025": 0.950,
        "value_2030": 0.800,
        "value_2035": 0.600,
        "value_2040": 0.420,
        "value_2045": 0.280,
        "value_2050": 0.160,
    },
    "automotive": {
        "activity_metric": "vehicle_produced",
        "unit": "tCO2e/vehicle",
        "base_2020": 6.000,
        "value_2025": 5.000,
        "value_2030": 3.800,
        "value_2035": 2.600,
        "value_2040": 1.600,
        "value_2045": 0.900,
        "value_2050": 0.400,
    },
}

# SBTi SDA minimum annual reduction rates (1.5C aligned)
SDA_MIN_ANNUAL_REDUCTION_PCT = 4.2

# ACA comparison annual reduction rate (cross-sector 1.5C)
ACA_ANNUAL_REDUCTION_PCT = 4.2


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class SectorClassificationResult(BaseModel):
    """Output of sector classification phase."""

    sector: str = Field(default="")
    sector_display_name: str = Field(default="")
    is_sda_eligible: bool = Field(default=False)
    activity_metric: str = Field(default="")
    intensity_unit: str = Field(default="")
    benchmark_available: bool = Field(default=False)
    alternative_pathway: str = Field(default="absolute_contraction")


class BenchmarkPathwayPoint(BaseModel):
    """Single point on the sector benchmark pathway."""

    year: int = Field(default=2025)
    benchmark_intensity: float = Field(default=0.0)
    company_target_intensity: float = Field(default=0.0)
    reduction_from_base_pct: float = Field(default=0.0)


class BenchmarkPathway(BaseModel):
    """Complete sector benchmark convergence pathway."""

    sector: str = Field(default="")
    pathway_points: List[BenchmarkPathwayPoint] = Field(default_factory=list)
    convergence_year: int = Field(default=2050)
    base_year_benchmark: float = Field(default=0.0)
    target_year_benchmark: float = Field(default=0.0)
    total_reduction_pct: float = Field(default=0.0)


class CompanyIntensityTarget(BaseModel):
    """Company-specific intensity target for a given year."""

    year: int = Field(default=2030)
    target_intensity: float = Field(default=0.0)
    benchmark_intensity: float = Field(default=0.0)
    convergence_pct: float = Field(default=0.0, description="How close to benchmark (100% = converged)")
    emissions_budget_tco2e: float = Field(default=0.0)
    reduction_from_base_pct: float = Field(default=0.0)


class ACAComparison(BaseModel):
    """Comparison between SDA and ACA approaches."""

    aca_near_term_reduction_pct: float = Field(default=0.0)
    aca_long_term_reduction_pct: float = Field(default=0.0)
    sda_near_term_reduction_pct: float = Field(default=0.0)
    sda_long_term_reduction_pct: float = Field(default=0.0)
    more_ambitious_approach: str = Field(default="")
    recommendation: str = Field(default="")


class SDAValidationFinding(BaseModel):
    """A single SDA validation finding."""

    criterion: str = Field(default="")
    description: str = Field(default="")
    severity: str = Field(default="pass")
    detail: str = Field(default="")


class SDAValidationReport(BaseModel):
    """SDA validation report against SBTi criteria."""

    overall_valid: bool = Field(default=False)
    findings: List[SDAValidationFinding] = Field(default_factory=list)
    pass_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    fail_count: int = Field(default=0)
    aca_comparison: ACAComparison = Field(default_factory=ACAComparison)


class SDATargetConfig(BaseModel):
    """Configuration for the SDA target workflow."""

    sector: str = Field(default="power_generation", description="SDA sector classification")
    base_year: int = Field(default=2024, ge=2015, le=2050)
    base_year_intensity: float = Field(default=0.0, ge=0.0, description="Company intensity in base year")
    base_year_activity: float = Field(default=0.0, ge=0.0, description="Activity volume in base year")
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    activity_metric: str = Field(default="", description="Activity metric override")
    growth_rate: float = Field(default=0.02, ge=-0.10, le=0.20, description="Annual activity growth rate")
    convergence_year: int = Field(default=2050, ge=2040, le=2060)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("sector")
    @classmethod
    def _validate_sector(cls, v: str) -> str:
        v = v.lower().strip()
        return v


class SDATargetResult(BaseModel):
    """Complete result from the SDA target workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sda_target")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    classification: SectorClassificationResult = Field(default_factory=SectorClassificationResult)
    benchmark_pathway: BenchmarkPathway = Field(default_factory=BenchmarkPathway)
    company_targets: List[CompanyIntensityTarget] = Field(default_factory=list)
    near_term_target: Optional[CompanyIntensityTarget] = Field(None)
    long_term_target: Optional[CompanyIntensityTarget] = Field(None)
    validation: SDAValidationReport = Field(default_factory=SDAValidationReport)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SDATargetWorkflow:
    """
    4-phase SDA target-setting workflow.

    Classifies the company into an SDA sector, calculates the IEA NZE
    benchmark convergence pathway, sets company-specific intensity
    targets, and validates against SBTi SDA criteria.

    Zero-hallucination: all benchmark values, convergence calculations,
    and validation criteria come from deterministic IEA/SBTi reference
    data.  No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = SDATargetWorkflow()
        >>> config = SDATargetConfig(sector="steel", base_year_intensity=1.5)
        >>> result = await wf.execute(config)
        >>> assert result.validation.overall_valid
    """

    def __init__(self) -> None:
        """Initialise SDATargetWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._classification: SectorClassificationResult = SectorClassificationResult()
        self._benchmark: BenchmarkPathway = BenchmarkPathway()
        self._targets: List[CompanyIntensityTarget] = []
        self._near_term: Optional[CompanyIntensityTarget] = None
        self._long_term: Optional[CompanyIntensityTarget] = None
        self._validation: SDAValidationReport = SDAValidationReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: SDATargetConfig) -> SDATargetResult:
        """
        Execute the 4-phase SDA target workflow.

        Args:
            config: SDA target configuration with sector, base year
                intensity, activity metric, and growth rate.

        Returns:
            SDATargetResult with benchmark pathway, company targets,
            and SBTi validation.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting SDA target workflow %s, sector=%s",
            self.workflow_id, config.sector,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_sector_classification(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_benchmark_calculation(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_target_setting(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_validation(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("SDA target workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = SDATargetResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            classification=self._classification,
            benchmark_pathway=self._benchmark,
            company_targets=self._targets,
            near_term_target=self._near_term,
            long_term_target=self._long_term,
            validation=self._validation,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "SDA target workflow %s completed in %.2fs, valid=%s",
            self.workflow_id, elapsed, self._validation.overall_valid,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Sector Classification
    # -------------------------------------------------------------------------

    async def _phase_sector_classification(self, config: SDATargetConfig) -> PhaseResult:
        """Classify company into SDA sector, determine activity metrics."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector = config.sector.lower().strip()
        is_eligible = sector in SDA_BENCHMARKS
        benchmark = SDA_BENCHMARKS.get(sector)

        if not is_eligible:
            warnings.append(
                f"Sector '{sector}' is not SDA-eligible; "
                "ACA (Absolute Contraction Approach) will be recommended as alternative"
            )

        activity_metric = config.activity_metric or (benchmark["activity_metric"] if benchmark else "")
        intensity_unit = benchmark["unit"] if benchmark else ""

        self._classification = SectorClassificationResult(
            sector=sector,
            sector_display_name=sector.replace("_", " ").title(),
            is_sda_eligible=is_eligible,
            activity_metric=activity_metric,
            intensity_unit=intensity_unit,
            benchmark_available=benchmark is not None,
            alternative_pathway="absolute_contraction" if not is_eligible else "",
        )

        outputs["sector"] = sector
        outputs["is_sda_eligible"] = is_eligible
        outputs["activity_metric"] = activity_metric
        outputs["intensity_unit"] = intensity_unit

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Sector classification: %s, SDA-eligible=%s", sector, is_eligible)
        return PhaseResult(
            phase_name="sector_classification",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Benchmark Calculation
    # -------------------------------------------------------------------------

    async def _phase_benchmark_calculation(self, config: SDATargetConfig) -> PhaseResult:
        """Calculate sector benchmark convergence pathway from IEA NZE data."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector = self._classification.sector
        benchmark_data = SDA_BENCHMARKS.get(sector)

        if not benchmark_data:
            warnings.append("No SDA benchmark available; using linear reduction proxy")
            self._benchmark = self._generate_proxy_pathway(config)
        else:
            self._benchmark = self._calculate_benchmark_pathway(config, benchmark_data)

        outputs["sector"] = sector
        outputs["convergence_year"] = self._benchmark.convergence_year
        outputs["base_benchmark"] = round(self._benchmark.base_year_benchmark, 4)
        outputs["target_benchmark"] = round(self._benchmark.target_year_benchmark, 4)
        outputs["total_reduction_pct"] = round(self._benchmark.total_reduction_pct, 2)
        outputs["pathway_points"] = len(self._benchmark.pathway_points)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Benchmark: %s, base=%.4f, target=%.4f, reduction=%.1f%%",
            sector, self._benchmark.base_year_benchmark,
            self._benchmark.target_year_benchmark, self._benchmark.total_reduction_pct,
        )
        return PhaseResult(
            phase_name="benchmark_calculation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _calculate_benchmark_pathway(
        self, config: SDATargetConfig, benchmark_data: Dict[str, Any]
    ) -> BenchmarkPathway:
        """Calculate benchmark pathway from IEA NZE reference data."""
        ref_years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
        ref_keys = [
            "base_2020", "value_2025", "value_2030",
            "value_2035", "value_2040", "value_2045", "value_2050",
        ]
        ref_values = {y: benchmark_data[k] for y, k in zip(ref_years, ref_keys)}

        base_benchmark = self._interpolate_value(ref_values, config.base_year)
        target_benchmark = self._interpolate_value(ref_values, config.convergence_year)

        total_reduction = 0.0
        if base_benchmark > 0:
            total_reduction = ((base_benchmark - target_benchmark) / base_benchmark) * 100.0

        points: List[BenchmarkPathwayPoint] = []
        for year in range(config.base_year, config.convergence_year + 1, 5):
            bm_intensity = self._interpolate_value(ref_values, year)
            red_pct = 0.0
            if base_benchmark > 0:
                red_pct = ((base_benchmark - bm_intensity) / base_benchmark) * 100.0
            points.append(BenchmarkPathwayPoint(
                year=year,
                benchmark_intensity=round(bm_intensity, 6),
                reduction_from_base_pct=round(red_pct, 2),
            ))

        return BenchmarkPathway(
            sector=config.sector,
            pathway_points=points,
            convergence_year=config.convergence_year,
            base_year_benchmark=round(base_benchmark, 6),
            target_year_benchmark=round(target_benchmark, 6),
            total_reduction_pct=round(total_reduction, 2),
        )

    def _generate_proxy_pathway(self, config: SDATargetConfig) -> BenchmarkPathway:
        """Generate a proxy linear pathway when no benchmark is available."""
        base_intensity = config.base_year_intensity if config.base_year_intensity > 0 else 1.0
        target_intensity = base_intensity * 0.10  # 90% reduction proxy
        years_span = config.convergence_year - config.base_year

        points: List[BenchmarkPathwayPoint] = []
        for year in range(config.base_year, config.convergence_year + 1, 5):
            elapsed = year - config.base_year
            frac = elapsed / years_span if years_span > 0 else 1.0
            bm = base_intensity - (base_intensity - target_intensity) * frac
            red_pct = ((base_intensity - bm) / base_intensity) * 100.0 if base_intensity > 0 else 0.0
            points.append(BenchmarkPathwayPoint(
                year=year,
                benchmark_intensity=round(bm, 6),
                reduction_from_base_pct=round(red_pct, 2),
            ))

        return BenchmarkPathway(
            sector=config.sector,
            pathway_points=points,
            convergence_year=config.convergence_year,
            base_year_benchmark=round(base_intensity, 6),
            target_year_benchmark=round(target_intensity, 6),
            total_reduction_pct=90.0,
        )

    def _interpolate_value(self, ref: Dict[int, float], year: int) -> float:
        """Linearly interpolate a value from reference year-value pairs."""
        years = sorted(ref.keys())
        if year <= years[0]:
            return ref[years[0]]
        if year >= years[-1]:
            return ref[years[-1]]
        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                y0, y1 = years[i], years[i + 1]
                v0, v1 = ref[y0], ref[y1]
                frac = (year - y0) / (y1 - y0)
                return v0 + frac * (v1 - v0)
        return ref[years[-1]]

    # -------------------------------------------------------------------------
    # Phase 3: Target Setting
    # -------------------------------------------------------------------------

    async def _phase_target_setting(self, config: SDATargetConfig) -> PhaseResult:
        """Set company-specific intensity targets converging to benchmark."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        company_base = config.base_year_intensity
        if company_base <= 0:
            company_base = self._benchmark.base_year_benchmark
            warnings.append(
                f"No base year intensity provided; using sector benchmark: {company_base}"
            )

        base_activity = config.base_year_activity if config.base_year_activity > 0 else 1.0
        growth_rate = config.growth_rate

        self._targets = []
        bm_points = {p.year: p.benchmark_intensity for p in self._benchmark.pathway_points}

        for year in range(config.base_year, config.convergence_year + 1, 5):
            bm_intensity = self._interpolate_value(bm_points, year) if bm_points else 0.0

            # SDA convergence formula:
            # company_target(t) = benchmark(t) + (company_base - benchmark_base) * (1 - convergence_fraction)
            years_elapsed = year - config.base_year
            years_total = config.convergence_year - config.base_year
            convergence_frac = years_elapsed / years_total if years_total > 0 else 1.0
            convergence_frac = min(convergence_frac, 1.0)

            gap = company_base - self._benchmark.base_year_benchmark
            target_intensity = bm_intensity + gap * (1.0 - convergence_frac)
            target_intensity = max(target_intensity, 0.0)

            # Calculate emissions budget with activity growth
            activity_at_year = base_activity * ((1.0 + growth_rate) ** years_elapsed)
            emissions_budget = target_intensity * activity_at_year

            reduction_pct = 0.0
            if company_base > 0:
                reduction_pct = ((company_base - target_intensity) / company_base) * 100.0

            convergence_pct = 0.0
            if gap != 0:
                convergence_pct = convergence_frac * 100.0
            else:
                convergence_pct = 100.0

            target = CompanyIntensityTarget(
                year=year,
                target_intensity=round(target_intensity, 6),
                benchmark_intensity=round(bm_intensity, 6),
                convergence_pct=round(convergence_pct, 2),
                emissions_budget_tco2e=round(emissions_budget, 4),
                reduction_from_base_pct=round(max(reduction_pct, 0.0), 2),
            )
            self._targets.append(target)

            if year == config.near_term_target_year:
                self._near_term = target
            if year == config.convergence_year:
                self._long_term = target

        # Ensure near-term target exists (interpolate if needed)
        if self._near_term is None and self._targets:
            self._near_term = self._interpolate_target(
                config.near_term_target_year, config, company_base, base_activity
            )
            self._targets.append(self._near_term)
            self._targets.sort(key=lambda t: t.year)

        outputs["company_base_intensity"] = round(company_base, 6)
        outputs["target_count"] = len(self._targets)
        if self._near_term:
            outputs["near_term_year"] = self._near_term.year
            outputs["near_term_intensity"] = self._near_term.target_intensity
            outputs["near_term_reduction_pct"] = self._near_term.reduction_from_base_pct
        if self._long_term:
            outputs["long_term_year"] = self._long_term.year
            outputs["long_term_intensity"] = self._long_term.target_intensity
            outputs["long_term_reduction_pct"] = self._long_term.reduction_from_base_pct

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Targets set: %d points, NT=%.4f, LT=%.4f",
            len(self._targets),
            self._near_term.target_intensity if self._near_term else 0.0,
            self._long_term.target_intensity if self._long_term else 0.0,
        )
        return PhaseResult(
            phase_name="target_setting",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _interpolate_target(
        self,
        year: int,
        config: SDATargetConfig,
        company_base: float,
        base_activity: float,
    ) -> CompanyIntensityTarget:
        """Interpolate a company target for a year between pathway points."""
        bm_points = {p.year: p.benchmark_intensity for p in self._benchmark.pathway_points}
        bm_intensity = self._interpolate_value(bm_points, year) if bm_points else 0.0

        years_elapsed = year - config.base_year
        years_total = config.convergence_year - config.base_year
        convergence_frac = min(years_elapsed / years_total if years_total > 0 else 1.0, 1.0)

        gap = company_base - self._benchmark.base_year_benchmark
        target_intensity = max(bm_intensity + gap * (1.0 - convergence_frac), 0.0)

        activity_at_year = base_activity * ((1.0 + config.growth_rate) ** years_elapsed)
        emissions_budget = target_intensity * activity_at_year
        reduction_pct = ((company_base - target_intensity) / company_base * 100.0) if company_base > 0 else 0.0

        return CompanyIntensityTarget(
            year=year,
            target_intensity=round(target_intensity, 6),
            benchmark_intensity=round(bm_intensity, 6),
            convergence_pct=round(convergence_frac * 100.0, 2),
            emissions_budget_tco2e=round(emissions_budget, 4),
            reduction_from_base_pct=round(max(reduction_pct, 0.0), 2),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Validation
    # -------------------------------------------------------------------------

    async def _phase_validation(self, config: SDATargetConfig) -> PhaseResult:
        """Validate against SBTi SDA criteria and compare with ACA."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        findings: List[SDAValidationFinding] = []

        # V1: Sector eligibility
        if self._classification.is_sda_eligible:
            findings.append(SDAValidationFinding(
                criterion="SDA-V1",
                description="Sector is SDA-eligible",
                severity="pass",
                detail=f"Sector: {self._classification.sector}",
            ))
        else:
            findings.append(SDAValidationFinding(
                criterion="SDA-V1",
                description="Sector is not SDA-eligible",
                severity="fail",
                detail=f"Sector '{self._classification.sector}' not in SDA methodology scope",
            ))

        # V2: Company intensity converges to benchmark by 2050
        if self._long_term:
            if self._long_term.convergence_pct >= 99.0:
                findings.append(SDAValidationFinding(
                    criterion="SDA-V2",
                    description="Company intensity converges to sector benchmark",
                    severity="pass",
                    detail=f"Convergence: {self._long_term.convergence_pct:.1f}%",
                ))
            else:
                findings.append(SDAValidationFinding(
                    criterion="SDA-V2",
                    description="Company intensity does not fully converge",
                    severity="warning",
                    detail=f"Convergence: {self._long_term.convergence_pct:.1f}% (target: 100%)",
                ))

        # V3: Near-term reduction rate meets minimum
        if self._near_term:
            years = self._near_term.year - config.base_year
            min_reduction = SDA_MIN_ANNUAL_REDUCTION_PCT * years
            if self._near_term.reduction_from_base_pct >= min_reduction:
                findings.append(SDAValidationFinding(
                    criterion="SDA-V3",
                    description="Near-term reduction meets SBTi minimum rate",
                    severity="pass",
                    detail=f"{self._near_term.reduction_from_base_pct:.1f}% >= {min_reduction:.1f}%",
                ))
            else:
                findings.append(SDAValidationFinding(
                    criterion="SDA-V3",
                    description="Near-term reduction below SBTi minimum",
                    severity="fail",
                    detail=f"{self._near_term.reduction_from_base_pct:.1f}% < {min_reduction:.1f}%",
                ))

        # V4: Long-term reduction >= 90%
        if self._long_term:
            if self._long_term.reduction_from_base_pct >= 90.0:
                findings.append(SDAValidationFinding(
                    criterion="SDA-V4",
                    description="Long-term reduction meets 90% minimum",
                    severity="pass",
                    detail=f"Reduction: {self._long_term.reduction_from_base_pct:.1f}%",
                ))
            else:
                findings.append(SDAValidationFinding(
                    criterion="SDA-V4",
                    description="Long-term reduction below 90%",
                    severity="fail",
                    detail=f"Reduction: {self._long_term.reduction_from_base_pct:.1f}%",
                ))

        # V5: Target year not later than 2050
        if config.convergence_year <= 2050:
            findings.append(SDAValidationFinding(
                criterion="SDA-V5",
                description="Convergence year is 2050 or earlier",
                severity="pass",
            ))
        else:
            findings.append(SDAValidationFinding(
                criterion="SDA-V5",
                description="Convergence year is after 2050",
                severity="warning",
                detail=f"Convergence year: {config.convergence_year}",
            ))

        # ACA comparison
        aca_comparison = self._compare_with_aca(config)

        pass_count = sum(1 for f in findings if f.severity == "pass")
        warn_count = sum(1 for f in findings if f.severity == "warning")
        fail_count = sum(1 for f in findings if f.severity == "fail")

        self._validation = SDAValidationReport(
            overall_valid=fail_count == 0,
            findings=findings,
            pass_count=pass_count,
            warning_count=warn_count,
            fail_count=fail_count,
            aca_comparison=aca_comparison,
        )

        outputs["overall_valid"] = self._validation.overall_valid
        outputs["pass_count"] = pass_count
        outputs["warning_count"] = warn_count
        outputs["fail_count"] = fail_count
        outputs["more_ambitious"] = aca_comparison.more_ambitious_approach

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Validation: valid=%s, pass=%d, warn=%d, fail=%d",
            self._validation.overall_valid, pass_count, warn_count, fail_count,
        )
        return PhaseResult(
            phase_name="validation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _compare_with_aca(self, config: SDATargetConfig) -> ACAComparison:
        """Compare SDA targets with ACA alternative."""
        nt_years = config.near_term_target_year - config.base_year
        lt_years = config.convergence_year - config.base_year

        aca_nt = min(ACA_ANNUAL_REDUCTION_PCT * nt_years, 100.0)
        aca_lt = min(ACA_ANNUAL_REDUCTION_PCT * lt_years, 100.0)

        sda_nt = self._near_term.reduction_from_base_pct if self._near_term else 0.0
        sda_lt = self._long_term.reduction_from_base_pct if self._long_term else 0.0

        if sda_nt > aca_nt:
            more_ambitious = "SDA"
            rec = (
                "SDA approach produces more ambitious near-term targets for this sector. "
                "Recommend SDA pathway for sector-specific credibility."
            )
        elif aca_nt > sda_nt:
            more_ambitious = "ACA"
            rec = (
                "ACA approach produces more ambitious near-term targets. "
                "SBTi requires the more ambitious of ACA or SDA for validation."
            )
        else:
            more_ambitious = "equivalent"
            rec = "Both approaches yield equivalent near-term reductions."

        return ACAComparison(
            aca_near_term_reduction_pct=round(aca_nt, 2),
            aca_long_term_reduction_pct=round(aca_lt, 2),
            sda_near_term_reduction_pct=round(sda_nt, 2),
            sda_long_term_reduction_pct=round(sda_lt, 2),
            more_ambitious_approach=more_ambitious,
            recommendation=rec,
        )
