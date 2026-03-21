# -*- coding: utf-8 -*-
"""
SDA Pathway Workflow
========================

4-phase workflow for Sectoral Decarbonisation Approach (SDA) pathway
assessment within PACK-023 SBTi Alignment Pack.  The workflow
classifies the company's sector using NACE/GICS codes, loads the
appropriate IEA NZE sector benchmark, calculates the SDA convergence
pathway with annual intensity milestones, and validates results
against the SBTi SDA Tool V3.0.

Phases:
    1. SectorClassify    -- Classify sector via NACE/GICS codes and validate SDA eligibility
    2. BenchmarkLoad     -- Load IEA NZE 2050 sector benchmarks and interpolate pathway
    3. ConvergenceCalc   -- Calculate SDA convergence pathway with intensity milestones
    4. Validate          -- Cross-validate against SBTi SDA Tool V3.0 and ambition check

Regulatory references:
    - SBTi Sectoral Decarbonisation Approach (SDA) V2.1 (2024)
    - SBTi SDA Tool V3.0 (2024)
    - SBTi Corporate Manual V5.3 (2024)
    - IEA Net Zero Emissions by 2050 Scenario (NZE 2023)
    - IEA Energy Technology Perspectives 2023
    - IPCC AR6 WG3 (2022) - Sector mitigation pathways
    - NACE Rev. 2.1 Statistical Classification
    - GICS Industry Classification Standard

Zero-hallucination: all sector benchmarks from IEA NZE 2023 published
data, convergence formula from SBTi SDA V2.1 specification.  No LLM
calls in the numeric computation path.

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"


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


class SDAEligibility(str, Enum):
    """SDA eligibility status for a sector."""

    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    CONDITIONAL = "conditional"


class AmbitionLevel(str, Enum):
    """Temperature ambition level of SDA targets."""

    CELSIUS_1_5 = "1.5C"
    WELL_BELOW_2C = "WB2C"
    CELSIUS_2C = "2C"
    INSUFFICIENT = "insufficient"


class CrossValidationStatus(str, Enum):
    """SDA Tool cross-validation status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_VALIDATED = "not_validated"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination Lookups)
# =============================================================================

# NACE Rev. 2.1 to SDA sector mapping
NACE_TO_SDA_SECTOR: Dict[str, str] = {
    # Power generation
    "D35.11": "power",
    "D35.12": "power",
    "D35.13": "power",
    "D35.14": "power",
    # Cement
    "C23.51": "cement",
    "C23.52": "cement",
    # Steel
    "C24.10": "steel",
    "C24.20": "steel",
    "C24.31": "steel",
    "C24.32": "steel",
    "C24.33": "steel",
    "C24.34": "steel",
    # Aluminium
    "C24.42": "aluminium",
    "C24.43": "aluminium",
    "C24.44": "aluminium",
    # Pulp and paper
    "C17.11": "pulp_paper",
    "C17.12": "pulp_paper",
    "C17.21": "pulp_paper",
    "C17.22": "pulp_paper",
    "C17.23": "pulp_paper",
    "C17.24": "pulp_paper",
    "C17.29": "pulp_paper",
    # Chemicals
    "C20.11": "chemicals",
    "C20.12": "chemicals",
    "C20.13": "chemicals",
    "C20.14": "chemicals",
    "C20.15": "chemicals",
    "C20.16": "chemicals",
    "C20.17": "chemicals",
    "C20.20": "chemicals",
    # Aviation
    "H51.10": "aviation",
    "H51.21": "aviation",
    # Maritime
    "H50.10": "maritime",
    "H50.20": "maritime",
    # Road transport
    "H49.10": "road_transport",
    "H49.20": "road_transport",
    "H49.31": "road_transport",
    "H49.32": "road_transport",
    "H49.39": "road_transport",
    "H49.41": "road_transport",
    "H49.42": "road_transport",
    # Buildings commercial
    "L68.10": "buildings_commercial",
    "L68.20": "buildings_commercial",
    "L68.31": "buildings_commercial",
    "L68.32": "buildings_commercial",
    # Buildings residential
    "F41.10": "buildings_residential",
    "F41.20": "buildings_residential",
    # Food and beverage
    "C10.11": "food_beverage",
    "C10.12": "food_beverage",
    "C10.13": "food_beverage",
    "C10.20": "food_beverage",
    "C10.31": "food_beverage",
    "C10.32": "food_beverage",
    "C10.39": "food_beverage",
    "C10.41": "food_beverage",
    "C10.42": "food_beverage",
    "C10.51": "food_beverage",
    "C10.52": "food_beverage",
    "C10.61": "food_beverage",
    "C10.62": "food_beverage",
    "C10.71": "food_beverage",
    "C10.72": "food_beverage",
    "C10.73": "food_beverage",
    "C10.81": "food_beverage",
    "C10.82": "food_beverage",
    "C10.83": "food_beverage",
    "C10.84": "food_beverage",
    "C10.85": "food_beverage",
    "C10.86": "food_beverage",
    "C10.89": "food_beverage",
    "C10.91": "food_beverage",
    "C10.92": "food_beverage",
    "C11.01": "food_beverage",
    "C11.02": "food_beverage",
    "C11.03": "food_beverage",
    "C11.04": "food_beverage",
    "C11.05": "food_beverage",
    "C11.06": "food_beverage",
    "C11.07": "food_beverage",
}

# GICS sub-industry to SDA sector mapping
GICS_TO_SDA_SECTOR: Dict[str, str] = {
    # Power (GICS 5510)
    "55101010": "power",
    "55101020": "power",
    "55101030": "power",
    "55101040": "power",
    "55105010": "power",
    "55105020": "power",
    # Materials - Cement (GICS 1510)
    "15102010": "cement",
    # Materials - Steel (GICS 1510)
    "15104010": "steel",
    "15104020": "steel",
    # Materials - Aluminium
    "15104030": "aluminium",
    "15104040": "aluminium",
    # Materials - Paper (GICS 1510)
    "15105010": "pulp_paper",
    "15105020": "pulp_paper",
    # Materials - Chemicals (GICS 1510)
    "15101010": "chemicals",
    "15101020": "chemicals",
    "15101030": "chemicals",
    "15101040": "chemicals",
    "15101050": "chemicals",
    # Transportation - Airlines (GICS 2030)
    "20301010": "aviation",
    # Transportation - Marine (GICS 2030)
    "20302010": "maritime",
    # Transportation - Trucking / Road
    "20304010": "road_transport",
    "20304020": "road_transport",
    # Real Estate (GICS 6010)
    "60101010": "buildings_commercial",
    "60101020": "buildings_commercial",
    "60101030": "buildings_commercial",
    "60101040": "buildings_commercial",
    "60101050": "buildings_commercial",
    "60101060": "buildings_commercial",
    "60101070": "buildings_commercial",
    "60101080": "buildings_commercial",
    # Consumer Staples - Food (GICS 3020)
    "30201010": "food_beverage",
    "30201020": "food_beverage",
    "30201030": "food_beverage",
    "30202010": "food_beverage",
    "30202030": "food_beverage",
    # Consumer Staples - Beverages (GICS 3020)
    "30201030": "food_beverage",
}

# SDA-eligible sectors (from SBTi SDA Tool V3.0)
SDA_ELIGIBLE_SECTORS: List[str] = [
    "power", "cement", "steel", "aluminium", "pulp_paper",
    "chemicals", "aviation", "maritime", "road_transport",
    "buildings_commercial", "buildings_residential", "food_beverage",
]

# Sector display names
SECTOR_DISPLAY_NAMES: Dict[str, str] = {
    "power": "Power Generation",
    "cement": "Cement",
    "steel": "Iron & Steel",
    "aluminium": "Aluminium",
    "pulp_paper": "Pulp & Paper",
    "chemicals": "Chemicals",
    "aviation": "Aviation",
    "maritime": "Maritime / Shipping",
    "road_transport": "Road Transport",
    "buildings_commercial": "Commercial Buildings",
    "buildings_residential": "Residential Buildings",
    "food_beverage": "Food & Beverage",
}

# Physical intensity metrics by sector
SECTOR_INTENSITY_UNITS: Dict[str, str] = {
    "power": "tCO2e/MWh",
    "cement": "tCO2e/tonne clinker",
    "steel": "tCO2e/tonne crude steel",
    "aluminium": "tCO2e/tonne primary aluminium",
    "pulp_paper": "tCO2e/tonne paper",
    "chemicals": "tCO2e/tonne chemical output",
    "aviation": "gCO2e/RPK",
    "maritime": "gCO2e/tonne-nm",
    "road_transport": "gCO2e/vkm",
    "buildings_commercial": "kgCO2e/m2",
    "buildings_residential": "kgCO2e/m2",
    "food_beverage": "tCO2e/tonne product",
}

# IEA NZE 2023 sector benchmarks: (base_year_intensity_2020, target_intensity_2050)
# Units match SECTOR_INTENSITY_UNITS
# Source: IEA Net Zero Emissions by 2050 Scenario (NZE 2023)
IEA_NZE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "power": {
        "base_year": 2020,
        "base_intensity": 0.459,       # tCO2e/MWh global average 2020
        "target_year": 2050,
        "target_intensity": -0.011,    # Net negative with BECCS/CCS
        "milestone_2030": 0.138,       # tCO2e/MWh
        "milestone_2040": 0.020,       # tCO2e/MWh
    },
    "cement": {
        "base_year": 2020,
        "base_intensity": 0.610,       # tCO2e/t clinker
        "target_year": 2050,
        "target_intensity": 0.060,     # With CCS
        "milestone_2030": 0.480,
        "milestone_2040": 0.230,
    },
    "steel": {
        "base_year": 2020,
        "base_intensity": 1.400,       # tCO2e/t crude steel
        "target_year": 2050,
        "target_intensity": 0.050,     # With hydrogen DRI
        "milestone_2030": 1.050,
        "milestone_2040": 0.370,
    },
    "aluminium": {
        "base_year": 2020,
        "base_intensity": 12.100,      # tCO2e/t primary aluminium
        "target_year": 2050,
        "target_intensity": 0.600,     # With inert anodes
        "milestone_2030": 8.200,
        "milestone_2040": 3.500,
    },
    "pulp_paper": {
        "base_year": 2020,
        "base_intensity": 0.340,       # tCO2e/t paper
        "target_year": 2050,
        "target_intensity": 0.020,
        "milestone_2030": 0.230,
        "milestone_2040": 0.100,
    },
    "chemicals": {
        "base_year": 2020,
        "base_intensity": 1.200,       # tCO2e/t chemical output
        "target_year": 2050,
        "target_intensity": 0.110,
        "milestone_2030": 0.900,
        "milestone_2040": 0.400,
    },
    "aviation": {
        "base_year": 2020,
        "base_intensity": 93.0,        # gCO2e/RPK
        "target_year": 2050,
        "target_intensity": 10.0,      # With SAF
        "milestone_2030": 72.0,
        "milestone_2040": 35.0,
    },
    "maritime": {
        "base_year": 2020,
        "base_intensity": 7.700,       # gCO2e/tonne-nm
        "target_year": 2050,
        "target_intensity": 0.800,     # With ammonia/hydrogen
        "milestone_2030": 5.500,
        "milestone_2040": 2.500,
    },
    "road_transport": {
        "base_year": 2020,
        "base_intensity": 170.0,       # gCO2e/vkm
        "target_year": 2050,
        "target_intensity": 5.0,       # Full EV fleet
        "milestone_2030": 115.0,
        "milestone_2040": 40.0,
    },
    "buildings_commercial": {
        "base_year": 2020,
        "base_intensity": 36.0,        # kgCO2e/m2
        "target_year": 2050,
        "target_intensity": 2.0,
        "milestone_2030": 22.0,
        "milestone_2040": 8.0,
    },
    "buildings_residential": {
        "base_year": 2020,
        "base_intensity": 24.0,        # kgCO2e/m2
        "target_year": 2050,
        "target_intensity": 1.5,
        "milestone_2030": 15.0,
        "milestone_2040": 5.5,
    },
    "food_beverage": {
        "base_year": 2020,
        "base_intensity": 0.520,       # tCO2e/t product
        "target_year": 2050,
        "target_intensity": 0.050,
        "milestone_2030": 0.370,
        "milestone_2040": 0.170,
    },
}

# SDA Tool V3.0 cross-validation tolerance
SDA_TOOL_TOLERANCE_PCT = 2.0  # 2% maximum deviation


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


class SectorClassification(BaseModel):
    """Result of sector classification and SDA eligibility check."""

    sector_code: str = Field(default="", description="SDA sector identifier")
    sector_name: str = Field(default="", description="Display name")
    nace_codes: List[str] = Field(default_factory=list)
    gics_codes: List[str] = Field(default_factory=list)
    eligibility: SDAEligibility = Field(default=SDAEligibility.NOT_ELIGIBLE)
    intensity_unit: str = Field(default="")
    classification_source: str = Field(default="manual")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list)


class BenchmarkPoint(BaseModel):
    """A single year-intensity point on the sector benchmark pathway."""

    year: int = Field(...)
    sector_intensity: float = Field(default=0.0)
    source: str = Field(default="IEA NZE 2023")


class SectorBenchmark(BaseModel):
    """IEA NZE sector benchmark data for SDA convergence."""

    sector: str = Field(default="")
    intensity_unit: str = Field(default="")
    base_year: int = Field(default=2020)
    base_intensity: float = Field(default=0.0)
    target_year: int = Field(default=2050)
    target_intensity: float = Field(default=0.0)
    milestone_2030: float = Field(default=0.0)
    milestone_2040: float = Field(default=0.0)
    pathway_points: List[BenchmarkPoint] = Field(default_factory=list)
    source: str = Field(default="IEA NZE 2023")


class IntensityMilestone(BaseModel):
    """Annual intensity milestone on the SDA convergence pathway."""

    year: int = Field(...)
    company_intensity: float = Field(default=0.0)
    sector_intensity: float = Field(default=0.0)
    convergence_gap: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    absolute_emissions_tco2e: float = Field(default=0.0)


class ConvergenceResult(BaseModel):
    """Complete SDA convergence pathway calculation."""

    company_base_intensity: float = Field(default=0.0)
    company_target_intensity: float = Field(default=0.0)
    sector_base_intensity: float = Field(default=0.0)
    sector_target_intensity: float = Field(default=0.0)
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2030)
    cumulative_reduction_pct: float = Field(default=0.0)
    average_annual_reduction_rate: float = Field(default=0.0)
    milestones: List[IntensityMilestone] = Field(default_factory=list)
    converges_by_target: bool = Field(default=False)
    convergence_year: Optional[int] = Field(None)
    intensity_unit: str = Field(default="")


class CrossValidationResult(BaseModel):
    """SDA Tool V3.0 cross-validation assessment."""

    status: CrossValidationStatus = Field(default=CrossValidationStatus.NOT_VALIDATED)
    max_deviation_pct: float = Field(default=0.0)
    avg_deviation_pct: float = Field(default=0.0)
    deviations_by_year: Dict[int, float] = Field(default_factory=dict)
    tolerance_pct: float = Field(default=SDA_TOOL_TOLERANCE_PCT)
    years_exceeding_tolerance: List[int] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class AmbitionAssessment(BaseModel):
    """Ambition assessment of the SDA target."""

    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.INSUFFICIENT)
    annual_reduction_rate: float = Field(default=0.0)
    meets_1_5c: bool = Field(default=False)
    meets_wb2c: bool = Field(default=False)
    sector_aligned: bool = Field(default=False)
    assessment_notes: List[str] = Field(default_factory=list)


class SDAPathwayConfig(BaseModel):
    """Configuration for the SDA pathway workflow."""

    # Sector identification (at least one should be provided)
    nace_codes: List[str] = Field(default_factory=list,
                                  description="NACE Rev. 2.1 codes for sector classification")
    gics_codes: List[str] = Field(default_factory=list,
                                  description="GICS sub-industry codes for sector classification")
    sector_override: str = Field(default="",
                                 description="Manual sector override (bypasses NACE/GICS lookup)")

    # Company emissions and activity data
    base_year: int = Field(default=2022, ge=2015, le=2050)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    company_base_intensity: float = Field(default=0.0, ge=0.0,
                                          description="Company intensity at base year")
    company_base_emissions_tco2e: float = Field(default=0.0, ge=0.0,
                                                description="Company S1+S2 emissions at base year")
    activity_value: float = Field(default=0.0, ge=0.0,
                                  description="Activity metric value (MWh, tonnes, etc.)")
    activity_unit: str = Field(default="", description="Activity metric unit")
    projected_activity_growth_pct: float = Field(default=0.0,
                                                 description="Annual activity growth rate (%)")

    # SDA Tool reference values for cross-validation
    sda_tool_reference: Dict[int, float] = Field(
        default_factory=dict,
        description="SDA Tool V3.0 reference intensities by year for cross-validation",
    )

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("nace_codes")
    @classmethod
    def _validate_nace(cls, v: List[str]) -> List[str]:
        for code in v:
            parts = code.split(".")
            if len(parts) < 2:
                raise ValueError(f"NACE code must be in format X99.99, got '{code}'")
        return v


class SDAPathwayResult(BaseModel):
    """Complete result from the SDA pathway workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sda_pathway")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    sector_classification: Optional[SectorClassification] = Field(None)
    sector_benchmark: Optional[SectorBenchmark] = Field(None)
    convergence: Optional[ConvergenceResult] = Field(None)
    cross_validation: Optional[CrossValidationResult] = Field(None)
    ambition: Optional[AmbitionAssessment] = Field(None)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SDAPathwayWorkflow:
    """
    4-phase SDA sector pathway workflow for SBTi alignment.

    Classifies the company's sector using NACE/GICS codes, loads the
    IEA NZE benchmark for the matched sector, calculates the SDA
    convergence pathway from the company's base intensity to the
    sector target, and cross-validates against the SBTi SDA Tool V3.0.

    Zero-hallucination: all sector benchmarks are from IEA NZE 2023
    published data.  The SDA convergence formula is from SBTi SDA V2.1
    specification.  No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = SDAPathwayWorkflow()
        >>> config = SDAPathwayConfig(
        ...     nace_codes=["D35.11"],
        ...     base_year=2022,
        ...     target_year=2030,
        ...     company_base_intensity=0.520,
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self) -> None:
        """Initialise SDAPathwayWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._classification: Optional[SectorClassification] = None
        self._benchmark: Optional[SectorBenchmark] = None
        self._convergence: Optional[ConvergenceResult] = None
        self._cross_validation: Optional[CrossValidationResult] = None
        self._ambition: Optional[AmbitionAssessment] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: SDAPathwayConfig) -> SDAPathwayResult:
        """
        Execute the 4-phase SDA pathway workflow.

        Args:
            config: SDA pathway configuration with sector codes,
                base year intensity, and activity data.

        Returns:
            SDAPathwayResult with sector classification, benchmark,
            convergence pathway, and validation results.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting SDA pathway workflow %s, nace=%s, gics=%s",
            self.workflow_id, config.nace_codes, config.gics_codes,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Sector Classification
            phase1 = await self._phase_sector_classify(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Sector classification failed; cannot proceed")

            # Phase 2: Benchmark Load
            phase2 = await self._phase_benchmark_load(config)
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise ValueError("Benchmark load failed; cannot proceed")

            # Phase 3: Convergence Calculation
            phase3 = await self._phase_convergence_calc(config)
            self._phase_results.append(phase3)

            # Phase 4: Validation
            phase4 = await self._phase_validate(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("SDA pathway workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = SDAPathwayResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            sector_classification=self._classification,
            sector_benchmark=self._benchmark,
            convergence=self._convergence,
            cross_validation=self._cross_validation,
            ambition=self._ambition,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "SDA pathway workflow %s completed in %.2fs, sector=%s",
            self.workflow_id, elapsed,
            self._classification.sector_code if self._classification else "unknown",
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Sector Classification
    # -------------------------------------------------------------------------

    async def _phase_sector_classify(self, config: SDAPathwayConfig) -> PhaseResult:
        """Classify sector via NACE/GICS codes and validate SDA eligibility."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        matched_sector = ""
        classification_source = ""
        confidence = 0.0
        notes: List[str] = []

        # Priority: sector_override > NACE > GICS
        if config.sector_override:
            sector = config.sector_override.lower().strip()
            if sector in SDA_ELIGIBLE_SECTORS:
                matched_sector = sector
                classification_source = "manual_override"
                confidence = 1.0
                notes.append(f"Manual sector override: {sector}")
            else:
                errors.append(
                    f"Manual sector override '{sector}' is not a valid SDA sector. "
                    f"Valid sectors: {', '.join(SDA_ELIGIBLE_SECTORS)}"
                )
        elif config.nace_codes:
            # Try NACE code matching
            nace_matches: Dict[str, int] = {}
            for nace in config.nace_codes:
                sector = NACE_TO_SDA_SECTOR.get(nace, "")
                if sector:
                    nace_matches[sector] = nace_matches.get(sector, 0) + 1

            if nace_matches:
                # Use sector with most NACE code matches
                matched_sector = max(nace_matches, key=lambda s: nace_matches[s])
                total_codes = len(config.nace_codes)
                matching_codes = nace_matches[matched_sector]
                confidence = min(matching_codes / max(total_codes, 1), 1.0)
                classification_source = "nace"
                notes.append(
                    f"NACE classification: {matching_codes}/{total_codes} codes "
                    f"map to {matched_sector}"
                )

                # Check for ambiguous classification
                if len(nace_matches) > 1:
                    warnings.append(
                        f"Multiple SDA sectors matched from NACE codes: "
                        f"{list(nace_matches.keys())}; using '{matched_sector}' "
                        f"with highest match count"
                    )
            else:
                warnings.append(
                    f"No NACE codes matched SDA-eligible sectors: {config.nace_codes}"
                )
        elif config.gics_codes:
            # Try GICS code matching
            gics_matches: Dict[str, int] = {}
            for gics in config.gics_codes:
                sector = GICS_TO_SDA_SECTOR.get(gics, "")
                if sector:
                    gics_matches[sector] = gics_matches.get(sector, 0) + 1

            if gics_matches:
                matched_sector = max(gics_matches, key=lambda s: gics_matches[s])
                total_codes = len(config.gics_codes)
                matching_codes = gics_matches[matched_sector]
                confidence = min(matching_codes / max(total_codes, 1), 1.0)
                classification_source = "gics"
                notes.append(
                    f"GICS classification: {matching_codes}/{total_codes} codes "
                    f"map to {matched_sector}"
                )
            else:
                warnings.append(
                    f"No GICS codes matched SDA-eligible sectors: {config.gics_codes}"
                )

        # Determine eligibility
        if matched_sector and matched_sector in SDA_ELIGIBLE_SECTORS:
            eligibility = SDAEligibility.ELIGIBLE
        elif matched_sector:
            eligibility = SDAEligibility.CONDITIONAL
            warnings.append(
                f"Sector '{matched_sector}' is conditionally eligible for SDA; "
                "verify with SBTi before submission"
            )
        else:
            eligibility = SDAEligibility.NOT_ELIGIBLE
            if not errors:
                errors.append(
                    "Could not determine SDA-eligible sector from provided codes. "
                    "Provide NACE/GICS codes or use sector_override."
                )

        # Get intensity unit
        intensity_unit = SECTOR_INTENSITY_UNITS.get(matched_sector, "")

        self._classification = SectorClassification(
            sector_code=matched_sector,
            sector_name=SECTOR_DISPLAY_NAMES.get(matched_sector, matched_sector),
            nace_codes=config.nace_codes,
            gics_codes=config.gics_codes,
            eligibility=eligibility,
            intensity_unit=intensity_unit,
            classification_source=classification_source,
            confidence=round(confidence, 4),
            notes=notes,
        )

        outputs["sector_code"] = matched_sector
        outputs["sector_name"] = SECTOR_DISPLAY_NAMES.get(matched_sector, "")
        outputs["eligibility"] = eligibility.value
        outputs["classification_source"] = classification_source
        outputs["confidence"] = round(confidence, 4)
        outputs["intensity_unit"] = intensity_unit
        outputs["nace_codes_provided"] = len(config.nace_codes)
        outputs["gics_codes_provided"] = len(config.gics_codes)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Sector classify: sector=%s, eligibility=%s, confidence=%.2f",
            matched_sector, eligibility.value, confidence,
        )
        return PhaseResult(
            phase_name="sector_classify",
            status=PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Benchmark Load
    # -------------------------------------------------------------------------

    async def _phase_benchmark_load(self, config: SDAPathwayConfig) -> PhaseResult:
        """Load IEA NZE 2050 sector benchmarks and interpolate pathway."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        if not self._classification or not self._classification.sector_code:
            errors.append("No sector classified; cannot load benchmark")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="benchmark_load", status=PhaseStatus.FAILED,
                duration_seconds=round(elapsed, 4), errors=errors,
            )

        sector = self._classification.sector_code
        benchmark_data = IEA_NZE_BENCHMARKS.get(sector)

        if not benchmark_data:
            errors.append(f"No IEA NZE benchmark available for sector '{sector}'")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="benchmark_load", status=PhaseStatus.FAILED,
                duration_seconds=round(elapsed, 4), errors=errors,
            )

        # Build interpolated pathway from base to target
        bm_base_year = benchmark_data["base_year"]
        bm_target_year = benchmark_data["target_year"]
        bm_base_intensity = benchmark_data["base_intensity"]
        bm_target_intensity = benchmark_data["target_intensity"]
        bm_2030 = benchmark_data["milestone_2030"]
        bm_2040 = benchmark_data["milestone_2040"]

        # Create pathway points with piecewise linear interpolation
        # using IEA milestones at 2030 and 2040
        pathway_points: List[BenchmarkPoint] = []

        for year in range(bm_base_year, bm_target_year + 1):
            if year <= 2030:
                # Interpolate between base and 2030
                fraction = (year - bm_base_year) / max(2030 - bm_base_year, 1)
                intensity = bm_base_intensity + (bm_2030 - bm_base_intensity) * fraction
            elif year <= 2040:
                # Interpolate between 2030 and 2040
                fraction = (year - 2030) / 10.0
                intensity = bm_2030 + (bm_2040 - bm_2030) * fraction
            else:
                # Interpolate between 2040 and 2050
                fraction = (year - 2040) / 10.0
                intensity = bm_2040 + (bm_target_intensity - bm_2040) * fraction

            pathway_points.append(BenchmarkPoint(
                year=year,
                sector_intensity=round(intensity, 6),
                source="IEA NZE 2023",
            ))

        intensity_unit = SECTOR_INTENSITY_UNITS.get(sector, "")

        self._benchmark = SectorBenchmark(
            sector=sector,
            intensity_unit=intensity_unit,
            base_year=bm_base_year,
            base_intensity=bm_base_intensity,
            target_year=bm_target_year,
            target_intensity=bm_target_intensity,
            milestone_2030=bm_2030,
            milestone_2040=bm_2040,
            pathway_points=pathway_points,
            source="IEA NZE 2023",
        )

        # Validate company base year falls within benchmark range
        if config.base_year < bm_base_year:
            warnings.append(
                f"Company base year {config.base_year} is before benchmark base year "
                f"{bm_base_year}; extrapolation may be less reliable"
            )

        outputs["sector"] = sector
        outputs["benchmark_base_year"] = bm_base_year
        outputs["benchmark_base_intensity"] = bm_base_intensity
        outputs["benchmark_target_year"] = bm_target_year
        outputs["benchmark_target_intensity"] = bm_target_intensity
        outputs["benchmark_2030"] = bm_2030
        outputs["benchmark_2040"] = bm_2040
        outputs["pathway_points_count"] = len(pathway_points)
        outputs["intensity_unit"] = intensity_unit
        outputs["benchmark_source"] = "IEA NZE 2023"

        # Calculate total sector reduction
        if bm_base_intensity > 0:
            sector_reduction_pct = (1.0 - bm_target_intensity / bm_base_intensity) * 100.0
            outputs["sector_total_reduction_pct"] = round(sector_reduction_pct, 2)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Benchmark load: sector=%s, base=%.4f, target=%.4f %s, points=%d",
            sector, bm_base_intensity, bm_target_intensity, intensity_unit,
            len(pathway_points),
        )
        return PhaseResult(
            phase_name="benchmark_load",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Convergence Calculation
    # -------------------------------------------------------------------------

    async def _phase_convergence_calc(self, config: SDAPathwayConfig) -> PhaseResult:
        """Calculate SDA convergence pathway with intensity milestones."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        if not self._benchmark:
            errors.append("No benchmark loaded; cannot calculate convergence")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="convergence_calc", status=PhaseStatus.FAILED,
                duration_seconds=round(elapsed, 4), errors=errors,
            )

        bm = self._benchmark
        company_base = config.company_base_intensity

        if company_base <= 0:
            errors.append("Company base intensity must be > 0 for SDA calculation")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="convergence_calc", status=PhaseStatus.FAILED,
                duration_seconds=round(elapsed, 4), errors=errors,
            )

        # Get sector pathway intensities at the company's base year and target year
        sector_base = self._get_sector_intensity(config.base_year)
        sector_target = self._get_sector_intensity(config.target_year)
        sector_endpoint = self._get_sector_intensity(bm.target_year)

        # SDA Convergence Formula:
        # I(t) = I_sector(t) + (I_company(base) - I_sector(base))
        #        * ((I_sector(target) - I_sector(t))
        #           / (I_sector(target) - I_sector(base)))
        #
        # Where I_sector(target) is the endpoint year (2050) intensity

        milestones: List[IntensityMilestone] = []
        convergence_year: Optional[int] = None
        prev_company_intensity = company_base

        # Activity growth projection for absolute emissions
        base_activity = config.activity_value if config.activity_value > 0 else 1.0
        annual_growth = config.projected_activity_growth_pct / 100.0

        for year in range(config.base_year, config.target_year + 1):
            sector_at_year = self._get_sector_intensity(year)

            if year == config.base_year:
                company_at_year = company_base
            else:
                # SDA convergence formula
                denominator = sector_endpoint - sector_base
                if abs(denominator) < 1e-10:
                    # Sector is already at target; company converges linearly
                    fraction = (year - config.base_year) / max(config.target_year - config.base_year, 1)
                    company_at_year = company_base + (sector_target - company_base) * fraction
                else:
                    convergence_factor = (sector_endpoint - sector_at_year) / denominator
                    company_at_year = (
                        sector_at_year
                        + (company_base - sector_base) * convergence_factor
                    )

            # Calculate annual reduction rate
            annual_rate_pct = 0.0
            if prev_company_intensity > 0 and year > config.base_year:
                annual_rate_pct = (1.0 - company_at_year / prev_company_intensity) * 100.0

            # Cumulative reduction from base
            cumulative_pct = 0.0
            if company_base > 0:
                cumulative_pct = (1.0 - company_at_year / company_base) * 100.0

            # Convergence gap
            convergence_gap = company_at_year - sector_at_year

            # Absolute emissions (intensity * projected activity)
            years_from_base = year - config.base_year
            projected_activity = base_activity * ((1.0 + annual_growth) ** years_from_base)
            absolute_emissions = company_at_year * projected_activity

            # Check for convergence
            if convergence_gap <= 0 and convergence_year is None and year > config.base_year:
                convergence_year = year

            milestones.append(IntensityMilestone(
                year=year,
                company_intensity=round(company_at_year, 6),
                sector_intensity=round(sector_at_year, 6),
                convergence_gap=round(convergence_gap, 6),
                annual_reduction_rate_pct=round(annual_rate_pct, 4),
                cumulative_reduction_pct=round(cumulative_pct, 4),
                absolute_emissions_tco2e=round(absolute_emissions, 2),
            ))

            prev_company_intensity = company_at_year

        # Calculate company target intensity at target year
        company_target = milestones[-1].company_intensity if milestones else 0.0

        # Average annual reduction rate
        total_years = config.target_year - config.base_year
        avg_annual_rate = 0.0
        if total_years > 0 and company_base > 0 and company_target > 0:
            avg_annual_rate = (1.0 - (company_target / company_base) ** (1.0 / total_years)) * 100.0

        converges = company_target <= sector_target

        self._convergence = ConvergenceResult(
            company_base_intensity=round(company_base, 6),
            company_target_intensity=round(company_target, 6),
            sector_base_intensity=round(sector_base, 6),
            sector_target_intensity=round(sector_target, 6),
            base_year=config.base_year,
            target_year=config.target_year,
            cumulative_reduction_pct=round(
                (1.0 - company_target / company_base) * 100.0 if company_base > 0 else 0.0, 4
            ),
            average_annual_reduction_rate=round(avg_annual_rate, 4),
            milestones=milestones,
            converges_by_target=converges,
            convergence_year=convergence_year,
            intensity_unit=bm.intensity_unit,
        )

        # Warnings
        if not converges:
            warnings.append(
                f"Company intensity ({company_target:.4f}) does not converge to sector "
                f"target ({sector_target:.4f}) by {config.target_year}. "
                "Consider extending target year or increasing ambition."
            )

        if company_base < sector_base:
            warnings.append(
                f"Company base intensity ({company_base:.4f}) is below sector average "
                f"({sector_base:.4f}); company is already a sector leader. "
                "SDA may produce less aggressive targets than ACA."
            )

        outputs["company_base_intensity"] = round(company_base, 6)
        outputs["company_target_intensity"] = round(company_target, 6)
        outputs["sector_base_intensity"] = round(sector_base, 6)
        outputs["sector_target_intensity"] = round(sector_target, 6)
        outputs["cumulative_reduction_pct"] = round(
            (1.0 - company_target / company_base) * 100.0 if company_base > 0 else 0.0, 4
        )
        outputs["average_annual_reduction_rate_pct"] = round(avg_annual_rate, 4)
        outputs["milestones_count"] = len(milestones)
        outputs["converges_by_target"] = converges
        outputs["convergence_year"] = convergence_year

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Convergence: base=%.4f -> target=%.4f, reduction=%.1f%%, converges=%s",
            company_base, company_target,
            (1.0 - company_target / company_base) * 100.0 if company_base > 0 else 0.0,
            converges,
        )
        return PhaseResult(
            phase_name="convergence_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _get_sector_intensity(self, year: int) -> float:
        """Get sector pathway intensity at a specific year from the benchmark."""
        if not self._benchmark or not self._benchmark.pathway_points:
            return 0.0

        # Find exact match or interpolate
        points = self._benchmark.pathway_points
        for point in points:
            if point.year == year:
                return point.sector_intensity

        # If year is before first point, extrapolate backwards
        if year < points[0].year:
            if len(points) >= 2:
                rate = (points[1].sector_intensity - points[0].sector_intensity) / max(
                    points[1].year - points[0].year, 1
                )
                return points[0].sector_intensity + rate * (year - points[0].year)
            return points[0].sector_intensity

        # If year is after last point, extrapolate forwards
        if year > points[-1].year:
            return points[-1].sector_intensity

        # Interpolate between surrounding points
        for i in range(len(points) - 1):
            if points[i].year <= year <= points[i + 1].year:
                fraction = (year - points[i].year) / max(
                    points[i + 1].year - points[i].year, 1
                )
                return (
                    points[i].sector_intensity
                    + (points[i + 1].sector_intensity - points[i].sector_intensity) * fraction
                )

        return points[-1].sector_intensity

    # -------------------------------------------------------------------------
    # Phase 4: Validation
    # -------------------------------------------------------------------------

    async def _phase_validate(self, config: SDAPathwayConfig) -> PhaseResult:
        """Cross-validate against SBTi SDA Tool V3.0 and assess ambition."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # ----- Cross-validation against SDA Tool V3.0 -----
        cross_val_status = CrossValidationStatus.NOT_VALIDATED
        deviations: Dict[int, float] = {}
        max_deviation = 0.0
        avg_deviation = 0.0
        years_exceeding: List[int] = []
        cv_notes: List[str] = []

        if config.sda_tool_reference and self._convergence and self._convergence.milestones:
            # Compare calculated pathway against SDA Tool reference values
            deviation_sum = 0.0
            deviation_count = 0

            for milestone in self._convergence.milestones:
                ref_value = config.sda_tool_reference.get(milestone.year)
                if ref_value is not None and ref_value > 0:
                    deviation_pct = abs(
                        milestone.company_intensity - ref_value
                    ) / ref_value * 100.0
                    deviations[milestone.year] = round(deviation_pct, 4)
                    deviation_sum += deviation_pct
                    deviation_count += 1

                    if deviation_pct > max_deviation:
                        max_deviation = deviation_pct

                    if deviation_pct > SDA_TOOL_TOLERANCE_PCT:
                        years_exceeding.append(milestone.year)

            if deviation_count > 0:
                avg_deviation = deviation_sum / deviation_count

            if years_exceeding:
                cross_val_status = CrossValidationStatus.FAIL
                cv_notes.append(
                    f"Deviation exceeds {SDA_TOOL_TOLERANCE_PCT}% tolerance in "
                    f"{len(years_exceeding)} years: {years_exceeding}"
                )
            elif deviation_count > 0:
                cross_val_status = CrossValidationStatus.PASS
                cv_notes.append(
                    f"All {deviation_count} years within {SDA_TOOL_TOLERANCE_PCT}% tolerance"
                )
            else:
                cross_val_status = CrossValidationStatus.WARNING
                cv_notes.append("No comparable reference years found in SDA Tool data")
        else:
            cv_notes.append(
                "No SDA Tool V3.0 reference values provided; cross-validation skipped"
            )

        self._cross_validation = CrossValidationResult(
            status=cross_val_status,
            max_deviation_pct=round(max_deviation, 4),
            avg_deviation_pct=round(avg_deviation, 4),
            deviations_by_year=deviations,
            tolerance_pct=SDA_TOOL_TOLERANCE_PCT,
            years_exceeding_tolerance=years_exceeding,
            notes=cv_notes,
        )

        # ----- Ambition assessment -----
        annual_rate = 0.0
        if self._convergence:
            annual_rate = self._convergence.average_annual_reduction_rate

        # Classify ambition level
        if annual_rate >= 4.2:
            ambition_level = AmbitionLevel.CELSIUS_1_5
        elif annual_rate >= 2.5:
            ambition_level = AmbitionLevel.WELL_BELOW_2C
        elif annual_rate >= 1.5:
            ambition_level = AmbitionLevel.CELSIUS_2C
        else:
            ambition_level = AmbitionLevel.INSUFFICIENT

        meets_1_5c = annual_rate >= 4.2
        meets_wb2c = annual_rate >= 2.5
        sector_aligned = self._convergence.converges_by_target if self._convergence else False

        assessment_notes: List[str] = []
        assessment_notes.append(
            f"Average annual intensity reduction: {annual_rate:.2f}%/yr"
        )
        assessment_notes.append(f"Ambition classification: {ambition_level.value}")

        if sector_aligned:
            assessment_notes.append(
                "Company intensity converges to or below sector pathway by target year"
            )
        else:
            assessment_notes.append(
                "Company intensity does not converge to sector pathway by target year"
            )

        if not meets_wb2c:
            warnings.append(
                f"Annual reduction rate ({annual_rate:.2f}%) is below WB2C minimum (2.5%/yr). "
                "SBTi may not validate this target."
            )

        self._ambition = AmbitionAssessment(
            ambition_level=ambition_level,
            annual_reduction_rate=round(annual_rate, 4),
            meets_1_5c=meets_1_5c,
            meets_wb2c=meets_wb2c,
            sector_aligned=sector_aligned,
            assessment_notes=assessment_notes,
        )

        outputs["cross_validation_status"] = cross_val_status.value
        outputs["max_deviation_pct"] = round(max_deviation, 4)
        outputs["avg_deviation_pct"] = round(avg_deviation, 4)
        outputs["years_exceeding_tolerance"] = len(years_exceeding)
        outputs["ambition_level"] = ambition_level.value
        outputs["annual_reduction_rate_pct"] = round(annual_rate, 4)
        outputs["meets_1_5c"] = meets_1_5c
        outputs["meets_wb2c"] = meets_wb2c
        outputs["sector_aligned"] = sector_aligned

        if cross_val_status == CrossValidationStatus.FAIL:
            warnings.append(
                f"SDA Tool cross-validation failed: max deviation {max_deviation:.2f}% "
                f"exceeds {SDA_TOOL_TOLERANCE_PCT}% tolerance"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Validate: cross_val=%s, ambition=%s, annual_rate=%.2f%%",
            cross_val_status.value, ambition_level.value, annual_rate,
        )
        return PhaseResult(
            phase_name="validate",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
