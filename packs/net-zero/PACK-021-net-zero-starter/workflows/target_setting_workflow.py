# -*- coding: utf-8 -*-
"""
Target Setting Workflow
===========================

4-phase workflow for defining SBTi-aligned net-zero targets within
PACK-021 Net-Zero Starter Pack.  The workflow analyses the organisation's
sector classification, selects the appropriate SBTi pathway, defines
near-term and long-term targets, and validates them against the SBTi
Net-Zero Standard v1.2 criteria.

Phases:
    1. SectorAnalysis     -- Determine sector, applicable pathways, FLAG relevance
    2. PathwaySelection   -- Select pathway (ACA/SDA/FLAG), ambition level, reduction rates
    3. TargetDefinition   -- Define near-term / long-term targets with scope coverage
    4. Validation         -- Validate against SBTi Net-Zero Standard v1.2

Regulatory references:
    - SBTi Net-Zero Standard v1.2 (2024)
    - SBTi Corporate Near-Term Criteria v5.1
    - GHG Protocol Corporate Standard
    - Paris Agreement Art. 2.1(a) - 1.5 deg C goal

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"


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


class SBTiPathway(str, Enum):
    """SBTi target-setting pathways."""

    ACA = "absolute_contraction"
    SDA = "sectoral_decarbonisation"
    FLAG = "forest_land_agriculture"


class AmbitionLevel(str, Enum):
    """Target ambition levels."""

    CELSIUS_1_5 = "1.5C"
    WELL_BELOW_2 = "WB2C"


class TargetType(str, Enum):
    """Target time horizon types."""

    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"


class ValidationSeverity(str, Enum):
    """Validation finding severity."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


# =============================================================================
# SBTi REFERENCE DATA (Zero-Hallucination, from SBTi NZ Standard v1.2)
# =============================================================================

# Annual linear reduction rates required by SBTi
SBTI_REDUCTION_RATES: Dict[str, Dict[str, float]] = {
    "1.5C": {
        "absolute_contraction": 4.2,      # % per year, cross-sector
        "sectoral_decarbonisation": 4.2,   # % per year (sector-specific varies)
        "forest_land_agriculture": 3.0,    # FLAG-specific minimum
    },
    "WB2C": {
        "absolute_contraction": 2.5,
        "sectoral_decarbonisation": 2.5,
        "forest_land_agriculture": 2.0,
    },
}

# SBTi minimum scope coverage requirements
SBTI_SCOPE_COVERAGE: Dict[str, Dict[str, float]] = {
    "near_term": {
        "scope1": 95.0,    # % of Scope 1 emissions
        "scope2": 95.0,    # % of Scope 2 emissions
        "scope3": 67.0,    # % of Scope 3 if > 40% of total
    },
    "long_term": {
        "scope1": 95.0,
        "scope2": 95.0,
        "scope3": 90.0,
    },
}

# SBTi long-term target minimum reduction
SBTI_LONG_TERM_MINIMUM_REDUCTION_PCT = 90.0

# SBTi maximum target year constraints
SBTI_NEAR_TERM_MAX_YEARS = 10   # From base year, max 5-10 years
SBTI_LONG_TERM_TARGET_YEAR = 2050

# Sector classifications and applicable pathways
SECTOR_PATHWAY_MAP: Dict[str, List[str]] = {
    "power_generation": ["sectoral_decarbonisation"],
    "oil_and_gas": ["sectoral_decarbonisation"],
    "cement": ["sectoral_decarbonisation"],
    "steel": ["sectoral_decarbonisation"],
    "aluminum": ["sectoral_decarbonisation"],
    "pulp_and_paper": ["sectoral_decarbonisation"],
    "aviation": ["sectoral_decarbonisation"],
    "shipping": ["sectoral_decarbonisation"],
    "chemicals": ["sectoral_decarbonisation", "absolute_contraction"],
    "agriculture": ["forest_land_agriculture", "absolute_contraction"],
    "forestry": ["forest_land_agriculture"],
    "food_and_beverage": ["forest_land_agriculture", "absolute_contraction"],
    "apparel": ["absolute_contraction", "sectoral_decarbonisation"],
    "financial_services": ["absolute_contraction"],
    "technology": ["absolute_contraction"],
    "real_estate": ["sectoral_decarbonisation", "absolute_contraction"],
    "retail": ["absolute_contraction"],
    "healthcare": ["absolute_contraction"],
    "manufacturing": ["absolute_contraction", "sectoral_decarbonisation"],
    "services": ["absolute_contraction"],
    "other": ["absolute_contraction"],
}

# SDA intensity benchmarks (tCO2e per unit) for 1.5C in 2030
SDA_INTENSITY_BENCHMARKS_2030: Dict[str, Dict[str, Any]] = {
    "power_generation": {"metric": "tCO2e/MWh", "value_2030": 0.138, "value_2050": 0.0},
    "cement": {"metric": "tCO2e/tonne_cement", "value_2030": 0.469, "value_2050": 0.143},
    "steel": {"metric": "tCO2e/tonne_steel", "value_2030": 1.013, "value_2050": 0.050},
    "aluminum": {"metric": "tCO2e/tonne_al", "value_2030": 4.600, "value_2050": 1.100},
    "pulp_and_paper": {"metric": "tCO2e/tonne_product", "value_2030": 0.210, "value_2050": 0.040},
    "real_estate": {"metric": "kgCO2e/sqm", "value_2030": 22.0, "value_2050": 2.5},
}

# FLAG sectors requiring FLAG target
FLAG_SECTORS = {"agriculture", "forestry", "food_and_beverage"}


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


class BaselineEmissions(BaseModel):
    """Baseline emissions for target-setting context."""

    base_year: int = Field(default=2024, ge=2015, le=2050)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0, description="FLAG-related emissions")
    intensity_metric: Optional[str] = Field(None, description="e.g. tCO2e/revenue")
    intensity_value: Optional[float] = Field(None, ge=0.0)


class TargetSettingConfig(BaseModel):
    """Configuration for the target setting workflow."""

    base_year: int = Field(default=2024, ge=2015, le=2050)
    baseline_emissions: BaselineEmissions = Field(default_factory=BaselineEmissions)
    sector: str = Field(default="other", description="Primary sector classification")
    sub_sector: str = Field(default="", description="Sub-sector for SDA")
    preferred_pathway: Optional[str] = Field(None, description="Preferred SBTi pathway")
    ambition_level: str = Field(default="1.5C", description="1.5C or WB2C")
    near_term_target_year: Optional[int] = Field(None, ge=2025, le=2040)
    long_term_target_year: int = Field(default=2050, ge=2040, le=2055)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("ambition_level")
    @classmethod
    def _validate_ambition(cls, v: str) -> str:
        if v not in {"1.5C", "WB2C"}:
            raise ValueError("ambition_level must be '1.5C' or 'WB2C'")
        return v


class SectorAnalysisResult(BaseModel):
    """Output of sector analysis phase."""

    sector: str = Field(default="other")
    sub_sector: str = Field(default="")
    applicable_pathways: List[str] = Field(default_factory=list)
    flag_relevant: bool = Field(default=False)
    flag_share_pct: float = Field(default=0.0)
    sda_available: bool = Field(default=False)
    sda_benchmark: Optional[Dict[str, Any]] = Field(None)
    scope3_significant: bool = Field(default=False, description="Scope 3 > 40% of total")


class PathwayDetail(BaseModel):
    """Details of the selected pathway."""

    pathway: str = Field(default="absolute_contraction")
    ambition_level: str = Field(default="1.5C")
    annual_reduction_rate_pct: float = Field(default=4.2)
    rationale: str = Field(default="")


class TargetDefinition(BaseModel):
    """A single target definition (near-term or long-term)."""

    target_type: str = Field(default="near_term")
    target_year: int = Field(default=2030)
    base_year: int = Field(default=2024)
    scope1_coverage_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    scope2_coverage_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    scope3_coverage_pct: float = Field(default=67.0, ge=0.0, le=100.0)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    is_absolute: bool = Field(default=True, description="True = absolute, False = intensity")
    intensity_metric: Optional[str] = Field(None)
    pathway: str = Field(default="absolute_contraction")
    ambition_level: str = Field(default="1.5C")


class Milestone(BaseModel):
    """Interim milestone on the reduction trajectory."""

    year: int = Field(default=2025)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_from_base_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    label: str = Field(default="")


class ValidationFinding(BaseModel):
    """A single SBTi validation finding."""

    criterion: str = Field(default="", description="SBTi criterion reference")
    description: str = Field(default="")
    severity: ValidationSeverity = Field(default=ValidationSeverity.PASS)
    detail: str = Field(default="")


class ValidationReport(BaseModel):
    """Full SBTi validation report."""

    overall_valid: bool = Field(default=False)
    findings: List[ValidationFinding] = Field(default_factory=list)
    pass_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    fail_count: int = Field(default=0)
    sbti_standard_version: str = Field(default="Net-Zero Standard v1.2")


class TargetSettingResult(BaseModel):
    """Complete result from the target setting workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="target_setting")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    sector_analysis: SectorAnalysisResult = Field(default_factory=SectorAnalysisResult)
    pathway: PathwayDetail = Field(default_factory=PathwayDetail)
    near_term_target: Optional[TargetDefinition] = Field(None)
    long_term_target: Optional[TargetDefinition] = Field(None)
    targets: List[TargetDefinition] = Field(default_factory=list)
    milestones: List[Milestone] = Field(default_factory=list)
    validation_results: ValidationReport = Field(default_factory=ValidationReport)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class TargetSettingWorkflow:
    """
    4-phase target-setting workflow for SBTi-aligned net-zero targets.

    Analyses sector classification, selects the appropriate SBTi pathway,
    defines near-term and long-term targets with milestone trajectory,
    and validates them against SBTi Net-Zero Standard v1.2.

    Zero-hallucination: all reduction rates, coverage requirements, and
    validation criteria come from deterministic SBTi reference data.
    No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Target-setting configuration.

    Example:
        >>> wf = TargetSettingWorkflow()
        >>> config = TargetSettingConfig(sector="manufacturing", ...)
        >>> result = await wf.execute(config)
        >>> assert result.validation_results.overall_valid is True
    """

    def __init__(self) -> None:
        """Initialise TargetSettingWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._sector_analysis: SectorAnalysisResult = SectorAnalysisResult()
        self._pathway: PathwayDetail = PathwayDetail()
        self._near_term: Optional[TargetDefinition] = None
        self._long_term: Optional[TargetDefinition] = None
        self._milestones: List[Milestone] = []
        self._validation: ValidationReport = ValidationReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: TargetSettingConfig) -> TargetSettingResult:
        """
        Execute the 4-phase target-setting workflow.

        Args:
            config: Target-setting configuration with baseline emissions,
                sector classification, and ambition preferences.

        Returns:
            TargetSettingResult with targets, milestones, and validation.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting target setting workflow %s, sector=%s, ambition=%s",
            self.workflow_id, config.sector, config.ambition_level,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_sector_analysis(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_pathway_selection(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_target_definition(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_validation(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Target setting workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        targets = []
        if self._near_term:
            targets.append(self._near_term)
        if self._long_term:
            targets.append(self._long_term)

        result = TargetSettingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            sector_analysis=self._sector_analysis,
            pathway=self._pathway,
            near_term_target=self._near_term,
            long_term_target=self._long_term,
            targets=targets,
            milestones=self._milestones,
            validation_results=self._validation,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Target setting workflow %s completed in %.2fs, valid=%s",
            self.workflow_id, elapsed,
            self._validation.overall_valid,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Sector Analysis
    # -------------------------------------------------------------------------

    async def _phase_sector_analysis(self, config: TargetSettingConfig) -> PhaseResult:
        """Determine sector classification, applicable pathways, FLAG relevance."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector = config.sector.lower().strip()
        if sector not in SECTOR_PATHWAY_MAP:
            warnings.append(f"Sector '{sector}' not in SBTi sector map; defaulting to 'other'")
            sector = "other"

        applicable = SECTOR_PATHWAY_MAP.get(sector, ["absolute_contraction"])
        flag_relevant = sector in FLAG_SECTORS
        sda_available = "sectoral_decarbonisation" in applicable

        # Assess FLAG share
        base = config.baseline_emissions
        flag_share = 0.0
        if base.total_tco2e > 0 and base.flag_emissions_tco2e > 0:
            flag_share = (base.flag_emissions_tco2e / base.total_tco2e) * 100.0

        # Assess Scope 3 significance (> 40% of total triggers Scope 3 target requirement)
        scope3_significant = base.scope3_pct_of_total > 40.0

        # SDA benchmark lookup
        sda_benchmark = None
        if sda_available and sector in SDA_INTENSITY_BENCHMARKS_2030:
            sda_benchmark = SDA_INTENSITY_BENCHMARKS_2030[sector]

        self._sector_analysis = SectorAnalysisResult(
            sector=sector,
            sub_sector=config.sub_sector,
            applicable_pathways=applicable,
            flag_relevant=flag_relevant,
            flag_share_pct=round(flag_share, 2),
            sda_available=sda_available,
            sda_benchmark=sda_benchmark,
            scope3_significant=scope3_significant,
        )

        outputs["sector"] = sector
        outputs["applicable_pathways"] = applicable
        outputs["flag_relevant"] = flag_relevant
        outputs["sda_available"] = sda_available
        outputs["scope3_significant"] = scope3_significant

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Sector analysis: %s, pathways=%s, FLAG=%s", sector, applicable, flag_relevant)
        return PhaseResult(
            phase_name="sector_analysis",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Pathway Selection
    # -------------------------------------------------------------------------

    async def _phase_pathway_selection(self, config: TargetSettingConfig) -> PhaseResult:
        """Select SBTi pathway and determine required reduction rates."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        applicable = self._sector_analysis.applicable_pathways
        ambition = config.ambition_level

        # Determine pathway
        if config.preferred_pathway and config.preferred_pathway in applicable:
            pathway = config.preferred_pathway
        elif self._sector_analysis.sda_available:
            pathway = "sectoral_decarbonisation"
        else:
            pathway = "absolute_contraction"

        # FLAG override: if FLAG-relevant and FLAG share > 20%, FLAG target required
        if self._sector_analysis.flag_relevant and self._sector_analysis.flag_share_pct > 20.0:
            if "forest_land_agriculture" in applicable:
                pathway = "forest_land_agriculture"
                warnings.append(
                    f"FLAG share is {self._sector_analysis.flag_share_pct:.1f}%; "
                    "FLAG pathway selected per SBTi requirements"
                )

        # Look up reduction rate
        rate = SBTI_REDUCTION_RATES.get(ambition, {}).get(pathway, 4.2)

        # Build rationale
        rationale = self._build_pathway_rationale(pathway, ambition, rate)

        self._pathway = PathwayDetail(
            pathway=pathway,
            ambition_level=ambition,
            annual_reduction_rate_pct=rate,
            rationale=rationale,
        )

        outputs["pathway"] = pathway
        outputs["ambition_level"] = ambition
        outputs["annual_reduction_rate_pct"] = rate

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Pathway selected: %s at %s (%.1f%%/yr)", pathway, ambition, rate)
        return PhaseResult(
            phase_name="pathway_selection",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _build_pathway_rationale(self, pathway: str, ambition: str, rate: float) -> str:
        """Build human-readable rationale for pathway selection."""
        parts = []
        if pathway == "absolute_contraction":
            parts.append(
                "Absolute Contraction Approach (ACA) selected: applies a uniform annual "
                f"reduction rate of {rate}% across all sectors."
            )
        elif pathway == "sectoral_decarbonisation":
            parts.append(
                "Sectoral Decarbonisation Approach (SDA) selected: uses sector-specific "
                "intensity benchmarks from IEA scenarios."
            )
        elif pathway == "forest_land_agriculture":
            parts.append(
                "Forest, Land and Agriculture (FLAG) pathway selected: addresses "
                "land-use-related emissions per SBTi FLAG guidance."
            )
        parts.append(f"Ambition level: {ambition} aligned with Paris Agreement.")
        parts.append(f"Required minimum annual reduction: {rate}% per year.")
        return " ".join(parts)

    # -------------------------------------------------------------------------
    # Phase 3: Target Definition
    # -------------------------------------------------------------------------

    async def _phase_target_definition(self, config: TargetSettingConfig) -> PhaseResult:
        """Define near-term and long-term targets with scope coverage."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        base = config.baseline_emissions
        rate = self._pathway.annual_reduction_rate_pct

        # Near-term target
        nt_year = config.near_term_target_year or (config.base_year + 7)
        nt_year = min(nt_year, config.base_year + SBTI_NEAR_TERM_MAX_YEARS)
        years_to_nt = nt_year - config.base_year
        nt_reduction_pct = min(rate * years_to_nt, 95.0)

        # Scope 3 coverage: required if scope3 > 40% of total
        nt_scope3_cov = SBTI_SCOPE_COVERAGE["near_term"]["scope3"] if self._sector_analysis.scope3_significant else 0.0

        # Calculate target emissions
        nt_covered = (base.scope1_tco2e + base.scope2_tco2e) * 0.95
        if nt_scope3_cov > 0:
            nt_covered += base.scope3_tco2e * (nt_scope3_cov / 100.0)
        nt_target_emissions = nt_covered * (1.0 - nt_reduction_pct / 100.0)

        self._near_term = TargetDefinition(
            target_type="near_term",
            target_year=nt_year,
            base_year=config.base_year,
            scope1_coverage_pct=SBTI_SCOPE_COVERAGE["near_term"]["scope1"],
            scope2_coverage_pct=SBTI_SCOPE_COVERAGE["near_term"]["scope2"],
            scope3_coverage_pct=nt_scope3_cov,
            reduction_pct=round(nt_reduction_pct, 2),
            target_emissions_tco2e=round(max(nt_target_emissions, 0.0), 4),
            is_absolute=True,
            pathway=self._pathway.pathway,
            ambition_level=self._pathway.ambition_level,
        )

        # Long-term target
        lt_year = config.long_term_target_year
        lt_reduction_pct = max(SBTI_LONG_TERM_MINIMUM_REDUCTION_PCT, rate * (lt_year - config.base_year))
        lt_reduction_pct = min(lt_reduction_pct, 100.0)

        lt_covered = (base.scope1_tco2e + base.scope2_tco2e) * 0.95
        lt_covered += base.scope3_tco2e * (SBTI_SCOPE_COVERAGE["long_term"]["scope3"] / 100.0)
        lt_target_emissions = lt_covered * (1.0 - lt_reduction_pct / 100.0)

        self._long_term = TargetDefinition(
            target_type="long_term",
            target_year=lt_year,
            base_year=config.base_year,
            scope1_coverage_pct=SBTI_SCOPE_COVERAGE["long_term"]["scope1"],
            scope2_coverage_pct=SBTI_SCOPE_COVERAGE["long_term"]["scope2"],
            scope3_coverage_pct=SBTI_SCOPE_COVERAGE["long_term"]["scope3"],
            reduction_pct=round(lt_reduction_pct, 2),
            target_emissions_tco2e=round(max(lt_target_emissions, 0.0), 4),
            is_absolute=True,
            pathway=self._pathway.pathway,
            ambition_level=self._pathway.ambition_level,
        )

        # Generate milestones (every 5 years from base to 2050)
        self._milestones = self._generate_milestones(config, rate)

        outputs["near_term_year"] = nt_year
        outputs["near_term_reduction_pct"] = self._near_term.reduction_pct
        outputs["near_term_target_tco2e"] = self._near_term.target_emissions_tco2e
        outputs["long_term_year"] = lt_year
        outputs["long_term_reduction_pct"] = self._long_term.reduction_pct
        outputs["long_term_target_tco2e"] = self._long_term.target_emissions_tco2e
        outputs["milestone_count"] = len(self._milestones)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Targets defined: NT %d (-%.1f%%), LT %d (-%.1f%%)",
            nt_year, nt_reduction_pct, lt_year, lt_reduction_pct,
        )
        return PhaseResult(
            phase_name="target_definition",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_milestones(self, config: TargetSettingConfig, annual_rate: float) -> List[Milestone]:
        """Generate interim milestones every 5 years."""
        milestones: List[Milestone] = []
        base = config.baseline_emissions
        total_covered = base.scope1_tco2e + base.scope2_tco2e + base.scope3_tco2e
        base_year = config.base_year

        for year in range(base_year + 5, 2055, 5):
            if year > config.long_term_target_year:
                break
            years_elapsed = year - base_year
            reduction_pct = min(annual_rate * years_elapsed, 100.0)
            target_emissions = total_covered * (1.0 - reduction_pct / 100.0)
            milestones.append(Milestone(
                year=year,
                target_emissions_tco2e=round(max(target_emissions, 0.0), 4),
                reduction_from_base_pct=round(reduction_pct, 2),
                label=f"{year} milestone ({reduction_pct:.1f}% reduction)",
            ))
        return milestones

    # -------------------------------------------------------------------------
    # Phase 4: Validation
    # -------------------------------------------------------------------------

    async def _phase_validation(self, config: TargetSettingConfig) -> PhaseResult:
        """Validate targets against SBTi Net-Zero Standard v1.2."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        findings: List[ValidationFinding] = []

        # Validate near-term target
        if self._near_term:
            findings.extend(self._validate_near_term(config, self._near_term))

        # Validate long-term target
        if self._long_term:
            findings.extend(self._validate_long_term(config, self._long_term))

        # Validate pathway consistency
        findings.extend(self._validate_pathway_consistency(config))

        # Validate timeframe
        findings.extend(self._validate_timeframe(config))

        pass_count = sum(1 for f in findings if f.severity == ValidationSeverity.PASS)
        warn_count = sum(1 for f in findings if f.severity == ValidationSeverity.WARNING)
        fail_count = sum(1 for f in findings if f.severity == ValidationSeverity.FAIL)
        overall_valid = fail_count == 0

        self._validation = ValidationReport(
            overall_valid=overall_valid,
            findings=findings,
            pass_count=pass_count,
            warning_count=warn_count,
            fail_count=fail_count,
        )

        outputs["overall_valid"] = overall_valid
        outputs["pass_count"] = pass_count
        outputs["warning_count"] = warn_count
        outputs["fail_count"] = fail_count

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Validation: valid=%s, pass=%d, warn=%d, fail=%d",
            overall_valid, pass_count, warn_count, fail_count,
        )
        return PhaseResult(
            phase_name="validation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _validate_near_term(self, config: TargetSettingConfig, target: TargetDefinition) -> List[ValidationFinding]:
        """Validate near-term target against SBTi criteria."""
        findings: List[ValidationFinding] = []
        rate = self._pathway.annual_reduction_rate_pct
        years = target.target_year - config.base_year
        min_reduction = rate * years

        # C1: Minimum reduction ambition
        if target.reduction_pct >= min_reduction:
            findings.append(ValidationFinding(
                criterion="NZ-C1",
                description="Near-term reduction meets minimum ambition",
                severity=ValidationSeverity.PASS,
                detail=f"{target.reduction_pct:.1f}% >= {min_reduction:.1f}% required",
            ))
        else:
            findings.append(ValidationFinding(
                criterion="NZ-C1",
                description="Near-term reduction below minimum ambition",
                severity=ValidationSeverity.FAIL,
                detail=f"{target.reduction_pct:.1f}% < {min_reduction:.1f}% required",
            ))

        # C2: Scope 1+2 coverage >= 95%
        if target.scope1_coverage_pct >= 95.0 and target.scope2_coverage_pct >= 95.0:
            findings.append(ValidationFinding(
                criterion="NZ-C2",
                description="Scope 1+2 coverage meets 95% requirement",
                severity=ValidationSeverity.PASS,
            ))
        else:
            findings.append(ValidationFinding(
                criterion="NZ-C2",
                description="Scope 1+2 coverage below 95%",
                severity=ValidationSeverity.FAIL,
                detail=f"S1={target.scope1_coverage_pct}%, S2={target.scope2_coverage_pct}%",
            ))

        # C3: Scope 3 coverage if significant
        if self._sector_analysis.scope3_significant:
            if target.scope3_coverage_pct >= 67.0:
                findings.append(ValidationFinding(
                    criterion="NZ-C3",
                    description="Scope 3 coverage meets 67% near-term requirement",
                    severity=ValidationSeverity.PASS,
                ))
            else:
                findings.append(ValidationFinding(
                    criterion="NZ-C3",
                    description="Scope 3 coverage below 67%",
                    severity=ValidationSeverity.FAIL,
                    detail=f"Coverage: {target.scope3_coverage_pct}%, required: 67%",
                ))

        # C4: Target year within 5-10 years of base year
        if 5 <= years <= 10:
            findings.append(ValidationFinding(
                criterion="NZ-C4",
                description="Near-term target year within 5-10 year window",
                severity=ValidationSeverity.PASS,
            ))
        elif years < 5:
            findings.append(ValidationFinding(
                criterion="NZ-C4",
                description="Near-term target year too close to base year",
                severity=ValidationSeverity.WARNING,
                detail=f"Only {years} years; SBTi recommends 5-10 years",
            ))
        else:
            findings.append(ValidationFinding(
                criterion="NZ-C4",
                description="Near-term target year exceeds 10-year window",
                severity=ValidationSeverity.FAIL,
                detail=f"{years} years from base year (max 10)",
            ))

        return findings

    def _validate_long_term(self, config: TargetSettingConfig, target: TargetDefinition) -> List[ValidationFinding]:
        """Validate long-term target against SBTi criteria."""
        findings: List[ValidationFinding] = []

        # LT1: Minimum 90% reduction
        if target.reduction_pct >= 90.0:
            findings.append(ValidationFinding(
                criterion="NZ-LT1",
                description="Long-term target meets 90% minimum reduction",
                severity=ValidationSeverity.PASS,
                detail=f"Reduction: {target.reduction_pct:.1f}%",
            ))
        else:
            findings.append(ValidationFinding(
                criterion="NZ-LT1",
                description="Long-term target below 90% minimum",
                severity=ValidationSeverity.FAIL,
                detail=f"Reduction: {target.reduction_pct:.1f}%, required: >=90%",
            ))

        # LT2: Scope 3 coverage >= 90%
        if target.scope3_coverage_pct >= 90.0:
            findings.append(ValidationFinding(
                criterion="NZ-LT2",
                description="Long-term Scope 3 coverage meets 90% requirement",
                severity=ValidationSeverity.PASS,
            ))
        else:
            findings.append(ValidationFinding(
                criterion="NZ-LT2",
                description="Long-term Scope 3 coverage below 90%",
                severity=ValidationSeverity.FAIL,
                detail=f"Coverage: {target.scope3_coverage_pct}%, required: 90%",
            ))

        # LT3: Target year no later than 2050
        if target.target_year <= 2050:
            findings.append(ValidationFinding(
                criterion="NZ-LT3",
                description="Long-term target year by 2050",
                severity=ValidationSeverity.PASS,
            ))
        else:
            findings.append(ValidationFinding(
                criterion="NZ-LT3",
                description="Long-term target year after 2050",
                severity=ValidationSeverity.WARNING,
                detail=f"Target year: {target.target_year}",
            ))

        return findings

    def _validate_pathway_consistency(self, config: TargetSettingConfig) -> List[ValidationFinding]:
        """Validate pathway consistency with sector."""
        findings: List[ValidationFinding] = []
        pathway = self._pathway.pathway
        applicable = self._sector_analysis.applicable_pathways

        if pathway in applicable:
            findings.append(ValidationFinding(
                criterion="NZ-P1",
                description="Selected pathway is applicable to sector",
                severity=ValidationSeverity.PASS,
            ))
        else:
            findings.append(ValidationFinding(
                criterion="NZ-P1",
                description="Selected pathway may not suit sector",
                severity=ValidationSeverity.WARNING,
                detail=f"'{pathway}' not in {applicable}",
            ))

        # FLAG check
        if self._sector_analysis.flag_relevant and pathway != "forest_land_agriculture":
            if self._sector_analysis.flag_share_pct > 20.0:
                findings.append(ValidationFinding(
                    criterion="NZ-P2",
                    description="FLAG pathway required but not selected",
                    severity=ValidationSeverity.FAIL,
                    detail=f"FLAG share: {self._sector_analysis.flag_share_pct:.1f}%",
                ))
            else:
                findings.append(ValidationFinding(
                    criterion="NZ-P2",
                    description="FLAG relevance noted but share below 20%",
                    severity=ValidationSeverity.WARNING,
                ))

        return findings

    def _validate_timeframe(self, config: TargetSettingConfig) -> List[ValidationFinding]:
        """Validate target timeframe consistency."""
        findings: List[ValidationFinding] = []

        if self._near_term and self._long_term:
            if self._near_term.target_year < self._long_term.target_year:
                findings.append(ValidationFinding(
                    criterion="NZ-T1",
                    description="Near-term target precedes long-term target",
                    severity=ValidationSeverity.PASS,
                ))
            else:
                findings.append(ValidationFinding(
                    criterion="NZ-T1",
                    description="Near-term target year >= long-term target year",
                    severity=ValidationSeverity.FAIL,
                ))

        if self._near_term and self._long_term:
            if self._near_term.reduction_pct < self._long_term.reduction_pct:
                findings.append(ValidationFinding(
                    criterion="NZ-T2",
                    description="Long-term ambition exceeds near-term (consistent trajectory)",
                    severity=ValidationSeverity.PASS,
                ))
            else:
                findings.append(ValidationFinding(
                    criterion="NZ-T2",
                    description="Long-term reduction not greater than near-term",
                    severity=ValidationSeverity.WARNING,
                ))

        return findings
