# -*- coding: utf-8 -*-
"""
Target Setting Workflow
===========================

5-phase workflow for end-to-end SBTi target setting within PACK-023
SBTi Alignment Pack.  The workflow collects/validates emissions
inventory, screens Scope 3 categories for materiality, selects the
appropriate pathway method (ACA/SDA/FLAG), defines targets with
coverage requirements, and runs criteria validation with ambition
assessment.

Phases:
    1. Inventory       -- Collect and validate emissions inventory (S1+S2+S3)
    2. Screening       -- Screen Scope 3 categories for 40% materiality trigger
    3. PathwaySelect   -- Select pathway method (ACA/SDA/FLAG by sector)
    4. TargetDef       -- Define targets with coverage requirements
    5. Validate        -- Run criteria validation and ambition assessment

Regulatory references:
    - SBTi Corporate Manual V5.3 (2024)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - GHG Protocol Corporate Standard (2015)
    - GHG Protocol Scope 3 Standard (2011)

Zero-hallucination: all pathway calculations, coverage checks, and
ambition assessments use deterministic formulas and lookup tables.
No LLM calls in the numeric computation path.

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"

# =============================================================================
# HELPERS
# =============================================================================

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

class PathwayMethod(str, Enum):
    """SBTi target pathway methods."""

    ACA = "aca"           # Absolute Contraction Approach
    SDA = "sda"           # Sectoral Decarbonization Approach
    FLAG = "flag"         # Forest, Land and Agriculture
    ACA_FLAG = "aca_flag" # ACA + separate FLAG target

class AmbitionLevel(str, Enum):
    """Temperature ambition level of targets."""

    CELSIUS_1_5 = "1.5C"
    WELL_BELOW_2C = "WB2C"
    CELSIUS_2C = "2C"
    INSUFFICIENT = "insufficient"

class TargetType(str, Enum):
    """SBTi target types."""

    NEAR_TERM = "near_term"       # 5-10 years
    LONG_TERM = "long_term"       # >2035
    NET_ZERO = "net_zero"         # 2050 max

class TargetBoundary(str, Enum):
    """Organizational boundary approach."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination Lookups)
# =============================================================================

# ACA minimum annual reduction rates by ambition level
ACA_RATES: Dict[str, float] = {
    "1.5C": 0.042,   # 4.2% per year
    "WB2C": 0.025,   # 2.5% per year
    "2C": 0.015,     # 1.5% per year (below WB2C)
}

# SDA-eligible sectors (from SBTi SDA Tool V3.0)
SDA_ELIGIBLE_SECTORS: List[str] = [
    "power", "cement", "steel", "aluminium", "pulp_paper",
    "chemicals", "aviation", "maritime", "road_transport",
    "buildings_commercial", "buildings_residential", "food_beverage",
]

# FLAG commodity categories
FLAG_COMMODITIES: List[str] = [
    "cattle", "soy", "palm_oil", "timber", "cocoa",
    "coffee", "rubber", "rice", "sugarcane", "maize", "wheat",
]

# Scope 3 category names (GHG Protocol)
SCOPE3_CATEGORIES: Dict[int, str] = {
    1: "Purchased Goods & Services",
    2: "Capital Goods",
    3: "Fuel & Energy Activities",
    4: "Upstream Transportation",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

# Minimum coverage requirements
MIN_SCOPE12_COVERAGE_PCT = 95.0
MIN_SCOPE3_NEAR_TERM_COVERAGE_PCT = 67.0
MIN_SCOPE3_LONG_TERM_COVERAGE_PCT = 90.0
SCOPE3_MATERIALITY_TRIGGER_PCT = 40.0
FLAG_TRIGGER_PCT = 20.0
BASE_YEAR_MIN = 2015
BASE_YEAR_MAX_AGE = 5  # years from submission

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

class EmissionsInventory(BaseModel):
    """Complete emissions inventory for target setting."""

    base_year: int = Field(..., ge=2015, le=2050, description="Base year")
    scope1_tco2e: float = Field(default=0.0, ge=0.0, description="Scope 1 total tCO2e")
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_categories: Dict[int, float] = Field(
        default_factory=dict,
        description="Scope 3 emissions by category (1-15), tCO2e",
    )
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    boundary: TargetBoundary = Field(default=TargetBoundary.OPERATIONAL_CONTROL)
    revenue_usd: float = Field(default=0.0, ge=0.0, description="Revenue for intensity")
    activity_unit: str = Field(default="", description="Activity metric unit (for SDA)")
    activity_value: float = Field(default=0.0, ge=0.0, description="Activity metric value")

    @field_validator("scope3_categories")
    @classmethod
    def _validate_categories(cls, v: Dict[int, float]) -> Dict[int, float]:
        for cat_id in v:
            if cat_id < 1 or cat_id > 15:
                raise ValueError(f"Scope 3 category must be 1-15, got {cat_id}")
        return v

class Scope3CategoryResult(BaseModel):
    """Screening result for a single Scope 3 category."""

    category_id: int = Field(..., ge=1, le=15)
    category_name: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    pct_of_scope3: float = Field(default=0.0, ge=0.0)
    pct_of_total: float = Field(default=0.0, ge=0.0)
    is_material: bool = Field(default=False)
    data_quality: str = Field(default="estimated")
    included_in_target: bool = Field(default=False)

class PathwaySelection(BaseModel):
    """Selected pathway method and parameters."""

    method: PathwayMethod = Field(...)
    sector: str = Field(default="")
    annual_reduction_rate: float = Field(default=0.0)
    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.CELSIUS_1_5)
    requires_flag: bool = Field(default=False)
    sda_eligible: bool = Field(default=False)
    rationale: str = Field(default="")

class TargetDefinition(BaseModel):
    """A single SBTi target definition."""

    target_id: str = Field(default="")
    target_type: TargetType = Field(...)
    scopes_covered: List[str] = Field(default_factory=list)
    base_year: int = Field(..., ge=2015)
    target_year: int = Field(..., ge=2025)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    pathway_method: PathwayMethod = Field(default=PathwayMethod.ACA)
    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.CELSIUS_1_5)
    is_intensity: bool = Field(default=False)
    intensity_metric: str = Field(default="")
    base_year_intensity: float = Field(default=0.0)
    target_intensity: float = Field(default=0.0)

class PathwayMilestone(BaseModel):
    """Annual milestone point on the reduction pathway."""

    year: int = Field(...)
    target_emissions_tco2e: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    annual_reduction_pct: float = Field(default=0.0)

class AmbitionAssessment(BaseModel):
    """Temperature alignment assessment of defined targets."""

    overall_ambition: AmbitionLevel = Field(default=AmbitionLevel.INSUFFICIENT)
    scope12_ambition: AmbitionLevel = Field(default=AmbitionLevel.INSUFFICIENT)
    scope3_ambition: AmbitionLevel = Field(default=AmbitionLevel.INSUFFICIENT)
    scope12_annual_rate: float = Field(default=0.0)
    scope3_annual_rate: float = Field(default=0.0)
    meets_minimum_ambition: bool = Field(default=False)
    assessment_notes: List[str] = Field(default_factory=list)

class TargetSettingConfig(BaseModel):
    """Configuration for the target setting workflow."""

    inventory: EmissionsInventory = Field(...)
    sector: str = Field(default="other")
    preferred_pathway: Optional[PathwayMethod] = Field(None)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    long_term_target_year: int = Field(default=2040, ge=2035, le=2060)
    net_zero_target_year: int = Field(default=2050, ge=2040, le=2060)
    target_ambition: AmbitionLevel = Field(default=AmbitionLevel.CELSIUS_1_5)
    include_net_zero: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class TargetSettingResult(BaseModel):
    """Complete result from the target setting workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="target_setting")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    validated_inventory: Optional[EmissionsInventory] = Field(None)
    scope3_screening: List[Scope3CategoryResult] = Field(default_factory=list)
    pathway_selection: Optional[PathwaySelection] = Field(None)
    targets: List[TargetDefinition] = Field(default_factory=list)
    milestones: List[PathwayMilestone] = Field(default_factory=list)
    ambition_assessment: Optional[AmbitionAssessment] = Field(None)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class TargetSettingWorkflow:
    """
    5-phase target setting workflow for SBTi alignment.

    Collects and validates emissions inventory, screens Scope 3
    categories for materiality, selects the appropriate pathway
    method (ACA/SDA/FLAG), defines near-term/long-term/net-zero
    targets with coverage requirements, and runs criteria
    validation with ambition assessment.

    Zero-hallucination: all reduction rates, coverage thresholds,
    and ambition assessments use SBTi-specified values from lookup
    tables.  No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = TargetSettingWorkflow()
        >>> config = TargetSettingConfig(
        ...     inventory=EmissionsInventory(base_year=2022, scope1_tco2e=5000),
        ...     sector="manufacturing",
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self) -> None:
        """Initialise TargetSettingWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._inventory: Optional[EmissionsInventory] = None
        self._scope3_screening: List[Scope3CategoryResult] = []
        self._pathway: Optional[PathwaySelection] = None
        self._targets: List[TargetDefinition] = []
        self._milestones: List[PathwayMilestone] = []
        self._ambition: Optional[AmbitionAssessment] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: TargetSettingConfig) -> TargetSettingResult:
        """
        Execute the 5-phase target setting workflow.

        Args:
            config: Target setting configuration with emissions
                inventory, sector, and target preferences.

        Returns:
            TargetSettingResult with defined targets and ambition assessment.
        """
        started_at = utcnow()
        self.logger.info(
            "Starting target setting workflow %s, sector=%s, base_year=%d",
            self.workflow_id, config.sector, config.inventory.base_year,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Inventory validation
            phase1 = await self._phase_inventory(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Inventory phase failed; cannot proceed")

            # Phase 2: Scope 3 Screening
            phase2 = await self._phase_screening(config)
            self._phase_results.append(phase2)

            # Phase 3: Pathway Selection
            phase3 = await self._phase_pathway_select(config)
            self._phase_results.append(phase3)

            # Phase 4: Target Definition
            phase4 = await self._phase_target_def(config)
            self._phase_results.append(phase4)

            # Phase 5: Validation
            phase5 = await self._phase_validate(config)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Target setting workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = TargetSettingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            validated_inventory=self._inventory,
            scope3_screening=self._scope3_screening,
            pathway_selection=self._pathway,
            targets=self._targets,
            milestones=self._milestones,
            ambition_assessment=self._ambition,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Target setting workflow %s completed in %.2fs, targets=%d",
            self.workflow_id, elapsed, len(self._targets),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Inventory
    # -------------------------------------------------------------------------

    async def _phase_inventory(self, config: TargetSettingConfig) -> PhaseResult:
        """Collect and validate emissions inventory (Scope 1+2+3)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        inv = config.inventory
        self._inventory = inv

        # Validate base year
        current_year = utcnow().year
        if inv.base_year < BASE_YEAR_MIN:
            errors.append(f"Base year {inv.base_year} is before minimum {BASE_YEAR_MIN}")
        if current_year - inv.base_year > BASE_YEAR_MAX_AGE:
            warnings.append(
                f"Base year {inv.base_year} is >{BASE_YEAR_MAX_AGE} years old; "
                "recalculation may be needed for new submissions"
            )

        # Validate Scope 1+2 presence
        scope12_total = inv.scope1_tco2e + inv.scope2_location_tco2e
        if scope12_total <= 0:
            errors.append("Scope 1+2 emissions must be > 0")

        # Calculate totals
        scope3_total = sum(inv.scope3_categories.values())
        total_emissions = scope12_total + scope3_total

        # Validate Scope 2 market-based availability
        if inv.scope2_market_tco2e <= 0 and inv.scope2_location_tco2e > 0:
            warnings.append(
                "Market-based Scope 2 not provided; location-based will be used "
                "but market-based is preferred per SBTi guidance"
            )

        # Calculate FLAG fraction
        flag_fraction = 0.0
        if total_emissions > 0:
            flag_fraction = (inv.flag_emissions_tco2e / total_emissions) * 100.0

        outputs["base_year"] = inv.base_year
        outputs["scope1_tco2e"] = round(inv.scope1_tco2e, 2)
        outputs["scope2_location_tco2e"] = round(inv.scope2_location_tco2e, 2)
        outputs["scope2_market_tco2e"] = round(inv.scope2_market_tco2e, 2)
        outputs["scope3_total_tco2e"] = round(scope3_total, 2)
        outputs["total_emissions_tco2e"] = round(total_emissions, 2)
        outputs["scope3_categories_reported"] = len(inv.scope3_categories)
        outputs["flag_emissions_tco2e"] = round(inv.flag_emissions_tco2e, 2)
        outputs["flag_fraction_pct"] = round(flag_fraction, 2)
        outputs["boundary"] = inv.boundary.value

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Inventory: total=%.2f tCO2e, S3 categories=%d, FLAG=%.1f%%",
            total_emissions, len(inv.scope3_categories), flag_fraction,
        )
        return PhaseResult(
            phase_name="inventory",
            status=PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Scope 3 Screening
    # -------------------------------------------------------------------------

    async def _phase_screening(self, config: TargetSettingConfig) -> PhaseResult:
        """Screen Scope 3 categories for 40% materiality trigger."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        inv = config.inventory
        scope12_total = inv.scope1_tco2e + inv.scope2_location_tco2e
        scope3_total = sum(inv.scope3_categories.values())
        total_emissions = scope12_total + scope3_total

        self._scope3_screening = []
        scope3_pct_of_total = 0.0
        if total_emissions > 0:
            scope3_pct_of_total = (scope3_total / total_emissions) * 100.0

        # Screen each category
        for cat_id in range(1, 16):
            cat_emissions = inv.scope3_categories.get(cat_id, 0.0)
            pct_of_scope3 = (cat_emissions / scope3_total * 100.0) if scope3_total > 0 else 0.0
            pct_of_total = (cat_emissions / total_emissions * 100.0) if total_emissions > 0 else 0.0

            # Material if >1% of total or >5% of Scope 3
            is_material = pct_of_total > 1.0 or pct_of_scope3 > 5.0

            self._scope3_screening.append(Scope3CategoryResult(
                category_id=cat_id,
                category_name=SCOPE3_CATEGORIES.get(cat_id, f"Category {cat_id}"),
                emissions_tco2e=round(cat_emissions, 2),
                pct_of_scope3=round(pct_of_scope3, 2),
                pct_of_total=round(pct_of_total, 2),
                is_material=is_material,
                data_quality="reported" if cat_emissions > 0 else "not_reported",
                included_in_target=is_material,
            ))

        # Determine if Scope 3 targets required (40% trigger)
        scope3_target_required = scope3_pct_of_total >= SCOPE3_MATERIALITY_TRIGGER_PCT

        # Calculate coverage of material categories
        material_emissions = sum(
            c.emissions_tco2e for c in self._scope3_screening if c.is_material
        )
        coverage_pct = (material_emissions / scope3_total * 100.0) if scope3_total > 0 else 0.0

        # Sort by Pareto (descending emissions)
        self._scope3_screening.sort(key=lambda c: c.emissions_tco2e, reverse=True)

        # Identify top categories for coverage
        cumulative_pct = 0.0
        for cat in self._scope3_screening:
            cumulative_pct += cat.pct_of_scope3
            if not cat.is_material and cumulative_pct <= 80.0 and cat.emissions_tco2e > 0:
                cat.included_in_target = True
                warnings.append(
                    f"Category {cat.category_id} ({cat.category_name}) added to "
                    "target for Pareto coverage"
                )

        material_count = sum(1 for c in self._scope3_screening if c.is_material)
        included_count = sum(1 for c in self._scope3_screening if c.included_in_target)

        outputs["scope3_pct_of_total"] = round(scope3_pct_of_total, 2)
        outputs["scope3_target_required"] = scope3_target_required
        outputs["material_categories"] = material_count
        outputs["included_categories"] = included_count
        outputs["coverage_pct"] = round(coverage_pct, 2)
        outputs["scope3_total_tco2e"] = round(scope3_total, 2)

        if not scope3_target_required:
            warnings.append(
                f"Scope 3 is {scope3_pct_of_total:.1f}% of total (<{SCOPE3_MATERIALITY_TRIGGER_PCT}%); "
                "Scope 3 target not required but recommended"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Screening: S3=%.1f%% of total, required=%s, material=%d categories",
            scope3_pct_of_total, scope3_target_required, material_count,
        )
        return PhaseResult(
            phase_name="screening",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Pathway Selection
    # -------------------------------------------------------------------------

    async def _phase_pathway_select(self, config: TargetSettingConfig) -> PhaseResult:
        """Select pathway method (ACA/SDA/FLAG based on sector)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        inv = config.inventory
        sector = config.sector.lower().strip()
        scope12_total = inv.scope1_tco2e + inv.scope2_location_tco2e
        scope3_total = sum(inv.scope3_categories.values())
        total_emissions = scope12_total + scope3_total

        sda_eligible = sector in SDA_ELIGIBLE_SECTORS
        flag_fraction = (inv.flag_emissions_tco2e / total_emissions * 100.0) if total_emissions > 0 else 0.0
        requires_flag = flag_fraction >= FLAG_TRIGGER_PCT

        # Determine pathway
        if config.preferred_pathway is not None:
            method = config.preferred_pathway
            rationale = f"User-preferred pathway: {method.value}"
            if method == PathwayMethod.SDA and not sda_eligible:
                warnings.append(
                    f"SDA preferred but sector '{sector}' is not SDA-eligible; "
                    "falling back to ACA"
                )
                method = PathwayMethod.ACA
                rationale = "Fallback to ACA: sector not SDA-eligible"
        elif requires_flag and sda_eligible:
            method = PathwayMethod.ACA_FLAG
            rationale = (
                f"FLAG emissions >{FLAG_TRIGGER_PCT}% and SDA-eligible; "
                "combined ACA+FLAG pathway recommended"
            )
        elif requires_flag:
            method = PathwayMethod.ACA_FLAG
            rationale = f"FLAG emissions >{FLAG_TRIGGER_PCT}%; ACA+FLAG required"
        elif sda_eligible:
            method = PathwayMethod.SDA
            rationale = f"Sector '{sector}' is SDA-eligible; SDA pathway recommended"
        else:
            method = PathwayMethod.ACA
            rationale = "Default ACA pathway for non-SDA sectors"

        # Determine annual reduction rate
        ambition = config.target_ambition
        if method in (PathwayMethod.ACA, PathwayMethod.ACA_FLAG):
            annual_rate = ACA_RATES.get(ambition.value, ACA_RATES["1.5C"])
        elif method == PathwayMethod.SDA:
            # SDA rate is sector-specific; use ACA as reference
            annual_rate = ACA_RATES.get(ambition.value, ACA_RATES["1.5C"])
        elif method == PathwayMethod.FLAG:
            annual_rate = 0.0303  # 3.03% per year for FLAG
        else:
            annual_rate = ACA_RATES["1.5C"]

        self._pathway = PathwaySelection(
            method=method,
            sector=sector,
            annual_reduction_rate=round(annual_rate, 4),
            ambition_level=ambition,
            requires_flag=requires_flag,
            sda_eligible=sda_eligible,
            rationale=rationale,
        )

        outputs["method"] = method.value
        outputs["sector"] = sector
        outputs["sda_eligible"] = sda_eligible
        outputs["requires_flag"] = requires_flag
        outputs["flag_fraction_pct"] = round(flag_fraction, 2)
        outputs["annual_reduction_rate"] = round(annual_rate, 4)
        outputs["ambition_level"] = ambition.value
        outputs["rationale"] = rationale

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Pathway: method=%s, rate=%.2f%%, FLAG=%s, SDA=%s",
            method.value, annual_rate * 100, requires_flag, sda_eligible,
        )
        return PhaseResult(
            phase_name="pathway_select",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Target Definition
    # -------------------------------------------------------------------------

    async def _phase_target_def(self, config: TargetSettingConfig) -> PhaseResult:
        """Define targets with coverage requirements."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._targets = []
        self._milestones = []

        inv = config.inventory
        scope12_total = inv.scope1_tco2e + inv.scope2_location_tco2e
        scope3_total = sum(inv.scope3_categories.values())

        if self._pathway is None:
            warnings.append("No pathway selected; using ACA defaults")
            annual_rate = ACA_RATES["1.5C"]
            method = PathwayMethod.ACA
        else:
            annual_rate = self._pathway.annual_reduction_rate
            method = self._pathway.method

        # ----- Near-term target (Scope 1+2) -----
        nt_years = config.near_term_target_year - inv.base_year
        nt_reduction_pct = min(1.0 - (1.0 - annual_rate) ** nt_years, 0.95) * 100.0
        nt_target_emissions = scope12_total * (1.0 - nt_reduction_pct / 100.0)

        nt_target = TargetDefinition(
            target_id=f"NT-S12-{_new_uuid()[:8]}",
            target_type=TargetType.NEAR_TERM,
            scopes_covered=["scope1", "scope2"],
            base_year=inv.base_year,
            target_year=config.near_term_target_year,
            base_year_emissions_tco2e=round(scope12_total, 2),
            target_reduction_pct=round(nt_reduction_pct, 2),
            target_emissions_tco2e=round(nt_target_emissions, 2),
            coverage_pct=MIN_SCOPE12_COVERAGE_PCT,
            pathway_method=method if method != PathwayMethod.FLAG else PathwayMethod.ACA,
            ambition_level=config.target_ambition,
        )
        self._targets.append(nt_target)

        # ----- Near-term target (Scope 3, if required) -----
        scope3_pct = (scope3_total / (scope12_total + scope3_total) * 100.0) if (scope12_total + scope3_total) > 0 else 0.0
        if scope3_pct >= SCOPE3_MATERIALITY_TRIGGER_PCT and scope3_total > 0:
            s3_annual_rate = annual_rate * 0.7  # Scope 3 rate typically lower
            s3_reduction_pct = min(1.0 - (1.0 - s3_annual_rate) ** nt_years, 0.90) * 100.0
            s3_target_emissions = scope3_total * (1.0 - s3_reduction_pct / 100.0)

            s3_target = TargetDefinition(
                target_id=f"NT-S3-{_new_uuid()[:8]}",
                target_type=TargetType.NEAR_TERM,
                scopes_covered=["scope3"],
                base_year=inv.base_year,
                target_year=config.near_term_target_year,
                base_year_emissions_tco2e=round(scope3_total, 2),
                target_reduction_pct=round(s3_reduction_pct, 2),
                target_emissions_tco2e=round(s3_target_emissions, 2),
                coverage_pct=MIN_SCOPE3_NEAR_TERM_COVERAGE_PCT,
                pathway_method=PathwayMethod.ACA,
                ambition_level=config.target_ambition,
            )
            self._targets.append(s3_target)

        # ----- Long-term target (if include_net_zero) -----
        if config.include_net_zero:
            lt_years = config.long_term_target_year - inv.base_year
            lt_reduction_pct = min(1.0 - (1.0 - annual_rate) ** lt_years, 0.95) * 100.0
            total_base = scope12_total + scope3_total
            lt_target_emissions = total_base * (1.0 - lt_reduction_pct / 100.0)

            lt_target = TargetDefinition(
                target_id=f"LT-ALL-{_new_uuid()[:8]}",
                target_type=TargetType.LONG_TERM,
                scopes_covered=["scope1", "scope2", "scope3"],
                base_year=inv.base_year,
                target_year=config.long_term_target_year,
                base_year_emissions_tco2e=round(total_base, 2),
                target_reduction_pct=round(lt_reduction_pct, 2),
                target_emissions_tco2e=round(lt_target_emissions, 2),
                coverage_pct=MIN_SCOPE3_LONG_TERM_COVERAGE_PCT,
                pathway_method=method if method != PathwayMethod.FLAG else PathwayMethod.ACA,
                ambition_level=config.target_ambition,
            )
            self._targets.append(lt_target)

            # ----- Net-zero target -----
            nz_years = config.net_zero_target_year - inv.base_year
            nz_reduction_pct = 90.0  # SBTi minimum for net-zero
            nz_target_emissions = total_base * (1.0 - nz_reduction_pct / 100.0)

            nz_target = TargetDefinition(
                target_id=f"NZ-ALL-{_new_uuid()[:8]}",
                target_type=TargetType.NET_ZERO,
                scopes_covered=["scope1", "scope2", "scope3"],
                base_year=inv.base_year,
                target_year=config.net_zero_target_year,
                base_year_emissions_tco2e=round(total_base, 2),
                target_reduction_pct=nz_reduction_pct,
                target_emissions_tco2e=round(nz_target_emissions, 2),
                coverage_pct=MIN_SCOPE3_LONG_TERM_COVERAGE_PCT,
                pathway_method=method if method != PathwayMethod.FLAG else PathwayMethod.ACA,
                ambition_level=AmbitionLevel.CELSIUS_1_5,
            )
            self._targets.append(nz_target)

        # ----- FLAG target (if required) -----
        if self._pathway and self._pathway.requires_flag and inv.flag_emissions_tco2e > 0:
            flag_years = config.near_term_target_year - inv.base_year
            flag_rate = 0.0303
            flag_reduction_pct = min(1.0 - (1.0 - flag_rate) ** flag_years, 0.50) * 100.0
            flag_target_emissions = inv.flag_emissions_tco2e * (1.0 - flag_reduction_pct / 100.0)

            flag_target = TargetDefinition(
                target_id=f"FLAG-{_new_uuid()[:8]}",
                target_type=TargetType.NEAR_TERM,
                scopes_covered=["flag"],
                base_year=inv.base_year,
                target_year=config.near_term_target_year,
                base_year_emissions_tco2e=round(inv.flag_emissions_tco2e, 2),
                target_reduction_pct=round(flag_reduction_pct, 2),
                target_emissions_tco2e=round(flag_target_emissions, 2),
                coverage_pct=100.0,
                pathway_method=PathwayMethod.FLAG,
                ambition_level=AmbitionLevel.CELSIUS_1_5,
            )
            self._targets.append(flag_target)

        # ----- Generate milestones for S1+2 near-term -----
        self._milestones = self._generate_milestones(
            inv.base_year, config.near_term_target_year,
            scope12_total, annual_rate,
        )

        outputs["targets_defined"] = len(self._targets)
        outputs["milestones_generated"] = len(self._milestones)
        for t in self._targets:
            outputs[f"{t.target_id}_type"] = t.target_type.value
            outputs[f"{t.target_id}_reduction_pct"] = round(t.target_reduction_pct, 2)
            outputs[f"{t.target_id}_target_year"] = t.target_year

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Target definition: %d targets, %d milestones defined",
            len(self._targets), len(self._milestones),
        )
        return PhaseResult(
            phase_name="target_def",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_milestones(
        self, base_year: int, target_year: int,
        base_emissions: float, annual_rate: float,
    ) -> List[PathwayMilestone]:
        """Generate annual milestone points on the reduction pathway."""
        milestones: List[PathwayMilestone] = []
        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            remaining_factor = (1.0 - annual_rate) ** years_elapsed
            target_emissions = base_emissions * remaining_factor
            cumulative_reduction = (1.0 - remaining_factor) * 100.0

            milestones.append(PathwayMilestone(
                year=year,
                target_emissions_tco2e=round(target_emissions, 2),
                cumulative_reduction_pct=round(cumulative_reduction, 2),
                annual_reduction_pct=round(annual_rate * 100.0, 2),
            ))
        return milestones

    # -------------------------------------------------------------------------
    # Phase 5: Validation
    # -------------------------------------------------------------------------

    async def _phase_validate(self, config: TargetSettingConfig) -> PhaseResult:
        """Run criteria validation and ambition assessment."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        assessment_notes: List[str] = []

        # Assess Scope 1+2 ambition
        s12_target = next(
            (t for t in self._targets if t.target_type == TargetType.NEAR_TERM and "scope1" in t.scopes_covered),
            None,
        )
        s12_ambition = AmbitionLevel.INSUFFICIENT
        s12_annual_rate = 0.0
        if s12_target:
            years = s12_target.target_year - s12_target.base_year
            if years > 0:
                s12_annual_rate = 1.0 - (1.0 - s12_target.target_reduction_pct / 100.0) ** (1.0 / years)
            s12_ambition = self._classify_ambition(s12_annual_rate)
            assessment_notes.append(
                f"S1+2 annual rate: {s12_annual_rate * 100:.2f}% -> {s12_ambition.value}"
            )

            # Coverage check
            if s12_target.coverage_pct < MIN_SCOPE12_COVERAGE_PCT:
                warnings.append(
                    f"S1+2 coverage {s12_target.coverage_pct}% < "
                    f"minimum {MIN_SCOPE12_COVERAGE_PCT}%"
                )

        # Assess Scope 3 ambition
        s3_target = next(
            (t for t in self._targets if t.target_type == TargetType.NEAR_TERM and t.scopes_covered == ["scope3"]),
            None,
        )
        s3_ambition = AmbitionLevel.INSUFFICIENT
        s3_annual_rate = 0.0
        if s3_target:
            years = s3_target.target_year - s3_target.base_year
            if years > 0:
                s3_annual_rate = 1.0 - (1.0 - s3_target.target_reduction_pct / 100.0) ** (1.0 / years)
            s3_ambition = self._classify_ambition(s3_annual_rate)
            assessment_notes.append(
                f"S3 annual rate: {s3_annual_rate * 100:.2f}% -> {s3_ambition.value}"
            )

        # Overall ambition (most conservative)
        overall = self._overall_ambition(s12_ambition, s3_ambition)
        meets_minimum = overall in (
            AmbitionLevel.CELSIUS_1_5,
            AmbitionLevel.WELL_BELOW_2C,
        )

        # Timeframe checks
        for t in self._targets:
            if t.target_type == TargetType.NEAR_TERM:
                years_from_base = t.target_year - t.base_year
                if years_from_base < 5 or years_from_base > 10:
                    warnings.append(
                        f"Target {t.target_id}: near-term timeframe "
                        f"{years_from_base} years (should be 5-10)"
                    )
            elif t.target_type == TargetType.NET_ZERO:
                if t.target_year > 2050:
                    warnings.append(
                        f"Target {t.target_id}: net-zero year {t.target_year} > 2050"
                    )

        # Net-zero residual check
        nz_target = next(
            (t for t in self._targets if t.target_type == TargetType.NET_ZERO), None
        )
        if nz_target and nz_target.target_reduction_pct < 90.0:
            warnings.append(
                f"Net-zero target reduction {nz_target.target_reduction_pct}% "
                "is below 90% SBTi minimum"
            )
            assessment_notes.append("Net-zero reduction below 90% threshold")

        self._ambition = AmbitionAssessment(
            overall_ambition=overall,
            scope12_ambition=s12_ambition,
            scope3_ambition=s3_ambition,
            scope12_annual_rate=round(s12_annual_rate, 4),
            scope3_annual_rate=round(s3_annual_rate, 4),
            meets_minimum_ambition=meets_minimum,
            assessment_notes=assessment_notes,
        )

        outputs["overall_ambition"] = overall.value
        outputs["scope12_ambition"] = s12_ambition.value
        outputs["scope3_ambition"] = s3_ambition.value
        outputs["scope12_annual_rate_pct"] = round(s12_annual_rate * 100, 2)
        outputs["scope3_annual_rate_pct"] = round(s3_annual_rate * 100, 2)
        outputs["meets_minimum_ambition"] = meets_minimum
        outputs["targets_validated"] = len(self._targets)
        outputs["warnings_count"] = len(warnings)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Validation: ambition=%s, meets_minimum=%s, warnings=%d",
            overall.value, meets_minimum, len(warnings),
        )
        return PhaseResult(
            phase_name="validate",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _classify_ambition(self, annual_rate: float) -> AmbitionLevel:
        """Classify annual reduction rate into ambition level."""
        if annual_rate >= 0.042:
            return AmbitionLevel.CELSIUS_1_5
        elif annual_rate >= 0.025:
            return AmbitionLevel.WELL_BELOW_2C
        elif annual_rate >= 0.015:
            return AmbitionLevel.CELSIUS_2C
        else:
            return AmbitionLevel.INSUFFICIENT

    def _overall_ambition(
        self, s12: AmbitionLevel, s3: AmbitionLevel,
    ) -> AmbitionLevel:
        """Determine overall ambition from scope-level ambitions."""
        ranking = {
            AmbitionLevel.CELSIUS_1_5: 4,
            AmbitionLevel.WELL_BELOW_2C: 3,
            AmbitionLevel.CELSIUS_2C: 2,
            AmbitionLevel.INSUFFICIENT: 1,
        }
        # Overall is the less ambitious of the two
        s12_rank = ranking.get(s12, 1)
        s3_rank = ranking.get(s3, 1)
        min_rank = min(s12_rank, s3_rank)

        # If no S3 target at all, overall is S1+2 ambition
        if s3 == AmbitionLevel.INSUFFICIENT and s12 != AmbitionLevel.INSUFFICIENT:
            return s12

        reverse = {v: k for k, v in ranking.items()}
        return reverse.get(min_rank, AmbitionLevel.INSUFFICIENT)
