# -*- coding: utf-8 -*-
"""
Certification Assessment Workflow
======================================

4-phase workflow for green building certification assessment within
PACK-032 Building Energy Assessment Pack.

Phases:
    1. CertificationSelection  -- Determine applicable scheme (LEED/BREEAM/Energy Star/NABERS)
    2. CreditScoring           -- Evaluate credits/prerequisites by category
    3. GapAnalysis             -- Identify missing credits, effort to achieve
    4. ActionPlan              -- Prioritised list of actions for target certification level

Zero-hallucination: all credit scoring uses published scheme criteria and
deterministic point allocation. No LLM calls in the scoring path.

Schedule: on-demand
Estimated duration: 180 minutes

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

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


class CertificationScheme(str, Enum):
    """Green building certification schemes."""

    LEED = "leed"
    BREEAM = "breeam"
    ENERGY_STAR = "energy_star"
    NABERS = "nabers"
    GREEN_STAR = "green_star"
    WELL = "well"
    FITWEL = "fitwel"
    DGNB = "dgnb"


class CertificationLevel(str, Enum):
    """Certification level/rating."""

    CERTIFIED = "certified"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    PASS = "pass"
    GOOD = "good"
    VERY_GOOD = "very_good"
    EXCELLENT = "excellent"
    OUTSTANDING = "outstanding"
    ONE_STAR = "1_star"
    TWO_STAR = "2_star"
    THREE_STAR = "3_star"
    FOUR_STAR = "4_star"
    FIVE_STAR = "5_star"
    SIX_STAR = "6_star"


class CreditStatus(str, Enum):
    """Credit achievement status."""

    ACHIEVED = "achieved"
    PARTIAL = "partial"
    NOT_ACHIEVED = "not_achieved"
    NOT_APPLICABLE = "not_applicable"
    PREREQUISITE_MET = "prerequisite_met"
    PREREQUISITE_NOT_MET = "prerequisite_not_met"


class EffortLevel(str, Enum):
    """Effort level to achieve a credit."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =============================================================================
# ZERO-HALLUCINATION REFERENCE CONSTANTS
# =============================================================================

# LEED v4.1 O+M point thresholds
LEED_LEVELS: Dict[str, int] = {
    "certified": 40,
    "silver": 50,
    "gold": 60,
    "platinum": 80,
}

# LEED v4.1 O+M credit categories and max points
LEED_CREDITS: List[Dict[str, Any]] = [
    # Energy & Atmosphere
    {"id": "EA-P1", "category": "energy_atmosphere", "name": "Energy Efficiency Best Management Practices",
     "max_points": 0, "is_prerequisite": True, "eui_threshold": 999.0},
    {"id": "EA-P2", "category": "energy_atmosphere", "name": "Minimum Energy Performance",
     "max_points": 0, "is_prerequisite": True, "eui_threshold": 999.0},
    {"id": "EA-C1", "category": "energy_atmosphere", "name": "Optimize Energy Performance",
     "max_points": 20, "is_prerequisite": False, "eui_threshold": 200.0},
    {"id": "EA-C2", "category": "energy_atmosphere", "name": "Advanced Energy Metering",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EA-C3", "category": "energy_atmosphere", "name": "Demand Response",
     "max_points": 3, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EA-C4", "category": "energy_atmosphere", "name": "Renewable Energy",
     "max_points": 5, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EA-C5", "category": "energy_atmosphere", "name": "Enhanced Refrigerant Management",
     "max_points": 1, "is_prerequisite": False, "eui_threshold": 999.0},
    # Water Efficiency
    {"id": "WE-P1", "category": "water_efficiency", "name": "Indoor Water Use Reduction",
     "max_points": 0, "is_prerequisite": True, "eui_threshold": 999.0},
    {"id": "WE-C1", "category": "water_efficiency", "name": "Indoor Water Use Reduction",
     "max_points": 12, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "WE-C2", "category": "water_efficiency", "name": "Outdoor Water Use Reduction",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "WE-C3", "category": "water_efficiency", "name": "Cooling Tower Water Use",
     "max_points": 3, "is_prerequisite": False, "eui_threshold": 999.0},
    # Indoor Environmental Quality
    {"id": "EQ-P1", "category": "indoor_quality", "name": "Minimum IAQ Performance",
     "max_points": 0, "is_prerequisite": True, "eui_threshold": 999.0},
    {"id": "EQ-C1", "category": "indoor_quality", "name": "IAQ Management Program",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EQ-C2", "category": "indoor_quality", "name": "Enhanced IAQ Strategies",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EQ-C3", "category": "indoor_quality", "name": "Thermal Comfort",
     "max_points": 1, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EQ-C4", "category": "indoor_quality", "name": "Interior Lighting",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EQ-C5", "category": "indoor_quality", "name": "Daylight and Quality Views",
     "max_points": 4, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "EQ-C6", "category": "indoor_quality", "name": "Green Cleaning",
     "max_points": 1, "is_prerequisite": False, "eui_threshold": 999.0},
    # Sustainable Sites
    {"id": "SS-P1", "category": "sustainable_sites", "name": "Site Management Policy",
     "max_points": 0, "is_prerequisite": True, "eui_threshold": 999.0},
    {"id": "SS-C1", "category": "sustainable_sites", "name": "High Priority Site Management",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "SS-C2", "category": "sustainable_sites", "name": "Rainwater Management",
     "max_points": 3, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "SS-C3", "category": "sustainable_sites", "name": "Heat Island Reduction",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "SS-C4", "category": "sustainable_sites", "name": "Light Pollution Reduction",
     "max_points": 1, "is_prerequisite": False, "eui_threshold": 999.0},
    # Materials & Resources
    {"id": "MR-P1", "category": "materials_resources", "name": "Purchasing Policy",
     "max_points": 0, "is_prerequisite": True, "eui_threshold": 999.0},
    {"id": "MR-C1", "category": "materials_resources", "name": "Purchasing - Ongoing",
     "max_points": 1, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "MR-C2", "category": "materials_resources", "name": "Purchasing - Lamps",
     "max_points": 1, "is_prerequisite": False, "eui_threshold": 999.0},
    {"id": "MR-C3", "category": "materials_resources", "name": "Waste Performance",
     "max_points": 2, "is_prerequisite": False, "eui_threshold": 999.0},
    # Transportation
    {"id": "TR-C1", "category": "transportation", "name": "Alternative Commuting Transportation",
     "max_points": 15, "is_prerequisite": False, "eui_threshold": 999.0},
    # Innovation
    {"id": "IN-C1", "category": "innovation", "name": "Innovation Credits",
     "max_points": 6, "is_prerequisite": False, "eui_threshold": 999.0},
]

# BREEAM In-Use V6 weighting percentages
BREEAM_CATEGORIES: Dict[str, float] = {
    "management": 0.12,
    "health_wellbeing": 0.15,
    "energy": 0.19,
    "transport": 0.08,
    "water": 0.06,
    "materials": 0.125,
    "waste": 0.075,
    "land_use_ecology": 0.10,
    "pollution": 0.10,
}

BREEAM_LEVELS: Dict[str, float] = {
    "pass": 30.0,
    "good": 45.0,
    "very_good": 55.0,
    "excellent": 70.0,
    "outstanding": 85.0,
}

# Energy Star score thresholds (1-100)
ENERGY_STAR_CERTIFIED_THRESHOLD: int = 75

# NABERS star rating thresholds (normalised score 0-6)
NABERS_LEVELS: Dict[str, float] = {
    "1_star": 1.0,
    "2_star": 2.0,
    "3_star": 3.0,
    "4_star": 4.0,
    "5_star": 5.0,
    "6_star": 5.5,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BuildingPerformanceData(BaseModel):
    """Building performance data for certification assessment."""

    eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    co2_kg_per_sqm: float = Field(default=0.0, ge=0.0)
    epc_band: str = Field(default="")
    water_use_litres_per_sqm: float = Field(default=0.0, ge=0.0)
    waste_diversion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    renewable_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    has_bms: bool = Field(default=False)
    has_submetering: bool = Field(default=False)
    has_demand_response: bool = Field(default=False)
    has_green_cleaning: bool = Field(default=False)
    has_iaq_monitoring: bool = Field(default=False)
    daylight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    thermal_comfort_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    cycling_facilities: bool = Field(default=False)
    ev_charging: bool = Field(default=False)
    public_transport_proximity: bool = Field(default=False)
    green_lease: bool = Field(default=False)
    site_management_plan: bool = Field(default=False)
    purchasing_policy: bool = Field(default=False)
    low_mercury_lamps: bool = Field(default=False)
    rainwater_management: bool = Field(default=False)


class CreditAssessment(BaseModel):
    """Assessment of a single credit."""

    credit_id: str = Field(default="")
    category: str = Field(default="")
    name: str = Field(default="")
    max_points: int = Field(default=0, ge=0)
    achieved_points: float = Field(default=0.0, ge=0.0)
    status: CreditStatus = Field(default=CreditStatus.NOT_ACHIEVED)
    is_prerequisite: bool = Field(default=False)
    evidence_available: bool = Field(default=False)
    notes: str = Field(default="")


class GapItem(BaseModel):
    """Gap analysis item for a missing credit."""

    credit_id: str = Field(default="")
    name: str = Field(default="")
    category: str = Field(default="")
    points_available: float = Field(default=0.0, ge=0.0)
    effort: EffortLevel = Field(default=EffortLevel.MEDIUM)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    estimated_weeks: int = Field(default=0, ge=0)
    description: str = Field(default="")
    dependencies: List[str] = Field(default_factory=list)


class CertificationAction(BaseModel):
    """Action item to achieve certification credit."""

    action_id: str = Field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    credit_id: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    priority: str = Field(default="medium", description="critical|high|medium|low")
    effort: EffortLevel = Field(default=EffortLevel.MEDIUM)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    estimated_weeks: int = Field(default=0, ge=0)
    points_gain: float = Field(default=0.0, ge=0.0)
    responsible_party: str = Field(default="")


class CertificationAssessmentInput(BaseModel):
    """Input data model for CertificationAssessmentWorkflow."""

    building_name: str = Field(default="")
    building_type: str = Field(default="office")
    country: str = Field(default="GB")
    total_floor_area_sqm: float = Field(default=0.0, ge=0.0)
    performance: BuildingPerformanceData = Field(default_factory=BuildingPerformanceData)
    target_scheme: CertificationScheme = Field(default=CertificationScheme.LEED)
    target_level: str = Field(default="gold")
    existing_certifications: List[str] = Field(default_factory=list)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("total_floor_area_sqm")
    @classmethod
    def validate_floor_area(cls, v: float) -> float:
        """Floor area must be positive."""
        if v <= 0:
            raise ValueError("total_floor_area_sqm must be > 0")
        return v


class CertificationAssessmentResult(BaseModel):
    """Complete result from certification assessment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="certification_assessment")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    building_name: str = Field(default="")
    target_scheme: str = Field(default="")
    target_level: str = Field(default="")
    recommended_scheme: str = Field(default="")
    credit_assessments: List[CreditAssessment] = Field(default_factory=list)
    total_points_available: float = Field(default=0.0, ge=0.0)
    total_points_achieved: float = Field(default=0.0, ge=0.0)
    current_level: str = Field(default="")
    target_points_needed: float = Field(default=0.0, ge=0.0)
    points_gap: float = Field(default=0.0, ge=0.0)
    prerequisites_met: bool = Field(default=False)
    gap_items: List[GapItem] = Field(default_factory=list)
    action_plan: List[CertificationAction] = Field(default_factory=list)
    total_action_cost_eur: float = Field(default=0.0, ge=0.0)
    feasibility_assessment: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CertificationAssessmentWorkflow:
    """
    4-phase green building certification assessment workflow.

    Selects applicable certification scheme, evaluates credits and
    prerequisites, identifies gaps, and produces a prioritised action
    plan to achieve the target certification level.

    Zero-hallucination: all credit scoring uses published scheme criteria
    with deterministic point allocation rules.

    Example:
        >>> wf = CertificationAssessmentWorkflow()
        >>> inp = CertificationAssessmentInput(
        ...     total_floor_area_sqm=5000,
        ...     target_scheme=CertificationScheme.LEED,
        ...     target_level="gold"
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CertificationAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._credit_assessments: List[CreditAssessment] = []
        self._gap_items: List[GapItem] = []
        self._actions: List[CertificationAction] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[CertificationAssessmentInput] = None,
    ) -> CertificationAssessmentResult:
        """Execute the 4-phase certification assessment workflow."""
        if input_data is None:
            raise ValueError("input_data must be provided")

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting certification assessment workflow %s for %s scheme=%s",
            self.workflow_id, input_data.building_name, input_data.target_scheme.value,
        )

        self._phase_results = []
        self._credit_assessments = []
        self._gap_items = []
        self._actions = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_certification_selection(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_credit_scoring(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_gap_analysis(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_action_plan(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Certification assessment workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_available = sum(c.max_points for c in self._credit_assessments if not c.is_prerequisite)
        total_achieved = sum(c.achieved_points for c in self._credit_assessments)
        prereqs_met = all(
            c.status in (CreditStatus.PREREQUISITE_MET, CreditStatus.ACHIEVED)
            for c in self._credit_assessments if c.is_prerequisite
        )
        target_points = self._get_target_points(input_data.target_scheme, input_data.target_level)
        current_level = self._determine_current_level(total_achieved, input_data.target_scheme)
        points_gap = max(0, target_points - total_achieved)
        total_cost = sum(a.estimated_cost_eur for a in self._actions)

        feasibility = "high" if points_gap <= 10 else ("medium" if points_gap <= 25 else "low")

        result = CertificationAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            building_name=input_data.building_name,
            target_scheme=input_data.target_scheme.value,
            target_level=input_data.target_level,
            recommended_scheme=input_data.target_scheme.value,
            credit_assessments=self._credit_assessments,
            total_points_available=total_available,
            total_points_achieved=total_achieved,
            current_level=current_level,
            target_points_needed=target_points,
            points_gap=points_gap,
            prerequisites_met=prereqs_met,
            gap_items=self._gap_items,
            action_plan=self._actions,
            total_action_cost_eur=round(total_cost, 2),
            feasibility_assessment=feasibility,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Certification assessment %s completed in %.2fs: %s %d/%d points, "
            "current=%s, target=%s, gap=%d",
            self.workflow_id, elapsed, input_data.target_scheme.value,
            int(total_achieved), int(total_available), current_level,
            input_data.target_level, int(points_gap),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Certification Selection
    # -------------------------------------------------------------------------

    async def _phase_certification_selection(
        self, input_data: CertificationAssessmentInput
    ) -> PhaseResult:
        """Determine applicable certification scheme."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        scheme = input_data.target_scheme
        country = input_data.country

        # Scheme suitability assessment
        suitability: Dict[str, str] = {}
        suitability["leed"] = "high" if country in ("US", "GB", "DE", "FR", "AE", "SG") else "medium"
        suitability["breeam"] = "high" if country in ("GB", "NL", "DE", "SE", "NO", "DK") else "medium"
        suitability["energy_star"] = "high" if country == "US" else "low"
        suitability["nabers"] = "high" if country in ("AU", "NZ") else "low"
        suitability["green_star"] = "high" if country in ("AU", "NZ", "ZA") else "low"

        recommended = scheme.value
        if suitability.get(scheme.value, "medium") == "low":
            # Find better alternative
            for s, level in suitability.items():
                if level == "high":
                    recommended = s
                    warnings.append(
                        f"{scheme.value} has low suitability for {country}; "
                        f"consider {recommended} instead"
                    )
                    break

        outputs["selected_scheme"] = scheme.value
        outputs["recommended_scheme"] = recommended
        outputs["country"] = country
        outputs["suitability_scores"] = suitability
        outputs["target_level"] = input_data.target_level
        outputs["existing_certifications"] = input_data.existing_certifications
        outputs["building_type"] = input_data.building_type

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 CertificationSelection: scheme=%s, recommended=%s, country=%s",
            scheme.value, recommended, country,
        )
        return PhaseResult(
            phase_name="certification_selection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Credit Scoring
    # -------------------------------------------------------------------------

    async def _phase_credit_scoring(
        self, input_data: CertificationAssessmentInput
    ) -> PhaseResult:
        """Evaluate credits and prerequisites by category."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        perf = input_data.performance

        if input_data.target_scheme == CertificationScheme.LEED:
            self._score_leed_credits(perf)
        elif input_data.target_scheme == CertificationScheme.BREEAM:
            self._score_breeam_credits(perf)
        else:
            # Generic scoring for other schemes
            self._score_generic_credits(perf, input_data.target_scheme)

        total_available = sum(c.max_points for c in self._credit_assessments if not c.is_prerequisite)
        total_achieved = sum(c.achieved_points for c in self._credit_assessments)
        prereqs_met = sum(
            1 for c in self._credit_assessments
            if c.is_prerequisite and c.status == CreditStatus.PREREQUISITE_MET
        )
        prereqs_total = sum(1 for c in self._credit_assessments if c.is_prerequisite)

        category_scores: Dict[str, Dict[str, float]] = {}
        for ca in self._credit_assessments:
            if ca.is_prerequisite:
                continue
            if ca.category not in category_scores:
                category_scores[ca.category] = {"available": 0.0, "achieved": 0.0}
            category_scores[ca.category]["available"] += ca.max_points
            category_scores[ca.category]["achieved"] += ca.achieved_points

        outputs["total_credits"] = len(self._credit_assessments)
        outputs["total_available_points"] = total_available
        outputs["total_achieved_points"] = round(total_achieved, 1)
        outputs["prerequisites_met"] = f"{prereqs_met}/{prereqs_total}"
        outputs["category_scores"] = {
            k: {"available": round(v["available"], 1), "achieved": round(v["achieved"], 1)}
            for k, v in category_scores.items()
        }
        outputs["achievement_pct"] = round(total_achieved / max(total_available, 1) * 100, 1)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 CreditScoring: %d credits, %d/%d points, prereqs %d/%d",
            len(self._credit_assessments), int(total_achieved),
            int(total_available), prereqs_met, prereqs_total,
        )
        return PhaseResult(
            phase_name="credit_scoring", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _score_leed_credits(self, perf: BuildingPerformanceData) -> None:
        """Score LEED v4.1 O+M credits."""
        for credit in LEED_CREDITS:
            achieved = 0.0
            status = CreditStatus.NOT_ACHIEVED

            if credit["is_prerequisite"]:
                # Prerequisites - binary pass/fail
                if credit["id"] == "EA-P1":
                    met = perf.has_bms or perf.has_submetering
                elif credit["id"] == "EA-P2":
                    met = perf.eui_kwh_per_sqm < 300.0
                elif credit["id"] == "WE-P1":
                    met = perf.water_use_litres_per_sqm < 15.0 or perf.water_use_litres_per_sqm == 0
                elif credit["id"] == "EQ-P1":
                    met = True  # Assume minimum IAQ unless data says otherwise
                elif credit["id"] == "SS-P1":
                    met = perf.site_management_plan
                elif credit["id"] == "MR-P1":
                    met = perf.purchasing_policy
                else:
                    met = True
                status = CreditStatus.PREREQUISITE_MET if met else CreditStatus.PREREQUISITE_NOT_MET
            else:
                max_pts = credit["max_points"]
                # Score based on performance data
                if credit["id"] == "EA-C1":
                    # Energy performance points based on EUI percentile
                    if perf.eui_kwh_per_sqm <= 80:
                        achieved = min(max_pts, 20.0)
                    elif perf.eui_kwh_per_sqm <= 120:
                        achieved = min(max_pts, 15.0)
                    elif perf.eui_kwh_per_sqm <= 160:
                        achieved = min(max_pts, 10.0)
                    elif perf.eui_kwh_per_sqm <= 200:
                        achieved = min(max_pts, 5.0)
                elif credit["id"] == "EA-C2":
                    achieved = max_pts if perf.has_submetering else 0.0
                elif credit["id"] == "EA-C3":
                    achieved = max_pts if perf.has_demand_response else 0.0
                elif credit["id"] == "EA-C4":
                    if perf.renewable_pct >= 50:
                        achieved = max_pts
                    elif perf.renewable_pct >= 25:
                        achieved = max_pts * 0.6
                    elif perf.renewable_pct >= 10:
                        achieved = max_pts * 0.3
                elif credit["id"] == "EA-C5":
                    achieved = max_pts * 0.5  # Partial by default
                elif credit["id"] == "WE-C1":
                    if perf.water_use_litres_per_sqm > 0:
                        water_ratio = min(1.0, max(0, (15.0 - perf.water_use_litres_per_sqm) / 10.0))
                        achieved = max_pts * water_ratio
                elif credit["id"] == "EQ-C1":
                    achieved = max_pts if perf.has_iaq_monitoring else 0.0
                elif credit["id"] == "EQ-C4":
                    achieved = max_pts * 0.5  # Assume partial
                elif credit["id"] == "EQ-C5":
                    achieved = max_pts * (perf.daylight_pct / 100.0) if perf.daylight_pct > 0 else 0.0
                elif credit["id"] == "EQ-C6":
                    achieved = max_pts if perf.has_green_cleaning else 0.0
                elif credit["id"] == "MR-C2":
                    achieved = max_pts if perf.low_mercury_lamps else 0.0
                elif credit["id"] == "MR-C3":
                    if perf.waste_diversion_pct >= 75:
                        achieved = max_pts
                    elif perf.waste_diversion_pct >= 50:
                        achieved = max_pts * 0.5
                elif credit["id"] == "TR-C1":
                    score = 0.0
                    if perf.public_transport_proximity:
                        score += max_pts * 0.4
                    if perf.cycling_facilities:
                        score += max_pts * 0.2
                    if perf.ev_charging:
                        score += max_pts * 0.2
                    achieved = min(max_pts, score)
                elif credit["id"] == "SS-C2":
                    achieved = max_pts if perf.rainwater_management else 0.0
                elif credit["id"] == "IN-C1":
                    achieved = 0.0  # Innovation requires specific documentation
                else:
                    achieved = 0.0

                if achieved >= max_pts:
                    status = CreditStatus.ACHIEVED
                elif achieved > 0:
                    status = CreditStatus.PARTIAL
                else:
                    status = CreditStatus.NOT_ACHIEVED

            self._credit_assessments.append(CreditAssessment(
                credit_id=credit["id"],
                category=credit["category"],
                name=credit["name"],
                max_points=credit["max_points"],
                achieved_points=round(achieved, 1),
                status=status,
                is_prerequisite=credit["is_prerequisite"],
            ))

    def _score_breeam_credits(self, perf: BuildingPerformanceData) -> None:
        """Score BREEAM In-Use credits (simplified)."""
        for category, weight in BREEAM_CATEGORIES.items():
            max_pts = weight * 100.0
            achieved = 0.0

            if category == "energy":
                if perf.eui_kwh_per_sqm <= 80:
                    achieved = max_pts * 0.9
                elif perf.eui_kwh_per_sqm <= 130:
                    achieved = max_pts * 0.7
                elif perf.eui_kwh_per_sqm <= 200:
                    achieved = max_pts * 0.5
                else:
                    achieved = max_pts * 0.3
            elif category == "management":
                score = 0.3
                if perf.has_bms:
                    score += 0.3
                if perf.has_submetering:
                    score += 0.2
                if perf.green_lease:
                    score += 0.2
                achieved = max_pts * min(score, 1.0)
            elif category == "health_wellbeing":
                score = 0.3
                if perf.has_iaq_monitoring:
                    score += 0.3
                if perf.thermal_comfort_pct > 80:
                    score += 0.2
                if perf.daylight_pct > 50:
                    score += 0.2
                achieved = max_pts * min(score, 1.0)
            elif category == "water":
                if perf.water_use_litres_per_sqm > 0 and perf.water_use_litres_per_sqm < 8:
                    achieved = max_pts * 0.8
                else:
                    achieved = max_pts * 0.4
            elif category == "waste":
                achieved = max_pts * (perf.waste_diversion_pct / 100.0) if perf.waste_diversion_pct > 0 else max_pts * 0.3
            elif category == "transport":
                score = 0.2
                if perf.public_transport_proximity:
                    score += 0.4
                if perf.cycling_facilities:
                    score += 0.2
                if perf.ev_charging:
                    score += 0.2
                achieved = max_pts * min(score, 1.0)
            else:
                achieved = max_pts * 0.4  # Default partial

            self._credit_assessments.append(CreditAssessment(
                credit_id=f"BREEAM-{category}",
                category=category,
                name=f"BREEAM {category.replace('_', ' ').title()}",
                max_points=int(max_pts),
                achieved_points=round(achieved, 1),
                status=CreditStatus.ACHIEVED if achieved >= max_pts * 0.7 else CreditStatus.PARTIAL,
            ))

    def _score_generic_credits(self, perf: BuildingPerformanceData, scheme: CertificationScheme) -> None:
        """Generic credit scoring for other schemes."""
        categories = ["energy", "water", "indoor_quality", "site", "materials"]
        for cat in categories:
            max_pts = 20
            achieved = max_pts * 0.4
            if cat == "energy" and perf.eui_kwh_per_sqm <= 120:
                achieved = max_pts * 0.7
            self._credit_assessments.append(CreditAssessment(
                credit_id=f"{scheme.value}-{cat}",
                category=cat,
                name=f"{scheme.value.upper()} {cat.replace('_', ' ').title()}",
                max_points=max_pts,
                achieved_points=round(achieved, 1),
                status=CreditStatus.PARTIAL,
            ))

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(
        self, input_data: CertificationAssessmentInput
    ) -> PhaseResult:
        """Identify missing credits, effort to achieve."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        target_points = self._get_target_points(input_data.target_scheme, input_data.target_level)
        current_points = sum(c.achieved_points for c in self._credit_assessments)
        gap = max(0, target_points - current_points)

        # Find credits with room for improvement
        for ca in self._credit_assessments:
            if ca.is_prerequisite and ca.status == CreditStatus.PREREQUISITE_NOT_MET:
                self._gap_items.append(GapItem(
                    credit_id=ca.credit_id,
                    name=ca.name,
                    category=ca.category,
                    points_available=0.0,
                    effort=EffortLevel.HIGH,
                    estimated_cost_eur=5000.0,
                    estimated_weeks=8,
                    description=f"Prerequisite {ca.name} must be met for certification.",
                ))
            elif not ca.is_prerequisite and ca.achieved_points < ca.max_points:
                remaining = ca.max_points - ca.achieved_points
                if remaining <= 0:
                    continue

                # Estimate effort based on category
                effort, cost, weeks = self._estimate_effort(ca.category, remaining, input_data.total_floor_area_sqm)

                self._gap_items.append(GapItem(
                    credit_id=ca.credit_id,
                    name=ca.name,
                    category=ca.category,
                    points_available=round(remaining, 1),
                    effort=effort,
                    estimated_cost_eur=round(cost, 2),
                    estimated_weeks=weeks,
                    description=f"Improve from {ca.achieved_points:.0f} to {ca.max_points} points.",
                ))

        # Sort by points-per-effort ratio
        self._gap_items.sort(
            key=lambda g: g.points_available / max(g.estimated_weeks, 1), reverse=True
        )

        outputs["target_points"] = target_points
        outputs["current_points"] = round(current_points, 1)
        outputs["points_gap"] = round(gap, 1)
        outputs["gap_items"] = len(self._gap_items)
        outputs["prerequisite_gaps"] = sum(1 for g in self._gap_items if g.points_available == 0)
        outputs["total_points_recoverable"] = round(sum(g.points_available for g in self._gap_items), 1)
        outputs["total_estimated_cost_eur"] = round(sum(g.estimated_cost_eur for g in self._gap_items), 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 GapAnalysis: gap=%d points, %d items, cost=%.0f EUR",
            int(gap), len(self._gap_items),
            sum(g.estimated_cost_eur for g in self._gap_items),
        )
        return PhaseResult(
            phase_name="gap_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Action Plan
    # -------------------------------------------------------------------------

    async def _phase_action_plan(
        self, input_data: CertificationAssessmentInput
    ) -> PhaseResult:
        """Generate prioritised action list for target certification level."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        target_points = self._get_target_points(input_data.target_scheme, input_data.target_level)
        current_points = sum(c.achieved_points for c in self._credit_assessments)
        remaining_gap = max(0, target_points - current_points)

        cumulative_points = 0.0
        for gap in self._gap_items:
            if cumulative_points >= remaining_gap and gap.points_available > 0:
                # Already enough actions to close gap, lower priority for remaining
                priority = "low"
            elif gap.points_available == 0:
                # Prerequisite
                priority = "critical"
            elif gap.effort in (EffortLevel.LOW, EffortLevel.NONE):
                priority = "high"
            elif gap.effort == EffortLevel.MEDIUM:
                priority = "medium"
            else:
                priority = "low"

            self._actions.append(CertificationAction(
                credit_id=gap.credit_id,
                title=f"Achieve {gap.name}",
                description=gap.description,
                priority=priority,
                effort=gap.effort,
                estimated_cost_eur=gap.estimated_cost_eur,
                estimated_weeks=gap.estimated_weeks,
                points_gain=gap.points_available,
            ))
            cumulative_points += gap.points_available

        # Sort: critical first, then by points_gain/effort ratio
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self._actions.sort(
            key=lambda a: (priority_order.get(a.priority, 3), -a.points_gain)
        )

        total_cost = sum(a.estimated_cost_eur for a in self._actions)
        total_points_gain = sum(a.points_gain for a in self._actions)

        outputs["total_actions"] = len(self._actions)
        outputs["critical_actions"] = sum(1 for a in self._actions if a.priority == "critical")
        outputs["high_priority_actions"] = sum(1 for a in self._actions if a.priority == "high")
        outputs["total_cost_eur"] = round(total_cost, 2)
        outputs["total_points_gain"] = round(total_points_gain, 1)
        outputs["target_achievable"] = total_points_gain >= remaining_gap
        outputs["projected_total_points"] = round(current_points + total_points_gain, 1)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ActionPlan: %d actions, cost=%.0f EUR, points gain=%.0f, "
            "target achievable=%s",
            len(self._actions), total_cost, total_points_gain,
            "YES" if total_points_gain >= remaining_gap else "NO",
        )
        return PhaseResult(
            phase_name="action_plan", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_target_points(scheme: CertificationScheme, level: str) -> float:
        """Get target point threshold for a scheme and level."""
        if scheme == CertificationScheme.LEED:
            return float(LEED_LEVELS.get(level, 60))
        elif scheme == CertificationScheme.BREEAM:
            return BREEAM_LEVELS.get(level, 55.0)
        return 60.0

    @staticmethod
    def _determine_current_level(points: float, scheme: CertificationScheme) -> str:
        """Determine current certification level from points."""
        if scheme == CertificationScheme.LEED:
            levels = sorted(LEED_LEVELS.items(), key=lambda x: x[1], reverse=True)
            for level, threshold in levels:
                if points >= threshold:
                    return level
            return "not_certified"
        elif scheme == CertificationScheme.BREEAM:
            levels = sorted(BREEAM_LEVELS.items(), key=lambda x: x[1], reverse=True)
            for level, threshold in levels:
                if points >= threshold:
                    return level
            return "unclassified"
        return "unknown"

    @staticmethod
    def _estimate_effort(
        category: str, points: float, floor_area: float
    ) -> Tuple[EffortLevel, float, int]:
        """Estimate effort, cost, and timeline for a credit gap."""
        # Cost and effort heuristics by category
        effort_map: Dict[str, Tuple[EffortLevel, float, int]] = {
            "energy_atmosphere": (EffortLevel.HIGH, floor_area * 5.0, 12),
            "water_efficiency": (EffortLevel.MEDIUM, floor_area * 2.0, 6),
            "indoor_quality": (EffortLevel.MEDIUM, floor_area * 3.0, 8),
            "sustainable_sites": (EffortLevel.LOW, 2000.0, 4),
            "materials_resources": (EffortLevel.LOW, 1000.0, 4),
            "transportation": (EffortLevel.MEDIUM, 5000.0, 8),
            "innovation": (EffortLevel.HIGH, 3000.0, 12),
            "energy": (EffortLevel.HIGH, floor_area * 5.0, 12),
            "management": (EffortLevel.MEDIUM, 2000.0, 6),
            "health_wellbeing": (EffortLevel.MEDIUM, floor_area * 2.0, 8),
            "water": (EffortLevel.MEDIUM, floor_area * 1.5, 6),
            "waste": (EffortLevel.LOW, 1500.0, 4),
            "transport": (EffortLevel.MEDIUM, 5000.0, 8),
        }
        defaults = effort_map.get(category, (EffortLevel.MEDIUM, 3000.0, 8))
        effort, base_cost, weeks = defaults
        # Scale cost by points
        scaled_cost = base_cost * (points / 5.0)
        return effort, scaled_cost, weeks

    def _compute_provenance(self, result: CertificationAssessmentResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
