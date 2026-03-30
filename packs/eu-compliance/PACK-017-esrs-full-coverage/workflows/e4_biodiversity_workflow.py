# -*- coding: utf-8 -*-
"""
ESRS E4 Biodiversity and Ecosystems Workflow
==============================================

6-phase workflow for ESRS E4 Biodiversity and Ecosystems disclosure covering
transition plan review, policy assessment, site-level biodiversity assessment,
species impact analysis, target evaluation, and financial effects with full
provenance tracking.

Phases:
    1. TransitionPlan      -- Review biodiversity transition plan (E4-1)
    2. PolicyReview        -- Review biodiversity policies and actions (E4-2, E4-3)
    3. SiteAssessment      -- Assess biodiversity at operational sites (E4-4)
    4. SpeciesImpact       -- Evaluate impact on species and habitats (E4-5)
    5. TargetEvaluation    -- Evaluate biodiversity targets (E4-3)
    6. FinancialEffects    -- Assess financial effects from biodiversity (E4-6)

ESRS E4 Disclosure Requirements (6 DRs):
    E4-1: Transition plan and consideration of biodiversity in strategy
    E4-2: Policies related to biodiversity and ecosystems
    E4-3: Actions and resources related to biodiversity and ecosystems
    E4-4: Targets related to biodiversity and ecosystems
    E4-5: Impact metrics related to biodiversity and ecosystems change
    E4-6: Anticipated financial effects from biodiversity-related risks

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

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

class WorkflowPhase(str, Enum):
    """Phases of the E4 biodiversity workflow."""
    TRANSITION_PLAN = "transition_plan"
    POLICY_REVIEW = "policy_review"
    SITE_ASSESSMENT = "site_assessment"
    SPECIES_IMPACT = "species_impact"
    TARGET_EVALUATION = "target_evaluation"
    FINANCIAL_EFFECTS = "financial_effects"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class SensitivityLevel(str, Enum):
    """Biodiversity sensitivity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProtectedAreaType(str, Enum):
    """Protected area classification."""
    NATURA_2000 = "natura_2000"
    KEY_BIODIVERSITY_AREA = "key_biodiversity_area"
    UNESCO_WORLD_HERITAGE = "unesco_world_heritage"
    RAMSAR_WETLAND = "ramsar_wetland"
    NATIONAL_PARK = "national_park"
    OTHER_PROTECTED = "other_protected"
    NONE = "none"

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

class SiteRecord(BaseModel):
    """Operational site biodiversity assessment record per E4-4/E4-5."""
    site_id: str = Field(default_factory=lambda: f"site-{_new_uuid()[:8]}")
    site_name: str = Field(default="")
    latitude: float = Field(default=0.0, ge=-90.0, le=90.0)
    longitude: float = Field(default=0.0, ge=-180.0, le=180.0)
    area_hectares: float = Field(default=0.0, ge=0.0)
    sensitivity_level: SensitivityLevel = Field(default=SensitivityLevel.LOW)
    protected_area_type: ProtectedAreaType = Field(default=ProtectedAreaType.NONE)
    land_use_change_hectares: float = Field(default=0.0, description="Land use change in hectares")
    land_degradation_hectares: float = Field(default=0.0, description="Degraded land in hectares")
    restoration_hectares: float = Field(default=0.0, ge=0.0, description="Restored land in hectares")

class SpeciesImpactRecord(BaseModel):
    """Species impact assessment record per E4-5."""
    record_id: str = Field(default_factory=lambda: f"sp-{_new_uuid()[:8]}")
    species_name: str = Field(default="")
    iucn_category: str = Field(default="", description="LC, NT, VU, EN, CR, EW, EX")
    impact_type: str = Field(default="", description="habitat_loss, pollution, disturbance, etc.")
    affected_population_estimate: int = Field(default=0, ge=0)
    mitigation_in_place: bool = Field(default=False)
    site_id: str = Field(default="")

class BiodiversityTarget(BaseModel):
    """Biodiversity target per E4-4."""
    target_id: str = Field(default_factory=lambda: f"bt-{_new_uuid()[:8]}")
    target_name: str = Field(default="")
    metric: str = Field(default="", description="Target metric (e.g., no_net_loss, net_positive)")
    base_year: int = Field(default=2019)
    target_year: int = Field(default=2030)
    current_progress_pct: float = Field(default=0.0, ge=0.0, le=200.0)
    on_track: bool = Field(default=False)
    aligned_with_gbf: bool = Field(default=False, description="Aligned with Global Biodiversity Framework")

class E4BiodiversityInput(BaseModel):
    """Input data model for E4BiodiversityWorkflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    e4_is_material: bool = Field(default=True, description="Whether E4 is material")
    transition_plan_data: Dict[str, Any] = Field(
        default_factory=dict, description="E4-1 transition plan"
    )
    policies: List[Dict[str, Any]] = Field(
        default_factory=list, description="E4-2 biodiversity policies"
    )
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="E4-3 actions and resources"
    )
    site_records: List[SiteRecord] = Field(
        default_factory=list, description="E4-4/E4-5 site assessments"
    )
    species_impacts: List[SpeciesImpactRecord] = Field(
        default_factory=list, description="E4-5 species impact records"
    )
    targets: List[BiodiversityTarget] = Field(
        default_factory=list, description="E4-4 biodiversity targets"
    )
    financial_effects_data: Dict[str, Any] = Field(
        default_factory=dict, description="E4-6 financial effects"
    )
    config: Dict[str, Any] = Field(default_factory=dict)

class E4BiodiversityWorkflowResult(BaseModel):
    """Complete result from E4 biodiversity workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="e4_biodiversity")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    e4_is_material: bool = Field(default=True)
    total_sites_assessed: int = Field(default=0)
    sites_in_protected_areas: int = Field(default=0)
    sites_critical_sensitivity: int = Field(default=0)
    total_land_use_change_ha: float = Field(default=0.0)
    total_restoration_ha: float = Field(default=0.0)
    threatened_species_count: int = Field(default=0)
    has_transition_plan: bool = Field(default=False)
    targets_on_track: int = Field(default=0)
    targets_total: int = Field(default=0)
    has_financial_effects: bool = Field(default=False)
    overall_completeness_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class E4BiodiversityWorkflow:
    """
    6-phase ESRS E4 Biodiversity and Ecosystems workflow.

    Orchestrates transition plan review, policy assessment, site assessment,
    species impact analysis, target evaluation, and financial effects for
    complete E4 disclosure covering E4-1 through E4-6.

    Zero-hallucination: all site and species aggregations use deterministic
    arithmetic. No LLM in numeric calculation paths.

    Example:
        >>> wf = E4BiodiversityWorkflow()
        >>> inp = E4BiodiversityInput(site_records=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_sites_assessed >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize E4BiodiversityWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.TRANSITION_PLAN.value, "description": "Review biodiversity transition plan"},
            {"name": WorkflowPhase.POLICY_REVIEW.value, "description": "Review biodiversity policies"},
            {"name": WorkflowPhase.SITE_ASSESSMENT.value, "description": "Assess operational sites"},
            {"name": WorkflowPhase.SPECIES_IMPACT.value, "description": "Evaluate species impacts"},
            {"name": WorkflowPhase.TARGET_EVALUATION.value, "description": "Evaluate biodiversity targets"},
            {"name": WorkflowPhase.FINANCIAL_EFFECTS.value, "description": "Assess financial effects"},
        ]

    def validate_inputs(self, input_data: E4BiodiversityInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.e4_is_material:
            issues.append("E4 is not material; full disclosure not required")
        if not input_data.site_records:
            issues.append("No site records provided for biodiversity assessment")
        return issues

    async def execute(
        self,
        input_data: Optional[E4BiodiversityInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> E4BiodiversityWorkflowResult:
        """
        Execute the 6-phase E4 biodiversity workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            E4BiodiversityWorkflowResult with site and species assessments.
        """
        if input_data is None:
            input_data = E4BiodiversityInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting E4 biodiversity workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_transition_plan(input_data))
            phase_results.append(await self._phase_policy_review(input_data))
            phase_results.append(await self._phase_site_assessment(input_data))
            phase_results.append(await self._phase_species_impact(input_data))
            phase_results.append(await self._phase_target_evaluation(input_data))
            phase_results.append(await self._phase_financial_effects(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("E4 biodiversity workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        protected = sum(1 for s in input_data.site_records if s.protected_area_type != ProtectedAreaType.NONE)
        critical = sum(1 for s in input_data.site_records if s.sensitivity_level == SensitivityLevel.CRITICAL)
        luc = sum(s.land_use_change_hectares for s in input_data.site_records)
        restoration = sum(s.restoration_hectares for s in input_data.site_records)
        threatened = sum(
            1 for sp in input_data.species_impacts
            if sp.iucn_category in ("VU", "EN", "CR")
        )
        on_track = sum(1 for t in input_data.targets if t.on_track)
        completeness = self._calculate_completeness(input_data)

        result = E4BiodiversityWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            e4_is_material=input_data.e4_is_material,
            total_sites_assessed=len(input_data.site_records),
            sites_in_protected_areas=protected,
            sites_critical_sensitivity=critical,
            total_land_use_change_ha=round(luc, 2),
            total_restoration_ha=round(restoration, 2),
            threatened_species_count=threatened,
            has_transition_plan=bool(input_data.transition_plan_data),
            targets_on_track=on_track,
            targets_total=len(input_data.targets),
            has_financial_effects=bool(input_data.financial_effects_data),
            overall_completeness_pct=completeness,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "E4 biodiversity %s completed in %.2fs: %d sites, %d in protected areas",
            self.workflow_id, elapsed, len(input_data.site_records), protected,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Transition Plan (E4-1)
    # -------------------------------------------------------------------------

    async def _phase_transition_plan(self, input_data: E4BiodiversityInput) -> PhaseResult:
        """Review biodiversity transition plan and strategy alignment."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.transition_plan_data
        outputs["has_transition_plan"] = bool(data)
        outputs["has_no_net_loss_commitment"] = bool(data.get("no_net_loss_commitment"))
        outputs["has_net_positive_goal"] = bool(data.get("net_positive_goal"))
        outputs["aligned_with_gbf"] = bool(data.get("gbf_alignment"))
        outputs["has_deforestation_commitment"] = bool(data.get("deforestation_free"))

        if not data:
            warnings.append("No biodiversity transition plan data provided (E4-1)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 1 TransitionPlan: has_plan=%s", bool(data))
        return PhaseResult(
            phase_name=WorkflowPhase.TRANSITION_PLAN.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Policy Review (E4-2, E4-3)
    # -------------------------------------------------------------------------

    async def _phase_policy_review(self, input_data: E4BiodiversityInput) -> PhaseResult:
        """Review biodiversity policies and actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["policies_count"] = len(input_data.policies)
        outputs["actions_count"] = len(input_data.actions)
        outputs["has_mitigation_hierarchy"] = any(
            p.get("mitigation_hierarchy") for p in input_data.policies
        )

        if not input_data.policies:
            warnings.append("No biodiversity policies defined (E4-2)")
        if not input_data.actions:
            warnings.append("No biodiversity actions defined (E4-3)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 2 PolicyReview: %d policies, %d actions",
                         len(input_data.policies), len(input_data.actions))
        return PhaseResult(
            phase_name=WorkflowPhase.POLICY_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Site Assessment (E4-4, E4-5)
    # -------------------------------------------------------------------------

    async def _phase_site_assessment(self, input_data: E4BiodiversityInput) -> PhaseResult:
        """Assess biodiversity at operational sites."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sites = input_data.site_records
        protected = [s for s in sites if s.protected_area_type != ProtectedAreaType.NONE]
        critical = [s for s in sites if s.sensitivity_level == SensitivityLevel.CRITICAL]
        total_area = sum(s.area_hectares for s in sites)
        total_luc = sum(s.land_use_change_hectares for s in sites)
        total_degradation = sum(s.land_degradation_hectares for s in sites)
        total_restoration = sum(s.restoration_hectares for s in sites)

        outputs["sites_assessed"] = len(sites)
        outputs["in_protected_areas"] = len(protected)
        outputs["critical_sensitivity"] = len(critical)
        outputs["total_area_ha"] = round(total_area, 2)
        outputs["total_land_use_change_ha"] = round(total_luc, 2)
        outputs["total_degradation_ha"] = round(total_degradation, 2)
        outputs["total_restoration_ha"] = round(total_restoration, 2)
        outputs["sensitivity_distribution"] = {
            level.value: sum(1 for s in sites if s.sensitivity_level == level)
            for level in SensitivityLevel
        }

        if not sites:
            warnings.append("No site records provided for biodiversity assessment")
        if critical:
            warnings.append(f"{len(critical)} sites at critical biodiversity sensitivity")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 3 SiteAssessment: %d sites, %d in protected areas",
                         len(sites), len(protected))
        return PhaseResult(
            phase_name=WorkflowPhase.SITE_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Species Impact (E4-5)
    # -------------------------------------------------------------------------

    async def _phase_species_impact(self, input_data: E4BiodiversityInput) -> PhaseResult:
        """Evaluate impact on species and habitats."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        species = input_data.species_impacts
        threatened = [sp for sp in species if sp.iucn_category in ("VU", "EN", "CR")]
        mitigated = [sp for sp in species if sp.mitigation_in_place]

        outputs["species_assessed"] = len(species)
        outputs["threatened_species"] = len(threatened)
        outputs["with_mitigation"] = len(mitigated)
        outputs["iucn_distribution"] = {}
        for sp in species:
            cat = sp.iucn_category or "unclassified"
            outputs["iucn_distribution"][cat] = outputs["iucn_distribution"].get(cat, 0) + 1
        outputs["impact_types"] = list(set(sp.impact_type for sp in species if sp.impact_type))

        if threatened and not mitigated:
            warnings.append("Threatened species impacted but no mitigation measures in place")
        if not species:
            warnings.append("No species impact records provided (E4-5)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 4 SpeciesImpact: %d species, %d threatened",
                         len(species), len(threatened))
        return PhaseResult(
            phase_name=WorkflowPhase.SPECIES_IMPACT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Target Evaluation (E4-4)
    # -------------------------------------------------------------------------

    async def _phase_target_evaluation(self, input_data: E4BiodiversityInput) -> PhaseResult:
        """Evaluate biodiversity targets and progress."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        targets = input_data.targets
        on_track = [t for t in targets if t.on_track]
        gbf_aligned = [t for t in targets if t.aligned_with_gbf]

        outputs["targets_count"] = len(targets)
        outputs["on_track_count"] = len(on_track)
        outputs["gbf_aligned_count"] = len(gbf_aligned)
        outputs["avg_progress_pct"] = round(
            sum(t.current_progress_pct for t in targets) / len(targets)
            if targets else 0.0, 1
        )

        if not targets:
            warnings.append("No biodiversity targets defined (E4-4)")
        if targets and not gbf_aligned:
            warnings.append("No targets aligned with Global Biodiversity Framework")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 5 TargetEvaluation: %d targets, %d on track",
                         len(targets), len(on_track))
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_EVALUATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Financial Effects (E4-6)
    # -------------------------------------------------------------------------

    async def _phase_financial_effects(self, input_data: E4BiodiversityInput) -> PhaseResult:
        """Assess anticipated financial effects from biodiversity risks."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.financial_effects_data
        outputs["has_financial_effects"] = bool(data)
        outputs["ecosystem_services_dependency_eur"] = data.get("ecosystem_services_dependency_eur", 0)
        outputs["remediation_costs_eur"] = data.get("remediation_costs_eur", 0)
        outputs["biodiversity_offset_costs_eur"] = data.get("biodiversity_offset_costs_eur", 0)
        outputs["total_exposure_eur"] = (
            data.get("ecosystem_services_dependency_eur", 0)
            + data.get("remediation_costs_eur", 0)
            + data.get("biodiversity_offset_costs_eur", 0)
        )

        if not data:
            warnings.append("No financial effects data provided (E4-6)")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Phase 6 FinancialEffects: exposure=%d EUR", outputs["total_exposure_eur"])
        return PhaseResult(
            phase_name=WorkflowPhase.FINANCIAL_EFFECTS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_completeness(self, input_data: E4BiodiversityInput) -> float:
        """Calculate overall E4 completeness percentage."""
        scores: List[float] = []
        scores.append(100.0 if input_data.transition_plan_data else 0.0)
        scores.append(100.0 if input_data.policies else 0.0)
        scores.append(100.0 if input_data.actions else 0.0)
        scores.append(100.0 if input_data.targets else 0.0)
        scores.append(100.0 if input_data.site_records else 0.0)
        scores.append(100.0 if input_data.financial_effects_data else 0.0)
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    def _compute_provenance(self, result: E4BiodiversityWorkflowResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
