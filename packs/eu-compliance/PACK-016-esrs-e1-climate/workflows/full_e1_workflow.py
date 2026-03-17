# -*- coding: utf-8 -*-
"""
Full E1 Workflow
==============================

10-phase end-to-end ESRS E1 disclosure generation workflow. Orchestrates
all sub-workflows into a complete climate disclosure covering E1-1 through
E1-9 with aggregated results and completeness tracking.

Phases:
    1.  MaterialityCheck       -- Verify E1 is material for this entity
    2.  GHGInventory           -- Run GHG inventory (E1-6)
    3.  EnergyAssessment       -- Run energy assessment (E1-5)
    4.  TransitionPlan         -- Run transition plan (E1-1)
    5.  TargetReview           -- Run target review (E1-4)
    6.  ActionTracking         -- Run action tracking (E1-2/E1-3)
    7.  CreditReview           -- Run carbon credit review (E1-7)
    8.  PricingDisclosure      -- Run carbon pricing (E1-8)
    9.  RiskAssessment         -- Run climate risk assessment (E1-9)
    10. ReportAssembly         -- Assemble full E1 disclosure package

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


class WorkflowPhase(str, Enum):
    """Phases of the full E1 workflow."""
    MATERIALITY_CHECK = "materiality_check"
    GHG_INVENTORY = "ghg_inventory"
    ENERGY_ASSESSMENT = "energy_assessment"
    TRANSITION_PLAN = "transition_plan"
    TARGET_REVIEW = "target_review"
    ACTION_TRACKING = "action_tracking"
    CREDIT_REVIEW = "credit_review"
    PRICING_DISCLOSURE = "pricing_disclosure"
    RISK_ASSESSMENT = "risk_assessment"
    REPORT_ASSEMBLY = "report_assembly"


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


class E1DisclosureStatus(str, Enum):
    """Status of individual E1 disclosure requirement."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_MATERIAL = "not_material"
    SKIPPED = "skipped"


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


class DisclosureItem(BaseModel):
    """Status of an individual E1 disclosure requirement."""
    disclosure_id: str = Field(..., description="E1-X identifier")
    name: str = Field(default="")
    status: E1DisclosureStatus = Field(default=E1DisclosureStatus.MISSING)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_points_required: int = Field(default=0, ge=0)
    data_points_completed: int = Field(default=0, ge=0)
    warnings: List[str] = Field(default_factory=list)


class FullE1Input(BaseModel):
    """Input data model for FullE1Workflow."""
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    e1_is_material: bool = Field(default=True, description="Whether E1 is material")
    ghg_inventory_data: Dict[str, Any] = Field(default_factory=dict)
    energy_data: Dict[str, Any] = Field(default_factory=dict)
    transition_plan_data: Dict[str, Any] = Field(default_factory=dict)
    target_data: Dict[str, Any] = Field(default_factory=dict)
    actions_data: Dict[str, Any] = Field(default_factory=dict)
    credits_data: Dict[str, Any] = Field(default_factory=dict)
    pricing_data: Dict[str, Any] = Field(default_factory=dict)
    risk_data: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class FullE1Result(BaseModel):
    """Complete result from full E1 workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="full_e1")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, description="Number of phases completed")
    duration_ms: float = Field(default=0.0, description="Total duration in milliseconds")
    total_duration_seconds: float = Field(default=0.0)
    disclosures: List[DisclosureItem] = Field(default_factory=list)
    overall_completeness_pct: float = Field(default=0.0)
    disclosures_complete: int = Field(default=0)
    disclosures_partial: int = Field(default=0)
    disclosures_missing: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    total_energy_mwh: float = Field(default=0.0)
    has_transition_plan: bool = Field(default=False)
    targets_on_track: int = Field(default=0)
    high_risks: int = Field(default=0)
    e1_is_material: bool = Field(default=True)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# E1 DISCLOSURE REQUIREMENTS
# =============================================================================

E1_DISCLOSURES: List[Dict[str, Any]] = [
    {
        "id": "E1-1", "name": "Transition plan for climate change mitigation",
        "data_points": 12, "data_key": "transition_plan_data",
    },
    {
        "id": "E1-2", "name": "Policies related to climate change mitigation and adaptation",
        "data_points": 8, "data_key": "actions_data",
    },
    {
        "id": "E1-3", "name": "Actions and resources in relation to climate change",
        "data_points": 10, "data_key": "actions_data",
    },
    {
        "id": "E1-4", "name": "Targets related to climate change mitigation and adaptation",
        "data_points": 14, "data_key": "target_data",
    },
    {
        "id": "E1-5", "name": "Energy consumption and mix",
        "data_points": 8, "data_key": "energy_data",
    },
    {
        "id": "E1-6", "name": "Gross Scopes 1, 2, 3 and Total GHG emissions",
        "data_points": 18, "data_key": "ghg_inventory_data",
    },
    {
        "id": "E1-7", "name": "GHG removals and GHG mitigation projects financed through carbon credits",
        "data_points": 6, "data_key": "credits_data",
    },
    {
        "id": "E1-8", "name": "Internal carbon pricing",
        "data_points": 6, "data_key": "pricing_data",
    },
    {
        "id": "E1-9", "name": "Anticipated financial effects from material physical and transition risks",
        "data_points": 12, "data_key": "risk_data",
    },
]


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullE1Workflow:
    """
    10-phase end-to-end ESRS E1 disclosure workflow.

    Orchestrates all sub-workflows (GHG inventory, energy assessment,
    transition plan, target setting, climate actions, carbon credits,
    carbon pricing, climate risk) into a complete E1 disclosure with
    completeness tracking per disclosure requirement (E1-1 through E1-9).

    Zero-hallucination: completeness assessment uses deterministic
    data point counting.

    Example:
        >>> wf = FullE1Workflow()
        >>> inp = FullE1Input(ghg_inventory_data={...}, energy_data={...})
        >>> result = await wf.execute(inp)
        >>> assert result.overall_completeness_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize FullE1Workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._disclosures: List[DisclosureItem] = []
        self._sub_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.MATERIALITY_CHECK.value, "description": "Verify E1 materiality"},
            {"name": WorkflowPhase.GHG_INVENTORY.value, "description": "Run GHG inventory (E1-6)"},
            {"name": WorkflowPhase.ENERGY_ASSESSMENT.value, "description": "Run energy assessment (E1-5)"},
            {"name": WorkflowPhase.TRANSITION_PLAN.value, "description": "Run transition plan (E1-1)"},
            {"name": WorkflowPhase.TARGET_REVIEW.value, "description": "Run target review (E1-4)"},
            {"name": WorkflowPhase.ACTION_TRACKING.value, "description": "Run action tracking (E1-2/E1-3)"},
            {"name": WorkflowPhase.CREDIT_REVIEW.value, "description": "Run carbon credit review (E1-7)"},
            {"name": WorkflowPhase.PRICING_DISCLOSURE.value, "description": "Run carbon pricing (E1-8)"},
            {"name": WorkflowPhase.RISK_ASSESSMENT.value, "description": "Run climate risk assessment (E1-9)"},
            {"name": WorkflowPhase.REPORT_ASSEMBLY.value, "description": "Assemble full E1 disclosure"},
        ]

    def validate_inputs(self, input_data: FullE1Input) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.e1_is_material:
            issues.append("E1 is not material; full disclosure not required")
        data_sections = [
            "ghg_inventory_data", "energy_data", "transition_plan_data",
            "target_data", "actions_data", "credits_data",
            "pricing_data", "risk_data",
        ]
        empty = [s for s in data_sections if not getattr(input_data, s, {})]
        if empty:
            issues.append(f"Empty data sections: {', '.join(empty)}")
        return issues

    async def execute(
        self,
        input_data: Optional[FullE1Input] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> FullE1Result:
        """
        Execute the 10-phase full E1 workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            FullE1Result with completeness tracking for all E1 disclosures.
        """
        if input_data is None:
            input_data = FullE1Input(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting full E1 workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            # Phase 1: Materiality check
            phase_results.append(await self._phase_materiality_check(input_data))

            if not input_data.e1_is_material:
                overall_status = WorkflowStatus.COMPLETED
            else:
                # Phases 2-9: Sub-workflows
                phase_results.append(await self._phase_ghg_inventory(input_data))
                phase_results.append(await self._phase_energy_assessment(input_data))
                phase_results.append(await self._phase_transition_plan(input_data))
                phase_results.append(await self._phase_target_review(input_data))
                phase_results.append(await self._phase_action_tracking(input_data))
                phase_results.append(await self._phase_credit_review(input_data))
                phase_results.append(await self._phase_pricing_disclosure(input_data))
                phase_results.append(await self._phase_risk_assessment(input_data))

                # Phase 10: Report assembly
                phase_results.append(await self._phase_report_assembly(input_data))

                overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Full E1 workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)
        complete = sum(1 for d in self._disclosures if d.status == E1DisclosureStatus.COMPLETE)
        partial = sum(1 for d in self._disclosures if d.status == E1DisclosureStatus.PARTIAL)
        missing = sum(1 for d in self._disclosures if d.status == E1DisclosureStatus.MISSING)
        overall_pct = round(
            sum(d.completeness_pct for d in self._disclosures) / len(self._disclosures)
            if self._disclosures else 0.0, 1
        )

        result = FullE1Result(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            disclosures=self._disclosures,
            overall_completeness_pct=overall_pct,
            disclosures_complete=complete,
            disclosures_partial=partial,
            disclosures_missing=missing,
            total_emissions_tco2e=self._sub_results.get("ghg", {}).get("total_tco2e", 0.0),
            total_energy_mwh=self._sub_results.get("energy", {}).get("total_mwh", 0.0),
            has_transition_plan=bool(input_data.transition_plan_data),
            targets_on_track=self._sub_results.get("targets", {}).get("on_track", 0),
            high_risks=self._sub_results.get("risk", {}).get("high_count", 0),
            e1_is_material=input_data.e1_is_material,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Full E1 %s completed in %.2fs: %d complete, %d partial, %d missing (%.1f%%)",
            self.workflow_id, elapsed, complete, partial, missing, overall_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Materiality Check
    # -------------------------------------------------------------------------

    async def _phase_materiality_check(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Verify E1 Climate Change is material for this entity."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        outputs["e1_is_material"] = input_data.e1_is_material
        outputs["entity_name"] = input_data.entity_name

        if not input_data.e1_is_material:
            warnings.append(
                "E1 Climate Change is not material; only IRO-1 disclosure required"
            )
            # Set all disclosures to not_material
            for disc_def in E1_DISCLOSURES:
                self._disclosures.append(DisclosureItem(
                    disclosure_id=disc_def["id"],
                    name=disc_def["name"],
                    status=E1DisclosureStatus.NOT_MATERIAL,
                    completeness_pct=100.0,
                    data_points_required=0,
                    data_points_completed=0,
                ))

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 1 MaterialityCheck: E1 material=%s", input_data.e1_is_material)
        return PhaseResult(
            phase_name=WorkflowPhase.MATERIALITY_CHECK.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: GHG Inventory
    # -------------------------------------------------------------------------

    async def _phase_ghg_inventory(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process GHG inventory data for E1-6."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.ghg_inventory_data
        has_scope_1 = bool(data.get("scope_1_tco2e", 0))
        has_scope_2 = bool(data.get("scope_2_location_tco2e", 0) or data.get("scope_2_market_tco2e", 0))
        has_scope_3 = bool(data.get("scope_3_tco2e", 0))

        completed_points = sum([
            has_scope_1, has_scope_2, has_scope_3,
            bool(data.get("base_year")),
            bool(data.get("gas_disaggregation")),
            bool(data.get("consolidation_approach")),
        ])
        total_points = 18

        total_tco2e = (
            data.get("scope_1_tco2e", 0)
            + max(data.get("scope_2_location_tco2e", 0), data.get("scope_2_market_tco2e", 0))
            + data.get("scope_3_tco2e", 0)
        )
        self._sub_results["ghg"] = {"total_tco2e": total_tco2e}

        completeness = round((completed_points / total_points * 100) if total_points > 0 else 0.0, 1)
        status = (
            E1DisclosureStatus.COMPLETE if completeness >= 80
            else E1DisclosureStatus.PARTIAL if completeness > 0
            else E1DisclosureStatus.MISSING
        )

        outputs["ghg_total_tco2e"] = total_tco2e
        outputs["ghg_completeness_pct"] = completeness
        outputs["data_points_completed"] = completed_points

        if not has_scope_1:
            warnings.append("Scope 1 emissions data missing")
        if not has_scope_2:
            warnings.append("Scope 2 emissions data missing")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 2 GHGInventory: %.0f tCO2e, %.1f%% complete", total_tco2e, completeness)
        return PhaseResult(
            phase_name=WorkflowPhase.GHG_INVENTORY.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Energy Assessment
    # -------------------------------------------------------------------------

    async def _phase_energy_assessment(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process energy data for E1-5."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.energy_data
        total_mwh = data.get("total_consumption_mwh", 0)
        self._sub_results["energy"] = {"total_mwh": total_mwh}

        completed_points = sum([
            bool(data.get("total_consumption_mwh")),
            bool(data.get("fossil_mwh")),
            bool(data.get("renewable_mwh")),
            bool(data.get("renewable_share_pct") is not None),
            bool(data.get("energy_intensity")),
        ])
        total_points = 8
        completeness = round((completed_points / total_points * 100) if total_points > 0 else 0.0, 1)

        outputs["energy_total_mwh"] = total_mwh
        outputs["energy_completeness_pct"] = completeness

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 3 EnergyAssessment: %.0f MWh, %.1f%% complete", total_mwh, completeness)
        return PhaseResult(
            phase_name=WorkflowPhase.ENERGY_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Transition Plan
    # -------------------------------------------------------------------------

    async def _phase_transition_plan(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process transition plan data for E1-1."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.transition_plan_data
        has_plan = bool(data)

        completed_points = sum([
            has_plan,
            bool(data.get("scenario_used")),
            bool(data.get("target_year")),
            bool(data.get("levers")),
            bool(data.get("locked_in_emissions")),
            bool(data.get("capex_allocation")),
        ])
        total_points = 12
        completeness = round((completed_points / total_points * 100) if total_points > 0 else 0.0, 1)

        outputs["has_transition_plan"] = has_plan
        outputs["transition_plan_completeness_pct"] = completeness

        if not has_plan:
            warnings.append("No transition plan data provided")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 4 TransitionPlan: has_plan=%s, %.1f%% complete", has_plan, completeness)
        return PhaseResult(
            phase_name=WorkflowPhase.TRANSITION_PLAN.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Target Review
    # -------------------------------------------------------------------------

    async def _phase_target_review(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process target data for E1-4."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.target_data
        targets = data.get("targets", [])
        on_track = sum(1 for t in targets if t.get("progress_status") in ("on_track", "exceeded"))
        self._sub_results["targets"] = {"on_track": on_track}

        completed_points = sum([
            bool(targets),
            any(t.get("sbti_compliant") for t in targets),
            any(t.get("scope") in ("scope_1_2", "all_scopes") for t in targets),
            any(t.get("scope") in ("scope_3", "all_scopes") for t in targets),
        ])
        total_points = 14
        completeness = round((completed_points / total_points * 100) if total_points > 0 else 0.0, 1)

        outputs["targets_count"] = len(targets)
        outputs["on_track_count"] = on_track
        outputs["target_completeness_pct"] = completeness

        if not targets:
            warnings.append("No climate targets defined")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 5 TargetReview: %d targets, %d on track", len(targets), on_track)
        return PhaseResult(
            phase_name=WorkflowPhase.TARGET_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Action Tracking
    # -------------------------------------------------------------------------

    async def _phase_action_tracking(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process actions/policies data for E1-2 and E1-3."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.actions_data
        policies = data.get("policies", [])
        actions = data.get("actions", [])

        completed_points = sum([
            bool(policies),
            bool(actions),
            any(a.get("is_taxonomy_aligned") for a in actions),
            bool(data.get("total_capex_eur")),
        ])

        outputs["policies_count"] = len(policies)
        outputs["actions_count"] = len(actions)
        outputs["action_completeness_pct"] = round((completed_points / 10 * 100), 1)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 6 ActionTracking: %d policies, %d actions", len(policies), len(actions))
        return PhaseResult(
            phase_name=WorkflowPhase.ACTION_TRACKING.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 7: Credit Review
    # -------------------------------------------------------------------------

    async def _phase_credit_review(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process carbon credit data for E1-7."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.credits_data
        credits = data.get("credits", [])

        completed_points = sum([
            bool(credits),
            bool(data.get("removal_credits_tco2e")),
            bool(data.get("standards_used")),
        ])

        outputs["credits_count"] = len(credits)
        outputs["credit_completeness_pct"] = round((completed_points / 6 * 100), 1)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 7 CreditReview: %d credits", len(credits))
        return PhaseResult(
            phase_name=WorkflowPhase.CREDIT_REVIEW.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Pricing Disclosure
    # -------------------------------------------------------------------------

    async def _phase_pricing_disclosure(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process carbon pricing data for E1-8."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.pricing_data
        mechanisms = data.get("mechanisms", [])

        completed_points = sum([
            bool(mechanisms),
            bool(data.get("shadow_price_eur")),
            bool(data.get("coverage_pct")),
        ])

        outputs["mechanisms_count"] = len(mechanisms)
        outputs["pricing_completeness_pct"] = round((completed_points / 6 * 100), 1)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 8 PricingDisclosure: %d mechanisms", len(mechanisms))
        return PhaseResult(
            phase_name=WorkflowPhase.PRICING_DISCLOSURE.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 9: Risk Assessment
    # -------------------------------------------------------------------------

    async def _phase_risk_assessment(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Process climate risk data for E1-9."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        data = input_data.risk_data
        risks = data.get("risks", [])
        high_count = sum(1 for r in risks if r.get("composite_risk_score", 0) >= 16.0)
        self._sub_results["risk"] = {"high_count": high_count}

        completed_points = sum([
            bool(risks),
            bool(data.get("opportunities")),
            bool(data.get("scenarios_analyzed")),
            bool(data.get("financial_effects")),
        ])

        outputs["risks_count"] = len(risks)
        outputs["high_risk_count"] = high_count
        outputs["risk_completeness_pct"] = round((completed_points / 12 * 100), 1)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 9 RiskAssessment: %d risks, %d high", len(risks), high_count)
        return PhaseResult(
            phase_name=WorkflowPhase.RISK_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 10: Report Assembly
    # -------------------------------------------------------------------------

    async def _phase_report_assembly(
        self, input_data: FullE1Input,
    ) -> PhaseResult:
        """Assemble complete E1 disclosure from all sub-workflow results."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._disclosures = []

        for disc_def in E1_DISCLOSURES:
            data = getattr(input_data, disc_def["data_key"], {})
            data_points = disc_def["data_points"]

            # Count available data points (non-empty values)
            completed = sum(1 for v in data.values() if v) if isinstance(data, dict) else 0
            completed = min(completed, data_points)
            completeness = round((completed / data_points * 100) if data_points > 0 else 0.0, 1)

            if completeness >= 80:
                status = E1DisclosureStatus.COMPLETE
            elif completeness > 0:
                status = E1DisclosureStatus.PARTIAL
            else:
                status = E1DisclosureStatus.MISSING

            disc_warnings = []
            if status == E1DisclosureStatus.MISSING:
                disc_warnings.append(f"{disc_def['id']} has no data provided")

            self._disclosures.append(DisclosureItem(
                disclosure_id=disc_def["id"],
                name=disc_def["name"],
                status=status,
                completeness_pct=completeness,
                data_points_required=data_points,
                data_points_completed=completed,
                warnings=disc_warnings,
            ))

        complete = sum(1 for d in self._disclosures if d.status == E1DisclosureStatus.COMPLETE)
        partial = sum(1 for d in self._disclosures if d.status == E1DisclosureStatus.PARTIAL)
        missing = sum(1 for d in self._disclosures if d.status == E1DisclosureStatus.MISSING)

        outputs["disclosures_complete"] = complete
        outputs["disclosures_partial"] = partial
        outputs["disclosures_missing"] = missing
        outputs["overall_completeness_pct"] = round(
            sum(d.completeness_pct for d in self._disclosures) / len(self._disclosures)
            if self._disclosures else 0.0, 1
        )

        if missing > 0:
            warnings.append(f"{missing} E1 disclosures have no data")
        if complete == 9:
            outputs["disclosure_ready"] = True
        else:
            outputs["disclosure_ready"] = False
            warnings.append(
                f"E1 disclosure not fully ready: {complete}/9 complete, "
                f"{partial} partial, {missing} missing"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 10 ReportAssembly: %d complete, %d partial, %d missing",
            complete, partial, missing,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_ASSEMBLY.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: FullE1Result) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
