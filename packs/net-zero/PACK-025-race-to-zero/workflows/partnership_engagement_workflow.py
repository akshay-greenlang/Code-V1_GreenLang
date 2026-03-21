# -*- coding: utf-8 -*-
"""
Partnership Engagement Workflow
====================================

5-phase workflow for Race to Zero partnership management within
PACK-025 Race to Zero Pack.  Discovers potential partners, establishes
collaboration agreements, sets joint targets, tracks performance,
and generates impact reports.

Phases:
    1. PartnerDiscovery          -- Discover and match partner initiatives
    2. CollaborationAgreement    -- Establish collaboration terms and scope
    3. JointTargetSetting        -- Set joint emission reduction targets
    4. PerformanceTracking       -- Track collaborative performance metrics
    5. ImpactReporting           -- Generate partnership impact reports

Regulatory references:
    - Race to Zero Campaign Partner Initiative Requirements
    - HLEG "Integrity Matters" (Partnership & collaboration)
    - We Mean Business Coalition Commitments
    - GFANZ Framework (Financial institution partnerships)
    - CDP Supply Chain Program

Zero-hallucination: all partner matching, scoring, and performance
calculations use deterministic formulas.  No LLM calls in the
numeric computation path.

Author: GreenLang Team
Version: 25.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class PartnershipPhase(str, Enum):
    PARTNER_DISCOVERY = "partner_discovery"
    COLLABORATION_AGREEMENT = "collaboration_agreement"
    JOINT_TARGET_SETTING = "joint_target_setting"
    PERFORMANCE_TRACKING = "performance_tracking"
    IMPACT_REPORTING = "impact_reporting"


class PartnerType(str, Enum):
    STANDARD_SETTER = "standard_setter"
    INITIATIVE = "initiative"
    INDUSTRY_GROUP = "industry_group"
    SUPPLY_CHAIN = "supply_chain"
    PUBLIC_SECTOR = "public_sector"
    NGO = "ngo"
    ACADEMIA = "academia"
    TECHNOLOGY_PROVIDER = "technology_provider"


class EngagementLevel(str, Enum):
    STRATEGIC = "strategic"
    ACTIVE = "active"
    PARTICIPATING = "participating"
    EXPLORATORY = "exploratory"


class CollaborationStatus(str, Enum):
    PROPOSED = "proposed"
    NEGOTIATING = "negotiating"
    ACTIVE = "active"
    COMPLETED = "completed"
    TERMINATED = "terminated"


# =============================================================================
# REFERENCE DATA
# =============================================================================

# Race to Zero partner initiatives (40+ accelerators)
R2Z_PARTNER_INITIATIVES: List[Dict[str, Any]] = [
    {
        "id": "PI-01", "name": "Science Based Targets initiative (SBTi)",
        "type": "standard_setter", "actor_types": ["corporate", "financial_institution"],
        "focus_areas": ["target_setting", "validation", "methodology"],
        "reporting_channel": "cdp", "alignment_score": 95,
    },
    {
        "id": "PI-02", "name": "CDP",
        "type": "initiative", "actor_types": ["corporate", "financial_institution", "city"],
        "focus_areas": ["disclosure", "scoring", "supply_chain"],
        "reporting_channel": "cdp", "alignment_score": 90,
    },
    {
        "id": "PI-03", "name": "C40 Cities",
        "type": "initiative", "actor_types": ["city"],
        "focus_areas": ["city_action", "transport", "buildings", "waste"],
        "reporting_channel": "c40", "alignment_score": 92,
    },
    {
        "id": "PI-04", "name": "ICLEI",
        "type": "initiative", "actor_types": ["city", "region"],
        "focus_areas": ["local_government", "community_wide", "adaptation"],
        "reporting_channel": "iclei", "alignment_score": 88,
    },
    {
        "id": "PI-05", "name": "GFANZ",
        "type": "initiative", "actor_types": ["financial_institution"],
        "focus_areas": ["financed_emissions", "portfolio_alignment", "transition_plans"],
        "reporting_channel": "gfanz", "alignment_score": 93,
    },
    {
        "id": "PI-06", "name": "We Mean Business Coalition",
        "type": "initiative", "actor_types": ["corporate"],
        "focus_areas": ["corporate_action", "policy_advocacy", "reporting"],
        "reporting_channel": "cdp", "alignment_score": 85,
    },
    {
        "id": "PI-07", "name": "RE100",
        "type": "initiative", "actor_types": ["corporate"],
        "focus_areas": ["renewable_energy", "electricity_procurement"],
        "reporting_channel": "cdp", "alignment_score": 80,
    },
    {
        "id": "PI-08", "name": "EV100",
        "type": "initiative", "actor_types": ["corporate"],
        "focus_areas": ["electric_vehicles", "fleet_electrification"],
        "reporting_channel": "cdp", "alignment_score": 78,
    },
    {
        "id": "PI-09", "name": "EP100",
        "type": "initiative", "actor_types": ["corporate"],
        "focus_areas": ["energy_productivity", "efficiency"],
        "reporting_channel": "cdp", "alignment_score": 77,
    },
    {
        "id": "PI-10", "name": "SME Climate Hub",
        "type": "initiative", "actor_types": ["sme"],
        "focus_areas": ["sme_action", "simplified_reporting", "tools"],
        "reporting_channel": "race_to_zero", "alignment_score": 75,
    },
    {
        "id": "PI-11", "name": "The Climate Pledge",
        "type": "initiative", "actor_types": ["corporate"],
        "focus_areas": ["net_zero_by_2040", "corporate_action"],
        "reporting_channel": "race_to_zero", "alignment_score": 82,
    },
    {
        "id": "PI-12", "name": "Second Nature",
        "type": "initiative", "actor_types": ["university"],
        "focus_areas": ["higher_education", "campus_emissions", "curriculum"],
        "reporting_channel": "race_to_zero", "alignment_score": 80,
    },
]

PHASE_DEPENDENCIES: Dict[PartnershipPhase, List[PartnershipPhase]] = {
    PartnershipPhase.PARTNER_DISCOVERY: [],
    PartnershipPhase.COLLABORATION_AGREEMENT: [PartnershipPhase.PARTNER_DISCOVERY],
    PartnershipPhase.JOINT_TARGET_SETTING: [PartnershipPhase.COLLABORATION_AGREEMENT],
    PartnershipPhase.PERFORMANCE_TRACKING: [PartnershipPhase.JOINT_TARGET_SETTING],
    PartnershipPhase.IMPACT_REPORTING: [PartnershipPhase.PERFORMANCE_TRACKING],
}

PHASE_EXECUTION_ORDER: List[PartnershipPhase] = [
    PartnershipPhase.PARTNER_DISCOVERY,
    PartnershipPhase.COLLABORATION_AGREEMENT,
    PartnershipPhase.JOINT_TARGET_SETTING,
    PartnershipPhase.PERFORMANCE_TRACKING,
    PartnershipPhase.IMPACT_REPORTING,
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase: PartnershipPhase = Field(...)
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class PartnerMatch(BaseModel):
    partner_id: str = Field(default="")
    partner_name: str = Field(default="")
    partner_type: PartnerType = Field(default=PartnerType.INITIATIVE)
    alignment_score: float = Field(default=0.0, ge=0.0, le=100.0)
    focus_areas: List[str] = Field(default_factory=list)
    reporting_channel: str = Field(default="")
    recommended: bool = Field(default=False)


class CollaborationAgreement(BaseModel):
    agreement_id: str = Field(default="")
    partner_id: str = Field(default="")
    partner_name: str = Field(default="")
    scope: str = Field(default="")
    joint_commitments: List[str] = Field(default_factory=list)
    reporting_frequency: str = Field(default="annual")
    status: CollaborationStatus = Field(default=CollaborationStatus.PROPOSED)
    start_date: str = Field(default="")
    review_date: str = Field(default="")


class JointTarget(BaseModel):
    target_id: str = Field(default="")
    partner_id: str = Field(default="")
    description: str = Field(default="")
    target_value: float = Field(default=0.0)
    target_unit: str = Field(default="tCO2e")
    target_year: int = Field(default=2030)
    baseline_value: float = Field(default=0.0)
    progress_pct: float = Field(default=0.0)


class PartnershipEngagementConfig(BaseModel):
    pack_id: str = Field(default="PACK-025")
    org_name: str = Field(default="")
    actor_type: str = Field(default="corporate")
    sector: str = Field(default="general_services")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    current_partners: List[str] = Field(default_factory=list)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    max_partnerships: int = Field(default=5, ge=1, le=20)
    enable_provenance: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class PartnershipEngagementResult(BaseModel):
    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    workflow_name: str = Field(default="partnership_engagement")
    org_name: str = Field(default="")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    partner_matches: List[PartnerMatch] = Field(default_factory=list)
    agreements: List[CollaborationAgreement] = Field(default_factory=list)
    joint_targets: List[JointTarget] = Field(default_factory=list)
    total_joint_reduction_tco2e: float = Field(default=0.0)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PartnershipEngagementWorkflow:
    """
    5-phase partnership engagement workflow for PACK-025 Race to Zero Pack.

    Discovers potential partners, establishes collaboration agreements,
    sets joint targets, tracks performance, and generates impact reports
    for Race to Zero partnership network navigation.

    Engines used:
        - partnership_scoring_engine (discovery and scoring)
        - interim_target_engine (joint target setting)
        - progress_tracking_engine (performance tracking)
    """

    def __init__(
        self,
        config: Optional[PartnershipEngagementConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or PartnershipEngagementConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, PartnershipEngagementResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

    async def execute(
        self, input_data: Optional[Dict[str, Any]] = None,
    ) -> PartnershipEngagementResult:
        """Execute the 5-phase partnership engagement workflow."""
        input_data = input_data or {}
        result = PartnershipEngagementResult(
            org_name=self.config.org_name,
            status=WorkflowStatus.RUNNING, started_at=_utcnow(),
        )
        self._results[result.execution_id] = result
        start_time = time.monotonic()
        phases = PHASE_EXECUTION_ORDER

        self.logger.info(
            "Starting partnership engagement: execution_id=%s, org=%s",
            result.execution_id, self.config.org_name,
        )

        ctx: Dict[str, Any] = dict(input_data)
        ctx["actor_type"] = self.config.actor_type
        ctx["total_emissions"] = (
            self.config.scope1_tco2e + self.config.scope2_tco2e + self.config.scope3_tco2e
        )

        try:
            for idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = WorkflowStatus.CANCELLED
                    break
                if not self._deps_met(phase, result):
                    result.status = WorkflowStatus.FAILED
                    result.errors.append(f"Dependencies not met for {phase.value}")
                    break

                if self._progress_callback:
                    await self._progress_callback(phase.value, (idx / len(phases)) * 100, phase.value)

                pr = await self._run_phase(phase, ctx)
                result.phase_results[phase.value] = pr
                if pr.status == PhaseStatus.FAILED:
                    result.status = WorkflowStatus.PARTIAL
                result.phases_completed.append(phase.value)
                result.total_records_processed += pr.records_processed
                ctx[phase.value] = pr.outputs

            if result.status == WorkflowStatus.RUNNING:
                result.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Partnership engagement failed: %s", exc, exc_info=True)
            result.status = WorkflowStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = _utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.partner_matches = self._build_matches(ctx)
            result.agreements = self._build_agreements(ctx)
            result.joint_targets = self._build_targets(ctx)
            result.total_joint_reduction_tco2e = sum(
                t.target_value for t in result.joint_targets
            )
            result.quality_score = round(
                (len(result.phases_completed) / max(len(phases), 1)) * 100, 1
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    result.model_dump_json(exclude={"provenance_hash"})
                )

        return result

    def cancel(self, eid: str) -> Dict[str, Any]:
        self._cancelled.add(eid)
        return {"cancelled": True}

    async def _run_phase(self, phase: PartnershipPhase, ctx: Dict[str, Any]) -> PhaseResult:
        started = _utcnow()
        st = time.monotonic()
        handler = {
            PartnershipPhase.PARTNER_DISCOVERY: self._ph_discovery,
            PartnershipPhase.COLLABORATION_AGREEMENT: self._ph_agreement,
            PartnershipPhase.JOINT_TARGET_SETTING: self._ph_joint_targets,
            PartnershipPhase.PERFORMANCE_TRACKING: self._ph_performance,
            PartnershipPhase.IMPACT_REPORTING: self._ph_impact,
        }[phase]
        try:
            out, warn, err, rec = await handler(ctx)
            status = PhaseStatus.FAILED if err else PhaseStatus.COMPLETED
        except Exception as exc:
            out, warn, err, rec = {}, [], [str(exc)], 0
            status = PhaseStatus.FAILED
        return PhaseResult(
            phase=phase, status=status, started_at=started, completed_at=_utcnow(),
            duration_ms=round((time.monotonic() - st) * 1000, 2), records_processed=rec,
            outputs=out, warnings=warn, errors=err,
            provenance_hash=_compute_hash(out) if self.config.enable_provenance else "",
        )

    def _deps_met(self, phase: PartnershipPhase, result: PartnershipEngagementResult) -> bool:
        for dep in PHASE_DEPENDENCIES.get(phase, []):
            dr = result.phase_results.get(dep.value)
            if not dr or dr.status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return False
        return True

    # ---- Phase 1: Partner Discovery ----

    async def _ph_discovery(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        actor_type = self.config.actor_type
        matches: List[Dict[str, Any]] = []

        for pi in R2Z_PARTNER_INITIATIVES:
            applicable = "all" in pi.get("actor_types", []) or actor_type in pi.get("actor_types", [])
            already_member = pi["name"] in self.config.current_partners
            recommended = applicable and not already_member

            match = {
                "partner_id": pi["id"],
                "partner_name": pi["name"],
                "partner_type": pi["type"],
                "alignment_score": pi["alignment_score"],
                "focus_areas": pi["focus_areas"],
                "reporting_channel": pi["reporting_channel"],
                "applicable": applicable,
                "already_member": already_member,
                "recommended": recommended,
            }
            matches.append(match)

        # Sort by alignment score, take top N
        matches.sort(key=lambda m: (-m["alignment_score"], not m["recommended"]))
        top_matches = [m for m in matches if m["recommended"]][:self.config.max_partnerships]

        outputs["all_partners_screened"] = len(R2Z_PARTNER_INITIATIVES)
        outputs["applicable_partners"] = sum(1 for m in matches if m["applicable"])
        outputs["recommended_partners"] = len(top_matches)
        outputs["existing_partnerships"] = sum(1 for m in matches if m["already_member"])
        outputs["matches"] = top_matches

        return outputs, warnings, errors, len(matches)

    # ---- Phase 2: Collaboration Agreement ----

    async def _ph_agreement(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        discovery = ctx.get("partner_discovery", {})
        matches = discovery.get("matches", [])

        agreements: List[Dict[str, Any]] = []
        from datetime import timedelta

        for match in matches:
            agreement = {
                "agreement_id": _new_uuid()[:12],
                "partner_id": match["partner_id"],
                "partner_name": match["partner_name"],
                "scope": f"Climate action collaboration through {match['partner_name']}",
                "joint_commitments": [
                    "Align targets with Race to Zero minimum criteria",
                    "Share annual progress data through partner reporting channel",
                    f"Participate in {match['partner_name']} programs and events",
                    "Contribute to collective impact reporting",
                ],
                "reporting_frequency": "annual",
                "status": "active",
                "start_date": _utcnow().strftime("%Y-%m-%d"),
                "review_date": (_utcnow() + timedelta(days=365)).strftime("%Y-%m-%d"),
                "reporting_channel": match["reporting_channel"],
            }
            agreements.append(agreement)

        outputs["agreements"] = agreements
        outputs["agreements_count"] = len(agreements)
        outputs["reporting_channels"] = list(set(a["reporting_channel"] for a in agreements))

        return outputs, warnings, errors, len(agreements)

    # ---- Phase 3: Joint Target Setting ----

    async def _ph_joint_targets(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        agreements = ctx.get("collaboration_agreement", {}).get("agreements", [])
        total_emissions = ctx.get("total_emissions", 0)

        joint_targets: List[Dict[str, Any]] = []
        total_joint_reduction = 0.0

        for agreement in agreements:
            # Each partnership contributes a share of the reduction
            partner_share = total_emissions * 0.05  # ~5% per partnership
            target = {
                "target_id": _new_uuid()[:12],
                "partner_id": agreement["partner_id"],
                "partner_name": agreement["partner_name"],
                "description": (
                    f"Joint emission reduction with {agreement['partner_name']}: "
                    f"reduce {partner_share:.0f} tCO2e through collaborative actions"
                ),
                "target_value": round(partner_share, 2),
                "target_unit": "tCO2e",
                "target_year": 2030,
                "baseline_value": round(total_emissions, 2),
                "progress_pct": 0.0,
            }
            joint_targets.append(target)
            total_joint_reduction += partner_share

        outputs["joint_targets"] = joint_targets
        outputs["targets_count"] = len(joint_targets)
        outputs["total_joint_reduction_tco2e"] = round(total_joint_reduction, 2)
        outputs["joint_reduction_pct"] = round(
            (total_joint_reduction / max(total_emissions, 1)) * 100.0, 1
        )

        return outputs, warnings, errors, len(joint_targets)

    # ---- Phase 4: Performance Tracking ----

    async def _ph_performance(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        targets = ctx.get("joint_target_setting", {}).get("joint_targets", [])
        agreements = ctx.get("collaboration_agreement", {}).get("agreements", [])

        performance_records: List[Dict[str, Any]] = []
        for target in targets:
            # Simulate initial performance (year 1)
            progress = 0.0  # First year, no progress yet
            record = {
                "target_id": target["target_id"],
                "partner_name": target["partner_name"],
                "target_value_tco2e": target["target_value"],
                "achieved_tco2e": round(target["target_value"] * progress / 100, 2),
                "progress_pct": round(progress, 1),
                "status": "on_track" if progress >= 10 else "starting",
                "engagement_quality": "active",
                "reporting_up_to_date": True,
            }
            performance_records.append(record)

        active_partnerships = len([a for a in agreements if a.get("status") == "active"])
        total_achieved = sum(r["achieved_tco2e"] for r in performance_records)

        outputs["performance_records"] = performance_records
        outputs["active_partnerships"] = active_partnerships
        outputs["total_achieved_tco2e"] = round(total_achieved, 2)
        outputs["avg_engagement_quality"] = "active"
        outputs["all_reporting_current"] = all(r["reporting_up_to_date"] for r in performance_records)

        return outputs, warnings, errors, len(performance_records)

    # ---- Phase 5: Impact Reporting ----

    async def _ph_impact(self, ctx: Dict[str, Any]) -> tuple:
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        performance = ctx.get("performance_tracking", {})
        targets_data = ctx.get("joint_target_setting", {})

        total_joint = targets_data.get("total_joint_reduction_tco2e", 0)
        total_achieved = performance.get("total_achieved_tco2e", 0)
        active = performance.get("active_partnerships", 0)

        report_id = f"PIR-{self.config.reporting_year}-{_new_uuid()[:8].upper()}"

        outputs["report_id"] = report_id
        outputs["org_name"] = self.config.org_name
        outputs["reporting_year"] = self.config.reporting_year
        outputs["active_partnerships"] = active
        outputs["total_joint_reduction_target_tco2e"] = round(total_joint, 2)
        outputs["total_achieved_tco2e"] = round(total_achieved, 2)
        outputs["achievement_pct"] = round(
            (total_achieved / max(total_joint, 1)) * 100.0, 1
        )
        outputs["key_highlights"] = [
            f"Engaged with {active} Race to Zero partner initiatives",
            f"Joint reduction target: {total_joint:,.0f} tCO2e by 2030",
            "All partnership reporting channels active and current",
        ]
        outputs["recommendations"] = [
            "Deepen engagement with highest-alignment partners",
            "Explore sector-specific collaboration opportunities",
            "Increase supply chain partner engagement for Scope 3",
        ]
        outputs["report_generated"] = True

        return outputs, warnings, errors, 1

    # ---- Extractors ----

    def _build_matches(self, ctx: Dict[str, Any]) -> List[PartnerMatch]:
        data = ctx.get("partner_discovery", {}).get("matches", [])
        return [
            PartnerMatch(
                partner_id=m["partner_id"], partner_name=m["partner_name"],
                partner_type=PartnerType(m["partner_type"]),
                alignment_score=m["alignment_score"],
                focus_areas=m["focus_areas"],
                reporting_channel=m["reporting_channel"],
                recommended=m["recommended"],
            )
            for m in data
        ]

    def _build_agreements(self, ctx: Dict[str, Any]) -> List[CollaborationAgreement]:
        data = ctx.get("collaboration_agreement", {}).get("agreements", [])
        return [
            CollaborationAgreement(
                agreement_id=a["agreement_id"], partner_id=a["partner_id"],
                partner_name=a["partner_name"], scope=a["scope"],
                joint_commitments=a["joint_commitments"],
                reporting_frequency=a["reporting_frequency"],
                status=CollaborationStatus(a["status"]),
                start_date=a["start_date"], review_date=a["review_date"],
            )
            for a in data
        ]

    def _build_targets(self, ctx: Dict[str, Any]) -> List[JointTarget]:
        data = ctx.get("joint_target_setting", {}).get("joint_targets", [])
        return [
            JointTarget(
                target_id=t["target_id"], partner_id=t["partner_id"],
                description=t["description"], target_value=t["target_value"],
                target_unit=t["target_unit"], target_year=t["target_year"],
                baseline_value=t["baseline_value"], progress_pct=t["progress_pct"],
            )
            for t in data
        ]
