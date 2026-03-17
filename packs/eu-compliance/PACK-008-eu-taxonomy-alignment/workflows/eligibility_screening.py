# -*- coding: utf-8 -*-
"""
Eligibility Screening Workflow
=================================

Four-phase workflow for screening an organization's economic activities against
the EU Taxonomy Regulation (EU) 2020/852 to determine taxonomy eligibility.

This workflow enables:
- Comprehensive inventory of all economic activities
- NACE code mapping to ~240 taxonomy-eligible activities
- Per-objective eligibility assessment across 6 environmental objectives
- Eligibility matrix report generation with revenue-weighted ratios

Phases:
    1. Activity Inventory - Collect all economic activities, map to NACE codes
    2. NACE Mapping - Map NACE codes to taxonomy economic activities (~240)
    3. Eligibility Assessment - Screen each activity for eligibility per objective
    4. Eligibility Report - Generate eligibility matrix report

Regulatory Context:
    EU Taxonomy Regulation Article 1 defines taxonomy eligibility as the first
    gate: an activity must appear in the Delegated Acts (Climate DA or
    Environmental DA) to be considered taxonomy-eligible. Eligibility does NOT
    imply alignment -- it only indicates the activity is described in the taxonomy.

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Phase(str, Enum):
    """Workflow phases."""
    ACTIVITY_INVENTORY = "activity_inventory"
    NACE_MAPPING = "nace_mapping"
    ELIGIBILITY_ASSESSMENT = "eligibility_assessment"
    ELIGIBILITY_REPORT = "eligibility_report"


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""
    CCM = "climate_change_mitigation"
    CCA = "climate_change_adaptation"
    WTR = "water_and_marine_resources"
    CE = "circular_economy"
    PPC = "pollution_prevention_and_control"
    BIO = "biodiversity_and_ecosystems"


# =============================================================================
# DATA MODELS
# =============================================================================


class EligibilityScreeningConfig(BaseModel):
    """Configuration for eligibility screening workflow."""
    organization_id: Optional[str] = Field(None, description="Organization identifier")
    reporting_period: str = Field(default="2025", description="Reporting period (year)")
    include_all_objectives: bool = Field(default=True, description="Screen against all 6 objectives")
    objectives_in_scope: List[str] = Field(
        default_factory=lambda: [obj.value for obj in EnvironmentalObjective],
        description="Environmental objectives to screen against",
    )
    revenue_weighted: bool = Field(default=True, description="Calculate revenue-weighted eligibility")
    nace_version: str = Field(default="rev2", description="NACE classification version")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase: Phase = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution duration")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique execution ID")
    config: EligibilityScreeningConfig = Field(default_factory=EligibilityScreeningConfig)
    phase_results: List[PhaseResult] = Field(default_factory=list, description="Completed phase results")
    state: Dict[str, Any] = Field(default_factory=dict, description="Shared state data")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Workflow start time")

    class Config:
        arbitrary_types_allowed = True


class WorkflowResult(BaseModel):
    """Complete result from the eligibility screening workflow."""
    workflow_name: str = Field(default="eligibility_screening", description="Workflow identifier")
    phases: List[PhaseResult] = Field(default_factory=list, description="All phase results")
    overall_status: PhaseStatus = Field(..., description="Overall workflow status")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total execution time")
    provenance_hash: str = Field(default="", description="Workflow-level provenance hash")
    execution_id: str = Field(..., description="Execution identifier")
    total_activities: int = Field(default=0, ge=0, description="Total activities inventoried")
    eligible_count: int = Field(default=0, ge=0, description="Taxonomy-eligible activities")
    not_eligible_count: int = Field(default=0, ge=0, description="Non-eligible activities")
    eligible_by_objective: Dict[str, int] = Field(
        default_factory=dict, description="Eligible count per environmental objective"
    )
    eligibility_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Eligible / total ratio")
    revenue_weighted_eligibility: Optional[float] = Field(
        None, description="Revenue-weighted eligibility ratio"
    )
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")


# =============================================================================
# SAMPLE TAXONOMY ACTIVITY CATALOG (SUBSET)
# =============================================================================

_TAXONOMY_ACTIVITY_CATALOG: List[Dict[str, Any]] = [
    {"nace": "D35.11", "activity": "Electricity generation using solar photovoltaic technology", "objectives": ["CCM"]},
    {"nace": "D35.11", "activity": "Electricity generation from wind power", "objectives": ["CCM"]},
    {"nace": "D35.11", "activity": "Electricity generation from hydropower", "objectives": ["CCM", "WTR"]},
    {"nace": "D35.30", "activity": "District heating/cooling distribution", "objectives": ["CCM"]},
    {"nace": "F41.10", "activity": "Construction of new buildings", "objectives": ["CCM", "CCA"]},
    {"nace": "F41.20", "activity": "Renovation of existing buildings", "objectives": ["CCM", "CE"]},
    {"nace": "F42.11", "activity": "Infrastructure for rail transport", "objectives": ["CCM"]},
    {"nace": "C20.11", "activity": "Manufacture of hydrogen", "objectives": ["CCM"]},
    {"nace": "C23.51", "activity": "Manufacture of cement", "objectives": ["CCM"]},
    {"nace": "C24.10", "activity": "Manufacture of iron and steel", "objectives": ["CCM", "CE"]},
    {"nace": "C29.10", "activity": "Manufacture of motor vehicles", "objectives": ["CCM", "PPC"]},
    {"nace": "H49.10", "activity": "Passenger interurban rail transport", "objectives": ["CCM"]},
    {"nace": "H49.32", "activity": "Urban and suburban passenger land transport", "objectives": ["CCM", "PPC"]},
    {"nace": "H50.10", "activity": "Sea and coastal freight water transport", "objectives": ["CCM", "PPC"]},
    {"nace": "J61.10", "activity": "Data processing, hosting and related activities", "objectives": ["CCM"]},
    {"nace": "A02.10", "activity": "Afforestation", "objectives": ["CCM", "BIO"]},
    {"nace": "A02.10", "activity": "Restoration of forests", "objectives": ["CCM", "BIO"]},
    {"nace": "E36.00", "activity": "Water supply", "objectives": ["WTR"]},
    {"nace": "E38.11", "activity": "Collection and transport of non-hazardous waste", "objectives": ["CE"]},
    {"nace": "E38.21", "activity": "Material recovery from non-hazardous waste", "objectives": ["CE"]},
    {"nace": "M71.12", "activity": "Engineering activities and related technical consultancy", "objectives": ["CCM", "CCA"]},
    {"nace": "K64.19", "activity": "Other monetary intermediation", "objectives": ["CCM", "CCA"]},
    {"nace": "K65.12", "activity": "Non-life insurance", "objectives": ["CCA"]},
]


# =============================================================================
# ELIGIBILITY SCREENING WORKFLOW
# =============================================================================


class EligibilityScreeningWorkflow:
    """
    Four-phase eligibility screening workflow.

    Screens an organization's economic activities against the EU Taxonomy
    to determine which activities are taxonomy-eligible:
    - Inventory all economic activities with NACE codes and revenues
    - Map NACE codes to the ~240 taxonomy economic activities
    - Assess eligibility per environmental objective
    - Generate an eligibility matrix report

    Example:
        >>> config = EligibilityScreeningConfig(
        ...     organization_id="ORG-001",
        ...     reporting_period="2025",
        ... )
        >>> workflow = EligibilityScreeningWorkflow(config)
        >>> result = await workflow.run(WorkflowContext(config=config))
        >>> assert result.overall_status == PhaseStatus.COMPLETED
        >>> assert result.total_activities > 0
    """

    def __init__(self, config: Optional[EligibilityScreeningConfig] = None) -> None:
        """Initialize the eligibility screening workflow."""
        self.config = config or EligibilityScreeningConfig()
        self.logger = logging.getLogger(f"{__name__}.EligibilityScreeningWorkflow")

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """
        Execute the full 4-phase eligibility screening workflow.

        Args:
            context: Workflow context with configuration and initial state.

        Returns:
            WorkflowResult with eligibility counts, ratios, and per-objective breakdown.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting eligibility screening workflow execution_id=%s period=%s",
            context.execution_id,
            self.config.reporting_period,
        )

        context.config = self.config

        phase_handlers = [
            (Phase.ACTIVITY_INVENTORY, self._phase_1_activity_inventory),
            (Phase.NACE_MAPPING, self._phase_2_nace_mapping),
            (Phase.ELIGIBILITY_ASSESSMENT, self._phase_3_eligibility_assessment),
            (Phase.ELIGIBILITY_REPORT, self._phase_4_eligibility_report),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase.value)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_result.timestamp = datetime.utcnow()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase.value, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    data={"error": str(exc)},
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    provenance_hash=self._hash({"error": str(exc)}),
                    timestamp=datetime.utcnow(),
                )

            context.phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                self.logger.error("Phase '%s' failed; halting workflow.", phase.value)
                break

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        # Extract final outputs
        total_activities = context.state.get("total_activities", 0)
        eligible_count = context.state.get("eligible_count", 0)
        not_eligible_count = total_activities - eligible_count
        eligible_by_objective = context.state.get("eligible_by_objective", {})
        eligibility_ratio = eligible_count / max(total_activities, 1)
        revenue_weighted = context.state.get("revenue_weighted_eligibility")

        provenance = self._hash({
            "execution_id": context.execution_id,
            "phases": [p.provenance_hash for p in context.phase_results],
            "total_activities": total_activities,
            "eligible_count": eligible_count,
        })

        self.logger.info(
            "Eligibility screening finished execution_id=%s status=%s "
            "total=%d eligible=%d ratio=%.1f%%",
            context.execution_id,
            overall_status.value,
            total_activities,
            eligible_count,
            eligibility_ratio * 100,
        )

        return WorkflowResult(
            phases=context.phase_results,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            provenance_hash=provenance,
            execution_id=context.execution_id,
            total_activities=total_activities,
            eligible_count=eligible_count,
            not_eligible_count=not_eligible_count,
            eligible_by_objective=eligible_by_objective,
            eligibility_ratio=round(eligibility_ratio, 4),
            revenue_weighted_eligibility=revenue_weighted,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Activity Inventory
    # -------------------------------------------------------------------------

    async def _phase_1_activity_inventory(self, context: WorkflowContext) -> PhaseResult:
        """
        Collect all economic activities from the organization, map to NACE codes.

        Data collected per activity:
        - Activity description and internal identifier
        - NACE code (Rev. 2) classification
        - Annual revenue attributable to this activity
        - Business unit and geographic region
        - CapEx and OpEx attributable to this activity
        """
        phase = Phase.ACTIVITY_INVENTORY

        self.logger.info(
            "Inventorying economic activities for organization=%s",
            self.config.organization_id,
        )

        await asyncio.sleep(0.05)

        # Simulate activity inventory (replace with actual ERP/financial system queries)
        nace_codes = [
            "D35.11", "F41.10", "F41.20", "C29.10", "H49.10",
            "J61.10", "A02.10", "E38.11", "C24.10", "C23.51",
            "H49.32", "M71.12", "K64.19", "G47.11", "N82.11",
            "L68.20", "I55.10", "Q86.10",
        ]

        activity_count = random.randint(10, 30)
        activities = []
        total_revenue = 0.0

        for i in range(activity_count):
            nace = random.choice(nace_codes)
            revenue = round(random.uniform(500_000, 50_000_000), 2)
            capex = round(revenue * random.uniform(0.05, 0.25), 2)
            opex = round(revenue * random.uniform(0.02, 0.10), 2)
            total_revenue += revenue

            activities.append({
                "activity_id": f"ACT-{uuid.uuid4().hex[:8]}",
                "description": f"Activity {i + 1}",
                "nace_code": nace,
                "revenue": revenue,
                "capex": capex,
                "opex": opex,
                "business_unit": random.choice(["Energy", "Manufacturing", "Services", "Real Estate"]),
                "region": random.choice(["EU", "APAC", "Americas"]),
            })

        context.state["activities"] = activities
        context.state["total_activities"] = activity_count
        context.state["total_revenue"] = round(total_revenue, 2)

        provenance = self._hash({
            "phase": phase.value,
            "activity_count": activity_count,
            "total_revenue": total_revenue,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "activity_count": activity_count,
                "total_revenue": round(total_revenue, 2),
                "unique_nace_codes": len(set(a["nace_code"] for a in activities)),
                "business_units": list(set(a["business_unit"] for a in activities)),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: NACE Mapping
    # -------------------------------------------------------------------------

    async def _phase_2_nace_mapping(self, context: WorkflowContext) -> PhaseResult:
        """
        Map NACE codes to the ~240 taxonomy economic activities.

        For each NACE code found in the activity inventory, look up matching
        taxonomy activities from the Delegated Acts. A single NACE code may
        map to multiple taxonomy activities (e.g., D35.11 maps to solar,
        wind, hydro, geothermal generation).
        """
        phase = Phase.NACE_MAPPING
        activities = context.state.get("activities", [])

        self.logger.info("Mapping %d activities to taxonomy catalog", len(activities))

        mapped_activities = []
        unmapped_activities = []

        for activity in activities:
            nace = activity["nace_code"]
            matching = [
                entry for entry in _TAXONOMY_ACTIVITY_CATALOG if entry["nace"] == nace
            ]

            if matching:
                for match in matching:
                    mapped_activities.append({
                        **activity,
                        "taxonomy_activity": match["activity"],
                        "eligible_objectives": match["objectives"],
                        "mapping_confidence": "HIGH",
                    })
            else:
                unmapped_activities.append({
                    **activity,
                    "taxonomy_activity": None,
                    "eligible_objectives": [],
                    "mapping_confidence": "NOT_FOUND",
                })

        context.state["mapped_activities"] = mapped_activities
        context.state["unmapped_activities"] = unmapped_activities

        provenance = self._hash({
            "phase": phase.value,
            "mapped_count": len(mapped_activities),
            "unmapped_count": len(unmapped_activities),
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "mapped_count": len(mapped_activities),
                "unmapped_count": len(unmapped_activities),
                "catalog_activities_referenced": len(set(
                    m["taxonomy_activity"] for m in mapped_activities if m["taxonomy_activity"]
                )),
                "coverage_ratio": round(
                    len(mapped_activities) / max(len(mapped_activities) + len(unmapped_activities), 1), 3
                ),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Eligibility Assessment
    # -------------------------------------------------------------------------

    async def _phase_3_eligibility_assessment(self, context: WorkflowContext) -> PhaseResult:
        """
        Screen each activity for eligibility per environmental objective.

        An activity is taxonomy-eligible if it appears in any Delegated Act
        under any of the 6 environmental objectives. This phase:
        - Marks each mapped activity as ELIGIBLE or NOT_ELIGIBLE
        - Records which objectives the activity is eligible under
        - Calculates per-objective eligibility counts
        - Prevents double-counting across objectives
        """
        phase = Phase.ELIGIBILITY_ASSESSMENT
        mapped = context.state.get("mapped_activities", [])
        unmapped = context.state.get("unmapped_activities", [])
        objectives_in_scope = self.config.objectives_in_scope

        self.logger.info(
            "Assessing eligibility for %d mapped + %d unmapped activities",
            len(mapped), len(unmapped),
        )

        objective_short_map = {
            "climate_change_mitigation": "CCM",
            "climate_change_adaptation": "CCA",
            "water_and_marine_resources": "WTR",
            "circular_economy": "CE",
            "pollution_prevention_and_control": "PPC",
            "biodiversity_and_ecosystems": "BIO",
        }

        eligible_activities = []
        eligible_by_objective: Dict[str, int] = {obj: 0 for obj in EnvironmentalObjective.__members__}
        unique_eligible_ids = set()

        for activity in mapped:
            obj_codes = activity.get("eligible_objectives", [])
            is_eligible = False

            for obj_full, obj_short in objective_short_map.items():
                if obj_full in objectives_in_scope and obj_short in obj_codes:
                    eligible_by_objective[obj_short] = eligible_by_objective.get(obj_short, 0) + 1
                    is_eligible = True

            if is_eligible:
                activity["eligibility_status"] = "ELIGIBLE"
                unique_eligible_ids.add(activity["activity_id"])
            else:
                activity["eligibility_status"] = "NOT_ELIGIBLE"

            eligible_activities.append(activity)

        # Mark unmapped as not eligible
        for activity in unmapped:
            activity["eligibility_status"] = "NOT_ELIGIBLE"
            eligible_activities.append(activity)

        eligible_count = len(unique_eligible_ids)
        context.state["eligible_activities"] = eligible_activities
        context.state["eligible_count"] = eligible_count
        context.state["eligible_by_objective"] = eligible_by_objective

        provenance = self._hash({
            "phase": phase.value,
            "eligible_count": eligible_count,
            "eligible_by_objective": eligible_by_objective,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "eligible_count": eligible_count,
                "not_eligible_count": len(eligible_activities) - eligible_count,
                "eligible_by_objective": eligible_by_objective,
                "objectives_screened": len(objectives_in_scope),
            },
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Eligibility Report
    # -------------------------------------------------------------------------

    async def _phase_4_eligibility_report(self, context: WorkflowContext) -> PhaseResult:
        """
        Generate eligibility matrix report.

        Report contents:
        - Activity-level eligibility results per objective
        - NACE sector breakdown (eligible vs. not-eligible per sector)
        - Revenue-weighted eligibility ratio
        - Summary statistics and executive narrative
        """
        phase = Phase.ELIGIBILITY_REPORT
        activities = context.state.get("eligible_activities", [])
        total_revenue = context.state.get("total_revenue", 0.0)
        eligible_by_objective = context.state.get("eligible_by_objective", {})

        self.logger.info("Generating eligibility matrix report")

        # Calculate revenue-weighted eligibility
        eligible_revenue = sum(
            a["revenue"] for a in activities if a.get("eligibility_status") == "ELIGIBLE"
        )
        revenue_weighted_eligibility = eligible_revenue / max(total_revenue, 1.0)

        context.state["revenue_weighted_eligibility"] = round(revenue_weighted_eligibility, 4)

        # Sector breakdown
        sector_breakdown: Dict[str, Dict[str, int]] = {}
        for activity in activities:
            nace_sector = activity["nace_code"][:1] if activity.get("nace_code") else "X"
            if nace_sector not in sector_breakdown:
                sector_breakdown[nace_sector] = {"eligible": 0, "not_eligible": 0}
            if activity.get("eligibility_status") == "ELIGIBLE":
                sector_breakdown[nace_sector]["eligible"] += 1
            else:
                sector_breakdown[nace_sector]["not_eligible"] += 1

        # Generate report metadata
        report = {
            "report_id": f"ELIG-{uuid.uuid4().hex[:8]}",
            "organization_id": self.config.organization_id,
            "reporting_period": self.config.reporting_period,
            "generated_at": datetime.utcnow().isoformat(),
            "total_activities": len(activities),
            "eligible_count": context.state.get("eligible_count", 0),
            "eligible_by_objective": eligible_by_objective,
            "revenue_weighted_eligibility": round(revenue_weighted_eligibility, 4),
            "eligible_revenue": round(eligible_revenue, 2),
            "total_revenue": round(total_revenue, 2),
            "sector_breakdown": sector_breakdown,
        }

        context.state["eligibility_report"] = report

        provenance = self._hash({
            "phase": phase.value,
            "report_id": report["report_id"],
            "revenue_weighted_eligibility": revenue_weighted_eligibility,
        })

        return PhaseResult(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            data={
                "report_id": report["report_id"],
                "revenue_weighted_eligibility": round(revenue_weighted_eligibility, 4),
                "eligible_revenue": round(eligible_revenue, 2),
                "total_revenue": round(total_revenue, 2),
                "sectors_covered": len(sector_breakdown),
            },
            provenance_hash=provenance,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
