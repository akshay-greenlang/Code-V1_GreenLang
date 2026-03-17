# -*- coding: utf-8 -*-
"""
ESRS Mapping Workflow
=========================

3-phase workflow for mapping double materiality results to ESRS disclosure
requirements within PACK-015 Double Materiality Pack. Selects material ESRS
topics, maps to specific disclosure requirements (DRs), and performs gap
analysis against current data availability.

Phases:
    1. TopicSelection     -- Select material ESRS topics from DMA results
    2. DisclosureMapping  -- Map to specific ESRS disclosure requirements
    3. GapAnalysis        -- Identify disclosure gaps and effort estimates

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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


class DisclosureStatus(str, Enum):
    """Status of a disclosure requirement."""
    COVERED = "covered"
    PARTIAL = "partial"
    GAP = "gap"
    NOT_APPLICABLE = "not_applicable"


class EffortLevel(str, Enum):
    """Effort level to close a disclosure gap."""
    LOW = "low"           # <1 week
    MEDIUM = "medium"     # 1-4 weeks
    HIGH = "high"         # 1-3 months
    VERY_HIGH = "very_high"  # >3 months


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


class MaterialTopic(BaseModel):
    """A material ESRS topic selected from DMA results."""
    topic_id: str = Field(default="", description="ESRS topic code")
    topic_name: str = Field(default="")
    impact_score: float = Field(default=0.0, ge=0.0, le=5.0)
    financial_score: float = Field(default=0.0, ge=0.0, le=5.0)
    materiality_type: str = Field(
        default="double_material",
        description="double_material|impact_only|financial_only",
    )


class DisclosureRequirement(BaseModel):
    """An ESRS disclosure requirement mapping."""
    dr_id: str = Field(default="", description="Disclosure requirement ID (e.g., E1-1)")
    dr_name: str = Field(default="", description="Disclosure requirement name")
    esrs_topic: str = Field(default="", description="Parent ESRS topic")
    esrs_paragraph: str = Field(default="", description="ESRS paragraph reference")
    is_mandatory: bool = Field(default=True)
    datapoints_required: int = Field(default=0, ge=0)
    datapoints_available: int = Field(default=0, ge=0)
    status: DisclosureStatus = Field(default=DisclosureStatus.GAP)
    description: str = Field(default="")


class DisclosureGap(BaseModel):
    """An identified gap in ESRS disclosure coverage."""
    gap_id: str = Field(default_factory=lambda: f"gap-{uuid.uuid4().hex[:8]}")
    dr_id: str = Field(default="")
    dr_name: str = Field(default="")
    esrs_topic: str = Field(default="")
    missing_datapoints: int = Field(default=0, ge=0)
    effort_level: EffortLevel = Field(default=EffortLevel.MEDIUM)
    estimated_weeks: float = Field(default=0.0, ge=0.0)
    responsible_team: str = Field(default="")
    description: str = Field(default="")
    priority: int = Field(default=0, ge=0, description="1=highest priority")


class ESRSMappingInput(BaseModel):
    """Input data model for ESRSMappingWorkflow."""
    material_topics: List[MaterialTopic] = Field(
        default_factory=list, description="Material topics from DMA"
    )
    existing_disclosures: List[DisclosureRequirement] = Field(
        default_factory=list, description="Current disclosure coverage"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    include_voluntary: bool = Field(
        default=False, description="Include voluntary disclosures"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class ESRSMappingResult(BaseModel):
    """Complete result from ESRS mapping workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="esrs_mapping")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    material_topics_count: int = Field(default=0, ge=0)
    disclosure_requirements: List[DisclosureRequirement] = Field(default_factory=list)
    total_drs_mapped: int = Field(default=0, ge=0)
    gaps_identified: List[DisclosureGap] = Field(default_factory=list)
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_effort_weeks: float = Field(default=0.0, ge=0.0)
    topic_coverage: Dict[str, float] = Field(default_factory=dict)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# ESRS DISCLOSURE REQUIREMENT REFERENCE DATA
# =============================================================================

# Disclosure requirements per ESRS topic (DR ID, name, mandatory, datapoints)
ESRS_DR_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "E1": [
        {"dr_id": "E1-1", "name": "Transition plan for climate change mitigation", "mandatory": True, "datapoints": 12},
        {"dr_id": "E1-2", "name": "Policies related to climate change mitigation and adaptation", "mandatory": True, "datapoints": 8},
        {"dr_id": "E1-3", "name": "Actions and resources in relation to climate change policies", "mandatory": True, "datapoints": 10},
        {"dr_id": "E1-4", "name": "Targets related to climate change mitigation and adaptation", "mandatory": True, "datapoints": 14},
        {"dr_id": "E1-5", "name": "Energy consumption and mix", "mandatory": True, "datapoints": 9},
        {"dr_id": "E1-6", "name": "Gross Scopes 1, 2, 3 and Total GHG emissions", "mandatory": True, "datapoints": 18},
        {"dr_id": "E1-7", "name": "GHG removals and GHG mitigation projects", "mandatory": False, "datapoints": 6},
        {"dr_id": "E1-8", "name": "Internal carbon pricing", "mandatory": False, "datapoints": 5},
        {"dr_id": "E1-9", "name": "Anticipated financial effects from climate change", "mandatory": True, "datapoints": 8},
    ],
    "E2": [
        {"dr_id": "E2-1", "name": "Policies related to pollution", "mandatory": True, "datapoints": 6},
        {"dr_id": "E2-2", "name": "Actions and resources related to pollution", "mandatory": True, "datapoints": 8},
        {"dr_id": "E2-3", "name": "Targets related to pollution", "mandatory": True, "datapoints": 7},
        {"dr_id": "E2-4", "name": "Pollution of air, water and soil", "mandatory": True, "datapoints": 10},
        {"dr_id": "E2-5", "name": "Substances of concern and substances of very high concern", "mandatory": True, "datapoints": 6},
        {"dr_id": "E2-6", "name": "Anticipated financial effects from pollution", "mandatory": True, "datapoints": 5},
    ],
    "E3": [
        {"dr_id": "E3-1", "name": "Policies related to water and marine resources", "mandatory": True, "datapoints": 6},
        {"dr_id": "E3-2", "name": "Actions and resources related to water", "mandatory": True, "datapoints": 7},
        {"dr_id": "E3-3", "name": "Targets related to water and marine resources", "mandatory": True, "datapoints": 6},
        {"dr_id": "E3-4", "name": "Water consumption", "mandatory": True, "datapoints": 8},
        {"dr_id": "E3-5", "name": "Anticipated financial effects from water", "mandatory": True, "datapoints": 5},
    ],
    "E4": [
        {"dr_id": "E4-1", "name": "Transition plan on biodiversity and ecosystems", "mandatory": True, "datapoints": 8},
        {"dr_id": "E4-2", "name": "Policies related to biodiversity and ecosystems", "mandatory": True, "datapoints": 6},
        {"dr_id": "E4-3", "name": "Actions and resources related to biodiversity", "mandatory": True, "datapoints": 7},
        {"dr_id": "E4-4", "name": "Targets related to biodiversity and ecosystems", "mandatory": True, "datapoints": 6},
        {"dr_id": "E4-5", "name": "Impact metrics related to biodiversity", "mandatory": True, "datapoints": 9},
        {"dr_id": "E4-6", "name": "Anticipated financial effects from biodiversity", "mandatory": True, "datapoints": 5},
    ],
    "E5": [
        {"dr_id": "E5-1", "name": "Policies related to resource use and circular economy", "mandatory": True, "datapoints": 6},
        {"dr_id": "E5-2", "name": "Actions and resources related to circular economy", "mandatory": True, "datapoints": 7},
        {"dr_id": "E5-3", "name": "Targets related to resource use and circular economy", "mandatory": True, "datapoints": 6},
        {"dr_id": "E5-4", "name": "Resource inflows", "mandatory": True, "datapoints": 7},
        {"dr_id": "E5-5", "name": "Resource outflows", "mandatory": True, "datapoints": 8},
        {"dr_id": "E5-6", "name": "Anticipated financial effects from circular economy", "mandatory": True, "datapoints": 5},
    ],
    "S1": [
        {"dr_id": "S1-1", "name": "Policies related to own workforce", "mandatory": True, "datapoints": 8},
        {"dr_id": "S1-2", "name": "Processes for engaging with own workers", "mandatory": True, "datapoints": 5},
        {"dr_id": "S1-3", "name": "Processes to remediate negative impacts", "mandatory": True, "datapoints": 5},
        {"dr_id": "S1-4", "name": "Taking action on material impacts", "mandatory": True, "datapoints": 7},
        {"dr_id": "S1-5", "name": "Targets related to managing impacts", "mandatory": True, "datapoints": 6},
        {"dr_id": "S1-6", "name": "Characteristics of the undertaking employees", "mandatory": True, "datapoints": 12},
    ],
    "S2": [
        {"dr_id": "S2-1", "name": "Policies related to value chain workers", "mandatory": True, "datapoints": 6},
        {"dr_id": "S2-2", "name": "Processes for engaging with value chain workers", "mandatory": True, "datapoints": 5},
        {"dr_id": "S2-3", "name": "Processes to remediate negative impacts", "mandatory": True, "datapoints": 5},
        {"dr_id": "S2-4", "name": "Taking action on material impacts", "mandatory": True, "datapoints": 6},
        {"dr_id": "S2-5", "name": "Targets related to managing impacts", "mandatory": True, "datapoints": 5},
    ],
    "S3": [
        {"dr_id": "S3-1", "name": "Policies related to affected communities", "mandatory": True, "datapoints": 6},
        {"dr_id": "S3-2", "name": "Processes for engaging with affected communities", "mandatory": True, "datapoints": 5},
        {"dr_id": "S3-3", "name": "Processes to remediate negative impacts", "mandatory": True, "datapoints": 5},
        {"dr_id": "S3-4", "name": "Taking action on material impacts", "mandatory": True, "datapoints": 6},
        {"dr_id": "S3-5", "name": "Targets related to managing impacts", "mandatory": True, "datapoints": 5},
    ],
    "S4": [
        {"dr_id": "S4-1", "name": "Policies related to consumers and end-users", "mandatory": True, "datapoints": 6},
        {"dr_id": "S4-2", "name": "Processes for engaging with consumers", "mandatory": True, "datapoints": 5},
        {"dr_id": "S4-3", "name": "Processes to remediate negative impacts", "mandatory": True, "datapoints": 5},
        {"dr_id": "S4-4", "name": "Taking action on material impacts", "mandatory": True, "datapoints": 6},
        {"dr_id": "S4-5", "name": "Targets related to managing impacts", "mandatory": True, "datapoints": 5},
    ],
    "G1": [
        {"dr_id": "G1-1", "name": "Business conduct policies and corporate culture", "mandatory": True, "datapoints": 8},
        {"dr_id": "G1-2", "name": "Management of relationships with suppliers", "mandatory": True, "datapoints": 6},
        {"dr_id": "G1-3", "name": "Prevention and detection of corruption and bribery", "mandatory": True, "datapoints": 7},
        {"dr_id": "G1-4", "name": "Incidents of corruption or bribery", "mandatory": True, "datapoints": 5},
        {"dr_id": "G1-5", "name": "Political influence and lobbying activities", "mandatory": True, "datapoints": 5},
        {"dr_id": "G1-6", "name": "Payment practices", "mandatory": True, "datapoints": 4},
    ],
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ESRSMappingWorkflow:
    """
    3-phase ESRS disclosure mapping workflow.

    Selects material ESRS topics from double materiality assessment results,
    maps them to specific disclosure requirements using the ESRS DR catalog,
    and performs gap analysis to identify missing datapoints with effort
    estimates and prioritization.

    Zero-hallucination: all mapping uses the deterministic ESRS DR catalog.
    Gap identification is based on datapoint counts. No LLM in numeric paths.

    Example:
        >>> wf = ESRSMappingWorkflow()
        >>> inp = ESRSMappingInput(material_topics=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.coverage_pct >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ESRSMappingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._material_topics: List[MaterialTopic] = []
        self._disclosure_reqs: List[DisclosureRequirement] = []
        self._gaps: List[DisclosureGap] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[ESRSMappingInput] = None,
        material_topics: Optional[List[MaterialTopic]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ESRSMappingResult:
        """
        Execute the 3-phase ESRS mapping workflow.

        Args:
            input_data: Full input model (preferred).
            material_topics: Material topics (fallback).
            config: Configuration overrides.

        Returns:
            ESRSMappingResult with disclosure mappings, gaps, coverage.
        """
        if input_data is None:
            input_data = ESRSMappingInput(
                material_topics=material_topics or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting ESRS mapping %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_topic_selection(input_data))
            phase_results.append(await self._phase_disclosure_mapping(input_data))
            phase_results.append(await self._phase_gap_analysis(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "ESRS mapping workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_dps_required = sum(dr.datapoints_required for dr in self._disclosure_reqs)
        total_dps_available = sum(dr.datapoints_available for dr in self._disclosure_reqs)
        coverage_pct = (
            (total_dps_available / total_dps_required * 100)
            if total_dps_required > 0 else 0.0
        )

        topic_coverage: Dict[str, float] = {}
        for mt in self._material_topics:
            topic_drs = [dr for dr in self._disclosure_reqs if dr.esrs_topic == mt.topic_id]
            req = sum(dr.datapoints_required for dr in topic_drs)
            avail = sum(dr.datapoints_available for dr in topic_drs)
            topic_coverage[mt.topic_id] = round((avail / req * 100) if req > 0 else 0.0, 1)

        total_effort = sum(g.estimated_weeks for g in self._gaps)

        result = ESRSMappingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            material_topics_count=len(self._material_topics),
            disclosure_requirements=self._disclosure_reqs,
            total_drs_mapped=len(self._disclosure_reqs),
            gaps_identified=self._gaps,
            coverage_pct=round(coverage_pct, 1),
            total_effort_weeks=round(total_effort, 1),
            topic_coverage=topic_coverage,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "ESRS mapping %s completed in %.2fs: %d DRs, %.1f%% coverage, %d gaps",
            self.workflow_id, elapsed, len(self._disclosure_reqs),
            coverage_pct, len(self._gaps),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Topic Selection
    # -------------------------------------------------------------------------

    async def _phase_topic_selection(
        self, input_data: ESRSMappingInput,
    ) -> PhaseResult:
        """Select material ESRS topics from DMA results."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._material_topics = list(input_data.material_topics)

        topics_selected = [mt.topic_id for mt in self._material_topics]
        materiality_types: Dict[str, int] = {}
        for mt in self._material_topics:
            materiality_types[mt.materiality_type] = (
                materiality_types.get(mt.materiality_type, 0) + 1
            )

        outputs["topics_selected"] = topics_selected
        outputs["topics_count"] = len(topics_selected)
        outputs["materiality_type_distribution"] = materiality_types

        if not topics_selected:
            warnings.append("No material topics provided; ESRS mapping will be empty")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 TopicSelection: %d material topics selected",
            len(topics_selected),
        )
        return PhaseResult(
            phase_name="topic_selection", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Disclosure Mapping
    # -------------------------------------------------------------------------

    async def _phase_disclosure_mapping(
        self, input_data: ESRSMappingInput,
    ) -> PhaseResult:
        """Map material topics to specific ESRS disclosure requirements."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._disclosure_reqs = []

        # Build lookup for existing disclosures
        existing_lookup: Dict[str, DisclosureRequirement] = {}
        for dr in input_data.existing_disclosures:
            existing_lookup[dr.dr_id] = dr

        for mt in self._material_topics:
            catalog_drs = ESRS_DR_CATALOG.get(mt.topic_id, [])
            for cat_dr in catalog_drs:
                if not cat_dr["mandatory"] and not input_data.include_voluntary:
                    continue

                existing = existing_lookup.get(cat_dr["dr_id"])
                datapoints_available = (
                    existing.datapoints_available if existing else 0
                )
                datapoints_required = cat_dr["datapoints"]

                if datapoints_available >= datapoints_required:
                    status = DisclosureStatus.COVERED
                elif datapoints_available > 0:
                    status = DisclosureStatus.PARTIAL
                else:
                    status = DisclosureStatus.GAP

                self._disclosure_reqs.append(DisclosureRequirement(
                    dr_id=cat_dr["dr_id"],
                    dr_name=cat_dr["name"],
                    esrs_topic=mt.topic_id,
                    is_mandatory=cat_dr["mandatory"],
                    datapoints_required=datapoints_required,
                    datapoints_available=datapoints_available,
                    status=status,
                ))

        status_counts: Dict[str, int] = {}
        for dr in self._disclosure_reqs:
            status_counts[dr.status.value] = status_counts.get(dr.status.value, 0) + 1

        outputs["total_drs_mapped"] = len(self._disclosure_reqs)
        outputs["status_distribution"] = status_counts
        outputs["mandatory_drs"] = sum(1 for dr in self._disclosure_reqs if dr.is_mandatory)
        outputs["voluntary_drs"] = sum(1 for dr in self._disclosure_reqs if not dr.is_mandatory)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DisclosureMapping: %d DRs mapped across %d topics",
            len(self._disclosure_reqs), len(self._material_topics),
        )
        return PhaseResult(
            phase_name="disclosure_mapping", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(
        self, input_data: ESRSMappingInput,
    ) -> PhaseResult:
        """Identify disclosure gaps with effort estimates."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._gaps = []

        priority_counter = 0
        for dr in self._disclosure_reqs:
            if dr.status in (DisclosureStatus.GAP, DisclosureStatus.PARTIAL):
                missing = dr.datapoints_required - dr.datapoints_available
                effort = self._estimate_effort(missing)
                priority_counter += 1

                self._gaps.append(DisclosureGap(
                    dr_id=dr.dr_id,
                    dr_name=dr.dr_name,
                    esrs_topic=dr.esrs_topic,
                    missing_datapoints=missing,
                    effort_level=effort["level"],
                    estimated_weeks=effort["weeks"],
                    description=f"Missing {missing} of {dr.datapoints_required} datapoints",
                    priority=priority_counter,
                ))

        # Sort gaps by effort (highest first for prioritization)
        self._gaps.sort(key=lambda g: g.estimated_weeks, reverse=True)
        for i, gap in enumerate(self._gaps, start=1):
            gap.priority = i

        total_effort = sum(g.estimated_weeks for g in self._gaps)
        effort_by_topic: Dict[str, float] = {}
        for gap in self._gaps:
            effort_by_topic[gap.esrs_topic] = (
                effort_by_topic.get(gap.esrs_topic, 0.0) + gap.estimated_weeks
            )

        outputs["gaps_identified"] = len(self._gaps)
        outputs["total_missing_datapoints"] = sum(g.missing_datapoints for g in self._gaps)
        outputs["total_effort_weeks"] = round(total_effort, 1)
        outputs["effort_by_topic"] = {k: round(v, 1) for k, v in effort_by_topic.items()}
        outputs["effort_level_distribution"] = self._count_effort_levels()

        if len(self._gaps) > len(self._disclosure_reqs) * 0.5:
            warnings.append(
                f"More than 50% of disclosure requirements have gaps "
                f"({len(self._gaps)}/{len(self._disclosure_reqs)})"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 GapAnalysis: %d gaps, %.1f weeks total effort",
            len(self._gaps), total_effort,
        )
        return PhaseResult(
            phase_name="gap_analysis", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _estimate_effort(self, missing_datapoints: int) -> Dict[str, Any]:
        """Estimate effort to close a disclosure gap based on missing datapoints."""
        if missing_datapoints <= 2:
            return {"level": EffortLevel.LOW, "weeks": 0.5}
        elif missing_datapoints <= 5:
            return {"level": EffortLevel.MEDIUM, "weeks": 2.0}
        elif missing_datapoints <= 10:
            return {"level": EffortLevel.HIGH, "weeks": 6.0}
        else:
            return {"level": EffortLevel.VERY_HIGH, "weeks": 12.0}

    def _count_effort_levels(self) -> Dict[str, int]:
        """Count gaps by effort level."""
        counts: Dict[str, int] = {}
        for gap in self._gaps:
            counts[gap.effort_level.value] = counts.get(gap.effort_level.value, 0) + 1
        return counts

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ESRSMappingResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
