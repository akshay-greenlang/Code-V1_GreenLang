# -*- coding: utf-8 -*-
"""
ESRS Retail Disclosure Workflow
====================================

4-phase workflow for generating ESRS disclosure content tailored
to the retail and consumer goods sector within PACK-014.

Phases:
    1. MaterialityAssessment   -- Identify material ESRS topics for retail
    2. DataPointCollection     -- Gather required datapoints per topic
    3. DisclosureGeneration    -- Generate ESRS chapter content
    4. AuditPreparation        -- Prepare audit trail and evidence packages

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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


class ESRSTopic(str, Enum):
    """ESRS topics."""
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    G1 = "G1"
    ESRS2 = "ESRS2"


class MaterialityLevel(str, Enum):
    """Materiality assessment level."""
    MATERIAL = "material"
    NOT_MATERIAL = "not_material"
    CONDITIONALLY_MATERIAL = "conditionally_material"


class DataPointStatus(str, Enum):
    """Status of a data point."""
    COLLECTED = "collected"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class MaterialityResult(BaseModel):
    """Materiality assessment result for an ESRS topic."""
    topic: str = Field(..., description="ESRS topic code")
    topic_name: str = Field(default="")
    level: MaterialityLevel = Field(default=MaterialityLevel.NOT_MATERIAL)
    impact_score: float = Field(default=0.0, ge=0.0, le=10.0)
    financial_score: float = Field(default=0.0, ge=0.0, le=10.0)
    combined_score: float = Field(default=0.0, ge=0.0, le=10.0)
    rationale: str = Field(default="")
    stakeholder_relevance: str = Field(default="", description="high|medium|low")


class ESRSDataPoint(BaseModel):
    """Individual ESRS data point."""
    datapoint_id: str = Field(default="")
    topic: str = Field(default="")
    disclosure_requirement: str = Field(default="", description="DR code e.g. E1-1")
    description: str = Field(default="")
    value: Optional[Any] = Field(None)
    unit: str = Field(default="")
    status: DataPointStatus = Field(default=DataPointStatus.MISSING)
    source: str = Field(default="")
    evidence_ref: str = Field(default="")


class DisclosureChapter(BaseModel):
    """Generated ESRS disclosure chapter."""
    topic: str = Field(...)
    topic_name: str = Field(default="")
    chapter_title: str = Field(default="")
    content_sections: List[Dict[str, Any]] = Field(default_factory=list)
    datapoints_used: int = Field(default=0)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    word_count: int = Field(default=0)


class EvidencePackage(BaseModel):
    """Audit evidence package for a topic."""
    topic: str = Field(...)
    evidence_items: List[Dict[str, str]] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    provenance_hashes: List[str] = Field(default_factory=list)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class ESRSDisclosureInput(BaseModel):
    """Input data model for ESRSRetailDisclosureWorkflow."""
    materiality_results: List[MaterialityResult] = Field(default_factory=list)
    datapoints: List[ESRSDataPoint] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class ESRSDisclosureResult(BaseModel):
    """Complete result from ESRS disclosure workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="esrs_retail_disclosure")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    disclosure_chapters: List[DisclosureChapter] = Field(default_factory=list)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    audit_trail: List[EvidencePackage] = Field(default_factory=list)
    material_topics: List[str] = Field(default_factory=list)
    total_datapoints: int = Field(default=0)
    collected_datapoints: int = Field(default=0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# ESRS TOPIC METADATA
# =============================================================================

ESRS_TOPIC_NAMES: Dict[str, str] = {
    "E1": "Climate Change",
    "E2": "Pollution",
    "E3": "Water and Marine Resources",
    "E4": "Biodiversity and Ecosystems",
    "E5": "Resource Use and Circular Economy",
    "S1": "Own Workforce",
    "S2": "Workers in the Value Chain",
    "S3": "Affected Communities",
    "S4": "Consumers and End-Users",
    "G1": "Business Conduct",
    "ESRS2": "General Disclosures",
}

# Retail-sector default materiality scores (0-10)
RETAIL_MATERIALITY_DEFAULTS: Dict[str, Dict[str, float]] = {
    "E1": {"impact": 8.5, "financial": 7.5},
    "E2": {"impact": 5.0, "financial": 4.0},
    "E3": {"impact": 4.5, "financial": 3.5},
    "E4": {"impact": 5.5, "financial": 4.5},
    "E5": {"impact": 8.0, "financial": 7.0},
    "S1": {"impact": 7.5, "financial": 6.5},
    "S2": {"impact": 8.0, "financial": 7.0},
    "S3": {"impact": 4.0, "financial": 3.0},
    "S4": {"impact": 7.0, "financial": 6.0},
    "G1": {"impact": 6.5, "financial": 6.0},
    "ESRS2": {"impact": 10.0, "financial": 10.0},
}

# Key disclosure requirements per topic (simplified)
TOPIC_DRS: Dict[str, List[str]] = {
    "E1": ["E1-1", "E1-2", "E1-3", "E1-4", "E1-5", "E1-6", "E1-7", "E1-8", "E1-9"],
    "E5": ["E5-1", "E5-2", "E5-3", "E5-4", "E5-5", "E5-6"],
    "S1": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5", "S1-6", "S1-7", "S1-8", "S1-9", "S1-10", "S1-11", "S1-12", "S1-13", "S1-14", "S1-15", "S1-16", "S1-17"],
    "S2": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"],
    "S4": ["S4-1", "S4-2", "S4-3", "S4-4", "S4-5"],
    "G1": ["G1-1", "G1-2", "G1-3", "G1-4", "G1-5", "G1-6"],
    "ESRS2": ["ESRS2-BP1", "ESRS2-BP2", "ESRS2-GOV1", "ESRS2-GOV2", "ESRS2-GOV3", "ESRS2-GOV4", "ESRS2-GOV5", "ESRS2-SBM1", "ESRS2-SBM2", "ESRS2-SBM3", "ESRS2-IRO1", "ESRS2-IRO2"],
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ESRSRetailDisclosureWorkflow:
    """
    4-phase ESRS retail disclosure workflow.

    Performs materiality assessment for retail, collects datapoints,
    generates disclosure chapters per material topic, and prepares
    audit evidence packages.

    Example:
        >>> wf = ESRSRetailDisclosureWorkflow()
        >>> inp = ESRSDisclosureInput(datapoints=[...])
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ESRSRetailDisclosureWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._material_topics: List[str] = []
        self._materiality: List[MaterialityResult] = []
        self._chapters: List[DisclosureChapter] = []
        self._evidence: List[EvidencePackage] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[ESRSDisclosureInput] = None,
        materiality_results: Optional[List[MaterialityResult]] = None,
        datapoints: Optional[List[ESRSDataPoint]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ESRSDisclosureResult:
        """Execute the 4-phase ESRS disclosure workflow."""
        if input_data is None:
            input_data = ESRSDisclosureInput(
                materiality_results=materiality_results or [],
                datapoints=datapoints or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting ESRS disclosure workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_materiality(input_data))
            phase_results.append(await self._phase_data_collection(input_data))
            phase_results.append(await self._phase_disclosure_generation(input_data))
            phase_results.append(await self._phase_audit_preparation(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("ESRS disclosure workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_dp = len(input_data.datapoints)
        collected_dp = sum(1 for dp in input_data.datapoints if dp.status == DataPointStatus.COLLECTED)
        completeness = (collected_dp / max(total_dp, 1)) * 100

        result = ESRSDisclosureResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            disclosure_chapters=self._chapters,
            completeness_pct=round(completeness, 2),
            audit_trail=self._evidence,
            material_topics=self._material_topics,
            total_datapoints=total_dp,
            collected_datapoints=collected_dp,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Materiality Assessment
    # -------------------------------------------------------------------------

    async def _phase_materiality(self, input_data: ESRSDisclosureInput) -> PhaseResult:
        """Identify material ESRS topics for retail."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}

        if input_data.materiality_results:
            self._materiality = input_data.materiality_results
        else:
            self._materiality = []
            for topic, scores in RETAIL_MATERIALITY_DEFAULTS.items():
                combined = (scores["impact"] + scores["financial"]) / 2
                level = MaterialityLevel.MATERIAL if combined >= 5.0 else MaterialityLevel.NOT_MATERIAL
                self._materiality.append(MaterialityResult(
                    topic=topic,
                    topic_name=ESRS_TOPIC_NAMES.get(topic, topic),
                    level=level,
                    impact_score=scores["impact"],
                    financial_score=scores["financial"],
                    combined_score=round(combined, 2),
                    stakeholder_relevance="high" if combined >= 7.0 else "medium" if combined >= 5.0 else "low",
                ))

        self._material_topics = [m.topic for m in self._materiality if m.level == MaterialityLevel.MATERIAL]
        outputs["total_topics_assessed"] = len(self._materiality)
        outputs["material_topics"] = self._material_topics
        outputs["non_material_topics"] = [m.topic for m in self._materiality if m.level == MaterialityLevel.NOT_MATERIAL]

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 1 Materiality: %d material topics", len(self._material_topics))
        return PhaseResult(
            phase_name="materiality_assessment", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Point Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(self, input_data: ESRSDisclosureInput) -> PhaseResult:
        """Gather required datapoints per material topic."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        topic_status: Dict[str, Dict[str, int]] = {}
        for topic in self._material_topics:
            topic_dps = [dp for dp in input_data.datapoints if dp.topic == topic]
            collected = sum(1 for dp in topic_dps if dp.status == DataPointStatus.COLLECTED)
            missing = sum(1 for dp in topic_dps if dp.status == DataPointStatus.MISSING)
            total = len(topic_dps)

            if total == 0:
                expected_drs = TOPIC_DRS.get(topic, [])
                warnings.append(f"Topic {topic}: no datapoints provided ({len(expected_drs)} DRs expected)")

            topic_status[topic] = {
                "total": total, "collected": collected, "missing": missing,
                "completeness_pct": round(collected / max(total, 1) * 100, 2),
            }

        outputs["topic_data_status"] = topic_status
        outputs["total_datapoints"] = len(input_data.datapoints)
        outputs["total_collected"] = sum(1 for dp in input_data.datapoints if dp.status == DataPointStatus.COLLECTED)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 2 DataCollection: %d collected of %d", outputs["total_collected"], outputs["total_datapoints"])
        return PhaseResult(
            phase_name="data_point_collection", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Disclosure Generation
    # -------------------------------------------------------------------------

    async def _phase_disclosure_generation(self, input_data: ESRSDisclosureInput) -> PhaseResult:
        """Generate ESRS disclosure chapter content."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._chapters = []

        for topic in self._material_topics:
            topic_dps = [dp for dp in input_data.datapoints if dp.topic == topic]
            collected = sum(1 for dp in topic_dps if dp.status == DataPointStatus.COLLECTED)
            total = max(len(topic_dps), 1)
            completeness = (collected / total) * 100

            sections: List[Dict[str, Any]] = []
            drs = TOPIC_DRS.get(topic, [])
            for dr in drs:
                dr_dps = [dp for dp in topic_dps if dp.disclosure_requirement == dr]
                dr_collected = sum(1 for dp in dr_dps if dp.status == DataPointStatus.COLLECTED)
                sections.append({
                    "disclosure_requirement": dr,
                    "datapoints_available": len(dr_dps),
                    "datapoints_collected": dr_collected,
                    "status": "complete" if dr_collected == len(dr_dps) and dr_dps else "incomplete",
                })

            word_count = collected * 150  # Estimated words per datapoint
            self._chapters.append(DisclosureChapter(
                topic=topic,
                topic_name=ESRS_TOPIC_NAMES.get(topic, topic),
                chapter_title=f"ESRS {topic} - {ESRS_TOPIC_NAMES.get(topic, topic)}",
                content_sections=sections,
                datapoints_used=collected,
                completeness_pct=round(completeness, 2),
                word_count=word_count,
            ))

        outputs["chapters_generated"] = len(self._chapters)
        outputs["avg_completeness_pct"] = round(
            sum(c.completeness_pct for c in self._chapters) / max(len(self._chapters), 1), 2
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 3 DisclosureGeneration: %d chapters", len(self._chapters))
        return PhaseResult(
            phase_name="disclosure_generation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Audit Preparation
    # -------------------------------------------------------------------------

    async def _phase_audit_preparation(self, input_data: ESRSDisclosureInput) -> PhaseResult:
        """Prepare audit trail and evidence packages."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        self._evidence = []

        for topic in self._material_topics:
            topic_dps = [dp for dp in input_data.datapoints if dp.topic == topic]
            evidence_items = []
            sources: List[str] = []
            hashes: List[str] = []

            for dp in topic_dps:
                if dp.status == DataPointStatus.COLLECTED:
                    evidence_items.append({
                        "datapoint_id": dp.datapoint_id,
                        "disclosure_requirement": dp.disclosure_requirement,
                        "source": dp.source,
                        "evidence_ref": dp.evidence_ref,
                    })
                    if dp.source and dp.source not in sources:
                        sources.append(dp.source)
                    dp_hash = hashlib.sha256(
                        json.dumps({"id": dp.datapoint_id, "value": str(dp.value)}).encode()
                    ).hexdigest()
                    hashes.append(dp_hash)

            collected = len(evidence_items)
            total = max(len(topic_dps), 1)
            self._evidence.append(EvidencePackage(
                topic=topic,
                evidence_items=evidence_items,
                data_sources=sources,
                provenance_hashes=hashes,
                completeness_pct=round(collected / total * 100, 2),
            ))

        outputs["evidence_packages"] = len(self._evidence)
        outputs["total_evidence_items"] = sum(len(e.evidence_items) for e in self._evidence)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 4 AuditPreparation: %d evidence packages", len(self._evidence))
        return PhaseResult(
            phase_name="audit_preparation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ESRSDisclosureResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
