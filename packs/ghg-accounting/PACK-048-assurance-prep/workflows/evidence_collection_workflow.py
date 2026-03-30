# -*- coding: utf-8 -*-
"""
Evidence Collection Workflow
====================================

5-phase workflow for GHG assurance evidence collection covering scope
inventory, source identification, document collection, quality grading,
and evidence package build within PACK-048 GHG Assurance Prep Pack.

Phases:
    1. ScopeInventory              -- Inventory all emission sources across
                                      Scope 1, 2, and 3 categories, recording
                                      source type, facility, emission quantity,
                                      and data origin for each.
    2. SourceIdentification        -- Identify required evidence items for
                                      each emission source based on the
                                      assurance standard requirements, mapping
                                      source to evidence type (invoice, meter
                                      reading, calculation, contract, etc.).
    3. DocumentCollection          -- Collect/link evidence items from data
                                      bridges (ERP, meter systems, document
                                      management), recording availability,
                                      format, and access path.
    4. QualityGrading              -- Grade each evidence item on a 5-level
                                      scale (EXCELLENT / GOOD / ADEQUATE /
                                      MARGINAL / INSUFFICIENT) based on
                                      provenance, completeness, timeliness,
                                      and corroboration criteria.
    5. PackageBuild                -- Build a consolidated evidence package
                                      with a structured index, SHA-256 hashes
                                      for each document, cross-reference to
                                      checklist items, and overall package
                                      completeness metrics.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ISAE 3410 (2012) - Evidence requirements for GHG assurance
    ISO 14064-3:2019 - Validation/verification evidence criteria
    AA1000AS v3 (2020) - Underlying evidence expectations
    GHG Protocol Corporate Standard (2015) - Source documentation
    ESRS E1 (2024) - Assurance-ready disclosure evidence
    CSRD (2022/2464) - Documentation requirements for assurance

Schedule: Prior to assurance engagement commencement
Estimated duration: 3-6 weeks depending on source complexity

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

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

class EvidenceCollectionPhase(str, Enum):
    """Evidence collection workflow phases."""

    SCOPE_INVENTORY = "scope_inventory"
    SOURCE_IDENTIFICATION = "source_identification"
    DOCUMENT_COLLECTION = "document_collection"
    QUALITY_GRADING = "quality_grading"
    PACKAGE_BUILD = "package_build"

class EmissionScope(str, Enum):
    """GHG emission scope classification."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"

class SourceType(str, Enum):
    """Emission source type."""

    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    PURCHASED_ELECTRICITY = "purchased_electricity"
    PURCHASED_HEAT_STEAM = "purchased_heat_steam"
    PURCHASED_GOODS = "purchased_goods"
    CAPITAL_GOODS = "capital_goods"
    FUEL_ENERGY = "fuel_energy"
    TRANSPORT_UPSTREAM = "transport_upstream"
    WASTE_GENERATED = "waste_generated"
    BUSINESS_TRAVEL = "business_travel"
    EMPLOYEE_COMMUTING = "employee_commuting"
    OTHER = "other"

class EvidenceType(str, Enum):
    """Type of evidence document."""

    INVOICE = "invoice"
    METER_READING = "meter_reading"
    CALCULATION_SHEET = "calculation_sheet"
    CONTRACT = "contract"
    SUPPLIER_STATEMENT = "supplier_statement"
    UTILITY_BILL = "utility_bill"
    TRANSPORT_LOG = "transport_log"
    INTERNAL_REPORT = "internal_report"
    THIRD_PARTY_DATA = "third_party_data"
    EMISSION_FACTOR_SOURCE = "emission_factor_source"
    METHODOLOGY_DOCUMENT = "methodology_document"
    AUDIT_TRAIL = "audit_trail"

class EvidenceQualityGrade(str, Enum):
    """Evidence quality grading scale."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    MARGINAL = "marginal"
    INSUFFICIENT = "insufficient"

class DocumentStatus(str, Enum):
    """Status of a document in collection process."""

    COLLECTED = "collected"
    LINKED = "linked"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"

# =============================================================================
# EVIDENCE REQUIREMENTS REFERENCE DATA (Zero-Hallucination)
# =============================================================================

SOURCE_EVIDENCE_MAP: Dict[str, List[str]] = {
    "stationary_combustion": [
        "invoice", "meter_reading", "calculation_sheet",
        "emission_factor_source", "methodology_document",
    ],
    "mobile_combustion": [
        "invoice", "transport_log", "calculation_sheet",
        "emission_factor_source",
    ],
    "process_emissions": [
        "calculation_sheet", "methodology_document",
        "emission_factor_source", "internal_report",
    ],
    "fugitive_emissions": [
        "internal_report", "calculation_sheet",
        "emission_factor_source", "methodology_document",
    ],
    "purchased_electricity": [
        "utility_bill", "contract", "meter_reading",
        "emission_factor_source", "supplier_statement",
    ],
    "purchased_heat_steam": [
        "utility_bill", "contract", "meter_reading",
        "emission_factor_source",
    ],
    "purchased_goods": [
        "supplier_statement", "invoice", "calculation_sheet",
        "emission_factor_source", "third_party_data",
    ],
    "capital_goods": [
        "invoice", "supplier_statement", "calculation_sheet",
        "emission_factor_source",
    ],
    "fuel_energy": [
        "utility_bill", "invoice", "emission_factor_source",
        "calculation_sheet",
    ],
    "transport_upstream": [
        "transport_log", "invoice", "third_party_data",
        "calculation_sheet",
    ],
    "waste_generated": [
        "invoice", "internal_report", "calculation_sheet",
        "emission_factor_source",
    ],
    "business_travel": [
        "invoice", "internal_report", "third_party_data",
        "calculation_sheet",
    ],
    "employee_commuting": [
        "internal_report", "calculation_sheet",
        "third_party_data",
    ],
    "other": [
        "calculation_sheet", "emission_factor_source",
        "internal_report",
    ],
}

QUALITY_GRADE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "excellent": (90.0, 100.1),
    "good": (75.0, 90.0),
    "adequate": (55.0, 75.0),
    "marginal": (35.0, 55.0),
    "insufficient": (0.0, 35.0),
}

QUALITY_CRITERIA_WEIGHTS: Dict[str, Decimal] = {
    "provenance": Decimal("0.30"),
    "completeness": Decimal("0.30"),
    "timeliness": Decimal("0.20"),
    "corroboration": Decimal("0.20"),
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
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class EmissionSourceRecord(BaseModel):
    """Record of an emission source in the inventory."""

    source_id: str = Field(default_factory=lambda: f"src-{_new_uuid()[:8]}")
    scope: EmissionScope = Field(...)
    source_type: SourceType = Field(...)
    facility_name: str = Field(default="")
    description: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    activity_data_value: float = Field(default=0.0, ge=0.0)
    activity_data_unit: str = Field(default="")
    data_origin: str = Field(default="")
    scope_3_category: int = Field(default=0, ge=0, le=15)
    required_evidence_types: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class EvidenceItem(BaseModel):
    """A single evidence document item."""

    evidence_id: str = Field(default_factory=lambda: f"ev-{_new_uuid()[:8]}")
    source_id: str = Field(default="")
    evidence_type: EvidenceType = Field(...)
    document_name: str = Field(default="")
    document_format: str = Field(default="")
    document_status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    access_path: str = Field(default="")
    data_bridge_source: str = Field(default="")
    provenance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    timeliness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    corroboration_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_quality_score: str = Field(default="0.00")
    quality_grade: EvidenceQualityGrade = Field(default=EvidenceQualityGrade.INSUFFICIENT)
    document_hash: str = Field(default="")
    provenance_hash: str = Field(default="")

class PackageIndex(BaseModel):
    """Index entry in the evidence package."""

    index_number: int = Field(default=0)
    source_id: str = Field(default="")
    evidence_id: str = Field(default="")
    document_name: str = Field(default="")
    quality_grade: str = Field(default="")
    document_hash: str = Field(default="")
    checklist_reference: str = Field(default="")

class PackageSummary(BaseModel):
    """Summary of the evidence package."""

    total_sources: int = Field(default=0, ge=0)
    total_evidence_items: int = Field(default=0, ge=0)
    collected_count: int = Field(default=0, ge=0)
    pending_count: int = Field(default=0, ge=0)
    unavailable_count: int = Field(default=0, ge=0)
    completeness_pct: str = Field(default="0.00")
    avg_quality_score: str = Field(default="0.00")
    package_hash: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class EvidenceCollectionInput(BaseModel):
    """Input data model for EvidenceCollectionWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    emission_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of emission sources with scope, type, facility, emissions",
    )
    existing_documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pre-existing evidence documents with type, source_id, scores",
    )
    data_bridge_connections: List[str] = Field(
        default_factory=list,
        description="Connected data bridges (erp, meter_system, dms)",
    )
    reporting_period: str = Field(default="2025")
    include_scope_3: bool = Field(default=False)
    scope_3_categories: List[int] = Field(
        default_factory=list, description="Scope 3 categories to include (1-15)",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class EvidenceCollectionResult(BaseModel):
    """Complete result from evidence collection workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="evidence_collection")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    source_inventory: List[EmissionSourceRecord] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    package_index: List[PackageIndex] = Field(default_factory=list)
    package_summary: Optional[PackageSummary] = Field(default=None)
    total_sources: int = Field(default=0)
    total_evidence_items: int = Field(default=0)
    completeness_pct: str = Field(default="0.00")
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class EvidenceCollectionWorkflow:
    """
    5-phase workflow for GHG assurance evidence collection.

    Inventories all emission sources, identifies required evidence per source,
    collects/links documents from data bridges, grades evidence quality, and
    builds a consolidated evidence package with index and hashes.

    Zero-hallucination: all quality scoring uses Decimal arithmetic with
    ROUND_HALF_UP; evidence requirements are deterministic per source type;
    no LLM calls in scoring path; SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _sources: Emission source inventory.
        _evidence_items: Collected evidence items.
        _package_index: Package index entries.
        _package_summary: Package summary statistics.

    Example:
        >>> wf = EvidenceCollectionWorkflow()
        >>> inp = EvidenceCollectionInput(
        ...     organization_id="org-001",
        ...     emission_sources=[...],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[EvidenceCollectionPhase] = [
        EvidenceCollectionPhase.SCOPE_INVENTORY,
        EvidenceCollectionPhase.SOURCE_IDENTIFICATION,
        EvidenceCollectionPhase.DOCUMENT_COLLECTION,
        EvidenceCollectionPhase.QUALITY_GRADING,
        EvidenceCollectionPhase.PACKAGE_BUILD,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize EvidenceCollectionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._sources: List[EmissionSourceRecord] = []
        self._evidence_items: List[EvidenceItem] = []
        self._package_index: List[PackageIndex] = []
        self._package_summary: Optional[PackageSummary] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: EvidenceCollectionInput,
    ) -> EvidenceCollectionResult:
        """
        Execute the 5-phase evidence collection workflow.

        Args:
            input_data: Organisation sources, documents, and data bridges.

        Returns:
            EvidenceCollectionResult with evidence package and quality grades.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting evidence collection %s org=%s sources=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.emission_sources),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_scope_inventory,
            self._phase_2_source_identification,
            self._phase_3_document_collection,
            self._phase_4_quality_grading,
            self._phase_5_package_build,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Evidence collection failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        completeness = "0.00"
        if self._package_summary:
            completeness = self._package_summary.completeness_pct

        result = EvidenceCollectionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            source_inventory=self._sources,
            evidence_items=self._evidence_items,
            package_index=self._package_index,
            package_summary=self._package_summary,
            total_sources=len(self._sources),
            total_evidence_items=len(self._evidence_items),
            completeness_pct=completeness,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Evidence collection %s completed in %.2fs status=%s sources=%d evidence=%d",
            self.workflow_id, elapsed, overall_status.value,
            len(self._sources), len(self._evidence_items),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Scope Inventory
    # -------------------------------------------------------------------------

    async def _phase_1_scope_inventory(
        self, input_data: EvidenceCollectionInput,
    ) -> PhaseResult:
        """Inventory all emission sources across scopes."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._sources = []
        scope_counts: Dict[str, int] = {"scope_1": 0, "scope_2": 0, "scope_3": 0}

        for src_data in input_data.emission_sources:
            scope_str = src_data.get("scope", "scope_1")
            try:
                scope = EmissionScope(scope_str)
            except ValueError:
                scope = EmissionScope.SCOPE_1
                warnings.append(f"Unknown scope '{scope_str}', defaulting to scope_1")

            # Skip scope 3 if not included
            if scope == EmissionScope.SCOPE_3 and not input_data.include_scope_3:
                continue

            source_type_str = src_data.get("source_type", "other")
            try:
                source_type = SourceType(source_type_str)
            except ValueError:
                source_type = SourceType.OTHER

            source = EmissionSourceRecord(
                scope=scope,
                source_type=source_type,
                facility_name=src_data.get("facility_name", ""),
                description=src_data.get("description", ""),
                emissions_tco2e=float(src_data.get("emissions_tco2e", 0.0)),
                activity_data_value=float(src_data.get("activity_data_value", 0.0)),
                activity_data_unit=src_data.get("activity_data_unit", ""),
                data_origin=src_data.get("data_origin", ""),
                scope_3_category=int(src_data.get("scope_3_category", 0)),
            )
            src_hash_data = {
                "scope": scope.value, "type": source_type.value,
                "emissions": source.emissions_tco2e,
            }
            source.provenance_hash = _compute_hash(src_hash_data)
            self._sources.append(source)
            scope_counts[scope.value] = scope_counts.get(scope.value, 0) + 1

        total_emissions = sum(s.emissions_tco2e for s in self._sources)

        outputs["total_sources"] = len(self._sources)
        outputs["scope_1_sources"] = scope_counts.get("scope_1", 0)
        outputs["scope_2_sources"] = scope_counts.get("scope_2", 0)
        outputs["scope_3_sources"] = scope_counts.get("scope_3", 0)
        outputs["total_emissions_tco2e"] = round(total_emissions, 2)

        if not self._sources:
            warnings.append("No emission sources provided; evidence collection limited")

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 ScopeInventory: %d sources (S1=%d S2=%d S3=%d) total=%.0f tCO2e",
            len(self._sources), scope_counts.get("scope_1", 0),
            scope_counts.get("scope_2", 0), scope_counts.get("scope_3", 0),
            total_emissions,
        )
        return PhaseResult(
            phase_name="scope_inventory", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Source Identification
    # -------------------------------------------------------------------------

    async def _phase_2_source_identification(
        self, input_data: EvidenceCollectionInput,
    ) -> PhaseResult:
        """Identify required evidence types for each emission source."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_evidence_required = 0
        for source in self._sources:
            st_key = source.source_type.value
            required_types = SOURCE_EVIDENCE_MAP.get(st_key, ["calculation_sheet"])
            source.required_evidence_types = required_types
            total_evidence_required += len(required_types)

        outputs["sources_mapped"] = len(self._sources)
        outputs["total_evidence_items_required"] = total_evidence_required
        outputs["avg_evidence_per_source"] = (
            round(total_evidence_required / max(len(self._sources), 1), 1)
        )

        # Count by evidence type
        type_counts: Dict[str, int] = {}
        for source in self._sources:
            for et in source.required_evidence_types:
                type_counts[et] = type_counts.get(et, 0) + 1
        outputs["evidence_type_distribution"] = type_counts

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 SourceIdentification: %d sources -> %d evidence items required",
            len(self._sources), total_evidence_required,
        )
        return PhaseResult(
            phase_name="source_identification", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Document Collection
    # -------------------------------------------------------------------------

    async def _phase_3_document_collection(
        self, input_data: EvidenceCollectionInput,
    ) -> PhaseResult:
        """Collect/link evidence items from data bridges."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build lookup of existing documents by source_id + type
        doc_lookup: Dict[str, Dict[str, Any]] = {}
        for doc in input_data.existing_documents:
            key = f"{doc.get('source_id', '')}|{doc.get('evidence_type', '')}"
            doc_lookup[key] = doc

        self._evidence_items = []
        collected = 0
        linked = 0
        pending = 0
        unavailable = 0

        for source in self._sources:
            for ev_type_str in source.required_evidence_types:
                try:
                    ev_type = EvidenceType(ev_type_str)
                except ValueError:
                    ev_type = EvidenceType.CALCULATION_SHEET

                lookup_key = f"{source.source_id}|{ev_type_str}"
                existing = doc_lookup.get(lookup_key)

                if existing:
                    status = DocumentStatus.COLLECTED
                    if existing.get("linked", False):
                        status = DocumentStatus.LINKED
                        linked += 1
                    else:
                        collected += 1

                    item = EvidenceItem(
                        source_id=source.source_id,
                        evidence_type=ev_type,
                        document_name=existing.get("document_name", f"{ev_type_str}_doc"),
                        document_format=existing.get("format", "pdf"),
                        document_status=status,
                        access_path=existing.get("access_path", ""),
                        data_bridge_source=existing.get("bridge", ""),
                        provenance_score=float(existing.get("provenance_score", 50.0)),
                        completeness_score=float(existing.get("completeness_score", 50.0)),
                        timeliness_score=float(existing.get("timeliness_score", 50.0)),
                        corroboration_score=float(existing.get("corroboration_score", 50.0)),
                    )
                elif ev_type_str in ("invoice", "utility_bill", "meter_reading") and \
                        "erp" in input_data.data_bridge_connections:
                    item = EvidenceItem(
                        source_id=source.source_id,
                        evidence_type=ev_type,
                        document_name=f"{ev_type_str}_from_erp",
                        document_format="electronic",
                        document_status=DocumentStatus.LINKED,
                        data_bridge_source="erp",
                        provenance_score=70.0,
                        completeness_score=60.0,
                        timeliness_score=80.0,
                        corroboration_score=50.0,
                    )
                    linked += 1
                else:
                    item = EvidenceItem(
                        source_id=source.source_id,
                        evidence_type=ev_type,
                        document_name=f"{ev_type_str}_pending",
                        document_status=DocumentStatus.PENDING,
                    )
                    pending += 1

                # Generate document hash
                doc_data = {
                    "source": source.source_id, "type": ev_type_str,
                    "name": item.document_name,
                }
                item.document_hash = _compute_hash(doc_data)
                self._evidence_items.append(item)

        outputs["total_evidence_items"] = len(self._evidence_items)
        outputs["collected"] = collected
        outputs["linked"] = linked
        outputs["pending"] = pending
        outputs["unavailable"] = unavailable
        outputs["data_bridges_used"] = input_data.data_bridge_connections

        if pending > len(self._evidence_items) * 0.3:
            warnings.append(
                f"More than 30% of evidence items pending collection "
                f"({pending}/{len(self._evidence_items)})"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 DocumentCollection: %d items (collected=%d linked=%d pending=%d)",
            len(self._evidence_items), collected, linked, pending,
        )
        return PhaseResult(
            phase_name="document_collection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Quality Grading
    # -------------------------------------------------------------------------

    async def _phase_4_quality_grading(
        self, input_data: EvidenceCollectionInput,
    ) -> PhaseResult:
        """Grade each evidence item on provenance, completeness, timeliness, corroboration."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        grade_counts: Dict[str, int] = {g.value: 0 for g in EvidenceQualityGrade}

        for item in self._evidence_items:
            if item.document_status in (DocumentStatus.PENDING, DocumentStatus.UNAVAILABLE):
                item.quality_grade = EvidenceQualityGrade.INSUFFICIENT
                item.overall_quality_score = "0.00"
                grade_counts["insufficient"] += 1
                continue

            # Compute weighted quality score
            provenance = Decimal(str(item.provenance_score))
            completeness = Decimal(str(item.completeness_score))
            timeliness = Decimal(str(item.timeliness_score))
            corroboration = Decimal(str(item.corroboration_score))

            weighted = (
                provenance * QUALITY_CRITERIA_WEIGHTS["provenance"]
                + completeness * QUALITY_CRITERIA_WEIGHTS["completeness"]
                + timeliness * QUALITY_CRITERIA_WEIGHTS["timeliness"]
                + corroboration * QUALITY_CRITERIA_WEIGHTS["corroboration"]
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            item.overall_quality_score = str(weighted)
            item.quality_grade = self._classify_quality_grade(float(weighted))

            grade_data = {
                "id": item.evidence_id, "score": str(weighted),
                "grade": item.quality_grade.value,
            }
            item.provenance_hash = _compute_hash(grade_data)
            grade_counts[item.quality_grade.value] += 1

        outputs["items_graded"] = len(self._evidence_items)
        outputs["grade_distribution"] = grade_counts
        outputs["excellent_pct"] = round(
            grade_counts.get("excellent", 0) / max(len(self._evidence_items), 1) * 100.0, 1,
        )
        outputs["insufficient_pct"] = round(
            grade_counts.get("insufficient", 0) / max(len(self._evidence_items), 1) * 100.0, 1,
        )

        # Average quality score for collected items
        scored_items = [
            i for i in self._evidence_items
            if i.document_status not in (DocumentStatus.PENDING, DocumentStatus.UNAVAILABLE)
        ]
        if scored_items:
            avg_score = sum(
                Decimal(i.overall_quality_score) for i in scored_items
            ) / Decimal(str(len(scored_items)))
            outputs["avg_quality_score"] = str(
                avg_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            )
        else:
            outputs["avg_quality_score"] = "0.00"

        if grade_counts.get("insufficient", 0) > len(self._evidence_items) * 0.2:
            warnings.append("More than 20% of evidence items graded INSUFFICIENT")

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 QualityGrading: %d items graded, dist=%s",
            len(self._evidence_items), grade_counts,
        )
        return PhaseResult(
            phase_name="quality_grading", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Package Build
    # -------------------------------------------------------------------------

    async def _phase_5_package_build(
        self, input_data: EvidenceCollectionInput,
    ) -> PhaseResult:
        """Build consolidated evidence package with index and hashes."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._package_index = []
        for idx, item in enumerate(self._evidence_items, start=1):
            self._package_index.append(PackageIndex(
                index_number=idx,
                source_id=item.source_id,
                evidence_id=item.evidence_id,
                document_name=item.document_name,
                quality_grade=item.quality_grade.value,
                document_hash=item.document_hash,
                checklist_reference=f"EV-{idx:04d}",
            ))

        # Package summary
        collected = sum(
            1 for i in self._evidence_items
            if i.document_status in (DocumentStatus.COLLECTED, DocumentStatus.LINKED)
        )
        pending = sum(
            1 for i in self._evidence_items if i.document_status == DocumentStatus.PENDING
        )
        unavailable = sum(
            1 for i in self._evidence_items if i.document_status == DocumentStatus.UNAVAILABLE
        )
        total = len(self._evidence_items)

        completeness = Decimal("0.00")
        if total > 0:
            completeness = (
                Decimal(str(collected)) / Decimal(str(total)) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        scored_items = [
            i for i in self._evidence_items
            if i.document_status not in (DocumentStatus.PENDING, DocumentStatus.UNAVAILABLE)
        ]
        avg_quality = Decimal("0.00")
        if scored_items:
            avg_quality = (
                sum(Decimal(i.overall_quality_score) for i in scored_items)
                / Decimal(str(len(scored_items)))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Package hash = hash of all document hashes
        all_doc_hashes = "|".join(i.document_hash for i in self._evidence_items)
        package_hash = hashlib.sha256(all_doc_hashes.encode("utf-8")).hexdigest()

        summary_data = {
            "total": total, "collected": collected,
            "completeness": str(completeness),
        }
        self._package_summary = PackageSummary(
            total_sources=len(self._sources),
            total_evidence_items=total,
            collected_count=collected,
            pending_count=pending,
            unavailable_count=unavailable,
            completeness_pct=str(completeness),
            avg_quality_score=str(avg_quality),
            package_hash=package_hash,
            provenance_hash=_compute_hash(summary_data),
        )

        outputs["package_index_entries"] = len(self._package_index)
        outputs["completeness_pct"] = str(completeness)
        outputs["avg_quality_score"] = str(avg_quality)
        outputs["package_hash"] = package_hash

        if float(completeness) < 80.0:
            warnings.append(
                f"Evidence package completeness ({completeness}%) below 80% threshold"
            )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 PackageBuild: %d entries, completeness=%s%%, hash=%s",
            len(self._package_index), str(completeness), package_hash[:16],
        )
        return PhaseResult(
            phase_name="package_build", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: EvidenceCollectionInput,
        phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio

                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Calculation Helpers
    # -------------------------------------------------------------------------

    def _classify_quality_grade(self, score: float) -> EvidenceQualityGrade:
        """Classify quality score into evidence quality grade."""
        for grade_name, (lower, upper) in QUALITY_GRADE_THRESHOLDS.items():
            if lower <= score < upper:
                return EvidenceQualityGrade(grade_name)
        return EvidenceQualityGrade.INSUFFICIENT

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._sources = []
        self._evidence_items = []
        self._package_index = []
        self._package_summary = None

    def _compute_provenance(self, result: EvidenceCollectionResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.total_sources}|{result.completeness_pct}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
