# -*- coding: utf-8 -*-
"""
Mitigation Measure Designer Models - AGENT-EUDR-029

Pydantic v2 models for mitigation measure design, strategy orchestration,
effectiveness estimation, implementation tracking, verification, workflow
state management, and mitigation reporting.

All models use Decimal for numeric scores to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field
from greenlang.schemas import GreenLangBase


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EUDRCommodity(str, enum.Enum):
    """EUDR regulated commodities (Article 1)."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class RiskLevel(str, enum.Enum):
    """Risk classification levels per EUDR Article 10(2)."""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class Article11Category(str, enum.Enum):
    """EUDR Article 11 mitigation measure categories.

    Article 11 specifies the types of additional measures operators
    must take when risk assessment identifies non-negligible risk:
    - Additional information gathering
    - Independent survey/audit
    - Other risk mitigation measures
    """

    ADDITIONAL_INFO = "additional_info"
    INDEPENDENT_AUDIT = "independent_audit"
    OTHER_MEASURES = "other_measures"


class MeasureStatus(str, enum.Enum):
    """Lifecycle status of a single mitigation measure."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class WorkflowStatus(str, enum.Enum):
    """Lifecycle status of the overall mitigation workflow."""

    INITIATED = "initiated"
    STRATEGY_DESIGNED = "strategy_designed"
    MEASURES_APPROVED = "measures_approved"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    CLOSED = "closed"
    ESCALATED = "escalated"
    FAILED = "failed"


class MeasurePriority(str, enum.Enum):
    """Priority classification for mitigation measures."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EffectivenessLevel(str, enum.Enum):
    """Projected effectiveness classification for a mitigation measure."""

    HIGH_IMPACT = "high_impact"
    MEDIUM_IMPACT = "medium_impact"
    LOW_IMPACT = "low_impact"
    MINIMAL_IMPACT = "minimal_impact"


class VerificationResult(str, enum.Enum):
    """Outcome of post-implementation verification assessment."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class RiskDimension(str, enum.Enum):
    """Risk dimensions contributing to the composite risk score."""

    COUNTRY = "country"
    COMMODITY = "commodity"
    SUPPLIER = "supplier"
    DEFORESTATION = "deforestation"
    CORRUPTION = "corruption"
    SUPPLY_CHAIN_COMPLEXITY = "supply_chain_complexity"
    MIXING_RISK = "mixing_risk"
    CIRCUMVENTION_RISK = "circumvention_risk"


class EvidenceType(str, enum.Enum):
    """Types of evidence supporting mitigation measure completion."""

    DOCUMENT = "document"
    CERTIFICATE = "certificate"
    AUDIT_REPORT = "audit_report"
    SATELLITE_IMAGE = "satellite_image"
    SITE_VISIT_REPORT = "site_visit_report"
    SUPPLIER_DECLARATION = "supplier_declaration"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-MMD-029"
AGENT_VERSION = "1.0.0"

DEFAULT_RISK_WEIGHTS: Dict[RiskDimension, Decimal] = {
    RiskDimension.COUNTRY: Decimal("0.20"),
    RiskDimension.COMMODITY: Decimal("0.15"),
    RiskDimension.SUPPLIER: Decimal("0.20"),
    RiskDimension.DEFORESTATION: Decimal("0.20"),
    RiskDimension.CORRUPTION: Decimal("0.10"),
    RiskDimension.SUPPLY_CHAIN_COMPLEXITY: Decimal("0.05"),
    RiskDimension.MIXING_RISK: Decimal("0.05"),
    RiskDimension.CIRCUMVENTION_RISK: Decimal("0.05"),
}

RISK_THRESHOLDS: Dict[RiskLevel, Decimal] = {
    RiskLevel.NEGLIGIBLE: Decimal("15"),
    RiskLevel.LOW: Decimal("30"),
    RiskLevel.STANDARD: Decimal("60"),
    RiskLevel.HIGH: Decimal("80"),
    RiskLevel.CRITICAL: Decimal("100"),
}

SUPPORTED_COMMODITIES: List[str] = [c.value for c in EUDRCommodity]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RiskTrigger(GreenLangBase):
    """Upstream risk assessment result that triggers mitigation design.

    Captures the composite risk score and per-dimension breakdown
    from AGENT-EUDR-028 that warrants mitigation action.
    """

    assessment_id: str = Field(..., description="Source risk assessment ID")
    operator_id: str = Field(..., description="Operator under assessment")
    commodity: EUDRCommodity
    composite_score: Decimal = Field(
        ..., ge=0, le=100, description="Composite risk score 0-100"
    )
    risk_level: RiskLevel
    risk_dimensions: Dict[RiskDimension, Decimal] = Field(
        default_factory=dict,
        description="Per-dimension risk scores",
    )
    triggered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class MeasureTemplate(GreenLangBase):
    """Reusable template for a mitigation measure.

    Templates define the standard structure for mitigation measures
    aligned to EUDR Article 11 categories with expected effectiveness,
    applicable dimensions, and evidence requirements.
    """

    template_id: str = Field(..., description="Unique template identifier")
    title: str = Field(..., description="Human-readable measure title")
    description: str = Field(default="", description="Detailed description")
    article11_category: Article11Category
    applicable_dimensions: List[RiskDimension] = Field(
        default_factory=list,
        description="Risk dimensions this measure addresses",
    )
    applicable_commodities: List[EUDRCommodity] = Field(
        default_factory=list,
        description="Commodities this template applies to (empty = all)",
    )
    base_effectiveness: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=100,
        description="Expected base risk reduction percentage",
    )
    typical_timeline_days: int = Field(
        default=30, ge=1, description="Typical implementation timeline in days"
    )
    evidence_requirements: List[str] = Field(
        default_factory=list,
        description="Evidence types required for verification",
    )
    regulatory_reference: str = Field(
        default="",
        description="EUDR article/section reference",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class MitigationMeasure(GreenLangBase):
    """A single mitigation measure within a strategy.

    Tracks the full lifecycle from proposal through verification,
    including assignment, deadlines, evidence collection, and
    actual vs. expected risk reduction.
    """

    measure_id: str = Field(..., description="Unique measure identifier")
    strategy_id: str = Field(..., description="Parent strategy ID")
    template_id: Optional[str] = Field(
        default=None, description="Source template ID if template-based"
    )
    title: str = Field(..., description="Measure title")
    description: str = Field(default="", description="Detailed description")
    article11_category: Article11Category
    target_dimension: RiskDimension
    status: MeasureStatus = MeasureStatus.PROPOSED
    priority: MeasurePriority = MeasurePriority.MEDIUM
    assigned_to: Optional[str] = Field(
        default=None, description="User/team assigned to implement"
    )
    deadline: Optional[datetime] = Field(
        default=None, description="Implementation deadline"
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence items supporting completion",
    )
    expected_risk_reduction: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=100,
        description="Expected risk reduction percentage",
    )
    actual_risk_reduction: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=100,
        description="Actual risk reduction after verification",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class MitigationStrategy(GreenLangBase):
    """Collection of mitigation measures forming a coherent strategy.

    Designed to reduce the composite risk score from pre-mitigation
    level down to the target score, with tracking of post-mitigation
    outcomes.
    """

    strategy_id: str = Field(..., description="Unique strategy identifier")
    workflow_id: str = Field(..., description="Parent workflow ID")
    risk_trigger: RiskTrigger
    measures: List[MitigationMeasure] = Field(
        default_factory=list,
        description="Ordered list of mitigation measures",
    )
    pre_mitigation_score: Decimal = Field(
        ..., ge=0, le=100, description="Score before mitigation"
    )
    target_score: Decimal = Field(
        ..., ge=0, le=100, description="Target score after mitigation"
    )
    post_mitigation_score: Optional[Decimal] = Field(
        default=None, ge=0, le=100, description="Actual score after verification"
    )
    status: WorkflowStatus = WorkflowStatus.STRATEGY_DESIGNED
    designed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    designed_by: str = Field(default=AGENT_ID, description="Agent or user who designed")
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class EffectivenessEstimate(GreenLangBase):
    """Three-scenario effectiveness projection for a measure.

    Provides conservative, moderate, and optimistic estimates of
    risk reduction with applicability factor and confidence level.
    """

    estimate_id: str = Field(..., description="Unique estimate identifier")
    measure_id: str = Field(..., description="Associated measure ID")
    conservative: Decimal = Field(
        ..., ge=0, le=100, description="Conservative risk reduction estimate"
    )
    moderate: Decimal = Field(
        ..., ge=0, le=100, description="Moderate risk reduction estimate"
    )
    optimistic: Decimal = Field(
        ..., ge=0, le=100, description="Optimistic risk reduction estimate"
    )
    applicability_factor: Decimal = Field(
        default=Decimal("1.00"),
        ge=0,
        le=1,
        description="Factor adjusting effectiveness for context",
    )
    confidence: Decimal = Field(
        default=Decimal("0.50"),
        ge=0,
        le=1,
        description="Confidence level in the estimate",
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class VerificationReport(GreenLangBase):
    """Post-implementation verification of mitigation effectiveness.

    Compares pre- and post-mitigation risk scores to determine
    whether the strategy achieved sufficient risk reduction.
    """

    verification_id: str = Field(..., description="Unique verification ID")
    strategy_id: str = Field(..., description="Strategy being verified")
    pre_score: Decimal = Field(
        ..., ge=0, le=100, description="Risk score before mitigation"
    )
    post_score: Decimal = Field(
        ..., ge=0, le=100, description="Risk score after mitigation"
    )
    risk_reduction: Decimal = Field(
        ..., description="Absolute risk score reduction"
    )
    result: VerificationResult
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    verified_by: str = Field(default="", description="Verifier identity")
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class WorkflowState(GreenLangBase):
    """Top-level workflow state for the mitigation lifecycle.

    Tracks the overall workflow from initiation through closure,
    including escalation and failure states.
    """

    workflow_id: str = Field(..., description="Unique workflow identifier")
    operator_id: str = Field(..., description="Operator under mitigation")
    commodity: EUDRCommodity
    status: WorkflowStatus = WorkflowStatus.INITIATED
    strategy_id: Optional[str] = Field(
        default=None, description="Associated strategy ID"
    )
    current_phase: str = Field(
        default="initiation", description="Current workflow phase"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    closed_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ImplementationMilestone(GreenLangBase):
    """A milestone within a mitigation measure implementation.

    Tracks intermediate deliverables and checkpoints during
    measure execution.
    """

    milestone_id: str = Field(..., description="Unique milestone identifier")
    measure_id: str = Field(..., description="Parent measure ID")
    title: str = Field(..., description="Milestone title")
    description: str = Field(default="", description="Milestone description")
    due_date: Optional[datetime] = Field(
        default=None, description="Target completion date"
    )
    completed_at: Optional[datetime] = None
    status: MeasureStatus = MeasureStatus.PROPOSED

    model_config = {"frozen": False, "extra": "ignore"}


class MeasureEvidence(GreenLangBase):
    """Evidence item supporting mitigation measure completion.

    Documents, certificates, audit reports, satellite imagery,
    and other evidence types collected during implementation.
    """

    evidence_id: str = Field(..., description="Unique evidence identifier")
    measure_id: str = Field(..., description="Associated measure ID")
    evidence_type: EvidenceType
    title: str = Field(..., description="Evidence title")
    file_reference: str = Field(
        default="", description="File path or object storage key"
    )
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    uploaded_by: str = Field(default="", description="Uploader identity")

    model_config = {"frozen": False, "extra": "ignore"}


class MeasureSummary(GreenLangBase):
    """Summary of a single measure for inclusion in reports."""

    measure_id: str
    title: str
    article11_category: Article11Category
    target_dimension: RiskDimension
    status: MeasureStatus
    priority: MeasurePriority
    expected_risk_reduction: Decimal = Decimal("0")
    actual_risk_reduction: Optional[Decimal] = None


class MitigationReport(GreenLangBase):
    """Complete mitigation report for audit and DDS submission.

    Summarizes the mitigation strategy, measure outcomes,
    verification results, and provenance chain for regulatory
    record-keeping per EUDR Article 31.
    """

    report_id: str = Field(..., description="Unique report identifier")
    strategy_id: str = Field(..., description="Associated strategy ID")
    operator_id: str = Field(..., description="Operator identifier")
    commodity: EUDRCommodity
    pre_score: Decimal = Field(
        ..., ge=0, le=100, description="Pre-mitigation risk score"
    )
    post_score: Decimal = Field(
        ..., ge=0, le=100, description="Post-mitigation risk score"
    )
    measures_summary: List[MeasureSummary] = Field(
        default_factory=list,
        description="Summary of all measures in the strategy",
    )
    verification_result: Optional[VerificationResult] = None
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(GreenLangBase):
    """Health check response for the Mitigation Measure Designer."""

    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}
