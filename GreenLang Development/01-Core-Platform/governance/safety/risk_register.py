r"""
RiskRegister - Comprehensive Risk Register Management

This module implements a full-featured Risk Register for process safety management
per IEC 61511, IEC 61882 (HAZOP), and IEC 60812 (FMEA) standards.

Key Features:
- RiskEntry model with ID, description, likelihood, severity, controls, and residual risk
- RiskRegister class with full CRUD operations
- Risk ranking and prioritization algorithms
- Risk acceptance workflow with approval levels
- Export to standard formats (CSV, JSON, Excel-compatible)
- Integration with HAZOP, FMEA, and LOPA outputs
- Complete audit trail with provenance tracking

Risk Level Matrix (5x5):
    Severity\Likelihood   1        2        3        4        5
    1 (Minor)             LOW      LOW      LOW      MEDIUM   MEDIUM
    2 (Significant)       LOW      LOW      MEDIUM   MEDIUM   HIGH
    3 (Serious)           LOW      MEDIUM   MEDIUM   HIGH     HIGH
    4 (Major)             MEDIUM   MEDIUM   HIGH     HIGH     CRITICAL
    5 (Catastrophic)      MEDIUM   HIGH     HIGH     CRITICAL CRITICAL

Reference:
- IEC 61511-1:2016 - Functional Safety
- IEC 61882:2016 - Hazard and Operability Studies (HAZOP)
- IEC 60812:2018 - Failure Mode and Effects Analysis (FMEA)
- ISO 31000:2018 - Risk Management

Example:
    >>> from greenlang.safety.risk_register import RiskRegister, RiskEntry
    >>> register = RiskRegister()
    >>> entry = RiskEntry(
    ...     title="High temperature excursion in reactor",
    ...     description="Temperature exceeds design limits during exothermic reaction",
    ...     category=RiskCategory.SAFETY,
    ...     severity=4,
    ...     likelihood=3,
    ...     controls=["Temperature alarm", "Emergency shutdown system"]
    ... )
    >>> register.add_risk(entry)
"""

from typing import Dict, List, Optional, Any, ClassVar, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
import json
import csv
from io import StringIO
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Risk classification per IEC 61511."""

    SAFETY = "safety"  # Personnel, equipment injury
    ENVIRONMENTAL = "environmental"  # Emissions, spills, pollution
    OPERATIONAL = "operational"  # Downtime, quality loss
    COMPLIANCE = "compliance"  # Regulatory, financial penalties
    FINANCIAL = "financial"  # Financial impact
    REPUTATION = "reputation"  # Reputational damage


class RiskStatus(str, Enum):
    """Risk management status."""

    IDENTIFIED = "identified"  # Newly identified
    ASSESSED = "assessed"  # Assessment complete
    MITIGATION_PLANNED = "mitigation_planned"  # Mitigation in planning
    MITIGATION_IN_PROGRESS = "mitigation_in_progress"  # Mitigation underway
    MITIGATED = "mitigated"  # Mitigation implemented
    MONITORING = "monitoring"  # Ongoing monitoring
    ACCEPTED = "accepted"  # Risk formally accepted
    CLOSED = "closed"  # Risk no longer applicable
    TRANSFERRED = "transferred"  # Risk transferred (e.g., insurance)


class RiskSource(str, Enum):
    """Source of risk identification."""

    HAZOP = "hazop"
    FMEA = "fmea"
    LOPA = "lopa"
    INCIDENT = "incident"
    AUDIT = "audit"
    INSPECTION = "inspection"
    REVIEW = "review"
    DESIGN_REVIEW = "design_review"
    WHAT_IF = "what_if"
    CHECKLIST = "checklist"
    EXTERNAL = "external"
    OTHER = "other"


class AcceptanceLevel(str, Enum):
    """Authority level required for risk acceptance."""

    SUPERVISOR = "supervisor"  # LOW risk
    MANAGER = "manager"  # MEDIUM risk
    DIRECTOR = "director"  # HIGH risk
    EXECUTIVE = "executive"  # CRITICAL risk
    BOARD = "board"  # Exceptional cases


class SafetyIntegrityLevel(str, Enum):
    """IEC 61511 Safety Integrity Level mapping."""

    SIL_4 = "sil_4"  # 10,000-100,000x risk reduction
    SIL_3 = "sil_3"  # 1,000-10,000x risk reduction
    SIL_2 = "sil_2"  # 100-1,000x risk reduction
    SIL_1 = "sil_1"  # 10-100x risk reduction
    NO_SIL = "no_sil"  # No SIL required


# =============================================================================
# RISK MATRIX DEFINITION
# =============================================================================

class RiskMatrix:
    """
    5x5 Risk Matrix per IEC 61511.

    Static utility class for risk level calculations and mappings.
    """

    # Risk level matrix: [severity-1][likelihood-1]
    RISK_MATRIX: ClassVar[List[List[RiskLevel]]] = [
        [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM],
        [RiskLevel.LOW, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH],
        [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH],
        [RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL],
        [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.CRITICAL],
    ]

    # Color mapping for visualization
    COLOR_MAP: ClassVar[Dict[RiskLevel, str]] = {
        RiskLevel.LOW: "green",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "orange",
        RiskLevel.CRITICAL: "red",
    }

    # Numeric score for sorting/ranking
    RISK_SCORE: ClassVar[Dict[RiskLevel, int]] = {
        RiskLevel.LOW: 1,
        RiskLevel.MEDIUM: 2,
        RiskLevel.HIGH: 3,
        RiskLevel.CRITICAL: 4,
    }

    # SIL mapping per IEC 61511
    SIL_MAP: ClassVar[Dict[RiskLevel, SafetyIntegrityLevel]] = {
        RiskLevel.LOW: SafetyIntegrityLevel.NO_SIL,
        RiskLevel.MEDIUM: SafetyIntegrityLevel.SIL_1,
        RiskLevel.HIGH: SafetyIntegrityLevel.SIL_2,
        RiskLevel.CRITICAL: SafetyIntegrityLevel.SIL_3,
    }

    # Acceptance criteria (days for action)
    ACCEPTANCE_DAYS: ClassVar[Dict[RiskLevel, int]] = {
        RiskLevel.CRITICAL: 7,
        RiskLevel.HIGH: 30,
        RiskLevel.MEDIUM: 90,
        RiskLevel.LOW: 365,
    }

    # Acceptance authority
    ACCEPTANCE_AUTHORITY: ClassVar[Dict[RiskLevel, AcceptanceLevel]] = {
        RiskLevel.LOW: AcceptanceLevel.SUPERVISOR,
        RiskLevel.MEDIUM: AcceptanceLevel.MANAGER,
        RiskLevel.HIGH: AcceptanceLevel.DIRECTOR,
        RiskLevel.CRITICAL: AcceptanceLevel.EXECUTIVE,
    }

    @staticmethod
    def calculate_risk_level(severity: int, likelihood: int) -> RiskLevel:
        """
        Calculate risk level from severity and likelihood.

        Args:
            severity: 1-5 (1=Minor, 5=Catastrophic)
            likelihood: 1-5 (1=Remote, 5=Almost Certain)

        Returns:
            RiskLevel: LOW, MEDIUM, HIGH, or CRITICAL

        Raises:
            ValueError: If scores are outside 1-5 range
        """
        if not (1 <= severity <= 5 and 1 <= likelihood <= 5):
            raise ValueError("Severity and likelihood must be 1-5")

        return RiskMatrix.RISK_MATRIX[severity - 1][likelihood - 1]

    @staticmethod
    def get_risk_score(severity: int, likelihood: int) -> int:
        """Calculate numeric risk score (1-25) for ranking."""
        return severity * likelihood

    @staticmethod
    def get_risk_color(risk_level: RiskLevel) -> str:
        """Get visualization color for risk level."""
        return RiskMatrix.COLOR_MAP.get(risk_level, "gray")

    @staticmethod
    def get_required_sil(risk_level: RiskLevel) -> SafetyIntegrityLevel:
        """Get required Safety Integrity Level per IEC 61511."""
        return RiskMatrix.SIL_MAP.get(risk_level, SafetyIntegrityLevel.NO_SIL)

    @staticmethod
    def get_acceptance_days(risk_level: RiskLevel) -> int:
        """Get days allowed for risk mitigation."""
        return RiskMatrix.ACCEPTANCE_DAYS.get(risk_level, 365)

    @staticmethod
    def get_acceptance_authority(risk_level: RiskLevel) -> AcceptanceLevel:
        """Get required authority level for acceptance."""
        return RiskMatrix.ACCEPTANCE_AUTHORITY.get(
            risk_level, AcceptanceLevel.SUPERVISOR
        )


# =============================================================================
# DATA MODELS
# =============================================================================

class Control(BaseModel):
    """Risk control measure."""

    control_id: str = Field(
        default_factory=lambda: f"CTL-{uuid.uuid4().hex[:6].upper()}",
        description="Unique control identifier"
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Control description"
    )
    control_type: str = Field(
        default="preventive",
        description="Type: preventive, detective, corrective"
    )
    effectiveness: str = Field(
        default="medium",
        description="Effectiveness: high, medium, low"
    )
    is_implemented: bool = Field(
        default=False,
        description="Whether control is implemented"
    )
    implementation_date: Optional[datetime] = Field(
        None,
        description="Date control was implemented"
    )
    owner: str = Field(
        default="",
        description="Control owner"
    )
    verification_method: str = Field(
        default="",
        description="How control effectiveness is verified"
    )
    last_verified: Optional[datetime] = Field(
        None,
        description="Date of last verification"
    )


class RiskAcceptance(BaseModel):
    """Risk acceptance record."""

    acceptance_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique acceptance identifier"
    )
    accepted_by: str = Field(
        ...,
        description="Person accepting risk"
    )
    acceptance_level: AcceptanceLevel = Field(
        ...,
        description="Authority level of acceptor"
    )
    acceptance_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date of acceptance"
    )
    justification: str = Field(
        default="",
        description="Justification for acceptance"
    )
    review_date: Optional[datetime] = Field(
        None,
        description="Date for acceptance review"
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions for acceptance"
    )


class RiskTrend(BaseModel):
    """Risk trend data point."""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp"
    )
    severity: int = Field(..., ge=1, le=5)
    likelihood: int = Field(..., ge=1, le=5)
    risk_level: RiskLevel = Field(...)
    risk_score: int = Field(...)
    status: RiskStatus = Field(...)
    note: str = Field(default="")


class RiskEntry(BaseModel):
    """Comprehensive risk register entry."""

    risk_id: str = Field(
        default_factory=lambda: f"RISK-{uuid.uuid4().hex[:8].upper()}",
        description="Unique risk identifier"
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Risk title"
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed risk description"
    )

    # Classification
    category: RiskCategory = Field(
        default=RiskCategory.OPERATIONAL,
        description="Risk category"
    )
    status: RiskStatus = Field(
        default=RiskStatus.IDENTIFIED,
        description="Current status"
    )

    # Source traceability
    source: RiskSource = Field(
        default=RiskSource.OTHER,
        description="Source of identification"
    )
    source_reference: str = Field(
        default="",
        description="Reference ID from source"
    )
    linked_hazard_ids: List[str] = Field(
        default_factory=list,
        description="Linked hazard IDs"
    )
    linked_action_ids: List[str] = Field(
        default_factory=list,
        description="Linked action IDs"
    )

    # Initial (inherent) risk assessment
    severity: int = Field(
        ...,
        ge=1,
        le=5,
        description="Severity 1-5 (before controls)"
    )
    likelihood: int = Field(
        ...,
        ge=1,
        le=5,
        description="Likelihood 1-5 (before controls)"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW,
        description="Calculated risk level"
    )
    risk_score: int = Field(
        default=1,
        description="Numeric risk score (1-25)"
    )

    # Controls
    controls: List[Control] = Field(
        default_factory=list,
        description="Control measures"
    )
    control_descriptions: List[str] = Field(
        default_factory=list,
        description="Simple control descriptions (legacy)"
    )

    # Residual risk (after controls)
    residual_severity: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Residual severity"
    )
    residual_likelihood: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Residual likelihood"
    )
    residual_risk_level: Optional[RiskLevel] = Field(
        None,
        description="Residual risk level"
    )
    residual_risk_score: Optional[int] = Field(
        None,
        description="Residual risk score"
    )

    # SIL requirements
    required_sil: SafetyIntegrityLevel = Field(
        default=SafetyIntegrityLevel.NO_SIL,
        description="Required SIL per IEC 61511"
    )

    # Ownership
    risk_owner: str = Field(
        default="",
        description="Person responsible for risk"
    )
    responsible_department: str = Field(
        default="",
        description="Responsible department"
    )

    # Dates
    identified_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date risk was identified"
    )
    assessment_date: Optional[datetime] = Field(
        None,
        description="Date risk was assessed"
    )
    target_mitigation_date: Optional[datetime] = Field(
        None,
        description="Target date for mitigation"
    )
    actual_mitigation_date: Optional[datetime] = Field(
        None,
        description="Actual mitigation completion date"
    )
    review_date: Optional[datetime] = Field(
        None,
        description="Next review date"
    )

    # Acceptance
    acceptance: Optional[RiskAcceptance] = Field(
        None,
        description="Risk acceptance record"
    )

    # Trend tracking
    trend_history: List[RiskTrend] = Field(
        default_factory=list,
        description="Risk trend history"
    )

    # Consequences and causes
    causes: List[str] = Field(
        default_factory=list,
        description="Root causes"
    )
    consequences: List[str] = Field(
        default_factory=list,
        description="Potential consequences"
    )

    # Metadata
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for filtering"
    )
    location: str = Field(
        default="",
        description="Physical location/area"
    )
    asset: str = Field(
        default="",
        description="Related asset/equipment"
    )
    notes: str = Field(
        default="",
        description="Additional notes"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {datetime: lambda v: v.isoformat()}

    @field_validator('severity', 'likelihood', 'residual_severity', 'residual_likelihood')
    @classmethod
    def validate_risk_scores(cls, v):
        """Validate risk scores are in valid range."""
        if v is not None and not (1 <= v <= 5):
            raise ValueError("Risk scores must be between 1 and 5")
        return v


class RiskRegisterConfig(BaseModel):
    """Configuration for risk register."""

    auto_calculate_risk_level: bool = Field(
        default=True,
        description="Automatically calculate risk level from severity/likelihood"
    )
    auto_assign_sil: bool = Field(
        default=True,
        description="Automatically assign SIL based on risk level"
    )
    auto_set_target_date: bool = Field(
        default=True,
        description="Auto-set target mitigation date based on risk level"
    )
    track_trend_history: bool = Field(
        default=True,
        description="Track risk trend history on updates"
    )
    require_residual_assessment: bool = Field(
        default=False,
        description="Require residual risk assessment for closure"
    )


# =============================================================================
# RISK REGISTER
# =============================================================================

class RiskRegister:
    """
    Comprehensive Risk Register for process safety management.

    Implements risk tracking, assessment, and reporting per IEC 61511
    and ISO 31000 standards.

    Attributes:
        risks: Dict of risk_id to RiskEntry
        config: RiskRegisterConfig
        audit_trail: List of audit events

    Example:
        >>> register = RiskRegister()
        >>> entry = RiskEntry(
        ...     title="Overpressure scenario",
        ...     category=RiskCategory.SAFETY,
        ...     severity=4,
        ...     likelihood=3
        ... )
        >>> register.add_risk(entry)
    """

    def __init__(self, config: Optional[RiskRegisterConfig] = None):
        """
        Initialize RiskRegister.

        Args:
            config: Optional register configuration
        """
        self.risks: Dict[str, RiskEntry] = {}
        self.config = config or RiskRegisterConfig()
        self.audit_trail: List[Dict[str, Any]] = []

        logger.info("RiskRegister initialized")

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    def add_risk(self, risk: RiskEntry) -> RiskEntry:
        """
        Add a new risk to the register.

        Args:
            risk: RiskEntry to add

        Returns:
            Added risk with calculated fields

        Raises:
            ValueError: If risk already exists
        """
        if risk.risk_id in self.risks:
            raise ValueError(f"Risk already exists: {risk.risk_id}")

        # Calculate risk level and score
        if self.config.auto_calculate_risk_level:
            risk.risk_level = RiskMatrix.calculate_risk_level(
                risk.severity, risk.likelihood
            )
            risk.risk_score = RiskMatrix.get_risk_score(
                risk.severity, risk.likelihood
            )

        # Assign SIL
        if self.config.auto_assign_sil:
            risk.required_sil = RiskMatrix.get_required_sil(risk.risk_level)

        # Set target mitigation date
        if self.config.auto_set_target_date and not risk.target_mitigation_date:
            days_allowed = RiskMatrix.get_acceptance_days(risk.risk_level)
            risk.target_mitigation_date = (
                datetime.utcnow() + timedelta(days=days_allowed)
            )

        # Set review date (quarterly by default)
        if not risk.review_date:
            risk.review_date = datetime.utcnow() + timedelta(days=90)

        # Initialize trend history
        if self.config.track_trend_history:
            risk.trend_history.append(RiskTrend(
                severity=risk.severity,
                likelihood=risk.likelihood,
                risk_level=risk.risk_level,
                risk_score=risk.risk_score,
                status=risk.status,
                note="Initial assessment"
            ))

        # Calculate provenance hash
        risk.provenance_hash = self._calculate_provenance(risk)

        # Store risk
        self.risks[risk.risk_id] = risk

        # Log audit trail
        self._log_audit("RISK_CREATED", risk.risk_id, {
            "title": risk.title,
            "category": risk.category.value,
            "risk_level": risk.risk_level.value
        })

        logger.info(
            f"Risk added: {risk.risk_id} ({risk.risk_level.value}) - {risk.title}"
        )
        return risk

    def get_risk(self, risk_id: str) -> Optional[RiskEntry]:
        """Get risk by ID."""
        return self.risks.get(risk_id)

    def update_risk(
        self,
        risk_id: str,
        updates: Dict[str, Any]
    ) -> RiskEntry:
        """
        Update risk entry.

        Args:
            risk_id: Risk identifier
            updates: Dictionary of field updates

        Returns:
            Updated RiskEntry

        Raises:
            ValueError: If risk not found
        """
        if risk_id not in self.risks:
            raise ValueError(f"Risk not found: {risk_id}")

        risk = self.risks[risk_id]
        old_level = risk.risk_level

        # Apply updates
        for key, value in updates.items():
            if hasattr(risk, key):
                setattr(risk, key, value)

        risk.updated_at = datetime.utcnow()

        # Recalculate risk level if severity/likelihood changed
        if "severity" in updates or "likelihood" in updates:
            if self.config.auto_calculate_risk_level:
                risk.risk_level = RiskMatrix.calculate_risk_level(
                    risk.severity, risk.likelihood
                )
                risk.risk_score = RiskMatrix.get_risk_score(
                    risk.severity, risk.likelihood
                )

            if self.config.auto_assign_sil:
                risk.required_sil = RiskMatrix.get_required_sil(risk.risk_level)

        # Update residual risk if applicable
        if risk.residual_severity and risk.residual_likelihood:
            risk.residual_risk_level = RiskMatrix.calculate_risk_level(
                risk.residual_severity, risk.residual_likelihood
            )
            risk.residual_risk_score = RiskMatrix.get_risk_score(
                risk.residual_severity, risk.residual_likelihood
            )

        # Track trend history
        if self.config.track_trend_history and (
            "severity" in updates or "likelihood" in updates or "status" in updates
        ):
            risk.trend_history.append(RiskTrend(
                severity=risk.severity,
                likelihood=risk.likelihood,
                risk_level=risk.risk_level,
                risk_score=risk.risk_score,
                status=risk.status,
                note="Risk updated"
            ))

        risk.provenance_hash = self._calculate_provenance(risk)

        self._log_audit("RISK_UPDATED", risk_id, {
            "updates": list(updates.keys()),
            "old_level": old_level.value,
            "new_level": risk.risk_level.value
        })

        logger.info(f"Risk updated: {risk_id}")
        return risk

    def delete_risk(
        self,
        risk_id: str,
        reason: str = "",
        deleted_by: str = ""
    ) -> bool:
        """
        Delete a risk (soft delete by closing).

        Args:
            risk_id: Risk identifier
            reason: Reason for deletion
            deleted_by: Person performing deletion

        Returns:
            True if deleted

        Raises:
            ValueError: If risk not found
        """
        if risk_id not in self.risks:
            raise ValueError(f"Risk not found: {risk_id}")

        risk = self.risks[risk_id]
        risk.status = RiskStatus.CLOSED
        risk.notes += f"\n[CLOSED] {reason} by {deleted_by}"
        risk.updated_at = datetime.utcnow()
        risk.provenance_hash = self._calculate_provenance(risk)

        self._log_audit("RISK_CLOSED", risk_id, {
            "reason": reason,
            "deleted_by": deleted_by
        })

        logger.info(f"Risk closed: {risk_id}")
        return True

    def add_control(
        self,
        risk_id: str,
        control: Control
    ) -> RiskEntry:
        """
        Add a control to a risk.

        Args:
            risk_id: Risk identifier
            control: Control to add

        Returns:
            Updated RiskEntry
        """
        if risk_id not in self.risks:
            raise ValueError(f"Risk not found: {risk_id}")

        risk = self.risks[risk_id]
        risk.controls.append(control)
        risk.updated_at = datetime.utcnow()
        risk.provenance_hash = self._calculate_provenance(risk)

        self._log_audit("CONTROL_ADDED", risk_id, {
            "control_id": control.control_id,
            "description": control.description
        })

        logger.info(f"Control added to risk {risk_id}")
        return risk

    # =========================================================================
    # RISK ASSESSMENT
    # =========================================================================

    def assess_residual_risk(
        self,
        risk_id: str,
        residual_severity: int,
        residual_likelihood: int,
        assessed_by: str = ""
    ) -> RiskEntry:
        """
        Assess residual risk after controls.

        Args:
            risk_id: Risk identifier
            residual_severity: Residual severity (1-5)
            residual_likelihood: Residual likelihood (1-5)
            assessed_by: Person performing assessment

        Returns:
            Updated RiskEntry
        """
        if risk_id not in self.risks:
            raise ValueError(f"Risk not found: {risk_id}")

        risk = self.risks[risk_id]

        risk.residual_severity = residual_severity
        risk.residual_likelihood = residual_likelihood
        risk.residual_risk_level = RiskMatrix.calculate_risk_level(
            residual_severity, residual_likelihood
        )
        risk.residual_risk_score = RiskMatrix.get_risk_score(
            residual_severity, residual_likelihood
        )
        risk.assessment_date = datetime.utcnow()
        risk.updated_at = datetime.utcnow()

        # Update status
        if risk.status == RiskStatus.IDENTIFIED:
            risk.status = RiskStatus.ASSESSED

        # Track trend
        if self.config.track_trend_history:
            risk.trend_history.append(RiskTrend(
                severity=residual_severity,
                likelihood=residual_likelihood,
                risk_level=risk.residual_risk_level,
                risk_score=risk.residual_risk_score,
                status=risk.status,
                note=f"Residual assessment by {assessed_by}"
            ))

        risk.provenance_hash = self._calculate_provenance(risk)

        self._log_audit("RESIDUAL_ASSESSED", risk_id, {
            "residual_severity": residual_severity,
            "residual_likelihood": residual_likelihood,
            "residual_level": risk.residual_risk_level.value,
            "assessed_by": assessed_by
        })

        logger.info(
            f"Residual risk assessed for {risk_id}: {risk.residual_risk_level.value}"
        )
        return risk

    # =========================================================================
    # RISK ACCEPTANCE
    # =========================================================================

    def accept_risk(
        self,
        risk_id: str,
        accepted_by: str,
        acceptance_level: AcceptanceLevel,
        justification: str,
        conditions: Optional[List[str]] = None,
        review_period_days: int = 365
    ) -> RiskEntry:
        """
        Formally accept a risk.

        Args:
            risk_id: Risk identifier
            accepted_by: Person accepting risk
            acceptance_level: Authority level
            justification: Justification for acceptance
            conditions: Conditions for acceptance
            review_period_days: Days until review

        Returns:
            Updated RiskEntry

        Raises:
            ValueError: If acceptance level is insufficient
        """
        if risk_id not in self.risks:
            raise ValueError(f"Risk not found: {risk_id}")

        risk = self.risks[risk_id]

        # Validate acceptance authority
        required_level = RiskMatrix.get_acceptance_authority(
            risk.residual_risk_level or risk.risk_level
        )
        level_order = list(AcceptanceLevel)

        if level_order.index(acceptance_level) < level_order.index(required_level):
            raise ValueError(
                f"Insufficient authority. Required: {required_level.value}, "
                f"Provided: {acceptance_level.value}"
            )

        # Create acceptance record
        risk.acceptance = RiskAcceptance(
            accepted_by=accepted_by,
            acceptance_level=acceptance_level,
            justification=justification,
            review_date=datetime.utcnow() + timedelta(days=review_period_days),
            conditions=conditions or []
        )

        risk.status = RiskStatus.ACCEPTED
        risk.updated_at = datetime.utcnow()
        risk.provenance_hash = self._calculate_provenance(risk)

        self._log_audit("RISK_ACCEPTED", risk_id, {
            "accepted_by": accepted_by,
            "acceptance_level": acceptance_level.value,
            "justification": justification
        })

        logger.info(f"Risk {risk_id} accepted by {accepted_by}")
        return risk

    # =========================================================================
    # QUERIES AND FILTERING
    # =========================================================================

    def get_all_risks(
        self,
        include_closed: bool = False
    ) -> List[RiskEntry]:
        """Get all risks, optionally excluding closed."""
        if include_closed:
            return list(self.risks.values())
        return [r for r in self.risks.values() if r.status != RiskStatus.CLOSED]

    def get_risks_by_level(
        self,
        risk_level: RiskLevel,
        use_residual: bool = False
    ) -> List[RiskEntry]:
        """Get risks filtered by level."""
        if use_residual:
            return [
                r for r in self.risks.values()
                if (r.residual_risk_level or r.risk_level) == risk_level
            ]
        return [r for r in self.risks.values() if r.risk_level == risk_level]

    def get_risks_by_category(
        self,
        category: RiskCategory
    ) -> List[RiskEntry]:
        """Get risks filtered by category."""
        return [r for r in self.risks.values() if r.category == category]

    def get_risks_by_status(
        self,
        status: RiskStatus
    ) -> List[RiskEntry]:
        """Get risks filtered by status."""
        return [r for r in self.risks.values() if r.status == status]

    def get_critical_risks(self) -> List[RiskEntry]:
        """Get all critical risks."""
        return sorted(
            [r for r in self.risks.values() if r.risk_level == RiskLevel.CRITICAL],
            key=lambda x: x.risk_score,
            reverse=True
        )

    def get_overdue_risks(self) -> List[RiskEntry]:
        """Get risks with overdue mitigation targets."""
        now = datetime.utcnow()
        overdue = [
            r for r in self.risks.values()
            if r.target_mitigation_date
            and r.target_mitigation_date < now
            and r.status not in [RiskStatus.CLOSED, RiskStatus.MITIGATED]
        ]
        return sorted(overdue, key=lambda x: x.target_mitigation_date)

    def get_risks_for_review(self) -> List[RiskEntry]:
        """Get risks due for review."""
        now = datetime.utcnow()
        for_review = [
            r for r in self.risks.values()
            if r.review_date and r.review_date <= now
            and r.status not in [RiskStatus.CLOSED]
        ]
        return sorted(for_review, key=lambda x: x.review_date)

    # =========================================================================
    # RANKING AND PRIORITIZATION
    # =========================================================================

    def get_ranked_risks(
        self,
        top_n: Optional[int] = None,
        use_residual: bool = False
    ) -> List[RiskEntry]:
        """
        Get risks ranked by severity/likelihood score.

        Args:
            top_n: Return only top N risks
            use_residual: Use residual risk for ranking

        Returns:
            List of risks sorted by risk score (highest first)
        """
        open_risks = [
            r for r in self.risks.values()
            if r.status not in [RiskStatus.CLOSED]
        ]

        if use_residual:
            ranked = sorted(
                open_risks,
                key=lambda x: x.residual_risk_score or x.risk_score,
                reverse=True
            )
        else:
            ranked = sorted(
                open_risks,
                key=lambda x: x.risk_score,
                reverse=True
            )

        if top_n:
            return ranked[:top_n]
        return ranked

    def get_pareto_risks(
        self,
        percentage: float = 0.8
    ) -> List[RiskEntry]:
        """
        Get risks contributing to top percentage of total risk.

        Args:
            percentage: Percentage of total risk (0-1)

        Returns:
            List of high-impact risks
        """
        ranked = self.get_ranked_risks()
        if not ranked:
            return []

        total_score = sum(r.risk_score for r in ranked)
        target_score = total_score * percentage

        cumulative = 0
        pareto_risks = []
        for risk in ranked:
            pareto_risks.append(risk)
            cumulative += risk.risk_score
            if cumulative >= target_score:
                break

        return pareto_risks

    # =========================================================================
    # IMPORT FROM HAZOP/FMEA
    # =========================================================================

    def import_from_hazop(
        self,
        hazop_deviations: List[Dict[str, Any]]
    ) -> List[RiskEntry]:
        """
        Import risks from HAZOP study.

        Args:
            hazop_deviations: List of HAZOP deviation dictionaries

        Returns:
            List of created RiskEntry objects
        """
        created = []

        for dev in hazop_deviations:
            entry = RiskEntry(
                title=f"HAZOP: {dev.get('deviation_description', 'Unknown')}",
                description=f"Causes: {', '.join(dev.get('causes', []))}\n"
                           f"Consequences: {', '.join(dev.get('consequences', []))}",
                category=RiskCategory.SAFETY,
                source=RiskSource.HAZOP,
                source_reference=dev.get("deviation_id", ""),
                severity=dev.get("severity", 1),
                likelihood=dev.get("likelihood", 1),
                control_descriptions=dev.get("existing_safeguards", []),
                causes=dev.get("causes", []),
                consequences=dev.get("consequences", [])
            )

            created.append(self.add_risk(entry))

        logger.info(f"Imported {len(created)} risks from HAZOP")
        return created

    def import_from_fmea(
        self,
        failure_modes: List[Dict[str, Any]]
    ) -> List[RiskEntry]:
        """
        Import risks from FMEA study.

        Args:
            failure_modes: List of FMEA failure mode dictionaries

        Returns:
            List of created RiskEntry objects
        """
        created = []

        for fm in failure_modes:
            # Map RPN to severity/likelihood
            rpn = fm.get("rpn", 0)
            severity = min(fm.get("severity", 1), 5)
            occurrence = min(fm.get("occurrence", 1), 5)

            entry = RiskEntry(
                title=f"FMEA: {fm.get('component_name', '')} - {fm.get('failure_mode', '')}",
                description=f"End Effect: {fm.get('end_effect', '')}\nRPN: {rpn}",
                category=RiskCategory.OPERATIONAL,
                source=RiskSource.FMEA,
                source_reference=fm.get("fm_id", ""),
                severity=severity,
                likelihood=occurrence,
                control_descriptions=[fm.get("detection_method", "")] if fm.get("detection_method") else [],
                causes=[fm.get("cause", "")],
                consequences=[fm.get("end_effect", "")],
                asset=fm.get("component_name", "")
            )

            created.append(self.add_risk(entry))

        logger.info(f"Imported {len(created)} risks from FMEA")
        return created

    # =========================================================================
    # REPORTING AND EXPORT
    # =========================================================================

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate risk register summary.

        Returns:
            Summary dictionary with key metrics
        """
        now = datetime.utcnow()
        open_risks = [r for r in self.risks.values() if r.status != RiskStatus.CLOSED]

        if not open_risks:
            return {
                "total_risks": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "average_severity": 0,
                "average_likelihood": 0,
                "overdue_count": 0,
                "report_date": now.isoformat()
            }

        # Level counts
        level_counts = {
            RiskLevel.CRITICAL: sum(1 for r in open_risks if r.risk_level == RiskLevel.CRITICAL),
            RiskLevel.HIGH: sum(1 for r in open_risks if r.risk_level == RiskLevel.HIGH),
            RiskLevel.MEDIUM: sum(1 for r in open_risks if r.risk_level == RiskLevel.MEDIUM),
            RiskLevel.LOW: sum(1 for r in open_risks if r.risk_level == RiskLevel.LOW),
        }

        # Category counts
        category_counts = {}
        for cat in RiskCategory:
            category_counts[cat.value] = sum(
                1 for r in open_risks if r.category == cat
            )

        # Status counts
        status_counts = {}
        for status in RiskStatus:
            status_counts[status.value] = sum(
                1 for r in self.risks.values() if r.status == status
            )

        # Overdue
        overdue = self.get_overdue_risks()

        return {
            "report_date": now.isoformat(),
            "total_risks": len(self.risks),
            "open_risks": len(open_risks),
            "level_breakdown": {
                "critical": level_counts[RiskLevel.CRITICAL],
                "high": level_counts[RiskLevel.HIGH],
                "medium": level_counts[RiskLevel.MEDIUM],
                "low": level_counts[RiskLevel.LOW],
            },
            "category_breakdown": category_counts,
            "status_breakdown": status_counts,
            "average_severity": statistics.mean([r.severity for r in open_risks]),
            "average_likelihood": statistics.mean([r.likelihood for r in open_risks]),
            "average_risk_score": statistics.mean([r.risk_score for r in open_risks]),
            "overdue_count": len(overdue),
            "for_review_count": len(self.get_risks_for_review()),
            "provenance_hash": hashlib.sha256(
                f"{now.isoformat()}|{len(self.risks)}".encode()
            ).hexdigest()
        }

    def generate_heatmap_data(self) -> Dict[str, Any]:
        """
        Generate data for risk heatmap visualization.

        Returns:
            Heatmap data dictionary
        """
        # Initialize 5x5 matrix
        matrix = [[0 for _ in range(5)] for _ in range(5)]
        colors = [["green" for _ in range(5)] for _ in range(5)]

        open_risks = [r for r in self.risks.values() if r.status != RiskStatus.CLOSED]

        # Populate matrix
        for risk in open_risks:
            s_idx = risk.severity - 1
            l_idx = risk.likelihood - 1
            matrix[s_idx][l_idx] += 1

            # Update color
            level = RiskMatrix.calculate_risk_level(risk.severity, risk.likelihood)
            colors[s_idx][l_idx] = RiskMatrix.get_risk_color(level)

        return {
            "matrix": matrix,
            "colors": colors,
            "total_risks": sum(sum(row) for row in matrix),
            "generated_at": datetime.utcnow().isoformat()
        }

    def export_to_json(self) -> str:
        """Export register to JSON format."""
        data = {
            "risks": [r.model_dump() for r in self.risks.values()],
            "summary": self.generate_summary(),
            "export_date": datetime.utcnow().isoformat()
        }
        return json.dumps(data, indent=2, default=str)

    def export_to_csv(self) -> str:
        """Export register to CSV format."""
        output = StringIO()
        writer = csv.writer(output)

        # Header
        headers = [
            "risk_id", "title", "category", "status",
            "severity", "likelihood", "risk_level", "risk_score",
            "residual_severity", "residual_likelihood", "residual_level",
            "risk_owner", "target_date", "controls", "source"
        ]
        writer.writerow(headers)

        # Data rows
        for risk in self.risks.values():
            controls_str = "; ".join(
                c.description for c in risk.controls
            ) or "; ".join(risk.control_descriptions)

            row = [
                risk.risk_id,
                risk.title,
                risk.category.value,
                risk.status.value,
                risk.severity,
                risk.likelihood,
                risk.risk_level.value,
                risk.risk_score,
                risk.residual_severity or "",
                risk.residual_likelihood or "",
                risk.residual_risk_level.value if risk.residual_risk_level else "",
                risk.risk_owner,
                risk.target_mitigation_date.isoformat() if risk.target_mitigation_date else "",
                controls_str,
                risk.source.value
            ]
            writer.writerow(row)

        return output.getvalue()

    def export_compliance_report(self, format_type: str = "text") -> str:
        """
        Generate formatted compliance report.

        Args:
            format_type: "text", "csv", or "json"

        Returns:
            Formatted report string
        """
        summary = self.generate_summary()

        if format_type == "json":
            return json.dumps(summary, indent=2, default=str)

        if format_type == "csv":
            return self.export_to_csv()

        # Text format
        level_breakdown = summary.get("level_breakdown", {})
        lines = [
            "=" * 70,
            "RISK REGISTER COMPLIANCE REPORT",
            "=" * 70,
            f"Report Generated: {summary['report_date']}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 70,
            f"Total Risks in Register: {summary['total_risks']}",
            f"Open Risks: {summary['open_risks']}",
            "",
            "RISK LEVEL BREAKDOWN",
            "-" * 70,
            f"  CRITICAL: {level_breakdown.get('critical', 0)} (Immediate action required)",
            f"  HIGH:     {level_breakdown.get('high', 0)} (Action within 30 days)",
            f"  MEDIUM:   {level_breakdown.get('medium', 0)} (Action within 90 days)",
            f"  LOW:      {level_breakdown.get('low', 0)} (Monitor and review)",
            "",
            f"Risks Overdue: {summary['overdue_count']}",
            f"Risks Due for Review: {summary['for_review_count']}",
            "",
            f"Average Risk Score: {summary['average_risk_score']:.1f}/25",
            "",
            "=" * 70,
        ]

        # Add critical risks
        critical = self.get_critical_risks()
        if critical:
            lines.extend([
                "",
                "CRITICAL RISKS REQUIRING IMMEDIATE ACTION",
                "-" * 70,
            ])
            for risk in critical[:10]:
                lines.append(
                    f"  {risk.risk_id}: {risk.title}\n"
                    f"    Risk Level: {risk.risk_level.value.upper()}\n"
                    f"    Required SIL: {risk.required_sil.value}\n"
                    f"    Target Date: {risk.target_mitigation_date.date() if risk.target_mitigation_date else 'Not set'}"
                )

        return "\n".join(lines)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_provenance(self, risk: RiskEntry) -> str:
        """Calculate SHA-256 provenance hash for risk."""
        data_str = (
            f"{risk.risk_id}|"
            f"{risk.title}|"
            f"{risk.severity}|"
            f"{risk.likelihood}|"
            f"{risk.status.value}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _log_audit(
        self,
        event_type: str,
        risk_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log event to audit trail."""
        self.audit_trail.append({
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "risk_id": risk_id,
            "details": details
        })


if __name__ == "__main__":
    # Example usage
    print("RiskRegister module loaded successfully")

    # Create register
    register = RiskRegister()

    # Create a risk
    entry = RiskEntry(
        title="High temperature in reactor during exothermic reaction",
        description="Temperature may exceed design limits due to insufficient cooling",
        category=RiskCategory.SAFETY,
        severity=4,
        likelihood=3,
        causes=["Cooling water failure", "Runaway reaction"],
        consequences=["Equipment damage", "Potential release"]
    )

    added = register.add_risk(entry)
    print(f"Added: {added.risk_id} ({added.risk_level.value})")

    # Generate summary
    summary = register.generate_summary()
    print(f"Total Risks: {summary['total_risks']}")
    print(f"Critical: {summary['level_breakdown']['critical']}")
