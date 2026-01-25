r"""
SafeguardVerification - Safeguard Registry and Verification Workflow

This module implements comprehensive safeguard (barrier) tracking and verification
per IEC 61511 and IEC 61508 standards for process safety management. It provides:

- SafeguardRegistry for tracking all safety barriers and Independent Protection Layers (IPLs)
- Verification workflow with status tracking and approval processes
- IPL documentation with PFD values and credit validation
- Safeguard effectiveness validation against risk reduction requirements
- Periodic verification scheduling with overdue tracking

Key concepts:
- Safeguard: Any barrier that reduces risk (engineering, administrative, or inherent)
- IPL: Independent Protection Layer with quantifiable risk reduction credit
- Verification: Periodic confirmation that safeguard remains effective
- Credit: Risk reduction factor that can be claimed in LOPA analysis

Reference:
- IEC 61511-1:2016 Clause 9.4 - Independent Protection Layers
- IEC 61511-3:2016 Annex F - LOPA requirements for IPL independence
- CCPS Guidelines for IPL Credit (2001)

Example:
    >>> from greenlang.safety.safeguard_verification import SafeguardRegistry, Safeguard
    >>> registry = SafeguardRegistry()
    >>> safeguard = Safeguard(
    ...     name="High Pressure Relief Valve",
    ...     safeguard_type=SafeguardType.RELIEF_DEVICE,
    ...     pfd=0.01,
    ...     is_ipl=True
    ... )
    >>> registry.register_safeguard(safeguard)
    >>> verification = registry.schedule_verification(safeguard.safeguard_id)
"""

from typing import Dict, List, Optional, Any, ClassVar
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SafeguardType(str, Enum):
    """Safeguard classification per IEC 61511."""

    # Engineering safeguards
    BPCS = "bpcs"  # Basic Process Control System
    SIS = "sis"  # Safety Instrumented System
    RELIEF_DEVICE = "relief_device"  # PSVs, rupture disks
    MECHANICAL_INTERLOCK = "mechanical_interlock"  # Mechanical barriers
    CHECK_VALVE = "check_valve"  # Backflow prevention
    FIRE_PROTECTION = "fire_protection"  # Fire suppression systems

    # Administrative safeguards
    OPERATOR_RESPONSE = "operator_response"  # Operator action on alarm
    PROCEDURE = "procedure"  # Operating procedures
    INSPECTION = "inspection"  # Regular inspections
    TRAINING = "training"  # Operator training programs

    # Inherent safety
    INHERENT_DESIGN = "inherent_design"  # Design eliminates hazard
    PASSIVE_PROTECTION = "passive_protection"  # Dikes, containment


class SafeguardStatus(str, Enum):
    """Safeguard operational status."""

    ACTIVE = "active"  # In service and verified
    BYPASSED = "bypassed"  # Temporarily bypassed
    DEGRADED = "degraded"  # Reduced effectiveness
    FAILED = "failed"  # Not functional
    PENDING_VERIFICATION = "pending_verification"  # Awaiting verification
    DECOMMISSIONED = "decommissioned"  # No longer in service


class VerificationStatus(str, Enum):
    """Verification workflow status."""

    SCHEDULED = "scheduled"  # Verification scheduled
    IN_PROGRESS = "in_progress"  # Verification underway
    AWAITING_REVIEW = "awaiting_review"  # Awaiting supervisor review
    APPROVED = "approved"  # Verification approved
    REJECTED = "rejected"  # Verification rejected, action required
    OVERDUE = "overdue"  # Past due date


class VerificationType(str, Enum):
    """Type of verification activity."""

    INITIAL = "initial"  # Initial installation verification
    PERIODIC = "periodic"  # Scheduled periodic verification
    POST_MODIFICATION = "post_modification"  # After MOC
    POST_INCIDENT = "post_incident"  # After incident/near-miss
    REVALIDATION = "revalidation"  # Full revalidation of credit


class IPLCategory(str, Enum):
    """IPL categories per CCPS guidelines."""

    BPCS = "bpcs"  # PFD typically 0.1
    ALARM_RESPONSE = "alarm_response"  # PFD typically 0.1
    SAFETY_INTERLOCK = "safety_interlock"  # PFD 0.01-0.001
    RELIEF_DEVICE = "relief_device"  # PFD typically 0.01
    RUPTURE_DISK = "rupture_disk"  # PFD typically 0.001
    CHECK_VALVE = "check_valve"  # PFD typically 0.01
    DIKE_CONTAINMENT = "dike_containment"  # PFD typically 0.01
    HUMAN_ACTION = "human_action"  # PFD typically 0.1-1.0


# =============================================================================
# DATA MODELS
# =============================================================================

class IPLCredit(BaseModel):
    """IPL credit documentation for LOPA analysis."""

    credit_id: str = Field(
        default_factory=lambda: f"IPL-{uuid.uuid4().hex[:8].upper()}",
        description="Unique IPL credit identifier"
    )
    category: IPLCategory = Field(
        ...,
        description="IPL category per CCPS guidelines"
    )
    pfd: float = Field(
        ...,
        ge=1e-5,
        le=1.0,
        description="Probability of Failure on Demand (0-1)"
    )
    pfd_basis: str = Field(
        default="",
        description="Basis for PFD claim (vendor data, operating history, etc.)"
    )
    is_independent: bool = Field(
        default=True,
        description="Confirmed independent from initiating event and other IPLs"
    )
    independence_justification: str = Field(
        default="",
        description="Justification for independence claim"
    )
    is_auditable: bool = Field(
        default=True,
        description="IPL can be tested and audited"
    )
    audit_method: str = Field(
        default="",
        description="Method for verifying IPL effectiveness"
    )
    common_cause_exclusions: List[str] = Field(
        default_factory=list,
        description="Common cause failures this IPL is NOT credited for"
    )
    scenarios_credited: List[str] = Field(
        default_factory=list,
        description="LOPA scenarios where this IPL is credited"
    )
    credit_approved_by: Optional[str] = Field(
        None,
        description="Person who approved IPL credit"
    )
    credit_approved_date: Optional[datetime] = Field(
        None,
        description="Date credit was approved"
    )

    @field_validator('pfd')
    @classmethod
    def validate_pfd_range(cls, v: float) -> float:
        """Validate PFD is within typical IPL credit range."""
        if v < 1e-4 and v != 0:
            logger.warning(
                f"PFD {v:.2e} is very low. Verify credit is justified per IEC 61511."
            )
        return v


class Safeguard(BaseModel):
    """Comprehensive safeguard (barrier) definition."""

    safeguard_id: str = Field(
        default_factory=lambda: f"SG-{uuid.uuid4().hex[:8].upper()}",
        description="Unique safeguard identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Safeguard name"
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Detailed description"
    )
    safeguard_type: SafeguardType = Field(
        ...,
        description="Type of safeguard"
    )
    tag_number: str = Field(
        default="",
        description="Equipment tag number (e.g., PSV-001)"
    )
    location: str = Field(
        default="",
        description="Physical location or P&ID reference"
    )

    # IPL-specific fields
    is_ipl: bool = Field(
        default=False,
        description="Qualifies as Independent Protection Layer"
    )
    ipl_credit: Optional[IPLCredit] = Field(
        None,
        description="IPL credit documentation if is_ipl=True"
    )
    pfd: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Probability of Failure on Demand"
    )

    # Status and tracking
    status: SafeguardStatus = Field(
        default=SafeguardStatus.ACTIVE,
        description="Current operational status"
    )
    installation_date: Optional[datetime] = Field(
        None,
        description="Date safeguard was installed"
    )
    last_verification_date: Optional[datetime] = Field(
        None,
        description="Date of last verification"
    )
    next_verification_date: Optional[datetime] = Field(
        None,
        description="Next scheduled verification"
    )
    verification_interval_days: int = Field(
        default=365,
        ge=1,
        description="Verification interval in days"
    )

    # Ownership and documentation
    responsible_department: str = Field(
        default="",
        description="Department responsible for safeguard"
    )
    responsible_person: str = Field(
        default="",
        description="Person responsible for maintenance"
    )
    linked_hazards: List[str] = Field(
        default_factory=list,
        description="Hazard IDs this safeguard mitigates"
    )
    linked_lopa_scenarios: List[str] = Field(
        default_factory=list,
        description="LOPA scenario IDs where credited"
    )
    documentation_links: List[str] = Field(
        default_factory=list,
        description="Links to supporting documentation"
    )

    # Metadata
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


class VerificationRecord(BaseModel):
    """Safeguard verification record."""

    verification_id: str = Field(
        default_factory=lambda: f"VER-{uuid.uuid4().hex[:8].upper()}",
        description="Unique verification identifier"
    )
    safeguard_id: str = Field(
        ...,
        description="Associated safeguard identifier"
    )
    verification_type: VerificationType = Field(
        ...,
        description="Type of verification"
    )
    status: VerificationStatus = Field(
        default=VerificationStatus.SCHEDULED,
        description="Current verification status"
    )

    # Scheduling
    scheduled_date: datetime = Field(
        ...,
        description="Scheduled verification date"
    )
    actual_date: Optional[datetime] = Field(
        None,
        description="Actual verification date"
    )
    due_date: datetime = Field(
        ...,
        description="Due date including grace period"
    )

    # Verification details
    verified_by: Optional[str] = Field(
        None,
        description="Person who performed verification"
    )
    reviewed_by: Optional[str] = Field(
        None,
        description="Supervisor who reviewed"
    )
    review_date: Optional[datetime] = Field(
        None,
        description="Date of supervisor review"
    )

    # Results
    effectiveness_confirmed: bool = Field(
        default=False,
        description="Safeguard effectiveness confirmed"
    )
    pfd_verified: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Verified PFD value"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Verification findings"
    )
    corrective_actions: List[str] = Field(
        default_factory=list,
        description="Required corrective actions"
    )
    evidence_documents: List[str] = Field(
        default_factory=list,
        description="Supporting evidence documents"
    )

    # Checklist
    checklist_completed: bool = Field(
        default=False,
        description="Verification checklist completed"
    )
    checklist_items: Dict[str, bool] = Field(
        default_factory=dict,
        description="Checklist items and completion status"
    )

    # Metadata
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


class VerificationScheduleConfig(BaseModel):
    """Configuration for verification scheduling."""

    default_interval_days: int = Field(
        default=365,
        ge=1,
        description="Default verification interval"
    )
    grace_period_days: int = Field(
        default=30,
        ge=0,
        description="Grace period before overdue"
    )
    escalation_days: List[int] = Field(
        default_factory=lambda: [7, 14, 30],
        description="Days before due to send escalation notifications"
    )
    require_supervisor_approval: bool = Field(
        default=True,
        description="Require supervisor approval for verification"
    )
    auto_schedule_next: bool = Field(
        default=True,
        description="Automatically schedule next verification"
    )


# =============================================================================
# SAFEGUARD REGISTRY
# =============================================================================

class SafeguardRegistry:
    """
    Registry for tracking all safety barriers and safeguards.

    Implements comprehensive safeguard management per IEC 61511:
    - Registration and tracking of all safeguards
    - IPL credit documentation and validation
    - Verification workflow management
    - Periodic verification scheduling
    - Compliance reporting

    Attributes:
        safeguards: Dict of safeguard_id to Safeguard
        verifications: Dict of verification_id to VerificationRecord
        schedule_config: Verification scheduling configuration

    Example:
        >>> registry = SafeguardRegistry()
        >>> safeguard = Safeguard(
        ...     name="High Pressure Alarm",
        ...     safeguard_type=SafeguardType.BPCS,
        ...     is_ipl=True,
        ...     pfd=0.1
        ... )
        >>> registry.register_safeguard(safeguard)
        >>> registry.schedule_verification(safeguard.safeguard_id)
    """

    # Standard PFD values per IPL category (CCPS guidelines)
    STANDARD_PFD_VALUES: ClassVar[Dict[IPLCategory, float]] = {
        IPLCategory.BPCS: 0.1,
        IPLCategory.ALARM_RESPONSE: 0.1,
        IPLCategory.SAFETY_INTERLOCK: 0.01,
        IPLCategory.RELIEF_DEVICE: 0.01,
        IPLCategory.RUPTURE_DISK: 0.001,
        IPLCategory.CHECK_VALVE: 0.01,
        IPLCategory.DIKE_CONTAINMENT: 0.01,
        IPLCategory.HUMAN_ACTION: 0.1,
    }

    # Verification intervals by safeguard type (days)
    DEFAULT_VERIFICATION_INTERVALS: ClassVar[Dict[SafeguardType, int]] = {
        SafeguardType.BPCS: 365,
        SafeguardType.SIS: 365,
        SafeguardType.RELIEF_DEVICE: 365,
        SafeguardType.MECHANICAL_INTERLOCK: 180,
        SafeguardType.CHECK_VALVE: 365,
        SafeguardType.FIRE_PROTECTION: 90,
        SafeguardType.OPERATOR_RESPONSE: 365,
        SafeguardType.PROCEDURE: 365,
        SafeguardType.INSPECTION: 365,
        SafeguardType.TRAINING: 365,
        SafeguardType.INHERENT_DESIGN: 730,  # 2 years
        SafeguardType.PASSIVE_PROTECTION: 365,
    }

    def __init__(
        self,
        schedule_config: Optional[VerificationScheduleConfig] = None
    ):
        """
        Initialize SafeguardRegistry.

        Args:
            schedule_config: Optional verification scheduling configuration
        """
        self.safeguards: Dict[str, Safeguard] = {}
        self.verifications: Dict[str, VerificationRecord] = {}
        self.schedule_config = schedule_config or VerificationScheduleConfig()
        self.audit_trail: List[Dict[str, Any]] = []

        logger.info("SafeguardRegistry initialized")

    def register_safeguard(self, safeguard: Safeguard) -> Safeguard:
        """
        Register a new safeguard in the registry.

        Args:
            safeguard: Safeguard to register

        Returns:
            Registered safeguard with updated fields

        Raises:
            ValueError: If safeguard already exists or validation fails
        """
        if safeguard.safeguard_id in self.safeguards:
            raise ValueError(f"Safeguard already exists: {safeguard.safeguard_id}")

        # Validate IPL requirements
        if safeguard.is_ipl and not safeguard.ipl_credit:
            logger.warning(
                f"Safeguard {safeguard.safeguard_id} marked as IPL but no credit defined"
            )

        # Set default verification interval based on type
        if safeguard.verification_interval_days == 365:  # Default value
            safeguard.verification_interval_days = self.DEFAULT_VERIFICATION_INTERVALS.get(
                safeguard.safeguard_type, 365
            )

        # Calculate next verification date
        if safeguard.last_verification_date:
            safeguard.next_verification_date = (
                safeguard.last_verification_date +
                timedelta(days=safeguard.verification_interval_days)
            )
        else:
            # New safeguard - schedule initial verification
            safeguard.next_verification_date = (
                datetime.utcnow() +
                timedelta(days=self.schedule_config.default_interval_days)
            )

        # Calculate provenance hash
        safeguard.provenance_hash = self._calculate_safeguard_provenance(safeguard)

        # Store safeguard
        self.safeguards[safeguard.safeguard_id] = safeguard

        # Log audit trail
        self._log_audit("SAFEGUARD_REGISTERED", safeguard.safeguard_id, {
            "name": safeguard.name,
            "type": safeguard.safeguard_type.value,
            "is_ipl": safeguard.is_ipl
        })

        logger.info(f"Safeguard registered: {safeguard.safeguard_id} - {safeguard.name}")
        return safeguard

    def update_safeguard(
        self,
        safeguard_id: str,
        updates: Dict[str, Any]
    ) -> Safeguard:
        """
        Update safeguard details.

        Args:
            safeguard_id: Safeguard identifier
            updates: Dictionary of field updates

        Returns:
            Updated safeguard

        Raises:
            ValueError: If safeguard not found
        """
        if safeguard_id not in self.safeguards:
            raise ValueError(f"Safeguard not found: {safeguard_id}")

        safeguard = self.safeguards[safeguard_id]
        old_status = safeguard.status

        # Apply updates
        for key, value in updates.items():
            if hasattr(safeguard, key):
                setattr(safeguard, key, value)

        safeguard.updated_at = datetime.utcnow()
        safeguard.provenance_hash = self._calculate_safeguard_provenance(safeguard)

        # Log status changes
        if safeguard.status != old_status:
            self._log_audit("SAFEGUARD_STATUS_CHANGED", safeguard_id, {
                "old_status": old_status.value,
                "new_status": safeguard.status.value
            })

        logger.info(f"Safeguard updated: {safeguard_id}")
        return safeguard

    def get_safeguard(self, safeguard_id: str) -> Optional[Safeguard]:
        """Get safeguard by ID."""
        return self.safeguards.get(safeguard_id)

    def get_all_ipls(self) -> List[Safeguard]:
        """
        Get all safeguards qualified as Independent Protection Layers.

        Returns:
            List of IPL-qualified safeguards
        """
        return [s for s in self.safeguards.values() if s.is_ipl]

    def get_safeguards_by_type(
        self,
        safeguard_type: SafeguardType
    ) -> List[Safeguard]:
        """
        Get safeguards filtered by type.

        Args:
            safeguard_type: Type to filter by

        Returns:
            List of matching safeguards
        """
        return [
            s for s in self.safeguards.values()
            if s.safeguard_type == safeguard_type
        ]

    def get_safeguards_by_hazard(self, hazard_id: str) -> List[Safeguard]:
        """
        Get safeguards linked to a specific hazard.

        Args:
            hazard_id: Hazard identifier

        Returns:
            List of safeguards mitigating the hazard
        """
        return [
            s for s in self.safeguards.values()
            if hazard_id in s.linked_hazards
        ]

    def validate_ipl_credit(self, safeguard_id: str) -> Dict[str, Any]:
        """
        Validate IPL credit for LOPA analysis.

        Checks independence, auditability, and PFD reasonableness
        per IEC 61511-3 Annex F.

        Args:
            safeguard_id: Safeguard identifier

        Returns:
            Validation result dictionary

        Raises:
            ValueError: If safeguard not found or not an IPL
        """
        if safeguard_id not in self.safeguards:
            raise ValueError(f"Safeguard not found: {safeguard_id}")

        safeguard = self.safeguards[safeguard_id]

        if not safeguard.is_ipl:
            raise ValueError(f"Safeguard {safeguard_id} is not marked as IPL")

        issues: List[str] = []
        warnings: List[str] = []

        # Check IPL credit documentation
        if not safeguard.ipl_credit:
            issues.append("No IPL credit documentation provided")
        else:
            credit = safeguard.ipl_credit

            # Check independence
            if not credit.is_independent:
                issues.append("IPL not confirmed as independent")
            elif not credit.independence_justification:
                warnings.append("Independence justification not documented")

            # Check auditability
            if not credit.is_auditable:
                issues.append("IPL not confirmed as auditable")
            elif not credit.audit_method:
                warnings.append("Audit method not documented")

            # Check PFD basis
            if not credit.pfd_basis:
                warnings.append("PFD basis not documented")

            # Check PFD reasonableness
            standard_pfd = self.STANDARD_PFD_VALUES.get(credit.category, 0.1)
            if credit.pfd < standard_pfd / 10:
                warnings.append(
                    f"Claimed PFD {credit.pfd:.2e} is significantly lower than "
                    f"standard value {standard_pfd:.2e} for {credit.category.value}"
                )

            # Check approval
            if not credit.credit_approved_by:
                warnings.append("IPL credit not formally approved")

        # Check verification status
        if safeguard.status != SafeguardStatus.ACTIVE:
            issues.append(f"Safeguard status is {safeguard.status.value}, not ACTIVE")

        is_valid = len(issues) == 0

        return {
            "safeguard_id": safeguard_id,
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "pfd": safeguard.pfd,
            "category": safeguard.ipl_credit.category.value if safeguard.ipl_credit else None,
            "validation_timestamp": datetime.utcnow().isoformat()
        }

    # =========================================================================
    # VERIFICATION WORKFLOW
    # =========================================================================

    def schedule_verification(
        self,
        safeguard_id: str,
        verification_type: VerificationType = VerificationType.PERIODIC,
        scheduled_date: Optional[datetime] = None
    ) -> VerificationRecord:
        """
        Schedule a verification for a safeguard.

        Args:
            safeguard_id: Safeguard identifier
            verification_type: Type of verification
            scheduled_date: Optional specific date (defaults to next due date)

        Returns:
            VerificationRecord for the scheduled verification

        Raises:
            ValueError: If safeguard not found
        """
        if safeguard_id not in self.safeguards:
            raise ValueError(f"Safeguard not found: {safeguard_id}")

        safeguard = self.safeguards[safeguard_id]

        # Determine scheduled date
        if scheduled_date is None:
            scheduled_date = safeguard.next_verification_date or datetime.utcnow()

        # Calculate due date with grace period
        due_date = scheduled_date + timedelta(
            days=self.schedule_config.grace_period_days
        )

        # Create verification record
        verification = VerificationRecord(
            safeguard_id=safeguard_id,
            verification_type=verification_type,
            scheduled_date=scheduled_date,
            due_date=due_date,
            status=VerificationStatus.SCHEDULED,
            checklist_items=self._get_verification_checklist(safeguard.safeguard_type)
        )

        verification.provenance_hash = self._calculate_verification_provenance(
            verification
        )

        # Store verification
        self.verifications[verification.verification_id] = verification

        # Update safeguard status if appropriate
        if safeguard.status == SafeguardStatus.ACTIVE:
            safeguard.status = SafeguardStatus.PENDING_VERIFICATION
            safeguard.updated_at = datetime.utcnow()

        self._log_audit("VERIFICATION_SCHEDULED", verification.verification_id, {
            "safeguard_id": safeguard_id,
            "type": verification_type.value,
            "scheduled_date": scheduled_date.isoformat()
        })

        logger.info(
            f"Verification scheduled: {verification.verification_id} "
            f"for {safeguard_id} on {scheduled_date.date()}"
        )
        return verification

    def start_verification(
        self,
        verification_id: str,
        verified_by: str
    ) -> VerificationRecord:
        """
        Start a scheduled verification.

        Args:
            verification_id: Verification identifier
            verified_by: Person performing verification

        Returns:
            Updated VerificationRecord

        Raises:
            ValueError: If verification not found or not in scheduled status
        """
        if verification_id not in self.verifications:
            raise ValueError(f"Verification not found: {verification_id}")

        verification = self.verifications[verification_id]

        if verification.status != VerificationStatus.SCHEDULED:
            raise ValueError(
                f"Cannot start verification in status: {verification.status.value}"
            )

        verification.status = VerificationStatus.IN_PROGRESS
        verification.verified_by = verified_by
        verification.actual_date = datetime.utcnow()
        verification.updated_at = datetime.utcnow()

        self._log_audit("VERIFICATION_STARTED", verification_id, {
            "verified_by": verified_by
        })

        logger.info(f"Verification started: {verification_id} by {verified_by}")
        return verification

    def complete_verification(
        self,
        verification_id: str,
        effectiveness_confirmed: bool,
        findings: Optional[List[str]] = None,
        corrective_actions: Optional[List[str]] = None,
        pfd_verified: Optional[float] = None,
        checklist_items: Optional[Dict[str, bool]] = None,
        evidence_documents: Optional[List[str]] = None,
        notes: str = ""
    ) -> VerificationRecord:
        """
        Complete a verification with results.

        Args:
            verification_id: Verification identifier
            effectiveness_confirmed: Whether safeguard is effective
            findings: List of findings during verification
            corrective_actions: Required corrective actions
            pfd_verified: Verified PFD value (if measured)
            checklist_items: Completed checklist items
            evidence_documents: Supporting evidence
            notes: Additional notes

        Returns:
            Updated VerificationRecord

        Raises:
            ValueError: If verification not in progress
        """
        if verification_id not in self.verifications:
            raise ValueError(f"Verification not found: {verification_id}")

        verification = self.verifications[verification_id]

        if verification.status != VerificationStatus.IN_PROGRESS:
            raise ValueError(
                f"Cannot complete verification in status: {verification.status.value}"
            )

        # Update verification record
        verification.effectiveness_confirmed = effectiveness_confirmed
        verification.findings = findings or []
        verification.corrective_actions = corrective_actions or []
        verification.pfd_verified = pfd_verified
        verification.evidence_documents = evidence_documents or []
        verification.notes = notes
        verification.updated_at = datetime.utcnow()

        if checklist_items:
            verification.checklist_items.update(checklist_items)

        verification.checklist_completed = all(verification.checklist_items.values())

        # Set status based on configuration
        if self.schedule_config.require_supervisor_approval:
            verification.status = VerificationStatus.AWAITING_REVIEW
        else:
            verification.status = VerificationStatus.APPROVED
            self._finalize_verification(verification)

        verification.provenance_hash = self._calculate_verification_provenance(
            verification
        )

        self._log_audit("VERIFICATION_COMPLETED", verification_id, {
            "effectiveness_confirmed": effectiveness_confirmed,
            "findings_count": len(verification.findings),
            "corrective_actions_count": len(verification.corrective_actions)
        })

        logger.info(
            f"Verification completed: {verification_id}, "
            f"effectiveness={effectiveness_confirmed}"
        )
        return verification

    def approve_verification(
        self,
        verification_id: str,
        reviewed_by: str,
        approved: bool = True,
        review_notes: str = ""
    ) -> VerificationRecord:
        """
        Approve or reject a completed verification.

        Args:
            verification_id: Verification identifier
            reviewed_by: Supervisor performing review
            approved: Whether verification is approved
            review_notes: Review notes

        Returns:
            Updated VerificationRecord

        Raises:
            ValueError: If verification not awaiting review
        """
        if verification_id not in self.verifications:
            raise ValueError(f"Verification not found: {verification_id}")

        verification = self.verifications[verification_id]

        if verification.status != VerificationStatus.AWAITING_REVIEW:
            raise ValueError(
                f"Cannot review verification in status: {verification.status.value}"
            )

        verification.reviewed_by = reviewed_by
        verification.review_date = datetime.utcnow()
        verification.updated_at = datetime.utcnow()

        if review_notes:
            verification.notes += f"\n[Supervisor Review] {review_notes}"

        if approved:
            verification.status = VerificationStatus.APPROVED
            self._finalize_verification(verification)
            self._log_audit("VERIFICATION_APPROVED", verification_id, {
                "reviewed_by": reviewed_by
            })
        else:
            verification.status = VerificationStatus.REJECTED
            self._log_audit("VERIFICATION_REJECTED", verification_id, {
                "reviewed_by": reviewed_by,
                "reason": review_notes
            })

        logger.info(
            f"Verification {'approved' if approved else 'rejected'}: "
            f"{verification_id} by {reviewed_by}"
        )
        return verification

    def _finalize_verification(self, verification: VerificationRecord) -> None:
        """Finalize verification and update safeguard."""
        safeguard_id = verification.safeguard_id

        if safeguard_id not in self.safeguards:
            return

        safeguard = self.safeguards[safeguard_id]
        safeguard.last_verification_date = verification.actual_date or datetime.utcnow()

        # Update PFD if verified
        if verification.pfd_verified is not None:
            safeguard.pfd = verification.pfd_verified
            if safeguard.ipl_credit:
                safeguard.ipl_credit.pfd = verification.pfd_verified

        # Set status based on effectiveness
        if verification.effectiveness_confirmed:
            safeguard.status = SafeguardStatus.ACTIVE
        else:
            safeguard.status = SafeguardStatus.DEGRADED

        # Schedule next verification if configured
        if self.schedule_config.auto_schedule_next:
            safeguard.next_verification_date = (
                safeguard.last_verification_date +
                timedelta(days=safeguard.verification_interval_days)
            )

        safeguard.updated_at = datetime.utcnow()
        safeguard.provenance_hash = self._calculate_safeguard_provenance(safeguard)

    # =========================================================================
    # VERIFICATION SCHEDULING
    # =========================================================================

    def get_overdue_verifications(self) -> List[VerificationRecord]:
        """
        Get all overdue verifications.

        Returns:
            List of overdue VerificationRecords
        """
        now = datetime.utcnow()
        overdue = []

        for verification in self.verifications.values():
            if verification.status in [
                VerificationStatus.SCHEDULED,
                VerificationStatus.IN_PROGRESS
            ] and verification.due_date < now:
                verification.status = VerificationStatus.OVERDUE
                overdue.append(verification)

        if overdue:
            logger.warning(f"{len(overdue)} verifications are overdue")

        return overdue

    def get_upcoming_verifications(
        self,
        days_ahead: int = 30
    ) -> List[VerificationRecord]:
        """
        Get verifications due within specified period.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of upcoming VerificationRecords
        """
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)

        upcoming = [
            v for v in self.verifications.values()
            if v.status == VerificationStatus.SCHEDULED
            and v.scheduled_date <= cutoff
        ]

        return sorted(upcoming, key=lambda x: x.scheduled_date)

    def get_safeguards_needing_verification(self) -> List[Safeguard]:
        """
        Get safeguards that need verification scheduled.

        Returns:
            List of safeguards needing verification
        """
        now = datetime.utcnow()

        needing_verification = []
        for safeguard in self.safeguards.values():
            # Check if verification is needed
            if safeguard.status == SafeguardStatus.DECOMMISSIONED:
                continue

            if safeguard.next_verification_date and safeguard.next_verification_date <= now:
                # Check if verification already scheduled
                has_pending = any(
                    v.safeguard_id == safeguard.safeguard_id
                    and v.status in [
                        VerificationStatus.SCHEDULED,
                        VerificationStatus.IN_PROGRESS
                    ]
                    for v in self.verifications.values()
                )

                if not has_pending:
                    needing_verification.append(safeguard)

        return needing_verification

    def _get_verification_checklist(
        self,
        safeguard_type: SafeguardType
    ) -> Dict[str, bool]:
        """Get standard verification checklist for safeguard type."""
        base_checklist = {
            "Visual inspection completed": False,
            "Documentation reviewed": False,
            "Functionality tested": False,
            "Calibration verified": False,
            "As-built matches design": False,
        }

        # Type-specific checklist items
        type_checklists = {
            SafeguardType.RELIEF_DEVICE: {
                "Set pressure verified": False,
                "Leak test passed": False,
                "Inlet/outlet blockage checked": False,
            },
            SafeguardType.SIS: {
                "Logic solver functional": False,
                "Sensors calibrated": False,
                "Final element tested": False,
                "Trip setpoint verified": False,
            },
            SafeguardType.FIRE_PROTECTION: {
                "Detection system tested": False,
                "Suppression agent level checked": False,
                "Activation mechanism verified": False,
            },
        }

        # Combine base and type-specific
        checklist = base_checklist.copy()
        if safeguard_type in type_checklists:
            checklist.update(type_checklists[safeguard_type])

        return checklist

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.

        Returns:
            Compliance report dictionary
        """
        now = datetime.utcnow()

        # Update overdue status
        self.get_overdue_verifications()

        # Calculate statistics
        total_safeguards = len(self.safeguards)
        active_safeguards = sum(
            1 for s in self.safeguards.values()
            if s.status == SafeguardStatus.ACTIVE
        )
        ipl_count = sum(1 for s in self.safeguards.values() if s.is_ipl)

        # Verification status
        overdue_count = sum(
            1 for v in self.verifications.values()
            if v.status == VerificationStatus.OVERDUE
        )
        pending_count = sum(
            1 for v in self.verifications.values()
            if v.status == VerificationStatus.SCHEDULED
        )

        # Status breakdown
        status_breakdown = {}
        for status in SafeguardStatus:
            status_breakdown[status.value] = sum(
                1 for s in self.safeguards.values()
                if s.status == status
            )

        # Type breakdown
        type_breakdown = {}
        for sg_type in SafeguardType:
            type_breakdown[sg_type.value] = sum(
                1 for s in self.safeguards.values()
                if s.safeguard_type == sg_type
            )

        return {
            "report_date": now.isoformat(),
            "summary": {
                "total_safeguards": total_safeguards,
                "active_safeguards": active_safeguards,
                "ipl_count": ipl_count,
                "compliance_rate": (
                    active_safeguards / total_safeguards * 100
                    if total_safeguards > 0 else 100
                )
            },
            "verification_status": {
                "overdue": overdue_count,
                "pending": pending_count,
                "awaiting_review": sum(
                    1 for v in self.verifications.values()
                    if v.status == VerificationStatus.AWAITING_REVIEW
                )
            },
            "safeguard_status_breakdown": status_breakdown,
            "safeguard_type_breakdown": type_breakdown,
            "overdue_verifications": [
                {
                    "verification_id": v.verification_id,
                    "safeguard_id": v.safeguard_id,
                    "safeguard_name": self.safeguards[v.safeguard_id].name
                    if v.safeguard_id in self.safeguards else "Unknown",
                    "due_date": v.due_date.isoformat(),
                    "days_overdue": (now - v.due_date).days
                }
                for v in self.verifications.values()
                if v.status == VerificationStatus.OVERDUE
            ],
            "audit_trail_count": len(self.audit_trail),
            "provenance_hash": hashlib.sha256(
                f"{now.isoformat()}|{total_safeguards}|{overdue_count}".encode()
            ).hexdigest()
        }

    def export_to_json(self, include_verifications: bool = True) -> str:
        """
        Export registry to JSON format.

        Args:
            include_verifications: Include verification records

        Returns:
            JSON string
        """
        data = {
            "safeguards": [
                s.model_dump() for s in self.safeguards.values()
            ],
            "export_date": datetime.utcnow().isoformat()
        }

        if include_verifications:
            data["verifications"] = [
                v.model_dump() for v in self.verifications.values()
            ]

        return json.dumps(data, indent=2, default=str)

    def export_to_csv(self) -> str:
        """
        Export safeguards to CSV format.

        Returns:
            CSV string
        """
        headers = [
            "safeguard_id", "name", "type", "is_ipl", "pfd",
            "status", "last_verification", "next_verification"
        ]
        lines = [",".join(headers)]

        for s in self.safeguards.values():
            row = [
                s.safeguard_id,
                f'"{s.name}"',
                s.safeguard_type.value,
                str(s.is_ipl),
                str(s.pfd or ""),
                s.status.value,
                s.last_verification_date.isoformat() if s.last_verification_date else "",
                s.next_verification_date.isoformat() if s.next_verification_date else ""
            ]
            lines.append(",".join(row))

        return "\n".join(lines)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_safeguard_provenance(self, safeguard: Safeguard) -> str:
        """Calculate SHA-256 provenance hash for safeguard."""
        data_str = (
            f"{safeguard.safeguard_id}|"
            f"{safeguard.name}|"
            f"{safeguard.safeguard_type.value}|"
            f"{safeguard.status.value}|"
            f"{safeguard.pfd or 0}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _calculate_verification_provenance(
        self,
        verification: VerificationRecord
    ) -> str:
        """Calculate SHA-256 provenance hash for verification."""
        data_str = (
            f"{verification.verification_id}|"
            f"{verification.safeguard_id}|"
            f"{verification.status.value}|"
            f"{verification.effectiveness_confirmed}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _log_audit(
        self,
        event_type: str,
        entity_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log event to audit trail."""
        self.audit_trail.append({
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "entity_id": entity_id,
            "details": details
        })


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_ipl_from_standard(
    category: IPLCategory,
    name: str,
    independence_justification: str,
    audit_method: str
) -> IPLCredit:
    """
    Create IPL credit using standard PFD values.

    Args:
        category: IPL category
        name: IPL name
        independence_justification: Justification for independence
        audit_method: Method for auditing effectiveness

    Returns:
        IPLCredit with standard PFD value
    """
    standard_pfd = SafeguardRegistry.STANDARD_PFD_VALUES.get(category, 0.1)

    return IPLCredit(
        category=category,
        pfd=standard_pfd,
        pfd_basis=f"CCPS standard value for {category.value}",
        is_independent=True,
        independence_justification=independence_justification,
        is_auditable=True,
        audit_method=audit_method
    )


if __name__ == "__main__":
    # Example usage
    print("SafeguardVerification module loaded successfully")

    # Create registry
    registry = SafeguardRegistry()

    # Create an IPL safeguard
    ipl_credit = create_ipl_from_standard(
        category=IPLCategory.RELIEF_DEVICE,
        name="High Pressure Relief Valve",
        independence_justification="Mechanical device independent of control system",
        audit_method="Annual bench test with documented set pressure verification"
    )

    safeguard = Safeguard(
        name="Reactor High Pressure PSV",
        description="Pressure Safety Valve for reactor overpressure protection",
        safeguard_type=SafeguardType.RELIEF_DEVICE,
        tag_number="PSV-101A",
        is_ipl=True,
        ipl_credit=ipl_credit,
        pfd=0.01
    )

    # Register safeguard
    registered = registry.register_safeguard(safeguard)
    print(f"Registered: {registered.safeguard_id}")

    # Validate IPL credit
    validation = registry.validate_ipl_credit(registered.safeguard_id)
    print(f"IPL Valid: {validation['is_valid']}")
