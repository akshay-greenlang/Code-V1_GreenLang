"""
ManualOverride - Override Management with Logging

This module implements manual override management for Safety Instrumented
Systems per IEC 61511-1 Clause 11.7. Overrides (bypasses) must be:
- Properly authorized
- Time-limited
- Logged for audit trail
- Alarmed

Reference: IEC 61511-1 Clause 11.7

Example:
    >>> from greenlang.safety.failsafe.manual_override import ManualOverride
    >>> override_mgr = ManualOverride()
    >>> request = OverrideRequest(
    ...     equipment_id="XV-001",
    ...     reason="Maintenance",
    ...     requested_by="J.Smith"
    ... )
    >>> record = override_mgr.request_override(request)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class OverrideType(str, Enum):
    """Types of overrides per IEC 61511."""

    MAINTENANCE = "maintenance"  # For maintenance activities
    TESTING = "testing"  # For proof testing
    STARTUP = "startup"  # For plant startup
    PROCESS = "process"  # For process reasons
    EMERGENCY = "emergency"  # Emergency bypass


class OverrideStatus(str, Enum):
    """Override request status."""

    PENDING = "pending"  # Awaiting approval
    APPROVED = "approved"  # Approved, not yet active
    ACTIVE = "active"  # Currently active
    EXPIRED = "expired"  # Time limit exceeded
    CANCELLED = "cancelled"  # Manually cancelled
    DENIED = "denied"  # Request denied


class ApprovalLevel(str, Enum):
    """Approval levels for overrides."""

    OPERATOR = "operator"  # Operator level (short duration)
    SUPERVISOR = "supervisor"  # Supervisor level
    ENGINEER = "engineer"  # Safety engineer level
    MANAGER = "manager"  # Management level (extended duration)


class OverrideRequest(BaseModel):
    """Override request specification."""

    request_id: str = Field(
        default_factory=lambda: f"OR-{uuid.uuid4().hex[:8].upper()}",
        description="Request identifier"
    )
    equipment_id: str = Field(
        ...,
        description="Equipment/SIF to override"
    )
    override_type: OverrideType = Field(
        default=OverrideType.MAINTENANCE,
        description="Type of override"
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Detailed reason for override"
    )
    requested_by: str = Field(
        ...,
        description="Person requesting override"
    )
    requested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request timestamp"
    )
    duration_hours: float = Field(
        default=4.0,
        gt=0,
        le=168,  # Max 1 week
        description="Requested duration in hours"
    )
    compensating_measures: List[str] = Field(
        default_factory=list,
        description="Compensating measures during override"
    )
    work_permit_ref: Optional[str] = Field(
        None,
        description="Work permit reference"
    )
    moc_ref: Optional[str] = Field(
        None,
        description="Management of Change reference"
    )


class OverrideRecord(BaseModel):
    """Complete override record for audit trail."""

    record_id: str = Field(
        default_factory=lambda: f"OVR-{uuid.uuid4().hex[:8].upper()}",
        description="Record identifier"
    )
    request: OverrideRequest = Field(
        ...,
        description="Original request"
    )
    status: OverrideStatus = Field(
        default=OverrideStatus.PENDING,
        description="Current status"
    )
    approval_level: Optional[ApprovalLevel] = Field(
        None,
        description="Approval level granted"
    )
    approved_by: Optional[str] = Field(
        None,
        description="Person who approved"
    )
    approved_at: Optional[datetime] = Field(
        None,
        description="Approval timestamp"
    )
    activated_at: Optional[datetime] = Field(
        None,
        description="Activation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Expiration timestamp"
    )
    deactivated_at: Optional[datetime] = Field(
        None,
        description="Deactivation timestamp"
    )
    deactivated_by: Optional[str] = Field(
        None,
        description="Person who deactivated"
    )
    deactivation_reason: Optional[str] = Field(
        None,
        description="Reason for deactivation"
    )
    extension_count: int = Field(
        default=0,
        description="Number of extensions granted"
    )
    alarm_acknowledged: bool = Field(
        default=False,
        description="Override alarm acknowledged"
    )
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Complete audit trail"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ManualOverride:
    """
    Manual Override Manager for SIS.

    Manages safety system overrides per IEC 61511-1 Clause 11.7.
    Features:
    - Authorization workflow
    - Time-limited overrides
    - Complete audit trail
    - Alarm generation
    - Extension management

    The manager follows zero-hallucination principles:
    - All actions are logged
    - Time limits enforced
    - No implicit approvals

    Attributes:
        overrides: Dict of active override records
        max_extensions: Maximum extension count

    Example:
        >>> manager = ManualOverride()
        >>> request = OverrideRequest(...)
        >>> record = manager.request_override(request)
        >>> record = manager.approve_override(record.record_id, "supervisor", "J.Doe")
    """

    # Maximum duration by approval level (hours)
    MAX_DURATION_BY_LEVEL: Dict[ApprovalLevel, float] = {
        ApprovalLevel.OPERATOR: 4.0,  # 4 hours
        ApprovalLevel.SUPERVISOR: 12.0,  # 12 hours
        ApprovalLevel.ENGINEER: 24.0,  # 24 hours
        ApprovalLevel.MANAGER: 168.0,  # 1 week
    }

    # Required approval level by duration
    REQUIRED_LEVEL_BY_DURATION: List[tuple] = [
        (4.0, ApprovalLevel.OPERATOR),
        (12.0, ApprovalLevel.SUPERVISOR),
        (24.0, ApprovalLevel.ENGINEER),
        (168.0, ApprovalLevel.MANAGER),
    ]

    def __init__(self, max_extensions: int = 2):
        """
        Initialize ManualOverride manager.

        Args:
            max_extensions: Maximum number of extensions allowed
        """
        self.overrides: Dict[str, OverrideRecord] = {}
        self.max_extensions = max_extensions
        logger.info("ManualOverride manager initialized")

    def request_override(
        self,
        request: OverrideRequest
    ) -> OverrideRecord:
        """
        Submit override request.

        Args:
            request: OverrideRequest specification

        Returns:
            OverrideRecord in pending status
        """
        logger.info(
            f"Override requested for {request.equipment_id} "
            f"by {request.requested_by}"
        )

        # Create record
        record = OverrideRecord(
            request=request,
            status=OverrideStatus.PENDING,
        )

        # Add to audit trail
        record.audit_trail.append({
            "action": "request_submitted",
            "timestamp": datetime.utcnow().isoformat(),
            "user": request.requested_by,
            "details": {
                "equipment_id": request.equipment_id,
                "duration_hours": request.duration_hours,
                "reason": request.reason,
            }
        })

        # Calculate required approval level
        required_level = self._get_required_approval_level(request.duration_hours)
        record.audit_trail.append({
            "action": "approval_level_determined",
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "required_level": required_level.value,
                "duration_hours": request.duration_hours,
            }
        })

        # Calculate provenance hash
        record.provenance_hash = self._calculate_provenance(record)

        # Store record
        self.overrides[record.record_id] = record

        logger.info(f"Override request created: {record.record_id}")

        return record

    def approve_override(
        self,
        record_id: str,
        approval_level: str,
        approved_by: str,
        modified_duration_hours: Optional[float] = None
    ) -> OverrideRecord:
        """
        Approve override request.

        Args:
            record_id: Override record ID
            approval_level: Level of approval
            approved_by: Approver name
            modified_duration_hours: Optional modified duration

        Returns:
            Updated OverrideRecord

        Raises:
            ValueError: If approval is invalid
        """
        if record_id not in self.overrides:
            raise ValueError(f"Override record not found: {record_id}")

        record = self.overrides[record_id]

        if record.status != OverrideStatus.PENDING:
            raise ValueError(
                f"Cannot approve override in status: {record.status.value}"
            )

        # Validate approval level
        level = ApprovalLevel(approval_level)
        duration = modified_duration_hours or record.request.duration_hours

        required_level = self._get_required_approval_level(duration)
        if not self._is_approval_level_sufficient(level, required_level):
            raise ValueError(
                f"Approval level {level.value} insufficient. "
                f"Required: {required_level.value}"
            )

        # Update record
        record.status = OverrideStatus.APPROVED
        record.approval_level = level
        record.approved_by = approved_by
        record.approved_at = datetime.utcnow()

        if modified_duration_hours:
            record.request.duration_hours = modified_duration_hours

        # Add to audit trail
        record.audit_trail.append({
            "action": "approved",
            "timestamp": datetime.utcnow().isoformat(),
            "user": approved_by,
            "details": {
                "approval_level": level.value,
                "duration_hours": duration,
            }
        })

        # Update provenance
        record.provenance_hash = self._calculate_provenance(record)

        logger.info(
            f"Override {record_id} approved by {approved_by} "
            f"at level {level.value}"
        )

        return record

    def activate_override(
        self,
        record_id: str,
        activated_by: str
    ) -> OverrideRecord:
        """
        Activate approved override.

        Args:
            record_id: Override record ID
            activated_by: Person activating

        Returns:
            Updated OverrideRecord

        Raises:
            ValueError: If activation is invalid
        """
        if record_id not in self.overrides:
            raise ValueError(f"Override record not found: {record_id}")

        record = self.overrides[record_id]

        if record.status != OverrideStatus.APPROVED:
            raise ValueError(
                f"Cannot activate override in status: {record.status.value}"
            )

        # Update record
        record.status = OverrideStatus.ACTIVE
        record.activated_at = datetime.utcnow()
        record.expires_at = (
            record.activated_at +
            timedelta(hours=record.request.duration_hours)
        )

        # Add to audit trail
        record.audit_trail.append({
            "action": "activated",
            "timestamp": datetime.utcnow().isoformat(),
            "user": activated_by,
            "details": {
                "expires_at": record.expires_at.isoformat(),
            }
        })

        # Update provenance
        record.provenance_hash = self._calculate_provenance(record)

        logger.warning(
            f"OVERRIDE ACTIVATED: {record.request.equipment_id} "
            f"(expires: {record.expires_at.isoformat()})"
        )

        return record

    def deactivate_override(
        self,
        record_id: str,
        deactivated_by: str,
        reason: str
    ) -> OverrideRecord:
        """
        Deactivate active override.

        Args:
            record_id: Override record ID
            deactivated_by: Person deactivating
            reason: Reason for deactivation

        Returns:
            Updated OverrideRecord
        """
        if record_id not in self.overrides:
            raise ValueError(f"Override record not found: {record_id}")

        record = self.overrides[record_id]

        if record.status not in [OverrideStatus.ACTIVE, OverrideStatus.APPROVED]:
            raise ValueError(
                f"Cannot deactivate override in status: {record.status.value}"
            )

        # Update record
        previous_status = record.status
        record.status = OverrideStatus.CANCELLED
        record.deactivated_at = datetime.utcnow()
        record.deactivated_by = deactivated_by
        record.deactivation_reason = reason

        # Add to audit trail
        record.audit_trail.append({
            "action": "deactivated",
            "timestamp": datetime.utcnow().isoformat(),
            "user": deactivated_by,
            "details": {
                "previous_status": previous_status.value,
                "reason": reason,
            }
        })

        # Update provenance
        record.provenance_hash = self._calculate_provenance(record)

        logger.info(
            f"Override {record_id} deactivated by {deactivated_by}: {reason}"
        )

        return record

    def extend_override(
        self,
        record_id: str,
        extension_hours: float,
        extended_by: str,
        approval_level: str,
        reason: str
    ) -> OverrideRecord:
        """
        Extend active override.

        Args:
            record_id: Override record ID
            extension_hours: Hours to extend
            extended_by: Person extending
            approval_level: Approval level for extension
            reason: Reason for extension

        Returns:
            Updated OverrideRecord

        Raises:
            ValueError: If extension is invalid
        """
        if record_id not in self.overrides:
            raise ValueError(f"Override record not found: {record_id}")

        record = self.overrides[record_id]

        if record.status != OverrideStatus.ACTIVE:
            raise ValueError(
                f"Cannot extend override in status: {record.status.value}"
            )

        if record.extension_count >= self.max_extensions:
            raise ValueError(
                f"Maximum extensions ({self.max_extensions}) reached"
            )

        # Validate approval level for total duration
        total_hours = record.request.duration_hours + extension_hours
        level = ApprovalLevel(approval_level)
        required_level = self._get_required_approval_level(total_hours)

        if not self._is_approval_level_sufficient(level, required_level):
            raise ValueError(
                f"Approval level {level.value} insufficient for "
                f"{total_hours} hours. Required: {required_level.value}"
            )

        # Update record
        record.request.duration_hours = total_hours
        record.expires_at = (
            record.activated_at +
            timedelta(hours=total_hours)
        )
        record.extension_count += 1

        # Add to audit trail
        record.audit_trail.append({
            "action": "extended",
            "timestamp": datetime.utcnow().isoformat(),
            "user": extended_by,
            "details": {
                "extension_hours": extension_hours,
                "total_hours": total_hours,
                "extension_count": record.extension_count,
                "new_expires_at": record.expires_at.isoformat(),
                "reason": reason,
            }
        })

        # Update provenance
        record.provenance_hash = self._calculate_provenance(record)

        logger.warning(
            f"Override {record_id} extended by {extension_hours} hours "
            f"(total: {total_hours} hours)"
        )

        return record

    def check_expired_overrides(self) -> List[OverrideRecord]:
        """
        Check and expire overrides that have exceeded time limit.

        Returns:
            List of newly expired overrides
        """
        now = datetime.utcnow()
        expired = []

        for record in self.overrides.values():
            if record.status == OverrideStatus.ACTIVE:
                if record.expires_at and now > record.expires_at:
                    record.status = OverrideStatus.EXPIRED
                    record.deactivated_at = now
                    record.deactivation_reason = "Time limit exceeded"

                    record.audit_trail.append({
                        "action": "expired",
                        "timestamp": now.isoformat(),
                        "details": {
                            "expired_at": record.expires_at.isoformat(),
                        }
                    })

                    record.provenance_hash = self._calculate_provenance(record)

                    expired.append(record)

                    logger.warning(
                        f"Override {record.record_id} EXPIRED for "
                        f"{record.request.equipment_id}"
                    )

        return expired

    def get_active_overrides(self) -> List[OverrideRecord]:
        """Get all active overrides."""
        return [
            r for r in self.overrides.values()
            if r.status == OverrideStatus.ACTIVE
        ]

    def get_overrides_for_equipment(
        self,
        equipment_id: str
    ) -> List[OverrideRecord]:
        """Get all overrides for specific equipment."""
        return [
            r for r in self.overrides.values()
            if r.request.equipment_id == equipment_id
        ]

    def get_override_report(self) -> Dict[str, Any]:
        """
        Generate override status report.

        Returns:
            Report dictionary
        """
        now = datetime.utcnow()

        active = [r for r in self.overrides.values()
                  if r.status == OverrideStatus.ACTIVE]

        expiring_soon = [
            r for r in active
            if r.expires_at and (r.expires_at - now).total_seconds() < 3600
        ]

        return {
            "report_timestamp": now.isoformat(),
            "total_records": len(self.overrides),
            "active_overrides": len(active),
            "expiring_within_hour": len(expiring_soon),
            "pending_requests": sum(
                1 for r in self.overrides.values()
                if r.status == OverrideStatus.PENDING
            ),
            "active_details": [
                {
                    "record_id": r.record_id,
                    "equipment_id": r.request.equipment_id,
                    "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                    "remaining_hours": (
                        (r.expires_at - now).total_seconds() / 3600
                        if r.expires_at else None
                    ),
                }
                for r in active
            ],
            "provenance_hash": hashlib.sha256(
                f"{now.isoformat()}|{len(active)}".encode()
            ).hexdigest()
        }

    def _get_required_approval_level(
        self,
        duration_hours: float
    ) -> ApprovalLevel:
        """Determine required approval level based on duration."""
        for threshold, level in self.REQUIRED_LEVEL_BY_DURATION:
            if duration_hours <= threshold:
                return level
        return ApprovalLevel.MANAGER

    def _is_approval_level_sufficient(
        self,
        provided: ApprovalLevel,
        required: ApprovalLevel
    ) -> bool:
        """Check if provided approval level is sufficient."""
        level_order = [
            ApprovalLevel.OPERATOR,
            ApprovalLevel.SUPERVISOR,
            ApprovalLevel.ENGINEER,
            ApprovalLevel.MANAGER,
        ]
        return level_order.index(provided) >= level_order.index(required)

    def _calculate_provenance(self, record: OverrideRecord) -> str:
        """Calculate SHA-256 provenance hash for record."""
        provenance_str = (
            f"{record.record_id}|"
            f"{record.request.equipment_id}|"
            f"{record.status.value}|"
            f"{len(record.audit_trail)}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
