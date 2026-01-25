"""
BypassManager - Bypass Management per IEC 61511

This module implements bypass (override) management for Emergency Shutdown
Systems per IEC 61511-1 Clause 11.7. Bypasses are temporary deactivations
of safety functions that must be:
- Properly authorized
- Time-limited
- Alarmed
- Logged

Reference: IEC 61511-1 Clause 11.7, ISA TR84.00.09

Example:
    >>> from greenlang.safety.esd.bypass_manager import BypassManager
    >>> manager = BypassManager()
    >>> record = manager.request_bypass(request)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class BypassType(str, Enum):
    """Types of bypasses per IEC 61511."""

    MAINTENANCE = "maintenance"  # For maintenance activities
    TESTING = "testing"  # For proof testing
    STARTUP = "startup"  # For plant startup/commissioning
    PROCESS = "process"  # For process upset handling
    FAULT = "fault"  # Due to equipment fault


class BypassStatus(str, Enum):
    """Bypass status."""

    REQUESTED = "requested"
    APPROVED = "approved"
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class BypassRequest(BaseModel):
    """Bypass request specification."""

    request_id: str = Field(
        default_factory=lambda: f"BYP-{uuid.uuid4().hex[:8].upper()}",
        description="Request identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF to bypass"
    )
    equipment_ids: List[str] = Field(
        default_factory=list,
        description="Equipment affected by bypass"
    )
    bypass_type: BypassType = Field(
        ...,
        description="Type of bypass"
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Detailed reason for bypass"
    )
    requested_by: str = Field(
        ...,
        description="Person requesting bypass"
    )
    requested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request timestamp"
    )
    duration_hours: float = Field(
        ...,
        gt=0,
        le=24,  # Max 24 hours per IEC 61511
        description="Requested duration (hours)"
    )
    compensating_measures: List[str] = Field(
        default_factory=list,
        description="Compensating measures during bypass"
    )
    work_permit_ref: Optional[str] = Field(
        None,
        description="Associated work permit"
    )
    risk_assessment_ref: Optional[str] = Field(
        None,
        description="Risk assessment reference"
    )


class BypassRecord(BaseModel):
    """Complete bypass record for audit trail."""

    record_id: str = Field(
        default_factory=lambda: f"BYPR-{uuid.uuid4().hex[:8].upper()}",
        description="Record identifier"
    )
    request: BypassRequest = Field(
        ...,
        description="Original request"
    )
    status: BypassStatus = Field(
        default=BypassStatus.REQUESTED,
        description="Current status"
    )
    approved_by: Optional[str] = Field(
        None,
        description="Approver"
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
    alarm_generated: bool = Field(
        default=False,
        description="Was alarm generated"
    )
    alarm_acknowledged: bool = Field(
        default=False,
        description="Was alarm acknowledged"
    )
    extension_count: int = Field(
        default=0,
        description="Number of extensions"
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


class BypassManager:
    """
    Bypass Manager for ESD/SIS Systems.

    Manages safety system bypasses per IEC 61511-1 Clause 11.7.
    Key requirements:
    - Bypasses must be authorized
    - Maximum duration limits apply
    - Alarms must be generated
    - Complete logging required
    - Compensating measures documented

    The manager follows IEC 61511 principles:
    - No indefinite bypasses
    - Clear accountability
    - Full traceability

    Attributes:
        bypasses: Active bypass records
        max_bypass_hours: Maximum bypass duration

    Example:
        >>> manager = BypassManager()
        >>> request = BypassRequest(sif_id="SIF-001", ...)
        >>> record = manager.request_bypass(request)
    """

    # Maximum bypass durations by type (hours)
    MAX_DURATION_BY_TYPE: Dict[BypassType, float] = {
        BypassType.MAINTENANCE: 8.0,
        BypassType.TESTING: 4.0,
        BypassType.STARTUP: 24.0,
        BypassType.PROCESS: 8.0,
        BypassType.FAULT: 24.0,  # Longer for fault investigation
    }

    def __init__(
        self,
        max_extensions: int = 2,
        alarm_callback: Optional[callable] = None
    ):
        """
        Initialize BypassManager.

        Args:
            max_extensions: Maximum number of extensions allowed
            alarm_callback: Callback for alarm generation
        """
        self.bypasses: Dict[str, BypassRecord] = {}
        self.max_extensions = max_extensions
        self.alarm_callback = alarm_callback or self._default_alarm

        logger.info("BypassManager initialized")

    def request_bypass(
        self,
        request: BypassRequest
    ) -> BypassRecord:
        """
        Submit a bypass request.

        Args:
            request: BypassRequest specification

        Returns:
            BypassRecord in requested status
        """
        logger.warning(
            f"Bypass requested for {request.sif_id} "
            f"by {request.requested_by}: {request.reason}"
        )

        # Validate duration against type limits
        max_allowed = self.MAX_DURATION_BY_TYPE.get(
            request.bypass_type, 8.0
        )
        if request.duration_hours > max_allowed:
            logger.warning(
                f"Requested duration {request.duration_hours}h exceeds "
                f"maximum {max_allowed}h for {request.bypass_type.value}"
            )
            request.duration_hours = max_allowed

        # Check for existing active bypass
        existing = self._get_active_bypass(request.sif_id)
        if existing:
            logger.warning(
                f"Active bypass already exists for {request.sif_id}: "
                f"{existing.record_id}"
            )

        # Create record
        record = BypassRecord(
            request=request,
            status=BypassStatus.REQUESTED,
        )

        # Add to audit trail
        record.audit_trail.append({
            "action": "requested",
            "timestamp": datetime.utcnow().isoformat(),
            "user": request.requested_by,
            "details": {
                "sif_id": request.sif_id,
                "bypass_type": request.bypass_type.value,
                "duration_hours": request.duration_hours,
                "reason": request.reason,
            }
        })

        # Calculate provenance
        record.provenance_hash = self._calculate_provenance(record)

        # Store
        self.bypasses[record.record_id] = record

        logger.info(f"Bypass request created: {record.record_id}")

        return record

    def approve_bypass(
        self,
        record_id: str,
        approved_by: str,
        modified_duration: Optional[float] = None
    ) -> BypassRecord:
        """
        Approve a bypass request.

        Args:
            record_id: Bypass record ID
            approved_by: Approver name
            modified_duration: Optional modified duration

        Returns:
            Updated BypassRecord

        Raises:
            ValueError: If approval is invalid
        """
        if record_id not in self.bypasses:
            raise ValueError(f"Bypass record not found: {record_id}")

        record = self.bypasses[record_id]

        if record.status != BypassStatus.REQUESTED:
            raise ValueError(
                f"Cannot approve bypass in status: {record.status.value}"
            )

        # Update duration if modified
        if modified_duration:
            record.request.duration_hours = modified_duration

        record.status = BypassStatus.APPROVED
        record.approved_by = approved_by
        record.approved_at = datetime.utcnow()

        record.audit_trail.append({
            "action": "approved",
            "timestamp": datetime.utcnow().isoformat(),
            "user": approved_by,
            "details": {
                "final_duration_hours": record.request.duration_hours,
            }
        })

        record.provenance_hash = self._calculate_provenance(record)

        logger.info(f"Bypass {record_id} approved by {approved_by}")

        return record

    def activate_bypass(
        self,
        record_id: str,
        activated_by: str
    ) -> BypassRecord:
        """
        Activate an approved bypass.

        Args:
            record_id: Bypass record ID
            activated_by: Person activating

        Returns:
            Updated BypassRecord
        """
        if record_id not in self.bypasses:
            raise ValueError(f"Bypass record not found: {record_id}")

        record = self.bypasses[record_id]

        if record.status != BypassStatus.APPROVED:
            raise ValueError(
                f"Cannot activate bypass in status: {record.status.value}"
            )

        record.status = BypassStatus.ACTIVE
        record.activated_at = datetime.utcnow()
        record.expires_at = record.activated_at + timedelta(
            hours=record.request.duration_hours
        )

        # Generate alarm
        record.alarm_generated = True
        self.alarm_callback(
            f"BYPASS ACTIVE: {record.request.sif_id}",
            record_id
        )

        record.audit_trail.append({
            "action": "activated",
            "timestamp": datetime.utcnow().isoformat(),
            "user": activated_by,
            "details": {
                "expires_at": record.expires_at.isoformat(),
            }
        })

        record.provenance_hash = self._calculate_provenance(record)

        logger.warning(
            f"BYPASS ACTIVATED: {record.request.sif_id} "
            f"(expires: {record.expires_at.isoformat()})"
        )

        return record

    def deactivate_bypass(
        self,
        record_id: str,
        deactivated_by: str,
        reason: str = "Normal deactivation"
    ) -> BypassRecord:
        """
        Deactivate an active bypass.

        Args:
            record_id: Bypass record ID
            deactivated_by: Person deactivating
            reason: Deactivation reason

        Returns:
            Updated BypassRecord
        """
        if record_id not in self.bypasses:
            raise ValueError(f"Bypass record not found: {record_id}")

        record = self.bypasses[record_id]

        if record.status not in [BypassStatus.ACTIVE, BypassStatus.APPROVED]:
            raise ValueError(
                f"Cannot deactivate bypass in status: {record.status.value}"
            )

        previous_status = record.status
        record.status = BypassStatus.CANCELLED
        record.deactivated_at = datetime.utcnow()
        record.deactivated_by = deactivated_by

        record.audit_trail.append({
            "action": "deactivated",
            "timestamp": datetime.utcnow().isoformat(),
            "user": deactivated_by,
            "details": {
                "previous_status": previous_status.value,
                "reason": reason,
            }
        })

        record.provenance_hash = self._calculate_provenance(record)

        logger.info(f"Bypass {record_id} deactivated by {deactivated_by}")

        return record

    def extend_bypass(
        self,
        record_id: str,
        extension_hours: float,
        extended_by: str,
        reason: str
    ) -> BypassRecord:
        """
        Extend an active bypass.

        Args:
            record_id: Bypass record ID
            extension_hours: Hours to extend
            extended_by: Person extending
            reason: Extension reason

        Returns:
            Updated BypassRecord
        """
        if record_id not in self.bypasses:
            raise ValueError(f"Bypass record not found: {record_id}")

        record = self.bypasses[record_id]

        if record.status != BypassStatus.ACTIVE:
            raise ValueError(
                f"Cannot extend bypass in status: {record.status.value}"
            )

        if record.extension_count >= self.max_extensions:
            raise ValueError(
                f"Maximum extensions ({self.max_extensions}) reached"
            )

        # Check against maximum allowed
        new_total = record.request.duration_hours + extension_hours
        max_allowed = self.MAX_DURATION_BY_TYPE.get(
            record.request.bypass_type, 8.0
        ) * 2  # Allow up to 2x for extensions

        if new_total > max_allowed:
            raise ValueError(
                f"Extended duration {new_total}h exceeds maximum {max_allowed}h"
            )

        record.request.duration_hours = new_total
        record.expires_at = record.activated_at + timedelta(hours=new_total)
        record.extension_count += 1

        record.audit_trail.append({
            "action": "extended",
            "timestamp": datetime.utcnow().isoformat(),
            "user": extended_by,
            "details": {
                "extension_hours": extension_hours,
                "new_total_hours": new_total,
                "new_expires_at": record.expires_at.isoformat(),
                "extension_count": record.extension_count,
                "reason": reason,
            }
        })

        record.provenance_hash = self._calculate_provenance(record)

        logger.warning(
            f"Bypass {record_id} extended by {extension_hours}h "
            f"(total: {new_total}h)"
        )

        return record

    def check_expired_bypasses(self) -> List[BypassRecord]:
        """
        Check and expire bypasses that have exceeded time limit.

        Returns:
            List of newly expired bypasses
        """
        now = datetime.utcnow()
        expired = []

        for record in self.bypasses.values():
            if record.status == BypassStatus.ACTIVE:
                if record.expires_at and now > record.expires_at:
                    record.status = BypassStatus.EXPIRED
                    record.deactivated_at = now

                    record.audit_trail.append({
                        "action": "expired",
                        "timestamp": now.isoformat(),
                        "details": {
                            "scheduled_expiry": record.expires_at.isoformat(),
                        }
                    })

                    record.provenance_hash = self._calculate_provenance(record)

                    expired.append(record)

                    logger.warning(
                        f"BYPASS EXPIRED: {record.request.sif_id} "
                        f"({record.record_id})"
                    )

        return expired

    def get_active_bypasses(self) -> List[BypassRecord]:
        """Get all active bypasses."""
        return [
            r for r in self.bypasses.values()
            if r.status == BypassStatus.ACTIVE
        ]

    def get_bypasses_for_sif(self, sif_id: str) -> List[BypassRecord]:
        """Get all bypasses for a specific SIF."""
        return [
            r for r in self.bypasses.values()
            if r.request.sif_id == sif_id
        ]

    def get_bypass_report(self) -> Dict[str, Any]:
        """Generate bypass status report."""
        now = datetime.utcnow()

        active = [r for r in self.bypasses.values()
                  if r.status == BypassStatus.ACTIVE]

        expiring_soon = [
            r for r in active
            if r.expires_at and (r.expires_at - now).total_seconds() < 3600
        ]

        return {
            "report_timestamp": now.isoformat(),
            "total_bypasses": len(self.bypasses),
            "active_count": len(active),
            "expiring_within_hour": len(expiring_soon),
            "pending_requests": sum(
                1 for r in self.bypasses.values()
                if r.status == BypassStatus.REQUESTED
            ),
            "active_details": [
                {
                    "record_id": r.record_id,
                    "sif_id": r.request.sif_id,
                    "bypass_type": r.request.bypass_type.value,
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

    def _get_active_bypass(self, sif_id: str) -> Optional[BypassRecord]:
        """Get active bypass for a SIF."""
        for record in self.bypasses.values():
            if (record.request.sif_id == sif_id and
                record.status == BypassStatus.ACTIVE):
                return record
        return None

    def _default_alarm(self, message: str, bypass_id: str) -> None:
        """Default alarm callback."""
        logger.warning(f"ALARM: {message} (Bypass: {bypass_id})")

    def _calculate_provenance(self, record: BypassRecord) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{record.record_id}|"
            f"{record.request.sif_id}|"
            f"{record.status.value}|"
            f"{len(record.audit_trail)}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
