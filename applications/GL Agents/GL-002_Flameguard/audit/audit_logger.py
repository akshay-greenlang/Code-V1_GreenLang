"""
GL-002 FLAMEGUARD - Audit Logger

Comprehensive audit logging for regulatory compliance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Audit event types."""
    # Configuration
    CONFIG_CHANGE = "config_change"
    SETPOINT_CHANGE = "setpoint_change"

    # Operations
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    MODE_CHANGE = "mode_change"

    # Safety
    SAFETY_TRIP = "safety_trip"
    SAFETY_RESET = "safety_reset"
    BYPASS_SET = "bypass_set"
    BYPASS_CLEARED = "bypass_cleared"
    ALARM_ACTIVE = "alarm_active"
    ALARM_ACKNOWLEDGED = "alarm_acknowledged"

    # Calculations
    EFFICIENCY_CALCULATION = "efficiency_calculation"
    EMISSIONS_CALCULATION = "emissions_calculation"
    OPTIMIZATION_RUN = "optimization_run"
    OPTIMIZATION_APPLIED = "optimization_applied"

    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    AUTH_FAILURE = "auth_failure"

    # Data
    DATA_EXPORT = "data_export"
    REPORT_GENERATED = "report_generated"

    # System
    SYSTEM_ERROR = "system_error"
    MAINTENANCE_START = "maintenance_start"
    MAINTENANCE_END = "maintenance_end"


@dataclass
class AuditEntry:
    """Single audit log entry."""
    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    boiler_id: str
    operator: Optional[str] = None
    source: str = "system"  # system, manual, optimization, safety

    # Event details
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Before/after values for changes
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    # Safety classification
    safety_related: bool = False
    requires_review: bool = False

    # Verification
    entry_hash: str = ""

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute entry hash for tamper detection."""
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "boiler_id": self.boiler_id,
            "operator": self.operator,
            "description": self.description,
            "details": self.details,
        }
        json_data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "boiler_id": self.boiler_id,
            "operator": self.operator,
            "source": self.source,
            "description": self.description,
            "details": self.details,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "safety_related": self.safety_related,
            "requires_review": self.requires_review,
            "entry_hash": self.entry_hash,
        }


class AuditLogger:
    """
    Audit logging system for regulatory compliance.

    Features:
    - Tamper-evident logging
    - Safety event classification
    - Operator tracking
    - Change tracking (before/after)
    - Export for regulatory reporting
    """

    def __init__(
        self,
        agent_id: str = "GL-002",
        on_safety_event: Optional[Callable[[AuditEntry], None]] = None,
        max_entries: int = 100000,
    ) -> None:
        self.agent_id = agent_id
        self._on_safety_event = on_safety_event
        self._max_entries = max_entries

        # Audit log storage
        self._entries: List[AuditEntry] = []

        # Chain hash for tamper detection
        self._chain_hash = "0" * 64

        # Statistics
        self._stats = {
            "total_entries": 0,
            "safety_events": 0,
            "config_changes": 0,
            "optimization_events": 0,
        }

        logger.info(f"AuditLogger initialized: {agent_id}")

    def log(
        self,
        event_type: AuditEventType,
        boiler_id: str,
        description: str,
        operator: Optional[str] = None,
        source: str = "system",
        details: Optional[Dict[str, Any]] = None,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
    ) -> AuditEntry:
        """Create audit log entry."""
        # Determine safety classification
        safety_events = [
            AuditEventType.SAFETY_TRIP,
            AuditEventType.SAFETY_RESET,
            AuditEventType.BYPASS_SET,
            AuditEventType.BYPASS_CLEARED,
        ]
        safety_related = event_type in safety_events

        review_events = [
            AuditEventType.SAFETY_TRIP,
            AuditEventType.BYPASS_SET,
            AuditEventType.SYSTEM_ERROR,
        ]
        requires_review = event_type in review_events

        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            boiler_id=boiler_id,
            operator=operator,
            source=source,
            description=description,
            details=details or {},
            old_value=old_value,
            new_value=new_value,
            safety_related=safety_related,
            requires_review=requires_review,
        )

        # Update chain hash
        self._update_chain(entry)

        # Store entry
        self._entries.append(entry)
        self._stats["total_entries"] += 1

        if safety_related:
            self._stats["safety_events"] += 1
            if self._on_safety_event:
                self._on_safety_event(entry)

        if event_type == AuditEventType.CONFIG_CHANGE:
            self._stats["config_changes"] += 1

        if event_type in [AuditEventType.OPTIMIZATION_RUN, AuditEventType.OPTIMIZATION_APPLIED]:
            self._stats["optimization_events"] += 1

        # Trim if needed
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        logger.debug(f"Audit: {event_type.value} - {description}")
        return entry

    def _update_chain(self, entry: AuditEntry) -> None:
        """Update chain hash for tamper detection."""
        combined = f"{self._chain_hash}:{entry.entry_hash}"
        self._chain_hash = hashlib.sha256(combined.encode()).hexdigest()

    def log_setpoint_change(
        self,
        boiler_id: str,
        setpoint_tag: str,
        old_value: float,
        new_value: float,
        operator: Optional[str] = None,
        source: str = "manual",
        reason: str = "",
    ) -> AuditEntry:
        """Log setpoint change."""
        return self.log(
            event_type=AuditEventType.SETPOINT_CHANGE,
            boiler_id=boiler_id,
            description=f"Setpoint {setpoint_tag} changed: {old_value} -> {new_value}",
            operator=operator,
            source=source,
            details={
                "setpoint_tag": setpoint_tag,
                "reason": reason,
            },
            old_value=old_value,
            new_value=new_value,
        )

    def log_safety_trip(
        self,
        boiler_id: str,
        trip_cause: str,
        interlock_tag: Optional[str] = None,
        trip_value: Optional[float] = None,
        setpoint: Optional[float] = None,
    ) -> AuditEntry:
        """Log safety trip event."""
        return self.log(
            event_type=AuditEventType.SAFETY_TRIP,
            boiler_id=boiler_id,
            description=f"Safety trip: {trip_cause}",
            source="safety",
            details={
                "trip_cause": trip_cause,
                "interlock_tag": interlock_tag,
                "trip_value": trip_value,
                "setpoint": setpoint,
            },
        )

    def log_bypass(
        self,
        boiler_id: str,
        interlock_tag: str,
        reason: str,
        operator: str,
        supervisor: str,
        duration_minutes: int,
    ) -> AuditEntry:
        """Log safety bypass."""
        return self.log(
            event_type=AuditEventType.BYPASS_SET,
            boiler_id=boiler_id,
            description=f"Bypass set on {interlock_tag}",
            operator=operator,
            source="manual",
            details={
                "interlock_tag": interlock_tag,
                "reason": reason,
                "supervisor": supervisor,
                "duration_minutes": duration_minutes,
            },
        )

    def log_optimization(
        self,
        boiler_id: str,
        mode: str,
        current_efficiency: float,
        predicted_efficiency: float,
        recommendations: Dict[str, float],
        applied: bool,
        operator: Optional[str] = None,
    ) -> AuditEntry:
        """Log optimization run."""
        event_type = (
            AuditEventType.OPTIMIZATION_APPLIED if applied
            else AuditEventType.OPTIMIZATION_RUN
        )

        return self.log(
            event_type=event_type,
            boiler_id=boiler_id,
            description=f"Optimization ({mode}): {current_efficiency:.1f}% -> {predicted_efficiency:.1f}%",
            operator=operator,
            source="optimization",
            details={
                "mode": mode,
                "current_efficiency": current_efficiency,
                "predicted_efficiency": predicted_efficiency,
                "recommendations": recommendations,
                "applied": applied,
            },
        )

    def log_calculation(
        self,
        boiler_id: str,
        calculation_type: str,
        result: Dict[str, float],
        provenance_hash: str,
    ) -> AuditEntry:
        """Log calculation for audit trail."""
        event_type = {
            "efficiency": AuditEventType.EFFICIENCY_CALCULATION,
            "emissions": AuditEventType.EMISSIONS_CALCULATION,
        }.get(calculation_type, AuditEventType.EFFICIENCY_CALCULATION)

        return self.log(
            event_type=event_type,
            boiler_id=boiler_id,
            description=f"{calculation_type} calculation completed",
            source="calculation",
            details={
                "calculation_type": calculation_type,
                "result": result,
                "provenance_hash": provenance_hash,
            },
        )

    def get_entries(
        self,
        boiler_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operator: Optional[str] = None,
        safety_only: bool = False,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit entries with filters."""
        entries = self._entries

        if boiler_id:
            entries = [e for e in entries if e.boiler_id == boiler_id]

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]

        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        if operator:
            entries = [e for e in entries if e.operator == operator]

        if safety_only:
            entries = [e for e in entries if e.safety_related]

        # Sort by timestamp descending
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    def get_entries_requiring_review(self) -> List[AuditEntry]:
        """Get entries requiring review."""
        return [e for e in self._entries if e.requires_review]

    def export_for_compliance(
        self,
        boiler_id: str,
        start_time: datetime,
        end_time: datetime,
        include_chain_verification: bool = True,
    ) -> Dict:
        """Export audit data for regulatory compliance."""
        entries = self.get_entries(
            boiler_id=boiler_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        export = {
            "agent_id": self.agent_id,
            "boiler_id": boiler_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "entry_count": len(entries),
            "entries": [e.to_dict() for e in reversed(entries)],
        }

        if include_chain_verification:
            export["chain_hash"] = self._chain_hash
            export["chain_verified"] = self._verify_chain()

        return export

    def _verify_chain(self) -> bool:
        """Verify chain hash integrity."""
        computed_hash = "0" * 64

        for entry in self._entries:
            combined = f"{computed_hash}:{entry.entry_hash}"
            computed_hash = hashlib.sha256(combined.encode()).hexdigest()

        return computed_hash == self._chain_hash

    def get_statistics(self) -> Dict:
        """Get audit log statistics."""
        return {
            **self._stats,
            "entries_count": len(self._entries),
            "chain_hash": self._chain_hash[:16] + "...",
            "chain_verified": self._verify_chain(),
        }
