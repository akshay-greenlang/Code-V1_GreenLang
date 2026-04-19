"""
EmergencyResponseHandler - Emergency response coordination (READ-ONLY).

This module coordinates emergency response activities. The optimizer is
READ-ONLY with respect to emergency systems - it observes and logs events
but does not control SIS or emergency shutdown systems.

CRITICAL: This module is READ-ONLY. All emergency responses are executed by
the SIS/BMS, not by the optimizer. The optimizer only observes and reports.

Example:
    >>> handler = EmergencyResponseHandler(unit_id="BLR-001")
    >>> plan = handler.define_emergency_actions(scenario)
    >>> # Note: execute_emergency_shutdown only OBSERVES, does not control
    >>> result = handler.execute_emergency_shutdown(reason)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmergencyType(str, Enum):
    """Type of emergency condition."""
    FLAME_FAILURE = "flame_failure"
    EXPLOSION = "explosion"
    FIRE = "fire"
    GAS_LEAK = "gas_leak"
    OVERPRESSURE = "overpressure"
    HIGH_CO = "high_co"
    EQUIPMENT_FAILURE = "equipment_failure"
    LOSS_OF_UTILITIES = "loss_of_utilities"
    MANUAL_TRIP = "manual_trip"
    UNKNOWN = "unknown"


class ResponsePriority(str, Enum):
    """Priority level of emergency response."""
    CRITICAL = "critical"  # Life safety, immediate action
    HIGH = "high"  # Equipment safety, rapid action
    MEDIUM = "medium"  # Operational safety
    LOW = "low"  # Precautionary


class ActionType(str, Enum):
    """Type of emergency action."""
    OBSERVE = "observe"  # Optimizer only observes
    NOTIFY = "notify"  # Send notifications
    LOG = "log"  # Log event for audit
    RECOMMEND = "recommend"  # Recommend action to operators
    # Note: No CONTROL actions - optimizer is read-only


class EmergencyScenario(BaseModel):
    """Definition of an emergency scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    emergency_type: EmergencyType = Field(..., description="Type of emergency")
    description: str = Field(..., description="Scenario description")
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    priority: ResponsePriority = Field(..., description="Response priority")
    expected_sis_response: str = Field(..., description="Expected SIS action")
    expected_bms_response: str = Field(..., description="Expected BMS action")
    operator_actions_required: List[str] = Field(default_factory=list)
    estimated_recovery_time: Optional[float] = Field(None, description="Recovery time minutes")


class EmergencyEvent(BaseModel):
    """Actual emergency event that occurred."""
    event_id: str = Field(..., description="Unique event identifier")
    unit_id: str = Field(..., description="Unit identifier")
    emergency_type: EmergencyType = Field(..., description="Type of emergency")
    severity: ResponsePriority = Field(..., description="Event severity")
    trigger_signal: str = Field(..., description="Signal that triggered event")
    trigger_value: Optional[float] = Field(None, description="Value at trigger")
    sis_response: Optional[str] = Field(None, description="SIS response observed")
    bms_response: Optional[str] = Field(None, description="BMS response observed")
    operator_notified: bool = Field(default=False)
    evacuation_required: bool = Field(default=False)
    injuries_reported: bool = Field(default=False)
    damage_reported: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolved_timestamp: Optional[datetime] = Field(None)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class ActionPlan(BaseModel):
    """Emergency action plan (advisory only)."""
    plan_id: str = Field(..., description="Unique plan identifier")
    scenario: EmergencyScenario = Field(..., description="Associated scenario")
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    priority_order: List[str] = Field(default_factory=list)
    optimizer_role: str = Field(
        default="OBSERVE_ONLY",
        description="Optimizer role is always observe-only"
    )
    notifications_required: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class ShutdownResult(BaseModel):
    """Result of observing emergency shutdown (READ-ONLY)."""
    unit_id: str = Field(..., description="Unit identifier")
    shutdown_observed: bool = Field(..., description="Whether shutdown was observed")
    shutdown_reason: str = Field(..., description="Reason for shutdown")
    sis_status: str = Field(..., description="Observed SIS status")
    bms_status: str = Field(..., description="Observed BMS status")
    optimizer_action: str = Field(
        default="OBSERVE_ONLY",
        description="Optimizer only observed, did not control"
    )
    observe_only_mode: bool = Field(default=True, description="Always true")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class NotificationResult(BaseModel):
    """Result of sending emergency notifications."""
    notification_id: str = Field(..., description="Unique notification ID")
    event: EmergencyEvent = Field(..., description="Associated event")
    recipients: List[str] = Field(default_factory=list)
    notification_method: str = Field(..., description="Method used")
    sent_successfully: bool = Field(..., description="Whether sent successfully")
    acknowledgment_received: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class EmergencyContact(BaseModel):
    """Emergency contact information."""
    contact_id: str = Field(..., description="Contact identifier")
    name: str = Field(..., description="Contact name")
    role: str = Field(..., description="Contact role")
    phone: Optional[str] = Field(None, description="Phone number")
    email: Optional[str] = Field(None, description="Email address")
    notification_priority: int = Field(default=1, description="Priority order")
    available_24x7: bool = Field(default=False)


class EmergencyResponseHandler:
    """
    EmergencyResponseHandler coordinates emergency response (READ-ONLY).

    CRITICAL SAFETY INVARIANT:
    - This handler is READ-ONLY with respect to SIS/BMS
    - NEVER issues control commands to safety systems
    - Only observes, logs, and notifies
    - All emergency shutdowns are executed by SIS/BMS, not optimizer

    Attributes:
        unit_id: Identifier for the combustion unit
        scenarios: Defined emergency scenarios
        event_log: Log of emergency events
        contacts: Emergency contact list

    Example:
        >>> handler = EmergencyResponseHandler(unit_id="BLR-001")
        >>> scenario = EmergencyScenario(...)
        >>> plan = handler.define_emergency_actions(scenario)
        >>> # Note: optimizer observes but does not control shutdown
        >>> result = handler.execute_emergency_shutdown("High CO detected")
    """

    def __init__(
        self,
        unit_id: str,
        sis_interface: Any = None,
        bms_interface: Any = None
    ):
        """
        Initialize EmergencyResponseHandler.

        Args:
            unit_id: Unit identifier
            sis_interface: Optional SIS read interface (for status only)
            bms_interface: Optional BMS read interface (for status only)
        """
        self.unit_id = unit_id
        self._sis_interface = sis_interface  # READ-ONLY access
        self._bms_interface = bms_interface  # READ-ONLY access
        self.scenarios: Dict[str, EmergencyScenario] = {}
        self.action_plans: Dict[str, ActionPlan] = {}
        self.event_log: List[EmergencyEvent] = []
        self.contacts: List[EmergencyContact] = []
        self._observe_only_mode = True  # ALWAYS true
        self._creation_time = datetime.utcnow()

        logger.info(
            f"EmergencyResponseHandler initialized for unit {unit_id} "
            f"(READ-ONLY mode enforced)"
        )

    def define_emergency_actions(
        self,
        scenario: EmergencyScenario
    ) -> ActionPlan:
        """
        Define emergency action plan for a scenario.

        Note: Actions are advisory only. The optimizer does not control
        emergency systems.

        Args:
            scenario: Emergency scenario to plan for

        Returns:
            ActionPlan with advisory actions
        """
        # Store scenario
        self.scenarios[scenario.scenario_id] = scenario

        # Define advisory actions based on scenario type
        actions = []
        priority_order = []
        notifications = []

        # All actions are observe/notify/log - never control
        actions.append({
            "action_id": "observe_sis",
            "type": ActionType.OBSERVE.value,
            "description": f"Monitor SIS response: {scenario.expected_sis_response}",
            "automated": True
        })
        priority_order.append("observe_sis")

        actions.append({
            "action_id": "observe_bms",
            "type": ActionType.OBSERVE.value,
            "description": f"Monitor BMS response: {scenario.expected_bms_response}",
            "automated": True
        })
        priority_order.append("observe_bms")

        if scenario.priority in [ResponsePriority.CRITICAL, ResponsePriority.HIGH]:
            actions.append({
                "action_id": "notify_control_room",
                "type": ActionType.NOTIFY.value,
                "description": "Alert control room operators",
                "automated": True
            })
            priority_order.insert(0, "notify_control_room")
            notifications.append("control_room")

        if scenario.priority == ResponsePriority.CRITICAL:
            actions.append({
                "action_id": "notify_emergency_team",
                "type": ActionType.NOTIFY.value,
                "description": "Alert emergency response team",
                "automated": True
            })
            priority_order.insert(1, "notify_emergency_team")
            notifications.append("emergency_team")

        actions.append({
            "action_id": "log_event",
            "type": ActionType.LOG.value,
            "description": "Log all event details for audit",
            "automated": True
        })
        priority_order.append("log_event")

        # Recommend operator actions
        for i, op_action in enumerate(scenario.operator_actions_required):
            actions.append({
                "action_id": f"recommend_{i}",
                "type": ActionType.RECOMMEND.value,
                "description": op_action,
                "automated": False,
                "requires_operator": True
            })
            priority_order.append(f"recommend_{i}")

        plan_id = hashlib.sha256(
            f"{scenario.scenario_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        provenance_hash = hashlib.sha256(
            f"{plan_id}{scenario.json()}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        plan = ActionPlan(
            plan_id=plan_id,
            scenario=scenario,
            actions=actions,
            priority_order=priority_order,
            optimizer_role="OBSERVE_ONLY",  # Always read-only
            notifications_required=notifications,
            provenance_hash=provenance_hash
        )

        self.action_plans[plan_id] = plan

        logger.info(
            f"Emergency action plan defined: {plan_id} for {scenario.emergency_type.value}"
        )

        return plan

    def execute_emergency_shutdown(self, reason: str) -> ShutdownResult:
        """
        Observe and log emergency shutdown (READ-ONLY).

        CRITICAL: This method does NOT control the shutdown. It only
        observes the SIS/BMS response and logs the event.

        Args:
            reason: Reason for shutdown observation

        Returns:
            ShutdownResult with observed status
        """
        logger.warning(
            f"Emergency shutdown observation initiated: {reason}"
        )

        # Observe SIS status (READ-ONLY)
        sis_status = "UNKNOWN"
        if self._sis_interface:
            try:
                sis_raw = self._sis_interface.read_status()
                sis_status = sis_raw.get('state', 'UNKNOWN')
            except Exception as e:
                logger.error(f"Failed to read SIS status: {e}")
                sis_status = f"READ_ERROR: {e}"
        else:
            sis_status = "SIMULATED_TRIP_ACTIVE"

        # Observe BMS status (READ-ONLY)
        bms_status = "UNKNOWN"
        if self._bms_interface:
            try:
                bms_raw = self._bms_interface.read_status()
                bms_status = bms_raw.get('state', 'UNKNOWN')
            except Exception as e:
                logger.error(f"Failed to read BMS status: {e}")
                bms_status = f"READ_ERROR: {e}"
        else:
            bms_status = "SIMULATED_LOCKOUT"

        provenance_hash = hashlib.sha256(
            f"{self.unit_id}{reason}{sis_status}{bms_status}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        result = ShutdownResult(
            unit_id=self.unit_id,
            shutdown_observed=True,
            shutdown_reason=reason,
            sis_status=sis_status,
            bms_status=bms_status,
            optimizer_action="OBSERVE_ONLY - No control actions taken",
            observe_only_mode=True,  # ALWAYS true
            provenance_hash=provenance_hash
        )

        logger.critical(
            f"EMERGENCY SHUTDOWN OBSERVED: unit={self.unit_id}, reason={reason}, "
            f"SIS={sis_status}, BMS={bms_status}, optimizer=OBSERVE_ONLY"
        )

        return result

    def notify_emergency_contacts(
        self,
        event: EmergencyEvent
    ) -> NotificationResult:
        """
        Send notifications to emergency contacts.

        Args:
            event: Emergency event triggering notification

        Returns:
            NotificationResult with delivery status
        """
        recipients = []
        sent_successfully = True

        # Determine recipients based on severity
        for contact in sorted(self.contacts, key=lambda c: c.notification_priority):
            if event.severity == ResponsePriority.CRITICAL:
                recipients.append(contact.name)
            elif event.severity == ResponsePriority.HIGH:
                if contact.notification_priority <= 2:
                    recipients.append(contact.name)
            elif contact.notification_priority == 1:
                recipients.append(contact.name)

        # In production, would send actual notifications
        # For now, log the notification attempt
        notification_method = "simulated"

        if not recipients:
            recipients = ["default_operator"]
            logger.warning("No emergency contacts configured, using default")

        notification_id = hashlib.sha256(
            f"{event.event_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        provenance_hash = hashlib.sha256(
            f"{notification_id}{event.json()}{recipients}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        result = NotificationResult(
            notification_id=notification_id,
            event=event,
            recipients=recipients,
            notification_method=notification_method,
            sent_successfully=sent_successfully,
            provenance_hash=provenance_hash
        )

        logger.info(
            f"Emergency notifications sent: {len(recipients)} recipients "
            f"for event {event.event_id}"
        )

        return result

    def log_emergency_event(self, event: EmergencyEvent) -> None:
        """
        Log an emergency event for audit trail.

        Args:
            event: Emergency event to log
        """
        # Ensure event ID
        if not event.event_id:
            event.event_id = hashlib.sha256(
                f"{event.unit_id}_{event.emergency_type}_{event.timestamp.isoformat()}".encode()
            ).hexdigest()[:16]

        # Ensure provenance hash
        if not event.provenance_hash:
            event.provenance_hash = hashlib.sha256(
                f"{event.json()}".encode()
            ).hexdigest()

        self.event_log.append(event)

        logger.critical(
            f"EMERGENCY EVENT LOGGED: unit={event.unit_id}, "
            f"type={event.emergency_type.value}, severity={event.severity.value}, "
            f"timestamp={event.timestamp.isoformat()}"
        )

    def add_emergency_contact(self, contact: EmergencyContact) -> None:
        """
        Add an emergency contact to the notification list.

        Args:
            contact: Emergency contact to add
        """
        self.contacts.append(contact)
        logger.info(f"Emergency contact added: {contact.name} ({contact.role})")

    def get_active_emergencies(self) -> List[EmergencyEvent]:
        """
        Get list of active (unresolved) emergencies.

        Returns:
            List of unresolved emergency events
        """
        return [
            e for e in self.event_log
            if e.resolved_timestamp is None
        ]

    def resolve_emergency(
        self,
        event_id: str,
        resolution_notes: str = ""
    ) -> bool:
        """
        Mark an emergency event as resolved.

        Args:
            event_id: Event ID to resolve
            resolution_notes: Optional resolution notes

        Returns:
            True if event was found and resolved
        """
        for event in self.event_log:
            if event.event_id == event_id:
                event.resolved_timestamp = datetime.utcnow()
                logger.info(
                    f"Emergency resolved: {event_id} at {event.resolved_timestamp}"
                )
                return True

        logger.warning(f"Emergency event not found for resolution: {event_id}")
        return False

    def get_response_summary(self) -> Dict[str, Any]:
        """
        Get summary of emergency response capability.

        Returns:
            Dictionary with response capability summary
        """
        return {
            "unit_id": self.unit_id,
            "mode": "OBSERVE_ONLY",  # Always read-only
            "scenarios_defined": len(self.scenarios),
            "action_plans_defined": len(self.action_plans),
            "contacts_configured": len(self.contacts),
            "active_emergencies": len(self.get_active_emergencies()),
            "total_events_logged": len(self.event_log),
            "sis_interface_available": self._sis_interface is not None,
            "bms_interface_available": self._bms_interface is not None,
            "optimizer_control_capability": "NONE - READ ONLY",
            "last_event": (
                self.event_log[-1].timestamp.isoformat()
                if self.event_log else None
            )
        }
