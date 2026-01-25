"""
GL-001 ThermalCommand Orchestrator - Event Handlers

This module provides event handlers for various orchestrator events
including system events, safety events, and compliance events.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_001_thermal_command.schemas import (
    OrchestratorEvent,
    SafetyEvent,
    Priority,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE EVENT HANDLER
# =============================================================================

class EventHandler(ABC):
    """
    Base class for event handlers.

    Event handlers process events from the orchestrator and
    trigger appropriate actions.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the event handler.

        Args:
            name: Handler name
        """
        self.name = name
        self._callbacks: Dict[str, List[Callable]] = {}
        self._event_count = 0

    @abstractmethod
    async def handle(self, event: OrchestratorEvent) -> None:
        """
        Handle an event.

        Args:
            event: Event to handle
        """
        pass

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[OrchestratorEvent], None],
    ) -> None:
        """Register a callback for an event type."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def _invoke_callbacks(self, event: OrchestratorEvent) -> None:
        """Invoke registered callbacks for an event."""
        callbacks = self._callbacks.get(event.event_type, [])
        callbacks.extend(self._callbacks.get("*", []))  # Wildcard callbacks

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error in {self.name}: {e}")

    @property
    def event_count(self) -> int:
        """Get total events handled."""
        return self._event_count


# =============================================================================
# SAFETY EVENT HANDLER
# =============================================================================

class SafetyEventHandler(EventHandler):
    """
    Handler for safety-related events.

    This handler processes safety events and can trigger
    emergency actions when necessary.
    """

    def __init__(
        self,
        name: str = "SafetyHandler",
        esd_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Initialize the safety event handler.

        Args:
            name: Handler name
            esd_callback: Callback to trigger ESD
        """
        super().__init__(name)
        self._esd_callback = esd_callback
        self._active_alarms: List[SafetyEvent] = []
        self._alarm_history: List[SafetyEvent] = []

    async def handle(self, event: OrchestratorEvent) -> None:
        """
        Handle a safety event.

        Args:
            event: Event to handle
        """
        self._event_count += 1

        # Convert to SafetyEvent if needed
        if isinstance(event.payload, dict):
            safety_event = SafetyEvent(
                event_type=event.event_type,
                severity=event.payload.get("severity", "warning"),
                description=event.payload.get("description", ""),
                source_equipment=event.payload.get("source_equipment"),
                threshold_value=event.payload.get("threshold_value"),
                actual_value=event.payload.get("actual_value"),
            )
        else:
            safety_event = SafetyEvent(
                event_type=event.event_type,
                severity="info",
                description=str(event.payload),
            )

        logger.warning(
            f"Safety event: {safety_event.event_type} - "
            f"{safety_event.description}"
        )

        # Track alarm
        self._active_alarms.append(safety_event)
        self._alarm_history.append(safety_event)

        # Check for emergency conditions
        if safety_event.severity == "critical":
            await self._handle_critical_event(safety_event)

        # Invoke callbacks
        self._invoke_callbacks(event)

    async def _handle_critical_event(self, event: SafetyEvent) -> None:
        """Handle a critical safety event."""
        logger.critical(f"CRITICAL SAFETY EVENT: {event.description}")

        # Determine if ESD should be triggered
        esd_triggers = [
            "OVERPRESSURE",
            "OVERTEMPERATURE",
            "FLAME_FAILURE",
            "EMERGENCY_STOP",
            "GAS_LEAK",
            "RUNAWAY",
        ]

        if any(trigger in event.event_type.upper() for trigger in esd_triggers):
            if self._esd_callback:
                logger.critical("Triggering Emergency Shutdown")
                self._esd_callback()

    def acknowledge_alarm(self, event_id: str, user: str) -> bool:
        """
        Acknowledge an alarm.

        Args:
            event_id: Event ID to acknowledge
            user: User acknowledging

        Returns:
            True if acknowledged successfully
        """
        for alarm in self._active_alarms:
            if alarm.event_id == event_id:
                alarm.acknowledged = True
                alarm.acknowledged_by = user
                logger.info(f"Alarm {event_id} acknowledged by {user}")
                return True
        return False

    def clear_alarm(self, event_id: str) -> bool:
        """
        Clear an alarm.

        Args:
            event_id: Event ID to clear

        Returns:
            True if cleared successfully
        """
        self._active_alarms = [
            a for a in self._active_alarms if a.event_id != event_id
        ]
        return True

    @property
    def active_alarm_count(self) -> int:
        """Get count of active alarms."""
        return len(self._active_alarms)

    @property
    def active_alarms(self) -> List[SafetyEvent]:
        """Get list of active alarms."""
        return list(self._active_alarms)

    def get_alarm_summary(self) -> Dict[str, Any]:
        """Get alarm summary."""
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        for alarm in self._active_alarms:
            severity_counts[alarm.severity] = severity_counts.get(
                alarm.severity, 0
            ) + 1

        return {
            "total_active": len(self._active_alarms),
            "by_severity": severity_counts,
            "total_historical": len(self._alarm_history),
            "unacknowledged": sum(
                1 for a in self._active_alarms if not a.acknowledged
            ),
        }


# =============================================================================
# COMPLIANCE EVENT HANDLER
# =============================================================================

class ComplianceEventHandler(EventHandler):
    """
    Handler for compliance-related events.

    This handler tracks compliance events and maintains
    audit trails for regulatory reporting.
    """

    def __init__(
        self,
        name: str = "ComplianceHandler",
        compliance_frameworks: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the compliance event handler.

        Args:
            name: Handler name
            compliance_frameworks: Active compliance frameworks
        """
        super().__init__(name)
        self._frameworks = compliance_frameworks or [
            "ISO_14064",
            "GHG_PROTOCOL",
            "EPA_40_CFR",
        ]
        self._compliance_events: Dict[str, List[OrchestratorEvent]] = {
            fw: [] for fw in self._frameworks
        }
        self._violations: List[Dict[str, Any]] = []

    async def handle(self, event: OrchestratorEvent) -> None:
        """
        Handle a compliance event.

        Args:
            event: Event to handle
        """
        self._event_count += 1

        # Determine applicable frameworks
        applicable_frameworks = event.payload.get(
            "compliance_frameworks",
            self._frameworks
        )

        # Store event per framework
        for framework in applicable_frameworks:
            if framework in self._compliance_events:
                self._compliance_events[framework].append(event)

        # Check for violations
        if event.payload.get("is_violation", False):
            self._violations.append({
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "framework": applicable_frameworks,
                "description": event.payload.get("violation_description", ""),
                "severity": event.payload.get("severity", "warning"),
            })
            logger.warning(f"Compliance violation recorded: {event.event_id}")

        logger.info(f"Compliance event: {event.event_type}")

        # Invoke callbacks
        self._invoke_callbacks(event)

    def get_framework_events(
        self,
        framework: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[OrchestratorEvent]:
        """
        Get events for a specific compliance framework.

        Args:
            framework: Compliance framework
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            List of events
        """
        events = self._compliance_events.get(framework, [])

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events

    def get_violations(
        self,
        framework: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get compliance violations.

        Args:
            framework: Optional framework filter

        Returns:
            List of violations
        """
        if framework:
            return [v for v in self._violations if framework in v["framework"]]
        return list(self._violations)

    def generate_compliance_report(
        self,
        framework: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """
        Generate a compliance report.

        Args:
            framework: Target compliance framework
            start_time: Report start time
            end_time: Report end time

        Returns:
            Compliance report data
        """
        events = self.get_framework_events(framework, start_time, end_time)
        violations = [
            v for v in self._violations
            if framework in v.get("framework", [])
            and start_time <= v["timestamp"] <= end_time
        ]

        return {
            "framework": framework,
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(events),
            "total_violations": len(violations),
            "violation_details": violations,
            "event_summary": self._summarize_events(events),
        }

    def _summarize_events(
        self,
        events: List[OrchestratorEvent],
    ) -> Dict[str, Any]:
        """Summarize events by type."""
        type_counts: Dict[str, int] = {}
        for event in events:
            type_counts[event.event_type] = type_counts.get(
                event.event_type, 0
            ) + 1

        return {
            "by_type": type_counts,
            "total": len(events),
        }


# =============================================================================
# WORKFLOW EVENT HANDLER
# =============================================================================

class WorkflowEventHandler(EventHandler):
    """
    Handler for workflow-related events.

    Tracks workflow lifecycle events and maintains
    execution history.
    """

    def __init__(self, name: str = "WorkflowHandler") -> None:
        """Initialize the workflow event handler."""
        super().__init__(name)
        self._workflow_states: Dict[str, str] = {}
        self._workflow_history: List[Dict[str, Any]] = []

    async def handle(self, event: OrchestratorEvent) -> None:
        """
        Handle a workflow event.

        Args:
            event: Event to handle
        """
        self._event_count += 1

        workflow_id = event.payload.get("workflow_id")
        if workflow_id:
            new_state = event.payload.get("status", event.event_type)
            old_state = self._workflow_states.get(workflow_id, "unknown")

            self._workflow_states[workflow_id] = new_state
            self._workflow_history.append({
                "workflow_id": workflow_id,
                "timestamp": event.timestamp,
                "old_state": old_state,
                "new_state": new_state,
                "event_type": event.event_type,
            })

            logger.info(
                f"Workflow {workflow_id}: {old_state} -> {new_state}"
            )

        self._invoke_callbacks(event)

    def get_workflow_state(self, workflow_id: str) -> Optional[str]:
        """Get current workflow state."""
        return self._workflow_states.get(workflow_id)

    def get_workflow_history(
        self,
        workflow_id: str,
    ) -> List[Dict[str, Any]]:
        """Get workflow state history."""
        return [
            h for h in self._workflow_history
            if h["workflow_id"] == workflow_id
        ]


# =============================================================================
# METRICS EVENT HANDLER
# =============================================================================

class MetricsEventHandler(EventHandler):
    """
    Handler for metrics events.

    Collects and aggregates metrics for Prometheus export.
    """

    def __init__(self, name: str = "MetricsHandler") -> None:
        """Initialize the metrics event handler."""
        super().__init__(name)
        self._metrics: Dict[str, List[float]] = {}
        self._counters: Dict[str, int] = {}

    async def handle(self, event: OrchestratorEvent) -> None:
        """
        Handle a metrics event.

        Args:
            event: Event to handle
        """
        self._event_count += 1

        # Extract metrics from payload
        metrics = event.payload.get("metrics", {})

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric_name not in self._metrics:
                    self._metrics[metric_name] = []
                self._metrics[metric_name].append(float(value))

        # Update counters
        counters = event.payload.get("counters", {})
        for counter_name, delta in counters.items():
            self._counters[counter_name] = self._counters.get(
                counter_name, 0
            ) + delta

        self._invoke_callbacks(event)

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self._metrics.get(metric_name, [])
        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "last": values[-1],
        }

    def get_counter(self, counter_name: str) -> int:
        """Get counter value."""
        return self._counters.get(counter_name, 0)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Export gauges/histograms
        for metric_name, values in self._metrics.items():
            if values:
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {values[-1]}")

        # Export counters
        for counter_name, value in self._counters.items():
            lines.append(f"# TYPE {counter_name} counter")
            lines.append(f"{counter_name} {value}")

        return "\n".join(lines)
