"""GL-070: Emergency Response Agent (EMERGENCY-RESPONSE).

Manages emergency response procedures for thermal systems.

Standards: OSHA 1910.38, NFPA 1600
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmergencyType(str, Enum):
    FIRE = "FIRE"
    EXPLOSION = "EXPLOSION"
    GAS_LEAK = "GAS_LEAK"
    PRESSURE_FAILURE = "PRESSURE_FAILURE"
    THERMAL_RUNAWAY = "THERMAL_RUNAWAY"
    POWER_FAILURE = "POWER_FAILURE"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EmergencyInput(BaseModel):
    facility_id: str
    emergency_type: EmergencyType
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM)
    affected_systems: List[str] = Field(default_factory=list)
    personnel_count: int = Field(default=10, ge=0)
    has_hazmat: bool = Field(default=False)
    fire_detection_active: bool = Field(default=True)
    suppression_available: bool = Field(default=True)
    evacuation_routes_clear: bool = Field(default=True)
    current_temp_c: Optional[float] = None
    current_pressure_bar: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResponseAction(BaseModel):
    priority: int
    action: str
    responsible: str
    time_critical: bool
    completed: bool = False


class EmergencyResponseOutput(BaseModel):
    facility_id: str
    emergency_type: str
    severity: str
    response_level: str
    immediate_actions: List[ResponseAction]
    evacuation_required: bool
    shutdown_sequence: List[str]
    notification_list: List[str]
    estimated_response_time_min: int
    safety_critical_systems: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class EmergencyResponseAgent:
    AGENT_ID = "GL-070"
    AGENT_NAME = "EMERGENCY-RESPONSE"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"EmergencyResponseAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = EmergencyInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _get_response_level(self, severity: SeverityLevel, emergency: EmergencyType) -> str:
        """Determine response level based on severity and type."""
        if severity == SeverityLevel.CRITICAL:
            return "LEVEL 3 - Full Emergency Response"
        elif severity == SeverityLevel.HIGH:
            return "LEVEL 2 - Major Incident Response"
        elif emergency in [EmergencyType.FIRE, EmergencyType.EXPLOSION]:
            return "LEVEL 2 - Major Incident Response"
        else:
            return "LEVEL 1 - Local Response"

    def _generate_actions(self, inp: EmergencyInput) -> List[ResponseAction]:
        """Generate prioritized response actions."""
        actions = []
        priority = 1

        # Universal first actions
        actions.append(ResponseAction(
            priority=priority,
            action="Sound alarm and initiate emergency notification",
            responsible="Control Room Operator",
            time_critical=True
        ))
        priority += 1

        # Type-specific actions
        if inp.emergency_type == EmergencyType.FIRE:
            actions.append(ResponseAction(
                priority=priority,
                action="Activate fire suppression system",
                responsible="Fire Team",
                time_critical=True
            ))
            priority += 1
            actions.append(ResponseAction(
                priority=priority,
                action="Isolate fuel supply to affected area",
                responsible="Operations",
                time_critical=True
            ))
            priority += 1

        elif inp.emergency_type == EmergencyType.GAS_LEAK:
            actions.append(ResponseAction(
                priority=priority,
                action="Isolate gas supply valves",
                responsible="Operations",
                time_critical=True
            ))
            priority += 1
            actions.append(ResponseAction(
                priority=priority,
                action="Activate ventilation and gas detection",
                responsible="Safety Team",
                time_critical=True
            ))
            priority += 1

        elif inp.emergency_type == EmergencyType.PRESSURE_FAILURE:
            actions.append(ResponseAction(
                priority=priority,
                action="Open pressure relief valves",
                responsible="Operations",
                time_critical=True
            ))
            priority += 1
            actions.append(ResponseAction(
                priority=priority,
                action="Initiate controlled depressurization",
                responsible="Operations",
                time_critical=True
            ))
            priority += 1

        elif inp.emergency_type == EmergencyType.THERMAL_RUNAWAY:
            actions.append(ResponseAction(
                priority=priority,
                action="Initiate emergency cooling",
                responsible="Operations",
                time_critical=True
            ))
            priority += 1
            actions.append(ResponseAction(
                priority=priority,
                action="Cut all heat sources",
                responsible="Operations",
                time_critical=True
            ))
            priority += 1

        # Personnel evacuation
        if inp.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            actions.append(ResponseAction(
                priority=priority,
                action=f"Evacuate {inp.personnel_count} personnel via designated routes",
                responsible="Safety Coordinator",
                time_critical=True
            ))
            priority += 1

        # External notification
        actions.append(ResponseAction(
            priority=priority,
            action="Notify emergency services if required",
            responsible="Emergency Coordinator",
            time_critical=inp.severity == SeverityLevel.CRITICAL
        ))

        return actions

    def _process(self, inp: EmergencyInput) -> EmergencyResponseOutput:
        recommendations = []

        response_level = self._get_response_level(inp.severity, inp.emergency_type)
        actions = self._generate_actions(inp)

        # Evacuation decision
        evacuation = inp.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        evacuation = evacuation or inp.emergency_type in [EmergencyType.FIRE, EmergencyType.EXPLOSION]
        evacuation = evacuation or inp.has_hazmat

        # Shutdown sequence
        shutdown = []
        if inp.affected_systems:
            shutdown.append(f"1. Isolate affected systems: {', '.join(inp.affected_systems)}")
        shutdown.append("2. Secure fuel and energy supplies")
        shutdown.append("3. Activate backup cooling if available")
        shutdown.append("4. Depressurize vessels to safe levels")
        shutdown.append("5. Confirm all personnel accounted for")

        # Notification list
        notifications = ["Control Room", "Facility Manager", "Safety Officer"]
        if inp.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            notifications.extend(["Fire Department", "Emergency Medical Services"])
        if inp.has_hazmat:
            notifications.append("HAZMAT Team")
        if inp.emergency_type == EmergencyType.GAS_LEAK:
            notifications.append("Gas Utility Company")

        # Response time estimate
        if inp.severity == SeverityLevel.CRITICAL:
            response_time = 5
        elif inp.severity == SeverityLevel.HIGH:
            response_time = 10
        else:
            response_time = 15

        # Safety critical systems
        critical_systems = []
        if not inp.fire_detection_active:
            critical_systems.append("Fire Detection - OFFLINE")
            recommendations.append("CRITICAL: Fire detection system offline - manual monitoring required")
        if not inp.suppression_available:
            critical_systems.append("Fire Suppression - UNAVAILABLE")
            recommendations.append("CRITICAL: Suppression system unavailable - prepare portable extinguishers")
        if not inp.evacuation_routes_clear:
            critical_systems.append("Evacuation Routes - BLOCKED")
            recommendations.append("CRITICAL: Clear evacuation routes immediately")

        # Additional recommendations
        if inp.personnel_count > 50:
            recommendations.append("Large personnel count - assign evacuation marshals for each area")
        if inp.current_temp_c and inp.current_temp_c > 500:
            recommendations.append(f"High temperature ({inp.current_temp_c}°C) - maintain safe distance")
        if inp.current_pressure_bar and inp.current_pressure_bar > 10:
            recommendations.append(f"High pressure ({inp.current_pressure_bar} bar) - explosion risk zone")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "emergency": inp.emergency_type.value,
            "severity": inp.severity.value
        }).encode()).hexdigest()

        return EmergencyResponseOutput(
            facility_id=inp.facility_id,
            emergency_type=inp.emergency_type.value,
            severity=inp.severity.value,
            response_level=response_level,
            immediate_actions=actions,
            evacuation_required=evacuation,
            shutdown_sequence=shutdown,
            notification_list=notifications,
            estimated_response_time_min=response_time,
            safety_critical_systems=critical_systems,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-070", "name": "EMERGENCY-RESPONSE", "version": "1.0.0",
    "summary": "Emergency response management for thermal systems",
    "standards": [{"ref": "OSHA 1910.38"}, {"ref": "NFPA 1600"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
