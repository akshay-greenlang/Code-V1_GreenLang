"""GL-040 SAFETYSENTRY: Process Safety Monitor Agent.

Monitors process heat safety parameters and prevents incidents through
early warning and automated response recommendations.

Standards: IEC 61511, NFPA 85/86, OSHA PSM, API 2510
"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SafetyAlarm(BaseModel):
    """Safety alarm definition."""
    alarm_id: str
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    parameter: str
    current_value: float
    limit_value: float
    unit: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SafetyInput(BaseModel):
    """Input for safety monitoring."""

    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: str = Field("boiler", description="boiler, furnace, heater, oven")

    # Pressure parameters
    pressure_bar: float = Field(..., description="Operating pressure (bar)")
    pressure_high_limit_bar: float = Field(..., description="High pressure limit")
    pressure_hh_limit_bar: float = Field(..., description="High-high pressure limit")
    pressure_low_limit_bar: float = Field(0, description="Low pressure limit")

    # Temperature parameters
    temperature_c: float = Field(..., description="Operating temperature (°C)")
    temperature_high_limit_c: float = Field(..., description="High temp limit")
    temperature_hh_limit_c: float = Field(..., description="High-high temp limit")
    tube_metal_temp_c: Optional[float] = Field(None, description="Tube metal temp")
    tube_metal_limit_c: Optional[float] = Field(None, description="TMT limit")

    # Combustion safety
    flame_detected: bool = Field(True, description="Flame present")
    burner_status: str = Field("RUNNING", description="Burner status")
    fuel_pressure_bar: float = Field(..., ge=0, description="Fuel pressure")
    fuel_pressure_min_bar: float = Field(1.0, description="Min fuel pressure")
    combustion_air_flow_pct: float = Field(100, description="Air flow % of setpoint")
    air_flow_min_pct: float = Field(80, description="Minimum air flow %")

    # Safety interlocks
    interlock_status: Dict[str, bool] = Field(
        default_factory=dict,
        description="Interlock status (True=OK, False=Tripped)"
    )
    sis_status: str = Field("NORMAL", description="SIS system status")
    manual_override_active: bool = Field(False, description="Manual override active")

    # Gas detection
    combustible_gas_pct_lel: float = Field(0, ge=0, description="% LEL")
    combustible_alarm_pct: float = Field(20, description="Alarm setpoint % LEL")
    toxic_gas_ppm: Dict[str, float] = Field(default_factory=dict, description="Toxic gas readings")

    # Operating conditions
    operating_hours_since_inspection: int = Field(0, description="Hours since inspection")
    days_since_sis_test: int = Field(0, description="Days since SIS proof test")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SafetyOutput(BaseModel):
    """Output from safety monitor."""

    # Overall status
    safety_status: str = Field(..., description="SAFE, CAUTION, WARNING, DANGER, EMERGENCY")
    risk_level: int = Field(..., ge=1, le=5, description="Risk level 1-5")

    # Active alarms
    active_alarms: List[SafetyAlarm] = Field(default_factory=list)
    alarm_count_by_severity: Dict[str, int] = Field(default_factory=dict)

    # Parameter status
    pressure_status: str = Field(..., description="NORMAL, HIGH, HIGH_HIGH, LOW")
    temperature_status: str = Field(..., description="NORMAL, HIGH, HIGH_HIGH")
    flame_status: str = Field(..., description="STABLE, UNSTABLE, LOST")
    combustion_status: str = Field(..., description="NORMAL, LEAN, RICH, UNSAFE")

    # Interlock summary
    interlocks_healthy: bool = Field(..., description="All interlocks OK")
    tripped_interlocks: List[str] = Field(default_factory=list)

    # Recommendations
    immediate_actions: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Compliance
    sis_test_overdue: bool = Field(False, description="SIS test overdue")
    inspection_overdue: bool = Field(False, description="Inspection overdue")
    compliance_status: str = Field(..., description="COMPLIANT, AT_RISK, NON_COMPLIANT")

    # Emergency shutdown recommendation
    shutdown_recommended: bool = Field(False, description="ESD recommended")
    shutdown_reason: Optional[str] = Field(None)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def assess_parameter_status(
    value: float,
    high_limit: float,
    hh_limit: float,
    low_limit: float = 0
) -> str:
    """Assess parameter against limits."""
    if value >= hh_limit:
        return "HIGH_HIGH"
    elif value >= high_limit:
        return "HIGH"
    elif value <= low_limit:
        return "LOW"
    return "NORMAL"


def calculate_risk_level(
    alarms: List[SafetyAlarm],
    shutdown_recommended: bool,
    flame_lost: bool
) -> int:
    """Calculate overall risk level (1=lowest, 5=highest)."""
    if shutdown_recommended or flame_lost:
        return 5

    critical_count = sum(1 for a in alarms if a.severity == "CRITICAL")
    high_count = sum(1 for a in alarms if a.severity == "HIGH")

    if critical_count > 0:
        return 5
    elif high_count >= 2:
        return 4
    elif high_count >= 1:
        return 3
    elif alarms:
        return 2
    return 1


class ProcessSafetyMonitorAgent:
    """Process safety monitoring agent."""

    AGENT_ID = "GL-040"
    AGENT_NAME = "SAFETYSENTRY"
    VERSION = "1.0.0"

    # Test intervals
    SIS_TEST_INTERVAL_DAYS = 90
    INSPECTION_INTERVAL_HOURS = 8760  # Annual

    # Gas detection limits
    LEL_WARNING_PCT = 10
    LEL_ALARM_PCT = 20
    LEL_DANGER_PCT = 40

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = SafetyInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: SafetyInput) -> SafetyOutput:
        alarms = []
        immediate_actions = []
        recommendations = []
        warnings = []
        shutdown_recommended = False
        shutdown_reason = None

        # Pressure assessment
        pressure_status = assess_parameter_status(
            inp.pressure_bar,
            inp.pressure_high_limit_bar,
            inp.pressure_hh_limit_bar,
            inp.pressure_low_limit_bar
        )

        if pressure_status == "HIGH_HIGH":
            alarms.append(SafetyAlarm(
                alarm_id="PAHH", description="Pressure very high",
                severity="CRITICAL", parameter="pressure",
                current_value=inp.pressure_bar, limit_value=inp.pressure_hh_limit_bar,
                unit="bar"
            ))
            shutdown_recommended = True
            shutdown_reason = f"Pressure {inp.pressure_bar:.1f} bar exceeds HH limit"
            immediate_actions.append("EMERGENCY: Initiate pressure relief")
        elif pressure_status == "HIGH":
            alarms.append(SafetyAlarm(
                alarm_id="PAH", description="Pressure high",
                severity="HIGH", parameter="pressure",
                current_value=inp.pressure_bar, limit_value=inp.pressure_high_limit_bar,
                unit="bar"
            ))
            immediate_actions.append("Reduce firing rate to lower pressure")

        # Temperature assessment
        temp_status = assess_parameter_status(
            inp.temperature_c,
            inp.temperature_high_limit_c,
            inp.temperature_hh_limit_c
        )

        if temp_status == "HIGH_HIGH":
            alarms.append(SafetyAlarm(
                alarm_id="TAHH", description="Temperature very high",
                severity="CRITICAL", parameter="temperature",
                current_value=inp.temperature_c, limit_value=inp.temperature_hh_limit_c,
                unit="°C"
            ))
            if not shutdown_recommended:
                shutdown_recommended = True
                shutdown_reason = f"Temperature {inp.temperature_c:.0f}°C exceeds HH limit"
            immediate_actions.append("EMERGENCY: Initiate emergency cooling")
        elif temp_status == "HIGH":
            alarms.append(SafetyAlarm(
                alarm_id="TAH", description="Temperature high",
                severity="HIGH", parameter="temperature",
                current_value=inp.temperature_c, limit_value=inp.temperature_high_limit_c,
                unit="°C"
            ))

        # Tube metal temperature
        if inp.tube_metal_temp_c and inp.tube_metal_limit_c:
            if inp.tube_metal_temp_c >= inp.tube_metal_limit_c:
                alarms.append(SafetyAlarm(
                    alarm_id="TMT_HIGH", description="Tube metal temp critical",
                    severity="CRITICAL", parameter="tube_metal_temp",
                    current_value=inp.tube_metal_temp_c, limit_value=inp.tube_metal_limit_c,
                    unit="°C"
                ))
                immediate_actions.append("Reduce heat input - tube damage imminent")
            elif inp.tube_metal_temp_c >= inp.tube_metal_limit_c * 0.95:
                alarms.append(SafetyAlarm(
                    alarm_id="TMT_WARN", description="Tube metal temp high",
                    severity="HIGH", parameter="tube_metal_temp",
                    current_value=inp.tube_metal_temp_c, limit_value=inp.tube_metal_limit_c,
                    unit="°C"
                ))

        # Flame monitoring
        flame_status = "STABLE"
        if not inp.flame_detected and inp.burner_status == "RUNNING":
            flame_status = "LOST"
            alarms.append(SafetyAlarm(
                alarm_id="FLAME_LOSS", description="Flame loss detected",
                severity="CRITICAL", parameter="flame",
                current_value=0, limit_value=1, unit="status"
            ))
            shutdown_recommended = True
            shutdown_reason = "Flame loss - fuel safety shutdown required"
            immediate_actions.append("EMERGENCY: Flame loss - immediate fuel shutoff required")

        # Combustion air
        combustion_status = "NORMAL"
        if inp.combustion_air_flow_pct < inp.air_flow_min_pct:
            combustion_status = "UNSAFE"
            alarms.append(SafetyAlarm(
                alarm_id="AIR_LOW", description="Combustion air low",
                severity="HIGH", parameter="air_flow",
                current_value=inp.combustion_air_flow_pct, limit_value=inp.air_flow_min_pct,
                unit="%"
            ))
            immediate_actions.append("Check FD fan and air dampers")

        # Fuel pressure
        if inp.fuel_pressure_bar < inp.fuel_pressure_min_bar:
            alarms.append(SafetyAlarm(
                alarm_id="FUEL_LOW", description="Fuel pressure low",
                severity="MEDIUM", parameter="fuel_pressure",
                current_value=inp.fuel_pressure_bar, limit_value=inp.fuel_pressure_min_bar,
                unit="bar"
            ))

        # Gas detection
        if inp.combustible_gas_pct_lel >= self.LEL_DANGER_PCT:
            alarms.append(SafetyAlarm(
                alarm_id="GAS_DANGER", description="Combustible gas DANGER",
                severity="CRITICAL", parameter="combustible_gas",
                current_value=inp.combustible_gas_pct_lel, limit_value=self.LEL_DANGER_PCT,
                unit="% LEL"
            ))
            shutdown_recommended = True
            shutdown_reason = f"Combustible gas at {inp.combustible_gas_pct_lel:.0f}% LEL"
            immediate_actions.append("EMERGENCY: Evacuate area, ventilate, eliminate ignition sources")
        elif inp.combustible_gas_pct_lel >= self.LEL_ALARM_PCT:
            alarms.append(SafetyAlarm(
                alarm_id="GAS_ALARM", description="Combustible gas high",
                severity="HIGH", parameter="combustible_gas",
                current_value=inp.combustible_gas_pct_lel, limit_value=self.LEL_ALARM_PCT,
                unit="% LEL"
            ))

        # Interlock status
        tripped = [k for k, v in inp.interlock_status.items() if not v]
        interlocks_healthy = len(tripped) == 0

        if tripped:
            for interlock in tripped:
                alarms.append(SafetyAlarm(
                    alarm_id=f"INTLK_{interlock}", description=f"Interlock {interlock} tripped",
                    severity="HIGH", parameter="interlock",
                    current_value=0, limit_value=1, unit="status"
                ))
            warnings.append(f"Tripped interlocks: {', '.join(tripped)}")

        # Override warning
        if inp.manual_override_active:
            warnings.append("CAUTION: Manual override active - automated protections bypassed")
            alarms.append(SafetyAlarm(
                alarm_id="OVERRIDE", description="Manual override active",
                severity="MEDIUM", parameter="override",
                current_value=1, limit_value=0, unit="status"
            ))

        # Compliance checks
        sis_overdue = inp.days_since_sis_test > self.SIS_TEST_INTERVAL_DAYS
        inspection_overdue = inp.operating_hours_since_inspection > self.INSPECTION_INTERVAL_HOURS

        if sis_overdue:
            warnings.append(f"SIS proof test overdue by {inp.days_since_sis_test - self.SIS_TEST_INTERVAL_DAYS} days")
        if inspection_overdue:
            warnings.append("Annual inspection overdue")

        compliance_status = "COMPLIANT"
        if sis_overdue or inspection_overdue:
            compliance_status = "AT_RISK"
        if sis_overdue and inspection_overdue:
            compliance_status = "NON_COMPLIANT"

        # Calculate risk level
        risk_level = calculate_risk_level(alarms, shutdown_recommended, flame_status == "LOST")

        # Determine overall status
        if shutdown_recommended:
            safety_status = "EMERGENCY"
        elif risk_level >= 4:
            safety_status = "DANGER"
        elif risk_level >= 3:
            safety_status = "WARNING"
        elif risk_level >= 2:
            safety_status = "CAUTION"
        else:
            safety_status = "SAFE"

        # Alarm summary
        alarm_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for alarm in alarms:
            alarm_counts[alarm.severity] = alarm_counts.get(alarm.severity, 0) + 1

        # Recommendations
        if not shutdown_recommended and alarms:
            recommendations.append("Review and acknowledge all active alarms")
        if compliance_status != "COMPLIANT":
            recommendations.append("Schedule overdue safety tests/inspections")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "status": safety_status,
            "alarms": len(alarms),
            "risk": risk_level
        }).encode()).hexdigest()

        return SafetyOutput(
            safety_status=safety_status,
            risk_level=risk_level,
            active_alarms=alarms,
            alarm_count_by_severity=alarm_counts,
            pressure_status=pressure_status,
            temperature_status=temp_status,
            flame_status=flame_status,
            combustion_status=combustion_status,
            interlocks_healthy=interlocks_healthy,
            tripped_interlocks=tripped,
            immediate_actions=immediate_actions,
            recommendations=recommendations,
            warnings=warnings,
            sis_test_overdue=sis_overdue,
            inspection_overdue=inspection_overdue,
            compliance_status=compliance_status,
            shutdown_recommended=shutdown_recommended,
            shutdown_reason=shutdown_reason,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Safety",
            "type": "Monitor",
            "complexity": "High",
            "priority": "P0",
            "standards": ["IEC 61511", "NFPA 85", "NFPA 86", "OSHA PSM"]
        }
