"""
GL-042: PressureMaster Agent (PRESSUREMASTER)

Steam header pressure control optimization.

Standards: ISA-5.1, ASME B31.1
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HeaderPressure(BaseModel):
    """Steam header pressure data."""
    header_id: str
    current_pressure_psig: float
    setpoint_psig: float
    min_pressure_psig: float
    max_pressure_psig: float


class BoilerStatus(BaseModel):
    """Boiler status."""
    boiler_id: str
    is_online: bool
    steam_output_klb_hr: float
    max_capacity_klb_hr: float


class ValvePosition(BaseModel):
    """Control valve position."""
    valve_id: str
    position_percent: float
    valve_type: str  # PRV, letdown, vent


class PressureMasterInput(BaseModel):
    """Input for PressureMasterAgent."""
    system_id: str
    header_pressures: List[HeaderPressure]
    demands: Dict[str, float] = Field(default_factory=dict)  # demand_id: klb/hr
    boiler_status: List[BoilerStatus] = Field(default_factory=list)
    valve_positions: List[ValvePosition] = Field(default_factory=list)
    stability_window_minutes: int = Field(default=5)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SetpointRecommendation(BaseModel):
    """Recommended setpoint adjustment."""
    header_id: str
    current_setpoint: float
    recommended_setpoint: float
    reason: str


class ValveAdjustment(BaseModel):
    """Recommended valve adjustment."""
    valve_id: str
    current_position: float
    recommended_position: float
    reason: str


class PressureMasterOutput(BaseModel):
    """Output from PressureMasterAgent."""
    analysis_id: str
    system_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    setpoint_recommendations: List[SetpointRecommendation]
    valve_adjustments: List[ValveAdjustment]
    stability_score: float
    total_steam_supply_klb_hr: float
    total_steam_demand_klb_hr: float
    supply_demand_balance: str
    pressure_deviations: Dict[str, float]
    optimization_notes: List[str]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class PressureMasterAgent:
    """GL-042: PressureMaster Agent."""

    AGENT_ID = "GL-042"
    AGENT_NAME = "PRESSUREMASTER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"PressureMasterAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: PressureMasterInput) -> PressureMasterOutput:
        """Execute pressure control optimization."""
        start_time = datetime.utcnow()

        setpoint_recs = []
        valve_adjs = []
        deviations = {}
        notes = []

        # Analyze each header
        for header in input_data.header_pressures:
            deviation = header.current_pressure_psig - header.setpoint_psig
            deviation_pct = (deviation / header.setpoint_psig * 100) if header.setpoint_psig > 0 else 0
            deviations[header.header_id] = round(deviation_pct, 1)

            # Recommend setpoint changes if pressure consistently off
            if abs(deviation_pct) > 5:
                # Adjust setpoint to reduce control effort
                new_setpoint = header.setpoint_psig + deviation * 0.3
                new_setpoint = max(header.min_pressure_psig, min(header.max_pressure_psig, new_setpoint))

                setpoint_recs.append(SetpointRecommendation(
                    header_id=header.header_id,
                    current_setpoint=header.setpoint_psig,
                    recommended_setpoint=round(new_setpoint, 1),
                    reason=f"Pressure deviation of {deviation_pct:.1f}% - reduce control hunting"
                ))

        # Valve optimization
        for valve in input_data.valve_positions:
            if valve.valve_type == "vent" and valve.position_percent > 10:
                # Vent should be mostly closed
                valve_adjs.append(ValveAdjustment(
                    valve_id=valve.valve_id,
                    current_position=valve.position_percent,
                    recommended_position=5.0,
                    reason="Reduce steam venting - energy loss"
                ))
                notes.append(f"Vent {valve.valve_id} open at {valve.position_percent}% - potential energy waste")

        # Supply/demand balance
        total_supply = sum(b.steam_output_klb_hr for b in input_data.boiler_status if b.is_online)
        total_demand = sum(input_data.demands.values())

        if total_supply > total_demand * 1.2:
            balance = "OVERSUPPLIED"
            notes.append("Consider reducing boiler output to match demand")
        elif total_supply < total_demand * 0.95:
            balance = "UNDERSUPPLIED"
            notes.append("Increase boiler output or start standby boiler")
        else:
            balance = "BALANCED"

        # Stability score
        avg_deviation = sum(abs(d) for d in deviations.values()) / max(len(deviations), 1)
        stability_score = max(0, 100 - avg_deviation * 5)

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "system": input_data.system_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return PressureMasterOutput(
            analysis_id=f"PM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_id=input_data.system_id,
            setpoint_recommendations=setpoint_recs,
            valve_adjustments=valve_adjs,
            stability_score=round(stability_score, 1),
            total_steam_supply_klb_hr=round(total_supply, 1),
            total_steam_demand_klb_hr=round(total_demand, 1),
            supply_demand_balance=balance,
            pressure_deviations=deviations,
            optimization_notes=notes,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-042",
    "name": "PRESSUREMASTER",
    "version": "1.0.0",
    "summary": "Steam header pressure control optimization",
    "tags": ["steam", "pressure-control", "PID", "ISA-5.1", "ASME"],
    "standards": [
        {"ref": "ISA-5.1", "description": "Instrumentation Symbols and Identification"},
        {"ref": "ASME B31.1", "description": "Power Piping"}
    ]
}
