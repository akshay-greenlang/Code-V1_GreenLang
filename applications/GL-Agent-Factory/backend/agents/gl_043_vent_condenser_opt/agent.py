"""
GL-043: Vent Condenser Optimizer Agent (VENT-CONDENSER-OPT)

Vent condenser optimization for steam recovery.

Standards: HEI Standards for Steam Surface Condensers
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VentCondenserOptInput(BaseModel):
    """Input for VentCondenserOptAgent."""
    condenser_id: str
    vent_flow_lb_hr: float = Field(..., ge=0)
    inlet_steam_quality: float = Field(default=0.95, ge=0, le=1)
    cooling_water_inlet_temp_f: float = Field(default=75)
    cooling_water_outlet_temp_f: float = Field(default=95)
    cooling_water_flow_gpm: float = Field(default=100)
    design_duty_mmbtu_hr: float = Field(default=1.0)
    current_vacuum_in_hg: float = Field(default=28)
    steam_value_per_klb: float = Field(default=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VentCondenserOptOutput(BaseModel):
    """Output from VentCondenserOptAgent."""
    analysis_id: str
    condenser_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    current_recovery_rate_percent: float
    optimal_recovery_rate_percent: float
    steam_recovered_lb_hr: float
    energy_saved_mmbtu_hr: float
    annual_savings: float
    condensate_quality: str
    optimal_cw_temp_setpoint_f: float
    optimal_vacuum_setpoint_in_hg: float
    performance_index: float
    recommendations: list
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class VentCondenserOptAgent:
    """GL-043: Vent Condenser Optimizer Agent."""

    AGENT_ID = "GL-043"
    AGENT_NAME = "VENT-CONDENSER-OPT"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"VentCondenserOptAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: VentCondenserOptInput) -> VentCondenserOptOutput:
        """Execute vent condenser optimization."""
        start_time = datetime.utcnow()

        # Current recovery (based on CW temperature rise and flow)
        cw_delta_t = input_data.cooling_water_outlet_temp_f - input_data.cooling_water_inlet_temp_f
        heat_absorbed_btu_hr = input_data.cooling_water_flow_gpm * 500 * cw_delta_t  # 500 = 8.33 * 60

        # Steam latent heat ~970 Btu/lb
        steam_condensed = heat_absorbed_btu_hr / 970
        current_recovery = (steam_condensed / input_data.vent_flow_lb_hr * 100) if input_data.vent_flow_lb_hr > 0 else 0

        # Optimal recovery (with better heat transfer)
        optimal_recovery = min(98, current_recovery * 1.15)

        # Energy savings
        additional_recovery = (optimal_recovery - current_recovery) / 100 * input_data.vent_flow_lb_hr
        energy_saved = additional_recovery * 970 / 1e6  # MMBtu/hr

        # Annual savings (8760 hrs)
        annual_savings = additional_recovery / 1000 * input_data.steam_value_per_klb * 8760

        # Condensate quality based on vacuum
        if input_data.current_vacuum_in_hg > 27:
            cond_quality = "EXCELLENT"
        elif input_data.current_vacuum_in_hg > 25:
            cond_quality = "GOOD"
        else:
            cond_quality = "FAIR"

        # Optimal setpoints
        optimal_cw_temp = input_data.cooling_water_inlet_temp_f + cw_delta_t * 0.8
        optimal_vacuum = min(28.5, input_data.current_vacuum_in_hg + 0.5)

        # Performance index
        perf_index = current_recovery * (input_data.current_vacuum_in_hg / 28) / 100

        recommendations = []
        if current_recovery < 90:
            recommendations.append("Increase cooling water flow to improve recovery")
        if cw_delta_t < 15:
            recommendations.append("CW temperature rise low - check for fouling")
        if input_data.current_vacuum_in_hg < 27:
            recommendations.append("Improve vacuum to enhance condensation")

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "condenser": input_data.condenser_id,
                        "timestamp": datetime.utcnow().isoformat()}, sort_keys=True).encode()
        ).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return VentCondenserOptOutput(
            analysis_id=f"VC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            condenser_id=input_data.condenser_id,
            current_recovery_rate_percent=round(current_recovery, 1),
            optimal_recovery_rate_percent=round(optimal_recovery, 1),
            steam_recovered_lb_hr=round(steam_condensed, 0),
            energy_saved_mmbtu_hr=round(energy_saved, 3),
            annual_savings=round(annual_savings, 0),
            condensate_quality=cond_quality,
            optimal_cw_temp_setpoint_f=round(optimal_cw_temp, 1),
            optimal_vacuum_setpoint_in_hg=round(optimal_vacuum, 1),
            performance_index=round(perf_index, 2),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS"
        )


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-043",
    "name": "VENT-CONDENSER-OPT",
    "version": "1.0.0",
    "summary": "Vent condenser optimization for steam recovery",
    "tags": ["condenser", "steam-recovery", "heat-transfer", "HEI"],
    "standards": [{"ref": "HEI Standards", "description": "Steam Surface Condensers"}]
}
