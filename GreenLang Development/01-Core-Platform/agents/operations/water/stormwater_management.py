# -*- coding: utf-8 -*-
"""
GL-OPS-WAT-008: Stormwater Management Agent
==========================================

Operations agent for stormwater system management.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class StormSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    EXTREME = "extreme"


class InfrastructureType(str, Enum):
    DETENTION_BASIN = "detention_basin"
    RETENTION_POND = "retention_pond"
    GREEN_INFRASTRUCTURE = "green_infrastructure"
    STORM_DRAIN = "storm_drain"
    PUMP_STATION = "pump_station"


class StormEvent(BaseModel):
    """Storm event definition."""
    event_id: str
    start_time: datetime
    duration_hours: float
    total_precipitation_mm: float
    peak_intensity_mm_hr: float
    return_period_years: Optional[float] = None
    severity: StormSeverity


class InfrastructureStatus(BaseModel):
    """Infrastructure asset status."""
    asset_id: str
    asset_type: InfrastructureType
    capacity_m3: float
    current_level_percent: float
    available_capacity_m3: float
    condition_score: float  # 1-10
    pump_status: Optional[str] = None


class FloodRiskZone(BaseModel):
    """Flood risk zone."""
    zone_id: str
    zone_name: str
    risk_level: str
    population_at_risk: int
    property_value_at_risk: float


class StormwaterInput(BaseModel):
    """Input for stormwater management."""
    catchment_id: str
    catchment_area_km2: float
    infrastructure: List[InfrastructureStatus]
    forecast_events: List[StormEvent] = Field(default_factory=list)
    current_conditions: Dict[str, Any] = Field(default_factory=dict)
    flood_risk_zones: List[FloodRiskZone] = Field(default_factory=list)
    runoff_coefficient: float = Field(default=0.5)


class StormwaterOutput(BaseModel):
    """Output from stormwater management."""
    catchment_id: str
    system_capacity_summary: Dict[str, float]
    event_response_plans: List[Dict[str, Any]]
    infrastructure_recommendations: List[Dict[str, Any]]
    flood_risk_assessment: Dict[str, Any]
    operational_recommendations: List[str]
    alerts: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class StormwaterManagementAgent(BaseAgent):
    """
    GL-OPS-WAT-008: Stormwater Management Agent

    Manages stormwater systems and flood preparedness.
    """

    AGENT_ID = "GL-OPS-WAT-008"
    AGENT_NAME = "Stormwater Management Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Stormwater system management",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            sw_input = StormwaterInput(**input_data)
            alerts = []
            recommendations = []

            # System capacity summary
            total_capacity = sum(i.capacity_m3 for i in sw_input.infrastructure)
            available_capacity = sum(i.available_capacity_m3 for i in sw_input.infrastructure)
            avg_condition = sum(i.condition_score for i in sw_input.infrastructure) / max(1, len(sw_input.infrastructure))

            capacity_summary = {
                "total_capacity_m3": round(total_capacity, 0),
                "available_capacity_m3": round(available_capacity, 0),
                "utilization_percent": round((total_capacity - available_capacity) / total_capacity * 100, 2) if total_capacity > 0 else 0,
                "average_condition_score": round(avg_condition, 1),
            }

            # Event response plans
            event_plans = []
            for event in sw_input.forecast_events:
                # Calculate runoff volume
                # Q = C * I * A (simplified rational method)
                runoff_m3 = (
                    sw_input.runoff_coefficient *
                    event.total_precipitation_mm / 1000 *
                    sw_input.catchment_area_km2 * 1e6
                )

                # Check capacity
                capacity_gap = runoff_m3 - available_capacity
                overflow_risk = capacity_gap > 0

                plan = {
                    "event_id": event.event_id,
                    "severity": event.severity.value,
                    "expected_runoff_m3": round(runoff_m3, 0),
                    "available_capacity_m3": round(available_capacity, 0),
                    "capacity_gap_m3": round(max(0, capacity_gap), 0),
                    "overflow_risk": overflow_risk,
                    "recommended_actions": [],
                }

                if overflow_risk:
                    plan["recommended_actions"].append("Pre-empty detention basins before event")
                    plan["recommended_actions"].append("Activate all available pump stations")
                    alerts.append(f"ALERT: Capacity gap of {capacity_gap:,.0f} m3 for event {event.event_id}")

                if event.severity in [StormSeverity.MAJOR, StormSeverity.EXTREME]:
                    plan["recommended_actions"].append("Issue flood warnings to affected areas")
                    plan["recommended_actions"].append("Deploy emergency response teams")

                event_plans.append(plan)

            # Infrastructure recommendations
            infra_recs = []
            for asset in sw_input.infrastructure:
                if asset.condition_score < 5:
                    infra_recs.append({
                        "asset_id": asset.asset_id,
                        "issue": "Poor condition",
                        "recommendation": "Schedule maintenance inspection",
                        "priority": "high" if asset.condition_score < 3 else "medium",
                    })
                if asset.current_level_percent > 80:
                    infra_recs.append({
                        "asset_id": asset.asset_id,
                        "issue": "High utilization",
                        "recommendation": "Reduce level before forecast events",
                        "priority": "medium",
                    })

            # Flood risk assessment
            total_population_at_risk = sum(z.population_at_risk for z in sw_input.flood_risk_zones)
            total_property_at_risk = sum(z.property_value_at_risk for z in sw_input.flood_risk_zones)

            flood_risk = {
                "zones_analyzed": len(sw_input.flood_risk_zones),
                "total_population_at_risk": total_population_at_risk,
                "total_property_value_at_risk": round(total_property_at_risk, 0),
                "high_risk_zones": [z.zone_id for z in sw_input.flood_risk_zones if z.risk_level == "high"],
            }

            # General recommendations
            if avg_condition < 6:
                recommendations.append("Implement comprehensive infrastructure rehabilitation program")
            if len(alerts) > 0:
                recommendations.append("Review emergency response procedures")
            recommendations.append("Continue monitoring weather forecasts for updates")
            recommendations.append("Ensure all pump stations are operational")

            provenance_hash = hashlib.sha256(
                json.dumps({"catchment": sw_input.catchment_id, "events": len(sw_input.forecast_events)}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = StormwaterOutput(
                catchment_id=sw_input.catchment_id,
                system_capacity_summary=capacity_summary,
                event_response_plans=event_plans,
                infrastructure_recommendations=infra_recs,
                flood_risk_assessment=flood_risk,
                operational_recommendations=recommendations,
                alerts=alerts,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Stormwater management failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
