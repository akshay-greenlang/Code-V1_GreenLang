# -*- coding: utf-8 -*-
"""
GL-OPS-WAT-007: Reservoir Management Agent
==========================================

Operations agent for reservoir optimization.

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


class ReservoirType(str, Enum):
    STORAGE = "storage"
    SERVICE = "service"
    ELEVATED = "elevated"
    GROUND_LEVEL = "ground_level"


class ReservoirState(BaseModel):
    """Current reservoir state."""
    reservoir_id: str
    reservoir_name: Optional[str] = None
    reservoir_type: ReservoirType
    capacity_m3: float
    current_level_m3: float
    current_level_percent: float
    inflow_m3_hr: float
    outflow_m3_hr: float
    min_operating_level_percent: float = Field(default=20)
    max_operating_level_percent: float = Field(default=95)


class ReleaseScheduleEntry(BaseModel):
    """Release schedule entry."""
    timestamp: datetime
    release_m3_hr: float
    target_level_percent: float
    energy_optimized: bool = False


class ReleaseSchedule(BaseModel):
    """Complete release schedule."""
    reservoir_id: str
    schedule_start: datetime
    schedule_end: datetime
    entries: List[ReleaseScheduleEntry]
    optimization_goal: str
    expected_final_level_percent: float


class ReservoirManagementInput(BaseModel):
    """Input for reservoir management."""
    reservoirs: List[ReservoirState]
    demand_forecast_m3_hr: List[float] = Field(default_factory=list)
    inflow_forecast_m3_hr: List[float] = Field(default_factory=list)
    optimization_horizon_hours: int = Field(default=24)
    electricity_tariffs: List[Dict[str, Any]] = Field(default_factory=list)
    emergency_reserve_percent: float = Field(default=25)


class ReservoirManagementOutput(BaseModel):
    """Output from reservoir management."""
    schedules: List[ReleaseSchedule]
    reservoir_summaries: Dict[str, Dict[str, Any]]
    total_storage_m3: float
    total_available_m3: float
    system_reserve_days: float
    alerts: List[str]
    recommendations: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class ReservoirManagementAgent(BaseAgent):
    """
    GL-OPS-WAT-007: Reservoir Management Agent

    Optimizes reservoir operations and release schedules.
    """

    AGENT_ID = "GL-OPS-WAT-007"
    AGENT_NAME = "Reservoir Management Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Reservoir optimization and management",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            rm_input = ReservoirManagementInput(**input_data)
            schedules = []
            summaries = {}
            alerts = []

            total_storage = sum(r.capacity_m3 for r in rm_input.reservoirs)
            total_available = sum(r.current_level_m3 for r in rm_input.reservoirs)

            base_time = DeterministicClock.now()

            for reservoir in rm_input.reservoirs:
                # Generate release schedule
                entries = []
                current_level = reservoir.current_level_m3

                for hour in range(rm_input.optimization_horizon_hours):
                    timestamp = base_time + timedelta(hours=hour)

                    # Get demand and inflow for this hour
                    demand = rm_input.demand_forecast_m3_hr[hour] if hour < len(rm_input.demand_forecast_m3_hr) else reservoir.outflow_m3_hr
                    inflow = rm_input.inflow_forecast_m3_hr[hour] if hour < len(rm_input.inflow_forecast_m3_hr) else reservoir.inflow_m3_hr

                    # Calculate release (simple balance)
                    release = demand / len(rm_input.reservoirs)

                    # Adjust for level constraints
                    projected_level = current_level + inflow - release
                    level_percent = projected_level / reservoir.capacity_m3 * 100

                    if level_percent < reservoir.min_operating_level_percent:
                        release = max(0, current_level + inflow - reservoir.capacity_m3 * reservoir.min_operating_level_percent / 100)
                    elif level_percent > reservoir.max_operating_level_percent:
                        release = current_level + inflow - reservoir.capacity_m3 * reservoir.max_operating_level_percent / 100

                    current_level = current_level + inflow - release
                    level_percent = current_level / reservoir.capacity_m3 * 100

                    entry = ReleaseScheduleEntry(
                        timestamp=timestamp,
                        release_m3_hr=round(release, 2),
                        target_level_percent=round(level_percent, 2),
                        energy_optimized=False,
                    )
                    entries.append(entry)

                final_level = current_level / reservoir.capacity_m3 * 100

                schedule = ReleaseSchedule(
                    reservoir_id=reservoir.reservoir_id,
                    schedule_start=base_time,
                    schedule_end=base_time + timedelta(hours=rm_input.optimization_horizon_hours),
                    entries=entries,
                    optimization_goal="demand_balance",
                    expected_final_level_percent=round(final_level, 2),
                )
                schedules.append(schedule)

                # Summary
                summaries[reservoir.reservoir_id] = {
                    "current_level_percent": reservoir.current_level_percent,
                    "expected_final_level_percent": round(final_level, 2),
                    "capacity_m3": reservoir.capacity_m3,
                    "status": "normal" if 30 < final_level < 90 else "attention",
                }

                # Alerts
                if final_level < rm_input.emergency_reserve_percent:
                    alerts.append(f"ALERT: {reservoir.reservoir_id} projected to fall below emergency reserve")
                if final_level > 95:
                    alerts.append(f"WARNING: {reservoir.reservoir_id} projected near overflow")

            # System reserve
            avg_demand = sum(rm_input.demand_forecast_m3_hr) / max(1, len(rm_input.demand_forecast_m3_hr)) * 24
            reserve_days = total_available / avg_demand if avg_demand > 0 else 999

            # Recommendations
            recommendations = []
            if reserve_days < 3:
                recommendations.append("Increase water production to rebuild reserves")
            if alerts:
                recommendations.append("Monitor reservoir levels closely over next 24 hours")
            recommendations.append("Review release schedules daily based on updated forecasts")

            provenance_hash = hashlib.sha256(
                json.dumps({"reservoirs": len(rm_input.reservoirs), "horizon": rm_input.optimization_horizon_hours}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = ReservoirManagementOutput(
                schedules=schedules,
                reservoir_summaries=summaries,
                total_storage_m3=round(total_storage, 0),
                total_available_m3=round(total_available, 0),
                system_reserve_days=round(reserve_days, 1),
                alerts=alerts,
                recommendations=recommendations,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Reservoir management failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
