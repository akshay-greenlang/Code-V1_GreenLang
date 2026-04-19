# -*- coding: utf-8 -*-
"""
GL-DECARB-X-009: Carbon Intensity Tracker Agent
=================================================

Tracks and analyzes carbon intensity metrics over time, supporting
both absolute emissions and intensity-based target monitoring.

Author: GreenLang Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig
from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, content_hash, deterministic_id

logger = logging.getLogger(__name__)


class IntensityMetricType(str, Enum):
    PER_REVENUE = "per_revenue"  # tCO2e/MUSD
    PER_PRODUCTION = "per_production"  # tCO2e/unit
    PER_FLOOR_AREA = "per_floor_area"  # tCO2e/m2
    PER_EMPLOYEE = "per_employee"  # tCO2e/FTE
    PER_ENERGY = "per_energy"  # tCO2e/MWh


class IntensityDataPoint(BaseModel):
    period: str = Field(..., description="Period (e.g., 2024-Q1)")
    year: int = Field(...)
    quarter: Optional[int] = Field(None, ge=1, le=4)
    month: Optional[int] = Field(None, ge=1, le=12)

    # Emissions
    absolute_emissions_tco2e: float = Field(..., ge=0)
    scope_1_tco2e: float = Field(default=0, ge=0)
    scope_2_tco2e: float = Field(default=0, ge=0)
    scope_3_tco2e: float = Field(default=0, ge=0)

    # Intensity metrics
    intensity_value: float = Field(...)
    intensity_metric: IntensityMetricType = Field(...)
    intensity_unit: str = Field(...)

    # Activity data
    activity_value: float = Field(..., ge=0)
    activity_unit: str = Field(...)


class IntensityTrend(BaseModel):
    metric_type: IntensityMetricType = Field(...)
    base_period: str = Field(...)
    base_value: float = Field(...)
    current_period: str = Field(...)
    current_value: float = Field(...)
    change_absolute: float = Field(...)
    change_percent: float = Field(...)
    trend_direction: str = Field(...)  # improving, worsening, stable
    periods_analyzed: int = Field(...)


class CarbonIntensityInput(BaseModel):
    operation: str = Field(default="track")
    metric_type: IntensityMetricType = Field(default=IntensityMetricType.PER_REVENUE)
    data_points: List[Dict[str, Any]] = Field(default_factory=list)
    target_intensity: Optional[float] = Field(None, ge=0)
    base_year: Optional[int] = Field(None)


class CarbonIntensityOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    data_points: List[IntensityDataPoint] = Field(default_factory=list)
    trend: Optional[IntensityTrend] = Field(None)
    on_track_to_target: Optional[bool] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class CarbonIntensityTracker(DeterministicAgent):
    """
    GL-DECARB-X-009: Carbon Intensity Tracker Agent

    Tracks carbon intensity metrics and trends.
    """

    AGENT_ID = "GL-DECARB-X-009"
    AGENT_NAME = "Carbon Intensity Tracker"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="CarbonIntensityTracker",
        category=AgentCategory.CRITICAL,
        description="Tracks carbon intensity metrics over time"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Tracks carbon intensity", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            track_input = CarbonIntensityInput(**inputs)
            calculation_trace.append(f"Operation: {track_input.operation}")

            if track_input.operation == "track":
                # Parse data points
                parsed_points = [IntensityDataPoint(**dp) for dp in track_input.data_points]

                if len(parsed_points) >= 2:
                    # Calculate trend
                    sorted_points = sorted(parsed_points, key=lambda p: (p.year, p.quarter or 0, p.month or 0))
                    base = sorted_points[0]
                    current = sorted_points[-1]

                    change_abs = current.intensity_value - base.intensity_value
                    change_pct = (change_abs / base.intensity_value * 100) if base.intensity_value != 0 else 0

                    if change_pct < -2:
                        direction = "improving"
                    elif change_pct > 2:
                        direction = "worsening"
                    else:
                        direction = "stable"

                    trend = IntensityTrend(
                        metric_type=track_input.metric_type,
                        base_period=base.period,
                        base_value=base.intensity_value,
                        current_period=current.period,
                        current_value=current.intensity_value,
                        change_absolute=change_abs,
                        change_percent=change_pct,
                        trend_direction=direction,
                        periods_analyzed=len(sorted_points)
                    )

                    # Check against target
                    on_track = None
                    if track_input.target_intensity:
                        on_track = current.intensity_value <= track_input.target_intensity

                    calculation_trace.append(f"Trend: {direction} ({change_pct:.1f}%)")
                else:
                    trend = None
                    on_track = None

                self._capture_audit_entry(
                    operation="track",
                    inputs={"points_count": len(parsed_points)},
                    outputs={"trend": trend.trend_direction if trend else None},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "track",
                    "success": True,
                    "data_points": [dp.model_dump() for dp in parsed_points],
                    "trend": trend.model_dump() if trend else None,
                    "on_track_to_target": on_track,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {track_input.operation}")

        except Exception as e:
            self.logger.error(f"Tracking failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
