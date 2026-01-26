# -*- coding: utf-8 -*-
"""
GL-DECARB-X-018: Progress Monitoring Agent
===========================================

Monitors decarbonization progress against targets and milestones.

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


class ProgressStatus(str, Enum):
    ON_TRACK = "on_track"
    AHEAD = "ahead"
    BEHIND = "behind"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"


class ProgressDataPoint(BaseModel):
    period: str = Field(...)
    year: int = Field(...)
    actual_emissions_tco2e: float = Field(..., ge=0)
    target_emissions_tco2e: float = Field(..., ge=0)
    variance_tco2e: float = Field(...)
    variance_percent: float = Field(...)
    status: ProgressStatus = Field(...)


class ProgressReport(BaseModel):
    report_id: str = Field(...)
    report_date: datetime = Field(default_factory=DeterministicClock.now)

    # Target info
    base_year: int = Field(...)
    base_year_emissions_tco2e: float = Field(..., ge=0)
    target_year: int = Field(...)
    target_emissions_tco2e: float = Field(..., ge=0)
    target_reduction_percent: float = Field(...)

    # Current state
    current_year: int = Field(...)
    current_emissions_tco2e: float = Field(..., ge=0)
    current_reduction_percent: float = Field(...)
    required_reduction_percent: float = Field(...)  # Based on linear trajectory

    # Status
    overall_status: ProgressStatus = Field(...)
    variance_from_trajectory_tco2e: float = Field(...)
    variance_from_trajectory_percent: float = Field(...)

    # Time series
    progress_data: List[ProgressDataPoint] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    provenance_hash: str = Field(default="")


class ProgressMonitoringInput(BaseModel):
    operation: str = Field(default="monitor")
    base_year: int = Field(default=2020)
    base_year_emissions_tco2e: float = Field(default=100000, ge=0)
    target_year: int = Field(default=2030)
    target_reduction_percent: float = Field(default=50, ge=0, le=100)
    emission_history: List[Dict[str, Any]] = Field(default_factory=list)


class ProgressMonitoringOutput(BaseModel):
    operation: str = Field(...)
    success: bool = Field(...)
    report: Optional[ProgressReport] = Field(None)
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


class ProgressMonitoringAgent(DeterministicAgent):
    """GL-DECARB-X-018: Progress Monitoring Agent"""

    AGENT_ID = "GL-DECARB-X-018"
    AGENT_NAME = "Progress Monitoring Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="ProgressMonitoringAgent",
        category=AgentCategory.CRITICAL,
        description="Monitors decarbonization progress"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        super().__init__(enable_audit_trail=enable_audit_trail)
        self.config = config or AgentConfig(
            name=self.AGENT_NAME, description="Monitors progress", version=self.VERSION
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        calculation_trace = []

        try:
            pm_input = ProgressMonitoringInput(**inputs)
            calculation_trace.append(f"Operation: {pm_input.operation}")

            if pm_input.operation == "monitor":
                # Calculate target trajectory
                target_emissions = pm_input.base_year_emissions_tco2e * (1 - pm_input.target_reduction_percent / 100)
                annual_reduction = (pm_input.base_year_emissions_tco2e - target_emissions) / (pm_input.target_year - pm_input.base_year)

                # Process emission history
                progress_data = []
                current_year = pm_input.base_year
                current_emissions = pm_input.base_year_emissions_tco2e

                for record in pm_input.emission_history:
                    year = record.get("year", pm_input.base_year)
                    actual = record.get("emissions_tco2e", pm_input.base_year_emissions_tco2e)

                    # Calculate expected (linear trajectory)
                    years_elapsed = year - pm_input.base_year
                    expected = pm_input.base_year_emissions_tco2e - (annual_reduction * years_elapsed)

                    variance = expected - actual
                    variance_pct = (variance / expected * 100) if expected > 0 else 0

                    # Determine status
                    if variance_pct > 5:
                        status = ProgressStatus.AHEAD
                    elif variance_pct > 0:
                        status = ProgressStatus.ON_TRACK
                    elif variance_pct > -5:
                        status = ProgressStatus.BEHIND
                    elif variance_pct > -15:
                        status = ProgressStatus.AT_RISK
                    else:
                        status = ProgressStatus.OFF_TRACK

                    progress_data.append(ProgressDataPoint(
                        period=str(year),
                        year=year,
                        actual_emissions_tco2e=actual,
                        target_emissions_tco2e=expected,
                        variance_tco2e=variance,
                        variance_percent=variance_pct,
                        status=status
                    ))

                    current_year = year
                    current_emissions = actual

                # Overall status
                if progress_data:
                    overall_status = progress_data[-1].status
                    variance = progress_data[-1].variance_tco2e
                    variance_pct = progress_data[-1].variance_percent
                else:
                    overall_status = ProgressStatus.ON_TRACK
                    variance = 0
                    variance_pct = 0

                current_reduction = ((pm_input.base_year_emissions_tco2e - current_emissions) / pm_input.base_year_emissions_tco2e * 100) if pm_input.base_year_emissions_tco2e > 0 else 0
                required_reduction = ((current_year - pm_input.base_year) / (pm_input.target_year - pm_input.base_year)) * pm_input.target_reduction_percent

                # Generate recommendations
                recommendations = []
                if overall_status in [ProgressStatus.BEHIND, ProgressStatus.AT_RISK]:
                    recommendations.append("Accelerate implementation of planned abatement measures")
                    recommendations.append("Review project portfolio for quick-win opportunities")
                if overall_status == ProgressStatus.OFF_TRACK:
                    recommendations.append("Consider additional abatement options")
                    recommendations.append("Evaluate offset strategy to bridge gap")
                    recommendations.append("Conduct root cause analysis of delays")

                report = ProgressReport(
                    report_id=deterministic_id({"year": current_year}, "progress_"),
                    base_year=pm_input.base_year,
                    base_year_emissions_tco2e=pm_input.base_year_emissions_tco2e,
                    target_year=pm_input.target_year,
                    target_emissions_tco2e=target_emissions,
                    target_reduction_percent=pm_input.target_reduction_percent,
                    current_year=current_year,
                    current_emissions_tco2e=current_emissions,
                    current_reduction_percent=current_reduction,
                    required_reduction_percent=required_reduction,
                    overall_status=overall_status,
                    variance_from_trajectory_tco2e=variance,
                    variance_from_trajectory_percent=variance_pct,
                    progress_data=progress_data,
                    recommendations=recommendations
                )
                report.provenance_hash = content_hash(report.model_dump(exclude={"provenance_hash"}))

                calculation_trace.append(f"Generated progress report: {overall_status.value}")

                self._capture_audit_entry(
                    operation="monitor",
                    inputs=inputs,
                    outputs={"status": overall_status.value},
                    calculation_trace=calculation_trace
                )

                return {
                    "operation": "monitor",
                    "success": True,
                    "report": report.model_dump(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "timestamp": DeterministicClock.now().isoformat()
                }
            else:
                raise ValueError(f"Unknown operation: {pm_input.operation}")

        except Exception as e:
            self.logger.error(f"Monitoring failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }
