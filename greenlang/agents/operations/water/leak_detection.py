# -*- coding: utf-8 -*-
"""
GL-OPS-WAT-003: Leak Detection Agent
====================================

Operations agent for water loss and leak detection using
flow balance analysis and statistical methods.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class LeakSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DMASummary(BaseModel):
    """District Metered Area summary."""
    dma_id: str = Field(..., description="DMA identifier")
    dma_name: Optional[str] = None
    inflow_m3: float = Field(..., ge=0)
    billed_consumption_m3: float = Field(..., ge=0)
    authorized_unbilled_m3: float = Field(default=0, ge=0)
    apparent_losses_m3: float = Field(default=0, ge=0)
    real_losses_m3: float = Field(default=0, ge=0)
    nrw_percent: float = Field(default=0)
    ili: Optional[float] = Field(None, description="Infrastructure Leakage Index")


class LeakCandidate(BaseModel):
    """Potential leak location."""
    candidate_id: str
    dma_id: str
    location_description: str
    estimated_flow_m3_hr: float
    confidence_percent: float
    severity: LeakSeverity
    detection_method: str
    coordinates: Optional[Dict[str, float]] = None
    recommended_action: str


class FlowReading(BaseModel):
    """Flow meter reading."""
    meter_id: str
    dma_id: str
    timestamp: datetime
    flow_m3_hr: float
    reading_type: str = "automatic"


class LeakDetectionInput(BaseModel):
    """Input for leak detection."""
    dma_summaries: List[DMASummary]
    flow_readings: List[FlowReading] = Field(default_factory=list)
    analysis_period_start: datetime
    analysis_period_end: datetime
    minimum_night_flow_threshold_m3_hr: float = Field(default=0.5)
    nrw_threshold_percent: float = Field(default=25)


class LeakDetectionOutput(BaseModel):
    """Output from leak detection."""
    analysis_period: str
    total_dmass_analyzed: int
    total_inflow_m3: float
    total_nrw_m3: float
    overall_nrw_percent: float
    leak_candidates: List[LeakCandidate]
    dma_rankings: List[Dict[str, Any]]
    estimated_annual_loss_m3: float
    estimated_annual_loss_value: float
    recommendations: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


class LeakDetectionAgent(BaseAgent):
    """
    GL-OPS-WAT-003: Leak Detection Agent

    Detects and prioritizes water leaks using flow balance and MNF analysis.
    """

    AGENT_ID = "GL-OPS-WAT-003"
    AGENT_NAME = "Leak Detection Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Water loss and leak detection",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            ld_input = LeakDetectionInput(**input_data)

            leak_candidates = []
            dma_rankings = []
            total_inflow = 0.0
            total_nrw = 0.0

            for dma in ld_input.dma_summaries:
                total_inflow += dma.inflow_m3
                nrw = dma.real_losses_m3 + dma.apparent_losses_m3
                total_nrw += nrw

                # Calculate NRW percentage
                nrw_pct = (nrw / dma.inflow_m3 * 100) if dma.inflow_m3 > 0 else 0

                # Determine severity
                if nrw_pct > 50:
                    severity = LeakSeverity.CRITICAL
                elif nrw_pct > 35:
                    severity = LeakSeverity.HIGH
                elif nrw_pct > 25:
                    severity = LeakSeverity.MEDIUM
                else:
                    severity = LeakSeverity.LOW

                # Add leak candidate if above threshold
                if nrw_pct > ld_input.nrw_threshold_percent:
                    candidate = LeakCandidate(
                        candidate_id=f"LC-{dma.dma_id}",
                        dma_id=dma.dma_id,
                        location_description=f"DMA {dma.dma_id} - High NRW Zone",
                        estimated_flow_m3_hr=dma.real_losses_m3 / 720,  # Monthly to hourly
                        confidence_percent=min(95, 50 + nrw_pct),
                        severity=severity,
                        detection_method="flow_balance",
                        recommended_action="Deploy acoustic leak detection equipment",
                    )
                    leak_candidates.append(candidate)

                dma_rankings.append({
                    "dma_id": dma.dma_id,
                    "nrw_percent": round(nrw_pct, 2),
                    "real_losses_m3": round(dma.real_losses_m3, 2),
                    "severity": severity.value,
                    "priority_score": round(nrw_pct * dma.real_losses_m3 / 1000, 2),
                })

            # Sort by priority
            dma_rankings.sort(key=lambda x: x["priority_score"], reverse=True)

            overall_nrw_pct = (total_nrw / total_inflow * 100) if total_inflow > 0 else 0

            # Estimate annual losses
            period_days = (ld_input.analysis_period_end - ld_input.analysis_period_start).days
            annual_multiplier = 365 / max(1, period_days)
            annual_loss_m3 = total_nrw * annual_multiplier
            annual_loss_value = annual_loss_m3 * 1.50  # $1.50/m3 average

            # Generate recommendations
            recommendations = []
            if overall_nrw_pct > 30:
                recommendations.append("Implement comprehensive pressure management program")
            if len(leak_candidates) > 3:
                recommendations.append("Prioritize leak repair in top 3 DMAs by priority score")
            recommendations.append("Establish continuous monitoring using smart meters")

            provenance_hash = hashlib.sha256(
                json.dumps({"dmass": len(ld_input.dma_summaries), "nrw": total_nrw}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = LeakDetectionOutput(
                analysis_period=f"{ld_input.analysis_period_start.date()} to {ld_input.analysis_period_end.date()}",
                total_dmass_analyzed=len(ld_input.dma_summaries),
                total_inflow_m3=round(total_inflow, 2),
                total_nrw_m3=round(total_nrw, 2),
                overall_nrw_percent=round(overall_nrw_pct, 2),
                leak_candidates=leak_candidates,
                dma_rankings=dma_rankings[:10],
                estimated_annual_loss_m3=round(annual_loss_m3, 0),
                estimated_annual_loss_value=round(annual_loss_value, 0),
                recommendations=recommendations,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Leak detection failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
