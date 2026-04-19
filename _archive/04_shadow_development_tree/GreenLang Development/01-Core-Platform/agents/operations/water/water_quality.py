# -*- coding: utf-8 -*-
"""
GL-OPS-WAT-004: Water Quality Monitor Agent
==========================================

Operations agent for water quality monitoring and compliance.

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


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WaterQualityParameter(BaseModel):
    """Water quality parameter definition."""
    parameter_name: str
    unit: str
    min_limit: Optional[float] = None
    max_limit: Optional[float] = None
    target_value: Optional[float] = None


class WaterQualitySample(BaseModel):
    """Water quality sample record."""
    sample_id: str
    location_id: str
    sample_time: datetime
    parameters: Dict[str, float]
    sample_type: str = "routine"
    sampler_id: Optional[str] = None


class QualityAlert(BaseModel):
    """Water quality alert."""
    alert_id: str
    location_id: str
    parameter_name: str
    measured_value: float
    limit_value: float
    violation_type: str
    priority: AlertPriority
    alert_time: datetime
    recommended_action: str


class WaterQualityInput(BaseModel):
    """Input for water quality monitoring."""
    samples: List[WaterQualitySample]
    parameters: List[WaterQualityParameter] = Field(default_factory=list)
    regulatory_limits: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    analysis_period_start: datetime
    analysis_period_end: datetime


class WaterQualityOutput(BaseModel):
    """Output from water quality monitoring."""
    analysis_period: str
    samples_analyzed: int
    compliance_summary: Dict[str, Any]
    parameter_statistics: Dict[str, Dict[str, float]]
    alerts: List[QualityAlert]
    locations_with_issues: List[str]
    overall_compliance_percent: float
    recommendations: List[str]
    provenance_hash: str
    calculation_timestamp: datetime
    processing_time_ms: float


# Default regulatory limits (EPA/WHO standards)
DEFAULT_LIMITS = {
    "turbidity_ntu": {"max": 4.0, "target": 1.0},
    "ph": {"min": 6.5, "max": 8.5, "target": 7.5},
    "chlorine_residual_mg_l": {"min": 0.2, "max": 4.0, "target": 0.5},
    "total_coliform_cfu_100ml": {"max": 0},
    "e_coli_cfu_100ml": {"max": 0},
    "lead_mg_l": {"max": 0.015},
    "arsenic_mg_l": {"max": 0.010},
    "nitrate_mg_l": {"max": 10.0},
}


class WaterQualityMonitorAgent(BaseAgent):
    """
    GL-OPS-WAT-004: Water Quality Monitor Agent

    Monitors water quality parameters and compliance.
    """

    AGENT_ID = "GL-OPS-WAT-004"
    AGENT_NAME = "Water Quality Monitor Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Water quality monitoring and compliance",
                version=self.VERSION,
            )
        super().__init__(config)
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            wq_input = WaterQualityInput(**input_data)
            limits = wq_input.regulatory_limits or DEFAULT_LIMITS

            alerts = []
            parameter_stats: Dict[str, Dict[str, List[float]]] = {}
            violations_count = 0
            total_checks = 0
            locations_with_issues = set()

            for sample in wq_input.samples:
                for param_name, value in sample.parameters.items():
                    # Initialize stats tracking
                    if param_name not in parameter_stats:
                        parameter_stats[param_name] = {"values": []}
                    parameter_stats[param_name]["values"].append(value)

                    # Check against limits
                    if param_name in limits:
                        param_limits = limits[param_name]
                        total_checks += 1

                        # Check max limit
                        if "max" in param_limits and value > param_limits["max"]:
                            violations_count += 1
                            locations_with_issues.add(sample.location_id)
                            alert = QualityAlert(
                                alert_id=f"ALT-{sample.sample_id}-{param_name}",
                                location_id=sample.location_id,
                                parameter_name=param_name,
                                measured_value=value,
                                limit_value=param_limits["max"],
                                violation_type="max_exceeded",
                                priority=AlertPriority.HIGH if param_name in ["e_coli_cfu_100ml", "total_coliform_cfu_100ml"] else AlertPriority.MEDIUM,
                                alert_time=sample.sample_time,
                                recommended_action=f"Investigate elevated {param_name} at location {sample.location_id}",
                            )
                            alerts.append(alert)

                        # Check min limit
                        if "min" in param_limits and value < param_limits["min"]:
                            violations_count += 1
                            locations_with_issues.add(sample.location_id)
                            alert = QualityAlert(
                                alert_id=f"ALT-{sample.sample_id}-{param_name}",
                                location_id=sample.location_id,
                                parameter_name=param_name,
                                measured_value=value,
                                limit_value=param_limits["min"],
                                violation_type="min_not_met",
                                priority=AlertPriority.MEDIUM,
                                alert_time=sample.sample_time,
                                recommended_action=f"Address low {param_name} at location {sample.location_id}",
                            )
                            alerts.append(alert)

            # Calculate statistics
            param_statistics = {}
            for param_name, data in parameter_stats.items():
                values = data["values"]
                if values:
                    param_statistics[param_name] = {
                        "count": len(values),
                        "min": round(min(values), 4),
                        "max": round(max(values), 4),
                        "mean": round(sum(values) / len(values), 4),
                    }

            # Compliance
            compliance_pct = ((total_checks - violations_count) / total_checks * 100) if total_checks > 0 else 100

            # Generate recommendations
            recommendations = []
            if len(alerts) > 0:
                recommendations.append(f"Address {len(alerts)} quality alerts immediately")
            if compliance_pct < 95:
                recommendations.append("Review treatment processes to improve compliance rate")
            recommendations.append("Continue routine monitoring program")

            provenance_hash = hashlib.sha256(
                json.dumps({"samples": len(wq_input.samples), "alerts": len(alerts)}, sort_keys=True).encode()
            ).hexdigest()[:16]

            processing_time = (time.time() - start_time) * 1000

            output = WaterQualityOutput(
                analysis_period=f"{wq_input.analysis_period_start.date()} to {wq_input.analysis_period_end.date()}",
                samples_analyzed=len(wq_input.samples),
                compliance_summary={
                    "total_checks": total_checks,
                    "violations": violations_count,
                    "compliance_rate": round(compliance_pct, 2),
                },
                parameter_statistics=param_statistics,
                alerts=alerts,
                locations_with_issues=list(locations_with_issues),
                overall_compliance_percent=round(compliance_pct, 2),
                recommendations=recommendations,
                provenance_hash=provenance_hash,
                calculation_timestamp=DeterministicClock.now(),
                processing_time_ms=processing_time,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Water quality monitoring failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))
