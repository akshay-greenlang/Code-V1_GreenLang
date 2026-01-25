"""GL-094: Stakeholder Reporter Agent (STAKEHOLDER-REPORTER).

Generates stakeholder reports for energy programs.

Standards: GRI, SASB, Integrated Reporting
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StakeholderType(str, Enum):
    EXECUTIVE = "EXECUTIVE"
    BOARD = "BOARD"
    INVESTOR = "INVESTOR"
    REGULATOR = "REGULATOR"
    EMPLOYEE = "EMPLOYEE"
    CUSTOMER = "CUSTOMER"


class ReportMetric(BaseModel):
    metric_id: str
    name: str
    value: float
    unit: str
    target: Optional[float] = None
    trend: str = Field(default="STABLE")


class StakeholderReporterInput(BaseModel):
    report_id: str
    report_period: str = Field(default="Q4 2024")
    stakeholder_type: StakeholderType = Field(default=StakeholderType.EXECUTIVE)
    metrics: List[ReportMetric] = Field(default_factory=list)
    narrative_highlights: List[str] = Field(default_factory=list)
    include_financials: bool = Field(default=True)
    include_sustainability: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    name: str
    value: str
    status: str
    trend_indicator: str


class StakeholderReporterOutput(BaseModel):
    report_id: str
    report_period: str
    stakeholder_type: str
    executive_summary: str
    metric_summaries: List[MetricSummary]
    key_achievements: List[str]
    areas_of_concern: List[str]
    action_items: List[str]
    overall_status: str
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class StakeholderReporterAgent:
    AGENT_ID = "GL-094"
    AGENT_NAME = "STAKEHOLDER-REPORTER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"StakeholderReporterAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = StakeholderReporterInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: StakeholderReporterInput) -> StakeholderReporterOutput:
        achievements = []
        concerns = []
        actions = []
        summaries = []

        on_target = 0
        off_target = 0

        for metric in inp.metrics:
            # Determine status
            if metric.target:
                variance = (metric.value - metric.target) / metric.target * 100 if metric.target != 0 else 0
                if abs(variance) < 5:
                    status = "ON_TARGET"
                    on_target += 1
                elif variance > 0:
                    status = "ABOVE_TARGET"
                    on_target += 1
                    achievements.append(f"{metric.name} exceeded target by {variance:.1f}%")
                else:
                    status = "BELOW_TARGET"
                    off_target += 1
                    concerns.append(f"{metric.name} {abs(variance):.1f}% below target")
                    actions.append(f"Develop action plan to improve {metric.name}")
            else:
                status = "NO_TARGET"

            # Trend indicator
            trend_map = {"IMPROVING": "↑", "DECLINING": "↓", "STABLE": "→"}
            trend_indicator = trend_map.get(metric.trend, "→")

            summaries.append(MetricSummary(
                name=metric.name,
                value=f"{metric.value:,.2f} {metric.unit}",
                status=status,
                trend_indicator=trend_indicator
            ))

        # Overall status
        if len(inp.metrics) > 0:
            target_pct = on_target / len(inp.metrics) * 100
            if target_pct >= 80:
                overall = "GREEN"
            elif target_pct >= 60:
                overall = "YELLOW"
            else:
                overall = "RED"
        else:
            overall = "NO_DATA"

        # Executive summary
        if inp.stakeholder_type == StakeholderType.EXECUTIVE:
            summary = f"Energy program {overall} status for {inp.report_period}. {on_target} of {len(inp.metrics)} metrics on target."
        elif inp.stakeholder_type == StakeholderType.BOARD:
            summary = f"Board summary: Program performing at {target_pct:.0f}% target achievement."
        elif inp.stakeholder_type == StakeholderType.INVESTOR:
            summary = f"Investment performance update: {overall} status with {len(achievements)} positive developments."
        else:
            summary = f"Stakeholder report for {inp.report_period}: Overall status {overall}."

        # Add narrative highlights to achievements
        achievements.extend(inp.narrative_highlights)

        calc_hash = hashlib.sha256(json.dumps({
            "report": inp.report_id,
            "period": inp.report_period,
            "overall": overall
        }).encode()).hexdigest()

        return StakeholderReporterOutput(
            report_id=inp.report_id,
            report_period=inp.report_period,
            stakeholder_type=inp.stakeholder_type.value,
            executive_summary=summary,
            metric_summaries=summaries,
            key_achievements=achievements[:5],
            areas_of_concern=concerns[:5],
            action_items=actions[:5],
            overall_status=overall,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-094", "name": "STAKEHOLDER-REPORTER", "version": "1.0.0",
    "summary": "Stakeholder reporting for energy programs",
    "standards": [{"ref": "GRI"}, {"ref": "SASB"}, {"ref": "Integrated Reporting"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
