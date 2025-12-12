"""GL-088: Incentive Maximizer Agent (INCENTIVE).

Maximizes energy incentives and rebates.

Standards: DSIRE Database, Utility Programs
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IncentiveType(str, Enum):
    UTILITY_REBATE = "UTILITY_REBATE"
    TAX_CREDIT = "TAX_CREDIT"
    GRANT = "GRANT"
    LOW_INTEREST_LOAN = "LOW_INTEREST_LOAN"
    PERFORMANCE_INCENTIVE = "PERFORMANCE_INCENTIVE"


class EligibleIncentive(BaseModel):
    incentive_id: str
    name: str
    incentive_type: IncentiveType
    provider: str
    max_value_usd: float = Field(ge=0)
    rate_per_unit: float = Field(default=0, ge=0)
    unit: str = Field(default="kWh")
    application_deadline: Optional[datetime] = None
    requirements: List[str] = Field(default_factory=list)
    stackable: bool = Field(default=True)


class ProjectMetrics(BaseModel):
    annual_kwh_savings: float = Field(default=0, ge=0)
    annual_therm_savings: float = Field(default=0, ge=0)
    peak_kw_reduction: float = Field(default=0, ge=0)
    renewable_capacity_kw: float = Field(default=0, ge=0)
    project_cost_usd: float = Field(default=0, ge=0)


class IncentiveInput(BaseModel):
    project_id: str
    project_name: str = Field(default="Energy Project")
    location_state: str = Field(default="CA")
    utility: str = Field(default="PG&E")
    project_metrics: ProjectMetrics
    incentives: List[EligibleIncentive] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IncentiveCapture(BaseModel):
    incentive_name: str
    incentive_type: str
    provider: str
    estimated_value_usd: float
    application_status: str
    priority: str
    action_required: str


class IncentiveOutput(BaseModel):
    project_id: str
    total_incentive_value_usd: float
    incentive_as_pct_of_cost: float
    captures: List[IncentiveCapture]
    stackable_incentives: int
    non_stackable_incentives: int
    time_sensitive_count: int
    net_project_cost_usd: float
    effective_payback_reduction_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class IncentiveMaximizerAgent:
    AGENT_ID = "GL-088"
    AGENT_NAME = "INCENTIVE"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"IncentiveMaximizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = IncentiveInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_incentive_value(self, incentive: EligibleIncentive, metrics: ProjectMetrics) -> float:
        """Calculate incentive value based on project metrics."""
        if incentive.unit == "kWh":
            value = metrics.annual_kwh_savings * incentive.rate_per_unit
        elif incentive.unit == "therm":
            value = metrics.annual_therm_savings * incentive.rate_per_unit
        elif incentive.unit == "kW":
            value = metrics.peak_kw_reduction * incentive.rate_per_unit
        elif incentive.unit == "kW_capacity":
            value = metrics.renewable_capacity_kw * incentive.rate_per_unit
        elif incentive.unit == "percent":
            value = metrics.project_cost_usd * incentive.rate_per_unit / 100
        else:
            value = incentive.max_value_usd

        return min(value, incentive.max_value_usd)

    def _process(self, inp: IncentiveInput) -> IncentiveOutput:
        recommendations = []
        captures = []
        stackable = 0
        non_stackable = 0
        time_sensitive = 0
        total_value = 0

        now = datetime.utcnow()

        for incentive in inp.incentives:
            value = self._calculate_incentive_value(incentive, inp.project_metrics)

            # Check deadline
            if incentive.application_deadline:
                days_until = (incentive.application_deadline - now).days
                if days_until < 0:
                    status = "EXPIRED"
                    priority = "NONE"
                    action = "Incentive expired"
                elif days_until < 30:
                    status = "URGENT"
                    priority = "HIGH"
                    action = f"Apply immediately - {days_until} days remaining"
                    time_sensitive += 1
                elif days_until < 90:
                    status = "ACTIVE"
                    priority = "MEDIUM"
                    action = f"Apply soon - {days_until} days remaining"
                    time_sensitive += 1
                else:
                    status = "ACTIVE"
                    priority = "NORMAL"
                    action = "Submit application"
            else:
                status = "ACTIVE"
                priority = "NORMAL"
                action = "Submit application"

            if status != "EXPIRED":
                total_value += value
                if incentive.stackable:
                    stackable += 1
                else:
                    non_stackable += 1

            captures.append(IncentiveCapture(
                incentive_name=incentive.name,
                incentive_type=incentive.incentive_type.value,
                provider=incentive.provider,
                estimated_value_usd=round(value, 2),
                application_status=status,
                priority=priority,
                action_required=action
            ))

        # Sort by priority and value
        priority_order = {"HIGH": 0, "MEDIUM": 1, "NORMAL": 2, "NONE": 3}
        captures.sort(key=lambda x: (priority_order.get(x.priority, 3), -x.estimated_value_usd))

        # Calculations
        project_cost = inp.project_metrics.project_cost_usd
        incentive_pct = (total_value / project_cost * 100) if project_cost > 0 else 0
        net_cost = project_cost - total_value
        payback_reduction = incentive_pct

        # Recommendations
        if time_sensitive > 0:
            recommendations.append(f"{time_sensitive} time-sensitive incentives - prioritize applications")

        high_value = [c for c in captures if c.estimated_value_usd > project_cost * 0.1]
        if high_value:
            recommendations.append(f"{len(high_value)} high-value incentives (>10% of project cost)")

        tax_credits = [c for c in captures if c.incentive_type == "TAX_CREDIT"]
        if tax_credits:
            recommendations.append("Tax credits available - engage tax advisor")

        if incentive_pct < 10:
            recommendations.append("Low incentive capture (<10%) - research additional programs")
        elif incentive_pct > 30:
            recommendations.append(f"Strong incentive capture {incentive_pct:.0f}% - excellent project economics")

        if non_stackable > 1:
            recommendations.append("Multiple non-stackable incentives - choose highest value option")

        calc_hash = hashlib.sha256(json.dumps({
            "project": inp.project_id,
            "total_value": round(total_value, 2),
            "incentive_pct": round(incentive_pct, 1)
        }).encode()).hexdigest()

        return IncentiveOutput(
            project_id=inp.project_id,
            total_incentive_value_usd=round(total_value, 2),
            incentive_as_pct_of_cost=round(incentive_pct, 1),
            captures=captures,
            stackable_incentives=stackable,
            non_stackable_incentives=non_stackable,
            time_sensitive_count=time_sensitive,
            net_project_cost_usd=round(net_cost, 2),
            effective_payback_reduction_pct=round(payback_reduction, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-088", "name": "INCENTIVE", "version": "1.0.0",
    "summary": "Energy incentive and rebate maximization",
    "standards": [{"ref": "DSIRE Database"}, {"ref": "Utility Programs"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
