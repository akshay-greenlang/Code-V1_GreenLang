"""GL-080: Grid Services Agent (GRID-SERVICES).

Optimizes participation in grid services and demand response.

Standards: IEEE 2030, FERC Order 2222
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GridService(str, Enum):
    DEMAND_RESPONSE = "DEMAND_RESPONSE"
    FREQUENCY_REGULATION = "FREQUENCY_REGULATION"
    SPINNING_RESERVE = "SPINNING_RESERVE"
    CAPACITY = "CAPACITY"
    ENERGY_ARBITRAGE = "ENERGY_ARBITRAGE"


class AssetType(str, Enum):
    HVAC = "HVAC"
    THERMAL_STORAGE = "THERMAL_STORAGE"
    BATTERY = "BATTERY"
    GENERATOR = "GENERATOR"
    FLEXIBLE_LOAD = "FLEXIBLE_LOAD"


class FlexibleAsset(BaseModel):
    asset_id: str
    asset_type: AssetType
    capacity_kw: float = Field(ge=0)
    response_time_min: float = Field(ge=0)
    duration_hours: float = Field(ge=0)
    availability_pct: float = Field(ge=0, le=100)
    cycling_cost_usd_kwh: float = Field(default=0.01, ge=0)


class GridServicesInput(BaseModel):
    facility_id: str
    utility_territory: str = Field(default="PJM")
    assets: List[FlexibleAsset] = Field(default_factory=list)
    peak_demand_kw: float = Field(default=1000, gt=0)
    annual_consumption_kwh: float = Field(default=5000000, gt=0)
    demand_charge_kw: float = Field(default=15, ge=0)
    energy_rate_peak_kwh: float = Field(default=0.15, ge=0)
    energy_rate_offpeak_kwh: float = Field(default=0.08, ge=0)
    dr_event_price_kwh: float = Field(default=0.50, ge=0)
    capacity_price_kw_year: float = Field(default=50, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ServiceOpportunity(BaseModel):
    service: str
    asset_id: str
    capacity_kw: float
    annual_revenue_usd: float
    annual_cost_usd: float
    net_value_usd: float
    risk_level: str


class GridServicesOutput(BaseModel):
    facility_id: str
    total_flexible_capacity_kw: float
    dr_eligible_capacity_kw: float
    frequency_reg_eligible_kw: float
    capacity_eligible_kw: float
    opportunities: List[ServiceOpportunity]
    total_annual_revenue_usd: float
    total_annual_cost_usd: float
    net_annual_value_usd: float
    demand_charge_savings_usd: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class GridServicesAgent:
    AGENT_ID = "GL-080"
    AGENT_NAME = "GRID-SERVICES"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"GridServicesAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = GridServicesInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _evaluate_service(self, asset: FlexibleAsset, service: GridService, inp: GridServicesInput) -> Optional[ServiceOpportunity]:
        """Evaluate a grid service opportunity for an asset."""

        # Check eligibility
        if service == GridService.FREQUENCY_REGULATION:
            if asset.response_time_min > 5:
                return None
            events_per_year = 8760  # Continuous
            revenue_rate = 0.02  # $/kW-hr
            duration = 1
        elif service == GridService.DEMAND_RESPONSE:
            if asset.duration_hours < 2:
                return None
            events_per_year = 50  # DR events
            revenue_rate = inp.dr_event_price_kwh
            duration = 4
        elif service == GridService.CAPACITY:
            if asset.availability_pct < 90:
                return None
            events_per_year = 1  # Annual capacity
            revenue_rate = inp.capacity_price_kw_year
            duration = 1
        elif service == GridService.SPINNING_RESERVE:
            if asset.response_time_min > 10:
                return None
            events_per_year = 100
            revenue_rate = 0.03
            duration = 1
        else:  # Energy arbitrage
            events_per_year = 250  # Trading days
            revenue_rate = inp.energy_rate_peak_kwh - inp.energy_rate_offpeak_kwh
            duration = asset.duration_hours

        # Calculate revenue
        capacity = asset.capacity_kw * (asset.availability_pct / 100)

        if service == GridService.CAPACITY:
            revenue = capacity * revenue_rate
        else:
            revenue = capacity * duration * events_per_year * revenue_rate

        # Calculate costs
        cycles = events_per_year if service != GridService.CAPACITY else 50
        cost = capacity * duration * cycles * asset.cycling_cost_usd_kwh

        net = revenue - cost

        if net <= 0:
            return None

        # Risk assessment
        if service == GridService.FREQUENCY_REGULATION:
            risk = "MEDIUM"
        elif service == GridService.DEMAND_RESPONSE:
            risk = "LOW"
        elif service == GridService.CAPACITY:
            risk = "LOW"
        else:
            risk = "HIGH"

        return ServiceOpportunity(
            service=service.value,
            asset_id=asset.asset_id,
            capacity_kw=round(capacity, 1),
            annual_revenue_usd=round(revenue, 2),
            annual_cost_usd=round(cost, 2),
            net_value_usd=round(net, 2),
            risk_level=risk
        )

    def _process(self, inp: GridServicesInput) -> GridServicesOutput:
        recommendations = []

        # Calculate capacity by eligibility
        total_flexible = sum(a.capacity_kw for a in inp.assets)
        dr_eligible = sum(a.capacity_kw for a in inp.assets if a.duration_hours >= 2)
        freq_eligible = sum(a.capacity_kw for a in inp.assets if a.response_time_min <= 5)
        cap_eligible = sum(a.capacity_kw for a in inp.assets if a.availability_pct >= 90)

        # Evaluate all opportunities
        opportunities = []
        for asset in inp.assets:
            for service in GridService:
                opp = self._evaluate_service(asset, service, inp)
                if opp:
                    opportunities.append(opp)

        # Sort by net value
        opportunities.sort(key=lambda x: -x.net_value_usd)

        # Totals (avoid double-counting same asset)
        asset_best = {}
        for opp in opportunities:
            if opp.asset_id not in asset_best or opp.net_value_usd > asset_best[opp.asset_id].net_value_usd:
                asset_best[opp.asset_id] = opp

        best_opportunities = list(asset_best.values())
        total_revenue = sum(o.annual_revenue_usd for o in best_opportunities)
        total_cost = sum(o.annual_cost_usd for o in best_opportunities)
        net_value = total_revenue - total_cost

        # Demand charge savings (assuming DR reduces peak by 10%)
        demand_savings = inp.peak_demand_kw * 0.1 * inp.demand_charge_kw * 12

        # Recommendations
        if not inp.assets:
            recommendations.append("No flexible assets identified - evaluate load flexibility potential")
        if dr_eligible < inp.peak_demand_kw * 0.1:
            recommendations.append("Limited DR capacity - consider thermal storage or battery")
        if freq_eligible > 0:
            recommendations.append(f"Frequency regulation eligible capacity: {freq_eligible:.0f} kW")
        if total_flexible > inp.peak_demand_kw * 0.2:
            recommendations.append("Significant flexible capacity - enroll in multiple programs")

        low_risk = [o for o in best_opportunities if o.risk_level == "LOW"]
        if low_risk:
            recommendations.append(f"{len(low_risk)} low-risk opportunities totaling ${sum(o.net_value_usd for o in low_risk):,.0f}/year")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "flexible_kw": round(total_flexible, 1),
            "net_value": round(net_value, 2)
        }).encode()).hexdigest()

        return GridServicesOutput(
            facility_id=inp.facility_id,
            total_flexible_capacity_kw=round(total_flexible, 1),
            dr_eligible_capacity_kw=round(dr_eligible, 1),
            frequency_reg_eligible_kw=round(freq_eligible, 1),
            capacity_eligible_kw=round(cap_eligible, 1),
            opportunities=opportunities[:10],  # Top 10
            total_annual_revenue_usd=round(total_revenue, 2),
            total_annual_cost_usd=round(total_cost, 2),
            net_annual_value_usd=round(net_value, 2),
            demand_charge_savings_usd=round(demand_savings, 2),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-080", "name": "GRID-SERVICES", "version": "1.0.0",
    "summary": "Grid services and demand response optimization",
    "standards": [{"ref": "IEEE 2030"}, {"ref": "FERC Order 2222"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
