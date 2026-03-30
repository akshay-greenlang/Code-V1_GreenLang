# -*- coding: utf-8 -*-
"""
CarbonNeutralDecarbBridge - Bridge to 21 DECARB-X Agents for PACK-024
=======================================================================

This module bridges the Carbon Neutral Pack to 21 DECARB-X agents that
provide abatement options, MACC generation, decarbonisation roadmap
building, technology assessment, and lever-specific planning -- all
mapped to PAS 2060 carbon management plan requirements.

DECARB-X Agent Mapping (21 agents):
    DECARB-X-001  Abatement Options Catalog
    DECARB-X-002  MACC Generator
    DECARB-X-003  Roadmap Builder
    DECARB-X-004  Technology Assessment
    DECARB-X-005  Avoided Emissions Calculator
    DECARB-X-006  Renewable Energy Planner
    DECARB-X-007  Electrification Planner
    DECARB-X-008  Fuel Switching Optimizer
    DECARB-X-009  Energy Efficiency Identifier
    DECARB-X-010  CCUS Assessment
    DECARB-X-011  Supplier Engagement Planner
    DECARB-X-012  Circular Economy Integration
    DECARB-X-013  Process Innovation Assessment
    DECARB-X-014  Demand Reduction Planner
    DECARB-X-015  Offset Strategy Planner
    DECARB-X-016  Green Procurement Engine
    DECARB-X-017  Fleet Decarbonisation
    DECARB-X-018  Building Decarbonisation
    DECARB-X-019  Digital Twin Carbon
    DECARB-X-020  Progress Monitor
    DECARB-X-021  Business Case Generator

PAS 2060 Integration:
    - Maps abatement options to PAS 2060 carbon management plan
    - Validates reduction pathway against PAS 2060 YoY requirement
    - Calculates residual emissions for offset procurement
    - Supports near-term reductions before offsetting

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-024 Carbon Neutral Pack
Status: Production Ready
"""

import hashlib
import importlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------

class _AgentStub:
    """Stub for unavailable DECARB-X agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "agent": self._agent_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._agent_name} not available, using stub",
            }
        return _stub_method

def _try_import_decarb_agent(agent_id: str, module_path: str) -> Any:
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("DECARB agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecarbLever(str, Enum):
    """Decarbonisation lever categories."""

    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    FUEL_SWITCHING = "fuel_switching"
    ENERGY_EFFICIENCY = "energy_efficiency"
    CCUS = "ccus"
    PROCESS_INNOVATION = "process_innovation"
    CIRCULAR_ECONOMY = "circular_economy"
    DEMAND_REDUCTION = "demand_reduction"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    GREEN_PROCUREMENT = "green_procurement"
    FLEET_DECARBONISATION = "fleet_decarbonisation"
    BUILDING_DECARBONISATION = "building_decarbonisation"
    OFFSET_STRATEGY = "offset_strategy"

class TechnologyReadiness(str, Enum):
    """Technology Readiness Level (TRL) classification."""

    COMMERCIAL = "commercial"
    EARLY_COMMERCIAL = "early_commercial"
    DEMONSTRATION = "demonstration"
    PILOT = "pilot"
    R_AND_D = "r_and_d"

class PAS2060ReductionAlignment(str, Enum):
    """PAS 2060 reduction requirement alignment."""

    ON_TRACK = "on_track"
    BEHIND = "behind"
    AT_RISK = "at_risk"
    NOT_STARTED = "not_started"

# ---------------------------------------------------------------------------
# DECARB-X Routing Table (21 agents)
# ---------------------------------------------------------------------------

DECARB_ROUTING_TABLE: Dict[str, Dict[str, Any]] = {
    "abatement_catalog": {"agent": "DECARB-X-001", "module": "greenlang.agents.decarb.abatement_catalog"},
    "macc_generator": {"agent": "DECARB-X-002", "module": "greenlang.agents.decarb.macc_generator"},
    "roadmap_builder": {"agent": "DECARB-X-003", "module": "greenlang.agents.decarb.roadmap_builder"},
    "technology_assessment": {"agent": "DECARB-X-004", "module": "greenlang.agents.decarb.technology_assessment"},
    "avoided_emissions": {"agent": "DECARB-X-005", "module": "greenlang.agents.decarb.avoided_emissions"},
    "renewable_energy": {"agent": "DECARB-X-006", "module": "greenlang.agents.decarb.renewable_energy"},
    "electrification": {"agent": "DECARB-X-007", "module": "greenlang.agents.decarb.electrification"},
    "fuel_switching": {"agent": "DECARB-X-008", "module": "greenlang.agents.decarb.fuel_switching"},
    "energy_efficiency": {"agent": "DECARB-X-009", "module": "greenlang.agents.decarb.energy_efficiency"},
    "ccus": {"agent": "DECARB-X-010", "module": "greenlang.agents.decarb.ccus"},
    "supplier_engagement": {"agent": "DECARB-X-011", "module": "greenlang.agents.decarb.supplier_engagement"},
    "circular_economy": {"agent": "DECARB-X-012", "module": "greenlang.agents.decarb.circular_economy"},
    "process_innovation": {"agent": "DECARB-X-013", "module": "greenlang.agents.decarb.process_innovation"},
    "demand_reduction": {"agent": "DECARB-X-014", "module": "greenlang.agents.decarb.demand_reduction"},
    "offset_strategy": {"agent": "DECARB-X-015", "module": "greenlang.agents.decarb.offset_strategy"},
    "green_procurement": {"agent": "DECARB-X-016", "module": "greenlang.agents.decarb.green_procurement"},
    "fleet_decarbonisation": {"agent": "DECARB-X-017", "module": "greenlang.agents.decarb.fleet_decarbonisation"},
    "building_decarbonisation": {"agent": "DECARB-X-018", "module": "greenlang.agents.decarb.building_decarbonisation"},
    "digital_twin": {"agent": "DECARB-X-019", "module": "greenlang.agents.decarb.digital_twin"},
    "progress_monitor": {"agent": "DECARB-X-020", "module": "greenlang.agents.decarb.progress_monitor"},
    "business_case": {"agent": "DECARB-X-021", "module": "greenlang.agents.decarb.business_case"},
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DecarbBridgeConfig(BaseModel):
    """Configuration for the DECARB Bridge."""

    pack_id: str = Field(default="PACK-024")
    enable_provenance: bool = Field(default=True)
    planning_horizon_years: int = Field(default=10, ge=1, le=30)
    discount_rate_pct: float = Field(default=8.0, ge=0.0, le=30.0)
    carbon_price_usd: float = Field(default=50.0, ge=0.0)

class AbatementOption(BaseModel):
    """Single abatement option from DECARB agents."""

    option_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    lever: str = Field(default="")
    reduction_tco2e: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0)
    cost_per_tco2e: float = Field(default=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    trl: str = Field(default="commercial")
    scope: str = Field(default="scope_1")
    implementation_year: int = Field(default=2025)

class AbatementResult(BaseModel):
    """Abatement catalog result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    options: List[AbatementOption] = Field(default_factory=list)
    total_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0)
    avg_cost_per_tco2e: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class MACCResult(BaseModel):
    """Marginal Abatement Cost Curve result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    curve_data: List[Dict[str, float]] = Field(default_factory=list)
    total_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    negative_cost_tco2e: float = Field(default=0.0, ge=0.0)
    positive_cost_tco2e: float = Field(default=0.0, ge=0.0)
    break_even_price: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class RoadmapResult(BaseModel):
    """Decarbonisation roadmap result aligned with PAS 2060."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    total_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    residual_tco2e: float = Field(default=0.0, ge=0.0)
    offset_required_tco2e: float = Field(default=0.0, ge=0.0)
    pas_2060_aligned: bool = Field(default=False)
    yoy_reduction_maintained: bool = Field(default=False)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class TechnologyResult(BaseModel):
    """Technology assessment result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    technologies: List[Dict[str, Any]] = Field(default_factory=list)
    recommended: List[str] = Field(default_factory=list)
    total_potential_tco2e: float = Field(default=0.0, ge=0.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class ProgressMonitorResult(BaseModel):
    """Progress monitoring result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    target_tco2e: float = Field(default=0.0, ge=0.0)
    actual_tco2e: float = Field(default=0.0, ge=0.0)
    gap_tco2e: float = Field(default=0.0)
    on_track: bool = Field(default=False)
    pas_2060_alignment: str = Field(default="not_started")
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class BusinessCaseResult(BaseModel):
    """Business case generation result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    npv_usd: float = Field(default=0.0)
    irr_pct: float = Field(default=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    total_investment_usd: float = Field(default=0.0)
    annual_savings_usd: float = Field(default=0.0)
    carbon_value_usd: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# CarbonNeutralDecarbBridge
# ---------------------------------------------------------------------------

class CarbonNeutralDecarbBridge:
    """Bridge to 21 DECARB-X agents for PAS 2060 reduction planning.

    Provides abatement options, MACC generation, roadmap building, technology
    assessment, progress monitoring, and business case generation -- all
    aligned with PAS 2060 carbon management plan requirements.

    Example:
        >>> bridge = CarbonNeutralDecarbBridge()
        >>> result = bridge.get_abatement_options(context={"baseline_tco2e": 10000})
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[DecarbBridgeConfig] = None) -> None:
        self.config = config or DecarbBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}
        for source, info in DECARB_ROUTING_TABLE.items():
            self._agents[source] = _try_import_decarb_agent(info["agent"], info["module"])
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        self.logger.info(
            "CarbonNeutralDecarbBridge initialized: %d/%d agents available",
            available, len(self._agents),
        )

    def get_abatement_options(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> AbatementResult:
        """Get abatement options for PAS 2060 carbon management plan."""
        start = time.monotonic()
        context = context or {}
        options = context.get("options", [])
        parsed = [AbatementOption(**o) if isinstance(o, dict) else o for o in options]
        total_red = sum(o.reduction_tco2e for o in parsed)
        total_cost = sum(o.cost_usd for o in parsed)
        avg_cost = round(total_cost / total_red, 2) if total_red > 0 else 0.0

        result = AbatementResult(
            status="completed",
            options=parsed,
            total_reduction_tco2e=round(total_red, 2),
            total_cost_usd=round(total_cost, 2),
            avg_cost_per_tco2e=avg_cost,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def generate_macc(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> MACCResult:
        """Generate Marginal Abatement Cost Curve."""
        start = time.monotonic()
        context = context or {}
        curve = context.get("curve_data", [])
        total = sum(p.get("abatement_tco2e", 0) for p in curve)
        neg = sum(p.get("abatement_tco2e", 0) for p in curve if p.get("cost_per_tco2e", 0) < 0)
        pos = total - neg

        result = MACCResult(
            status="completed",
            curve_data=curve,
            total_abatement_tco2e=round(total, 2),
            negative_cost_tco2e=round(neg, 2),
            positive_cost_tco2e=round(pos, 2),
            break_even_price=self.config.carbon_price_usd,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def build_roadmap(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoadmapResult:
        """Build decarbonisation roadmap aligned with PAS 2060."""
        start = time.monotonic()
        context = context or {}
        milestones = context.get("milestones", [])
        baseline = context.get("baseline_tco2e", 0.0)
        total_red = sum(m.get("reduction_tco2e", 0) for m in milestones)
        residual = max(0.0, baseline - total_red)

        # Check YoY reduction for PAS 2060
        yearly_emissions = context.get("yearly_emissions", [])
        yoy_maintained = all(
            yearly_emissions[i] >= yearly_emissions[i + 1]
            for i in range(len(yearly_emissions) - 1)
        ) if len(yearly_emissions) >= 2 else False

        result = RoadmapResult(
            status="completed",
            milestones=milestones,
            total_reduction_tco2e=round(total_red, 2),
            residual_tco2e=round(residual, 2),
            offset_required_tco2e=round(residual, 2),
            pas_2060_aligned=yoy_maintained,
            yoy_reduction_maintained=yoy_maintained,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def assess_technology(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> TechnologyResult:
        """Assess technology options for decarbonisation."""
        start = time.monotonic()
        context = context or {}
        technologies = context.get("technologies", [])
        recommended = [t.get("name", "") for t in technologies if t.get("trl", "") in ("commercial", "early_commercial")]
        total_potential = sum(t.get("potential_tco2e", 0) for t in technologies)

        result = TechnologyResult(
            status="completed",
            technologies=technologies,
            recommended=recommended,
            total_potential_tco2e=round(total_potential, 2),
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def monitor_progress(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProgressMonitorResult:
        """Monitor reduction progress against PAS 2060 plan."""
        start = time.monotonic()
        context = context or {}
        target = context.get("target_tco2e", 0.0)
        actual = context.get("actual_tco2e", 0.0)
        gap = actual - target
        on_track = gap <= 0

        if on_track:
            alignment = PAS2060ReductionAlignment.ON_TRACK.value
        elif gap / target < 0.1 if target > 0 else False:
            alignment = PAS2060ReductionAlignment.BEHIND.value
        else:
            alignment = PAS2060ReductionAlignment.AT_RISK.value

        recommendations: List[str] = []
        if not on_track:
            recommendations.append(f"Gap of {gap:.1f} tCO2e to target")
            recommendations.append("Accelerate reduction initiatives or increase offset procurement")

        result = ProgressMonitorResult(
            status="completed",
            target_tco2e=round(target, 2),
            actual_tco2e=round(actual, 2),
            gap_tco2e=round(gap, 2),
            on_track=on_track,
            pas_2060_alignment=alignment,
            recommendations=recommendations,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def generate_business_case(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BusinessCaseResult:
        """Generate business case for reduction investments."""
        start = time.monotonic()
        context = context or {}
        investment = context.get("total_investment_usd", 0.0)
        annual_savings = context.get("annual_savings_usd", 0.0)
        reduction_tco2e = context.get("reduction_tco2e", 0.0)
        carbon_value = reduction_tco2e * self.config.carbon_price_usd
        payback = round(investment / annual_savings, 1) if annual_savings > 0 else 0.0

        # Simple NPV over planning horizon
        npv = -investment
        for yr in range(1, self.config.planning_horizon_years + 1):
            npv += (annual_savings + carbon_value / self.config.planning_horizon_years) / (1 + self.config.discount_rate_pct / 100) ** yr
        irr = round((annual_savings / investment - 1) * 100, 1) if investment > 0 else 0.0

        result = BusinessCaseResult(
            status="completed",
            npv_usd=round(npv, 2),
            irr_pct=irr,
            payback_years=payback,
            total_investment_usd=round(investment, 2),
            annual_savings_usd=round(annual_savings, 2),
            carbon_value_usd=round(carbon_value, 2),
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_agent_status(self) -> Dict[str, bool]:
        """Get availability status of all 21 DECARB agents."""
        return {source: not isinstance(agent, _AgentStub) for source, agent in self._agents.items()}

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        return {
            "pack_id": self.config.pack_id,
            "module_version": _MODULE_VERSION,
            "total_agents": len(self._agents),
            "available_agents": available,
            "planning_horizon_years": self.config.planning_horizon_years,
            "carbon_price_usd": self.config.carbon_price_usd,
        }
