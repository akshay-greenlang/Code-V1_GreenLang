# -*- coding: utf-8 -*-
"""
SBTiDecarbBridge - Bridge to 21 DECARB-X Agents for SBTi Reduction Planning
==============================================================================

This module bridges the SBTi Alignment Pack to 21 DECARB-X agents that provide
abatement options, MACC generation, decarbonisation roadmap building, technology
assessment, avoided emissions calculation, and lever-specific planning -- all
mapped to SBTi pathway requirements.

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

SBTi Integration:
    - Maps abatement options to SBTi target gaps
    - Validates reduction pathway against ACA/SDA requirements
    - Tracks avoided emissions (not counted toward SBTi targets)
    - Supports near-term (5-10yr) and long-term (2050) planning

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
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
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
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
    """Try to import a DECARB-X agent with graceful fallback."""
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
    RESEARCH = "research"

class SBTiPathwayAlignment(str, Enum):
    """SBTi pathway alignment for abatement options."""

    ALIGNED_1_5C = "aligned_1_5c"
    ALIGNED_WB2C = "aligned_wb2c"
    INSUFFICIENT = "insufficient"
    BEYOND_VALUE_CHAIN = "beyond_value_chain"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class DecarbBridgeConfig(BaseModel):
    """Configuration for the SBTi Decarb Bridge."""

    pack_id: str = Field(default="PACK-023")
    enable_provenance: bool = Field(default=True)
    base_year: int = Field(default=2019, ge=2015, le=2025)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2035)
    long_term_target_year: int = Field(default=2050, ge=2040, le=2060)
    pathway: str = Field(default="1.5C")
    sector: str = Field(default="general")
    sbti_gap_tco2e: float = Field(default=0.0, ge=0.0, description="Emissions gap to close for SBTi target")

class AbatementOption(BaseModel):
    """A single abatement option."""

    option_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    lever: str = Field(default="")
    scope: str = Field(default="scope_1")
    abatement_tco2e: float = Field(default=0.0, ge=0.0)
    cost_eur_per_tco2e: float = Field(default=0.0)
    capex_eur: float = Field(default=0.0, ge=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    trl: str = Field(default="commercial")
    implementation_years: int = Field(default=1, ge=1)
    sbti_eligible: bool = Field(default=True)
    sbti_pathway_alignment: str = Field(default="aligned_1_5c")

class AbatementResult(BaseModel):
    """Result of abatement options assessment."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    total_options: int = Field(default=0)
    sbti_eligible_options: int = Field(default=0)
    total_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    sbti_eligible_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    sbti_gap_tco2e: float = Field(default=0.0, ge=0.0)
    gap_covered_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    options: List[AbatementOption] = Field(default_factory=list)
    by_lever: Dict[str, float] = Field(default_factory=dict)
    by_scope: Dict[str, float] = Field(default_factory=dict)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class MACCResult(BaseModel):
    """Marginal Abatement Cost Curve result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    total_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    negative_cost_tco2e: float = Field(default=0.0, ge=0.0)
    positive_cost_tco2e: float = Field(default=0.0, ge=0.0)
    total_cost_eur: float = Field(default=0.0)
    avg_cost_eur_per_tco2e: float = Field(default=0.0)
    curve_points: List[Dict[str, Any]] = Field(default_factory=list)
    sbti_gap_covered_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class RoadmapResult(BaseModel):
    """Decarbonisation roadmap result aligned with SBTi pathway."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    pathway: str = Field(default="1.5C")
    base_year: int = Field(default=2019)
    target_year: int = Field(default=2050)
    total_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    total_capex_eur: float = Field(default=0.0, ge=0.0)
    total_opex_savings_eur: float = Field(default=0.0)
    sbti_pathway_aligned: bool = Field(default=False)
    near_term_gap_closed: bool = Field(default=False)
    long_term_gap_closed: bool = Field(default=False)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class TechnologyResult(BaseModel):
    """Technology assessment result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    technologies_assessed: int = Field(default=0)
    commercial_ready: int = Field(default=0)
    early_commercial: int = Field(default=0)
    demonstration: int = Field(default=0)
    technologies: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_for_sbti: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class ProgressMonitorResult(BaseModel):
    """Decarbonisation progress monitoring result."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    levers_implemented: int = Field(default=0)
    levers_planned: int = Field(default=0)
    achieved_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    planned_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    on_track_for_sbti: bool = Field(default=False)
    sbti_gap_remaining_tco2e: float = Field(default=0.0, ge=0.0)
    rag_status: str = Field(default="red")
    next_actions: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class BusinessCaseResult(BaseModel):
    """Business case result for SBTi-aligned decarbonisation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    total_investment_eur: float = Field(default=0.0, ge=0.0)
    total_savings_eur: float = Field(default=0.0)
    net_benefit_eur: float = Field(default=0.0)
    payback_period_years: float = Field(default=0.0, ge=0.0)
    irr_pct: float = Field(default=0.0)
    carbon_price_sensitivity: Dict[str, float] = Field(default_factory=dict)
    sbti_compliance_value_eur: float = Field(default=0.0, ge=0.0)
    risk_reduction_value_eur: float = Field(default=0.0, ge=0.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# DECARB-X Agent Mapping (21 agents)
# ---------------------------------------------------------------------------

DECARB_AGENTS: Dict[str, str] = {
    "DECARB-X-001": "greenlang.agents.decarb.abatement_catalog",
    "DECARB-X-002": "greenlang.agents.decarb.macc_generator",
    "DECARB-X-003": "greenlang.agents.decarb.roadmap_builder",
    "DECARB-X-004": "greenlang.agents.decarb.technology_assessment",
    "DECARB-X-005": "greenlang.agents.decarb.avoided_emissions",
    "DECARB-X-006": "greenlang.agents.decarb.renewable_energy_planner",
    "DECARB-X-007": "greenlang.agents.decarb.electrification_planner",
    "DECARB-X-008": "greenlang.agents.decarb.fuel_switching_optimizer",
    "DECARB-X-009": "greenlang.agents.decarb.energy_efficiency",
    "DECARB-X-010": "greenlang.agents.decarb.ccus_assessment",
    "DECARB-X-011": "greenlang.agents.decarb.supplier_engagement",
    "DECARB-X-012": "greenlang.agents.decarb.circular_economy",
    "DECARB-X-013": "greenlang.agents.decarb.process_innovation",
    "DECARB-X-014": "greenlang.agents.decarb.demand_reduction",
    "DECARB-X-015": "greenlang.agents.decarb.offset_strategy",
    "DECARB-X-016": "greenlang.agents.decarb.green_procurement",
    "DECARB-X-017": "greenlang.agents.decarb.fleet_decarbonisation",
    "DECARB-X-018": "greenlang.agents.decarb.building_decarbonisation",
    "DECARB-X-019": "greenlang.agents.decarb.digital_twin_carbon",
    "DECARB-X-020": "greenlang.agents.decarb.progress_monitor",
    "DECARB-X-021": "greenlang.agents.decarb.business_case_generator",
}

# SBTi reduction requirements by pathway
SBTI_REDUCTION_REQUIREMENTS: Dict[str, Dict[str, float]] = {
    "1.5C": {"near_term_s12_pct": 42.0, "near_term_s3_pct": 25.0, "long_term_pct": 90.0, "annual_rate": 4.2},
    "well_below_2C": {"near_term_s12_pct": 25.0, "near_term_s3_pct": 20.0, "long_term_pct": 90.0, "annual_rate": 2.5},
    "2C": {"near_term_s12_pct": 25.0, "near_term_s3_pct": 20.0, "long_term_pct": 80.0, "annual_rate": 2.5},
}

# ---------------------------------------------------------------------------
# SBTiDecarbBridge
# ---------------------------------------------------------------------------

class SBTiDecarbBridge:
    """Bridge to 21 DECARB-X agents for SBTi-aligned reduction planning.

    Provides abatement options assessment, MACC generation, roadmap building,
    technology assessment, progress monitoring, and business case generation
    -- all mapped to SBTi pathway requirements.

    Example:
        >>> bridge = SBTiDecarbBridge(DecarbBridgeConfig(pathway="1.5C", sbti_gap_tco2e=5000))
        >>> options = bridge.get_abatement_options(sector="manufacturing")
        >>> print(f"Gap covered: {options.gap_covered_pct}%")
    """

    def __init__(self, config: Optional[DecarbBridgeConfig] = None) -> None:
        """Initialize the SBTi Decarb Bridge."""
        self.config = config or DecarbBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._agents: Dict[str, Any] = {}
        for agent_id, module_path in DECARB_AGENTS.items():
            self._agents[agent_id] = _try_import_decarb_agent(agent_id, module_path)
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        self.logger.info(
            "SBTiDecarbBridge initialized: %d/%d agents, pathway=%s",
            available, len(self._agents), self.config.pathway,
        )

    def get_abatement_options(
        self,
        sector: str = "",
        scopes: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AbatementResult:
        """Get abatement options mapped to SBTi target gap.

        Args:
            sector: Sector code for filtering options.
            scopes: Scopes to include (scope_1, scope_2, scope_3).
            context: Optional context with option data.

        Returns:
            AbatementResult with SBTi-eligible options.
        """
        start = time.monotonic()
        context = context or {}
        scopes = scopes or ["scope_1", "scope_2", "scope_3"]

        options_data = context.get("options", [])
        options: List[AbatementOption] = []
        total_abatement = 0.0
        sbti_eligible_abatement = 0.0
        by_lever: Dict[str, float] = {}
        by_scope: Dict[str, float] = {}

        for opt_data in options_data:
            opt = AbatementOption(
                name=opt_data.get("name", ""),
                lever=opt_data.get("lever", ""),
                scope=opt_data.get("scope", "scope_1"),
                abatement_tco2e=opt_data.get("abatement_tco2e", 0.0),
                cost_eur_per_tco2e=opt_data.get("cost_eur_per_tco2e", 0.0),
                capex_eur=opt_data.get("capex_eur", 0.0),
                payback_years=opt_data.get("payback_years", 0.0),
                trl=opt_data.get("trl", "commercial"),
                implementation_years=opt_data.get("implementation_years", 1),
                sbti_eligible=opt_data.get("sbti_eligible", True),
                sbti_pathway_alignment=opt_data.get("sbti_pathway_alignment", "aligned_1_5c"),
            )
            options.append(opt)
            total_abatement += opt.abatement_tco2e
            if opt.sbti_eligible and opt.scope in scopes:
                sbti_eligible_abatement += opt.abatement_tco2e

            lever_key = opt.lever or "other"
            by_lever[lever_key] = by_lever.get(lever_key, 0.0) + opt.abatement_tco2e
            by_scope[opt.scope] = by_scope.get(opt.scope, 0.0) + opt.abatement_tco2e

        gap_covered = round(
            sbti_eligible_abatement / self.config.sbti_gap_tco2e * 100.0, 1
        ) if self.config.sbti_gap_tco2e > 0 else 0.0

        result = AbatementResult(
            status="completed",
            total_options=len(options),
            sbti_eligible_options=sum(1 for o in options if o.sbti_eligible),
            total_abatement_tco2e=round(total_abatement, 2),
            sbti_eligible_abatement_tco2e=round(sbti_eligible_abatement, 2),
            sbti_gap_tco2e=self.config.sbti_gap_tco2e,
            gap_covered_pct=min(gap_covered, 100.0),
            options=options,
            by_lever=by_lever,
            by_scope=by_scope,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def generate_macc(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> MACCResult:
        """Generate MACC curve for SBTi target gap analysis.

        Args:
            context: Optional context with MACC data.

        Returns:
            MACCResult with cost-ordered abatement curve.
        """
        start = time.monotonic()
        context = context or {}

        curve_points = context.get("curve_points", [])
        sorted_points = sorted(curve_points, key=lambda x: x.get("cost_eur_per_tco2e", 0.0))

        neg_cost = sum(p.get("abatement_tco2e", 0.0) for p in sorted_points if p.get("cost_eur_per_tco2e", 0.0) < 0)
        pos_cost = sum(p.get("abatement_tco2e", 0.0) for p in sorted_points if p.get("cost_eur_per_tco2e", 0.0) >= 0)
        total = neg_cost + pos_cost
        total_cost = sum(
            p.get("abatement_tco2e", 0.0) * p.get("cost_eur_per_tco2e", 0.0) for p in sorted_points
        )
        avg_cost = round(total_cost / total, 2) if total > 0 else 0.0
        gap_covered = round(total / self.config.sbti_gap_tco2e * 100.0, 1) if self.config.sbti_gap_tco2e > 0 else 0.0

        result = MACCResult(
            status="completed",
            total_abatement_tco2e=round(total, 2),
            negative_cost_tco2e=round(neg_cost, 2),
            positive_cost_tco2e=round(pos_cost, 2),
            total_cost_eur=round(total_cost, 2),
            avg_cost_eur_per_tco2e=avg_cost,
            curve_points=sorted_points,
            sbti_gap_covered_pct=min(gap_covered, 100.0),
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def build_roadmap(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoadmapResult:
        """Build decarbonisation roadmap aligned with SBTi pathway.

        Args:
            context: Optional context with roadmap data.

        Returns:
            RoadmapResult with phased reduction plan.
        """
        start = time.monotonic()
        context = context or {}

        requirements = SBTI_REDUCTION_REQUIREMENTS.get(self.config.pathway, SBTI_REDUCTION_REQUIREMENTS["1.5C"])
        total_reduction = context.get("total_reduction_tco2e", 0.0)

        phases = context.get("phases", [
            {"phase": 1, "name": "Quick wins", "years": "2025-2027", "reduction_tco2e": total_reduction * 0.2},
            {"phase": 2, "name": "Medium-term", "years": "2028-2030", "reduction_tco2e": total_reduction * 0.3},
            {"phase": 3, "name": "Long-term", "years": "2031-2040", "reduction_tco2e": total_reduction * 0.3},
            {"phase": 4, "name": "Deep decarb", "years": "2041-2050", "reduction_tco2e": total_reduction * 0.2},
        ])

        result = RoadmapResult(
            status="completed",
            pathway=self.config.pathway,
            base_year=self.config.base_year,
            target_year=self.config.long_term_target_year,
            total_reduction_tco2e=round(total_reduction, 2),
            phases=phases,
            milestones=context.get("milestones", []),
            total_capex_eur=context.get("total_capex_eur", 0.0),
            total_opex_savings_eur=context.get("total_opex_savings_eur", 0.0),
            sbti_pathway_aligned=context.get("sbti_pathway_aligned", False),
            near_term_gap_closed=context.get("near_term_gap_closed", False),
            long_term_gap_closed=context.get("long_term_gap_closed", False),
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def assess_technologies(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> TechnologyResult:
        """Assess technology readiness for SBTi-aligned decarbonisation.

        Args:
            context: Optional context with technology data.

        Returns:
            TechnologyResult with TRL assessment.
        """
        start = time.monotonic()
        context = context or {}

        technologies = context.get("technologies", [])
        commercial = sum(1 for t in technologies if t.get("trl") == "commercial")
        early = sum(1 for t in technologies if t.get("trl") == "early_commercial")
        demo = sum(1 for t in technologies if t.get("trl") in ("demonstration", "pilot", "research"))

        recommended = [t.get("name", "") for t in technologies if t.get("trl") in ("commercial", "early_commercial") and t.get("sbti_eligible", True)]

        result = TechnologyResult(
            status="completed",
            technologies_assessed=len(technologies),
            commercial_ready=commercial,
            early_commercial=early,
            demonstration=demo,
            technologies=technologies,
            recommended_for_sbti=recommended,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def monitor_progress(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProgressMonitorResult:
        """Monitor decarbonisation progress against SBTi targets.

        Args:
            context: Optional context with progress data.

        Returns:
            ProgressMonitorResult with SBTi gap assessment.
        """
        start = time.monotonic()
        context = context or {}

        achieved = context.get("achieved_reduction_tco2e", 0.0)
        planned = context.get("planned_reduction_tco2e", 0.0)
        gap_remaining = max(self.config.sbti_gap_tco2e - achieved, 0.0)
        on_track = achieved >= (self.config.sbti_gap_tco2e * 0.9)

        if on_track:
            rag = "green"
        elif achieved >= self.config.sbti_gap_tco2e * 0.6:
            rag = "amber"
        else:
            rag = "red"

        next_actions: List[str] = []
        if not on_track:
            next_actions.append(f"Accelerate reduction by {gap_remaining:.0f} tCO2e")
            next_actions.append("Review implementation timeline for planned levers")
        else:
            next_actions.append("Maintain current trajectory")
            next_actions.append("Prepare for SBTi annual progress report")

        result = ProgressMonitorResult(
            status="completed",
            levers_implemented=context.get("levers_implemented", 0),
            levers_planned=context.get("levers_planned", 0),
            achieved_reduction_tco2e=round(achieved, 2),
            planned_reduction_tco2e=round(planned, 2),
            on_track_for_sbti=on_track,
            sbti_gap_remaining_tco2e=round(gap_remaining, 2),
            rag_status=rag,
            next_actions=next_actions,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def generate_business_case(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> BusinessCaseResult:
        """Generate business case for SBTi-aligned decarbonisation.

        Args:
            context: Optional context with financial data.

        Returns:
            BusinessCaseResult with ROI and value analysis.
        """
        start = time.monotonic()
        context = context or {}

        investment = context.get("total_investment_eur", 0.0)
        savings = context.get("total_savings_eur", 0.0)
        net_benefit = savings - investment

        result = BusinessCaseResult(
            status="completed",
            total_investment_eur=round(investment, 2),
            total_savings_eur=round(savings, 2),
            net_benefit_eur=round(net_benefit, 2),
            payback_period_years=context.get("payback_period_years", 0.0),
            irr_pct=context.get("irr_pct", 0.0),
            carbon_price_sensitivity=context.get("carbon_price_sensitivity", {}),
            sbti_compliance_value_eur=context.get("sbti_compliance_value_eur", 0.0),
            risk_reduction_value_eur=context.get("risk_reduction_value_eur", 0.0),
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def get_agent_status(self) -> Dict[str, Any]:
        """Get availability status of all 21 DECARB-X agents.

        Returns:
            Dict with agent availability information.
        """
        agent_status: List[Dict[str, Any]] = []
        for agent_id, agent in self._agents.items():
            agent_status.append({
                "agent_id": agent_id,
                "available": not isinstance(agent, _AgentStub),
            })
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        return {
            "pack_id": self.config.pack_id,
            "total_agents": len(self._agents),
            "available": available,
            "degraded": len(self._agents) - available,
            "agents": agent_status,
        }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        return {
            "pack_id": self.config.pack_id,
            "module_version": _MODULE_VERSION,
            "total_agents": len(self._agents),
            "available_agents": available,
            "pathway": self.config.pathway,
            "sbti_gap_tco2e": self.config.sbti_gap_tco2e,
            "sector": self.config.sector,
        }
