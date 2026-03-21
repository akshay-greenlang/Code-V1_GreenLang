# -*- coding: utf-8 -*-
"""
Reduction Planning Workflow
================================

5-phase workflow for building a prioritised emissions reduction roadmap
within PACK-021 Net-Zero Starter Pack.  The workflow profiles emissions
hotspots, identifies abatement actions from a technology catalog,
performs cost-benefit analysis, ranks actions by cost-effectiveness,
and generates a phased implementation roadmap.

Phases:
    1. EmissionsProfile     -- Analyse emissions by scope/category/source
    2. ActionIdentification -- Match hotspots to abatement options
    3. CostAnalysis         -- Calculate cost/tCO2e, NPV, IRR, payback
    4. Prioritization       -- Rank actions; apply budget constraints
    5. RoadmapGeneration    -- Generate phased roadmap (short/medium/long)

Zero-hallucination: all cost and abatement calculations use deterministic
formulas and published abatement cost data.  No LLM calls in the numeric
computation path.  SHA-256 provenance hashes for auditability.

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class TimeHorizon(str, Enum):
    """Implementation time horizon for actions."""

    SHORT = "short_term"     # 0-2 years
    MEDIUM = "medium_term"   # 3-5 years
    LONG = "long_term"       # 6-10 years


class FeasibilityRating(str, Enum):
    """Action feasibility rating."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionCategory(str, Enum):
    """Abatement action categories."""

    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    FUEL_SWITCHING = "fuel_switching"
    ELECTRIFICATION = "electrification"
    PROCESS_OPTIMISATION = "process_optimisation"
    SUPPLY_CHAIN = "supply_chain"
    FLEET_TRANSITION = "fleet_transition"
    BUILDING_RETROFIT = "building_retrofit"
    BEHAVIOURAL_CHANGE = "behavioural_change"
    CIRCULAR_ECONOMY = "circular_economy"
    CARBON_CAPTURE = "carbon_capture"
    OTHER = "other"


# =============================================================================
# ABATEMENT TECHNOLOGY CATALOG (Zero-Hallucination, based on McKinsey MACC
# and IEA Net Zero Roadmap 2023 cost ranges)
# =============================================================================

ABATEMENT_CATALOG: List[Dict[str, Any]] = [
    {
        "action_id": "ACT-001",
        "name": "LED lighting upgrade",
        "category": "energy_efficiency",
        "applicable_scopes": ["scope2"],
        "applicable_sources": ["electricity"],
        "typical_reduction_pct": 15.0,
        "cost_per_tco2e_usd": -50.0,
        "capex_per_unit_usd": 5000.0,
        "payback_years": 2.0,
        "lifetime_years": 10,
        "feasibility": "high",
        "co_benefits": ["lower_energy_bills", "improved_lighting_quality"],
        "time_horizon": "short_term",
    },
    {
        "action_id": "ACT-002",
        "name": "HVAC system optimisation",
        "category": "energy_efficiency",
        "applicable_scopes": ["scope1", "scope2"],
        "applicable_sources": ["heating", "cooling", "electricity"],
        "typical_reduction_pct": 20.0,
        "cost_per_tco2e_usd": -30.0,
        "capex_per_unit_usd": 25000.0,
        "payback_years": 3.5,
        "lifetime_years": 15,
        "feasibility": "high",
        "co_benefits": ["thermal_comfort", "lower_energy_bills"],
        "time_horizon": "short_term",
    },
    {
        "action_id": "ACT-003",
        "name": "On-site solar PV installation",
        "category": "renewable_energy",
        "applicable_scopes": ["scope2"],
        "applicable_sources": ["electricity"],
        "typical_reduction_pct": 30.0,
        "cost_per_tco2e_usd": 10.0,
        "capex_per_unit_usd": 80000.0,
        "payback_years": 6.0,
        "lifetime_years": 25,
        "feasibility": "high",
        "co_benefits": ["energy_independence", "hedge_against_price_volatility"],
        "time_horizon": "medium_term",
    },
    {
        "action_id": "ACT-004",
        "name": "Renewable electricity PPA",
        "category": "renewable_energy",
        "applicable_scopes": ["scope2"],
        "applicable_sources": ["electricity"],
        "typical_reduction_pct": 80.0,
        "cost_per_tco2e_usd": 5.0,
        "capex_per_unit_usd": 0.0,
        "payback_years": 0.0,
        "lifetime_years": 15,
        "feasibility": "high",
        "co_benefits": ["price_certainty", "additionality"],
        "time_horizon": "short_term",
    },
    {
        "action_id": "ACT-005",
        "name": "Natural gas to heat pump conversion",
        "category": "electrification",
        "applicable_scopes": ["scope1"],
        "applicable_sources": ["heating", "natural_gas"],
        "typical_reduction_pct": 60.0,
        "cost_per_tco2e_usd": 40.0,
        "capex_per_unit_usd": 35000.0,
        "payback_years": 7.0,
        "lifetime_years": 20,
        "feasibility": "medium",
        "co_benefits": ["cooling_capability", "efficiency_gain"],
        "time_horizon": "medium_term",
    },
    {
        "action_id": "ACT-006",
        "name": "Fleet electrification (passenger vehicles)",
        "category": "fleet_transition",
        "applicable_scopes": ["scope1"],
        "applicable_sources": ["fleet", "mobile"],
        "typical_reduction_pct": 70.0,
        "cost_per_tco2e_usd": 60.0,
        "capex_per_unit_usd": 15000.0,
        "payback_years": 5.0,
        "lifetime_years": 10,
        "feasibility": "medium",
        "co_benefits": ["lower_fuel_costs", "noise_reduction"],
        "time_horizon": "medium_term",
    },
    {
        "action_id": "ACT-007",
        "name": "Supplier engagement programme",
        "category": "supply_chain",
        "applicable_scopes": ["scope3"],
        "applicable_sources": ["cat1_purchased_goods", "cat4_upstream_transport"],
        "typical_reduction_pct": 10.0,
        "cost_per_tco2e_usd": 15.0,
        "capex_per_unit_usd": 50000.0,
        "payback_years": 3.0,
        "lifetime_years": 5,
        "feasibility": "medium",
        "co_benefits": ["supply_chain_resilience", "transparency"],
        "time_horizon": "short_term",
    },
    {
        "action_id": "ACT-008",
        "name": "Building envelope retrofit (insulation)",
        "category": "building_retrofit",
        "applicable_scopes": ["scope1", "scope2"],
        "applicable_sources": ["heating", "cooling"],
        "typical_reduction_pct": 25.0,
        "cost_per_tco2e_usd": 20.0,
        "capex_per_unit_usd": 60000.0,
        "payback_years": 8.0,
        "lifetime_years": 30,
        "feasibility": "medium",
        "co_benefits": ["thermal_comfort", "property_value"],
        "time_horizon": "medium_term",
    },
    {
        "action_id": "ACT-009",
        "name": "Employee commuting programme (remote work + shuttle)",
        "category": "behavioural_change",
        "applicable_scopes": ["scope3"],
        "applicable_sources": ["cat7_commuting"],
        "typical_reduction_pct": 25.0,
        "cost_per_tco2e_usd": -10.0,
        "capex_per_unit_usd": 10000.0,
        "payback_years": 1.0,
        "lifetime_years": 5,
        "feasibility": "high",
        "co_benefits": ["employee_satisfaction", "reduced_traffic"],
        "time_horizon": "short_term",
    },
    {
        "action_id": "ACT-010",
        "name": "Business travel reduction (virtual meetings)",
        "category": "behavioural_change",
        "applicable_scopes": ["scope3"],
        "applicable_sources": ["cat6_business_travel"],
        "typical_reduction_pct": 40.0,
        "cost_per_tco2e_usd": -80.0,
        "capex_per_unit_usd": 5000.0,
        "payback_years": 0.5,
        "lifetime_years": 5,
        "feasibility": "high",
        "co_benefits": ["cost_savings", "work_life_balance"],
        "time_horizon": "short_term",
    },
    {
        "action_id": "ACT-011",
        "name": "Waste reduction and recycling programme",
        "category": "circular_economy",
        "applicable_scopes": ["scope3"],
        "applicable_sources": ["cat5_waste"],
        "typical_reduction_pct": 30.0,
        "cost_per_tco2e_usd": -20.0,
        "capex_per_unit_usd": 15000.0,
        "payback_years": 2.0,
        "lifetime_years": 10,
        "feasibility": "high",
        "co_benefits": ["waste_cost_reduction", "regulatory_compliance"],
        "time_horizon": "short_term",
    },
    {
        "action_id": "ACT-012",
        "name": "Heavy fleet transition (hydrogen/electric trucks)",
        "category": "fleet_transition",
        "applicable_scopes": ["scope1", "scope3"],
        "applicable_sources": ["fleet", "cat4_upstream_transport", "cat9_downstream_transport"],
        "typical_reduction_pct": 50.0,
        "cost_per_tco2e_usd": 120.0,
        "capex_per_unit_usd": 150000.0,
        "payback_years": 10.0,
        "lifetime_years": 12,
        "feasibility": "low",
        "co_benefits": ["air_quality", "noise_reduction"],
        "time_horizon": "long_term",
    },
    {
        "action_id": "ACT-013",
        "name": "Green procurement policy",
        "category": "supply_chain",
        "applicable_scopes": ["scope3"],
        "applicable_sources": ["cat1_purchased_goods", "cat2_capital_goods"],
        "typical_reduction_pct": 15.0,
        "cost_per_tco2e_usd": 25.0,
        "capex_per_unit_usd": 20000.0,
        "payback_years": 4.0,
        "lifetime_years": 10,
        "feasibility": "medium",
        "co_benefits": ["brand_reputation", "supply_chain_innovation"],
        "time_horizon": "medium_term",
    },
    {
        "action_id": "ACT-014",
        "name": "Process heat electrification",
        "category": "electrification",
        "applicable_scopes": ["scope1"],
        "applicable_sources": ["process_heat", "natural_gas", "heating"],
        "typical_reduction_pct": 55.0,
        "cost_per_tco2e_usd": 80.0,
        "capex_per_unit_usd": 200000.0,
        "payback_years": 9.0,
        "lifetime_years": 20,
        "feasibility": "low",
        "co_benefits": ["process_efficiency", "air_quality"],
        "time_horizon": "long_term",
    },
    {
        "action_id": "ACT-015",
        "name": "Biogas / biomethane fuel switching",
        "category": "fuel_switching",
        "applicable_scopes": ["scope1"],
        "applicable_sources": ["natural_gas", "heating"],
        "typical_reduction_pct": 80.0,
        "cost_per_tco2e_usd": 45.0,
        "capex_per_unit_usd": 40000.0,
        "payback_years": 6.0,
        "lifetime_years": 15,
        "feasibility": "medium",
        "co_benefits": ["waste_valorisation", "circular_economy"],
        "time_horizon": "medium_term",
    },
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class EmissionSource(BaseModel):
    """An identified emission source / hotspot."""

    source_id: str = Field(default="")
    scope: str = Field(default="scope1", description="scope1|scope2|scope3")
    category: str = Field(default="", description="Emission category or Scope 3 cat")
    source_label: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0, ge=0.0)
    share_of_total_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class EmissionsProfile(BaseModel):
    """Baseline emissions profile for hotspot analysis."""

    scope1_stationary_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_mobile_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_by_category: Dict[str, float] = Field(default_factory=dict)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)


class AbatementAction(BaseModel):
    """A matched abatement action with cost analysis."""

    action_id: str = Field(default="")
    name: str = Field(default="")
    category: str = Field(default="")
    target_scope: str = Field(default="")
    target_source: str = Field(default="")
    reduction_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    cost_per_tco2e_usd: float = Field(default=0.0)
    total_capex_usd: float = Field(default=0.0, ge=0.0)
    annual_opex_savings_usd: float = Field(default=0.0)
    npv_usd: float = Field(default=0.0)
    irr_pct: float = Field(default=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    lifetime_years: int = Field(default=10, ge=1)
    feasibility: str = Field(default="medium")
    co_benefits: List[str] = Field(default_factory=list)
    time_horizon: str = Field(default="medium_term")
    priority_rank: int = Field(default=0, ge=0)
    cumulative_reduction_tco2e: float = Field(default=0.0, ge=0.0)


class MACCDataPoint(BaseModel):
    """Single data point for the Marginal Abatement Cost Curve."""

    action_id: str = Field(default="")
    action_name: str = Field(default="")
    cost_per_tco2e_usd: float = Field(default=0.0)
    reduction_tco2e: float = Field(default=0.0, ge=0.0)
    cumulative_reduction_tco2e: float = Field(default=0.0, ge=0.0)


class RoadmapPhase(BaseModel):
    """A phase of the implementation roadmap."""

    phase_label: str = Field(default="")
    time_horizon: str = Field(default="short_term")
    year_start: int = Field(default=2025)
    year_end: int = Field(default=2027)
    actions: List[str] = Field(default_factory=list, description="Action IDs")
    total_capex_usd: float = Field(default=0.0, ge=0.0)
    total_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    cumulative_reduction_tco2e: float = Field(default=0.0, ge=0.0)


class ReductionPlanningConfig(BaseModel):
    """Configuration for the reduction planning workflow."""

    emissions_profile: EmissionsProfile = Field(default_factory=EmissionsProfile)
    budget_constraint_usd: Optional[float] = Field(None, ge=0.0, description="Total budget cap")
    max_actions: int = Field(default=20, ge=1, le=50)
    planning_horizon_years: int = Field(default=10, ge=1, le=30)
    base_year: int = Field(default=2024, ge=2015, le=2050)
    include_scope3_actions: bool = Field(default=True)
    discount_rate_pct: float = Field(default=8.0, ge=0.0, le=30.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ReductionPlanningResult(BaseModel):
    """Complete result from the reduction planning workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="reduction_planning")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    hotspots: List[EmissionSource] = Field(default_factory=list)
    prioritized_actions: List[AbatementAction] = Field(default_factory=list)
    macc_data: List[MACCDataPoint] = Field(default_factory=list)
    roadmap: List[RoadmapPhase] = Field(default_factory=list)
    total_reduction_tco2e: float = Field(default=0.0, ge=0.0)
    total_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_capex_usd: float = Field(default=0.0, ge=0.0)
    average_cost_per_tco2e_usd: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ReductionPlanningWorkflow:
    """
    5-phase reduction planning workflow.

    Profiles emissions hotspots, identifies abatement actions, performs
    cost analysis, prioritises by cost-effectiveness, and generates a
    phased implementation roadmap.

    Zero-hallucination: all cost and reduction calculations use
    deterministic formulas.  No LLM calls in the numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Workflow configuration.

    Example:
        >>> wf = ReductionPlanningWorkflow()
        >>> cfg = ReductionPlanningConfig(emissions_profile=profile)
        >>> result = await wf.execute(cfg)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self) -> None:
        """Initialise ReductionPlanningWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._hotspots: List[EmissionSource] = []
        self._matched_actions: List[AbatementAction] = []
        self._macc_data: List[MACCDataPoint] = []
        self._roadmap: List[RoadmapPhase] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: ReductionPlanningConfig) -> ReductionPlanningResult:
        """
        Execute the 5-phase reduction planning workflow.

        Args:
            config: Reduction planning configuration with emissions profile,
                budget constraints, and planning parameters.

        Returns:
            ReductionPlanningResult with prioritised actions and roadmap.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting reduction planning workflow %s, total=%.2f tCO2e, budget=%s",
            self.workflow_id, config.emissions_profile.total_tco2e,
            config.budget_constraint_usd,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_emissions_profile(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_action_identification(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_cost_analysis(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_prioritization(config)
            self._phase_results.append(phase4)

            phase5 = await self._phase_roadmap_generation(config)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Reduction planning workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        total_red = sum(a.reduction_tco2e for a in self._matched_actions)
        total_capex = sum(a.total_capex_usd for a in self._matched_actions)
        total_pct = (total_red / config.emissions_profile.total_tco2e * 100) if config.emissions_profile.total_tco2e > 0 else 0
        avg_cost = (total_capex / total_red) if total_red > 0 else 0

        result = ReductionPlanningResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            hotspots=self._hotspots,
            prioritized_actions=self._matched_actions,
            macc_data=self._macc_data,
            roadmap=self._roadmap,
            total_reduction_tco2e=round(total_red, 4),
            total_reduction_pct=round(total_pct, 2),
            total_capex_usd=round(total_capex, 2),
            average_cost_per_tco2e_usd=round(avg_cost, 2),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Reduction planning %s completed in %.2fs: %d actions, %.1f tCO2e reduction (%.1f%%)",
            self.workflow_id, elapsed, len(self._matched_actions), total_red, total_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Emissions Profile
    # -------------------------------------------------------------------------

    async def _phase_emissions_profile(self, config: ReductionPlanningConfig) -> PhaseResult:
        """Analyse emissions profile to identify hotspots."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        profile = config.emissions_profile
        total = profile.total_tco2e or 1.0

        hotspots: List[EmissionSource] = []

        # Scope 1 stationary
        if profile.scope1_stationary_tco2e > 0:
            hotspots.append(EmissionSource(
                source_id="HS-S1-STAT",
                scope="scope1", category="stationary_combustion",
                source_label="Scope 1 - Stationary Combustion",
                emissions_tco2e=profile.scope1_stationary_tco2e,
                share_of_total_pct=round((profile.scope1_stationary_tco2e / total) * 100, 2),
            ))

        # Scope 1 mobile
        if profile.scope1_mobile_tco2e > 0:
            hotspots.append(EmissionSource(
                source_id="HS-S1-MOB",
                scope="scope1", category="mobile_combustion",
                source_label="Scope 1 - Mobile Combustion",
                emissions_tco2e=profile.scope1_mobile_tco2e,
                share_of_total_pct=round((profile.scope1_mobile_tco2e / total) * 100, 2),
            ))

        # Scope 2
        if profile.scope2_location_tco2e > 0:
            hotspots.append(EmissionSource(
                source_id="HS-S2-ELEC",
                scope="scope2", category="purchased_electricity",
                source_label="Scope 2 - Purchased Electricity",
                emissions_tco2e=profile.scope2_location_tco2e,
                share_of_total_pct=round((profile.scope2_location_tco2e / total) * 100, 2),
            ))

        # Scope 3 by category
        for cat, val in profile.scope3_by_category.items():
            if val > 0:
                hotspots.append(EmissionSource(
                    source_id=f"HS-S3-{cat.upper()}",
                    scope="scope3", category=cat,
                    source_label=f"Scope 3 - {cat}",
                    emissions_tco2e=val,
                    share_of_total_pct=round((val / total) * 100, 2),
                ))

        hotspots.sort(key=lambda h: h.emissions_tco2e, reverse=True)
        self._hotspots = hotspots

        outputs["hotspot_count"] = len(hotspots)
        outputs["top_hotspot"] = hotspots[0].source_label if hotspots else ""
        outputs["top_hotspot_share_pct"] = hotspots[0].share_of_total_pct if hotspots else 0

        if not hotspots:
            warnings.append("No emissions hotspots identified; emissions profile may be incomplete")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Emissions profile: %d hotspots identified", len(hotspots))
        return PhaseResult(
            phase_name="emissions_profile",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Action Identification
    # -------------------------------------------------------------------------

    async def _phase_action_identification(self, config: ReductionPlanningConfig) -> PhaseResult:
        """Match emissions hotspots to abatement options from catalog."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        matched: List[AbatementAction] = []

        for hotspot in self._hotspots:
            if hotspot.scope == "scope3" and not config.include_scope3_actions:
                continue
            matching_techs = self._find_matching_technologies(hotspot)
            for tech in matching_techs:
                reduction_tco2e = hotspot.emissions_tco2e * (tech["typical_reduction_pct"] / 100.0)
                action = AbatementAction(
                    action_id=tech["action_id"],
                    name=tech["name"],
                    category=tech["category"],
                    target_scope=hotspot.scope,
                    target_source=hotspot.category,
                    reduction_tco2e=round(reduction_tco2e, 4),
                    reduction_pct=tech["typical_reduction_pct"],
                    cost_per_tco2e_usd=tech["cost_per_tco2e_usd"],
                    total_capex_usd=tech["capex_per_unit_usd"],
                    payback_years=tech["payback_years"],
                    lifetime_years=tech["lifetime_years"],
                    feasibility=tech["feasibility"],
                    co_benefits=tech.get("co_benefits", []),
                    time_horizon=tech["time_horizon"],
                )
                matched.append(action)

        # Deduplicate by action_id (keep highest reduction)
        seen: Dict[str, AbatementAction] = {}
        for act in matched:
            key = act.action_id
            if key not in seen or act.reduction_tco2e > seen[key].reduction_tco2e:
                seen[key] = act
        self._matched_actions = list(seen.values())

        outputs["matched_action_count"] = len(self._matched_actions)
        outputs["categories_covered"] = list({a.category for a in self._matched_actions})
        unmatched = [h for h in self._hotspots if not self._find_matching_technologies(h)]
        if unmatched:
            warnings.append(f"{len(unmatched)} hotspot(s) have no matching abatement actions")
            outputs["unmatched_hotspots"] = [h.source_label for h in unmatched]

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Action identification: %d actions matched", len(self._matched_actions))
        return PhaseResult(
            phase_name="action_identification",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _find_matching_technologies(self, hotspot: EmissionSource) -> List[Dict[str, Any]]:
        """Find technologies from catalog applicable to a hotspot."""
        matches = []
        for tech in ABATEMENT_CATALOG:
            scope_match = hotspot.scope in tech["applicable_scopes"]
            source_match = any(
                s in hotspot.category.lower() or hotspot.category.lower() in s
                for s in tech["applicable_sources"]
            )
            if scope_match and source_match:
                matches.append(tech)
        return matches

    # -------------------------------------------------------------------------
    # Phase 3: Cost Analysis
    # -------------------------------------------------------------------------

    async def _phase_cost_analysis(self, config: ReductionPlanningConfig) -> PhaseResult:
        """Calculate cost/tCO2e, NPV, IRR, payback for each action."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        discount_rate = config.discount_rate_pct / 100.0

        for action in self._matched_actions:
            # Annual savings = reduction * cost_per_tco2e (negative cost = savings)
            annual_saving = action.reduction_tco2e * abs(action.cost_per_tco2e_usd) if action.cost_per_tco2e_usd < 0 else 0.0
            annual_cost = action.reduction_tco2e * action.cost_per_tco2e_usd if action.cost_per_tco2e_usd > 0 else 0.0
            action.annual_opex_savings_usd = round(annual_saving, 2)

            # NPV calculation
            npv = -action.total_capex_usd
            annual_cf = annual_saving - annual_cost
            for year in range(1, action.lifetime_years + 1):
                npv += annual_cf / ((1 + discount_rate) ** year)
            action.npv_usd = round(npv, 2)

            # IRR approximation (bisection method)
            action.irr_pct = round(self._calc_irr(action.total_capex_usd, annual_cf, action.lifetime_years), 2)

            # Payback recalculation
            if annual_cf > 0 and action.total_capex_usd > 0:
                action.payback_years = round(action.total_capex_usd / annual_cf, 1)
            elif action.total_capex_usd <= 0:
                action.payback_years = 0.0

        net_negative = [a for a in self._matched_actions if a.cost_per_tco2e_usd < 0]
        outputs["actions_with_npv"] = len(self._matched_actions)
        outputs["net_negative_cost_actions"] = len(net_negative)
        total_npv = sum(a.npv_usd for a in self._matched_actions)
        outputs["total_portfolio_npv_usd"] = round(total_npv, 2)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Cost analysis complete: portfolio NPV=%.0f USD", total_npv)
        return PhaseResult(
            phase_name="cost_analysis",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _calc_irr(self, capex: float, annual_cf: float, years: int) -> float:
        """Calculate IRR using bisection method.  Returns percentage."""
        if capex <= 0 or annual_cf <= 0:
            return 0.0

        def npv_at_rate(rate: float) -> float:
            val = -capex
            for y in range(1, years + 1):
                val += annual_cf / ((1 + rate) ** y)
            return val

        lo, hi = -0.5, 5.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if npv_at_rate(mid) > 0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 0.0001:
                break
        return mid * 100.0

    # -------------------------------------------------------------------------
    # Phase 4: Prioritization
    # -------------------------------------------------------------------------

    async def _phase_prioritization(self, config: ReductionPlanningConfig) -> PhaseResult:
        """Rank actions by cost-effectiveness and apply budget constraints."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Score each action: lower cost/tCO2e + higher feasibility = better
        feasibility_weights = {"high": 1.0, "medium": 0.7, "low": 0.4}
        for action in self._matched_actions:
            feas_w = feasibility_weights.get(action.feasibility, 0.5)
            # Normalised score: negative cost is best; add feasibility bonus
            action.priority_rank = 0  # Will be set after sorting
            action._sort_key = action.cost_per_tco2e_usd - (feas_w * 50)  # type: ignore[attr-defined]

        # Sort by cost-effectiveness (lowest cost_per_tco2e first)
        self._matched_actions.sort(key=lambda a: getattr(a, "_sort_key", a.cost_per_tco2e_usd))

        # Apply budget constraint
        if config.budget_constraint_usd is not None:
            budget_remaining = config.budget_constraint_usd
            selected: List[AbatementAction] = []
            for action in self._matched_actions:
                if action.total_capex_usd <= budget_remaining:
                    selected.append(action)
                    budget_remaining -= action.total_capex_usd
                elif len(selected) < config.max_actions:
                    warnings.append(
                        f"Action {action.action_id} ({action.name}) exceeds remaining budget"
                    )
            self._matched_actions = selected[:config.max_actions]
        else:
            self._matched_actions = self._matched_actions[:config.max_actions]

        # Assign priority ranks and cumulative reduction
        cumulative = 0.0
        for i, action in enumerate(self._matched_actions, 1):
            action.priority_rank = i
            cumulative += action.reduction_tco2e
            action.cumulative_reduction_tco2e = round(cumulative, 4)

        # Build MACC data
        self._macc_data = []
        cum_red = 0.0
        for action in self._matched_actions:
            cum_red += action.reduction_tco2e
            self._macc_data.append(MACCDataPoint(
                action_id=action.action_id,
                action_name=action.name,
                cost_per_tco2e_usd=action.cost_per_tco2e_usd,
                reduction_tco2e=action.reduction_tco2e,
                cumulative_reduction_tco2e=round(cum_red, 4),
            ))

        outputs["prioritized_action_count"] = len(self._matched_actions)
        outputs["total_reduction_tco2e"] = round(cumulative, 4)
        outputs["macc_datapoints"] = len(self._macc_data)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Prioritization: %d actions selected, %.1f tCO2e total reduction",
            len(self._matched_actions), cumulative,
        )
        return PhaseResult(
            phase_name="prioritization",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Roadmap Generation
    # -------------------------------------------------------------------------

    async def _phase_roadmap_generation(self, config: ReductionPlanningConfig) -> PhaseResult:
        """Generate phased implementation roadmap."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        base = config.base_year

        horizon_map = {
            "short_term": {"label": "Short-term (0-2 years)", "start": base, "end": base + 2},
            "medium_term": {"label": "Medium-term (3-5 years)", "start": base + 3, "end": base + 5},
            "long_term": {"label": "Long-term (6-10 years)", "start": base + 6, "end": base + 10},
        }

        phases: List[RoadmapPhase] = []
        cumulative_reduction = 0.0

        for horizon_key in ["short_term", "medium_term", "long_term"]:
            info = horizon_map[horizon_key]
            horizon_actions = [a for a in self._matched_actions if a.time_horizon == horizon_key]
            phase_capex = sum(a.total_capex_usd for a in horizon_actions)
            phase_reduction = sum(a.reduction_tco2e for a in horizon_actions)
            cumulative_reduction += phase_reduction

            phases.append(RoadmapPhase(
                phase_label=info["label"],
                time_horizon=horizon_key,
                year_start=info["start"],
                year_end=info["end"],
                actions=[a.action_id for a in horizon_actions],
                total_capex_usd=round(phase_capex, 2),
                total_reduction_tco2e=round(phase_reduction, 4),
                cumulative_reduction_tco2e=round(cumulative_reduction, 4),
            ))

        self._roadmap = phases

        outputs["roadmap_phases"] = len(phases)
        for rp in phases:
            outputs[f"{rp.time_horizon}_actions"] = len(rp.actions)
            outputs[f"{rp.time_horizon}_capex_usd"] = rp.total_capex_usd
            outputs[f"{rp.time_horizon}_reduction_tco2e"] = rp.total_reduction_tco2e

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Roadmap: %d phases generated", len(phases))
        return PhaseResult(
            phase_name="roadmap_generation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
