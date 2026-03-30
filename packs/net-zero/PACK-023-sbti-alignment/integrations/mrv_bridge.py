# -*- coding: utf-8 -*-
"""
SBTiMRVBridge - Bridge to 30 MRV Agents for SBTi GHG Inventory (PACK-023)
============================================================================

This module routes emissions calculation requests to the appropriate MRV
(Monitoring, Reporting, Verification) agents across all three GHG Protocol
scopes, formatting outputs for SBTi alignment validation. It provides the
calculation backbone for the SBTi inventory phase, feeding data into
target setting, pathway calculation, Scope 3 screening, FLAG assessment,
and progress tracking engines.

Routing Table (30 agents):
    Scope 1 (8 agents):
        Stationary Combustion  --> MRV-001
        Refrigerants & F-Gas   --> MRV-002
        Mobile Combustion      --> MRV-003
        Process Emissions      --> MRV-004
        Fugitive Emissions     --> MRV-005
        Land Use Emissions     --> MRV-006
        Waste Treatment        --> MRV-007
        Agricultural Emissions --> MRV-008
    Scope 2 (5 agents):
        Location-Based         --> MRV-009
        Market-Based           --> MRV-010
        Steam/Heat Purchase    --> MRV-011
        Cooling Purchase       --> MRV-012
        Dual Reporting Recon   --> MRV-013
    Scope 3 (15 category agents):
        Categories 1-15        --> MRV-014..028
    Cross-Cutting (2 agents):
        Category Mapper        --> MRV-029
        Audit Trail & Lineage  --> MRV-030

SBTi-Specific Mapping:
    - Routes MRV outputs to SBTi target boundary definitions
    - Maps Scope 3 category outputs to 40% materiality screening
    - Routes FLAG-relevant sources (MRV-006, MRV-008) for 20% threshold
    - Tracks location- vs. market-based Scope 2 for dual reporting
    - Aggregates by base year for recalculation triggers
    - SHA-256 provenance on all routing operations

Features:
    - Route calculation requests to correct MRV agent
    - Graceful degradation with _AgentStub when agents not importable
    - SBTi format mapping on all outputs
    - Batch routing for multi-source portfolios
    - Scope 3 screening data collection
    - FLAG emissions extraction
    - Base year inventory aggregation
    - Agent status checking

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
from typing import Any, Dict, List, Optional, Set, Tuple

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
    """Stub for unavailable MRV agent modules.

    Returns informative defaults when MRV agents are not installed,
    allowing PACK-023 to operate in standalone mode with degraded
    calculation capability.
    """

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
                "emissions_tco2e": 0.0,
            }
        return _stub_method

def _try_import_mrv_agent(agent_id: str, module_path: str) -> Any:
    """Try to import an MRV agent with graceful fallback.

    Args:
        agent_id: Agent identifier (e.g., 'MRV-001').
        module_path: Python module path for the agent.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("MRV agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmissionSource(str, Enum):
    """Emission source categories mapped to MRV agents."""

    # Scope 1
    STATIONARY_COMBUSTION = "stationary_combustion"
    REFRIGERANTS = "refrigerants"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"
    # Scope 2
    ELECTRICITY_LOCATION = "electricity_location"
    ELECTRICITY_MARKET = "electricity_market"
    STEAM_HEAT = "steam_heat"
    COOLING = "cooling"
    DUAL_REPORTING = "dual_reporting"
    # Scope 3
    PURCHASED_GOODS = "purchased_goods"
    CAPITAL_GOODS = "capital_goods"
    FUEL_ENERGY_ACTIVITIES = "fuel_energy_activities"
    UPSTREAM_TRANSPORT = "upstream_transport"
    WASTE_GENERATED = "waste_generated"
    BUSINESS_TRAVEL = "business_travel"
    EMPLOYEE_COMMUTING = "employee_commuting"
    UPSTREAM_LEASED = "upstream_leased"
    DOWNSTREAM_TRANSPORT = "downstream_transport"
    PROCESSING_SOLD = "processing_sold"
    USE_OF_SOLD = "use_of_sold"
    END_OF_LIFE = "end_of_life"
    DOWNSTREAM_LEASED = "downstream_leased"
    FRANCHISES = "franchises"
    INVESTMENTS = "investments"
    # Cross-cutting
    SCOPE3_MAPPER = "scope3_mapper"
    AUDIT_TRAIL = "audit_trail"

class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    CROSS_CUTTING = "cross_cutting"

class SBTiTargetBoundary(str, Enum):
    """SBTi target boundary classifications."""

    SCOPE_1_2 = "scope_1_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2_3 = "scope_1_2_3"
    FLAG = "flag"
    FI_PORTFOLIO = "fi_portfolio"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MRVAgentRoute(BaseModel):
    """Routing entry mapping an emission source to an MRV agent."""

    source: EmissionSource = Field(...)
    mrv_agent_id: str = Field(..., description="MRV agent identifier (e.g., MRV-001)")
    mrv_agent_name: str = Field(default="", description="Human-readable agent name")
    scope: MRVScope = Field(...)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    module_path: str = Field(default="", description="Python module path")
    description: str = Field(default="")
    sbti_boundary: SBTiTargetBoundary = Field(
        default=SBTiTargetBoundary.SCOPE_1_2,
        description="SBTi target boundary this source maps to",
    )
    flag_relevant: bool = Field(
        default=False,
        description="Whether this source contributes to FLAG emissions",
    )

class RoutingResult(BaseModel):
    """Result of routing a calculation request to an MRV agent."""

    routing_id: str = Field(default_factory=_new_uuid)
    source: str = Field(default="")
    mrv_agent_id: str = Field(default="")
    scope: str = Field(default="")
    scope3_category: Optional[int] = Field(None)
    sbti_boundary: str = Field(default="scope_1_2")
    flag_relevant: bool = Field(default=False)
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    emissions_tco2e: float = Field(default=0.0)
    calculation_details: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class MRVBridgeConfig(BaseModel):
    """Configuration for the SBTi MRV Bridge."""

    pack_id: str = Field(default="PACK-023")
    enable_provenance: bool = Field(default=True)
    enable_batch_routing: bool = Field(default=True)
    max_concurrent_agents: int = Field(default=10, ge=1, le=30)
    scopes_included: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2", "scope_3"],
    )
    base_year: int = Field(default=2019, ge=2015, le=2025)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    flag_threshold_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    scope3_materiality_threshold_pct: float = Field(default=40.0, ge=0.0, le=100.0)

class BatchRoutingResult(BaseModel):
    """Result of routing multiple calculation requests."""

    batch_id: str = Field(default_factory=_new_uuid)
    total_sources: int = Field(default=0)
    successful: int = Field(default=0)
    degraded: int = Field(default=0)
    failed: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_location_tco2e: float = Field(default=0.0)
    scope2_market_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    flag_emissions_tco2e: float = Field(default=0.0)
    flag_emissions_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    flag_triggered: bool = Field(default=False)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    scope3_materiality_triggered: bool = Field(default=False)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    results: List[RoutingResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class SBTiInventoryResult(BaseModel):
    """SBTi-formatted GHG inventory result."""

    inventory_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    base_year: int = Field(default=2019)
    reporting_year: int = Field(default=2025)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    flag_emissions_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    flag_triggered: bool = Field(default=False)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    scope3_materiality_triggered: bool = Field(default=False)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    scope3_material_categories: List[int] = Field(default_factory=list)
    scope3_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    sources_count: int = Field(default=0)
    agents_used: int = Field(default=0)
    agents_degraded: int = Field(default=0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class Scope3ScreeningResult(BaseModel):
    """Scope 3 screening result for SBTi 40% materiality assessment."""

    screening_id: str = Field(default_factory=_new_uuid)
    categories_screened: int = Field(default=15)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    materiality_threshold_pct: float = Field(default=40.0)
    materiality_triggered: bool = Field(default=False)
    category_results: List[Dict[str, Any]] = Field(default_factory=list)
    material_categories: List[int] = Field(default_factory=list)
    near_term_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    near_term_coverage_required_pct: float = Field(default=67.0)
    long_term_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    long_term_coverage_required_pct: float = Field(default=90.0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class FLAGEmissionsResult(BaseModel):
    """FLAG emissions extraction result for SBTi 20% threshold."""

    flag_id: str = Field(default_factory=_new_uuid)
    land_use_tco2e: float = Field(default=0.0, ge=0.0)
    agricultural_tco2e: float = Field(default=0.0, ge=0.0)
    total_flag_tco2e: float = Field(default=0.0, ge=0.0)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    flag_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    threshold_pct: float = Field(default=20.0)
    flag_triggered: bool = Field(default=False)
    sources: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# MRV Agent Routing Table (30 agents)
# ---------------------------------------------------------------------------

MRV_ROUTING_TABLE: List[MRVAgentRoute] = [
    # Scope 1 (8 agents)
    MRVAgentRoute(
        source=EmissionSource.STATIONARY_COMBUSTION, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Boilers, furnaces, heaters, turbines",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.REFRIGERANTS, mrv_agent_id="MRV-002",
        mrv_agent_name="Refrigerants & F-Gas", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.refrigerants_fgas",
        description="HFC/HCFC/PFC refrigerant leakage",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.MOBILE_COMBUSTION, mrv_agent_id="MRV-003",
        mrv_agent_name="Mobile Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.mobile_combustion",
        description="Company vehicles, fleet, forklifts",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.PROCESS_EMISSIONS, mrv_agent_id="MRV-004",
        mrv_agent_name="Process Emissions", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.process_emissions",
        description="Chemical/physical process emissions",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.FUGITIVE_EMISSIONS, mrv_agent_id="MRV-005",
        mrv_agent_name="Fugitive Emissions", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.fugitive_emissions",
        description="Intentional/unintentional releases",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.LAND_USE, mrv_agent_id="MRV-006",
        mrv_agent_name="Land Use Emissions", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.land_use_emissions",
        description="Land use change emissions (FLAG-relevant)",
        sbti_boundary=SBTiTargetBoundary.FLAG,
        flag_relevant=True,
    ),
    MRVAgentRoute(
        source=EmissionSource.WASTE_TREATMENT, mrv_agent_id="MRV-007",
        mrv_agent_name="Waste Treatment Emissions", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.waste_treatment_emissions",
        description="On-site waste treatment",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.AGRICULTURAL, mrv_agent_id="MRV-008",
        mrv_agent_name="Agricultural Emissions", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.agricultural_emissions",
        description="Enteric fermentation, manure, soils (FLAG-relevant)",
        sbti_boundary=SBTiTargetBoundary.FLAG,
        flag_relevant=True,
    ),
    # Scope 2 (5 agents)
    MRVAgentRoute(
        source=EmissionSource.ELECTRICITY_LOCATION, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_location_based",
        description="Grid-average emission factors",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.ELECTRICITY_MARKET, mrv_agent_id="MRV-010",
        mrv_agent_name="Scope 2 Market-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_market_based",
        description="Contractual/residual emission factors",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.STEAM_HEAT, mrv_agent_id="MRV-011",
        mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.steam_heat_purchase",
        description="Purchased steam and district heating",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.COOLING, mrv_agent_id="MRV-012",
        mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.cooling_purchase",
        description="Purchased cooling",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    MRVAgentRoute(
        source=EmissionSource.DUAL_REPORTING, mrv_agent_id="MRV-013",
        mrv_agent_name="Dual Reporting Reconciliation", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.dual_reporting_reconciliation",
        description="Location vs market-based reconciliation",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
    # Scope 3 (15 category agents)
    MRVAgentRoute(
        source=EmissionSource.PURCHASED_GOODS, mrv_agent_id="MRV-014",
        mrv_agent_name="Purchased Goods & Services (Cat 1)", scope=MRVScope.SCOPE_3,
        scope3_category=1,
        module_path="greenlang.agents.mrv.scope3_cat1",
        description="Upstream emissions from purchased goods and services",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.CAPITAL_GOODS, mrv_agent_id="MRV-015",
        mrv_agent_name="Capital Goods (Cat 2)", scope=MRVScope.SCOPE_3,
        scope3_category=2,
        module_path="greenlang.agents.mrv.scope3_cat2",
        description="Upstream emissions from capital goods",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.FUEL_ENERGY_ACTIVITIES, mrv_agent_id="MRV-016",
        mrv_agent_name="Fuel & Energy Activities (Cat 3)", scope=MRVScope.SCOPE_3,
        scope3_category=3,
        module_path="greenlang.agents.mrv.scope3_cat3",
        description="Upstream energy-related emissions not in Scope 1/2",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.UPSTREAM_TRANSPORT, mrv_agent_id="MRV-017",
        mrv_agent_name="Upstream Transportation (Cat 4)", scope=MRVScope.SCOPE_3,
        scope3_category=4,
        module_path="greenlang.agents.mrv.scope3_cat4",
        description="Inbound logistics and distribution",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.WASTE_GENERATED, mrv_agent_id="MRV-018",
        mrv_agent_name="Waste Generated (Cat 5)", scope=MRVScope.SCOPE_3,
        scope3_category=5,
        module_path="greenlang.agents.mrv.scope3_cat5",
        description="Third-party waste disposal",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.BUSINESS_TRAVEL, mrv_agent_id="MRV-019",
        mrv_agent_name="Business Travel (Cat 6)", scope=MRVScope.SCOPE_3,
        scope3_category=6,
        module_path="greenlang.agents.mrv.scope3_cat6",
        description="Employee business travel",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.EMPLOYEE_COMMUTING, mrv_agent_id="MRV-020",
        mrv_agent_name="Employee Commuting (Cat 7)", scope=MRVScope.SCOPE_3,
        scope3_category=7,
        module_path="greenlang.agents.mrv.scope3_cat7",
        description="Employee commuting and remote work",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.UPSTREAM_LEASED, mrv_agent_id="MRV-021",
        mrv_agent_name="Upstream Leased Assets (Cat 8)", scope=MRVScope.SCOPE_3,
        scope3_category=8,
        module_path="greenlang.agents.mrv.scope3_cat8",
        description="Leased assets upstream",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.DOWNSTREAM_TRANSPORT, mrv_agent_id="MRV-022",
        mrv_agent_name="Downstream Transportation (Cat 9)", scope=MRVScope.SCOPE_3,
        scope3_category=9,
        module_path="greenlang.agents.mrv.scope3_cat9",
        description="Outbound logistics and distribution",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.PROCESSING_SOLD, mrv_agent_id="MRV-023",
        mrv_agent_name="Processing of Sold Products (Cat 10)", scope=MRVScope.SCOPE_3,
        scope3_category=10,
        module_path="greenlang.agents.mrv.scope3_cat10",
        description="Downstream processing of sold products",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.USE_OF_SOLD, mrv_agent_id="MRV-024",
        mrv_agent_name="Use of Sold Products (Cat 11)", scope=MRVScope.SCOPE_3,
        scope3_category=11,
        module_path="greenlang.agents.mrv.scope3_cat11",
        description="Customer use-phase emissions",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.END_OF_LIFE, mrv_agent_id="MRV-025",
        mrv_agent_name="End-of-Life Treatment (Cat 12)", scope=MRVScope.SCOPE_3,
        scope3_category=12,
        module_path="greenlang.agents.mrv.scope3_cat12",
        description="End-of-life treatment of sold products",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.DOWNSTREAM_LEASED, mrv_agent_id="MRV-026",
        mrv_agent_name="Downstream Leased Assets (Cat 13)", scope=MRVScope.SCOPE_3,
        scope3_category=13,
        module_path="greenlang.agents.mrv.scope3_cat13",
        description="Leased assets downstream",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.FRANCHISES, mrv_agent_id="MRV-027",
        mrv_agent_name="Franchises (Cat 14)", scope=MRVScope.SCOPE_3,
        scope3_category=14,
        module_path="greenlang.agents.mrv.scope3_cat14",
        description="Franchise operations",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.INVESTMENTS, mrv_agent_id="MRV-028",
        mrv_agent_name="Investments (Cat 15)", scope=MRVScope.SCOPE_3,
        scope3_category=15,
        module_path="greenlang.agents.mrv.scope3_cat15",
        description="Financed emissions from investments",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    # Cross-cutting (2 agents)
    MRVAgentRoute(
        source=EmissionSource.SCOPE3_MAPPER, mrv_agent_id="MRV-029",
        mrv_agent_name="Scope 3 Category Mapper", scope=MRVScope.CROSS_CUTTING,
        module_path="greenlang.agents.mrv.scope3_category_mapper",
        description="Maps activities to Scope 3 categories",
        sbti_boundary=SBTiTargetBoundary.SCOPE_3,
    ),
    MRVAgentRoute(
        source=EmissionSource.AUDIT_TRAIL, mrv_agent_id="MRV-030",
        mrv_agent_name="Audit Trail & Lineage", scope=MRVScope.CROSS_CUTTING,
        module_path="greenlang.agents.mrv.audit_trail_lineage",
        description="Provenance and audit trail tracking",
        sbti_boundary=SBTiTargetBoundary.SCOPE_1_2,
    ),
]

# Scope 3 category names for screening
SCOPE3_CATEGORY_NAMES: Dict[int, str] = {
    1: "Purchased Goods & Services",
    2: "Capital Goods",
    3: "Fuel & Energy Activities",
    4: "Upstream Transportation & Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation & Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

# FLAG-relevant emission source IDs
FLAG_RELEVANT_SOURCES: Set[EmissionSource] = {
    EmissionSource.LAND_USE,
    EmissionSource.AGRICULTURAL,
}

# ---------------------------------------------------------------------------
# SBTiMRVBridge
# ---------------------------------------------------------------------------

class SBTiMRVBridge:
    """Bridge to 30 MRV agents for SBTi GHG inventory and screening.

    Routes emission source calculation requests to the appropriate MRV
    agent across Scope 1, 2, and 3. Formats outputs for SBTi alignment
    including target boundary mapping, Scope 3 materiality screening,
    FLAG emissions extraction, and dual Scope 2 reporting.

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded MRV agent modules/stubs.
        _routing_table: Active routing table.

    Example:
        >>> bridge = SBTiMRVBridge()
        >>> result = bridge.route_calculation(
        ...     EmissionSource.STATIONARY_COMBUSTION,
        ...     {"fuel_type": "natural_gas", "consumption_m3": 10000}
        ... )
        >>> print(f"Emissions: {result.emissions_tco2e} tCO2e")
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize the SBTi MRV Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or MRVBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._routing_table = list(MRV_ROUTING_TABLE)

        # Load MRV agents with graceful fallback
        self._agents: Dict[str, Any] = {}
        unique_agents = {r.mrv_agent_id: r.module_path for r in self._routing_table}
        for agent_id, module_path in unique_agents.items():
            self._agents[agent_id] = _try_import_mrv_agent(agent_id, module_path)

        available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        self.logger.info(
            "SBTiMRVBridge initialized: %d/%d agents available, pack=%s",
            available, len(self._agents), self.config.pack_id,
        )

    # -------------------------------------------------------------------------
    # Single Source Routing
    # -------------------------------------------------------------------------

    def route_calculation(
        self,
        source: EmissionSource,
        data: Dict[str, Any],
    ) -> RoutingResult:
        """Route a calculation request to the appropriate MRV agent.

        Args:
            source: Emission source category.
            data: Input data for the calculation.

        Returns:
            RoutingResult with calculation output or degraded status.
        """
        start = time.monotonic()

        route = self._find_route(source)
        if route is None:
            return RoutingResult(
                source=source.value,
                success=False,
                message=f"No routing entry for source '{source.value}'",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        agent = self._agents.get(route.mrv_agent_id)
        if agent is None or isinstance(agent, _AgentStub):
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                sbti_boundary=route.sbti_boundary.value,
                flag_relevant=route.flag_relevant,
                success=False,
                degraded=True,
                message=f"MRV agent {route.mrv_agent_id} not available (stub mode)",
                duration_ms=(time.monotonic() - start) * 1000,
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)
            return result

        try:
            agent_result = agent.calculate(data)
            emissions = agent_result.get("emissions_tco2e", 0.0) if isinstance(agent_result, dict) else 0.0

            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                sbti_boundary=route.sbti_boundary.value,
                flag_relevant=route.flag_relevant,
                success=True,
                emissions_tco2e=emissions,
                calculation_details=agent_result if isinstance(agent_result, dict) else {},
                message="Calculation completed",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            self.logger.error("MRV agent %s calculation failed: %s", route.mrv_agent_id, exc)
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                sbti_boundary=route.sbti_boundary.value,
                flag_relevant=route.flag_relevant,
                success=False,
                message=f"Calculation error: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Batch Routing
    # -------------------------------------------------------------------------

    def route_batch(
        self,
        sources: List[Tuple[EmissionSource, Dict[str, Any]]],
    ) -> BatchRoutingResult:
        """Route multiple calculation requests and aggregate for SBTi.

        Args:
            sources: List of (EmissionSource, data) tuples.

        Returns:
            BatchRoutingResult with SBTi-formatted aggregations.
        """
        start = time.monotonic()
        batch = BatchRoutingResult(total_sources=len(sources))

        for source, data in sources:
            routing_result = self.route_calculation(source, data)
            batch.results.append(routing_result)

            if routing_result.success:
                batch.successful += 1
                batch.total_emissions_tco2e += routing_result.emissions_tco2e
                if routing_result.scope == MRVScope.SCOPE_1.value:
                    batch.scope1_tco2e += routing_result.emissions_tco2e
                elif routing_result.scope == MRVScope.SCOPE_2.value:
                    if routing_result.source == EmissionSource.ELECTRICITY_LOCATION.value:
                        batch.scope2_location_tco2e += routing_result.emissions_tco2e
                    elif routing_result.source == EmissionSource.ELECTRICITY_MARKET.value:
                        batch.scope2_market_tco2e += routing_result.emissions_tco2e
                    else:
                        batch.scope2_location_tco2e += routing_result.emissions_tco2e
                elif routing_result.scope == MRVScope.SCOPE_3.value:
                    batch.scope3_tco2e += routing_result.emissions_tco2e
                    if routing_result.scope3_category:
                        cat = routing_result.scope3_category
                        batch.scope3_by_category[cat] = (
                            batch.scope3_by_category.get(cat, 0.0) + routing_result.emissions_tco2e
                        )
                if routing_result.flag_relevant:
                    batch.flag_emissions_tco2e += routing_result.emissions_tco2e
            elif routing_result.degraded:
                batch.degraded += 1
            else:
                batch.failed += 1

        # SBTi-specific calculations
        if batch.total_emissions_tco2e > 0:
            batch.flag_emissions_pct = round(
                batch.flag_emissions_tco2e / batch.total_emissions_tco2e * 100.0, 2
            )
            batch.scope3_pct_of_total = round(
                batch.scope3_tco2e / batch.total_emissions_tco2e * 100.0, 2
            )

        batch.flag_triggered = batch.flag_emissions_pct >= self.config.flag_threshold_pct
        batch.scope3_materiality_triggered = (
            batch.scope3_pct_of_total >= self.config.scope3_materiality_threshold_pct
        )
        batch.duration_ms = (time.monotonic() - start) * 1000

        if self.config.enable_provenance:
            batch.provenance_hash = _compute_hash(batch)
        return batch

    # -------------------------------------------------------------------------
    # SBTi Inventory Compilation
    # -------------------------------------------------------------------------

    def compile_sbti_inventory(
        self,
        sources: List[Tuple[EmissionSource, Dict[str, Any]]],
        organization_name: str = "",
    ) -> SBTiInventoryResult:
        """Compile a full SBTi-formatted GHG inventory from MRV agents.

        Args:
            sources: List of (EmissionSource, data) tuples.
            organization_name: Organization name for the inventory.

        Returns:
            SBTiInventoryResult with full SBTi-formatted inventory.
        """
        start = time.monotonic()
        batch = self.route_batch(sources)

        # Determine material Scope 3 categories (>1% of total each)
        material_categories: List[int] = []
        if batch.total_emissions_tco2e > 0:
            for cat, emissions in sorted(batch.scope3_by_category.items()):
                cat_pct = emissions / batch.total_emissions_tco2e * 100.0
                if cat_pct >= 1.0:
                    material_categories.append(cat)

        # Coverage calculation: categories with data / 15
        cats_with_data = len([c for c in batch.scope3_by_category if batch.scope3_by_category[c] > 0])
        coverage_pct = round(cats_with_data / 15.0 * 100.0, 1) if cats_with_data > 0 else 0.0

        result = SBTiInventoryResult(
            organization_name=organization_name,
            base_year=self.config.base_year,
            reporting_year=self.config.reporting_year,
            scope1_tco2e=round(batch.scope1_tco2e, 2),
            scope2_location_tco2e=round(batch.scope2_location_tco2e, 2),
            scope2_market_tco2e=round(batch.scope2_market_tco2e, 2),
            scope3_tco2e=round(batch.scope3_tco2e, 2),
            total_tco2e=round(batch.total_emissions_tco2e, 2),
            flag_emissions_tco2e=round(batch.flag_emissions_tco2e, 2),
            flag_emissions_pct=batch.flag_emissions_pct,
            flag_triggered=batch.flag_triggered,
            scope3_pct_of_total=batch.scope3_pct_of_total,
            scope3_materiality_triggered=batch.scope3_materiality_triggered,
            scope3_by_category=batch.scope3_by_category,
            scope3_material_categories=material_categories,
            scope3_coverage_pct=coverage_pct,
            data_quality_score=self._assess_data_quality(batch),
            sources_count=batch.total_sources,
            agents_used=batch.successful,
            agents_degraded=batch.degraded,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Scope 3 Screening
    # -------------------------------------------------------------------------

    def screen_scope3(
        self,
        scope3_data: Dict[int, Dict[str, Any]],
        total_emissions_tco2e: float = 0.0,
    ) -> Scope3ScreeningResult:
        """Screen all 15 Scope 3 categories for SBTi materiality.

        Args:
            scope3_data: Dict mapping category number (1-15) to input data.
            total_emissions_tco2e: Total company emissions for percentage calc.

        Returns:
            Scope3ScreeningResult with materiality assessment.
        """
        start = time.monotonic()
        scope3_sources = [
            EmissionSource.PURCHASED_GOODS, EmissionSource.CAPITAL_GOODS,
            EmissionSource.FUEL_ENERGY_ACTIVITIES, EmissionSource.UPSTREAM_TRANSPORT,
            EmissionSource.WASTE_GENERATED, EmissionSource.BUSINESS_TRAVEL,
            EmissionSource.EMPLOYEE_COMMUTING, EmissionSource.UPSTREAM_LEASED,
            EmissionSource.DOWNSTREAM_TRANSPORT, EmissionSource.PROCESSING_SOLD,
            EmissionSource.USE_OF_SOLD, EmissionSource.END_OF_LIFE,
            EmissionSource.DOWNSTREAM_LEASED, EmissionSource.FRANCHISES,
            EmissionSource.INVESTMENTS,
        ]

        category_results: List[Dict[str, Any]] = []
        scope3_total = 0.0
        material_cats: List[int] = []

        for idx, source in enumerate(scope3_sources, start=1):
            cat_data = scope3_data.get(idx, {})
            routing = self.route_calculation(source, cat_data)
            cat_emissions = routing.emissions_tco2e
            scope3_total += cat_emissions

            cat_result = {
                "category": idx,
                "name": SCOPE3_CATEGORY_NAMES.get(idx, f"Category {idx}"),
                "emissions_tco2e": round(cat_emissions, 2),
                "pct_of_scope3": 0.0,
                "pct_of_total": 0.0,
                "material": False,
                "data_available": bool(cat_data),
                "agent_id": routing.mrv_agent_id,
                "status": "calculated" if routing.success else ("degraded" if routing.degraded else "failed"),
            }
            category_results.append(cat_result)

        # Post-process percentages and materiality
        for cat_result in category_results:
            if scope3_total > 0:
                cat_result["pct_of_scope3"] = round(
                    cat_result["emissions_tco2e"] / scope3_total * 100.0, 2
                )
            if total_emissions_tco2e > 0:
                cat_result["pct_of_total"] = round(
                    cat_result["emissions_tco2e"] / total_emissions_tco2e * 100.0, 2
                )
            if cat_result["pct_of_total"] >= 1.0:
                cat_result["material"] = True
                material_cats.append(cat_result["category"])

        scope3_pct = round(scope3_total / total_emissions_tco2e * 100.0, 2) if total_emissions_tco2e > 0 else 0.0
        materiality_triggered = scope3_pct >= self.config.scope3_materiality_threshold_pct

        # Coverage: material categories with data
        cats_with_data = sum(1 for c in category_results if c["data_available"] and c["emissions_tco2e"] > 0)
        material_with_data = sum(1 for c in category_results if c["material"] and c["emissions_tco2e"] > 0)
        near_term_coverage = round(material_with_data / len(material_cats) * 100.0, 1) if material_cats else 0.0
        long_term_coverage = round(cats_with_data / 15 * 100.0, 1)

        result = Scope3ScreeningResult(
            scope3_total_tco2e=round(scope3_total, 2),
            scope3_pct_of_total=scope3_pct,
            materiality_triggered=materiality_triggered,
            category_results=category_results,
            material_categories=material_cats,
            near_term_coverage_pct=near_term_coverage,
            long_term_coverage_pct=long_term_coverage,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # FLAG Emissions Extraction
    # -------------------------------------------------------------------------

    def extract_flag_emissions(
        self,
        land_use_data: Optional[Dict[str, Any]] = None,
        agricultural_data: Optional[Dict[str, Any]] = None,
        total_emissions_tco2e: float = 0.0,
    ) -> FLAGEmissionsResult:
        """Extract FLAG-relevant emissions for SBTi 20% threshold check.

        Args:
            land_use_data: Input data for land use emissions (MRV-006).
            agricultural_data: Input data for agricultural emissions (MRV-008).
            total_emissions_tco2e: Total company emissions for percentage calc.

        Returns:
            FLAGEmissionsResult with threshold assessment.
        """
        start = time.monotonic()

        lu_result = self.route_calculation(EmissionSource.LAND_USE, land_use_data or {})
        ag_result = self.route_calculation(EmissionSource.AGRICULTURAL, agricultural_data or {})

        land_use_tco2e = lu_result.emissions_tco2e
        agricultural_tco2e = ag_result.emissions_tco2e
        total_flag = land_use_tco2e + agricultural_tco2e

        flag_pct = round(total_flag / total_emissions_tco2e * 100.0, 2) if total_emissions_tco2e > 0 else 0.0

        sources_used: List[str] = []
        if lu_result.success:
            sources_used.append("MRV-006_land_use")
        if ag_result.success:
            sources_used.append("MRV-008_agricultural")

        result = FLAGEmissionsResult(
            land_use_tco2e=round(land_use_tco2e, 2),
            agricultural_tco2e=round(agricultural_tco2e, 2),
            total_flag_tco2e=round(total_flag, 2),
            total_emissions_tco2e=round(total_emissions_tco2e, 2),
            flag_pct_of_total=flag_pct,
            flag_triggered=flag_pct >= self.config.flag_threshold_pct,
            sources=sources_used,
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Agent Status
    # -------------------------------------------------------------------------

    def get_agent_status(self) -> Dict[str, Any]:
        """Get availability status of all 30 MRV agents.

        Returns:
            Dict with agent availability by scope.
        """
        status: Dict[str, List[Dict[str, Any]]] = {
            "scope_1": [], "scope_2": [], "scope_3": [], "cross_cutting": [],
        }

        for route in self._routing_table:
            agent = self._agents.get(route.mrv_agent_id)
            available = agent is not None and not isinstance(agent, _AgentStub)
            entry = {
                "agent_id": route.mrv_agent_id,
                "agent_name": route.mrv_agent_name,
                "source": route.source.value,
                "available": available,
                "sbti_boundary": route.sbti_boundary.value,
                "flag_relevant": route.flag_relevant,
            }
            if route.scope3_category:
                entry["scope3_category"] = route.scope3_category

            scope_key = route.scope.value
            if scope_key in status:
                status[scope_key].append(entry)

        available_count = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        return {
            "pack_id": self.config.pack_id,
            "total_agents": len(self._agents),
            "available": available_count,
            "degraded": len(self._agents) - available_count,
            "agents_by_scope": status,
        }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status summary.

        Returns:
            Dict with bridge status information.
        """
        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        return {
            "pack_id": self.config.pack_id,
            "module_version": _MODULE_VERSION,
            "total_agents": len(self._agents),
            "available_agents": available,
            "base_year": self.config.base_year,
            "reporting_year": self.config.reporting_year,
            "flag_threshold_pct": self.config.flag_threshold_pct,
            "scope3_materiality_threshold_pct": self.config.scope3_materiality_threshold_pct,
            "scopes_included": self.config.scopes_included,
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _find_route(self, source: EmissionSource) -> Optional[MRVAgentRoute]:
        """Find the routing entry for a given emission source.

        Args:
            source: Emission source to look up.

        Returns:
            MRVAgentRoute or None if not found.
        """
        for route in self._routing_table:
            if route.source == source:
                return route
        return None

    def _assess_data_quality(self, batch: BatchRoutingResult) -> float:
        """Assess overall data quality based on agent availability and success.

        Args:
            batch: Batch routing result.

        Returns:
            Data quality score between 0 and 100.
        """
        if batch.total_sources == 0:
            return 0.0

        success_score = (batch.successful / batch.total_sources) * 60.0
        coverage_score = min(len(batch.scope3_by_category) / 15.0 * 30.0, 30.0)
        no_degraded_score = max(0.0, 10.0 - batch.degraded * 1.0)

        return round(min(success_score + coverage_score + no_degraded_score, 100.0), 1)
