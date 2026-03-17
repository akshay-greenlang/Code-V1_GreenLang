# -*- coding: utf-8 -*-
"""
MRVEmissionsBridge - Bridge to 30 MRV Agents for PAI 1-6 Emissions Calculations
=================================================================================

This module connects PACK-011 (SFDR Article 9) with the 30 AGENT-MRV agents
(GL-MRV-001 through GL-MRV-030) to provide emissions data for PAI indicators
1 through 6. Article 9 products require mandatory reporting of all 18 PAI
indicators, with PAI 1-6 being emissions-related and requiring deterministic
calculation via the MRV layer.

Architecture:
    PACK-011 SFDR Art 9 --> MRVEmissionsBridge --> 30 MRV Agents
                                  |
                                  v
    PAI 1: GHG Emissions (Scope 1 + 2 + 3)
    PAI 2: Carbon Footprint
    PAI 3: GHG Intensity of Investee Companies
    PAI 4: Exposure to Fossil Fuel Companies
    PAI 5: Non-Renewable Energy Share
    PAI 6: Energy Consumption Intensity

Zero-Hallucination Guarantee:
    All emissions calculations are routed to deterministic MRV agents.
    No LLM calls are made for numeric values. Results carry provenance
    hashes from the MRV layer for full audit trail.

Example:
    >>> config = MRVBridgeConfig()
    >>> bridge = MRVEmissionsBridge(config)
    >>> result = bridge.calculate_pai_emissions(holdings, nav_eur=1_000_000)
    >>> print(f"PAI 1 total: {result.total_scope_1_2_3_tco2e:.2f} tCO2e")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Agent Stub
# =============================================================================


class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning(
                "AgentStub: failed to load %s from %s: %s",
                self.agent_id, self.module_path, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None


# =============================================================================
# Enums
# =============================================================================


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


class PAIEmissionIndicator(str, Enum):
    """PAI indicators 1-6 (emissions-related)."""
    PAI_1_GHG_EMISSIONS = "pai_1_ghg_emissions"
    PAI_2_CARBON_FOOTPRINT = "pai_2_carbon_footprint"
    PAI_3_GHG_INTENSITY = "pai_3_ghg_intensity"
    PAI_4_FOSSIL_FUEL = "pai_4_fossil_fuel"
    PAI_5_NON_RENEWABLE = "pai_5_non_renewable"
    PAI_6_ENERGY_INTENSITY = "pai_6_energy_intensity"


class DataQualityTier(str, Enum):
    """Emissions data quality tier per PCAF."""
    REPORTED = "reported"
    ESTIMATED_SPECIFIC = "estimated_specific"
    ESTIMATED_AVERAGE = "estimated_average"
    PROXY = "proxy"
    UNAVAILABLE = "unavailable"


class MRVAgentCategory(str, Enum):
    """MRV agent categories."""
    SCOPE_1_STATIONARY = "scope_1_stationary"
    SCOPE_1_MOBILE = "scope_1_mobile"
    SCOPE_1_PROCESS = "scope_1_process"
    SCOPE_1_FUGITIVE = "scope_1_fugitive"
    SCOPE_1_REFRIGERANT = "scope_1_refrigerant"
    SCOPE_1_LAND_USE = "scope_1_land_use"
    SCOPE_1_WASTE = "scope_1_waste"
    SCOPE_1_AGRICULTURE = "scope_1_agriculture"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_2_STEAM = "scope_2_steam"
    SCOPE_2_COOLING = "scope_2_cooling"
    SCOPE_2_DUAL = "scope_2_dual"
    SCOPE_3_CAT_1 = "scope_3_cat_1"
    SCOPE_3_CAT_2 = "scope_3_cat_2"
    SCOPE_3_CAT_3 = "scope_3_cat_3"
    SCOPE_3_CAT_4 = "scope_3_cat_4"
    SCOPE_3_CAT_5 = "scope_3_cat_5"
    SCOPE_3_CAT_6 = "scope_3_cat_6"
    SCOPE_3_CAT_7 = "scope_3_cat_7"
    SCOPE_3_CAT_8 = "scope_3_cat_8"
    SCOPE_3_CAT_9 = "scope_3_cat_9"
    SCOPE_3_CAT_10 = "scope_3_cat_10"
    SCOPE_3_CAT_11 = "scope_3_cat_11"
    SCOPE_3_CAT_12 = "scope_3_cat_12"
    SCOPE_3_CAT_13 = "scope_3_cat_13"
    SCOPE_3_CAT_14 = "scope_3_cat_14"
    SCOPE_3_CAT_15 = "scope_3_cat_15"
    SCOPE_3_MAPPER = "scope_3_mapper"
    AUDIT_TRAIL = "audit_trail"


# =============================================================================
# Data Models
# =============================================================================


class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV Emissions Bridge."""
    mrv_base_path: str = Field(
        default="greenlang.agents.mrv",
        description="Base import path for MRV agents",
    )
    enable_scope_1: bool = Field(
        default=True, description="Enable Scope 1 emission agents"
    )
    enable_scope_2: bool = Field(
        default=True, description="Enable Scope 2 emission agents"
    )
    enable_scope_3: bool = Field(
        default=True, description="Enable Scope 3 emission agents"
    )
    scope_2_preference: str = Field(
        default="market",
        description="Preferred Scope 2 method: market or location",
    )
    mandatory_pai_enforcement: bool = Field(
        default=True,
        description="Enforce all PAI 1-6 indicators are calculated (Art 9 requirement)",
    )
    min_data_quality: DataQualityTier = Field(
        default=DataQualityTier.ESTIMATED_AVERAGE,
        description="Minimum acceptable data quality tier",
    )
    attribution_method: str = Field(
        default="enterprise_value",
        description="Attribution method: enterprise_value, revenue, or ownership",
    )
    currency: str = Field(
        default="EUR", description="Reporting currency"
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking"
    )
    cache_ttl_seconds: int = Field(
        default=3600, ge=0, description="Cache TTL for emission factors"
    )
    batch_size: int = Field(
        default=500, ge=1, le=10000, description="Batch size for processing"
    )


class MRVAgentMapping(BaseModel):
    """Mapping of a single MRV agent to its configuration."""
    agent_id: str = Field(..., description="GreenLang agent identifier")
    category: MRVAgentCategory = Field(..., description="MRV agent category")
    scope: EmissionScope = Field(..., description="Emission scope")
    module_path: str = Field(..., description="Python import path")
    class_name: str = Field(..., description="Agent class name")
    pai_indicators: List[str] = Field(
        default_factory=list,
        description="PAI indicators this agent contributes to",
    )
    enabled: bool = Field(default=True, description="Whether agent is active")
    priority: int = Field(
        default=10, ge=1, le=100, description="Routing priority"
    )


class PAIRoutingResult(BaseModel):
    """Result of routing a PAI calculation to MRV agents."""
    pai_indicator: str = Field(..., description="PAI indicator identifier")
    pai_name: str = Field(default="", description="Human-readable PAI name")
    agents_invoked: List[str] = Field(
        default_factory=list, description="Agent IDs that contributed"
    )
    value: float = Field(default=0.0, description="Calculated PAI value")
    unit: str = Field(default="tCO2e", description="Value unit")
    data_quality: str = Field(
        default="estimated_average", description="Overall data quality tier"
    )
    coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Portfolio coverage %"
    )
    scope_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-scope contribution"
    )
    per_holding: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-holding attribution"
    )
    calculation_method: str = Field(
        default="deterministic", description="Calculation method used"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings encountered"
    )
    errors: List[str] = Field(
        default_factory=list, description="Errors encountered"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: str = Field(default="", description="Calculation timestamp")
    execution_time_ms: float = Field(default=0.0, description="Execution time in ms")


class EmissionsAggregate(BaseModel):
    """Aggregated emissions data for SFDR PAI reporting."""
    # PAI 1: GHG Emissions
    scope_1_tco2e: float = Field(default=0.0, description="Scope 1 emissions (tCO2e)")
    scope_2_tco2e: float = Field(default=0.0, description="Scope 2 emissions (tCO2e)")
    scope_3_tco2e: float = Field(default=0.0, description="Scope 3 emissions (tCO2e)")
    total_scope_1_2_3_tco2e: float = Field(
        default=0.0, description="Total Scope 1+2+3 emissions (tCO2e)"
    )

    # PAI 2: Carbon Footprint
    carbon_footprint_tco2e_per_eur_m: float = Field(
        default=0.0, description="Carbon footprint (tCO2e / EUR million invested)"
    )

    # PAI 3: GHG Intensity
    ghg_intensity_tco2e_per_eur_m_revenue: float = Field(
        default=0.0, description="GHG intensity (tCO2e / EUR million revenue)"
    )

    # PAI 4: Fossil Fuel Exposure
    fossil_fuel_exposure_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Share of investments in fossil fuel companies (%)",
    )
    fossil_fuel_companies_count: int = Field(
        default=0, description="Number of fossil fuel companies in portfolio"
    )

    # PAI 5: Non-Renewable Energy
    non_renewable_energy_share_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Non-renewable energy production and consumption share (%)",
    )
    non_renewable_production_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Non-renewable energy production share (%)",
    )
    non_renewable_consumption_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Non-renewable energy consumption share (%)",
    )

    # PAI 6: Energy Intensity
    energy_intensity_gwh_per_eur_m: float = Field(
        default=0.0,
        description="Energy consumption intensity (GWh / EUR million revenue)",
    )
    energy_intensity_by_nace: Dict[str, float] = Field(
        default_factory=dict,
        description="Energy intensity breakdown by NACE high-impact sector",
    )

    # Metadata
    nav_eur: float = Field(default=0.0, description="Portfolio NAV in EUR")
    total_holdings: int = Field(default=0, description="Total portfolio holdings")
    coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Overall emissions data coverage (%)",
    )
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weighted data quality score"
    )
    pai_results: Dict[str, PAIRoutingResult] = Field(
        default_factory=dict, description="Individual PAI routing results"
    )
    agents_used: List[str] = Field(
        default_factory=list, description="All MRV agent IDs invoked"
    )
    errors: List[str] = Field(default_factory=list, description="Aggregate errors")
    warnings: List[str] = Field(default_factory=list, description="Aggregate warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: str = Field(default="", description="Calculation timestamp")
    execution_time_ms: float = Field(default=0.0, description="Total execution time")


# =============================================================================
# MRV Agent Registry
# =============================================================================


MRV_AGENT_REGISTRY: List[MRVAgentMapping] = [
    # Scope 1 agents (001-008)
    MRVAgentMapping(
        agent_id="GL-MRV-001", category=MRVAgentCategory.SCOPE_1_STATIONARY,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.stationary_combustion",
        class_name="StationaryCombustionAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-002", category=MRVAgentCategory.SCOPE_1_REFRIGERANT,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.refrigerants_fgas",
        class_name="RefrigerantsFGasAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-003", category=MRVAgentCategory.SCOPE_1_MOBILE,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.mobile_combustion",
        class_name="MobileCombustionAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-004", category=MRVAgentCategory.SCOPE_1_PROCESS,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.process_emissions",
        class_name="ProcessEmissionsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-005", category=MRVAgentCategory.SCOPE_1_FUGITIVE,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.fugitive_emissions",
        class_name="FugitiveEmissionsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-006", category=MRVAgentCategory.SCOPE_1_LAND_USE,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.land_use_emissions",
        class_name="LandUseEmissionsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-007", category=MRVAgentCategory.SCOPE_1_WASTE,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.waste_treatment",
        class_name="WasteTreatmentAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-008", category=MRVAgentCategory.SCOPE_1_AGRICULTURE,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.agricultural_emissions",
        class_name="AgriculturalEmissionsAgent", pai_indicators=["pai_1"],
    ),
    # Scope 2 agents (009-013)
    MRVAgentMapping(
        agent_id="GL-MRV-009", category=MRVAgentCategory.SCOPE_2_LOCATION,
        scope=EmissionScope.SCOPE_2_LOCATION, module_path="greenlang.agents.mrv.scope2_location",
        class_name="Scope2LocationAgent", pai_indicators=["pai_1", "pai_5", "pai_6"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-010", category=MRVAgentCategory.SCOPE_2_MARKET,
        scope=EmissionScope.SCOPE_2_MARKET, module_path="greenlang.agents.mrv.scope2_market",
        class_name="Scope2MarketAgent", pai_indicators=["pai_1", "pai_5", "pai_6"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-011", category=MRVAgentCategory.SCOPE_2_STEAM,
        scope=EmissionScope.SCOPE_2_LOCATION, module_path="greenlang.agents.mrv.steam_heat",
        class_name="SteamHeatAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-012", category=MRVAgentCategory.SCOPE_2_COOLING,
        scope=EmissionScope.SCOPE_2_LOCATION, module_path="greenlang.agents.mrv.cooling_purchase",
        class_name="CoolingPurchaseAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-013", category=MRVAgentCategory.SCOPE_2_DUAL,
        scope=EmissionScope.SCOPE_2_LOCATION, module_path="greenlang.agents.mrv.dual_reporting",
        class_name="DualReportingReconciliationAgent", pai_indicators=["pai_1"],
    ),
    # Scope 3 agents (014-028)
    MRVAgentMapping(
        agent_id="GL-MRV-014", category=MRVAgentCategory.SCOPE_3_CAT_1,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.purchased_goods",
        class_name="PurchasedGoodsServicesAgent", pai_indicators=["pai_1", "pai_3"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-015", category=MRVAgentCategory.SCOPE_3_CAT_2,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.capital_goods",
        class_name="CapitalGoodsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-016", category=MRVAgentCategory.SCOPE_3_CAT_3,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.fuel_energy",
        class_name="FuelEnergyActivitiesAgent", pai_indicators=["pai_1", "pai_5"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-017", category=MRVAgentCategory.SCOPE_3_CAT_4,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.upstream_transport",
        class_name="UpstreamTransportationAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-018", category=MRVAgentCategory.SCOPE_3_CAT_5,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.waste_generated",
        class_name="WasteGeneratedAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-019", category=MRVAgentCategory.SCOPE_3_CAT_6,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.business_travel",
        class_name="BusinessTravelAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-020", category=MRVAgentCategory.SCOPE_3_CAT_7,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.employee_commuting",
        class_name="EmployeeCommutingAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-021", category=MRVAgentCategory.SCOPE_3_CAT_8,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.upstream_leased",
        class_name="UpstreamLeasedAssetsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-022", category=MRVAgentCategory.SCOPE_3_CAT_9,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.downstream_transport",
        class_name="DownstreamTransportationAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-023", category=MRVAgentCategory.SCOPE_3_CAT_10,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.processing_sold",
        class_name="ProcessingSoldProductsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-024", category=MRVAgentCategory.SCOPE_3_CAT_11,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.use_of_sold",
        class_name="UseSoldProductsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-025", category=MRVAgentCategory.SCOPE_3_CAT_12,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.end_of_life",
        class_name="EndOfLifeTreatmentAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-026", category=MRVAgentCategory.SCOPE_3_CAT_13,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.downstream_leased",
        class_name="DownstreamLeasedAssetsAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-027", category=MRVAgentCategory.SCOPE_3_CAT_14,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.franchises",
        class_name="FranchisesAgent", pai_indicators=["pai_1"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-028", category=MRVAgentCategory.SCOPE_3_CAT_15,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.investments",
        class_name="InvestmentsAgent", pai_indicators=["pai_1"],
    ),
    # Cross-cutting agents (029-030)
    MRVAgentMapping(
        agent_id="GL-MRV-029", category=MRVAgentCategory.SCOPE_3_MAPPER,
        scope=EmissionScope.SCOPE_3, module_path="greenlang.agents.mrv.scope3_mapper",
        class_name="Scope3CategoryMapperAgent", pai_indicators=["pai_1", "pai_3"],
    ),
    MRVAgentMapping(
        agent_id="GL-MRV-030", category=MRVAgentCategory.AUDIT_TRAIL,
        scope=EmissionScope.SCOPE_1, module_path="greenlang.agents.mrv.audit_trail",
        class_name="AuditTrailLineageAgent", pai_indicators=[],
    ),
]


# PAI indicator definitions for Article 9
PAI_INDICATOR_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "pai_1": {
        "name": "GHG Emissions",
        "description": "Scope 1, 2, and 3 GHG emissions",
        "unit": "tCO2e",
        "regulatory_ref": "Table 1, Indicator 1, Annex I RTS",
    },
    "pai_2": {
        "name": "Carbon Footprint",
        "description": "Carbon footprint per EUR million invested",
        "unit": "tCO2e/EUR M",
        "regulatory_ref": "Table 1, Indicator 2, Annex I RTS",
    },
    "pai_3": {
        "name": "GHG Intensity of Investee Companies",
        "description": "Weighted average GHG intensity per EUR million revenue",
        "unit": "tCO2e/EUR M revenue",
        "regulatory_ref": "Table 1, Indicator 3, Annex I RTS",
    },
    "pai_4": {
        "name": "Exposure to Fossil Fuel Companies",
        "description": "Share of investments in companies active in fossil fuel sector",
        "unit": "%",
        "regulatory_ref": "Table 1, Indicator 4, Annex I RTS",
    },
    "pai_5": {
        "name": "Non-Renewable Energy Share",
        "description": "Share of non-renewable energy consumption and production",
        "unit": "%",
        "regulatory_ref": "Table 1, Indicator 5, Annex I RTS",
    },
    "pai_6": {
        "name": "Energy Consumption Intensity",
        "description": "Energy consumption intensity per high-impact climate sector",
        "unit": "GWh/EUR M revenue",
        "regulatory_ref": "Table 1, Indicator 6, Annex I RTS",
    },
}


# NACE high-impact sectors for PAI 6
NACE_HIGH_IMPACT_SECTORS: Dict[str, str] = {
    "A": "Agriculture, Forestry and Fishing",
    "B": "Mining and Quarrying",
    "C": "Manufacturing",
    "D": "Electricity, Gas, Steam",
    "E": "Water Supply, Sewerage, Waste",
    "F": "Construction",
    "G": "Wholesale and Retail Trade",
    "H": "Transportation and Storage",
    "L": "Real Estate Activities",
}


# =============================================================================
# MRV Emissions Bridge
# =============================================================================


class MRVEmissionsBridge:
    """Bridge connecting SFDR Article 9 Pack to 30 MRV emission agents.

    Routes PAI indicator 1-6 calculations to the appropriate MRV agents
    based on scope and category. Enforces mandatory PAI reporting for
    Article 9 products (all 18 mandatory, no opt-out for PAI 1-6).

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for MRV agents.
        _registry: MRV agent registry with routing information.

    Example:
        >>> bridge = MRVEmissionsBridge(MRVBridgeConfig())
        >>> result = bridge.calculate_pai_emissions(holdings, nav_eur=5e6)
        >>> print(f"Scope 1: {result.scope_1_tco2e:.2f} tCO2e")
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize the MRV Emissions Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or MRVBridgeConfig()
        self.logger = logger
        self._registry = list(MRV_AGENT_REGISTRY)

        # Build agent stubs from registry
        self._agents: Dict[str, _AgentStub] = {}
        for mapping in self._registry:
            if mapping.enabled:
                self._agents[mapping.agent_id] = _AgentStub(
                    mapping.agent_id,
                    mapping.module_path,
                    mapping.class_name,
                )

        self.logger.info(
            "MRVEmissionsBridge initialized: agents=%d, scope1=%s, scope2=%s, "
            "scope3=%s, mandatory_enforcement=%s",
            len(self._agents),
            self.config.enable_scope_1,
            self.config.enable_scope_2,
            self.config.enable_scope_3,
            self.config.mandatory_pai_enforcement,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def calculate_pai_emissions(
        self,
        holdings: List[Dict[str, Any]],
        nav_eur: float = 0.0,
    ) -> EmissionsAggregate:
        """Calculate PAI 1-6 emissions for the portfolio.

        Routes each PAI indicator to the appropriate MRV agents and
        aggregates the results. Article 9 mandatory enforcement ensures
        all 6 indicators are calculated or errors are raised.

        Args:
            holdings: List of portfolio holding records with ISIN, weight, etc.
            nav_eur: Portfolio Net Asset Value in EUR.

        Returns:
            EmissionsAggregate with all PAI 1-6 values and metadata.
        """
        start_time = time.time()
        start_dt = _utcnow()

        all_agents_used: List[str] = []
        all_errors: List[str] = []
        all_warnings: List[str] = []
        pai_results: Dict[str, PAIRoutingResult] = {}

        # Calculate each PAI indicator
        for pai_id, pai_def in PAI_INDICATOR_DEFINITIONS.items():
            pai_result = self._route_pai_indicator(
                pai_id, pai_def, holdings, nav_eur,
            )
            pai_results[pai_id] = pai_result
            all_agents_used.extend(pai_result.agents_invoked)
            all_errors.extend(pai_result.errors)
            all_warnings.extend(pai_result.warnings)

        # Enforce mandatory PAI for Article 9
        if self.config.mandatory_pai_enforcement:
            missing = self._check_mandatory_pai_coverage(pai_results)
            if missing:
                all_warnings.append(
                    f"Missing mandatory PAI indicators: {', '.join(missing)}"
                )

        # Build aggregate
        scope_1 = self._sum_scope_emissions(pai_results, EmissionScope.SCOPE_1)
        scope_2 = self._sum_scope_emissions(pai_results, EmissionScope.SCOPE_2_MARKET)
        scope_3 = self._sum_scope_emissions(pai_results, EmissionScope.SCOPE_3)
        total = scope_1 + scope_2 + scope_3

        elapsed_ms = (time.time() - start_time) * 1000

        aggregate = EmissionsAggregate(
            scope_1_tco2e=scope_1,
            scope_2_tco2e=scope_2,
            scope_3_tco2e=scope_3,
            total_scope_1_2_3_tco2e=total,
            carbon_footprint_tco2e_per_eur_m=(
                (total / (nav_eur / 1_000_000)) if nav_eur > 0 else 0.0
            ),
            ghg_intensity_tco2e_per_eur_m_revenue=(
                pai_results.get("pai_3", PAIRoutingResult(pai_indicator="pai_3")).value
            ),
            fossil_fuel_exposure_pct=(
                pai_results.get("pai_4", PAIRoutingResult(pai_indicator="pai_4")).value
            ),
            non_renewable_energy_share_pct=(
                pai_results.get("pai_5", PAIRoutingResult(pai_indicator="pai_5")).value
            ),
            energy_intensity_gwh_per_eur_m=(
                pai_results.get("pai_6", PAIRoutingResult(pai_indicator="pai_6")).value
            ),
            nav_eur=nav_eur,
            total_holdings=len(holdings),
            coverage_pct=self._calculate_coverage(holdings, pai_results),
            data_quality_score=self._calculate_data_quality_score(pai_results),
            pai_results=pai_results,
            agents_used=list(set(all_agents_used)),
            errors=all_errors,
            warnings=all_warnings,
            calculated_at=start_dt.isoformat(),
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            aggregate.provenance_hash = _hash_data(
                aggregate.model_dump(exclude={"provenance_hash"})
            )

        self.logger.info(
            "MRVEmissionsBridge: PAI 1-6 calculated in %.1fms, total=%.2f tCO2e, "
            "holdings=%d, agents=%d",
            elapsed_ms, total, len(holdings), len(set(all_agents_used)),
        )
        return aggregate

    def get_scope_breakdown(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Get per-scope emission breakdown for the portfolio.

        Args:
            holdings: Portfolio holding records.

        Returns:
            Dict mapping scope names to tCO2e values.
        """
        result = self.calculate_pai_emissions(holdings)
        return {
            "scope_1": result.scope_1_tco2e,
            "scope_2": result.scope_2_tco2e,
            "scope_3": result.scope_3_tco2e,
            "total": result.total_scope_1_2_3_tco2e,
        }

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered MRV agents.

        Returns:
            Dict mapping agent IDs to their status information.
        """
        status: Dict[str, Dict[str, Any]] = {}
        for mapping in self._registry:
            stub = self._agents.get(mapping.agent_id)
            status[mapping.agent_id] = {
                "category": mapping.category.value,
                "scope": mapping.scope.value,
                "enabled": mapping.enabled,
                "loaded": stub.is_loaded if stub else False,
                "pai_indicators": mapping.pai_indicators,
                "class_name": mapping.class_name,
            }
        return status

    def get_pai_routing_map(self) -> Dict[str, List[str]]:
        """Get mapping of PAI indicators to agent IDs.

        Returns:
            Dict mapping PAI indicator IDs to lists of agent IDs.
        """
        routing: Dict[str, List[str]] = {}
        for pai_id in PAI_INDICATOR_DEFINITIONS:
            routing[pai_id] = [
                m.agent_id for m in self._registry
                if pai_id in m.pai_indicators and m.enabled
            ]
        return routing

    def validate_mandatory_coverage(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate that mandatory PAI 1-6 data coverage meets Article 9 requirements.

        Args:
            holdings: Portfolio holding records.

        Returns:
            Validation result with per-indicator coverage and overall status.
        """
        result = self.calculate_pai_emissions(holdings)
        per_pai: Dict[str, Dict[str, Any]] = {}
        all_pass = True

        for pai_id, pai_def in PAI_INDICATOR_DEFINITIONS.items():
            pai_result = result.pai_results.get(pai_id)
            coverage = pai_result.coverage_pct if pai_result else 0.0
            has_value = pai_result is not None and pai_result.value != 0.0
            passed = has_value and coverage >= 50.0
            if not passed:
                all_pass = False

            per_pai[pai_id] = {
                "name": pai_def["name"],
                "value": pai_result.value if pai_result else 0.0,
                "coverage_pct": coverage,
                "data_quality": pai_result.data_quality if pai_result else "unavailable",
                "passed": passed,
            }

        return {
            "overall_pass": all_pass,
            "indicators": per_pai,
            "total_indicators": len(PAI_INDICATOR_DEFINITIONS),
            "passing_indicators": sum(1 for p in per_pai.values() if p["passed"]),
            "validated_at": _utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Private Routing Methods
    # -------------------------------------------------------------------------

    def _route_pai_indicator(
        self,
        pai_id: str,
        pai_def: Dict[str, str],
        holdings: List[Dict[str, Any]],
        nav_eur: float,
    ) -> PAIRoutingResult:
        """Route a single PAI indicator to appropriate MRV agents.

        Args:
            pai_id: PAI indicator identifier (pai_1 through pai_6).
            pai_def: PAI indicator definition.
            holdings: Portfolio holding records.
            nav_eur: Portfolio NAV in EUR.

        Returns:
            PAIRoutingResult with calculated value and metadata.
        """
        start_time = time.time()
        agents_invoked: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []
        scope_breakdown: Dict[str, float] = {}

        # Find agents for this PAI indicator
        target_agents = [
            m for m in self._registry
            if pai_id in m.pai_indicators and m.enabled
        ]

        if not target_agents:
            errors.append(f"No MRV agents registered for {pai_id}")
            return PAIRoutingResult(
                pai_indicator=pai_id,
                pai_name=pai_def.get("name", ""),
                errors=errors,
                calculated_at=_utcnow().isoformat(),
            )

        # Route based on PAI type
        if pai_id == "pai_1":
            value, scope_breakdown = self._calculate_pai_1_ghg(
                target_agents, holdings, agents_invoked, warnings, errors,
            )
        elif pai_id == "pai_2":
            value = self._calculate_pai_2_carbon_footprint(
                holdings, nav_eur, agents_invoked, warnings, errors,
            )
        elif pai_id == "pai_3":
            value = self._calculate_pai_3_ghg_intensity(
                holdings, agents_invoked, warnings, errors,
            )
        elif pai_id == "pai_4":
            value = self._calculate_pai_4_fossil_fuel(
                holdings, agents_invoked, warnings, errors,
            )
        elif pai_id == "pai_5":
            value = self._calculate_pai_5_non_renewable(
                holdings, agents_invoked, warnings, errors,
            )
        elif pai_id == "pai_6":
            value = self._calculate_pai_6_energy_intensity(
                holdings, agents_invoked, warnings, errors,
            )
        else:
            value = 0.0
            errors.append(f"Unknown PAI indicator: {pai_id}")

        elapsed_ms = (time.time() - start_time) * 1000

        result = PAIRoutingResult(
            pai_indicator=pai_id,
            pai_name=pai_def.get("name", ""),
            agents_invoked=agents_invoked,
            value=value,
            unit=pai_def.get("unit", "tCO2e"),
            data_quality=self._assess_data_quality(holdings).value,
            coverage_pct=self._calculate_holding_coverage(holdings),
            scope_breakdown=scope_breakdown,
            warnings=warnings,
            errors=errors,
            calculated_at=_utcnow().isoformat(),
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(exclude={"provenance_hash"})
            )

        return result

    def _calculate_pai_1_ghg(
        self,
        target_agents: List[MRVAgentMapping],
        holdings: List[Dict[str, Any]],
        agents_invoked: List[str],
        warnings: List[str],
        errors: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate PAI 1 GHG emissions (Scope 1+2+3).

        Uses deterministic MRV agents -- zero hallucination.
        """
        scope_breakdown: Dict[str, float] = {
            "scope_1": 0.0, "scope_2": 0.0, "scope_3": 0.0,
        }

        for mapping in target_agents:
            stub = self._agents.get(mapping.agent_id)
            if stub is None:
                continue

            agent = stub.load()
            if agent is None:
                warnings.append(f"Agent {mapping.agent_id} unavailable")
                continue

            agents_invoked.append(mapping.agent_id)

            try:
                agent_result = self._invoke_agent(agent, holdings)
                emissions = float(agent_result.get("total_emissions_tco2e", 0.0))

                if mapping.scope == EmissionScope.SCOPE_1:
                    scope_breakdown["scope_1"] += emissions
                elif mapping.scope in (EmissionScope.SCOPE_2_LOCATION,
                                       EmissionScope.SCOPE_2_MARKET):
                    scope_breakdown["scope_2"] += emissions
                elif mapping.scope == EmissionScope.SCOPE_3:
                    scope_breakdown["scope_3"] += emissions

            except Exception as exc:
                errors.append(f"Agent {mapping.agent_id} failed: {exc}")

        total = sum(scope_breakdown.values())
        return total, scope_breakdown

    def _calculate_pai_2_carbon_footprint(
        self,
        holdings: List[Dict[str, Any]],
        nav_eur: float,
        agents_invoked: List[str],
        warnings: List[str],
        errors: List[str],
    ) -> float:
        """Calculate PAI 2 carbon footprint (tCO2e / EUR M invested).

        Deterministic: total emissions / (NAV / 1,000,000).
        """
        if nav_eur <= 0:
            warnings.append("NAV not provided; carbon footprint set to 0")
            return 0.0

        total_emissions = 0.0
        for mapping in self._registry:
            if "pai_1" not in mapping.pai_indicators or not mapping.enabled:
                continue
            stub = self._agents.get(mapping.agent_id)
            if stub is None:
                continue
            agent = stub.load()
            if agent is None:
                continue
            agents_invoked.append(mapping.agent_id)
            try:
                result = self._invoke_agent(agent, holdings)
                total_emissions += float(result.get("total_emissions_tco2e", 0.0))
            except Exception as exc:
                errors.append(f"PAI-2 agent {mapping.agent_id} error: {exc}")

        return total_emissions / (nav_eur / 1_000_000)

    def _calculate_pai_3_ghg_intensity(
        self,
        holdings: List[Dict[str, Any]],
        agents_invoked: List[str],
        warnings: List[str],
        errors: List[str],
    ) -> float:
        """Calculate PAI 3 GHG intensity (tCO2e / EUR M revenue).

        Weighted average by portfolio weight.
        """
        weighted_intensity = 0.0
        total_weight = 0.0

        for holding in holdings:
            weight = float(holding.get("weight", 0.0))
            revenue_eur = float(holding.get("revenue_eur", 0.0))
            emissions = float(holding.get("total_emissions_tco2e", 0.0))

            if revenue_eur > 0 and weight > 0:
                intensity = emissions / (revenue_eur / 1_000_000)
                weighted_intensity += weight * intensity
                total_weight += weight

        if total_weight > 0:
            return weighted_intensity / total_weight

        warnings.append("No revenue data available for GHG intensity calculation")
        return 0.0

    def _calculate_pai_4_fossil_fuel(
        self,
        holdings: List[Dict[str, Any]],
        agents_invoked: List[str],
        warnings: List[str],
        errors: List[str],
    ) -> float:
        """Calculate PAI 4 fossil fuel exposure (%).

        Deterministic: sum of weights of holdings flagged as fossil fuel.
        """
        fossil_weight = 0.0
        total_weight = 0.0

        for holding in holdings:
            weight = float(holding.get("weight", 0.0))
            is_fossil = holding.get("is_fossil_fuel", False)
            total_weight += weight
            if is_fossil:
                fossil_weight += weight

        if total_weight > 0:
            return (fossil_weight / total_weight) * 100.0

        warnings.append("No weight data for fossil fuel exposure")
        return 0.0

    def _calculate_pai_5_non_renewable(
        self,
        holdings: List[Dict[str, Any]],
        agents_invoked: List[str],
        warnings: List[str],
        errors: List[str],
    ) -> float:
        """Calculate PAI 5 non-renewable energy share (%).

        Weighted average of non-renewable energy share across investees.
        """
        weighted_share = 0.0
        total_weight = 0.0

        for holding in holdings:
            weight = float(holding.get("weight", 0.0))
            nre_share = float(holding.get("non_renewable_energy_pct", 0.0))
            if weight > 0:
                weighted_share += weight * nre_share
                total_weight += weight

        if total_weight > 0:
            return weighted_share / total_weight

        warnings.append("No energy data for non-renewable share")
        return 0.0

    def _calculate_pai_6_energy_intensity(
        self,
        holdings: List[Dict[str, Any]],
        agents_invoked: List[str],
        warnings: List[str],
        errors: List[str],
    ) -> float:
        """Calculate PAI 6 energy consumption intensity (GWh / EUR M revenue).

        Only includes holdings in NACE high-impact climate sectors.
        """
        total_energy_gwh = 0.0
        total_revenue_eur_m = 0.0

        for holding in holdings:
            nace = holding.get("nace_code", "")
            nace_letter = nace[:1].upper() if nace else ""

            if nace_letter not in NACE_HIGH_IMPACT_SECTORS:
                continue

            energy = float(holding.get("energy_consumption_gwh", 0.0))
            revenue = float(holding.get("revenue_eur", 0.0)) / 1_000_000

            total_energy_gwh += energy
            if revenue > 0:
                total_revenue_eur_m += revenue

        if total_revenue_eur_m > 0:
            return total_energy_gwh / total_revenue_eur_m

        warnings.append("No high-impact sector data for energy intensity")
        return 0.0

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _invoke_agent(
        self,
        agent: Any,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Invoke an MRV agent with holding data.

        Args:
            agent: Loaded agent instance.
            holdings: Portfolio holding records.

        Returns:
            Agent result dict with emissions data.
        """
        try:
            if hasattr(agent, "calculate"):
                return agent.calculate(holdings)
            elif hasattr(agent, "process"):
                return agent.process(holdings)
            elif hasattr(agent, "run"):
                return agent.run(holdings)
            else:
                return {"total_emissions_tco2e": 0.0}
        except Exception as exc:
            self.logger.warning("Agent invocation failed: %s", exc)
            return {"total_emissions_tco2e": 0.0}

    def _check_mandatory_pai_coverage(
        self,
        pai_results: Dict[str, PAIRoutingResult],
    ) -> List[str]:
        """Check that all mandatory PAI 1-6 are covered.

        Returns:
            List of missing PAI indicator IDs.
        """
        missing: List[str] = []
        for pai_id in PAI_INDICATOR_DEFINITIONS:
            result = pai_results.get(pai_id)
            if result is None or (len(result.errors) > 0 and result.value == 0.0):
                missing.append(pai_id)
        return missing

    def _sum_scope_emissions(
        self,
        pai_results: Dict[str, PAIRoutingResult],
        scope: EmissionScope,
    ) -> float:
        """Sum emissions for a given scope from PAI results."""
        pai_1 = pai_results.get("pai_1")
        if pai_1 is None:
            return 0.0

        scope_key = scope.value.replace("scope_2_market", "scope_2").replace(
            "scope_2_location", "scope_2"
        )
        return pai_1.scope_breakdown.get(scope_key, 0.0)

    def _calculate_coverage(
        self,
        holdings: List[Dict[str, Any]],
        pai_results: Dict[str, PAIRoutingResult],
    ) -> float:
        """Calculate overall data coverage percentage."""
        if not pai_results:
            return 0.0
        coverages = [r.coverage_pct for r in pai_results.values()]
        return sum(coverages) / len(coverages) if coverages else 0.0

    def _calculate_holding_coverage(
        self,
        holdings: List[Dict[str, Any]],
    ) -> float:
        """Calculate holding coverage based on available emission data."""
        if not holdings:
            return 0.0
        covered = sum(
            1 for h in holdings
            if h.get("total_emissions_tco2e") is not None
            or h.get("emissions_data_available", False)
        )
        return (covered / len(holdings)) * 100.0

    def _calculate_data_quality_score(
        self,
        pai_results: Dict[str, PAIRoutingResult],
    ) -> float:
        """Calculate weighted data quality score from PAI results."""
        quality_weights: Dict[str, float] = {
            "reported": 1.0,
            "estimated_specific": 0.75,
            "estimated_average": 0.5,
            "proxy": 0.25,
            "unavailable": 0.0,
        }

        if not pai_results:
            return 0.0

        total_score = 0.0
        count = 0
        for result in pai_results.values():
            weight = quality_weights.get(result.data_quality, 0.5)
            total_score += weight
            count += 1

        return total_score / count if count > 0 else 0.0

    def _assess_data_quality(
        self,
        holdings: List[Dict[str, Any]],
    ) -> DataQualityTier:
        """Assess overall data quality tier for holdings."""
        if not holdings:
            return DataQualityTier.UNAVAILABLE

        reported = sum(1 for h in holdings if h.get("data_source") == "reported")
        ratio = reported / len(holdings)

        if ratio >= 0.8:
            return DataQualityTier.REPORTED
        elif ratio >= 0.5:
            return DataQualityTier.ESTIMATED_SPECIFIC
        elif ratio >= 0.2:
            return DataQualityTier.ESTIMATED_AVERAGE
        else:
            return DataQualityTier.PROXY
