"""
MRV Taxonomy Bridge - PACK-008 EU Taxonomy Alignment

This module routes Scope 1/2/3 emissions data from MRV agents to Climate Change
Mitigation (CCM) and Climate Change Adaptation (CCA) Technical Screening Criteria
evaluations. Each taxonomy activity maps to specific MRV agents for emissions
data retrieval.

MRV Routing Coverage:
- Scope 1: Stationary combustion, mobile combustion, process, fugitive, land use
- Scope 2: Location-based, market-based, steam/heat, cooling
- Scope 3: All 15 categories via category mapper

Example:
    >>> config = MRVTaxonomyBridgeConfig(
    ...     mrv_agents_enabled=True,
    ...     scope_coverage=["scope_1", "scope_2", "scope_3"]
    ... )
    >>> bridge = MRVTaxonomyBridge(config)
    >>> emissions = await bridge.get_emissions_for_activity("4.1")
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import logging
import asyncio

logger = logging.getLogger(__name__)


class MRVTaxonomyBridgeConfig(BaseModel):
    """Configuration for MRV Taxonomy Bridge."""

    mrv_agents_enabled: bool = Field(
        default=True,
        description="Enable MRV agent data retrieval"
    )
    scope_coverage: List[str] = Field(
        default=["scope_1", "scope_2", "scope_3"],
        description="Emission scope coverage"
    )
    emission_unit: str = Field(
        default="tCO2e",
        description="Standard emission unit"
    )
    reporting_year: int = Field(
        default=2025,
        ge=2020,
        description="Reporting period year"
    )
    parallel_fetch: bool = Field(
        default=True,
        description="Fetch MRV data in parallel"
    )
    cache_emissions: bool = Field(
        default=True,
        description="Cache emissions data during assessment"
    )


class _AgentStub:
    """Stub for MRV agent injection pattern."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._real_agent: Any = None

    def inject(self, real_agent: Any) -> None:
        """Inject real agent instance."""
        self._real_agent = real_agent
        logger.info(f"Injected real MRV agent for {self.agent_name}")

    async def execute(self, method_name: str, **kwargs) -> Any:
        """Execute agent method (real or fallback)."""
        if self._real_agent and hasattr(self._real_agent, method_name):
            method = getattr(self._real_agent, method_name)
            if asyncio.iscoroutinefunction(method):
                return await method(**kwargs)
            return method(**kwargs)

        return {
            "status": "fallback",
            "agent": self.agent_name,
            "method": method_name,
            "message": f"MRV fallback for {self.agent_name}.{method_name}",
            "data": kwargs
        }


# Mapping from EU Taxonomy activity codes to MRV agent IDs
MRV_ROUTING_TABLE: Dict[str, Dict[str, Any]] = {
    # Energy sector (4.x)
    "4.1": {"agent": "mrv_001_stationary_combustion", "metric": "direct_emissions_intensity", "scope": "scope_1", "description": "Electricity generation using solar PV"},
    "4.2": {"agent": "mrv_001_stationary_combustion", "metric": "direct_emissions_intensity", "scope": "scope_1", "description": "Electricity generation using CSP"},
    "4.3": {"agent": "mrv_003_mobile_combustion", "metric": "lifecycle_emissions", "scope": "scope_1", "description": "Electricity generation from wind power"},
    "4.5": {"agent": "mrv_001_stationary_combustion", "metric": "direct_emissions_intensity", "scope": "scope_1", "description": "Electricity generation from hydropower"},
    "4.7": {"agent": "mrv_001_stationary_combustion", "metric": "direct_emissions_intensity", "scope": "scope_1", "description": "Electricity generation from renewable non-fossil gaseous and liquid fuels"},
    "4.8": {"agent": "mrv_001_stationary_combustion", "metric": "direct_emissions_intensity", "scope": "scope_1", "description": "Electricity generation from bioenergy"},
    "4.9": {"agent": "mrv_011_steam_heat", "metric": "heat_generation_emissions", "scope": "scope_2", "description": "Transmission and distribution of electricity"},
    "4.10": {"agent": "mrv_011_steam_heat", "metric": "heat_generation_emissions", "scope": "scope_2", "description": "Storage of electricity"},
    "4.13": {"agent": "mrv_001_stationary_combustion", "metric": "direct_emissions_intensity", "scope": "scope_1", "description": "Manufacture of biogas and biofuels"},
    "4.14": {"agent": "mrv_011_steam_heat", "metric": "heat_generation_emissions", "scope": "scope_2", "description": "Transmission and distribution networks for renewable and low-carbon gases"},
    "4.15": {"agent": "mrv_011_steam_heat", "metric": "heat_generation_emissions", "scope": "scope_2", "description": "District heating/cooling distribution"},
    "4.16": {"agent": "mrv_001_stationary_combustion", "metric": "direct_emissions_intensity", "scope": "scope_1", "description": "Installation and operation of electric heat pumps"},
    "4.17": {"agent": "mrv_001_stationary_combustion", "metric": "cogeneration_emissions", "scope": "scope_1", "description": "Cogeneration of heat/cool and power from solar energy"},
    "4.18": {"agent": "mrv_001_stationary_combustion", "metric": "cogeneration_emissions", "scope": "scope_1", "description": "Cogeneration of heat/cool and power from geothermal energy"},
    "4.19": {"agent": "mrv_001_stationary_combustion", "metric": "cogeneration_emissions", "scope": "scope_1", "description": "Cogeneration of heat/cool and power from renewable non-fossil fuels"},
    "4.20": {"agent": "mrv_001_stationary_combustion", "metric": "cogeneration_emissions", "scope": "scope_1", "description": "Cogeneration of heat/cool and power from bioenergy"},
    # Transport sector (6.x)
    "6.1": {"agent": "mrv_003_mobile_combustion", "metric": "rail_emissions", "scope": "scope_1", "description": "Passenger interurban rail transport"},
    "6.2": {"agent": "mrv_003_mobile_combustion", "metric": "rail_emissions", "scope": "scope_1", "description": "Freight rail transport"},
    "6.3": {"agent": "mrv_003_mobile_combustion", "metric": "urban_transport_emissions", "scope": "scope_1", "description": "Urban and suburban transport"},
    "6.4": {"agent": "mrv_003_mobile_combustion", "metric": "cycling_infrastructure", "scope": "scope_1", "description": "Infrastructure for personal mobility"},
    "6.5": {"agent": "mrv_003_mobile_combustion", "metric": "road_transport_emissions", "scope": "scope_1", "description": "Transport by motorbikes, passenger cars and light commercial vehicles"},
    "6.6": {"agent": "mrv_003_mobile_combustion", "metric": "road_freight_emissions", "scope": "scope_1", "description": "Freight transport services by road"},
    "6.7": {"agent": "mrv_003_mobile_combustion", "metric": "inland_waterway_emissions", "scope": "scope_1", "description": "Inland passenger water transport"},
    "6.10": {"agent": "mrv_003_mobile_combustion", "metric": "sea_transport_emissions", "scope": "scope_1", "description": "Sea and coastal freight water transport"},
    "6.11": {"agent": "mrv_003_mobile_combustion", "metric": "sea_transport_emissions", "scope": "scope_1", "description": "Sea and coastal passenger water transport"},
    # Building sector (7.x)
    "7.1": {"agent": "mrv_009_scope2_location", "metric": "building_energy_emissions", "scope": "scope_2", "description": "Construction of new buildings"},
    "7.2": {"agent": "mrv_009_scope2_location", "metric": "building_energy_emissions", "scope": "scope_2", "description": "Renovation of existing buildings"},
    "7.3": {"agent": "mrv_012_cooling", "metric": "cooling_energy_emissions", "scope": "scope_2", "description": "Installation, maintenance and repair of energy efficiency equipment"},
    "7.4": {"agent": "mrv_012_cooling", "metric": "cooling_energy_emissions", "scope": "scope_2", "description": "Installation, maintenance and repair of charging stations"},
    "7.5": {"agent": "mrv_010_scope2_market", "metric": "renewable_energy_purchase", "scope": "scope_2", "description": "Installation, maintenance and repair of instruments for measuring, regulation and controlling energy performance"},
    "7.6": {"agent": "mrv_010_scope2_market", "metric": "renewable_energy_purchase", "scope": "scope_2", "description": "Installation, maintenance and repair of renewable energy technologies"},
    "7.7": {"agent": "mrv_009_scope2_location", "metric": "building_energy_emissions", "scope": "scope_2", "description": "Acquisition and ownership of buildings"},
    # Manufacturing sector (3.x)
    "3.1": {"agent": "mrv_004_process_emissions", "metric": "process_emissions_intensity", "scope": "scope_1", "description": "Manufacture of renewable energy technologies"},
    "3.2": {"agent": "mrv_004_process_emissions", "metric": "process_emissions_intensity", "scope": "scope_1", "description": "Manufacture of equipment for production and use of hydrogen"},
    "3.3": {"agent": "mrv_004_process_emissions", "metric": "process_emissions_intensity", "scope": "scope_1", "description": "Manufacture of low carbon technologies for transport"},
    "3.4": {"agent": "mrv_004_process_emissions", "metric": "process_emissions_intensity", "scope": "scope_1", "description": "Manufacture of batteries"},
    "3.5": {"agent": "mrv_004_process_emissions", "metric": "process_emissions_intensity", "scope": "scope_1", "description": "Manufacture of energy efficiency equipment for buildings"},
    "3.6": {"agent": "mrv_004_process_emissions", "metric": "process_emissions_intensity", "scope": "scope_1", "description": "Manufacture of other low carbon technologies"},
    "3.7": {"agent": "mrv_004_process_emissions", "metric": "cement_emissions_intensity", "scope": "scope_1", "description": "Manufacture of cement"},
    "3.8": {"agent": "mrv_004_process_emissions", "metric": "aluminium_emissions_intensity", "scope": "scope_1", "description": "Manufacture of aluminium"},
    "3.9": {"agent": "mrv_004_process_emissions", "metric": "steel_emissions_intensity", "scope": "scope_1", "description": "Manufacture of iron and steel"},
    # Forestry/Agriculture (1.x / 2.x)
    "1.1": {"agent": "mrv_006_land_use", "metric": "afforestation_sequestration", "scope": "scope_1", "description": "Afforestation"},
    "1.2": {"agent": "mrv_006_land_use", "metric": "forest_management_emissions", "scope": "scope_1", "description": "Rehabilitation and restoration of forests"},
    "1.3": {"agent": "mrv_006_land_use", "metric": "forest_management_emissions", "scope": "scope_1", "description": "Forest management"},
    "1.4": {"agent": "mrv_006_land_use", "metric": "conservation_emissions", "scope": "scope_1", "description": "Conservation forestry"},
    "2.1": {"agent": "mrv_006_land_use", "metric": "restoration_emissions", "scope": "scope_1", "description": "Restoration of wetlands"},
    # Waste/Water (5.x)
    "5.1": {"agent": "mrv_007_waste_treatment", "metric": "wastewater_emissions", "scope": "scope_1", "description": "Construction, extension and operation of water collection"},
    "5.2": {"agent": "mrv_007_waste_treatment", "metric": "wastewater_treatment_emissions", "scope": "scope_1", "description": "Renewal of water collection, treatment and supply systems"},
    "5.3": {"agent": "mrv_007_waste_treatment", "metric": "wastewater_treatment_emissions", "scope": "scope_1", "description": "Construction, extension and operation of waste water collection"},
    "5.5": {"agent": "mrv_007_waste_treatment", "metric": "material_recovery_emissions", "scope": "scope_1", "description": "Collection and transport of non-hazardous waste"},
    "5.9": {"agent": "mrv_007_waste_treatment", "metric": "material_recovery_emissions", "scope": "scope_1", "description": "Material recovery from non-hazardous waste"},
    # Scope 3 downstream
    "scope3_cat1": {"agent": "mrv_014_purchased_goods", "metric": "purchased_goods_emissions", "scope": "scope_3", "description": "Purchased goods and services"},
    "scope3_cat2": {"agent": "mrv_015_capital_goods", "metric": "capital_goods_emissions", "scope": "scope_3", "description": "Capital goods"},
    "scope3_cat4": {"agent": "mrv_017_upstream_transport", "metric": "transport_emissions", "scope": "scope_3", "description": "Upstream transportation and distribution"},
    "scope3_cat5": {"agent": "mrv_018_waste_generated", "metric": "waste_emissions", "scope": "scope_3", "description": "Waste generated in operations"},
    "scope3_cat11": {"agent": "mrv_024_use_sold", "metric": "use_phase_emissions", "scope": "scope_3", "description": "Use of sold products"},
    "scope3_cat12": {"agent": "mrv_025_end_of_life", "metric": "eol_emissions", "scope": "scope_3", "description": "End-of-life treatment of sold products"},
}


class MRVTaxonomyBridge:
    """
    Bridge routing taxonomy activity emissions queries to MRV agents.

    Routes Scope 1/2/3 emissions data from 30 MRV agents to support CCM and CCA
    Technical Screening Criteria evaluation for EU Taxonomy alignment assessment.

    Example:
        >>> config = MRVTaxonomyBridgeConfig()
        >>> bridge = MRVTaxonomyBridge(config)
        >>> bridge.inject_agent("mrv_001_stationary_combustion", real_agent)
        >>> data = await bridge.get_emissions_for_activity("4.1")
    """

    def __init__(self, config: MRVTaxonomyBridgeConfig):
        """Initialize bridge with MRV agent stubs."""
        self.config = config
        self._agents: Dict[str, _AgentStub] = {}
        self._emissions_cache: Dict[str, Any] = {}
        self._initialize_agent_stubs()
        logger.info(
            f"MRVTaxonomyBridge initialized with {len(self._agents)} MRV agent stubs"
        )

    def _initialize_agent_stubs(self) -> None:
        """Create stubs for all MRV agents referenced in routing table."""
        agent_names = set()
        for route in MRV_ROUTING_TABLE.values():
            agent_names.add(route["agent"])

        # Ensure all 30 MRV agents have stubs
        all_mrv = [
            "mrv_001_stationary_combustion", "mrv_002_refrigerants",
            "mrv_003_mobile_combustion", "mrv_004_process_emissions",
            "mrv_005_fugitive_emissions", "mrv_006_land_use",
            "mrv_007_waste_treatment", "mrv_008_agricultural",
            "mrv_009_scope2_location", "mrv_010_scope2_market",
            "mrv_011_steam_heat", "mrv_012_cooling",
            "mrv_013_dual_reporting", "mrv_014_purchased_goods",
            "mrv_015_capital_goods", "mrv_016_fuel_energy",
            "mrv_017_upstream_transport", "mrv_018_waste_generated",
            "mrv_019_business_travel", "mrv_020_employee_commuting",
            "mrv_021_upstream_leased", "mrv_022_downstream_transport",
            "mrv_023_processing_sold", "mrv_024_use_sold",
            "mrv_025_end_of_life", "mrv_026_downstream_leased",
            "mrv_027_franchises", "mrv_028_investments",
            "mrv_029_scope3_mapper", "mrv_030_audit_trail"
        ]

        for name in all_mrv:
            self._agents[name] = _AgentStub(name)

    def inject_agent(self, agent_name: str, real_agent: Any) -> None:
        """Inject real MRV agent instance."""
        if agent_name in self._agents:
            self._agents[agent_name].inject(real_agent)
        else:
            logger.warning(f"Unknown MRV agent name: {agent_name}")

    async def get_emissions_for_activity(
        self,
        activity_code: str
    ) -> Dict[str, Any]:
        """
        Get emissions data for a taxonomy activity from the appropriate MRV agent.

        Args:
            activity_code: EU Taxonomy activity code (e.g., "4.1", "6.5")

        Returns:
            Emissions data from the routed MRV agent
        """
        try:
            # Check cache
            cache_key = f"{activity_code}_{self.config.reporting_year}"
            if self.config.cache_emissions and cache_key in self._emissions_cache:
                logger.debug(f"Cache hit for activity {activity_code}")
                return self._emissions_cache[cache_key]

            # Look up routing
            route = MRV_ROUTING_TABLE.get(activity_code)
            if not route:
                logger.warning(f"No MRV routing for activity {activity_code}")
                return {
                    "status": "no_routing",
                    "activity_code": activity_code,
                    "message": f"No MRV agent mapped for activity {activity_code}",
                    "timestamp": datetime.utcnow().isoformat()
                }

            agent_name = route["agent"]
            metric_code = route["metric"]
            scope = route["scope"]

            # Check scope coverage
            if scope not in self.config.scope_coverage:
                return {
                    "status": "scope_excluded",
                    "activity_code": activity_code,
                    "scope": scope,
                    "message": f"Scope {scope} not in coverage",
                    "timestamp": datetime.utcnow().isoformat()
                }

            # Route to MRV agent
            result = await self._agents[agent_name].execute(
                "get_emissions",
                metric=metric_code,
                reporting_year=self.config.reporting_year,
                unit=self.config.emission_unit
            )

            emissions_data = {
                "activity_code": activity_code,
                "activity_description": route["description"],
                "agent": agent_name,
                "scope": scope,
                "metric": metric_code,
                "emissions": result,
                "unit": self.config.emission_unit,
                "reporting_year": self.config.reporting_year,
                "provenance_hash": self._calculate_hash(result),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Cache result
            if self.config.cache_emissions:
                self._emissions_cache[cache_key] = emissions_data

            return emissions_data

        except Exception as e:
            logger.error(f"Emissions fetch failed for activity {activity_code}: {str(e)}")
            return {
                "status": "error",
                "activity_code": activity_code,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def route_to_mrv_agent(
        self,
        activity_code: str,
        metric_code: str
    ) -> Any:
        """
        Route a specific metric query to the mapped MRV agent.

        Args:
            activity_code: EU Taxonomy activity code
            metric_code: Specific metric to fetch

        Returns:
            MRV agent response
        """
        try:
            route = MRV_ROUTING_TABLE.get(activity_code)
            if not route:
                return {
                    "status": "no_routing",
                    "activity_code": activity_code,
                    "metric": metric_code
                }

            agent_name = route["agent"]
            result = await self._agents[agent_name].execute(
                "get_metric",
                metric=metric_code,
                reporting_year=self.config.reporting_year
            )

            return result

        except Exception as e:
            logger.error(f"MRV routing failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_scope_summary(
        self,
        scope: Literal["scope_1", "scope_2", "scope_3"]
    ) -> Dict[str, Any]:
        """
        Get aggregated emissions summary for an entire scope.

        Args:
            scope: Emission scope to summarize

        Returns:
            Aggregated scope emissions data
        """
        try:
            # Collect all activities for this scope
            scope_activities = {
                code: route for code, route in MRV_ROUTING_TABLE.items()
                if route["scope"] == scope
            }

            if self.config.parallel_fetch:
                tasks = [
                    self.get_emissions_for_activity(code)
                    for code in scope_activities.keys()
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []
                for code in scope_activities.keys():
                    result = await self.get_emissions_for_activity(code)
                    results.append(result)

            # Filter out errors
            valid_results = [
                r for r in results
                if isinstance(r, dict) and r.get("status") != "error"
            ]

            return {
                "scope": scope,
                "total_activities": len(scope_activities),
                "data_available": len(valid_results),
                "activities": valid_results,
                "provenance_hash": self._calculate_hash(valid_results),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Scope summary failed for {scope}: {str(e)}")
            return {
                "scope": scope,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_all_emissions_for_tsc(
        self,
        activity_codes: List[str]
    ) -> Dict[str, Any]:
        """
        Fetch emissions data for multiple activities in bulk for TSC evaluation.

        Args:
            activity_codes: List of taxonomy activity codes

        Returns:
            Bulk emissions data keyed by activity code
        """
        try:
            if self.config.parallel_fetch:
                tasks = [
                    self.get_emissions_for_activity(code)
                    for code in activity_codes
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []
                for code in activity_codes:
                    result = await self.get_emissions_for_activity(code)
                    results.append(result)

            emissions_map = {}
            for i, code in enumerate(activity_codes):
                if i < len(results) and isinstance(results[i], dict):
                    emissions_map[code] = results[i]
                else:
                    emissions_map[code] = {
                        "status": "error",
                        "activity_code": code
                    }

            return {
                "total_requested": len(activity_codes),
                "data_available": sum(
                    1 for v in emissions_map.values()
                    if v.get("status") != "error"
                ),
                "emissions": emissions_map,
                "provenance_hash": self._calculate_hash(emissions_map),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Bulk emissions fetch failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def get_routing_table(self) -> Dict[str, Dict[str, Any]]:
        """Return the complete MRV routing table for inspection."""
        return MRV_ROUTING_TABLE.copy()

    def clear_cache(self) -> None:
        """Clear the emissions data cache."""
        self._emissions_cache.clear()
        logger.info("Emissions cache cleared")

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
