"""
GL-Agent-Factory: Unified Agent Registry

This registry provides a single point of access to all 100 Process Heat agents,
supporting dynamic discovery, instantiation, and metadata queries.

Usage:
    from agents.registry import AgentRegistry

    registry = AgentRegistry()
    agent = registry.get_agent("GL-022")
    result = agent.run(input_data)
"""
import importlib
import logging
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Metadata for a registered agent."""
    agent_id: str
    agent_name: str
    module_path: str
    class_name: str
    category: str
    agent_type: str
    complexity: str = "Medium"
    priority: str = "P2"
    market_size: str = ""
    description: str = ""
    standards: List[str] = field(default_factory=list)
    status: str = "Implemented"


# Complete registry of all 100 GL Process Heat agents
AGENT_DEFINITIONS = [
    # Climate & Compliance (GL-001 to GL-021) - Pre-existing
    AgentInfo("GL-001", "THERMALCOMMAND", "gl_001_carbon_emissions", "CarbonEmissionsAgent", "Orchestration", "Coordinator", "High", "P0"),
    AgentInfo("GL-002", "FLAMEGUARD", "gl_002_cbam_compliance", "CBAMComplianceAgent", "Compliance", "Monitor", "Medium", "P0"),
    AgentInfo("GL-003", "UNIFIEDSTEAM", "gl_003_csrd_reporting", "CSRDReportingAgent", "Reporting", "Reporter", "High", "P1"),
    AgentInfo("GL-004", "BURNMASTER", "gl_004_eudr_compliance", "EUDRComplianceAgent", "Compliance", "Monitor", "Medium", "P1"),
    AgentInfo("GL-005", "COMBUSENSE", "gl_005_building_energy", "BuildingEnergyAgent", "Building", "Analyzer", "Medium", "P1"),
    AgentInfo("GL-006", "HEATRECLAIM", "gl_006_scope3_emissions", "Scope3EmissionsAgent", "Emissions", "Calculator", "High", "P0"),
    AgentInfo("GL-007", "FURNACEPULSE", "gl_007_eu_taxonomy", "EUTaxonomyAgent", "Compliance", "Analyzer", "High", "P1"),
    AgentInfo("GL-008", "TRAPCATCHER", "gl_008_green_claims", "GreenClaimsAgent", "Compliance", "Validator", "Medium", "P2"),
    AgentInfo("GL-009", "THERMALIQ", "gl_009_product_carbon_footprint", "ProductCarbonFootprintAgent", "Carbon", "Calculator", "Medium", "P1"),
    AgentInfo("GL-010", "EMISSIONSGUARDIAN", "gl_010_sbti_validation", "SBTiValidationAgent", "Compliance", "Validator", "High", "P0"),
    AgentInfo("GL-011", "FUELCRAFT", "gl_011_climate_risk", "ClimateRiskAgent", "Risk", "Analyzer", "High", "P1"),
    AgentInfo("GL-012", "STEAMQUAL", "gl_012_carbon_offset", "CarbonOffsetAgent", "Carbon", "Manager", "Medium", "P2"),
    AgentInfo("GL-013", "PREDICTMAINT", "gl_013_sb253_disclosure", "SB253DisclosureAgent", "Compliance", "Reporter", "High", "P0"),

    # Process Heat Baseline (GL-020 to GL-021) - Pre-existing
    AgentInfo("GL-020", "ECONOPULSE", "gl_020_economizer_performance", "EconomizerPerformanceAgent", "Heat Recovery", "Monitor", "Medium", "P2"),
    AgentInfo("GL-021", "BURNERSENTRY", "gl_021_burner_maintenance", "BurnerMaintenancePredictorAgent", "Maintenance", "Predictor", "Medium", "P2"),

    # Steam System Agents (GL-022 to GL-030)
    AgentInfo("GL-022", "SUPERHEAT-CTRL", "gl_022_superheater_control", "SuperheaterControlAgent", "Steam Systems", "Controller", "Medium", "P2", "$5B"),
    AgentInfo("GL-023", "LOADBALANCER", "gl_023_heat_load_balancer", "HeatLoadBalancerAgent", "Optimization", "Optimizer", "High", "P1", "$9B"),
    AgentInfo("GL-024", "AIRPREHEATER", "gl_024_air_preheater", "AirPreheaterOptimizerAgent", "Heat Recovery", "Optimizer", "Medium", "P2", "$4B"),
    AgentInfo("GL-025", "COGENMAX", "gl_025_cogeneration", "CogenerationOptimizerAgent", "Cogeneration", "Optimizer", "High", "P0", "$15B"),
    AgentInfo("GL-026", "SOOTBLAST", "gl_026_soot_blower", "SootBlowerControllerAgent", "Combustion", "Controller", "Low", "P2", "$2B"),
    AgentInfo("GL-027", "RADIANT-OPT", "gl_027_radiant_heat", "RadiantHeatOptimizerAgent", "Furnaces", "Optimizer", "Medium", "P2", "$5B"),
    AgentInfo("GL-028", "CONVECTION-WATCH", "gl_028_convection_analyzer", "ConvectionSectionAnalyzerAgent", "Furnaces", "Analyzer", "Medium", "P2", "$4B"),
    AgentInfo("GL-029", "FUELCONDITIONER", "gl_029_fuel_conditioner", "FuelGasConditioningAgent", "Fuel Systems", "Controller", "Low", "P2", "$3B"),
    AgentInfo("GL-030", "HEATINTEGRATOR", "gl_030_heat_integrator", "HeatIntegrationOptimizerAgent", "Process Integration", "Optimizer", "High", "P1", "$12B"),

    # Safety & Optimization Agents (GL-031 to GL-045)
    AgentInfo("GL-031", "THERMALSTORAGE", "gl_031_thermal_storage", "ThermalEnergyStorageAgent", "Energy Storage", "Controller", "High", "P2", "$8B"),
    AgentInfo("GL-032", "HEATREPORTER", "gl_032_heat_reporter", "ProcessHeatReporterAgent", "Reporting", "Reporter", "Medium", "P1", "$6B"),
    AgentInfo("GL-033", "DISTRICT-LINK", "gl_033_district_heating", "DistrictHeatingIntegratorAgent", "Heat Networks", "Integrator", "High", "P2", "$10B"),
    AgentInfo("GL-034", "CARBONCAPTURE-HEAT", "gl_034_carbon_capture_heat", "CarbonCaptureHeatAgent", "Decarbonization", "Optimizer", "High", "P0", "$18B"),
    AgentInfo("GL-035", "H2-BURNER", "gl_035_hydrogen_burner", "HydrogenCombustionAgent", "Future Fuels", "Controller", "High", "P1", "$14B"),
    AgentInfo("GL-036", "ELECTRIFY-SCAN", "gl_036_electrification", "ElectrificationAnalyzerAgent", "Decarbonization", "Analyzer", "High", "P1", "$16B"),
    AgentInfo("GL-037", "BIOMASS-OPT", "gl_037_biomass", "BiomassCombustionOptimizerAgent", "Renewable Heat", "Optimizer", "Medium", "P2", "$7B"),
    AgentInfo("GL-038", "SOLAR-THERMAL", "gl_038_solar_thermal", "SolarThermalIntegratorAgent", "Renewable Heat", "Integrator", "Medium", "P2", "$9B"),
    AgentInfo("GL-039", "HEATPUMP-PRO", "gl_039_heat_pump", "HeatPumpOptimizerAgent", "Heat Pumps", "Optimizer", "High", "P1", "$11B"),
    AgentInfo("GL-040", "SAFETYSENTRY", "gl_040_safety_monitor", "ProcessSafetyMonitorAgent", "Safety", "Monitor", "High", "P0", "$13B"),
    AgentInfo("GL-041", "ENERGYVIZ", "gl_041_energy_dashboard", "EnergyManagementDashboardAgent", "Visualization", "Reporter", "Medium", "P1", "$8B"),
    AgentInfo("GL-042", "PRESSUREMASTER", "gl_042_steam_pressure", "SteamPressureOptimizerAgent", "Steam Systems", "Optimizer", "Medium", "P2", "$5B"),
    AgentInfo("GL-043", "CONDENSATE-RECLAIM", "gl_043_condensate_recovery", "CondensateRecoveryAgent", "Steam Systems", "Monitor", "Low", "P2", "$3B"),
    AgentInfo("GL-044", "FLASHSTEAM", "gl_044_flash_steam", "FlashSteamRecoveryAgent", "Heat Recovery", "Optimizer", "Medium", "P2", "$4B"),
    AgentInfo("GL-045", "AIROPTIMIZER", "gl_045_combustion_air", "CombustionAirOptimizerAgent", "Combustion", "Optimizer", "Medium", "P1", "$6B"),

    # Analytics Agents (GL-046 to GL-060)
    AgentInfo("GL-046", "DRAFTCONTROL", "gl_046_draft_control", "FurnaceDraftControllerAgent", "Furnaces", "Controller", "Low", "P2", "$3B"),
    AgentInfo("GL-047", "REFRACTORYWATCH", "gl_047_refractory", "RefractionMaterialMonitorAgent", "Furnaces", "Monitor", "Medium", "P2", "$4B"),
    AgentInfo("GL-048", "HEATLOSS-CALC", "gl_048_heat_loss", "HeatLossCalculatorAgent", "Analytics", "Calculator", "Medium", "P2", "$5B"),
    AgentInfo("GL-049", "PROCESS-INTEGRATOR", "gl_049_process_control", "ProcessControlIntegratorAgent", "Integration", "Integrator", "High", "P1", "$10B"),
    AgentInfo("GL-050", "VFD-OPTIMIZER", "gl_050_vfd", "VFDOptimizationAgent", "Motors", "Optimizer", "Medium", "P2", "$6B"),
    AgentInfo("GL-051", "STARTUP-CTRL", "gl_051_startup_shutdown", "StartupShutdownOptimizerAgent", "Operations", "Optimizer", "Medium", "P2", "$5B"),
    AgentInfo("GL-052", "HEAT-TRACER", "gl_052_heat_tracing", "HeatTracingOptimizerAgent", "Heat Tracing", "Optimizer", "Low", "P2", "$3B"),
    AgentInfo("GL-053", "THERMAL-OXIDIZER", "gl_053_thermal_oxidizer", "ThermalOxidizerOptimizerAgent", "Emissions Control", "Optimizer", "Medium", "P2", "$7B"),
    AgentInfo("GL-054", "HEAT-TREAT", "gl_054_heat_treatment", "HeatTreatmentOptimizerAgent", "Process", "Optimizer", "High", "P1", "$9B"),
    AgentInfo("GL-055", "DRYMASTER", "gl_055_drying", "DryingProcessOptimizerAgent", "Process", "Optimizer", "Medium", "P1", "$8B"),
    AgentInfo("GL-056", "CURE-CTRL", "gl_056_curing_oven", "CuringOvenControllerAgent", "Process", "Controller", "Medium", "P2", "$5B"),
    AgentInfo("GL-057", "INDUCTION-OPT", "gl_057_induction_heating", "InductionHeatingOptimizerAgent", "Process", "Optimizer", "Medium", "P2", "$6B"),
    AgentInfo("GL-058", "INFRARED-CTRL", "gl_058_infrared_heating", "InfraredHeatingControllerAgent", "Process", "Controller", "Low", "P2", "$4B"),
    AgentInfo("GL-059", "MICROWAVE-HEAT", "gl_059_microwave_heating", "MicrowaveHeatingAgent", "Process", "Controller", "Medium", "P3", "$5B"),
    AgentInfo("GL-060", "RESISTANCE-OPT", "gl_060_resistance_heating", "ResistanceHeatingOptimizerAgent", "Process", "Optimizer", "Low", "P2", "$3B"),

    # Digital Twin & Simulation Agents (GL-061 to GL-075)
    AgentInfo("GL-061", "BALANCE-ANALYZER", "gl_061_heat_balance_analyzer", "HeatBalanceAnalyzerAgent", "Analytics", "Analyzer", "High", "P1", "$8B"),
    AgentInfo("GL-062", "EXERGY-SCAN", "gl_062_exergy_analyzer", "ExergyAnalyzerAgent", "Analytics", "Analyzer", "High", "P2", "$7B"),
    AgentInfo("GL-063", "BENCHMARKIQ", "gl_063_benchmarking", "BenchmarkingAgent", "Analytics", "Analyzer", "Medium", "P1", "$6B"),
    AgentInfo("GL-064", "COST-ALLOCATOR", "gl_064_cost_allocation", "CostAllocationAgent", "Financial", "Calculator", "Medium", "P1", "$5B"),
    AgentInfo("GL-065", "CARBON-ACCOUNTANT", "gl_065_carbon_accounting", "CarbonAccountingAgent", "Sustainability", "Calculator", "High", "P0", "$12B"),
    AgentInfo("GL-066", "ENERGY-AUDITOR", "gl_066_energy_audit", "EnergyAuditAgent", "Compliance", "Auditor", "High", "P1", "$9B"),
    AgentInfo("GL-067", "COMMISSIONING-AGENT", "gl_067_commissioning", "ContinuousCommissioningAgent", "Optimization", "Optimizer", "High", "P1", "$10B"),
    AgentInfo("GL-068", "DIGITAL-TWIN", "gl_068_digital_twin", "DigitalTwinOrchestratorAgent", "Digital Twin", "Coordinator", "High", "P1", "$15B"),
    AgentInfo("GL-069", "DEMAND-FORECASTER", "gl_069_demand_forecast", "PredictiveDemandForecasterAgent", "Planning", "Predictor", "High", "P1", "$8B"),
    AgentInfo("GL-070", "EMERGENCY-RESPONDER", "gl_070_emergency_response", "EmergencyResponseAgent", "Safety", "Automator", "High", "P0", "$11B"),
    AgentInfo("GL-071", "REGULATORY-GUARDIAN", "gl_071_regulatory", "RegulatoryComplianceAgent", "Compliance", "Monitor", "High", "P0", "$10B"),
    AgentInfo("GL-072", "TRAINING-SIM", "gl_072_training_sim", "TrainingSimulatorAgent", "Training", "Simulator", "Medium", "P2", "$4B"),
    AgentInfo("GL-073", "MAINT-SCHEDULER", "gl_073_maint_scheduler", "MaintenanceScheduleOptimizerAgent", "Maintenance", "Optimizer", "Medium", "P1", "$7B"),
    AgentInfo("GL-074", "SPAREPARTS-IQ", "gl_074_spare_parts", "SparePartsOptimizerAgent", "Inventory", "Optimizer", "Low", "P2", "$3B"),
    AgentInfo("GL-075", "CONTRACTOR-WATCH", "gl_075_contractor", "ContractorPerformanceMonitorAgent", "Procurement", "Monitor", "Low", "P2", "$2B"),

    # Financial & Business Agents (GL-076 to GL-100)
    AgentInfo("GL-076", "VENDOR-SELECT", "gl_076_vendor_select", "VendorSelectionAgent", "Procurement", "Analyzer", "Medium", "P2", "$4B"),
    AgentInfo("GL-077", "LCA-ANALYST", "gl_077_lca", "LifecycleAssessmentAgent", "Sustainability", "Analyzer", "High", "P1", "$8B"),
    AgentInfo("GL-078", "CIRCULAR-ECONOMY", "gl_078_circular_economy", "CircularEconomyAgent", "Sustainability", "Analyzer", "Medium", "P2", "$6B"),
    AgentInfo("GL-079", "WATER-ENERGY-NEXUS", "gl_079_water_energy", "WaterEnergyNexusAgent", "Integration", "Optimizer", "High", "P2", "$9B"),
    AgentInfo("GL-080", "GRID-SERVICES", "gl_080_grid_services", "GridServicesAgent", "Grid Integration", "Coordinator", "High", "P1", "$12B"),
    AgentInfo("GL-081", "RENEWABLE-INTEGRATOR", "gl_081_renewable", "RenewableIntegrationAgent", "Renewable Energy", "Integrator", "High", "P1", "$14B"),
    AgentInfo("GL-082", "H2-PRODUCTION-HEAT", "gl_082_h2_production", "HydrogenProductionHeatAgent", "Hydrogen", "Optimizer", "High", "P1", "$13B"),
    AgentInfo("GL-083", "CCS-INTEGRATOR", "gl_083_ccs_integration", "CCSIntegrationOptimizerAgent", "Carbon Capture", "Optimizer", "High", "P0", "$16B"),
    AgentInfo("GL-084", "NET-ZERO-PATH", "gl_084_net_zero", "NetZeroPathwayAgent", "Decarbonization", "Planner", "High", "P0", "$20B"),
    AgentInfo("GL-085", "INNOVATION-SCOUT", "gl_085_innovation", "InnovationScoutAgent", "Innovation", "Analyzer", "Medium", "P2", "$5B"),
    AgentInfo("GL-086", "RISK-ASSESSOR", "gl_086_risk", "RiskAssessmentAgent", "Risk", "Analyzer", "High", "P1", "$7B"),
    AgentInfo("GL-087", "BUSINESS-BUILDER", "gl_087_business_case", "BusinessCaseBuilderAgent", "Financial", "Analyzer", "High", "P1", "$9B"),
    AgentInfo("GL-088", "INCENTIVE-MAX", "gl_088_incentive", "IncentiveMaximizerAgent", "Financial", "Optimizer", "Medium", "P1", "$8B"),
    AgentInfo("GL-089", "FINANCING-OPT", "gl_089_financing", "FinancingOptimizerAgent", "Financial", "Optimizer", "Medium", "P2", "$6B"),
    AgentInfo("GL-090", "ASSET-VALUATOR", "gl_090_asset_value", "AssetValuationAgent", "Financial", "Calculator", "Medium", "P2", "$4B"),
    AgentInfo("GL-091", "INSURANCE-OPT", "gl_091_insurance", "InsuranceOptimizerAgent", "Risk", "Optimizer", "Low", "P2", "$3B"),
    AgentInfo("GL-092", "SUPPLY-CHAIN-LINK", "gl_092_supply_chain", "SupplyChainIntegratorAgent", "Supply Chain", "Integrator", "Medium", "P2", "$5B"),
    AgentInfo("GL-093", "QUALITY-INTEGRATOR", "gl_093_quality", "ProductQualityIntegratorAgent", "Quality", "Integrator", "High", "P1", "$10B"),
    AgentInfo("GL-094", "OEE-MAXIMIZER", "gl_094_oee", "OEEMaximizerAgent", "Operations", "Optimizer", "High", "P1", "$11B"),
    AgentInfo("GL-095", "ALARM-MANAGER", "gl_095_alarm", "AlarmManagementAgent", "Operations", "Optimizer", "Medium", "P2", "$4B"),
    AgentInfo("GL-096", "CYBER-SHIELD", "gl_096_cybersecurity", "CybersecurityAgent", "Security", "Monitor", "High", "P0", "$9B"),
    AgentInfo("GL-097", "DATA-QUALITY", "gl_097_data_quality", "DataQualityAgent", "Data", "Monitor", "Medium", "P1", "$5B"),
    AgentInfo("GL-098", "INTEROP-BRIDGE", "gl_098_interoperability", "InteroperabilityAgent", "Integration", "Integrator", "High", "P1", "$8B"),
    AgentInfo("GL-099", "KNOWLEDGE-VAULT", "gl_099_knowledge", "KnowledgeManagementAgent", "Knowledge", "Coordinator", "Medium", "P2", "$6B"),
    AgentInfo("GL-100", "KAIZEN-DRIVER", "gl_100_kaizen", "ContinuousImprovementAgent", "Operations", "Coordinator", "High", "P1", "$10B"),
]


class AgentRegistry:
    """
    Central registry for all GL Process Heat agents.

    Features:
    - Dynamic agent discovery and loading
    - Lazy instantiation for memory efficiency
    - Category and type filtering
    - Metadata queries
    """

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._load_definitions()

    def _load_definitions(self):
        """Load all agent definitions into registry."""
        for agent_def in AGENT_DEFINITIONS:
            self._agents[agent_def.agent_id] = agent_def
            self._agents[agent_def.agent_name] = agent_def
        logger.info(f"Loaded {len(AGENT_DEFINITIONS)} agent definitions")

    def get_agent(self, identifier: str, config: Optional[Dict] = None) -> Any:
        """
        Get an agent instance by ID or name.

        Args:
            identifier: Agent ID (e.g., "GL-022") or name (e.g., "SUPERHEAT-CTRL")
            config: Optional configuration dict for agent

        Returns:
            Instantiated agent object
        """
        agent_info = self._agents.get(identifier)
        if not agent_info:
            raise ValueError(f"Unknown agent: {identifier}")

        cache_key = f"{agent_info.agent_id}:{hash(str(config))}"

        if cache_key not in self._instances:
            try:
                module = importlib.import_module(
                    f"agents.{agent_info.module_path}"
                )
                agent_class = getattr(module, agent_info.class_name)
                self._instances[cache_key] = agent_class(config or {})
            except Exception as e:
                logger.error(f"Failed to load agent {identifier}: {e}")
                raise

        return self._instances[cache_key]

    def get_info(self, identifier: str) -> Optional[AgentInfo]:
        """Get agent metadata by ID or name."""
        return self._agents.get(identifier)

    def list_agents(
        self,
        category: Optional[str] = None,
        agent_type: Optional[str] = None,
        priority: Optional[str] = None
    ) -> List[AgentInfo]:
        """
        List agents with optional filtering.

        Args:
            category: Filter by category (e.g., "Steam Systems")
            agent_type: Filter by type (e.g., "Optimizer")
            priority: Filter by priority (e.g., "P0")
        """
        results = []
        seen_ids = set()

        for agent_info in self._agents.values():
            if agent_info.agent_id in seen_ids:
                continue
            seen_ids.add(agent_info.agent_id)

            if category and agent_info.category != category:
                continue
            if agent_type and agent_info.agent_type != agent_type:
                continue
            if priority and agent_info.priority != priority:
                continue

            results.append(agent_info)

        return sorted(results, key=lambda x: x.agent_id)

    def get_categories(self) -> List[str]:
        """Get list of all agent categories."""
        categories = set()
        for info in self._agents.values():
            categories.add(info.category)
        return sorted(categories)

    def get_types(self) -> List[str]:
        """Get list of all agent types."""
        types = set()
        for info in self._agents.values():
            types.add(info.agent_type)
        return sorted(types)

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        agents = self.list_agents()

        by_category = {}
        by_type = {}
        by_priority = {}
        by_complexity = {}
        total_market = 0

        for agent in agents:
            by_category[agent.category] = by_category.get(agent.category, 0) + 1
            by_type[agent.agent_type] = by_type.get(agent.agent_type, 0) + 1
            by_priority[agent.priority] = by_priority.get(agent.priority, 0) + 1
            by_complexity[agent.complexity] = by_complexity.get(agent.complexity, 0) + 1

            if agent.market_size:
                # Parse market size like "$15B"
                try:
                    size = float(agent.market_size.replace("$", "").replace("B", ""))
                    total_market += size
                except:
                    pass

        return {
            "total_agents": len(agents),
            "by_category": by_category,
            "by_type": by_type,
            "by_priority": by_priority,
            "by_complexity": by_complexity,
            "total_addressable_market_billions": round(total_market, 1),
            "loaded_instances": len(self._instances)
        }

    def health_check(self) -> Dict[str, Any]:
        """Check health of all registered agents."""
        results = {
            "total": len(self.list_agents()),
            "loadable": 0,
            "failed": [],
            "status": "HEALTHY"
        }

        for agent_info in self.list_agents():
            try:
                # Try to import module
                importlib.import_module(f"agents.{agent_info.module_path}")
                results["loadable"] += 1
            except Exception as e:
                results["failed"].append({
                    "agent_id": agent_info.agent_id,
                    "error": str(e)
                })

        if results["failed"]:
            results["status"] = "DEGRADED" if results["loadable"] > 0 else "UNHEALTHY"

        return results


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Get global registry instance (singleton)."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


# Convenience functions
def get_agent(identifier: str, config: Optional[Dict] = None) -> Any:
    """Get agent by ID or name."""
    return get_registry().get_agent(identifier, config)


def list_agents(**filters) -> List[AgentInfo]:
    """List agents with optional filters."""
    return get_registry().list_agents(**filters)


def get_statistics() -> Dict[str, Any]:
    """Get registry statistics."""
    return get_registry().get_statistics()
