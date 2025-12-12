"""
GL-Agent-Factory: Unified Agent Registry

This registry provides a single point of access to all 143 Process Heat agents,
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


# Complete registry of all 143 GL Process Heat agents
AGENT_DEFINITIONS = [
    # ========================================
    # Climate & Compliance (GL-001 to GL-013)
    # ========================================
    AgentInfo("GL-001", "CARBON-EMISSIONS", "gl_001_carbon_emissions", "CarbonEmissionsAgent", "Emissions", "Calculator", "Medium", "P0"),
    AgentInfo("GL-002", "CBAM-COMPLIANCE", "gl_002_cbam_compliance", "CBAMComplianceAgent", "Compliance", "Monitor", "Medium", "P0"),
    AgentInfo("GL-003", "CSRD-REPORTING", "gl_003_csrd_reporting", "CSRDReportingAgent", "Reporting", "Reporter", "High", "P1"),
    AgentInfo("GL-004", "EUDR-COMPLIANCE", "gl_004_eudr_compliance", "EUDRComplianceAgent", "Compliance", "Monitor", "High", "P1"),
    AgentInfo("GL-005", "BUILDING-ENERGY", "gl_005_building_energy", "BuildingEnergyAgent", "Building", "Analyzer", "Medium", "P1"),
    AgentInfo("GL-006", "SCOPE3-EMISSIONS", "gl_006_scope3_emissions", "Scope3EmissionsAgent", "Emissions", "Calculator", "High", "P0"),
    AgentInfo("GL-007", "EU-TAXONOMY", "gl_007_eu_taxonomy", "EUTaxonomyAgent", "Compliance", "Analyzer", "High", "P1"),
    AgentInfo("GL-008", "GREEN-CLAIMS", "gl_008_green_claims", "GreenClaimsAgent", "Compliance", "Validator", "Medium", "P2"),
    AgentInfo("GL-009", "PRODUCT-CARBON-FOOTPRINT", "gl_009_product_carbon_footprint", "ProductCarbonFootprintAgent", "Carbon", "Calculator", "Medium", "P1"),
    AgentInfo("GL-010", "SBTI-VALIDATION", "gl_010_sbti_validation", "SBTiValidationAgent", "Compliance", "Validator", "High", "P0"),
    AgentInfo("GL-011", "CLIMATE-RISK", "gl_011_climate_risk", "ClimateRiskAgent", "Risk", "Analyzer", "High", "P1"),
    AgentInfo("GL-012", "CARBON-OFFSET", "gl_012_carbon_offset", "CarbonOffsetAgent", "Carbon", "Manager", "Medium", "P2"),
    AgentInfo("GL-013", "SB253-DISCLOSURE", "gl_013_sb253_disclosure", "SB253DisclosureAgent", "Compliance", "Reporter", "High", "P0"),

    # ========================================
    # Process Heat Foundation (GL-014 to GL-019)
    # Added: December 2025 - SHAP/LIME Explainability Enabled
    # ========================================
    AgentInfo("GL-014", "EXCHANGERPRO", "gl_014_heat_exchanger", "HeatExchangerOptimizerAgent",
              "Heat Exchangers", "Optimizer", "Medium", "P1", "$6B",
              "TEMA-compliant heat exchanger optimization with epsilon-NTU, LMTD analysis, "
              "fouling prediction, cleaning schedule optimization, SHAP/LIME explainability.",
              ["TEMA", "ASME"]),
    AgentInfo("GL-015", "INSULSCAN", "gl_015_insulation", "InsulationAnalysisAgent",
              "Energy Conservation", "Monitor", "Low", "P2", "$3B",
              "Comprehensive insulation analysis with 50+ material database, economic thickness "
              "calculations, thermal imaging integration, SHAP/LIME explainability.",
              ["ASTM C680"]),
    AgentInfo("GL-016", "WATERGUARD", "gl_016_boiler_water", "BoilerWaterTreatmentAgent",
              "Boiler Systems", "Controller", "Medium", "P1", "$5B",
              "ASME/ABMA compliant water treatment with cycles of concentration optimization, "
              "blowdown control, chemical dosing optimization, SHAP/LIME explainability.",
              ["ASME", "ABMA"]),
    AgentInfo("GL-017", "CONDENSYNC", "gl_017_condenser", "CondenserOptimizationAgent",
              "Steam Systems", "Optimizer", "Medium", "P2", "$4B",
              "HEI Standards compliant condenser optimization with cleanliness factor tracking, "
              "vacuum optimization, ML fouling prediction, SHAP/LIME explainability.",
              ["HEI"]),
    AgentInfo("GL-018", "UNIFIEDCOMBUSTION", "gl_018_unified_combustion", "UnifiedCombustionOptimizerAgent",
              "Combustion", "Optimizer", "High", "P0", "$24B",
              "NFPA 85/86 compliant unified combustion optimizer with O2 trim, CO optimization, "
              "excess air control, safety interlocks, causal inference, SHAP/LIME explainability.",
              ["NFPA 85", "NFPA 86"]),
    AgentInfo("GL-019", "HEATSCHEDULER", "gl_019_heat_scheduler", "ProcessHeatingSchedulerAgent",
              "Planning", "Coordinator", "High", "P1", "$7B",
              "ML-based demand forecasting with thermal storage optimization, TOU tariff arbitrage, "
              "uncertainty quantification, SSE streaming, SHAP/LIME explainability.",
              ["ISO 50001", "OpenADR 2.0"]),

    # ========================================
    # Process Heat Baseline (GL-020 to GL-030)
    # ========================================
    AgentInfo("GL-020", "ECONOMIZER-PERFORMANCE", "gl_020_economizer_performance", "EconomizerPerformanceAgent", "Heat Recovery", "Monitor", "High", "P2"),
    AgentInfo("GL-021", "BURNER-MAINTENANCE", "gl_021_burner_maintenance", "BurnerMaintenancePredictorAgent", "Maintenance", "Predictor", "Medium", "P2"),
    AgentInfo("GL-022", "SUPERHEAT-CTRL", "gl_022_superheater_control", "SuperheaterControlAgent", "Steam Systems", "Controller", "Medium", "P2", "$5B"),
    AgentInfo("GL-023", "LOADBALANCER", "gl_023_heat_load_balancer", "HeatLoadBalancerAgent", "Optimization", "Optimizer", "High", "P1", "$9B"),
    AgentInfo("GL-024", "AIRPREHEATER", "gl_024_air_preheater", "AirPreheaterOptimizerAgent", "Heat Recovery", "Optimizer", "Medium", "P2", "$4B"),
    AgentInfo("GL-025", "COGENMAX", "gl_025_cogeneration", "CogenerationOptimizerAgent", "Cogeneration", "Optimizer", "High", "P0", "$15B"),
    AgentInfo("GL-026", "SOOTBLAST", "gl_026_soot_blower", "SootBlowerControllerAgent", "Combustion", "Controller", "Low", "P2", "$2B"),
    AgentInfo("GL-027", "RADIANT-OPT", "gl_027_radiant_heat", "RadiantHeatOptimizerAgent", "Furnaces", "Optimizer", "Medium", "P2", "$5B"),
    AgentInfo("GL-028", "CONVECTION-WATCH", "gl_028_convection_analyzer", "ConvectionSectionAnalyzerAgent", "Furnaces", "Analyzer", "Medium", "P2", "$4B"),
    AgentInfo("GL-029", "FUELCONDITIONER", "gl_029_fuel_conditioner", "FuelGasConditioningAgent", "Fuel Systems", "Controller", "Low", "P2", "$3B"),
    AgentInfo("GL-030", "HEATINTEGRATOR", "gl_030_heat_integrator", "HeatIntegrationOptimizerAgent", "Process Integration", "Optimizer", "High", "P1", "$12B"),

    # ========================================
    # Safety & Optimization Agents (GL-031 to GL-045)
    # With duplicate implementations for some IDs
    # ========================================
    # GL-031: Two implementations
    AgentInfo("GL-031", "FURNACE-GUARDIAN", "gl_031_furnace_guardian", "FurnaceGuardianAgent", "Furnaces", "Monitor", "High", "P0", "$8B"),
    AgentInfo("GL-031B", "THERMAL-STORAGE", "gl_031_thermal_storage", "ThermalEnergyStorageAgent", "Energy Storage", "Controller", "High", "P2", "$8B"),

    # GL-032: Two implementations
    AgentInfo("GL-032", "HEAT-REPORTER", "gl_032_heat_reporter", "HeatReporterAgent", "Reporting", "Reporter", "Medium", "P1", "$6B"),
    AgentInfo("GL-032B", "REFRACTORY-MONITOR", "gl_032_refractory_monitor", "RefractoryMonitorAgent", "Furnaces", "Monitor", "Medium", "P2", "$4B"),

    # GL-033: Two implementations
    AgentInfo("GL-033", "BURNER-BALANCER", "gl_033_burner_balancer", "BurnerBalancerAgent", "Combustion", "Controller", "Medium", "P2", "$5B"),
    AgentInfo("GL-033B", "DISTRICT-HEATING", "gl_033_district_heating", "DistrictHeatingAgent", "Heat Networks", "Integrator", "High", "P2", "$10B"),

    # GL-034: Two implementations
    AgentInfo("GL-034", "CARBON-CAPTURE-HEAT", "gl_034_carbon_capture_heat", "CarbonCaptureHeatAgent", "Decarbonization", "Optimizer", "High", "P0", "$18B"),
    AgentInfo("GL-034B", "HEAT-RECOVERY-SCOUT", "gl_034_heat_recovery_scout", "HeatRecoveryScoutAgent", "Heat Recovery", "Analyzer", "Medium", "P2", "$6B"),

    # GL-035: Two implementations
    AgentInfo("GL-035", "HYDROGEN-BURNER", "gl_035_hydrogen_burner", "HydrogenBurnerAgent", "Future Fuels", "Controller", "High", "P1", "$14B"),
    AgentInfo("GL-035B", "THERMAL-STORAGE-OPT", "gl_035_thermal_storage_optimizer", "ThermalStorageOptimizerAgent", "Energy Storage", "Optimizer", "High", "P2", "$8B"),

    # GL-036: Two implementations
    AgentInfo("GL-036", "ELECTRIFICATION", "gl_036_electrification", "ElectrificationAgent", "Decarbonization", "Analyzer", "High", "P1", "$16B"),
    AgentInfo("GL-036B", "CHP-COORDINATOR", "gl_036_chp_coordinator", "CHPCoordinatorAgent", "Cogeneration", "Coordinator", "High", "P1", "$12B"),

    # GL-037: Two implementations
    AgentInfo("GL-037", "BIOMASS", "gl_037_biomass", "BiomassCombustionAgent", "Renewable Heat", "Optimizer", "Medium", "P2", "$7B"),
    AgentInfo("GL-037B", "FLARE-MINIMIZER", "gl_037_flare_minimizer", "FlareMinimzerAgent", "Emissions Control", "Controller", "Medium", "P2", "$5B"),

    # GL-038: Two implementations
    AgentInfo("GL-038", "SOLAR-THERMAL", "gl_038_solar_thermal", "SolarThermalAgent", "Renewable Heat", "Integrator", "Medium", "P2", "$9B"),
    AgentInfo("GL-038B", "INSULATION-AUDITOR", "gl_038_insulation_auditor", "InsulationAuditorAgent", "Efficiency", "Auditor", "Medium", "P2", "$4B"),

    # GL-039: Two implementations
    AgentInfo("GL-039", "HEAT-PUMP", "gl_039_heat_pump", "HeatPumpOptimizerAgent", "Heat Pumps", "Optimizer", "High", "P1", "$11B"),
    AgentInfo("GL-039B", "ENERGY-BENCHMARK", "gl_039_energy_benchmark", "EnergyBenchmarkAgent", "Analytics", "Analyzer", "Medium", "P2", "$6B"),

    # GL-040: Two implementations
    AgentInfo("GL-040", "SAFETY-MONITOR", "gl_040_safety_monitor", "ProcessSafetyMonitorAgent", "Safety", "Monitor", "High", "P0", "$13B"),
    AgentInfo("GL-040B", "LOAD-FORECASTER", "gl_040_load_forecaster", "LoadForecasterAgent", "Planning", "Predictor", "Medium", "P2", "$6B"),

    # GL-041: Two implementations
    AgentInfo("GL-041", "ENERGY-DASHBOARD", "gl_041_energy_dashboard", "EnergyVizAgent", "Visualization", "Reporter", "Medium", "P1", "$8B"),
    AgentInfo("GL-041B", "STARTUP-OPTIMIZER", "gl_041_startup_optimizer", "StartupOptimizerAgent", "Operations", "Optimizer", "Medium", "P2", "$5B"),

    # GL-042: Two implementations
    AgentInfo("GL-042", "STEAM-PRESSURE", "gl_042_steam_pressure", "PressureMasterAgent", "Steam Systems", "Optimizer", "Medium", "P2", "$5B"),
    AgentInfo("GL-042B", "PRESSUREMASTER", "gl_042_pressuremaster", "PressureMasterAgent", "Steam Systems", "Controller", "Medium", "P2", "$5B"),

    # GL-043: Two implementations
    AgentInfo("GL-043", "CONDENSATE-RECOVERY", "gl_043_condensate_recovery", "CondensateReclaimAgent", "Steam Systems", "Monitor", "Low", "P2", "$3B"),
    AgentInfo("GL-043B", "VENT-CONDENSER-OPT", "gl_043_vent_condenser_opt", "VentCondenserOptAgent", "Steam Systems", "Optimizer", "Low", "P2", "$3B"),

    # GL-044: Two implementations
    AgentInfo("GL-044", "FLASH-STEAM", "gl_044_flash_steam", "FlashSteamAgent", "Heat Recovery", "Optimizer", "Medium", "P2", "$4B"),
    AgentInfo("GL-044B", "WATER-TREATMENT", "gl_044_water_treatment_advisor", "WaterTreatmentAdvisorAgent", "Water", "Advisor", "Medium", "P2", "$4B"),

    # GL-045: Single implementation
    AgentInfo("GL-045", "CARBON-INTENSITY", "gl_045_carbon_intensity_tracker", "CarbonIntensityTrackerAgent", "Emissions", "Tracker", "Medium", "P1", "$6B"),

    # ========================================
    # Analytics Agents (GL-046 to GL-060)
    # ========================================
    AgentInfo("GL-046", "DRAFT-CONTROL", "gl_046_draft_control", "DraftControlAgent", "Furnaces", "Controller", "Low", "P2", "$3B"),
    AgentInfo("GL-047", "REFRACTORY", "gl_047_refractory", "RefractoryAgent", "Furnaces", "Monitor", "Medium", "P2", "$4B"),
    AgentInfo("GL-048", "HEAT-LOSS", "gl_048_heat_loss", "HeatLossAgent", "Analytics", "Calculator", "Medium", "P2", "$5B"),
    AgentInfo("GL-049", "PROCESS-CONTROL", "gl_049_process_control", "ProcessControlAgent", "Integration", "Integrator", "High", "P1", "$10B"),
    AgentInfo("GL-050", "VFD", "gl_050_vfd", "VFDAgent", "Motors", "Optimizer", "Medium", "P2", "$6B"),
    AgentInfo("GL-051", "STARTUP-SHUTDOWN", "gl_051_startup_shutdown", "StartupShutdownAgent", "Operations", "Optimizer", "Medium", "P2", "$5B"),
    AgentInfo("GL-052", "HEAT-TRACING", "gl_052_heat_tracing", "HeatTracingAgent", "Heat Tracing", "Optimizer", "Low", "P2", "$3B"),
    AgentInfo("GL-053", "THERMAL-OXIDIZER", "gl_053_thermal_oxidizer", "ThermalOxidizerAgent", "Emissions Control", "Optimizer", "Medium", "P2", "$7B"),
    AgentInfo("GL-054", "HEAT-TREATMENT", "gl_054_heat_treatment", "HeatTreatmentAgent", "Process", "Optimizer", "High", "P1", "$9B"),
    AgentInfo("GL-055", "DRYING", "gl_055_drying", "DryingAgent", "Process", "Optimizer", "Medium", "P1", "$8B"),
    AgentInfo("GL-056", "CURING-OVEN", "gl_056_curing_oven", "CuringOvenAgent", "Process", "Controller", "Medium", "P2", "$5B"),
    AgentInfo("GL-057", "INDUCTION-HEATING", "gl_057_induction_heating", "InductionHeatingAgent", "Process", "Optimizer", "Medium", "P2", "$6B"),
    AgentInfo("GL-058", "INFRARED-HEATING", "gl_058_infrared_heating", "InfraredHeatingAgent", "Process", "Controller", "Low", "P2", "$4B"),
    AgentInfo("GL-059", "MICROWAVE-HEATING", "gl_059_microwave_heating", "MicrowaveHeatingAgent", "Process", "Controller", "Medium", "P3", "$5B"),
    AgentInfo("GL-060", "RESISTANCE-HEATING", "gl_060_resistance_heating", "ResistanceHeatingAgent", "Process", "Optimizer", "Low", "P2", "$3B"),

    # ========================================
    # Digital Twin & Simulation Agents (GL-061 to GL-075)
    # ========================================
    AgentInfo("GL-061", "HEAT-BALANCE-ANALYZER", "gl_061_heat_balance_analyzer", "HeatBalanceAnalyzerAgent", "Analytics", "Analyzer", "High", "P1", "$8B"),

    # GL-062: Two implementations
    AgentInfo("GL-062", "EXERGY-ANALYZER", "gl_062_exergy_analyzer", "ExergyAnalyzerAgent", "Analytics", "Analyzer", "High", "P2", "$7B"),
    AgentInfo("GL-062B", "EXERGY-SCAN", "gl_062_exergy_scan", "ExergyScanAgent", "Analytics", "Scanner", "High", "P2", "$7B"),

    # GL-063: Two implementations
    AgentInfo("GL-063", "BENCHMARKING", "gl_063_benchmarking", "BenchmarkingAgent", "Analytics", "Analyzer", "Medium", "P1", "$6B"),
    AgentInfo("GL-063B", "PINCH-ANALYZER", "gl_063_pinch_analyzer", "PinchAnalyzerAgent", "Process Integration", "Analyzer", "High", "P1", "$8B"),

    # GL-064: Two implementations
    AgentInfo("GL-064", "COST-ALLOCATION", "gl_064_cost_allocation", "CostAllocationAgent", "Financial", "Calculator", "Medium", "P1", "$5B"),
    AgentInfo("GL-064B", "PROCESS-SIMULATOR", "gl_064_process_simulator", "ProcessSimulatorAgent", "Simulation", "Simulator", "High", "P2", "$8B"),

    # GL-065: Two implementations
    AgentInfo("GL-065", "CARBON-ACCOUNTING", "gl_065_carbon_accounting", "CarbonAccountingAgent", "Sustainability", "Calculator", "High", "P0", "$12B"),
    AgentInfo("GL-065B", "CARBON-ACCOUNTANT", "gl_065_carbon_accountant", "CarbonAccountantAgent", "Sustainability", "Calculator", "High", "P0", "$12B"),

    # GL-066: Two implementations
    AgentInfo("GL-066", "ENERGY-AUDIT", "gl_066_energy_audit", "EnergyAuditAgent", "Compliance", "Auditor", "High", "P1", "$9B"),
    AgentInfo("GL-066B", "SCENARIO-MODELER", "gl_066_scenario_modeler", "ScenarioModelerAgent", "Planning", "Modeler", "High", "P2", "$7B"),

    # GL-067: Two implementations
    AgentInfo("GL-067", "COMMISSIONING", "gl_067_commissioning", "CommissioningAgent", "Optimization", "Optimizer", "High", "P1", "$10B"),
    AgentInfo("GL-067B", "CONTINUOUS-COMMISSIONING", "gl_067_continuous_commissioning", "ContinuousCommissioningAgent", "Optimization", "Optimizer", "High", "P1", "$10B"),

    AgentInfo("GL-068", "DIGITAL-TWIN", "gl_068_digital_twin", "DigitalTwinAgent", "Digital Twin", "Coordinator", "High", "P1", "$15B"),

    # GL-069: Two implementations
    AgentInfo("GL-069", "DEMAND-FORECAST", "gl_069_demand_forecast", "DemandForecastAgent", "Planning", "Predictor", "High", "P1", "$8B"),
    AgentInfo("GL-069B", "LIFECYCLE-ANALYZER", "gl_069_lifecycle_analyzer", "LifecycleAnalyzerAgent", "Sustainability", "Analyzer", "High", "P2", "$7B"),

    # GL-070: Two implementations
    AgentInfo("GL-070", "EMERGENCY-RESPONSE", "gl_070_emergency_response", "EmergencyResponseAgent", "Safety", "Automator", "High", "P0", "$11B"),
    AgentInfo("GL-070B", "DECARBONIZATION-PLANNER", "gl_070_decarbonization_planner", "DecarbonizationPlannerAgent", "Decarbonization", "Planner", "High", "P1", "$15B"),

    # GL-071: Two implementations
    AgentInfo("GL-071", "REGULATORY", "gl_071_regulatory", "RegulatoryGuardianAgent", "Compliance", "Monitor", "High", "P0", "$10B"),
    AgentInfo("GL-071B", "HYDROGEN-READINESS", "gl_071_hydrogen_readiness", "HydrogenReadinessAgent", "Future Fuels", "Analyzer", "High", "P1", "$12B"),

    AgentInfo("GL-072", "TRAINING-SIM", "gl_072_training_sim", "TrainingSimAgent", "Training", "Simulator", "Medium", "P2", "$4B"),

    # GL-073: Two implementations
    AgentInfo("GL-073", "MAINT-SCHEDULER", "gl_073_maint_scheduler", "MaintenanceSchedulerAgent", "Maintenance", "Optimizer", "Medium", "P1", "$7B"),
    AgentInfo("GL-073B", "EMISSIONS-PREDICTOR", "gl_073_emissions_predictor", "EmissionsPredictorAgent", "Emissions", "Predictor", "High", "P1", "$8B"),

    # GL-074: Two implementations
    AgentInfo("GL-074", "SPARE-PARTS", "gl_074_spare_parts", "SparePartsOptimizerAgent", "Inventory", "Optimizer", "Low", "P2", "$3B"),
    AgentInfo("GL-074B", "GRID-OPTIMIZER", "gl_074_grid_optimizer", "GridOptimizerAgent", "Grid Integration", "Optimizer", "High", "P1", "$10B"),

    # GL-075: Two implementations
    AgentInfo("GL-075", "CONTRACTOR", "gl_075_contractor", "ContractorAgent", "Procurement", "Monitor", "Low", "P2", "$2B"),
    AgentInfo("GL-075B", "WATER-ENERGY-NEXUS", "gl_075_water_energy_nexus", "WaterEnergyNexusAgent", "Integration", "Analyzer", "High", "P1", "$9B"),

    # ========================================
    # Financial & Business Agents (GL-076 to GL-100)
    # ========================================
    # GL-076: Two implementations
    AgentInfo("GL-076", "VENDOR-SELECT", "gl_076_vendor_select", "VendorSelectAgent", "Procurement", "Analyzer", "Medium", "P2", "$4B"),
    AgentInfo("GL-076B", "CARBON-MARKET-TRADER", "gl_076_carbon_market_trader", "CarbonMarketTraderAgent", "Carbon", "Trader", "High", "P1", "$12B"),

    # GL-077: Two implementations
    AgentInfo("GL-077", "LCA", "gl_077_lca", "LCAAgent", "Sustainability", "Analyzer", "High", "P1", "$8B"),
    AgentInfo("GL-077B", "INCENTIVE-HUNTER", "gl_077_incentive_hunter", "IncentiveHunterAgent", "Financial", "Optimizer", "Medium", "P2", "$5B"),

    # GL-078: Two implementations
    AgentInfo("GL-078", "CIRCULAR-ECONOMY", "gl_078_circular_economy", "CircularEconomyAgent", "Sustainability", "Analyzer", "Medium", "P2", "$6B"),
    AgentInfo("GL-078B", "TARIFF-OPTIMIZER", "gl_078_tariff_optimizer", "TariffOptimizerAgent", "Financial", "Optimizer", "Medium", "P2", "$5B"),

    # GL-079: Two implementations
    AgentInfo("GL-079", "WATER-ENERGY", "gl_079_water_energy", "WaterEnergyAgent", "Integration", "Optimizer", "High", "P2", "$9B"),
    AgentInfo("GL-079B", "CAPEX-ANALYZER", "gl_079_capex_analyzer", "CapexAnalyzerAgent", "Financial", "Calculator", "High", "P1", "$10B"),

    # GL-080: Two implementations
    AgentInfo("GL-080", "GRID-SERVICES", "gl_080_grid_services", "GridServicesAgent", "Grid Integration", "Coordinator", "High", "P1", "$12B"),
    AgentInfo("GL-080B", "OPEX-OPTIMIZER", "gl_080_opex_optimizer", "OpexOptimizerAgent", "Financial", "Optimizer", "Medium", "P1", "$8B"),

    # GL-081: Two implementations
    AgentInfo("GL-081", "RENEWABLE", "gl_081_renewable", "RenewableIntegrationAgent", "Renewable Energy", "Integrator", "High", "P1", "$14B"),
    AgentInfo("GL-081B", "BUDGET-FORECASTER", "gl_081_budget_forecaster", "BudgetForecasterAgent", "Financial", "Predictor", "Medium", "P2", "$5B"),

    # GL-082: Two implementations
    AgentInfo("GL-082", "H2-PRODUCTION", "gl_082_h2_production", "HydrogenProductionHeatAgent", "Hydrogen", "Optimizer", "High", "P1", "$13B"),
    AgentInfo("GL-082B", "ROI-CALCULATOR", "gl_082_roi_calculator", "ROICalculatorAgent", "Financial", "Calculator", "Medium", "P2", "$4B"),

    # GL-083: Two implementations
    AgentInfo("GL-083", "CCS-INTEGRATION", "gl_083_ccs_integration", "CCSIntegrationAgent", "Carbon Capture", "Optimizer", "High", "P0", "$16B"),
    AgentInfo("GL-083B", "PROCUREMENT-ADVISOR", "gl_083_procurement_advisor", "ProcurementAdvisorAgent", "Procurement", "Advisor", "Medium", "P2", "$4B"),

    # GL-084: Two implementations
    AgentInfo("GL-084", "NET-ZERO", "gl_084_net_zero", "NetZeroAgent", "Decarbonization", "Planner", "High", "P0", "$20B"),
    AgentInfo("GL-084B", "SUSTAINABILITY-REPORTER", "gl_084_sustainability_reporter", "SustainabilityReporterAgent", "Reporting", "Reporter", "High", "P1", "$8B"),

    # GL-085: Two implementations
    AgentInfo("GL-085", "BENCHMARK-COMPARATOR", "gl_085_benchmark_comparator", "BenchmarkComparatorAgent", "Analytics", "Analyzer", "Medium", "P2", "$5B"),
    AgentInfo("GL-085B", "INNOVATION", "gl_085_innovation", "InnovationScoutAgent", "Innovation", "Analyzer", "Medium", "P2", "$5B"),

    # GL-086: Two implementations
    AgentInfo("GL-086", "RISK", "gl_086_risk", "RiskAssessmentAgent", "Risk", "Analyzer", "High", "P1", "$7B"),
    AgentInfo("GL-086B", "RISK-ASSESSOR", "gl_086_risk_assessor", "RiskAssessorAgent", "Risk", "Assessor", "High", "P1", "$7B"),

    # GL-087: Two implementations
    AgentInfo("GL-087", "BUSINESS-BUILDER", "gl_087_business_builder", "BusinessBuilderAgent", "Financial", "Analyzer", "High", "P1", "$9B"),
    AgentInfo("GL-087B", "BUSINESS-CASE", "gl_087_business_case", "BusinessCaseAgent", "Financial", "Builder", "High", "P1", "$9B"),

    # GL-088: Two implementations
    AgentInfo("GL-088", "INCENTIVE", "gl_088_incentive", "IncentiveMaximizerAgent", "Financial", "Optimizer", "Medium", "P1", "$8B"),
    AgentInfo("GL-088B", "VENDOR-EVALUATOR", "gl_088_vendor_evaluator", "VendorEvaluatorAgent", "Procurement", "Evaluator", "Medium", "P2", "$4B"),

    # GL-089: Two implementations
    AgentInfo("GL-089", "FINANCING", "gl_089_financing", "FinancingAgent", "Financial", "Optimizer", "Medium", "P2", "$6B"),
    AgentInfo("GL-089B", "FINANCING-OPTIMIZER", "gl_089_financing_optimizer", "FinancingOptimizerAgent", "Financial", "Optimizer", "Medium", "P2", "$6B"),

    # GL-090: Two implementations
    AgentInfo("GL-090", "ASSET-VALUATION", "gl_090_asset_valuation", "AssetValuationAgent", "Financial", "Calculator", "Medium", "P2", "$4B"),
    AgentInfo("GL-090B", "ASSET-VALUE", "gl_090_asset_value", "AssetValueAgent", "Financial", "Calculator", "Medium", "P2", "$4B"),

    # GL-091: Two implementations
    AgentInfo("GL-091", "INSURANCE", "gl_091_insurance", "InsuranceOptimizerAgent", "Risk", "Optimizer", "Low", "P2", "$3B"),
    AgentInfo("GL-091B", "MA-ANALYZER", "gl_091_ma_analyzer", "MAAnalyzerAgent", "Financial", "Analyzer", "High", "P2", "$8B"),

    # GL-092: Two implementations
    AgentInfo("GL-092", "SUPPLY-CHAIN", "gl_092_supply_chain", "SupplyChainIntegratorAgent", "Supply Chain", "Integrator", "Medium", "P2", "$5B"),
    AgentInfo("GL-092B", "INSURANCE-ADVISOR", "gl_092_insurance_advisor", "InsuranceAdvisorAgent", "Risk", "Advisor", "Low", "P2", "$3B"),

    # GL-093: Two implementations
    AgentInfo("GL-093", "QUALITY", "gl_093_quality", "ProductQualityIntegratorAgent", "Quality", "Integrator", "High", "P1", "$10B"),
    AgentInfo("GL-093B", "TAX-OPTIMIZER", "gl_093_tax_optimizer", "TaxOptimizerAgent", "Financial", "Optimizer", "Medium", "P2", "$5B"),

    # GL-094: Two implementations
    AgentInfo("GL-094", "OEE", "gl_094_oee", "OEEMaximizerAgent", "Operations", "Optimizer", "High", "P1", "$11B"),
    AgentInfo("GL-094B", "STAKEHOLDER-REPORTER", "gl_094_stakeholder_reporter", "StakeholderReporterAgent", "Reporting", "Reporter", "Medium", "P2", "$4B"),

    # GL-095: Two implementations
    AgentInfo("GL-095", "ALARM", "gl_095_alarm", "AlarmManagementAgent", "Operations", "Optimizer", "Medium", "P2", "$4B"),
    AgentInfo("GL-095B", "STRATEGIC-PLANNER", "gl_095_strategic_planner", "StrategicPlannerAgent", "Planning", "Planner", "High", "P1", "$9B"),

    # GL-096: Two implementations
    AgentInfo("GL-096", "CYBER-SHIELD", "gl_096_cyber_shield", "CyberShieldAgent", "Security", "Monitor", "High", "P0", "$9B"),
    AgentInfo("GL-096B", "CYBERSECURITY", "gl_096_cybersecurity", "CybersecurityAgent", "Security", "Monitor", "High", "P0", "$9B"),

    # GL-097: Two implementations
    AgentInfo("GL-097", "DATA-QUALITY", "gl_097_data_quality", "DataQualityAgent", "Data", "Monitor", "Medium", "P1", "$5B"),
    AgentInfo("GL-097B", "REGULATORY-TRACKER", "gl_097_regulatory_tracker", "RegulatoryTrackerAgent", "Compliance", "Tracker", "High", "P1", "$7B"),

    # GL-098: Two implementations
    AgentInfo("GL-098", "INNOVATION-SCOUT", "gl_098_innovation_scout", "InnovationScoutAgent", "Innovation", "Analyzer", "Medium", "P2", "$5B"),
    AgentInfo("GL-098B", "INTEROPERABILITY", "gl_098_interoperability", "InteroperabilityAgent", "Integration", "Integrator", "High", "P1", "$8B"),

    # GL-099: Two implementations
    AgentInfo("GL-099", "KNOWLEDGE", "gl_099_knowledge", "KnowledgeAgent", "Knowledge", "Coordinator", "Medium", "P2", "$6B"),
    AgentInfo("GL-099B", "WORKFORCE-PLANNER", "gl_099_workforce_planner", "WorkforcePlannerAgent", "HR", "Planner", "Medium", "P2", "$5B"),

    # GL-100: Two implementations
    AgentInfo("GL-100", "KAIZEN", "gl_100_kaizen", "KaizenAgent", "Operations", "Coordinator", "High", "P1", "$10B"),
    AgentInfo("GL-100B", "KAIZEN-DRIVER", "gl_100_kaizen_driver", "KaizenDriverAgent", "Operations", "Driver", "High", "P1", "$10B"),
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
