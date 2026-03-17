# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Engines Module
====================================================

Calculation engines for manufacturing companies implementing CSRD disclosures
under ESRS E1-E5 with sector-specific manufacturing requirements. This pack
covers process emissions (EU ETS/CBAM), energy intensity (EED/BAT), product
carbon footprints (ISO 14067/DPP), circular economy (WFD/EPR), water pollution
(IED/WFD), BAT compliance (IED/BREF), supply chain emissions (Scope 3), and
manufacturing benchmarking (SBTi sectoral pathways).

Engines:
    1. ProcessEmissionsEngine           - Direct process emissions (ETS/CBAM scope)
    2. EnergyIntensityEngine            - Energy intensity metrics and EED compliance
    3. ProductCarbonFootprintEngine      - ISO 14067 PCF and Digital Product Passport
    4. CircularEconomyEngine            - Material circularity and EPR compliance
    5. WaterPollutionEngine             - Water intake/discharge and IED pollutant limits
    6. BATComplianceEngine              - BAT-AEL compliance and BREF assessment
    7. SupplyChainEmissionsEngine       - Scope 3 upstream supply chain emissions
    8. ManufacturingBenchmarkEngine     - Sector benchmarking and SBTi trajectory

Regulatory Basis:
    EU Directive 2022/2464 (CSRD)
    EU Delegated Regulation 2023/2772 (ESRS)
    EU Regulation 2023/956 (CBAM)
    EU Directive 2023/1791 (Energy Efficiency Directive recast)
    EU Directive 2010/75/EU (Industrial Emissions Directive)
    EU Directive 2008/98/EC (Waste Framework Directive)
    EU Regulation 2024/1781 (Ecodesign for Sustainable Products)
    ISO 14067:2018 (Carbon Footprint of Products)

Pack Tier: Sector-Specific (PACK-013)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-013"
__pack_name__: str = "CSRD Manufacturing Pack"
__engines_count__: int = 8

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Process Emissions
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "ProcessEmissionsEngine",
    "ProcessEmissionsConfig",
    "FacilityData",
    "ProcessLine",
    "RawMaterial",
    "FuelConsumption",
    "ProcessEmissionsResult",
    "CBAMEmbeddedEmissions",
    "ETSBenchmarkComparison",
    "AbatementRecord",
    "ManufacturingSubSector",
    "ProcessType",
    "FuelType",
    "PROCESS_EMISSION_FACTORS",
    "ETS_PRODUCT_BENCHMARKS",
    "FUEL_EMISSION_FACTORS",
]

try:
    from .process_emissions_engine import (
        AbatementRecord,
        CBAMEmbeddedEmissions,
        ETS_PRODUCT_BENCHMARKS,
        ETSBenchmarkComparison,
        FUEL_EMISSION_FACTORS,
        PROCESS_EMISSION_FACTORS,
        FacilityData,
        FuelConsumption,
        FuelType,
        ManufacturingSubSector,
        ProcessEmissionsConfig,
        ProcessEmissionsEngine,
        ProcessEmissionsResult,
        ProcessLine,
        ProcessType,
        RawMaterial,
    )
    _loaded_engines.append("ProcessEmissionsEngine")
except ImportError as e:
    logger.debug("Engine 1 (ProcessEmissionsEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Energy Intensity
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "EnergyIntensityEngine",
    "EnergyIntensityConfig",
    "EnergyConsumptionData",
    "ProductionVolumeData",
    "FacilityEnergyData",
    "EnergyIntensityResult",
    "BenchmarkComparison",
    "EEDCompliance",
    "DecarbonizationOpportunity",
    "EnergySource",
    "ProductionUnit",
    "EEDTier",
    "BAT_ENERGY_BENCHMARKS",
]

try:
    from .energy_intensity_engine import (
        BAT_ENERGY_BENCHMARKS,
        BenchmarkComparison,
        DecarbonizationOpportunity,
        EEDCompliance,
        EEDTier,
        EnergyConsumptionData,
        EnergyIntensityConfig,
        EnergyIntensityEngine,
        EnergyIntensityResult,
        EnergySource,
        FacilityEnergyData,
        ProductionUnit,
        ProductionVolumeData,
    )
    _loaded_engines.append("EnergyIntensityEngine")
except ImportError as e:
    logger.debug("Engine 2 (EnergyIntensityEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Product Carbon Footprint
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "ProductCarbonFootprintEngine",
    "PCFConfig",
    "ProductData",
    "BOMComponent",
    "ManufacturingProcess",
    "DistributionData",
    "UsePhaseData",
    "EndOfLifeData",
    "PCFResult",
    "DPPData",
    "LifecycleScope",
    "AllocationMethod",
    "LifecycleStage",
    "DataQualityLevel",
    "MATERIAL_EMISSION_FACTORS",
    "TRANSPORT_EMISSION_FACTORS",
]

try:
    from .product_carbon_footprint_engine import (
        MATERIAL_EMISSION_FACTORS,
        TRANSPORT_EMISSION_FACTORS,
        AllocationMethod,
        BOMComponent,
        DPPData,
        DataQualityLevel,
        DistributionData,
        EndOfLifeData,
        LifecycleScope,
        LifecycleStage,
        ManufacturingProcess,
        PCFConfig,
        PCFResult,
        ProductCarbonFootprintEngine,
        ProductData,
        UsePhaseData,
    )
    _loaded_engines.append("ProductCarbonFootprintEngine")
except ImportError as e:
    logger.debug("Engine 3 (ProductCarbonFootprintEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Circular Economy
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "CircularEconomyEngine",
    "CircularEconomyConfig",
    "MaterialFlowData",
    "WasteStreamData",
    "ProductRecyclability",
    "CircularEconomyResult",
    "MCIResult",
    "WasteCategory",
    "WasteType",
    "WasteDestination",
    "EPRScheme",
    "WASTE_HIERARCHY_WEIGHTS",
    "EPR_RECYCLING_TARGETS",
    "CRM_RECYCLING_TARGETS",
]

try:
    from .circular_economy_engine import (
        CRM_RECYCLING_TARGETS,
        EPR_RECYCLING_TARGETS,
        WASTE_HIERARCHY_WEIGHTS,
        CircularEconomyConfig,
        CircularEconomyEngine,
        CircularEconomyResult,
        EPRScheme,
        MCIResult,
        MaterialFlowData,
        ProductRecyclability,
        WasteCategory,
        WasteDestination,
        WasteStreamData,
        WasteType,
    )
    _loaded_engines.append("CircularEconomyEngine")
except ImportError as e:
    logger.debug("Engine 4 (CircularEconomyEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Water Pollution
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "WaterPollutionEngine",
    "WaterPollutionConfig",
    "WaterIntakeData",
    "WaterDischargeData",
    "PollutantEmission",
    "SVHCSubstance",
    "WaterPollutionResult",
    "WaterStressAssessment",
    "WaterSource",
    "WaterStressLevel",
    "PollutantCategory",
    "PollutantType",
    "WATER_STRESS_THRESHOLDS",
    "IED_EMISSION_LIMITS",
]

try:
    from .water_pollution_engine import (
        IED_EMISSION_LIMITS,
        WATER_STRESS_THRESHOLDS,
        PollutantCategory,
        PollutantEmission,
        PollutantType,
        SVHCSubstance,
        WaterDischargeData,
        WaterIntakeData,
        WaterPollutionConfig,
        WaterPollutionEngine,
        WaterPollutionResult,
        WaterSource,
        WaterStressAssessment,
        WaterStressLevel,
    )
    _loaded_engines.append("WaterPollutionEngine")
except ImportError as e:
    logger.debug("Engine 5 (WaterPollutionEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: BAT Compliance
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "BATComplianceEngine",
    "BATConfig",
    "FacilityBATData",
    "BREFReference",
    "MeasuredParameter",
    "BATComplianceResult",
    "ParameterResult",
    "TransformationPlan",
    "AbatementOption",
    "BREFDocument",
    "ComplianceStatus",
    "TechnologyReadinessLevel",
    "BAT_AEL_DATABASE",
    "IED_PENALTY_MINIMUM",
]

try:
    from .bat_compliance_engine import (
        BAT_AEL_DATABASE,
        IED_PENALTY_MINIMUM,
        AbatementOption,
        BATComplianceEngine,
        BATComplianceResult,
        BATConfig,
        BREFDocument,
        BREFReference,
        ComplianceStatus,
        FacilityBATData,
        MeasuredParameter,
        ParameterResult,
        TechnologyReadinessLevel,
        TransformationPlan,
    )
    _loaded_engines.append("BATComplianceEngine")
except ImportError as e:
    logger.debug("Engine 6 (BATComplianceEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Supply Chain Emissions
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "SupplyChainEmissionsEngine",
    "SupplyChainConfig",
    "SupplierData",
    "BOMEmissionData",
    "TransportData",
    "SupplyChainResult",
    "SupplierHotspot",
    "EngagementRecommendation",
    "CalculationMethod",
    "Scope3Category",
    "SupplierTier",
    "DataQualityScore",
    "SPEND_EMISSION_FACTORS",
    "SC_MATERIAL_EMISSION_FACTORS",
]

try:
    from .supply_chain_emissions_engine import (
        MATERIAL_EMISSION_FACTORS as SC_MATERIAL_EMISSION_FACTORS,
        SPEND_EMISSION_FACTORS,
        BOMEmissionData,
        CalculationMethod,
        DataQualityScore,
        EngagementRecommendation,
        Scope3Category,
        SupplierData,
        SupplierHotspot,
        SupplierTier,
        SupplyChainConfig,
        SupplyChainEmissionsEngine,
        SupplyChainResult,
        TransportData,
    )
    _loaded_engines.append("SupplyChainEmissionsEngine")
except ImportError as e:
    logger.debug("Engine 7 (SupplyChainEmissionsEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Manufacturing Benchmark
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "ManufacturingBenchmarkEngine",
    "BenchmarkConfig",
    "FacilityKPIs",
    "SectorBenchmark",
    "BenchmarkResult",
    "SBTiAlignment",
    "TrajectoryAnalysis",
    "PeerComparison",
    "BenchmarkKPI",
    "PercentileRank",
    "SBTiPathway",
    "SECTOR_BENCHMARKS",
    "SBTI_PATHWAYS",
]

try:
    from .manufacturing_benchmark_engine import (
        SBTI_PATHWAYS,
        SECTOR_BENCHMARKS,
        BenchmarkConfig,
        BenchmarkKPI,
        BenchmarkResult,
        FacilityKPIs,
        ManufacturingBenchmarkEngine,
        PeerComparison,
        PercentileRank,
        SBTiAlignment,
        SBTiPathway,
        SectorBenchmark,
        TrajectoryAnalysis,
    )
    _loaded_engines.append("ManufacturingBenchmarkEngine")
except ImportError as e:
    logger.debug("Engine 8 (ManufacturingBenchmarkEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Public API - dynamically collected from successfully loaded engines
# ===================================================================

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    # Engine 1: Process Emissions
    *_ENGINE_1_SYMBOLS,
    # Engine 2: Energy Intensity
    *_ENGINE_2_SYMBOLS,
    # Engine 3: Product Carbon Footprint
    *_ENGINE_3_SYMBOLS,
    # Engine 4: Circular Economy
    *_ENGINE_4_SYMBOLS,
    # Engine 5: Water Pollution
    *_ENGINE_5_SYMBOLS,
    # Engine 6: BAT Compliance
    *_ENGINE_6_SYMBOLS,
    # Engine 7: Supply Chain Emissions
    *_ENGINE_7_SYMBOLS,
    # Engine 8: Manufacturing Benchmark
    *_ENGINE_8_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-013 CSRD Manufacturing engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
