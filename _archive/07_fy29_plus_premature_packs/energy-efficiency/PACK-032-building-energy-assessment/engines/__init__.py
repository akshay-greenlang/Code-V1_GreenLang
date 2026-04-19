# -*- coding: utf-8 -*-
"""
PACK-032 Building Energy Assessment -- Calculation Engines
==========================================================

Zero-hallucination, deterministic calculation engines for building
energy performance assessment, EPC rating, HVAC efficiency, DHW,
lighting, renewable integration, benchmarking, retrofit analysis,
indoor environment quality, and whole life carbon assessment.

Engines:
    1. BuildingEnvelopeEngine      -- Fabric / envelope thermal performance
    2. EPCRatingEngine             -- Energy Performance Certificate rating
    3. HVACAssessmentEngine        -- HVAC system efficiency assessment
    4. DomesticHotWaterEngine      -- DHW system assessment per EN 15316-3
    5. LightingAssessmentEngine    -- Lighting energy/quality per EN 12464-1, EN 15193
    6. RenewableIntegrationEngine  -- Building-integrated renewables assessment
    7. BuildingBenchmarkEngine     -- Energy benchmarking (DEC, CRREM, ENERGY STAR)
    8. RetrofitAnalysisEngine      -- Retrofit cost-benefit, MACC, nZEB gap, roadmap
    9. IndoorEnvironmentEngine     -- IEQ per EN 16798-1, ISO 7730 (PMV/PPD/adaptive)
   10. WholeLifeCarbonEngine       -- Whole life carbon per EN 15978 (A1-D)

Regulatory Basis:
    EN 15316-3:2017 (DHW energy performance)
    EN 12464-1:2021 (Lighting requirements for indoor workplaces)
    EN 15193-1:2017 (Lighting Energy Numeric Indicator)
    CIBSE TM46 (Energy benchmarks for buildings)
    CRREM (Carbon Risk Real Estate Monitor pathways)
    HSG274 Part 2 (Legionella control in hot/cold water systems)
    IEC 61724 (PV system performance monitoring)
    EN 14825 (Heat pump seasonal performance)
    EN 15459:2017 (Economic evaluation of energy systems in buildings)
    EN 16798-1:2019 (Indoor environmental input parameters)
    ISO 7730:2005 (Ergonomics of the thermal environment)
    EN 15978:2011 (Sustainability of construction works)
    EN 15804:2012+A2:2019 (Environmental Product Declarations)

All engines use Decimal arithmetic, SHA-256 provenance hashing,
and complete audit trails.  No LLM is used in any calculation path.

Pack Tier: Professional (PACK-032)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-032"
__pack_name__: str = "Building Energy Assessment Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Building Envelope
# ===================================================================
try:
    from .building_envelope_engine import BuildingEnvelopeEngine
    _loaded_engines.append("BuildingEnvelopeEngine")
except ImportError as e:
    logger.debug("Engine 1 (BuildingEnvelopeEngine) not available: %s", e)


# ===================================================================
# Engine 2: EPC Rating
# ===================================================================
try:
    from .epc_rating_engine import EPCRatingEngine
    _loaded_engines.append("EPCRatingEngine")
except ImportError as e:
    logger.debug("Engine 2 (EPCRatingEngine) not available: %s", e)


# ===================================================================
# Engine 3: HVAC Assessment
# ===================================================================
try:
    from .hvac_assessment_engine import HVACAssessmentEngine
    _loaded_engines.append("HVACAssessmentEngine")
except ImportError as e:
    logger.debug("Engine 3 (HVACAssessmentEngine) not available: %s", e)


# ===================================================================
# Engine 4: Domestic Hot Water
# ===================================================================
try:
    from .domestic_hot_water_engine import (
        DomesticHotWaterEngine,
        DHWAssessmentInput,
        DHWAssessmentResult,
        DHWDemandResult,
        GenerationResult,
        StorageResult,
        DistributionResult,
        SolarThermalResult,
        LegionellaResult,
        DHWSystemType,
        SolarCollectorType,
        StorageType,
        BuildingOccupancyType,
        InsulationType,
        ComplianceStatus,
        DHW_DEMAND_LITRES_PER_DAY,
        SYSTEM_EFFICIENCY,
        STORAGE_LOSS_WK,
        LEGIONELLA_REQUIREMENTS,
    )
    _loaded_engines.append("DomesticHotWaterEngine")
except ImportError as e:
    logger.debug("Engine 4 (DomesticHotWaterEngine) not available: %s", e)


# ===================================================================
# Engine 5: Lighting Assessment
# ===================================================================
try:
    from .lighting_assessment_engine import (
        LightingAssessmentEngine,
        LightingAssessmentInput,
        LightingAssessmentResult,
        ZoneLPDResult,
        LENIResult,
        DaylightResult,
        ControlsResult,
        RetrofitResult,
        VisualQualityResult,
        LampType,
        ControlType,
        SpaceCategory,
        LPD_BENCHMARKS,
        LENI_BENCHMARKS,
        ILLUMINANCE_REQUIREMENTS,
        LAMP_EFFICACY,
        CONTROL_FACTOR,
    )
    _loaded_engines.append("LightingAssessmentEngine")
except ImportError as e:
    logger.debug("Engine 5 (LightingAssessmentEngine) not available: %s", e)


# ===================================================================
# Engine 6: Renewable Integration
# ===================================================================
try:
    from .renewable_integration_engine import (
        RenewableIntegrationEngine,
        RenewableAssessmentInput,
        RenewableAssessmentResult,
        SolarPVResult,
        HeatPumpResult,
        BiomassResult,
        RenewableType,
        PVMountType,
        PVModuleType,
        BuildingLoadProfile,
        HeatDistributionType,
        SOLAR_IRRADIANCE_BY_LOCATION,
        PV_MODULE_EFFICIENCY,
        HEAT_PUMP_SEASONAL_COP,
        SELF_CONSUMPTION_PROFILES,
    )
    _loaded_engines.append("RenewableIntegrationEngine")
except ImportError as e:
    logger.debug("Engine 6 (RenewableIntegrationEngine) not available: %s", e)


# ===================================================================
# Engine 7: Building Benchmark
# ===================================================================
try:
    from .building_benchmark_engine import (
        BuildingBenchmarkEngine,
        BenchmarkInput,
        BuildingBenchmarkResult,
        EUIResult,
        DECResult,
        CRREMResult,
        EnergyStarResult,
        PeerComparisonResult,
        GapAnalysisResult,
        BuildingType,
        BenchmarkStandard,
        DECRating,
        CRREMScenario,
        EUI_BENCHMARKS,
        CRREM_PATHWAYS,
        DEC_RATING_THRESHOLDS,
        TYPICAL_END_USE_SPLIT,
    )
    _loaded_engines.append("BuildingBenchmarkEngine")
except ImportError as e:
    logger.debug("Engine 7 (BuildingBenchmarkEngine) not available: %s", e)


# ===================================================================
# Engine 8: Retrofit Analysis
# ===================================================================
try:
    from .retrofit_analysis_engine import (
        RetrofitAnalysisEngine,
        RetrofitAnalysisInput,
        RetrofitAnalysisResult,
        MeasureInput,
        MeasureFinancials,
        InteractionResult,
        MACCEntry,
        RoadmapPhase,
        NZEBAssessment,
        FinancingSummary,
        CarbonValueSummary,
        RetrofitCategory,
        RetrofitPriority,
        NZEBLevel,
        MeasureComplexity,
        DisruptionLevel,
        CarbonPriceScenario,
        RETROFIT_MEASURE_LIBRARY,
        MEASURE_INTERACTION_MATRIX,
        NZEB_TARGETS,
        FINANCING_OPTIONS,
        CARBON_PRICE_PROJECTIONS,
    )
    _loaded_engines.append("RetrofitAnalysisEngine")
except ImportError as e:
    logger.debug("Engine 8 (RetrofitAnalysisEngine) not available: %s", e)


# ===================================================================
# Engine 9: Indoor Environment
# ===================================================================
try:
    from .indoor_environment_engine import (
        IndoorEnvironmentEngine,
        IndoorEnvironmentInput,
        IndoorEnvironmentResult,
        ThermalComfortInput,
        PMVPPDResult,
        AdaptiveComfortResult,
        IAQMeasurement,
        IAQAssessmentResult,
        SpaceVentilationInput,
        VentilationResult,
        OverheatingInput,
        OverheatingResult,
        DaylightInput,
        IEQScoreBreakdown,
        IEQCategory,
        ThermalComfortMethod,
        IAQParameter,
        VentilationStandard,
        SpaceType,
        THERMAL_COMFORT_CATEGORIES,
        IAQ_LIMITS,
        VENTILATION_RATES,
        OVERHEATING_CRITERIA,
        METABOLIC_RATES,
        CLOTHING_INSULATION,
        DAYLIGHTING_REQUIREMENTS,
    )
    _loaded_engines.append("IndoorEnvironmentEngine")
except ImportError as e:
    logger.debug("Engine 9 (IndoorEnvironmentEngine) not available: %s", e)


# ===================================================================
# Engine 10: Whole Life Carbon
# ===================================================================
try:
    from .whole_life_carbon_engine import (
        WholeLifeCarbonEngine,
        WholeLifeCarbonInput,
        WholeLifeCarbonResult,
        MaterialInput,
        MaterialEmbodiedResult,
        LifecycleStageResult,
        TargetComparison,
        TopMaterialContributor,
        LifecycleStage,
        MaterialCategory,
        CarbonTarget,
        TransportMode,
        BuildingTypeWLC,
        EMBODIED_CARBON_FACTORS,
        MATERIAL_LIFETIME,
        TRANSPORT_EMISSION_FACTORS,
        CARBON_BUDGETS,
        GRID_DECARBONISATION,
        BIOGENIC_CARBON_FACTORS,
    )
    _loaded_engines.append("WholeLifeCarbonEngine")
except ImportError as e:
    logger.debug("Engine 10 (WholeLifeCarbonEngine) not available: %s", e)


# ===================================================================
# Public API
# ===================================================================

__all__: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
    # Engine 1
    "BuildingEnvelopeEngine",
    # Engine 2
    "EPCRatingEngine",
    # Engine 3
    "HVACAssessmentEngine",
    # Engine 4
    "DomesticHotWaterEngine",
    "DHWAssessmentInput",
    "DHWAssessmentResult",
    "DHWDemandResult",
    "GenerationResult",
    "StorageResult",
    "DistributionResult",
    "SolarThermalResult",
    "LegionellaResult",
    "DHWSystemType",
    "SolarCollectorType",
    "StorageType",
    "BuildingOccupancyType",
    "InsulationType",
    "ComplianceStatus",
    "DHW_DEMAND_LITRES_PER_DAY",
    "SYSTEM_EFFICIENCY",
    "STORAGE_LOSS_WK",
    "LEGIONELLA_REQUIREMENTS",
    # Engine 5
    "LightingAssessmentEngine",
    "LightingAssessmentInput",
    "LightingAssessmentResult",
    "ZoneLPDResult",
    "LENIResult",
    "DaylightResult",
    "ControlsResult",
    "RetrofitResult",
    "VisualQualityResult",
    "LampType",
    "ControlType",
    "SpaceCategory",
    "LPD_BENCHMARKS",
    "LENI_BENCHMARKS",
    "ILLUMINANCE_REQUIREMENTS",
    "LAMP_EFFICACY",
    "CONTROL_FACTOR",
    # Engine 6
    "RenewableIntegrationEngine",
    "RenewableAssessmentInput",
    "RenewableAssessmentResult",
    "SolarPVResult",
    "HeatPumpResult",
    "BiomassResult",
    "RenewableType",
    "PVMountType",
    "PVModuleType",
    "BuildingLoadProfile",
    "HeatDistributionType",
    "SOLAR_IRRADIANCE_BY_LOCATION",
    "PV_MODULE_EFFICIENCY",
    "HEAT_PUMP_SEASONAL_COP",
    "SELF_CONSUMPTION_PROFILES",
    # Engine 7
    "BuildingBenchmarkEngine",
    "BenchmarkInput",
    "BuildingBenchmarkResult",
    "EUIResult",
    "DECResult",
    "CRREMResult",
    "EnergyStarResult",
    "PeerComparisonResult",
    "GapAnalysisResult",
    "BuildingType",
    "BenchmarkStandard",
    "DECRating",
    "CRREMScenario",
    "EUI_BENCHMARKS",
    "CRREM_PATHWAYS",
    "DEC_RATING_THRESHOLDS",
    "TYPICAL_END_USE_SPLIT",
    # Engine 8
    "RetrofitAnalysisEngine",
    "RetrofitAnalysisInput",
    "RetrofitAnalysisResult",
    "MeasureInput",
    "MeasureFinancials",
    "InteractionResult",
    "MACCEntry",
    "RoadmapPhase",
    "NZEBAssessment",
    "FinancingSummary",
    "CarbonValueSummary",
    "RetrofitCategory",
    "RetrofitPriority",
    "NZEBLevel",
    "MeasureComplexity",
    "DisruptionLevel",
    "CarbonPriceScenario",
    "RETROFIT_MEASURE_LIBRARY",
    "MEASURE_INTERACTION_MATRIX",
    "NZEB_TARGETS",
    "FINANCING_OPTIONS",
    "CARBON_PRICE_PROJECTIONS",
    # Engine 9
    "IndoorEnvironmentEngine",
    "IndoorEnvironmentInput",
    "IndoorEnvironmentResult",
    "ThermalComfortInput",
    "PMVPPDResult",
    "AdaptiveComfortResult",
    "IAQMeasurement",
    "IAQAssessmentResult",
    "SpaceVentilationInput",
    "VentilationResult",
    "OverheatingInput",
    "OverheatingResult",
    "DaylightInput",
    "IEQScoreBreakdown",
    "IEQCategory",
    "ThermalComfortMethod",
    "IAQParameter",
    "VentilationStandard",
    "SpaceType",
    "THERMAL_COMFORT_CATEGORIES",
    "IAQ_LIMITS",
    "VENTILATION_RATES",
    "OVERHEATING_CRITERIA",
    "METABOLIC_RATES",
    "CLOTHING_INSULATION",
    "DAYLIGHTING_REQUIREMENTS",
    # Engine 10
    "WholeLifeCarbonEngine",
    "WholeLifeCarbonInput",
    "WholeLifeCarbonResult",
    "MaterialInput",
    "MaterialEmbodiedResult",
    "LifecycleStageResult",
    "TargetComparison",
    "TopMaterialContributor",
    "LifecycleStage",
    "MaterialCategory",
    "CarbonTarget",
    "TransportMode",
    "BuildingTypeWLC",
    "EMBODIED_CARBON_FACTORS",
    "MATERIAL_LIFETIME",
    "TRANSPORT_EMISSION_FACTORS",
    "CARBON_BUDGETS",
    "GRID_DECARBONISATION",
    "BIOGENIC_CARBON_FACTORS",
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-032 Building Energy Assessment engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
