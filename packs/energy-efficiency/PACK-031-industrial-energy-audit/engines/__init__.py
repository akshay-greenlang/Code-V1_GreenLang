# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Engines Module
========================================================

Calculation engines for industrial energy auditing, optimization,
and regulatory compliance per EU EED, ISO 50001, and EN 16247.

Engines:
    1. EnergyBaselineEngine              - ISO 50006 EnPI, regression, degree-day normalization
    2. EnergyAuditEngine                 - EN 16247 Type 1/2/3 audits, EED Article 8
    3. ProcessEnergyMappingEngine        - Sankey diagrams, energy balance, process efficiency
    4. EquipmentEfficiencyEngine         - Motors IE1-IE5, pumps, compressors, boilers, HVAC
    5. EnergySavingsEngine               - ECM identification, IPMVP M&V, NPV/IRR/payback
    6. WasteHeatRecoveryEngine           - Pinch analysis, heat exchanger sizing, ORC
    7. CompressedAirEngine               - Leak detection, specific power, VSD, pressure optimization
    8. SteamOptimizationEngine           - Boiler efficiency, trap survey, condensate, flash steam
    9. LightingHVACEngine                - LED retrofit, daylight harvesting, VSD, economizers
    10. EnergyBenchmarkEngine            - SEC, EnPI, BAT-AEL from BREF documents

Regulatory Basis:
    EU Directive 2023/1791 (EED - Energy Efficiency Directive)
    ISO 50001:2018 (Energy Management Systems)
    ISO 50006:2023 (Energy Performance Indicators)
    EN 16247-1:2022 through EN 16247-5 (Energy Audits)
    EU ETS Directive 2003/87/EC
    IEC 60034-30-1 (Motor Efficiency Classes)
    IPMVP (International Performance Measurement & Verification Protocol)
    EU BAT/BREF documents (Best Available Techniques)

Pack Tier: Professional (PACK-031)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-031"
__pack_name__: str = "Industrial Energy Audit Pack"
__engines_count__: int = 10

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Energy Baseline
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "EnergyBaselineEngine",
    "FacilityData",
    "EnergyMeterReading",
    "ProductionData",
    "WeatherData",
    "BaselineModel",
    "EnPIResult",
    "EnergyBaselineResult",
    "EnergyCarrier",
    "RegressionType",
    "EnPIType",
    "FacilitySector",
    "BaselineStatus",
    "ENERGY_CONVERSION_FACTORS",
    "DEGREE_DAY_BASE_TEMPS",
]

try:
    from .energy_baseline_engine import (
        DEGREE_DAY_BASE_TEMPS,
        ENERGY_CONVERSION_FACTORS,
        BaselineModel,
        BaselineStatus,
        EnergyBaselineEngine,
        EnergyBaselineResult,
        EnergyCarrier,
        EnergyMeterReading,
        EnPIResult,
        EnPIType,
        FacilityData,
        FacilitySector,
        ProductionData,
        RegressionType,
        WeatherData,
    )
    _loaded_engines.append("EnergyBaselineEngine")
except ImportError as e:
    logger.debug("Engine 1 (EnergyBaselineEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Energy Audit
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "EnergyAuditEngine",
    "AuditScope",
    "EnergyEndUse",
    "AuditFinding",
    "EN16247Checklist",
    "EEDComplianceStatus",
    "EnergyAuditResult",
    "AuditType",
    "AuditPriority",
    "AuditComplexity",
    "EndUseCategory",
    "EN16247Part",
    "ComplianceStatus",
]

try:
    from .energy_audit_engine import (
        AuditComplexity,
        AuditFinding,
        AuditPriority,
        AuditScope,
        AuditType,
        ComplianceStatus,
        EEDComplianceStatus,
        EN16247Checklist,
        EN16247Part,
        EndUseCategory,
        EnergyAuditEngine,
        EnergyAuditResult,
        EnergyEndUse,
    )
    _loaded_engines.append("EnergyAuditEngine")
except ImportError as e:
    logger.debug("Engine 2 (EnergyAuditEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Process Energy Mapping
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "ProcessEnergyMappingEngine",
    "ProcessNode",
    "EnergyFlow",
    "ProductionLine",
    "SankeyData",
    "LossBreakdown",
    "ProcessEnergyResult",
    "ProcessType",
    "EnergyType",
    "LossType",
    "TemperatureGrade",
]

try:
    from .process_energy_mapping_engine import (
        EnergyFlow,
        EnergyType,
        LossBreakdown,
        LossType,
        ProcessEnergyMappingEngine,
        ProcessEnergyResult,
        ProcessNode,
        ProcessType,
        ProductionLine,
        SankeyData,
        TemperatureGrade,
    )
    _loaded_engines.append("ProcessEnergyMappingEngine")
except ImportError as e:
    logger.debug("Engine 3 (ProcessEnergyMappingEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: Equipment Efficiency
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "EquipmentEfficiencyEngine",
    "Equipment",
    "MotorData",
    "PumpData",
    "CompressorData",
    "BoilerData",
    "HVACData",
    "EquipmentEfficiencyResult",
    "EquipmentType",
    "MotorEfficiencyClass",
    "CompressorType",
    "BoilerType",
    "FuelType",
    "HVACType",
]

try:
    from .equipment_efficiency_engine import (
        BoilerData,
        BoilerType,
        CompressorData,
        CompressorType,
        Equipment,
        EquipmentEfficiencyEngine,
        EquipmentEfficiencyResult,
        EquipmentType,
        FuelType,
        HVACData,
        HVACType,
        MotorData,
        MotorEfficiencyClass,
        PumpData,
    )
    _loaded_engines.append("EquipmentEfficiencyEngine")
except ImportError as e:
    logger.debug("Engine 4 (EquipmentEfficiencyEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Energy Savings
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "EnergySavingsEngine",
    "EnergySavingsMeasure",
    "IPMVPPlan",
    "FinancialAnalysis",
    "InteractionEffect",
    "MACCPoint",
    "EnergySavingsResult",
    "ECMCategory",
    "IPMVPOption",
    "ImplementationComplexity",
    "PriorityLevel",
    "MeasureStatus",
]

try:
    from .energy_savings_engine import (
        ECMCategory,
        EnergySavingsEngine,
        EnergySavingsMeasure,
        EnergySavingsResult,
        FinancialAnalysis,
        IPMVPOption,
        IPMVPPlan,
        ImplementationComplexity,
        InteractionEffect,
        MACCPoint,
        MeasureStatus,
        PriorityLevel,
    )
    _loaded_engines.append("EnergySavingsEngine")
except ImportError as e:
    logger.debug("Engine 5 (EnergySavingsEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: Waste Heat Recovery
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "WasteHeatRecoveryEngine",
    "WasteHeatSource",
    "HeatSink",
    "PinchAnalysisResult",
    "HeatExchangerDesign",
    "RecoveryTechnology",
    "WasteHeatResult",
    "HeatSourceType",
    "TemperatureGrade",
    "HeatExchangerType",
]

try:
    from .waste_heat_recovery_engine import (
        HeatExchangerDesign,
        HeatExchangerType,
        HeatSink,
        HeatSourceType,
        PinchAnalysisResult,
        RecoveryTechnology,
        WasteHeatRecoveryEngine,
        WasteHeatResult,
        WasteHeatSource,
    )
    _loaded_engines.append("WasteHeatRecoveryEngine")
except ImportError as e:
    logger.debug("Engine 6 (WasteHeatRecoveryEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Compressed Air
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "CompressedAirEngine",
    "CompressedAirSystem",
    "Compressor",
    "LeakSurvey",
    "PressureProfile",
    "AirReceiver",
    "CompressedAirResult",
    "CompressorType",
    "CompressorControl",
    "DryerType",
]

try:
    from .compressed_air_engine import (
        AirReceiver,
        CompressedAirEngine,
        CompressedAirResult,
        CompressedAirSystem,
        Compressor,
        CompressorControl,
        DryerType,
        LeakSurvey,
        PressureProfile,
    )
    _loaded_engines.append("CompressedAirEngine")
except ImportError as e:
    logger.debug("Engine 7 (CompressedAirEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: Steam Optimization
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "SteamOptimizationEngine",
    "SteamSystem",
    "Boiler",
    "SteamTrapSurvey",
    "InsulationAssessment",
    "CondensateSystem",
    "FlueGasAnalysis",
    "SteamOptimizationResult",
    "SteamBoilerType",
    "SteamTrapType",
    "TrapStatus",
    "InsulationMaterial",
]

try:
    from .steam_optimization_engine import (
        Boiler,
        CondensateSystem,
        FlueGasAnalysis,
        InsulationAssessment,
        InsulationMaterial,
        SteamBoilerType,
        SteamOptimizationEngine,
        SteamOptimizationResult,
        SteamSystem,
        SteamTrapSurvey,
        SteamTrapType,
        TrapStatus,
    )
    _loaded_engines.append("SteamOptimizationEngine")
except ImportError as e:
    logger.debug("Engine 8 (SteamOptimizationEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Engine 9: Lighting & HVAC
# ===================================================================
_ENGINE_9_SYMBOLS: list[str] = [
    "LightingHVACEngine",
    "LightingZone",
    "LightingRetrofit",
    "HVACSystem",
    "VSDRetrofit",
    "EconomizerAnalysis",
    "BuildingEnvelope",
    "LightingHVACResult",
    "FixtureType",
    "HVACSystemType",
    "ClimateZone",
    "VentilationStrategy",
]

try:
    from .lighting_hvac_engine import (
        BuildingEnvelope,
        ClimateZone,
        EconomizerAnalysis,
        FixtureType,
        HVACSystem,
        HVACSystemType,
        LightingHVACEngine,
        LightingHVACResult,
        LightingRetrofit,
        LightingZone,
        VSDRetrofit,
        VentilationStrategy,
    )
    _loaded_engines.append("LightingHVACEngine")
except ImportError as e:
    logger.debug("Engine 9 (LightingHVACEngine) not available: %s", e)
    _ENGINE_9_SYMBOLS = []


# ===================================================================
# Engine 10: Energy Benchmark
# ===================================================================
_ENGINE_10_SYMBOLS: list[str] = [
    "EnergyBenchmarkEngine",
    "BenchmarkFacility",
    "SECResult",
    "BATBenchmark",
    "EnergyRating",
    "PeerComparison",
    "EnergyBenchmarkResult",
    "IndustrySector",
    "EnergyRatingClass",
    "BREFDocument",
    "BenchmarkMetric",
]

try:
    from .energy_benchmark_engine import (
        BATBenchmark,
        BREFDocument,
        BenchmarkFacility,
        BenchmarkMetric,
        EnergyBenchmarkEngine,
        EnergyBenchmarkResult,
        EnergyRating,
        EnergyRatingClass,
        IndustrySector,
        PeerComparison,
        SECResult,
    )
    _loaded_engines.append("EnergyBenchmarkEngine")
except ImportError as e:
    logger.debug("Engine 10 (EnergyBenchmarkEngine) not available: %s", e)
    _ENGINE_10_SYMBOLS = []


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
    *_ENGINE_1_SYMBOLS,
    *_ENGINE_2_SYMBOLS,
    *_ENGINE_3_SYMBOLS,
    *_ENGINE_4_SYMBOLS,
    *_ENGINE_5_SYMBOLS,
    *_ENGINE_6_SYMBOLS,
    *_ENGINE_7_SYMBOLS,
    *_ENGINE_8_SYMBOLS,
    *_ENGINE_9_SYMBOLS,
    *_ENGINE_10_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-031 Industrial Energy Audit engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
