"""
GL-020 ECONOPULSE - Economizer Optimization Agent

This agent optimizes economizer performance through:
- Gas-side fouling analysis and differentiation
- Water-side scaling/fouling monitoring
- Soot blower optimization
- Acid dew point calculations for cold-end corrosion prevention
- Heat transfer effectiveness tracking
- Steaming economizer detection and prevention

Standards Compliance:
- ASME PTC 4.3 (Air Heater Testing)
- ASME PTC 4.1 (Steam Generating Units)
- EPA Method 6C (SO2 measurement)

Target Score: 95+/100
"""

from .config import (
    EconomizerOptimizationConfig,
    EconomizerDesignConfig,
    PerformanceBaselineConfig,
    GasSideFoulingConfig,
    WaterSideFoulingConfig,
    SootBlowerConfig,
    AcidDewPointConfig,
    EffectivenessConfig,
    SteamingConfig,
    EconomizerType,
    EconomizerArrangement,
    TubeMaterial,
    FuelType,
    SootBlowerType,
)

from .schemas import (
    EconomizerInput,
    EconomizerOutput,
    EconomizerStatus,
    FoulingType,
    FoulingSeverity,
    AlertSeverity,
    CleaningStatus,
    SootBlowingStatus,
    GasSideFoulingResult as GasSideAnalysis,
    WaterSideFoulingResult as WaterSideAnalysis,
    SootBlowerResult as SootBlowingRecommendation,
    AcidDewPointResult,
    EffectivenessResult,
    SteamingResult as SteamingAnalysis,
    OptimizationRecommendation,
    Alert as EconomizerAlert,
)

from .gas_side import (
    GasSideFoulingAnalyzer,
    GasSideFoulingInput,
    GasSideFoulingResult,
    create_gas_side_fouling_analyzer,
)
from .water_side import (
    WaterSideFoulingAnalyzer,
    WaterSideFoulingInput,
    WaterSideFoulingResult,
    WaterChemistryData,
    create_water_side_fouling_analyzer,
)
from .soot_blowing import (
    SootBlowerOptimizer,
    SootBlowerConfig as SootBlowerOptConfig,
    SootBlowerInput,
    SootBlowerResult,
    BlowEffectivenessRecord,
    create_soot_blower_optimizer,
)
from .acid_dew_point import AcidDewPointCalculator, create_acid_dew_point_calculator
from .effectiveness import EffectivenessCalculator, create_effectiveness_calculator
from .steaming import (
    SteamingDetector,
    SteamingConfig as SteamingDetectorConfig,
    SteamingInput,
    SteamingResult,
    create_steaming_detector,
)
from .optimizer import EconomizerOptimizer, create_economizer_optimizer

__all__ = [
    # Configuration
    "EconomizerOptimizationConfig",
    "EconomizerDesignConfig",
    "PerformanceBaselineConfig",
    "GasSideFoulingConfig",
    "WaterSideFoulingConfig",
    "SootBlowerConfig",
    "AcidDewPointConfig",
    "EffectivenessConfig",
    "SteamingConfig",
    "EconomizerType",
    "EconomizerArrangement",
    "TubeMaterial",
    "FuelType",
    "SootBlowerType",
    # Schemas
    "EconomizerInput",
    "EconomizerOutput",
    "EconomizerStatus",
    "FoulingType",
    "FoulingSeverity",
    "AlertSeverity",
    "CleaningStatus",
    "SootBlowingStatus",
    "GasSideAnalysis",
    "WaterSideAnalysis",
    "SootBlowingRecommendation",
    "AcidDewPointResult",
    "EffectivenessResult",
    "SteamingAnalysis",
    "OptimizationRecommendation",
    "EconomizerAlert",
    # Gas-Side Fouling
    "GasSideFoulingAnalyzer",
    "GasSideFoulingInput",
    "GasSideFoulingResult",
    "create_gas_side_fouling_analyzer",
    # Water-Side Fouling
    "WaterSideFoulingAnalyzer",
    "WaterSideFoulingInput",
    "WaterSideFoulingResult",
    "WaterChemistryData",
    "create_water_side_fouling_analyzer",
    # Soot Blowing
    "SootBlowerOptimizer",
    "SootBlowerOptConfig",
    "SootBlowerInput",
    "SootBlowerResult",
    "BlowEffectivenessRecord",
    "create_soot_blower_optimizer",
    # Acid Dew Point
    "AcidDewPointCalculator",
    "create_acid_dew_point_calculator",
    # Effectiveness
    "EffectivenessCalculator",
    "create_effectiveness_calculator",
    # Steaming Detection
    "SteamingDetector",
    "SteamingDetectorConfig",
    "SteamingInput",
    "SteamingResult",
    "create_steaming_detector",
    # Main Agent
    "EconomizerOptimizer",
    "create_economizer_optimizer",
]

__version__ = "1.0.0"
__agent_id__ = "GL-020"
__agent_name__ = "ECONOPULSE"
