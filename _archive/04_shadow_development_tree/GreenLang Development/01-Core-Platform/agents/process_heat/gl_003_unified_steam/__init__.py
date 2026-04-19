"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER

This module consolidates GL-003 (STEAMWISE) and GL-012 (STEAMQUAL) -
agents with 60% functional overlap in steam distribution and quality monitoring.

Features:
    - Steam header pressure balancing with exergy-based optimization
    - Steam quality monitoring per ASME standards (dryness, TDS, conductivity)
    - Flash steam recovery calculations (thermodynamic fraction extraction)
    - PRV sizing and optimization per ASME B31.1 (50-70% opening targets)
    - Desuperheating control
    - Condensate return temperature maximization for fuel savings
    - Steam trap survey integration points
    - IAPWS-IF97 steam property calculations
    - Zero-hallucination: All calculations deterministic with provenance

Standards References:
    - ASME B31.1 Power Piping Code
    - ASME Boiler and Pressure Vessel Code
    - IAPWS-IF97 Steam Properties
    - ABMA Recommended Practices
    - ANSI/ISA-75.01 Control Valve Sizing

Example Usage:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam import (
    ...     UnifiedSteamOptimizer,
    ...     UnifiedSteamConfig,
    ...     UnifiedSteamOptimizerInput,
    ...     create_default_config,
    ... )
    >>>
    >>> # Create configuration
    >>> config = create_default_config()
    >>>
    >>> # Initialize optimizer
    >>> optimizer = UnifiedSteamOptimizer(config)
    >>>
    >>> # Process steam system data
    >>> input_data = UnifiedSteamOptimizerInput(
    ...     header_readings=[...],
    ...     quality_readings=[...],
    ...     total_steam_flow_lb_hr=100000,
    ... )
    >>>
    >>> result = optimizer.process(input_data)
    >>> print(f"System efficiency: {result.system_efficiency_pct:.1f}%")

Module Structure:
    - config.py: Configuration schemas and defaults
    - schemas.py: Pydantic data models
    - distribution.py: Steam header pressure balancing
    - quality.py: Steam quality monitoring per ASME
    - condensate.py: Condensate return optimization
    - flash_recovery.py: Flash steam calculations
    - prv_optimization.py: PRV sizing per ASME B31.1
    - optimizer.py: Main UnifiedSteamOptimizer class

Author: GreenLang Engineering Team
Version: 2.0.0
License: Proprietary
"""

# Version info
__version__ = "2.0.0"
__author__ = "GreenLang Engineering"
__agent_id__ = "GL-003"
__agent_name__ = "Unified Steam System Optimizer"

# =============================================================================
# CONFIGURATION EXPORTS
# =============================================================================
from .config import (
    # Main configuration
    UnifiedSteamConfig,
    create_default_config,

    # Sub-configurations
    SteamHeaderConfig,
    SteamHeaderLevel,
    PRVConfig,
    PRVSizingMethod,
    DesuperheaterConfig,
    DesuperheaterType,
    QualityMonitoringConfig,
    SteamQualityStandard,
    CondensateConfig,
    CondensateFlashMethod,
    FlashRecoveryConfig,
    SteamTrapSurveyConfig,
    ExergyOptimizationConfig,
)

# =============================================================================
# SCHEMA EXPORTS
# =============================================================================
from .schemas import (
    # Enums
    SteamPhase,
    ValidationStatus,
    OptimizationStatus,
    TrapStatus,

    # Steam properties
    SteamProperties,
    SteamFlowMeasurement,

    # Header models
    HeaderReading,
    HeaderBalanceInput,
    HeaderBalanceOutput,

    # Quality models
    SteamQualityReading,
    SteamQualityAnalysis,

    # PRV models
    PRVOperatingPoint,
    PRVSizingInput,
    PRVSizingOutput,

    # Condensate models
    CondensateReading,
    CondensateReturnAnalysis,

    # Flash steam models
    FlashSteamInput,
    FlashSteamOutput,

    # Trap models
    SteamTrapReading,
    TrapSurveyAnalysis,

    # Recommendation model
    OptimizationRecommendation,

    # Main output
    UnifiedSteamOptimizerOutput,
)

# =============================================================================
# MAIN OPTIMIZER EXPORTS
# =============================================================================
from .optimizer import (
    UnifiedSteamOptimizer,
    UnifiedSteamOptimizerInput,
)

# =============================================================================
# DISTRIBUTION MODULE EXPORTS
# =============================================================================
from .distribution import (
    SteamDistributionOptimizer,
    SteamPropertyCalculator,
    HeaderBalanceCalculator,
)

# =============================================================================
# QUALITY MODULE EXPORTS
# =============================================================================
from .quality import (
    SteamQualityMonitor,
    QualityLimitCalculator,
    DrynessFractionCalculator,
    CarryoverAnalyzer,
    ASMEQualityLimits,
)

# =============================================================================
# CONDENSATE MODULE EXPORTS
# =============================================================================
from .condensate import (
    CondensateReturnOptimizer,
    CondensateHeatCalculator,
    CondensateQualityAnalyzer,
    SteamTrapSurveyAnalyzer,
)

# =============================================================================
# FLASH RECOVERY EXPORTS
# =============================================================================
from .flash_recovery import (
    FlashRecoveryOptimizer,
    FlashSteamCalculator,
    FlashTankSizer,
    MultiStageFlashOptimizer,
)

# =============================================================================
# PRV OPTIMIZATION EXPORTS
# =============================================================================
from .prv_optimization import (
    PRVOptimizer,
    CvCalculator,
    DesuperheaterCalculator,
    MultiPRVCoordinator,
)

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    "__author__",
    "__agent_id__",
    "__agent_name__",

    # Main classes
    "UnifiedSteamOptimizer",
    "UnifiedSteamOptimizerInput",
    "UnifiedSteamOptimizerOutput",

    # Configuration
    "UnifiedSteamConfig",
    "create_default_config",
    "SteamHeaderConfig",
    "SteamHeaderLevel",
    "PRVConfig",
    "PRVSizingMethod",
    "DesuperheaterConfig",
    "DesuperheaterType",
    "QualityMonitoringConfig",
    "SteamQualityStandard",
    "CondensateConfig",
    "CondensateFlashMethod",
    "FlashRecoveryConfig",
    "SteamTrapSurveyConfig",
    "ExergyOptimizationConfig",

    # Schemas - Enums
    "SteamPhase",
    "ValidationStatus",
    "OptimizationStatus",
    "TrapStatus",

    # Schemas - Steam
    "SteamProperties",
    "SteamFlowMeasurement",

    # Schemas - Headers
    "HeaderReading",
    "HeaderBalanceInput",
    "HeaderBalanceOutput",

    # Schemas - Quality
    "SteamQualityReading",
    "SteamQualityAnalysis",

    # Schemas - PRV
    "PRVOperatingPoint",
    "PRVSizingInput",
    "PRVSizingOutput",

    # Schemas - Condensate
    "CondensateReading",
    "CondensateReturnAnalysis",

    # Schemas - Flash
    "FlashSteamInput",
    "FlashSteamOutput",

    # Schemas - Traps
    "SteamTrapReading",
    "TrapSurveyAnalysis",

    # Schemas - Recommendations
    "OptimizationRecommendation",

    # Distribution
    "SteamDistributionOptimizer",
    "SteamPropertyCalculator",
    "HeaderBalanceCalculator",

    # Quality
    "SteamQualityMonitor",
    "QualityLimitCalculator",
    "DrynessFractionCalculator",
    "CarryoverAnalyzer",
    "ASMEQualityLimits",

    # Condensate
    "CondensateReturnOptimizer",
    "CondensateHeatCalculator",
    "CondensateQualityAnalyzer",
    "SteamTrapSurveyAnalyzer",

    # Flash Recovery
    "FlashRecoveryOptimizer",
    "FlashSteamCalculator",
    "FlashTankSizer",
    "MultiStageFlashOptimizer",

    # PRV
    "PRVOptimizer",
    "CvCalculator",
    "DesuperheaterCalculator",
    "MultiPRVCoordinator",
]
