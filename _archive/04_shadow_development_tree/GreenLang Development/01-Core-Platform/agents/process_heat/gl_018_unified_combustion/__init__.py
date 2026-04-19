"""
GL-018 UnifiedCombustionOptimizer Agent Package

This package provides the GL-018 UnifiedCombustionOptimizer agent that consolidates
GL-002 (FLAMEGUARD), GL-004 (BURNMASTER), and GL-018 (FLUEFLOW) - agents with
70-80% functional overlap.

Features:
    - ASME PTC 4.1 efficiency calculations (input-output and losses methods)
    - API 560 combustion analysis
    - Air-fuel ratio optimization with O2 trim and cross-limiting per NFPA 85
    - Burner tuning with Flame Stability Index (FSI)
    - NOx/CO emissions control (Low NOx Burners, FGR, SCR optimization)
    - Soot blowing and blowdown optimization
    - BMS coordination per NFPA 85 startup/shutdown sequences
    - Zero-hallucination: All calculations are deterministic with provenance
    - Full audit trail with SHA-256 hashing

Example:
    >>> from greenlang.agents.process_heat.gl_018_unified_combustion import (
    ...     UnifiedCombustionOptimizer,
    ...     UnifiedCombustionConfig,
    ...     BurnerConfig,
    ...     CombustionInput,
    ...     FlueGasReading,
    ... )
    >>>
    >>> # Create configuration
    >>> config = UnifiedCombustionConfig(
    ...     equipment_id="BOILER-001",
    ...     burner=BurnerConfig(
    ...         burner_id="BNR-001",
    ...         capacity_mmbtu_hr=50.0
    ...     )
    ... )
    >>>
    >>> # Initialize agent
    >>> agent = UnifiedCombustionOptimizer(config)
    >>>
    >>> # Create input data
    >>> input_data = CombustionInput(
    ...     equipment_id="BOILER-001",
    ...     load_pct=75.0,
    ...     fuel_flow_rate=1000.0,
    ...     flue_gas=FlueGasReading(
    ...         o2_pct=3.5,
    ...         co_ppm=25.0,
    ...         temperature_f=400.0
    ...     )
    ... )
    >>>
    >>> # Process and get results
    >>> result = agent.process(input_data)
    >>> print(f"Efficiency: {result.efficiency.net_efficiency_pct:.1f}%")
    >>> print(f"Recommendations: {len(result.recommendations)}")

Module Structure:
    - config.py: Configuration schemas (UnifiedCombustionConfig)
    - schemas.py: Pydantic data models (CombustionInput, CombustionOutput)
    - flue_gas.py: Flue gas analysis per API 560
    - burner_control.py: Burner tuning and BMS coordination per NFPA 85
    - efficiency.py: ASME PTC 4.1 efficiency calculations
    - emissions.py: NOx/CO emissions control and EPA compliance
    - optimizer.py: Main UnifiedCombustionOptimizer agent class
"""

# Configuration
from .config import (
    # Enums
    FuelType,
    EquipmentType,
    BurnerType,
    ControlMode,
    EmissionControlTechnology,
    BMSSequence,
    # Sub-configurations
    BurnerConfig,
    AirFuelConfig,
    FlueGasConfig,
    FlameStabilityConfig,
    EmissionsConfig,
    BMSConfig,
    SootBlowingConfig,
    BlowdownConfig,
    EfficiencyConfig,
    # Main configuration
    UnifiedCombustionConfig,
)

# Schemas
from .schemas import (
    # Enums
    OperatingStatus,
    AlertSeverity,
    RecommendationPriority,
    # Input models
    BurnerStatus,
    FlueGasReading,
    CombustionInput,
    # Analysis results
    FlueGasAnalysis,
    FlameStabilityAnalysis,
    BurnerTuningResult,
    EfficiencyResult,
    EmissionsAnalysis,
    BMSStatus,
    # Recommendations and alerts
    OptimizationRecommendation,
    Alert,
    # Output
    CombustionOutput,
)

# Analysis components
from .flue_gas import (
    FlueGasAnalyzer,
    AirFuelOptimizer,
    FUEL_PROPERTIES,
)

from .burner_control import (
    FlameStabilityAnalyzer,
    BurnerTuningController,
    BMSSequenceController,
)

from .efficiency import (
    EfficiencyCalculator,
)

from .emissions import (
    EmissionsController,
)

# Main agent
from .optimizer import (
    UnifiedCombustionOptimizer,
)

__all__ = [
    # Main agent
    "UnifiedCombustionOptimizer",
    # Configuration
    "UnifiedCombustionConfig",
    "BurnerConfig",
    "AirFuelConfig",
    "FlueGasConfig",
    "FlameStabilityConfig",
    "EmissionsConfig",
    "BMSConfig",
    "SootBlowingConfig",
    "BlowdownConfig",
    "EfficiencyConfig",
    # Enums
    "FuelType",
    "EquipmentType",
    "BurnerType",
    "ControlMode",
    "EmissionControlTechnology",
    "BMSSequence",
    "OperatingStatus",
    "AlertSeverity",
    "RecommendationPriority",
    # Input schemas
    "CombustionInput",
    "FlueGasReading",
    "BurnerStatus",
    # Output schemas
    "CombustionOutput",
    "EfficiencyResult",
    "FlueGasAnalysis",
    "FlameStabilityAnalysis",
    "EmissionsAnalysis",
    "BMSStatus",
    "BurnerTuningResult",
    "OptimizationRecommendation",
    "Alert",
    # Analysis components
    "FlueGasAnalyzer",
    "AirFuelOptimizer",
    "FlameStabilityAnalyzer",
    "BurnerTuningController",
    "BMSSequenceController",
    "EfficiencyCalculator",
    "EmissionsController",
    # Constants
    "FUEL_PROPERTIES",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-018"
__agent_name__ = "UnifiedCombustionOptimizer"
