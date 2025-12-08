# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY - Burner Maintenance and Flame Analysis Module
==================================================================

This module provides comprehensive burner health monitoring, flame pattern
analysis, and predictive maintenance capabilities for industrial combustion
systems per NFPA 85, NFPA 86, and API 535.

Modules:
    - flame_analysis: Flame pattern recognition and anomaly detection
    - burner_health: Burner component health scoring and maintenance planning
    - maintenance_prediction: Weibull-based RUL prediction + ML failure prediction
    - fuel_impact: Fuel quality impact on burner degradation

Standards Compliance:
    - API 535: Burners for Fired Heaters in General Refinery Services
    - API 560: Fired Heaters for General Refinery Service
    - API 571: Damage Mechanisms Affecting Fixed Equipment
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - IEC 61511: Functional Safety for Process Industry

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance import (
    ...     FlamePatternAnalyzer,
    ...     BurnerHealthAnalyzer,
    ...     MaintenancePredictionEngine,
    ...     FuelImpactAnalyzer,
    ... )
    >>>
    >>> # Flame analysis
    >>> flame_analyzer = FlamePatternAnalyzer()
    >>> flame_result = flame_analyzer.analyze(flame_input)
    >>> print(f"Flame Quality Score: {flame_result.quality_score:.1f}")
    >>>
    >>> # Burner health
    >>> health_analyzer = BurnerHealthAnalyzer()
    >>> health_result = health_analyzer.analyze(health_input)
    >>> print(f"Overall Health: {health_result.overall_health_score:.1f}")
    >>>
    >>> # Maintenance prediction
    >>> engine = MaintenancePredictionEngine()
    >>> rul_result = engine.predict_rul(current_age_hours=15000)
    >>> print(f"RUL P50: {rul_result.rul_p50_hours:.0f} hours")
    >>>
    >>> # Fuel impact analysis
    >>> fuel_analyzer = FuelImpactAnalyzer()
    >>> quality = fuel_analyzer.calculate_fuel_quality_score(fuel_properties)
    >>> print(f"Quality: {quality.overall_score:.0f}")

Author: GreenLang Process Heat Team
Version: 1.1.0
Status: Production Ready
"""

from greenlang.agents.process_heat.gl_021_burner_maintenance.flame_analysis import (
    # Main analyzer
    FlamePatternAnalyzer,
    # Input schemas
    FlameAnalysisInput,
    FlameScannerSignal,
    FlameGeometryInput,
    FlameTemperatureProfile,
    # Output schemas
    FlameAnalysisOutput,
    FlameStabilityResult,
    FlameGeometryResult,
    TemperatureProfileResult,
    ColorIndexResult,
    PulsationResult,
    SPCResult,
    # Enums
    FlameStatus,
    FlameColor,
    AnomalyType,
    AlertSeverity as FlameAlertSeverity,
    # Utility functions
    create_default_flame_analyzer,
    quick_stability_check,
)

from greenlang.agents.process_heat.gl_021_burner_maintenance.burner_health import (
    # Main analyzer
    BurnerHealthAnalyzer,
    # Input schemas
    BurnerHealthInput,
    NozzleData,
    RefractoryTileData,
    IgniterData,
    FlameScannerData,
    AirRegisterData,
    FuelValveData,
    # Output schemas
    BurnerHealthOutput,
    ComponentHealthResult,
    # Component health models
    NozzleHealthModel,
    RefractoryHealthModel,
    IgniterHealthModel,
    ScannerHealthModel,
    AirRegisterHealthModel,
    FuelValveHealthModel,
    # Enums
    HealthStatus,
    MaintenancePriority,
    FailureMode,
    AlertSeverity as HealthAlertSeverity,
    # Utility functions
    create_default_health_analyzer,
    quick_health_check,
)

# Maintenance Prediction Engine (NEW)
from greenlang.agents.process_heat.gl_021_burner_maintenance.maintenance_prediction import (
    # Main engine
    MaintenancePredictionEngine,
    PredictionEngineConfig,
    # Sub-components
    WeibullAnalyzer,
    WeibullAnalysisConfig,
    ProportionalHazardsModel,
    MLFailurePredictor,
    EnsemblePrediction,
    # Data classes
    FailureData,
    WeibullParameters,
    OperatingConditions as PredictionOperatingConditions,
    # Pydantic models
    RULPredictionResult,
    FailurePredictionResult,
    # Enums
    BurnerComponent,
    FailureMode as PredictionFailureMode,
    PredictionConfidence,
    MaintenanceUrgency,
    # Convenience functions
    quick_rul_prediction,
    calculate_b10_life,
    get_component_mtbf,
)

# Fuel Impact Analysis (NEW)
from greenlang.agents.process_heat.gl_021_burner_maintenance.fuel_impact import (
    # Main analyzer
    FuelImpactAnalyzer,
    # Data classes
    FuelProperties,
    OperatingConditions as FuelOperatingConditions,
    FoulingResult,
    CokingResult,
    # Pydantic models
    FuelQualityScore,
    DegradationImpact,
    FuelSwitchingImpact,
    # Enums
    FuelType,
    DamageMechanism,
    ImpactSeverity,
    FoulingLevel,
    # Convenience functions
    quick_fuel_quality_check,
    calculate_sulfur_corrosion_rate,
    get_vanadium_inhibitor_ratio,
    estimate_fouling_efficiency_loss,
)

# Replacement Planner (NEW)
from greenlang.agents.process_heat.gl_021_burner_maintenance.replacement_planner import (
    # Main class
    ReplacementPlanner,
    # Supporting classes
    EconomicReplacementModel,
    OptimalTimingCalculator,
    GroupReplacementStrategy,
    InventoryOptimizer,
    MonteCarloSimulator,
    # Data models
    BurnerAsset,
    EconomicParameters,
    ReplacementAnalysisResult,
    GroupReplacementResult,
    SparePartRecommendation,
    InventoryOptimizationResult,
    FailureCostModel,
    # Enums
    BurnerType,
    ReplacementStrategy,
    CriticalityLevel,
    SparePartCategory,
    OutageType,
    # Factory functions
    create_replacement_planner,
    create_burner_asset,
)

# CMMS Integration (NEW)
from greenlang.agents.process_heat.gl_021_burner_maintenance.cmms_integration import (
    # Main class
    CMSIntegration,
    # Supporting classes
    WorkOrderGenerator,
    PriorityCalculator,
    ResourcePlanner,
    # Adapters
    CMMSAdapter,
    SAPPMAdapter,
    MaximoAdapter,
    eMaintAdapter,
    MockCMMSAdapter,
    # Configs
    SAPPMConfig,
    MaximoConfig,
    eMaintConfig,
    # Data models
    WorkOrder,
    WorkOrderTemplate,
    SparePart as CMSSSparePart,
    LaborResource,
    SafetyRequirement,
    CMMSResponse,
    PredictionInput,
    # Enums
    WorkOrderPriority,
    WorkOrderType,
    WorkOrderStatus,
    CMMSType,
    MaintenanceTask,
    FailureMode as CMSSFailureMode,
    CriticalityLevel as CMSSCriticalityLevel,
    # Factory functions
    create_cms_integration,
)

# Monitoring Interface (NEW)
from greenlang.agents.process_heat.gl_021_burner_maintenance.monitoring_interface import (
    # Main class
    BurnerMonitoringInterface,
    # Supporting classes
    FlamescannerInterface,
    BMSInterface,
    AnalyzerInterface,
    HistorianInterface,
    # Data models
    FlamescannerReading,
    BMSStatus,
    AnalyzerReading,
    HistorianDataPoint,
    HistorianQuery,
    HistorianTag,
    MonitoringConfig,
    # Enums
    FlameDetectorType,
    FlameStatus as MonitoringFlameStatus,
    BMSState,
    InterlockStatus,
    AnalyzerType,
    CommunicationProtocol,
    HistorianType,
    # Factory functions
    create_monitoring_interface,
    quick_burner_status,
)

__all__ = [
    # Flame Analysis
    "FlamePatternAnalyzer",
    "FlameAnalysisInput",
    "FlameScannerSignal",
    "FlameGeometryInput",
    "FlameTemperatureProfile",
    "FlameAnalysisOutput",
    "FlameStabilityResult",
    "FlameGeometryResult",
    "TemperatureProfileResult",
    "ColorIndexResult",
    "PulsationResult",
    "SPCResult",
    "FlameStatus",
    "FlameColor",
    "AnomalyType",
    "FlameAlertSeverity",
    "create_default_flame_analyzer",
    "quick_stability_check",
    # Burner Health
    "BurnerHealthAnalyzer",
    "BurnerHealthInput",
    "NozzleData",
    "RefractoryTileData",
    "IgniterData",
    "FlameScannerData",
    "AirRegisterData",
    "FuelValveData",
    "BurnerHealthOutput",
    "ComponentHealthResult",
    "NozzleHealthModel",
    "RefractoryHealthModel",
    "IgniterHealthModel",
    "ScannerHealthModel",
    "AirRegisterHealthModel",
    "FuelValveHealthModel",
    "HealthStatus",
    "MaintenancePriority",
    "FailureMode",
    "HealthAlertSeverity",
    "create_default_health_analyzer",
    "quick_health_check",
    # Maintenance Prediction (NEW)
    "MaintenancePredictionEngine",
    "PredictionEngineConfig",
    "WeibullAnalyzer",
    "WeibullAnalysisConfig",
    "ProportionalHazardsModel",
    "MLFailurePredictor",
    "EnsemblePrediction",
    "FailureData",
    "WeibullParameters",
    "PredictionOperatingConditions",
    "RULPredictionResult",
    "FailurePredictionResult",
    "BurnerComponent",
    "PredictionFailureMode",
    "PredictionConfidence",
    "MaintenanceUrgency",
    "quick_rul_prediction",
    "calculate_b10_life",
    "get_component_mtbf",
    # Fuel Impact Analysis (NEW)
    "FuelImpactAnalyzer",
    "FuelProperties",
    "FuelOperatingConditions",
    "FoulingResult",
    "CokingResult",
    "FuelQualityScore",
    "DegradationImpact",
    "FuelSwitchingImpact",
    "FuelType",
    "DamageMechanism",
    "ImpactSeverity",
    "FoulingLevel",
    "quick_fuel_quality_check",
    "calculate_sulfur_corrosion_rate",
    "get_vanadium_inhibitor_ratio",
    "estimate_fouling_efficiency_loss",
    # Replacement Planner (NEW)
    "ReplacementPlanner",
    "EconomicReplacementModel",
    "OptimalTimingCalculator",
    "GroupReplacementStrategy",
    "InventoryOptimizer",
    "MonteCarloSimulator",
    "BurnerAsset",
    "EconomicParameters",
    "ReplacementAnalysisResult",
    "GroupReplacementResult",
    "SparePartRecommendation",
    "InventoryOptimizationResult",
    "FailureCostModel",
    "BurnerType",
    "ReplacementStrategy",
    "CriticalityLevel",
    "SparePartCategory",
    "OutageType",
    "create_replacement_planner",
    "create_burner_asset",
    # CMMS Integration (NEW)
    "CMSIntegration",
    "WorkOrderGenerator",
    "PriorityCalculator",
    "ResourcePlanner",
    "CMMSAdapter",
    "SAPPMAdapter",
    "MaximoAdapter",
    "eMaintAdapter",
    "MockCMMSAdapter",
    "SAPPMConfig",
    "MaximoConfig",
    "eMaintConfig",
    "WorkOrder",
    "WorkOrderTemplate",
    "CMSSSparePart",
    "LaborResource",
    "SafetyRequirement",
    "CMMSResponse",
    "PredictionInput",
    "WorkOrderPriority",
    "WorkOrderType",
    "WorkOrderStatus",
    "CMMSType",
    "MaintenanceTask",
    "CMSSFailureMode",
    "CMSSCriticalityLevel",
    "create_cms_integration",
    # Monitoring Interface (NEW)
    "BurnerMonitoringInterface",
    "FlamescannerInterface",
    "BMSInterface",
    "AnalyzerInterface",
    "HistorianInterface",
    "FlamescannerReading",
    "BMSStatus",
    "AnalyzerReading",
    "HistorianDataPoint",
    "HistorianQuery",
    "HistorianTag",
    "MonitoringConfig",
    "FlameDetectorType",
    "MonitoringFlameStatus",
    "BMSState",
    "InterlockStatus",
    "AnalyzerType",
    "CommunicationProtocol",
    "HistorianType",
    "create_monitoring_interface",
    "quick_burner_status",
]

__version__ = "1.2.0"
__author__ = "GreenLang Process Heat Team"
