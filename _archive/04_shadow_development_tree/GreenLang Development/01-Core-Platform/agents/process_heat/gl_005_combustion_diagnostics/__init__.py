# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE - Combustion Diagnostics Agent
================================================

This module provides the GL-005 COMBUSENSE (Combustion Diagnostics) agent
and all its components for comprehensive combustion analysis.

AGENT BOUNDARY:
    GL-005 is a DIAGNOSTICS-ONLY agent. It analyzes combustion data and
    generates recommendations but does NOT execute control actions.

    - READS: Sensor data from GL-018 or direct sensors
    - ANALYZES: CQI, anomalies, fuel quality, equipment health
    - RECOMMENDS: Maintenance actions, control adjustments
    - GENERATES: Work orders, compliance reports, trending analysis

Components:
    - CombustionDiagnosticsAgent: Main agent class
    - CombustionQualityCalculator: CQI calculation engine
    - CombustionAnomalyDetector: SPC + ML anomaly detection
    - FuelCharacterizationEngine: Fuel analysis from flue gas
    - MaintenanceAdvisor: Maintenance prediction and work orders
    - TrendingEngine: Long-term trending and analysis

Example:
    >>> from greenlang.agents.process_heat.gl_005_combustion_diagnostics import (
    ...     CombustionDiagnosticsAgent,
    ...     GL005Config,
    ...     FuelCategory,
    ... )
    >>>
    >>> config = GL005Config(
    ...     agent_id="GL005-BOILER-01",
    ...     equipment_id="BLR-001",
    ...     primary_fuel=FuelCategory.NATURAL_GAS,
    ... )
    >>> agent = CombustionDiagnosticsAgent(config)

ZERO-HALLUCINATION GUARANTEE:
    All calculations are deterministic using documented engineering methods.
    No AI/ML in critical calculation paths.
    Full audit trail with SHA-256 provenance hashes.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

# Configuration
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    GL005Config,
    CQIConfig,
    CQIWeights,
    CQIThresholds,
    AnomalyDetectionConfig,
    SPCConfig,
    MLAnomalyConfig,
    FuelCharacterizationConfig,
    MaintenanceAdvisoryConfig,
    FoulingPredictionConfig,
    BurnerWearConfig,
    TrendingConfig,
    ComplianceConfig,
    DiagnosticMode,
    ComplianceFramework,
    FuelCategory,
    MaintenancePriority,
    AnomalyType,
    create_default_config,
    create_high_precision_config,
    create_compliance_focused_config,
)

# Schemas
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    # Input schemas
    FlueGasReading,
    CombustionOperatingData,
    DiagnosticsInput,
    # Output schemas
    CQIResult,
    CQIComponentScore,
    CQIRating,
    AnomalyDetectionResult,
    AnomalyEvent,
    AnomalySeverity,
    FuelCharacterizationResult,
    FuelProperties,
    MaintenanceAdvisoryResult,
    MaintenanceRecommendation,
    FoulingAssessment,
    BurnerWearAssessment,
    CMMSWorkOrder,
    ComplianceReportResult,
    ComplianceStatus,
    DiagnosticsOutput,
    # Enums
    AnalysisStatus,
    TrendDirection,
)

# Main agent
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.diagnostics import (
    CombustionDiagnosticsAgent,
    create_combustion_diagnostics_agent,
)

# Calculation engines
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.combustion_quality import (
    CombustionQualityCalculator,
    calculate_cqi_quick,
    create_default_cqi_calculator,
)

# Anomaly detection
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.anomaly_detection import (
    CombustionAnomalyDetector,
    SPCAnalyzer,
    MLAnomalyDetector,
    RuleBasedDetector,
)

# Fuel characterization
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.fuel_characterization import (
    FuelCharacterizationEngine,
    get_fuel_reference,
    calculate_emission_factor,
    estimate_fuel_consumption,
)

# Maintenance advisory
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.maintenance_advisor import (
    MaintenanceAdvisor,
    FoulingPredictor,
    BurnerWearAssessor,
    WorkOrderGenerator,
)

# Trending
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.trending import (
    TrendingEngine,
    TrendAnalyzer,
    TimeSeriesStore,
    TrendAnalysisResult,
    SeasonalityResult,
    BaselineComparison,
    calculate_moving_average,
    calculate_exponential_moving_average,
)


__all__ = [
    # Main agent
    "CombustionDiagnosticsAgent",
    "create_combustion_diagnostics_agent",

    # Configuration
    "GL005Config",
    "CQIConfig",
    "CQIWeights",
    "CQIThresholds",
    "AnomalyDetectionConfig",
    "SPCConfig",
    "MLAnomalyConfig",
    "FuelCharacterizationConfig",
    "MaintenanceAdvisoryConfig",
    "FoulingPredictionConfig",
    "BurnerWearConfig",
    "TrendingConfig",
    "ComplianceConfig",
    "DiagnosticMode",
    "ComplianceFramework",
    "FuelCategory",
    "MaintenancePriority",
    "AnomalyType",
    "create_default_config",
    "create_high_precision_config",
    "create_compliance_focused_config",

    # Input schemas
    "FlueGasReading",
    "CombustionOperatingData",
    "DiagnosticsInput",

    # Output schemas
    "CQIResult",
    "CQIComponentScore",
    "CQIRating",
    "AnomalyDetectionResult",
    "AnomalyEvent",
    "AnomalySeverity",
    "FuelCharacterizationResult",
    "FuelProperties",
    "MaintenanceAdvisoryResult",
    "MaintenanceRecommendation",
    "FoulingAssessment",
    "BurnerWearAssessment",
    "CMMSWorkOrder",
    "ComplianceReportResult",
    "ComplianceStatus",
    "DiagnosticsOutput",
    "AnalysisStatus",
    "TrendDirection",

    # Calculation engines
    "CombustionQualityCalculator",
    "calculate_cqi_quick",
    "create_default_cqi_calculator",

    # Anomaly detection
    "CombustionAnomalyDetector",
    "SPCAnalyzer",
    "MLAnomalyDetector",
    "RuleBasedDetector",

    # Fuel characterization
    "FuelCharacterizationEngine",
    "get_fuel_reference",
    "calculate_emission_factor",
    "estimate_fuel_consumption",

    # Maintenance advisory
    "MaintenanceAdvisor",
    "FoulingPredictor",
    "BurnerWearAssessor",
    "WorkOrderGenerator",

    # Trending
    "TrendingEngine",
    "TrendAnalyzer",
    "TimeSeriesStore",
    "TrendAnalysisResult",
    "SeasonalityResult",
    "BaselineComparison",
    "calculate_moving_average",
    "calculate_exponential_moving_average",
]

__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"
__agent_type__ = "GL-005"
__agent_name__ = "COMBUSENSE"
