# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY - Burner Maintenance Predictor
===================================================

This module implements the main BurnerMaintenancePredictor class that orchestrates
all sub-analyzers for comprehensive industrial burner maintenance prediction.

The predictor integrates:
    - FlamePatternAnalyzer: Flame stability, geometry, and anomaly detection
    - BurnerHealthAnalyzer: Component health scoring per API 535
    - FuelImpactAnalyzer: Fuel quality impact on degradation
    - MaintenancePredictionEngine: Weibull RUL + ML failure prediction
    - ReplacementPlanner: Economic replacement optimization
    - CMSIntegration: CMMS work order generation
    - GL021Explainer: SHAP/LIME explainability for predictions

ZERO-HALLUCINATION GUARANTEE:
    All calculations are deterministic using documented engineering formulas.
    No LLM/AI is used in the calculation path.
    Full provenance tracking with SHA-256 hashes.

Standards Compliance:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - API 535: Burners for Fired Heaters in General Refinery Service
    - API 560: Fired Heaters for General Refinery Service
    - API 556: Instrumentation, Control, and Protective Systems
    - ISA 84: Safety Instrumented Systems

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.predictor import (
    ...     BurnerMaintenancePredictor
    ... )
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.config import GL021Config
    >>> config = GL021Config(burner=BurnerConfig(burner_id="BNR-001", ...))
    >>> predictor = BurnerMaintenancePredictor(config)
    >>> result = predictor.analyze(input_data)
    >>> print(f"Health Score: {result.health_score:.1f}")
    >>> print(f"RUL P50: {result.rul_p50_hours:.0f} hours")

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

# Import sub-analyzers
from greenlang.agents.process_heat.gl_021_burner_maintenance.flame_analysis import (
    FlamePatternAnalyzer,
    FlameAnalysisInput,
    FlameAnalysisOutput,
    FlameScannerSignal,
    FlameGeometryInput,
    FlameTemperatureProfile,
)
from greenlang.agents.process_heat.gl_021_burner_maintenance.burner_health import (
    BurnerHealthAnalyzer,
    BurnerHealthInput,
    BurnerHealthOutput,
    NozzleData,
    RefractoryTileData,
    IgniterData,
    FlameScannerData,
    AirRegisterData,
    FuelValveData,
    HealthStatus,
    MaintenancePriority,
)
from greenlang.agents.process_heat.gl_021_burner_maintenance.fuel_impact import (
    FuelImpactAnalyzer,
    FuelProperties,
    OperatingConditions as FuelOperatingConditions,
    FuelQualityScore,
    DegradationImpact,
    FuelType,
)
from greenlang.agents.process_heat.gl_021_burner_maintenance.maintenance_prediction import (
    MaintenancePredictionEngine,
    PredictionEngineConfig,
    RULPredictionResult,
    FailurePredictionResult,
    FailureData,
    OperatingConditions as PredictionOperatingConditions,
    BurnerComponent,
    MaintenanceUrgency,
    PredictionConfidence,
)
from greenlang.agents.process_heat.gl_021_burner_maintenance.replacement_planner import (
    ReplacementPlanner,
    BurnerAsset,
    EconomicParameters,
    ReplacementAnalysisResult,
    GroupReplacementResult,
    BurnerType as ReplacementBurnerType,
)
from greenlang.agents.process_heat.gl_021_burner_maintenance.cmms_integration import (
    CMSIntegration,
    WorkOrderGenerator,
    PredictionInput,
    WorkOrder,
    WorkOrderPriority,
    CriticalityLevel,
)
from greenlang.agents.process_heat.gl_021_burner_maintenance.explainability import (
    GL021SHAPExplainer,
    GL021ExplanationResult,
    ExplanationAudience,
)
from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
    GL021Config,
    BurnerConfig,
    FlameAnalysisConfig,
    MaintenancePredictionConfig,
    FuelQualityConfig,
    ReplacementPlanningConfig,
    CMSIntegrationConfig,
    SafetyConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class PredictionStatus(str, Enum):
    """Prediction status indicators."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


class OverallHealthStatus(str, Enum):
    """Overall burner health status classification."""
    EXCELLENT = "excellent"  # Score >= 90
    GOOD = "good"            # Score 70-89
    FAIR = "fair"            # Score 50-69
    POOR = "poor"            # Score 25-49
    CRITICAL = "critical"    # Score < 25


class AlertLevel(str, Enum):
    """Alert levels for maintenance notifications."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class GL021Input(BaseModel):
    """
    Comprehensive input data for GL-021 BURNERSENTRY analysis.

    This model encapsulates all operational data required for complete
    burner maintenance prediction including flame data, component health,
    fuel quality, and operating conditions.

    Attributes:
        burner_id: Unique burner identifier
        timestamp: Analysis timestamp
        flame_data: Flame scanner and geometry data
        component_data: Component health metrics
        fuel_data: Fuel quality properties
        operating_data: Current operating conditions

    Example:
        >>> input_data = GL021Input(
        ...     burner_id="BNR-001",
        ...     flame_data=flame_input,
        ...     component_data=component_input,
        ... )
    """

    # Identification
    request_id: str = Field(
        default_factory=lambda: f"gl021_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
        description="Unique request identifier"
    )
    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Flame analysis data
    flame_scanner_signals: List[FlameScannerSignal] = Field(
        default_factory=list,
        description="Flame scanner signal readings"
    )
    flame_signal_history: List[float] = Field(
        default_factory=list,
        description="Historical flame signal for FFT analysis"
    )
    flame_geometry: Optional[FlameGeometryInput] = Field(
        default=None,
        description="Flame geometry measurements"
    )
    flame_temperature_profile: Optional[FlameTemperatureProfile] = Field(
        default=None,
        description="Flame temperature distribution"
    )
    flame_color_rgb: Optional[Tuple[int, int, int]] = Field(
        default=None,
        description="Flame color as RGB"
    )
    flame_luminosity_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Flame luminosity percentage"
    )

    # Component health data
    nozzle_data: Optional[NozzleData] = Field(
        default=None,
        description="Nozzle operational data"
    )
    refractory_data: Optional[RefractoryTileData] = Field(
        default=None,
        description="Refractory tile data"
    )
    igniter_data: Optional[IgniterData] = Field(
        default=None,
        description="Igniter system data"
    )
    scanner_data: Optional[FlameScannerData] = Field(
        default=None,
        description="Flame scanner health data"
    )
    air_register_data: Optional[AirRegisterData] = Field(
        default=None,
        description="Air register/damper data"
    )
    fuel_valve_data: Optional[FuelValveData] = Field(
        default=None,
        description="Fuel valve data"
    )

    # Fuel quality data
    fuel_type: str = Field(
        default="natural_gas",
        description="Fuel type"
    )
    fuel_sulfur_pct: float = Field(
        default=0.0,
        ge=0,
        description="Fuel sulfur content (%)"
    )
    fuel_h2s_ppm: float = Field(
        default=0.0,
        ge=0,
        description="H2S content for gas fuels (ppm)"
    )
    fuel_vanadium_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Vanadium content (ppm)"
    )
    fuel_sodium_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Sodium content (ppm)"
    )
    fuel_ash_pct: float = Field(
        default=0.0,
        ge=0,
        description="Ash content (%)"
    )
    fuel_water_pct: float = Field(
        default=0.0,
        ge=0,
        description="Water content (%)"
    )
    fuel_carbon_residue_pct: float = Field(
        default=0.0,
        ge=0,
        description="Carbon residue (%)"
    )
    fuel_heating_value_mj_kg: float = Field(
        default=42.5,
        gt=0,
        description="Lower heating value (MJ/kg)"
    )

    # Operating conditions
    operating_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total operating hours"
    )
    start_stop_cycles: int = Field(
        default=0,
        ge=0,
        description="Total start/stop cycles"
    )
    firing_rate_pct: float = Field(
        default=75.0,
        ge=0,
        le=100,
        description="Current firing rate (%)"
    )
    flame_temperature_c: float = Field(
        default=1200.0,
        description="Average flame temperature (C)"
    )
    excess_air_pct: float = Field(
        default=15.0,
        ge=0,
        description="Excess air percentage"
    )
    air_fuel_ratio: float = Field(
        default=10.5,
        gt=0,
        description="Air-fuel ratio"
    )
    flue_gas_temperature_c: float = Field(
        default=350.0,
        description="Flue gas temperature (C)"
    )
    combustion_air_temp_c: float = Field(
        default=25.0,
        description="Combustion air temperature (C)"
    )
    ambient_humidity_pct: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Ambient humidity (%)"
    )
    cycling_frequency_per_day: float = Field(
        default=2.0,
        ge=0,
        description="Start/stop cycles per day"
    )

    # Historical data
    historical_stability_indices: List[float] = Field(
        default_factory=list,
        description="Historical flame stability indices"
    )
    failure_history: List[FailureData] = Field(
        default_factory=list,
        description="Historical failure data for Weibull"
    )
    previous_health_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Previous health assessment score"
    )
    previous_assessment_date: Optional[datetime] = Field(
        default=None,
        description="Date of previous assessment"
    )

    # Sensor data (for ML features)
    sensor_data: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional sensor readings"
    )

    # Analysis options
    include_flame_analysis: bool = Field(
        default=True,
        description="Include flame pattern analysis"
    )
    include_health_analysis: bool = Field(
        default=True,
        description="Include component health analysis"
    )
    include_fuel_analysis: bool = Field(
        default=True,
        description="Include fuel impact analysis"
    )
    include_rul_prediction: bool = Field(
        default=True,
        description="Include RUL prediction"
    )
    include_replacement_analysis: bool = Field(
        default=False,
        description="Include replacement economics"
    )
    include_work_order_generation: bool = Field(
        default=False,
        description="Generate CMMS work orders"
    )
    include_explainability: bool = Field(
        default=True,
        description="Include SHAP/LIME explanations"
    )
    explanation_audience: ExplanationAudience = Field(
        default=ExplanationAudience.ENGINEER,
        description="Target audience for explanations"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GL021Alert(BaseModel):
    """Alert generated by GL-021 analysis."""

    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp"
    )
    level: AlertLevel = Field(..., description="Alert severity level")
    category: str = Field(..., description="Alert category")
    message: str = Field(..., description="Alert message")
    source_component: str = Field(..., description="Source of alert")
    action_required: bool = Field(default=False, description="Action required")
    recommended_action: Optional[str] = Field(default=None, description="Recommended action")


class GL021Result(BaseModel):
    """
    Comprehensive result from GL-021 BURNERSENTRY analysis.

    This model contains all analysis outputs including flame quality,
    component health, RUL predictions, recommendations, and work orders.

    Attributes:
        burner_id: Burner identifier
        overall_health_score: Weighted overall health (0-100)
        health_status: Status classification
        flame_quality_score: Flame quality score (0-100)
        rul_p50_hours: Median remaining useful life
        maintenance_urgency: Urgency classification
        recommendations: Prioritized recommendations
        work_orders: Generated work orders
        explanation: Multi-audience explanations

    Example:
        >>> result = predictor.analyze(input_data)
        >>> print(f"Health: {result.overall_health_score:.1f}")
        >>> for rec in result.recommendations[:3]:
        ...     print(f"  - {rec}")
    """

    # Identification
    request_id: str = Field(..., description="Original request ID")
    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    status: PredictionStatus = Field(..., description="Analysis status")
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total processing time (ms)"
    )

    # Overall assessment
    overall_health_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall burner health score (0-100)"
    )
    health_status: OverallHealthStatus = Field(
        ...,
        description="Health status classification"
    )

    # Flame analysis results
    flame_quality_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Flame quality score (0-100)"
    )
    flame_stability_index: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Flame stability index"
    )
    flame_status: Optional[str] = Field(
        default=None,
        description="Flame status (stable, marginal, unstable)"
    )
    flame_anomalies: List[str] = Field(
        default_factory=list,
        description="Detected flame anomalies"
    )

    # Component health results
    component_health_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Component health score (0-100)"
    )
    limiting_component: Optional[str] = Field(
        default=None,
        description="Component with lowest health"
    )
    limiting_component_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Score of limiting component"
    )
    component_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual component scores"
    )

    # Fuel impact results
    fuel_quality_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Fuel quality score (0-100)"
    )
    fuel_life_reduction_factor: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=10.0,
        description="Life reduction from fuel quality"
    )
    fuel_concerns: List[str] = Field(
        default_factory=list,
        description="Fuel quality concerns"
    )

    # RUL prediction results
    rul_p10_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="RUL at 10% failure probability"
    )
    rul_p50_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="RUL at 50% failure probability"
    )
    rul_p90_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="RUL at 90% failure probability"
    )
    current_failure_probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Current cumulative failure probability"
    )
    failure_probability_30d: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="30-day failure probability"
    )
    maintenance_urgency: MaintenanceUrgency = Field(
        default=MaintenanceUrgency.MONITOR,
        description="Maintenance urgency"
    )
    prediction_confidence: PredictionConfidence = Field(
        default=PredictionConfidence.MEDIUM,
        description="Prediction confidence level"
    )

    # Weibull parameters
    weibull_beta: Optional[float] = Field(
        default=None,
        gt=0,
        description="Weibull shape parameter"
    )
    weibull_eta_hours: Optional[float] = Field(
        default=None,
        gt=0,
        description="Weibull scale parameter"
    )

    # Failure mode predictions
    predicted_failure_modes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Predicted failure modes with probabilities"
    )
    primary_failure_mode: Optional[str] = Field(
        default=None,
        description="Most likely failure mode"
    )

    # Replacement analysis
    replacement_npv: Optional[float] = Field(
        default=None,
        description="NPV of replacement ($)"
    )
    replacement_irr: Optional[float] = Field(
        default=None,
        description="IRR of replacement"
    )
    optimal_replacement_date: Optional[datetime] = Field(
        default=None,
        description="Optimal replacement date"
    )
    annual_savings_potential: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual savings from replacement ($)"
    )

    # Recommendations and alerts
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritized maintenance recommendations"
    )
    alerts: List[GL021Alert] = Field(
        default_factory=list,
        description="Generated alerts"
    )

    # Work orders
    work_orders: List[WorkOrder] = Field(
        default_factory=list,
        description="Generated CMMS work orders"
    )

    # Trend analysis
    health_trend: str = Field(
        default="stable",
        description="Health trend (improving, stable, degrading)"
    )
    trend_rate_pct_per_month: Optional[float] = Field(
        default=None,
        description="Health change rate (% per month)"
    )

    # Explanation
    explanation: Optional[GL021ExplanationResult] = Field(
        default=None,
        description="Multi-audience explanations"
    )
    natural_language_summary: Optional[str] = Field(
        default=None,
        description="Natural language summary"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    sub_analysis_hashes: Dict[str, str] = Field(
        default_factory=dict,
        description="Provenance hashes from sub-analyzers"
    )
    calculation_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Calculation details for audit"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# BURNER MAINTENANCE PREDICTOR CLASS
# =============================================================================

class BurnerMaintenancePredictor:
    """
    GL-021 BURNERSENTRY - Main Burner Maintenance Predictor.

    Orchestrates comprehensive burner maintenance prediction by integrating:
    - Flame pattern analysis (stability, geometry, color, pulsation)
    - Component health scoring (nozzle, refractory, igniter, scanner, valves)
    - Fuel quality impact assessment
    - Weibull-based RUL prediction with ML failure mode prediction
    - Economic replacement optimization
    - CMMS work order generation
    - SHAP/LIME explainability

    ZERO-HALLUCINATION: All predictions are deterministic using documented
    engineering formulas. No LLM in calculation path.

    AUDITABLE: Full calculation trace with SHA-256 provenance hashes.

    Standards:
        - NFPA 85: Boiler and Combustion Systems Hazards Code
        - API 535: Burners for Fired Heaters
        - API 560: Fired Heaters for General Refinery Service
        - API 556: Instrumentation, Control, and Protective Systems

    Attributes:
        config: GL021 configuration
        flame_analyzer: Flame pattern analyzer
        health_analyzer: Component health analyzer
        fuel_analyzer: Fuel impact analyzer
        prediction_engine: RUL/failure prediction engine
        replacement_planner: Economic replacement planner
        cmms_integration: CMMS integration for work orders
        explainer: SHAP/LIME explainer

    Example:
        >>> config = GL021Config(burner=BurnerConfig(burner_id="BNR-001", ...))
        >>> predictor = BurnerMaintenancePredictor(config)
        >>> result = predictor.analyze(input_data)
        >>> print(f"Health: {result.overall_health_score:.1f}")
        >>> print(f"RUL: {result.rul_p50_hours:.0f} hours")
        >>> print(f"Urgency: {result.maintenance_urgency}")
    """

    # Component weights for overall health score
    DEFAULT_HEALTH_WEIGHTS = {
        "flame": 0.25,
        "component": 0.35,
        "fuel": 0.15,
        "prediction": 0.25,
    }

    def __init__(
        self,
        config: GL021Config,
        health_weights: Optional[Dict[str, float]] = None,
        enable_async: bool = True,
        thread_pool_size: int = 4,
    ) -> None:
        """
        Initialize BurnerMaintenancePredictor.

        Args:
            config: GL021 master configuration
            health_weights: Custom weights for overall health calculation
            enable_async: Enable async analysis methods
            thread_pool_size: Thread pool size for parallel analysis
        """
        self.config = config
        self.health_weights = health_weights or self.DEFAULT_HEALTH_WEIGHTS
        self.enable_async = enable_async

        # Validate weights sum to 1.0
        total_weight = sum(self.health_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Health weights must sum to 1.0, got {total_weight:.3f}")

        # Initialize sub-analyzers
        self._init_flame_analyzer()
        self._init_health_analyzer()
        self._init_fuel_analyzer()
        self._init_prediction_engine()
        self._init_replacement_planner()
        self._init_cmms_integration()
        self._init_explainer()

        # Thread pool for parallel analysis
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)

        # Tracking
        self._analysis_count = 0
        self._audit_trail: List[Dict[str, Any]] = []

        logger.info(
            f"BurnerMaintenancePredictor initialized for {config.burner.burner_id}: "
            f"flame={self.flame_analyzer is not None}, "
            f"health={self.health_analyzer is not None}, "
            f"fuel={self.fuel_analyzer is not None}, "
            f"prediction={self.prediction_engine is not None}, "
            f"replacement={self.replacement_planner is not None}, "
            f"cmms={self.cmms_integration is not None}, "
            f"explainer={self.explainer is not None}"
        )

    def _init_flame_analyzer(self) -> None:
        """Initialize flame pattern analyzer."""
        if self.config.flame_analysis.enabled:
            self.flame_analyzer = FlamePatternAnalyzer(
                stability_weight=0.30,
                geometry_weight=0.20,
                temperature_weight=0.15,
                color_weight=0.15,
                pulsation_weight=0.20,
            )
        else:
            self.flame_analyzer = None
            logger.info("Flame analysis disabled in configuration")

    def _init_health_analyzer(self) -> None:
        """Initialize burner health analyzer."""
        self.health_analyzer = BurnerHealthAnalyzer()

    def _init_fuel_analyzer(self) -> None:
        """Initialize fuel impact analyzer."""
        self.fuel_analyzer = FuelImpactAnalyzer()

    def _init_prediction_engine(self) -> None:
        """Initialize maintenance prediction engine."""
        pred_config = PredictionEngineConfig(
            component=BurnerComponent.BURNER_TIP,
            enable_cox_model=True,
            enable_ml_prediction=self.config.maintenance_prediction.ml_model_enabled,
            enable_ensemble=True,
            ensemble_weights={
                "weibull": 0.40,
                "cox": 0.30,
                "ml": 0.30,
            },
        )
        self.prediction_engine = MaintenancePredictionEngine(pred_config)

    def _init_replacement_planner(self) -> None:
        """Initialize replacement planner."""
        if self.config.replacement_planning.economic_optimization_enabled:
            self.replacement_planner = ReplacementPlanner(
                enable_monte_carlo=True,
                monte_carlo_simulations=500,
            )
        else:
            self.replacement_planner = None
            logger.info("Replacement planning disabled in configuration")

    def _init_cmms_integration(self) -> None:
        """Initialize CMMS integration."""
        if self.config.cmms.enabled:
            self.cmms_integration = CMSIntegration(self.config.cmms)
        else:
            self.cmms_integration = None
            logger.info("CMMS integration disabled in configuration")

    def _init_explainer(self) -> None:
        """Initialize SHAP/LIME explainer."""
        self.explainer = GL021SHAPExplainer()

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(self, input_data: GL021Input) -> GL021Result:
        """
        Perform comprehensive burner maintenance analysis.

        This is the main entry point for GL-021 analysis. It orchestrates:
        1. Flame pattern analysis (stability, geometry, anomalies)
        2. Component health scoring (per API 535)
        3. Fuel quality impact assessment
        4. RUL prediction (Weibull + ML)
        5. Replacement economics (optional)
        6. Work order generation (optional)
        7. Explainability generation

        Args:
            input_data: Comprehensive input data for analysis

        Returns:
            GL021Result with complete analysis results

        Raises:
            ValueError: If input data is insufficient
        """
        start_time = datetime.now(timezone.utc)
        self._analysis_count += 1
        self._audit_trail = []

        logger.info(
            f"Starting GL-021 analysis #{self._analysis_count} for {input_data.burner_id}"
        )

        self._add_audit_entry("analysis_start", {
            "request_id": input_data.request_id,
            "burner_id": input_data.burner_id,
            "analysis_number": self._analysis_count,
            "options": {
                "flame": input_data.include_flame_analysis,
                "health": input_data.include_health_analysis,
                "fuel": input_data.include_fuel_analysis,
                "rul": input_data.include_rul_prediction,
                "replacement": input_data.include_replacement_analysis,
                "work_orders": input_data.include_work_order_generation,
                "explainability": input_data.include_explainability,
            }
        })

        # Initialize result containers
        flame_result: Optional[FlameAnalysisOutput] = None
        health_result: Optional[BurnerHealthOutput] = None
        fuel_quality: Optional[FuelQualityScore] = None
        fuel_impact: Optional[DegradationImpact] = None
        rul_result: Optional[RULPredictionResult] = None
        failure_predictions: List[FailurePredictionResult] = []
        replacement_result: Optional[ReplacementAnalysisResult] = None
        work_orders: List[WorkOrder] = []
        explanation: Optional[GL021ExplanationResult] = None
        sub_hashes: Dict[str, str] = {}

        try:
            # Step 1: Flame Pattern Analysis
            if input_data.include_flame_analysis and self.flame_analyzer:
                flame_result = self._analyze_flame_patterns(input_data)
                if flame_result:
                    sub_hashes["flame"] = flame_result.provenance_hash
                    self._add_audit_entry("flame_analysis", {
                        "quality_score": flame_result.quality_score,
                        "stability_index": flame_result.stability.stability_index,
                        "status": flame_result.flame_status.value if hasattr(flame_result.flame_status, 'value') else flame_result.flame_status,
                    })

            # Step 2: Component Health Analysis
            if input_data.include_health_analysis:
                health_result = self._assess_burner_health(input_data)
                if health_result:
                    sub_hashes["health"] = health_result.provenance_hash
                    self._add_audit_entry("health_analysis", {
                        "overall_score": health_result.overall_health_score,
                        "limiting_component": health_result.limiting_component,
                        "limiting_score": health_result.limiting_score,
                    })

            # Step 3: Fuel Impact Analysis
            if input_data.include_fuel_analysis:
                fuel_quality, fuel_impact = self._evaluate_fuel_impact(input_data)
                if fuel_quality:
                    sub_hashes["fuel"] = fuel_quality.provenance_hash
                    self._add_audit_entry("fuel_analysis", {
                        "quality_score": fuel_quality.overall_score,
                        "quality_class": fuel_quality.quality_class,
                        "life_reduction": fuel_impact.life_reduction_factor if fuel_impact else 1.0,
                    })

            # Step 4: RUL Prediction
            if input_data.include_rul_prediction:
                rul_result, failure_predictions = self._predict_maintenance(input_data)
                if rul_result:
                    sub_hashes["rul"] = rul_result.provenance_hash
                    self._add_audit_entry("rul_prediction", {
                        "rul_p50_hours": rul_result.rul_p50_hours,
                        "failure_probability": rul_result.current_failure_probability,
                        "urgency": rul_result.maintenance_urgency.value if hasattr(rul_result.maintenance_urgency, 'value') else rul_result.maintenance_urgency,
                    })

            # Step 5: Replacement Analysis
            if input_data.include_replacement_analysis and self.replacement_planner:
                replacement_result = self._plan_replacement(input_data, rul_result)
                if replacement_result:
                    sub_hashes["replacement"] = replacement_result.provenance_hash
                    self._add_audit_entry("replacement_analysis", {
                        "npv": replacement_result.npv_replacement,
                        "optimal_date": replacement_result.optimal_replacement_date.isoformat() if replacement_result.optimal_replacement_date else None,
                    })

            # Step 6: Work Order Generation
            if input_data.include_work_order_generation and self.cmms_integration:
                work_orders = self._generate_work_orders(
                    input_data,
                    health_result,
                    rul_result,
                    failure_predictions,
                )
                self._add_audit_entry("work_order_generation", {
                    "work_orders_created": len(work_orders),
                })

            # Step 7: Generate Recommendations
            recommendations = self._generate_recommendations(
                flame_result,
                health_result,
                fuel_quality,
                fuel_impact,
                rul_result,
                failure_predictions,
            )

            # Step 8: Generate Explanations
            if input_data.include_explainability and self.explainer:
                explanation = self._generate_explanation(
                    input_data,
                    flame_result,
                    health_result,
                    fuel_quality,
                    rul_result,
                    input_data.explanation_audience,
                )
                if explanation:
                    sub_hashes["explanation"] = explanation.provenance_hash

            # Step 9: Calculate Overall Health Score
            overall_score, component_scores = self._calculate_overall_health(
                flame_result,
                health_result,
                fuel_quality,
                rul_result,
            )

            # Step 10: Generate Alerts
            alerts = self._generate_alerts(
                flame_result,
                health_result,
                fuel_quality,
                rul_result,
                overall_score,
            )

            # Step 11: Determine Health Status and Trend
            health_status = self._determine_health_status(overall_score)
            health_trend, trend_rate = self._analyze_health_trend(
                overall_score,
                input_data.previous_health_score,
                input_data.previous_assessment_date,
            )

            # Step 12: Calculate Processing Time
            processing_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Step 13: Calculate Provenance Hash
            provenance_hash = self._calculate_provenance_hash(
                input_data,
                overall_score,
                sub_hashes,
            )

            # Step 14: Generate Natural Language Summary
            nl_summary = self._generate_natural_language_summary(
                input_data.burner_id,
                overall_score,
                health_status,
                rul_result,
                health_result,
                recommendations,
            )

            # Step 15: Determine Status
            status = self._determine_analysis_status(
                flame_result,
                health_result,
                rul_result,
            )

            # Build result
            result = GL021Result(
                request_id=input_data.request_id,
                burner_id=input_data.burner_id,
                timestamp=datetime.now(timezone.utc),
                status=status,
                processing_time_ms=round(processing_time_ms, 2),

                # Overall assessment
                overall_health_score=round(overall_score, 2),
                health_status=health_status,

                # Flame results
                flame_quality_score=round(flame_result.quality_score, 2) if flame_result else None,
                flame_stability_index=round(flame_result.stability.stability_index, 2) if flame_result else None,
                flame_status=flame_result.flame_status.value if flame_result and hasattr(flame_result.flame_status, 'value') else (flame_result.flame_status if flame_result else None),
                flame_anomalies=[a.value if hasattr(a, 'value') else str(a) for a in (flame_result.anomalies if flame_result else [])],

                # Component results
                component_health_score=round(health_result.overall_health_score, 2) if health_result else None,
                limiting_component=health_result.limiting_component if health_result else None,
                limiting_component_score=round(health_result.limiting_score, 2) if health_result and health_result.limiting_score else None,
                component_scores=component_scores,

                # Fuel results
                fuel_quality_score=round(fuel_quality.overall_score, 2) if fuel_quality else None,
                fuel_life_reduction_factor=round(fuel_impact.life_reduction_factor, 3) if fuel_impact else None,
                fuel_concerns=fuel_quality.concerns if fuel_quality else [],

                # RUL results
                rul_p10_hours=round(rul_result.rul_p10_hours, 0) if rul_result else None,
                rul_p50_hours=round(rul_result.rul_p50_hours, 0) if rul_result else None,
                rul_p90_hours=round(rul_result.rul_p90_hours, 0) if rul_result else None,
                current_failure_probability=round(rul_result.current_failure_probability, 4) if rul_result else None,
                failure_probability_30d=round(rul_result.conditional_failure_prob_30d, 4) if rul_result else None,
                maintenance_urgency=rul_result.maintenance_urgency if rul_result else MaintenanceUrgency.MONITOR,
                prediction_confidence=rul_result.prediction_confidence if rul_result else PredictionConfidence.MEDIUM,
                weibull_beta=round(rul_result.beta, 3) if rul_result else None,
                weibull_eta_hours=round(rul_result.eta_hours, 0) if rul_result else None,

                # Failure mode predictions
                predicted_failure_modes=[
                    {
                        "mode": fp.failure_mode.value if hasattr(fp.failure_mode, 'value') else fp.failure_mode,
                        "probability": round(fp.probability, 4),
                        "confidence": round(fp.confidence, 4),
                        "risk_level": fp.risk_level,
                    }
                    for fp in failure_predictions[:5]
                ],
                primary_failure_mode=failure_predictions[0].failure_mode.value if failure_predictions and hasattr(failure_predictions[0].failure_mode, 'value') else (failure_predictions[0].failure_mode if failure_predictions else None),

                # Replacement results
                replacement_npv=round(replacement_result.npv_replacement, 0) if replacement_result else None,
                replacement_irr=round(replacement_result.irr_replacement, 4) if replacement_result and replacement_result.irr_replacement else None,
                optimal_replacement_date=replacement_result.optimal_replacement_date if replacement_result else None,
                annual_savings_potential=round(replacement_result.annual_savings, 0) if replacement_result else None,

                # Recommendations and alerts
                recommendations=recommendations[:15],
                alerts=alerts,

                # Work orders
                work_orders=work_orders,

                # Trend
                health_trend=health_trend,
                trend_rate_pct_per_month=round(trend_rate, 2) if trend_rate else None,

                # Explanation
                explanation=explanation,
                natural_language_summary=nl_summary,

                # Provenance
                provenance_hash=provenance_hash,
                sub_analysis_hashes=sub_hashes,
                calculation_details={
                    "health_weights": self.health_weights,
                    "component_scores": component_scores,
                    "audit_trail": self._audit_trail,
                },
            )

            logger.info(
                f"GL-021 analysis complete for {input_data.burner_id}: "
                f"Health={overall_score:.1f} ({health_status.value}), "
                f"RUL_P50={rul_result.rul_p50_hours:.0f}h, " if rul_result else ""
                f"Urgency={result.maintenance_urgency.value if hasattr(result.maintenance_urgency, 'value') else result.maintenance_urgency}, "
                f"Alerts={len(alerts)}, "
                f"Time={processing_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"GL-021 analysis failed for {input_data.burner_id}: {str(e)}", exc_info=True)
            raise

    # =========================================================================
    # ASYNC ANALYSIS METHODS
    # =========================================================================

    async def analyze_async(self, input_data: GL021Input) -> GL021Result:
        """
        Perform comprehensive burner analysis asynchronously.

        Runs sub-analyzers in parallel where possible for improved performance.

        Args:
            input_data: Comprehensive input data

        Returns:
            GL021Result with complete analysis results
        """
        if not self.enable_async:
            return self.analyze(input_data)

        loop = asyncio.get_event_loop()

        # Run analysis in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self._executor,
            self.analyze,
            input_data
        )

        return result

    def analyze_batch(
        self,
        data_list: List[GL021Input],
        parallel: bool = True,
    ) -> List[GL021Result]:
        """
        Analyze multiple burners in batch.

        Args:
            data_list: List of input data for multiple burners
            parallel: Run analyses in parallel

        Returns:
            List of GL021Result for each burner
        """
        logger.info(f"Starting batch analysis for {len(data_list)} burners")
        start_time = datetime.now(timezone.utc)

        if parallel and len(data_list) > 1:
            # Parallel execution
            futures = [
                self._executor.submit(self.analyze, data)
                for data in data_list
            ]
            results = [f.result() for f in futures]
        else:
            # Sequential execution
            results = [self.analyze(data) for data in data_list]

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            f"Batch analysis complete: {len(results)} burners in {elapsed:.2f}s "
            f"({elapsed/len(results)*1000:.1f}ms avg)"
        )

        return results

    # =========================================================================
    # SUB-ANALYZER METHODS
    # =========================================================================

    def _analyze_flame_patterns(
        self,
        input_data: GL021Input
    ) -> Optional[FlameAnalysisOutput]:
        """
        Analyze flame patterns using FlamePatternAnalyzer.

        Performs:
        - Flame stability index calculation
        - Geometry analysis (length, width, cone angle)
        - Temperature profile analysis
        - Color index calculation
        - Pulsation detection via FFT
        - Anomaly detection

        Args:
            input_data: GL021 input data

        Returns:
            FlameAnalysisOutput or None if insufficient data
        """
        if not self.flame_analyzer:
            return None

        # Build flame analysis input
        flame_input = FlameAnalysisInput(
            request_id=f"{input_data.request_id}_flame",
            burner_id=input_data.burner_id,
            scanner_signals=input_data.flame_scanner_signals,
            signal_history=input_data.flame_signal_history,
            geometry=input_data.flame_geometry,
            temperature_profile=input_data.flame_temperature_profile,
            flame_color_rgb=input_data.flame_color_rgb,
            flame_luminosity_pct=input_data.flame_luminosity_pct,
            fuel_type=input_data.fuel_type,
            fuel_flow_rate_kg_hr=0,  # Could be derived from firing rate
            air_flow_rate_kg_hr=0,
            excess_air_pct=input_data.excess_air_pct,
            historical_stability_indices=input_data.historical_stability_indices,
        )

        try:
            result = self.flame_analyzer.analyze(flame_input)
            logger.debug(f"Flame analysis: FSI={result.stability.stability_index:.1f}, FQS={result.quality_score:.1f}")
            return result
        except ValueError as e:
            logger.warning(f"Flame analysis skipped: {str(e)}")
            return None

    def _assess_burner_health(
        self,
        input_data: GL021Input
    ) -> Optional[BurnerHealthOutput]:
        """
        Assess burner component health using BurnerHealthAnalyzer.

        Evaluates:
        - Nozzle/tip health (erosion, coking, plugging)
        - Refractory tile integrity (Coffin-Manson thermal fatigue)
        - Igniter/pilot system health
        - Flame scanner reliability
        - Air register/damper operation
        - Fuel valve performance

        Args:
            input_data: GL021 input data

        Returns:
            BurnerHealthOutput or None if insufficient data
        """
        # Build health analysis input
        health_input = BurnerHealthInput(
            request_id=f"{input_data.request_id}_health",
            burner_id=input_data.burner_id,
            nozzle=input_data.nozzle_data,
            refractory_tile=input_data.refractory_data,
            igniter=input_data.igniter_data,
            scanner=input_data.scanner_data,
            air_register=input_data.air_register_data,
            fuel_valve=input_data.fuel_valve_data,
            total_burner_hours=input_data.operating_hours,
            total_startups=input_data.start_stop_cycles,
            fuel_type=input_data.fuel_type,
            previous_health_score=input_data.previous_health_score,
            previous_assessment_date=input_data.previous_assessment_date,
        )

        # Check if any component data is available
        has_data = any([
            input_data.nozzle_data,
            input_data.refractory_data,
            input_data.igniter_data,
            input_data.scanner_data,
            input_data.air_register_data,
            input_data.fuel_valve_data,
        ])

        if not has_data:
            logger.warning("No component data available for health analysis")
            return None

        try:
            result = self.health_analyzer.analyze(health_input)
            logger.debug(f"Health analysis: Score={result.overall_health_score:.1f}, Limiting={result.limiting_component}")
            return result
        except ValueError as e:
            logger.warning(f"Health analysis skipped: {str(e)}")
            return None

    def _evaluate_fuel_impact(
        self,
        input_data: GL021Input
    ) -> Tuple[Optional[FuelQualityScore], Optional[DegradationImpact]]:
        """
        Evaluate fuel quality impact using FuelImpactAnalyzer.

        Analyzes:
        - Fuel quality scoring (0-100)
        - Contaminant impacts (sulfur, vanadium, sodium)
        - Fouling rate prediction (Kern-Seaton)
        - Coking tendency
        - Life reduction factor

        Args:
            input_data: GL021 input data

        Returns:
            Tuple of (FuelQualityScore, DegradationImpact) or (None, None)
        """
        # Map fuel type string to enum
        fuel_type_map = {
            "natural_gas": FuelType.NATURAL_GAS,
            "refinery_gas": FuelType.REFINERY_GAS,
            "fuel_oil_2": FuelType.FUEL_OIL_NO2,
            "fuel_oil_no2": FuelType.FUEL_OIL_NO2,
            "fuel_oil_6": FuelType.FUEL_OIL_NO6,
            "fuel_oil_no6": FuelType.FUEL_OIL_NO6,
            "diesel": FuelType.DIESEL,
            "propane": FuelType.LPG,
            "lpg": FuelType.LPG,
        }
        fuel_enum = fuel_type_map.get(input_data.fuel_type.lower(), FuelType.NATURAL_GAS)

        # Build fuel properties
        fuel_props = FuelProperties(
            fuel_type=fuel_enum,
            sulfur_pct=input_data.fuel_sulfur_pct,
            h2s_ppm=input_data.fuel_h2s_ppm,
            vanadium_ppm=input_data.fuel_vanadium_ppm,
            sodium_ppm=input_data.fuel_sodium_ppm,
            ash_pct=input_data.fuel_ash_pct,
            water_pct=input_data.fuel_water_pct,
            carbon_residue_pct=input_data.fuel_carbon_residue_pct,
            heating_value_mj_kg=input_data.fuel_heating_value_mj_kg,
        )

        # Build operating conditions
        fuel_conditions = FuelOperatingConditions(
            flame_temperature_c=input_data.flame_temperature_c,
            flue_gas_temperature_c=input_data.flue_gas_temperature_c,
            tube_metal_temperature_c=input_data.flue_gas_temperature_c + 100,  # Approximate
            excess_air_pct=input_data.excess_air_pct,
            firing_rate_pct=input_data.firing_rate_pct,
            operating_hours_per_year=8000,
            thermal_cycles_per_year=input_data.cycling_frequency_per_day * 365,
        )

        try:
            # Calculate quality score
            quality = self.fuel_analyzer.calculate_fuel_quality_score(fuel_props)

            # Analyze degradation impact
            impact = self.fuel_analyzer.analyze_degradation_impact(fuel_props, fuel_conditions)

            logger.debug(f"Fuel analysis: Quality={quality.overall_score:.1f}, LRF={impact.life_reduction_factor:.2f}")
            return quality, impact

        except Exception as e:
            logger.warning(f"Fuel analysis failed: {str(e)}")
            return None, None

    def _predict_maintenance(
        self,
        input_data: GL021Input
    ) -> Tuple[Optional[RULPredictionResult], List[FailurePredictionResult]]:
        """
        Predict maintenance using MaintenancePredictionEngine.

        Performs:
        - Weibull-based RUL estimation
        - Cox Proportional Hazards adjustment
        - ML failure mode prediction
        - Ensemble combination

        Args:
            input_data: GL021 input data

        Returns:
            Tuple of (RULPredictionResult, List[FailurePredictionResult])
        """
        # Build operating conditions for prediction
        pred_conditions = PredictionOperatingConditions(
            avg_flame_temp_c=input_data.flame_temperature_c,
            firing_rate_pct=input_data.firing_rate_pct,
            air_fuel_ratio=input_data.air_fuel_ratio,
            combustion_air_temp_c=input_data.combustion_air_temp_c,
            flue_gas_temp_c=input_data.flue_gas_temperature_c,
            cycling_frequency=input_data.cycling_frequency_per_day,
            ambient_humidity_pct=input_data.ambient_humidity_pct,
            fuel_sulfur_content_pct=input_data.fuel_sulfur_pct,
            excess_air_pct=input_data.excess_air_pct,
        )

        # Calculate fuel quality score for ML
        fuel_quality_score = 100.0
        if input_data.include_fuel_analysis:
            try:
                fuel_type_map = {"natural_gas": FuelType.NATURAL_GAS}
                fuel_props = FuelProperties(
                    fuel_type=fuel_type_map.get(input_data.fuel_type.lower(), FuelType.NATURAL_GAS),
                    sulfur_pct=input_data.fuel_sulfur_pct,
                )
                fq = self.fuel_analyzer.calculate_fuel_quality_score(fuel_props)
                fuel_quality_score = fq.overall_score
            except Exception:
                pass

        try:
            # Predict RUL
            rul_result = self.prediction_engine.predict_rul(
                current_age_hours=input_data.operating_hours,
                failure_history=input_data.failure_history or [],
                operating_conditions=pred_conditions,
                sensor_data=input_data.sensor_data,
                fuel_quality_score=fuel_quality_score,
            )

            # Predict failure modes
            failure_predictions = self.prediction_engine.predict_failure_modes(
                operating_conditions=pred_conditions,
                sensor_data=input_data.sensor_data,
                equipment_age_hours=input_data.operating_hours,
                fuel_quality_score=fuel_quality_score,
            )

            logger.debug(
                f"RUL prediction: P50={rul_result.rul_p50_hours:.0f}h, "
                f"P_fail={rul_result.current_failure_probability:.3f}"
            )

            return rul_result, failure_predictions

        except Exception as e:
            logger.warning(f"RUL prediction failed: {str(e)}")
            return None, []

    def _plan_replacement(
        self,
        input_data: GL021Input,
        rul_result: Optional[RULPredictionResult],
    ) -> Optional[ReplacementAnalysisResult]:
        """
        Plan replacement using ReplacementPlanner.

        Performs:
        - NPV/IRR analysis
        - Optimal replacement timing
        - Monte Carlo uncertainty

        Args:
            input_data: GL021 input data
            rul_result: RUL prediction result

        Returns:
            ReplacementAnalysisResult or None
        """
        if not self.replacement_planner:
            return None

        # Build burner asset
        burner_type_map = {
            "gas": ReplacementBurnerType.NOZZLE_MIX,
            "oil": ReplacementBurnerType.NOZZLE_MIX,
            "low_nox": ReplacementBurnerType.LOW_NOX,
            "ultra_low_nox": ReplacementBurnerType.ULTRA_LOW_NOX,
        }

        # Get burner config
        burner_cfg = self.config.burner
        replacement_cfg = self.config.replacement_planning

        asset = BurnerAsset(
            asset_id=input_data.burner_id,
            tag_number=burner_cfg.burner_tag or input_data.burner_id,
            burner_type=burner_type_map.get(
                burner_cfg.burner_type.value if hasattr(burner_cfg.burner_type, 'value') else burner_cfg.burner_type,
                ReplacementBurnerType.NOZZLE_MIX
            ),
            installation_date=burner_cfg.installation_date or datetime.now(timezone.utc) - timedelta(days=365*3),
            current_age_hours=input_data.operating_hours,
            expected_life_hours=burner_cfg.expected_lifetime_hours,
            heat_input_mmbtu_hr=burner_cfg.capacity_mmbtu_hr,
            current_efficiency_pct=burner_cfg.design_efficiency_pct - 5,  # Assume some degradation
            design_efficiency_pct=burner_cfg.design_efficiency_pct,
            original_cost=replacement_cfg.burner_replacement_cost_usd * 0.8,
            replacement_cost=replacement_cfg.burner_replacement_cost_usd,
            installation_cost=replacement_cfg.planned_outage_duration_hours * replacement_cfg.labor_rate_usd_hr,
            weibull_beta=rul_result.beta if rul_result else self.config.maintenance_prediction.weibull_beta,
            weibull_eta=rul_result.eta_hours if rul_result else self.config.maintenance_prediction.weibull_eta_hours,
        )

        # Build economic parameters
        econ_params = EconomicParameters(
            discount_rate=replacement_cfg.discount_rate_pct / 100,
            analysis_horizon_years=replacement_cfg.planning_horizon_months // 12,
            operating_hours_per_year=8000,
            fuel_cost_per_mmbtu=replacement_cfg.fuel_cost_usd_mmbtu,
            preventive_maint_cost_per_year=5000,
            unplanned_failure_cost=replacement_cfg.downtime_cost_usd_hr * replacement_cfg.unplanned_outage_duration_hours,
            lost_production_cost_per_hour=replacement_cfg.downtime_cost_usd_hr,
            avg_repair_time_hours=replacement_cfg.unplanned_outage_duration_hours,
            carbon_cost_per_ton=replacement_cfg.carbon_price_usd_ton if replacement_cfg.carbon_cost_enabled else 0,
        )

        try:
            result = self.replacement_planner.analyze_replacement(
                asset=asset,
                params=econ_params,
                include_sensitivity=True,
            )
            logger.debug(f"Replacement analysis: NPV=${result.npv_replacement:,.0f}")
            return result

        except Exception as e:
            logger.warning(f"Replacement analysis failed: {str(e)}")
            return None

    def _generate_work_orders(
        self,
        input_data: GL021Input,
        health_result: Optional[BurnerHealthOutput],
        rul_result: Optional[RULPredictionResult],
        failure_predictions: List[FailurePredictionResult],
    ) -> List[WorkOrder]:
        """
        Generate CMMS work orders using CMSIntegration.

        Args:
            input_data: GL021 input data
            health_result: Health analysis result
            rul_result: RUL prediction result
            failure_predictions: Failure predictions

        Returns:
            List of generated WorkOrder objects
        """
        if not self.cmms_integration:
            return []

        work_orders = []

        # Generate work order from top failure prediction
        if failure_predictions and failure_predictions[0].probability > 0.3:
            top_pred = failure_predictions[0]

            pred_input = PredictionInput(
                burner_id=input_data.burner_id,
                failure_mode=top_pred.failure_mode.value if hasattr(top_pred.failure_mode, 'value') else top_pred.failure_mode,
                failure_probability=top_pred.probability,
                confidence=top_pred.confidence,
                time_to_failure_hours=top_pred.time_to_failure_hours or 0,
                health_index=health_result.overall_health_score / 100 if health_result else 0.7,
                contributing_factors=top_pred.top_contributing_features,
                recommendation=f"Predicted failure mode: {top_pred.failure_mode}",
            )

            # Determine criticality
            if top_pred.probability > 0.7:
                criticality = CriticalityLevel.CRITICAL
            elif top_pred.probability > 0.5:
                criticality = CriticalityLevel.HIGH
            else:
                criticality = CriticalityLevel.MEDIUM

            try:
                wo = self.cmms_integration.create_work_order_from_prediction(
                    prediction=pred_input,
                    criticality=criticality,
                    equipment_tag=input_data.burner_id,
                    functional_location=self.config.cmms.plant_code,
                    production_impact_per_hour=self.config.replacement_planning.downtime_cost_usd_hr,
                )
                work_orders.append(wo)
            except Exception as e:
                logger.warning(f"Work order generation failed: {str(e)}")

        return work_orders

    def _generate_recommendations(
        self,
        flame_result: Optional[FlameAnalysisOutput],
        health_result: Optional[BurnerHealthOutput],
        fuel_quality: Optional[FuelQualityScore],
        fuel_impact: Optional[DegradationImpact],
        rul_result: Optional[RULPredictionResult],
        failure_predictions: List[FailurePredictionResult],
    ) -> List[str]:
        """
        Generate prioritized maintenance recommendations.

        Args:
            flame_result: Flame analysis result
            health_result: Health analysis result
            fuel_quality: Fuel quality score
            fuel_impact: Fuel degradation impact
            rul_result: RUL prediction result
            failure_predictions: Failure predictions

        Returns:
            Prioritized list of recommendations
        """
        recommendations = []

        # RUL-based recommendations
        if rul_result:
            recommendations.extend(rul_result.recommendations)

        # Health-based recommendations
        if health_result:
            recommendations.extend(health_result.recommendations)

        # Flame-based recommendations
        if flame_result:
            recommendations.extend(flame_result.recommendations)

        # Fuel-based recommendations
        if fuel_quality:
            for concern in fuel_quality.concerns:
                recommendations.append(f"[FUEL] {concern}")

        if fuel_impact:
            recommendations.extend([f"[FUEL] {r}" for r in fuel_impact.recommendations])

        # Failure mode recommendations
        if failure_predictions:
            top_pred = failure_predictions[0]
            if top_pred.probability > 0.3:
                recommendations.insert(0,
                    f"[ML PREDICTION] {top_pred.failure_mode.value if hasattr(top_pred.failure_mode, 'value') else top_pred.failure_mode} "
                    f"risk at {top_pred.probability:.0%} probability. "
                    f"Top factors: {', '.join(top_pred.top_contributing_features[:3])}"
                )

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations

    # =========================================================================
    # SCORING AND STATUS METHODS
    # =========================================================================

    def _calculate_overall_health(
        self,
        flame_result: Optional[FlameAnalysisOutput],
        health_result: Optional[BurnerHealthOutput],
        fuel_quality: Optional[FuelQualityScore],
        rul_result: Optional[RULPredictionResult],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall weighted health score.

        Formula: Overall = sum(weight_i * score_i)

        Args:
            flame_result: Flame analysis result
            health_result: Health analysis result
            fuel_quality: Fuel quality score
            rul_result: RUL prediction result

        Returns:
            Tuple of (overall_score, component_scores)
        """
        scores = {}
        weights_used = {}
        total_weight = 0.0

        # Flame score
        if flame_result:
            scores["flame"] = flame_result.quality_score
            weights_used["flame"] = self.health_weights["flame"]
            total_weight += self.health_weights["flame"]

        # Component health score
        if health_result:
            scores["component"] = health_result.overall_health_score
            weights_used["component"] = self.health_weights["component"]
            total_weight += self.health_weights["component"]

        # Fuel quality score
        if fuel_quality:
            scores["fuel"] = fuel_quality.overall_score
            weights_used["fuel"] = self.health_weights["fuel"]
            total_weight += self.health_weights["fuel"]

        # RUL-based score (convert failure probability to health)
        if rul_result:
            # Health = 100 * (1 - failure_probability)
            rul_health = 100.0 * (1.0 - rul_result.current_failure_probability)
            scores["prediction"] = rul_health
            weights_used["prediction"] = self.health_weights["prediction"]
            total_weight += self.health_weights["prediction"]

        # Calculate weighted average
        if total_weight > 0:
            overall_score = sum(
                scores[k] * weights_used[k] / total_weight
                for k in scores
            )
        else:
            # Default to 70 if no data
            overall_score = 70.0

        return overall_score, scores

    def _determine_health_status(self, score: float) -> OverallHealthStatus:
        """Determine health status from score."""
        if score >= 90:
            return OverallHealthStatus.EXCELLENT
        elif score >= 70:
            return OverallHealthStatus.GOOD
        elif score >= 50:
            return OverallHealthStatus.FAIR
        elif score >= 25:
            return OverallHealthStatus.POOR
        else:
            return OverallHealthStatus.CRITICAL

    def _analyze_health_trend(
        self,
        current_score: float,
        previous_score: Optional[float],
        previous_date: Optional[datetime],
    ) -> Tuple[str, Optional[float]]:
        """Analyze health trend from historical data."""
        if previous_score is None or previous_date is None:
            return "stable", None

        score_change = current_score - previous_score
        days_elapsed = (datetime.now(timezone.utc) - previous_date).days

        if days_elapsed <= 0:
            return "stable", None

        # Calculate monthly rate
        rate_per_month = score_change / (days_elapsed / 30)

        if rate_per_month > 2.0:
            return "improving", rate_per_month
        elif rate_per_month < -2.0:
            return "degrading", rate_per_month
        else:
            return "stable", rate_per_month

    def _determine_analysis_status(
        self,
        flame_result: Optional[FlameAnalysisOutput],
        health_result: Optional[BurnerHealthOutput],
        rul_result: Optional[RULPredictionResult],
    ) -> PredictionStatus:
        """Determine overall analysis status."""
        has_flame = flame_result is not None
        has_health = health_result is not None
        has_rul = rul_result is not None

        if has_flame and has_health and has_rul:
            return PredictionStatus.SUCCESS
        elif has_flame or has_health or has_rul:
            return PredictionStatus.PARTIAL
        else:
            return PredictionStatus.INSUFFICIENT_DATA

    # =========================================================================
    # ALERT GENERATION
    # =========================================================================

    def _generate_alerts(
        self,
        flame_result: Optional[FlameAnalysisOutput],
        health_result: Optional[BurnerHealthOutput],
        fuel_quality: Optional[FuelQualityScore],
        rul_result: Optional[RULPredictionResult],
        overall_score: float,
    ) -> List[GL021Alert]:
        """Generate alerts based on analysis results."""
        alerts = []
        timestamp = datetime.now(timezone.utc)

        # Overall health alert
        if overall_score < 50:
            alerts.append(GL021Alert(
                alert_id=f"GL021-{timestamp.strftime('%Y%m%d%H%M%S')}-HEALTH",
                level=AlertLevel.CRITICAL if overall_score < 25 else AlertLevel.ALARM,
                category="overall_health",
                message=f"Overall burner health is {overall_score:.0f}%",
                source_component="overall",
                action_required=True,
                recommended_action="Schedule comprehensive inspection",
            ))

        # Flame alerts
        if flame_result:
            if flame_result.quality_score < 60:
                alerts.append(GL021Alert(
                    alert_id=f"GL021-{timestamp.strftime('%Y%m%d%H%M%S')}-FLAME",
                    level=AlertLevel.ALARM,
                    category="flame_quality",
                    message=f"Flame quality degraded to {flame_result.quality_score:.0f}%",
                    source_component="flame_analyzer",
                    action_required=True,
                    recommended_action="Inspect burner and combustion settings",
                ))

        # Component health alerts
        if health_result and health_result.limiting_score:
            if health_result.limiting_score < 50:
                alerts.append(GL021Alert(
                    alert_id=f"GL021-{timestamp.strftime('%Y%m%d%H%M%S')}-COMP",
                    level=AlertLevel.ALARM,
                    category="component_health",
                    message=f"{health_result.limiting_component} health critical at {health_result.limiting_score:.0f}%",
                    source_component=health_result.limiting_component or "unknown",
                    action_required=True,
                    recommended_action=f"Inspect/replace {health_result.limiting_component}",
                ))

        # RUL alerts
        if rul_result:
            if rul_result.maintenance_urgency in [MaintenanceUrgency.IMMEDIATE, MaintenanceUrgency.URGENT]:
                alerts.append(GL021Alert(
                    alert_id=f"GL021-{timestamp.strftime('%Y%m%d%H%M%S')}-RUL",
                    level=AlertLevel.CRITICAL if rul_result.maintenance_urgency == MaintenanceUrgency.IMMEDIATE else AlertLevel.ALARM,
                    category="rul_prediction",
                    message=f"RUL P50 is {rul_result.rul_p50_hours:.0f} hours - {rul_result.maintenance_urgency.value} maintenance required",
                    source_component="prediction_engine",
                    action_required=True,
                    recommended_action=rul_result.recommendations[0] if rul_result.recommendations else "Schedule maintenance",
                ))

        return alerts

    # =========================================================================
    # EXPLANATION GENERATION
    # =========================================================================

    def _generate_explanation(
        self,
        input_data: GL021Input,
        flame_result: Optional[FlameAnalysisOutput],
        health_result: Optional[BurnerHealthOutput],
        fuel_quality: Optional[FuelQualityScore],
        rul_result: Optional[RULPredictionResult],
        audience: ExplanationAudience,
    ) -> Optional[GL021ExplanationResult]:
        """
        Generate multi-audience explanations using GL021Explainer.

        Args:
            input_data: GL021 input data
            flame_result: Flame analysis result
            health_result: Health analysis result
            fuel_quality: Fuel quality score
            rul_result: RUL prediction result
            audience: Target audience

        Returns:
            GL021ExplanationResult or None
        """
        if not self.explainer:
            return None

        # Build feature values for SHAP
        feature_values = {
            "operating_hours": input_data.operating_hours,
            "firing_rate_pct": input_data.firing_rate_pct,
            "flame_temperature_c": input_data.flame_temperature_c,
            "excess_air_pct": input_data.excess_air_pct,
            "fuel_sulfur_pct": input_data.fuel_sulfur_pct,
            "start_stop_cycles": input_data.start_stop_cycles,
        }

        if flame_result:
            feature_values["flame_stability_index"] = flame_result.stability.stability_index
            feature_values["flame_quality_score"] = flame_result.quality_score

        if health_result:
            feature_values["component_health_score"] = health_result.overall_health_score

        try:
            explanation = self.explainer.explain(
                X=None,  # Not using actual model features here
                feature_values=feature_values,
            )
            return explanation
        except Exception as e:
            logger.warning(f"Explanation generation failed: {str(e)}")
            return None

    def _generate_natural_language_summary(
        self,
        burner_id: str,
        overall_score: float,
        health_status: OverallHealthStatus,
        rul_result: Optional[RULPredictionResult],
        health_result: Optional[BurnerHealthOutput],
        recommendations: List[str],
    ) -> str:
        """Generate natural language summary of analysis."""
        parts = []

        # Overall status
        parts.append(
            f"Burner {burner_id} is in {health_status.value} condition "
            f"with an overall health score of {overall_score:.0f}/100."
        )

        # RUL summary
        if rul_result:
            urgency_text = {
                MaintenanceUrgency.IMMEDIATE: "Immediate maintenance is required.",
                MaintenanceUrgency.URGENT: "Urgent maintenance is recommended within 1 week.",
                MaintenanceUrgency.PLANNED: "Planned maintenance should be scheduled within 30 days.",
                MaintenanceUrgency.SCHEDULED: "Maintenance can be scheduled for the next turnaround.",
                MaintenanceUrgency.MONITOR: "Continue routine monitoring.",
            }
            parts.append(
                f"Estimated remaining useful life is {rul_result.rul_p50_hours:.0f} hours (median). "
                f"{urgency_text.get(rul_result.maintenance_urgency, '')}"
            )

        # Limiting component
        if health_result and health_result.limiting_component:
            parts.append(
                f"The {health_result.limiting_component} is the limiting component "
                f"at {health_result.limiting_score:.0f}% health."
            )

        # Top recommendation
        if recommendations:
            parts.append(f"Primary action: {recommendations[0]}")

        return " ".join(parts)

    # =========================================================================
    # PROVENANCE AND AUDIT
    # =========================================================================

    def _calculate_provenance_hash(
        self,
        input_data: GL021Input,
        overall_score: float,
        sub_hashes: Dict[str, str],
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "request_id": input_data.request_id,
            "burner_id": input_data.burner_id,
            "timestamp": input_data.timestamp.isoformat(),
            "overall_score": overall_score,
            "sub_hashes": sub_hashes,
            "analyzer_version": "1.0.0",
            "config_hash": hashlib.sha256(
                json.dumps(self.health_weights, sort_keys=True).encode()
            ).hexdigest()[:16],
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _add_audit_entry(self, operation: str, data: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "data": data,
        })

    # =========================================================================
    # PUBLIC UTILITY METHODS
    # =========================================================================

    def get_explanation(
        self,
        result: GL021Result,
        audience: ExplanationAudience = ExplanationAudience.ENGINEER,
    ) -> str:
        """
        Get explanation for a specific audience.

        Args:
            result: GL021Result from analysis
            audience: Target audience

        Returns:
            Explanation text for the specified audience
        """
        if result.explanation is None:
            return result.natural_language_summary or "No explanation available."

        audience_map = {
            ExplanationAudience.OPERATOR: result.explanation.operator_explanation,
            ExplanationAudience.ENGINEER: result.explanation.engineer_explanation,
            ExplanationAudience.MANAGER: result.explanation.manager_explanation,
            ExplanationAudience.AUDITOR: result.explanation.auditor_explanation,
        }

        explanation = audience_map.get(audience)
        if explanation and hasattr(explanation, 'explanation_text'):
            return explanation.explanation_text
        return result.natural_language_summary or "No explanation available."

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get calculation audit trail from last analysis."""
        return self._audit_trail.copy()

    def get_analysis_count(self) -> int:
        """Get total number of analyses performed."""
        return self._analysis_count

    def shutdown(self) -> None:
        """Shutdown predictor and release resources."""
        self._executor.shutdown(wait=True)
        logger.info("BurnerMaintenancePredictor shutdown complete")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_predictor(config: GL021Config) -> BurnerMaintenancePredictor:
    """
    Factory function to create BurnerMaintenancePredictor.

    Args:
        config: GL021 configuration

    Returns:
        Configured BurnerMaintenancePredictor
    """
    return BurnerMaintenancePredictor(config)


def quick_analysis(
    burner_id: str,
    operating_hours: float,
    flame_stability: float = 90.0,
    firing_rate_pct: float = 80.0,
) -> GL021Result:
    """
    Quick analysis with minimal configuration.

    Args:
        burner_id: Burner identifier
        operating_hours: Operating hours
        flame_stability: Flame stability index (0-100)
        firing_rate_pct: Current firing rate

    Returns:
        GL021Result
    """
    from greenlang.agents.process_heat.gl_021_burner_maintenance.config import BurnerConfig

    config = GL021Config(
        burner=BurnerConfig(burner_id=burner_id)
    )

    predictor = BurnerMaintenancePredictor(config)

    input_data = GL021Input(
        burner_id=burner_id,
        operating_hours=operating_hours,
        firing_rate_pct=firing_rate_pct,
        historical_stability_indices=[flame_stability] * 10,
        include_replacement_analysis=False,
        include_work_order_generation=False,
    )

    return predictor.analyze(input_data)
