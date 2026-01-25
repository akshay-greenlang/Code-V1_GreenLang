"""
GL-004 BURNMASTER Data Schemas

Pydantic schemas for burner process data, sensor readings, setpoints,
recommendations, and optimization results. All schemas include
uncertainty quantification and provenance tracking.

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SensorQuality(str, Enum):
    """Sensor data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    SUBSTITUTED = "substituted"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RecommendationCategory(str, Enum):
    """Categories of optimization recommendations."""
    AIR_FUEL_RATIO = "air_fuel_ratio"
    FLAME_STABILITY = "flame_stability"
    EMISSIONS = "emissions"
    EFFICIENCY = "efficiency"
    TURNDOWN = "turndown"
    MAINTENANCE = "maintenance"
    SAFETY = "safety"


class UncertainValue(BaseModel):
    """Value with uncertainty bounds."""
    value: float
    uncertainty: float = Field(ge=0.0, description="1-sigma uncertainty")
    lower_95: float | None = None
    upper_95: float | None = None
    quality: SensorQuality = SensorQuality.GOOD

    def __post_init__(self) -> None:
        """Calculate 95% confidence bounds if not provided."""
        if self.lower_95 is None:
            self.lower_95 = self.value - 1.96 * self.uncertainty
        if self.upper_95 is None:
            self.upper_95 = self.value + 1.96 * self.uncertainty


class Provenance(BaseModel):
    """Provenance information for audit trail."""
    calculation_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(description="Source of the calculation/data")
    method: str = Field(description="Method/formula used")
    version: str = Field(default="1.0.0")
    inputs_hash: str | None = None
    model_version: str | None = None


class BurnerSensorData(BaseModel):
    """Real-time sensor data from a single burner."""

    burner_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Flow rates
    fuel_flow_rate_kg_s: UncertainValue
    air_flow_rate_kg_s: UncertainValue
    flue_gas_flow_rate_kg_s: UncertainValue | None = None

    # Temperatures (°C)
    combustion_air_temp: UncertainValue
    flame_temp: UncertainValue | None = None
    furnace_temp: UncertainValue
    stack_temp: UncertainValue

    # Pressures (mbar)
    furnace_pressure: UncertainValue
    air_damper_position_pct: UncertainValue
    fuel_valve_position_pct: UncertainValue

    # Flue gas analysis
    excess_o2_pct: UncertainValue
    co_ppm: UncertainValue
    co2_pct: UncertainValue | None = None
    nox_ppm: UncertainValue
    sox_ppm: UncertainValue | None = None

    # Flame monitoring
    flame_intensity: UncertainValue | None = None
    flame_stability_index: UncertainValue | None = None
    flame_color_r: float | None = None
    flame_color_g: float | None = None
    flame_color_b: float | None = None

    # Acoustic/vibration (for stability detection)
    acoustic_rms: UncertainValue | None = None
    dominant_frequency_hz: float | None = None
    vibration_rms: UncertainValue | None = None

    # Calculated fields (optional - computed by agent)
    lambda_ratio: float | None = None
    thermal_input_mw: float | None = None
    efficiency_pct: float | None = None


class ProcessState(BaseModel):
    """Current state of the combustion process."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    burners: list[BurnerSensorData]

    # Aggregate metrics
    total_fuel_flow_kg_s: float
    total_thermal_input_mw: float
    average_excess_o2_pct: float
    average_lambda: float
    aggregate_co_ppm: float
    aggregate_nox_ppm: float

    # System state
    load_pct: float = Field(ge=0, le=100)
    operating_point: str = Field(description="e.g., 'high-fire', 'low-fire', 'modulating'")
    in_turndown: bool = False
    turndown_ratio: float | None = None

    # Stability assessment
    overall_stability_index: float = Field(ge=0, le=1)
    stability_warning: bool = False
    instability_type: str | None = None

    # Efficiency
    current_efficiency_pct: float
    heat_rate_btu_kwh: float | None = None

    # Calculated lambda distribution
    lambda_distribution: dict[str, float] | None = None


class Setpoint(BaseModel):
    """Setpoint for a control variable."""

    variable: str
    burner_id: str | None = None
    current_value: float
    target_value: float
    min_value: float
    max_value: float
    rate_limit: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SetpointRecommendation(BaseModel):
    """Recommended setpoint change with explanation."""

    recommendation_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    setpoint: Setpoint
    category: RecommendationCategory
    priority: RecommendationPriority

    # Expected impact
    expected_benefit: str
    expected_fuel_savings_pct: float | None = None
    expected_emissions_reduction_pct: float | None = None
    expected_efficiency_gain_pct: float | None = None

    # Confidence and uncertainty
    confidence: float = Field(ge=0, le=1)
    uncertainty_impact: dict[str, float] | None = None

    # Explainability
    explanation: str
    contributing_factors: list[str]
    physics_basis: str | None = None
    shap_values: dict[str, float] | None = None
    counterfactual: str | None = None

    # Safety check results
    safety_verified: bool = True
    safety_margin: float | None = None

    # Provenance
    provenance: Provenance


class OptimizationResult(BaseModel):
    """Result of multi-objective optimization."""

    optimization_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Status
    success: bool
    iterations: int
    convergence_achieved: bool
    computation_time_ms: float

    # Objective values
    total_objective: float
    fuel_cost_component: float
    emissions_cost_component: float
    co_penalty_component: float
    stability_penalty_component: float
    actuator_move_component: float

    # Optimal setpoints
    optimal_setpoints: list[Setpoint]
    recommendations: list[SetpointRecommendation]

    # Constraints
    active_constraints: list[str]
    constraint_violations: list[str]

    # Pareto analysis (for multi-objective)
    pareto_front: list[dict[str, float]] | None = None
    selected_solution_index: int | None = None

    # Sensitivity analysis
    sensitivity: dict[str, float] | None = None

    # Provenance
    provenance: Provenance


class CausalFactor(BaseModel):
    """A causal factor in root cause analysis."""

    factor_id: str
    variable: str
    description: str
    causal_strength: float = Field(ge=0, le=1)
    time_lag_s: float | None = None
    mechanism: str
    evidence: list[str]
    intervention_recommended: str | None = None


class RootCauseAnalysis(BaseModel):
    """Root cause analysis result."""

    analysis_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Target
    anomaly_description: str
    target_variable: str
    deviation_magnitude: float
    deviation_direction: str  # "above_normal" or "below_normal"

    # Causes
    root_causes: list[CausalFactor]
    contributing_factors: list[CausalFactor]

    # Counterfactuals
    counterfactual_scenarios: list[dict[str, Any]]

    # Confidence
    confidence: float = Field(ge=0, le=1)
    methodology: str

    # Provenance
    provenance: Provenance


class HealthStatus(BaseModel):
    """Health status of the burner optimization agent."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Agent status
    agent_id: str
    is_healthy: bool
    uptime_s: float
    current_mode: str

    # Data quality
    sensor_health: dict[str, SensorQuality]
    data_freshness_s: float
    missing_sensors: list[str]

    # Model health
    model_last_retrained: datetime | None = None
    model_performance_metrics: dict[str, float] | None = None
    model_drift_detected: bool = False

    # Performance
    optimization_latency_ms: float
    inference_latency_ms: float
    last_optimization_time: datetime | None = None

    # Errors
    error_count_24h: int
    warning_count_24h: int
    last_error: str | None = None


class Alert(BaseModel):
    """Alert from the monitoring system."""

    alert_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    severity: str  # "critical", "warning", "info"
    category: str
    title: str
    description: str
    affected_burners: list[str]

    # Values
    current_value: float | None = None
    threshold_value: float | None = None
    deviation_pct: float | None = None

    # Recommendations
    recommended_action: str | None = None
    auto_action_taken: str | None = None

    # State
    acknowledged: bool = False
    acknowledged_by: str | None = None
    resolved: bool = False
    resolved_at: datetime | None = None


class EmissionsReport(BaseModel):
    """Emissions report for climate/regulatory compliance."""

    report_id: UUID = Field(default_factory=uuid4)
    period_start: datetime
    period_end: datetime
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Fuel consumption
    total_fuel_kg: float
    fuel_type: str
    fuel_hhv_mj_kg: float

    # Energy
    total_thermal_input_mj: float
    total_useful_output_mj: float
    average_efficiency_pct: float

    # Emissions
    total_co2_kg: float
    total_co2e_kg: float  # Includes CH4, N2O
    total_nox_kg: float
    total_sox_kg: float
    total_co_kg: float
    total_pm_kg: float | None = None

    # Intensity metrics
    co2_intensity_kg_mwh: float
    nox_intensity_lb_mmbtu: float

    # Comparison
    baseline_co2_kg: float | None = None
    reduction_pct: float | None = None

    # Provenance
    provenance: Provenance


class BurnerDiagnostics(BaseModel):
    """Diagnostics for a single burner."""

    burner_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Health scores (0-1)
    overall_health: float = Field(ge=0, le=1)
    flame_health: float = Field(ge=0, le=1)
    fuel_system_health: float = Field(ge=0, le=1)
    air_system_health: float = Field(ge=0, le=1)
    ignition_health: float = Field(ge=0, le=1)

    # Anomalies detected
    anomalies: list[str]
    anomaly_scores: dict[str, float]

    # Maintenance predictions
    predicted_maintenance_days: int | None = None
    maintenance_priority: str | None = None
    recommended_maintenance: list[str]

    # Performance trends
    efficiency_trend: str  # "improving", "stable", "degrading"
    stability_trend: str
    emissions_trend: str

    # Last maintenance
    last_maintenance_date: datetime | None = None
    operating_hours_since_maintenance: float | None = None
