# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Schema Definitions

This module defines all Pydantic models for inputs, outputs, analysis results,
and status reporting for the Predictive Maintenance Agent.

All models include comprehensive validation, documentation, and support for
provenance tracking through SHA-256 hashes.

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    ...     PredictiveMaintenanceInput,
    ...     PredictiveMaintenanceOutput,
    ... )
"""

from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    EquipmentType,
    FailureMode,
    MaintenanceStrategy,
)


# =============================================================================
# ENUMS
# =============================================================================

class HealthStatus(str, Enum):
    """Equipment health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class TrendDirection(str, Enum):
    """Parameter trend direction."""
    STABLE = "stable"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    ERRATIC = "erratic"


class DiagnosisConfidence(str, Enum):
    """Confidence level in diagnosis."""
    HIGH = "high"  # >90%
    MEDIUM = "medium"  # 70-90%
    LOW = "low"  # 50-70%
    UNCERTAIN = "uncertain"  # <50%


class WorkOrderType(str, Enum):
    """Work order types."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    INSPECTION = "inspection"
    LUBRICATION = "lubrication"
    CALIBRATION = "calibration"


class WorkOrderPriority(str, Enum):
    """Work order priority levels."""
    EMERGENCY = "emergency"  # Immediate
    URGENT = "urgent"  # 24 hours
    HIGH = "high"  # 48 hours
    MEDIUM = "medium"  # 1 week
    LOW = "low"  # 2 weeks
    SCHEDULED = "scheduled"  # Next planned outage


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class VibrationReading(BaseModel):
    """Single vibration reading from sensor."""

    sensor_id: str = Field(..., description="Sensor identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )
    location: str = Field(..., description="Sensor location (DE, NDE, etc.)")
    orientation: str = Field(
        default="radial",
        description="Measurement orientation"
    )
    velocity_rms_mm_s: float = Field(
        ...,
        ge=0,
        description="Velocity RMS (mm/s)"
    )
    acceleration_rms_g: float = Field(
        ...,
        ge=0,
        description="Acceleration RMS (g)"
    )
    displacement_um: Optional[float] = Field(
        default=None,
        ge=0,
        description="Displacement peak-to-peak (um)"
    )
    spectrum: Optional[List[float]] = Field(
        default=None,
        description="FFT spectrum amplitudes"
    )
    frequency_resolution_hz: Optional[float] = Field(
        default=None,
        description="FFT frequency resolution"
    )
    operating_speed_rpm: float = Field(
        ...,
        gt=0,
        description="Operating speed at measurement"
    )
    temperature_c: Optional[float] = Field(
        default=None,
        description="Sensor/bearing temperature"
    )


class OilAnalysisReading(BaseModel):
    """Oil analysis sample results."""

    sample_id: str = Field(..., description="Sample identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Sample timestamp"
    )
    sample_point: str = Field(..., description="Sampling location")

    # Physical properties
    viscosity_40c_cst: float = Field(
        ...,
        gt=0,
        description="Kinematic viscosity at 40C (cSt)"
    )
    viscosity_100c_cst: Optional[float] = Field(
        default=None,
        gt=0,
        description="Kinematic viscosity at 100C (cSt)"
    )
    viscosity_index: Optional[float] = Field(
        default=None,
        description="Viscosity index"
    )

    # Acid/base
    tan_mg_koh_g: float = Field(
        ...,
        ge=0,
        description="Total Acid Number (mg KOH/g)"
    )
    tbn_mg_koh_g: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total Base Number (mg KOH/g)"
    )

    # Contamination
    water_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Water content (ppm)"
    )
    particle_count_iso_4406: Optional[str] = Field(
        default=None,
        description="Particle count ISO 4406 code"
    )

    # Wear metals (ppm)
    iron_ppm: float = Field(default=0.0, ge=0)
    copper_ppm: float = Field(default=0.0, ge=0)
    chromium_ppm: float = Field(default=0.0, ge=0)
    aluminum_ppm: float = Field(default=0.0, ge=0)
    lead_ppm: float = Field(default=0.0, ge=0)
    tin_ppm: float = Field(default=0.0, ge=0)
    nickel_ppm: float = Field(default=0.0, ge=0)
    silver_ppm: float = Field(default=0.0, ge=0)

    # Contaminants (ppm)
    silicon_ppm: float = Field(default=0.0, ge=0, description="Dirt ingress")
    sodium_ppm: float = Field(default=0.0, ge=0, description="Coolant leak")

    # Additives (ppm)
    zinc_ppm: float = Field(default=0.0, ge=0)
    phosphorus_ppm: float = Field(default=0.0, ge=0)
    calcium_ppm: float = Field(default=0.0, ge=0)
    magnesium_ppm: float = Field(default=0.0, ge=0)

    # Condition indicators
    oxidation_abs_cm: Optional[float] = Field(
        default=None,
        description="Oxidation (Abs/cm)"
    )
    nitration_abs_cm: Optional[float] = Field(
        default=None,
        description="Nitration (Abs/cm)"
    )
    soot_pct: Optional[float] = Field(
        default=None,
        ge=0,
        description="Soot content (%)"
    )


class TemperatureReading(BaseModel):
    """Temperature measurement."""

    sensor_id: str = Field(..., description="Sensor identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )
    location: str = Field(..., description="Measurement location")
    temperature_c: float = Field(..., description="Temperature (Celsius)")
    ambient_c: Optional[float] = Field(
        default=None,
        description="Ambient temperature"
    )
    delta_c: Optional[float] = Field(
        default=None,
        description="Temperature differential"
    )


class ThermalImage(BaseModel):
    """Thermal image data."""

    image_id: str = Field(..., description="Image identifier")
    camera_id: str = Field(..., description="Camera identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Image timestamp"
    )
    min_temperature_c: float = Field(..., description="Minimum temperature")
    max_temperature_c: float = Field(..., description="Maximum temperature")
    avg_temperature_c: float = Field(..., description="Average temperature")
    hot_spots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected hot spots with coordinates"
    )
    emissivity: float = Field(
        default=0.95,
        ge=0.1,
        le=1.0,
        description="Surface emissivity used"
    )
    ambient_c: Optional[float] = Field(
        default=None,
        description="Ambient temperature"
    )


class CurrentReading(BaseModel):
    """Motor current measurement for MCSA."""

    sensor_id: str = Field(..., description="Sensor identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )
    phase_a_rms_a: float = Field(..., ge=0, description="Phase A RMS current")
    phase_b_rms_a: float = Field(..., ge=0, description="Phase B RMS current")
    phase_c_rms_a: float = Field(..., ge=0, description="Phase C RMS current")
    current_unbalance_pct: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current unbalance percentage"
    )
    spectrum_phase_a: Optional[List[float]] = Field(
        default=None,
        description="Phase A current spectrum"
    )
    frequency_resolution_hz: Optional[float] = Field(
        default=None,
        description="Spectrum frequency resolution"
    )
    line_frequency_hz: float = Field(
        default=60.0,
        description="Line frequency (50 or 60 Hz)"
    )
    operating_speed_rpm: Optional[float] = Field(
        default=None,
        description="Motor operating speed"
    )


class PredictiveMaintenanceInput(BaseModel):
    """
    Input data for predictive maintenance analysis.

    This model encapsulates all sensor readings and operational data
    required for comprehensive equipment health assessment.

    Attributes:
        equipment_id: Unique equipment identifier
        timestamp: Analysis timestamp
        vibration_readings: List of vibration measurements
        oil_analysis: Oil analysis results
        temperature_readings: Temperature measurements
        thermal_images: Infrared thermal images
        current_readings: Motor current measurements

    Example:
        >>> input_data = PredictiveMaintenanceInput(
        ...     equipment_id="PUMP-001",
        ...     vibration_readings=[vib_reading],
        ...     oil_analysis=oil_sample
        ... )
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis request timestamp"
    )

    # Sensor readings
    vibration_readings: List[VibrationReading] = Field(
        default_factory=list,
        description="Vibration measurements"
    )
    oil_analysis: Optional[OilAnalysisReading] = Field(
        default=None,
        description="Oil analysis results"
    )
    temperature_readings: List[TemperatureReading] = Field(
        default_factory=list,
        description="Temperature measurements"
    )
    thermal_images: List[ThermalImage] = Field(
        default_factory=list,
        description="Thermal images"
    )
    current_readings: List[CurrentReading] = Field(
        default_factory=list,
        description="Motor current readings"
    )

    # Operating conditions
    operating_speed_rpm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Current operating speed"
    )
    load_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=150,
        description="Current load percentage"
    )
    running_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total running hours"
    )

    # Context
    recent_maintenance: Optional[str] = Field(
        default=None,
        description="Recent maintenance performed"
    )
    operator_notes: Optional[str] = Field(
        default=None,
        description="Operator observations"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# ANALYSIS RESULT SCHEMAS
# =============================================================================

class WeibullAnalysisResult(BaseModel):
    """Weibull distribution analysis result for RUL estimation."""

    beta: float = Field(
        ...,
        gt=0,
        description="Shape parameter (Weibull slope)"
    )
    eta_hours: float = Field(
        ...,
        gt=0,
        description="Scale parameter (characteristic life in hours)"
    )
    gamma_hours: float = Field(
        default=0.0,
        ge=0,
        description="Location parameter (failure-free life)"
    )

    # Remaining Useful Life estimates
    rul_p10_hours: float = Field(
        ...,
        description="RUL at 10% probability of failure"
    )
    rul_p50_hours: float = Field(
        ...,
        description="RUL at 50% probability of failure (median)"
    )
    rul_p90_hours: float = Field(
        ...,
        description="RUL at 90% probability of failure"
    )

    # Current failure probability
    current_age_hours: float = Field(
        ...,
        ge=0,
        description="Current equipment age in hours"
    )
    current_failure_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current cumulative failure probability"
    )
    conditional_failure_probability_30d: float = Field(
        ...,
        ge=0,
        le=1,
        description="Conditional probability of failure in next 30 days"
    )

    # Confidence bounds
    beta_ci_lower: Optional[float] = Field(
        default=None,
        description="Beta lower confidence bound"
    )
    beta_ci_upper: Optional[float] = Field(
        default=None,
        description="Beta upper confidence bound"
    )
    eta_ci_lower_hours: Optional[float] = Field(
        default=None,
        description="Eta lower confidence bound"
    )
    eta_ci_upper_hours: Optional[float] = Field(
        default=None,
        description="Eta upper confidence bound"
    )
    confidence_level: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Confidence level for intervals"
    )

    # Failure mode interpretation
    failure_mode_interpretation: str = Field(
        ...,
        description="Interpretation of beta value"
    )

    # Fit statistics
    r_squared: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="R-squared of fit"
    )
    n_failures: int = Field(
        default=0,
        ge=0,
        description="Number of failures in dataset"
    )
    n_censored: int = Field(
        default=0,
        ge=0,
        description="Number of censored observations"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class VibrationAnalysisResult(BaseModel):
    """Vibration analysis result."""

    sensor_id: str = Field(..., description="Analyzed sensor")
    timestamp: datetime = Field(..., description="Analysis timestamp")

    # Overall levels
    overall_velocity_mm_s: float = Field(
        ...,
        ge=0,
        description="Overall velocity (mm/s RMS)"
    )
    overall_acceleration_g: float = Field(
        ...,
        ge=0,
        description="Overall acceleration (g RMS)"
    )
    overall_displacement_um: Optional[float] = Field(
        default=None,
        ge=0,
        description="Overall displacement (um p-p)"
    )

    # ISO 10816 assessment
    iso_zone: AlertSeverity = Field(
        ...,
        description="ISO 10816 zone classification"
    )

    # Spectral analysis
    dominant_frequency_hz: float = Field(
        ...,
        description="Dominant frequency in spectrum"
    )
    dominant_amplitude: float = Field(
        ...,
        description="Amplitude at dominant frequency"
    )
    harmonics_detected: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Detected harmonic frequencies"
    )

    # Fault indicators
    bearing_defect_detected: bool = Field(
        default=False,
        description="Bearing defect indicator"
    )
    bearing_defect_type: Optional[str] = Field(
        default=None,
        description="Type of bearing defect (BPFO, BPFI, BSF, FTF)"
    )
    imbalance_detected: bool = Field(
        default=False,
        description="Imbalance detected"
    )
    imbalance_severity: Optional[str] = Field(
        default=None,
        description="Imbalance severity"
    )
    misalignment_detected: bool = Field(
        default=False,
        description="Misalignment detected"
    )
    misalignment_type: Optional[str] = Field(
        default=None,
        description="Angular, parallel, or combined"
    )
    looseness_detected: bool = Field(
        default=False,
        description="Mechanical looseness detected"
    )

    # Trend
    trend_direction: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Trend over recent history"
    )
    trend_rate_pct_per_day: Optional[float] = Field(
        default=None,
        description="Rate of change per day"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Analysis recommendations"
    )


class OilAnalysisResult(BaseModel):
    """Oil analysis interpretation result."""

    sample_id: str = Field(..., description="Analyzed sample")
    timestamp: datetime = Field(..., description="Analysis timestamp")

    # Overall condition
    oil_condition: HealthStatus = Field(
        ...,
        description="Overall oil condition"
    )
    remaining_useful_life_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Estimated remaining useful life"
    )

    # Viscosity assessment
    viscosity_status: str = Field(
        ...,
        description="Viscosity status: normal, low, high"
    )
    viscosity_change_pct: float = Field(
        ...,
        description="Change from baseline"
    )

    # Acid number assessment
    tan_status: AlertSeverity = Field(
        ...,
        description="TAN status"
    )

    # Contamination assessment
    water_status: AlertSeverity = Field(
        ...,
        description="Water contamination status"
    )
    particle_status: AlertSeverity = Field(
        ...,
        description="Particle contamination status"
    )

    # Wear assessment
    wear_trend: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Wear metals trend"
    )
    primary_wear_metal: Optional[str] = Field(
        default=None,
        description="Primary wear metal detected"
    )
    wear_source_probable: Optional[str] = Field(
        default=None,
        description="Probable wear source"
    )

    # Recommendations
    oil_change_recommended: bool = Field(
        default=False,
        description="Oil change recommended"
    )
    filtration_recommended: bool = Field(
        default=False,
        description="Enhanced filtration recommended"
    )
    investigation_needed: bool = Field(
        default=False,
        description="Further investigation needed"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Detailed recommendations"
    )


class ThermographyResult(BaseModel):
    """Thermal analysis result."""

    image_id: str = Field(..., description="Analyzed image")
    timestamp: datetime = Field(..., description="Analysis timestamp")

    # Temperature analysis
    max_temperature_c: float = Field(..., description="Maximum temperature")
    reference_temperature_c: Optional[float] = Field(
        default=None,
        description="Reference/baseline temperature"
    )
    delta_t_c: Optional[float] = Field(
        default=None,
        description="Temperature differential"
    )

    # Hot spot detection
    hot_spots_detected: int = Field(
        default=0,
        ge=0,
        description="Number of hot spots detected"
    )
    hot_spots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Hot spot details"
    )

    # Severity assessment
    thermal_severity: AlertSeverity = Field(
        ...,
        description="Overall thermal severity"
    )

    # Probable causes
    probable_causes: List[str] = Field(
        default_factory=list,
        description="Probable causes of anomaly"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )


class MCSAResult(BaseModel):
    """Motor Current Signature Analysis result."""

    sensor_id: str = Field(..., description="Analyzed sensor")
    timestamp: datetime = Field(..., description="Analysis timestamp")

    # Current levels
    avg_current_a: float = Field(..., ge=0, description="Average phase current")
    current_unbalance_pct: float = Field(
        ...,
        ge=0,
        description="Current unbalance percentage"
    )

    # Fault detection
    bearing_defect_detected: bool = Field(
        default=False,
        description="Bearing defect from current signature"
    )
    bearing_defect_severity_db: Optional[float] = Field(
        default=None,
        description="Bearing defect sideband amplitude (dB)"
    )
    rotor_bar_fault_detected: bool = Field(
        default=False,
        description="Broken rotor bar detected"
    )
    rotor_bar_fault_severity_db: Optional[float] = Field(
        default=None,
        description="Rotor bar fault sideband amplitude (dB)"
    )
    eccentricity_detected: bool = Field(
        default=False,
        description="Eccentricity detected"
    )
    eccentricity_severity_db: Optional[float] = Field(
        default=None,
        description="Eccentricity sideband amplitude (dB)"
    )
    stator_fault_detected: bool = Field(
        default=False,
        description="Stator winding fault detected"
    )

    # Overall assessment
    motor_health: HealthStatus = Field(
        ...,
        description="Overall motor health"
    )
    confidence: DiagnosisConfidence = Field(
        default=DiagnosisConfidence.MEDIUM,
        description="Diagnosis confidence"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )


class FailurePrediction(BaseModel):
    """ML-based failure prediction result."""

    failure_mode: FailureMode = Field(
        ...,
        description="Predicted failure mode"
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Failure probability"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Prediction confidence"
    )
    time_to_failure_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated time to failure"
    )
    uncertainty_lower_hours: Optional[float] = Field(
        default=None,
        description="Lower uncertainty bound"
    )
    uncertainty_upper_hours: Optional[float] = Field(
        default=None,
        description="Upper uncertainty bound"
    )
    feature_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="SHAP feature importance"
    )
    top_contributing_features: List[str] = Field(
        default_factory=list,
        description="Top features driving prediction"
    )
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")


class MaintenanceRecommendation(BaseModel):
    """Maintenance action recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique recommendation ID"
    )
    failure_mode: FailureMode = Field(
        ...,
        description="Related failure mode"
    )
    priority: WorkOrderPriority = Field(
        ...,
        description="Action priority"
    )
    action_type: str = Field(
        ...,
        description="Type of action required"
    )
    description: str = Field(
        ...,
        description="Detailed recommendation"
    )
    deadline_hours: Optional[float] = Field(
        default=None,
        description="Recommended completion deadline"
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated cost"
    )
    estimated_duration_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated duration"
    )
    parts_required: List[str] = Field(
        default_factory=list,
        description="Parts/materials required"
    )
    skills_required: List[str] = Field(
        default_factory=list,
        description="Required skills/certifications"
    )
    risk_if_delayed: str = Field(
        default="",
        description="Risk of delaying action"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting recommendation"
    )


class WorkOrderRequest(BaseModel):
    """Request to create a CMMS work order."""

    work_order_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Internal work order ID"
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_tag: Optional[str] = Field(
        default=None,
        description="Plant equipment tag"
    )
    order_type: WorkOrderType = Field(
        ...,
        description="Work order type"
    )
    priority: WorkOrderPriority = Field(
        ...,
        description="Work order priority"
    )
    title: str = Field(
        ...,
        max_length=100,
        description="Work order title"
    )
    description: str = Field(
        ...,
        max_length=2000,
        description="Detailed description"
    )
    failure_modes: List[FailureMode] = Field(
        default_factory=list,
        description="Related failure modes"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended maintenance actions"
    )
    parts_required: List[str] = Field(
        default_factory=list,
        description="Required parts"
    )
    estimated_duration_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated duration"
    )
    required_by_date: Optional[datetime] = Field(
        default=None,
        description="Required completion date"
    )
    created_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    source_analysis_id: str = Field(
        ...,
        description="Source analysis request ID"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class PredictiveMaintenanceOutput(BaseModel):
    """
    Output from predictive maintenance analysis.

    This comprehensive output model contains all analysis results,
    health assessments, failure predictions, and recommendations.

    Attributes:
        equipment_id: Equipment identifier
        health_status: Overall equipment health
        failure_predictions: ML failure predictions
        recommendations: Maintenance recommendations
        work_orders: Generated work order requests

    Example:
        >>> output = predictor.process(input_data)
        >>> print(f"Equipment health: {output.health_status}")
        >>> for pred in output.failure_predictions:
        ...     print(f"{pred.failure_mode}: {pred.probability:.1%}")
    """

    request_id: str = Field(..., description="Original request ID")
    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    status: str = Field(
        default="success",
        description="Analysis status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing duration"
    )

    # Overall health assessment
    health_status: HealthStatus = Field(
        ...,
        description="Overall equipment health"
    )
    health_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Health score (0-100)"
    )
    health_trend: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Health trend direction"
    )

    # Weibull RUL analysis
    weibull_analysis: Optional[WeibullAnalysisResult] = Field(
        default=None,
        description="Weibull RUL analysis"
    )

    # Sensor analysis results
    vibration_analysis: List[VibrationAnalysisResult] = Field(
        default_factory=list,
        description="Vibration analysis results"
    )
    oil_analysis_result: Optional[OilAnalysisResult] = Field(
        default=None,
        description="Oil analysis interpretation"
    )
    thermography_results: List[ThermographyResult] = Field(
        default_factory=list,
        description="Thermography analysis"
    )
    mcsa_results: List[MCSAResult] = Field(
        default_factory=list,
        description="MCSA results"
    )

    # ML failure predictions
    failure_predictions: List[FailurePrediction] = Field(
        default_factory=list,
        description="ML failure predictions"
    )
    highest_risk_failure_mode: Optional[FailureMode] = Field(
        default=None,
        description="Highest risk failure mode"
    )
    overall_failure_probability_30d: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Overall 30-day failure probability"
    )

    # Remaining useful life
    rul_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated remaining useful life (hours)"
    )
    rul_confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="RUL confidence interval (P10, P90)"
    )

    # Active alerts
    active_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently active alerts"
    )
    alert_count_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Alert counts by severity"
    )

    # Recommendations and work orders
    recommendations: List[MaintenanceRecommendation] = Field(
        default_factory=list,
        description="Maintenance recommendations"
    )
    work_orders: List[WorkOrderRequest] = Field(
        default_factory=list,
        description="Generated work orders"
    )

    # Key performance indicators
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Metadata
    analysis_methods: List[str] = Field(
        default_factory=list,
        description="Analysis methods used"
    )
    data_quality_score: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Input data quality score"
    )
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Model versions used"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# UPDATE FORWARD REFERENCES
# =============================================================================

PredictiveMaintenanceInput.update_forward_refs()
PredictiveMaintenanceOutput.update_forward_refs()
