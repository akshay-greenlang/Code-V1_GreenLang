# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Data Schemas Module
=====================================

This module defines all Pydantic data models (schemas) for the GL-005
Combustion Diagnostics Agent. These schemas ensure type safety, validation,
and clear data contracts throughout the agent.

Schema Categories:
    - Input schemas (sensor data, analysis requests)
    - Output schemas (diagnostics results, recommendations)
    - Internal state schemas (trending data, ML features)
    - Integration schemas (CMMS work orders, compliance reports)

ZERO-HALLUCINATION: All calculations use these validated schemas to ensure
deterministic, reproducible results with full provenance tracking.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    AnomalyType,
    ComplianceFramework,
    FuelCategory,
    MaintenancePriority,
)


# =============================================================================
# ENUMS FOR SCHEMAS
# =============================================================================

class CQIRating(str, Enum):
    """CQI rating categories."""
    EXCELLENT = "excellent"   # 90-100
    GOOD = "good"             # 75-89
    ACCEPTABLE = "acceptable" # 60-74
    POOR = "poor"             # 40-59
    CRITICAL = "critical"     # 0-39


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


class TrendDirection(str, Enum):
    """Trend direction indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


class AnalysisStatus(str, Enum):
    """Analysis status indicators."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class FlueGasReading(BaseModel):
    """
    Single flue gas composition reading from sensors.

    All concentration values should be on a DRY BASIS unless otherwise noted.
    This is the primary input from GL-018 or direct sensors.
    """

    timestamp: datetime = Field(
        ...,
        description="Reading timestamp (UTC)"
    )

    # Primary gas components (% or ppm as noted)
    oxygen_pct: float = Field(
        ...,
        ge=0.0,
        le=21.0,
        description="Oxygen concentration (% dry basis)"
    )
    co2_pct: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Carbon dioxide concentration (% dry basis)"
    )
    co_ppm: float = Field(
        ...,
        ge=0.0,
        le=100000.0,
        description="Carbon monoxide concentration (ppm dry basis)"
    )
    nox_ppm: float = Field(
        default=0.0,
        ge=0.0,
        le=10000.0,
        description="NOx concentration (ppm dry basis, as NO2)"
    )

    # Optional additional components
    so2_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10000.0,
        description="SO2 concentration (ppm dry basis)"
    )
    combustibles_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Unburned combustibles (% by volume)"
    )
    moisture_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50.0,
        description="Flue gas moisture content (%)"
    )

    # Temperature and pressure
    flue_gas_temp_c: float = Field(
        ...,
        ge=50.0,
        le=1500.0,
        description="Flue gas temperature (Celsius)"
    )
    ambient_temp_c: Optional[float] = Field(
        default=25.0,
        ge=-40.0,
        le=60.0,
        description="Ambient temperature (Celsius)"
    )
    barometric_pressure_kpa: Optional[float] = Field(
        default=101.325,
        ge=80.0,
        le=110.0,
        description="Barometric pressure (kPa)"
    )

    # Sensor quality indicators
    sensor_status: str = Field(
        default="ok",
        description="Sensor status: ok, degraded, fault"
    )
    data_quality_flag: str = Field(
        default="good",
        description="Data quality: good, suspect, bad"
    )

    @validator("oxygen_pct")
    def validate_oxygen_range(cls, v):
        """Validate oxygen is within reasonable combustion range."""
        if v < 0.5:
            # Below 0.5% O2 is dangerous - incomplete combustion
            raise ValueError("Oxygen below safe combustion level (0.5%)")
        return v


class CombustionOperatingData(BaseModel):
    """
    Operating data for combustion equipment.

    Provides context for diagnostics including load, fuel, and control states.
    """

    timestamp: datetime = Field(..., description="Data timestamp (UTC)")

    # Load information
    firing_rate_pct: float = Field(
        ...,
        ge=0.0,
        le=110.0,
        description="Current firing rate (% of capacity)"
    )
    steam_flow_kg_h: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Steam flow rate (kg/h)"
    )
    heat_output_mw: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Heat output (MW thermal)"
    )

    # Fuel data
    fuel_flow_rate: float = Field(
        ...,
        ge=0.0,
        description="Fuel flow rate (units depend on fuel type)"
    )
    fuel_flow_unit: str = Field(
        default="m3/h",
        description="Fuel flow unit: m3/h, kg/h, L/h"
    )
    fuel_type: FuelCategory = Field(
        default=FuelCategory.NATURAL_GAS,
        description="Current fuel type"
    )
    fuel_hhv_mj_kg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=60.0,
        description="Fuel higher heating value (MJ/kg)"
    )

    # Combustion air
    combustion_air_flow_m3_h: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Combustion air flow (m3/h at STP)"
    )
    air_temp_c: Optional[float] = Field(
        default=25.0,
        ge=-40.0,
        le=400.0,
        description="Combustion air temperature (Celsius)"
    )
    air_humidity_pct: Optional[float] = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Combustion air relative humidity (%)"
    )

    # Control states
    burner_status: str = Field(
        default="firing",
        description="Burner status: off, pilot, low_fire, modulating, high_fire"
    )
    damper_position_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Air damper position (%)"
    )
    control_mode: str = Field(
        default="auto",
        description="Control mode: manual, auto, cascade"
    )

    # Operating hours
    operating_hours_total: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total operating hours"
    )
    operating_hours_since_maintenance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Operating hours since last maintenance"
    )


class DiagnosticsInput(BaseModel):
    """
    Complete input for combustion diagnostics analysis.

    Combines flue gas readings with operating data for comprehensive analysis.
    """

    # Identification
    equipment_id: str = Field(..., description="Equipment identifier")
    request_id: str = Field(..., description="Unique analysis request ID")

    # Data
    flue_gas: FlueGasReading = Field(..., description="Flue gas reading")
    operating_data: CombustionOperatingData = Field(
        ...,
        description="Operating data"
    )

    # Analysis options
    run_cqi_analysis: bool = Field(default=True)
    run_anomaly_detection: bool = Field(default=True)
    run_fuel_characterization: bool = Field(default=True)
    run_maintenance_prediction: bool = Field(default=True)

    # Historical context (optional)
    historical_readings: Optional[List[FlueGasReading]] = Field(
        default=None,
        description="Recent historical readings for trend analysis"
    )


# =============================================================================
# OUTPUT SCHEMAS - CQI
# =============================================================================

class CQIComponentScore(BaseModel):
    """Score for individual CQI component."""

    component: str = Field(..., description="Component name")
    raw_value: float = Field(..., description="Raw measurement value")
    normalized_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Normalized score (0-100)"
    )
    weight: float = Field(..., ge=0.0, le=1.0, description="Component weight")
    weighted_score: float = Field(..., description="Weight * normalized score")
    status: str = Field(..., description="Status: optimal, acceptable, warning, critical")


class CQIResult(BaseModel):
    """
    Combustion Quality Index calculation result.

    The CQI is a proprietary diagnostic metric (0-100) that provides a single
    number assessment of combustion quality based on multiple factors.
    """

    # Overall CQI
    cqi_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall CQI score (0-100)"
    )
    cqi_rating: CQIRating = Field(..., description="CQI rating category")

    # Component breakdown
    components: List[CQIComponentScore] = Field(
        ...,
        description="Individual component scores"
    )

    # Corrected values (to reference O2)
    co_corrected_ppm: float = Field(
        ...,
        description="CO corrected to reference O2 (ppm)"
    )
    nox_corrected_ppm: float = Field(
        ...,
        description="NOx corrected to reference O2 (ppm)"
    )
    o2_reference_pct: float = Field(
        ...,
        description="Reference O2 percentage used"
    )

    # Efficiency indicators
    excess_air_pct: float = Field(..., description="Excess air percentage")
    combustion_efficiency_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Estimated combustion efficiency (%)"
    )

    # Trend compared to baseline
    trend_vs_baseline: TrendDirection = Field(
        default=TrendDirection.UNKNOWN,
        description="Trend compared to baseline"
    )
    baseline_cqi: Optional[float] = Field(
        default=None,
        description="Baseline CQI for comparison"
    )

    # Calculation metadata
    calculation_timestamp: datetime = Field(
        ...,
        description="Calculation timestamp"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash"
    )


# =============================================================================
# OUTPUT SCHEMAS - ANOMALY DETECTION
# =============================================================================

class AnomalyEvent(BaseModel):
    """Single anomaly detection event."""

    anomaly_id: str = Field(..., description="Unique anomaly identifier")
    timestamp: datetime = Field(..., description="Detection timestamp")

    # Classification
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    severity: AnomalySeverity = Field(..., description="Severity level")

    # Detection details
    detection_method: str = Field(
        ...,
        description="Detection method: spc, ml, rule_based"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence (0-1)"
    )

    # Values
    observed_value: float = Field(..., description="Observed value")
    expected_value: float = Field(..., description="Expected/normal value")
    deviation_pct: float = Field(..., description="Deviation percentage")

    # Context
    affected_parameter: str = Field(..., description="Affected parameter name")
    potential_causes: List[str] = Field(
        default_factory=list,
        description="Potential root causes"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended corrective actions"
    )

    # SPC specific (if applicable)
    control_limit_upper: Optional[float] = Field(default=None)
    control_limit_lower: Optional[float] = Field(default=None)
    sigma_deviation: Optional[float] = Field(default=None)

    # ML specific (if applicable)
    anomaly_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ML anomaly score"
    )
    feature_contributions: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature contributions to anomaly"
    )


class AnomalyDetectionResult(BaseModel):
    """Complete anomaly detection analysis result."""

    # Overall status
    status: AnalysisStatus = Field(..., description="Analysis status")
    anomaly_detected: bool = Field(..., description="Any anomaly detected")
    total_anomalies: int = Field(default=0, description="Total anomalies found")

    # Detected anomalies
    anomalies: List[AnomalyEvent] = Field(
        default_factory=list,
        description="List of detected anomalies"
    )

    # Summary by severity
    critical_count: int = Field(default=0)
    alarm_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    info_count: int = Field(default=0)

    # SPC status
    spc_in_control: bool = Field(
        default=True,
        description="SPC control status"
    )
    spc_violations: List[str] = Field(
        default_factory=list,
        description="SPC rule violations"
    )

    # ML status
    ml_health_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall ML health score"
    )

    # Metadata
    analysis_timestamp: datetime = Field(...)
    samples_analyzed: int = Field(default=1)
    provenance_hash: str = Field(...)


# =============================================================================
# OUTPUT SCHEMAS - FUEL CHARACTERIZATION
# =============================================================================

class FuelProperties(BaseModel):
    """Characterized fuel properties."""

    fuel_category: FuelCategory = Field(..., description="Detected fuel category")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Characterization confidence"
    )

    # Composition (dry basis, mole %)
    carbon_content_pct: float = Field(..., ge=0.0, le=100.0)
    hydrogen_content_pct: float = Field(..., ge=0.0, le=100.0)
    oxygen_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    nitrogen_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    sulfur_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)

    # Heating values
    hhv_mj_kg: float = Field(..., ge=0.0, description="Higher heating value (MJ/kg)")
    lhv_mj_kg: float = Field(..., ge=0.0, description="Lower heating value (MJ/kg)")

    # Stoichiometric values
    stoich_air_fuel_ratio: float = Field(
        ...,
        ge=0.0,
        description="Stoichiometric air-fuel ratio (mass basis)"
    )
    theoretical_co2_pct: float = Field(
        ...,
        ge=0.0,
        le=25.0,
        description="Theoretical CO2 at stoichiometric combustion (%)"
    )

    # Emission factors
    co2_emission_factor_kg_mj: float = Field(
        ...,
        ge=0.0,
        description="CO2 emission factor (kg CO2/MJ)"
    )


class FuelCharacterizationResult(BaseModel):
    """Complete fuel characterization result."""

    status: AnalysisStatus = Field(...)

    # Primary fuel
    primary_fuel: FuelProperties = Field(...)

    # Blend detection (if applicable)
    is_fuel_blend: bool = Field(default=False)
    blend_components: Optional[List[FuelProperties]] = Field(default=None)
    blend_fractions: Optional[List[float]] = Field(default=None)

    # Comparison to expected
    matches_configured_fuel: bool = Field(default=True)
    deviation_from_expected_pct: float = Field(default=0.0)

    # Quality indicators
    fuel_quality_rating: str = Field(
        default="normal",
        description="Quality: excellent, normal, poor, suspect"
    )
    quality_concerns: List[str] = Field(default_factory=list)

    # Metadata
    analysis_timestamp: datetime = Field(...)
    provenance_hash: str = Field(...)


# =============================================================================
# OUTPUT SCHEMAS - MAINTENANCE ADVISORY
# =============================================================================

class MaintenanceRecommendation(BaseModel):
    """Single maintenance recommendation."""

    recommendation_id: str = Field(...)
    timestamp: datetime = Field(...)

    # Classification
    maintenance_type: str = Field(
        ...,
        description="Type: inspection, cleaning, repair, replacement, calibration"
    )
    priority: MaintenancePriority = Field(...)
    component: str = Field(..., description="Affected component")

    # Description
    title: str = Field(..., description="Short title")
    description: str = Field(..., description="Detailed description")
    justification: str = Field(..., description="Why this is recommended")

    # Timing
    recommended_by_date: Optional[datetime] = Field(default=None)
    estimated_duration_hours: Optional[float] = Field(default=None)

    # Impact if not addressed
    risk_if_deferred: str = Field(
        default="medium",
        description="Risk level: low, medium, high, critical"
    )
    potential_consequences: List[str] = Field(default_factory=list)

    # Cost estimates (if available)
    estimated_labor_hours: Optional[float] = Field(default=None)
    estimated_parts_cost: Optional[float] = Field(default=None)


class FoulingAssessment(BaseModel):
    """Fouling/deposit assessment."""

    fouling_detected: bool = Field(default=False)
    fouling_severity: str = Field(
        default="none",
        description="Severity: none, light, moderate, heavy, severe"
    )

    # Indicators
    efficiency_loss_pct: float = Field(default=0.0, ge=0.0)
    stack_temp_increase_c: float = Field(default=0.0)
    delta_t_degradation_pct: float = Field(default=0.0)

    # Prediction
    days_until_cleaning_recommended: Optional[int] = Field(default=None)
    predicted_efficiency_loss_30d: Optional[float] = Field(default=None)

    # Confidence
    assessment_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class BurnerWearAssessment(BaseModel):
    """Burner wear assessment."""

    wear_detected: bool = Field(default=False)
    wear_level: str = Field(
        default="normal",
        description="Level: normal, early_wear, moderate_wear, significant_wear, replacement_needed"
    )

    # Indicators
    operating_hours: float = Field(default=0.0, ge=0.0)
    expected_life_remaining_pct: float = Field(default=100.0, ge=0.0, le=100.0)

    # Symptoms
    co_trend_slope: float = Field(default=0.0, description="CO trend (ppm/day)")
    flame_stability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    ignition_reliability: float = Field(default=1.0, ge=0.0, le=1.0)

    # Prediction
    estimated_remaining_life_hours: Optional[float] = Field(default=None)
    replacement_recommended_by: Optional[datetime] = Field(default=None)

    # Confidence
    assessment_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MaintenanceAdvisoryResult(BaseModel):
    """Complete maintenance advisory result."""

    status: AnalysisStatus = Field(...)

    # Assessments
    fouling: FoulingAssessment = Field(...)
    burner_wear: BurnerWearAssessment = Field(...)

    # Recommendations
    recommendations: List[MaintenanceRecommendation] = Field(default_factory=list)
    urgent_actions_required: bool = Field(default=False)

    # Overall equipment health
    equipment_health_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall equipment health (0-100)"
    )
    health_trend: TrendDirection = Field(default=TrendDirection.UNKNOWN)

    # Next scheduled maintenance
    next_recommended_maintenance: Optional[MaintenanceRecommendation] = Field(
        default=None
    )

    # Metadata
    analysis_timestamp: datetime = Field(...)
    provenance_hash: str = Field(...)


# =============================================================================
# OUTPUT SCHEMAS - WORK ORDER
# =============================================================================

class CMMSWorkOrder(BaseModel):
    """CMMS work order for maintenance integration."""

    work_order_id: str = Field(..., description="Generated work order ID")
    created_timestamp: datetime = Field(...)

    # Equipment
    equipment_id: str = Field(...)
    equipment_name: Optional[str] = Field(default=None)
    location: Optional[str] = Field(default=None)

    # Work details
    work_type: str = Field(..., description="PM, CM, INSP, EMRG")
    priority: MaintenancePriority = Field(...)
    title: str = Field(...)
    description: str = Field(...)

    # Scheduling
    requested_start_date: Optional[datetime] = Field(default=None)
    requested_end_date: Optional[datetime] = Field(default=None)
    estimated_hours: Optional[float] = Field(default=None)

    # Assignment
    assigned_craft: Optional[str] = Field(default=None)
    assigned_crew: Optional[str] = Field(default=None)

    # Source information
    source_agent: str = Field(default="GL-005")
    source_analysis_id: str = Field(...)
    source_recommendations: List[str] = Field(default_factory=list)

    # Status
    status: str = Field(
        default="pending_approval",
        description="Status: pending_approval, approved, scheduled, in_progress, completed"
    )

    # Audit
    provenance_hash: str = Field(...)


# =============================================================================
# OUTPUT SCHEMAS - COMPLIANCE
# =============================================================================

class ComplianceStatus(BaseModel):
    """Compliance status for a single framework."""

    framework: ComplianceFramework = Field(...)
    compliant: bool = Field(...)

    # Limit checks
    limits_checked: int = Field(default=0)
    limits_passed: int = Field(default=0)
    limits_exceeded: int = Field(default=0)

    # Exceedances
    exceedances: List[Dict[str, Any]] = Field(default_factory=list)

    # Warnings (approaching limits)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)


class ComplianceReportResult(BaseModel):
    """Complete compliance report result."""

    status: AnalysisStatus = Field(...)

    # Overall compliance
    overall_compliant: bool = Field(...)
    frameworks_checked: List[ComplianceFramework] = Field(...)

    # Per-framework status
    framework_status: List[ComplianceStatus] = Field(default_factory=list)

    # Summary statistics
    total_limits_checked: int = Field(default=0)
    total_exceedances: int = Field(default=0)
    total_warnings: int = Field(default=0)

    # Period covered
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)

    # Metadata
    report_timestamp: datetime = Field(...)
    provenance_hash: str = Field(...)


# =============================================================================
# COMPLETE DIAGNOSTICS OUTPUT
# =============================================================================

class DiagnosticsOutput(BaseModel):
    """
    Complete output from GL-005 Combustion Diagnostics Agent.

    This is the master output schema that combines all diagnostic analyses
    into a single comprehensive result.

    IMPORTANT: This output is ADVISORY ONLY. GL-005 does not execute any
    control actions. Recommendations should be reviewed and implemented
    by operators or passed to GL-018 for automated control.
    """

    # Identification
    request_id: str = Field(..., description="Original request ID")
    equipment_id: str = Field(..., description="Equipment identifier")
    agent_id: str = Field(default="GL-005", description="Agent identifier")
    agent_version: str = Field(default="1.0.0", description="Agent version")

    # Overall status
    status: AnalysisStatus = Field(..., description="Overall analysis status")
    processing_time_ms: float = Field(..., description="Processing time in ms")

    # Analysis results
    cqi: Optional[CQIResult] = Field(
        default=None,
        description="Combustion Quality Index result"
    )
    anomaly_detection: Optional[AnomalyDetectionResult] = Field(
        default=None,
        description="Anomaly detection result"
    )
    fuel_characterization: Optional[FuelCharacterizationResult] = Field(
        default=None,
        description="Fuel characterization result"
    )
    maintenance_advisory: Optional[MaintenanceAdvisoryResult] = Field(
        default=None,
        description="Maintenance advisory result"
    )
    compliance: Optional[ComplianceReportResult] = Field(
        default=None,
        description="Compliance report result"
    )

    # Generated work orders (if CMMS integration enabled)
    work_orders: List[CMMSWorkOrder] = Field(
        default_factory=list,
        description="Generated CMMS work orders"
    )

    # Summary alerts
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary alerts for operator attention"
    )

    # Recommendations (consolidated from all analyses)
    recommendations: List[str] = Field(
        default_factory=list,
        description="Consolidated recommendations"
    )

    # Control suggestions (for GL-018)
    control_suggestions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Suggested control actions for GL-018"
    )

    # Timestamps
    input_timestamp: datetime = Field(..., description="Input data timestamp")
    output_timestamp: datetime = Field(..., description="Output generation timestamp")

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash of complete output for audit trail"
    )

    # Audit metadata
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Calculation audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
