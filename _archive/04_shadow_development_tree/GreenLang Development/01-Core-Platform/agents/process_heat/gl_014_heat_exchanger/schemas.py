# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Schema Definitions

This module defines all Pydantic models for inputs, outputs, analysis results,
and status reporting for the Heat Exchanger Optimization Agent.

All models include comprehensive validation, documentation, and support for
provenance tracking through SHA-256 hashes.

References:
    - TEMA Standards 9th Edition
    - ASME PTC 12.5 Performance Test Code
    - Heat Exchanger Design Handbook

Example:
    >>> from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    ...     HeatExchangerInput,
    ...     HeatExchangerOutput,
    ... )
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    AlertSeverity,
    CleaningMethod,
    ExchangerType,
    FailureMode,
    FlowArrangement,
    FoulingCategory,
)


# =============================================================================
# ENUMS
# =============================================================================

class HealthStatus(str, Enum):
    """Equipment health status."""
    EXCELLENT = "excellent"  # >90% effectiveness
    GOOD = "good"  # 80-90% effectiveness
    FAIR = "fair"  # 70-80% effectiveness
    POOR = "poor"  # 60-70% effectiveness
    CRITICAL = "critical"  # <60% effectiveness


class TrendDirection(str, Enum):
    """Parameter trend direction."""
    STABLE = "stable"
    IMPROVING = "improving"
    DEGRADING = "degrading"
    RAPID_DEGRADATION = "rapid_degradation"


class OperatingMode(str, Enum):
    """Exchanger operating mode."""
    NORMAL = "normal"
    REDUCED_CAPACITY = "reduced_capacity"
    BYPASS = "bypass"
    STANDBY = "standby"
    SHUTDOWN = "shutdown"
    CLEANING = "cleaning"


class TestCompliance(str, Enum):
    """ASME PTC 12.5 test compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_TESTED = "not_tested"


# =============================================================================
# INPUT SCHEMAS - PROCESS DATA
# =============================================================================

class StreamConditions(BaseModel):
    """Stream conditions (temperature, pressure, flow)."""

    temperature_c: float = Field(
        ...,
        description="Temperature (Celsius)"
    )
    pressure_barg: float = Field(
        ...,
        ge=0,
        description="Pressure (barg)"
    )
    mass_flow_kg_s: float = Field(
        ...,
        gt=0,
        description="Mass flow rate (kg/s)"
    )
    density_kg_m3: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fluid density (kg/m3)"
    )
    viscosity_cp: Optional[float] = Field(
        default=None,
        gt=0,
        description="Dynamic viscosity (centipoise)"
    )
    specific_heat_kj_kgk: Optional[float] = Field(
        default=None,
        gt=0,
        description="Specific heat capacity (kJ/kg-K)"
    )
    thermal_conductivity_w_mk: Optional[float] = Field(
        default=None,
        gt=0,
        description="Thermal conductivity (W/m-K)"
    )


class ProcessMeasurement(BaseModel):
    """Single process measurement with metadata."""

    tag: str = Field(..., description="Instrument tag")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Engineering unit")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp"
    )
    quality: str = Field(
        default="good",
        description="Data quality: good, uncertain, bad"
    )
    sensor_type: Optional[str] = Field(
        default=None,
        description="Sensor type"
    )


class HeatExchangerOperatingData(BaseModel):
    """Current operating data for heat exchanger."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Shell side conditions
    shell_inlet: StreamConditions = Field(
        ...,
        description="Shell side inlet conditions"
    )
    shell_outlet: StreamConditions = Field(
        ...,
        description="Shell side outlet conditions"
    )
    shell_pressure_drop_bar: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured shell side pressure drop"
    )

    # Tube side conditions
    tube_inlet: StreamConditions = Field(
        ...,
        description="Tube side inlet conditions"
    )
    tube_outlet: StreamConditions = Field(
        ...,
        description="Tube side outlet conditions"
    )
    tube_pressure_drop_bar: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured tube side pressure drop"
    )

    # Operating mode
    operating_mode: OperatingMode = Field(
        default=OperatingMode.NORMAL,
        description="Current operating mode"
    )
    load_percent: float = Field(
        default=100.0,
        ge=0,
        le=150,
        description="Operating load percentage"
    )

    # Ambient conditions (for air-cooled)
    ambient_temperature_c: Optional[float] = Field(
        default=None,
        description="Ambient temperature for air-cooled"
    )
    ambient_humidity_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Ambient relative humidity"
    )

    @property
    def shell_delta_t(self) -> float:
        """Shell side temperature change."""
        return abs(
            self.shell_inlet.temperature_c - self.shell_outlet.temperature_c
        )

    @property
    def tube_delta_t(self) -> float:
        """Tube side temperature change."""
        return abs(
            self.tube_inlet.temperature_c - self.tube_outlet.temperature_c
        )


class TubeInspectionData(BaseModel):
    """Tube inspection results from eddy current testing."""

    inspection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Inspection identifier"
    )
    inspection_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Inspection date"
    )
    inspection_method: str = Field(
        default="eddy_current",
        description="Inspection method"
    )

    total_tubes: int = Field(
        ...,
        ge=1,
        description="Total number of tubes"
    )
    tubes_inspected: int = Field(
        ...,
        ge=0,
        description="Number of tubes inspected"
    )
    tubes_with_defects: int = Field(
        default=0,
        ge=0,
        description="Tubes with detected defects"
    )
    tubes_plugged: int = Field(
        default=0,
        ge=0,
        description="Number of plugged tubes"
    )

    # Defect summary
    wall_loss_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Wall loss categories: {'<20%': 50, '20-40%': 10, ...}"
    )
    defect_locations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of defect locations and severities"
    )

    # Recommendations
    tubes_recommended_for_plugging: List[int] = Field(
        default_factory=list,
        description="Tube numbers recommended for plugging"
    )
    retube_recommended: bool = Field(
        default=False,
        description="Retubing recommended"
    )

    @property
    def defect_rate(self) -> float:
        """Calculate defect rate."""
        if self.tubes_inspected == 0:
            return 0.0
        return self.tubes_with_defects / self.tubes_inspected

    @property
    def plugging_rate(self) -> float:
        """Calculate plugging rate."""
        if self.total_tubes == 0:
            return 0.0
        return self.tubes_plugged / self.total_tubes


class CleaningRecord(BaseModel):
    """Record of cleaning event."""

    cleaning_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Cleaning record identifier"
    )
    cleaning_date: datetime = Field(
        ...,
        description="Cleaning date"
    )
    cleaning_method: CleaningMethod = Field(
        ...,
        description="Cleaning method used"
    )
    duration_hours: float = Field(
        ...,
        gt=0,
        description="Cleaning duration"
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Cleaning cost"
    )

    # Effectiveness metrics
    u_before_cleaning: float = Field(
        ...,
        gt=0,
        description="U value before cleaning (W/m2K)"
    )
    u_after_cleaning: float = Field(
        ...,
        gt=0,
        description="U value after cleaning (W/m2K)"
    )
    effectiveness_before: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Thermal effectiveness before"
    )
    effectiveness_after: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Thermal effectiveness after"
    )

    notes: Optional[str] = Field(
        default=None,
        description="Cleaning notes"
    )

    @property
    def u_improvement_percent(self) -> float:
        """Calculate U value improvement."""
        return (
            (self.u_after_cleaning - self.u_before_cleaning)
            / self.u_before_cleaning * 100
        )


class HeatExchangerInput(BaseModel):
    """
    Input data for heat exchanger optimization analysis.

    This model encapsulates all process data, inspection results,
    and historical cleaning data for comprehensive analysis.

    Attributes:
        exchanger_id: Unique exchanger identifier
        operating_data: Current operating conditions
        inspection_data: Recent tube inspection results
        cleaning_history: Historical cleaning records

    Example:
        >>> input_data = HeatExchangerInput(
        ...     exchanger_id="E-1001",
        ...     operating_data=operating_data,
        ... )
    """

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    exchanger_id: str = Field(..., description="Exchanger identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis request timestamp"
    )

    # Current operating data
    operating_data: HeatExchangerOperatingData = Field(
        ...,
        description="Current operating conditions"
    )

    # Historical data (optional)
    operating_history: List[HeatExchangerOperatingData] = Field(
        default_factory=list,
        description="Historical operating data for trending"
    )

    # Inspection data (optional)
    inspection_data: Optional[TubeInspectionData] = Field(
        default=None,
        description="Recent tube inspection data"
    )

    # Cleaning history (optional)
    cleaning_history: List[CleaningRecord] = Field(
        default_factory=list,
        description="Historical cleaning records"
    )

    # Additional measurements
    additional_measurements: List[ProcessMeasurement] = Field(
        default_factory=list,
        description="Additional process measurements"
    )

    # Context
    time_since_last_cleaning_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Days since last cleaning"
    )
    running_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total running hours"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# OUTPUT SCHEMAS - ANALYSIS RESULTS
# =============================================================================

class ThermalPerformanceResult(BaseModel):
    """Thermal performance analysis results."""

    # Heat duty
    actual_duty_kw: float = Field(
        ...,
        description="Actual heat duty (kW)"
    )
    design_duty_kw: float = Field(
        ...,
        description="Design heat duty (kW)"
    )
    duty_ratio: float = Field(
        ...,
        ge=0,
        description="Actual/design duty ratio"
    )

    # Temperature differences
    lmtd_c: float = Field(
        ...,
        description="Log Mean Temperature Difference (C)"
    )
    lmtd_correction_factor: float = Field(
        default=1.0,
        ge=0,
        le=1.0,
        description="LMTD correction factor (F)"
    )
    corrected_lmtd_c: float = Field(
        ...,
        description="Corrected LMTD (C)"
    )
    approach_temperature_c: float = Field(
        ...,
        description="Temperature approach (C)"
    )

    # Overall heat transfer coefficient
    u_clean_w_m2k: float = Field(
        ...,
        gt=0,
        description="Clean U value (W/m2K)"
    )
    u_actual_w_m2k: float = Field(
        ...,
        gt=0,
        description="Current U value (W/m2K)"
    )
    u_design_w_m2k: float = Field(
        ...,
        gt=0,
        description="Design U value (W/m2K)"
    )
    u_degradation_percent: float = Field(
        ...,
        description="U value degradation from clean (%)"
    )

    # Effectiveness (e-NTU method)
    ntu: float = Field(
        ...,
        ge=0,
        description="Number of Transfer Units"
    )
    heat_capacity_ratio: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Heat capacity ratio (Cmin/Cmax)"
    )
    thermal_effectiveness: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Thermal effectiveness"
    )
    design_effectiveness: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Design thermal effectiveness"
    )
    effectiveness_ratio: float = Field(
        ...,
        ge=0,
        description="Actual/design effectiveness ratio"
    )

    # Individual heat transfer coefficients
    shell_htc_w_m2k: Optional[float] = Field(
        default=None,
        description="Shell side heat transfer coefficient"
    )
    tube_htc_w_m2k: Optional[float] = Field(
        default=None,
        description="Tube side heat transfer coefficient"
    )

    # Calculated fouling
    calculated_fouling_m2kw: float = Field(
        ...,
        ge=0,
        description="Calculated total fouling resistance"
    )

    # Trends
    effectiveness_trend: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Effectiveness trend"
    )
    trend_rate_per_day: Optional[float] = Field(
        default=None,
        description="Rate of effectiveness change per day"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class FoulingAnalysisResult(BaseModel):
    """Fouling analysis results with prediction."""

    # Current fouling state
    shell_fouling_m2kw: float = Field(
        ...,
        ge=0,
        description="Shell side fouling resistance (m2K/W)"
    )
    tube_fouling_m2kw: float = Field(
        ...,
        ge=0,
        description="Tube side fouling resistance (m2K/W)"
    )
    total_fouling_m2kw: float = Field(
        ...,
        ge=0,
        description="Total fouling resistance (m2K/W)"
    )
    fouling_factor_ratio: float = Field(
        ...,
        ge=0,
        description="Actual/design fouling factor ratio"
    )

    # Fouling rate
    fouling_rate_m2kw_per_day: float = Field(
        ...,
        ge=0,
        description="Current fouling rate (m2K/W per day)"
    )
    fouling_rate_trend: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Fouling rate trend"
    )

    # Predictions (ML-based)
    predicted_fouling_30d_m2kw: Optional[float] = Field(
        default=None,
        description="Predicted fouling in 30 days"
    )
    predicted_fouling_60d_m2kw: Optional[float] = Field(
        default=None,
        description="Predicted fouling in 60 days"
    )
    predicted_fouling_90d_m2kw: Optional[float] = Field(
        default=None,
        description="Predicted fouling in 90 days"
    )
    prediction_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description="Prediction confidence"
    )
    prediction_uncertainty_m2kw: Optional[float] = Field(
        default=None,
        description="Prediction uncertainty (m2K/W)"
    )

    # Time estimates
    days_to_cleaning_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated days until cleaning threshold"
    )
    days_to_critical_fouling: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated days until critical fouling"
    )

    # Fouling mechanism
    probable_fouling_mechanism: FoulingCategory = Field(
        default=FoulingCategory.COMBINED,
        description="Most probable fouling mechanism"
    )
    fouling_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Fouling attribution by mechanism (%)"
    )

    # Feature importance (from ML)
    feature_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="SHAP feature importance"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class HydraulicAnalysisResult(BaseModel):
    """Hydraulic performance analysis results."""

    # Shell side
    shell_pressure_drop_bar: float = Field(
        ...,
        ge=0,
        description="Calculated shell side pressure drop"
    )
    shell_dp_design_bar: float = Field(
        ...,
        ge=0,
        description="Design shell side pressure drop"
    )
    shell_dp_measured_bar: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured shell side pressure drop"
    )
    shell_dp_ratio: float = Field(
        ...,
        ge=0,
        description="Actual/design shell DP ratio"
    )
    shell_velocity_m_s: float = Field(
        ...,
        ge=0,
        description="Shell side velocity"
    )

    # Tube side
    tube_pressure_drop_bar: float = Field(
        ...,
        ge=0,
        description="Calculated tube side pressure drop"
    )
    tube_dp_design_bar: float = Field(
        ...,
        ge=0,
        description="Design tube side pressure drop"
    )
    tube_dp_measured_bar: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured tube side pressure drop"
    )
    tube_dp_ratio: float = Field(
        ...,
        ge=0,
        description="Actual/design tube DP ratio"
    )
    tube_velocity_m_s: float = Field(
        ...,
        ge=0,
        description="Tube side velocity"
    )

    # Fouling impact on pressure drop
    shell_dp_fouling_contribution_bar: float = Field(
        default=0.0,
        ge=0,
        description="Shell DP from fouling"
    )
    tube_dp_fouling_contribution_bar: float = Field(
        default=0.0,
        ge=0,
        description="Tube DP from fouling"
    )

    # Reynolds numbers
    shell_reynolds: float = Field(
        ...,
        gt=0,
        description="Shell side Reynolds number"
    )
    tube_reynolds: float = Field(
        ...,
        gt=0,
        description="Tube side Reynolds number"
    )

    # Alerts
    shell_dp_alarm: bool = Field(
        default=False,
        description="Shell DP exceeds limit"
    )
    tube_dp_alarm: bool = Field(
        default=False,
        description="Tube DP exceeds limit"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class TubeIntegrityResult(BaseModel):
    """Tube integrity analysis and prediction results."""

    # Current state
    current_wall_thickness_mm: float = Field(
        ...,
        gt=0,
        description="Estimated current wall thickness"
    )
    minimum_required_thickness_mm: float = Field(
        ...,
        gt=0,
        description="Minimum required wall thickness"
    )
    thickness_margin_mm: float = Field(
        ...,
        description="Thickness margin above minimum"
    )
    wall_loss_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Average wall loss percentage"
    )

    # Plugging status
    tubes_plugged: int = Field(
        default=0,
        ge=0,
        description="Number of plugged tubes"
    )
    plugging_rate_percent: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Plugging rate percentage"
    )
    tubes_at_risk: int = Field(
        default=0,
        ge=0,
        description="Tubes requiring attention"
    )

    # Predictions
    estimated_remaining_life_years: float = Field(
        ...,
        ge=0,
        description="Estimated remaining tube life"
    )
    remaining_life_confidence: float = Field(
        default=0.8,
        ge=0,
        le=1.0,
        description="Life estimate confidence"
    )
    predicted_failures_1yr: int = Field(
        default=0,
        ge=0,
        description="Predicted tube failures in 1 year"
    )
    predicted_failures_5yr: int = Field(
        default=0,
        ge=0,
        description="Predicted tube failures in 5 years"
    )

    # Weibull parameters (if available)
    weibull_beta: Optional[float] = Field(
        default=None,
        description="Weibull shape parameter"
    )
    weibull_eta_years: Optional[float] = Field(
        default=None,
        description="Weibull scale parameter (years)"
    )

    # Recommendations
    retube_recommended: bool = Field(
        default=False,
        description="Retubing recommended"
    )
    next_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Recommended next inspection date"
    )
    inspection_urgency: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Inspection urgency"
    )

    # Failure mode analysis
    failure_modes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active failure modes and risks"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class CleaningRecommendation(BaseModel):
    """Cleaning recommendation with optimization."""

    recommended: bool = Field(
        ...,
        description="Cleaning recommended"
    )
    urgency: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Cleaning urgency"
    )
    recommended_method: CleaningMethod = Field(
        ...,
        description="Recommended cleaning method"
    )
    alternative_methods: List[CleaningMethod] = Field(
        default_factory=list,
        description="Alternative cleaning methods"
    )

    # Timing
    optimal_cleaning_date: Optional[datetime] = Field(
        default=None,
        description="Optimal cleaning date"
    )
    latest_cleaning_date: Optional[datetime] = Field(
        default=None,
        description="Latest acceptable cleaning date"
    )
    days_until_recommended: float = Field(
        default=0.0,
        description="Days until recommended cleaning"
    )

    # Economics
    estimated_cleaning_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated cleaning cost"
    )
    production_loss_usd: float = Field(
        default=0.0,
        ge=0,
        description="Production loss during cleaning"
    )
    energy_savings_potential_usd_per_month: float = Field(
        default=0.0,
        ge=0,
        description="Energy savings from cleaning"
    )
    npv_of_cleaning_usd: float = Field(
        default=0.0,
        description="NPV of cleaning vs. waiting"
    )

    # Expected results
    expected_u_after_w_m2k: float = Field(
        default=0.0,
        gt=0,
        description="Expected U after cleaning"
    )
    expected_effectiveness_after: float = Field(
        default=0.0,
        ge=0,
        le=1.0,
        description="Expected effectiveness after"
    )
    expected_fouling_removal_percent: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Expected fouling removal"
    )

    reasoning: str = Field(
        default="",
        description="Recommendation reasoning"
    )


class EconomicAnalysisResult(BaseModel):
    """Economic analysis results."""

    # Current operating costs
    energy_loss_kw: float = Field(
        default=0.0,
        ge=0,
        description="Energy loss due to fouling (kW)"
    )
    energy_cost_usd_per_day: float = Field(
        default=0.0,
        ge=0,
        description="Daily energy cost from degradation"
    )
    energy_cost_usd_per_month: float = Field(
        default=0.0,
        ge=0,
        description="Monthly energy cost from degradation"
    )
    energy_cost_usd_per_year: float = Field(
        default=0.0,
        ge=0,
        description="Annual energy cost from degradation"
    )

    # Cleaning economics
    cleaning_roi_percent: float = Field(
        default=0.0,
        description="ROI from cleaning"
    )
    payback_period_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Payback period for cleaning"
    )
    optimal_cleaning_frequency_days: float = Field(
        default=365.0,
        gt=0,
        description="Optimal cleaning frequency"
    )
    annual_cleaning_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual cleaning cost"
    )

    # Replacement economics
    remaining_value_usd: float = Field(
        default=0.0,
        ge=0,
        description="Remaining equipment value"
    )
    replacement_timing_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optimal replacement timing"
    )
    replace_vs_maintain_npv_usd: float = Field(
        default=0.0,
        description="NPV: replace vs. maintain"
    )

    # Total cost of ownership
    annual_tco_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual total cost of ownership"
    )
    lifecycle_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Remaining lifecycle cost"
    )

    # Optimization opportunities
    optimization_savings_usd_per_year: float = Field(
        default=0.0,
        ge=0,
        description="Potential annual savings from optimization"
    )
    optimization_recommendations: List[str] = Field(
        default_factory=list,
        description="Cost optimization recommendations"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class Alert(BaseModel):
    """System alert."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Alert identifier"
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity"
    )
    category: str = Field(
        ...,
        description="Alert category"
    )
    message: str = Field(
        ...,
        description="Alert message"
    )
    parameter: Optional[str] = Field(
        default=None,
        description="Related parameter"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current value"
    )
    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold value"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp"
    )
    acknowledged: bool = Field(
        default=False,
        description="Alert acknowledged"
    )


class ASMEPTC125Result(BaseModel):
    """ASME PTC 12.5 compliance test results."""

    test_compliant: TestCompliance = Field(
        ...,
        description="Overall compliance status"
    )
    test_date: datetime = Field(
        ...,
        description="Test date"
    )

    # Uncertainty analysis
    duty_uncertainty_percent: float = Field(
        ...,
        ge=0,
        description="Heat duty uncertainty (%)"
    )
    u_uncertainty_percent: float = Field(
        ...,
        ge=0,
        description="U value uncertainty (%)"
    )
    meets_uncertainty_requirements: bool = Field(
        default=True,
        description="Meets PTC 12.5 uncertainty requirements"
    )

    # Test conditions
    steady_state_achieved: bool = Field(
        default=True,
        description="Steady state conditions achieved"
    )
    test_duration_hours: float = Field(
        ...,
        gt=0,
        description="Test duration"
    )
    data_points_collected: int = Field(
        ...,
        ge=1,
        description="Number of data points"
    )

    # Deviations
    deviations: List[str] = Field(
        default_factory=list,
        description="Deviations from standard"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Test recommendations"
    )

    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )


class HeatExchangerOutput(BaseModel):
    """
    Output from heat exchanger optimization analysis.

    This comprehensive output model contains all analysis results,
    performance assessments, predictions, and recommendations.

    Attributes:
        exchanger_id: Exchanger identifier
        health_status: Overall equipment health
        thermal_performance: Thermal analysis results
        fouling_analysis: Fouling analysis and predictions
        hydraulic_analysis: Pressure drop analysis
        tube_integrity: Tube condition assessment
        cleaning_recommendation: Cleaning recommendations
        economic_analysis: Economic analysis

    Example:
        >>> output = optimizer.process(input_data)
        >>> print(f"Effectiveness: {output.thermal_performance.thermal_effectiveness:.1%}")
        >>> if output.cleaning_recommendation.recommended:
        ...     print(f"Cleaning recommended in {output.cleaning_recommendation.days_until_recommended:.0f} days")
    """

    request_id: str = Field(..., description="Original request ID")
    exchanger_id: str = Field(..., description="Exchanger identifier")
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

    # Performance analysis
    thermal_performance: ThermalPerformanceResult = Field(
        ...,
        description="Thermal performance analysis"
    )
    fouling_analysis: FoulingAnalysisResult = Field(
        ...,
        description="Fouling analysis"
    )
    hydraulic_analysis: HydraulicAnalysisResult = Field(
        ...,
        description="Hydraulic analysis"
    )

    # Tube integrity (optional)
    tube_integrity: Optional[TubeIntegrityResult] = Field(
        default=None,
        description="Tube integrity analysis"
    )

    # Recommendations
    cleaning_recommendation: CleaningRecommendation = Field(
        ...,
        description="Cleaning recommendation"
    )

    # Economic analysis
    economic_analysis: EconomicAnalysisResult = Field(
        ...,
        description="Economic analysis"
    )

    # ASME PTC 12.5 (optional)
    asme_ptc_result: Optional[ASMEPTC125Result] = Field(
        default=None,
        description="ASME PTC 12.5 test results"
    )

    # Alerts
    active_alerts: List[Alert] = Field(
        default_factory=list,
        description="Active alerts"
    )
    alert_count_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Alert counts by severity"
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

HeatExchangerInput.update_forward_refs()
HeatExchangerOutput.update_forward_refs()
