# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC API Schemas

Pydantic v2 request/response models for Condenser Optimization Agent REST API.

This module defines all data models for:
- Diagnostic requests and responses
- Vacuum optimization endpoints
- Fouling prediction models
- Cleaning recommendation schemas
- Health and status monitoring
- KPI tracking and metrics

All models include:
- Comprehensive field validation
- JSON Schema examples
- Provenance tracking fields
- Correlation ID support

Author: GL-APIDeveloper
Date: December 2025
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CondenserType(str, Enum):
    """Types of condensers supported by Condensync."""
    SURFACE = "surface"
    DIRECT_CONTACT = "direct_contact"
    AIR_COOLED = "air_cooled"
    EVAPORATIVE = "evaporative"
    SHELL_AND_TUBE = "shell_and_tube"
    PLATE = "plate"


class ConditionStatus(str, Enum):
    """Condenser operating condition status."""
    OPTIMAL = "optimal"
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    """Severity levels for issues and alerts."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertLevel(str, Enum):
    """Alert notification levels."""
    NONE = "none"
    ADVISORY = "advisory"
    WARNING = "warning"
    ALARM = "alarm"
    EMERGENCY = "emergency"


class FoulingType(str, Enum):
    """Types of condenser fouling."""
    BIOLOGICAL = "biological"
    SCALING = "scaling"
    CORROSION = "corrosion"
    PARTICULATE = "particulate"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CleaningMethod(str, Enum):
    """Condenser cleaning methods."""
    MECHANICAL_BRUSH = "mechanical_brush"
    CHEMICAL_TREATMENT = "chemical_treatment"
    HIGH_PRESSURE_WATER = "high_pressure_water"
    BACKFLUSHING = "backflushing"
    BALL_CLEANING = "ball_cleaning"
    ULTRASONIC = "ultrasonic"
    THERMAL_SHOCK = "thermal_shock"


class OptimizationMode(str, Enum):
    """Vacuum optimization modes."""
    EFFICIENCY = "efficiency"
    COST = "cost"
    EMISSIONS = "emissions"
    BALANCED = "balanced"
    PERFORMANCE = "performance"


class MetricTimeRange(str, Enum):
    """Time ranges for metric queries."""
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"
    YEAR = "365d"


class HealthStatus(str, Enum):
    """System health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# BASE MODELS
# =============================================================================

class BaseRequest(BaseModel):
    """Base request model with correlation tracking."""

    correlation_id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique correlation ID for request tracking",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    request_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp in UTC"
    )
    client_id: Optional[str] = Field(
        default=None,
        description="Client application identifier"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "request_timestamp": "2025-12-30T10:00:00Z",
                    "client_id": "condensync-dashboard-v1"
                }
            ]
        }
    }


class BaseResponse(BaseModel):
    """Base response model with provenance tracking."""

    correlation_id: str = Field(
        ...,
        description="Correlation ID from request"
    )
    response_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp in UTC"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    agent_id: str = Field(
        default="GL-017",
        description="Agent identifier"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail and reproducibility",
        min_length=64,
        max_length=64
    )


class ProvenanceMetadata(BaseModel):
    """Provenance metadata for audit trails."""

    computation_hash: str = Field(
        ...,
        description="SHA-256 hash of computation",
        min_length=64,
        max_length=64
    )
    inputs_hash: str = Field(
        ...,
        description="SHA-256 hash of inputs",
        min_length=64,
        max_length=64
    )
    algorithm_version: str = Field(
        ...,
        description="Version of algorithm used"
    )
    timestamp: datetime = Field(
        ...,
        description="Computation timestamp UTC"
    )
    deterministic: bool = Field(
        default=True,
        description="Whether computation is deterministic"
    )


# =============================================================================
# CONDENSER DATA MODELS
# =============================================================================

class CondenserOperatingData(BaseModel):
    """Real-time condenser operating data."""

    condenser_id: str = Field(
        ...,
        description="Unique condenser identifier",
        min_length=1,
        max_length=100,
        examples=["COND-001"]
    )
    condenser_type: CondenserType = Field(
        default=CondenserType.SURFACE,
        description="Type of condenser"
    )

    # Temperature data
    steam_inlet_temp_c: float = Field(
        ...,
        ge=0,
        le=200,
        description="Steam inlet temperature in Celsius"
    )
    hotwell_temp_c: float = Field(
        ...,
        ge=0,
        le=100,
        description="Hotwell temperature in Celsius"
    )
    cooling_water_inlet_temp_c: float = Field(
        ...,
        ge=0,
        le=50,
        description="Cooling water inlet temperature in Celsius"
    )
    cooling_water_outlet_temp_c: float = Field(
        ...,
        ge=0,
        le=60,
        description="Cooling water outlet temperature in Celsius"
    )

    # Pressure data
    condenser_vacuum_mbar_abs: float = Field(
        ...,
        ge=0,
        le=200,
        description="Condenser vacuum in mbar absolute"
    )
    design_vacuum_mbar_abs: float = Field(
        default=50.0,
        ge=0,
        le=200,
        description="Design condenser vacuum in mbar absolute"
    )

    # Flow data
    cooling_water_flow_m3h: float = Field(
        ...,
        ge=0,
        description="Cooling water flow rate in m3/h"
    )
    steam_flow_kg_h: float = Field(
        ...,
        ge=0,
        description="Steam flow rate to condenser in kg/h"
    )

    # Air ingress data
    air_ingress_kg_h: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated air ingress rate in kg/h"
    )

    # Equipment data
    tube_count: int = Field(
        default=5000,
        ge=100,
        description="Number of condenser tubes"
    )
    tube_length_m: float = Field(
        default=10.0,
        ge=1,
        description="Tube length in meters"
    )
    tube_od_mm: float = Field(
        default=25.4,
        ge=10,
        le=50,
        description="Tube outer diameter in mm"
    )
    tube_material: str = Field(
        default="titanium",
        description="Tube material"
    )
    surface_area_m2: Optional[float] = Field(
        default=None,
        ge=100,
        description="Total heat transfer surface area in m2"
    )

    @field_validator("cooling_water_outlet_temp_c")
    @classmethod
    def validate_outlet_temp(cls, v: float, info) -> float:
        """Validate outlet temp is greater than inlet temp."""
        inlet_temp = info.data.get("cooling_water_inlet_temp_c")
        if inlet_temp is not None and v <= inlet_temp:
            raise ValueError("Cooling water outlet temp must be greater than inlet temp")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "condenser_id": "COND-001",
                    "condenser_type": "surface",
                    "steam_inlet_temp_c": 45.0,
                    "hotwell_temp_c": 35.0,
                    "cooling_water_inlet_temp_c": 20.0,
                    "cooling_water_outlet_temp_c": 30.0,
                    "condenser_vacuum_mbar_abs": 55.0,
                    "design_vacuum_mbar_abs": 50.0,
                    "cooling_water_flow_m3h": 15000.0,
                    "steam_flow_kg_h": 100000.0,
                    "tube_count": 8000,
                    "tube_length_m": 12.0,
                    "tube_od_mm": 25.4,
                    "tube_material": "titanium"
                }
            ]
        }
    }


class CondenserHistoricalData(BaseModel):
    """Historical data point for trend analysis."""

    timestamp: datetime = Field(
        ...,
        description="Data point timestamp UTC"
    )
    condenser_vacuum_mbar_abs: float = Field(
        ...,
        ge=0,
        le=200,
        description="Condenser vacuum in mbar absolute"
    )
    cooling_water_inlet_temp_c: float = Field(
        ...,
        ge=0,
        le=50,
        description="Cooling water inlet temperature"
    )
    steam_flow_kg_h: float = Field(
        ...,
        ge=0,
        description="Steam flow rate"
    )
    heat_transfer_coefficient_w_m2k: Optional[float] = Field(
        default=None,
        ge=0,
        description="Calculated heat transfer coefficient"
    )
    cleanliness_factor: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Cleanliness factor (0-1)"
    )


# =============================================================================
# DIAGNOSTIC SCHEMAS
# =============================================================================

class DiagnosticRequest(BaseRequest):
    """Request model for condenser diagnostic analysis."""

    operating_data: CondenserOperatingData = Field(
        ...,
        description="Current condenser operating data"
    )
    historical_data: Optional[List[CondenserHistoricalData]] = Field(
        default=None,
        description="Historical data for trend analysis (last 30 days recommended)",
        max_length=1000
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include optimization recommendations in response"
    )
    include_explanation: bool = Field(
        default=True,
        description="Include detailed calculation explanations"
    )
    sensitivity_analysis: bool = Field(
        default=False,
        description="Perform sensitivity analysis on key parameters"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "correlation_id": "diag-001",
                    "operating_data": {
                        "condenser_id": "COND-001",
                        "steam_inlet_temp_c": 45.0,
                        "hotwell_temp_c": 35.0,
                        "cooling_water_inlet_temp_c": 20.0,
                        "cooling_water_outlet_temp_c": 30.0,
                        "condenser_vacuum_mbar_abs": 55.0,
                        "cooling_water_flow_m3h": 15000.0,
                        "steam_flow_kg_h": 100000.0
                    },
                    "include_recommendations": True,
                    "include_explanation": True
                }
            ]
        }
    }


class PerformanceMetrics(BaseModel):
    """Calculated performance metrics for condenser."""

    actual_vacuum_mbar_abs: float = Field(
        ...,
        description="Actual condenser vacuum"
    )
    expected_vacuum_mbar_abs: float = Field(
        ...,
        description="Expected vacuum based on conditions"
    )
    vacuum_deviation_mbar: float = Field(
        ...,
        description="Deviation from expected vacuum"
    )
    vacuum_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Vacuum efficiency percentage"
    )
    heat_transfer_coefficient_w_m2k: float = Field(
        ...,
        ge=0,
        description="Overall heat transfer coefficient"
    )
    design_heat_transfer_coefficient_w_m2k: float = Field(
        ...,
        ge=0,
        description="Design heat transfer coefficient"
    )
    cleanliness_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="Cleanliness factor (1.0 = clean)"
    )
    terminal_temperature_difference_c: float = Field(
        ...,
        description="TTD in Celsius"
    )
    subcooling_c: float = Field(
        ...,
        description="Condensate subcooling in Celsius"
    )
    log_mean_temp_diff_c: float = Field(
        ...,
        description="Log mean temperature difference"
    )
    heat_duty_mw: float = Field(
        ...,
        ge=0,
        description="Heat duty in MW"
    )


class EnergyImpact(BaseModel):
    """Energy and cost impact assessment."""

    power_loss_mw: float = Field(
        ...,
        ge=0,
        description="Turbine power loss due to poor vacuum in MW"
    )
    annual_energy_loss_mwh: float = Field(
        ...,
        ge=0,
        description="Annual energy loss in MWh"
    )
    annual_cost_usd: float = Field(
        ...,
        ge=0,
        description="Annual cost impact in USD"
    )
    annual_co2_tonnes: float = Field(
        ...,
        ge=0,
        description="Annual CO2 emissions impact in tonnes"
    )
    efficiency_loss_pct: float = Field(
        ...,
        ge=0,
        description="Plant efficiency loss percentage"
    )
    specific_fuel_increase_pct: float = Field(
        ...,
        ge=0,
        description="Specific fuel consumption increase percentage"
    )


class DiagnosticIssue(BaseModel):
    """Identified diagnostic issue."""

    issue_id: str = Field(
        ...,
        description="Unique issue identifier"
    )
    issue_type: str = Field(
        ...,
        description="Type of issue identified"
    )
    description: str = Field(
        ...,
        description="Detailed issue description"
    )
    severity: SeverityLevel = Field(
        ...,
        description="Issue severity level"
    )
    affected_parameter: str = Field(
        ...,
        description="Primary parameter affected"
    )
    deviation_pct: float = Field(
        ...,
        description="Deviation from normal as percentage"
    )
    recommended_action: str = Field(
        ...,
        description="Recommended corrective action"
    )


class DiagnosticResponse(BaseResponse):
    """Response model for condenser diagnostic analysis."""

    condenser_id: str = Field(
        ...,
        description="Condenser identifier"
    )
    condition_status: ConditionStatus = Field(
        ...,
        description="Overall condenser condition"
    )
    alert_level: AlertLevel = Field(
        ...,
        description="Alert level for operations"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score of diagnosis"
    )
    performance_metrics: PerformanceMetrics = Field(
        ...,
        description="Calculated performance metrics"
    )
    energy_impact: EnergyImpact = Field(
        ...,
        description="Energy and cost impact"
    )
    issues_identified: List[DiagnosticIssue] = Field(
        default_factory=list,
        description="List of identified issues"
    )
    explanation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed calculation explanation"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Optimization recommendations"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "correlation_id": "diag-001",
                    "response_timestamp": "2025-12-30T10:00:05Z",
                    "processing_time_ms": 125.5,
                    "agent_id": "GL-017",
                    "agent_version": "1.0.0",
                    "provenance_hash": "a" * 64,
                    "condenser_id": "COND-001",
                    "condition_status": "degraded",
                    "alert_level": "warning",
                    "confidence_score": 0.92,
                    "performance_metrics": {
                        "actual_vacuum_mbar_abs": 55.0,
                        "expected_vacuum_mbar_abs": 50.0,
                        "vacuum_deviation_mbar": 5.0,
                        "vacuum_efficiency_pct": 91.0,
                        "heat_transfer_coefficient_w_m2k": 2800.0,
                        "design_heat_transfer_coefficient_w_m2k": 3500.0,
                        "cleanliness_factor": 0.80,
                        "terminal_temperature_difference_c": 3.5,
                        "subcooling_c": 2.0,
                        "log_mean_temp_diff_c": 12.5,
                        "heat_duty_mw": 250.0
                    },
                    "energy_impact": {
                        "power_loss_mw": 2.5,
                        "annual_energy_loss_mwh": 21900.0,
                        "annual_cost_usd": 1095000.0,
                        "annual_co2_tonnes": 8760.0,
                        "efficiency_loss_pct": 0.5,
                        "specific_fuel_increase_pct": 0.6
                    },
                    "issues_identified": [
                        {
                            "issue_id": "ISSUE-001",
                            "issue_type": "fouling",
                            "description": "Tube fouling detected",
                            "severity": "medium",
                            "affected_parameter": "heat_transfer_coefficient",
                            "deviation_pct": 20.0,
                            "recommended_action": "Schedule tube cleaning"
                        }
                    ]
                }
            ]
        }
    }


# =============================================================================
# VACUUM OPTIMIZATION SCHEMAS
# =============================================================================

class VacuumOptimizationRequest(BaseRequest):
    """Request model for vacuum optimization analysis."""

    operating_data: CondenserOperatingData = Field(
        ...,
        description="Current condenser operating data"
    )
    optimization_mode: OptimizationMode = Field(
        default=OptimizationMode.BALANCED,
        description="Optimization objective mode"
    )
    target_vacuum_mbar_abs: Optional[float] = Field(
        default=None,
        ge=20,
        le=100,
        description="Target vacuum if specific value desired"
    )

    # Constraints
    max_cooling_water_flow_m3h: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum available cooling water flow"
    )
    min_cooling_water_flow_m3h: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum required cooling water flow"
    )
    electricity_cost_usd_kwh: float = Field(
        default=0.08,
        ge=0,
        description="Electricity cost for pump power"
    )
    co2_factor_kg_kwh: float = Field(
        default=0.4,
        ge=0,
        description="CO2 emission factor for electricity"
    )

    # Analysis options
    include_sensitivity: bool = Field(
        default=True,
        description="Include sensitivity analysis"
    )
    include_pareto: bool = Field(
        default=False,
        description="Include Pareto frontier for multi-objective"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "correlation_id": "vac-opt-001",
                    "operating_data": {
                        "condenser_id": "COND-001",
                        "steam_inlet_temp_c": 45.0,
                        "hotwell_temp_c": 35.0,
                        "cooling_water_inlet_temp_c": 20.0,
                        "cooling_water_outlet_temp_c": 30.0,
                        "condenser_vacuum_mbar_abs": 55.0,
                        "cooling_water_flow_m3h": 15000.0,
                        "steam_flow_kg_h": 100000.0
                    },
                    "optimization_mode": "balanced",
                    "electricity_cost_usd_kwh": 0.08
                }
            ]
        }
    }


class OptimizationSetpoint(BaseModel):
    """Optimized setpoint recommendation."""

    parameter_name: str = Field(
        ...,
        description="Parameter name"
    )
    current_value: float = Field(
        ...,
        description="Current value"
    )
    optimal_value: float = Field(
        ...,
        description="Recommended optimal value"
    )
    unit: str = Field(
        ...,
        description="Engineering unit"
    )
    change_pct: float = Field(
        ...,
        description="Percentage change required"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in recommendation"
    )


class OptimizationBenefit(BaseModel):
    """Benefits from optimization."""

    vacuum_improvement_mbar: float = Field(
        ...,
        description="Expected vacuum improvement in mbar"
    )
    power_gain_mw: float = Field(
        ...,
        ge=0,
        description="Expected power output gain in MW"
    )
    annual_energy_gain_mwh: float = Field(
        ...,
        ge=0,
        description="Annual energy gain in MWh"
    )
    annual_savings_usd: float = Field(
        ...,
        ge=0,
        description="Annual cost savings in USD"
    )
    annual_co2_reduction_tonnes: float = Field(
        ...,
        ge=0,
        description="Annual CO2 reduction in tonnes"
    )
    efficiency_improvement_pct: float = Field(
        ...,
        ge=0,
        description="Efficiency improvement percentage"
    )
    payback_period_months: Optional[float] = Field(
        default=None,
        ge=0,
        description="Payback period in months if investment required"
    )


class SensitivityResult(BaseModel):
    """Sensitivity analysis result for a parameter."""

    parameter_name: str = Field(
        ...,
        description="Parameter analyzed"
    )
    base_value: float = Field(
        ...,
        description="Base value used"
    )
    sensitivity_coefficient: float = Field(
        ...,
        description="Sensitivity coefficient (dOutput/dInput)"
    )
    impact_ranking: int = Field(
        ...,
        ge=1,
        description="Ranking by impact (1 = highest)"
    )
    variation_range: List[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="[min, max] variation tested"
    )
    output_range: List[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="[min, max] output observed"
    )


class VacuumOptimizationResponse(BaseResponse):
    """Response model for vacuum optimization analysis."""

    condenser_id: str = Field(
        ...,
        description="Condenser identifier"
    )
    optimization_mode: OptimizationMode = Field(
        ...,
        description="Mode used for optimization"
    )
    optimization_status: str = Field(
        ...,
        description="Status: optimal, improved, no_improvement, infeasible"
    )
    current_vacuum_mbar_abs: float = Field(
        ...,
        description="Current vacuum"
    )
    optimal_vacuum_mbar_abs: float = Field(
        ...,
        description="Achievable optimal vacuum"
    )
    setpoint_recommendations: List[OptimizationSetpoint] = Field(
        default_factory=list,
        description="Recommended setpoint changes"
    )
    expected_benefits: OptimizationBenefit = Field(
        ...,
        description="Expected benefits from optimization"
    )
    sensitivity_analysis: Optional[List[SensitivityResult]] = Field(
        default=None,
        description="Sensitivity analysis results"
    )
    constraints_active: List[str] = Field(
        default_factory=list,
        description="Active constraints at optimum"
    )
    implementation_notes: List[str] = Field(
        default_factory=list,
        description="Notes for implementation"
    )


# =============================================================================
# FOULING PREDICTION SCHEMAS
# =============================================================================

class FoulingPredictionRequest(BaseRequest):
    """Request model for fouling prediction analysis."""

    condenser_id: str = Field(
        ...,
        description="Condenser identifier"
    )
    historical_data: List[CondenserHistoricalData] = Field(
        ...,
        description="Historical operating data for trend analysis",
        min_length=10,
        max_length=10000
    )
    current_cleanliness_factor: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Current measured cleanliness factor"
    )
    last_cleaning_date: Optional[datetime] = Field(
        default=None,
        description="Date of last cleaning"
    )
    cooling_water_quality: Optional[Dict[str, float]] = Field(
        default=None,
        description="Water quality parameters (TDS, pH, etc.)"
    )
    prediction_horizon_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Prediction horizon in days"
    )
    cleanliness_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=0.95,
        description="Cleanliness factor threshold for cleaning trigger"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "correlation_id": "foul-pred-001",
                    "condenser_id": "COND-001",
                    "historical_data": [
                        {
                            "timestamp": "2025-12-01T00:00:00Z",
                            "condenser_vacuum_mbar_abs": 52.0,
                            "cooling_water_inlet_temp_c": 18.0,
                            "steam_flow_kg_h": 95000.0,
                            "cleanliness_factor": 0.95
                        }
                    ],
                    "prediction_horizon_days": 90,
                    "cleanliness_threshold": 0.85
                }
            ]
        }
    }


class FoulingTrendPoint(BaseModel):
    """Point in fouling trend prediction."""

    date: datetime = Field(
        ...,
        description="Prediction date"
    )
    predicted_cleanliness_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="Predicted cleanliness factor"
    )
    confidence_lower: float = Field(
        ...,
        ge=0,
        le=1,
        description="Lower confidence bound"
    )
    confidence_upper: float = Field(
        ...,
        ge=0,
        le=1,
        description="Upper confidence bound"
    )
    predicted_vacuum_mbar_abs: float = Field(
        ...,
        ge=0,
        description="Predicted vacuum at this fouling level"
    )


class FoulingPredictionResponse(BaseResponse):
    """Response model for fouling prediction analysis."""

    condenser_id: str = Field(
        ...,
        description="Condenser identifier"
    )
    current_cleanliness_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current estimated cleanliness factor"
    )
    fouling_type: FoulingType = Field(
        ...,
        description="Identified fouling type"
    )
    fouling_rate_per_day: float = Field(
        ...,
        description="Fouling rate (cleanliness factor loss per day)"
    )
    days_to_threshold: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days until cleaning threshold reached"
    )
    recommended_cleaning_date: Optional[datetime] = Field(
        default=None,
        description="Recommended cleaning date"
    )
    prediction_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall prediction confidence"
    )
    fouling_trend: List[FoulingTrendPoint] = Field(
        default_factory=list,
        description="Predicted fouling trend"
    )
    cumulative_energy_loss_mwh: float = Field(
        ...,
        ge=0,
        description="Cumulative energy loss until threshold"
    )
    cumulative_cost_usd: float = Field(
        ...,
        ge=0,
        description="Cumulative cost impact until threshold"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Identified risk factors for fouling"
    )


# =============================================================================
# CLEANING RECOMMENDATION SCHEMAS
# =============================================================================

class CleaningRecommendationRequest(BaseRequest):
    """Request model for cleaning recommendation analysis."""

    condenser_id: str = Field(
        ...,
        description="Condenser identifier"
    )
    current_cleanliness_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current cleanliness factor"
    )
    fouling_type: FoulingType = Field(
        default=FoulingType.UNKNOWN,
        description="Type of fouling if known"
    )
    condenser_type: CondenserType = Field(
        default=CondenserType.SURFACE,
        description="Type of condenser"
    )
    tube_material: str = Field(
        default="titanium",
        description="Tube material"
    )
    tube_count: int = Field(
        default=5000,
        ge=100,
        description="Number of tubes"
    )
    available_methods: Optional[List[CleaningMethod]] = Field(
        default=None,
        description="Available cleaning methods at site"
    )
    outage_cost_usd_hour: float = Field(
        default=50000.0,
        ge=0,
        description="Cost of plant outage per hour"
    )
    max_outage_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum acceptable outage duration"
    )
    online_cleaning_available: bool = Field(
        default=True,
        description="Whether online cleaning is possible"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "correlation_id": "clean-rec-001",
                    "condenser_id": "COND-001",
                    "current_cleanliness_factor": 0.78,
                    "fouling_type": "biological",
                    "tube_material": "titanium",
                    "tube_count": 8000,
                    "outage_cost_usd_hour": 75000.0,
                    "online_cleaning_available": True
                }
            ]
        }
    }


class CleaningMethodRecommendation(BaseModel):
    """Recommendation for a specific cleaning method."""

    method: CleaningMethod = Field(
        ...,
        description="Cleaning method"
    )
    effectiveness_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Expected effectiveness percentage"
    )
    expected_cleanliness_after: float = Field(
        ...,
        ge=0,
        le=1,
        description="Expected cleanliness factor after cleaning"
    )
    duration_hours: float = Field(
        ...,
        ge=0,
        description="Expected duration in hours"
    )
    requires_outage: bool = Field(
        ...,
        description="Whether method requires plant outage"
    )
    estimated_cost_usd: float = Field(
        ...,
        ge=0,
        description="Estimated cleaning cost in USD"
    )
    total_cost_usd: float = Field(
        ...,
        ge=0,
        description="Total cost including outage"
    )
    risk_level: SeverityLevel = Field(
        ...,
        description="Risk level of procedure"
    )
    suitability_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Suitability score for this application"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Implementation notes"
    )


class CleaningRecommendationResponse(BaseResponse):
    """Response model for cleaning recommendation analysis."""

    condenser_id: str = Field(
        ...,
        description="Condenser identifier"
    )
    cleaning_urgency: SeverityLevel = Field(
        ...,
        description="Urgency level for cleaning"
    )
    recommended_method: CleaningMethod = Field(
        ...,
        description="Primary recommended cleaning method"
    )
    all_method_recommendations: List[CleaningMethodRecommendation] = Field(
        default_factory=list,
        description="All evaluated cleaning methods ranked"
    )
    expected_benefits: OptimizationBenefit = Field(
        ...,
        description="Expected benefits from cleaning"
    )
    optimal_cleaning_window: Optional[str] = Field(
        default=None,
        description="Recommended timing for cleaning"
    )
    pre_cleaning_actions: List[str] = Field(
        default_factory=list,
        description="Actions to take before cleaning"
    )
    post_cleaning_monitoring: List[str] = Field(
        default_factory=list,
        description="Monitoring recommendations after cleaning"
    )


# =============================================================================
# HEALTH AND STATUS SCHEMAS
# =============================================================================

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: HealthStatus = Field(
        ...,
        description="Overall health status"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp"
    )
    agent_id: str = Field(
        default="GL-017",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="CONDENSYNC",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    uptime_seconds: float = Field(
        ...,
        ge=0,
        description="Agent uptime in seconds"
    )
    checks: Dict[str, HealthStatus] = Field(
        default_factory=dict,
        description="Individual component health checks"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2025-12-30T10:00:00Z",
                    "agent_id": "GL-017",
                    "agent_name": "CONDENSYNC",
                    "version": "1.0.0",
                    "uptime_seconds": 86400.0,
                    "checks": {
                        "calculators": "healthy",
                        "diagnostics": "healthy",
                        "optimization": "healthy"
                    }
                }
            ]
        }
    }


class ComponentStatus(BaseModel):
    """Status of an individual component."""

    name: str = Field(
        ...,
        description="Component name"
    )
    status: HealthStatus = Field(
        ...,
        description="Component status"
    )
    last_used: Optional[datetime] = Field(
        default=None,
        description="Last time component was used"
    )
    call_count: int = Field(
        default=0,
        ge=0,
        description="Number of calls to component"
    )
    avg_latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Average latency in milliseconds"
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors"
    )


class StatusResponse(BaseModel):
    """Response model for agent status endpoint."""

    agent_id: str = Field(
        default="GL-017",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="CONDENSYNC",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    status: HealthStatus = Field(
        ...,
        description="Overall status"
    )
    mode: str = Field(
        default="production",
        description="Operating mode"
    )
    started_at: datetime = Field(
        ...,
        description="Agent start timestamp"
    )
    uptime_seconds: float = Field(
        ...,
        ge=0,
        description="Uptime in seconds"
    )
    statistics: Dict[str, int] = Field(
        default_factory=dict,
        description="Request statistics by endpoint"
    )
    components: List[ComponentStatus] = Field(
        default_factory=list,
        description="Component statuses"
    )
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current configuration summary"
    )


# =============================================================================
# METRICS AND KPI SCHEMAS
# =============================================================================

class MetricsRequest(BaseRequest):
    """Request model for metrics endpoint."""

    time_range: MetricTimeRange = Field(
        default=MetricTimeRange.DAY,
        description="Time range for metrics"
    )
    condenser_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter by condenser IDs"
    )
    include_histograms: bool = Field(
        default=False,
        description="Include histogram data"
    )


class MetricValue(BaseModel):
    """A single metric value."""

    name: str = Field(
        ...,
        description="Metric name"
    )
    value: float = Field(
        ...,
        description="Metric value"
    )
    unit: str = Field(
        ...,
        description="Unit of measurement"
    )
    timestamp: datetime = Field(
        ...,
        description="Metric timestamp"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Metric labels"
    )


class MetricsResponse(BaseResponse):
    """Response model for metrics endpoint."""

    time_range: MetricTimeRange = Field(
        ...,
        description="Time range of metrics"
    )
    metrics: List[MetricValue] = Field(
        default_factory=list,
        description="Metric values"
    )
    summary: Dict[str, float] = Field(
        default_factory=dict,
        description="Summary statistics"
    )


class KPIValue(BaseModel):
    """A Key Performance Indicator value."""

    kpi_id: str = Field(
        ...,
        description="KPI identifier"
    )
    name: str = Field(
        ...,
        description="KPI display name"
    )
    value: float = Field(
        ...,
        description="Current KPI value"
    )
    target: Optional[float] = Field(
        default=None,
        description="Target value"
    )
    unit: str = Field(
        ...,
        description="Unit of measurement"
    )
    trend: str = Field(
        default="stable",
        description="Trend: improving, stable, declining"
    )
    status: ConditionStatus = Field(
        ...,
        description="KPI status vs target"
    )
    last_updated: datetime = Field(
        ...,
        description="Last update timestamp"
    )


class CurrentKPIsResponse(BaseResponse):
    """Response model for current KPIs endpoint."""

    kpis: List[KPIValue] = Field(
        default_factory=list,
        description="Current KPI values"
    )
    overall_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall performance score"
    )
    improvement_opportunities: List[str] = Field(
        default_factory=list,
        description="Identified improvement opportunities"
    )


class HistoricalKPIPoint(BaseModel):
    """Historical KPI data point."""

    timestamp: datetime = Field(
        ...,
        description="Data point timestamp"
    )
    value: float = Field(
        ...,
        description="KPI value at timestamp"
    )


class HistoricalKPI(BaseModel):
    """Historical data for a single KPI."""

    kpi_id: str = Field(
        ...,
        description="KPI identifier"
    )
    name: str = Field(
        ...,
        description="KPI display name"
    )
    unit: str = Field(
        ...,
        description="Unit of measurement"
    )
    data_points: List[HistoricalKPIPoint] = Field(
        default_factory=list,
        description="Historical data points"
    )
    statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Statistics (min, max, avg, std)"
    )


class HistoricalKPIsRequest(BaseRequest):
    """Request model for historical KPIs endpoint."""

    kpi_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter by KPI IDs"
    )
    time_range: MetricTimeRange = Field(
        default=MetricTimeRange.WEEK,
        description="Time range for history"
    )
    aggregation_interval: str = Field(
        default="1h",
        description="Aggregation interval (1h, 6h, 1d)"
    )
    condenser_id: Optional[str] = Field(
        default=None,
        description="Filter by condenser ID"
    )


class HistoricalKPIsResponse(BaseResponse):
    """Response model for historical KPIs endpoint."""

    time_range: MetricTimeRange = Field(
        ...,
        description="Time range of data"
    )
    start_time: datetime = Field(
        ...,
        description="Start of time range"
    )
    end_time: datetime = Field(
        ...,
        description="End of time range"
    )
    kpis: List[HistoricalKPI] = Field(
        default_factory=list,
        description="Historical KPI data"
    )
    data_quality: float = Field(
        ...,
        ge=0,
        le=1,
        description="Data quality score (1.0 = complete)"
    )


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: Optional[str] = Field(
        default=None,
        description="Field that caused error"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    code: str = Field(
        ...,
        description="Error code"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID if available"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "validation_error",
                    "message": "Invalid input data",
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2025-12-30T10:00:00Z",
                    "details": [
                        {
                            "field": "condenser_vacuum_mbar_abs",
                            "message": "Value must be between 0 and 200",
                            "code": "value_out_of_range"
                        }
                    ]
                }
            ]
        }
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enumerations
    "CondenserType",
    "ConditionStatus",
    "SeverityLevel",
    "AlertLevel",
    "FoulingType",
    "CleaningMethod",
    "OptimizationMode",
    "MetricTimeRange",
    "HealthStatus",
    # Base models
    "BaseRequest",
    "BaseResponse",
    "ProvenanceMetadata",
    # Condenser data
    "CondenserOperatingData",
    "CondenserHistoricalData",
    # Diagnostic
    "DiagnosticRequest",
    "DiagnosticResponse",
    "PerformanceMetrics",
    "EnergyImpact",
    "DiagnosticIssue",
    # Vacuum optimization
    "VacuumOptimizationRequest",
    "VacuumOptimizationResponse",
    "OptimizationSetpoint",
    "OptimizationBenefit",
    "SensitivityResult",
    # Fouling prediction
    "FoulingPredictionRequest",
    "FoulingPredictionResponse",
    "FoulingTrendPoint",
    # Cleaning recommendation
    "CleaningRecommendationRequest",
    "CleaningRecommendationResponse",
    "CleaningMethodRecommendation",
    # Health and status
    "HealthResponse",
    "StatusResponse",
    "ComponentStatus",
    # Metrics and KPIs
    "MetricsRequest",
    "MetricsResponse",
    "MetricValue",
    "CurrentKPIsResponse",
    "KPIValue",
    "HistoricalKPIsRequest",
    "HistoricalKPIsResponse",
    "HistoricalKPI",
    "HistoricalKPIPoint",
    # Error handling
    "ErrorResponse",
    "ErrorDetail",
]
