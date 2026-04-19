"""
GL-020 ECONOPULSE Pydantic Schemas

Request and response models for all API endpoints.
Includes validation rules and OpenAPI documentation examples.

Agent ID: GL-020
Codename: ECONOPULSE
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


# =============================================================================
# Enumerations
# =============================================================================

class EconomizerStatus(str, Enum):
    """Economizer operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class EconomizerType(str, Enum):
    """Type of economizer."""
    BARE_TUBE = "bare_tube"
    FINNED_TUBE = "finned_tube"
    CAST_IRON = "cast_iron"
    CONDENSING = "condensing"
    NON_CONDENSING = "non_condensing"


class FoulingSeverity(str, Enum):
    """Fouling severity levels."""
    CLEAN = "clean"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Alert status values."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertType(str, Enum):
    """Types of alerts."""
    FOULING = "fouling"
    EFFICIENCY = "efficiency"
    TEMPERATURE = "temperature"
    PRESSURE_DROP = "pressure_drop"
    FLOW_RATE = "flow_rate"
    CLEANING_DUE = "cleaning_due"
    SENSOR_FAULT = "sensor_fault"


class SootBlowerStatus(str, Enum):
    """Soot blower operational status."""
    IDLE = "idle"
    OPERATING = "operating"
    COOLDOWN = "cooldown"
    FAULT = "fault"
    DISABLED = "disabled"


class ReportFormat(str, Enum):
    """Report export formats."""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"


class TrendDirection(str, Enum):
    """Trend direction indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


# =============================================================================
# Health Check Schemas
# =============================================================================

class HealthStatus(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-11-09T10:30:00Z",
                "service": "gl-020-econopulse",
                "version": "1.0.0"
            }
        }


class ReadinessStatus(BaseModel):
    """Readiness check response model."""
    status: str = Field(..., description="Readiness status")
    timestamp: datetime = Field(..., description="Check timestamp")
    checks: Dict[str, bool] = Field(..., description="Dependency check results")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ready",
                "timestamp": "2025-11-09T10:30:00Z",
                "checks": {
                    "database": True,
                    "redis": True,
                    "historian": True,
                    "message_queue": True
                },
                "message": "All dependencies healthy"
            }
        }


# =============================================================================
# Economizer Schemas
# =============================================================================

class EconomizerBase(BaseModel):
    """Base economizer model."""
    name: str = Field(..., min_length=1, max_length=100, description="Economizer name")
    description: Optional[str] = Field(None, max_length=500, description="Description")
    type: EconomizerType = Field(..., description="Economizer type")
    location: str = Field(..., description="Physical location")
    boiler_id: str = Field(..., description="Associated boiler ID")
    design_capacity_kw: float = Field(..., gt=0, description="Design heat recovery capacity (kW)")
    design_pressure_drop_kpa: float = Field(..., gt=0, description="Design pressure drop (kPa)")
    surface_area_m2: float = Field(..., gt=0, description="Heat transfer surface area (m2)")


class EconomizerCreate(EconomizerBase):
    """Create economizer request model."""
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Metadata tags")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Economizer Unit A1",
                "description": "Primary economizer for Boiler 1",
                "type": "finned_tube",
                "location": "Building A, Level 2",
                "boiler_id": "boiler-001",
                "design_capacity_kw": 500.0,
                "design_pressure_drop_kpa": 1.5,
                "surface_area_m2": 150.0,
                "tags": {"plant": "main", "zone": "north"}
            }
        }


class EconomizerUpdate(BaseModel):
    """Update economizer request model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    location: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class Economizer(EconomizerBase):
    """Economizer response model."""
    id: str = Field(..., description="Unique identifier")
    status: EconomizerStatus = Field(..., description="Current operational status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_cleaned: Optional[datetime] = Field(None, description="Last cleaning timestamp")
    tags: Dict[str, str] = Field(default_factory=dict, description="Metadata tags")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "econ-001",
                "name": "Economizer Unit A1",
                "description": "Primary economizer for Boiler 1",
                "type": "finned_tube",
                "location": "Building A, Level 2",
                "boiler_id": "boiler-001",
                "design_capacity_kw": 500.0,
                "design_pressure_drop_kpa": 1.5,
                "surface_area_m2": 150.0,
                "status": "online",
                "created_at": "2025-01-15T08:00:00Z",
                "updated_at": "2025-11-09T10:30:00Z",
                "last_cleaned": "2025-10-15T14:00:00Z",
                "tags": {"plant": "main", "zone": "north"}
            }
        }


class EconomizerList(BaseModel):
    """List economizers response model."""
    items: List[Economizer] = Field(..., description="List of economizers")
    total: int = Field(..., ge=0, description="Total count")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


# =============================================================================
# Performance Schemas
# =============================================================================

class PerformanceMetrics(BaseModel):
    """Current performance metrics model."""
    economizer_id: str = Field(..., description="Economizer ID")
    timestamp: datetime = Field(..., description="Measurement timestamp")

    # Temperature measurements
    gas_inlet_temp_c: float = Field(..., description="Flue gas inlet temperature (C)")
    gas_outlet_temp_c: float = Field(..., description="Flue gas outlet temperature (C)")
    water_inlet_temp_c: float = Field(..., description="Feedwater inlet temperature (C)")
    water_outlet_temp_c: float = Field(..., description="Feedwater outlet temperature (C)")

    # Flow rates
    gas_flow_rate_kg_s: float = Field(..., ge=0, description="Flue gas mass flow rate (kg/s)")
    water_flow_rate_kg_s: float = Field(..., ge=0, description="Water mass flow rate (kg/s)")

    # Pressure measurements
    gas_pressure_drop_kpa: float = Field(..., ge=0, description="Flue gas pressure drop (kPa)")
    water_pressure_drop_kpa: float = Field(..., ge=0, description="Water side pressure drop (kPa)")

    # Calculated metrics
    heat_transfer_kw: float = Field(..., description="Current heat transfer rate (kW)")
    effectiveness_percent: float = Field(..., ge=0, le=100, description="Heat exchanger effectiveness (%)")
    overall_htc_w_m2k: float = Field(..., ge=0, description="Overall heat transfer coefficient (W/m2K)")
    approach_temp_c: float = Field(..., description="Approach temperature (C)")

    # Status
    data_quality: str = Field(..., description="Data quality indicator")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "timestamp": "2025-11-09T10:30:00Z",
                "gas_inlet_temp_c": 320.5,
                "gas_outlet_temp_c": 180.2,
                "water_inlet_temp_c": 105.0,
                "water_outlet_temp_c": 140.5,
                "gas_flow_rate_kg_s": 15.2,
                "water_flow_rate_kg_s": 8.5,
                "gas_pressure_drop_kpa": 1.8,
                "water_pressure_drop_kpa": 0.5,
                "heat_transfer_kw": 485.3,
                "effectiveness_percent": 78.5,
                "overall_htc_w_m2k": 45.2,
                "approach_temp_c": 39.7,
                "data_quality": "good"
            }
        }


class PerformanceDataPoint(BaseModel):
    """Single performance data point for history."""
    timestamp: datetime
    heat_transfer_kw: float
    effectiveness_percent: float
    overall_htc_w_m2k: float
    gas_pressure_drop_kpa: float


class PerformanceHistory(BaseModel):
    """Historical performance data model."""
    economizer_id: str = Field(..., description="Economizer ID")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    resolution: str = Field(..., description="Data resolution (1m, 5m, 1h, 1d)")
    data_points: List[PerformanceDataPoint] = Field(..., description="Performance data points")
    statistics: Dict[str, Any] = Field(..., description="Summary statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "start_time": "2025-11-08T00:00:00Z",
                "end_time": "2025-11-09T00:00:00Z",
                "resolution": "1h",
                "data_points": [
                    {
                        "timestamp": "2025-11-08T00:00:00Z",
                        "heat_transfer_kw": 480.5,
                        "effectiveness_percent": 77.8,
                        "overall_htc_w_m2k": 44.5,
                        "gas_pressure_drop_kpa": 1.75
                    }
                ],
                "statistics": {
                    "avg_heat_transfer_kw": 478.2,
                    "min_effectiveness_percent": 75.0,
                    "max_effectiveness_percent": 82.0,
                    "avg_pressure_drop_kpa": 1.78
                }
            }
        }


class PerformanceTrend(BaseModel):
    """Performance trend analysis model."""
    economizer_id: str = Field(..., description="Economizer ID")
    analysis_period_days: int = Field(..., ge=1, description="Analysis period in days")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")

    # Trend indicators
    effectiveness_trend: TrendDirection = Field(..., description="Effectiveness trend")
    htc_trend: TrendDirection = Field(..., description="Heat transfer coefficient trend")
    pressure_drop_trend: TrendDirection = Field(..., description="Pressure drop trend")

    # Trend values
    effectiveness_change_percent: float = Field(..., description="Effectiveness change (%)")
    htc_change_percent: float = Field(..., description="HTC change (%)")
    pressure_drop_change_percent: float = Field(..., description="Pressure drop change (%)")

    # Predictions
    days_until_intervention: Optional[int] = Field(None, description="Estimated days until cleaning needed")
    confidence_percent: float = Field(..., ge=0, le=100, description="Prediction confidence (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "analysis_period_days": 30,
                "analyzed_at": "2025-11-09T10:30:00Z",
                "effectiveness_trend": "degrading",
                "htc_trend": "degrading",
                "pressure_drop_trend": "degrading",
                "effectiveness_change_percent": -5.2,
                "htc_change_percent": -8.1,
                "pressure_drop_change_percent": 12.5,
                "days_until_intervention": 14,
                "confidence_percent": 85.0
            }
        }


# =============================================================================
# Fouling Schemas
# =============================================================================

class FoulingStatus(BaseModel):
    """Current fouling status model."""
    economizer_id: str = Field(..., description="Economizer ID")
    timestamp: datetime = Field(..., description="Assessment timestamp")
    severity: FoulingSeverity = Field(..., description="Fouling severity level")
    fouling_factor: float = Field(..., ge=0, description="Fouling factor (m2K/W)")
    fouling_score: float = Field(..., ge=0, le=100, description="Fouling score (0-100)")
    effectiveness_loss_percent: float = Field(..., ge=0, description="Effectiveness loss due to fouling (%)")
    pressure_drop_increase_percent: float = Field(..., ge=0, description="Pressure drop increase (%)")
    estimated_deposit_thickness_mm: float = Field(..., ge=0, description="Estimated deposit thickness (mm)")
    cleaning_recommended: bool = Field(..., description="Cleaning recommended flag")
    last_cleaned: Optional[datetime] = Field(None, description="Last cleaning timestamp")
    days_since_cleaning: Optional[int] = Field(None, ge=0, description="Days since last cleaning")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "timestamp": "2025-11-09T10:30:00Z",
                "severity": "moderate",
                "fouling_factor": 0.00025,
                "fouling_score": 45.0,
                "effectiveness_loss_percent": 8.5,
                "pressure_drop_increase_percent": 15.2,
                "estimated_deposit_thickness_mm": 1.2,
                "cleaning_recommended": False,
                "last_cleaned": "2025-10-15T14:00:00Z",
                "days_since_cleaning": 25
            }
        }


class FoulingDataPoint(BaseModel):
    """Single fouling data point for history."""
    timestamp: datetime
    fouling_factor: float
    fouling_score: float
    severity: FoulingSeverity


class FoulingHistory(BaseModel):
    """Fouling history model."""
    economizer_id: str = Field(..., description="Economizer ID")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    data_points: List[FoulingDataPoint] = Field(..., description="Fouling data points")
    cleaning_events: List[datetime] = Field(..., description="Cleaning event timestamps")
    average_fouling_rate: float = Field(..., description="Average fouling rate per day")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "start_time": "2025-10-01T00:00:00Z",
                "end_time": "2025-11-09T00:00:00Z",
                "data_points": [
                    {
                        "timestamp": "2025-10-15T00:00:00Z",
                        "fouling_factor": 0.0001,
                        "fouling_score": 10.0,
                        "severity": "light"
                    }
                ],
                "cleaning_events": ["2025-10-15T14:00:00Z"],
                "average_fouling_rate": 0.000008
            }
        }


class FoulingPrediction(BaseModel):
    """Fouling prediction model."""
    economizer_id: str = Field(..., description="Economizer ID")
    predicted_at: datetime = Field(..., description="Prediction timestamp")
    current_fouling_score: float = Field(..., ge=0, le=100, description="Current fouling score")
    predicted_fouling_score_7d: float = Field(..., ge=0, le=100, description="Predicted score in 7 days")
    predicted_fouling_score_14d: float = Field(..., ge=0, le=100, description="Predicted score in 14 days")
    predicted_fouling_score_30d: float = Field(..., ge=0, le=100, description="Predicted score in 30 days")
    days_until_cleaning_threshold: int = Field(..., ge=0, description="Days until cleaning threshold")
    recommended_cleaning_date: datetime = Field(..., description="Recommended cleaning date")
    confidence_percent: float = Field(..., ge=0, le=100, description="Prediction confidence (%)")
    model_version: str = Field(..., description="Prediction model version")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "predicted_at": "2025-11-09T10:30:00Z",
                "current_fouling_score": 45.0,
                "predicted_fouling_score_7d": 52.0,
                "predicted_fouling_score_14d": 60.0,
                "predicted_fouling_score_30d": 75.0,
                "days_until_cleaning_threshold": 18,
                "recommended_cleaning_date": "2025-11-27T08:00:00Z",
                "confidence_percent": 82.5,
                "model_version": "v2.1.0"
            }
        }


class FoulingBaseline(BaseModel):
    """Fouling baseline request model."""
    economizer_id: str = Field(..., description="Economizer ID")
    baseline_type: str = Field(..., description="Baseline type: clean, as_found, custom")
    reference_date: Optional[datetime] = Field(None, description="Reference date for baseline")
    effectiveness_percent: Optional[float] = Field(None, ge=0, le=100, description="Baseline effectiveness (%)")
    pressure_drop_kpa: Optional[float] = Field(None, ge=0, description="Baseline pressure drop (kPa)")
    overall_htc_w_m2k: Optional[float] = Field(None, ge=0, description="Baseline HTC (W/m2K)")
    notes: Optional[str] = Field(None, max_length=500, description="Baseline notes")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "baseline_type": "clean",
                "reference_date": "2025-10-15T14:00:00Z",
                "effectiveness_percent": 85.0,
                "pressure_drop_kpa": 1.5,
                "overall_htc_w_m2k": 52.0,
                "notes": "Baseline set after chemical cleaning"
            }
        }


class FoulingBaselineResponse(FoulingBaseline):
    """Fouling baseline response model."""
    id: str = Field(..., description="Baseline ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    created_by: str = Field(..., description="Created by user ID")


# =============================================================================
# Alert Schemas
# =============================================================================

class AlertBase(BaseModel):
    """Base alert model."""
    economizer_id: str = Field(..., description="Economizer ID")
    alert_type: AlertType = Field(..., description="Alert type")
    severity: AlertSeverity = Field(..., description="Alert severity")
    title: str = Field(..., max_length=200, description="Alert title")
    message: str = Field(..., max_length=1000, description="Alert message")
    metric_name: str = Field(..., description="Triggering metric name")
    metric_value: float = Field(..., description="Triggering metric value")
    threshold_value: float = Field(..., description="Threshold value")


class Alert(AlertBase):
    """Alert response model."""
    id: str = Field(..., description="Alert ID")
    status: AlertStatus = Field(..., description="Alert status")
    triggered_at: datetime = Field(..., description="Alert trigger timestamp")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    acknowledged_by: Optional[str] = Field(None, description="Acknowledged by user ID")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolved_by: Optional[str] = Field(None, description="Resolved by user ID")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "alert-001",
                "economizer_id": "econ-001",
                "alert_type": "fouling",
                "severity": "warning",
                "status": "active",
                "title": "Elevated Fouling Detected",
                "message": "Fouling score has exceeded warning threshold of 50",
                "metric_name": "fouling_score",
                "metric_value": 55.2,
                "threshold_value": 50.0,
                "triggered_at": "2025-11-09T08:15:00Z",
                "acknowledged_at": None,
                "acknowledged_by": None,
                "resolved_at": None,
                "resolved_by": None,
                "resolution_notes": None
            }
        }


class AlertList(BaseModel):
    """List alerts response model."""
    items: List[Alert] = Field(..., description="List of alerts")
    total: int = Field(..., ge=0, description="Total count")
    active_count: int = Field(..., ge=0, description="Active alerts count")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")


class AlertAcknowledge(BaseModel):
    """Acknowledge alert request model."""
    notes: Optional[str] = Field(None, max_length=500, description="Acknowledgment notes")

    class Config:
        json_schema_extra = {
            "example": {
                "notes": "Acknowledged. Scheduling cleaning for next maintenance window."
            }
        }


class AlertThreshold(BaseModel):
    """Single alert threshold configuration."""
    metric_name: str = Field(..., description="Metric name")
    warning_threshold: float = Field(..., description="Warning threshold value")
    critical_threshold: float = Field(..., description="Critical threshold value")
    emergency_threshold: Optional[float] = Field(None, description="Emergency threshold value")
    enabled: bool = Field(True, description="Threshold enabled flag")
    hysteresis: float = Field(0.0, ge=0, description="Hysteresis value to prevent flapping")


class AlertThresholdConfig(BaseModel):
    """Alert threshold configuration request model."""
    economizer_id: Optional[str] = Field(None, description="Economizer ID (None for global)")
    thresholds: List[AlertThreshold] = Field(..., description="Threshold configurations")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "thresholds": [
                    {
                        "metric_name": "fouling_score",
                        "warning_threshold": 50.0,
                        "critical_threshold": 70.0,
                        "emergency_threshold": 85.0,
                        "enabled": True,
                        "hysteresis": 2.0
                    },
                    {
                        "metric_name": "effectiveness_percent",
                        "warning_threshold": 70.0,
                        "critical_threshold": 60.0,
                        "emergency_threshold": 50.0,
                        "enabled": True,
                        "hysteresis": 1.0
                    }
                ]
            }
        }


class AlertThresholdConfigResponse(AlertThresholdConfig):
    """Alert threshold configuration response model."""
    id: str = Field(..., description="Configuration ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# =============================================================================
# Efficiency Schemas
# =============================================================================

class EfficiencyMetrics(BaseModel):
    """Efficiency metrics model."""
    economizer_id: str = Field(..., description="Economizer ID")
    timestamp: datetime = Field(..., description="Measurement timestamp")

    # Current efficiency
    current_efficiency_percent: float = Field(..., ge=0, le=100, description="Current efficiency (%)")
    design_efficiency_percent: float = Field(..., ge=0, le=100, description="Design efficiency (%)")
    clean_efficiency_percent: float = Field(..., ge=0, le=100, description="Clean baseline efficiency (%)")

    # Efficiency breakdown
    thermal_efficiency_percent: float = Field(..., ge=0, le=100, description="Thermal efficiency (%)")
    heat_recovery_percent: float = Field(..., ge=0, le=100, description="Heat recovery rate (%)")

    # Comparison
    efficiency_vs_design_percent: float = Field(..., description="Efficiency vs design (%)")
    efficiency_vs_clean_percent: float = Field(..., description="Efficiency vs clean baseline (%)")

    # Energy metrics
    heat_recovered_kw: float = Field(..., ge=0, description="Heat recovered (kW)")
    potential_heat_recovery_kw: float = Field(..., ge=0, description="Potential heat recovery (kW)")
    heat_loss_kw: float = Field(..., ge=0, description="Heat loss due to inefficiency (kW)")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "timestamp": "2025-11-09T10:30:00Z",
                "current_efficiency_percent": 78.5,
                "design_efficiency_percent": 85.0,
                "clean_efficiency_percent": 83.0,
                "thermal_efficiency_percent": 80.2,
                "heat_recovery_percent": 76.8,
                "efficiency_vs_design_percent": -7.6,
                "efficiency_vs_clean_percent": -5.4,
                "heat_recovered_kw": 485.0,
                "potential_heat_recovery_kw": 520.0,
                "heat_loss_kw": 35.0
            }
        }


class EfficiencyLoss(BaseModel):
    """Efficiency loss quantification model."""
    economizer_id: str = Field(..., description="Economizer ID")
    analysis_period_start: datetime = Field(..., description="Analysis period start")
    analysis_period_end: datetime = Field(..., description="Analysis period end")

    # Loss breakdown
    total_efficiency_loss_percent: float = Field(..., description="Total efficiency loss (%)")
    fouling_loss_percent: float = Field(..., description="Loss due to fouling (%)")
    operational_loss_percent: float = Field(..., description="Loss due to off-design operation (%)")
    other_loss_percent: float = Field(..., description="Other losses (%)")

    # Energy impact
    energy_loss_kwh: float = Field(..., ge=0, description="Energy loss during period (kWh)")
    energy_loss_gj: float = Field(..., ge=0, description="Energy loss during period (GJ)")

    # Monetary impact
    fuel_cost_usd: float = Field(..., ge=0, description="Fuel cost of losses (USD)")
    carbon_emissions_kg: float = Field(..., ge=0, description="Additional CO2 emissions (kg)")

    # Trends
    loss_trend: TrendDirection = Field(..., description="Loss trend direction")
    loss_change_percent: float = Field(..., description="Loss change from previous period (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "analysis_period_start": "2025-11-01T00:00:00Z",
                "analysis_period_end": "2025-11-09T00:00:00Z",
                "total_efficiency_loss_percent": 7.6,
                "fouling_loss_percent": 5.2,
                "operational_loss_percent": 1.8,
                "other_loss_percent": 0.6,
                "energy_loss_kwh": 8520.0,
                "energy_loss_gj": 30.7,
                "fuel_cost_usd": 425.0,
                "carbon_emissions_kg": 1850.0,
                "loss_trend": "degrading",
                "loss_change_percent": 1.2
            }
        }


class EfficiencySavings(BaseModel):
    """Potential efficiency savings model."""
    economizer_id: str = Field(..., description="Economizer ID")
    calculated_at: datetime = Field(..., description="Calculation timestamp")

    # Current state
    current_efficiency_percent: float = Field(..., ge=0, le=100, description="Current efficiency (%)")
    target_efficiency_percent: float = Field(..., ge=0, le=100, description="Target efficiency after cleaning (%)")

    # Projected savings (annual)
    energy_savings_kwh_year: float = Field(..., ge=0, description="Annual energy savings (kWh)")
    energy_savings_gj_year: float = Field(..., ge=0, description="Annual energy savings (GJ)")
    fuel_savings_usd_year: float = Field(..., ge=0, description="Annual fuel cost savings (USD)")
    carbon_reduction_kg_year: float = Field(..., ge=0, description="Annual CO2 reduction (kg)")

    # Cleaning economics
    estimated_cleaning_cost_usd: float = Field(..., ge=0, description="Estimated cleaning cost (USD)")
    payback_period_days: float = Field(..., ge=0, description="Cleaning payback period (days)")
    roi_percent: float = Field(..., description="Return on investment (%)")

    # Recommendations
    cleaning_recommended: bool = Field(..., description="Cleaning recommended flag")
    optimal_cleaning_date: Optional[datetime] = Field(None, description="Optimal cleaning date")
    recommendation_notes: str = Field(..., description="Recommendation notes")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "calculated_at": "2025-11-09T10:30:00Z",
                "current_efficiency_percent": 78.5,
                "target_efficiency_percent": 83.0,
                "energy_savings_kwh_year": 45600.0,
                "energy_savings_gj_year": 164.2,
                "fuel_savings_usd_year": 2280.0,
                "carbon_reduction_kg_year": 9880.0,
                "estimated_cleaning_cost_usd": 500.0,
                "payback_period_days": 80.0,
                "roi_percent": 356.0,
                "cleaning_recommended": True,
                "optimal_cleaning_date": "2025-11-20T08:00:00Z",
                "recommendation_notes": "Cleaning recommended based on fouling level and economic analysis."
            }
        }


# =============================================================================
# Soot Blower Schemas
# =============================================================================

class SootBlower(BaseModel):
    """Soot blower model."""
    id: str = Field(..., description="Soot blower ID")
    name: str = Field(..., description="Soot blower name")
    type: str = Field(..., description="Soot blower type (steam, air, sonic)")
    status: SootBlowerStatus = Field(..., description="Operational status")
    economizer_id: str = Field(..., description="Associated economizer ID")
    position: str = Field(..., description="Position on economizer")
    last_operated: Optional[datetime] = Field(None, description="Last operation timestamp")
    operation_count: int = Field(0, ge=0, description="Total operation count")
    steam_consumption_kg_cycle: Optional[float] = Field(None, description="Steam consumption per cycle (kg)")


class SootBlowerStatusResponse(BaseModel):
    """Soot blower status response model."""
    economizer_id: str = Field(..., description="Economizer ID")
    timestamp: datetime = Field(..., description="Status timestamp")
    soot_blowers: List[SootBlower] = Field(..., description="Soot blower status list")
    total_count: int = Field(..., ge=0, description="Total soot blowers")
    available_count: int = Field(..., ge=0, description="Available for operation")
    operating_count: int = Field(..., ge=0, description="Currently operating")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "timestamp": "2025-11-09T10:30:00Z",
                "soot_blowers": [
                    {
                        "id": "sb-001",
                        "name": "Soot Blower 1",
                        "type": "steam",
                        "status": "idle",
                        "economizer_id": "econ-001",
                        "position": "inlet",
                        "last_operated": "2025-11-08T14:00:00Z",
                        "operation_count": 1250,
                        "steam_consumption_kg_cycle": 15.5
                    }
                ],
                "total_count": 4,
                "available_count": 4,
                "operating_count": 0
            }
        }


class SootBlowerTrigger(BaseModel):
    """Trigger soot blower request model."""
    soot_blower_ids: Optional[List[str]] = Field(None, description="Specific soot blower IDs (None for all)")
    sequence: str = Field("standard", description="Cleaning sequence: standard, intensive, custom")
    delay_seconds: int = Field(0, ge=0, le=3600, description="Delay before starting (seconds)")
    reason: str = Field(..., max_length=200, description="Trigger reason")

    class Config:
        json_schema_extra = {
            "example": {
                "soot_blower_ids": ["sb-001", "sb-002"],
                "sequence": "standard",
                "delay_seconds": 0,
                "reason": "Scheduled cleaning based on fouling level"
            }
        }


class SootBlowerTriggerResponse(BaseModel):
    """Trigger soot blower response model."""
    operation_id: str = Field(..., description="Operation ID for tracking")
    economizer_id: str = Field(..., description="Economizer ID")
    triggered_at: datetime = Field(..., description="Trigger timestamp")
    scheduled_start: datetime = Field(..., description="Scheduled start time")
    soot_blowers_triggered: List[str] = Field(..., description="Triggered soot blower IDs")
    sequence: str = Field(..., description="Cleaning sequence")
    estimated_duration_seconds: int = Field(..., ge=0, description="Estimated duration (seconds)")
    status: str = Field(..., description="Operation status")


class CleaningEvent(BaseModel):
    """Cleaning event model."""
    id: str = Field(..., description="Event ID")
    economizer_id: str = Field(..., description="Economizer ID")
    cleaning_type: str = Field(..., description="Cleaning type (soot_blower, chemical, manual)")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Duration (seconds)")
    soot_blowers_used: List[str] = Field(default_factory=list, description="Soot blowers used")
    effectiveness_before: float = Field(..., ge=0, le=100, description="Effectiveness before cleaning (%)")
    effectiveness_after: Optional[float] = Field(None, ge=0, le=100, description="Effectiveness after cleaning (%)")
    fouling_score_before: float = Field(..., ge=0, le=100, description="Fouling score before")
    fouling_score_after: Optional[float] = Field(None, ge=0, le=100, description="Fouling score after")
    success: Optional[bool] = Field(None, description="Cleaning success flag")
    notes: Optional[str] = Field(None, description="Cleaning notes")


class CleaningHistory(BaseModel):
    """Cleaning history response model."""
    economizer_id: str = Field(..., description="Economizer ID")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    events: List[CleaningEvent] = Field(..., description="Cleaning events")
    total_events: int = Field(..., ge=0, description="Total events in period")
    successful_events: int = Field(..., ge=0, description="Successful cleaning events")
    average_effectiveness_improvement: float = Field(..., description="Average effectiveness improvement (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "start_time": "2025-10-01T00:00:00Z",
                "end_time": "2025-11-09T00:00:00Z",
                "events": [
                    {
                        "id": "clean-001",
                        "economizer_id": "econ-001",
                        "cleaning_type": "soot_blower",
                        "started_at": "2025-10-15T14:00:00Z",
                        "completed_at": "2025-10-15T14:15:00Z",
                        "duration_seconds": 900,
                        "soot_blowers_used": ["sb-001", "sb-002", "sb-003", "sb-004"],
                        "effectiveness_before": 72.5,
                        "effectiveness_after": 82.0,
                        "fouling_score_before": 55.0,
                        "fouling_score_after": 15.0,
                        "success": True,
                        "notes": "Standard cleaning cycle completed"
                    }
                ],
                "total_events": 5,
                "successful_events": 5,
                "average_effectiveness_improvement": 8.5
            }
        }


class CleaningOptimization(BaseModel):
    """Cleaning optimization recommendations model."""
    economizer_id: str = Field(..., description="Economizer ID")
    calculated_at: datetime = Field(..., description="Calculation timestamp")

    # Current schedule analysis
    current_interval_hours: float = Field(..., ge=0, description="Current cleaning interval (hours)")
    optimal_interval_hours: float = Field(..., ge=0, description="Optimal cleaning interval (hours)")

    # Optimization metrics
    cleaning_efficiency_percent: float = Field(..., ge=0, le=100, description="Cleaning efficiency (%)")
    over_cleaning_risk: bool = Field(..., description="Over-cleaning risk flag")
    under_cleaning_risk: bool = Field(..., description="Under-cleaning risk flag")

    # Recommended schedule
    next_recommended_cleaning: datetime = Field(..., description="Next recommended cleaning")
    recommended_sequence: str = Field(..., description="Recommended cleaning sequence")
    recommended_soot_blowers: List[str] = Field(..., description="Recommended soot blowers to use")

    # Cost optimization
    current_annual_cleaning_cost_usd: float = Field(..., ge=0, description="Current annual cleaning cost (USD)")
    optimized_annual_cleaning_cost_usd: float = Field(..., ge=0, description="Optimized annual cleaning cost (USD)")
    annual_savings_usd: float = Field(..., ge=0, description="Annual savings with optimization (USD)")

    # Notes
    optimization_notes: List[str] = Field(..., description="Optimization recommendations")

    class Config:
        json_schema_extra = {
            "example": {
                "economizer_id": "econ-001",
                "calculated_at": "2025-11-09T10:30:00Z",
                "current_interval_hours": 168.0,
                "optimal_interval_hours": 144.0,
                "cleaning_efficiency_percent": 85.0,
                "over_cleaning_risk": False,
                "under_cleaning_risk": True,
                "next_recommended_cleaning": "2025-11-15T08:00:00Z",
                "recommended_sequence": "intensive",
                "recommended_soot_blowers": ["sb-001", "sb-002", "sb-003", "sb-004"],
                "current_annual_cleaning_cost_usd": 12000.0,
                "optimized_annual_cleaning_cost_usd": 11000.0,
                "annual_savings_usd": 1000.0,
                "optimization_notes": [
                    "Increase cleaning frequency by 14% based on fouling rate analysis",
                    "Use intensive sequence due to elevated fouling in inlet section",
                    "Consider chemical cleaning if soot blower effectiveness drops below 70%"
                ]
            }
        }


# =============================================================================
# Report Schemas
# =============================================================================

class DailyReportSummary(BaseModel):
    """Daily summary for a single economizer."""
    economizer_id: str
    economizer_name: str
    avg_effectiveness_percent: float
    avg_fouling_score: float
    heat_recovered_kwh: float
    cleaning_events: int
    alerts_triggered: int
    status: str


class DailyReport(BaseModel):
    """Daily performance report model."""
    report_date: datetime = Field(..., description="Report date")
    generated_at: datetime = Field(..., description="Generation timestamp")
    report_period_start: datetime = Field(..., description="Period start")
    report_period_end: datetime = Field(..., description="Period end")

    # Summary metrics
    total_economizers: int = Field(..., ge=0, description="Total monitored economizers")
    online_economizers: int = Field(..., ge=0, description="Online economizers")
    total_heat_recovered_kwh: float = Field(..., ge=0, description="Total heat recovered (kWh)")
    average_effectiveness_percent: float = Field(..., ge=0, le=100, description="Fleet average effectiveness (%)")
    average_fouling_score: float = Field(..., ge=0, le=100, description="Fleet average fouling score")

    # Alerts summary
    total_alerts: int = Field(..., ge=0, description="Total alerts triggered")
    critical_alerts: int = Field(..., ge=0, description="Critical alerts")
    acknowledged_alerts: int = Field(..., ge=0, description="Acknowledged alerts")

    # Cleaning summary
    cleaning_events: int = Field(..., ge=0, description="Total cleaning events")
    successful_cleanings: int = Field(..., ge=0, description="Successful cleanings")

    # Economizer summaries
    economizer_summaries: List[DailyReportSummary] = Field(..., description="Per-economizer summaries")

    class Config:
        json_schema_extra = {
            "example": {
                "report_date": "2025-11-08T00:00:00Z",
                "generated_at": "2025-11-09T06:00:00Z",
                "report_period_start": "2025-11-08T00:00:00Z",
                "report_period_end": "2025-11-09T00:00:00Z",
                "total_economizers": 5,
                "online_economizers": 5,
                "total_heat_recovered_kwh": 58500.0,
                "average_effectiveness_percent": 79.2,
                "average_fouling_score": 38.5,
                "total_alerts": 3,
                "critical_alerts": 0,
                "acknowledged_alerts": 2,
                "cleaning_events": 1,
                "successful_cleanings": 1,
                "economizer_summaries": [
                    {
                        "economizer_id": "econ-001",
                        "economizer_name": "Economizer Unit A1",
                        "avg_effectiveness_percent": 78.5,
                        "avg_fouling_score": 45.0,
                        "heat_recovered_kwh": 11700.0,
                        "cleaning_events": 0,
                        "alerts_triggered": 1,
                        "status": "online"
                    }
                ]
            }
        }


class WeeklyReport(BaseModel):
    """Weekly summary report model."""
    report_week: str = Field(..., description="Report week (ISO format)")
    generated_at: datetime = Field(..., description="Generation timestamp")
    report_period_start: datetime = Field(..., description="Period start")
    report_period_end: datetime = Field(..., description="Period end")

    # Trend analysis
    effectiveness_trend: TrendDirection = Field(..., description="Fleet effectiveness trend")
    fouling_trend: TrendDirection = Field(..., description="Fleet fouling trend")
    effectiveness_change_percent: float = Field(..., description="Effectiveness change (%)")

    # Energy metrics
    total_heat_recovered_kwh: float = Field(..., ge=0, description="Total heat recovered (kWh)")
    energy_savings_vs_previous_week_kwh: float = Field(..., description="Energy savings vs previous week (kWh)")

    # Cost impact
    estimated_fuel_savings_usd: float = Field(..., description="Estimated fuel savings (USD)")
    carbon_reduction_kg: float = Field(..., ge=0, description="CO2 reduction (kg)")

    # Maintenance summary
    total_cleaning_events: int = Field(..., ge=0, description="Total cleaning events")
    maintenance_hours: float = Field(..., ge=0, description="Total maintenance hours")

    # Alerts summary
    total_alerts: int = Field(..., ge=0, description="Total alerts")
    resolved_alerts: int = Field(..., ge=0, description="Resolved alerts")

    # Recommendations
    recommendations: List[str] = Field(..., description="Weekly recommendations")

    class Config:
        json_schema_extra = {
            "example": {
                "report_week": "2025-W45",
                "generated_at": "2025-11-10T06:00:00Z",
                "report_period_start": "2025-11-03T00:00:00Z",
                "report_period_end": "2025-11-10T00:00:00Z",
                "effectiveness_trend": "stable",
                "fouling_trend": "degrading",
                "effectiveness_change_percent": -0.5,
                "total_heat_recovered_kwh": 405000.0,
                "energy_savings_vs_previous_week_kwh": 2500.0,
                "estimated_fuel_savings_usd": 125.0,
                "carbon_reduction_kg": 542.0,
                "total_cleaning_events": 3,
                "maintenance_hours": 2.5,
                "total_alerts": 12,
                "resolved_alerts": 10,
                "recommendations": [
                    "Schedule cleaning for Economizer A1 within next 7 days",
                    "Review alert thresholds for Economizer B2 - false positive rate high",
                    "Consider chemical cleaning for Economizer C1 - soot blower effectiveness declining"
                ]
            }
        }


class EfficiencyReport(BaseModel):
    """Efficiency analysis report model."""
    report_id: str = Field(..., description="Report ID")
    generated_at: datetime = Field(..., description="Generation timestamp")
    report_period_start: datetime = Field(..., description="Period start")
    report_period_end: datetime = Field(..., description="Period end")

    # Fleet summary
    total_economizers: int = Field(..., ge=0, description="Total economizers analyzed")
    fleet_average_efficiency_percent: float = Field(..., ge=0, le=100, description="Fleet average efficiency (%)")
    fleet_design_efficiency_percent: float = Field(..., ge=0, le=100, description="Fleet design efficiency (%)")

    # Loss analysis
    total_energy_loss_kwh: float = Field(..., ge=0, description="Total energy loss (kWh)")
    total_fuel_cost_impact_usd: float = Field(..., ge=0, description="Total fuel cost impact (USD)")
    total_carbon_impact_kg: float = Field(..., ge=0, description="Total CO2 impact (kg)")

    # Savings potential
    total_savings_potential_usd_year: float = Field(..., ge=0, description="Annual savings potential (USD)")
    total_carbon_reduction_potential_kg_year: float = Field(..., ge=0, description="Annual CO2 reduction potential (kg)")

    # Per-economizer analysis
    economizer_efficiency: List[Dict[str, Any]] = Field(..., description="Per-economizer efficiency data")

    # Recommendations
    priority_actions: List[str] = Field(..., description="Priority action items")

    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "eff-rpt-001",
                "generated_at": "2025-11-09T10:30:00Z",
                "report_period_start": "2025-10-01T00:00:00Z",
                "report_period_end": "2025-11-01T00:00:00Z",
                "total_economizers": 5,
                "fleet_average_efficiency_percent": 78.5,
                "fleet_design_efficiency_percent": 85.0,
                "total_energy_loss_kwh": 125000.0,
                "total_fuel_cost_impact_usd": 6250.0,
                "total_carbon_impact_kg": 27125.0,
                "total_savings_potential_usd_year": 75000.0,
                "total_carbon_reduction_potential_kg_year": 162750.0,
                "economizer_efficiency": [
                    {
                        "economizer_id": "econ-001",
                        "name": "Economizer Unit A1",
                        "current_efficiency": 78.5,
                        "design_efficiency": 85.0,
                        "loss_kwh": 25000.0,
                        "savings_potential_usd_year": 15000.0
                    }
                ],
                "priority_actions": [
                    "Clean Economizer A1 - highest efficiency loss",
                    "Investigate sensor calibration on Economizer B2",
                    "Schedule preventive maintenance for Economizer C1"
                ]
            }
        }


class ReportExport(BaseModel):
    """Report export request model."""
    report_type: str = Field(..., description="Report type: daily, weekly, efficiency, custom")
    format: ReportFormat = Field(..., description="Export format")
    start_date: datetime = Field(..., description="Report period start")
    end_date: datetime = Field(..., description="Report period end")
    economizer_ids: Optional[List[str]] = Field(None, description="Filter by economizer IDs")
    include_charts: bool = Field(True, description="Include charts in report")
    include_raw_data: bool = Field(False, description="Include raw data appendix")
    email_recipients: Optional[List[str]] = Field(None, description="Email recipients for delivery")

    class Config:
        json_schema_extra = {
            "example": {
                "report_type": "efficiency",
                "format": "pdf",
                "start_date": "2025-10-01T00:00:00Z",
                "end_date": "2025-11-01T00:00:00Z",
                "economizer_ids": ["econ-001", "econ-002"],
                "include_charts": True,
                "include_raw_data": False,
                "email_recipients": ["engineer@example.com"]
            }
        }


class ReportExportResponse(BaseModel):
    """Report export response model."""
    export_id: str = Field(..., description="Export job ID")
    status: str = Field(..., description="Export status: pending, processing, completed, failed")
    requested_at: datetime = Field(..., description="Request timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    download_url: Optional[str] = Field(None, description="Download URL (valid for 24 hours)")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    expires_at: Optional[datetime] = Field(None, description="Download URL expiration")

    class Config:
        json_schema_extra = {
            "example": {
                "export_id": "export-001",
                "status": "completed",
                "requested_at": "2025-11-09T10:30:00Z",
                "completed_at": "2025-11-09T10:30:45Z",
                "download_url": "https://api.greenlang.io/exports/export-001/report.pdf",
                "file_size_bytes": 2548000,
                "expires_at": "2025-11-10T10:30:45Z"
            }
        }


# =============================================================================
# Query Parameters Schemas
# =============================================================================

class PaginationParams(BaseModel):
    """Pagination query parameters."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")


class TimeRangeParams(BaseModel):
    """Time range query parameters."""
    start_time: Optional[datetime] = Field(None, description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    resolution: str = Field("1h", description="Data resolution: 1m, 5m, 15m, 1h, 1d")

    @validator("resolution")
    def validate_resolution(cls, v):
        valid_resolutions = ["1m", "5m", "15m", "1h", "1d"]
        if v not in valid_resolutions:
            raise ValueError(f"Resolution must be one of: {valid_resolutions}")
        return v


class AlertFilterParams(BaseModel):
    """Alert filter query parameters."""
    status: Optional[AlertStatus] = Field(None, description="Filter by status")
    severity: Optional[AlertSeverity] = Field(None, description="Filter by severity")
    alert_type: Optional[AlertType] = Field(None, description="Filter by type")
    economizer_id: Optional[str] = Field(None, description="Filter by economizer")


# =============================================================================
# Error Response Schemas
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid economizer ID format",
                "details": {"field": "economizer_id", "value": "invalid"},
                "request_id": "req-123456"
            }
        }


class NotFoundResponse(BaseModel):
    """Not found error response model."""
    error: str = Field("not_found", description="Error code")
    message: str = Field(..., description="Error message")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource ID")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "not_found",
                "message": "Economizer not found",
                "resource_type": "economizer",
                "resource_id": "econ-999"
            }
        }
