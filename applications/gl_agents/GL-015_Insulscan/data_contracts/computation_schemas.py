# -*- coding: utf-8 -*-
"""
GL-015 Insulscan: Computation Schemas - Version 1.0

Provides validated data schemas for heat loss computations, condition
assessment results, and prediction records with complete auditability.

This module defines Pydantic v2 models for:
- HeatLossInput: Input data for heat loss calculations
- HeatLossOutput: Heat loss calculation results with breakdown
- ConditionAssessmentInput: Input for condition assessment
- ConditionAssessmentOutput: Condition assessment results with factors

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ComputationType(str, Enum):
    """Types of computations performed."""
    HEAT_LOSS_STEADY_STATE = "heat_loss_steady_state"
    HEAT_LOSS_TRANSIENT = "heat_loss_transient"
    SURFACE_TEMPERATURE = "surface_temperature"
    THICKNESS_CALCULATION = "thickness_calculation"
    CONDITION_ASSESSMENT = "condition_assessment"
    ENERGY_SAVINGS = "energy_savings"
    ROI_ANALYSIS = "roi_analysis"
    REMAINING_LIFE = "remaining_life"


class ComputationStatus(str, Enum):
    """Status of computation."""
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


class ValidityFlag(str, Enum):
    """Validity flags for computation results."""
    VALID = "valid"
    EXTRAPOLATED = "extrapolated"
    APPROXIMATE = "approximate"
    LOW_CONFIDENCE = "low_confidence"
    CHECK_REQUIRED = "check_required"
    INVALID = "invalid"


class HeatTransferMechanism(str, Enum):
    """Heat transfer mechanisms."""
    CONDUCTION = "conduction"
    CONVECTION_NATURAL = "convection_natural"
    CONVECTION_FORCED = "convection_forced"
    RADIATION = "radiation"
    COMBINED = "combined"


class ConditionCategory(str, Enum):
    """Categories for condition assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class SeverityLevel(str, Enum):
    """Severity levels for issues."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WarningCode(str, Enum):
    """Warning codes for computation issues."""
    TEMPERATURE_EXCEEDED = "temperature_exceeded"
    SURFACE_TEMP_HIGH = "surface_temp_high"
    HEAT_LOSS_EXCESSIVE = "heat_loss_excessive"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    EXTRAPOLATION_REQUIRED = "extrapolation_required"
    PROPERTY_UNCERTAINTY = "property_uncertainty"
    CONVERGENCE_ISSUE = "convergence_issue"
    ASSUMPTION_MADE = "assumption_made"


# =============================================================================
# COMPUTATION WARNING
# =============================================================================

class ComputationWarning(BaseModel):
    """Warning generated during computation."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "code": "surface_temp_high",
                    "message": "Calculated surface temperature exceeds safe touch limit",
                    "severity": "warning",
                    "value": 65.5,
                    "threshold": 60.0
                }
            ]
        }
    )

    code: WarningCode = Field(
        ...,
        description="Warning code for programmatic handling"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable warning message"
    )
    severity: Literal["info", "warning", "error"] = Field(
        default="warning",
        description="Warning severity level"
    )
    value: Optional[float] = Field(
        None,
        description="Value that triggered the warning"
    )
    threshold: Optional[float] = Field(
        None,
        description="Threshold that was exceeded"
    )
    location: Optional[str] = Field(
        None,
        max_length=200,
        description="Location in calculation where warning occurred"
    )


# =============================================================================
# OPERATING CONDITIONS
# =============================================================================

class OperatingConditions(BaseModel):
    """Operating conditions for heat loss calculation."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "process_temp_c": 180.0,
                    "process_pressure_bar_g": 10.0,
                    "operating_mode": "continuous",
                    "load_factor": 1.0
                }
            ]
        }
    )

    # Process conditions
    process_temp_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Process/pipe temperature in Celsius"
    )
    process_pressure_bar_g: Optional[float] = Field(
        None,
        ge=-1,
        le=500,
        description="Process pressure in bar gauge"
    )

    # Operating mode
    operating_mode: Literal[
        "continuous", "batch", "cyclic", "intermittent", "standby"
    ] = Field(
        default="continuous",
        description="Operating mode"
    )
    load_factor: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Load factor (fraction of full operation)"
    )

    # Operating hours
    annual_operating_hours: Optional[float] = Field(
        None,
        ge=0,
        le=8760,
        description="Annual operating hours"
    )

    # Temperature profile for cyclic operation
    temp_profile: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Temperature profile for cyclic operation [{time_h, temp_c}]"
    )


# =============================================================================
# HEAT LOSS INPUT
# =============================================================================

class HeatLossInput(BaseModel):
    """
    Input data for heat loss calculations.

    Contains all parameters required for steady-state or transient
    heat loss calculations through insulated surfaces.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "asset_id": "INS-1001",
                    "operating_conditions": {
                        "process_temp_c": 180.0,
                        "operating_mode": "continuous"
                    },
                    "ambient_conditions": {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "temp_c": 22.0,
                        "humidity_percent": 50.0,
                        "wind_speed_ms": 2.0,
                        "solar_radiation_w_m2": 0.0
                    }
                }
            ]
        }
    )

    # Identifiers
    calculation_id: str = Field(
        default_factory=lambda: f"CALC-{uuid.uuid4().hex[:12].upper()}",
        description="Unique calculation identifier"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Asset configuration reference (can be loaded separately)
    asset_config_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to asset configuration"
    )

    # Operating conditions
    operating_conditions: OperatingConditions = Field(
        ...,
        description="Operating conditions"
    )

    # Ambient conditions - inline for convenience
    ambient_conditions: Dict[str, Any] = Field(
        ...,
        description="Ambient conditions (AmbientConditions compatible)"
    )

    # Calculation parameters
    calculation_method: Literal[
        "astm_c680", "iso_12241", "vdi_2055", "simplified"
    ] = Field(
        default="astm_c680",
        description="Heat loss calculation standard/method"
    )
    include_radiation: bool = Field(
        default=True,
        description="Include radiation heat transfer"
    )
    include_convection: bool = Field(
        default=True,
        description="Include convection heat transfer"
    )

    # Property overrides (if not using asset defaults)
    insulation_conductivity_override: Optional[float] = Field(
        None,
        gt=0,
        le=5,
        description="Override thermal conductivity in W/(m.K)"
    )
    jacket_emissivity_override: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Override jacket emissivity"
    )

    # Defect modeling
    include_defects: bool = Field(
        default=False,
        description="Include known defects in calculation"
    )
    defect_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Total defect/bare area in m^2"
    )

    def compute_input_hash(self) -> str:
        """Compute SHA-256 hash of input data for provenance."""
        content = (
            f"{self.asset_id}"
            f"{self.operating_conditions.process_temp_c}"
            f"{self.ambient_conditions.get('temp_c', 0)}"
            f"{self.ambient_conditions.get('wind_speed_ms', 0)}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# HEAT LOSS BREAKDOWN
# =============================================================================

class HeatLossBreakdown(BaseModel):
    """Breakdown of heat loss by mechanism."""

    model_config = ConfigDict(frozen=True)

    # By mechanism
    conduction_w: float = Field(
        ...,
        ge=0,
        description="Heat loss through conduction in Watts"
    )
    convection_w: float = Field(
        ...,
        ge=0,
        description="Heat loss through convection in Watts"
    )
    radiation_w: float = Field(
        ...,
        ge=0,
        description="Heat loss through radiation in Watts"
    )

    # By location (if applicable)
    insulated_area_w: Optional[float] = Field(
        None,
        ge=0,
        description="Heat loss from insulated area in Watts"
    )
    bare_area_w: Optional[float] = Field(
        None,
        ge=0,
        description="Heat loss from bare/defect area in Watts"
    )
    support_w: Optional[float] = Field(
        None,
        ge=0,
        description="Heat loss through supports in Watts"
    )

    # Percentages
    conduction_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Conduction as percentage of total"
    )
    convection_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Convection as percentage of total"
    )
    radiation_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Radiation as percentage of total"
    )


class UncertaintyBounds(BaseModel):
    """Uncertainty bounds for calculated values."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(
        ...,
        description="Central/nominal value"
    )
    lower_bound: float = Field(
        ...,
        description="Lower uncertainty bound"
    )
    upper_bound: float = Field(
        ...,
        description="Upper uncertainty bound"
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Confidence level (e.g., 0.95 for 95%)"
    )
    uncertainty_sources: List[str] = Field(
        default_factory=list,
        description="Sources of uncertainty"
    )


# =============================================================================
# HEAT LOSS OUTPUT
# =============================================================================

class HeatLossOutput(BaseModel):
    """
    Complete heat loss calculation result.

    Contains calculated heat loss, surface temperature, breakdown
    by mechanism, and uncertainty analysis.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "calculation_id": "CALC-2024-00001234",
                    "asset_id": "INS-1001",
                    "heat_loss_w": 1250.5,
                    "heat_loss_w_m2": 117.3,
                    "surface_temp_c": 42.5,
                    "status": "success",
                    "validity": "valid"
                }
            ]
        }
    )

    # Link to input
    calculation_id: str = Field(
        ...,
        description="Reference to calculation input"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation completion timestamp"
    )

    # Primary results
    heat_loss_w: float = Field(
        ...,
        ge=0,
        description="Total heat loss in Watts"
    )
    heat_loss_w_m: Optional[float] = Field(
        None,
        ge=0,
        description="Heat loss per unit length in W/m (for pipes)"
    )
    heat_loss_w_m2: float = Field(
        ...,
        ge=0,
        description="Heat loss per unit area in W/m^2"
    )

    # Surface temperature
    surface_temp_c: float = Field(
        ...,
        ge=-50,
        le=1500,
        description="Calculated outer surface temperature in Celsius"
    )
    surface_temp_safe: bool = Field(
        default=True,
        description="Whether surface temp is within safe touch limit"
    )
    safe_touch_limit_c: float = Field(
        default=60.0,
        description="Safe touch temperature limit used"
    )

    # Breakdown
    breakdown_by_mechanism: HeatLossBreakdown = Field(
        ...,
        description="Heat loss breakdown by mechanism"
    )

    # Uncertainty
    uncertainty_bounds: Optional[UncertaintyBounds] = Field(
        None,
        description="Uncertainty bounds for heat loss"
    )

    # Comparison to design
    design_heat_loss_w_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Design heat loss for comparison"
    )
    heat_loss_ratio: Optional[float] = Field(
        None,
        ge=0,
        description="Actual/design heat loss ratio"
    )
    excess_heat_loss_percent: Optional[float] = Field(
        None,
        description="Percentage above design heat loss"
    )

    # Energy and cost impact
    annual_energy_loss_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Annual energy loss in kWh"
    )
    annual_energy_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Annual energy cost in local currency"
    )
    energy_price_per_kwh: Optional[float] = Field(
        None,
        ge=0,
        description="Energy price used for cost calculation"
    )
    co2_emissions_kg_yr: Optional[float] = Field(
        None,
        ge=0,
        description="Annual CO2 emissions in kg"
    )

    # Intermediate calculations (for audit)
    convection_coefficient_w_m2k: Optional[float] = Field(
        None,
        gt=0,
        description="Calculated convection coefficient"
    )
    thermal_resistance_m2k_w: Optional[float] = Field(
        None,
        gt=0,
        description="Total thermal resistance"
    )
    effective_conductivity_w_mk: Optional[float] = Field(
        None,
        gt=0,
        description="Effective insulation conductivity at mean temp"
    )

    # Computation metadata
    status: ComputationStatus = Field(
        ...,
        description="Computation status"
    )
    validity: ValidityFlag = Field(
        default=ValidityFlag.VALID,
        description="Result validity flag"
    )
    warnings: List[ComputationWarning] = Field(
        default_factory=list,
        description="Warnings from computation"
    )
    calculation_method: str = Field(
        ...,
        description="Calculation method used"
    )
    execution_time_ms: float = Field(
        ...,
        ge=0,
        description="Execution time in milliseconds"
    )

    # Provenance
    input_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of input data"
    )
    output_hash: Optional[str] = Field(
        None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of output data"
    )

    def compute_output_hash(self) -> str:
        """Compute SHA-256 hash of output for provenance."""
        content = (
            f"{self.calculation_id}"
            f"{self.heat_loss_w:.4f}"
            f"{self.surface_temp_c:.4f}"
            f"{self.timestamp.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def is_valid(self) -> bool:
        """Check if result is valid for use."""
        return self.validity in [ValidityFlag.VALID, ValidityFlag.APPROXIMATE]


# =============================================================================
# CONDITION ASSESSMENT INPUT
# =============================================================================

class ConditionAssessmentInput(BaseModel):
    """
    Input data for condition assessment.

    Combines thermal data, inspection data, and asset age
    for comprehensive condition scoring.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "assessment_id": "CA-2024-00001234",
                    "asset_id": "INS-1001",
                    "thermal_data": {
                        "avg_surface_temp_c": 45.2,
                        "max_surface_temp_c": 62.5,
                        "hotspot_count": 2
                    },
                    "age_years": 5.5
                }
            ]
        }
    )

    # Identifiers
    assessment_id: str = Field(
        default_factory=lambda: f"CA-{uuid.uuid4().hex[:12].upper()}",
        description="Unique assessment identifier"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Assessment timestamp"
    )

    # Thermal data
    thermal_data: Dict[str, Any] = Field(
        ...,
        description="Thermal measurement data"
    )
    thermal_image_refs: List[str] = Field(
        default_factory=list,
        description="References to thermal images used"
    )

    # Inspection data
    inspection_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Recent inspection findings"
    )
    inspection_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to inspection record"
    )

    # Age data
    age_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Asset age and lifecycle data"
    )
    installation_date: Optional[datetime] = Field(
        None,
        description="Installation date"
    )
    age_years: Optional[float] = Field(
        None,
        ge=0,
        description="Asset age in years"
    )
    design_life_years: Optional[float] = Field(
        None,
        gt=0,
        description="Design life in years"
    )

    # Operating history
    operating_history: Optional[Dict[str, Any]] = Field(
        None,
        description="Operating history data"
    )
    excursion_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of temperature excursions"
    )
    moisture_exposure_events: Optional[int] = Field(
        None,
        ge=0,
        description="Number of moisture exposure events"
    )

    # Previous assessments
    previous_assessment_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to previous assessment"
    )
    previous_condition_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Previous condition score"
    )


# =============================================================================
# CONTRIBUTING FACTOR
# =============================================================================

class ContributingFactor(BaseModel):
    """Individual factor contributing to condition assessment."""

    model_config = ConfigDict(frozen=True)

    factor_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of contributing factor"
    )
    factor_category: Literal[
        "thermal", "physical", "age", "environmental", "operational"
    ] = Field(
        ...,
        description="Category of factor"
    )
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Factor score (0-100)"
    )
    weight: float = Field(
        ...,
        ge=0,
        le=1,
        description="Weight in overall calculation"
    )
    weighted_contribution: float = Field(
        ...,
        description="Weighted contribution to overall score"
    )
    severity: SeverityLevel = Field(
        ...,
        description="Severity level of this factor"
    )
    details: Optional[str] = Field(
        None,
        max_length=500,
        description="Details about this factor"
    )
    recommendation: Optional[str] = Field(
        None,
        max_length=500,
        description="Recommendation based on this factor"
    )


# =============================================================================
# CONDITION ASSESSMENT OUTPUT
# =============================================================================

class ConditionAssessmentOutput(BaseModel):
    """
    Complete condition assessment result.

    Provides overall condition score, category, contributing factors,
    and recommendations for maintenance planning.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "assessment_id": "CA-2024-00001234",
                    "asset_id": "INS-1001",
                    "condition_score": 72.5,
                    "condition_category": "fair",
                    "severity": "medium",
                    "status": "success"
                }
            ]
        }
    )

    # Link to input
    assessment_id: str = Field(
        ...,
        description="Reference to assessment input"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Assessment completion timestamp"
    )

    # Overall condition
    condition_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall condition score (0-100, higher is better)"
    )
    condition_category: ConditionCategory = Field(
        ...,
        description="Condition category"
    )
    severity: SeverityLevel = Field(
        ...,
        description="Overall severity level"
    )

    # Trend analysis
    score_change: Optional[float] = Field(
        None,
        ge=-100,
        le=100,
        description="Change from previous assessment"
    )
    trend: Optional[Literal["improving", "stable", "degrading"]] = Field(
        None,
        description="Condition trend"
    )
    degradation_rate: Optional[float] = Field(
        None,
        description="Estimated degradation rate (points per year)"
    )

    # Contributing factors
    contributing_factors: List[ContributingFactor] = Field(
        default_factory=list,
        description="Factors contributing to assessment"
    )
    primary_concern: Optional[str] = Field(
        None,
        max_length=500,
        description="Primary concern identified"
    )

    # Specific assessments
    thermal_performance_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Thermal performance sub-score"
    )
    physical_condition_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Physical condition sub-score"
    )
    age_factor_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Age-related sub-score"
    )

    # Remaining life estimation
    estimated_remaining_life_years: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated remaining useful life in years"
    )
    remaining_life_confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence in remaining life estimate"
    )
    end_of_life_date: Optional[datetime] = Field(
        None,
        description="Estimated end of life date"
    )

    # Risk assessment
    failure_risk: Optional[Literal["low", "medium", "high", "critical"]] = Field(
        None,
        description="Risk of insulation failure"
    )
    cui_risk: Optional[Literal["low", "medium", "high", "confirmed"]] = Field(
        None,
        description="Risk of corrosion under insulation"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )
    priority_actions: List[str] = Field(
        default_factory=list,
        description="Priority actions required"
    )
    next_assessment_date: Optional[datetime] = Field(
        None,
        description="Recommended next assessment date"
    )
    repair_recommended: bool = Field(
        default=False,
        description="Whether repair is recommended"
    )
    replacement_recommended: bool = Field(
        default=False,
        description="Whether replacement is recommended"
    )

    # Economic impact
    estimated_annual_energy_loss: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated annual energy loss from degradation (kWh)"
    )
    estimated_annual_cost_impact: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated annual cost impact"
    )
    repair_cost_estimate: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated repair cost"
    )
    replacement_cost_estimate: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated replacement cost"
    )

    # Computation metadata
    status: ComputationStatus = Field(
        ...,
        description="Computation status"
    )
    validity: ValidityFlag = Field(
        default=ValidityFlag.VALID,
        description="Result validity flag"
    )
    warnings: List[ComputationWarning] = Field(
        default_factory=list,
        description="Warnings from assessment"
    )
    assessment_method: str = Field(
        default="standard",
        description="Assessment methodology used"
    )
    execution_time_ms: float = Field(
        ...,
        ge=0,
        description="Execution time in milliseconds"
    )

    # Provenance
    input_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of input data"
    )
    output_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of output data"
    )

    def compute_output_hash(self) -> str:
        """Compute SHA-256 hash of output for provenance."""
        content = (
            f"{self.assessment_id}"
            f"{self.condition_score:.4f}"
            f"{self.condition_category.value}"
            f"{self.timestamp.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def requires_action(self) -> bool:
        """Check if immediate action is required."""
        return self.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]


# =============================================================================
# EXPORTS
# =============================================================================

COMPUTATION_SCHEMAS = {
    "ComputationType": ComputationType,
    "ComputationStatus": ComputationStatus,
    "ValidityFlag": ValidityFlag,
    "HeatTransferMechanism": HeatTransferMechanism,
    "ConditionCategory": ConditionCategory,
    "SeverityLevel": SeverityLevel,
    "WarningCode": WarningCode,
    "ComputationWarning": ComputationWarning,
    "OperatingConditions": OperatingConditions,
    "HeatLossInput": HeatLossInput,
    "HeatLossBreakdown": HeatLossBreakdown,
    "UncertaintyBounds": UncertaintyBounds,
    "HeatLossOutput": HeatLossOutput,
    "ConditionAssessmentInput": ConditionAssessmentInput,
    "ContributingFactor": ContributingFactor,
    "ConditionAssessmentOutput": ConditionAssessmentOutput,
}

__all__ = [
    # Enumerations
    "ComputationType",
    "ComputationStatus",
    "ValidityFlag",
    "HeatTransferMechanism",
    "ConditionCategory",
    "SeverityLevel",
    "WarningCode",
    # Supporting models
    "ComputationWarning",
    "OperatingConditions",
    "HeatLossBreakdown",
    "UncertaintyBounds",
    "ContributingFactor",
    # Main schemas
    "HeatLossInput",
    "HeatLossOutput",
    "ConditionAssessmentInput",
    "ConditionAssessmentOutput",
    # Export dictionary
    "COMPUTATION_SCHEMAS",
]
