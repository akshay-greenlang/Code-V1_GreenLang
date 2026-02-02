# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro: Computation Schemas - Version 1.0

Provides validated data schemas for computation records, thermal calculation
results, and prediction records for complete auditability.

This module defines Pydantic v2 models for:
- ComputationRecord: Base computation record with provenance tracking
- ThermalComputationResult: Heat transfer calculation results (Q, UA, LMTD, NTU)
- PredictionRecord: ML model prediction with confidence intervals and explanations

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
    THERMAL_RATING = "thermal_rating"  # Calculate duty for given conditions
    THERMAL_SIMULATION = "thermal_simulation"  # Simulate outlet conditions
    UA_CALCULATION = "ua_calculation"  # Calculate current UA
    FOULING_FACTOR = "fouling_factor"  # Calculate current fouling resistance
    PRESSURE_DROP = "pressure_drop"  # Calculate pressure drops
    LMTD_CORRECTION = "lmtd_correction"  # Calculate LMTD correction factor
    NTU_EFFECTIVENESS = "ntu_effectiveness"  # NTU-effectiveness calculation
    CLEANING_OPTIMIZATION = "cleaning_optimization"  # Optimal cleaning time
    FOULING_PREDICTION = "fouling_prediction"  # Future fouling prediction
    RUL_PREDICTION = "rul_prediction"  # Remaining useful life prediction


class ComputationStatus(str, Enum):
    """Status of computation."""
    SUCCESS = "success"
    WARNING = "warning"  # Completed with warnings
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


class ValidityFlag(str, Enum):
    """Validity flags for computation results."""
    VALID = "valid"
    EXTRAPOLATED = "extrapolated"  # Outside normal operating range
    APPROXIMATE = "approximate"  # Used simplified method
    LOW_CONFIDENCE = "low_confidence"
    CHECK_REQUIRED = "check_required"
    INVALID = "invalid"


class WarningCode(str, Enum):
    """Warning codes for computation issues."""
    TEMPERATURE_CROSS = "temperature_cross"  # Temperature cross detected
    LOW_LMTD = "low_lmtd"  # Very low LMTD
    HIGH_FOULING = "high_fouling"  # Fouling above threshold
    ENERGY_IMBALANCE = "energy_imbalance"  # Q_hot != Q_cold
    LOW_FLOW = "low_flow"  # Below minimum flow
    HIGH_DP = "high_dp"  # Pressure drop above threshold
    PROPERTY_EXTRAPOLATION = "property_extrapolation"  # Properties extrapolated
    CONVERGENCE_ISSUE = "convergence_issue"  # Iteration convergence issue
    DATA_QUALITY = "data_quality"  # Poor input data quality


class PredictionType(str, Enum):
    """Types of predictions."""
    FOULING_TRAJECTORY = "fouling_trajectory"  # Future fouling prediction
    CLEANING_TIME = "cleaning_time"  # Time until cleaning needed
    UA_FORECAST = "ua_forecast"  # Future UA prediction
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FAILURE_PROBABILITY = "failure_probability"


class ModelType(str, Enum):
    """Types of prediction models."""
    PHYSICS_BASED = "physics_based"  # First principles
    MACHINE_LEARNING = "machine_learning"  # ML model
    HYBRID = "hybrid"  # Physics-informed ML
    ENSEMBLE = "ensemble"  # Multiple models
    STATISTICAL = "statistical"  # Statistical forecasting


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
                    "code": "energy_imbalance",
                    "message": "Heat duty imbalance of 5.2% between hot and cold sides",
                    "severity": "warning",
                    "value": 5.2,
                    "threshold": 5.0
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
# COMPUTATION RECORD
# =============================================================================

class ComputationRecord(BaseModel):
    """
    Base computation record for audit trail.

    Tracks all computations with full provenance including:
    - Input/output hashes for reproducibility
    - Algorithm and property library versions
    - Execution timing and status
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "record_id": "COMP-2024-00001234",
                    "computation_type": "ua_calculation",
                    "exchanger_id": "HX-1001",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "inputs_hash": "a1b2c3d4e5f6...",
                    "outputs_hash": "f6e5d4c3b2a1...",
                    "algorithm_version": "1.2.0",
                    "property_library_version": "REFPROP-10.0",
                    "execution_time_ms": 45.2,
                    "status": "success"
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    record_id: str = Field(
        default_factory=lambda: f"COMP-{uuid.uuid4().hex[:12].upper()}",
        description="Unique computation record identifier"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )
    computation_type: ComputationType = Field(
        ...,
        description="Type of computation performed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Computation timestamp"
    )

    # Provenance hashes
    inputs_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of all input data"
    )
    outputs_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of all output data"
    )

    # Version tracking
    algorithm_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of calculation algorithm"
    )
    algorithm_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Name of algorithm used"
    )
    property_library_version: Optional[str] = Field(
        None,
        max_length=100,
        description="Version of thermodynamic property library"
    )
    property_library_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Name of property library (e.g., REFPROP, CoolProp)"
    )

    # Execution metrics
    execution_time_ms: float = Field(
        ...,
        ge=0,
        description="Computation execution time in milliseconds"
    )
    iterations: Optional[int] = Field(
        None,
        ge=0,
        description="Number of iterations (for iterative methods)"
    )
    convergence_tolerance: Optional[float] = Field(
        None,
        gt=0,
        description="Convergence tolerance achieved"
    )

    # Status
    status: ComputationStatus = Field(
        ...,
        description="Computation status"
    )
    error_message: Optional[str] = Field(
        None,
        max_length=1000,
        description="Error message if status is failed"
    )
    warnings: List[ComputationWarning] = Field(
        default_factory=list,
        description="List of warnings generated"
    )

    # Context
    triggered_by: Optional[Literal[
        "scheduled", "manual", "event", "threshold", "api"
    ]] = Field(
        None,
        description="What triggered this computation"
    )
    user_id: Optional[str] = Field(
        None,
        max_length=100,
        description="User who triggered computation (if manual)"
    )

    # Input data reference
    measurement_set_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to input measurement set"
    )

    @classmethod
    def compute_hash(cls, data: Any) -> str:
        """Compute SHA-256 hash of data for provenance."""
        if isinstance(data, BaseModel):
            content = data.model_dump_json(exclude_none=True)
        elif isinstance(data, dict):
            import json
            content = json.dumps(data, sort_keys=True, default=str)
        else:
            content = str(data)

        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# THERMAL COMPUTATION RESULT
# =============================================================================

class LMTDResult(BaseModel):
    """Log Mean Temperature Difference calculation result."""

    model_config = ConfigDict(frozen=True)

    lmtd_c: float = Field(
        ...,
        description="Log Mean Temperature Difference in Celsius"
    )
    delta_t1_c: float = Field(
        ...,
        description="Temperature difference at one end in Celsius"
    )
    delta_t2_c: float = Field(
        ...,
        description="Temperature difference at other end in Celsius"
    )
    correction_factor: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="LMTD correction factor (F) for non-counter-current flow"
    )
    corrected_lmtd_c: float = Field(
        ...,
        description="Corrected LMTD (LMTD * F) in Celsius"
    )
    method: Literal[
        "arithmetic", "log_mean", "weighted"
    ] = Field(
        default="log_mean",
        description="Method used for temperature difference"
    )


class NTUResult(BaseModel):
    """NTU-Effectiveness calculation result."""

    model_config = ConfigDict(frozen=True)

    ntu: float = Field(
        ...,
        ge=0,
        description="Number of Transfer Units"
    )
    effectiveness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Heat exchanger effectiveness (epsilon)"
    )
    capacity_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Heat capacity ratio (C_min/C_max)"
    )
    c_min_w_k: float = Field(
        ...,
        gt=0,
        description="Minimum heat capacity rate in W/K"
    )
    c_max_w_k: float = Field(
        ...,
        gt=0,
        description="Maximum heat capacity rate in W/K"
    )
    min_side: Literal["hot", "cold"] = Field(
        ...,
        description="Which side has minimum heat capacity rate"
    )


class HeatDutyResult(BaseModel):
    """Heat duty calculation result."""

    model_config = ConfigDict(frozen=True)

    q_hot_kw: float = Field(
        ...,
        description="Heat duty from hot side in kW"
    )
    q_cold_kw: float = Field(
        ...,
        description="Heat duty from cold side in kW"
    )
    q_average_kw: float = Field(
        ...,
        description="Average heat duty in kW"
    )
    q_imbalance_percent: float = Field(
        ...,
        description="Imbalance between hot and cold duties as percentage"
    )
    heat_loss_kw: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated heat loss to ambient in kW"
    )


class FoulingResult(BaseModel):
    """Fouling factor calculation result."""

    model_config = ConfigDict(frozen=True)

    total_fouling_m2k_w: float = Field(
        ...,
        ge=0,
        description="Total fouling resistance (shell + tube) in m^2.K/W"
    )
    shell_fouling_m2k_w: Optional[float] = Field(
        None,
        ge=0,
        description="Shell-side fouling resistance in m^2.K/W"
    )
    tube_fouling_m2k_w: Optional[float] = Field(
        None,
        ge=0,
        description="Tube-side fouling resistance in m^2.K/W"
    )
    fouling_factor_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Fouling as percentage of design fouling allowance"
    )
    excess_fouling_m2k_w: Optional[float] = Field(
        None,
        description="Fouling above design allowance (can be negative)"
    )
    clean_ua_w_k: float = Field(
        ...,
        gt=0,
        description="Clean UA used as reference in W/K"
    )


class ThermalComputationResult(BaseModel):
    """
    Complete thermal computation result for heat exchanger.

    Contains all calculated thermal performance parameters including:
    - Heat duty (Q)
    - Overall heat transfer coefficient times area (UA)
    - Log Mean Temperature Difference (LMTD)
    - Effectiveness and NTU
    - Fouling factors
    - Validity flags and warnings
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "computation_record": {
                        "record_id": "COMP-2024-00001234",
                        "computation_type": "ua_calculation",
                        "exchanger_id": "HX-1001",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "inputs_hash": "a" * 64,
                        "outputs_hash": "b" * 64,
                        "algorithm_version": "1.2.0",
                        "execution_time_ms": 45.2,
                        "status": "success"
                    },
                    "q_kw": 2350.5,
                    "ua_w_k": 38500.0,
                    "ua_clean_w_k": 45000.0,
                    "ua_ratio": 0.856,
                    "overall_validity": "valid"
                }
            ]
        }
    )

    # Link to computation record
    computation_record: ComputationRecord = Field(
        ...,
        description="Base computation record with provenance"
    )

    # Primary results
    q_kw: float = Field(
        ...,
        description="Heat duty in kW"
    )
    ua_w_k: float = Field(
        ...,
        gt=0,
        description="Current overall heat transfer coefficient times area in W/K"
    )
    ua_clean_w_k: float = Field(
        ...,
        gt=0,
        description="Reference clean UA in W/K"
    )
    ua_ratio: float = Field(
        ...,
        gt=0,
        le=1.5,
        description="Ratio of current UA to clean UA"
    )

    # LMTD calculation
    lmtd_result: Optional[LMTDResult] = Field(
        None,
        description="LMTD calculation details"
    )

    # NTU-Effectiveness
    ntu_result: Optional[NTUResult] = Field(
        None,
        description="NTU-Effectiveness calculation details"
    )

    # Heat duty details
    duty_result: Optional[HeatDutyResult] = Field(
        None,
        description="Heat duty calculation details"
    )

    # Fouling
    fouling_result: Optional[FoulingResult] = Field(
        None,
        description="Fouling calculation details"
    )

    # Heat transfer coefficients
    shell_htc_w_m2k: Optional[float] = Field(
        None,
        gt=0,
        description="Shell-side heat transfer coefficient in W/(m^2.K)"
    )
    tube_htc_w_m2k: Optional[float] = Field(
        None,
        gt=0,
        description="Tube-side heat transfer coefficient in W/(m^2.K)"
    )
    overall_htc_w_m2k: Optional[float] = Field(
        None,
        gt=0,
        description="Overall heat transfer coefficient in W/(m^2.K)"
    )

    # Pressure drops (if calculated)
    dp_shell_bar: Optional[float] = Field(
        None,
        ge=0,
        description="Calculated shell-side pressure drop in bar"
    )
    dp_tube_bar: Optional[float] = Field(
        None,
        ge=0,
        description="Calculated tube-side pressure drop in bar"
    )

    # Validity and quality
    overall_validity: ValidityFlag = Field(
        default=ValidityFlag.VALID,
        description="Overall validity of computation result"
    )
    validity_flags: Dict[str, ValidityFlag] = Field(
        default_factory=dict,
        description="Validity flags for individual calculated values"
    )
    warnings: List[ComputationWarning] = Field(
        default_factory=list,
        description="Warnings from computation"
    )

    # Operating point context
    operating_point: Optional[Dict[str, float]] = Field(
        None,
        description="Key operating conditions for context"
    )

    @property
    def is_valid(self) -> bool:
        """Check if result is valid for use."""
        return self.overall_validity in [
            ValidityFlag.VALID,
            ValidityFlag.APPROXIMATE,
        ]


# =============================================================================
# PREDICTION RECORD
# =============================================================================

class ConfidenceInterval(BaseModel):
    """Confidence interval for a prediction."""

    model_config = ConfigDict(frozen=True)

    lower: float = Field(..., description="Lower bound of interval")
    upper: float = Field(..., description="Upper bound of interval")
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Confidence level (e.g., 0.95 for 95%)"
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "ConfidenceInterval":
        """Validate lower <= upper."""
        if self.lower > self.upper:
            raise ValueError("Lower bound must be <= upper bound")
        return self


class FeatureImportance(BaseModel):
    """Feature importance for model explainability."""

    model_config = ConfigDict(frozen=True)

    feature_name: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")
    contribution: float = Field(
        ...,
        description="Contribution to prediction (can be positive or negative)"
    )
    value: Optional[float] = Field(
        None,
        description="Current value of feature"
    )
    baseline: Optional[float] = Field(
        None,
        description="Baseline value for comparison"
    )


class ExplanationPayload(BaseModel):
    """
    Explanation payload for prediction interpretability.

    Supports multiple explanation methods (SHAP, LIME, etc.)
    """

    model_config = ConfigDict(frozen=True)

    method: Literal["shap", "lime", "feature_importance", "rule_based"] = Field(
        ...,
        description="Explanation method used"
    )
    feature_contributions: List[FeatureImportance] = Field(
        default_factory=list,
        description="Feature contributions to prediction"
    )
    explanation_text: Optional[str] = Field(
        None,
        max_length=2000,
        description="Natural language explanation"
    )
    counterfactual: Optional[Dict[str, float]] = Field(
        None,
        description="Counterfactual feature values for different outcome"
    )
    attention_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Attention weights (for attention-based models)"
    )


class PredictionRecord(BaseModel):
    """
    ML model prediction record with full explainability.

    Captures the complete prediction including:
    - Model version and metadata
    - Feature values used
    - Prediction with confidence intervals
    - Explanation payload for interpretability
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "prediction_id": "PRED-2024-00005678",
                    "exchanger_id": "HX-1001",
                    "prediction_type": "cleaning_time",
                    "model_name": "fouling_trajectory_v2",
                    "model_version": "2.1.0",
                    "prediction_value": 42.5,
                    "prediction_unit": "days",
                    "confidence_interval": {
                        "lower": 35.0,
                        "upper": 52.0,
                        "confidence_level": 0.90
                    }
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    prediction_id: str = Field(
        default_factory=lambda: f"PRED-{uuid.uuid4().hex[:12].upper()}",
        description="Unique prediction identifier"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Prediction timestamp"
    )

    # Prediction type
    prediction_type: PredictionType = Field(
        ...,
        description="Type of prediction"
    )

    # Model metadata
    model_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of prediction model"
    )
    model_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of prediction model"
    )
    model_type: ModelType = Field(
        default=ModelType.HYBRID,
        description="Type of model (physics-based, ML, hybrid)"
    )
    model_training_date: Optional[datetime] = Field(
        None,
        description="When model was last trained"
    )
    model_performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Model performance metrics (RMSE, MAE, etc.)"
    )

    # Feature values
    feature_values: Dict[str, float] = Field(
        ...,
        description="Feature values used for prediction"
    )
    feature_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of feature values"
    )

    # Prediction output
    prediction_value: float = Field(
        ...,
        description="Primary prediction value"
    )
    prediction_unit: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unit of prediction value"
    )
    confidence_interval: Optional[ConfidenceInterval] = Field(
        None,
        description="Confidence interval for prediction"
    )
    prediction_percentiles: Optional[Dict[str, float]] = Field(
        None,
        description="Prediction percentiles (e.g., p10, p50, p90)"
    )

    # Probability predictions
    probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Probability (for classification predictions)"
    )
    class_probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Class probabilities for multi-class"
    )

    # Explainability
    explanation_payload: Optional[ExplanationPayload] = Field(
        None,
        description="Explanation of prediction"
    )

    # Validity and quality
    data_quality_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Quality score of input data"
    )
    prediction_quality: Literal[
        "high", "medium", "low", "unreliable"
    ] = Field(
        default="medium",
        description="Quality assessment of prediction"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about prediction"
    )

    # Execution
    execution_time_ms: float = Field(
        ...,
        ge=0,
        description="Prediction execution time in milliseconds"
    )

    # Reference data
    computation_record_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to thermal computation used as input"
    )
    measurement_set_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to measurement set used"
    )

    @classmethod
    def compute_feature_hash(cls, features: Dict[str, float]) -> str:
        """Compute SHA-256 hash of feature values."""
        import json
        content = json.dumps(features, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# EXPORTS
# =============================================================================

COMPUTATION_SCHEMAS = {
    "ComputationType": ComputationType,
    "ComputationStatus": ComputationStatus,
    "ValidityFlag": ValidityFlag,
    "WarningCode": WarningCode,
    "PredictionType": PredictionType,
    "ModelType": ModelType,
    "ComputationWarning": ComputationWarning,
    "ComputationRecord": ComputationRecord,
    "LMTDResult": LMTDResult,
    "NTUResult": NTUResult,
    "HeatDutyResult": HeatDutyResult,
    "FoulingResult": FoulingResult,
    "ThermalComputationResult": ThermalComputationResult,
    "ConfidenceInterval": ConfidenceInterval,
    "FeatureImportance": FeatureImportance,
    "ExplanationPayload": ExplanationPayload,
    "PredictionRecord": PredictionRecord,
}

__all__ = [
    # Enumerations
    "ComputationType",
    "ComputationStatus",
    "ValidityFlag",
    "WarningCode",
    "PredictionType",
    "ModelType",
    # Supporting models
    "ComputationWarning",
    "LMTDResult",
    "NTUResult",
    "HeatDutyResult",
    "FoulingResult",
    "ConfidenceInterval",
    "FeatureImportance",
    "ExplanationPayload",
    # Main schemas
    "ComputationRecord",
    "ThermalComputationResult",
    "PredictionRecord",
    # Export dictionary
    "COMPUTATION_SCHEMAS",
]
