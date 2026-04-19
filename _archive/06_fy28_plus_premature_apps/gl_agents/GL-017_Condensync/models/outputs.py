# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Output Models

Comprehensive output models for condenser diagnostic results, optimization
recommendations, fouling predictions, and maintenance guidance. All outputs
include provenance tracking via SHA-256 hashes for complete audit trails.

Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- EPRI Condenser Performance Guidelines

Zero-Hallucination Guarantee:
All output values are deterministically calculated.
No LLM-generated numeric values in calculation paths.
Complete provenance tracking for regulatory compliance.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from .domain import (
    AlertLevel,
    CleaningMethod,
    CMMSWorkOrder,
    FailureMode,
    FailureSeverity,
    OperatingMode,
    TemperatureDifferential,
)


# ============================================================================
# BASE OUTPUT MODEL WITH PROVENANCE
# ============================================================================

class BaseOutputModel(BaseModel):
    """
    Base output model with provenance tracking.

    All output models inherit from this to ensure consistent
    audit trail and traceability.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    # Identification
    output_id: str = Field(
        ...,
        description="Unique output identifier (UUID)",
    )
    condenser_id: str = Field(
        ...,
        description="Condenser identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output generation timestamp (UTC)",
    )

    # Processing metadata
    agent_id: str = Field(
        "GL-017",
        description="Agent identifier",
    )
    agent_version: str = Field(
        "1.0.0",
        description="Agent version",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        description="Processing time (milliseconds)",
        ge=Decimal("0"),
    )

    # Provenance
    inputs_hash: str = Field(
        ...,
        description="SHA-256 hash of input data",
    )
    computation_hash: str = Field(
        "",
        description="SHA-256 hash of computation (inputs + outputs + params)",
    )

    # Quality
    confidence_score: Decimal = Field(
        Decimal("0.95"),
        description="Confidence in output (0.0-1.0)",
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )
    data_quality_score: Decimal = Field(
        Decimal("1.0"),
        description="Input data quality score (0.0-1.0)",
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Processing warnings",
    )

    def compute_provenance_hash(self, output_data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for complete provenance."""
        combined = {
            "inputs_hash": self.inputs_hash,
            "output_data": output_data,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "timestamp": self.timestamp.isoformat(),
        }
        json_str = json.dumps(combined, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# ============================================================================
# PERFORMANCE OUTPUT MODEL
# ============================================================================

class CondenserPerformanceOutput(BaseOutputModel):
    """
    Comprehensive condenser performance output.

    Contains all key performance indicators (KPIs) for condenser health
    assessment, including cleanliness factor, heat transfer, and
    temperature differentials.

    Example:
        >>> output = CondenserPerformanceOutput(
        ...     output_id="uuid-here",
        ...     condenser_id="COND-001",
        ...     inputs_hash="sha256-hash",
        ...     cleanliness_factor=Decimal("0.82"),
        ...     ua_actual_kw_k=Decimal("25000.0"),
        ...     ttd_c=Decimal("3.5"),
        ...     approach_c=Decimal("8.5"),
        ...     heat_duty_kw=Decimal("450000.0"),
        ... )
    """

    # Cleanliness Factor
    cleanliness_factor: Decimal = Field(
        ...,
        description="Cleanliness Factor (CF) = UA_actual / UA_design",
        ge=Decimal("0.0"),
        le=Decimal("1.5"),
    )
    cleanliness_factor_trend: str = Field(
        "stable",
        description="CF trend (improving, stable, degrading)",
    )
    cleanliness_factor_rate_per_day: Optional[Decimal] = Field(
        None,
        description="CF degradation rate (per day)",
    )

    # Heat Transfer
    ua_actual_kw_k: Decimal = Field(
        ...,
        description="Actual overall heat transfer coefficient-area (kW/K)",
        ge=Decimal("0.0"),
    )
    ua_design_kw_k: Decimal = Field(
        ...,
        description="Design overall heat transfer coefficient-area (kW/K)",
        ge=Decimal("0.0"),
    )
    fouling_resistance_m2_k_kw: Optional[Decimal] = Field(
        None,
        description="Calculated fouling resistance (m2-K/kW)",
    )

    # Temperature Differentials
    ttd_c: Decimal = Field(
        ...,
        description="Terminal Temperature Difference (Celsius)",
        ge=Decimal("-10.0"),
        le=Decimal("30.0"),
    )
    approach_c: Decimal = Field(
        ...,
        description="CW approach temperature (Celsius)",
        ge=Decimal("0.0"),
    )
    lmtd_c: Decimal = Field(
        ...,
        description="Log Mean Temperature Difference (Celsius)",
        ge=Decimal("0.0"),
    )
    cw_temperature_rise_c: Decimal = Field(
        ...,
        description="CW temperature rise (Celsius)",
        ge=Decimal("0.0"),
    )
    subcooling_c: Decimal = Field(
        Decimal("0.0"),
        description="Condensate subcooling (Celsius)",
    )

    # Heat Duty
    heat_duty_kw: Decimal = Field(
        ...,
        description="Heat duty (kW)",
        ge=Decimal("0.0"),
    )
    heat_duty_design_kw: Decimal = Field(
        ...,
        description="Design heat duty (kW)",
        ge=Decimal("0.0"),
    )
    heat_duty_percent: Decimal = Field(
        ...,
        description="Heat duty as percent of design",
        ge=Decimal("0.0"),
    )

    # Vacuum Performance
    condenser_pressure_kpa_abs: Decimal = Field(
        ...,
        description="Condenser pressure (kPa abs)",
    )
    saturation_temp_c: Decimal = Field(
        ...,
        description="Saturation temperature at condenser pressure (Celsius)",
    )
    pressure_deviation_kpa: Optional[Decimal] = Field(
        None,
        description="Deviation from expected pressure (kPa)",
    )

    # Backpressure Impact
    backpressure_mw_impact: Optional[Decimal] = Field(
        None,
        description="MW output impact due to backpressure deviation",
    )
    heat_rate_impact_btu_kwh: Optional[Decimal] = Field(
        None,
        description="Heat rate impact (BTU/kWh)",
    )
    annual_cost_impact_usd: Optional[Decimal] = Field(
        None,
        description="Annual cost impact ($)",
    )

    # Air In-leakage
    air_inleakage_scfm: Optional[Decimal] = Field(
        None,
        description="Air in-leakage (SCFM)",
    )
    air_inleakage_impact_kpa: Optional[Decimal] = Field(
        None,
        description="Backpressure impact from air in-leakage (kPa)",
    )

    # Status
    overall_status: str = Field(
        "NORMAL",
        description="Overall condenser status",
    )
    alert_level: AlertLevel = Field(
        AlertLevel.INFO,
        description="Alert level",
    )
    detected_issues: List[FailureMode] = Field(
        default_factory=list,
        description="Detected failure/degradation modes",
    )

    @computed_field
    @property
    def performance_score(self) -> Decimal:
        """Calculate overall performance score (0-100)."""
        # Simple scoring based on CF and pressure deviation
        cf_score = float(self.cleanliness_factor) * 100
        return Decimal(str(min(100, max(0, cf_score))))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all fields."""
        return self.model_dump(mode="json")

    def to_summary(self) -> Dict[str, Any]:
        """Convert to summary dictionary for dashboards."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "cleanliness_factor": float(self.cleanliness_factor),
            "ttd_c": float(self.ttd_c),
            "pressure_kpa": float(self.condenser_pressure_kpa_abs),
            "heat_duty_mw": float(self.heat_duty_kw / 1000),
            "status": self.overall_status,
            "alert_level": self.alert_level.value,
        }


# ============================================================================
# VACUUM OPTIMIZATION OUTPUT
# ============================================================================

class VacuumOptimizationOutput(BaseOutputModel):
    """
    Vacuum optimization recommendations.

    Provides optimal vacuum setpoint recommendations based on
    ambient conditions, load, and economic factors.

    Example:
        >>> output = VacuumOptimizationOutput(
        ...     output_id="uuid-here",
        ...     condenser_id="COND-001",
        ...     inputs_hash="sha256-hash",
        ...     current_pressure_kpa=Decimal("5.0"),
        ...     recommended_pressure_kpa=Decimal("4.5"),
        ...     expected_mw_gain=Decimal("1.2"),
        ... )
    """

    # Current State
    current_pressure_kpa: Decimal = Field(
        ...,
        description="Current condenser pressure (kPa abs)",
    )
    current_cw_flow_kg_s: Decimal = Field(
        ...,
        description="Current CW flow rate (kg/s)",
    )
    current_cw_pumps_running: int = Field(
        ...,
        description="Number of CW pumps currently running",
    )

    # Recommendations
    recommended_pressure_kpa: Decimal = Field(
        ...,
        description="Recommended condenser pressure setpoint (kPa abs)",
    )
    recommended_cw_flow_kg_s: Decimal = Field(
        ...,
        description="Recommended CW flow rate (kg/s)",
    )
    recommended_cw_pumps: int = Field(
        ...,
        description="Recommended number of CW pumps",
    )
    vacuum_control_mode: str = Field(
        "automatic",
        description="Recommended vacuum control mode",
    )

    # Expected Benefits
    expected_mw_gain: Decimal = Field(
        ...,
        description="Expected MW output improvement",
    )
    expected_heat_rate_improvement_btu_kwh: Decimal = Field(
        Decimal("0"),
        description="Expected heat rate improvement (BTU/kWh)",
    )
    expected_cw_pump_power_change_kw: Decimal = Field(
        Decimal("0"),
        description="Expected CW pump power change (kW)",
    )
    net_benefit_kw: Decimal = Field(
        ...,
        description="Net power benefit (MW gain - pump power change)",
    )
    annual_savings_usd: Optional[Decimal] = Field(
        None,
        description="Estimated annual savings ($)",
    )

    # Economic Analysis
    power_price_usd_mwh: Optional[Decimal] = Field(
        None,
        description="Power price used in analysis ($/MWh)",
    )
    fuel_cost_usd_mmbtu: Optional[Decimal] = Field(
        None,
        description="Fuel cost used in analysis ($/MMBtu)",
    )

    # Constraints Considered
    ambient_temp_c: Optional[Decimal] = Field(
        None,
        description="Ambient temperature considered (Celsius)",
    )
    unit_load_mw: Optional[Decimal] = Field(
        None,
        description="Unit load considered (MW)",
    )
    within_constraints: bool = Field(
        True,
        description="Recommendation within operating constraints",
    )
    constraint_violations: List[str] = Field(
        default_factory=list,
        description="Any constraint violations",
    )

    # Action Items
    actions: List[str] = Field(
        default_factory=list,
        description="Recommended operator actions",
    )
    implementation_priority: str = Field(
        "medium",
        description="Implementation priority (low, medium, high)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")


# ============================================================================
# FOULING PREDICTION OUTPUT
# ============================================================================

class FoulingPredictionOutput(BaseOutputModel):
    """
    Fouling prediction and cleaning ROI analysis.

    Predicts condenser fouling trajectory and recommends optimal
    cleaning timing based on economic analysis.

    Example:
        >>> output = FoulingPredictionOutput(
        ...     output_id="uuid-here",
        ...     condenser_id="COND-001",
        ...     inputs_hash="sha256-hash",
        ...     current_cf=Decimal("0.82"),
        ...     predicted_cf_30d=Decimal("0.78"),
        ...     days_to_cleaning_threshold=45,
        ...     cleaning_roi_percent=Decimal("250.0"),
        ... )
    """

    # Current State
    current_cf: Decimal = Field(
        ...,
        description="Current cleanliness factor",
    )
    current_cf_timestamp: datetime = Field(
        ...,
        description="Timestamp of current CF measurement",
    )
    cf_degradation_rate_per_day: Decimal = Field(
        ...,
        description="CF degradation rate (per day)",
    )

    # Predictions
    predicted_cf_7d: Decimal = Field(
        ...,
        description="Predicted CF in 7 days",
    )
    predicted_cf_14d: Decimal = Field(
        ...,
        description="Predicted CF in 14 days",
    )
    predicted_cf_30d: Decimal = Field(
        ...,
        description="Predicted CF in 30 days",
    )
    predicted_cf_90d: Optional[Decimal] = Field(
        None,
        description="Predicted CF in 90 days",
    )

    # Cleaning Threshold
    cleaning_threshold_cf: Decimal = Field(
        Decimal("0.75"),
        description="CF threshold triggering cleaning recommendation",
    )
    days_to_cleaning_threshold: int = Field(
        ...,
        description="Days until CF reaches cleaning threshold",
    )
    recommended_cleaning_date: Optional[datetime] = Field(
        None,
        description="Recommended cleaning date",
    )

    # Fouling Analysis
    fouling_type: str = Field(
        "biological",
        description="Predicted fouling type (biological, scale, debris)",
    )
    fouling_severity: FailureSeverity = Field(
        FailureSeverity.LOW,
        description="Fouling severity level",
    )
    fouling_rate_category: str = Field(
        "normal",
        description="Fouling rate category (low, normal, high, excessive)",
    )

    # MW Impact Projection
    current_mw_loss: Decimal = Field(
        ...,
        description="Current MW loss due to fouling",
    )
    projected_mw_loss_30d: Decimal = Field(
        ...,
        description="Projected MW loss in 30 days",
    )
    cumulative_energy_loss_mwh: Decimal = Field(
        ...,
        description="Cumulative energy loss until cleaning (MWh)",
    )
    cumulative_cost_usd: Decimal = Field(
        ...,
        description="Cumulative cost until cleaning ($)",
    )

    # Cleaning ROI Analysis
    recommended_cleaning_method: CleaningMethod = Field(
        ...,
        description="Recommended cleaning method",
    )
    cleaning_cost_usd: Decimal = Field(
        ...,
        description="Estimated cleaning cost ($)",
    )
    expected_cf_recovery: Decimal = Field(
        ...,
        description="Expected CF after cleaning",
    )
    cleaning_benefit_usd: Decimal = Field(
        ...,
        description="Economic benefit from cleaning ($)",
    )
    cleaning_roi_percent: Decimal = Field(
        ...,
        description="Cleaning ROI (%)",
    )
    payback_period_days: Decimal = Field(
        ...,
        description="Cleaning payback period (days)",
    )

    # Online vs Offline Comparison
    online_cleaning_available: bool = Field(
        False,
        description="Online cleaning system available",
    )
    online_cleaning_sufficient: bool = Field(
        False,
        description="Online cleaning sufficient for current fouling",
    )
    offline_cleaning_required: bool = Field(
        False,
        description="Offline cleaning required",
    )

    # Historical Context
    last_cleaning_date: Optional[datetime] = Field(
        None,
        description="Date of last cleaning",
    )
    last_cleaning_method: Optional[CleaningMethod] = Field(
        None,
        description="Method of last cleaning",
    )
    days_since_last_cleaning: Optional[int] = Field(
        None,
        description="Days since last cleaning",
    )
    average_cleaning_interval_days: Optional[int] = Field(
        None,
        description="Average cleaning interval (days)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")


# ============================================================================
# MAINTENANCE RECOMMENDATION OUTPUT
# ============================================================================

class MaintenanceRecommendation(BaseOutputModel):
    """
    Maintenance recommendation with evidence and CMMS integration.

    Provides actionable maintenance recommendations with supporting
    evidence and optional CMMS work order creation data.

    Example:
        >>> output = MaintenanceRecommendation(
        ...     output_id="uuid-here",
        ...     condenser_id="COND-001",
        ...     inputs_hash="sha256-hash",
        ...     action="Perform online tube cleaning",
        ...     severity=FailureSeverity.MEDIUM,
        ...     priority=2,
        ...     evidence=["CF declined 5% in 7 days", "TTD increased 0.8 C"],
        ... )
    """

    # Recommendation
    action: str = Field(
        ...,
        description="Recommended maintenance action",
    )
    action_category: str = Field(
        "cleaning",
        description="Action category (cleaning, inspection, repair, replacement)",
    )
    severity: FailureSeverity = Field(
        ...,
        description="Issue severity",
    )
    priority: int = Field(
        ...,
        description="Priority (1=highest, 5=lowest)",
        ge=1,
        le=5,
    )
    alert_level: AlertLevel = Field(
        AlertLevel.WARNING,
        description="Alert level",
    )

    # Evidence
    evidence: List[str] = Field(
        ...,
        description="Supporting evidence for recommendation",
        min_length=1,
    )
    failure_modes_detected: List[FailureMode] = Field(
        default_factory=list,
        description="Detected failure modes",
    )
    confidence_in_diagnosis: Decimal = Field(
        Decimal("0.9"),
        description="Confidence in failure mode diagnosis",
    )

    # Impact Assessment
    current_impact_mw: Decimal = Field(
        Decimal("0"),
        description="Current MW impact if issue unresolved",
    )
    projected_impact_mw: Decimal = Field(
        Decimal("0"),
        description="Projected MW impact if action delayed",
    )
    daily_cost_usd: Decimal = Field(
        Decimal("0"),
        description="Daily cost of inaction ($)",
    )
    risk_of_escalation: str = Field(
        "low",
        description="Risk of issue escalation (low, medium, high)",
    )

    # Timing
    recommended_completion_date: Optional[datetime] = Field(
        None,
        description="Recommended completion date",
    )
    requires_outage: bool = Field(
        False,
        description="Requires unit outage",
    )
    estimated_duration_hours: Decimal = Field(
        Decimal("4.0"),
        description="Estimated maintenance duration (hours)",
    )

    # Cost Estimate
    estimated_labor_hours: Decimal = Field(
        Decimal("0"),
        description="Estimated labor hours",
    )
    estimated_material_cost_usd: Decimal = Field(
        Decimal("0"),
        description="Estimated material cost ($)",
    )
    estimated_total_cost_usd: Decimal = Field(
        Decimal("0"),
        description="Estimated total cost ($)",
    )

    # CMMS Integration
    cmms_work_order: Optional[Dict[str, Any]] = Field(
        None,
        description="CMMS work order data",
    )
    equipment_tag: Optional[str] = Field(
        None,
        description="Equipment tag for CMMS",
    )
    failure_code: Optional[str] = Field(
        None,
        description="Standard failure code",
    )

    # Follow-up
    follow_up_required: bool = Field(
        False,
        description="Follow-up inspection required",
    )
    follow_up_interval_days: Optional[int] = Field(
        None,
        description="Follow-up inspection interval (days)",
    )
    related_recommendations: List[str] = Field(
        default_factory=list,
        description="Related recommendation IDs",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    def to_cmms_format(self) -> Dict[str, Any]:
        """Convert to CMMS work order format."""
        return {
            "equipment_id": self.equipment_tag or self.condenser_id,
            "description": self.action,
            "priority": self.priority,
            "work_type": "CM" if self.severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL] else "PM",
            "estimated_hours": float(self.estimated_labor_hours),
            "estimated_cost": float(self.estimated_total_cost_usd),
            "target_date": self.recommended_completion_date.isoformat() if self.recommended_completion_date else None,
            "failure_code": self.failure_code,
            "notes": "; ".join(self.evidence),
        }


# ============================================================================
# EXPLAINABILITY OUTPUT
# ============================================================================

class FeatureContribution(BaseModel):
    """Single feature contribution to model output."""

    model_config = ConfigDict(frozen=True)

    feature_name: str = Field(..., description="Feature name")
    feature_value: Decimal = Field(..., description="Feature value")
    contribution: Decimal = Field(..., description="SHAP contribution value")
    contribution_percent: Decimal = Field(..., description="Contribution as percent of total")
    direction: str = Field(..., description="Direction of impact (positive, negative)")
    explanation: str = Field(..., description="Human-readable explanation")


class Counterfactual(BaseModel):
    """Counterfactual scenario for what-if analysis."""

    model_config = ConfigDict(frozen=True)

    scenario_name: str = Field(..., description="Scenario name")
    changes: Dict[str, Tuple[Decimal, Decimal]] = Field(
        ...,
        description="Feature changes (feature: (original, counterfactual))",
    )
    predicted_outcome: Decimal = Field(..., description="Predicted outcome under scenario")
    outcome_change: Decimal = Field(..., description="Change from baseline outcome")
    explanation: str = Field(..., description="Scenario explanation")


class ExplainabilityOutput(BaseOutputModel):
    """
    Explainability output with SHAP values and counterfactuals.

    Provides transparent explanations of diagnostic and optimization
    results using SHAP-compatible feature attributions.

    Example:
        >>> output = ExplainabilityOutput(
        ...     output_id="uuid-here",
        ...     condenser_id="COND-001",
        ...     inputs_hash="sha256-hash",
        ...     prediction_type="cleanliness_factor",
        ...     predicted_value=Decimal("0.82"),
        ...     feature_contributions=[...],
        ...     physics_narrative="The condenser CF of 0.82...",
        ... )
    """

    # Prediction Context
    prediction_type: str = Field(
        ...,
        description="Type of prediction being explained",
    )
    predicted_value: Decimal = Field(
        ...,
        description="Predicted/calculated value",
    )
    baseline_value: Decimal = Field(
        ...,
        description="Baseline value for SHAP",
    )

    # SHAP Values
    feature_contributions: List[FeatureContribution] = Field(
        ...,
        description="SHAP feature contributions",
    )
    top_positive_features: List[str] = Field(
        default_factory=list,
        description="Top features contributing positively",
    )
    top_negative_features: List[str] = Field(
        default_factory=list,
        description="Top features contributing negatively",
    )

    # Physics Narrative
    physics_narrative: str = Field(
        ...,
        description="Physics-based explanation narrative",
    )
    governing_equations: List[str] = Field(
        default_factory=list,
        description="Governing equations used",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Key assumptions in analysis",
    )

    # Counterfactuals
    counterfactuals: List[Counterfactual] = Field(
        default_factory=list,
        description="What-if counterfactual scenarios",
    )

    # Uncertainty
    uncertainty_range: Optional[Tuple[Decimal, Decimal]] = Field(
        None,
        description="Uncertainty range (low, high)",
    )
    sensitivity_analysis: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Sensitivity of output to inputs",
    )

    # References
    calculation_method: str = Field(
        "HEI_Standard",
        description="Calculation method used",
    )
    reference_standards: List[str] = Field(
        default_factory=list,
        description="Reference standards applied",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    def to_operator_summary(self) -> str:
        """Generate operator-friendly summary."""
        lines = [
            f"Analysis: {self.prediction_type}",
            f"Result: {self.predicted_value}",
            "",
            "Key Factors:",
        ]
        for fc in self.feature_contributions[:5]:
            lines.append(f"  - {fc.feature_name}: {fc.explanation}")
        lines.append("")
        lines.append(self.physics_narrative)
        return "\n".join(lines)


# ============================================================================
# DIAGNOSTIC REPORT OUTPUT
# ============================================================================

class CondenserDiagnosticReport(BaseOutputModel):
    """
    Comprehensive diagnostic report combining all analyses.

    Aggregates performance, optimization, fouling, and maintenance
    outputs into a single report for operator dashboards and
    historical logging.
    """

    # Component Outputs
    performance: CondenserPerformanceOutput = Field(
        ...,
        description="Performance analysis output",
    )
    vacuum_optimization: Optional[VacuumOptimizationOutput] = Field(
        None,
        description="Vacuum optimization output",
    )
    fouling_prediction: Optional[FoulingPredictionOutput] = Field(
        None,
        description="Fouling prediction output",
    )
    maintenance_recommendations: List[MaintenanceRecommendation] = Field(
        default_factory=list,
        description="Maintenance recommendations",
    )
    explainability: Optional[ExplainabilityOutput] = Field(
        None,
        description="Explainability output",
    )

    # Summary
    overall_health_score: Decimal = Field(
        ...,
        description="Overall condenser health score (0-100)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    alert_count: int = Field(
        0,
        description="Number of active alerts",
    )
    critical_issues: List[str] = Field(
        default_factory=list,
        description="Critical issues requiring immediate attention",
    )

    # Trend Indicators
    performance_trend: str = Field(
        "stable",
        description="Performance trend (improving, stable, degrading)",
    )
    days_to_intervention: Optional[int] = Field(
        None,
        description="Days until intervention likely needed",
    )

    # Economic Summary
    current_loss_mw: Decimal = Field(
        Decimal("0"),
        description="Current MW loss from all factors",
    )
    potential_recovery_mw: Decimal = Field(
        Decimal("0"),
        description="Potential MW recovery if all actions taken",
    )
    estimated_annual_savings_usd: Optional[Decimal] = Field(
        None,
        description="Estimated annual savings from recommendations ($)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    def to_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary."""
        return {
            "condenser_id": self.condenser_id,
            "timestamp": self.timestamp.isoformat(),
            "health_score": float(self.overall_health_score),
            "cleanliness_factor": float(self.performance.cleanliness_factor),
            "current_loss_mw": float(self.current_loss_mw),
            "potential_recovery_mw": float(self.potential_recovery_mw),
            "alert_count": self.alert_count,
            "critical_issues": len(self.critical_issues),
            "performance_trend": self.performance_trend,
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Base
    "BaseOutputModel",
    # Performance
    "CondenserPerformanceOutput",
    # Optimization
    "VacuumOptimizationOutput",
    # Fouling
    "FoulingPredictionOutput",
    # Maintenance
    "MaintenanceRecommendation",
    # Explainability
    "FeatureContribution",
    "Counterfactual",
    "ExplainabilityOutput",
    # Report
    "CondenserDiagnosticReport",
]
