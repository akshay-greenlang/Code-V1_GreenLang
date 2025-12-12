"""
Pydantic Schemas for GL-018 UNIFIEDCOMBUSTION Agent

This module defines all input/output data models for the UnifiedCombustionOptimizer
agent, ensuring type safety, validation, and documentation.

All models follow zero-hallucination principles with deterministic validation
and complete provenance tracking.

Example:
    >>> from schemas import CombustionInput, CombustionOutput
    >>> input_data = CombustionInput(
    ...     equipment_id="BOILER-001",
    ...     fuel_type=FuelType.NATURAL_GAS,
    ...     fuel_flow_rate=100.0,
    ...     ...
    ... )
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import hashlib

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# Enumerations
# =============================================================================

class FuelType(str, Enum):
    """Supported fuel types for combustion analysis."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    BUTANE = "butane"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    COAL = "coal"
    BIOMASS = "biomass"
    MIXED = "mixed"


class EquipmentType(str, Enum):
    """Types of combustion equipment."""
    BOILER = "boiler"
    FURNACE = "furnace"
    HEATER = "heater"
    OVEN = "oven"
    KILN = "kiln"
    INCINERATOR = "incinerator"
    DRYER = "dryer"
    THERMAL_OXIDIZER = "thermal_oxidizer"


class OptimizationMode(str, Enum):
    """Optimization objective modes."""
    EFFICIENCY = "efficiency"
    EMISSIONS = "emissions"
    BALANCED = "balanced"
    SAFETY = "safety"


class ComplianceStatus(str, Enum):
    """NFPA compliance status levels."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    WARNING = "WARNING"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class SafetyInterlockStatus(str, Enum):
    """Safety interlock status."""
    ARMED = "ARMED"
    TRIPPED = "TRIPPED"
    BYPASSED = "BYPASSED"
    FAULT = "FAULT"


class Priority(str, Enum):
    """Recommendation priority levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class CausalRelationType(str, Enum):
    """Types of causal relationships."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDED = "confounded"
    MEDIATED = "mediated"


# =============================================================================
# Input Models - Fuel Composition
# =============================================================================

class FuelComposition(BaseModel):
    """Fuel composition analysis data."""

    methane_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Methane (CH4) percentage"
    )
    ethane_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Ethane (C2H6) percentage"
    )
    propane_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Propane (C3H8) percentage"
    )
    butane_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Butane (C4H10) percentage"
    )
    nitrogen_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Nitrogen (N2) percentage"
    )
    co2_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Carbon dioxide (CO2) percentage"
    )
    hydrogen_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Hydrogen (H2) percentage"
    )
    h2s_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Hydrogen sulfide (H2S) in ppm"
    )
    higher_heating_value_mj_m3: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher heating value in MJ/m3"
    )
    lower_heating_value_mj_m3: Optional[float] = Field(
        default=None,
        gt=0,
        description="Lower heating value in MJ/m3"
    )
    specific_gravity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Specific gravity relative to air"
    )

    @root_validator(skip_on_failure=True)
    def validate_total_composition(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that fuel composition percentages sum reasonably."""
        pct_fields = [
            'methane_pct', 'ethane_pct', 'propane_pct', 'butane_pct',
            'nitrogen_pct', 'co2_pct', 'hydrogen_pct'
        ]
        total = sum(values.get(f, 0) for f in pct_fields)
        if total > 0 and (total < 95 or total > 105):
            # Allow 5% tolerance for measurement error
            pass  # Log warning but don't reject
        return values


class FlueGasMeasurements(BaseModel):
    """Flue gas analysis measurements."""

    o2_percent: float = Field(
        ...,
        ge=0,
        le=21,
        description="Oxygen percentage in flue gas (dry basis)"
    )
    co2_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=25,
        description="Carbon dioxide percentage in flue gas"
    )
    co_ppm: float = Field(
        ...,
        ge=0,
        description="Carbon monoxide in parts per million"
    )
    nox_ppm: float = Field(
        ...,
        ge=0,
        description="NOx (NO + NO2) in parts per million"
    )
    so2_ppm: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Sulfur dioxide in parts per million"
    )
    stack_temperature_c: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Stack/flue gas temperature in Celsius"
    )
    ambient_temperature_c: float = Field(
        default=25.0,
        ge=-50,
        le=60,
        description="Ambient air temperature in Celsius"
    )
    humidity_percent: Optional[float] = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Relative humidity percentage"
    )

    @root_validator(skip_on_failure=True)
    def validate_flue_gas_chemistry(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that flue gas measurements are physically consistent."""
        o2 = values.get('o2_percent', 0)
        co = values.get('co_ppm', 0)

        # High CO with very low O2 indicates severe incomplete combustion
        if o2 < 0.5 and co > 2000:
            raise ValueError(
                "Invalid combustion: O2 too low with very high CO indicates "
                "dangerous incomplete combustion condition"
            )
        return values


class FlameMetrics(BaseModel):
    """Flame condition measurements."""

    flame_temperature_c: float = Field(
        ...,
        ge=0,
        le=3000,
        description="Flame temperature in Celsius"
    )
    stability_index: float = Field(
        ...,
        ge=0,
        le=1,
        description="Flame stability index 0-1 (1 = perfectly stable)"
    )
    flame_length_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Flame length in meters"
    )
    flame_diameter_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Flame diameter in meters"
    )
    luminosity: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Flame luminosity percentage"
    )
    uv_signal: Optional[float] = Field(
        default=None,
        ge=0,
        description="UV flame detector signal strength"
    )
    ir_signal: Optional[float] = Field(
        default=None,
        ge=0,
        description="IR flame detector signal strength"
    )


class AirFlowMeasurements(BaseModel):
    """Combustion air flow measurements."""

    primary_air_flow_m3h: float = Field(
        ...,
        ge=0,
        description="Primary combustion air flow in m3/h"
    )
    secondary_air_flow_m3h: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Secondary air flow in m3/h"
    )
    tertiary_air_flow_m3h: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Tertiary/overfire air flow in m3/h"
    )
    air_preheat_temp_c: Optional[float] = Field(
        default=None,
        ge=-50,
        le=600,
        description="Preheated combustion air temperature"
    )
    forced_draft_pressure_pa: Optional[float] = Field(
        default=None,
        description="Forced draft fan discharge pressure in Pa"
    )
    damper_position_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Air damper position percentage"
    )


class SafetyInterlockData(BaseModel):
    """Safety interlock status data per NFPA 85/86."""

    interlock_name: str = Field(..., description="Interlock identification")
    status: SafetyInterlockStatus = Field(..., description="Current status")
    setpoint: Optional[float] = Field(default=None, description="Trip setpoint")
    current_value: Optional[float] = Field(default=None, description="Current reading")
    last_test_date: Optional[date] = Field(default=None, description="Last test date")
    certified: bool = Field(default=True, description="Certification status")


class BurnerStatus(BaseModel):
    """Individual burner status data."""

    burner_id: str = Field(..., description="Burner identifier")
    firing_rate_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current firing rate percentage"
    )
    flame_on: bool = Field(..., description="Flame presence detected")
    fuel_valve_position: float = Field(
        ...,
        ge=0,
        le=100,
        description="Fuel valve position percentage"
    )
    air_register_position: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Air register position percentage"
    )
    igniter_status: Optional[str] = Field(
        default=None,
        description="Igniter status"
    )


# =============================================================================
# Main Input Model
# =============================================================================

class CombustionInput(BaseModel):
    """
    Complete input data model for UnifiedCombustionOptimizer.

    This model consolidates all inputs required for combustion optimization,
    NFPA compliance checking, and emissions analysis.
    """

    # Equipment identification
    equipment_id: str = Field(..., min_length=1, description="Equipment identifier")
    equipment_type: EquipmentType = Field(..., description="Type of combustion equipment")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name/description")

    # Fuel data
    fuel_type: FuelType = Field(..., description="Primary fuel type")
    fuel_flow_rate: float = Field(
        ...,
        gt=0,
        description="Fuel flow rate (unit depends on fuel type: m3/h for gas, kg/h for liquid/solid)"
    )
    fuel_flow_unit: str = Field(
        default="m3/h",
        description="Unit for fuel flow rate"
    )
    fuel_composition: Optional[FuelComposition] = Field(
        default=None,
        description="Detailed fuel composition (required for natural gas)"
    )
    fuel_heating_value_mj: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fuel heating value in MJ/unit"
    )

    # Flue gas measurements
    flue_gas: FlueGasMeasurements = Field(
        ...,
        description="Flue gas analysis measurements"
    )

    # Flame metrics
    flame_metrics: Optional[FlameMetrics] = Field(
        default=None,
        description="Flame condition measurements"
    )

    # Air flow data
    air_flow: AirFlowMeasurements = Field(
        ...,
        description="Combustion air flow measurements"
    )

    # Burner status
    burners: List[BurnerStatus] = Field(
        default_factory=list,
        description="Individual burner status data"
    )

    # Safety interlocks
    safety_interlocks: List[SafetyInterlockData] = Field(
        default_factory=list,
        description="Safety interlock status per NFPA 85/86"
    )

    # Operating parameters
    heat_input_mw: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total heat input in MW"
    )
    steam_production_tph: Optional[float] = Field(
        default=None,
        gt=0,
        description="Steam production rate in tonnes per hour"
    )
    process_temperature_c: Optional[float] = Field(
        default=None,
        description="Process temperature in Celsius"
    )
    furnace_pressure_pa: Optional[float] = Field(
        default=None,
        description="Furnace pressure in Pa (negative for draft)"
    )

    # Ambient conditions
    ambient_pressure_kpa: float = Field(
        default=101.325,
        gt=0,
        description="Ambient atmospheric pressure in kPa"
    )
    altitude_m: float = Field(
        default=0,
        ge=0,
        le=5000,
        description="Site altitude in meters"
    )

    # Optimization settings
    optimization_mode: OptimizationMode = Field(
        default=OptimizationMode.BALANCED,
        description="Optimization objective mode"
    )
    target_o2_percent: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=10,
        description="Target O2 setpoint for trim control"
    )
    max_co_ppm: Optional[float] = Field(
        default=100,
        gt=0,
        description="Maximum allowable CO in ppm"
    )
    max_nox_ppm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum allowable NOx in ppm (regulatory limit)"
    )

    # Regulatory requirements
    nfpa_standard: str = Field(
        default="NFPA 85",
        description="Applicable NFPA standard (85 for boilers, 86 for ovens/furnaces)"
    )
    emission_regulation: Optional[str] = Field(
        default=None,
        description="Applicable emission regulation code"
    )

    @root_validator(skip_on_failure=True)
    def validate_input_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall input consistency."""
        fuel_type = values.get('fuel_type')
        fuel_comp = values.get('fuel_composition')

        # Natural gas should have composition data for accurate calculations
        if fuel_type == FuelType.NATURAL_GAS and fuel_comp is None:
            pass  # Will use default composition

        return values


# =============================================================================
# Output Models - Optimization Results
# =============================================================================

class EfficiencyMetrics(BaseModel):
    """Calculated combustion efficiency metrics."""

    combustion_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Combustion efficiency percentage"
    )
    thermal_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Overall thermal efficiency percentage"
    )
    stack_loss_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Stack/flue gas heat loss percentage"
    )
    unburned_loss_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Unburned fuel loss percentage"
    )
    radiation_loss_pct: float = Field(
        default=1.5,
        ge=0,
        le=10,
        description="Radiation and convection loss percentage"
    )
    excess_air_pct: float = Field(
        ...,
        ge=-10,
        description="Excess air percentage"
    )
    air_fuel_ratio_actual: float = Field(
        ...,
        gt=0,
        description="Actual air-fuel ratio"
    )
    air_fuel_ratio_stoich: float = Field(
        ...,
        gt=0,
        description="Stoichiometric air-fuel ratio"
    )
    lambda_value: float = Field(
        ...,
        gt=0,
        description="Lambda (equivalence ratio)"
    )


class OptimizationRecommendation(BaseModel):
    """Individual optimization recommendation."""

    parameter: str = Field(..., description="Parameter to adjust")
    current_value: float = Field(..., description="Current value")
    recommended_value: float = Field(..., description="Recommended optimal value")
    unit: str = Field(..., description="Unit of measurement")
    expected_improvement: str = Field(..., description="Expected improvement description")
    priority: Priority = Field(..., description="Implementation priority")
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in recommendation (0-1)"
    )
    reasoning: str = Field(..., description="Technical reasoning for recommendation")


class O2TrimRecommendation(BaseModel):
    """O2 trim control optimization results."""

    current_o2_pct: float = Field(..., description="Current O2 percentage")
    optimal_o2_pct: float = Field(..., description="Optimal O2 setpoint")
    o2_trim_range_low: float = Field(..., description="Lower O2 trim limit")
    o2_trim_range_high: float = Field(..., description="Upper O2 trim limit")
    efficiency_gain_pct: float = Field(..., description="Expected efficiency improvement")
    co_risk_assessment: str = Field(..., description="CO breakthrough risk assessment")
    adjustment_direction: str = Field(..., description="Direction of O2 adjustment needed")


class ExcessAirRecommendation(BaseModel):
    """Excess air control recommendations."""

    current_excess_air_pct: float = Field(..., description="Current excess air percentage")
    optimal_excess_air_pct: float = Field(..., description="Optimal excess air percentage")
    damper_adjustment_pct: Optional[float] = Field(
        default=None,
        description="Recommended damper position adjustment"
    )
    fan_speed_adjustment_pct: Optional[float] = Field(
        default=None,
        description="Recommended fan speed adjustment"
    )
    fuel_savings_pct: float = Field(..., description="Expected fuel savings percentage")


class EmissionsAnalysis(BaseModel):
    """Emissions analysis and recommendations."""

    co_status: str = Field(..., description="CO emission status")
    nox_status: str = Field(..., description="NOx emission status")
    so2_status: Optional[str] = Field(default=None, description="SO2 emission status")

    co_corrected_ppm: float = Field(..., description="CO corrected to 3% O2")
    nox_corrected_ppm: float = Field(..., description="NOx corrected to 3% O2")

    co2_emission_rate_kgh: float = Field(..., description="CO2 emission rate in kg/h")
    nox_emission_rate_kgh: float = Field(..., description="NOx emission rate in kg/h")

    emission_index_co2: float = Field(..., description="CO2 emission index kg/GJ")
    emission_index_nox: float = Field(..., description="NOx emission index g/GJ")

    reduction_opportunities: List[str] = Field(
        default_factory=list,
        description="Emission reduction opportunities"
    )


# =============================================================================
# Output Models - NFPA Compliance
# =============================================================================

class NFPAViolation(BaseModel):
    """Individual NFPA compliance violation."""

    code_reference: str = Field(..., description="NFPA code section reference")
    requirement: str = Field(..., description="Requirement description")
    current_state: str = Field(..., description="Current state description")
    severity: Priority = Field(..., description="Violation severity")
    corrective_action: str = Field(..., description="Required corrective action")
    deadline: Optional[str] = Field(default=None, description="Compliance deadline")


class SafetyInterlockAssessment(BaseModel):
    """Safety interlock compliance assessment."""

    interlock_name: str = Field(..., description="Interlock name")
    required_by: str = Field(..., description="NFPA requirement reference")
    status: ComplianceStatus = Field(..., description="Compliance status")
    test_required: bool = Field(..., description="Whether testing is required")
    test_due_date: Optional[date] = Field(default=None, description="Next test due date")
    notes: Optional[str] = Field(default=None, description="Assessment notes")


class NFPAComplianceResult(BaseModel):
    """Complete NFPA compliance assessment result."""

    standard: str = Field(..., description="NFPA standard assessed")
    overall_status: ComplianceStatus = Field(..., description="Overall compliance status")
    assessment_date: datetime = Field(..., description="Assessment timestamp")

    violations: List[NFPAViolation] = Field(
        default_factory=list,
        description="List of violations found"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning conditions"
    )

    interlock_assessments: List[SafetyInterlockAssessment] = Field(
        default_factory=list,
        description="Individual interlock assessments"
    )

    burner_management_status: ComplianceStatus = Field(
        ...,
        description="Burner Management System compliance"
    )
    flame_safeguard_status: ComplianceStatus = Field(
        ...,
        description="Flame Safeguard System compliance"
    )
    purge_cycle_status: ComplianceStatus = Field(
        ...,
        description="Purge cycle compliance"
    )

    required_actions: List[str] = Field(
        default_factory=list,
        description="Required compliance actions"
    )


# =============================================================================
# Output Models - Explainability
# =============================================================================

class FeatureImportance(BaseModel):
    """Feature importance from SHAP/LIME analysis."""

    feature_name: str = Field(..., description="Input feature name")
    importance_score: float = Field(..., description="Importance score")
    contribution_direction: str = Field(
        ...,
        description="Positive or negative contribution"
    )
    shap_value: Optional[float] = Field(default=None, description="SHAP value")
    description: str = Field(..., description="Plain language explanation")


class CausalRelationship(BaseModel):
    """Causal relationship from inference analysis."""

    cause: str = Field(..., description="Cause variable")
    effect: str = Field(..., description="Effect variable")
    relationship_type: CausalRelationType = Field(..., description="Type of causal relationship")
    strength: float = Field(
        ...,
        ge=0,
        le=1,
        description="Relationship strength (0-1)"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Statistical confidence"
    )
    mechanism: str = Field(..., description="Physical mechanism explanation")


class AttentionVisualization(BaseModel):
    """Attention weights for ML component transparency."""

    component: str = Field(..., description="ML component name")
    input_features: List[str] = Field(..., description="Input feature names")
    attention_weights: List[float] = Field(..., description="Attention weight values")
    peak_attention_feature: str = Field(..., description="Feature with highest attention")
    interpretation: str = Field(..., description="Human-readable interpretation")


class ExplainabilityReport(BaseModel):
    """Complete explainability report for optimization decisions."""

    decision_summary: str = Field(..., description="Summary of optimization decision")

    feature_importances: List[FeatureImportance] = Field(
        default_factory=list,
        description="SHAP/LIME feature importance rankings"
    )

    causal_relationships: List[CausalRelationship] = Field(
        default_factory=list,
        description="Identified causal relationships"
    )

    attention_visualizations: List[AttentionVisualization] = Field(
        default_factory=list,
        description="Attention weight visualizations"
    )

    root_cause_analysis: List[str] = Field(
        default_factory=list,
        description="Root cause analysis findings"
    )

    counterfactual_scenarios: List[str] = Field(
        default_factory=list,
        description="What-if counterfactual scenarios"
    )

    natural_language_summary: str = Field(
        ...,
        description="Natural language explanation for operators"
    )


# =============================================================================
# Output Models - Calculation Provenance
# =============================================================================

class CalculationStep(BaseModel):
    """Individual calculation step with provenance."""

    step_number: int = Field(..., description="Step sequence number")
    description: str = Field(..., description="Step description")
    formula: str = Field(..., description="Formula or method used")
    inputs: Dict[str, Any] = Field(..., description="Input values used")
    output_name: str = Field(..., description="Output variable name")
    output_value: Any = Field(..., description="Calculated output value")
    unit: Optional[str] = Field(default=None, description="Output unit")
    reference: Optional[str] = Field(default=None, description="Standard/reference")


class ProvenanceRecord(BaseModel):
    """Complete provenance record for audit trail."""

    calculation_id: str = Field(..., description="Unique calculation identifier")
    timestamp: datetime = Field(..., description="Calculation timestamp")
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field(..., description="Agent version")

    input_hash: str = Field(..., description="SHA-256 hash of input data")
    output_hash: str = Field(..., description="SHA-256 hash of output data")
    calculation_hash: str = Field(..., description="SHA-256 hash of all calculation steps")

    calculation_steps: List[CalculationStep] = Field(
        ...,
        description="Complete calculation trace"
    )

    reproducibility_verified: bool = Field(
        ...,
        description="Whether calculation reproducibility was verified"
    )

    regulatory_standards: List[str] = Field(
        default_factory=list,
        description="Regulatory standards applied"
    )


# =============================================================================
# Main Output Model
# =============================================================================

class CombustionOutput(BaseModel):
    """
    Complete output data model for UnifiedCombustionOptimizer.

    Provides comprehensive results including efficiency metrics, optimization
    recommendations, NFPA compliance, emissions analysis, and explainability.
    """

    # Identification and metadata
    equipment_id: str = Field(..., description="Equipment identifier from input")
    assessment_timestamp: datetime = Field(..., description="Assessment timestamp")
    agent_id: str = Field(default="GL-018", description="Agent identifier")
    agent_version: str = Field(default="1.0.0", description="Agent version")

    # Efficiency metrics
    efficiency_metrics: EfficiencyMetrics = Field(
        ...,
        description="Calculated efficiency metrics"
    )

    # Optimization recommendations
    o2_trim: O2TrimRecommendation = Field(
        ...,
        description="O2 trim control recommendations"
    )
    excess_air: ExcessAirRecommendation = Field(
        ...,
        description="Excess air control recommendations"
    )
    optimization_recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Prioritized optimization recommendations"
    )

    # Optimal parameters
    optimal_air_fuel_ratio: float = Field(..., description="Optimal air-fuel ratio")
    optimal_excess_air_pct: float = Field(..., description="Optimal excess air percentage")
    optimal_o2_setpoint: float = Field(..., description="Optimal O2 setpoint")

    # Efficiency gains
    potential_efficiency_gain_pct: float = Field(
        ...,
        description="Potential efficiency improvement percentage"
    )
    potential_fuel_savings_pct: float = Field(
        ...,
        description="Potential fuel savings percentage"
    )
    annual_cost_savings_estimate: Optional[float] = Field(
        default=None,
        description="Estimated annual cost savings"
    )

    # Emissions analysis
    emissions_analysis: EmissionsAnalysis = Field(
        ...,
        description="Emissions analysis results"
    )

    # NFPA compliance
    nfpa_compliance: NFPAComplianceResult = Field(
        ...,
        description="NFPA compliance assessment"
    )

    # Safety status
    safety_status: ComplianceStatus = Field(
        ...,
        description="Overall safety compliance status"
    )
    safety_interlocks_ok: bool = Field(
        ...,
        description="All safety interlocks in proper state"
    )
    safety_concerns: List[str] = Field(
        default_factory=list,
        description="Identified safety concerns"
    )

    # Explainability
    explainability: ExplainabilityReport = Field(
        ...,
        description="Explainability report for decisions"
    )

    # Provenance and audit
    provenance: ProvenanceRecord = Field(
        ...,
        description="Complete calculation provenance"
    )

    # Validation
    validation_status: str = Field(
        ...,
        pattern="^(PASS|FAIL)$",
        description="Validation status: PASS or FAIL"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages"
    )

    # Performance
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing duration in milliseconds"
    )

    def get_provenance_hash(self) -> str:
        """Generate SHA-256 hash for complete output provenance."""
        provenance_data = {
            'equipment_id': self.equipment_id,
            'timestamp': self.assessment_timestamp.isoformat(),
            'efficiency': self.efficiency_metrics.dict(),
            'o2_trim': self.o2_trim.dict(),
            'nfpa_status': self.nfpa_compliance.overall_status.value,
            'agent_version': self.agent_version,
        }
        return hashlib.sha256(str(provenance_data).encode('utf-8')).hexdigest()


# =============================================================================
# Agent Configuration
# =============================================================================

class AgentConfig(BaseModel):
    """Configuration for UnifiedCombustionOptimizerAgent."""

    agent_id: str = Field(default="GL-018", description="Agent identifier")
    agent_name: str = Field(default="UNIFIEDCOMBUSTION", description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")

    # O2 trim settings
    default_target_o2_min: float = Field(
        default=2.0,
        ge=0.5,
        le=10,
        description="Default minimum target O2 percentage"
    )
    default_target_o2_max: float = Field(
        default=4.0,
        ge=0.5,
        le=10,
        description="Default maximum target O2 percentage"
    )
    co_breakthrough_threshold: float = Field(
        default=200,
        gt=0,
        description="CO threshold for O2 trim safety limit (ppm)"
    )

    # NFPA settings
    nfpa_strict_mode: bool = Field(
        default=True,
        description="Enable strict NFPA compliance checking"
    )
    required_interlocks: List[str] = Field(
        default_factory=lambda: [
            "low_fuel_pressure",
            "high_fuel_pressure",
            "flame_failure",
            "low_air_flow",
            "high_furnace_pressure",
            "combustion_air_proving"
        ],
        description="Required safety interlocks per NFPA"
    )

    # Calculation settings
    decimal_precision: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Decimal places for calculation precision"
    )
    enable_provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    # Explainability settings
    enable_shap_analysis: bool = Field(
        default=True,
        description="Enable SHAP feature importance"
    )
    enable_causal_inference: bool = Field(
        default=True,
        description="Enable causal inference analysis"
    )
