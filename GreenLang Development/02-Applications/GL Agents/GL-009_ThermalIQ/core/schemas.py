"""
GL-009 THERMALIQ - Schema Definitions

Pydantic models for all inputs, outputs, fluid properties, exergy results,
Sankey data, explainability reports, and provenance tracking for the
ThermalFluidAnalyzer agent.

All schemas support zero-hallucination principles with deterministic
calculations, SHA-256 provenance tracking, and regulatory compliance.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator, root_validator

from .config import (
    FluidPhase,
    CalculationMode,
    ExplainabilityMethod,
    SankeyOutputFormat,
)


# =============================================================================
# ENUMS
# =============================================================================

class CalculationStatus(Enum):
    """Calculation execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


class CalculationType(Enum):
    """Types of calculations performed."""
    EFFICIENCY = "efficiency"
    EXERGY = "exergy"
    SANKEY = "sankey"
    FLUID_PROPERTIES = "fluid_properties"
    FULL_ANALYSIS = "full_analysis"


class SeverityLevel(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecommendationType(Enum):
    """Types of efficiency recommendations."""
    HEAT_RECOVERY = "heat_recovery"
    INSULATION = "insulation"
    FLUID_CHANGE = "fluid_change"
    TEMPERATURE_OPTIMIZATION = "temperature_optimization"
    PRESSURE_OPTIMIZATION = "pressure_optimization"
    FLOW_OPTIMIZATION = "flow_optimization"
    EQUIPMENT_UPGRADE = "equipment_upgrade"
    MAINTENANCE = "maintenance"


# =============================================================================
# FLUID PROPERTIES SCHEMAS
# =============================================================================

class FluidProperties(BaseModel):
    """
    Thermal fluid properties at a specific state point.

    Contains all thermophysical properties needed for thermal
    and exergy calculations. Supports both constant and
    temperature/pressure-dependent properties.

    Attributes:
        name: Fluid identifier
        temperature_C: Temperature in Celsius
        pressure_kPa: Pressure in kPa
        phase: Current phase state
        density_kg_m3: Density
        Cp_kJ_kgK: Specific heat capacity
        viscosity_Pa_s: Dynamic viscosity
        conductivity_W_mK: Thermal conductivity
        enthalpy_kJ_kg: Specific enthalpy (optional)
        entropy_kJ_kgK: Specific entropy (optional)
        quality: Vapor quality for two-phase (0-1)
    """
    # Identification
    fluid_id: str = Field(..., description="Unique fluid identifier")
    name: str = Field(..., description="Human-readable fluid name")

    # State point
    temperature_C: float = Field(
        ...,
        ge=-273.15,
        le=2000.0,
        description="Temperature (Celsius)"
    )
    pressure_kPa: float = Field(
        ...,
        ge=0.1,
        le=100000.0,
        description="Pressure (kPa)"
    )
    phase: FluidPhase = Field(
        default=FluidPhase.LIQUID,
        description="Fluid phase"
    )

    # Thermophysical properties
    density_kg_m3: float = Field(
        ...,
        ge=0.001,
        le=25000.0,
        description="Density (kg/m3)"
    )
    Cp_kJ_kgK: float = Field(
        ...,
        ge=0.1,
        le=20.0,
        description="Specific heat capacity (kJ/kg-K)"
    )
    viscosity_Pa_s: Optional[float] = Field(
        default=None,
        ge=1e-7,
        le=1e6,
        description="Dynamic viscosity (Pa-s)"
    )
    conductivity_W_mK: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=500.0,
        description="Thermal conductivity (W/m-K)"
    )

    # Derived/advanced properties
    enthalpy_kJ_kg: Optional[float] = Field(
        default=None,
        description="Specific enthalpy (kJ/kg)"
    )
    entropy_kJ_kgK: Optional[float] = Field(
        default=None,
        description="Specific entropy (kJ/kg-K)"
    )
    exergy_kJ_kg: Optional[float] = Field(
        default=None,
        description="Specific exergy (kJ/kg)"
    )

    # Two-phase properties
    quality: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Vapor quality (two-phase only)"
    )

    # Metadata
    source: str = Field(
        default="internal",
        description="Property data source"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Config:
        use_enum_values = True


class OperatingConditions(BaseModel):
    """
    Operating conditions for thermal analysis.

    Defines the boundary conditions and operating parameters
    for thermal efficiency and exergy calculations.

    Attributes:
        inlet_temperature_C: Fluid inlet temperature
        outlet_temperature_C: Fluid outlet temperature
        mass_flow_kg_s: Mass flow rate
        pressure_kPa: Operating pressure
        ambient_temperature_C: Ambient temperature for losses
        reference_temperature_C: Dead state for exergy
    """
    # Temperature conditions
    inlet_temperature_C: float = Field(
        ...,
        ge=-200.0,
        le=1500.0,
        description="Inlet temperature (C)"
    )
    outlet_temperature_C: float = Field(
        ...,
        ge=-200.0,
        le=1500.0,
        description="Outlet temperature (C)"
    )
    ambient_temperature_C: float = Field(
        default=25.0,
        ge=-50.0,
        le=50.0,
        description="Ambient temperature (C)"
    )
    reference_temperature_C: float = Field(
        default=25.0,
        description="Dead state temperature for exergy (C)"
    )

    # Flow conditions
    mass_flow_kg_s: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Mass flow rate (kg/s)"
    )
    pressure_kPa: float = Field(
        default=101.325,
        ge=1.0,
        le=50000.0,
        description="Operating pressure (kPa)"
    )

    # Operational factors
    operating_hours_per_year: int = Field(
        default=8000,
        ge=0,
        le=8760,
        description="Annual operating hours"
    )
    load_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Load factor (0-1)"
    )

    @validator("outlet_temperature_C")
    def validate_temperature_change(cls, v, values):
        """Validate temperature change is reasonable."""
        if "inlet_temperature_C" in values:
            delta_t = abs(v - values["inlet_temperature_C"])
            if delta_t > 500:
                # Warning-level validation, not error
                pass
        return v


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class ThermalAnalysisInput(BaseModel):
    """
    Input data for thermal analysis.

    Contains all required parameters for thermal efficiency,
    exergy analysis, and Sankey diagram generation.

    Attributes:
        analysis_id: Unique analysis identifier
        energy_in_kW: Total energy input
        heat_out_kW: Useful heat output
        losses_kW: Identified thermal losses
        fluid_properties: Fluid state properties
        operating_conditions: Operating conditions
        mode: Analysis mode (efficiency, exergy, full)
    """
    # Identification
    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Energy balance inputs
    energy_in_kW: float = Field(
        ...,
        ge=0.0,
        le=1e9,
        description="Total energy input (kW)"
    )
    heat_out_kW: float = Field(
        ...,
        ge=0.0,
        le=1e9,
        description="Useful heat output (kW)"
    )
    losses_kW: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of thermal losses by category (kW)"
    )

    # Fluid information
    fluid_properties: FluidProperties = Field(
        ...,
        description="Fluid thermophysical properties"
    )
    operating_conditions: OperatingConditions = Field(
        ...,
        description="Operating conditions"
    )

    # Analysis settings
    mode: CalculationMode = Field(
        default=CalculationMode.FULL_ANALYSIS,
        description="Analysis mode"
    )
    include_exergy: bool = Field(
        default=True,
        description="Include exergy analysis"
    )
    include_sankey: bool = Field(
        default=True,
        description="Generate Sankey diagram data"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Generate improvement recommendations"
    )

    # Source tracking
    source_system: str = Field(
        default="api",
        description="Data source system"
    )
    requested_by: str = Field(
        default="system",
        description="Requester identifier"
    )

    @validator("heat_out_kW")
    def validate_energy_balance(cls, v, values):
        """Validate heat output does not exceed input."""
        if "energy_in_kW" in values and v > values["energy_in_kW"]:
            raise ValueError(
                f"Heat output ({v} kW) cannot exceed energy input "
                f"({values['energy_in_kW']} kW)"
            )
        return v

    class Config:
        use_enum_values = True


class ExergyResult(BaseModel):
    """
    Exergy (second-law) analysis results.

    Quantifies thermodynamic irreversibility and available
    work potential for thermal system improvement.

    Attributes:
        exergy_input_kW: Total exergy entering system
        exergy_output_kW: Useful exergy leaving system
        exergy_destruction_kW: Exergy destroyed (irreversibility)
        exergy_efficiency_pct: Second-law efficiency
        carnot_factor: Carnot efficiency at operating conditions
    """
    # Identification
    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Reference conditions
    reference_temperature_K: float = Field(
        default=298.15,
        description="Dead state temperature (K)"
    )
    reference_pressure_kPa: float = Field(
        default=101.325,
        description="Dead state pressure (kPa)"
    )

    # Exergy values
    exergy_input_kW: float = Field(
        ...,
        ge=0.0,
        description="Total exergy input (kW)"
    )
    exergy_output_kW: float = Field(
        ...,
        ge=0.0,
        description="Useful exergy output (kW)"
    )
    exergy_destruction_kW: float = Field(
        ...,
        ge=0.0,
        description="Exergy destruction/irreversibility (kW)"
    )
    exergy_loss_kW: float = Field(
        default=0.0,
        ge=0.0,
        description="Exergy loss to environment (kW)"
    )

    # Efficiency metrics
    exergy_efficiency_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Exergy (second-law) efficiency (%)"
    )
    carnot_factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Carnot factor (1 - T0/Tsource)"
    )

    # Breakdown by component (if available)
    exergy_by_component: Dict[str, float] = Field(
        default_factory=dict,
        description="Exergy destruction by component"
    )

    # Improvement potential
    improvement_potential_kW: float = Field(
        default=0.0,
        ge=0.0,
        description="Theoretical improvement potential (kW)"
    )
    improvement_potential_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Improvement potential (%)"
    )

    # Provenance
    input_hash: str = Field(default="", description="SHA-256 of inputs")
    output_hash: str = Field(default="", description="SHA-256 of outputs")
    formula_version: str = Field(default="EXERGY_v1.0")

    class Config:
        use_enum_values = True


class SankeyNode(BaseModel):
    """Node in Sankey diagram."""

    node_id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Display label")
    value_kW: float = Field(..., ge=0.0, description="Energy value (kW)")
    color: Optional[str] = Field(default=None, description="Node color")
    category: str = Field(default="flow", description="Node category")


class SankeyLink(BaseModel):
    """Link between nodes in Sankey diagram."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    value_kW: float = Field(..., ge=0.0, description="Flow value (kW)")
    color: Optional[str] = Field(default=None, description="Link color")
    label: Optional[str] = Field(default=None, description="Link label")


class SankeyData(BaseModel):
    """
    Sankey diagram data for energy flow visualization.

    Contains nodes and links for rendering an interactive
    Sankey diagram showing energy flows and losses.

    Attributes:
        nodes: List of Sankey nodes
        links: List of Sankey links
        total_input_kW: Total energy input
        total_output_kW: Total useful output
        total_losses_kW: Total losses
        format: Output format (plotly, d3, etc.)
    """
    # Identification
    diagram_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Diagram data
    nodes: List[SankeyNode] = Field(
        default_factory=list,
        description="Sankey diagram nodes"
    )
    links: List[SankeyLink] = Field(
        default_factory=list,
        description="Sankey diagram links"
    )

    # Summary values
    total_input_kW: float = Field(..., ge=0.0, description="Total input")
    total_output_kW: float = Field(..., ge=0.0, description="Total output")
    total_losses_kW: float = Field(..., ge=0.0, description="Total losses")

    # Efficiency
    thermal_efficiency_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Thermal efficiency (%)"
    )

    # Format and metadata
    format: SankeyOutputFormat = Field(
        default=SankeyOutputFormat.PLOTLY
    )
    units: str = Field(default="kW", description="Energy units")
    title: str = Field(default="Energy Flow Diagram")

    # Export helpers
    def to_plotly_dict(self) -> Dict[str, Any]:
        """Convert to Plotly Sankey format."""
        node_indices = {n.node_id: i for i, n in enumerate(self.nodes)}

        return {
            "type": "sankey",
            "node": {
                "label": [n.label for n in self.nodes],
                "color": [n.color or "#1f77b4" for n in self.nodes],
            },
            "link": {
                "source": [node_indices[l.source] for l in self.links],
                "target": [node_indices[l.target] for l in self.links],
                "value": [l.value_kW for l in self.links],
                "color": [l.color or "rgba(31,119,180,0.5)" for l in self.links],
            },
        }

    class Config:
        use_enum_values = True


# =============================================================================
# EXPLAINABILITY SCHEMAS
# =============================================================================

class FeatureImportance(BaseModel):
    """Feature importance from SHAP analysis."""

    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized importance (0-1)"
    )
    shap_value: float = Field(..., description="Raw SHAP value")
    direction: str = Field(
        default="neutral",
        description="Impact direction (positive/negative/neutral)"
    )


class LIMEExplanation(BaseModel):
    """Local explanation from LIME analysis."""

    feature_name: str = Field(..., description="Feature name")
    feature_value: float = Field(..., description="Feature value")
    weight: float = Field(..., description="LIME weight")
    contribution: str = Field(
        ...,
        description="Contribution description"
    )


class Recommendation(BaseModel):
    """
    Improvement recommendation from analysis.

    Contains actionable recommendations for improving
    thermal efficiency based on analysis results.
    """
    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    # Recommendation content
    type: RecommendationType = Field(..., description="Recommendation type")
    title: str = Field(..., description="Short title")
    description: str = Field(..., description="Detailed description")
    rationale: str = Field(..., description="Technical rationale")

    # Impact assessment
    estimated_savings_kW: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated energy savings (kW)"
    )
    estimated_savings_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated savings (%)"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level (0-1)"
    )

    # Implementation
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Priority (1=highest)"
    )
    complexity: str = Field(
        default="medium",
        description="Implementation complexity"
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated implementation cost"
    )

    class Config:
        use_enum_values = True


class ExplainabilityReport(BaseModel):
    """
    Comprehensive explainability report.

    Contains SHAP/LIME analysis results and actionable
    recommendations for thermal system improvement.

    Attributes:
        feature_importance: Ranked feature importances
        lime_explanations: Local explanations
        recommendations: Improvement recommendations
        summary: Executive summary
    """
    # Identification
    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    analysis_id: str = Field(..., description="Parent analysis ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # SHAP results
    feature_importance: List[FeatureImportance] = Field(
        default_factory=list,
        description="Feature importance ranking"
    )
    shap_base_value: float = Field(
        default=0.0,
        description="SHAP base (expected) value"
    )

    # LIME results
    lime_explanations: List[LIMEExplanation] = Field(
        default_factory=list,
        description="LIME local explanations"
    )
    lime_score: float = Field(
        default=0.0,
        description="LIME model fidelity score"
    )

    # Recommendations
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )

    # Summary
    executive_summary: str = Field(
        default="",
        description="Executive summary of findings"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================

class ThermalAnalysisOutput(BaseModel):
    """
    Complete output from thermal analysis.

    Contains all results including efficiency metrics, exergy
    analysis, Sankey data, and recommendations.

    Attributes:
        efficiency_percent: Thermal (first-law) efficiency
        exergy_result: Exergy analysis results
        sankey_data: Sankey diagram data
        explainability: Explainability report
        provenance: Audit provenance record
    """
    # Identification
    analysis_id: str = Field(..., description="Analysis identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: CalculationStatus = Field(
        default=CalculationStatus.COMPLETED
    )

    # Primary results - thermal efficiency
    efficiency_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Thermal (first-law) efficiency (%)"
    )
    energy_in_kW: float = Field(..., ge=0.0, description="Energy input")
    heat_out_kW: float = Field(..., ge=0.0, description="Heat output")
    total_losses_kW: float = Field(..., ge=0.0, description="Total losses")

    # Loss breakdown
    losses_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Losses by category (kW)"
    )

    # Exergy analysis results
    exergy_result: Optional[ExergyResult] = Field(
        default=None,
        description="Exergy analysis results"
    )

    # Sankey diagram data
    sankey_data: Optional[SankeyData] = Field(
        default=None,
        description="Sankey diagram data"
    )

    # Explainability and recommendations
    explainability: Optional[ExplainabilityReport] = Field(
        default=None,
        description="Explainability report"
    )

    # Fluid properties used
    fluid_properties: Optional[FluidProperties] = Field(
        default=None,
        description="Fluid properties at operating conditions"
    )

    # Performance metrics
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (ms)"
    )
    calculation_count: int = Field(
        default=1,
        ge=1,
        description="Number of calculations performed"
    )

    # Provenance and audit
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )
    input_hash: str = Field(
        default="",
        description="SHA-256 of inputs"
    )
    formula_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Formula versions used"
    )

    # Validation
    validation_passed: bool = Field(
        default=True,
        description="All validations passed"
    )
    validation_messages: List[str] = Field(
        default_factory=list,
        description="Validation messages/warnings"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# PROVENANCE AND AUDIT SCHEMAS
# =============================================================================

class ProvenanceRecord(BaseModel):
    """
    SHA-256 provenance record for audit trail.

    Tracks all inputs, outputs, and transformations
    for regulatory compliance and reproducibility.
    """
    # Identification
    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Hash values
    input_hash: str = Field(..., description="SHA-256 of inputs")
    output_hash: str = Field(..., description="SHA-256 of outputs")
    provenance_hash: str = Field(
        ...,
        description="Combined SHA-256 provenance hash"
    )

    # Calculation details
    calculation_type: CalculationType = Field(...)
    formula_id: str = Field(..., description="Formula identifier")
    formula_version: str = Field(default="1.0.0")

    # Agent information
    agent_id: str = Field(default="GL-009")
    agent_version: str = Field(default="1.0.0")

    # Reproducibility
    deterministic: bool = Field(default=True)
    reproducible: bool = Field(default=True)
    random_seed: Optional[int] = Field(default=None)

    # Input/output summaries (for quick reference)
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class CalculationEvent(BaseModel):
    """Calculation event for audit logging."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    calculation_type: CalculationType = Field(...)

    # Inputs
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    input_hash: str = Field(..., description="SHA-256 of inputs")

    # Outputs
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    output_hash: str = Field(..., description="SHA-256 of outputs")

    # Provenance
    formula_id: str = Field(..., description="Formula identifier")
    formula_version: str = Field(default="1.0.0")
    deterministic: bool = Field(default=True)
    reproducible: bool = Field(default=True)

    # Performance
    calculation_time_ms: float = Field(default=0.0, ge=0.0)

    class Config:
        use_enum_values = True


# =============================================================================
# STATUS SCHEMAS
# =============================================================================

class AgentStatus(BaseModel):
    """GL-009 THERMALIQ agent status."""

    agent_id: str = Field(default="GL-009")
    agent_name: str = Field(default="THERMALIQ")
    agent_version: str = Field(default="1.0.0")

    # Health
    status: str = Field(default="running")
    health: str = Field(default="healthy")
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Performance
    analyses_performed: int = Field(default=0, ge=0)
    analyses_successful: int = Field(default=0, ge=0)
    avg_processing_time_ms: float = Field(default=0.0, ge=0.0)

    # Capabilities
    available_fluids: List[str] = Field(default_factory=list)
    calculation_modes: List[str] = Field(default_factory=list)

    # Resources
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)

    # Integration
    kafka_connected: bool = Field(default=False)
    graphql_ready: bool = Field(default=True)
    explainability_ready: bool = Field(default=True)


class HealthCheckResponse(BaseModel):
    """Health check API response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    uptime_seconds: float = Field(default=0.0)
    checks: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# API RESPONSE SCHEMAS
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = Field(...)
    message: str = Field(default="")
    data: Optional[Any] = Field(default=None)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    request_id: Optional[str] = Field(default=None)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
