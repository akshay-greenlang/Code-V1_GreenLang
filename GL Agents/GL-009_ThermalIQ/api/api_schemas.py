"""
GL-009 ThermalIQ - API Schemas

Pydantic models for API request/response validation.
Includes models for:
- Thermal analysis
- Efficiency calculations
- Exergy analysis
- Fluid properties
- Sankey diagrams
- Fluid recommendations
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# Enums
# =============================================================================

class FluidPhase(str, Enum):
    """Fluid phase states."""
    LIQUID = "liquid"
    GAS = "gas"
    TWO_PHASE = "two_phase"
    SUPERCRITICAL = "supercritical"


class AnalysisMode(str, Enum):
    """Thermal analysis modes."""
    FULL = "full"
    QUICK = "quick"
    DETAILED = "detailed"


class EfficiencyMethod(str, Enum):
    """Efficiency calculation methods."""
    FIRST_LAW = "first_law"
    SECOND_LAW = "second_law"
    COMBINED = "combined"


class FluidCategory(str, Enum):
    """Fluid categories for recommendations."""
    REFRIGERANT = "refrigerant"
    HEAT_TRANSFER = "heat_transfer"
    THERMAL_OIL = "thermal_oil"
    WATER_GLYCOL = "water_glycol"
    STEAM = "steam"
    ORGANIC = "organic"
    INORGANIC = "inorganic"


# =============================================================================
# Common Models
# =============================================================================

class FluidState(BaseModel):
    """Fluid thermodynamic state."""
    temperature_C: float = Field(..., description="Temperature in Celsius")
    pressure_kPa: float = Field(..., description="Pressure in kPa")
    phase: Optional[FluidPhase] = Field(None, description="Fluid phase")
    quality: Optional[float] = Field(None, ge=0, le=1, description="Vapor quality for two-phase")

    model_config = ConfigDict(use_enum_values=True)


class FluidProperties(BaseModel):
    """Calculated fluid properties at a state point."""
    temperature_C: float = Field(..., description="Temperature (C)")
    pressure_kPa: float = Field(..., description="Pressure (kPa)")
    phase: FluidPhase = Field(..., description="Fluid phase")

    # Thermodynamic properties
    density_kg_m3: float = Field(..., description="Density (kg/m3)")
    specific_heat_kJ_kgK: float = Field(..., description="Specific heat capacity (kJ/kg-K)")
    enthalpy_kJ_kg: float = Field(..., description="Specific enthalpy (kJ/kg)")
    entropy_kJ_kgK: float = Field(..., description="Specific entropy (kJ/kg-K)")
    internal_energy_kJ_kg: float = Field(..., description="Internal energy (kJ/kg)")

    # Transport properties
    viscosity_Pa_s: float = Field(..., description="Dynamic viscosity (Pa-s)")
    thermal_conductivity_W_mK: float = Field(..., description="Thermal conductivity (W/m-K)")
    prandtl_number: float = Field(..., description="Prandtl number")

    # Quality indicators
    quality: Optional[float] = Field(None, description="Vapor quality (two-phase only)")
    compressibility_factor: Optional[float] = Field(None, description="Compressibility factor Z")

    model_config = ConfigDict(use_enum_values=True)


class StreamData(BaseModel):
    """Heat stream data for analysis."""
    stream_id: str = Field(..., description="Unique stream identifier")
    stream_name: Optional[str] = Field(None, description="Human-readable name")
    fluid_name: str = Field(..., description="Working fluid name")

    # State points
    inlet_temperature_C: float = Field(..., description="Inlet temperature (C)")
    outlet_temperature_C: float = Field(..., description="Outlet temperature (C)")
    pressure_kPa: float = Field(101.325, description="Operating pressure (kPa)")
    mass_flow_kg_s: float = Field(..., gt=0, description="Mass flow rate (kg/s)")

    # Optional properties
    specific_heat_kJ_kgK: Optional[float] = Field(None, description="Specific heat (kJ/kg-K)")
    phase: Optional[FluidPhase] = Field(None, description="Fluid phase")

    model_config = ConfigDict(use_enum_values=True)


class ExergyComponent(BaseModel):
    """Exergy breakdown component."""
    name: str = Field(..., description="Component name")
    exergy_input_kW: float = Field(..., description="Exergy input (kW)")
    exergy_output_kW: float = Field(..., description="Exergy output (kW)")
    exergy_destruction_kW: float = Field(..., description="Exergy destruction (kW)")
    exergy_efficiency_percent: float = Field(..., description="Exergetic efficiency (%)")
    irreversibility_kW: float = Field(..., description="Irreversibility rate (kW)")


class SankeyNode(BaseModel):
    """Node in Sankey diagram."""
    id: str = Field(..., description="Node identifier")
    name: str = Field(..., description="Node name")
    value: float = Field(..., description="Energy/exergy value (kW)")
    category: str = Field(..., description="Node category")
    color: Optional[str] = Field(None, description="Node color (hex)")


class SankeyLink(BaseModel):
    """Link between nodes in Sankey diagram."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    value: float = Field(..., description="Flow value (kW)")
    label: Optional[str] = Field(None, description="Link label")
    color: Optional[str] = Field(None, description="Link color (hex)")


# =============================================================================
# Analysis Request/Response
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for full thermal analysis."""
    streams: List[StreamData] = Field(..., min_length=1, description="Heat streams to analyze")
    ambient_temperature_C: float = Field(25.0, description="Ambient/dead-state temperature (C)")
    ambient_pressure_kPa: float = Field(101.325, description="Ambient pressure (kPa)")
    mode: AnalysisMode = Field(AnalysisMode.FULL, description="Analysis mode")

    # Options
    include_exergy: bool = Field(True, description="Include exergy analysis")
    include_sankey: bool = Field(True, description="Generate Sankey diagram")
    include_recommendations: bool = Field(False, description="Include fluid recommendations")

    # Correlation ID for tracing
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "streams": [
                    {
                        "stream_id": "hot_1",
                        "fluid_name": "Water",
                        "inlet_temperature_C": 150.0,
                        "outlet_temperature_C": 50.0,
                        "pressure_kPa": 500.0,
                        "mass_flow_kg_s": 2.5
                    }
                ],
                "ambient_temperature_C": 25.0,
                "mode": "full",
                "include_exergy": True
            }
        }
    )


class AnalyzeResponse(BaseModel):
    """Response model for full thermal analysis."""
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Analysis status")
    timestamp: datetime = Field(..., description="Analysis timestamp")

    # Analysis results
    total_heat_duty_kW: float = Field(..., description="Total heat duty (kW)")
    total_mass_flow_kg_s: float = Field(..., description="Total mass flow (kg/s)")

    # Efficiency metrics
    first_law_efficiency_percent: float = Field(..., description="First law efficiency (%)")
    second_law_efficiency_percent: Optional[float] = Field(None, description="Second law efficiency (%)")

    # Stream analysis
    stream_results: List[Dict[str, Any]] = Field(default_factory=list, description="Per-stream results")

    # Exergy analysis (if requested)
    exergy_analysis: Optional[Dict[str, Any]] = Field(None, description="Exergy analysis results")

    # Sankey diagram (if requested)
    sankey_diagram: Optional[Dict[str, Any]] = Field(None, description="Sankey diagram data")

    # Recommendations (if requested)
    recommendations: Optional[List[Dict[str, Any]]] = Field(None, description="Optimization recommendations")

    # Provenance
    computation_hash: str = Field(..., description="SHA-256 hash for provenance")
    processing_time_ms: float = Field(..., description="Processing time (ms)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req_abc123",
                "status": "completed",
                "timestamp": "2025-01-15T10:30:00Z",
                "total_heat_duty_kW": 523.5,
                "total_mass_flow_kg_s": 2.5,
                "first_law_efficiency_percent": 85.2,
                "second_law_efficiency_percent": 62.4,
                "computation_hash": "a1b2c3d4e5f6",
                "processing_time_ms": 125.3
            }
        }
    )


# =============================================================================
# Efficiency Request/Response
# =============================================================================

class EfficiencyRequest(BaseModel):
    """Request model for efficiency calculation."""
    streams: List[StreamData] = Field(..., min_length=1, description="Heat streams")
    ambient_temperature_C: float = Field(25.0, description="Ambient temperature (C)")
    method: EfficiencyMethod = Field(EfficiencyMethod.COMBINED, description="Calculation method")

    # Reference conditions
    reference_temperature_C: Optional[float] = Field(None, description="Reference temperature (C)")
    reference_pressure_kPa: Optional[float] = Field(None, description="Reference pressure (kPa)")

    model_config = ConfigDict(use_enum_values=True)


class EfficiencyResponse(BaseModel):
    """Response model for efficiency calculation."""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Calculation timestamp")

    # First law efficiency
    first_law_efficiency_percent: float = Field(..., description="First law (energy) efficiency")
    energy_input_kW: float = Field(..., description="Total energy input (kW)")
    energy_output_kW: float = Field(..., description="Useful energy output (kW)")
    energy_loss_kW: float = Field(..., description="Energy losses (kW)")

    # Second law efficiency (if applicable)
    second_law_efficiency_percent: Optional[float] = Field(None, description="Second law (exergy) efficiency")
    exergy_input_kW: Optional[float] = Field(None, description="Exergy input (kW)")
    exergy_output_kW: Optional[float] = Field(None, description="Exergy output (kW)")
    exergy_destruction_kW: Optional[float] = Field(None, description="Exergy destruction (kW)")

    # Breakdown by stream
    stream_efficiencies: List[Dict[str, Any]] = Field(default_factory=list)

    # Provenance
    computation_hash: str = Field(..., description="SHA-256 hash")
    method_used: str = Field(..., description="Calculation method used")


# =============================================================================
# Exergy Request/Response
# =============================================================================

class ExergyRequest(BaseModel):
    """Request model for exergy analysis."""
    streams: List[StreamData] = Field(..., min_length=1, description="Streams to analyze")

    # Dead state (reference environment)
    dead_state_temperature_C: float = Field(25.0, description="Dead state temperature (C)")
    dead_state_pressure_kPa: float = Field(101.325, description="Dead state pressure (kPa)")

    # Options
    include_chemical_exergy: bool = Field(False, description="Include chemical exergy")
    include_kinetic_exergy: bool = Field(False, description="Include kinetic exergy")
    include_potential_exergy: bool = Field(False, description="Include potential exergy")

    model_config = ConfigDict(use_enum_values=True)


class ExergyResponse(BaseModel):
    """Response model for exergy analysis."""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")

    # Dead state conditions
    dead_state_temperature_C: float = Field(..., description="Dead state temperature (C)")
    dead_state_pressure_kPa: float = Field(..., description="Dead state pressure (kPa)")

    # Total exergy metrics
    total_exergy_input_kW: float = Field(..., description="Total exergy input (kW)")
    total_exergy_output_kW: float = Field(..., description="Total exergy output (kW)")
    total_exergy_destruction_kW: float = Field(..., description="Total exergy destruction (kW)")
    exergy_efficiency_percent: float = Field(..., description="Overall exergetic efficiency (%)")

    # Exergy breakdown
    physical_exergy_kW: float = Field(..., description="Physical exergy (kW)")
    chemical_exergy_kW: Optional[float] = Field(None, description="Chemical exergy (kW)")
    kinetic_exergy_kW: Optional[float] = Field(None, description="Kinetic exergy (kW)")
    potential_exergy_kW: Optional[float] = Field(None, description="Potential exergy (kW)")

    # Component-level analysis
    components: List[ExergyComponent] = Field(default_factory=list)

    # Improvement potential
    improvement_potential_kW: float = Field(..., description="Maximum improvement potential (kW)")

    # Provenance
    computation_hash: str = Field(..., description="SHA-256 hash")
    processing_time_ms: float = Field(..., description="Processing time (ms)")


# =============================================================================
# Fluid Properties Request/Response
# =============================================================================

class FluidPropertiesRequest(BaseModel):
    """Request model for fluid properties lookup."""
    fluid_name: str = Field(..., description="Fluid name (e.g., 'Water', 'R134a')")
    temperature_C: float = Field(..., description="Temperature (C)")
    pressure_kPa: float = Field(..., description="Pressure (kPa)")

    # Optional phase specification for two-phase regions
    quality: Optional[float] = Field(None, ge=0, le=1, description="Vapor quality (0-1)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fluid_name": "Water",
                "temperature_C": 100.0,
                "pressure_kPa": 101.325
            }
        }
    )


class FluidPropertiesResponse(BaseModel):
    """Response model for fluid properties lookup."""
    fluid_name: str = Field(..., description="Fluid name")
    properties: FluidProperties = Field(..., description="Calculated properties")

    # Fluid information
    molecular_weight_g_mol: float = Field(..., description="Molecular weight (g/mol)")
    critical_temperature_C: float = Field(..., description="Critical temperature (C)")
    critical_pressure_kPa: float = Field(..., description="Critical pressure (kPa)")

    # Validity
    is_valid_state: bool = Field(..., description="Whether state point is valid")
    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")

    # Provenance
    data_source: str = Field(..., description="Property data source")
    computation_hash: str = Field(..., description="SHA-256 hash")


class FluidListResponse(BaseModel):
    """Response model for available fluids list."""
    fluids: List[Dict[str, Any]] = Field(..., description="Available fluids")
    categories: List[str] = Field(..., description="Fluid categories")
    total_count: int = Field(..., description="Total number of fluids")


# =============================================================================
# Sankey Diagram Request/Response
# =============================================================================

class SankeyRequest(BaseModel):
    """Request model for Sankey diagram generation."""
    streams: List[StreamData] = Field(..., min_length=1, description="Heat streams")

    # Diagram options
    diagram_type: str = Field("energy", description="Type: energy or exergy")
    show_losses: bool = Field(True, description="Show loss streams")
    show_percentages: bool = Field(True, description="Show percentage labels")

    # Styling
    color_scheme: str = Field("thermal", description="Color scheme: thermal, categorical, monochrome")
    min_width: float = Field(2.0, description="Minimum link width (px)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "streams": [
                    {
                        "stream_id": "hot_1",
                        "fluid_name": "Steam",
                        "inlet_temperature_C": 200.0,
                        "outlet_temperature_C": 100.0,
                        "mass_flow_kg_s": 5.0
                    }
                ],
                "diagram_type": "energy",
                "show_losses": True
            }
        }
    )


class SankeyResponse(BaseModel):
    """Response model for Sankey diagram."""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Generation timestamp")

    # Diagram data
    nodes: List[SankeyNode] = Field(..., description="Diagram nodes")
    links: List[SankeyLink] = Field(..., description="Diagram links")

    # Summary
    total_input_kW: float = Field(..., description="Total input (kW)")
    total_output_kW: float = Field(..., description="Total useful output (kW)")
    total_losses_kW: float = Field(..., description="Total losses (kW)")

    # Rendering hints
    diagram_type: str = Field(..., description="Diagram type generated")
    layout_direction: str = Field("left_to_right", description="Layout direction")

    # Provenance
    computation_hash: str = Field(..., description="SHA-256 hash")


# =============================================================================
# Fluid Recommendation Request/Response
# =============================================================================

class FluidRecommendationRequest(BaseModel):
    """Request model for fluid recommendations."""
    application: str = Field(..., description="Application type (e.g., 'heat_recovery', 'chiller')")

    # Operating conditions
    min_temperature_C: float = Field(..., description="Minimum operating temperature (C)")
    max_temperature_C: float = Field(..., description="Maximum operating temperature (C)")
    operating_pressure_kPa: float = Field(101.325, description="Operating pressure (kPa)")

    # Constraints
    max_gwp: Optional[float] = Field(None, description="Maximum GWP allowed")
    max_odp: Optional[float] = Field(None, description="Maximum ODP allowed")
    require_non_flammable: bool = Field(False, description="Require non-flammable fluid")
    require_non_toxic: bool = Field(False, description="Require non-toxic fluid")

    # Preferences
    preferred_categories: Optional[List[FluidCategory]] = Field(None, description="Preferred categories")
    exclude_fluids: Optional[List[str]] = Field(None, description="Fluids to exclude")

    # Number of recommendations
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations")

    model_config = ConfigDict(use_enum_values=True)


class FluidRecommendation(BaseModel):
    """Single fluid recommendation."""
    fluid_name: str = Field(..., description="Recommended fluid")
    category: FluidCategory = Field(..., description="Fluid category")

    # Suitability
    suitability_score: float = Field(..., ge=0, le=100, description="Suitability score (0-100)")
    ranking: int = Field(..., description="Ranking position")

    # Properties at operating conditions
    properties_at_conditions: Optional[FluidProperties] = Field(None)

    # Environmental metrics
    gwp: Optional[float] = Field(None, description="Global Warming Potential")
    odp: Optional[float] = Field(None, description="Ozone Depletion Potential")

    # Safety
    flammability_class: Optional[str] = Field(None, description="Flammability class")
    toxicity_class: Optional[str] = Field(None, description="Toxicity class")

    # Reasoning
    pros: List[str] = Field(default_factory=list, description="Advantages")
    cons: List[str] = Field(default_factory=list, description="Disadvantages")
    notes: Optional[str] = Field(None, description="Additional notes")

    model_config = ConfigDict(use_enum_values=True)


class FluidRecommendationResponse(BaseModel):
    """Response model for fluid recommendations."""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")

    # Application context
    application: str = Field(..., description="Application analyzed")
    temperature_range_C: tuple = Field(..., description="Temperature range (min, max)")

    # Recommendations
    recommendations: List[FluidRecommendation] = Field(..., description="Ranked recommendations")

    # Summary
    best_overall: str = Field(..., description="Best overall fluid")
    best_environmental: Optional[str] = Field(None, description="Best environmental option")
    best_performance: Optional[str] = Field(None, description="Best performance option")

    # Provenance
    computation_hash: str = Field(..., description="SHA-256 hash")


# =============================================================================
# Error Response
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(), description="Error timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Invalid fluid name 'XYZ'",
                "details": {"field": "fluid_name", "valid_options": ["Water", "R134a", "R410A"]},
                "request_id": "req_abc123",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }
    )


# =============================================================================
# Health and Metrics
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Check timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")
    uptime_seconds: float = Field(..., description="Service uptime")


class MetricsResponse(BaseModel):
    """Prometheus-compatible metrics response."""
    requests_total: int = Field(..., description="Total requests processed")
    requests_success: int = Field(..., description="Successful requests")
    requests_failed: int = Field(..., description="Failed requests")
    average_latency_ms: float = Field(..., description="Average request latency (ms)")
    active_connections: int = Field(..., description="Active connections")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)")

    # Component-specific metrics
    fluid_lookups: int = Field(..., description="Fluid property lookups")
    exergy_calculations: int = Field(..., description="Exergy calculations performed")
    sankey_generations: int = Field(..., description="Sankey diagrams generated")
