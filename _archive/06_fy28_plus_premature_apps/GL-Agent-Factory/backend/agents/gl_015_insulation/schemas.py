"""
Pydantic Schemas for GL-015 INSULSCAN Agent

This module defines all input/output models for the InsulationAnalysisAgent.
All models use Pydantic for validation, serialization, and documentation.

Models follow GreenLang standards:
- Complete field descriptions
- Appropriate validators
- Clear type hints
- Examples in docstrings
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator, root_validator
import hashlib


class SurfaceType(str, Enum):
    """Types of insulated surfaces."""
    PIPE = "pipe"
    FLAT = "flat"
    TANK = "tank"
    VESSEL = "vessel"
    DUCT = "duct"
    EQUIPMENT = "equipment"


class InsulationCondition(str, Enum):
    """Condition assessment of existing insulation."""
    EXCELLENT = "excellent"  # Like new, no damage
    GOOD = "good"  # Minor wear, functional
    FAIR = "fair"  # Moderate damage, reduced performance
    POOR = "poor"  # Significant damage, needs repair
    CRITICAL = "critical"  # Severe damage, immediate attention
    MISSING = "missing"  # No insulation present


class MaintenancePriority(str, Enum):
    """Maintenance priority levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class SurfaceGeometry(BaseModel):
    """
    Geometry specification for insulated surface.

    Supports both flat and cylindrical geometries.
    """
    surface_type: SurfaceType = Field(
        ...,
        description="Type of surface (pipe, flat, tank, etc.)"
    )

    # Flat surface dimensions
    area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Surface area in square meters (for flat surfaces)"
    )
    length_m: Optional[float] = Field(
        None,
        gt=0,
        description="Length in meters"
    )
    width_m: Optional[float] = Field(
        None,
        gt=0,
        description="Width in meters (for flat surfaces)"
    )

    # Cylindrical dimensions
    outer_diameter_m: Optional[float] = Field(
        None,
        gt=0,
        description="Outer diameter in meters (for pipes/cylinders)"
    )
    inner_diameter_m: Optional[float] = Field(
        None,
        gt=0,
        description="Inner diameter in meters (optional)"
    )

    # Current insulation
    current_insulation_thickness_m: Optional[float] = Field(
        None,
        ge=0,
        description="Current insulation thickness in meters (0 if bare)"
    )

    @root_validator(skip_on_failure=True)
    def validate_geometry(cls, values):
        """Validate geometry parameters are consistent."""
        surface_type = values.get('surface_type')

        if surface_type in [SurfaceType.PIPE, SurfaceType.DUCT]:
            if not values.get('outer_diameter_m'):
                raise ValueError(f"outer_diameter_m required for {surface_type}")
            if not values.get('length_m'):
                raise ValueError(f"length_m required for {surface_type}")
        elif surface_type == SurfaceType.FLAT:
            if not values.get('area_m2'):
                # Calculate from length x width if available
                length = values.get('length_m')
                width = values.get('width_m')
                if length and width:
                    values['area_m2'] = length * width
                else:
                    raise ValueError("area_m2 or (length_m and width_m) required for flat surface")

        return values


class TemperatureConditions(BaseModel):
    """Temperature conditions for analysis."""

    process_temp_c: float = Field(
        ...,
        description="Process/hot surface temperature in Celsius"
    )
    ambient_temp_c: float = Field(
        default=25.0,
        description="Ambient temperature in Celsius"
    )
    design_temp_c: Optional[float] = Field(
        None,
        description="Design temperature in Celsius (if different from process)"
    )
    max_surface_temp_c: Optional[float] = Field(
        None,
        description="Maximum allowable surface temperature for safety"
    )

    @validator('process_temp_c')
    def validate_process_temp(cls, v):
        """Validate process temperature is reasonable."""
        if v < -273:
            raise ValueError("Temperature cannot be below absolute zero")
        if v > 2000:
            raise ValueError("Temperature exceeds typical industrial range")
        return v

    @validator('max_surface_temp_c')
    def validate_max_surface(cls, v, values):
        """Validate max surface temp is below process temp."""
        if v is not None:
            process_temp = values.get('process_temp_c')
            if process_temp and v > process_temp:
                raise ValueError("Max surface temp cannot exceed process temp")
        return v


class IRCameraData(BaseModel):
    """
    Infrared camera measurement data.

    Supports thermal imaging integration for insulation analysis.
    """
    measurement_id: str = Field(
        ...,
        description="Unique measurement identifier"
    )
    measurement_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of IR measurement"
    )

    # Temperature measurements
    surface_temp_c: float = Field(
        ...,
        description="Measured surface temperature in Celsius"
    )
    ambient_temp_c: float = Field(
        ...,
        description="Ambient temperature at measurement time"
    )
    reflected_temp_c: Optional[float] = Field(
        None,
        description="Reflected apparent temperature for emissivity correction"
    )

    # Measurement parameters
    emissivity: float = Field(
        default=0.90,
        ge=0.01,
        le=1.0,
        description="Surface emissivity used in measurement"
    )
    distance_m: Optional[float] = Field(
        None,
        gt=0,
        description="Distance from camera to surface"
    )
    humidity_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Relative humidity during measurement"
    )

    # Area information
    area_m2: float = Field(
        ...,
        gt=0,
        description="Area of measured region in square meters"
    )
    spot_count: int = Field(
        default=1,
        ge=1,
        description="Number of measurement spots in area"
    )

    # Quality indicators
    confidence_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Measurement confidence (0-1)"
    )
    notes: Optional[str] = Field(
        None,
        description="Measurement notes or observations"
    )


class InsulationMaterialSpec(BaseModel):
    """Insulation material specification."""

    material_id: Optional[str] = Field(
        None,
        description="Material ID from database"
    )
    material_name: Optional[str] = Field(
        None,
        description="Material name"
    )
    thermal_conductivity: Optional[float] = Field(
        None,
        gt=0,
        description="Thermal conductivity in W/m-K (at mean temp)"
    )
    density_kg_m3: Optional[float] = Field(
        None,
        gt=0,
        description="Material density in kg/m3"
    )
    max_temp_c: Optional[float] = Field(
        None,
        description="Maximum service temperature in Celsius"
    )
    cost_per_m3_usd: Optional[float] = Field(
        None,
        ge=0,
        description="Material cost per cubic meter in USD"
    )

    @root_validator(skip_on_failure=True)
    def validate_material_spec(cls, values):
        """Ensure either material_id or properties are provided."""
        material_id = values.get('material_id')
        k_value = values.get('thermal_conductivity')

        if not material_id and not k_value:
            raise ValueError(
                "Either material_id or thermal_conductivity must be provided"
            )
        return values


class EconomicParameters(BaseModel):
    """Economic parameters for analysis."""

    energy_cost_per_kwh: float = Field(
        default=0.10,
        gt=0,
        description="Energy cost per kWh in USD"
    )
    energy_cost_per_therm: Optional[float] = Field(
        None,
        gt=0,
        description="Energy cost per therm (natural gas)"
    )
    boiler_efficiency: float = Field(
        default=0.85,
        gt=0,
        le=1,
        description="Boiler/heater efficiency"
    )
    operating_hours_per_year: float = Field(
        default=8760,
        gt=0,
        le=8760,
        description="Annual operating hours"
    )
    discount_rate: float = Field(
        default=0.08,
        ge=0,
        le=0.5,
        description="Discount rate for NPV calculations"
    )
    project_life_years: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Project life for economic analysis"
    )
    installation_factor: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Installation cost multiplier"
    )
    maintenance_factor: float = Field(
        default=0.02,
        ge=0,
        le=0.2,
        description="Annual maintenance cost as fraction of capital"
    )
    energy_escalation_rate: float = Field(
        default=0.03,
        ge=0,
        le=0.15,
        description="Annual energy price escalation rate"
    )


class EnvironmentalConditions(BaseModel):
    """Environmental conditions at the surface."""

    wind_speed_m_s: float = Field(
        default=0.0,
        ge=0,
        description="Wind speed in m/s"
    )
    surface_orientation: str = Field(
        default="vertical",
        description="Surface orientation (vertical, horizontal_up, horizontal_down)"
    )
    outdoor_exposure: bool = Field(
        default=False,
        description="True if surface is exposed to outdoor conditions"
    )
    corrosive_environment: bool = Field(
        default=False,
        description="True if corrosive environment (requires special materials)"
    )
    moisture_present: bool = Field(
        default=False,
        description="True if moisture is present"
    )


class InsulationAnalysisInput(BaseModel):
    """
    Input data model for InsulationAnalysisAgent.

    Comprehensive input model supporting thermal imaging integration,
    economic analysis, and multi-surface analysis.

    Example:
        >>> input_data = InsulationAnalysisInput(
        ...     analysis_id="INS-001",
        ...     geometry=SurfaceGeometry(
        ...         surface_type="pipe",
        ...         outer_diameter_m=0.1,
        ...         length_m=100
        ...     ),
        ...     temperature=TemperatureConditions(
        ...         process_temp_c=180,
        ...         ambient_temp_c=25
        ...     )
        ... )
    """
    # Identification
    analysis_id: str = Field(
        ...,
        min_length=1,
        description="Unique analysis identifier"
    )
    asset_id: Optional[str] = Field(
        None,
        description="Asset/equipment identifier"
    )
    location: Optional[str] = Field(
        None,
        description="Physical location description"
    )

    # Geometry
    geometry: SurfaceGeometry = Field(
        ...,
        description="Surface geometry specification"
    )

    # Temperature
    temperature: TemperatureConditions = Field(
        ...,
        description="Temperature conditions"
    )

    # Current insulation (if any)
    current_insulation: Optional[InsulationMaterialSpec] = Field(
        None,
        description="Current insulation material (if installed)"
    )
    insulation_condition: InsulationCondition = Field(
        default=InsulationCondition.MISSING,
        description="Condition of current insulation"
    )

    # Proposed insulation
    proposed_insulation: Optional[InsulationMaterialSpec] = Field(
        None,
        description="Proposed insulation material for analysis"
    )
    proposed_thickness_m: Optional[float] = Field(
        None,
        gt=0,
        description="Proposed insulation thickness in meters"
    )

    # IR camera data
    ir_measurements: List[IRCameraData] = Field(
        default_factory=list,
        description="IR camera measurements for this surface"
    )

    # Economic parameters
    economics: EconomicParameters = Field(
        default_factory=EconomicParameters,
        description="Economic parameters for analysis"
    )

    # Environmental conditions
    environment: EnvironmentalConditions = Field(
        default_factory=EnvironmentalConditions,
        description="Environmental conditions"
    )

    # Analysis options
    calculate_economic_thickness: bool = Field(
        default=True,
        description="Calculate optimal economic thickness"
    )
    compare_materials: bool = Field(
        default=False,
        description="Compare multiple insulation materials"
    )
    materials_to_compare: List[str] = Field(
        default_factory=list,
        description="List of material_ids to compare"
    )
    include_explainability: bool = Field(
        default=True,
        description="Include SHAP/LIME explainability in results"
    )


class HeatLossQuantification(BaseModel):
    """Heat loss quantification results."""

    heat_loss_w: float = Field(
        ...,
        ge=0,
        description="Heat loss in Watts"
    )
    heat_loss_btu_hr: float = Field(
        ...,
        ge=0,
        description="Heat loss in BTU/hr"
    )
    heat_flux_w_m2: float = Field(
        ...,
        ge=0,
        description="Heat flux per unit area in W/m2"
    )
    surface_temp_c: float = Field(
        ...,
        description="Calculated/measured surface temperature"
    )
    annual_energy_loss_kwh: float = Field(
        ...,
        ge=0,
        description="Annual energy loss in kWh"
    )
    annual_energy_loss_gj: float = Field(
        ...,
        ge=0,
        description="Annual energy loss in GJ"
    )
    annual_energy_cost_usd: float = Field(
        ...,
        ge=0,
        description="Annual energy cost in USD"
    )


class EconomicThicknessResult(BaseModel):
    """Economic thickness calculation results."""

    economic_thickness_mm: float = Field(
        ...,
        ge=0,
        description="Optimal insulation thickness in mm"
    )
    economic_thickness_inches: float = Field(
        ...,
        ge=0,
        description="Optimal thickness in inches"
    )
    minimum_total_cost_usd: float = Field(
        ...,
        ge=0,
        description="Minimum total annual cost at optimal thickness"
    )
    annual_energy_cost_usd: float = Field(
        ...,
        ge=0,
        description="Annual energy cost at optimal thickness"
    )
    annual_insulation_cost_usd: float = Field(
        ...,
        ge=0,
        description="Annualized insulation cost"
    )
    capital_cost_usd: float = Field(
        ...,
        ge=0,
        description="Capital cost of insulation installation"
    )
    simple_payback_years: float = Field(
        ...,
        ge=0,
        description="Simple payback period in years"
    )
    npv_usd: float = Field(
        ...,
        description="Net present value in USD"
    )
    irr_percent: float = Field(
        ...,
        description="Internal rate of return"
    )
    energy_savings_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Energy savings vs bare surface"
    )


class InsulationRecommendation(BaseModel):
    """Individual insulation recommendation."""

    recommendation_id: str = Field(
        ...,
        description="Unique recommendation identifier"
    )
    priority: MaintenancePriority = Field(
        ...,
        description="Priority level"
    )
    action: str = Field(
        ...,
        description="Recommended action"
    )
    reason: str = Field(
        ...,
        description="Reason for recommendation"
    )
    material_id: Optional[str] = Field(
        None,
        description="Recommended material ID"
    )
    material_name: Optional[str] = Field(
        None,
        description="Recommended material name"
    )
    thickness_mm: Optional[float] = Field(
        None,
        description="Recommended thickness in mm"
    )
    estimated_savings_usd: Optional[float] = Field(
        None,
        description="Estimated annual savings"
    )
    estimated_cost_usd: Optional[float] = Field(
        None,
        description="Estimated implementation cost"
    )
    roi_years: Optional[float] = Field(
        None,
        description="Return on investment period"
    )


class ExplainabilityFactor(BaseModel):
    """SHAP/LIME explainability factor."""

    factor_name: str = Field(
        ...,
        description="Name of the factor"
    )
    factor_value: float = Field(
        ...,
        description="Value of the factor"
    )
    contribution_percent: float = Field(
        ...,
        description="Contribution to result (%)"
    )
    direction: str = Field(
        ...,
        description="Direction of effect (increase/decrease)"
    )
    explanation: str = Field(
        ...,
        description="Plain-language explanation"
    )


class ExplainabilityReport(BaseModel):
    """SHAP/LIME style explainability report."""

    primary_drivers: List[ExplainabilityFactor] = Field(
        ...,
        description="Primary factors driving the result"
    )
    sensitivity_analysis: Dict[str, float] = Field(
        default_factory=dict,
        description="Sensitivity of result to input changes"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in the analysis"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Key assumptions in the analysis"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitations of the analysis"
    )


class MaterialComparison(BaseModel):
    """Comparison of insulation materials."""

    material_id: str = Field(..., description="Material identifier")
    material_name: str = Field(..., description="Material name")
    thermal_conductivity: float = Field(..., description="K-value in W/m-K")
    economic_thickness_mm: float = Field(..., description="Optimal thickness")
    total_annual_cost_usd: float = Field(..., description="Total annual cost")
    energy_savings_percent: float = Field(..., description="Energy savings")
    npv_usd: float = Field(..., description="NPV of investment")
    payback_years: float = Field(..., description="Simple payback")
    recommendation_rank: int = Field(..., description="Rank (1=best)")


class ThermalMapPoint(BaseModel):
    """Point in thermal map."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    temperature_c: float = Field(..., description="Temperature at point")
    heat_loss_w_m2: float = Field(..., description="Heat loss rate at point")
    condition: InsulationCondition = Field(..., description="Condition at point")


class InsulationAnalysisOutput(BaseModel):
    """
    Output data model for InsulationAnalysisAgent.

    Comprehensive output including heat loss quantification, economic analysis,
    recommendations, and explainability.
    """
    # Identification
    analysis_id: str = Field(
        ...,
        description="Analysis identifier from input"
    )
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of analysis"
    )

    # Heat Loss Results
    current_heat_loss: HeatLossQuantification = Field(
        ...,
        description="Current heat loss (with existing insulation or bare)"
    )
    proposed_heat_loss: Optional[HeatLossQuantification] = Field(
        None,
        description="Heat loss with proposed insulation"
    )
    bare_surface_heat_loss: Optional[HeatLossQuantification] = Field(
        None,
        description="Heat loss if surface were bare"
    )

    # Economic Analysis
    economic_thickness: Optional[EconomicThicknessResult] = Field(
        None,
        description="Economic thickness optimization results"
    )

    # ROI Analysis
    roi_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed ROI analysis"
    )

    # Recommendations
    recommendations: List[InsulationRecommendation] = Field(
        ...,
        description="Prioritized recommendations"
    )
    overall_priority: MaintenancePriority = Field(
        ...,
        description="Overall maintenance priority"
    )

    # Material Comparison
    material_comparisons: List[MaterialComparison] = Field(
        default_factory=list,
        description="Material comparison results"
    )
    recommended_material_id: Optional[str] = Field(
        None,
        description="ID of recommended material"
    )

    # Explainability
    explainability: Optional[ExplainabilityReport] = Field(
        None,
        description="SHAP/LIME style explainability"
    )

    # Thermal Map (if IR data available)
    thermal_map: List[ThermalMapPoint] = Field(
        default_factory=list,
        description="Thermal map points from IR data"
    )

    # Summary Statistics
    total_heat_loss_savings_w: float = Field(
        default=0,
        description="Total heat loss savings potential"
    )
    total_annual_savings_usd: float = Field(
        default=0,
        description="Total annual savings potential"
    )
    total_implementation_cost_usd: float = Field(
        default=0,
        description="Total implementation cost"
    )

    # Provenance and Audit
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing duration in milliseconds"
    )
    validation_status: str = Field(
        ...,
        pattern="^(PASS|FAIL)$",
        description="PASS or FAIL"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages if any"
    )
    calculation_method: str = Field(
        default="deterministic",
        description="Calculation method used"
    )


class AgentConfig(BaseModel):
    """Configuration for InsulationAnalysisAgent."""

    agent_id: str = Field(
        default="GL-015",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="INSULSCAN",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # Analysis defaults
    default_emissivity: float = Field(
        default=0.90,
        ge=0.01,
        le=1.0,
        description="Default surface emissivity"
    )
    min_thickness_m: float = Field(
        default=0.025,
        gt=0,
        description="Minimum thickness for analysis"
    )
    max_thickness_m: float = Field(
        default=0.300,
        gt=0,
        description="Maximum thickness for analysis"
    )
    thickness_increment_m: float = Field(
        default=0.0125,
        gt=0,
        description="Thickness increment for optimization"
    )

    # Priority thresholds
    critical_heat_loss_threshold_w_m2: float = Field(
        default=1000,
        gt=0,
        description="Heat flux threshold for critical priority"
    )
    high_heat_loss_threshold_w_m2: float = Field(
        default=500,
        gt=0,
        description="Heat flux threshold for high priority"
    )

    # Safety
    max_safe_surface_temp_c: float = Field(
        default=60,
        description="Maximum safe surface temperature for personnel protection"
    )
