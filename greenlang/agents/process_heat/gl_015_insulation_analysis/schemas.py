"""
GL-015 INSULSCAN - Schema Definitions

Pydantic models for insulation analysis inputs, outputs, and results.
Includes geometries, material specifications, and analysis results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field, validator


class GeometryType(Enum):
    """Geometry type for insulation analysis."""
    PIPE = "pipe"
    VESSEL = "vessel"
    FLAT_SURFACE = "flat_surface"
    TANK = "tank"
    DUCT = "duct"
    EQUIPMENT = "equipment"


class InsulationCondition(Enum):
    """Condition of existing insulation."""
    NEW = "new"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DAMAGED = "damaged"
    MISSING = "missing"
    SATURATED = "saturated"


class ServiceType(Enum):
    """Service type classification."""
    HOT = "hot"
    COLD = "cold"
    CRYOGENIC = "cryogenic"
    DUAL_TEMPERATURE = "dual_temperature"


class JacketingType(Enum):
    """Jacketing material type."""
    ALUMINUM = "aluminum"
    STAINLESS_STEEL = "stainless_steel"
    GALVANIZED = "galvanized"
    PVC = "pvc"
    PAINTED_ALUMINUM = "painted_aluminum"
    NONE = "none"


class PipeGeometry(BaseModel):
    """Pipe geometry specification."""

    nominal_pipe_size_in: float = Field(
        ...,
        gt=0,
        le=120,
        description="Nominal pipe size (inches)"
    )
    outer_diameter_in: Optional[float] = Field(
        default=None,
        gt=0,
        description="Actual outer diameter (inches)"
    )
    pipe_schedule: str = Field(
        default="40",
        description="Pipe schedule (e.g., 40, 80, STD, XS)"
    )
    pipe_length_ft: float = Field(
        ...,
        gt=0,
        description="Pipe length (feet)"
    )
    pipe_material: str = Field(
        default="carbon_steel",
        description="Pipe material"
    )
    orientation: str = Field(
        default="horizontal",
        description="Orientation: horizontal, vertical"
    )

    @validator('outer_diameter_in', always=True)
    def set_outer_diameter(cls, v, values):
        """Calculate outer diameter from NPS if not provided."""
        if v is not None:
            return v
        nps = values.get('nominal_pipe_size_in')
        if nps is None:
            return None
        # Standard pipe OD lookup (simplified)
        nps_to_od = {
            0.5: 0.840, 0.75: 1.050, 1.0: 1.315, 1.25: 1.660, 1.5: 1.900,
            2.0: 2.375, 2.5: 2.875, 3.0: 3.500, 3.5: 4.000, 4.0: 4.500,
            5.0: 5.563, 6.0: 6.625, 8.0: 8.625, 10.0: 10.750, 12.0: 12.750,
            14.0: 14.000, 16.0: 16.000, 18.0: 18.000, 20.0: 20.000, 24.0: 24.000,
            30.0: 30.000, 36.0: 36.000, 42.0: 42.000, 48.0: 48.000,
        }
        return nps_to_od.get(float(nps), nps + 0.5)


class VesselGeometry(BaseModel):
    """Vessel geometry specification."""

    vessel_diameter_ft: float = Field(
        ...,
        gt=0,
        description="Vessel diameter (feet)"
    )
    vessel_length_ft: float = Field(
        ...,
        gt=0,
        description="Vessel length (feet)"
    )
    vessel_type: str = Field(
        default="horizontal_cylinder",
        description="Vessel type: horizontal_cylinder, vertical_cylinder, sphere"
    )
    head_type: str = Field(
        default="2:1_elliptical",
        description="Head type: 2:1_elliptical, hemispherical, flat, torispherical"
    )
    include_heads: bool = Field(
        default=True,
        description="Include heads in analysis"
    )
    shell_thickness_in: float = Field(
        default=0.5,
        gt=0,
        description="Shell thickness (inches)"
    )


class FlatSurfaceGeometry(BaseModel):
    """Flat surface geometry specification."""

    length_ft: float = Field(
        ...,
        gt=0,
        description="Surface length (feet)"
    )
    width_ft: float = Field(
        ...,
        gt=0,
        description="Surface width (feet)"
    )
    surface_area_sqft: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total surface area (sqft) - calculated if not provided"
    )
    orientation: str = Field(
        default="vertical",
        description="Orientation: horizontal_up, horizontal_down, vertical"
    )
    base_material: str = Field(
        default="carbon_steel",
        description="Base surface material"
    )

    @validator('surface_area_sqft', always=True)
    def calculate_area(cls, v, values):
        """Calculate surface area if not provided."""
        if v is not None:
            return v
        length = values.get('length_ft', 0)
        width = values.get('width_ft', 0)
        return length * width


class InsulationLayer(BaseModel):
    """Insulation layer specification."""

    layer_number: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Layer number (1 = innermost)"
    )
    material_id: str = Field(
        ...,
        description="Insulation material identifier"
    )
    thickness_in: float = Field(
        ...,
        gt=0,
        le=24,
        description="Layer thickness (inches)"
    )
    density_pcf: Optional[float] = Field(
        default=None,
        gt=0,
        description="Material density (lb/ft3)"
    )
    condition: InsulationCondition = Field(
        default=InsulationCondition.GOOD,
        description="Layer condition"
    )
    condition_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Condition adjustment factor (1.0 = good)"
    )
    age_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="Age of insulation (years)"
    )

    class Config:
        use_enum_values = True


class JacketingSpec(BaseModel):
    """Jacketing specification."""

    jacketing_type: JacketingType = Field(
        default=JacketingType.ALUMINUM,
        description="Jacketing material type"
    )
    thickness_in: float = Field(
        default=0.016,
        gt=0,
        le=0.1,
        description="Jacketing thickness (inches)"
    )
    emissivity: float = Field(
        default=0.1,
        ge=0.03,
        le=0.95,
        description="Surface emissivity"
    )
    corroded: bool = Field(
        default=False,
        description="Jacketing is corroded"
    )
    paint_color: Optional[str] = Field(
        default=None,
        description="Paint color if painted"
    )

    class Config:
        use_enum_values = True


class InsulationInput(BaseModel):
    """Input data for insulation analysis."""

    # Identity
    item_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Item identifier"
    )
    item_name: str = Field(
        default="Unnamed Item",
        description="Item name/description"
    )
    tag_number: Optional[str] = Field(
        default=None,
        description="Equipment tag number"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Service conditions
    operating_temperature_f: float = Field(
        ...,
        ge=-459.67,
        le=2500,
        description="Operating temperature (F)"
    )
    ambient_temperature_f: float = Field(
        default=77.0,
        ge=-100,
        le=150,
        description="Ambient temperature (F)"
    )
    wind_speed_mph: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Wind speed (mph)"
    )
    relative_humidity_pct: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Relative humidity (%)"
    )
    service_type: ServiceType = Field(
        default=ServiceType.HOT,
        description="Service type classification"
    )

    # Geometry
    geometry_type: GeometryType = Field(
        ...,
        description="Geometry type"
    )
    pipe_geometry: Optional[PipeGeometry] = Field(
        default=None,
        description="Pipe geometry (if applicable)"
    )
    vessel_geometry: Optional[VesselGeometry] = Field(
        default=None,
        description="Vessel geometry (if applicable)"
    )
    flat_geometry: Optional[FlatSurfaceGeometry] = Field(
        default=None,
        description="Flat surface geometry (if applicable)"
    )

    # Insulation
    insulation_layers: List[InsulationLayer] = Field(
        default_factory=list,
        description="Insulation layers (empty for bare surface)"
    )
    jacketing: Optional[JacketingSpec] = Field(
        default=None,
        description="Jacketing specification"
    )

    # Analysis options
    calculate_economic_thickness: bool = Field(
        default=True,
        description="Calculate economic thickness"
    )
    check_surface_temperature: bool = Field(
        default=True,
        description="Check surface temperature compliance"
    )
    check_condensation: bool = Field(
        default=True,
        description="Check condensation prevention (cold service)"
    )
    personnel_protection_required: bool = Field(
        default=True,
        description="Personnel protection is required"
    )

    # Location
    location_indoor: bool = Field(
        default=True,
        description="Equipment is indoors"
    )
    location_elevation_ft: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Elevation above grade (feet)"
    )

    class Config:
        use_enum_values = True

    @validator('service_type', always=True)
    def auto_detect_service_type(cls, v, values):
        """Auto-detect service type from operating temperature."""
        if v is not None:
            return v
        temp = values.get('operating_temperature_f', 77)
        if temp < -100:
            return ServiceType.CRYOGENIC
        elif temp < 60:
            return ServiceType.COLD
        else:
            return ServiceType.HOT


class HeatLossResult(BaseModel):
    """Heat loss calculation result."""

    # Primary results
    heat_loss_btu_hr: float = Field(
        ...,
        description="Heat loss rate (BTU/hr)"
    )
    heat_loss_btu_hr_ft: Optional[float] = Field(
        default=None,
        description="Heat loss per linear foot (BTU/hr-ft)"
    )
    heat_loss_btu_hr_sqft: Optional[float] = Field(
        default=None,
        description="Heat loss per square foot (BTU/hr-sqft)"
    )

    # Surface conditions
    outer_surface_temperature_f: float = Field(
        ...,
        description="Outer surface temperature (F)"
    )
    inner_surface_temperature_f: Optional[float] = Field(
        default=None,
        description="Insulation inner surface temperature (F)"
    )

    # Heat transfer breakdown
    convection_heat_transfer_btu_hr: float = Field(
        ...,
        description="Convection heat transfer (BTU/hr)"
    )
    radiation_heat_transfer_btu_hr: float = Field(
        ...,
        description="Radiation heat transfer (BTU/hr)"
    )
    total_thermal_resistance_hr_f_btu: float = Field(
        ...,
        description="Total thermal resistance (hr-F/BTU)"
    )

    # Layer analysis
    layer_temperatures_f: List[float] = Field(
        default_factory=list,
        description="Interface temperatures between layers (F)"
    )
    layer_resistances_hr_f_btu: List[float] = Field(
        default_factory=list,
        description="Thermal resistance of each layer"
    )

    # Comparison
    bare_surface_heat_loss_btu_hr: Optional[float] = Field(
        default=None,
        description="Heat loss if bare (no insulation)"
    )
    heat_loss_reduction_pct: Optional[float] = Field(
        default=None,
        description="Heat loss reduction from insulation (%)"
    )

    # Methodology
    calculation_method: str = Field(
        default="ASTM_C680",
        description="Calculation method used"
    )
    formula_reference: str = Field(
        default="ASTM C680-19",
        description="Standard reference"
    )


class EconomicThicknessResult(BaseModel):
    """Economic thickness optimization result."""

    # Optimal thickness
    optimal_thickness_in: float = Field(
        ...,
        ge=0,
        description="Economically optimal thickness (inches)"
    )
    optimal_thickness_layers: List[float] = Field(
        default_factory=list,
        description="Optimal thickness for each layer (inches)"
    )
    recommended_material: str = Field(
        ...,
        description="Recommended insulation material"
    )

    # Current vs optimal
    current_thickness_in: float = Field(
        default=0.0,
        ge=0,
        description="Current insulation thickness (inches)"
    )
    additional_thickness_needed_in: float = Field(
        default=0.0,
        ge=0,
        description="Additional thickness needed (inches)"
    )

    # Heat loss comparison
    current_heat_loss_btu_hr: float = Field(
        ...,
        description="Current heat loss (BTU/hr)"
    )
    optimal_heat_loss_btu_hr: float = Field(
        ...,
        description="Heat loss at optimal thickness (BTU/hr)"
    )
    heat_loss_savings_btu_hr: float = Field(
        ...,
        description="Heat loss savings (BTU/hr)"
    )

    # Economic analysis
    annual_energy_cost_current_usd: float = Field(
        ...,
        description="Annual energy cost at current state ($)"
    )
    annual_energy_cost_optimal_usd: float = Field(
        ...,
        description="Annual energy cost at optimal ($)"
    )
    annual_savings_usd: float = Field(
        ...,
        description="Annual energy savings ($)"
    )
    insulation_cost_usd: float = Field(
        ...,
        description="Insulation material cost ($)"
    )
    installation_cost_usd: float = Field(
        ...,
        description="Installation labor cost ($)"
    )
    total_project_cost_usd: float = Field(
        ...,
        description="Total project cost ($)"
    )
    simple_payback_years: float = Field(
        ...,
        description="Simple payback period (years)"
    )
    npv_usd: float = Field(
        ...,
        description="Net present value ($)"
    )
    roi_pct: float = Field(
        ...,
        description="Return on investment (%)"
    )

    # Methodology
    calculation_method: str = Field(
        default="NAIMA_3E_PLUS",
        description="Calculation method"
    )


class SurfaceTemperatureResult(BaseModel):
    """Surface temperature compliance result."""

    # Temperature results
    calculated_surface_temp_f: float = Field(
        ...,
        description="Calculated surface temperature (F)"
    )
    calculated_surface_temp_c: float = Field(
        ...,
        description="Calculated surface temperature (C)"
    )
    osha_limit_temp_f: float = Field(
        default=140.0,
        description="OSHA limit temperature (F)"
    )
    osha_limit_temp_c: float = Field(
        default=60.0,
        description="OSHA limit temperature (C)"
    )

    # Compliance
    is_compliant: bool = Field(
        ...,
        description="Meets OSHA requirements"
    )
    margin_f: float = Field(
        ...,
        description="Margin below OSHA limit (F)"
    )
    margin_c: float = Field(
        ...,
        description="Margin below OSHA limit (C)"
    )

    # Required thickness
    minimum_thickness_for_compliance_in: Optional[float] = Field(
        default=None,
        description="Minimum thickness for OSHA compliance (inches)"
    )
    current_thickness_in: float = Field(
        default=0.0,
        description="Current insulation thickness (inches)"
    )
    additional_thickness_needed_in: float = Field(
        default=0.0,
        ge=0,
        description="Additional thickness needed for compliance (inches)"
    )

    # Contact hazard
    contact_burn_risk: str = Field(
        default="none",
        description="Contact burn risk level: none, low, medium, high, extreme"
    )
    time_to_burn_injury_sec: Optional[float] = Field(
        default=None,
        description="Time to burn injury on contact (seconds)"
    )

    # Personnel protection
    personnel_protection_required: bool = Field(
        ...,
        description="Personnel protection measures required"
    )
    recommended_protection: List[str] = Field(
        default_factory=list,
        description="Recommended protection measures"
    )


class CondensationAnalysisResult(BaseModel):
    """Condensation prevention analysis result."""

    # Dew point analysis
    ambient_dew_point_f: float = Field(
        ...,
        description="Ambient dew point temperature (F)"
    )
    surface_temperature_f: float = Field(
        ...,
        description="Outer surface temperature (F)"
    )
    margin_above_dew_point_f: float = Field(
        ...,
        description="Surface temp margin above dew point (F)"
    )

    # Condensation risk
    condensation_risk: bool = Field(
        ...,
        description="Condensation is likely"
    )
    condensation_risk_level: str = Field(
        default="none",
        description="Risk level: none, low, medium, high"
    )

    # Prevention requirements
    minimum_thickness_for_prevention_in: float = Field(
        ...,
        description="Minimum thickness to prevent condensation (inches)"
    )
    vapor_barrier_required: bool = Field(
        ...,
        description="Vapor barrier is required"
    )
    vapor_barrier_location: str = Field(
        default="innermost",
        description="Vapor barrier location: innermost, between_layers"
    )

    # Current status
    current_thickness_in: float = Field(
        default=0.0,
        description="Current insulation thickness (inches)"
    )
    additional_thickness_needed_in: float = Field(
        default=0.0,
        ge=0,
        description="Additional thickness needed (inches)"
    )
    has_adequate_vapor_barrier: bool = Field(
        default=False,
        description="Current system has adequate vapor barrier"
    )


class IRHotSpot(BaseModel):
    """IR survey hot spot identification."""

    spot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:6],
        description="Hot spot identifier"
    )
    location_description: str = Field(
        ...,
        description="Location description"
    )
    measured_temperature_f: float = Field(
        ...,
        description="Measured temperature (F)"
    )
    expected_temperature_f: float = Field(
        ...,
        description="Expected temperature (F)"
    )
    delta_t_f: float = Field(
        ...,
        description="Temperature deviation (F)"
    )
    severity: str = Field(
        default="medium",
        description="Severity: low, medium, high, critical"
    )
    estimated_heat_loss_btu_hr: Optional[float] = Field(
        default=None,
        description="Estimated additional heat loss (BTU/hr)"
    )
    probable_cause: str = Field(
        default="unknown",
        description="Probable cause of anomaly"
    )
    recommended_action: str = Field(
        ...,
        description="Recommended corrective action"
    )
    image_reference: Optional[str] = Field(
        default=None,
        description="Reference to thermal image"
    )


class IRSurveyResult(BaseModel):
    """IR thermography survey result."""

    # Survey identification
    survey_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Survey identifier"
    )
    survey_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Survey date"
    )

    # Survey conditions
    ambient_temperature_f: float = Field(
        ...,
        description="Ambient temperature during survey (F)"
    )
    wind_speed_mph: float = Field(
        default=0.0,
        description="Wind speed during survey (mph)"
    )
    humidity_pct: float = Field(
        default=50.0,
        description="Relative humidity (%)"
    )
    camera_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="IR camera settings"
    )

    # Analysis results
    items_surveyed: int = Field(
        default=0,
        ge=0,
        description="Number of items surveyed"
    )
    hot_spots_identified: List[IRHotSpot] = Field(
        default_factory=list,
        description="Hot spots identified"
    )
    total_anomalies: int = Field(
        default=0,
        ge=0,
        description="Total anomalies found"
    )

    # Heat loss impact
    total_excess_heat_loss_btu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Total excess heat loss from anomalies (BTU/hr)"
    )
    annual_excess_energy_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual excess energy cost ($)"
    )

    # Recommendations
    critical_repairs_needed: int = Field(
        default=0,
        ge=0,
        description="Number of critical repairs needed"
    )
    estimated_repair_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated repair cost ($)"
    )
    estimated_annual_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated annual savings from repairs ($)"
    )


class InsulationRecommendation(BaseModel):
    """Insulation improvement recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    category: str = Field(
        ...,
        description="Category: economic, safety, condensation, maintenance"
    )
    priority: str = Field(
        default="medium",
        description="Priority: low, medium, high, critical"
    )
    title: str = Field(
        ...,
        description="Recommendation title"
    )
    description: str = Field(
        ...,
        description="Detailed description"
    )

    # Current vs recommended
    current_state: str = Field(
        ...,
        description="Current state description"
    )
    recommended_action: str = Field(
        ...,
        description="Recommended action"
    )
    recommended_thickness_in: Optional[float] = Field(
        default=None,
        description="Recommended insulation thickness (inches)"
    )
    recommended_material: Optional[str] = Field(
        default=None,
        description="Recommended insulation material"
    )

    # Economics
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated implementation cost ($)"
    )
    annual_savings_usd: float = Field(
        default=0.0,
        ge=0,
        description="Estimated annual savings ($)"
    )
    payback_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="Simple payback (years)"
    )

    # Compliance
    addresses_compliance_issue: bool = Field(
        default=False,
        description="Addresses a compliance issue"
    )
    compliance_standard: Optional[str] = Field(
        default=None,
        description="Relevant compliance standard"
    )


class InsulationOutput(BaseModel):
    """Complete output from insulation analysis."""

    # Identity
    item_id: str = Field(
        ...,
        description="Item identifier"
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Status
    status: str = Field(
        default="success",
        description="Processing status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )

    # Analysis results
    heat_loss: HeatLossResult = Field(
        ...,
        description="Heat loss calculation results"
    )
    economic_thickness: Optional[EconomicThicknessResult] = Field(
        default=None,
        description="Economic thickness analysis"
    )
    surface_temperature: Optional[SurfaceTemperatureResult] = Field(
        default=None,
        description="Surface temperature compliance"
    )
    condensation_analysis: Optional[CondensationAnalysisResult] = Field(
        default=None,
        description="Condensation prevention analysis"
    )
    ir_survey: Optional[IRSurveyResult] = Field(
        default=None,
        description="IR survey results"
    )

    # Recommendations
    recommendations: List[InsulationRecommendation] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )

    # KPIs
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Alerts
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )
    input_hash: Optional[str] = Field(
        default=None,
        description="Input data hash"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
