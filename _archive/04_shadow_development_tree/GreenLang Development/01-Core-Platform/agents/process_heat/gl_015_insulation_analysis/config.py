"""
GL-015 INSULSCAN - Configuration Schemas

Configuration models for the Insulation Analysis Agent including
economic parameters, safety thresholds, and IR survey settings.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, validator


class TemperatureUnit(Enum):
    """Temperature unit options."""
    FAHRENHEIT = "F"
    CELSIUS = "C"
    KELVIN = "K"


class LengthUnit(Enum):
    """Length unit options."""
    INCHES = "in"
    FEET = "ft"
    METERS = "m"
    MILLIMETERS = "mm"


class CurrencyCode(Enum):
    """Currency codes."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"


class EconomicConfig(BaseModel):
    """Economic parameters for insulation optimization."""

    # Energy costs
    energy_cost_per_mmbtu: float = Field(
        default=8.50,
        gt=0,
        description="Energy cost ($/MMBTU)"
    )
    electricity_cost_per_kwh: float = Field(
        default=0.12,
        gt=0,
        description="Electricity cost ($/kWh)"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency code"
    )

    # Operating parameters
    operating_hours_per_year: int = Field(
        default=8760,
        ge=1000,
        le=8784,
        description="Annual operating hours"
    )
    plant_lifetime_years: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Expected plant lifetime (years)"
    )
    discount_rate_pct: float = Field(
        default=10.0,
        ge=0,
        le=30,
        description="Discount rate for NPV (%)"
    )
    inflation_rate_pct: float = Field(
        default=2.5,
        ge=0,
        le=15,
        description="Annual inflation rate (%)"
    )

    # Insulation costs (installed)
    insulation_cost_per_sqft: Dict[str, float] = Field(
        default={
            "calcium_silicate": 12.50,
            "mineral_wool": 8.75,
            "fiberglass": 7.50,
            "cellular_glass": 18.00,
            "perlite": 10.25,
            "polyurethane": 15.00,
            "polystyrene": 6.50,
            "phenolic": 22.00,
            "aerogel": 85.00,
        },
        description="Installed insulation cost per sqft by material type"
    )

    # Jacketing costs
    jacketing_cost_per_sqft: Dict[str, float] = Field(
        default={
            "aluminum": 4.50,
            "stainless_steel": 12.00,
            "pvc": 2.50,
            "galvanized": 3.75,
            "painted_aluminum": 5.50,
        },
        description="Jacketing cost per sqft by material type"
    )

    # Labor rates
    labor_rate_per_hour: float = Field(
        default=85.0,
        gt=0,
        description="Insulation labor rate ($/hr)"
    )
    scaffolding_cost_multiplier: float = Field(
        default=1.3,
        ge=1.0,
        le=3.0,
        description="Cost multiplier for elevated work"
    )

    # Economic analysis parameters
    minimum_payback_years: float = Field(
        default=2.0,
        ge=0.5,
        le=10,
        description="Minimum acceptable payback (years)"
    )
    target_roi_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Target ROI (%)"
    )

    class Config:
        use_enum_values = True


class SafetyConfig(BaseModel):
    """Safety configuration for surface temperature limits."""

    # OSHA surface temperature limits
    max_touch_temperature_c: float = Field(
        default=60.0,
        ge=40,
        le=80,
        description="Maximum touchable surface temperature (C) per OSHA"
    )
    max_touch_temperature_f: float = Field(
        default=140.0,
        ge=104,
        le=176,
        description="Maximum touchable surface temperature (F) per OSHA"
    )

    # Contact duration thresholds (seconds to burn injury)
    burn_threshold_map: Dict[str, Dict[str, float]] = Field(
        default={
            "metal": {
                "48C": 60.0,
                "51C": 10.0,
                "60C": 1.0,
                "70C": 0.1,
            },
            "non_metal": {
                "48C": 480.0,
                "51C": 60.0,
                "60C": 10.0,
                "70C": 1.0,
            },
        },
        description="Time to burn injury by surface material and temperature"
    )

    # Personnel protection settings
    personnel_protection_zone_ft: float = Field(
        default=3.0,
        ge=1,
        le=10,
        description="Protection zone radius (ft)"
    )
    warning_threshold_c: float = Field(
        default=50.0,
        ge=40,
        le=60,
        description="Warning threshold temperature (C)"
    )
    alarm_threshold_c: float = Field(
        default=55.0,
        ge=45,
        le=65,
        description="Alarm threshold temperature (C)"
    )

    # SIL level
    sil_level: int = Field(
        default=2,
        ge=0,
        le=4,
        description="Safety Integrity Level"
    )

    class Config:
        use_enum_values = True


class IRSurveyConfig(BaseModel):
    """Configuration for IR thermography surveys."""

    # Camera settings
    camera_model: Optional[str] = Field(
        default=None,
        description="IR camera model identifier"
    )
    emissivity_default: float = Field(
        default=0.95,
        ge=0.1,
        le=1.0,
        description="Default surface emissivity"
    )
    reflected_temperature_f: float = Field(
        default=70.0,
        description="Reflected apparent temperature (F)"
    )
    distance_ft: float = Field(
        default=6.0,
        gt=0,
        le=100,
        description="Default camera distance (ft)"
    )

    # Analysis thresholds
    hot_spot_threshold_delta_f: float = Field(
        default=15.0,
        ge=5,
        le=50,
        description="Delta-T threshold for hot spot detection (F)"
    )
    damaged_insulation_threshold_pct: float = Field(
        default=25.0,
        ge=10,
        le=100,
        description="Heat loss increase threshold for damaged insulation (%)"
    )
    missing_insulation_threshold_pct: float = Field(
        default=100.0,
        ge=50,
        le=500,
        description="Heat loss increase threshold for missing insulation (%)"
    )

    # Survey parameters
    ambient_temperature_correction: bool = Field(
        default=True,
        description="Apply ambient temperature correction"
    )
    wind_speed_correction: bool = Field(
        default=True,
        description="Apply wind speed correction for outdoor surveys"
    )
    solar_loading_correction: bool = Field(
        default=True,
        description="Apply solar loading correction for outdoor surveys"
    )

    # Reporting
    include_thermal_images: bool = Field(
        default=True,
        description="Include thermal images in report"
    )
    temperature_scale_range: str = Field(
        default="auto",
        description="Temperature scale: auto, fixed, or range like '50-250'"
    )

    class Config:
        use_enum_values = True


class CondensationConfig(BaseModel):
    """Configuration for condensation prevention analysis."""

    # Design conditions
    design_ambient_temp_f: float = Field(
        default=95.0,
        description="Design ambient temperature (F)"
    )
    design_relative_humidity_pct: float = Field(
        default=90.0,
        ge=0,
        le=100,
        description="Design relative humidity (%)"
    )
    design_dew_point_margin_f: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Minimum margin above dew point (F)"
    )

    # Vapor barrier settings
    vapor_barrier_required: bool = Field(
        default=True,
        description="Require vapor barrier for cold service"
    )
    vapor_barrier_perm_rating: float = Field(
        default=0.02,
        ge=0,
        le=1.0,
        description="Maximum vapor barrier perm rating"
    )

    # Cold service thresholds
    cold_service_threshold_f: float = Field(
        default=60.0,
        description="Temperature below which is considered cold service (F)"
    )
    cryogenic_threshold_f: float = Field(
        default=-100.0,
        description="Temperature below which is considered cryogenic (F)"
    )


class InsulationAnalysisConfig(BaseModel):
    """Complete configuration for GL-015 INSULSCAN agent."""

    # Identity
    config_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Configuration identifier"
    )
    facility_id: str = Field(
        ...,
        description="Facility identifier"
    )
    config_name: str = Field(
        default="Default Insulation Analysis Config",
        description="Configuration name"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )

    # Units
    temperature_unit: TemperatureUnit = Field(
        default=TemperatureUnit.FAHRENHEIT,
        description="Default temperature unit"
    )
    length_unit: LengthUnit = Field(
        default=LengthUnit.INCHES,
        description="Default length unit"
    )

    # Sub-configurations
    economic: EconomicConfig = Field(
        default_factory=EconomicConfig,
        description="Economic analysis parameters"
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )
    ir_survey: IRSurveyConfig = Field(
        default_factory=IRSurveyConfig,
        description="IR thermography configuration"
    )
    condensation: CondensationConfig = Field(
        default_factory=CondensationConfig,
        description="Condensation prevention configuration"
    )

    # Default ambient conditions
    default_ambient_temp_f: float = Field(
        default=77.0,
        description="Default ambient temperature (F)"
    )
    default_wind_speed_mph: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Default wind speed (mph)"
    )
    default_surface_orientation: str = Field(
        default="horizontal",
        description="Default surface orientation (horizontal, vertical)"
    )

    # Calculation settings
    convergence_tolerance: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.1,
        description="Iterative calculation convergence tolerance"
    )
    max_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum iterations for convergence"
    )
    include_radiation: bool = Field(
        default=True,
        description="Include radiation heat transfer"
    )
    include_convection: bool = Field(
        default=True,
        description="Include convection heat transfer"
    )

    # Reporting
    generate_recommendations: bool = Field(
        default=True,
        description="Generate optimization recommendations"
    )
    include_sensitivity_analysis: bool = Field(
        default=True,
        description="Include sensitivity analysis in results"
    )
    detailed_layer_analysis: bool = Field(
        default=True,
        description="Include detailed layer-by-layer analysis"
    )

    # Audit and provenance
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Configuration creation timestamp"
    )
    created_by: Optional[str] = Field(
        default=None,
        description="Configuration creator"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        use_enum_values = True

    @validator('safety')
    def validate_safety_consistency(cls, v):
        """Validate safety configuration consistency."""
        # Ensure F and C temperatures are consistent
        expected_f = v.max_touch_temperature_c * 9/5 + 32
        if abs(v.max_touch_temperature_f - expected_f) > 1.0:
            # Auto-correct to maintain consistency
            v.max_touch_temperature_f = expected_f
        return v
