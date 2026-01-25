# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY - Configuration Module

This module defines all configuration schemas for the Burner Maintenance
Predictor agent, including burner specifications, flame analysis thresholds,
maintenance prediction parameters, fuel quality limits, replacement planning
economics, CMMS integration settings, and safety compliance configuration.

Configuration follows GreenLang patterns with Pydantic validation and
sensible defaults for industrial burner applications.

Standards Compliance:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - API 535: Burners for Fired Heaters in General Refinery Service
    - API 560: Fired Heaters for General Refinery Service
    - API 556: Instrumentation, Control, and Protective Systems
    - ISA 84: Safety Instrumented Systems (SIL ratings)

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
    ...     GL021Config,
    ...     BurnerConfig,
    ...     BurnerType,
    ... )
    >>> config = GL021Config(
    ...     burner=BurnerConfig(
    ...         burner_id="BNR-001",
    ...         burner_type=BurnerType.LOW_NOX,
    ...         capacity_mmbtu_hr=50.0,
    ...     )
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS - BURNER TYPES AND CLASSIFICATIONS
# =============================================================================

class BurnerType(str, Enum):
    """
    Types of industrial burners supported by BURNERSENTRY.

    Classification based on fuel type, emissions profile, and design:
    - GAS: Standard gas-fired burners for natural gas, propane, etc.
    - OIL: Liquid fuel burners for fuel oil #2, #6, diesel
    - DUAL_FUEL: Burners capable of gas and oil operation
    - LOW_NOX: Low-NOx burners meeting EPA regulations
    - ULTRA_LOW_NOX: Ultra-low NOx (<9 ppm) for stringent air quality
    """
    GAS = "gas"
    OIL = "oil"
    DUAL_FUEL = "dual_fuel"
    LOW_NOX = "low_nox"
    ULTRA_LOW_NOX = "ultra_low_nox"
    STAGED_AIR = "staged_air"
    STAGED_FUEL = "staged_fuel"
    FLUE_GAS_RECIRCULATION = "fgr"
    REGENERATIVE = "regenerative"
    RADIANT_TUBE = "radiant_tube"
    LINE_BURNER = "line_burner"


class FuelType(str, Enum):
    """
    Fuel types supported by burner systems.

    Includes gaseous, liquid, and alternative fuels with their
    typical heating values and combustion characteristics.
    """
    # Gaseous fuels
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    BUTANE = "butane"
    LNG = "lng"
    LPG = "lpg"
    REFINERY_GAS = "refinery_gas"
    COKE_OVEN_GAS = "coke_oven_gas"
    BLAST_FURNACE_GAS = "blast_furnace_gas"
    BIOGAS = "biogas"
    LANDFILL_GAS = "landfill_gas"
    HYDROGEN = "hydrogen"
    SYNGAS = "syngas"

    # Liquid fuels
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_4 = "fuel_oil_4"
    FUEL_OIL_6 = "fuel_oil_6"
    DIESEL = "diesel"
    KEROSENE = "kerosene"
    WASTE_OIL = "waste_oil"
    BIODIESEL = "biodiesel"


class BurnerManufacturer(str, Enum):
    """Major industrial burner manufacturers."""
    ZEECO = "zeeco"
    JOHN_ZINK = "john_zink"
    FIVES = "fives"
    HONEYWELL = "honeywell"
    MAXON = "maxon"
    ECLIPSE = "eclipse"
    WEBSTER = "webster"
    RIELLO = "riello"
    WEISHAUPT = "weishaupt"
    BALTUR = "baltur"
    OILON = "oilon"
    DUNPHY = "dunphy"
    HAMWORTHY = "hamworthy"
    OTHER = "other"


class CMMSType(str, Enum):
    """Supported Computerized Maintenance Management Systems."""
    SAP_PM = "sap_pm"
    IBM_MAXIMO = "ibm_maximo"
    ORACLE_EAM = "oracle_eam"
    INFOR_EAM = "infor_eam"
    EMAINT = "emaint"
    FIIX = "fiix"
    UPTIMEWORKS = "uptime_works"
    MPULSE = "mpulse"
    LIMBLE = "limble"
    CUSTOM_API = "custom_api"


class SafetyIntegrityLevel(str, Enum):
    """IEC 61508 / ISA 84 Safety Integrity Levels."""
    SIL_1 = "sil_1"  # PFD 0.1 to 0.01
    SIL_2 = "sil_2"  # PFD 0.01 to 0.001
    SIL_3 = "sil_3"  # PFD 0.001 to 0.0001
    SIL_4 = "sil_4"  # PFD 0.0001 to 0.00001
    NON_SIL = "non_sil"


class BMSStandard(str, Enum):
    """Burner Management System standards compliance."""
    NFPA_85 = "nfpa_85"
    NFPA_86 = "nfpa_86"
    EN_298 = "en_298"
    EN_746 = "en_746"
    API_556 = "api_556"
    IEC_61511 = "iec_61511"


class FlameDetectorType(str, Enum):
    """Types of flame detection sensors."""
    UV = "uv"  # Ultraviolet
    IR = "ir"  # Infrared
    UV_IR = "uv_ir"  # Combined UV/IR
    FLAME_ROD = "flame_rod"  # Ionization
    VISIBLE = "visible"  # Visible light
    MULTI_SPECTRUM = "multi_spectrum"


class AlertSeverity(str, Enum):
    """Alert severity levels for burner monitoring."""
    GOOD = "good"
    ADVISORY = "advisory"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"


# =============================================================================
# BURNER CONFIGURATION
# =============================================================================

class BurnerNozzleConfig(BaseModel):
    """Configuration for burner nozzle/tip specifications."""

    nozzle_id: str = Field(
        default="primary",
        description="Nozzle identifier (primary, secondary, pilot)"
    )
    nozzle_type: str = Field(
        default="standard",
        description="Nozzle type (standard, swirl, staged)"
    )
    orifice_diameter_mm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Nozzle orifice diameter in mm"
    )
    spray_angle_deg: Optional[float] = Field(
        default=None,
        ge=0,
        le=180,
        description="Spray angle in degrees (for oil burners)"
    )
    flow_capacity_gph: Optional[float] = Field(
        default=None,
        gt=0,
        description="Flow capacity in gallons per hour"
    )
    material: str = Field(
        default="stainless_steel",
        description="Nozzle material"
    )
    expected_lifetime_hours: float = Field(
        default=8760.0,
        gt=0,
        description="Expected nozzle lifetime in hours"
    )


class BurnerRefractoryConfig(BaseModel):
    """Configuration for burner refractory/tile specifications."""

    refractory_type: str = Field(
        default="ceramic_fiber",
        description="Refractory type (ceramic_fiber, castable, brick)"
    )
    max_temperature_c: float = Field(
        default=1650.0,
        gt=0,
        description="Maximum service temperature in Celsius"
    )
    thickness_mm: float = Field(
        default=100.0,
        gt=0,
        description="Refractory thickness in mm"
    )
    thermal_conductivity_w_mk: float = Field(
        default=0.2,
        gt=0,
        description="Thermal conductivity at mean temperature"
    )
    expected_lifetime_hours: float = Field(
        default=35000.0,
        gt=0,
        description="Expected refractory lifetime in hours"
    )


class BurnerConfig(BaseModel):
    """
    Comprehensive burner specifications configuration.

    This configuration defines the physical characteristics, operational
    parameters, and expected performance of an industrial burner.

    Attributes:
        burner_id: Unique identifier for the burner
        burner_type: Type classification (gas, oil, low_nox, etc.)
        manufacturer: Burner manufacturer
        model: Manufacturer's model designation
        serial_number: Burner serial number for traceability
        capacity_mmbtu_hr: Maximum heat release capacity
        turndown_ratio: Minimum to maximum firing ratio
        fuel_type: Primary fuel type
        secondary_fuel_type: Secondary fuel (for dual-fuel burners)

    Example:
        >>> config = BurnerConfig(
        ...     burner_id="BNR-001",
        ...     burner_type=BurnerType.LOW_NOX,
        ...     manufacturer=BurnerManufacturer.ZEECO,
        ...     model="GLSF-500",
        ...     capacity_mmbtu_hr=50.0,
        ... )
    """

    # Identification
    burner_id: str = Field(
        ...,
        description="Unique burner identifier"
    )
    burner_tag: str = Field(
        default="",
        description="Plant equipment tag (e.g., B-101-A)"
    )
    burner_type: BurnerType = Field(
        default=BurnerType.GAS,
        description="Burner type classification"
    )
    manufacturer: BurnerManufacturer = Field(
        default=BurnerManufacturer.OTHER,
        description="Burner manufacturer"
    )
    model: str = Field(
        default="",
        description="Manufacturer model designation"
    )
    serial_number: str = Field(
        default="",
        description="Burner serial number"
    )

    # Capacity and performance
    capacity_mmbtu_hr: float = Field(
        default=50.0,
        gt=0,
        le=2000,
        description="Maximum heat release capacity (MMBtu/hr)"
    )
    capacity_mw: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum heat release capacity (MW thermal)"
    )
    turndown_ratio: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Turndown ratio (max:min firing rate)"
    )
    minimum_firing_rate_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Minimum stable firing rate (%)"
    )
    design_efficiency_pct: float = Field(
        default=85.0,
        ge=50.0,
        le=100.0,
        description="Design combustion efficiency (%)"
    )
    excess_air_design_pct: float = Field(
        default=15.0,
        ge=0.0,
        le=100.0,
        description="Design excess air (%)"
    )

    # Fuel specifications
    fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )
    secondary_fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Secondary fuel type (dual-fuel)"
    )
    fuel_pressure_psig_min: float = Field(
        default=3.0,
        ge=0,
        description="Minimum fuel gas pressure (psig)"
    )
    fuel_pressure_psig_max: float = Field(
        default=15.0,
        gt=0,
        description="Maximum fuel gas pressure (psig)"
    )

    # Combustion air
    air_supply_type: str = Field(
        default="forced_draft",
        description="Air supply type (forced_draft, induced_draft, natural)"
    )
    air_preheat_enabled: bool = Field(
        default=False,
        description="Combustion air preheat enabled"
    )
    air_preheat_temp_c: float = Field(
        default=25.0,
        description="Preheated air temperature (C)"
    )
    combustion_air_fan_hp: Optional[float] = Field(
        default=None,
        gt=0,
        description="Combustion air fan horsepower"
    )

    # Emissions specifications
    nox_design_ppm: float = Field(
        default=30.0,
        ge=0,
        description="Design NOx emissions (ppm @ 3% O2)"
    )
    co_design_ppm: float = Field(
        default=50.0,
        ge=0,
        description="Design CO emissions (ppm @ 3% O2)"
    )

    # Physical specifications
    nozzles: List[BurnerNozzleConfig] = Field(
        default_factory=list,
        description="Burner nozzle configurations"
    )
    refractory: Optional[BurnerRefractoryConfig] = Field(
        default=None,
        description="Burner refractory configuration"
    )

    # Lifecycle
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Burner installation date"
    )
    commissioning_date: Optional[datetime] = Field(
        default=None,
        description="Burner commissioning date"
    )
    last_overhaul_date: Optional[datetime] = Field(
        default=None,
        description="Last major overhaul date"
    )
    expected_lifetime_hours: float = Field(
        default=50000.0,
        gt=0,
        description="Expected burner lifetime (operating hours)"
    )
    warranty_expiry_date: Optional[datetime] = Field(
        default=None,
        description="Warranty expiration date"
    )

    # Current operating status
    current_operating_hours: float = Field(
        default=0.0,
        ge=0,
        description="Current total operating hours"
    )
    current_start_stop_cycles: int = Field(
        default=0,
        ge=0,
        description="Current start/stop cycle count"
    )

    @validator("capacity_mw", always=True)
    def calculate_mw_from_mmbtu(cls, v, values):
        """Auto-calculate MW if not provided (1 MMBtu/hr = 0.293 MW)."""
        if v is None and "capacity_mmbtu_hr" in values:
            return values["capacity_mmbtu_hr"] * 0.293071
        return v

    @validator("burner_type", pre=True)
    def convert_burner_type(cls, v):
        """Convert string to BurnerType enum."""
        if isinstance(v, str):
            return BurnerType(v)
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# FLAME ANALYSIS CONFIGURATION
# =============================================================================

class FlameAnalysisConfig(BaseModel):
    """
    Configuration for flame monitoring and analysis.

    Defines thresholds for flame characteristic monitoring including
    temperature, stability, shape, and color index parameters.

    Based on combustion engineering best practices and NFPA 85 requirements.

    Attributes:
        enabled: Enable flame analysis module
        temperature_variance_threshold_c: Max acceptable temp variance
        stability_index_warning: Stability index warning threshold
        stability_index_alarm: Stability index alarm threshold

    Example:
        >>> config = FlameAnalysisConfig(
        ...     stability_index_warning=0.85,
        ...     stability_index_alarm=0.75,
        ... )
    """

    enabled: bool = Field(
        default=True,
        description="Enable flame analysis"
    )

    # Temperature thresholds
    flame_temp_target_c: float = Field(
        default=1650.0,
        gt=500,
        lt=2500,
        description="Target flame temperature (C)"
    )
    flame_temp_min_c: float = Field(
        default=1200.0,
        gt=0,
        description="Minimum acceptable flame temperature (C)"
    )
    flame_temp_max_c: float = Field(
        default=1900.0,
        gt=0,
        description="Maximum acceptable flame temperature (C)"
    )
    temperature_variance_threshold_c: float = Field(
        default=50.0,
        gt=0,
        description="Max acceptable temperature variance (C)"
    )

    # Flame stability thresholds (0-1 scale, 1 = perfectly stable)
    stability_index_good: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Stability index for good condition"
    )
    stability_index_warning: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Stability index warning threshold"
    )
    stability_index_alarm: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Stability index alarm threshold"
    )
    stability_index_shutdown: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Stability index for safety shutdown"
    )

    # Flame shape thresholds (0-1 scale, 1 = ideal shape)
    shape_score_good: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Shape score for good condition"
    )
    shape_score_warning: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Shape score warning threshold"
    )
    shape_score_alarm: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Shape score alarm threshold"
    )

    # Color index thresholds (normalized 0-1, based on blackbody radiation)
    color_index_good: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Color index for good combustion"
    )
    color_index_warning: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Color index warning (incomplete combustion)"
    )
    color_index_alarm: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Color index alarm threshold"
    )

    # Flame detection
    flame_detector_type: FlameDetectorType = Field(
        default=FlameDetectorType.UV_IR,
        description="Flame detector sensor type"
    )
    flame_signal_min_pct: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Minimum flame signal strength (%)"
    )
    flame_failure_response_time_s: float = Field(
        default=3.0,
        gt=0,
        le=10,
        description="Flame failure response time per NFPA 85 (seconds)"
    )

    # Analysis frequency
    analysis_interval_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Flame analysis interval (seconds)"
    )
    trend_window_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Trend analysis window (hours)"
    )

    @validator("stability_index_warning")
    def validate_stability_warning(cls, v, values):
        """Ensure warning threshold is below good threshold."""
        if "stability_index_good" in values and v >= values["stability_index_good"]:
            raise ValueError("Warning threshold must be below good threshold")
        return v

    @validator("stability_index_alarm")
    def validate_stability_alarm(cls, v, values):
        """Ensure alarm threshold is below warning threshold."""
        if "stability_index_warning" in values and v >= values["stability_index_warning"]:
            raise ValueError("Alarm threshold must be below warning threshold")
        return v


# =============================================================================
# MAINTENANCE PREDICTION CONFIGURATION
# =============================================================================

class WeibullParameterConfig(BaseModel):
    """Weibull distribution parameters for component RUL prediction."""

    component: str = Field(
        ...,
        description="Component name (nozzle, refractory, ignitor, etc.)"
    )
    beta: float = Field(
        default=2.0,
        gt=0,
        le=10,
        description="Shape parameter (beta > 1 = wear-out failure)"
    )
    eta_hours: float = Field(
        default=25000.0,
        gt=0,
        description="Scale parameter (characteristic life in hours)"
    )
    gamma_hours: float = Field(
        default=0.0,
        ge=0,
        description="Location parameter (failure-free life)"
    )
    beta_uncertainty: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Beta parameter uncertainty (std dev)"
    )
    eta_uncertainty_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Eta parameter uncertainty (%)"
    )
    data_source: str = Field(
        default="manufacturer",
        description="Source of parameters (manufacturer, field_data, industry)"
    )


class MLModelConfig(BaseModel):
    """Machine learning model configuration for failure prediction."""

    enabled: bool = Field(
        default=True,
        description="Enable ML-based predictions"
    )
    model_type: str = Field(
        default="gradient_boosting",
        description="Model type (gradient_boosting, random_forest, lstm)"
    )
    ensemble_size: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of models in ensemble"
    )
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.50,
        le=0.99,
        description="Minimum confidence for predictions"
    )
    feature_importance_enabled: bool = Field(
        default=True,
        description="Calculate SHAP feature importance"
    )
    uncertainty_quantification: bool = Field(
        default=True,
        description="Enable uncertainty bounds"
    )
    online_learning_enabled: bool = Field(
        default=False,
        description="Enable incremental learning"
    )
    retrain_interval_days: int = Field(
        default=30,
        ge=7,
        le=365,
        description="Model retraining interval"
    )
    drift_detection_enabled: bool = Field(
        default=True,
        description="Enable concept drift detection"
    )


class MaintenancePredictionConfig(BaseModel):
    """
    Configuration for maintenance prediction and RUL estimation.

    Includes Weibull analysis parameters, ML model settings, and
    prediction thresholds for burner maintenance scheduling.

    Attributes:
        weibull_beta: Default Weibull shape parameter
        weibull_eta_hours: Default Weibull scale parameter
        confidence_level: Confidence level for intervals
        ml_model_enabled: Enable machine learning predictions

    Example:
        >>> config = MaintenancePredictionConfig(
        ...     weibull_beta=2.5,
        ...     weibull_eta_hours=50000,
        ...     confidence_level=0.90,
        ... )
    """

    # Weibull parameters (overall burner)
    weibull_beta: float = Field(
        default=2.0,
        gt=0,
        le=10,
        description="Default Weibull shape parameter (beta)"
    )
    weibull_eta_hours: float = Field(
        default=50000.0,
        gt=0,
        description="Default Weibull scale parameter (eta, hours)"
    )
    weibull_gamma_hours: float = Field(
        default=0.0,
        ge=0,
        description="Default Weibull location parameter (gamma)"
    )
    confidence_level: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Confidence level for RUL intervals"
    )

    # Component-specific Weibull parameters
    component_weibull_params: List[WeibullParameterConfig] = Field(
        default_factory=lambda: [
            WeibullParameterConfig(
                component="nozzle",
                beta=1.8,
                eta_hours=15000,
            ),
            WeibullParameterConfig(
                component="refractory",
                beta=2.5,
                eta_hours=35000,
            ),
            WeibullParameterConfig(
                component="ignitor",
                beta=1.5,
                eta_hours=20000,
            ),
            WeibullParameterConfig(
                component="flame_scanner",
                beta=2.0,
                eta_hours=25000,
            ),
            WeibullParameterConfig(
                component="fuel_valve",
                beta=2.2,
                eta_hours=40000,
            ),
        ],
        description="Component-specific Weibull parameters"
    )

    # ML model configuration
    ml_model_enabled: bool = Field(
        default=True,
        description="Enable ML-based predictions"
    )
    ml_config: MLModelConfig = Field(
        default_factory=MLModelConfig,
        description="ML model configuration"
    )

    # Cycle-based degradation
    cycle_degradation_enabled: bool = Field(
        default=True,
        description="Enable start/stop cycle degradation analysis"
    )
    cycles_per_equivalent_hour: float = Field(
        default=10.0,
        gt=0,
        description="Start/stop cycles equivalent to 1 operating hour"
    )
    max_daily_cycles_warning: int = Field(
        default=10,
        ge=1,
        description="Max daily cycles before warning"
    )

    # Fuel quality impact factors
    fuel_quality_impact_enabled: bool = Field(
        default=True,
        description="Include fuel quality in degradation model"
    )
    sulfur_impact_factor: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Degradation factor per ppm H2S"
    )
    moisture_impact_factor: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Degradation factor per % moisture"
    )

    # Prediction thresholds
    rul_warning_hours: float = Field(
        default=2000.0,
        gt=0,
        description="RUL threshold for warning (hours)"
    )
    rul_alarm_hours: float = Field(
        default=500.0,
        gt=0,
        description="RUL threshold for alarm (hours)"
    )
    failure_probability_warning: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Failure probability warning threshold"
    )
    failure_probability_alarm: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Failure probability alarm threshold"
    )

    # Historical data requirements
    minimum_history_days: int = Field(
        default=90,
        ge=7,
        description="Minimum historical data for predictions"
    )
    use_manufacturer_baseline: bool = Field(
        default=True,
        description="Use manufacturer data when field data insufficient"
    )


# =============================================================================
# FUEL QUALITY CONFIGURATION
# =============================================================================

class FuelQualityConfig(BaseModel):
    """
    Configuration for fuel quality monitoring and impact assessment.

    Defines acceptable ranges for fuel properties and their impact
    on burner degradation based on API and ASTM standards.

    Attributes:
        primary_fuel_type: Primary fuel being monitored
        hhv_nominal: Nominal higher heating value
        hhv_variance_tolerance_pct: Acceptable HHV variance
        max_h2s_ppm: Maximum H2S content

    Standards:
        - ASTM D1945: Natural gas composition
        - ASTM D240: Heat of combustion
        - ASTM D4809: Heat of combustion (calorimeter)
    """

    primary_fuel_type: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )

    # Natural gas specifications (ASTM D1945)
    hhv_nominal_mj_m3: float = Field(
        default=38.0,
        gt=0,
        description="Nominal HHV for natural gas (MJ/m3)"
    )
    hhv_min_mj_m3: float = Field(
        default=35.0,
        gt=0,
        description="Minimum acceptable HHV (MJ/m3)"
    )
    hhv_max_mj_m3: float = Field(
        default=42.0,
        gt=0,
        description="Maximum acceptable HHV (MJ/m3)"
    )
    hhv_variance_tolerance_pct: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Acceptable HHV variance from nominal (%)"
    )

    # Gas composition limits
    methane_min_pct: float = Field(
        default=85.0,
        ge=0,
        le=100,
        description="Minimum methane content (%)"
    )
    max_h2s_ppm: float = Field(
        default=4.0,
        ge=0,
        description="Maximum H2S content (ppm)"
    )
    max_total_sulfur_ppm: float = Field(
        default=20.0,
        ge=0,
        description="Maximum total sulfur content (ppm)"
    )
    max_moisture_pct: float = Field(
        default=1.0,
        ge=0,
        le=10,
        description="Maximum moisture content (%)"
    )
    max_co2_pct: float = Field(
        default=3.0,
        ge=0,
        le=20,
        description="Maximum CO2 content (%)"
    )
    max_n2_pct: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Maximum nitrogen content (%)"
    )
    max_oxygen_pct: float = Field(
        default=0.5,
        ge=0,
        le=5,
        description="Maximum oxygen content (%)"
    )

    # Liquid fuel specifications (ASTM D396)
    viscosity_nominal_cst: float = Field(
        default=5.0,
        gt=0,
        description="Nominal fuel oil viscosity at 40C (cSt)"
    )
    viscosity_min_cst: float = Field(
        default=2.0,
        gt=0,
        description="Minimum acceptable viscosity (cSt)"
    )
    viscosity_max_cst: float = Field(
        default=15.0,
        gt=0,
        description="Maximum acceptable viscosity (cSt)"
    )
    max_ash_content_pct: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="Maximum ash content (%)"
    )
    max_vanadium_ppm: float = Field(
        default=50.0,
        ge=0,
        description="Maximum vanadium content (ppm)"
    )
    max_sodium_ppm: float = Field(
        default=50.0,
        ge=0,
        description="Maximum sodium content (ppm)"
    )

    # Degradation impact factors
    h2s_nozzle_degradation_factor: float = Field(
        default=0.02,
        ge=0,
        le=0.5,
        description="Nozzle life reduction per ppm H2S above limit"
    )
    moisture_efficiency_impact_pct: float = Field(
        default=0.5,
        ge=0,
        le=5,
        description="Efficiency loss per % moisture (%)"
    )
    ash_refractory_degradation_factor: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Refractory life reduction per % ash"
    )
    wobbe_index_tolerance_pct: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Acceptable Wobbe Index variance (%)"
    )

    # Monitoring intervals
    quality_check_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Fuel quality check interval (hours)"
    )
    online_monitoring_enabled: bool = Field(
        default=False,
        description="Enable online fuel quality monitoring"
    )


# =============================================================================
# REPLACEMENT PLANNING CONFIGURATION
# =============================================================================

class ReplacementPlanningConfig(BaseModel):
    """
    Configuration for burner replacement planning and economic analysis.

    Includes cost parameters, lead times, and economic optimization
    settings for optimal replacement timing decisions.

    Attributes:
        economic_optimization_enabled: Enable NPV/TCO optimization
        planning_horizon_months: Planning time horizon
        burner_replacement_cost_usd: Full replacement cost
        spare_parts_lead_time_days: Parts procurement lead time

    Economic Model:
        Total Cost = Replacement Cost + Downtime Cost + Efficiency Loss
                   - Salvage Value
    """

    economic_optimization_enabled: bool = Field(
        default=True,
        description="Enable economic optimization"
    )
    planning_horizon_months: int = Field(
        default=24,
        ge=6,
        le=120,
        description="Planning horizon (months)"
    )

    # Cost parameters
    burner_replacement_cost_usd: float = Field(
        default=50000.0,
        ge=0,
        description="Full burner replacement cost ($)"
    )
    nozzle_replacement_cost_usd: float = Field(
        default=500.0,
        ge=0,
        description="Nozzle replacement cost ($)"
    )
    refractory_replacement_cost_usd: float = Field(
        default=15000.0,
        ge=0,
        description="Refractory replacement cost ($)"
    )
    ignitor_replacement_cost_usd: float = Field(
        default=2000.0,
        ge=0,
        description="Ignitor replacement cost ($)"
    )
    flame_scanner_replacement_cost_usd: float = Field(
        default=3000.0,
        ge=0,
        description="Flame scanner replacement cost ($)"
    )
    labor_rate_usd_hr: float = Field(
        default=150.0,
        ge=0,
        description="Labor rate for maintenance ($/hr)"
    )

    # Downtime costs
    downtime_cost_usd_hr: float = Field(
        default=5000.0,
        ge=0,
        description="Production downtime cost ($/hr)"
    )
    planned_outage_duration_hours: float = Field(
        default=8.0,
        ge=0,
        description="Typical planned maintenance duration (hours)"
    )
    unplanned_outage_duration_hours: float = Field(
        default=48.0,
        ge=0,
        description="Typical unplanned failure duration (hours)"
    )
    unplanned_failure_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        description="Cost multiplier for unplanned failures"
    )

    # Lead times
    spare_parts_lead_time_days: int = Field(
        default=14,
        ge=0,
        description="Standard spare parts lead time (days)"
    )
    burner_replacement_lead_time_days: int = Field(
        default=90,
        ge=0,
        description="Full burner replacement lead time (days)"
    )
    critical_spare_inventory_days: int = Field(
        default=30,
        ge=0,
        description="Critical spares inventory coverage (days)"
    )

    # Economic parameters
    discount_rate_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Discount rate for NPV calculations (%)"
    )
    fuel_cost_usd_mmbtu: float = Field(
        default=5.0,
        ge=0,
        description="Fuel cost ($/MMBtu)"
    )
    efficiency_degradation_cost_enabled: bool = Field(
        default=True,
        description="Include efficiency loss in cost model"
    )
    carbon_cost_enabled: bool = Field(
        default=False,
        description="Include carbon pricing in cost model"
    )
    carbon_price_usd_ton: float = Field(
        default=50.0,
        ge=0,
        description="Carbon price ($/ton CO2)"
    )

    # Salvage values
    burner_salvage_value_pct: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Salvage value as % of replacement cost"
    )

    # Optimization settings
    optimization_objective: str = Field(
        default="minimize_tco",
        description="Optimization objective (minimize_tco, maximize_availability)"
    )
    risk_tolerance: str = Field(
        default="medium",
        description="Risk tolerance (low, medium, high)"
    )
    min_availability_target_pct: float = Field(
        default=98.0,
        ge=90,
        le=100,
        description="Minimum availability target (%)"
    )


# =============================================================================
# CMMS INTEGRATION CONFIGURATION
# =============================================================================

class CMSIntegrationConfig(BaseModel):
    """
    Configuration for CMMS (Computerized Maintenance Management System)
    integration for work order generation.

    Supports SAP PM, IBM Maximo, Oracle EAM, and custom API integrations.

    Attributes:
        enabled: Enable CMMS integration
        system_type: Type of CMMS system
        api_endpoint: CMMS API endpoint URL
        auto_create_work_orders: Automatically create work orders

    Example:
        >>> config = CMSIntegrationConfig(
        ...     enabled=True,
        ...     system_type=CMMSType.SAP_PM,
        ...     plant_code="1000",
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable CMMS integration"
    )
    system_type: CMMSType = Field(
        default=CMMSType.SAP_PM,
        description="CMMS system type"
    )
    api_endpoint: Optional[str] = Field(
        default=None,
        description="CMMS API endpoint URL"
    )
    api_key_secret_name: Optional[str] = Field(
        default=None,
        description="Secret name for API key (use secrets manager)"
    )
    plant_code: str = Field(
        default="",
        description="Plant/facility code in CMMS"
    )
    work_center: str = Field(
        default="",
        description="Work center code for maintenance"
    )
    cost_center: str = Field(
        default="",
        description="Cost center for work orders"
    )

    # Work order settings
    auto_create_work_orders: bool = Field(
        default=False,
        description="Automatically create work orders"
    )
    require_approval: bool = Field(
        default=True,
        description="Require approval for auto-created WOs"
    )
    default_work_order_type: str = Field(
        default="PM01",
        description="Default work order type code"
    )
    priority_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "critical": "1",
            "high": "2",
            "medium": "3",
            "low": "4",
            "routine": "5",
        },
        description="Map alert severity to WO priority codes"
    )

    # Notification settings
    notification_enabled: bool = Field(
        default=True,
        description="Enable email notifications"
    )
    notification_recipients: List[str] = Field(
        default_factory=list,
        description="Notification email recipients"
    )
    notification_on_critical: bool = Field(
        default=True,
        description="Notify on critical alerts"
    )
    notification_on_work_order: bool = Field(
        default=True,
        description="Notify when work orders created"
    )

    # SAP-specific settings
    sap_client: str = Field(
        default="",
        description="SAP client number"
    )
    sap_maintenance_plan_type: str = Field(
        default="PM01",
        description="SAP maintenance plan type"
    )

    # Maximo-specific settings
    maximo_site_id: str = Field(
        default="",
        description="Maximo site ID"
    )
    maximo_org_id: str = Field(
        default="",
        description="Maximo organization ID"
    )

    # Data synchronization
    sync_interval_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Data sync interval (minutes)"
    )
    sync_maintenance_history: bool = Field(
        default=True,
        description="Sync historical maintenance data"
    )
    sync_spare_parts_inventory: bool = Field(
        default=True,
        description="Sync spare parts inventory"
    )


# =============================================================================
# SAFETY CONFIGURATION
# =============================================================================

class SafetyConfig(BaseModel):
    """
    Configuration for safety compliance and BMS coordination.

    Ensures compliance with NFPA 85, API 556, and ISA 84 standards
    for burner management and flame safeguard systems.

    Attributes:
        nfpa_85_enabled: Enable NFPA 85 compliance checks
        sil_rating: Safety Integrity Level for SIS
        bms_coordination_enabled: Enable BMS coordination
        flame_safeguard_type: Type of flame detection system

    Standards:
        - NFPA 85: Boiler and Combustion Systems Hazards Code
        - API 556: Instrumentation, Control, and Protective Systems
        - ISA 84: Safety Instrumented Systems
    """

    # Standards compliance
    nfpa_85_enabled: bool = Field(
        default=True,
        description="Enable NFPA 85 compliance"
    )
    nfpa_86_enabled: bool = Field(
        default=False,
        description="Enable NFPA 86 compliance (ovens/furnaces)"
    )
    api_556_enabled: bool = Field(
        default=True,
        description="Enable API 556 compliance (fired heaters)"
    )
    en_298_enabled: bool = Field(
        default=False,
        description="Enable EN 298 compliance (European)"
    )

    # SIS/SIL configuration
    sil_rating: SafetyIntegrityLevel = Field(
        default=SafetyIntegrityLevel.SIL_2,
        description="Safety Integrity Level rating"
    )
    sis_proof_test_interval_months: int = Field(
        default=12,
        ge=1,
        le=60,
        description="SIS proof test interval (months)"
    )
    pfd_target: float = Field(
        default=0.001,
        gt=0,
        lt=1,
        description="Probability of Failure on Demand target"
    )

    # BMS coordination
    bms_coordination_enabled: bool = Field(
        default=True,
        description="Enable BMS coordination"
    )
    bms_standard: BMSStandard = Field(
        default=BMSStandard.NFPA_85,
        description="BMS compliance standard"
    )
    bms_interface_protocol: str = Field(
        default="modbus_tcp",
        description="BMS communication protocol"
    )

    # Flame safeguard
    flame_safeguard_type: str = Field(
        default="UV_IR",
        description="Flame detection type (UV, IR, UV_IR, FLAME_ROD)"
    )
    flame_failure_response_time_s: float = Field(
        default=3.0,
        gt=0,
        le=10,
        description="Max flame failure response time (seconds)"
    )
    pilot_proving_time_s: float = Field(
        default=10.0,
        gt=0,
        le=30,
        description="Pilot proving time (seconds)"
    )
    main_flame_proving_time_s: float = Field(
        default=10.0,
        gt=0,
        le=30,
        description="Main flame proving time (seconds)"
    )

    # Purge requirements per NFPA 85
    prepurge_airflow_changes: int = Field(
        default=4,
        ge=4,
        le=8,
        description="Pre-purge air changes (min 4 per NFPA 85)"
    )
    postpurge_airflow_changes: int = Field(
        default=2,
        ge=0,
        le=4,
        description="Post-purge air changes"
    )
    purge_airflow_min_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Minimum purge airflow (%)"
    )

    # Interlock settings
    low_fuel_pressure_trip_psig: float = Field(
        default=2.0,
        ge=0,
        description="Low fuel pressure trip setpoint (psig)"
    )
    high_fuel_pressure_trip_psig: float = Field(
        default=20.0,
        gt=0,
        description="High fuel pressure trip setpoint (psig)"
    )
    high_firebox_pressure_trip_inwc: float = Field(
        default=0.5,
        gt=0,
        description="High firebox pressure trip (in. W.C.)"
    )
    low_combustion_air_trip: bool = Field(
        default=True,
        description="Enable low combustion air trip"
    )
    high_stack_temp_trip_c: float = Field(
        default=450.0,
        gt=0,
        description="High stack temperature trip (C)"
    )

    # Audit trail
    safety_event_logging: bool = Field(
        default=True,
        description="Enable safety event logging"
    )
    safety_event_retention_days: int = Field(
        default=365,
        ge=30,
        description="Safety event log retention (days)"
    )


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

class GL021Config(BaseModel):
    """
    Master configuration for GL-021 BURNERSENTRY agent.

    This configuration combines all component configurations for
    comprehensive burner maintenance prediction and planning.

    Attributes:
        burner: Burner specifications
        flame_analysis: Flame monitoring configuration
        maintenance_prediction: Prediction model settings
        fuel_quality: Fuel quality limits
        replacement_planning: Economic planning parameters
        cmms: CMMS integration settings
        safety: Safety compliance configuration

    Example:
        >>> config = GL021Config(
        ...     burner=BurnerConfig(
        ...         burner_id="BNR-001",
        ...         burner_type=BurnerType.LOW_NOX,
        ...         capacity_mmbtu_hr=50.0,
        ...     ),
        ...     cmms=CMSIntegrationConfig(
        ...         enabled=True,
        ...         system_type=CMMSType.SAP_PM,
        ...     ),
        ... )
    """

    # Component configurations
    burner: BurnerConfig = Field(
        ...,
        description="Burner specifications"
    )
    flame_analysis: FlameAnalysisConfig = Field(
        default_factory=FlameAnalysisConfig,
        description="Flame analysis configuration"
    )
    maintenance_prediction: MaintenancePredictionConfig = Field(
        default_factory=MaintenancePredictionConfig,
        description="Maintenance prediction configuration"
    )
    fuel_quality: FuelQualityConfig = Field(
        default_factory=FuelQualityConfig,
        description="Fuel quality configuration"
    )
    replacement_planning: ReplacementPlanningConfig = Field(
        default_factory=ReplacementPlanningConfig,
        description="Replacement planning configuration"
    )
    cmms: CMSIntegrationConfig = Field(
        default_factory=CMSIntegrationConfig,
        description="CMMS integration configuration"
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety compliance configuration"
    )

    # Data management
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Operational data retention (days)"
    )
    trend_history_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Trend analysis history (days)"
    )
    archive_enabled: bool = Field(
        default=True,
        description="Enable data archiving"
    )

    # Audit and provenance
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance hashes"
    )

    # Agent settings
    agent_id: str = Field(
        default="GL-021",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="BURNERSENTRY",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )

    @root_validator(skip_on_failure=True)
    def validate_config_consistency(cls, values):
        """Validate configuration consistency across components."""
        burner = values.get("burner")
        fuel_quality = values.get("fuel_quality")

        # Ensure fuel type consistency
        if burner and fuel_quality:
            if burner.fuel_type != fuel_quality.primary_fuel_type:
                # Auto-sync fuel types
                fuel_quality.primary_fuel_type = burner.fuel_type

        return values

    class Config:
        use_enum_values = True
        validate_assignment = True
