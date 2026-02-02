"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - Configuration Module

This module provides configuration schemas for the unified steam system optimizer.
Consolidates GL-003 (STEAMWISE) and GL-012 (STEAMQUAL) configurations.

Configuration Categories:
    - Steam header configurations
    - Quality monitoring thresholds (ASME standards)
    - PRV sizing parameters (ASME B31.1)
    - Condensate return optimization
    - Flash steam recovery settings
    - Desuperheating control parameters

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.config import (
    ...     UnifiedSteamConfig,
    ...     SteamHeaderConfig,
    ...     QualityMonitoringConfig,
    ... )
    >>>
    >>> config = UnifiedSteamConfig(
    ...     headers=[SteamHeaderConfig(name="HP", pressure_psig=600)],
    ...     quality=QualityMonitoringConfig(min_dryness_fraction=0.95),
    ... )
"""

from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


class SteamHeaderLevel(Enum):
    """Steam header pressure levels."""
    HIGH_PRESSURE = "HP"      # 250-600 psig
    MEDIUM_PRESSURE = "MP"    # 50-250 psig
    LOW_PRESSURE = "LP"       # 15-50 psig
    VERY_LOW_PRESSURE = "VLP" # 0-15 psig


class PRVSizingMethod(Enum):
    """PRV sizing calculation methods."""
    ASME_B31_1 = "asme_b31_1"
    API_520 = "api_520"
    MANUFACTURER = "manufacturer"


class DesuperheaterType(Enum):
    """Desuperheater types."""
    WATER_SPRAY = "water_spray"
    STEAM_ATOMIZING = "steam_atomizing"
    SURFACE_CONTACT = "surface_contact"
    VENTURI = "venturi"


class CondensateFlashMethod(Enum):
    """Condensate flash recovery methods."""
    FLASH_TANK = "flash_tank"
    FLASH_VESSEL = "flash_vessel"
    THERMODYNAMIC = "thermodynamic"


class SteamQualityStandard(Enum):
    """Steam quality standards."""
    ASME = "asme"
    ABMA = "abma"
    ISO_9806 = "iso_9806"


class SteamHeaderConfig(BaseModel):
    """Configuration for a single steam header."""

    name: str = Field(
        ...,
        description="Header identifier (e.g., 'HP-001', 'MP-MAIN')"
    )
    level: SteamHeaderLevel = Field(
        default=SteamHeaderLevel.MEDIUM_PRESSURE,
        description="Pressure level classification"
    )
    design_pressure_psig: float = Field(
        ...,
        ge=0,
        le=1500,
        description="Design operating pressure (psig)"
    )
    min_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Minimum operating pressure (psig)"
    )
    max_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Maximum operating pressure (psig)"
    )
    design_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Design flow rate (lb/hr)"
    )
    max_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Maximum flow capacity (lb/hr)"
    )
    design_temperature_f: Optional[float] = Field(
        default=None,
        description="Design temperature for superheated steam (F)"
    )
    connected_consumers: List[str] = Field(
        default_factory=list,
        description="List of connected consumer IDs"
    )
    connected_suppliers: List[str] = Field(
        default_factory=list,
        description="List of connected supplier IDs (boilers, PRVs)"
    )
    exergy_reference_temp_f: float = Field(
        default=77.0,
        description="Reference temperature for exergy calculations (F)"
    )

    @validator('max_pressure_psig')
    def max_pressure_greater_than_min(cls, v, values):
        """Ensure max pressure is greater than min pressure."""
        if 'min_pressure_psig' in values and v < values['min_pressure_psig']:
            raise ValueError(
                "max_pressure_psig must be >= min_pressure_psig"
            )
        return v

    @validator('design_pressure_psig')
    def design_pressure_in_range(cls, v, values):
        """Ensure design pressure is within operating range."""
        if 'min_pressure_psig' in values and v < values['min_pressure_psig']:
            raise ValueError(
                "design_pressure_psig must be >= min_pressure_psig"
            )
        if 'max_pressure_psig' in values and v > values['max_pressure_psig']:
            raise ValueError(
                "design_pressure_psig must be <= max_pressure_psig"
            )
        return v

    class Config:
        use_enum_values = True


class QualityMonitoringConfig(BaseModel):
    """Configuration for steam quality monitoring per ASME standards."""

    standard: SteamQualityStandard = Field(
        default=SteamQualityStandard.ASME,
        description="Quality monitoring standard"
    )

    # Dryness fraction limits
    min_dryness_fraction: float = Field(
        default=0.95,
        ge=0.80,
        le=1.0,
        description="Minimum acceptable dryness fraction (steam quality)"
    )
    target_dryness_fraction: float = Field(
        default=0.98,
        ge=0.90,
        le=1.0,
        description="Target dryness fraction"
    )

    # TDS limits (ppm) - per ABMA/ASME recommendations
    max_tds_ppm_lp: float = Field(
        default=3500.0,
        ge=0,
        description="Max TDS for LP steam (<300 psig)"
    )
    max_tds_ppm_mp: float = Field(
        default=3000.0,
        ge=0,
        description="Max TDS for MP steam (300-450 psig)"
    )
    max_tds_ppm_hp: float = Field(
        default=2500.0,
        ge=0,
        description="Max TDS for HP steam (>450 psig)"
    )

    # Cation conductivity limits (microS/cm)
    max_cation_conductivity_us_cm: float = Field(
        default=0.3,
        ge=0,
        le=10.0,
        description="Maximum cation conductivity (microS/cm)"
    )
    target_cation_conductivity_us_cm: float = Field(
        default=0.2,
        ge=0,
        le=5.0,
        description="Target cation conductivity (microS/cm)"
    )

    # Silica limits (ppm)
    max_silica_ppm: float = Field(
        default=0.02,
        ge=0,
        description="Maximum silica in steam (ppm)"
    )

    # Dissolved oxygen limits (ppb)
    max_dissolved_o2_ppb: float = Field(
        default=7.0,
        ge=0,
        description="Maximum dissolved O2 in feedwater (ppb)"
    )

    # Sampling frequency (seconds)
    sampling_interval_s: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Quality sampling interval (seconds)"
    )

    # Alert thresholds (percentage of limit)
    warning_threshold_pct: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description="Warning threshold as % of limit"
    )
    critical_threshold_pct: float = Field(
        default=95.0,
        ge=80.0,
        le=100.0,
        description="Critical threshold as % of limit"
    )

    class Config:
        use_enum_values = True


class PRVConfig(BaseModel):
    """Configuration for Pressure Reducing Valve per ASME B31.1."""

    prv_id: str = Field(
        ...,
        description="PRV identifier"
    )
    sizing_method: PRVSizingMethod = Field(
        default=PRVSizingMethod.ASME_B31_1,
        description="Sizing calculation method"
    )

    # Pressure settings
    inlet_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Inlet (upstream) pressure (psig)"
    )
    outlet_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Outlet (downstream) pressure (psig)"
    )
    min_pressure_drop_psi: float = Field(
        default=10.0,
        gt=0,
        description="Minimum required pressure drop (psi)"
    )

    # Flow capacity
    design_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Design flow rate (lb/hr)"
    )
    min_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Minimum controllable flow (lb/hr)"
    )
    max_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Maximum flow capacity (lb/hr)"
    )

    # Valve characteristics
    cv_rated: float = Field(
        ...,
        gt=0,
        description="Rated Cv value"
    )
    rangeability: float = Field(
        default=50.0,
        ge=10.0,
        le=100.0,
        description="Valve rangeability (turndown ratio)"
    )

    # Operating targets per ASME B31.1
    target_opening_min_pct: float = Field(
        default=50.0,
        ge=20.0,
        le=90.0,
        description="Target minimum opening percentage"
    )
    target_opening_max_pct: float = Field(
        default=70.0,
        ge=30.0,
        le=90.0,
        description="Target maximum opening percentage"
    )

    # Desuperheating
    desuperheater_enabled: bool = Field(
        default=False,
        description="Enable downstream desuperheating"
    )
    desuperheater_type: Optional[DesuperheaterType] = Field(
        default=None,
        description="Type of desuperheater"
    )
    target_superheat_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=200,
        description="Target outlet superheat (F)"
    )

    @validator('outlet_pressure_psig')
    def outlet_less_than_inlet(cls, v, values):
        """Ensure outlet pressure is less than inlet."""
        if 'inlet_pressure_psig' in values:
            if v >= values['inlet_pressure_psig']:
                raise ValueError(
                    "outlet_pressure_psig must be < inlet_pressure_psig"
                )
        return v

    @validator('target_opening_max_pct')
    def max_opening_greater_than_min(cls, v, values):
        """Ensure max opening is greater than min."""
        if 'target_opening_min_pct' in values:
            if v < values['target_opening_min_pct']:
                raise ValueError(
                    "target_opening_max_pct must be >= target_opening_min_pct"
                )
        return v

    class Config:
        use_enum_values = True


class DesuperheaterConfig(BaseModel):
    """Configuration for desuperheater control."""

    desuperheater_id: str = Field(
        ...,
        description="Desuperheater identifier"
    )
    type: DesuperheaterType = Field(
        default=DesuperheaterType.WATER_SPRAY,
        description="Desuperheater type"
    )

    # Temperature control
    target_outlet_temp_f: float = Field(
        ...,
        gt=0,
        description="Target outlet temperature (F)"
    )
    min_approach_temp_f: float = Field(
        default=20.0,
        ge=5.0,
        le=50.0,
        description="Minimum approach to saturation (F)"
    )

    # Spray water
    spray_water_temp_f: float = Field(
        default=200.0,
        gt=32,
        description="Spray water temperature (F)"
    )
    max_spray_rate_lb_hr: float = Field(
        ...,
        gt=0,
        description="Maximum spray water rate (lb/hr)"
    )
    spray_water_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Spray water supply pressure (psig)"
    )
    min_pressure_differential_psi: float = Field(
        default=50.0,
        ge=20.0,
        description="Minimum spray water pressure above steam (psi)"
    )

    # Control tuning
    control_deadband_f: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Control deadband (F)"
    )

    class Config:
        use_enum_values = True


class CondensateConfig(BaseModel):
    """Configuration for condensate return optimization."""

    # Return rate targets
    target_return_rate_pct: float = Field(
        default=85.0,
        ge=0,
        le=100,
        description="Target condensate return rate (%)"
    )
    min_acceptable_return_pct: float = Field(
        default=70.0,
        ge=0,
        le=100,
        description="Minimum acceptable return rate (%)"
    )

    # Temperature targets
    target_return_temp_f: float = Field(
        default=180.0,
        ge=100,
        le=300,
        description="Target condensate return temperature (F)"
    )
    min_return_temp_f: float = Field(
        default=140.0,
        ge=100,
        le=250,
        description="Minimum acceptable return temperature (F)"
    )

    # Quality limits
    max_contamination_tds_ppm: float = Field(
        default=50.0,
        ge=0,
        description="Maximum TDS for clean condensate (ppm)"
    )
    max_oil_ppm: float = Field(
        default=1.0,
        ge=0,
        description="Maximum oil content (ppm)"
    )
    max_iron_ppb: float = Field(
        default=100.0,
        ge=0,
        description="Maximum iron content (ppb)"
    )

    # Steam trap survey integration
    trap_survey_enabled: bool = Field(
        default=True,
        description="Enable steam trap survey integration"
    )
    trap_failure_rate_threshold_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Threshold for trap failure rate alert (%)"
    )

    # Flash steam recovery
    flash_recovery_enabled: bool = Field(
        default=True,
        description="Enable flash steam recovery"
    )
    flash_recovery_method: CondensateFlashMethod = Field(
        default=CondensateFlashMethod.FLASH_TANK,
        description="Flash steam recovery method"
    )

    class Config:
        use_enum_values = True


class FlashRecoveryConfig(BaseModel):
    """Configuration for flash steam recovery calculations."""

    flash_tank_id: str = Field(
        default="FT-001",
        description="Flash tank identifier"
    )

    # Operating conditions
    condensate_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Incoming condensate pressure (psig)"
    )
    flash_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Flash tank operating pressure (psig)"
    )
    condensate_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Condensate flow rate (lb/hr)"
    )

    # Efficiency targets
    min_recovery_efficiency_pct: float = Field(
        default=90.0,
        ge=50,
        le=100,
        description="Minimum flash recovery efficiency (%)"
    )

    # Destination header
    flash_steam_destination: Optional[str] = Field(
        default=None,
        description="Target header for flash steam"
    )

    # Economic factors
    fuel_cost_per_mmbtu: float = Field(
        default=5.0,
        ge=0,
        description="Fuel cost for savings calculations ($/MMBTU)"
    )
    operating_hours_per_year: int = Field(
        default=8000,
        ge=1000,
        le=8760,
        description="Operating hours per year"
    )

    @validator('flash_pressure_psig')
    def flash_less_than_condensate(cls, v, values):
        """Ensure flash pressure is less than condensate pressure."""
        if 'condensate_pressure_psig' in values:
            if v >= values['condensate_pressure_psig']:
                raise ValueError(
                    "flash_pressure_psig must be < condensate_pressure_psig"
                )
        return v

    class Config:
        use_enum_values = True


class SteamTrapSurveyConfig(BaseModel):
    """Configuration for steam trap survey integration."""

    survey_enabled: bool = Field(
        default=True,
        description="Enable trap survey integration"
    )
    survey_frequency_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Survey frequency (days)"
    )

    # Failure thresholds
    failed_open_threshold_pct: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Alert threshold for failed-open traps (%)"
    )
    failed_closed_threshold_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Alert threshold for failed-closed traps (%)"
    )

    # Economic factors
    steam_cost_per_mlb: float = Field(
        default=10.0,
        ge=0,
        description="Steam cost for loss calculations ($/Mlb)"
    )

    # Trap types for analysis
    tracked_trap_types: List[str] = Field(
        default_factory=lambda: [
            "thermodynamic",
            "thermostatic",
            "float_thermostatic",
            "inverted_bucket",
        ],
        description="Trap types to track"
    )


class ExergyOptimizationConfig(BaseModel):
    """Configuration for exergy-based optimization."""

    enabled: bool = Field(
        default=True,
        description="Enable exergy optimization"
    )

    # Reference conditions (dead state)
    reference_temperature_f: float = Field(
        default=77.0,
        description="Dead state temperature (F)"
    )
    reference_pressure_psia: float = Field(
        default=14.696,
        description="Dead state pressure (psia)"
    )

    # Optimization weights
    exergy_weight: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Weight for exergy efficiency in optimization"
    )
    cost_weight: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Weight for cost in optimization"
    )
    reliability_weight: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Weight for reliability in optimization"
    )

    # Thresholds
    min_exergy_efficiency_pct: float = Field(
        default=40.0,
        ge=10,
        le=90,
        description="Minimum acceptable exergy efficiency (%)"
    )

    @validator('reliability_weight')
    def weights_sum_to_one(cls, v, values):
        """Ensure optimization weights sum to 1.0."""
        total = v
        if 'exergy_weight' in values:
            total += values['exergy_weight']
        if 'cost_weight' in values:
            total += values['cost_weight']
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Optimization weights must sum to 1.0, got {total}"
            )
        return v


class UnifiedSteamConfig(BaseModel):
    """
    Complete configuration for GL-003 Unified Steam System Optimizer.

    This configuration consolidates settings from:
    - GL-003 STEAMWISE (steam distribution and header balancing)
    - GL-012 STEAMQUAL (steam quality monitoring)

    Example:
        >>> config = UnifiedSteamConfig(
        ...     agent_id="GL-003-001",
        ...     name="Plant Steam System",
        ...     headers=[
        ...         SteamHeaderConfig(
        ...             name="HP",
        ...             design_pressure_psig=600,
        ...             min_pressure_psig=580,
        ...             max_pressure_psig=620,
        ...         ),
        ...     ],
        ... )
    """

    # Agent identification
    agent_id: str = Field(
        default="GL-003-UNIFIED",
        description="Agent identifier"
    )
    name: str = Field(
        default="Unified Steam System Optimizer",
        description="Human-readable name"
    )
    version: str = Field(
        default="2.0.0",
        description="Configuration version"
    )

    # Steam headers
    headers: List[SteamHeaderConfig] = Field(
        default_factory=list,
        description="Steam header configurations"
    )

    # PRVs
    prvs: List[PRVConfig] = Field(
        default_factory=list,
        description="PRV configurations"
    )

    # Desuperheaters
    desuperheaters: List[DesuperheaterConfig] = Field(
        default_factory=list,
        description="Desuperheater configurations"
    )

    # Quality monitoring
    quality: QualityMonitoringConfig = Field(
        default_factory=QualityMonitoringConfig,
        description="Quality monitoring configuration"
    )

    # Condensate
    condensate: CondensateConfig = Field(
        default_factory=CondensateConfig,
        description="Condensate return configuration"
    )

    # Flash recovery
    flash_recovery: List[FlashRecoveryConfig] = Field(
        default_factory=list,
        description="Flash recovery configurations"
    )

    # Steam trap survey
    trap_survey: SteamTrapSurveyConfig = Field(
        default_factory=SteamTrapSurveyConfig,
        description="Steam trap survey configuration"
    )

    # Exergy optimization
    exergy: ExergyOptimizationConfig = Field(
        default_factory=ExergyOptimizationConfig,
        description="Exergy optimization configuration"
    )

    # Provenance
    provenance_enabled: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    # Performance
    calculation_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )

    class Config:
        use_enum_values = True


def create_default_config() -> UnifiedSteamConfig:
    """
    Create a default configuration for a typical industrial steam system.

    Returns:
        UnifiedSteamConfig with typical values for a 3-header system
    """
    return UnifiedSteamConfig(
        agent_id="GL-003-DEFAULT",
        name="Default Steam System",
        headers=[
            SteamHeaderConfig(
                name="HP-MAIN",
                level=SteamHeaderLevel.HIGH_PRESSURE,
                design_pressure_psig=600.0,
                min_pressure_psig=580.0,
                max_pressure_psig=620.0,
                design_flow_lb_hr=100000.0,
                max_flow_lb_hr=120000.0,
                design_temperature_f=750.0,
            ),
            SteamHeaderConfig(
                name="MP-MAIN",
                level=SteamHeaderLevel.MEDIUM_PRESSURE,
                design_pressure_psig=150.0,
                min_pressure_psig=140.0,
                max_pressure_psig=160.0,
                design_flow_lb_hr=50000.0,
                max_flow_lb_hr=65000.0,
            ),
            SteamHeaderConfig(
                name="LP-MAIN",
                level=SteamHeaderLevel.LOW_PRESSURE,
                design_pressure_psig=15.0,
                min_pressure_psig=10.0,
                max_pressure_psig=20.0,
                design_flow_lb_hr=25000.0,
                max_flow_lb_hr=35000.0,
            ),
        ],
        prvs=[
            PRVConfig(
                prv_id="PRV-HP-MP",
                inlet_pressure_psig=600.0,
                outlet_pressure_psig=150.0,
                design_flow_lb_hr=30000.0,
                max_flow_lb_hr=40000.0,
                cv_rated=150.0,
                target_opening_min_pct=50.0,
                target_opening_max_pct=70.0,
                desuperheater_enabled=True,
                desuperheater_type=DesuperheaterType.WATER_SPRAY,
                target_superheat_f=50.0,
            ),
            PRVConfig(
                prv_id="PRV-MP-LP",
                inlet_pressure_psig=150.0,
                outlet_pressure_psig=15.0,
                design_flow_lb_hr=15000.0,
                max_flow_lb_hr=20000.0,
                cv_rated=100.0,
                target_opening_min_pct=50.0,
                target_opening_max_pct=70.0,
            ),
        ],
        flash_recovery=[
            FlashRecoveryConfig(
                flash_tank_id="FT-HP",
                condensate_pressure_psig=150.0,
                flash_pressure_psig=15.0,
                flash_steam_destination="LP-MAIN",
            ),
        ],
    )
