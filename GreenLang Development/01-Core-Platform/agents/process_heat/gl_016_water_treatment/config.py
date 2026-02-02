"""
GL-016 WATERGUARD Agent - Configuration Module

Configuration schemas for water treatment monitoring including boiler water
chemistry limits, treatment programs, and optimization parameters.

References:
    - ASME Consensus on Operating Practices for Control of Feedwater/Boiler Water
    - ABMA Guidelines for Water Quality in Industrial Boilers
    - EPRI Boiler Water Chemistry Guidelines
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    BoilerPressureClass,
    TreatmentProgram,
    ChemicalType,
    BlowdownType,
)


# =============================================================================
# ASME/ABMA WATER QUALITY LIMITS
# =============================================================================

class ASMEBoilerWaterLimits(BaseModel):
    """ASME Consensus recommended boiler water limits by pressure class."""

    pressure_class: BoilerPressureClass = Field(
        ...,
        description="Boiler pressure classification"
    )
    pressure_range_psig: str = Field(..., description="Pressure range description")

    # pH limits
    ph_min: float = Field(..., ge=0, le=14, description="Minimum pH")
    ph_max: float = Field(..., ge=0, le=14, description="Maximum pH")

    # Total dissolved solids
    tds_max_ppm: float = Field(..., ge=0, description="Maximum TDS (ppm)")

    # Total alkalinity
    alkalinity_max_ppm: float = Field(..., ge=0, description="Maximum alkalinity (ppm)")

    # Suspended solids
    suspended_solids_max_ppm: float = Field(..., ge=0, description="Maximum suspended solids")

    # Silica
    silica_max_ppm: float = Field(..., ge=0, description="Maximum silica (ppm)")

    # Conductivity
    conductivity_max_umho: float = Field(..., ge=0, description="Maximum conductivity (umho/cm)")

    class Config:
        use_enum_values = True


# ASME recommended limits by pressure class
ASME_BOILER_WATER_LIMITS: Dict[str, ASMEBoilerWaterLimits] = {
    "low_pressure": ASMEBoilerWaterLimits(
        pressure_class=BoilerPressureClass.LOW_PRESSURE,
        pressure_range_psig="0-300",
        ph_min=10.0,
        ph_max=12.0,
        tds_max_ppm=3500,
        alkalinity_max_ppm=700,
        suspended_solids_max_ppm=15,
        silica_max_ppm=150,
        conductivity_max_umho=7000,
    ),
    "medium_pressure": ASMEBoilerWaterLimits(
        pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
        pressure_range_psig="300-900",
        ph_min=10.0,
        ph_max=11.5,
        tds_max_ppm=2500,
        alkalinity_max_ppm=500,
        suspended_solids_max_ppm=10,
        silica_max_ppm=30,
        conductivity_max_umho=5000,
    ),
    "high_pressure": ASMEBoilerWaterLimits(
        pressure_class=BoilerPressureClass.HIGH_PRESSURE,
        pressure_range_psig="900-1500",
        ph_min=9.5,
        ph_max=10.5,
        tds_max_ppm=1500,
        alkalinity_max_ppm=200,
        suspended_solids_max_ppm=5,
        silica_max_ppm=5,
        conductivity_max_umho=3000,
    ),
    "supercritical": ASMEBoilerWaterLimits(
        pressure_class=BoilerPressureClass.SUPERCRITICAL,
        pressure_range_psig=">1500",
        ph_min=9.0,
        ph_max=10.0,
        tds_max_ppm=100,
        alkalinity_max_ppm=50,
        suspended_solids_max_ppm=1,
        silica_max_ppm=0.5,
        conductivity_max_umho=200,
    ),
}


class ASMEFeedwaterLimits(BaseModel):
    """ASME Consensus recommended feedwater limits by pressure class."""

    pressure_class: BoilerPressureClass = Field(..., description="Pressure class")
    pressure_range_psig: str = Field(..., description="Pressure range")

    # Dissolved oxygen
    dissolved_oxygen_max_ppb: float = Field(
        ...,
        ge=0,
        description="Maximum dissolved O2 (ppb)"
    )

    # Total iron
    iron_max_ppb: float = Field(..., ge=0, description="Maximum iron (ppb)")

    # Total copper
    copper_max_ppb: float = Field(..., ge=0, description="Maximum copper (ppb)")

    # Total hardness
    total_hardness_max_ppm: float = Field(..., ge=0, description="Maximum hardness (ppm)")

    # pH
    ph_min: float = Field(..., ge=0, le=14, description="Minimum pH")
    ph_max: float = Field(..., ge=0, le=14, description="Maximum pH")

    # Silica
    silica_max_ppm: float = Field(..., ge=0, description="Maximum silica (ppm)")

    # Cation conductivity
    cation_conductivity_max_umho: float = Field(
        ...,
        ge=0,
        description="Maximum cation conductivity (umho/cm)"
    )

    class Config:
        use_enum_values = True


# ASME recommended feedwater limits by pressure class
ASME_FEEDWATER_LIMITS: Dict[str, ASMEFeedwaterLimits] = {
    "low_pressure": ASMEFeedwaterLimits(
        pressure_class=BoilerPressureClass.LOW_PRESSURE,
        pressure_range_psig="0-300",
        dissolved_oxygen_max_ppb=7,
        iron_max_ppb=100,
        copper_max_ppb=50,
        total_hardness_max_ppm=0.3,
        ph_min=8.3,
        ph_max=10.0,
        silica_max_ppm=1.0,
        cation_conductivity_max_umho=1.0,
    ),
    "medium_pressure": ASMEFeedwaterLimits(
        pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
        pressure_range_psig="300-900",
        dissolved_oxygen_max_ppb=7,
        iron_max_ppb=20,
        copper_max_ppb=15,
        total_hardness_max_ppm=0.1,
        ph_min=8.5,
        ph_max=9.5,
        silica_max_ppm=0.3,
        cation_conductivity_max_umho=0.5,
    ),
    "high_pressure": ASMEFeedwaterLimits(
        pressure_class=BoilerPressureClass.HIGH_PRESSURE,
        pressure_range_psig="900-1500",
        dissolved_oxygen_max_ppb=5,
        iron_max_ppb=10,
        copper_max_ppb=5,
        total_hardness_max_ppm=0.05,
        ph_min=8.8,
        ph_max=9.3,
        silica_max_ppm=0.1,
        cation_conductivity_max_umho=0.3,
    ),
    "supercritical": ASMEFeedwaterLimits(
        pressure_class=BoilerPressureClass.SUPERCRITICAL,
        pressure_range_psig=">1500",
        dissolved_oxygen_max_ppb=3,
        iron_max_ppb=5,
        copper_max_ppb=2,
        total_hardness_max_ppm=0.01,
        ph_min=9.0,
        ph_max=9.5,
        silica_max_ppm=0.02,
        cation_conductivity_max_umho=0.2,
    ),
}


# =============================================================================
# TREATMENT PROGRAM CONFIGURATIONS
# =============================================================================

class PhosphateTreatmentConfig(BaseModel):
    """Configuration for phosphate treatment programs."""

    program_type: TreatmentProgram = Field(..., description="Treatment program type")

    # Phosphate control range
    phosphate_min_ppm: float = Field(..., ge=0, description="Minimum PO4 (ppm)")
    phosphate_max_ppm: float = Field(..., ge=0, description="Maximum PO4 (ppm)")
    phosphate_target_ppm: float = Field(..., ge=0, description="Target PO4 (ppm)")

    # Na:PO4 ratio (for coordinated/congruent programs)
    na_po4_ratio_min: Optional[float] = Field(default=None, ge=2.0, description="Min Na:PO4")
    na_po4_ratio_max: Optional[float] = Field(default=None, le=3.5, description="Max Na:PO4")
    na_po4_ratio_target: Optional[float] = Field(default=None, description="Target Na:PO4")

    # pH control
    ph_min: float = Field(..., ge=0, le=14, description="Minimum pH")
    ph_max: float = Field(..., ge=0, le=14, description="Maximum pH")

    # Free hydroxide (for precipitate programs)
    free_oh_min_ppm: Optional[float] = Field(default=None, ge=0, description="Min free OH")
    free_oh_max_ppm: Optional[float] = Field(default=None, ge=0, description="Max free OH")

    class Config:
        use_enum_values = True


# Phosphate treatment configurations by program type
PHOSPHATE_TREATMENT_CONFIGS: Dict[str, PhosphateTreatmentConfig] = {
    "coordinated_phosphate": PhosphateTreatmentConfig(
        program_type=TreatmentProgram.COORDINATED_PHOSPHATE,
        phosphate_min_ppm=2.0,
        phosphate_max_ppm=12.0,
        phosphate_target_ppm=8.0,
        na_po4_ratio_min=2.6,
        na_po4_ratio_max=3.0,
        na_po4_ratio_target=2.8,
        ph_min=9.3,
        ph_max=10.0,
    ),
    "congruent_phosphate": PhosphateTreatmentConfig(
        program_type=TreatmentProgram.CONGRUENT_PHOSPHATE,
        phosphate_min_ppm=0.5,
        phosphate_max_ppm=3.0,
        phosphate_target_ppm=2.0,
        na_po4_ratio_min=2.2,
        na_po4_ratio_max=2.8,
        na_po4_ratio_target=2.5,
        ph_min=9.0,
        ph_max=9.6,
    ),
    "phosphate_precipitate": PhosphateTreatmentConfig(
        program_type=TreatmentProgram.PHOSPHATE_PRECIPITATE,
        phosphate_min_ppm=30.0,
        phosphate_max_ppm=60.0,
        phosphate_target_ppm=40.0,
        ph_min=10.0,
        ph_max=11.5,
        free_oh_min_ppm=50,
        free_oh_max_ppm=300,
    ),
    "phosphate_polymer": PhosphateTreatmentConfig(
        program_type=TreatmentProgram.PHOSPHATE_POLYMER,
        phosphate_min_ppm=10.0,
        phosphate_max_ppm=30.0,
        phosphate_target_ppm=15.0,
        ph_min=10.5,
        ph_max=11.5,
    ),
}


class OxygenScavengerConfig(BaseModel):
    """Configuration for oxygen scavenger treatment."""

    scavenger_type: ChemicalType = Field(..., description="Scavenger chemical type")

    # Stoichiometric ratios (lb scavenger per lb O2)
    stoichiometric_ratio: float = Field(
        ...,
        gt=0,
        description="Stoichiometric ratio (lb/lb O2)"
    )
    recommended_excess_pct: float = Field(
        default=50,
        ge=0,
        le=200,
        description="Recommended excess (%)"
    )

    # Residual targets
    residual_min_ppm: float = Field(..., ge=0, description="Minimum residual (ppm)")
    residual_max_ppm: float = Field(..., ge=0, description="Maximum residual (ppm)")
    residual_target_ppm: float = Field(..., ge=0, description="Target residual (ppm)")

    # Operating conditions
    max_temperature_f: float = Field(..., ge=0, description="Maximum operating temp (F)")
    decomposition_products: List[str] = Field(
        default_factory=list,
        description="Thermal decomposition products"
    )

    # Safety considerations
    passivating: bool = Field(default=False, description="Forms protective oxide layer")
    catalyzed_available: bool = Field(default=False, description="Catalyzed version available")

    class Config:
        use_enum_values = True


# Oxygen scavenger configurations
OXYGEN_SCAVENGER_CONFIGS: Dict[str, OxygenScavengerConfig] = {
    "sulfite": OxygenScavengerConfig(
        scavenger_type=ChemicalType.SULFITE,
        stoichiometric_ratio=7.9,  # 7.9 lb sodium sulfite per lb O2
        recommended_excess_pct=50,
        residual_min_ppm=20,
        residual_max_ppm=40,
        residual_target_ppm=30,
        max_temperature_f=700,
        decomposition_products=["SO2", "H2S"],
        passivating=False,
        catalyzed_available=True,
    ),
    "hydrazine": OxygenScavengerConfig(
        scavenger_type=ChemicalType.HYDRAZINE,
        stoichiometric_ratio=1.0,  # 1.0 lb hydrazine per lb O2
        recommended_excess_pct=100,
        residual_min_ppm=0.02,
        residual_max_ppm=0.05,
        residual_target_ppm=0.03,
        max_temperature_f=1000,
        decomposition_products=["NH3", "N2"],
        passivating=True,
        catalyzed_available=False,
    ),
    "carbohydrazide": OxygenScavengerConfig(
        scavenger_type=ChemicalType.CARBOHYDRAZIDE,
        stoichiometric_ratio=1.4,
        recommended_excess_pct=75,
        residual_min_ppm=0.05,
        residual_max_ppm=0.15,
        residual_target_ppm=0.10,
        max_temperature_f=900,
        decomposition_products=["NH3", "N2", "CO2"],
        passivating=True,
        catalyzed_available=False,
    ),
    "erythorbic_acid": OxygenScavengerConfig(
        scavenger_type=ChemicalType.ERYTHORBIC_ACID,
        stoichiometric_ratio=5.5,
        recommended_excess_pct=50,
        residual_min_ppm=5,
        residual_max_ppm=20,
        residual_target_ppm=10,
        max_temperature_f=600,
        decomposition_products=["CO2", "organic acids"],
        passivating=False,
        catalyzed_available=False,
    ),
}


class AmineConfig(BaseModel):
    """Configuration for amine treatment programs."""

    amine_type: ChemicalType = Field(..., description="Amine chemical type")

    # Properties
    distribution_ratio: float = Field(
        ...,
        gt=0,
        description="Steam/water distribution ratio"
    )
    neutralizing_capacity: float = Field(
        ...,
        gt=0,
        description="Neutralizing capacity (lb CO2/lb amine)"
    )
    basicity_constant: float = Field(..., description="Basicity constant (pKb)")

    # Application
    target_ph_range: tuple = Field(..., description="Target condensate pH range")
    typical_dose_ppm: float = Field(..., ge=0, description="Typical dose (ppm)")

    # Characteristics
    filming_capability: bool = Field(
        default=False,
        description="Has filming properties"
    )
    fda_approved: bool = Field(default=False, description="FDA approved for food contact")

    class Config:
        use_enum_values = True


# Amine configurations
AMINE_CONFIGS: Dict[str, AmineConfig] = {
    "morpholine": AmineConfig(
        amine_type=ChemicalType.MORPHOLINE,
        distribution_ratio=0.4,
        neutralizing_capacity=0.5,
        basicity_constant=5.5,
        target_ph_range=(8.5, 9.0),
        typical_dose_ppm=5.0,
        filming_capability=False,
        fda_approved=False,
    ),
    "cyclohexylamine": AmineConfig(
        amine_type=ChemicalType.CYCLOHEXYLAMINE,
        distribution_ratio=4.0,
        neutralizing_capacity=0.45,
        basicity_constant=3.4,
        target_ph_range=(8.8, 9.2),
        typical_dose_ppm=2.0,
        filming_capability=True,
        fda_approved=False,
    ),
    "diethylaminoethanol": AmineConfig(
        amine_type=ChemicalType.DIETHYLAMINOETHANOL,
        distribution_ratio=1.7,
        neutralizing_capacity=0.38,
        basicity_constant=4.2,
        target_ph_range=(8.5, 9.0),
        typical_dose_ppm=3.0,
        filming_capability=True,
        fda_approved=True,
    ),
}


# =============================================================================
# BLOWDOWN CONFIGURATION
# =============================================================================

class BlowdownConfig(BaseModel):
    """Configuration for blowdown optimization."""

    blowdown_type: BlowdownType = Field(
        default=BlowdownType.CONTINUOUS,
        description="Blowdown type"
    )

    # Cycles of concentration limits
    min_cycles: float = Field(default=3.0, ge=1, description="Minimum cycles")
    max_cycles: float = Field(default=10.0, ge=1, description="Maximum cycles")
    target_cycles: float = Field(default=6.0, ge=1, description="Target cycles")

    # Blowdown rate limits
    min_blowdown_rate_pct: float = Field(default=1.0, ge=0, description="Minimum rate (%)")
    max_blowdown_rate_pct: float = Field(default=10.0, ge=0, description="Maximum rate (%)")

    # Heat recovery
    heat_recovery_enabled: bool = Field(default=True, description="Enable heat recovery")
    flash_tank_available: bool = Field(default=True, description="Flash tank available")
    heat_exchanger_available: bool = Field(default=True, description="Heat exchanger available")

    # Control method
    tds_control: bool = Field(default=True, description="TDS-based control")
    conductivity_control: bool = Field(default=True, description="Conductivity-based control")
    silica_control: bool = Field(default=False, description="Silica-based control")

    class Config:
        use_enum_values = True


# =============================================================================
# DEAERATOR CONFIGURATION
# =============================================================================

class DeaeratorConfig(BaseModel):
    """Configuration for deaerator monitoring."""

    deaerator_type: str = Field(
        default="spray_tray",
        description="Type (spray, tray, spray_tray)"
    )

    # Operating pressure
    design_pressure_psig: float = Field(
        default=5.0,
        ge=0,
        le=30,
        description="Design operating pressure (psig)"
    )
    min_pressure_psig: float = Field(default=3.0, ge=0, description="Minimum pressure (psig)")
    max_pressure_psig: float = Field(default=7.0, ge=0, description="Maximum pressure (psig)")

    # Performance targets
    outlet_o2_target_ppb: float = Field(default=5.0, ge=0, description="Target outlet O2 (ppb)")
    outlet_o2_max_ppb: float = Field(default=7.0, ge=0, description="Maximum outlet O2 (ppb)")

    # Vent rate
    min_vent_rate_pct: float = Field(default=0.5, ge=0, description="Minimum vent rate (%)")
    max_vent_rate_pct: float = Field(default=2.0, ge=0, description="Maximum vent rate (%)")

    # Storage time
    min_storage_time_min: float = Field(default=10, ge=0, description="Minimum storage time (min)")


# =============================================================================
# MAIN WATER TREATMENT CONFIGURATION
# =============================================================================

class WaterTreatmentConfig(BaseModel):
    """
    Complete configuration for GL-016 WATERGUARD agent.

    This configuration defines all parameters for water treatment monitoring
    including chemistry limits, treatment programs, and optimization settings.
    """

    # Identity
    system_id: str = Field(..., description="Water treatment system identifier")
    name: str = Field(default="", description="System name")

    # Boiler configuration
    boiler_pressure_class: BoilerPressureClass = Field(
        default=BoilerPressureClass.MEDIUM_PRESSURE,
        description="Boiler pressure classification"
    )
    operating_pressure_psig: float = Field(
        default=150.0,
        ge=0,
        description="Normal operating pressure (psig)"
    )
    design_pressure_psig: float = Field(
        default=200.0,
        ge=0,
        description="Design pressure (psig)"
    )
    steam_capacity_lb_hr: float = Field(
        default=50000.0,
        gt=0,
        description="Steam capacity (lb/hr)"
    )

    # Treatment program
    treatment_program: TreatmentProgram = Field(
        default=TreatmentProgram.PHOSPHATE_POLYMER,
        description="Water treatment program"
    )
    phosphate_config: Optional[PhosphateTreatmentConfig] = Field(
        default=None,
        description="Phosphate treatment configuration"
    )

    # Oxygen scavenger
    oxygen_scavenger_type: ChemicalType = Field(
        default=ChemicalType.SULFITE,
        description="Oxygen scavenger type"
    )
    oxygen_scavenger_config: Optional[OxygenScavengerConfig] = Field(
        default=None,
        description="Oxygen scavenger configuration"
    )

    # Amine treatment
    amine_treatment_enabled: bool = Field(default=True, description="Enable amine treatment")
    amine_type: Optional[ChemicalType] = Field(
        default=ChemicalType.MORPHOLINE,
        description="Amine type for condensate protection"
    )
    amine_config: Optional[AmineConfig] = Field(
        default=None,
        description="Amine configuration"
    )

    # Blowdown
    blowdown_config: BlowdownConfig = Field(
        default_factory=BlowdownConfig,
        description="Blowdown configuration"
    )

    # Deaerator
    deaerator_config: DeaeratorConfig = Field(
        default_factory=DeaeratorConfig,
        description="Deaerator configuration"
    )

    # Condensate system
    condensate_return_pct: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Condensate return percentage"
    )

    # Makeup water
    makeup_water_source: str = Field(
        default="softened",
        description="Makeup water source (raw, softened, demin, RO)"
    )

    # Monitoring intervals
    continuous_monitoring_enabled: bool = Field(
        default=True,
        description="Enable continuous monitoring"
    )
    sample_interval_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="Manual sample interval (minutes)"
    )

    # Alarm settings
    alarm_on_out_of_spec: bool = Field(default=True, description="Alarm on out-of-spec")
    alarm_on_trend_deviation: bool = Field(default=True, description="Alarm on trend deviation")
    trend_deviation_threshold_pct: float = Field(
        default=10.0,
        ge=1,
        le=50,
        description="Trend deviation threshold (%)"
    )

    # Economic parameters
    fuel_cost_per_mmbtu: float = Field(default=5.0, gt=0, description="Fuel cost ($/MMBTU)")
    water_cost_per_kgal: float = Field(default=3.0, gt=0, description="Water cost ($/kgal)")
    chemical_cost_factor: float = Field(default=1.0, gt=0, description="Chemical cost factor")

    # Safety
    sil_level: int = Field(default=2, ge=1, le=3, description="Safety Integrity Level")
    high_alarm_enabled: bool = Field(default=True, description="Enable high alarms")
    low_alarm_enabled: bool = Field(default=True, description="Enable low alarms")

    # Data retention
    data_retention_days: int = Field(default=365, ge=30, description="Data retention (days)")
    historian_integration: bool = Field(default=True, description="Historian integration")

    class Config:
        use_enum_values = True

    @validator("name", always=True)
    def set_default_name(cls, v, values):
        """Set default name from system_id."""
        if not v and "system_id" in values:
            return f"Water Treatment System {values['system_id']}"
        return v

    @validator("operating_pressure_psig")
    def validate_operating_pressure(cls, v, values):
        """Validate operating pressure is less than design pressure."""
        if "design_pressure_psig" in values:
            if v > values["design_pressure_psig"]:
                raise ValueError("Operating pressure cannot exceed design pressure")
        return v


def get_boiler_water_limits(
    pressure_class: BoilerPressureClass
) -> ASMEBoilerWaterLimits:
    """
    Get ASME recommended boiler water limits for given pressure class.

    Args:
        pressure_class: Boiler pressure classification

    Returns:
        ASMEBoilerWaterLimits for the pressure class
    """
    key = pressure_class.value if hasattr(pressure_class, 'value') else str(pressure_class)
    return ASME_BOILER_WATER_LIMITS.get(key, ASME_BOILER_WATER_LIMITS["medium_pressure"])


def get_feedwater_limits(
    pressure_class: BoilerPressureClass
) -> ASMEFeedwaterLimits:
    """
    Get ASME recommended feedwater limits for given pressure class.

    Args:
        pressure_class: Boiler pressure classification

    Returns:
        ASMEFeedwaterLimits for the pressure class
    """
    key = pressure_class.value if hasattr(pressure_class, 'value') else str(pressure_class)
    return ASME_FEEDWATER_LIMITS.get(key, ASME_FEEDWATER_LIMITS["medium_pressure"])


def get_phosphate_config(
    program: TreatmentProgram
) -> Optional[PhosphateTreatmentConfig]:
    """
    Get phosphate treatment configuration for given program.

    Args:
        program: Treatment program type

    Returns:
        PhosphateTreatmentConfig or None
    """
    key = program.value if hasattr(program, 'value') else str(program)
    return PHOSPHATE_TREATMENT_CONFIGS.get(key)


def get_scavenger_config(
    scavenger_type: ChemicalType
) -> Optional[OxygenScavengerConfig]:
    """
    Get oxygen scavenger configuration.

    Args:
        scavenger_type: Scavenger chemical type

    Returns:
        OxygenScavengerConfig or None
    """
    key = scavenger_type.value if hasattr(scavenger_type, 'value') else str(scavenger_type)
    return OXYGEN_SCAVENGER_CONFIGS.get(key)


def get_amine_config(
    amine_type: ChemicalType
) -> Optional[AmineConfig]:
    """
    Get amine configuration.

    Args:
        amine_type: Amine chemical type

    Returns:
        AmineConfig or None
    """
    key = amine_type.value if hasattr(amine_type, 'value') else str(amine_type)
    return AMINE_CONFIGS.get(key)


def determine_pressure_class(pressure_psig: float) -> BoilerPressureClass:
    """
    Determine boiler pressure class from operating pressure.

    Args:
        pressure_psig: Operating pressure in psig

    Returns:
        BoilerPressureClass enumeration value
    """
    if pressure_psig < 300:
        return BoilerPressureClass.LOW_PRESSURE
    elif pressure_psig < 900:
        return BoilerPressureClass.MEDIUM_PRESSURE
    elif pressure_psig < 1500:
        return BoilerPressureClass.HIGH_PRESSURE
    else:
        return BoilerPressureClass.SUPERCRITICAL
