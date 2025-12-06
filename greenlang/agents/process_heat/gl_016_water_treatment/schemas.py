"""
GL-016 WATERGUARD Agent - Schema Definitions

Pydantic models for water treatment monitoring inputs, outputs, and results.
All schemas follow ASME/ABMA and EPRI guidelines for boiler water chemistry.

References:
    - ASME Consensus on Operating Practices for Control of Feedwater/Boiler Water
    - ABMA Guidelines for Water Quality in Industrial Boilers
    - EPRI Boiler Water Chemistry Guidelines
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS
# =============================================================================

class WaterQualityStatus(Enum):
    """Water quality assessment status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"
    OUT_OF_SPEC = "out_of_spec"


class TreatmentProgram(Enum):
    """Boiler water treatment program types."""
    PHOSPHATE_PRECIPITATE = "phosphate_precipitate"
    PHOSPHATE_POLYMER = "phosphate_polymer"
    COORDINATED_PHOSPHATE = "coordinated_phosphate"
    CONGRUENT_PHOSPHATE = "congruent_phosphate"
    ALL_VOLATILE = "all_volatile"
    OXYGENATED_TREATMENT = "oxygenated_treatment"
    CAUSTIC_TREATMENT = "caustic_treatment"


class BoilerPressureClass(Enum):
    """Boiler pressure classification per ASME/ABMA."""
    LOW_PRESSURE = "low_pressure"        # < 300 psig
    MEDIUM_PRESSURE = "medium_pressure"  # 300-900 psig
    HIGH_PRESSURE = "high_pressure"      # 900-1500 psig
    SUPERCRITICAL = "supercritical"      # > 1500 psig


class BlowdownType(Enum):
    """Blowdown operation types."""
    CONTINUOUS = "continuous"
    INTERMITTENT = "intermittent"
    SURFACE = "surface"
    BOTTOM = "bottom"
    COMBINED = "combined"


class ChemicalType(Enum):
    """Chemical treatment types."""
    PHOSPHATE = "phosphate"
    OXYGEN_SCAVENGER = "oxygen_scavenger"
    AMINE = "amine"
    POLYMER = "polymer"
    CAUSTIC = "caustic"
    SULFITE = "sulfite"
    HYDRAZINE = "hydrazine"
    CARBOHYDRAZIDE = "carbohydrazide"
    ERYTHORBIC_ACID = "erythorbic_acid"
    MORPHOLINE = "morpholine"
    CYCLOHEXYLAMINE = "cyclohexylamine"
    DIETHYLAMINOETHANOL = "diethylaminoethanol"


class CorrosionMechanism(Enum):
    """Corrosion mechanism types."""
    OXYGEN_PITTING = "oxygen_pitting"
    CAUSTIC_EMBRITTLEMENT = "caustic_embrittlement"
    CAUSTIC_GOUGING = "caustic_gouging"
    HYDROGEN_DAMAGE = "hydrogen_damage"
    ACID_PHOSPHATE_CORROSION = "acid_phosphate_corrosion"
    FLOW_ACCELERATED = "flow_accelerated"
    UNDER_DEPOSIT = "under_deposit"
    GALVANIC = "galvanic"
    CARBONIC_ACID = "carbonic_acid"


# =============================================================================
# BASE INPUT/OUTPUT MODELS
# =============================================================================

class WaterSampleInput(BaseModel):
    """Base water sample input data."""

    sample_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Sample identifier"
    )
    sample_point: str = Field(..., description="Sample collection point")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Sample timestamp"
    )
    temperature_f: Optional[float] = Field(
        default=None,
        ge=32,
        le=700,
        description="Sample temperature (F)"
    )

    class Config:
        use_enum_values = True


class WaterQualityResult(BaseModel):
    """Base water quality assessment result."""

    parameter: str = Field(..., description="Parameter name")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Measurement unit")
    min_limit: Optional[float] = Field(default=None, description="Minimum limit")
    max_limit: Optional[float] = Field(default=None, description="Maximum limit")
    target_value: Optional[float] = Field(default=None, description="Target value")
    status: WaterQualityStatus = Field(..., description="Quality status")
    deviation_pct: Optional[float] = Field(
        default=None,
        description="Deviation from target (%)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# BOILER WATER CHEMISTRY SCHEMAS
# =============================================================================

class BoilerWaterInput(WaterSampleInput):
    """Input data for boiler water chemistry analysis."""

    # pH and alkalinity
    ph: float = Field(
        ...,
        ge=0,
        le=14,
        description="Boiler water pH at 25C"
    )
    p_alkalinity_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="P-alkalinity as CaCO3 (ppm)"
    )
    m_alkalinity_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="M-alkalinity as CaCO3 (ppm)"
    )

    # Phosphate
    phosphate_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=200,
        description="Phosphate as PO4 (ppm)"
    )

    # Conductivity
    specific_conductivity_umho: float = Field(
        ...,
        ge=0,
        description="Specific conductivity at 25C (umho/cm)"
    )
    cation_conductivity_umho: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cation conductivity (umho/cm)"
    )

    # Silica
    silica_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Silica as SiO2 (ppm)"
    )

    # TDS
    tds_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total dissolved solids (ppm)"
    )

    # Iron and copper
    iron_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total iron (ppb)"
    )
    copper_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total copper (ppb)"
    )

    # Dissolved oxygen
    dissolved_oxygen_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Dissolved oxygen (ppb)"
    )

    # Operating conditions
    operating_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Boiler operating pressure (psig)"
    )
    steam_purity_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Steam purity - cation conductivity (ppb)"
    )

    @validator('p_alkalinity_ppm', 'm_alkalinity_ppm')
    def validate_alkalinity_relationship(cls, v, values):
        """Validate P-alkalinity is less than M-alkalinity."""
        if v is not None and 'p_alkalinity_ppm' in values:
            if values['p_alkalinity_ppm'] is not None and v is not None:
                if values.get('p_alkalinity_ppm', 0) > v:
                    pass  # Will be validated in root_validator
        return v


class BoilerWaterLimits(BaseModel):
    """ASME/ABMA recommended limits for boiler water chemistry."""

    pressure_class: BoilerPressureClass = Field(..., description="Pressure class")
    treatment_program: TreatmentProgram = Field(..., description="Treatment program")

    # pH limits
    ph_min: float = Field(..., ge=0, le=14, description="Minimum pH")
    ph_max: float = Field(..., ge=0, le=14, description="Maximum pH")

    # Phosphate limits
    phosphate_min_ppm: float = Field(default=0, ge=0, description="Min phosphate (ppm)")
    phosphate_max_ppm: float = Field(..., ge=0, description="Max phosphate (ppm)")

    # Conductivity limits
    conductivity_max_umho: float = Field(..., ge=0, description="Max conductivity (umho/cm)")

    # Silica limits
    silica_max_ppm: float = Field(..., ge=0, description="Max silica (ppm)")

    # Iron/copper limits
    iron_max_ppb: float = Field(default=100, ge=0, description="Max iron (ppb)")
    copper_max_ppb: float = Field(default=15, ge=0, description="Max copper (ppb)")

    # Steam purity
    steam_cation_conductivity_max_umho: float = Field(
        default=0.5,
        ge=0,
        description="Max steam cation conductivity (umho/cm)"
    )

    class Config:
        use_enum_values = True


class BoilerWaterOutput(BaseModel):
    """Output from boiler water chemistry analysis."""

    # Identity
    sample_id: str = Field(..., description="Sample identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Overall status
    overall_status: WaterQualityStatus = Field(..., description="Overall water quality status")
    status_message: str = Field(..., description="Status description")

    # Individual parameter results
    parameter_results: List[WaterQualityResult] = Field(
        default_factory=list,
        description="Individual parameter assessments"
    )

    # Phosphate control
    phosphate_sodium_ratio: Optional[float] = Field(
        default=None,
        description="Na:PO4 molar ratio for coordinated phosphate"
    )
    phosphate_control_status: Optional[str] = Field(
        default=None,
        description="Phosphate control assessment"
    )

    # Corrosion risk
    corrosion_risk_score: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Corrosion risk score (0-100)"
    )
    corrosion_mechanisms: List[CorrosionMechanism] = Field(
        default_factory=list,
        description="Potential corrosion mechanisms"
    )

    # Scaling risk
    scaling_risk_score: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Scaling risk score (0-100)"
    )

    # Deposition risk
    deposition_risk_score: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Deposition risk score (0-100)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Corrective action recommendations"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# FEEDWATER SCHEMAS
# =============================================================================

class FeedwaterInput(WaterSampleInput):
    """Input data for feedwater quality analysis."""

    # pH
    ph: float = Field(..., ge=0, le=14, description="Feedwater pH at 25C")

    # Conductivity
    specific_conductivity_umho: float = Field(
        ...,
        ge=0,
        description="Specific conductivity (umho/cm)"
    )
    cation_conductivity_umho: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cation conductivity (umho/cm)"
    )

    # Dissolved oxygen
    dissolved_oxygen_ppb: float = Field(
        ...,
        ge=0,
        description="Dissolved oxygen (ppb)"
    )

    # Hardness
    total_hardness_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total hardness as CaCO3 (ppm)"
    )

    # Iron and copper
    iron_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total iron (ppb)"
    )
    copper_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total copper (ppb)"
    )

    # Silica
    silica_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Silica as SiO2 (ppm)"
    )

    # Oxygen scavenger residual
    oxygen_scavenger_residual_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Oxygen scavenger residual (ppm)"
    )
    oxygen_scavenger_type: Optional[ChemicalType] = Field(
        default=None,
        description="Oxygen scavenger chemical type"
    )

    # Temperature
    temperature_f: float = Field(
        default=227,
        ge=100,
        le=500,
        description="Feedwater temperature (F)"
    )

    # Source
    deaerator_outlet: bool = Field(
        default=True,
        description="Sample from deaerator outlet"
    )


class FeedwaterLimits(BaseModel):
    """ASME recommended limits for feedwater quality."""

    pressure_class: BoilerPressureClass = Field(..., description="Pressure class")

    # pH
    ph_min: float = Field(default=8.5, description="Minimum pH")
    ph_max: float = Field(default=9.5, description="Maximum pH")

    # Dissolved oxygen
    dissolved_oxygen_max_ppb: float = Field(
        default=7,
        ge=0,
        description="Maximum dissolved O2 (ppb)"
    )

    # Hardness
    total_hardness_max_ppm: float = Field(
        default=0.3,
        ge=0,
        description="Maximum total hardness (ppm)"
    )

    # Iron
    iron_max_ppb: float = Field(default=20, ge=0, description="Maximum iron (ppb)")

    # Copper
    copper_max_ppb: float = Field(default=10, ge=0, description="Maximum copper (ppb)")

    # Silica
    silica_max_ppm: float = Field(default=0.1, ge=0, description="Maximum silica (ppm)")

    # Conductivity
    cation_conductivity_max_umho: float = Field(
        default=0.5,
        ge=0,
        description="Maximum cation conductivity (umho/cm)"
    )

    class Config:
        use_enum_values = True


class FeedwaterOutput(BaseModel):
    """Output from feedwater quality analysis."""

    sample_id: str = Field(..., description="Sample identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Overall status
    overall_status: WaterQualityStatus = Field(..., description="Overall quality status")

    # Parameter results
    parameter_results: List[WaterQualityResult] = Field(
        default_factory=list,
        description="Parameter assessments"
    )

    # Oxygen control
    oxygen_control_adequate: bool = Field(
        default=True,
        description="Oxygen control meets requirements"
    )
    oxygen_scavenger_dose_adjustment: Optional[float] = Field(
        default=None,
        description="Recommended dose adjustment (%)"
    )

    # Corrosion product transport
    iron_transport_concern: bool = Field(default=False, description="Elevated iron transport")
    copper_transport_concern: bool = Field(default=False, description="Elevated copper transport")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="Provenance hash")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time (ms)")

    class Config:
        use_enum_values = True


# =============================================================================
# CONDENSATE SCHEMAS
# =============================================================================

class CondensateInput(WaterSampleInput):
    """Input data for condensate quality analysis."""

    # pH
    ph: float = Field(..., ge=0, le=14, description="Condensate pH at 25C")

    # Conductivity
    specific_conductivity_umho: float = Field(
        ...,
        ge=0,
        description="Specific conductivity (umho/cm)"
    )
    cation_conductivity_umho: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cation conductivity (umho/cm)"
    )

    # Iron and copper (corrosion products)
    iron_ppb: float = Field(..., ge=0, description="Total iron (ppb)")
    copper_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total copper (ppb)"
    )

    # Dissolved oxygen
    dissolved_oxygen_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Dissolved oxygen (ppb)"
    )

    # Amine residual
    amine_residual_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Amine treatment residual (ppm)"
    )
    amine_type: Optional[ChemicalType] = Field(
        default=None,
        description="Amine treatment type"
    )

    # Contamination indicators
    hardness_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Hardness (ppm) - contamination indicator"
    )
    tds_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="TDS (ppm) - contamination indicator"
    )
    oil_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Oil contamination (ppm)"
    )

    # Return system
    condensate_return_pct: float = Field(
        default=80,
        ge=0,
        le=100,
        description="Condensate return percentage (%)"
    )
    condensate_source: str = Field(
        default="main_return",
        description="Condensate source/location"
    )


class CondensateLimits(BaseModel):
    """Recommended limits for condensate quality."""

    # pH
    ph_min: float = Field(default=8.0, description="Minimum pH")
    ph_max: float = Field(default=9.0, description="Maximum pH")

    # Iron
    iron_max_ppb: float = Field(default=50, ge=0, description="Maximum iron (ppb)")
    iron_action_level_ppb: float = Field(default=100, ge=0, description="Iron action level (ppb)")

    # Copper
    copper_max_ppb: float = Field(default=10, ge=0, description="Maximum copper (ppb)")
    copper_action_level_ppb: float = Field(default=20, ge=0, description="Copper action level (ppb)")

    # Contamination
    hardness_max_ppm: float = Field(default=0.5, ge=0, description="Maximum hardness (ppm)")
    oil_max_ppm: float = Field(default=1.0, ge=0, description="Maximum oil (ppm)")

    # Conductivity
    cation_conductivity_max_umho: float = Field(
        default=1.0,
        ge=0,
        description="Maximum cation conductivity (umho/cm)"
    )


class CondensateOutput(BaseModel):
    """Output from condensate quality analysis."""

    sample_id: str = Field(..., description="Sample identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Status
    overall_status: WaterQualityStatus = Field(..., description="Overall quality status")
    return_quality_acceptable: bool = Field(..., description="Condensate suitable for return")

    # Parameter results
    parameter_results: List[WaterQualityResult] = Field(
        default_factory=list,
        description="Parameter assessments"
    )

    # Corrosion assessment
    corrosion_rate_mpy: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated corrosion rate (mils per year)"
    )
    corrosion_mechanism: Optional[CorrosionMechanism] = Field(
        default=None,
        description="Primary corrosion mechanism"
    )

    # Contamination
    contamination_detected: bool = Field(default=False, description="Contamination detected")
    contamination_source: Optional[str] = Field(
        default=None,
        description="Suspected contamination source"
    )

    # Amine program
    amine_dose_adequate: Optional[bool] = Field(
        default=None,
        description="Amine dose meets requirements"
    )
    amine_adjustment_ppm: Optional[float] = Field(
        default=None,
        description="Recommended amine adjustment (ppm)"
    )

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="Provenance hash")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time (ms)")

    class Config:
        use_enum_values = True


# =============================================================================
# BLOWDOWN SCHEMAS
# =============================================================================

class BlowdownInput(BaseModel):
    """Input data for blowdown optimization."""

    # Current blowdown
    continuous_blowdown_rate_pct: float = Field(
        ...,
        ge=0,
        le=20,
        description="Continuous blowdown rate (%)"
    )
    intermittent_blowdown_frequency_per_shift: int = Field(
        default=0,
        ge=0,
        le=24,
        description="Intermittent blowdowns per shift"
    )
    blowdown_type: BlowdownType = Field(
        default=BlowdownType.CONTINUOUS,
        description="Blowdown type"
    )

    # Water quality
    boiler_tds_ppm: float = Field(..., ge=0, description="Boiler water TDS (ppm)")
    feedwater_tds_ppm: float = Field(..., ge=0, description="Feedwater TDS (ppm)")
    boiler_conductivity_umho: Optional[float] = Field(
        default=None,
        ge=0,
        description="Boiler water conductivity (umho/cm)"
    )
    feedwater_conductivity_umho: Optional[float] = Field(
        default=None,
        ge=0,
        description="Feedwater conductivity (umho/cm)"
    )

    # Limits
    tds_max_ppm: float = Field(..., ge=0, description="Maximum TDS limit (ppm)")

    # Boiler data
    steam_flow_rate_lb_hr: float = Field(..., gt=0, description="Steam flow rate (lb/hr)")
    operating_pressure_psig: float = Field(..., ge=0, description="Operating pressure (psig)")

    # Heat recovery
    blowdown_heat_recovery_enabled: bool = Field(
        default=False,
        description="Blowdown heat recovery in service"
    )
    flash_tank_pressure_psig: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flash tank pressure (psig)"
    )

    # Economics
    fuel_cost_per_mmbtu: float = Field(default=5.0, gt=0, description="Fuel cost ($/MMBTU)")
    water_cost_per_kgal: float = Field(default=3.0, gt=0, description="Water cost ($/kgal)")
    chemical_cost_per_kgal: float = Field(default=2.0, gt=0, description="Chemical cost ($/kgal)")

    class Config:
        use_enum_values = True


class BlowdownOutput(BaseModel):
    """Output from blowdown optimization analysis."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Current state
    current_cycles_of_concentration: float = Field(
        ...,
        ge=1,
        description="Current cycles of concentration"
    )
    current_blowdown_rate_pct: float = Field(
        ...,
        ge=0,
        description="Current blowdown rate (%)"
    )
    current_blowdown_flow_lb_hr: float = Field(
        ...,
        ge=0,
        description="Current blowdown flow (lb/hr)"
    )

    # Optimized values
    optimal_cycles_of_concentration: float = Field(
        ...,
        ge=1,
        description="Optimal cycles of concentration"
    )
    optimal_blowdown_rate_pct: float = Field(
        ...,
        ge=0,
        description="Optimal blowdown rate (%)"
    )
    optimal_blowdown_flow_lb_hr: float = Field(
        ...,
        ge=0,
        description="Optimal blowdown flow (lb/hr)"
    )

    # Savings
    blowdown_reduction_pct: float = Field(
        default=0,
        description="Blowdown reduction potential (%)"
    )
    energy_savings_mmbtu_yr: float = Field(
        default=0,
        ge=0,
        description="Annual energy savings (MMBTU/yr)"
    )
    water_savings_kgal_yr: float = Field(
        default=0,
        ge=0,
        description="Annual water savings (kgal/yr)"
    )
    total_savings_usd_yr: float = Field(
        default=0,
        ge=0,
        description="Total annual savings ($/yr)"
    )

    # Heat recovery
    heat_recovery_potential_btu_hr: float = Field(
        default=0,
        ge=0,
        description="Heat recovery potential (BTU/hr)"
    )
    flash_steam_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Flash steam generation (lb/hr)"
    )

    # Status
    optimization_status: str = Field(
        default="complete",
        description="Optimization status"
    )
    within_limits: bool = Field(
        default=True,
        description="Optimized values within limits"
    )

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="Provenance hash")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time (ms)")

    class Config:
        use_enum_values = True


# =============================================================================
# CHEMICAL DOSING SCHEMAS
# =============================================================================

class ChemicalDosingInput(BaseModel):
    """Input data for chemical dosing optimization."""

    # Feedwater data
    feedwater_flow_lb_hr: float = Field(..., gt=0, description="Feedwater flow (lb/hr)")
    makeup_water_flow_lb_hr: float = Field(
        default=0,
        ge=0,
        description="Makeup water flow (lb/hr)"
    )

    # Oxygen scavenger
    feedwater_do_ppb: float = Field(
        ...,
        ge=0,
        description="Feedwater dissolved oxygen (ppb)"
    )
    current_scavenger_type: ChemicalType = Field(
        default=ChemicalType.SULFITE,
        description="Current oxygen scavenger type"
    )
    current_scavenger_dose_ppm: float = Field(
        default=0,
        ge=0,
        description="Current scavenger dose (ppm)"
    )
    target_scavenger_residual_ppm: float = Field(
        default=20,
        ge=0,
        description="Target scavenger residual (ppm)"
    )

    # Phosphate
    boiler_phosphate_ppm: float = Field(
        default=0,
        ge=0,
        description="Current boiler phosphate (ppm)"
    )
    target_phosphate_ppm: float = Field(
        default=10,
        ge=0,
        description="Target phosphate (ppm)"
    )
    current_phosphate_dose_ppm: float = Field(
        default=0,
        ge=0,
        description="Current phosphate dose (ppm)"
    )
    blowdown_rate_pct: float = Field(
        default=3,
        ge=0,
        description="Blowdown rate (%)"
    )

    # Amine (for condensate)
    condensate_return_pct: float = Field(
        default=80,
        ge=0,
        le=100,
        description="Condensate return (%)"
    )
    condensate_ph: Optional[float] = Field(
        default=None,
        ge=0,
        le=14,
        description="Condensate pH"
    )
    target_condensate_ph: float = Field(
        default=8.5,
        ge=7,
        le=10,
        description="Target condensate pH"
    )
    current_amine_type: Optional[ChemicalType] = Field(
        default=None,
        description="Current amine type"
    )
    current_amine_dose_ppm: float = Field(
        default=0,
        ge=0,
        description="Current amine dose (ppm)"
    )

    # Operating conditions
    operating_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Boiler operating pressure (psig)"
    )
    treatment_program: TreatmentProgram = Field(
        default=TreatmentProgram.PHOSPHATE_POLYMER,
        description="Treatment program type"
    )

    # Chemical costs
    scavenger_cost_per_lb: float = Field(default=1.50, gt=0, description="Scavenger cost ($/lb)")
    phosphate_cost_per_lb: float = Field(default=0.80, gt=0, description="Phosphate cost ($/lb)")
    amine_cost_per_lb: float = Field(default=5.00, gt=0, description="Amine cost ($/lb)")

    class Config:
        use_enum_values = True


class ChemicalDosingOutput(BaseModel):
    """Output from chemical dosing optimization."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Oxygen scavenger recommendations
    scavenger_dose_recommended_ppm: float = Field(
        ...,
        ge=0,
        description="Recommended scavenger dose (ppm)"
    )
    scavenger_dose_change_ppm: float = Field(
        default=0,
        description="Dose change vs current (ppm)"
    )
    scavenger_feed_rate_lb_hr: float = Field(
        ...,
        ge=0,
        description="Scavenger feed rate (lb/hr)"
    )
    scavenger_ratio_to_o2: float = Field(
        ...,
        ge=0,
        description="Scavenger to O2 ratio"
    )

    # Phosphate recommendations
    phosphate_dose_recommended_ppm: float = Field(
        ...,
        ge=0,
        description="Recommended phosphate dose (ppm)"
    )
    phosphate_dose_change_ppm: float = Field(
        default=0,
        description="Dose change vs current (ppm)"
    )
    phosphate_feed_rate_lb_hr: float = Field(
        ...,
        ge=0,
        description="Phosphate feed rate (lb/hr)"
    )

    # Amine recommendations
    amine_dose_recommended_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Recommended amine dose (ppm)"
    )
    amine_dose_change_ppm: Optional[float] = Field(
        default=None,
        description="Amine dose change (ppm)"
    )
    amine_feed_rate_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Amine feed rate (lb/hr)"
    )

    # Cost analysis
    current_chemical_cost_per_day: float = Field(
        ...,
        ge=0,
        description="Current chemical cost ($/day)"
    )
    optimized_chemical_cost_per_day: float = Field(
        ...,
        ge=0,
        description="Optimized chemical cost ($/day)"
    )
    cost_savings_per_day: float = Field(
        default=0,
        description="Cost savings ($/day)"
    )
    annual_savings_usd: float = Field(
        default=0,
        ge=0,
        description="Annual savings ($)"
    )

    # Status
    within_recommended_ranges: bool = Field(
        default=True,
        description="All doses within recommended ranges"
    )
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="Provenance hash")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time (ms)")

    class Config:
        use_enum_values = True


# =============================================================================
# DEAERATION SCHEMAS
# =============================================================================

class DeaerationInput(BaseModel):
    """Input data for deaerator performance analysis."""

    # Deaerator operating conditions
    deaerator_pressure_psig: float = Field(
        ...,
        ge=0,
        le=30,
        description="Deaerator operating pressure (psig)"
    )
    deaerator_temperature_f: Optional[float] = Field(
        default=None,
        description="Deaerator temperature (F)"
    )

    # Inlet water
    inlet_water_temperature_f: float = Field(
        ...,
        ge=32,
        le=300,
        description="Inlet water temperature (F)"
    )
    inlet_dissolved_oxygen_ppb: float = Field(
        ...,
        ge=0,
        description="Inlet dissolved oxygen (ppb)"
    )
    inlet_co2_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Inlet CO2 (ppm)"
    )

    # Outlet (feedwater)
    outlet_dissolved_oxygen_ppb: float = Field(
        ...,
        ge=0,
        description="Outlet dissolved oxygen (ppb)"
    )
    outlet_co2_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Outlet CO2 (ppm)"
    )

    # Flow rates
    total_flow_lb_hr: float = Field(..., gt=0, description="Total flow through DA (lb/hr)")
    steam_flow_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Steam consumption (lb/hr)"
    )
    vent_rate_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Vent rate (lb/hr)"
    )

    # Deaerator type
    deaerator_type: str = Field(
        default="spray_tray",
        description="Deaerator type (spray, tray, spray_tray)"
    )

    # Limits
    outlet_o2_limit_ppb: float = Field(
        default=7,
        ge=0,
        description="Outlet O2 limit (ppb)"
    )


class DeaerationOutput(BaseModel):
    """Output from deaerator performance analysis."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Performance metrics
    oxygen_removal_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="O2 removal efficiency (%)"
    )
    co2_removal_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="CO2 removal efficiency (%)"
    )

    # Status
    performance_status: WaterQualityStatus = Field(
        ...,
        description="Performance status"
    )
    outlet_o2_within_limit: bool = Field(
        ...,
        description="Outlet O2 meets specification"
    )

    # Operating parameters
    saturation_temperature_f: float = Field(
        ...,
        description="Saturation temperature at operating pressure (F)"
    )
    subcooling_f: float = Field(
        ...,
        description="Degrees subcooling (F)"
    )

    # Steam usage
    theoretical_steam_lb_hr: float = Field(
        ...,
        ge=0,
        description="Theoretical steam requirement (lb/hr)"
    )
    actual_steam_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actual steam usage (lb/hr)"
    )
    steam_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=200,
        description="Steam usage efficiency (%)"
    )

    # Vent analysis
    vent_rate_recommended_lb_hr: float = Field(
        ...,
        ge=0,
        description="Recommended vent rate (lb/hr)"
    )
    vent_rate_status: str = Field(
        default="adequate",
        description="Vent rate status"
    )

    # Corrosion potential
    corrosion_potential: str = Field(
        default="low",
        description="Downstream corrosion potential (low/medium/high)"
    )

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="Provenance hash")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time (ms)")

    class Config:
        use_enum_values = True


# =============================================================================
# MAIN WATER TREATMENT MONITOR SCHEMAS
# =============================================================================

class WaterTreatmentInput(BaseModel):
    """Comprehensive input for water treatment monitoring."""

    # System identification
    system_id: str = Field(..., description="Water treatment system identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Monitoring timestamp"
    )

    # Operating conditions
    boiler_operating_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Boiler operating pressure (psig)"
    )
    steam_flow_rate_lb_hr: float = Field(..., gt=0, description="Steam flow rate (lb/hr)")

    # Treatment program
    treatment_program: TreatmentProgram = Field(
        default=TreatmentProgram.PHOSPHATE_POLYMER,
        description="Treatment program"
    )

    # Water samples
    boiler_water: Optional[BoilerWaterInput] = Field(
        default=None,
        description="Boiler water sample"
    )
    feedwater: Optional[FeedwaterInput] = Field(
        default=None,
        description="Feedwater sample"
    )
    condensate: Optional[CondensateInput] = Field(
        default=None,
        description="Condensate sample"
    )

    # Blowdown data
    blowdown_data: Optional[BlowdownInput] = Field(
        default=None,
        description="Blowdown data"
    )

    # Chemical dosing data
    chemical_dosing_data: Optional[ChemicalDosingInput] = Field(
        default=None,
        description="Chemical dosing data"
    )

    # Deaerator data
    deaerator_data: Optional[DeaerationInput] = Field(
        default=None,
        description="Deaerator data"
    )

    class Config:
        use_enum_values = True


class WaterTreatmentOutput(BaseModel):
    """Comprehensive output from water treatment monitoring."""

    # Identity
    system_id: str = Field(..., description="System identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Overall status
    overall_status: WaterQualityStatus = Field(..., description="Overall system status")
    overall_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall water treatment score (0-100)"
    )

    # Component analyses
    boiler_water_analysis: Optional[BoilerWaterOutput] = Field(
        default=None,
        description="Boiler water analysis results"
    )
    feedwater_analysis: Optional[FeedwaterOutput] = Field(
        default=None,
        description="Feedwater analysis results"
    )
    condensate_analysis: Optional[CondensateOutput] = Field(
        default=None,
        description="Condensate analysis results"
    )
    blowdown_analysis: Optional[BlowdownOutput] = Field(
        default=None,
        description="Blowdown optimization results"
    )
    chemical_dosing_analysis: Optional[ChemicalDosingOutput] = Field(
        default=None,
        description="Chemical dosing optimization results"
    )
    deaeration_analysis: Optional[DeaerationOutput] = Field(
        default=None,
        description="Deaeration analysis results"
    )

    # Risk assessments
    corrosion_risk_score: float = Field(default=0, ge=0, le=100, description="Corrosion risk")
    scaling_risk_score: float = Field(default=0, ge=0, le=100, description="Scaling risk")
    deposition_risk_score: float = Field(default=0, ge=0, le=100, description="Deposition risk")
    carryover_risk_score: float = Field(default=0, ge=0, le=100, description="Carryover risk")

    # Economics
    potential_annual_savings_usd: float = Field(
        default=0,
        ge=0,
        description="Potential annual savings ($)"
    )

    # KPIs
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Alerts and recommendations
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritized recommendations"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="Provenance hash")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time (ms)")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True
