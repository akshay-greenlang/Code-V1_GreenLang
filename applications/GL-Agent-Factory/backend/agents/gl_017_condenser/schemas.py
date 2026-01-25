"""
Pydantic Schemas for GL-017 CONDENSYNC Agent

This module defines all input/output data models for the Condenser
Optimization Agent using Pydantic for validation and serialization.

All models follow GreenLang standards for:
- Type safety with comprehensive validation
- Clear field descriptions for API documentation
- Sensible defaults where applicable
- Complete audit trail support
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator, root_validator


class TubeMaterial(str, Enum):
    """Supported condenser tube materials per HEI Standards."""
    ADMIRALTY_BRASS = "admiralty_brass"
    ALUMINUM_BRASS = "aluminum_brass"
    ALUMINUM_BRONZE = "aluminum_bronze"
    ARSENICAL_COPPER = "arsenical_copper"
    COPPER_NICKEL_90_10 = "copper_nickel_90_10"
    COPPER_NICKEL_70_30 = "copper_nickel_70_30"
    STAINLESS_STEEL_304 = "stainless_steel_304"
    STAINLESS_STEEL_316 = "stainless_steel_316"
    TITANIUM = "titanium"
    DUPLEX_STAINLESS = "duplex_stainless"
    CARBON_STEEL = "carbon_steel"


class CoolingWaterSource(str, Enum):
    """Types of cooling water sources."""
    SEAWATER_ONCE_THROUGH = "seawater_once_through"
    FRESHWATER_ONCE_THROUGH = "freshwater_once_through"
    COOLING_TOWER_TREATED = "cooling_tower_treated"
    COOLING_TOWER_UNTREATED = "cooling_tower_untreated"
    RIVER_WATER = "river_water"
    INDUSTRIAL_COOLING = "industrial_cooling"


class FoulingMechanism(str, Enum):
    """Types of fouling mechanisms."""
    BIOLOGICAL = "biological"
    SCALING = "scaling"
    PARTICULATE = "particulate"
    CORROSION = "corrosion"
    MIXED = "mixed"


class CleaningMethod(str, Enum):
    """Condenser tube cleaning methods."""
    MECHANICAL_BALL = "mechanical_ball"
    MECHANICAL_BRUSH = "mechanical_brush"
    CHEMICAL_ACID = "chemical_acid"
    CHEMICAL_BIODISPERSANT = "chemical_biodispersant"
    HIGH_PRESSURE_WATER = "high_pressure_water"
    BACKWASH = "backwash"


class OptimizationPriority(str, Enum):
    """Priority levels for optimization recommendations."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class CondenserDesignData(BaseModel):
    """Design specifications for the condenser."""

    surface_area_m2: float = Field(
        ...,
        gt=0,
        description="Total heat transfer surface area in square meters"
    )
    number_of_tubes: int = Field(
        ...,
        gt=0,
        description="Total number of condenser tubes"
    )
    tube_od_mm: float = Field(
        ...,
        gt=0,
        le=50,
        description="Tube outer diameter in millimeters"
    )
    tube_wall_mm: float = Field(
        ...,
        gt=0,
        le=5,
        description="Tube wall thickness in millimeters"
    )
    tube_length_m: float = Field(
        ...,
        gt=0,
        description="Effective tube length in meters"
    )
    tube_material: TubeMaterial = Field(
        ...,
        description="Tube material type"
    )
    number_of_passes: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of water passes"
    )
    design_heat_duty_kw: float = Field(
        ...,
        gt=0,
        description="Design heat duty in kilowatts"
    )
    design_u_clean: float = Field(
        ...,
        gt=0,
        description="Design clean U coefficient (W/m2-K)"
    )
    design_cleanliness_factor: float = Field(
        default=0.85,
        gt=0,
        le=1.0,
        description="Design cleanliness factor"
    )
    design_ttd_c: float = Field(
        ...,
        gt=0,
        description="Design terminal temperature difference (C)"
    )
    design_pressure_kpa: float = Field(
        ...,
        gt=0,
        description="Design condenser pressure (kPa absolute)"
    )
    cooling_water_source: CoolingWaterSource = Field(
        ...,
        description="Type of cooling water system"
    )

    @validator('tube_wall_mm')
    def validate_wall_thickness(cls, v: float, values: Dict[str, Any]) -> float:
        """Ensure wall thickness is appropriate for tube diameter."""
        od = values.get('tube_od_mm')
        if od and v >= od / 2:
            raise ValueError(
                f"Wall thickness ({v}mm) must be less than half OD ({od/2}mm)"
            )
        return v


class CoolingWaterData(BaseModel):
    """Cooling water operating data."""

    inlet_temp_c: float = Field(
        ...,
        ge=0,
        le=50,
        description="Cooling water inlet temperature (C)"
    )
    outlet_temp_c: float = Field(
        ...,
        ge=0,
        le=60,
        description="Cooling water outlet temperature (C)"
    )
    flow_rate_m3_hr: float = Field(
        ...,
        gt=0,
        description="Cooling water flow rate (m3/hr)"
    )
    velocity_m_s: Optional[float] = Field(
        default=None,
        gt=0,
        le=4.0,
        description="Water velocity in tubes (m/s)"
    )

    @root_validator(skip_on_failure=True)
    def validate_temperatures(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate outlet is greater than inlet."""
        inlet = values.get('inlet_temp_c')
        outlet = values.get('outlet_temp_c')
        if inlet is not None and outlet is not None:
            if outlet <= inlet:
                raise ValueError(
                    f"Outlet temp ({outlet}C) must be greater than inlet ({inlet}C)"
                )
        return values


class VacuumData(BaseModel):
    """Condenser vacuum/pressure data."""

    absolute_pressure_kpa: float = Field(
        ...,
        gt=0,
        lt=20,
        description="Absolute condenser pressure (kPa)"
    )
    saturation_temp_c: Optional[float] = Field(
        default=None,
        description="Steam saturation temperature (C)"
    )
    vacuum_in_hg: Optional[float] = Field(
        default=None,
        ge=0,
        le=30,
        description="Vacuum reading (inches Hg)"
    )
    barometric_pressure_kpa: float = Field(
        default=101.325,
        gt=90,
        lt=110,
        description="Local barometric pressure (kPa)"
    )


class AirLeakageData(BaseModel):
    """Air in-leakage measurement data."""

    measured_leakage_kg_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured air in-leakage rate (kg/hr)"
    )
    dissolved_oxygen_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Dissolved oxygen in condensate (ppb)"
    )
    air_removal_suction_temp_c: Optional[float] = Field(
        default=None,
        description="Air removal system suction temperature (C)"
    )


class TubeFoulingData(BaseModel):
    """Tube fouling measurement and history data."""

    current_u_coefficient: Optional[float] = Field(
        default=None,
        gt=0,
        description="Current overall U coefficient (W/m2-K)"
    )
    fouling_resistance_m2k_w: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured fouling resistance (m2-K/W)"
    )
    hours_since_cleaning: float = Field(
        default=0,
        ge=0,
        description="Operating hours since last tube cleaning"
    )
    last_cleaning_date: Optional[date] = Field(
        default=None,
        description="Date of last tube cleaning"
    )
    dominant_fouling_mechanism: Optional[FoulingMechanism] = Field(
        default=None,
        description="Dominant fouling mechanism if known"
    )
    fouling_history: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="Historical fouling data [(hours, resistance), ...]"
    )


class CondenserInput(BaseModel):
    """Complete input data model for CondenserOptimizationAgent."""

    condenser_id: str = Field(
        ...,
        min_length=1,
        description="Unique condenser identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of measurements"
    )

    # Design data
    design: CondenserDesignData = Field(
        ...,
        description="Condenser design specifications"
    )

    # Operating data
    cooling_water: CoolingWaterData = Field(
        ...,
        description="Cooling water measurements"
    )
    vacuum: VacuumData = Field(
        ...,
        description="Vacuum/pressure measurements"
    )

    # Optional detailed data
    air_leakage: Optional[AirLeakageData] = Field(
        default=None,
        description="Air in-leakage data if available"
    )
    fouling: Optional[TubeFoulingData] = Field(
        default=None,
        description="Tube fouling data if available"
    )

    # Operational context
    steam_flow_kg_hr: Optional[float] = Field(
        default=None,
        gt=0,
        description="Steam/condensate flow rate (kg/hr)"
    )
    turbine_power_mw: Optional[float] = Field(
        default=None,
        gt=0,
        description="Associated turbine power output (MW)"
    )
    operating_hours_per_day: float = Field(
        default=24,
        gt=0,
        le=24,
        description="Average daily operating hours"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "condenser_id": "COND-001",
                "design": {
                    "surface_area_m2": 5000,
                    "number_of_tubes": 10000,
                    "tube_od_mm": 25.4,
                    "tube_wall_mm": 1.245,
                    "tube_length_m": 8.0,
                    "tube_material": "admiralty_brass",
                    "design_heat_duty_kw": 250000,
                    "design_u_clean": 3500,
                    "design_ttd_c": 3.0,
                    "design_pressure_kpa": 5.0,
                    "cooling_water_source": "cooling_tower_treated",
                },
                "cooling_water": {
                    "inlet_temp_c": 25.0,
                    "outlet_temp_c": 35.0,
                    "flow_rate_m3_hr": 25000,
                },
                "vacuum": {
                    "absolute_pressure_kpa": 7.0,
                },
            }
        }


class HeatTransferAnalysis(BaseModel):
    """Heat transfer analysis results."""

    calculated_u_actual: float = Field(
        ...,
        description="Calculated actual U coefficient (W/m2-K)"
    )
    u_clean: float = Field(
        ...,
        description="Clean U coefficient (W/m2-K)"
    )
    lmtd: float = Field(
        ...,
        description="Log mean temperature difference (C)"
    )
    heat_duty_kw: float = Field(
        ...,
        description="Calculated heat duty (kW)"
    )
    ttd_c: float = Field(
        ...,
        description="Terminal temperature difference (C)"
    )
    cooling_water_rise_c: float = Field(
        ...,
        description="Cooling water temperature rise (C)"
    )


class CleanlinessAnalysis(BaseModel):
    """Cleanliness factor analysis results."""

    cleanliness_factor: float = Field(
        ...,
        ge=0,
        le=1.1,
        description="HEI cleanliness factor"
    )
    status: str = Field(
        ...,
        description="Cleanliness status (EXCELLENT/GOOD/ACCEPTABLE/MARGINAL/POOR)"
    )
    guidance: str = Field(
        ...,
        description="Status guidance text"
    )
    degradation_rate_per_day: Optional[float] = Field(
        default=None,
        description="CF degradation rate if trend data available"
    )
    projected_cf_30d: Optional[float] = Field(
        default=None,
        description="Projected CF in 30 days"
    )
    days_to_threshold: Optional[float] = Field(
        default=None,
        description="Days until CF reaches marginal threshold"
    )
    cf_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Composite CF score 0-100"
    )


class VacuumAnalysis(BaseModel):
    """Vacuum system analysis results."""

    theoretical_pressure_kpa: float = Field(
        ...,
        description="Theoretical minimum pressure (kPa)"
    )
    actual_pressure_kpa: float = Field(
        ...,
        description="Actual condenser pressure (kPa)"
    )
    pressure_deviation_kpa: float = Field(
        ...,
        description="Deviation from theoretical (kPa)"
    )
    vacuum_efficiency_pct: float = Field(
        ...,
        description="Vacuum efficiency percentage"
    )
    vacuum_in_hg: float = Field(
        ...,
        description="Vacuum in inches Hg"
    )
    status: str = Field(
        ...,
        description="Vacuum status (EXCELLENT/GOOD/MARGINAL/POOR)"
    )
    power_loss_mw: Optional[float] = Field(
        default=None,
        description="Estimated power loss from vacuum degradation (MW)"
    )


class FoulingAnalysis(BaseModel):
    """Fouling analysis results."""

    fouling_resistance_m2k_w: float = Field(
        ...,
        ge=0,
        description="Total fouling resistance (m2-K/W)"
    )
    design_fouling_m2k_w: float = Field(
        ...,
        description="Design fouling resistance (m2-K/W)"
    )
    fouling_ratio: float = Field(
        ...,
        description="Ratio of actual to design fouling"
    )
    severity: str = Field(
        ...,
        description="Fouling severity (CLEAN/LIGHT/MODERATE/HEAVY/SEVERE)"
    )
    fouling_rate_per_1000h: Optional[float] = Field(
        default=None,
        description="Fouling accumulation rate (m2-K/W per 1000h)"
    )
    hours_to_threshold: Optional[float] = Field(
        default=None,
        description="Hours until cleaning threshold"
    )
    dominant_mechanism: Optional[str] = Field(
        default=None,
        description="Dominant fouling mechanism"
    )


class AirLeakageAnalysis(BaseModel):
    """Air in-leakage analysis results."""

    measured_leakage_kg_hr: Optional[float] = Field(
        default=None,
        description="Measured air leakage (kg/hr)"
    )
    normalized_leakage: Optional[float] = Field(
        default=None,
        description="Normalized leakage (kg/hr per kW)"
    )
    severity: str = Field(
        ...,
        description="Leakage severity (EXCELLENT/ACCEPTABLE/HIGH/CRITICAL)"
    )
    hei_limit_kg_hr: float = Field(
        ...,
        description="HEI standard limit for this condenser (kg/hr)"
    )


class OptimizationRecommendation(BaseModel):
    """Individual optimization recommendation."""

    action: str = Field(
        ...,
        description="Recommended action"
    )
    priority: OptimizationPriority = Field(
        ...,
        description="Priority level"
    )
    reason: str = Field(
        ...,
        description="Reason for recommendation"
    )
    category: str = Field(
        ...,
        description="Category (vacuum/fouling/cleaning/air_leakage)"
    )
    estimated_benefit: Optional[str] = Field(
        default=None,
        description="Estimated benefit if implemented"
    )


class CleaningSchedule(BaseModel):
    """Recommended cleaning schedule."""

    recommended_date: date = Field(
        ...,
        description="Recommended cleaning date"
    )
    urgency: str = Field(
        ...,
        description="Urgency level"
    )
    method: CleaningMethod = Field(
        ...,
        description="Recommended cleaning method"
    )
    expected_cf_after: float = Field(
        ...,
        description="Expected CF after cleaning"
    )
    estimated_power_recovery_mw: Optional[float] = Field(
        default=None,
        description="Estimated power recovery from cleaning (MW)"
    )


class ExplainabilityReport(BaseModel):
    """SHAP/LIME-style explainability for optimization decisions."""

    feature_importance: Dict[str, float] = Field(
        ...,
        description="Feature importance scores for key metrics"
    )
    key_drivers: List[str] = Field(
        ...,
        description="Top factors driving current performance"
    )
    optimization_rationale: str = Field(
        ...,
        description="Human-readable explanation of optimization logic"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Confidence in analysis (0-1)"
    )


class CondenserOutput(BaseModel):
    """Complete output data model for CondenserOptimizationAgent."""

    condenser_id: str = Field(
        ...,
        description="Condenser identifier from input"
    )
    assessment_timestamp: datetime = Field(
        ...,
        description="Timestamp of assessment"
    )

    # Analysis results
    heat_transfer: HeatTransferAnalysis = Field(
        ...,
        description="Heat transfer analysis"
    )
    cleanliness: CleanlinessAnalysis = Field(
        ...,
        description="Cleanliness factor analysis"
    )
    vacuum: VacuumAnalysis = Field(
        ...,
        description="Vacuum system analysis"
    )
    fouling: FoulingAnalysis = Field(
        ...,
        description="Fouling analysis"
    )
    air_leakage: Optional[AirLeakageAnalysis] = Field(
        default=None,
        description="Air in-leakage analysis if data provided"
    )

    # Performance summary
    overall_efficiency_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall condenser efficiency score 0-100"
    )
    performance_status: str = Field(
        ...,
        description="Overall status (OPTIMAL/GOOD/DEGRADED/POOR/CRITICAL)"
    )

    # Optimization outputs
    recommendations: List[OptimizationRecommendation] = Field(
        ...,
        description="Prioritized optimization recommendations"
    )
    cleaning_schedule: Optional[CleaningSchedule] = Field(
        default=None,
        description="Recommended cleaning schedule"
    )
    explainability: ExplainabilityReport = Field(
        ...,
        description="Explainability report for decisions"
    )

    # Provenance and audit
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
    agent_version: str = Field(
        ...,
        description="Agent version that produced this output"
    )
    calculation_method: str = Field(
        default="HEI_STANDARDS",
        description="Calculation methodology used"
    )


class AgentConfig(BaseModel):
    """Configuration for CondenserOptimizationAgent."""

    agent_id: str = Field(
        default="GL-017",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="CONDENSYNC",
        description="Agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    cf_warning_threshold: float = Field(
        default=0.80,
        ge=0.5,
        le=1.0,
        description="CF threshold for warning status"
    )
    cf_critical_threshold: float = Field(
        default=0.70,
        ge=0.5,
        le=1.0,
        description="CF threshold for critical status"
    )
    vacuum_deviation_warning_kpa: float = Field(
        default=1.0,
        ge=0,
        description="Vacuum deviation for warning (kPa)"
    )
    vacuum_deviation_critical_kpa: float = Field(
        default=2.0,
        ge=0,
        description="Vacuum deviation for critical (kPa)"
    )
    fouling_threshold_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        description="Multiple of design fouling for cleaning trigger"
    )
    enable_fouling_prediction: bool = Field(
        default=True,
        description="Enable fouling trend prediction"
    )
    enable_explainability: bool = Field(
        default=True,
        description="Enable SHAP/LIME explainability reports"
    )
