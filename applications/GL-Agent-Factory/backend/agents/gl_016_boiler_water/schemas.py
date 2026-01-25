"""
Pydantic Schemas for GL-016 WATERGUARD (BoilerWaterTreatmentAgent)

This module defines all input/output data models for boiler water chemistry
optimization following ASME/ABMA standards.

All models use strict validation to ensure data quality for zero-hallucination
calculations in the water treatment optimization pipeline.

Standards Reference:
    - ASME Boiler and Pressure Vessel Code Section VII
    - ABMA (American Boiler Manufacturers Association) Guidelines
    - EPRI Water Chemistry Guidelines for Fossil Plants

Example:
    >>> from schemas import WaterChemistryInput, WaterTreatmentOutput
    >>> input_data = WaterChemistryInput(...)
    >>> # Process with agent
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator, root_validator


class BoilerPressureClass(str, Enum):
    """ASME pressure class for boiler water limits determination."""
    LOW_PRESSURE = "low_pressure"  # < 300 psig
    MEDIUM_PRESSURE = "medium_pressure"  # 300-600 psig
    HIGH_PRESSURE = "high_pressure"  # 600-900 psig
    VERY_HIGH_PRESSURE = "very_high_pressure"  # 900-1500 psig
    SUPERCRITICAL = "supercritical"  # > 1500 psig


class ChemicalType(str, Enum):
    """Types of water treatment chemicals."""
    OXYGEN_SCAVENGER = "oxygen_scavenger"
    PHOSPHATE = "phosphate"
    CAUSTIC_SODA = "caustic_soda"
    AMINE = "amine"
    POLYMER = "polymer"
    SULFITE = "sulfite"
    HYDRAZINE = "hydrazine"
    CHELANT = "chelant"
    DISPERSANT = "dispersant"


class ComplianceStatus(str, Enum):
    """Water chemistry compliance status."""
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    NON_COMPLIANT = "NON_COMPLIANT"
    CRITICAL = "CRITICAL"


class DosingPriority(str, Enum):
    """Chemical dosing priority levels."""
    IMMEDIATE = "IMMEDIATE"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    NONE = "NONE"


class WaterChemistryData(BaseModel):
    """Real-time water chemistry measurements."""

    conductivity_us_cm: float = Field(
        ...,
        ge=0,
        le=10000,
        description="Specific conductivity in microsiemens/cm"
    )
    ph: float = Field(
        ...,
        ge=0,
        le=14,
        description="pH value (0-14 scale)"
    )
    alkalinity_ppm_caco3: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Total alkalinity as ppm CaCO3"
    )
    silica_ppm: float = Field(
        ...,
        ge=0,
        le=500,
        description="Silica concentration in ppm"
    )
    total_hardness_ppm: float = Field(
        ...,
        ge=0,
        le=500,
        description="Total hardness as ppm CaCO3"
    )
    iron_ppm: float = Field(
        ...,
        ge=0,
        le=50,
        description="Iron concentration in ppm"
    )
    copper_ppm: float = Field(
        ...,
        ge=0,
        le=10,
        description="Copper concentration in ppm"
    )
    dissolved_oxygen_ppb: float = Field(
        ...,
        ge=0,
        le=10000,
        description="Dissolved oxygen in ppb"
    )
    phosphate_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="Phosphate residual in ppm (if phosphate program)"
    )
    sulfite_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Sulfite residual in ppm (if sulfite program)"
    )
    tds_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=5000,
        description="Total dissolved solids in ppm"
    )

    @validator('ph')
    def validate_ph_range(cls, v: float) -> float:
        """Validate pH is within measurable range."""
        if v < 2 or v > 12:
            import logging
            logging.getLogger(__name__).warning(
                f"pH value {v} is outside typical boiler water range [9-12]"
            )
        return v

    @root_validator(skip_on_failure=True)
    def validate_chemistry_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that chemistry values are internally consistent."""
        conductivity = values.get('conductivity_us_cm', 0)
        tds = values.get('tds_ppm')

        # TDS is approximately 0.5-0.7 * conductivity
        if tds is not None and conductivity > 0:
            ratio = tds / conductivity
            if ratio < 0.3 or ratio > 1.0:
                import logging
                logging.getLogger(__name__).warning(
                    f"TDS/Conductivity ratio {ratio:.2f} is outside typical range [0.5-0.7]"
                )

        return values


class FeedwaterQuality(BaseModel):
    """Feedwater (makeup water) quality data."""

    conductivity_us_cm: float = Field(
        ...,
        ge=0,
        le=2000,
        description="Feedwater conductivity in microsiemens/cm"
    )
    hardness_ppm: float = Field(
        ...,
        ge=0,
        le=500,
        description="Feedwater hardness as ppm CaCO3"
    )
    silica_ppm: float = Field(
        ...,
        ge=0,
        le=100,
        description="Feedwater silica in ppm"
    )
    dissolved_oxygen_ppb: float = Field(
        ...,
        ge=0,
        le=20000,
        description="Feedwater dissolved oxygen in ppb"
    )
    ph: float = Field(
        ...,
        ge=5,
        le=10,
        description="Feedwater pH"
    )
    iron_ppm: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Feedwater iron in ppm"
    )
    tds_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Feedwater TDS in ppm"
    )
    temperature_c: Optional[float] = Field(
        default=25.0,
        ge=0,
        le=150,
        description="Feedwater temperature in Celsius"
    )


class ChemicalInventory(BaseModel):
    """Chemical inventory and tank levels."""

    chemical_type: ChemicalType = Field(..., description="Type of chemical")
    tank_level_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current tank level percentage"
    )
    concentration_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Chemical concentration percentage"
    )
    days_supply_remaining: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated days of supply remaining"
    )
    cost_per_gallon: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cost per gallon in currency units"
    )


class OperatingParameters(BaseModel):
    """Boiler operating parameters."""

    operating_pressure_psig: float = Field(
        ...,
        ge=0,
        le=4000,
        description="Operating pressure in PSIG"
    )
    steam_production_rate_lb_hr: float = Field(
        ...,
        ge=0,
        description="Steam production rate in lb/hr"
    )
    feedwater_flow_gpm: float = Field(
        ...,
        ge=0,
        description="Feedwater flow rate in GPM"
    )
    blowdown_rate_percent: float = Field(
        default=0,
        ge=0,
        le=25,
        description="Current blowdown rate as percentage of steam"
    )
    condensate_return_percent: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Percentage of condensate being returned"
    )
    boiler_efficiency: Optional[float] = Field(
        default=80.0,
        ge=50,
        le=100,
        description="Boiler thermal efficiency percentage"
    )
    deaerator_pressure_psig: Optional[float] = Field(
        default=None,
        ge=0,
        description="Deaerator operating pressure in PSIG"
    )


class WaterTreatmentInput(BaseModel):
    """Complete input data for BoilerWaterTreatmentAgent."""

    # Identification
    boiler_id: str = Field(..., min_length=1, description="Unique boiler identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Measurement timestamp"
    )

    # Water chemistry data
    boiler_water_chemistry: WaterChemistryData = Field(
        ...,
        description="Current boiler water chemistry measurements"
    )
    feedwater_quality: FeedwaterQuality = Field(
        ...,
        description="Feedwater quality data"
    )

    # Operating parameters
    operating_parameters: OperatingParameters = Field(
        ...,
        description="Boiler operating parameters"
    )

    # Chemical inventory
    chemical_inventory: List[ChemicalInventory] = Field(
        default_factory=list,
        description="Chemical tank inventory"
    )

    # Historical data for trend analysis
    chemistry_history: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Historical chemistry readings (most recent last)"
    )
    history_interval_hours: float = Field(
        default=4,
        gt=0,
        description="Time interval between historical readings"
    )

    # Cost parameters
    water_cost_per_1000_gal: float = Field(
        default=5.0,
        ge=0,
        description="Water cost per 1000 gallons"
    )
    fuel_cost_per_mmbtu: float = Field(
        default=10.0,
        ge=0,
        description="Fuel cost per MMBtu"
    )

    @validator('operating_parameters')
    def validate_pressure_class(cls, v: OperatingParameters) -> OperatingParameters:
        """Validate operating pressure is within safe range."""
        if v.operating_pressure_psig > 3000:
            import logging
            logging.getLogger(__name__).warning(
                f"Operating pressure {v.operating_pressure_psig} PSIG is very high"
            )
        return v


class ChemistryLimitResult(BaseModel):
    """Result of chemistry limit check against ASME/ABMA standards."""

    parameter_name: str = Field(..., description="Chemistry parameter name")
    measured_value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    lower_limit: Optional[float] = Field(default=None, description="Lower limit")
    upper_limit: Optional[float] = Field(default=None, description="Upper limit")
    target_value: Optional[float] = Field(default=None, description="Target value")
    status: ComplianceStatus = Field(..., description="Compliance status")
    deviation_percent: float = Field(..., description="Deviation from target/limit")
    recommendation: Optional[str] = Field(default=None, description="Corrective action")


class BlowdownRecommendation(BaseModel):
    """Blowdown optimization recommendation."""

    current_rate_percent: float = Field(
        ...,
        ge=0,
        le=25,
        description="Current blowdown rate percentage"
    )
    optimal_rate_percent: float = Field(
        ...,
        ge=0,
        le=25,
        description="Recommended optimal blowdown rate"
    )
    cycles_of_concentration: float = Field(
        ...,
        ge=1,
        description="Calculated cycles of concentration"
    )
    optimal_cycles: float = Field(
        ...,
        ge=1,
        description="Optimal cycles of concentration"
    )
    water_savings_gpy: float = Field(
        ...,
        ge=0,
        description="Potential water savings (gallons per year)"
    )
    energy_savings_mmbtu_year: float = Field(
        ...,
        ge=0,
        description="Potential energy savings (MMBtu per year)"
    )
    cost_savings_per_year: float = Field(
        ...,
        ge=0,
        description="Total cost savings potential per year"
    )
    adjustment_action: str = Field(..., description="Recommended adjustment action")


class DosingRecommendation(BaseModel):
    """Chemical dosing recommendation."""

    chemical_type: ChemicalType = Field(..., description="Chemical type")
    current_dose_rate_gph: float = Field(
        ...,
        ge=0,
        description="Current dosing rate in gallons per hour"
    )
    recommended_dose_rate_gph: float = Field(
        ...,
        ge=0,
        description="Recommended dosing rate in gallons per hour"
    )
    target_residual: float = Field(..., description="Target residual level")
    current_residual: Optional[float] = Field(
        default=None,
        description="Current residual level"
    )
    priority: DosingPriority = Field(..., description="Dosing adjustment priority")
    reason: str = Field(..., description="Reason for recommendation")
    estimated_daily_cost_change: float = Field(
        default=0,
        description="Estimated daily cost change from adjustment"
    )


class ChemistryTrend(BaseModel):
    """Chemistry parameter trend analysis."""

    parameter_name: str = Field(..., description="Chemistry parameter name")
    trend_direction: str = Field(
        ...,
        pattern="^(increasing|decreasing|stable)$",
        description="Trend direction"
    )
    rate_of_change_per_hour: float = Field(
        ...,
        description="Rate of change per hour"
    )
    predicted_value_24h: Optional[float] = Field(
        default=None,
        description="Predicted value in 24 hours"
    )
    time_to_limit_hours: Optional[float] = Field(
        default=None,
        description="Hours until limit is reached (if trending toward limit)"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in trend prediction"
    )


class ExplainabilityReport(BaseModel):
    """SHAP/LIME style explainability for recommendations."""

    recommendation_type: str = Field(
        ...,
        description="Type of recommendation being explained"
    )
    feature_contributions: Dict[str, float] = Field(
        ...,
        description="Feature contributions to recommendation (SHAP values)"
    )
    top_factors: List[str] = Field(
        ...,
        description="Top 3-5 factors influencing recommendation"
    )
    calculation_breakdown: Dict[str, Any] = Field(
        ...,
        description="Step-by-step calculation breakdown"
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score for recommendation"
    )
    supporting_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Supporting data for audit trail"
    )


class WaterTreatmentOutput(BaseModel):
    """Complete output from BoilerWaterTreatmentAgent."""

    # Identification
    boiler_id: str = Field(..., description="Boiler identifier from input")
    assessment_timestamp: datetime = Field(..., description="Assessment timestamp")
    pressure_class: BoilerPressureClass = Field(..., description="Determined pressure class")

    # Compliance status
    overall_compliance_status: ComplianceStatus = Field(
        ...,
        description="Overall water chemistry compliance status"
    )
    chemistry_limit_checks: List[ChemistryLimitResult] = Field(
        ...,
        description="Individual parameter compliance checks"
    )
    compliance_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall compliance score (0-100)"
    )

    # Cycles of concentration
    current_cycles: float = Field(..., ge=1, description="Current cycles of concentration")
    optimal_cycles: float = Field(..., ge=1, description="Optimal cycles of concentration")
    max_cycles_by_silica: float = Field(..., ge=1, description="Max cycles limited by silica")
    max_cycles_by_alkalinity: float = Field(..., ge=1, description="Max cycles limited by alkalinity")
    max_cycles_by_conductivity: float = Field(..., ge=1, description="Max cycles limited by conductivity")

    # Blowdown optimization
    blowdown_recommendation: BlowdownRecommendation = Field(
        ...,
        description="Blowdown rate optimization"
    )

    # Chemical dosing recommendations
    dosing_recommendations: List[DosingRecommendation] = Field(
        ...,
        description="Chemical dosing recommendations"
    )

    # Water and cost savings
    water_savings_potential_gpy: float = Field(
        ...,
        ge=0,
        description="Potential water savings (gallons per year)"
    )
    energy_savings_potential_mmbtu: float = Field(
        ...,
        ge=0,
        description="Potential energy savings (MMBtu per year)"
    )
    total_cost_savings_per_year: float = Field(
        ...,
        ge=0,
        description="Total potential cost savings per year"
    )

    # Chemistry trends
    chemistry_trends: List[ChemistryTrend] = Field(
        default_factory=list,
        description="Chemistry parameter trends"
    )

    # Explainability
    explainability_reports: List[ExplainabilityReport] = Field(
        default_factory=list,
        description="SHAP/LIME style explainability reports"
    )

    # Provenance and audit
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in ms")
    validation_status: str = Field(
        ...,
        pattern="^(PASS|FAIL)$",
        description="Output validation status"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages"
    )

    # Standards references
    standards_applied: List[str] = Field(
        default_factory=lambda: ["ASME BPVC Section VII", "ABMA Guidelines"],
        description="Standards used for compliance assessment"
    )


class AgentConfig(BaseModel):
    """Configuration for BoilerWaterTreatmentAgent."""

    agent_id: str = Field(default="GL-016", description="Agent identifier")
    agent_name: str = Field(default="WATERGUARD", description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")

    # Compliance thresholds
    compliance_warning_threshold: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="Threshold for WARNING status (as fraction of limit)"
    )
    compliance_critical_threshold: float = Field(
        default=1.1,
        ge=0,
        description="Threshold for CRITICAL status (as fraction of limit)"
    )

    # Optimization parameters
    min_cycles_of_concentration: float = Field(
        default=3.0,
        ge=1,
        description="Minimum practical cycles of concentration"
    )
    max_blowdown_rate_percent: float = Field(
        default=10.0,
        ge=0,
        le=25,
        description="Maximum allowable blowdown rate"
    )

    # Cost defaults
    default_water_cost: float = Field(
        default=5.0,
        ge=0,
        description="Default water cost per 1000 gallons"
    )
    default_fuel_cost: float = Field(
        default=10.0,
        ge=0,
        description="Default fuel cost per MMBtu"
    )
