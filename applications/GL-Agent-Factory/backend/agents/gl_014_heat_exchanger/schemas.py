"""
Pydantic Schemas for GL-014 EXCHANGERPRO Agent

This module defines the input/output data models for the HeatExchangerOptimizerAgent
following GreenLang schema standards with comprehensive validation.

All models include:
- Type hints and field descriptions
- Validators for physical constraints
- TEMA-compliant default values
- Provenance tracking fields

Reference Standards:
- TEMA Standards (10th Edition)
- ASME Heat Exchanger Design Handbook
- API 660 Shell-and-Tube Heat Exchangers
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator, root_validator

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class FlowArrangement(str, Enum):
    """Heat exchanger flow arrangements."""
    COUNTERFLOW = "counterflow"
    PARALLEL_FLOW = "parallel_flow"
    SHELL_AND_TUBE_1_2 = "shell_and_tube_1_2"
    SHELL_AND_TUBE_2_4 = "shell_and_tube_2_4"
    CROSSFLOW_UNMIXED = "crossflow_unmixed"
    CROSSFLOW_MIXED_CMAX = "crossflow_mixed_cmax"
    CROSSFLOW_MIXED_CMIN = "crossflow_mixed_cmin"
    CROSSFLOW_BOTH_MIXED = "crossflow_both_mixed"


class ExchangerType(str, Enum):
    """TEMA heat exchanger types."""
    AES = "AES"  # Front head / Shell / Rear head
    AEL = "AEL"
    AEM = "AEM"
    AEP = "AEP"
    AET = "AET"
    AEU = "AEU"
    AEW = "AEW"
    AKT = "AKT"
    BEM = "BEM"
    BEU = "BEU"
    BKU = "BKU"
    CFU = "CFU"
    NEN = "NEN"
    # Generic types
    SHELL_AND_TUBE = "shell_and_tube"
    PLATE = "plate"
    AIR_COOLED = "air_cooled"
    DOUBLE_PIPE = "double_pipe"
    SPIRAL = "spiral"


class FluidCategory(str, Enum):
    """Fluid categories for fouling resistance."""
    WATER = "water"
    STEAM = "steam"
    GASES = "gases"
    LIQUIDS = "liquids"
    CHEMICALS = "chemicals"


class FoulingMechanism(str, Enum):
    """Types of fouling mechanisms."""
    PARTICULATE = "particulate"
    PRECIPITATION = "precipitation"
    CORROSION = "corrosion"
    BIOLOGICAL = "biological"
    COKING = "coking"


class MaintenanceUrgency(str, Enum):
    """Maintenance urgency levels."""
    IMMEDIATE = "IMMEDIATE"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class FoulingStatus(str, Enum):
    """Fouling condition status."""
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


# =============================================================================
# Input Models
# =============================================================================

class FluidProperties(BaseModel):
    """Physical properties of a fluid stream."""

    name: str = Field(..., description="Fluid name/identifier")
    category: FluidCategory = Field(
        default=FluidCategory.LIQUIDS,
        description="Fluid category for fouling estimation"
    )
    fluid_type: str = Field(
        default="process_fluids",
        description="Specific fluid type for TEMA fouling lookup"
    )

    # Thermal properties
    specific_heat_j_kg_k: float = Field(
        ...,
        gt=0,
        description="Specific heat capacity (J/kg-K)"
    )
    density_kg_m3: float = Field(
        ...,
        gt=0,
        description="Density (kg/m3)"
    )
    thermal_conductivity_w_m_k: float = Field(
        default=0.6,
        gt=0,
        description="Thermal conductivity (W/m-K)"
    )
    viscosity_pa_s: float = Field(
        default=0.001,
        gt=0,
        description="Dynamic viscosity (Pa-s)"
    )

    # Optional properties for advanced analysis
    ph: Optional[float] = Field(
        default=None,
        ge=0,
        le=14,
        description="pH for corrosion analysis"
    )
    fouling_tendency: Optional[str] = Field(
        default=None,
        description="Qualitative fouling tendency (low/medium/high)"
    )

    @validator('specific_heat_j_kg_k')
    def validate_cp(cls, v: float) -> float:
        """Validate specific heat is reasonable."""
        if v < 100 or v > 10000:
            logger.warning(f"Specific heat {v} J/kg-K is outside typical range")
        return v


class StreamData(BaseModel):
    """Operating data for a fluid stream (hot or cold side)."""

    # Temperature data
    inlet_temperature_c: float = Field(
        ...,
        description="Inlet temperature (Celsius)"
    )
    outlet_temperature_c: Optional[float] = Field(
        default=None,
        description="Outlet temperature (Celsius) - optional for sizing"
    )

    # Flow data
    mass_flow_kg_s: float = Field(
        ...,
        gt=0,
        description="Mass flow rate (kg/s)"
    )

    # Pressure data
    inlet_pressure_bar: float = Field(
        default=1.0,
        gt=0,
        description="Inlet pressure (bar absolute)"
    )
    pressure_drop_bar: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured/allowable pressure drop (bar)"
    )

    # Fluid properties
    fluid: FluidProperties = Field(
        ...,
        description="Fluid physical properties"
    )

    # Fouling
    fouling_resistance_m2_k_w: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fouling resistance (m2-K/W). If None, TEMA default used."
    )

    @root_validator(skip_on_failure=True)
    def validate_temperatures(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate temperature relationships."""
        t_in = values.get('inlet_temperature_c')
        t_out = values.get('outlet_temperature_c')

        if t_out is not None and t_in is not None:
            if abs(t_in - t_out) > 300:
                logger.warning(
                    f"Large temperature change ({abs(t_in-t_out):.0f}C) - verify data"
                )

        return values

    def heat_capacity_rate(self) -> float:
        """Calculate heat capacity rate C = m_dot * cp."""
        return self.mass_flow_kg_s * self.fluid.specific_heat_j_kg_k


class CleaningHistoryEntry(BaseModel):
    """Historical cleaning event record."""

    cleaning_date: date = Field(..., description="Date of cleaning")
    cleaning_type: str = Field(
        default="chemical",
        description="Type of cleaning (chemical/mechanical/hydroblast)"
    )
    pre_cleaning_ua: Optional[float] = Field(
        default=None,
        gt=0,
        description="UA before cleaning (W/K)"
    )
    post_cleaning_ua: Optional[float] = Field(
        default=None,
        gt=0,
        description="UA after cleaning (W/K)"
    )
    cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cleaning cost ($)"
    )
    downtime_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Downtime for cleaning (hours)"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class ExchangerGeometry(BaseModel):
    """Heat exchanger geometric specifications."""

    exchanger_type: ExchangerType = Field(
        default=ExchangerType.SHELL_AND_TUBE,
        description="TEMA exchanger type designation"
    )
    flow_arrangement: FlowArrangement = Field(
        default=FlowArrangement.SHELL_AND_TUBE_1_2,
        description="Flow arrangement type"
    )

    # Area
    heat_transfer_area_m2: float = Field(
        ...,
        gt=0,
        description="Heat transfer area (m2)"
    )

    # Shell-and-tube specific
    shell_diameter_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Shell inside diameter (m)"
    )
    tube_od_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Tube outside diameter (m)"
    )
    tube_length_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Tube length (m)"
    )
    number_of_tubes: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of tubes"
    )
    tube_passes: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Number of tube passes"
    )
    shell_passes: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of shell passes"
    )
    baffle_cut_percent: Optional[float] = Field(
        default=25,
        ge=15,
        le=45,
        description="Baffle cut percentage"
    )

    # Heat transfer coefficients (if known)
    h_shell_w_m2_k: Optional[float] = Field(
        default=None,
        gt=0,
        description="Shell-side heat transfer coefficient (W/m2-K)"
    )
    h_tube_w_m2_k: Optional[float] = Field(
        default=None,
        gt=0,
        description="Tube-side heat transfer coefficient (W/m2-K)"
    )
    wall_resistance_m2_k_w: Optional[float] = Field(
        default=0.0001,
        ge=0,
        description="Tube wall thermal resistance (m2-K/W)"
    )


class HeatExchangerInput(BaseModel):
    """Complete input data model for HeatExchangerOptimizerAgent."""

    # Identification
    exchanger_id: str = Field(
        ...,
        min_length=1,
        description="Unique heat exchanger identifier"
    )
    exchanger_name: Optional[str] = Field(
        default=None,
        description="Human-readable name"
    )
    location: Optional[str] = Field(
        default=None,
        description="Physical location/unit"
    )

    # Operating conditions
    hot_side: StreamData = Field(
        ...,
        description="Hot side stream data"
    )
    cold_side: StreamData = Field(
        ...,
        description="Cold side stream data"
    )

    # Equipment data
    geometry: ExchangerGeometry = Field(
        ...,
        description="Exchanger geometric specifications"
    )

    # Design/reference values
    ua_design_w_k: Optional[float] = Field(
        default=None,
        gt=0,
        description="Design UA value (W/K)"
    )
    ua_clean_w_k: Optional[float] = Field(
        default=None,
        gt=0,
        description="Clean UA value (W/K) - measured after last cleaning"
    )

    # Current performance
    ua_current_w_k: Optional[float] = Field(
        default=None,
        gt=0,
        description="Current measured UA value (W/K)"
    )

    # Operating history
    operating_hours_since_cleaning: float = Field(
        default=0,
        ge=0,
        description="Operating hours since last cleaning"
    )
    total_operating_hours: float = Field(
        default=0,
        ge=0,
        description="Total operating hours since installation"
    )
    installation_date: Optional[date] = Field(
        default=None,
        description="Equipment installation date"
    )

    # Cleaning history
    cleaning_history: List[CleaningHistoryEntry] = Field(
        default_factory=list,
        description="Historical cleaning records"
    )
    last_cleaning_date: Optional[date] = Field(
        default=None,
        description="Date of last cleaning"
    )

    # Economic parameters
    energy_cost_per_kwh: float = Field(
        default=0.10,
        gt=0,
        description="Energy cost ($/kWh)"
    )
    cleaning_cost: float = Field(
        default=5000,
        ge=0,
        description="Estimated cleaning cost ($)"
    )
    operating_hours_per_year: float = Field(
        default=8000,
        gt=0,
        le=8760,
        description="Annual operating hours"
    )

    @root_validator(skip_on_failure=True)
    def validate_temperatures(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hot/cold temperature relationships."""
        hot = values.get('hot_side')
        cold = values.get('cold_side')

        if hot and cold:
            if hot.inlet_temperature_c <= cold.inlet_temperature_c:
                logger.warning(
                    f"Hot inlet ({hot.inlet_temperature_c}C) <= cold inlet "
                    f"({cold.inlet_temperature_c}C)"
                )

        return values

    @validator('ua_current_w_k')
    def validate_ua_current(cls, v: Optional[float], values: Dict) -> Optional[float]:
        """Validate current UA against clean UA."""
        ua_clean = values.get('ua_clean_w_k')
        if v is not None and ua_clean is not None:
            if v > ua_clean * 1.1:  # 10% tolerance
                logger.warning(
                    f"Current UA ({v}) exceeds clean UA ({ua_clean}) - check data"
                )
        return v


# =============================================================================
# Output Models
# =============================================================================

class LMTDAnalysis(BaseModel):
    """LMTD calculation results."""

    lmtd_counterflow: float = Field(
        ...,
        description="LMTD assuming counterflow (K)"
    )
    f_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="LMTD correction factor"
    )
    lmtd_corrected: float = Field(
        ...,
        description="Corrected LMTD = F * LMTD_cf (K)"
    )
    p_parameter: float = Field(
        ...,
        description="Temperature effectiveness P"
    )
    r_parameter: float = Field(
        ...,
        description="Heat capacity ratio R"
    )


class EffectivenessAnalysis(BaseModel):
    """Epsilon-NTU analysis results."""

    ntu: float = Field(
        ...,
        ge=0,
        description="Number of Transfer Units"
    )
    effectiveness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Heat exchanger effectiveness (epsilon)"
    )
    capacity_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Capacity ratio Cr = Cmin/Cmax"
    )
    c_min_w_k: float = Field(
        ...,
        gt=0,
        description="Minimum heat capacity rate (W/K)"
    )
    c_max_w_k: float = Field(
        ...,
        gt=0,
        description="Maximum heat capacity rate (W/K)"
    )


class UADegradationAnalysis(BaseModel):
    """UA degradation analysis results."""

    ua_clean: float = Field(
        ...,
        gt=0,
        description="Clean UA value (W/K)"
    )
    ua_current: float = Field(
        ...,
        gt=0,
        description="Current UA value (W/K)"
    )
    ua_reduction_percent: float = Field(
        ...,
        ge=0,
        description="Percentage reduction in UA"
    )
    fouling_factor: float = Field(
        ...,
        gt=0,
        le=1,
        description="UA_current / UA_clean"
    )
    total_fouling_resistance: float = Field(
        ...,
        ge=0,
        description="Total fouling resistance (m2-K/W)"
    )


class FoulingPrediction(BaseModel):
    """Fouling prediction results."""

    current_rf: float = Field(
        ...,
        ge=0,
        description="Current fouling resistance (m2-K/W)"
    )
    fouling_rate: float = Field(
        ...,
        ge=0,
        description="Fouling rate (m2-K/W per 1000 hours)"
    )
    hours_to_critical: float = Field(
        ...,
        ge=0,
        description="Hours until UA drops to 70% of clean"
    )
    predicted_rf_1000h: float = Field(
        ...,
        ge=0,
        description="Predicted Rf in 1000 hours"
    )
    predicted_rf_5000h: float = Field(
        ...,
        ge=0,
        description="Predicted Rf in 5000 hours"
    )


class CleaningScheduleRecommendation(BaseModel):
    """Cleaning schedule optimization results."""

    optimal_interval_hours: float = Field(
        ...,
        gt=0,
        description="Optimal cleaning interval (hours)"
    )
    optimal_interval_days: float = Field(
        ...,
        gt=0,
        description="Optimal cleaning interval (days)"
    )
    next_cleaning_date: date = Field(
        ...,
        description="Recommended next cleaning date"
    )
    days_until_cleaning: float = Field(
        ...,
        description="Days until next cleaning"
    )
    cleanings_per_year: float = Field(
        ...,
        gt=0,
        description="Estimated cleanings per year"
    )
    annual_cleaning_cost: float = Field(
        ...,
        ge=0,
        description="Annual cleaning cost ($)"
    )
    annual_energy_loss: float = Field(
        ...,
        ge=0,
        description="Annual energy loss cost ($)"
    )
    total_annual_cost: float = Field(
        ...,
        ge=0,
        description="Total annual cost ($)"
    )


class EfficiencyGains(BaseModel):
    """Efficiency improvement potential."""

    current_efficiency_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current thermal efficiency (%)"
    )
    potential_efficiency_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Potential efficiency after cleaning (%)"
    )
    efficiency_gain_percent: float = Field(
        ...,
        ge=0,
        description="Efficiency improvement from cleaning (%)"
    )
    annual_energy_savings_kwh: float = Field(
        ...,
        ge=0,
        description="Annual energy savings (kWh)"
    )
    annual_cost_savings: float = Field(
        ...,
        ge=0,
        description="Annual cost savings ($)"
    )
    payback_hours: float = Field(
        ...,
        ge=0,
        description="Cleaning payback time (hours)"
    )


class ExplainabilityReport(BaseModel):
    """SHAP/LIME-style explainability for recommendations."""

    primary_factors: List[Dict[str, Any]] = Field(
        ...,
        description="Top factors influencing recommendations"
    )
    sensitivity_analysis: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameter sensitivity analysis"
    )
    confidence_level: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence level of recommendations"
    )
    data_quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Input data quality score"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Key assumptions made in analysis"
    )


class OptimizationRecommendation(BaseModel):
    """Individual optimization recommendation."""

    action: str = Field(
        ...,
        description="Recommended action"
    )
    urgency: MaintenanceUrgency = Field(
        ...,
        description="Action urgency level"
    )
    expected_benefit: str = Field(
        ...,
        description="Expected benefit from action"
    )
    estimated_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated cost ($)"
    )
    estimated_savings_per_year: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated annual savings ($)"
    )
    payback_months: Optional[float] = Field(
        default=None,
        ge=0,
        description="Payback period (months)"
    )
    rationale: str = Field(
        ...,
        description="Technical rationale for recommendation"
    )


class HeatExchangerOutput(BaseModel):
    """Complete output data model for HeatExchangerOptimizerAgent."""

    # Identification
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    assessment_timestamp: datetime = Field(
        ...,
        description="Timestamp of assessment"
    )

    # Thermal analysis
    heat_duty_w: float = Field(
        ...,
        description="Calculated heat duty (W)"
    )
    lmtd_analysis: LMTDAnalysis = Field(
        ...,
        description="LMTD calculation results"
    )
    effectiveness_analysis: EffectivenessAnalysis = Field(
        ...,
        description="Epsilon-NTU analysis results"
    )

    # Calculated outlet temperatures (if not provided)
    calculated_hot_outlet_c: Optional[float] = Field(
        default=None,
        description="Calculated hot outlet temperature (C)"
    )
    calculated_cold_outlet_c: Optional[float] = Field(
        default=None,
        description="Calculated cold outlet temperature (C)"
    )

    # UA analysis
    ua_degradation: UADegradationAnalysis = Field(
        ...,
        description="UA degradation analysis"
    )

    # Fouling analysis
    fouling_status: FoulingStatus = Field(
        ...,
        description="Current fouling status"
    )
    fouling_prediction: FoulingPrediction = Field(
        ...,
        description="Fouling prediction results"
    )

    # Cleaning optimization
    cleaning_schedule: CleaningScheduleRecommendation = Field(
        ...,
        description="Cleaning schedule optimization"
    )

    # Efficiency analysis
    efficiency_gains: EfficiencyGains = Field(
        ...,
        description="Efficiency improvement potential"
    )

    # Explainability
    explainability: ExplainabilityReport = Field(
        ...,
        description="Analysis explainability report"
    )

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(
        ...,
        description="Prioritized optimization recommendations"
    )

    # Audit and provenance
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
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation error messages if any"
    )


class AgentConfig(BaseModel):
    """Configuration for HeatExchangerOptimizerAgent."""

    agent_id: str = Field(default="GL-014", description="Agent identifier")
    agent_name: str = Field(default="EXCHANGERPRO", description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")

    # Thresholds
    critical_ua_threshold: float = Field(
        default=0.70,
        ge=0.5,
        le=0.95,
        description="UA ratio below which status is CRITICAL"
    )
    poor_ua_threshold: float = Field(
        default=0.80,
        ge=0.6,
        le=0.95,
        description="UA ratio below which status is POOR"
    )
    fair_ua_threshold: float = Field(
        default=0.90,
        ge=0.7,
        le=0.98,
        description="UA ratio below which status is FAIR"
    )

    # Analysis options
    enable_fouling_prediction: bool = Field(
        default=True,
        description="Enable fouling prediction"
    )
    enable_cleaning_optimization: bool = Field(
        default=True,
        description="Enable cleaning schedule optimization"
    )
    enable_explainability: bool = Field(
        default=True,
        description="Enable SHAP/LIME-style explainability"
    )

    # Default economic parameters
    default_energy_cost: float = Field(
        default=0.10,
        gt=0,
        description="Default energy cost ($/kWh)"
    )
    default_cleaning_cost: float = Field(
        default=5000,
        ge=0,
        description="Default cleaning cost ($)"
    )
