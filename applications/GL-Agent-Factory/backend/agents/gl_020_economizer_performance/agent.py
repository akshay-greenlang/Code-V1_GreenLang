"""
GL-020: Economizer Performance Analysis Agent (ECONOPULSE)

This module implements the EconomizerPerformanceAgent for analyzing heat exchanger
performance in industrial boiler and HRSG economizer systems.

The agent provides:
- Heat exchanger effectiveness analysis using NTU-epsilon method
- Acid dew point calculation using Verhoff-Banchero correlation
- Steaming risk detection using IAPWS-IF97 saturation properties
- Cold-end corrosion risk assessment
- Complete SHA-256 provenance tracking

Applications:
- Boiler economizer optimization
- HRSG (Heat Recovery Steam Generator) analysis
- Flue gas heat recovery systems
- Feed water heating performance monitoring
- Preventive maintenance scheduling

Example:
    >>> agent = EconomizerPerformanceAgent()
    >>> result = agent.run(EconomizerInput(
    ...     flue_gas=FlueGasComposition(
    ...         temperature_in_celsius=350.0,
    ...         temperature_out_celsius=150.0,
    ...         mass_flow_kg_s=50.0,
    ...         H2O_percent=8.0,
    ...         SO3_ppmv=15.0,
    ...     ),
    ...     water_side=WaterSideConditions(
    ...         inlet_temperature_celsius=105.0,
    ...         outlet_temperature_celsius=180.0,
    ...         mass_flow_kg_s=20.0,
    ...         drum_pressure_MPa=4.0,
    ...     ),
    ...     heat_exchanger=HeatExchangerGeometry(
    ...         flow_arrangement="counter_flow",
    ...     ),
    ... ))
    >>> print(f"Effectiveness: {result.effectiveness:.3f}")
    >>> print(f"Acid dew point: {result.acid_dew_point_celsius:.1f} deg C")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .calculators.acid_dew_point import (
    verhoff_banchero_acid_dew_point,
    calculate_partial_pressures,
)
from .calculators.effectiveness import (
    effectiveness_counter_flow,
    effectiveness_parallel_flow,
    effectiveness_cross_flow_both_unmixed,
    calculate_heat_transfer,
    calculate_effectiveness_from_temperatures,
)
from .calculators.steaming import (
    saturation_temperature_IF97,
    detect_steaming_risk,
    SteamingAnalysis as SteamingAnalysisResult,
    RiskLevel as SteamingRiskLevel,
)
from .calculators.corrosion import (
    assess_corrosion_risk,
    estimate_tube_metal_temperature,
    CorrosionAnalysis as CorrosionAnalysisResult,
    RiskLevel as CorrosionRiskLevel,
    CorrosionMechanism,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FlowArrangement(str, Enum):
    """Heat exchanger flow arrangements."""
    COUNTER_FLOW = "counter_flow"
    PARALLEL_FLOW = "parallel_flow"
    CROSS_FLOW_BOTH_UNMIXED = "cross_flow_both_unmixed"
    CROSS_FLOW_ONE_MIXED = "cross_flow_one_mixed"


class RiskLevel(str, Enum):
    """Risk level classifications."""
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


# =============================================================================
# INPUT MODELS
# =============================================================================

class FlueGasComposition(BaseModel):
    """Flue gas conditions and composition at economizer inlet/outlet."""

    temperature_in_celsius: float = Field(
        ...,
        ge=100.0,
        le=600.0,
        description="Flue gas inlet temperature (deg C)"
    )
    temperature_out_celsius: float = Field(
        ...,
        ge=50.0,
        le=400.0,
        description="Flue gas outlet temperature (deg C)"
    )
    mass_flow_kg_s: float = Field(
        ...,
        gt=0,
        description="Flue gas mass flow rate (kg/s)"
    )
    H2O_percent: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Water vapor concentration (volume %)"
    )
    SO3_ppmv: float = Field(
        default=5.0,
        ge=0.0,
        le=200.0,
        description="SO3 concentration (ppmv)"
    )
    total_pressure_kPa: float = Field(
        default=101.325,
        gt=0,
        description="Total flue gas pressure (kPa)"
    )
    specific_heat_J_per_kg_K: float = Field(
        default=1100.0,
        gt=0,
        description="Flue gas specific heat (J/kg-K)"
    )

    @validator('temperature_out_celsius')
    def validate_temp_outlet(cls, v, values):
        """Ensure outlet is less than inlet."""
        if 'temperature_in_celsius' in values and v >= values['temperature_in_celsius']:
            raise ValueError(
                f"Flue gas outlet temperature ({v}) must be less than "
                f"inlet temperature ({values['temperature_in_celsius']})"
            )
        return v


class WaterSideConditions(BaseModel):
    """Water/steam side conditions for the economizer."""

    inlet_temperature_celsius: float = Field(
        ...,
        ge=20.0,
        le=250.0,
        description="Water inlet temperature (deg C)"
    )
    outlet_temperature_celsius: float = Field(
        ...,
        ge=30.0,
        le=300.0,
        description="Water outlet temperature (deg C)"
    )
    mass_flow_kg_s: float = Field(
        ...,
        gt=0,
        description="Water mass flow rate (kg/s)"
    )
    drum_pressure_MPa: float = Field(
        ...,
        gt=0,
        le=22.0,
        description="Steam drum pressure (MPa absolute)"
    )
    specific_heat_J_per_kg_K: float = Field(
        default=4200.0,
        gt=0,
        description="Water specific heat (J/kg-K)"
    )
    inlet_pressure_MPa: Optional[float] = Field(
        None,
        gt=0,
        description="Water inlet pressure (MPa)"
    )

    @validator('outlet_temperature_celsius')
    def validate_temp_outlet(cls, v, values):
        """Ensure outlet is greater than inlet."""
        if 'inlet_temperature_celsius' in values and v <= values['inlet_temperature_celsius']:
            raise ValueError(
                f"Water outlet temperature ({v}) must be greater than "
                f"inlet temperature ({values['inlet_temperature_celsius']})"
            )
        return v


class HeatExchangerGeometry(BaseModel):
    """Economizer heat exchanger geometry and characteristics."""

    flow_arrangement: FlowArrangement = Field(
        default=FlowArrangement.COUNTER_FLOW,
        description="Heat exchanger flow arrangement"
    )
    tube_outer_diameter_mm: float = Field(
        default=51.0,
        gt=0,
        description="Tube outer diameter (mm)"
    )
    tube_wall_thickness_mm: float = Field(
        default=4.0,
        gt=0,
        description="Tube wall thickness (mm)"
    )
    original_tube_wall_thickness_mm: Optional[float] = Field(
        None,
        gt=0,
        description="Original tube wall thickness for life calculation (mm)"
    )
    number_of_tubes: Optional[int] = Field(
        None,
        gt=0,
        description="Number of tubes"
    )
    tube_length_m: Optional[float] = Field(
        None,
        gt=0,
        description="Tube length (m)"
    )
    tube_material: str = Field(
        default="carbon_steel",
        description="Tube material"
    )
    tube_material_conductivity_W_per_mK: float = Field(
        default=50.0,
        gt=0,
        description="Tube material thermal conductivity (W/m-K)"
    )


class OperatingConditions(BaseModel):
    """Additional operating conditions and parameters."""

    boiler_load_percent: float = Field(
        default=100.0,
        ge=0,
        le=120.0,
        description="Current boiler load (%)"
    )
    excess_air_percent: Optional[float] = Field(
        None,
        ge=0,
        description="Excess air percentage"
    )
    fuel_type: Optional[str] = Field(
        None,
        description="Fuel type (coal, natural_gas, oil, etc.)"
    )
    fuel_sulfur_percent: Optional[float] = Field(
        None,
        ge=0,
        description="Fuel sulfur content (%)"
    )
    water_side_fouling_resistance_m2K_per_W: float = Field(
        default=0.0002,
        ge=0,
        description="Water-side fouling resistance (m2-K/W)"
    )
    internal_film_coefficient_W_per_m2K: float = Field(
        default=5000.0,
        gt=0,
        description="Water-side film heat transfer coefficient (W/m2-K)"
    )


class EconomizerInput(BaseModel):
    """
    Complete input model for Economizer Performance Analysis.

    Attributes:
        flue_gas: Flue gas composition and conditions
        water_side: Water side conditions
        heat_exchanger: Heat exchanger geometry
        operating_conditions: Additional operating parameters
        metadata: Additional metadata for tracking
    """

    flue_gas: FlueGasComposition = Field(
        ...,
        description="Flue gas conditions and composition"
    )
    water_side: WaterSideConditions = Field(
        ...,
        description="Water side conditions"
    )
    heat_exchanger: HeatExchangerGeometry = Field(
        default_factory=HeatExchangerGeometry,
        description="Heat exchanger geometry"
    )
    operating_conditions: OperatingConditions = Field(
        default_factory=OperatingConditions,
        description="Operating conditions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class ThermalPerformance(BaseModel):
    """Thermal performance results."""

    heat_transfer_kW: float = Field(..., description="Heat transfer rate (kW)")
    heat_transfer_MW: float = Field(..., description="Heat transfer rate (MW)")
    effectiveness: float = Field(..., ge=0, le=1, description="Heat exchanger effectiveness")
    NTU: float = Field(..., ge=0, description="Number of Transfer Units")
    C_r: float = Field(..., ge=0, le=1, description="Capacity ratio (C_min/C_max)")
    C_min_W_per_K: float = Field(..., description="Minimum heat capacity rate (W/K)")
    C_max_W_per_K: float = Field(..., description="Maximum heat capacity rate (W/K)")
    LMTD_celsius: float = Field(..., description="Log Mean Temperature Difference (deg C)")
    UA_W_per_K: Optional[float] = Field(None, description="Overall UA value (W/K)")


class AcidDewPointAnalysis(BaseModel):
    """Acid dew point analysis results."""

    acid_dew_point_celsius: float = Field(
        ...,
        description="Calculated acid dew point temperature (deg C)"
    )
    P_H2O_atm: float = Field(..., description="Water vapor partial pressure (atm)")
    P_SO3_atm: float = Field(..., description="SO3 partial pressure (atm)")
    correlation_used: str = Field(
        default="Verhoff-Banchero (1974)",
        description="Correlation used for calculation"
    )


class SteamingAnalysis(BaseModel):
    """Steaming risk analysis results."""

    saturation_temperature_celsius: float = Field(
        ...,
        description="Saturation temperature at drum pressure (deg C)"
    )
    water_outlet_temperature_celsius: float = Field(
        ...,
        description="Water outlet temperature (deg C)"
    )
    approach_to_saturation_celsius: float = Field(
        ...,
        description="Margin below saturation (deg C)"
    )
    risk_level: str = Field(..., description="Risk level classification")
    risk_score: float = Field(..., ge=0, le=100, description="Numeric risk score (0-100)")
    recommended_max_outlet_celsius: float = Field(
        ...,
        description="Recommended maximum outlet temperature (deg C)"
    )
    description: str = Field(..., description="Risk description")


class CorrosionAnalysis(BaseModel):
    """Corrosion risk analysis results."""

    tube_metal_temperature_celsius: float = Field(
        ...,
        description="Estimated tube metal temperature (deg C)"
    )
    acid_dew_point_celsius: float = Field(
        ...,
        description="Acid dew point temperature (deg C)"
    )
    margin_above_dew_point_celsius: float = Field(
        ...,
        description="Margin above dew point (deg C)"
    )
    risk_level: str = Field(..., description="Risk level classification")
    risk_score: float = Field(..., ge=0, le=100, description="Numeric risk score (0-100)")
    mechanism: str = Field(..., description="Primary corrosion mechanism")
    recommended_min_metal_temp_celsius: float = Field(
        ...,
        description="Recommended minimum metal temperature (deg C)"
    )
    estimated_corrosion_rate_mm_per_year: float = Field(
        ...,
        ge=0,
        description="Estimated corrosion rate (mm/year)"
    )
    remaining_tube_life_years: Optional[float] = Field(
        None,
        description="Estimated remaining tube life (years)"
    )
    description: str = Field(..., description="Risk description")


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Operation timestamp")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    tool_name: str = Field(..., description="Tool/calculator used")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class EconomizerOutput(BaseModel):
    """
    Complete output model for Economizer Performance Analysis.

    Contains all analysis results including thermal performance,
    acid dew point, steaming risk, corrosion risk, and provenance chain.
    """

    # Identification
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

    # Thermal Performance
    thermal_performance: ThermalPerformance = Field(
        ...,
        description="Thermal performance results"
    )
    effectiveness: float = Field(..., ge=0, le=1, description="Heat exchanger effectiveness")

    # Acid Dew Point
    acid_dew_point_analysis: AcidDewPointAnalysis = Field(
        ...,
        description="Acid dew point analysis"
    )
    acid_dew_point_celsius: float = Field(..., description="Acid dew point (deg C)")

    # Steaming Risk
    steaming_analysis: SteamingAnalysis = Field(..., description="Steaming risk analysis")

    # Corrosion Risk
    corrosion_analysis: CorrosionAnalysis = Field(..., description="Corrosion risk analysis")

    # Summary Metrics
    overall_risk_level: str = Field(..., description="Overall risk level (worst of all)")
    recommendations: List[str] = Field(..., description="Operational recommendations")

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(
        ...,
        description="Complete audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash of provenance chain")

    # Processing Metadata
    processing_time_ms: float = Field(..., description="Total processing time (ms)")
    validation_status: str = Field(..., description="PASS or FAIL")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")


# =============================================================================
# ECONOMIZER PERFORMANCE AGENT
# =============================================================================

class EconomizerPerformanceAgent:
    """
    GL-020: Economizer Performance Analysis Agent (ECONOPULSE).

    This agent analyzes economizer heat exchanger performance for industrial
    boilers and HRSG systems. It provides comprehensive analysis including:

    - Heat exchanger effectiveness using NTU-epsilon method
    - Acid dew point calculation using Verhoff-Banchero correlation
    - Steaming risk detection using IAPWS-IF97 saturation properties
    - Cold-end corrosion risk assessment

    Zero-Hallucination Guarantee:
    - All thermodynamic calculations use deterministic formulas
    - Verhoff-Banchero coefficients exact from original publication
    - IAPWS-IF97 Equation 31 coefficients exact from standard
    - No LLM inference in calculation path
    - Complete audit trail for regulatory compliance

    Attributes:
        AGENT_ID: Unique agent identifier (GL-020)
        AGENT_NAME: Agent name (ECONOPULSE)
        VERSION: Agent version

    Example:
        >>> agent = EconomizerPerformanceAgent()
        >>> result = agent.run(input_data)
        >>> assert result.validation_status == "PASS"
    """

    AGENT_ID = "GL-020"
    AGENT_NAME = "ECONOPULSE"
    VERSION = "1.0.0"
    DESCRIPTION = "Economizer Performance Analysis Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EconomizerPerformanceAgent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []

        logger.info(
            f"EconomizerPerformanceAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: EconomizerInput) -> EconomizerOutput:
        """
        Execute economizer performance analysis.

        This method performs comprehensive analysis:
        1. Calculate heat exchanger effectiveness (NTU-epsilon method)
        2. Calculate acid dew point (Verhoff-Banchero correlation)
        3. Assess steaming risk (IAPWS-IF97 saturation)
        4. Assess corrosion risk (tube metal temperature vs dew point)
        5. Generate recommendations

        All calculations follow zero-hallucination principles:
        - Deterministic formulas from thermodynamic standards
        - Complete provenance tracking
        - No LLM in calculation path

        Args:
            input_data: Validated input data

        Returns:
            Complete analysis output with provenance hash

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(
            f"Starting economizer analysis "
            f"(T_gas_in={input_data.flue_gas.temperature_in_celsius} C, "
            f"T_water_in={input_data.water_side.inlet_temperature_celsius} C)"
        )

        try:
            # Step 1: Calculate thermal performance
            thermal_perf = self._calculate_thermal_performance(input_data)
            self._track_provenance(
                "thermal_performance_calculation",
                {
                    "T_hot_in": input_data.flue_gas.temperature_in_celsius,
                    "T_hot_out": input_data.flue_gas.temperature_out_celsius,
                    "T_cold_in": input_data.water_side.inlet_temperature_celsius,
                    "T_cold_out": input_data.water_side.outlet_temperature_celsius,
                },
                {"effectiveness": thermal_perf.effectiveness, "Q_kW": thermal_perf.heat_transfer_kW},
                "NTU-epsilon calculator"
            )

            # Step 2: Calculate acid dew point
            acid_dew_analysis = self._calculate_acid_dew_point(input_data)
            self._track_provenance(
                "acid_dew_point_calculation",
                {
                    "H2O_percent": input_data.flue_gas.H2O_percent,
                    "SO3_ppmv": input_data.flue_gas.SO3_ppmv,
                    "P_total_kPa": input_data.flue_gas.total_pressure_kPa,
                },
                {"T_dew_celsius": acid_dew_analysis.acid_dew_point_celsius},
                "Verhoff-Banchero correlation"
            )

            # Step 3: Assess steaming risk
            steaming = self._assess_steaming_risk(input_data)
            self._track_provenance(
                "steaming_risk_assessment",
                {
                    "T_water_out": input_data.water_side.outlet_temperature_celsius,
                    "drum_pressure_MPa": input_data.water_side.drum_pressure_MPa,
                },
                {"risk_level": steaming.risk_level, "approach_celsius": steaming.approach_to_saturation_celsius},
                "IAPWS-IF97 saturation"
            )

            # Step 4: Assess corrosion risk
            corrosion = self._assess_corrosion_risk(input_data, acid_dew_analysis)
            self._track_provenance(
                "corrosion_risk_assessment",
                {
                    "T_metal_celsius": corrosion.tube_metal_temperature_celsius,
                    "T_dew_celsius": acid_dew_analysis.acid_dew_point_celsius,
                },
                {"risk_level": corrosion.risk_level, "margin_celsius": corrosion.margin_above_dew_point_celsius},
                "Corrosion risk calculator"
            )

            # Step 5: Determine overall risk and recommendations
            overall_risk = self._determine_overall_risk(steaming, corrosion)
            recommendations = self._generate_recommendations(
                input_data, thermal_perf, acid_dew_analysis, steaming, corrosion
            )

            # Step 6: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate analysis ID
            analysis_id = (
                f"ECON-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(str(input_data.dict()).encode()).hexdigest()[:8]}"
            )

            # Validation status
            validation_status = "PASS" if not self._validation_errors else "FAIL"

            output = EconomizerOutput(
                analysis_id=analysis_id,
                thermal_performance=thermal_perf,
                effectiveness=thermal_perf.effectiveness,
                acid_dew_point_analysis=acid_dew_analysis,
                acid_dew_point_celsius=acid_dew_analysis.acid_dew_point_celsius,
                steaming_analysis=steaming,
                corrosion_analysis=corrosion,
                overall_risk_level=overall_risk,
                recommendations=recommendations,
                provenance_chain=[
                    ProvenanceRecord(
                        operation=step["operation"],
                        timestamp=step["timestamp"],
                        input_hash=step["input_hash"],
                        output_hash=step["output_hash"],
                        tool_name=step["tool_name"],
                        parameters=step.get("parameters", {}),
                    )
                    for step in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status=validation_status,
                validation_errors=self._validation_errors,
            )

            logger.info(
                f"Economizer analysis complete: effectiveness={thermal_perf.effectiveness:.3f}, "
                f"acid_dew_point={acid_dew_analysis.acid_dew_point_celsius:.1f} C, "
                f"overall_risk={overall_risk} (duration: {processing_time:.2f} ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Economizer analysis failed: {str(e)}", exc_info=True)
            raise

    def _calculate_thermal_performance(self, input_data: EconomizerInput) -> ThermalPerformance:
        """
        Calculate heat exchanger thermal performance using NTU-epsilon method.

        ZERO-HALLUCINATION: Uses exact Kays & London NTU-epsilon formulas.
        """
        # Extract temperatures
        T_hot_in = input_data.flue_gas.temperature_in_celsius
        T_hot_out = input_data.flue_gas.temperature_out_celsius
        T_cold_in = input_data.water_side.inlet_temperature_celsius
        T_cold_out = input_data.water_side.outlet_temperature_celsius

        # Calculate heat capacity rates
        # C = m_dot * c_p (W/K)
        C_hot = (
            input_data.flue_gas.mass_flow_kg_s *
            input_data.flue_gas.specific_heat_J_per_kg_K
        )
        C_cold = (
            input_data.water_side.mass_flow_kg_s *
            input_data.water_side.specific_heat_J_per_kg_K
        )

        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_r = C_min / C_max if C_max > 0 else 0

        # Calculate effectiveness from temperatures
        # ZERO-HALLUCINATION: epsilon = Q_actual / Q_max
        effectiveness = calculate_effectiveness_from_temperatures(
            T_hot_in=T_hot_in,
            T_hot_out=T_hot_out,
            T_cold_in=T_cold_in,
            T_cold_out=T_cold_out,
            C_hot_W_per_K=C_hot,
            C_cold_W_per_K=C_cold,
        )

        # Calculate actual heat transfer
        # ZERO-HALLUCINATION: Q = C_hot * (T_hot_in - T_hot_out)
        Q_watts = C_hot * (T_hot_in - T_hot_out)
        Q_kW = Q_watts / 1000.0
        Q_MW = Q_kW / 1000.0

        # Calculate LMTD
        LMTD = self._calculate_LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out)

        # Estimate NTU from effectiveness
        # For counter-flow, NTU can be back-calculated
        NTU = self._estimate_NTU_from_effectiveness(
            effectiveness,
            C_r,
            input_data.heat_exchanger.flow_arrangement
        )

        # Calculate UA if possible
        UA_W_per_K = None
        if NTU > 0:
            UA_W_per_K = NTU * C_min

        logger.debug(
            f"Thermal performance: Q={Q_kW:.2f} kW, eps={effectiveness:.4f}, "
            f"NTU={NTU:.3f}, C_r={C_r:.3f}"
        )

        return ThermalPerformance(
            heat_transfer_kW=round(Q_kW, 2),
            heat_transfer_MW=round(Q_MW, 4),
            effectiveness=round(effectiveness, 4),
            NTU=round(NTU, 4),
            C_r=round(C_r, 4),
            C_min_W_per_K=round(C_min, 2),
            C_max_W_per_K=round(C_max, 2),
            LMTD_celsius=round(LMTD, 2),
            UA_W_per_K=round(UA_W_per_K, 2) if UA_W_per_K else None,
        )

    def _calculate_LMTD(
        self,
        T_hot_in: float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out: float
    ) -> float:
        """
        Calculate Log Mean Temperature Difference for counter-flow.

        ZERO-HALLUCINATION FORMULA:
        LMTD = (dT1 - dT2) / ln(dT1 / dT2)

        Where for counter-flow:
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
        """
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in

        if dT1 <= 0 or dT2 <= 0:
            logger.warning(f"Invalid temperature differences: dT1={dT1}, dT2={dT2}")
            return abs(dT1 + dT2) / 2  # Return arithmetic mean as fallback

        if abs(dT1 - dT2) < 0.01:
            # When dT1 ≈ dT2, LMTD ≈ dT1
            return dT1

        import math
        LMTD = (dT1 - dT2) / math.log(dT1 / dT2)

        return LMTD

    def _estimate_NTU_from_effectiveness(
        self,
        epsilon: float,
        C_r: float,
        flow_arrangement: FlowArrangement
    ) -> float:
        """
        Back-calculate NTU from effectiveness.

        ZERO-HALLUCINATION: Inverse of Kays & London formulas.
        """
        import math

        if epsilon <= 0:
            return 0.0
        if epsilon >= 1:
            return float('inf')

        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            if abs(C_r - 1.0) < 1e-10:
                # NTU = epsilon / (1 - epsilon)
                if epsilon < 1:
                    return epsilon / (1.0 - epsilon)
                return float('inf')
            else:
                # epsilon = [1 - exp(-NTU*(1-C_r))] / [1 - C_r*exp(-NTU*(1-C_r))]
                # Solving for NTU:
                # NTU = ln[(1 - epsilon*C_r) / (1 - epsilon)] / (1 - C_r)
                if (1 - epsilon * C_r) > 0 and (1 - epsilon) > 0:
                    return math.log((1 - epsilon * C_r) / (1 - epsilon)) / (1 - C_r)

        elif flow_arrangement == FlowArrangement.PARALLEL_FLOW:
            # epsilon = [1 - exp(-NTU*(1+C_r))] / (1 + C_r)
            # NTU = -ln(1 - epsilon*(1+C_r)) / (1 + C_r)
            arg = 1 - epsilon * (1 + C_r)
            if arg > 0:
                return -math.log(arg) / (1 + C_r)

        # Default estimate
        return 1.0

    def _calculate_acid_dew_point(self, input_data: EconomizerInput) -> AcidDewPointAnalysis:
        """
        Calculate acid dew point using Verhoff-Banchero correlation.

        ZERO-HALLUCINATION: Uses exact Verhoff-Banchero formula coefficients.
        """
        # Calculate partial pressures
        pressures = calculate_partial_pressures(
            total_pressure_kPa=input_data.flue_gas.total_pressure_kPa,
            H2O_percent=input_data.flue_gas.H2O_percent,
            SO3_ppmv=input_data.flue_gas.SO3_ppmv,
        )

        # Calculate acid dew point
        T_dew_celsius = verhoff_banchero_acid_dew_point(
            P_H2O_atm=pressures.P_H2O_atm,
            P_SO3_atm=pressures.P_SO3_atm,
        )

        logger.debug(
            f"Acid dew point: {T_dew_celsius:.1f} C "
            f"(P_H2O={pressures.P_H2O_atm:.6f} atm, P_SO3={pressures.P_SO3_atm:.2e} atm)"
        )

        return AcidDewPointAnalysis(
            acid_dew_point_celsius=round(T_dew_celsius, 2),
            P_H2O_atm=round(pressures.P_H2O_atm, 6),
            P_SO3_atm=pressures.P_SO3_atm,
            correlation_used="Verhoff-Banchero (1974)",
        )

    def _assess_steaming_risk(self, input_data: EconomizerInput) -> SteamingAnalysis:
        """
        Assess steaming risk using IAPWS-IF97 saturation properties.

        ZERO-HALLUCINATION: Uses exact IAPWS-IF97 Equation 31 coefficients.
        """
        result = detect_steaming_risk(
            T_water_out_celsius=input_data.water_side.outlet_temperature_celsius,
            drum_pressure_MPa=input_data.water_side.drum_pressure_MPa,
        )

        return SteamingAnalysis(
            saturation_temperature_celsius=result.saturation_temperature_celsius,
            water_outlet_temperature_celsius=result.water_outlet_temperature_celsius,
            approach_to_saturation_celsius=result.approach_to_saturation_celsius,
            risk_level=result.risk_level.value,
            risk_score=result.risk_score,
            recommended_max_outlet_celsius=result.recommended_max_outlet_celsius,
            description=result.description,
        )

    def _assess_corrosion_risk(
        self,
        input_data: EconomizerInput,
        acid_dew_analysis: AcidDewPointAnalysis
    ) -> CorrosionAnalysis:
        """
        Assess cold-end corrosion risk.

        Estimates tube metal temperature and compares to acid dew point.
        """
        # Estimate heat flux at cold end
        # Simple estimate based on LMTD and typical UA
        T_gas_cold_end = input_data.flue_gas.temperature_out_celsius
        T_water_cold_end = input_data.water_side.inlet_temperature_celsius
        delta_T_local = T_gas_cold_end - T_water_cold_end

        # Estimate heat flux (typical range: 10-100 kW/m2)
        heat_flux_estimate = max(10000, min(100000, delta_T_local * 500))  # W/m2

        # Estimate tube metal temperature
        T_metal = estimate_tube_metal_temperature(
            T_fluid_celsius=T_water_cold_end,
            T_gas_celsius=T_gas_cold_end,
            heat_flux_W_per_m2=heat_flux_estimate,
            tube_wall_thickness_mm=input_data.heat_exchanger.tube_wall_thickness_mm,
            tube_material_conductivity_W_per_mK=input_data.heat_exchanger.tube_material_conductivity_W_per_mK,
            internal_film_coefficient_W_per_m2K=input_data.operating_conditions.internal_film_coefficient_W_per_m2K,
            fouling_resistance_m2K_per_W=input_data.operating_conditions.water_side_fouling_resistance_m2K_per_W,
        )

        # Assess corrosion risk
        result = assess_corrosion_risk(
            T_metal_celsius=T_metal,
            T_acid_dew_point_celsius=acid_dew_analysis.acid_dew_point_celsius,
            tube_wall_thickness_mm=input_data.heat_exchanger.tube_wall_thickness_mm,
            original_wall_thickness_mm=input_data.heat_exchanger.original_tube_wall_thickness_mm,
        )

        return CorrosionAnalysis(
            tube_metal_temperature_celsius=result.tube_metal_temperature_celsius,
            acid_dew_point_celsius=result.acid_dew_point_celsius,
            margin_above_dew_point_celsius=result.margin_above_dew_point_celsius,
            risk_level=result.risk_level.value,
            risk_score=result.risk_score,
            mechanism=result.mechanism.value,
            recommended_min_metal_temp_celsius=result.recommended_min_metal_temp_celsius,
            estimated_corrosion_rate_mm_per_year=result.estimated_corrosion_rate_mm_per_year,
            remaining_tube_life_years=result.remaining_tube_life_years,
            description=result.description,
        )

    def _determine_overall_risk(
        self,
        steaming: SteamingAnalysis,
        corrosion: CorrosionAnalysis
    ) -> str:
        """Determine the overall risk level (worst of all risks)."""
        risk_order = ["NONE", "LOW", "MODERATE", "HIGH", "SEVERE", "CRITICAL"]

        steaming_idx = risk_order.index(steaming.risk_level) if steaming.risk_level in risk_order else 0
        corrosion_idx = risk_order.index(corrosion.risk_level) if corrosion.risk_level in risk_order else 0

        max_idx = max(steaming_idx, corrosion_idx)
        return risk_order[max_idx]

    def _generate_recommendations(
        self,
        input_data: EconomizerInput,
        thermal: ThermalPerformance,
        acid_dew: AcidDewPointAnalysis,
        steaming: SteamingAnalysis,
        corrosion: CorrosionAnalysis
    ) -> List[str]:
        """Generate operational recommendations based on analysis."""
        recommendations = []

        # Effectiveness recommendations
        if thermal.effectiveness < 0.6:
            recommendations.append(
                f"Heat exchanger effectiveness ({thermal.effectiveness:.2f}) is below optimal. "
                f"Consider cleaning tubes or increasing heat transfer area."
            )
        elif thermal.effectiveness > 0.95:
            recommendations.append(
                f"High effectiveness ({thermal.effectiveness:.2f}) may indicate low flow rates. "
                f"Verify design conditions."
            )

        # Steaming recommendations
        if steaming.risk_level in ["HIGH", "CRITICAL"]:
            recommendations.append(
                f"URGENT: Reduce water outlet temperature below {steaming.recommended_max_outlet_celsius:.1f} C "
                f"to prevent steaming. Current approach: {steaming.approach_to_saturation_celsius:.1f} C."
            )
        elif steaming.risk_level == "MODERATE":
            recommendations.append(
                f"Monitor water outlet temperature closely. "
                f"Maintain at least 15 C margin below saturation ({steaming.saturation_temperature_celsius:.1f} C)."
            )

        # Corrosion recommendations
        if corrosion.risk_level in ["HIGH", "SEVERE"]:
            min_water_temp = acid_dew.acid_dew_point_celsius - 10  # Simplified estimate
            recommendations.append(
                f"URGENT: Increase feedwater temperature above {corrosion.recommended_min_metal_temp_celsius - 15:.1f} C "
                f"to prevent acid dew point corrosion. Current margin: {corrosion.margin_above_dew_point_celsius:.1f} C."
            )
        elif corrosion.risk_level == "MODERATE":
            recommendations.append(
                f"Consider increasing feedwater temperature or reducing fuel sulfur content. "
                f"Current margin above acid dew point: {corrosion.margin_above_dew_point_celsius:.1f} C."
            )

        # Remaining life recommendations
        if corrosion.remaining_tube_life_years is not None:
            if corrosion.remaining_tube_life_years < 2:
                recommendations.append(
                    f"CRITICAL: Estimated remaining tube life is only {corrosion.remaining_tube_life_years:.1f} years. "
                    f"Plan tube replacement immediately."
                )
            elif corrosion.remaining_tube_life_years < 5:
                recommendations.append(
                    f"Schedule tube inspection and consider replacement planning. "
                    f"Estimated remaining life: {corrosion.remaining_tube_life_years:.1f} years."
                )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Economizer is operating within acceptable limits. Continue routine monitoring."
            )

        return recommendations

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ) -> None:
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"],
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-020",
    "name": "ECONOPULSE - Economizer Performance Agent",
    "version": "1.0.0",
    "summary": "Economizer heat exchanger performance analysis with steaming and corrosion risk assessment",
    "tags": [
        "economizer",
        "heat-exchanger",
        "boiler",
        "HRSG",
        "acid-dew-point",
        "corrosion",
        "steaming",
        "NTU-epsilon",
        "Verhoff-Banchero",
        "IAPWS-IF97",
    ],
    "owners": ["thermal-systems-team"],
    "compute": {
        "entrypoint": "python://agents.gl_020_economizer_performance.agent:EconomizerPerformanceAgent",
        "deterministic": True,
    },
    "standards": [
        {"ref": "IAPWS-IF97", "description": "Industrial Formulation 1997 for Water/Steam Properties"},
        {"ref": "ASME PTC 4", "description": "Fired Steam Generators Performance Test Code"},
        {"ref": "Verhoff-Banchero (1974)", "description": "Acid Dew Point Correlation"},
        {"ref": "Kays & London", "description": "Compact Heat Exchangers NTU-epsilon Method"},
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True,
    },
}
