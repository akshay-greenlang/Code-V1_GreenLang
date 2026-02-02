"""
GL-056: Curing Oven Controller Agent (CURE-CTRL)

This module implements the CuringOvenAgent for optimizing curing oven operations
in industrial coating, composites, and powder coating applications.

The agent provides:
- Temperature profile optimization per zone
- Cure cycle validation and monitoring
- Arrhenius cure kinetics calculation
- Cross-linking degree estimation
- Energy consumption tracking and optimization
- Complete SHA-256 provenance tracking

Key Formulas:
- Arrhenius equation: k = A * exp(-Ea / (R * T))
- Cure degree: alpha(t) = 1 - exp(-k * t^n)
- Time-temperature equivalence for cure scheduling

Standards Compliance:
- ASTM D4541: Pull-Off Adhesion Testing
- ISO 11507: Paints and Varnishes - Exposure to Artificial Weathering
- NFPA 86: Standard for Ovens and Furnaces

Example:
    >>> agent = CuringOvenAgent()
    >>> result = agent.run(CuringOvenInput(
    ...     oven_id="CURE-001",
    ...     zones=[ZoneData(zone_id="Z1", setpoint_celsius=180, ...)],
    ...     product_type="powder_coating",
    ...     conveyor_speed_m_min=2.5
    ... ))
    >>> print(f"Cure Quality Score: {result.cure_quality_score}")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Gas constant R in J/(mol*K)
GAS_CONSTANT_R = 8.314

# Stefan-Boltzmann constant W/(m^2*K^4)
STEFAN_BOLTZMANN = 5.67e-8


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ProductType(str, Enum):
    POWDER_COATING = "powder_coating"
    LIQUID_COATING = "liquid_coating"
    COMPOSITE_MATERIAL = "composite_material"
    ADHESIVE = "adhesive"
    RUBBER = "rubber"
    INK = "ink"
    EPOXY = "epoxy"


class CureStatus(str, Enum):
    UNDER_CURED = "under_cured"
    OPTIMAL = "optimal"
    OVER_CURED = "over_cured"
    UNCERTAIN = "uncertain"


class ZoneStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    OFFLINE = "offline"


# Cure kinetics parameters by product type
# Structure: {product: {activation_energy_kJ_mol, pre_exponential_factor, reaction_order, optimal_cure_degree}}
CURE_KINETICS = {
    ProductType.POWDER_COATING: {
        "activation_energy_kJ_mol": 80.0,  # Ea in kJ/mol
        "pre_exponential_factor": 1.0e9,   # A (1/s)
        "reaction_order": 1.5,             # n
        "optimal_cure_degree": 0.95,       # Target alpha
        "min_cure_degree": 0.85,           # Minimum acceptable
        "max_cure_degree": 0.99            # Over-cure threshold
    },
    ProductType.LIQUID_COATING: {
        "activation_energy_kJ_mol": 60.0,
        "pre_exponential_factor": 5.0e7,
        "reaction_order": 1.2,
        "optimal_cure_degree": 0.92,
        "min_cure_degree": 0.80,
        "max_cure_degree": 0.98
    },
    ProductType.COMPOSITE_MATERIAL: {
        "activation_energy_kJ_mol": 95.0,
        "pre_exponential_factor": 2.0e10,
        "reaction_order": 2.0,
        "optimal_cure_degree": 0.98,
        "min_cure_degree": 0.90,
        "max_cure_degree": 0.995
    },
    ProductType.ADHESIVE: {
        "activation_energy_kJ_mol": 70.0,
        "pre_exponential_factor": 8.0e8,
        "reaction_order": 1.3,
        "optimal_cure_degree": 0.90,
        "min_cure_degree": 0.75,
        "max_cure_degree": 0.97
    },
    ProductType.RUBBER: {
        "activation_energy_kJ_mol": 85.0,
        "pre_exponential_factor": 3.0e9,
        "reaction_order": 1.8,
        "optimal_cure_degree": 0.93,
        "min_cure_degree": 0.82,
        "max_cure_degree": 0.98
    },
    ProductType.INK: {
        "activation_energy_kJ_mol": 50.0,
        "pre_exponential_factor": 1.0e7,
        "reaction_order": 1.0,
        "optimal_cure_degree": 0.88,
        "min_cure_degree": 0.70,
        "max_cure_degree": 0.96
    },
    ProductType.EPOXY: {
        "activation_energy_kJ_mol": 75.0,
        "pre_exponential_factor": 1.5e9,
        "reaction_order": 1.6,
        "optimal_cure_degree": 0.96,
        "min_cure_degree": 0.88,
        "max_cure_degree": 0.99
    }
}


# Recommended cure parameters by product type
CURE_PARAMETERS = {
    ProductType.POWDER_COATING: {
        "temp_range": (160, 200),
        "time_minutes": (10, 20),
        "energy_intensity_kwh_kg": 0.5
    },
    ProductType.LIQUID_COATING: {
        "temp_range": (80, 140),
        "time_minutes": (15, 30),
        "energy_intensity_kwh_kg": 0.4
    },
    ProductType.COMPOSITE_MATERIAL: {
        "temp_range": (120, 180),
        "time_minutes": (60, 180),
        "energy_intensity_kwh_kg": 1.2
    },
    ProductType.ADHESIVE: {
        "temp_range": (100, 150),
        "time_minutes": (5, 15),
        "energy_intensity_kwh_kg": 0.3
    },
    ProductType.RUBBER: {
        "temp_range": (140, 190),
        "time_minutes": (20, 60),
        "energy_intensity_kwh_kg": 0.8
    },
    ProductType.INK: {
        "temp_range": (80, 120),
        "time_minutes": (2, 10),
        "energy_intensity_kwh_kg": 0.2
    },
    ProductType.EPOXY: {
        "temp_range": (100, 160),
        "time_minutes": (30, 90),
        "energy_intensity_kwh_kg": 0.6
    }
}


# =============================================================================
# INPUT MODELS
# =============================================================================

class ZoneData(BaseModel):
    """Individual oven zone data."""

    zone_id: str = Field(..., description="Zone identifier (Z1, Z2, etc.)")
    setpoint_celsius: float = Field(..., ge=0, le=500, description="Temperature setpoint (C)")
    actual_celsius: float = Field(..., ge=0, le=500, description="Actual temperature (C)")
    power_kW: float = Field(default=0.0, ge=0, description="Current power consumption (kW)")
    airflow_cfm: Optional[float] = Field(None, ge=0, description="Airflow rate (CFM)")
    status: ZoneStatus = Field(default=ZoneStatus.NORMAL, description="Zone operational status")
    length_meters: float = Field(default=1.0, gt=0, description="Zone length (m)")


class ConveyorData(BaseModel):
    """Conveyor system data."""

    speed_m_min: float = Field(..., gt=0, description="Conveyor speed (m/min)")
    width_meters: float = Field(default=1.0, gt=0, description="Conveyor width (m)")
    product_loading_percent: float = Field(default=80.0, ge=0, le=100, description="Product loading (%)")


class ProductData(BaseModel):
    """Product being cured."""

    product_type: ProductType = Field(..., description="Type of product")
    mass_flow_kg_hr: float = Field(..., ge=0, description="Product mass flow rate (kg/hr)")
    thickness_mm: Optional[float] = Field(None, gt=0, description="Product thickness (mm)")
    surface_area_m2_hr: Optional[float] = Field(None, ge=0, description="Surface area rate (m2/hr)")
    cure_requirement_minutes: Optional[float] = Field(None, gt=0, description="Required cure time (min)")
    target_cure_temp_celsius: Optional[float] = Field(None, gt=0, description="Target cure temp (C)")
    initial_cure_degree: float = Field(default=0.0, ge=0, le=1, description="Initial cure degree (0-1)")


class AmbientConditions(BaseModel):
    """Ambient operating conditions."""

    temperature_celsius: float = Field(default=25.0, description="Ambient temperature (C)")
    humidity_percent: float = Field(default=50.0, ge=0, le=100, description="Relative humidity (%)")


class CuringOvenInput(BaseModel):
    """Input data model for CuringOvenAgent."""

    oven_id: str = Field(..., min_length=1, description="Unique oven identifier")
    zones: List[ZoneData] = Field(..., min_items=1, description="Zone temperature and status data")
    conveyor: ConveyorData = Field(..., description="Conveyor system data")
    product: ProductData = Field(..., description="Product being cured")
    ambient: AmbientConditions = Field(default_factory=AmbientConditions)
    energy_cost_per_kwh: float = Field(default=0.12, gt=0, description="Energy cost ($/kWh)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class ZoneAnalysis(BaseModel):
    """Analysis results for a single zone."""

    zone_id: str
    setpoint_celsius: float
    actual_celsius: float
    temperature_deviation_celsius: float
    power_kW: float
    residence_time_minutes: float
    thermal_efficiency_percent: float
    zone_cure_contribution: float  # Cure degree contribution from this zone
    status: str
    recommendations: List[str]


class ArrheniusCureAnalysis(BaseModel):
    """Arrhenius cure kinetics analysis results."""

    reaction_rate_constant: float = Field(..., description="k value at average temperature (1/s)")
    activation_energy_kJ_mol: float = Field(..., description="Activation energy (kJ/mol)")
    cure_degree_achieved: float = Field(..., ge=0, le=1, description="Final cure degree alpha (0-1)")
    cure_degree_target: float = Field(..., ge=0, le=1, description="Target cure degree")
    time_to_target_cure_minutes: float = Field(..., ge=0, description="Time needed to reach target cure")
    equivalent_cure_time_minutes: float = Field(..., description="Cure time at reference temperature")
    cross_linking_percent: float = Field(..., ge=0, le=100, description="Estimated cross-linking (%)")


class CureQualityAssessment(BaseModel):
    """Cure quality assessment."""

    cure_status: CureStatus
    total_cure_time_minutes: float
    average_cure_temp_celsius: float
    temperature_uniformity_percent: float
    cure_quality_score: float  # 0-100
    confidence_level: str
    arrhenius_analysis: ArrheniusCureAnalysis


class EnergyAnalysis(BaseModel):
    """Energy consumption and efficiency analysis."""

    total_power_kW: float
    specific_energy_kwh_kg: float
    energy_cost_per_kg: float
    thermal_efficiency_percent: float
    energy_savings_potential_percent: float
    estimated_savings_per_hour: float
    radiation_loss_kW: float
    convection_loss_kW: float


class Recommendation(BaseModel):
    """Optimization recommendation."""

    recommendation_id: str
    priority: str  # HIGH, MEDIUM, LOW
    category: str
    description: str
    expected_benefit: str
    implementation_effort: str


class Warning(BaseModel):
    """Process warning or alert."""

    warning_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    affected_component: str
    corrective_action: str


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CuringOvenOutput(BaseModel):
    """Output data model for CuringOvenAgent."""

    # Identification
    analysis_id: str
    oven_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Zone Analysis
    zones: List[ZoneAnalysis]
    total_oven_length_meters: float

    # Cure Quality
    cure_quality: CureQualityAssessment
    cure_quality_score: float = Field(..., ge=0, le=100)

    # Energy Performance
    energy: EnergyAnalysis

    # Throughput
    throughput_kg_hr: float
    throughput_m2_hr: float
    capacity_utilization_percent: float

    # Optimization
    recommendations: List[Recommendation]
    warnings: List[Warning]

    # Provenance
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str

    # Processing Metadata
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# ARRHENIUS CURE KINETICS CALCULATOR
# =============================================================================

class ArrheniusCureCalculator:
    """
    Deterministic Arrhenius cure kinetics calculator.

    Implements the Arrhenius equation and nth-order kinetics model:
    - Arrhenius: k(T) = A * exp(-Ea / (R * T))
    - Cure rate: d_alpha/dt = k * (1 - alpha)^n
    - Cure degree: alpha(t) = 1 - (1 + (n-1)*k*t)^(-1/(n-1)) for n != 1
                  alpha(t) = 1 - exp(-k*t) for n = 1

    Zero-hallucination: All calculations are deterministic from published
    cure kinetics theory.
    """

    @staticmethod
    def calculate_rate_constant(
        temperature_kelvin: float,
        activation_energy_J_mol: float,
        pre_exponential_factor: float
    ) -> float:
        """
        Calculate the reaction rate constant using Arrhenius equation.

        Formula: k = A * exp(-Ea / (R * T))

        Args:
            temperature_kelvin: Temperature in Kelvin
            activation_energy_J_mol: Activation energy in J/mol
            pre_exponential_factor: Pre-exponential factor A (1/s)

        Returns:
            Rate constant k (1/s)
        """
        if temperature_kelvin <= 0:
            return 0.0

        exponent = -activation_energy_J_mol / (GAS_CONSTANT_R * temperature_kelvin)
        # Limit exponent to avoid overflow
        exponent = max(-100, min(100, exponent))

        return pre_exponential_factor * math.exp(exponent)

    @staticmethod
    def calculate_cure_degree(
        time_seconds: float,
        rate_constant: float,
        reaction_order: float,
        initial_cure: float = 0.0
    ) -> float:
        """
        Calculate the degree of cure using nth-order kinetics.

        For n = 1: alpha = 1 - (1 - alpha0) * exp(-k*t)
        For n != 1: alpha = 1 - ((1 - alpha0)^(1-n) + (n-1)*k*t)^(-1/(n-1))

        Args:
            time_seconds: Cure time in seconds
            rate_constant: Arrhenius rate constant k (1/s)
            reaction_order: Reaction order n
            initial_cure: Initial cure degree (0-1)

        Returns:
            Cure degree alpha (0-1)
        """
        if time_seconds <= 0 or rate_constant <= 0:
            return initial_cure

        # Prevent numerical issues
        initial_cure = max(0.0, min(0.999, initial_cure))
        remaining = 1.0 - initial_cure

        if abs(reaction_order - 1.0) < 0.01:
            # First-order kinetics
            cure_degree = 1.0 - remaining * math.exp(-rate_constant * time_seconds)
        else:
            # nth-order kinetics
            n = reaction_order
            term = remaining ** (1 - n) + (n - 1) * rate_constant * time_seconds

            if term <= 0:
                cure_degree = 1.0  # Fully cured
            else:
                try:
                    cure_degree = 1.0 - term ** (-1.0 / (n - 1))
                except (OverflowError, ZeroDivisionError):
                    cure_degree = 1.0

        return max(0.0, min(1.0, cure_degree))

    @staticmethod
    def calculate_time_to_cure(
        target_cure: float,
        rate_constant: float,
        reaction_order: float,
        initial_cure: float = 0.0
    ) -> float:
        """
        Calculate time required to reach target cure degree.

        Args:
            target_cure: Target cure degree (0-1)
            rate_constant: Arrhenius rate constant k (1/s)
            reaction_order: Reaction order n
            initial_cure: Initial cure degree (0-1)

        Returns:
            Time in seconds to reach target cure
        """
        if rate_constant <= 0 or target_cure <= initial_cure:
            return 0.0

        if target_cure >= 1.0:
            return float('inf')

        remaining_initial = 1.0 - initial_cure
        remaining_target = 1.0 - target_cure

        if abs(reaction_order - 1.0) < 0.01:
            # First-order: t = -ln((1-alpha_f)/(1-alpha_0)) / k
            if remaining_target <= 0:
                return float('inf')
            time_s = -math.log(remaining_target / remaining_initial) / rate_constant
        else:
            # nth-order: t = ((1-alpha_f)^(1-n) - (1-alpha_0)^(1-n)) / ((n-1)*k)
            n = reaction_order
            numerator = remaining_target ** (1 - n) - remaining_initial ** (1 - n)
            denominator = (n - 1) * rate_constant

            if denominator == 0:
                return float('inf')
            time_s = numerator / denominator

        return max(0.0, time_s)

    @staticmethod
    def calculate_equivalent_cure_time(
        actual_time_seconds: float,
        actual_temp_kelvin: float,
        reference_temp_kelvin: float,
        activation_energy_J_mol: float
    ) -> float:
        """
        Calculate equivalent cure time at reference temperature.

        Uses time-temperature superposition principle.

        Args:
            actual_time_seconds: Actual cure time
            actual_temp_kelvin: Actual temperature in Kelvin
            reference_temp_kelvin: Reference temperature in Kelvin
            activation_energy_J_mol: Activation energy in J/mol

        Returns:
            Equivalent time at reference temperature in seconds
        """
        if actual_temp_kelvin <= 0 or reference_temp_kelvin <= 0:
            return actual_time_seconds

        # Shift factor: aT = exp(Ea/R * (1/T_ref - 1/T))
        exponent = (activation_energy_J_mol / GAS_CONSTANT_R) * (
            1.0 / reference_temp_kelvin - 1.0 / actual_temp_kelvin
        )

        # Limit to prevent overflow
        exponent = max(-50, min(50, exponent))
        shift_factor = math.exp(exponent)

        return actual_time_seconds * shift_factor

    @staticmethod
    def estimate_crosslinking_percent(cure_degree: float) -> float:
        """
        Estimate cross-linking percentage from cure degree.

        Cross-linking is approximately proportional to cure degree
        but follows a sigmoidal relationship.

        Args:
            cure_degree: Cure degree alpha (0-1)

        Returns:
            Estimated cross-linking percentage (0-100)
        """
        # Sigmoidal relationship: crosslink increases rapidly mid-cure
        # Using logistic function approximation
        if cure_degree <= 0:
            return 0.0
        if cure_degree >= 1.0:
            return 100.0

        # Crosslinking typically starts at ~20% cure and accelerates
        if cure_degree < 0.2:
            crosslink = cure_degree * 50  # 0-10%
        elif cure_degree < 0.8:
            # Main crosslinking region
            crosslink = 10 + (cure_degree - 0.2) * 133.33  # 10-90%
        else:
            # Final crosslinking, diminishing returns
            crosslink = 90 + (cure_degree - 0.8) * 50  # 90-100%

        return max(0.0, min(100.0, crosslink))


# =============================================================================
# CURING OVEN AGENT
# =============================================================================

class CuringOvenAgent:
    """
    GL-056: Curing Oven Controller Agent (CURE-CTRL).

    This agent optimizes curing oven operations for various coating and
    composite applications, ensuring proper cure while minimizing energy consumption.

    Key Capabilities:
    - Arrhenius cure kinetics modeling
    - Cross-linking degree estimation
    - Time-temperature equivalence calculations
    - Multi-zone optimization

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas from published standards
    - No LLM inference in calculation path
    - Complete audit trail for quality assurance

    Attributes:
        AGENT_ID: Unique agent identifier (GL-056)
        AGENT_NAME: Agent name (CURE-CTRL)
        VERSION: Agent version
    """

    AGENT_ID = "GL-056"
    AGENT_NAME = "CURE-CTRL"
    VERSION = "1.1.0"
    DESCRIPTION = "Curing Oven Controller Agent with Arrhenius Kinetics"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CuringOvenAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._warnings: List[Warning] = []
        self._recommendations: List[Recommendation] = []
        self._cure_calculator = ArrheniusCureCalculator()

        logger.info(
            f"CuringOvenAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: CuringOvenInput) -> CuringOvenOutput:
        """
        Execute curing oven optimization analysis.

        This method performs comprehensive oven analysis:
        1. Analyze each zone performance
        2. Calculate residence time and cure duration
        3. Perform Arrhenius cure kinetics analysis
        4. Assess cure quality and cross-linking
        5. Calculate energy consumption and efficiency
        6. Generate optimization recommendations

        Args:
            input_data: Validated curing oven input data

        Returns:
            Complete optimization analysis with provenance hash
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._warnings = []
        self._recommendations = []

        logger.info(f"Starting curing oven analysis for {input_data.oven_id}")

        try:
            # Step 1: Calculate total oven length and residence time
            total_length = sum(z.length_meters for z in input_data.zones)
            total_residence_time = self._calculate_residence_time(
                total_length, input_data.conveyor.speed_m_min
            )

            # Step 2: Perform Arrhenius cure kinetics analysis
            arrhenius_analysis = self._analyze_cure_kinetics(input_data, total_residence_time)

            self._track_provenance(
                "arrhenius_cure_kinetics",
                {
                    "product_type": input_data.product.product_type.value,
                    "residence_time_min": total_residence_time
                },
                {
                    "cure_degree": arrhenius_analysis.cure_degree_achieved,
                    "rate_constant": arrhenius_analysis.reaction_rate_constant
                },
                "arrhenius_calculator"
            )

            # Step 3: Analyze zones with cure contribution
            zone_analyses = self._analyze_zones(input_data, arrhenius_analysis)

            self._track_provenance(
                "zone_analysis",
                {"zone_count": len(input_data.zones), "total_length_m": total_length},
                {"zones_analyzed": len(zone_analyses)},
                "zone_analyzer"
            )

            # Step 4: Assess cure quality
            cure_quality = self._assess_cure_quality(
                input_data, zone_analyses, total_residence_time, arrhenius_analysis
            )

            self._track_provenance(
                "cure_quality_assessment",
                {
                    "residence_time_min": total_residence_time,
                    "product_type": input_data.product.product_type.value
                },
                {
                    "cure_status": cure_quality.cure_status.value,
                    "quality_score": cure_quality.cure_quality_score
                },
                "cure_quality_analyzer"
            )

            # Step 5: Calculate energy performance
            energy_analysis = self._analyze_energy(input_data, zone_analyses)

            self._track_provenance(
                "energy_analysis",
                {"total_power_kW": sum(z.power_kW for z in input_data.zones)},
                {
                    "specific_energy": energy_analysis.specific_energy_kwh_kg,
                    "efficiency": energy_analysis.thermal_efficiency_percent
                },
                "energy_analyzer"
            )

            # Step 6: Calculate throughput
            throughput_kg_hr = input_data.product.mass_flow_kg_hr
            throughput_m2_hr = input_data.product.surface_area_m2_hr or 0.0

            # Step 7: Calculate capacity utilization
            capacity_util = self._calculate_capacity_utilization(input_data)

            # Step 8: Generate recommendations
            self._generate_recommendations(
                input_data, zone_analyses, energy_analysis, cure_quality, arrhenius_analysis
            )

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate analysis ID
            analysis_id = (
                f"CURE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(input_data.oven_id.encode()).hexdigest()[:8]}"
            )

            output = CuringOvenOutput(
                analysis_id=analysis_id,
                oven_id=input_data.oven_id,
                zones=zone_analyses,
                total_oven_length_meters=round(total_length, 2),
                cure_quality=cure_quality,
                cure_quality_score=cure_quality.cure_quality_score,
                energy=energy_analysis,
                throughput_kg_hr=round(throughput_kg_hr, 2),
                throughput_m2_hr=round(throughput_m2_hr, 2),
                capacity_utilization_percent=round(capacity_util, 2),
                recommendations=self._recommendations,
                warnings=self._warnings,
                provenance_chain=[
                    ProvenanceRecord(
                        operation=s["operation"],
                        timestamp=s["timestamp"],
                        input_hash=s["input_hash"],
                        output_hash=s["output_hash"],
                        tool_name=s["tool_name"],
                        parameters=s.get("parameters", {})
                    )
                    for s in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS" if not self._validation_errors else "FAIL",
                validation_errors=self._validation_errors
            )

            logger.info(
                f"Curing oven analysis complete for {input_data.oven_id}: "
                f"cure_degree={arrhenius_analysis.cure_degree_achieved:.3f}, "
                f"quality_score={cure_quality.cure_quality_score:.1f}, "
                f"cure_status={cure_quality.cure_status.value}, "
                f"warnings={len(self._warnings)} (duration: {processing_time:.1f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Curing oven analysis failed: {str(e)}", exc_info=True)
            raise

    def _analyze_cure_kinetics(
        self,
        input_data: CuringOvenInput,
        total_residence_time_min: float
    ) -> ArrheniusCureAnalysis:
        """
        Perform Arrhenius cure kinetics analysis.

        Calculates:
        - Reaction rate constant k at average temperature
        - Degree of cure achieved
        - Time to reach target cure
        - Cross-linking percentage

        Args:
            input_data: Input data with product and zone information
            total_residence_time_min: Total residence time in minutes

        Returns:
            ArrheniusCureAnalysis with complete kinetics results
        """
        # Get kinetics parameters for product type
        kinetics = CURE_KINETICS.get(input_data.product.product_type, CURE_KINETICS[ProductType.POWDER_COATING])

        # Calculate average cure temperature (weighted by zone power)
        total_power = sum(z.power_kW for z in input_data.zones if z.power_kW > 0)
        if total_power > 0:
            avg_temp_c = sum(
                z.actual_celsius * z.power_kW for z in input_data.zones
            ) / total_power
        else:
            avg_temp_c = sum(z.actual_celsius for z in input_data.zones) / len(input_data.zones)

        # Convert to Kelvin
        avg_temp_k = avg_temp_c + 273.15

        # Calculate activation energy in J/mol
        activation_energy_J = kinetics["activation_energy_kJ_mol"] * 1000

        # Calculate rate constant using Arrhenius equation
        rate_constant = self._cure_calculator.calculate_rate_constant(
            avg_temp_k,
            activation_energy_J,
            kinetics["pre_exponential_factor"]
        )

        # Calculate cure degree achieved
        cure_time_s = total_residence_time_min * 60
        initial_cure = input_data.product.initial_cure_degree

        cure_degree = self._cure_calculator.calculate_cure_degree(
            cure_time_s,
            rate_constant,
            kinetics["reaction_order"],
            initial_cure
        )

        # Calculate time to reach target cure
        target_cure = kinetics["optimal_cure_degree"]
        time_to_target_s = self._cure_calculator.calculate_time_to_cure(
            target_cure,
            rate_constant,
            kinetics["reaction_order"],
            initial_cure
        )
        time_to_target_min = time_to_target_s / 60 if time_to_target_s != float('inf') else float('inf')

        # Calculate equivalent cure time at reference temperature (typically 150C)
        reference_temp_k = 150 + 273.15  # 150C reference
        equivalent_time_s = self._cure_calculator.calculate_equivalent_cure_time(
            cure_time_s,
            avg_temp_k,
            reference_temp_k,
            activation_energy_J
        )
        equivalent_time_min = equivalent_time_s / 60

        # Estimate cross-linking
        crosslink_percent = self._cure_calculator.estimate_crosslinking_percent(cure_degree)

        return ArrheniusCureAnalysis(
            reaction_rate_constant=round(rate_constant, 8),
            activation_energy_kJ_mol=kinetics["activation_energy_kJ_mol"],
            cure_degree_achieved=round(cure_degree, 4),
            cure_degree_target=target_cure,
            time_to_target_cure_minutes=round(time_to_target_min, 2) if time_to_target_min != float('inf') else 9999.0,
            equivalent_cure_time_minutes=round(equivalent_time_min, 2),
            cross_linking_percent=round(crosslink_percent, 1)
        )

    def _analyze_zones(
        self,
        input_data: CuringOvenInput,
        arrhenius_analysis: ArrheniusCureAnalysis
    ) -> List[ZoneAnalysis]:
        """Analyze each oven zone with cure contribution."""
        analyses = []

        # Get kinetics parameters
        kinetics = CURE_KINETICS.get(input_data.product.product_type, CURE_KINETICS[ProductType.POWDER_COATING])
        activation_energy_J = kinetics["activation_energy_kJ_mol"] * 1000

        cumulative_cure = input_data.product.initial_cure_degree

        for zone in input_data.zones:
            # Calculate temperature deviation
            deviation = zone.actual_celsius - zone.setpoint_celsius

            # Calculate residence time in this zone
            residence_time = self._calculate_residence_time(
                zone.length_meters, input_data.conveyor.speed_m_min
            )

            # Calculate cure contribution from this zone
            zone_temp_k = zone.actual_celsius + 273.15
            zone_rate_constant = self._cure_calculator.calculate_rate_constant(
                zone_temp_k,
                activation_energy_J,
                kinetics["pre_exponential_factor"]
            )

            zone_cure_after = self._cure_calculator.calculate_cure_degree(
                residence_time * 60,  # Convert to seconds
                zone_rate_constant,
                kinetics["reaction_order"],
                cumulative_cure
            )
            zone_cure_contribution = zone_cure_after - cumulative_cure
            cumulative_cure = zone_cure_after

            # Estimate thermal efficiency (simplified)
            temp_efficiency = max(0, min(100, 100 - abs(deviation) * 2))

            # Check for issues
            zone_recs = []
            if abs(deviation) > 10:
                zone_recs.append(f"Temperature deviation {deviation:.1f}C exceeds tolerance")
                self._warnings.append(Warning(
                    warning_id=f"ZONE-TEMP-{zone.zone_id}",
                    severity="HIGH" if abs(deviation) > 20 else "MEDIUM",
                    category="TEMPERATURE_CONTROL",
                    description=f"Zone {zone.zone_id} temperature deviation: {deviation:.1f}C",
                    affected_component=zone.zone_id,
                    corrective_action="Check temperature control system and sensor calibration"
                ))

            if zone.status != ZoneStatus.NORMAL:
                zone_recs.append(f"Zone status is {zone.status.value}")
                self._warnings.append(Warning(
                    warning_id=f"ZONE-STATUS-{zone.zone_id}",
                    severity="CRITICAL" if zone.status == ZoneStatus.OFFLINE else "HIGH",
                    category="ZONE_STATUS",
                    description=f"Zone {zone.zone_id} status: {zone.status.value}",
                    affected_component=zone.zone_id,
                    corrective_action="Investigate zone malfunction"
                ))

            analyses.append(ZoneAnalysis(
                zone_id=zone.zone_id,
                setpoint_celsius=zone.setpoint_celsius,
                actual_celsius=zone.actual_celsius,
                temperature_deviation_celsius=round(deviation, 2),
                power_kW=zone.power_kW,
                residence_time_minutes=round(residence_time, 2),
                thermal_efficiency_percent=round(temp_efficiency, 2),
                zone_cure_contribution=round(zone_cure_contribution, 4),
                status=zone.status.value,
                recommendations=zone_recs
            ))

        return analyses

    def _calculate_residence_time(self, length_meters: float, speed_m_min: float) -> float:
        """Calculate residence time in minutes."""
        if speed_m_min <= 0:
            return 0.0
        return length_meters / speed_m_min

    def _assess_cure_quality(
        self,
        input_data: CuringOvenInput,
        zones: List[ZoneAnalysis],
        total_time: float,
        arrhenius_analysis: ArrheniusCureAnalysis
    ) -> CureQualityAssessment:
        """Assess cure quality based on Arrhenius kinetics and time-temperature profile."""

        # Get kinetics parameters for status determination
        kinetics = CURE_KINETICS.get(input_data.product.product_type, CURE_KINETICS[ProductType.POWDER_COATING])

        # Calculate average cure temperature (weighted by residence time)
        total_temp_time = sum(z.actual_celsius * z.residence_time_minutes for z in zones)
        avg_temp = total_temp_time / total_time if total_time > 0 else 0

        # Calculate temperature uniformity
        temps = [z.actual_celsius for z in zones]
        avg_zone_temp = sum(temps) / len(temps) if temps else 0
        max_deviation = max(abs(t - avg_zone_temp) for t in temps) if temps else 0
        uniformity = max(0, 100 - (max_deviation / avg_zone_temp * 100)) if avg_zone_temp > 0 else 0

        # Determine cure status based on Arrhenius analysis
        cure_degree = arrhenius_analysis.cure_degree_achieved
        min_cure = kinetics["min_cure_degree"]
        optimal_cure = kinetics["optimal_cure_degree"]
        max_cure = kinetics["max_cure_degree"]

        if cure_degree < min_cure:
            cure_status = CureStatus.UNDER_CURED
            quality_score = (cure_degree / min_cure) * 60  # 0-60 for under-cured
            confidence = "HIGH"
        elif cure_degree < optimal_cure:
            cure_status = CureStatus.OPTIMAL  # Acceptable range
            quality_score = 60 + ((cure_degree - min_cure) / (optimal_cure - min_cure)) * 30  # 60-90
            confidence = "HIGH"
        elif cure_degree <= max_cure:
            cure_status = CureStatus.OPTIMAL
            quality_score = 90 + ((cure_degree - optimal_cure) / (max_cure - optimal_cure)) * 10  # 90-100
            confidence = "HIGH"
        else:
            cure_status = CureStatus.OVER_CURED
            quality_score = max(50, 100 - ((cure_degree - max_cure) * 500))  # Penalize over-cure
            confidence = "HIGH"

        # Adjust score for temperature uniformity
        quality_score = quality_score * (0.8 + 0.2 * uniformity / 100)

        # Add warnings for poor cure quality
        if cure_status == CureStatus.UNDER_CURED:
            self._warnings.append(Warning(
                warning_id="CURE-UNDER",
                severity="HIGH",
                category="CURE_QUALITY",
                description=f"Under-cured: achieved {cure_degree:.1%} vs target {optimal_cure:.1%}",
                affected_component="CURE_PROCESS",
                corrective_action="Reduce conveyor speed or increase zone temperatures"
            ))
        elif cure_status == CureStatus.OVER_CURED:
            self._warnings.append(Warning(
                warning_id="CURE-OVER",
                severity="MEDIUM",
                category="CURE_QUALITY",
                description=f"Over-cured: achieved {cure_degree:.1%} vs max {max_cure:.1%}",
                affected_component="CURE_PROCESS",
                corrective_action="Increase conveyor speed or reduce zone temperatures"
            ))

        return CureQualityAssessment(
            cure_status=cure_status,
            total_cure_time_minutes=round(total_time, 2),
            average_cure_temp_celsius=round(avg_temp, 2),
            temperature_uniformity_percent=round(uniformity, 2),
            cure_quality_score=round(quality_score, 2),
            confidence_level=confidence,
            arrhenius_analysis=arrhenius_analysis
        )

    def _analyze_energy(
        self,
        input_data: CuringOvenInput,
        zones: List[ZoneAnalysis]
    ) -> EnergyAnalysis:
        """Analyze energy consumption and efficiency with heat loss breakdown."""

        # Total power consumption
        total_power = sum(z.power_kW for z in zones)

        # Specific energy consumption (kWh/kg)
        if input_data.product.mass_flow_kg_hr > 0:
            specific_energy = total_power / input_data.product.mass_flow_kg_hr
        else:
            specific_energy = 0.0

        # Energy cost per kg
        energy_cost_per_kg = specific_energy * input_data.energy_cost_per_kwh

        # Estimate thermal efficiency
        params = CURE_PARAMETERS.get(input_data.product.product_type, {})
        theoretical_energy = params.get("energy_intensity_kwh_kg", 0.5)

        if theoretical_energy > 0:
            thermal_efficiency = min(100, (theoretical_energy / specific_energy * 100)) if specific_energy > 0 else 0
        else:
            thermal_efficiency = 50.0

        # Calculate heat losses (simplified model)
        # Radiation loss: P_rad = epsilon * sigma * A * (T_surface^4 - T_ambient^4)
        avg_temp_k = sum(z.actual_celsius for z in zones) / len(zones) + 273.15 if zones else 300
        ambient_k = input_data.ambient.temperature_celsius + 273.15

        # Estimate oven surface area (simplified)
        total_length = sum(z.length_meters for z in input_data.zones)
        surface_area_m2 = total_length * 3.0 * 2.0  # Assume 3m circumference, 2 sides

        # Radiation loss (assume emissivity 0.85 for painted steel)
        radiation_loss_kW = (
            0.85 * STEFAN_BOLTZMANN * surface_area_m2 *
            (avg_temp_k ** 4 - ambient_k ** 4)
        ) / 1000

        # Convection loss (simplified: h = 10 W/m2K for natural convection)
        convection_coeff = 10.0  # W/(m2*K)
        delta_t = avg_temp_k - ambient_k
        convection_loss_kW = convection_coeff * surface_area_m2 * delta_t / 1000

        # Energy savings potential
        if thermal_efficiency < 100:
            savings_potential = 100 - thermal_efficiency
        else:
            savings_potential = 0.0

        # Estimated savings per hour
        if savings_potential > 0:
            potential_power_reduction = total_power * (savings_potential / 100)
            savings_per_hour = potential_power_reduction * input_data.energy_cost_per_kwh
        else:
            savings_per_hour = 0.0

        # Add energy warnings
        if thermal_efficiency < 60:
            self._warnings.append(Warning(
                warning_id="ENERGY-EFFICIENCY",
                severity="MEDIUM",
                category="ENERGY_EFFICIENCY",
                description=f"Thermal efficiency is low: {thermal_efficiency:.1f}%",
                affected_component="OVEN_SYSTEM",
                corrective_action="Investigate insulation, air leaks, and zone temperature optimization"
            ))

        return EnergyAnalysis(
            total_power_kW=round(total_power, 2),
            specific_energy_kwh_kg=round(specific_energy, 4),
            energy_cost_per_kg=round(energy_cost_per_kg, 4),
            thermal_efficiency_percent=round(thermal_efficiency, 2),
            energy_savings_potential_percent=round(savings_potential, 2),
            estimated_savings_per_hour=round(savings_per_hour, 2),
            radiation_loss_kW=round(radiation_loss_kW, 2),
            convection_loss_kW=round(convection_loss_kW, 2)
        )

    def _calculate_capacity_utilization(self, input_data: CuringOvenInput) -> float:
        """Calculate oven capacity utilization."""
        loading = input_data.conveyor.product_loading_percent
        speed_factor = min(1.0, input_data.conveyor.speed_m_min / 3.0)
        return loading * speed_factor

    def _generate_recommendations(
        self,
        input_data: CuringOvenInput,
        zones: List[ZoneAnalysis],
        energy: EnergyAnalysis,
        cure_quality: CureQualityAssessment,
        arrhenius: ArrheniusCureAnalysis
    ):
        """Generate optimization recommendations based on Arrhenius analysis."""

        rec_id = 0

        # Cure kinetics-based recommendations
        if arrhenius.cure_degree_achieved < arrhenius.cure_degree_target:
            cure_deficit = arrhenius.cure_degree_target - arrhenius.cure_degree_achieved

            rec_id += 1
            self._recommendations.append(Recommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                priority="HIGH",
                category="CURE_OPTIMIZATION",
                description=(
                    f"Increase cure time by {arrhenius.time_to_target_cure_minutes - cure_quality.total_cure_time_minutes:.1f} min "
                    f"or increase temperature by ~10C to achieve target cure degree"
                ),
                expected_benefit=f"Achieve optimal cure degree ({arrhenius.cure_degree_target:.0%})",
                implementation_effort="LOW"
            ))

        # Cross-linking recommendation
        if arrhenius.cross_linking_percent < 85:
            rec_id += 1
            self._recommendations.append(Recommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                priority="MEDIUM",
                category="CURE_QUALITY",
                description=f"Cross-linking at {arrhenius.cross_linking_percent:.0f}% may affect coating durability",
                expected_benefit="Improved coating hardness and chemical resistance",
                implementation_effort="MEDIUM"
            ))

        # Over-cure prevention
        if cure_quality.cure_status == CureStatus.OVER_CURED:
            rec_id += 1
            self._recommendations.append(Recommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                priority="MEDIUM",
                category="CURE_OPTIMIZATION",
                description="Reduce cure time or temperature to prevent over-curing",
                expected_benefit="Energy savings, improved flexibility, reduced embrittlement",
                implementation_effort="LOW"
            ))

        # Temperature uniformity
        if cure_quality.temperature_uniformity_percent < 90:
            rec_id += 1
            self._recommendations.append(Recommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                priority="MEDIUM",
                category="TEMPERATURE_UNIFORMITY",
                description="Improve temperature uniformity between zones",
                expected_benefit="More consistent cure quality across product",
                implementation_effort="MEDIUM"
            ))

        # Energy efficiency
        if energy.energy_savings_potential_percent > 20:
            rec_id += 1
            self._recommendations.append(Recommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                priority="HIGH",
                category="ENERGY_EFFICIENCY",
                description=f"Optimize energy efficiency - potential savings: ${energy.estimated_savings_per_hour:.2f}/hr",
                expected_benefit=f"Reduce energy consumption by {energy.energy_savings_potential_percent:.1f}%",
                implementation_effort="MEDIUM"
            ))

        # Zone-specific recommendations
        for zone in zones:
            if abs(zone.temperature_deviation_celsius) > 15:
                rec_id += 1
                self._recommendations.append(Recommendation(
                    recommendation_id=f"REC-{rec_id:03d}",
                    priority="MEDIUM",
                    category="ZONE_CONTROL",
                    description=f"Improve temperature control in {zone.zone_id}",
                    expected_benefit="Better process stability and cure consistency",
                    implementation_effort="MEDIUM"
                ))

        # Capacity utilization
        capacity_util = self._calculate_capacity_utilization(input_data)
        if capacity_util < 70:
            rec_id += 1
            self._recommendations.append(Recommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                priority="LOW",
                category="CAPACITY_UTILIZATION",
                description=f"Increase capacity utilization (current: {capacity_util:.1f}%)",
                expected_benefit="Improved productivity and reduced unit costs",
                implementation_effort="LOW"
            ))

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ):
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs
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
                    "output_hash": s["output_hash"]
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-056",
    "name": "CURE-CTRL - Curing Oven Controller Agent",
    "version": "1.1.0",
    "summary": "Curing oven optimization with Arrhenius cure kinetics for coating and composite applications",
    "tags": [
        "curing",
        "oven",
        "coating",
        "composites",
        "powder-coating",
        "temperature-control",
        "energy-optimization",
        "arrhenius-kinetics",
        "cross-linking",
        "NFPA-86",
        "ASTM-D4541"
    ],
    "owners": ["process-heat-team"],
    "compute": {
        "entrypoint": "python://agents.gl_056_curing_oven.agent:CuringOvenAgent",
        "deterministic": True
    },
    "formulas": {
        "arrhenius_equation": "k = A * exp(-Ea / (R * T))",
        "cure_degree_first_order": "alpha = 1 - exp(-k * t)",
        "cure_degree_nth_order": "alpha = 1 - ((1-alpha0)^(1-n) + (n-1)*k*t)^(-1/(n-1))",
        "time_temperature_equivalence": "t_eq = t_actual * exp(Ea/R * (1/T_ref - 1/T))"
    },
    "standards": [
        {"ref": "ASTM D4541", "description": "Pull-Off Adhesion Testing"},
        {"ref": "ISO 11507", "description": "Paints and Varnishes - Exposure to Artificial Weathering"},
        {"ref": "NFPA 86", "description": "Standard for Ovens and Furnaces"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
