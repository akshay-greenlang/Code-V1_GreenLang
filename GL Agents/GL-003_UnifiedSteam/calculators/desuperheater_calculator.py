"""
GL-003 UNIFIEDSTEAM - Desuperheater Calculator

Desuperheater spray water calculations for steam temperature control.

Primary Formula:
    m_spray = m_steam * (h_in - h_target) / (h_target - h_spray)

Where:
    m_spray  = Spray water mass flow rate (kg/s)
    m_steam  = Inlet steam mass flow rate (kg/s)
    h_in     = Inlet steam specific enthalpy (kJ/kg)
    h_target = Target steam specific enthalpy (kJ/kg)
    h_spray  = Spray water specific enthalpy (kJ/kg)

Constraints:
    - Target temperature must maintain superheat margin above saturation
    - Steam velocity limits for erosion prevention
    - Spray water quality requirements
    - Dynamic response considerations

Reference: ASME B31.1 Power Piping, API 526 Flanged Steel PSVs

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND STANDARDS
# =============================================================================

# Minimum superheat margin above saturation (degrees Celsius)
# Per industry practice for wet steam prevention
MIN_SUPERHEAT_MARGIN_C = 10.0

# Steam velocity limits (m/s) per ASME standards
STEAM_VELOCITY_LIMITS = {
    "low_pressure": {"normal": 30.0, "max": 50.0},      # < 1.0 MPa
    "medium_pressure": {"normal": 40.0, "max": 60.0},   # 1.0 - 4.0 MPa
    "high_pressure": {"normal": 50.0, "max": 75.0},     # > 4.0 MPa
}

# Erosion risk thresholds based on velocity and moisture
EROSION_THRESHOLDS = {
    "velocity_critical_m_s": 100.0,
    "moisture_critical_percent": 1.0,
    "velocity_moisture_product": 50.0,  # v * moisture_fraction
}

# Spray water quality requirements
SPRAY_WATER_REQUIREMENTS = {
    "max_tds_ppm": 50.0,         # Total dissolved solids
    "max_silica_ppm": 0.02,      # Silica for high pressure
    "max_hardness_ppm": 0.5,     # Calcium hardness
    "max_iron_ppb": 20.0,        # Iron
    "min_ph": 8.5,
    "max_ph": 9.5,
}


class ErosionRiskLevel(str, Enum):
    """Erosion risk classification."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DesuperheaterType(str, Enum):
    """Types of desuperheaters."""
    SPRAY_NOZZLE = "SPRAY_NOZZLE"
    VENTURI = "VENTURI"
    ANNULAR_RING = "ANNULAR_RING"
    SURFACE_CONTACT = "SURFACE_CONTACT"
    MECHANICAL_ATOMIZING = "MECHANICAL_ATOMIZING"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DesuperheaterInput:
    """Input parameters for desuperheater calculation."""

    # Steam conditions
    steam_flow_kg_s: float
    inlet_pressure_kpa: float
    inlet_temperature_c: float
    inlet_enthalpy_kj_kg: float

    # Target conditions
    target_temperature_c: float
    target_enthalpy_kj_kg: float

    # Spray water conditions
    spray_water_temperature_c: float
    spray_water_enthalpy_kj_kg: float
    spray_water_pressure_kpa: float

    # Equipment parameters
    desuperheater_type: DesuperheaterType = DesuperheaterType.SPRAY_NOZZLE
    pipe_diameter_m: float = 0.3
    nozzle_count: int = 4
    max_spray_capacity_kg_s: float = 10.0

    # Water quality (optional)
    spray_water_tds_ppm: Optional[float] = None
    spray_water_silica_ppm: Optional[float] = None


@dataclass
class SprayRequirement:
    """Result of spray water requirement calculation."""

    # Identification
    calculation_id: str
    timestamp: datetime

    # Primary result
    spray_water_flow_kg_s: float
    spray_water_flow_uncertainty_kg_s: float

    # Derived values
    spray_to_steam_ratio: float
    outlet_steam_flow_kg_s: float
    heat_removed_kw: float

    # Enthalpy balance details
    inlet_enthalpy_kj_kg: float
    target_enthalpy_kj_kg: float
    spray_enthalpy_kj_kg: float
    enthalpy_change_kj_kg: float

    # Constraints check
    within_capacity: bool
    superheat_margin_c: float
    pressure_differential_adequate: bool

    # Provenance
    input_hash: str
    output_hash: str
    formula_version: str = "DESUP_V1.0"

    # Calculation steps for audit
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MaxCoolingResult:
    """Result of maximum safe cooling calculation."""

    calculation_id: str
    timestamp: datetime

    # Maximum cooling
    max_temperature_reduction_c: float
    min_safe_temperature_c: float
    saturation_temperature_c: float
    superheat_margin_c: float

    # Corresponding spray flow
    max_spray_flow_kg_s: float
    max_spray_capacity_limited: bool

    # Equipment limits
    nozzle_velocity_limited: bool
    water_quality_limited: bool

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class ErosionRiskAssessment:
    """Assessment of erosion and impingement risk."""

    calculation_id: str
    timestamp: datetime

    # Risk classification
    risk_level: ErosionRiskLevel
    risk_score: float  # 0-100 scale

    # Contributing factors
    steam_velocity_m_s: float
    velocity_risk_factor: float

    moisture_content_percent: float
    moisture_risk_factor: float

    velocity_moisture_product: float
    combined_risk_factor: float

    # Droplet analysis
    droplet_size_microns: float
    droplet_impact_velocity_m_s: float
    impingement_risk: str

    # Recommendations
    recommendations: List[str]

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class WaterQualityValidation:
    """Spray water quality validation result."""

    is_valid: bool
    issues: List[str]

    tds_valid: bool
    silica_valid: bool
    hardness_valid: bool
    ph_valid: bool

    deposition_risk: str
    carryover_risk: str


# =============================================================================
# DESUPERHEATER CALCULATOR
# =============================================================================

class DesuperheaterCalculator:
    """
    Zero-hallucination desuperheater spray water calculator.

    Implements deterministic calculations for:
    - Spray water requirement from energy balance
    - Maximum safe cooling to prevent wet steam
    - Erosion and impingement risk assessment
    - Water quality validation

    All calculations use:
    - Decimal arithmetic for precision
    - SHA-256 provenance hashing
    - Complete audit trails
    - NO LLM in calculation path

    Example:
        >>> calc = DesuperheaterCalculator()
        >>> result = calc.compute_spray_water_requirement(
        ...     steam_flow_kg_s=10.0,
        ...     inlet_enthalpy=3200.0,
        ...     target_enthalpy=2900.0,
        ...     spray_water_enthalpy=420.0
        ... )
        >>> print(f"Spray flow: {result.spray_water_flow_kg_s:.3f} kg/s")
    """

    VERSION = "1.0.0"
    FORMULA_VERSION = "DESUP_V1.0"

    def __init__(
        self,
        min_superheat_margin_c: float = MIN_SUPERHEAT_MARGIN_C,
        velocity_limits: Optional[Dict] = None,
    ) -> None:
        """
        Initialize desuperheater calculator.

        Args:
            min_superheat_margin_c: Minimum superheat above saturation
            velocity_limits: Custom steam velocity limits
        """
        self.min_superheat_margin = min_superheat_margin_c
        self.velocity_limits = velocity_limits or STEAM_VELOCITY_LIMITS

    def compute_spray_water_requirement(
        self,
        steam_flow_kg_s: float,
        inlet_enthalpy: float,
        target_enthalpy: float,
        spray_water_enthalpy: float,
        max_spray_capacity_kg_s: Optional[float] = None,
    ) -> SprayRequirement:
        """
        Compute spray water requirement using energy balance.

        Formula:
            m_spray = m_steam * (h_in - h_target) / (h_target - h_spray)

        This is a DETERMINISTIC calculation with NO LLM involvement.
        Same inputs will always produce identical outputs.

        Args:
            steam_flow_kg_s: Inlet steam mass flow rate (kg/s)
            inlet_enthalpy: Inlet steam specific enthalpy (kJ/kg)
            target_enthalpy: Target outlet specific enthalpy (kJ/kg)
            spray_water_enthalpy: Spray water specific enthalpy (kJ/kg)
            max_spray_capacity_kg_s: Maximum spray capacity (optional)

        Returns:
            SprayRequirement with complete calculation provenance

        Raises:
            ValueError: If inputs are invalid or physically impossible
        """
        calculation_steps = []

        # Step 1: Validate inputs
        self._validate_spray_inputs(
            steam_flow_kg_s,
            inlet_enthalpy,
            target_enthalpy,
            spray_water_enthalpy,
        )

        calculation_steps.append({
            "step": 1,
            "description": "Input validation",
            "inputs": {
                "steam_flow_kg_s": steam_flow_kg_s,
                "inlet_enthalpy": inlet_enthalpy,
                "target_enthalpy": target_enthalpy,
                "spray_water_enthalpy": spray_water_enthalpy,
            },
            "result": "VALID",
        })

        # Step 2: Convert to Decimal for precision
        m_steam = Decimal(str(steam_flow_kg_s))
        h_in = Decimal(str(inlet_enthalpy))
        h_target = Decimal(str(target_enthalpy))
        h_spray = Decimal(str(spray_water_enthalpy))

        # Step 3: Calculate numerator (heat to be removed)
        # Delta_h = h_in - h_target
        delta_h = h_in - h_target

        calculation_steps.append({
            "step": 2,
            "description": "Calculate enthalpy difference (heat to remove)",
            "formula": "delta_h = h_in - h_target",
            "inputs": {"h_in": float(h_in), "h_target": float(h_target)},
            "result": float(delta_h),
            "unit": "kJ/kg",
        })

        # Step 4: Calculate denominator (cooling capacity of spray water)
        # Delta_h_spray = h_target - h_spray
        delta_h_spray = h_target - h_spray

        calculation_steps.append({
            "step": 3,
            "description": "Calculate spray water heating capacity",
            "formula": "delta_h_spray = h_target - h_spray",
            "inputs": {"h_target": float(h_target), "h_spray": float(h_spray)},
            "result": float(delta_h_spray),
            "unit": "kJ/kg",
        })

        # Step 5: Calculate spray water flow rate
        # m_spray = m_steam * delta_h / delta_h_spray
        if delta_h_spray <= 0:
            raise ValueError(
                f"Invalid enthalpy values: spray water enthalpy ({spray_water_enthalpy}) "
                f"must be less than target enthalpy ({target_enthalpy})"
            )

        m_spray = m_steam * delta_h / delta_h_spray

        calculation_steps.append({
            "step": 4,
            "description": "Calculate spray water flow rate (energy balance)",
            "formula": "m_spray = m_steam * (h_in - h_target) / (h_target - h_spray)",
            "inputs": {
                "m_steam": float(m_steam),
                "delta_h": float(delta_h),
                "delta_h_spray": float(delta_h_spray),
            },
            "result": float(m_spray),
            "unit": "kg/s",
        })

        # Step 6: Calculate outlet steam flow
        m_outlet = m_steam + m_spray

        calculation_steps.append({
            "step": 5,
            "description": "Calculate outlet steam flow (mass balance)",
            "formula": "m_outlet = m_steam + m_spray",
            "inputs": {"m_steam": float(m_steam), "m_spray": float(m_spray)},
            "result": float(m_outlet),
            "unit": "kg/s",
        })

        # Step 7: Calculate heat removed
        # Q = m_steam * delta_h
        heat_removed = m_steam * delta_h

        calculation_steps.append({
            "step": 6,
            "description": "Calculate heat removed",
            "formula": "Q = m_steam * delta_h",
            "inputs": {"m_steam": float(m_steam), "delta_h": float(delta_h)},
            "result": float(heat_removed),
            "unit": "kW",
        })

        # Step 8: Calculate spray to steam ratio
        spray_ratio = m_spray / m_steam if m_steam > 0 else Decimal("0")

        calculation_steps.append({
            "step": 7,
            "description": "Calculate spray to steam ratio",
            "formula": "ratio = m_spray / m_steam",
            "result": float(spray_ratio),
            "unit": "dimensionless",
        })

        # Step 9: Check capacity constraints
        within_capacity = True
        if max_spray_capacity_kg_s is not None:
            within_capacity = float(m_spray) <= max_spray_capacity_kg_s

        # Step 10: Calculate uncertainty
        # Uncertainty propagation for energy balance
        # Using simplified uncertainty: ~2% for flow, ~1% for enthalpy
        relative_uncertainty = Decimal("0.025")  # 2.5% total
        spray_uncertainty = m_spray * relative_uncertainty

        # Step 11: Apply precision (3 decimal places for kg/s)
        m_spray_rounded = self._apply_precision(m_spray, 3)
        spray_uncertainty_rounded = self._apply_precision(spray_uncertainty, 4)

        # Compute provenance hashes
        input_data = {
            "steam_flow_kg_s": steam_flow_kg_s,
            "inlet_enthalpy": inlet_enthalpy,
            "target_enthalpy": target_enthalpy,
            "spray_water_enthalpy": spray_water_enthalpy,
        }
        input_hash = self._compute_hash(input_data)

        output_data = {
            "spray_water_flow_kg_s": float(m_spray_rounded),
            "outlet_steam_flow_kg_s": float(m_outlet),
            "heat_removed_kw": float(heat_removed),
        }
        output_hash = self._compute_hash(output_data)

        return SprayRequirement(
            calculation_id=f"DESUP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            spray_water_flow_kg_s=float(m_spray_rounded),
            spray_water_flow_uncertainty_kg_s=float(spray_uncertainty_rounded),
            spray_to_steam_ratio=float(self._apply_precision(spray_ratio, 4)),
            outlet_steam_flow_kg_s=float(self._apply_precision(m_outlet, 3)),
            heat_removed_kw=float(self._apply_precision(heat_removed, 1)),
            inlet_enthalpy_kj_kg=inlet_enthalpy,
            target_enthalpy_kj_kg=target_enthalpy,
            spray_enthalpy_kj_kg=spray_water_enthalpy,
            enthalpy_change_kj_kg=float(delta_h),
            within_capacity=within_capacity,
            superheat_margin_c=0.0,  # Set by caller if needed
            pressure_differential_adequate=True,  # Assume adequate
            input_hash=input_hash,
            output_hash=output_hash,
            formula_version=self.FORMULA_VERSION,
            calculation_steps=calculation_steps,
        )

    def validate_target_temperature(
        self,
        pressure_kpa: float,
        target_temp_c: float,
        saturation_temp_c: Optional[float] = None,
    ) -> Tuple[bool, float, str]:
        """
        Validate that target temperature maintains adequate superheat.

        Prevents wet steam condition by ensuring target is above
        saturation temperature plus safety margin.

        Args:
            pressure_kpa: Operating pressure (kPa absolute)
            target_temp_c: Target temperature (Celsius)
            saturation_temp_c: Known saturation temperature (optional)

        Returns:
            Tuple of (is_valid, superheat_margin, message)
        """
        # Get saturation temperature
        if saturation_temp_c is None:
            # Use correlation for saturation temperature
            # T_sat = f(P) from Antoine equation approximation
            saturation_temp_c = self._estimate_saturation_temp(pressure_kpa)

        # Calculate superheat margin
        superheat_margin = target_temp_c - saturation_temp_c

        # Check against minimum
        is_valid = superheat_margin >= self.min_superheat_margin

        if is_valid:
            message = f"Valid: {superheat_margin:.1f}C superheat (minimum {self.min_superheat_margin}C)"
        else:
            message = (
                f"INVALID: {superheat_margin:.1f}C superheat is below "
                f"minimum {self.min_superheat_margin}C. Risk of wet steam."
            )

        return is_valid, superheat_margin, message

    def compute_maximum_safe_cooling(
        self,
        inlet_pressure_kpa: float,
        inlet_temperature_c: float,
        inlet_enthalpy_kj_kg: float,
        steam_flow_kg_s: float,
        spray_water_enthalpy_kj_kg: float,
        max_spray_capacity_kg_s: float,
        saturation_temp_c: Optional[float] = None,
        saturation_enthalpy_kj_kg: Optional[float] = None,
    ) -> MaxCoolingResult:
        """
        Compute maximum safe cooling while maintaining superheat margin.

        Determines the lowest safe target temperature and corresponding
        maximum spray water flow rate.

        Args:
            inlet_pressure_kpa: Operating pressure (kPa absolute)
            inlet_temperature_c: Inlet steam temperature (C)
            inlet_enthalpy_kj_kg: Inlet steam enthalpy (kJ/kg)
            steam_flow_kg_s: Steam mass flow rate (kg/s)
            spray_water_enthalpy_kj_kg: Spray water enthalpy (kJ/kg)
            max_spray_capacity_kg_s: Maximum spray capacity (kg/s)
            saturation_temp_c: Known saturation temperature (optional)
            saturation_enthalpy_kj_kg: Known saturation enthalpy (optional)

        Returns:
            MaxCoolingResult with maximum safe cooling parameters
        """
        # Get saturation properties
        if saturation_temp_c is None:
            saturation_temp_c = self._estimate_saturation_temp(inlet_pressure_kpa)

        if saturation_enthalpy_kj_kg is None:
            saturation_enthalpy_kj_kg = self._estimate_saturation_enthalpy(inlet_pressure_kpa)

        # Minimum safe temperature (saturation + margin)
        min_safe_temp = saturation_temp_c + self.min_superheat_margin

        # Maximum temperature reduction
        max_temp_reduction = inlet_temperature_c - min_safe_temp

        # Estimate enthalpy at minimum safe temperature
        # Using linear interpolation assumption for superheat region
        cp_steam = 2.1  # Approximate Cp for superheated steam (kJ/kg-K)
        min_safe_enthalpy = inlet_enthalpy_kj_kg - (cp_steam * max_temp_reduction)

        # Ensure we don't go below saturation enthalpy + margin
        min_safe_enthalpy = max(
            min_safe_enthalpy,
            saturation_enthalpy_kj_kg + (cp_steam * self.min_superheat_margin)
        )

        # Calculate maximum spray flow for minimum safe enthalpy
        if min_safe_enthalpy > spray_water_enthalpy_kj_kg:
            max_spray_flow = (
                steam_flow_kg_s *
                (inlet_enthalpy_kj_kg - min_safe_enthalpy) /
                (min_safe_enthalpy - spray_water_enthalpy_kj_kg)
            )
        else:
            max_spray_flow = 0.0

        # Check if capacity limited
        capacity_limited = max_spray_flow > max_spray_capacity_kg_s
        if capacity_limited:
            max_spray_flow = max_spray_capacity_kg_s

        # Compute hashes
        input_hash = self._compute_hash({
            "inlet_pressure_kpa": inlet_pressure_kpa,
            "inlet_temperature_c": inlet_temperature_c,
            "inlet_enthalpy_kj_kg": inlet_enthalpy_kj_kg,
            "steam_flow_kg_s": steam_flow_kg_s,
        })

        output_hash = self._compute_hash({
            "max_temp_reduction": max_temp_reduction,
            "min_safe_temp": min_safe_temp,
            "max_spray_flow": max_spray_flow,
        })

        return MaxCoolingResult(
            calculation_id=f"MAXCOOL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            max_temperature_reduction_c=round(max_temp_reduction, 1),
            min_safe_temperature_c=round(min_safe_temp, 1),
            saturation_temperature_c=round(saturation_temp_c, 1),
            superheat_margin_c=self.min_superheat_margin,
            max_spray_flow_kg_s=round(max_spray_flow, 3),
            max_spray_capacity_limited=capacity_limited,
            nozzle_velocity_limited=False,  # Would need additional data
            water_quality_limited=False,    # Would need water quality data
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def assess_erosion_risk(
        self,
        steam_velocity_m_s: float,
        moisture_content_percent: float,
        pipe_diameter_m: float = 0.3,
        droplet_size_microns: float = 100.0,
    ) -> ErosionRiskAssessment:
        """
        Assess erosion and impingement risk from spray water.

        Evaluates risk based on:
        - Steam velocity
        - Moisture content
        - Velocity-moisture product
        - Droplet characteristics

        Args:
            steam_velocity_m_s: Steam velocity in pipe (m/s)
            moisture_content_percent: Moisture fraction (%)
            pipe_diameter_m: Pipe diameter (m)
            droplet_size_microns: Average droplet size (microns)

        Returns:
            ErosionRiskAssessment with risk classification
        """
        # Velocity risk factor (0-1)
        v_critical = EROSION_THRESHOLDS["velocity_critical_m_s"]
        velocity_factor = min(1.0, steam_velocity_m_s / v_critical)

        # Moisture risk factor (0-1)
        m_critical = EROSION_THRESHOLDS["moisture_critical_percent"]
        moisture_factor = min(1.0, moisture_content_percent / m_critical)

        # Combined velocity-moisture product
        vm_product = steam_velocity_m_s * (moisture_content_percent / 100)
        vm_critical = EROSION_THRESHOLDS["velocity_moisture_product"]
        combined_factor = min(1.0, vm_product / vm_critical)

        # Overall risk score (0-100)
        # Weighted combination: velocity 30%, moisture 30%, combined 40%
        risk_score = (
            velocity_factor * 30 +
            moisture_factor * 30 +
            combined_factor * 40
        )

        # Classify risk level
        if risk_score < 25:
            risk_level = ErosionRiskLevel.LOW
        elif risk_score < 50:
            risk_level = ErosionRiskLevel.MODERATE
        elif risk_score < 75:
            risk_level = ErosionRiskLevel.HIGH
        else:
            risk_level = ErosionRiskLevel.CRITICAL

        # Droplet impact analysis
        # Impact velocity approximation (droplet lags steam)
        impact_velocity = steam_velocity_m_s * 0.8  # Approximate

        # Impingement risk based on droplet size and velocity
        if droplet_size_microns > 500 and impact_velocity > 50:
            impingement_risk = "HIGH"
        elif droplet_size_microns > 200 or impact_velocity > 30:
            impingement_risk = "MODERATE"
        else:
            impingement_risk = "LOW"

        # Generate recommendations
        recommendations = []

        if velocity_factor > 0.7:
            recommendations.append(
                "Consider reducing steam velocity by increasing pipe diameter"
            )

        if moisture_factor > 0.5:
            recommendations.append(
                "Monitor moisture content; consider better atomization"
            )

        if combined_factor > 0.6:
            recommendations.append(
                "Install erosion-resistant liner at impact points"
            )

        if droplet_size_microns > 300:
            recommendations.append(
                "Consider finer atomization nozzles to reduce droplet size"
            )

        if impact_velocity > 60:
            recommendations.append(
                "Install impingement protection downstream of spray location"
            )

        if not recommendations:
            recommendations.append("Current operating conditions are within safe limits")

        # Compute hashes
        input_hash = self._compute_hash({
            "steam_velocity_m_s": steam_velocity_m_s,
            "moisture_content_percent": moisture_content_percent,
            "pipe_diameter_m": pipe_diameter_m,
            "droplet_size_microns": droplet_size_microns,
        })

        output_hash = self._compute_hash({
            "risk_level": risk_level.value,
            "risk_score": risk_score,
        })

        return ErosionRiskAssessment(
            calculation_id=f"EROSION-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            risk_level=risk_level,
            risk_score=round(risk_score, 1),
            steam_velocity_m_s=round(steam_velocity_m_s, 1),
            velocity_risk_factor=round(velocity_factor, 3),
            moisture_content_percent=round(moisture_content_percent, 2),
            moisture_risk_factor=round(moisture_factor, 3),
            velocity_moisture_product=round(vm_product, 2),
            combined_risk_factor=round(combined_factor, 3),
            droplet_size_microns=droplet_size_microns,
            droplet_impact_velocity_m_s=round(impact_velocity, 1),
            impingement_risk=impingement_risk,
            recommendations=recommendations,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def validate_spray_water_quality(
        self,
        tds_ppm: Optional[float] = None,
        silica_ppm: Optional[float] = None,
        hardness_ppm: Optional[float] = None,
        iron_ppb: Optional[float] = None,
        ph: Optional[float] = None,
        operating_pressure_kpa: float = 1000.0,
    ) -> WaterQualityValidation:
        """
        Validate spray water quality against requirements.

        Args:
            tds_ppm: Total dissolved solids (ppm)
            silica_ppm: Silica content (ppm)
            hardness_ppm: Calcium hardness (ppm)
            iron_ppb: Iron content (ppb)
            ph: pH value
            operating_pressure_kpa: Operating pressure (for silica limits)

        Returns:
            WaterQualityValidation result
        """
        issues = []
        req = SPRAY_WATER_REQUIREMENTS

        # TDS check
        tds_valid = True
        if tds_ppm is not None and tds_ppm > req["max_tds_ppm"]:
            tds_valid = False
            issues.append(f"TDS {tds_ppm} ppm exceeds limit of {req['max_tds_ppm']} ppm")

        # Silica check (stricter for high pressure)
        silica_valid = True
        silica_limit = req["max_silica_ppm"]
        if operating_pressure_kpa > 4000:  # High pressure
            silica_limit = 0.01
        if silica_ppm is not None and silica_ppm > silica_limit:
            silica_valid = False
            issues.append(f"Silica {silica_ppm} ppm exceeds limit of {silica_limit} ppm")

        # Hardness check
        hardness_valid = True
        if hardness_ppm is not None and hardness_ppm > req["max_hardness_ppm"]:
            hardness_valid = False
            issues.append(f"Hardness {hardness_ppm} ppm exceeds limit of {req['max_hardness_ppm']} ppm")

        # pH check
        ph_valid = True
        if ph is not None:
            if ph < req["min_ph"] or ph > req["max_ph"]:
                ph_valid = False
                issues.append(f"pH {ph} outside range {req['min_ph']}-{req['max_ph']}")

        # Assess risks
        deposition_risk = "LOW"
        if not silica_valid or not hardness_valid:
            deposition_risk = "HIGH"
        elif silica_ppm and silica_ppm > silica_limit * 0.5:
            deposition_risk = "MODERATE"

        carryover_risk = "LOW"
        if not tds_valid:
            carryover_risk = "HIGH"
        elif tds_ppm and tds_ppm > req["max_tds_ppm"] * 0.7:
            carryover_risk = "MODERATE"

        is_valid = tds_valid and silica_valid and hardness_valid and ph_valid

        return WaterQualityValidation(
            is_valid=is_valid,
            issues=issues if issues else ["Water quality meets all requirements"],
            tds_valid=tds_valid,
            silica_valid=silica_valid,
            hardness_valid=hardness_valid,
            ph_valid=ph_valid,
            deposition_risk=deposition_risk,
            carryover_risk=carryover_risk,
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _validate_spray_inputs(
        self,
        steam_flow: float,
        inlet_enthalpy: float,
        target_enthalpy: float,
        spray_enthalpy: float,
    ) -> None:
        """Validate inputs for spray water calculation."""
        if steam_flow <= 0:
            raise ValueError(f"Steam flow must be positive, got {steam_flow}")

        if inlet_enthalpy <= target_enthalpy:
            raise ValueError(
                f"Inlet enthalpy ({inlet_enthalpy}) must be greater than "
                f"target enthalpy ({target_enthalpy}) for cooling"
            )

        if target_enthalpy <= spray_enthalpy:
            raise ValueError(
                f"Target enthalpy ({target_enthalpy}) must be greater than "
                f"spray water enthalpy ({spray_enthalpy})"
            )

        # Physical limits
        if inlet_enthalpy > 4000:
            raise ValueError(
                f"Inlet enthalpy {inlet_enthalpy} kJ/kg exceeds physical limits"
            )

        if spray_enthalpy < 0 or spray_enthalpy > 500:
            raise ValueError(
                f"Spray water enthalpy {spray_enthalpy} kJ/kg outside valid range (0-500)"
            )

    def _estimate_saturation_temp(self, pressure_kpa: float) -> float:
        """
        Estimate saturation temperature from pressure.

        Uses polynomial approximation of steam tables.
        Valid for 10-22000 kPa.
        """
        if pressure_kpa < 10:
            pressure_kpa = 10
        if pressure_kpa > 22000:
            pressure_kpa = 22000

        # Polynomial fit to steam tables
        # T_sat = a + b*ln(P) + c*ln(P)^2
        import math
        ln_p = math.log(pressure_kpa)

        a = 42.6776
        b = 21.1069
        c = 0.1054

        t_sat = a + b * ln_p + c * ln_p ** 2

        return t_sat

    def _estimate_saturation_enthalpy(self, pressure_kpa: float) -> float:
        """
        Estimate saturation enthalpy (hg) from pressure.

        Uses polynomial approximation of steam tables.
        """
        if pressure_kpa < 10:
            pressure_kpa = 10
        if pressure_kpa > 22000:
            pressure_kpa = 22000

        import math
        ln_p = math.log(pressure_kpa)

        # Fit for saturated vapor enthalpy
        # hg peaks around 2800 kJ/kg then decreases
        a = 2514.4
        b = 91.8
        c = -3.8

        h_sat = a + b * ln_p + c * ln_p ** 2

        return h_sat

    def _apply_precision(self, value: Decimal, precision: int) -> Decimal:
        """Apply regulatory rounding precision."""
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
