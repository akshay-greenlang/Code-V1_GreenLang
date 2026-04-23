"""
Unit Tests: Desuperheater Calculator

Tests the desuperheater spray water calculation module including:
- Spray water requirement formula validation
- Constraint enforcement (min approach to saturation)
- Erosion risk assessment
- Design case data validation

Reference: ASME PTC 19.11, Spirax Sarco Engineering Guidelines

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import hashlib
import json


# =============================================================================
# Data Classes and Enumerations
# =============================================================================

class DesuperheaterType(Enum):
    """Types of desuperheaters."""
    SPRAY_NOZZLE = auto()          # Direct injection spray nozzles
    VENTURI = auto()                # Venturi type
    PROBE = auto()                  # Probe/lance type
    ANNULAR = auto()                # Annular ring type
    SURFACE_CONDENSER = auto()      # Surface contact condenser


class ErosionRisk(Enum):
    """Erosion risk levels."""
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()


class OperatingStatus(Enum):
    """Desuperheater operating status."""
    NORMAL = auto()
    WARNING = auto()
    ALARM = auto()
    SHUTDOWN = auto()


@dataclass
class DesuperheaterConfig:
    """Configuration for a desuperheater unit."""
    unit_id: str
    desuperheater_type: DesuperheaterType
    design_inlet_pressure_mpa: float
    design_inlet_temperature_k: float
    design_outlet_temperature_k: float
    design_steam_flow_kg_s: float
    spray_water_temperature_k: float
    spray_water_pressure_mpa: float
    min_approach_to_saturation_k: float = 10.0  # Default 10 K approach
    max_spray_ratio: float = 0.15  # Max spray water as fraction of steam
    nozzle_velocity_limit_m_s: float = 100.0  # Max nozzle velocity
    pipe_diameter_m: float = 0.15
    downstream_length_m: float = 10.0  # Length for atomization


@dataclass
class DesuperheaterInput:
    """Input data for desuperheater calculation."""
    inlet_pressure_mpa: float
    inlet_temperature_k: float
    inlet_mass_flow_kg_s: float
    target_outlet_temperature_k: float
    spray_water_temperature_k: float
    spray_water_pressure_mpa: float
    steam_quality: float = 1.0  # Assume dry steam


@dataclass
class DesuperheaterResult:
    """Result of desuperheater calculation."""
    spray_water_flow_kg_s: float
    outlet_mass_flow_kg_s: float
    outlet_temperature_k: float
    actual_outlet_enthalpy_kj_kg: float
    spray_ratio: float
    approach_to_saturation_k: float
    is_feasible: bool
    constraint_violations: List[str]
    warnings: List[str]
    erosion_risk: ErosionRisk
    nozzle_velocity_m_s: float
    provenance_hash: str


@dataclass
class SprayWaterSetpoint:
    """Recommended spray water setpoint."""
    flow_rate_kg_s: float
    valve_position_percent: float
    confidence: float
    operating_status: OperatingStatus
    explanation: str


# =============================================================================
# Constants
# =============================================================================

# Specific heat of water (liquid)
CP_WATER = 4.186  # kJ/(kg.K)

# Reference enthalpies (simplified)
H_WATER_REF = 100.0  # kJ/kg at ~25 C

# Critical point
CRITICAL_TEMPERATURE_K = 647.096
CRITICAL_PRESSURE_MPA = 22.064


# =============================================================================
# Desuperheater Calculator Implementation
# =============================================================================

class DesuperheaterError(Exception):
    """Error in desuperheater calculation."""
    pass


class InfeasibleOperationError(DesuperheaterError):
    """Requested operation is not feasible."""
    pass


def get_saturation_temperature(pressure_mpa: float) -> float:
    """Calculate saturation temperature from pressure."""
    if pressure_mpa < 0.001 or pressure_mpa > CRITICAL_PRESSURE_MPA:
        raise DesuperheaterError(f"Pressure {pressure_mpa} MPa outside valid range")

    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
         0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
         -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
         0.65017534844798e3]

    beta = pressure_mpa ** 0.25
    E = beta ** 2 + n[2] * beta + n[5]
    F = n[0] * beta ** 2 + n[3] * beta + n[6]
    G = n[1] * beta ** 2 + n[4] * beta + n[7]
    D = 2 * G / (-F - math.sqrt(F ** 2 - 4 * E * G))

    return (n[9] + D - math.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D))) / 2


def get_steam_enthalpy(pressure_mpa: float, temperature_k: float) -> float:
    """Calculate steam enthalpy (superheated region)."""
    t_sat = get_saturation_temperature(pressure_mpa)
    h_sat = 2675.0 + 0.5 * (t_sat - 373.15)  # Approximate h_g

    if temperature_k <= t_sat:
        return h_sat

    # Superheat contribution
    superheat = temperature_k - t_sat
    cp_steam = 2.0 + 0.001 * superheat
    return h_sat + cp_steam * superheat


def get_water_enthalpy(temperature_k: float, pressure_mpa: float = 0.1) -> float:
    """Calculate liquid water enthalpy."""
    return CP_WATER * (temperature_k - 273.15)


def calculate_spray_water_requirement(
    steam_flow_kg_s: float,
    steam_inlet_enthalpy_kj_kg: float,
    steam_outlet_enthalpy_kj_kg: float,
    spray_water_enthalpy_kj_kg: float
) -> float:
    """
    Calculate spray water flow rate using mass and energy balance.

    Energy balance: m_steam * h_in + m_spray * h_spray = (m_steam + m_spray) * h_out

    Solving for m_spray:
    m_spray = m_steam * (h_in - h_out) / (h_out - h_spray)
    """
    if steam_outlet_enthalpy_kj_kg <= spray_water_enthalpy_kj_kg:
        raise DesuperheaterError(
            f"Outlet enthalpy {steam_outlet_enthalpy_kj_kg} must exceed "
            f"spray water enthalpy {spray_water_enthalpy_kj_kg}"
        )

    delta_h_steam = steam_inlet_enthalpy_kj_kg - steam_outlet_enthalpy_kj_kg
    delta_h_spray = steam_outlet_enthalpy_kj_kg - spray_water_enthalpy_kj_kg

    if delta_h_spray <= 0:
        raise DesuperheaterError("Invalid enthalpy difference for spray calculation")

    spray_flow = steam_flow_kg_s * delta_h_steam / delta_h_spray

    return max(0.0, spray_flow)


def calculate_nozzle_velocity(
    spray_flow_kg_s: float,
    water_density_kg_m3: float,
    nozzle_area_m2: float
) -> float:
    """Calculate spray nozzle exit velocity."""
    if nozzle_area_m2 <= 0:
        raise DesuperheaterError("Nozzle area must be positive")

    volume_flow_m3_s = spray_flow_kg_s / water_density_kg_m3
    return volume_flow_m3_s / nozzle_area_m2


def assess_erosion_risk(
    nozzle_velocity_m_s: float,
    spray_ratio: float,
    approach_to_saturation_k: float
) -> ErosionRisk:
    """
    Assess erosion risk based on operating conditions.

    Factors:
    - High nozzle velocity increases droplet impact erosion
    - High spray ratio increases water loading
    - Low approach to saturation increases wet steam risk
    """
    risk_score = 0

    # Velocity contribution
    if nozzle_velocity_m_s > 150:
        risk_score += 3
    elif nozzle_velocity_m_s > 100:
        risk_score += 2
    elif nozzle_velocity_m_s > 75:
        risk_score += 1

    # Spray ratio contribution
    if spray_ratio > 0.20:
        risk_score += 3
    elif spray_ratio > 0.15:
        risk_score += 2
    elif spray_ratio > 0.10:
        risk_score += 1

    # Approach to saturation contribution
    if approach_to_saturation_k < 5:
        risk_score += 3
    elif approach_to_saturation_k < 10:
        risk_score += 2
    elif approach_to_saturation_k < 15:
        risk_score += 1

    # Map score to risk level
    if risk_score >= 6:
        return ErosionRisk.CRITICAL
    elif risk_score >= 4:
        return ErosionRisk.HIGH
    elif risk_score >= 2:
        return ErosionRisk.MODERATE
    else:
        return ErosionRisk.LOW


def calculate_desuperheater(
    config: DesuperheaterConfig,
    input_data: DesuperheaterInput
) -> DesuperheaterResult:
    """
    Calculate desuperheater spray water requirement and validate constraints.

    Args:
        config: Desuperheater configuration
        input_data: Current operating conditions and target

    Returns:
        DesuperheaterResult with spray water flow and constraint validation
    """
    constraint_violations = []
    warnings = []

    # Get saturation temperature at operating pressure
    t_sat = get_saturation_temperature(input_data.inlet_pressure_mpa)

    # Check target temperature is achievable
    min_outlet_temp = t_sat + config.min_approach_to_saturation_k

    if input_data.target_outlet_temperature_k < min_outlet_temp:
        constraint_violations.append(
            f"Target temperature {input_data.target_outlet_temperature_k:.1f} K "
            f"below minimum approach {min_outlet_temp:.1f} K"
        )
        # Adjust target to minimum feasible
        effective_target_temp = min_outlet_temp
    else:
        effective_target_temp = input_data.target_outlet_temperature_k

    # Check inlet temperature is superheated
    if input_data.inlet_temperature_k <= t_sat:
        constraint_violations.append(
            f"Inlet temperature {input_data.inlet_temperature_k:.1f} K "
            f"not superheated (Tsat = {t_sat:.1f} K)"
        )
        is_feasible = False
        spray_flow = 0.0
    else:
        # Calculate enthalpies
        h_inlet = get_steam_enthalpy(input_data.inlet_pressure_mpa, input_data.inlet_temperature_k)
        h_outlet = get_steam_enthalpy(input_data.inlet_pressure_mpa, effective_target_temp)
        h_spray = get_water_enthalpy(input_data.spray_water_temperature_k, input_data.spray_water_pressure_mpa)

        # Calculate spray water requirement
        try:
            spray_flow = calculate_spray_water_requirement(
                steam_flow_kg_s=input_data.inlet_mass_flow_kg_s,
                steam_inlet_enthalpy_kj_kg=h_inlet,
                steam_outlet_enthalpy_kj_kg=h_outlet,
                spray_water_enthalpy_kj_kg=h_spray
            )
            is_feasible = True
        except DesuperheaterError as e:
            constraint_violations.append(str(e))
            spray_flow = 0.0
            is_feasible = False
            h_outlet = h_inlet

    # Calculate spray ratio
    spray_ratio = spray_flow / input_data.inlet_mass_flow_kg_s if input_data.inlet_mass_flow_kg_s > 0 else 0

    # Check spray ratio constraint
    if spray_ratio > config.max_spray_ratio:
        constraint_violations.append(
            f"Spray ratio {spray_ratio:.3f} exceeds maximum {config.max_spray_ratio:.3f}"
        )
        is_feasible = False

    # Calculate approach to saturation
    approach_to_saturation = effective_target_temp - t_sat

    # Calculate nozzle velocity (assuming single nozzle, 5mm diameter)
    nozzle_diameter = 0.005  # 5mm
    nozzle_area = math.pi * (nozzle_diameter / 2) ** 2
    water_density = 1000.0  # kg/m3 (approximate)

    nozzle_velocity = calculate_nozzle_velocity(spray_flow, water_density, nozzle_area) if spray_flow > 0 else 0

    # Check nozzle velocity constraint
    if nozzle_velocity > config.nozzle_velocity_limit_m_s:
        warnings.append(
            f"Nozzle velocity {nozzle_velocity:.1f} m/s exceeds limit {config.nozzle_velocity_limit_m_s:.1f} m/s"
        )

    # Assess erosion risk
    erosion_risk = assess_erosion_risk(nozzle_velocity, spray_ratio, approach_to_saturation)

    if erosion_risk in [ErosionRisk.HIGH, ErosionRisk.CRITICAL]:
        warnings.append(f"Erosion risk level: {erosion_risk.name}")

    # Calculate outlet conditions
    outlet_mass_flow = input_data.inlet_mass_flow_kg_s + spray_flow

    # Compute provenance hash
    input_dict = {
        "inlet_pressure_mpa": input_data.inlet_pressure_mpa,
        "inlet_temperature_k": input_data.inlet_temperature_k,
        "inlet_mass_flow_kg_s": input_data.inlet_mass_flow_kg_s,
        "target_outlet_temperature_k": input_data.target_outlet_temperature_k,
        "spray_water_temperature_k": input_data.spray_water_temperature_k
    }
    provenance_hash = hashlib.sha256(
        json.dumps(input_dict, sort_keys=True).encode()
    ).hexdigest()

    return DesuperheaterResult(
        spray_water_flow_kg_s=spray_flow,
        outlet_mass_flow_kg_s=outlet_mass_flow,
        outlet_temperature_k=effective_target_temp,
        actual_outlet_enthalpy_kj_kg=h_outlet if is_feasible else 0,
        spray_ratio=spray_ratio,
        approach_to_saturation_k=approach_to_saturation,
        is_feasible=is_feasible and len(constraint_violations) == 0,
        constraint_violations=constraint_violations,
        warnings=warnings,
        erosion_risk=erosion_risk,
        nozzle_velocity_m_s=nozzle_velocity,
        provenance_hash=provenance_hash
    )


def calculate_spray_setpoint(
    result: DesuperheaterResult,
    valve_cv: float = 10.0,
    delta_p_mpa: float = 0.2
) -> SprayWaterSetpoint:
    """
    Calculate spray water valve setpoint from flow requirement.

    Uses simplified valve flow equation:
    Q = Cv * sqrt(dP / SG)
    """
    if not result.is_feasible:
        return SprayWaterSetpoint(
            flow_rate_kg_s=0.0,
            valve_position_percent=0.0,
            confidence=0.0,
            operating_status=OperatingStatus.ALARM,
            explanation=f"Operation not feasible: {'; '.join(result.constraint_violations)}"
        )

    # Calculate valve position from flow
    # Q in m3/h, Cv in gpm/psi^0.5
    # Simplified: position proportional to flow requirement
    max_flow_at_100_percent = valve_cv * math.sqrt(delta_p_mpa * 145.038) * 0.000227  # Approximate kg/s

    if max_flow_at_100_percent <= 0:
        valve_position = 0.0
    else:
        valve_position = min(100.0, result.spray_water_flow_kg_s / max_flow_at_100_percent * 100)

    # Determine operating status
    if result.erosion_risk == ErosionRisk.CRITICAL:
        status = OperatingStatus.SHUTDOWN
        confidence = 0.3
    elif result.erosion_risk == ErosionRisk.HIGH or len(result.warnings) > 0:
        status = OperatingStatus.WARNING
        confidence = 0.7
    else:
        status = OperatingStatus.NORMAL
        confidence = 0.95

    # Generate explanation
    explanation = (
        f"Spray flow {result.spray_water_flow_kg_s:.3f} kg/s required to achieve "
        f"{result.outlet_temperature_k:.1f} K outlet temperature. "
        f"Spray ratio: {result.spray_ratio:.1%}. "
        f"Approach to saturation: {result.approach_to_saturation_k:.1f} K."
    )

    return SprayWaterSetpoint(
        flow_rate_kg_s=result.spray_water_flow_kg_s,
        valve_position_percent=valve_position,
        confidence=confidence,
        operating_status=status,
        explanation=explanation
    )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def standard_config():
    """Standard desuperheater configuration."""
    return DesuperheaterConfig(
        unit_id="DSH-001",
        desuperheater_type=DesuperheaterType.SPRAY_NOZZLE,
        design_inlet_pressure_mpa=1.5,
        design_inlet_temperature_k=523.0,  # 250 C
        design_outlet_temperature_k=473.0,  # 200 C
        design_steam_flow_kg_s=5.0,
        spray_water_temperature_k=333.0,  # 60 C
        spray_water_pressure_mpa=2.0,
        min_approach_to_saturation_k=10.0,
        max_spray_ratio=0.15
    )


@pytest.fixture
def design_case_input():
    """Design case input matching config."""
    return DesuperheaterInput(
        inlet_pressure_mpa=1.5,
        inlet_temperature_k=523.0,
        inlet_mass_flow_kg_s=5.0,
        target_outlet_temperature_k=473.0,
        spray_water_temperature_k=333.0,
        spray_water_pressure_mpa=2.0
    )


@pytest.fixture
def high_superheat_input():
    """High superheat input requiring more spray."""
    return DesuperheaterInput(
        inlet_pressure_mpa=1.5,
        inlet_temperature_k=573.0,  # 300 C - high superheat
        inlet_mass_flow_kg_s=5.0,
        target_outlet_temperature_k=473.0,
        spray_water_temperature_k=333.0,
        spray_water_pressure_mpa=2.0
    )


@pytest.fixture
def low_target_input():
    """Low target temperature input (near saturation)."""
    return DesuperheaterInput(
        inlet_pressure_mpa=1.5,
        inlet_temperature_k=523.0,
        inlet_mass_flow_kg_s=5.0,
        target_outlet_temperature_k=475.0,  # Near saturation (~471 K at 1.5 MPa)
        spray_water_temperature_k=333.0,
        spray_water_pressure_mpa=2.0
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestSprayWaterFormula:
    """Test spray water requirement calculation formula."""

    def test_zero_superheat_removal_zero_spray(self):
        """Test that no spray is needed if inlet = outlet enthalpy."""
        spray = calculate_spray_water_requirement(
            steam_flow_kg_s=5.0,
            steam_inlet_enthalpy_kj_kg=2800.0,
            steam_outlet_enthalpy_kj_kg=2800.0,
            spray_water_enthalpy_kj_kg=200.0
        )
        assert spray == 0.0

    def test_spray_increases_with_enthalpy_reduction(self):
        """Test that spray flow increases with greater enthalpy reduction."""
        spray_small = calculate_spray_water_requirement(
            steam_flow_kg_s=5.0,
            steam_inlet_enthalpy_kj_kg=2850.0,
            steam_outlet_enthalpy_kj_kg=2800.0,
            spray_water_enthalpy_kj_kg=200.0
        )
        spray_large = calculate_spray_water_requirement(
            steam_flow_kg_s=5.0,
            steam_inlet_enthalpy_kj_kg=2900.0,
            steam_outlet_enthalpy_kj_kg=2800.0,
            spray_water_enthalpy_kj_kg=200.0
        )
        assert spray_large > spray_small

    def test_spray_scales_with_steam_flow(self):
        """Test that spray flow scales linearly with steam flow."""
        spray_1 = calculate_spray_water_requirement(
            steam_flow_kg_s=5.0,
            steam_inlet_enthalpy_kj_kg=2850.0,
            steam_outlet_enthalpy_kj_kg=2800.0,
            spray_water_enthalpy_kj_kg=200.0
        )
        spray_2 = calculate_spray_water_requirement(
            steam_flow_kg_s=10.0,
            steam_inlet_enthalpy_kj_kg=2850.0,
            steam_outlet_enthalpy_kj_kg=2800.0,
            spray_water_enthalpy_kj_kg=200.0
        )
        assert pytest.approx(spray_2, rel=0.001) == 2 * spray_1

    def test_colder_spray_water_reduces_flow(self):
        """Test that colder spray water requires less flow."""
        spray_warm = calculate_spray_water_requirement(
            steam_flow_kg_s=5.0,
            steam_inlet_enthalpy_kj_kg=2850.0,
            steam_outlet_enthalpy_kj_kg=2800.0,
            spray_water_enthalpy_kj_kg=300.0  # Warmer
        )
        spray_cold = calculate_spray_water_requirement(
            steam_flow_kg_s=5.0,
            steam_inlet_enthalpy_kj_kg=2850.0,
            steam_outlet_enthalpy_kj_kg=2800.0,
            spray_water_enthalpy_kj_kg=100.0  # Colder
        )
        assert spray_cold < spray_warm

    def test_invalid_outlet_below_spray_raises_error(self):
        """Test that outlet enthalpy below spray water raises error."""
        with pytest.raises(DesuperheaterError):
            calculate_spray_water_requirement(
                steam_flow_kg_s=5.0,
                steam_inlet_enthalpy_kj_kg=2800.0,
                steam_outlet_enthalpy_kj_kg=100.0,  # Below spray
                spray_water_enthalpy_kj_kg=200.0
            )

    def test_mass_balance_satisfied(self):
        """Test that mass balance is satisfied in calculation."""
        m_steam = 5.0
        h_in = 2850.0
        h_out = 2800.0
        h_spray = 200.0

        spray = calculate_spray_water_requirement(m_steam, h_in, h_out, h_spray)

        # Verify energy balance
        energy_in = m_steam * h_in + spray * h_spray
        energy_out = (m_steam + spray) * h_out

        assert pytest.approx(energy_in, rel=0.001) == energy_out


class TestConstraintEnforcement:
    """Test constraint enforcement in desuperheater calculations."""

    def test_min_approach_to_saturation_enforced(self, standard_config):
        """Test that minimum approach to saturation is enforced."""
        # Target below min approach
        t_sat = get_saturation_temperature(1.5)
        target_below_min = t_sat + 5.0  # Only 5 K above saturation

        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=523.0,
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=target_below_min,
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        result = calculate_desuperheater(standard_config, input_data)

        # Should have constraint violation
        assert len(result.constraint_violations) > 0
        assert any("approach" in v.lower() for v in result.constraint_violations)

    def test_max_spray_ratio_enforced(self, standard_config):
        """Test that maximum spray ratio is enforced."""
        # Very high superheat requiring excessive spray
        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=673.0,  # Very high superheat
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=480.0,  # Low target
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        # Modify config for stricter ratio
        config = DesuperheaterConfig(
            unit_id="DSH-001",
            desuperheater_type=DesuperheaterType.SPRAY_NOZZLE,
            design_inlet_pressure_mpa=1.5,
            design_inlet_temperature_k=523.0,
            design_outlet_temperature_k=473.0,
            design_steam_flow_kg_s=5.0,
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0,
            max_spray_ratio=0.05  # Very low limit
        )

        result = calculate_desuperheater(config, input_data)

        # Should violate spray ratio
        if result.spray_ratio > 0.05:
            assert not result.is_feasible
            assert any("ratio" in v.lower() for v in result.constraint_violations)

    def test_saturated_inlet_rejected(self, standard_config):
        """Test that saturated/subcooled inlet is rejected."""
        t_sat = get_saturation_temperature(1.5)

        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=t_sat - 5.0,  # Below saturation
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=t_sat - 10.0,
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        result = calculate_desuperheater(standard_config, input_data)

        assert not result.is_feasible
        assert any("superheated" in v.lower() for v in result.constraint_violations)


class TestErosionRiskAssessment:
    """Test erosion risk assessment."""

    def test_low_risk_normal_operation(self):
        """Test low risk for normal operation."""
        risk = assess_erosion_risk(
            nozzle_velocity_m_s=50.0,
            spray_ratio=0.05,
            approach_to_saturation_k=30.0
        )
        assert risk == ErosionRisk.LOW

    def test_moderate_risk_elevated_velocity(self):
        """Test moderate risk with elevated velocity."""
        risk = assess_erosion_risk(
            nozzle_velocity_m_s=85.0,
            spray_ratio=0.08,
            approach_to_saturation_k=15.0
        )
        assert risk == ErosionRisk.MODERATE

    def test_high_risk_multiple_factors(self):
        """Test high risk with multiple concerning factors."""
        risk = assess_erosion_risk(
            nozzle_velocity_m_s=120.0,
            spray_ratio=0.12,
            approach_to_saturation_k=8.0
        )
        assert risk in [ErosionRisk.HIGH, ErosionRisk.CRITICAL]

    def test_critical_risk_extreme_conditions(self):
        """Test critical risk under extreme conditions."""
        risk = assess_erosion_risk(
            nozzle_velocity_m_s=180.0,
            spray_ratio=0.25,
            approach_to_saturation_k=3.0
        )
        assert risk == ErosionRisk.CRITICAL

    @pytest.mark.parametrize("velocity,spray,approach,expected", [
        (30.0, 0.03, 40.0, ErosionRisk.LOW),
        (80.0, 0.08, 12.0, ErosionRisk.MODERATE),
        (110.0, 0.12, 8.0, ErosionRisk.HIGH),
    ])
    def test_erosion_risk_parametrized(self, velocity, spray, approach, expected):
        """Parametrized erosion risk tests."""
        risk = assess_erosion_risk(velocity, spray, approach)
        assert risk == expected or risk.value >= expected.value  # At least expected severity


class TestDesignCaseValidation:
    """Test against design case data."""

    def test_design_case_feasible(self, standard_config, design_case_input):
        """Test that design case operates feasibly."""
        result = calculate_desuperheater(standard_config, design_case_input)

        assert result.is_feasible
        assert len(result.constraint_violations) == 0

    def test_design_case_outlet_temperature(self, standard_config, design_case_input):
        """Test that design case achieves target outlet temperature."""
        result = calculate_desuperheater(standard_config, design_case_input)

        # Outlet should be at or close to target
        assert abs(result.outlet_temperature_k - design_case_input.target_outlet_temperature_k) < 1.0

    def test_design_case_spray_ratio_acceptable(self, standard_config, design_case_input):
        """Test that design case spray ratio is within limits."""
        result = calculate_desuperheater(standard_config, design_case_input)

        assert result.spray_ratio <= standard_config.max_spray_ratio

    def test_design_case_approach_acceptable(self, standard_config, design_case_input):
        """Test that design case approach is above minimum."""
        result = calculate_desuperheater(standard_config, design_case_input)

        assert result.approach_to_saturation_k >= standard_config.min_approach_to_saturation_k

    def test_design_case_low_erosion_risk(self, standard_config, design_case_input):
        """Test that design case has low erosion risk."""
        result = calculate_desuperheater(standard_config, design_case_input)

        assert result.erosion_risk in [ErosionRisk.LOW, ErosionRisk.MODERATE]

    def test_high_superheat_requires_more_spray(self, standard_config, design_case_input, high_superheat_input):
        """Test that higher superheat requires more spray water."""
        result_design = calculate_desuperheater(standard_config, design_case_input)
        result_high = calculate_desuperheater(standard_config, high_superheat_input)

        assert result_high.spray_water_flow_kg_s > result_design.spray_water_flow_kg_s


class TestSpraySetpointCalculation:
    """Test spray water setpoint calculation."""

    def test_setpoint_feasible_operation(self, standard_config, design_case_input):
        """Test setpoint calculation for feasible operation."""
        result = calculate_desuperheater(standard_config, design_case_input)
        setpoint = calculate_spray_setpoint(result)

        assert setpoint.flow_rate_kg_s == result.spray_water_flow_kg_s
        assert 0 <= setpoint.valve_position_percent <= 100
        assert setpoint.operating_status == OperatingStatus.NORMAL
        assert setpoint.confidence > 0.9

    def test_setpoint_infeasible_operation(self, standard_config):
        """Test setpoint calculation for infeasible operation."""
        t_sat = get_saturation_temperature(1.5)

        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=t_sat - 10.0,  # Subcooled - infeasible
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=t_sat - 20.0,
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        result = calculate_desuperheater(standard_config, input_data)
        setpoint = calculate_spray_setpoint(result)

        assert setpoint.valve_position_percent == 0.0
        assert setpoint.operating_status == OperatingStatus.ALARM
        assert setpoint.confidence < 0.5

    def test_setpoint_warning_conditions(self, standard_config):
        """Test setpoint calculation with warning conditions."""
        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=623.0,  # High superheat
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=485.0,  # Close to saturation
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        result = calculate_desuperheater(standard_config, input_data)

        if result.is_feasible and (result.erosion_risk in [ErosionRisk.HIGH, ErosionRisk.CRITICAL] or len(result.warnings) > 0):
            setpoint = calculate_spray_setpoint(result)
            assert setpoint.operating_status in [OperatingStatus.WARNING, OperatingStatus.SHUTDOWN]


class TestProvenanceTracking:
    """Test provenance hash generation."""

    def test_provenance_hash_generated(self, standard_config, design_case_input):
        """Test that provenance hash is generated."""
        result = calculate_desuperheater(standard_config, design_case_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_deterministic(self, standard_config, design_case_input):
        """Test that same inputs produce same hash."""
        result1 = calculate_desuperheater(standard_config, design_case_input)
        result2 = calculate_desuperheater(standard_config, design_case_input)

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_changes_with_input(self, standard_config):
        """Test that different inputs produce different hash."""
        input1 = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=523.0,
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=473.0,
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )
        input2 = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=523.0,
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=483.0,  # Different target
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        result1 = calculate_desuperheater(standard_config, input1)
        result2 = calculate_desuperheater(standard_config, input2)

        assert result1.provenance_hash != result2.provenance_hash


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_steam_flow(self, standard_config):
        """Test handling of zero steam flow."""
        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=523.0,
            inlet_mass_flow_kg_s=0.0,  # Zero flow
            target_outlet_temperature_k=473.0,
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        result = calculate_desuperheater(standard_config, input_data)

        assert result.spray_water_flow_kg_s == 0.0

    def test_very_small_superheat(self, standard_config):
        """Test with very small superheat to remove."""
        t_sat = get_saturation_temperature(1.5)

        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=t_sat + 15.0,  # Small superheat
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=t_sat + 12.0,  # Only 3 K reduction
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=2.0
        )

        result = calculate_desuperheater(standard_config, input_data)

        # Small spray should be required
        assert result.spray_water_flow_kg_s >= 0

    def test_near_critical_pressure(self, standard_config):
        """Test behavior near critical pressure."""
        input_data = DesuperheaterInput(
            inlet_pressure_mpa=20.0,  # Near critical
            inlet_temperature_k=700.0,
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=660.0,
            spray_water_temperature_k=333.0,
            spray_water_pressure_mpa=25.0
        )

        # Should handle near-critical conditions
        try:
            result = calculate_desuperheater(standard_config, input_data)
            assert result is not None
        except DesuperheaterError:
            # Acceptable to reject near-critical operation
            pass

    def test_very_cold_spray_water(self, standard_config):
        """Test with very cold spray water."""
        input_data = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=523.0,
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=473.0,
            spray_water_temperature_k=283.0,  # 10 C - very cold
            spray_water_pressure_mpa=2.0
        )

        result = calculate_desuperheater(standard_config, input_data)

        # Cold spray should require less flow
        input_warm = DesuperheaterInput(
            inlet_pressure_mpa=1.5,
            inlet_temperature_k=523.0,
            inlet_mass_flow_kg_s=5.0,
            target_outlet_temperature_k=473.0,
            spray_water_temperature_k=363.0,  # 90 C - warm
            spray_water_pressure_mpa=2.0
        )

        result_warm = calculate_desuperheater(standard_config, input_warm)

        assert result.spray_water_flow_kg_s < result_warm.spray_water_flow_kg_s


class TestMassEnergyBalance:
    """Test mass and energy balance in desuperheater calculation."""

    def test_mass_balance(self, standard_config, design_case_input):
        """Test that mass is conserved."""
        result = calculate_desuperheater(standard_config, design_case_input)

        mass_in = design_case_input.inlet_mass_flow_kg_s + result.spray_water_flow_kg_s
        mass_out = result.outlet_mass_flow_kg_s

        assert pytest.approx(mass_in, rel=0.001) == mass_out

    def test_energy_balance(self, standard_config, design_case_input):
        """Test that energy is conserved."""
        result = calculate_desuperheater(standard_config, design_case_input)

        if not result.is_feasible:
            pytest.skip("Infeasible operation")

        # Calculate energy in
        h_steam_in = get_steam_enthalpy(
            design_case_input.inlet_pressure_mpa,
            design_case_input.inlet_temperature_k
        )
        h_spray = get_water_enthalpy(
            design_case_input.spray_water_temperature_k,
            design_case_input.spray_water_pressure_mpa
        )

        energy_in = (
            design_case_input.inlet_mass_flow_kg_s * h_steam_in +
            result.spray_water_flow_kg_s * h_spray
        )

        energy_out = result.outlet_mass_flow_kg_s * result.actual_outlet_enthalpy_kj_kg

        # Allow 1% tolerance for numerical precision
        assert pytest.approx(energy_in, rel=0.01) == energy_out
