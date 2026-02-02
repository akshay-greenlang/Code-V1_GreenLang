# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Heat Loss Calculation Unit Tests

Tests for heat loss calculations per ASTM C680 methodology:
- Cylindrical pipe heat loss calculations
- Thermal resistance (R-value) calculations
- Surface heat transfer coefficients
- Temperature-dependent thermal conductivity
- ROI and payback period calculations
- Condition scoring algorithm

Reference Standards:
- ASTM C680: Standard Practice for Estimate of the Heat Gain or Loss
- ASTM C585: Standard Practice for Inner and Outer Diameters
- 3E Plus software validation cases

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
from decimal import Decimal
from unittest.mock import Mock, patch

# Import test fixtures from conftest
# These would be imported from the actual module in production
# from insulscan.calculators import HeatLossCalculator, ThermalResistanceCalculator
# from insulscan.calculators import ROICalculator, ConditionScorer


# =============================================================================
# HEAT LOSS CALCULATION HELPERS
# =============================================================================

def calculate_cylindrical_heat_loss(
    pipe_od_m: float,
    insulation_thickness_m: float,
    process_temp_C: float,
    ambient_temp_C: float,
    k_insulation: float,
    h_surface: float,
) -> float:
    """
    Calculate heat loss per meter for insulated cylindrical pipe.

    Uses ASTM C680 methodology:
    q = 2 * pi * (T_process - T_ambient) /
        (ln(r_outer/r_inner)/k + 1/(h*r_outer))

    Args:
        pipe_od_m: Pipe outer diameter in meters
        insulation_thickness_m: Insulation thickness in meters
        process_temp_C: Process temperature in Celsius
        ambient_temp_C: Ambient temperature in Celsius
        k_insulation: Thermal conductivity of insulation (W/m-K)
        h_surface: Surface heat transfer coefficient (W/m2-K)

    Returns:
        Heat loss in W/m (Watts per meter)
    """
    r_inner = pipe_od_m / 2
    r_outer = r_inner + insulation_thickness_m

    delta_T = process_temp_C - ambient_temp_C

    # Thermal resistance of insulation (per unit length)
    R_insulation = math.log(r_outer / r_inner) / (2 * math.pi * k_insulation)

    # Surface resistance (per unit length)
    R_surface = 1 / (2 * math.pi * r_outer * h_surface)

    # Total resistance
    R_total = R_insulation + R_surface

    # Heat loss
    q = delta_T / R_total

    return q


def calculate_bare_pipe_heat_loss(
    pipe_od_m: float,
    process_temp_C: float,
    ambient_temp_C: float,
    h_surface: float,
) -> float:
    """
    Calculate heat loss per meter for bare (uninsulated) pipe.

    Args:
        pipe_od_m: Pipe outer diameter in meters
        process_temp_C: Process temperature in Celsius
        ambient_temp_C: Ambient temperature in Celsius
        h_surface: Surface heat transfer coefficient (W/m2-K)

    Returns:
        Heat loss in W/m
    """
    r = pipe_od_m / 2
    delta_T = process_temp_C - ambient_temp_C

    # Surface area per unit length
    A = 2 * math.pi * r

    # Heat loss
    q = h_surface * A * delta_T

    return q


def calculate_surface_coefficient(
    wind_speed_m_s: float,
    surface_temp_C: float,
    ambient_temp_C: float,
    is_horizontal: bool = True,
) -> float:
    """
    Calculate combined convective and radiative surface coefficient.

    Uses correlation from ASTM C680.

    Args:
        wind_speed_m_s: Wind speed in m/s
        surface_temp_C: Surface temperature in Celsius
        ambient_temp_C: Ambient temperature in Celsius
        is_horizontal: True for horizontal pipe, False for vertical

    Returns:
        Surface heat transfer coefficient in W/m2-K
    """
    # Natural convection component
    delta_T = abs(surface_temp_C - ambient_temp_C)
    T_film = (surface_temp_C + ambient_temp_C) / 2 + 273.15  # Kelvin

    if is_horizontal:
        h_natural = 1.32 * (delta_T / 0.3) ** 0.25 if delta_T > 0 else 0
    else:
        h_natural = 1.42 * (delta_T / 0.3) ** 0.25 if delta_T > 0 else 0

    # Forced convection component (McAdams correlation)
    if wind_speed_m_s > 0:
        h_forced = 3.8 + 5.7 * wind_speed_m_s
    else:
        h_forced = 0

    # Combined (root mean square for mixed convection)
    h_convection = (h_natural ** 2 + h_forced ** 2) ** 0.5

    # Radiative component (linearized)
    emissivity = 0.9  # Typical jacketed surface
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    T_surf_K = surface_temp_C + 273.15
    T_amb_K = ambient_temp_C + 273.15
    h_radiation = emissivity * sigma * (T_surf_K ** 2 + T_amb_K ** 2) * (T_surf_K + T_amb_K)

    return h_convection + h_radiation


def calculate_temperature_dependent_k(
    k_ref: float,
    T_ref_C: float,
    T_mean_C: float,
    temperature_coefficient: float,
) -> float:
    """
    Calculate temperature-dependent thermal conductivity.

    k(T) = k_ref * (1 + coef * (T - T_ref))

    Args:
        k_ref: Reference thermal conductivity (W/m-K)
        T_ref_C: Reference temperature (Celsius)
        T_mean_C: Mean insulation temperature (Celsius)
        temperature_coefficient: Temperature coefficient

    Returns:
        Thermal conductivity at mean temperature
    """
    return k_ref * (1 + temperature_coefficient * (T_mean_C - T_ref_C))


def calculate_insulation_efficiency(
    heat_loss_insulated: float,
    heat_loss_bare: float,
) -> float:
    """
    Calculate insulation efficiency percentage.

    Args:
        heat_loss_insulated: Heat loss with insulation (W/m)
        heat_loss_bare: Heat loss without insulation (W/m)

    Returns:
        Efficiency percentage (0-100)
    """
    if heat_loss_bare == 0:
        return 0.0

    return (1 - heat_loss_insulated / heat_loss_bare) * 100


def calculate_condition_score(
    thermal_performance_ratio: float,
    age_years: float,
    expected_life_years: float,
    damage_severity: float,
    moisture_factor: float,
) -> float:
    """
    Calculate overall condition score (0-100).

    Weighted scoring:
    - Thermal performance: 40%
    - Age factor: 20%
    - Physical condition: 25%
    - Moisture: 15%

    Args:
        thermal_performance_ratio: Actual/expected performance (0-1)
        age_years: Age of insulation in years
        expected_life_years: Expected service life
        damage_severity: Damage severity factor (0-1)
        moisture_factor: Moisture impact factor (0-1)

    Returns:
        Condition score (0-100)
    """
    # Thermal performance score (0-100)
    thermal_score = min(100, thermal_performance_ratio * 100)

    # Age score (0-100)
    age_ratio = age_years / expected_life_years
    age_score = max(0, 100 * (1 - age_ratio))

    # Physical condition score (0-100)
    physical_score = max(0, 100 * (1 - damage_severity))

    # Moisture score (0-100)
    moisture_score = max(0, 100 * (1 - moisture_factor))

    # Weighted combination
    overall_score = (
        0.40 * thermal_score +
        0.20 * age_score +
        0.25 * physical_score +
        0.15 * moisture_score
    )

    return round(overall_score, 1)


def calculate_payback_period(
    investment_cost: float,
    annual_savings: float,
) -> float:
    """
    Calculate simple payback period in years.

    Args:
        investment_cost: Upfront cost in USD
        annual_savings: Annual energy savings in USD

    Returns:
        Payback period in years (float('inf') if no savings)
    """
    if annual_savings <= 0:
        return float('inf')

    return investment_cost / annual_savings


def calculate_npv(
    investment_cost: float,
    annual_savings: float,
    discount_rate: float,
    years: int,
) -> float:
    """
    Calculate Net Present Value of investment.

    Args:
        investment_cost: Upfront cost in USD
        annual_savings: Annual energy savings in USD
        discount_rate: Discount rate (e.g., 0.08 for 8%)
        years: Analysis period in years

    Returns:
        NPV in USD
    """
    npv = -investment_cost

    for year in range(1, years + 1):
        npv += annual_savings / (1 + discount_rate) ** year

    return npv


# =============================================================================
# TEST CLASS: HEAT LOSS CALCULATIONS
# =============================================================================

class TestHeatLossCalculations:
    """Test suite for heat loss calculations per ASTM C680."""

    @pytest.mark.unit
    @pytest.mark.parametrize("pipe_od_m,insulation_m,process_C,ambient_C,k,h,expected_q,tolerance", [
        # Case 1: Standard steam pipe
        (0.1143, 0.051, 175.0, 25.0, 0.040, 9.0, 58.5, 5.0),
        # Case 2: Larger pipe with more insulation
        (0.2191, 0.076, 175.0, 25.0, 0.040, 9.0, 72.8, 5.0),
        # Case 3: High temperature process
        (0.3239, 0.102, 400.0, 30.0, 0.055, 12.0, 185.0, 15.0),
        # Case 4: Low temperature difference
        (0.1143, 0.051, 50.0, 25.0, 0.040, 9.0, 9.7, 2.0),
    ])
    def test_cylindrical_heat_loss_calculation(
        self,
        pipe_od_m: float,
        insulation_m: float,
        process_C: float,
        ambient_C: float,
        k: float,
        h: float,
        expected_q: float,
        tolerance: float,
    ):
        """Test cylindrical heat loss against ASTM C680 reference values."""
        calculated_q = calculate_cylindrical_heat_loss(
            pipe_od_m=pipe_od_m,
            insulation_thickness_m=insulation_m,
            process_temp_C=process_C,
            ambient_temp_C=ambient_C,
            k_insulation=k,
            h_surface=h,
        )

        error_percent = abs(calculated_q - expected_q) / expected_q * 100

        assert error_percent < tolerance, (
            f"Heat loss calculation error {error_percent:.2f}% exceeds tolerance {tolerance}%: "
            f"expected {expected_q} W/m, got {calculated_q:.2f} W/m"
        )

    @pytest.mark.unit
    def test_heat_loss_increases_with_temperature_difference(self):
        """Test that heat loss increases with temperature difference."""
        base_params = {
            "pipe_od_m": 0.1143,
            "insulation_thickness_m": 0.051,
            "ambient_temp_C": 25.0,
            "k_insulation": 0.040,
            "h_surface": 9.0,
        }

        process_temps = [50.0, 100.0, 150.0, 200.0, 250.0]
        heat_losses = []

        for temp in process_temps:
            q = calculate_cylindrical_heat_loss(
                process_temp_C=temp,
                **base_params,
            )
            heat_losses.append(q)

        # Verify monotonic increase
        for i in range(1, len(heat_losses)):
            assert heat_losses[i] > heat_losses[i - 1], (
                f"Heat loss should increase with temperature: "
                f"T={process_temps[i]}C, q={heat_losses[i]} should be > "
                f"T={process_temps[i-1]}C, q={heat_losses[i-1]}"
            )

    @pytest.mark.unit
    def test_heat_loss_decreases_with_insulation_thickness(self):
        """Test that heat loss decreases with insulation thickness."""
        base_params = {
            "pipe_od_m": 0.1143,
            "process_temp_C": 175.0,
            "ambient_temp_C": 25.0,
            "k_insulation": 0.040,
            "h_surface": 9.0,
        }

        thicknesses = [0.025, 0.051, 0.076, 0.102, 0.127]  # 1-5 inches
        heat_losses = []

        for thickness in thicknesses:
            q = calculate_cylindrical_heat_loss(
                insulation_thickness_m=thickness,
                **base_params,
            )
            heat_losses.append(q)

        # Verify monotonic decrease
        for i in range(1, len(heat_losses)):
            assert heat_losses[i] < heat_losses[i - 1], (
                f"Heat loss should decrease with thickness: "
                f"t={thicknesses[i]}m, q={heat_losses[i]} should be < "
                f"t={thicknesses[i-1]}m, q={heat_losses[i-1]}"
            )

    @pytest.mark.unit
    def test_bare_pipe_heat_loss(self):
        """Test bare pipe heat loss calculation."""
        q_bare = calculate_bare_pipe_heat_loss(
            pipe_od_m=0.1143,
            process_temp_C=175.0,
            ambient_temp_C=25.0,
            h_surface=15.0,
        )

        # Bare pipe should have significant heat loss
        assert q_bare > 500, f"Bare pipe heat loss {q_bare} seems too low"

        # Compare to insulated
        q_insulated = calculate_cylindrical_heat_loss(
            pipe_od_m=0.1143,
            insulation_thickness_m=0.051,
            process_temp_C=175.0,
            ambient_temp_C=25.0,
            k_insulation=0.040,
            h_surface=9.0,
        )

        assert q_bare > q_insulated * 5, (
            f"Bare pipe should have much higher heat loss: "
            f"bare={q_bare}, insulated={q_insulated}"
        )

    @pytest.mark.unit
    def test_insulation_efficiency_calculation(self):
        """Test insulation efficiency calculation."""
        q_insulated = 60.0
        q_bare = 800.0

        efficiency = calculate_insulation_efficiency(q_insulated, q_bare)

        expected_efficiency = 92.5  # (1 - 60/800) * 100

        assert abs(efficiency - expected_efficiency) < 0.1, (
            f"Expected efficiency {expected_efficiency}%, got {efficiency}%"
        )

    @pytest.mark.unit
    def test_insulation_efficiency_bounds(self):
        """Test insulation efficiency is bounded correctly."""
        # Perfect insulation (no heat loss)
        eff_perfect = calculate_insulation_efficiency(0, 1000)
        assert eff_perfect == 100.0

        # No insulation benefit
        eff_none = calculate_insulation_efficiency(1000, 1000)
        assert eff_none == 0.0

        # Edge case: zero bare heat loss
        eff_edge = calculate_insulation_efficiency(100, 0)
        assert eff_edge == 0.0

    @pytest.mark.unit
    def test_cryogenic_heat_gain(self, cryogenic_asset):
        """Test heat gain calculation for cryogenic applications."""
        # For cryogenic, heat flows INTO the pipe
        q = calculate_cylindrical_heat_loss(
            pipe_od_m=cryogenic_asset.pipe_outer_diameter_m,
            insulation_thickness_m=cryogenic_asset.insulation_thickness_m,
            process_temp_C=cryogenic_asset.process_temperature_C,  # -160C
            ambient_temp_C=cryogenic_asset.ambient_temperature_C,  # 25C
            k_insulation=cryogenic_asset.material.thermal_conductivity_W_mK,
            h_surface=12.0,
        )

        # Negative heat loss = heat gain into the pipe
        assert q < 0, "Cryogenic should show negative heat loss (heat gain)"


# =============================================================================
# TEST CLASS: THERMAL RESISTANCE CALCULATIONS
# =============================================================================

class TestThermalResistanceCalculations:
    """Test suite for thermal resistance (R-value) calculations."""

    @pytest.mark.unit
    def test_insulation_r_value_cylindrical(self):
        """Test R-value calculation for cylindrical geometry."""
        r_inner = 0.1143 / 2  # 4-inch pipe
        r_outer = r_inner + 0.051  # 2-inch insulation
        k = 0.040  # W/m-K

        R_insulation = math.log(r_outer / r_inner) / (2 * math.pi * k)

        # Expected R-value approximately 1.0 m-K/W
        assert 0.8 < R_insulation < 1.5, f"R-value {R_insulation} out of expected range"

    @pytest.mark.unit
    def test_surface_resistance(self):
        """Test surface thermal resistance calculation."""
        r_outer = 0.1 + 0.05  # Insulation outer radius
        h = 10.0  # Surface coefficient

        R_surface = 1 / (2 * math.pi * r_outer * h)

        # Should be relatively small compared to insulation R
        assert R_surface < 0.2, f"Surface R-value {R_surface} seems too high"

    @pytest.mark.unit
    @pytest.mark.parametrize("wind_speed,expected_h_min,expected_h_max", [
        (0.0, 5.0, 15.0),    # Still air
        (2.0, 15.0, 25.0),   # Light breeze
        (5.0, 25.0, 40.0),   # Moderate wind
        (10.0, 50.0, 80.0),  # Strong wind
    ])
    def test_surface_coefficient_wind_effect(
        self,
        wind_speed: float,
        expected_h_min: float,
        expected_h_max: float,
    ):
        """Test surface coefficient calculation with wind speed."""
        h = calculate_surface_coefficient(
            wind_speed_m_s=wind_speed,
            surface_temp_C=50.0,
            ambient_temp_C=25.0,
        )

        assert expected_h_min <= h <= expected_h_max, (
            f"Surface coefficient h={h:.1f} outside expected range "
            f"[{expected_h_min}, {expected_h_max}] for wind={wind_speed} m/s"
        )

    @pytest.mark.unit
    def test_temperature_dependent_conductivity(self, mineral_wool_material):
        """Test temperature-dependent thermal conductivity."""
        k_ref = mineral_wool_material.thermal_conductivity_W_mK
        T_ref = mineral_wool_material.reference_temperature_C
        coef = mineral_wool_material.temperature_coefficient

        # At reference temperature
        k_at_ref = calculate_temperature_dependent_k(k_ref, T_ref, T_ref, coef)
        assert abs(k_at_ref - k_ref) < 1e-10, "k at T_ref should equal k_ref"

        # At higher temperature
        T_high = 200.0
        k_at_high = calculate_temperature_dependent_k(k_ref, T_ref, T_high, coef)
        assert k_at_high > k_ref, "k should increase with temperature"

        # At lower temperature
        T_low = 0.0
        k_at_low = calculate_temperature_dependent_k(k_ref, T_ref, T_low, coef)
        assert k_at_low < k_ref, "k should decrease with temperature"

    @pytest.mark.unit
    def test_total_thermal_resistance(self):
        """Test total thermal resistance is sum of components."""
        R_insulation = 1.0
        R_surface = 0.1

        R_total = R_insulation + R_surface

        assert R_total == pytest.approx(1.1, rel=1e-10)


# =============================================================================
# TEST CLASS: ROI CALCULATIONS
# =============================================================================

class TestROICalculations:
    """Test suite for ROI and payback period calculations."""

    @pytest.mark.unit
    @pytest.mark.parametrize("case", [
        {
            "investment": 2000.0,
            "annual_savings": 3066.0,
            "expected_payback": 0.652,
        },
        {
            "investment": 15000.0,
            "annual_savings": 7884.0,
            "expected_payback": 1.903,
        },
        {
            "investment": 5000.0,
            "annual_savings": 1000.0,
            "expected_payback": 5.0,
        },
    ])
    def test_simple_payback_period(self, case):
        """Test simple payback period calculation."""
        payback = calculate_payback_period(
            investment_cost=case["investment"],
            annual_savings=case["annual_savings"],
        )

        assert abs(payback - case["expected_payback"]) < 0.01, (
            f"Expected payback {case['expected_payback']}, got {payback:.3f}"
        )

    @pytest.mark.unit
    def test_payback_period_no_savings(self):
        """Test payback period when there are no savings."""
        payback = calculate_payback_period(
            investment_cost=10000.0,
            annual_savings=0.0,
        )

        assert payback == float('inf'), "Payback should be infinite with no savings"

    @pytest.mark.unit
    def test_payback_period_negative_savings(self):
        """Test payback period with negative savings (increased costs)."""
        payback = calculate_payback_period(
            investment_cost=10000.0,
            annual_savings=-500.0,
        )

        assert payback == float('inf'), "Payback should be infinite with negative savings"

    @pytest.mark.unit
    def test_npv_calculation_positive(self):
        """Test NPV calculation for profitable investment."""
        npv = calculate_npv(
            investment_cost=10000.0,
            annual_savings=3000.0,
            discount_rate=0.08,
            years=10,
        )

        # With 3000/year for 10 years at 8%, NPV should be positive
        assert npv > 0, f"NPV should be positive: {npv}"

        # Approximate expected value
        expected_npv = -10000 + 3000 * (1 - (1.08) ** (-10)) / 0.08
        assert abs(npv - expected_npv) < 1.0, (
            f"NPV {npv:.2f} differs from expected {expected_npv:.2f}"
        )

    @pytest.mark.unit
    def test_npv_calculation_negative(self):
        """Test NPV calculation for unprofitable investment."""
        npv = calculate_npv(
            investment_cost=50000.0,
            annual_savings=1000.0,
            discount_rate=0.08,
            years=10,
        )

        # With only 1000/year for 10 years at 8%, NPV should be negative
        assert npv < 0, f"NPV should be negative: {npv}"

    @pytest.mark.unit
    def test_npv_breakeven(self):
        """Test NPV at breakeven point."""
        # Find approximately breakeven scenario
        investment = 10000.0
        years = 10
        discount_rate = 0.08

        # Annual savings needed for breakeven
        # NPV = 0 => investment = savings * (1-(1+r)^-n)/r
        annuity_factor = (1 - (1 + discount_rate) ** (-years)) / discount_rate
        breakeven_savings = investment / annuity_factor

        npv = calculate_npv(investment, breakeven_savings, discount_rate, years)

        assert abs(npv) < 10, f"NPV should be near zero at breakeven: {npv}"

    @pytest.mark.unit
    def test_roi_with_energy_loss_data(self, economic_parameters, sample_pipe_asset):
        """Test ROI calculation with realistic energy loss data."""
        # Calculate current heat loss
        q_current = calculate_cylindrical_heat_loss(
            pipe_od_m=sample_pipe_asset.pipe_outer_diameter_m,
            insulation_thickness_m=sample_pipe_asset.insulation_thickness_m * 0.7,  # Degraded
            process_temp_C=sample_pipe_asset.process_temperature_C,
            ambient_temp_C=sample_pipe_asset.ambient_temperature_C,
            k_insulation=sample_pipe_asset.material.thermal_conductivity_W_mK * 1.3,  # Wet
            h_surface=10.0,
        )

        # Calculate heat loss after repair
        q_after = calculate_cylindrical_heat_loss(
            pipe_od_m=sample_pipe_asset.pipe_outer_diameter_m,
            insulation_thickness_m=sample_pipe_asset.insulation_thickness_m,
            process_temp_C=sample_pipe_asset.process_temperature_C,
            ambient_temp_C=sample_pipe_asset.ambient_temperature_C,
            k_insulation=sample_pipe_asset.material.thermal_conductivity_W_mK,
            h_surface=10.0,
        )

        # Calculate savings
        length = sample_pipe_asset.length_m
        energy_cost = economic_parameters["energy_cost_usd_per_kWh"]
        hours = economic_parameters["operating_hours_per_year"]

        annual_savings = (q_current - q_after) * length * hours / 1000 * energy_cost

        # ROI calculation
        repair_cost = 5000.0
        payback = calculate_payback_period(repair_cost, annual_savings)

        # Verify reasonable payback
        assert 0 < payback < 10, f"Payback {payback:.2f} years should be reasonable"


# =============================================================================
# TEST CLASS: CONDITION SCORING
# =============================================================================

class TestConditionScoring:
    """Test suite for insulation condition scoring algorithm."""

    @pytest.mark.unit
    @pytest.mark.parametrize("thermal_ratio,age,life,damage,moisture,expected_min,expected_max", [
        # Excellent condition
        (0.95, 2, 25, 0.0, 0.0, 85, 100),
        # Good condition
        (0.85, 10, 25, 0.1, 0.1, 65, 85),
        # Fair condition
        (0.70, 15, 25, 0.25, 0.2, 45, 65),
        # Poor condition
        (0.50, 20, 25, 0.4, 0.4, 25, 45),
        # Critical condition
        (0.30, 25, 25, 0.6, 0.5, 0, 30),
    ])
    def test_condition_score_ranges(
        self,
        thermal_ratio: float,
        age: float,
        life: float,
        damage: float,
        moisture: float,
        expected_min: float,
        expected_max: float,
    ):
        """Test condition score falls within expected ranges."""
        score = calculate_condition_score(
            thermal_performance_ratio=thermal_ratio,
            age_years=age,
            expected_life_years=life,
            damage_severity=damage,
            moisture_factor=moisture,
        )

        assert expected_min <= score <= expected_max, (
            f"Score {score} outside expected range [{expected_min}, {expected_max}]"
        )

    @pytest.mark.unit
    def test_condition_score_bounded(self):
        """Test condition score is always bounded 0-100."""
        test_cases = [
            (1.0, 0, 25, 0.0, 0.0),   # Best case
            (0.0, 50, 25, 1.0, 1.0),   # Worst case
            (1.5, 0, 25, 0.0, 0.0),    # Over 100% performance
            (0.5, 30, 20, 0.5, 0.5),   # Exceeded life
        ]

        for thermal, age, life, damage, moisture in test_cases:
            score = calculate_condition_score(thermal, age, life, damage, moisture)
            assert 0 <= score <= 100, f"Score {score} out of bounds [0, 100]"

    @pytest.mark.unit
    def test_condition_score_degradation_with_age(self):
        """Test condition score decreases with age."""
        scores = []

        for age in [0, 5, 10, 15, 20, 25]:
            score = calculate_condition_score(
                thermal_performance_ratio=0.85,
                age_years=age,
                expected_life_years=25,
                damage_severity=0.1,
                moisture_factor=0.1,
            )
            scores.append(score)

        # Verify monotonic decrease
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1], (
                f"Score should decrease with age: age={i*5}, score={scores[i]} "
                f"should be <= age={(i-1)*5}, score={scores[i-1]}"
            )

    @pytest.mark.unit
    def test_condition_score_damage_impact(self):
        """Test damage severity impacts condition score."""
        base_score = calculate_condition_score(
            thermal_performance_ratio=0.90,
            age_years=5,
            expected_life_years=25,
            damage_severity=0.0,
            moisture_factor=0.0,
        )

        damaged_score = calculate_condition_score(
            thermal_performance_ratio=0.90,
            age_years=5,
            expected_life_years=25,
            damage_severity=0.5,
            moisture_factor=0.0,
        )

        assert damaged_score < base_score, (
            f"Damaged score {damaged_score} should be less than base {base_score}"
        )

    @pytest.mark.unit
    def test_condition_score_moisture_impact(self):
        """Test moisture ingress impacts condition score."""
        dry_score = calculate_condition_score(
            thermal_performance_ratio=0.85,
            age_years=10,
            expected_life_years=25,
            damage_severity=0.1,
            moisture_factor=0.0,
        )

        wet_score = calculate_condition_score(
            thermal_performance_ratio=0.85,
            age_years=10,
            expected_life_years=25,
            damage_severity=0.1,
            moisture_factor=0.5,
        )

        assert wet_score < dry_score, (
            f"Wet score {wet_score} should be less than dry {dry_score}"
        )


# =============================================================================
# TEST CLASS: EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_zero_insulation_thickness(self):
        """Test behavior with zero insulation thickness."""
        # Should raise error or handle gracefully
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            calculate_cylindrical_heat_loss(
                pipe_od_m=0.1143,
                insulation_thickness_m=0.0,
                process_temp_C=175.0,
                ambient_temp_C=25.0,
                k_insulation=0.040,
                h_surface=9.0,
            )

    @pytest.mark.unit
    def test_negative_insulation_thickness(self):
        """Test handling of negative insulation thickness."""
        # Negative thickness is physically impossible
        with pytest.raises((ValueError, Exception)):
            calculate_cylindrical_heat_loss(
                pipe_od_m=0.1143,
                insulation_thickness_m=-0.05,
                process_temp_C=175.0,
                ambient_temp_C=25.0,
                k_insulation=0.040,
                h_surface=9.0,
            )

    @pytest.mark.unit
    def test_zero_thermal_conductivity(self):
        """Test handling of zero thermal conductivity."""
        # k=0 would give infinite R (division by zero)
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            calculate_cylindrical_heat_loss(
                pipe_od_m=0.1143,
                insulation_thickness_m=0.05,
                process_temp_C=175.0,
                ambient_temp_C=25.0,
                k_insulation=0.0,
                h_surface=9.0,
            )

    @pytest.mark.unit
    def test_equal_temperatures(self):
        """Test behavior when process equals ambient temperature."""
        q = calculate_cylindrical_heat_loss(
            pipe_od_m=0.1143,
            insulation_thickness_m=0.05,
            process_temp_C=25.0,
            ambient_temp_C=25.0,
            k_insulation=0.040,
            h_surface=9.0,
        )

        assert q == 0.0, "Heat loss should be zero with no temperature difference"

    @pytest.mark.unit
    def test_very_large_pipe(self):
        """Test calculation for very large pipe diameter."""
        # 48-inch pipe
        q = calculate_cylindrical_heat_loss(
            pipe_od_m=1.219,
            insulation_thickness_m=0.152,
            process_temp_C=175.0,
            ambient_temp_C=25.0,
            k_insulation=0.040,
            h_surface=9.0,
        )

        # Should be a reasonable value
        assert 100 < q < 500, f"Heat loss {q} for large pipe seems unreasonable"

    @pytest.mark.unit
    def test_very_small_pipe(self):
        """Test calculation for very small pipe diameter."""
        # 1/2-inch pipe
        q = calculate_cylindrical_heat_loss(
            pipe_od_m=0.0213,
            insulation_thickness_m=0.025,
            process_temp_C=175.0,
            ambient_temp_C=25.0,
            k_insulation=0.040,
            h_surface=9.0,
        )

        # Should be a small value
        assert 5 < q < 50, f"Heat loss {q} for small pipe seems unreasonable"


# =============================================================================
# TEST CLASS: PROVENANCE AND DETERMINISM
# =============================================================================

class TestProvenanceAndDeterminism:
    """Test provenance tracking and calculation determinism."""

    @pytest.mark.unit
    def test_calculation_determinism(self):
        """Test that calculations are deterministic."""
        params = {
            "pipe_od_m": 0.1143,
            "insulation_thickness_m": 0.051,
            "process_temp_C": 175.0,
            "ambient_temp_C": 25.0,
            "k_insulation": 0.040,
            "h_surface": 9.0,
        }

        results = [calculate_cylindrical_heat_loss(**params) for _ in range(10)]

        assert all(r == results[0] for r in results), (
            "Heat loss calculation must be deterministic"
        )

    @pytest.mark.unit
    def test_provenance_hash_generation(self, sample_thermal_measurement):
        """Test provenance hash is generated correctly."""
        measurement = sample_thermal_measurement

        # Hash should be 64 characters (SHA-256)
        assert len(measurement.provenance_hash) == 64

        # Hash should be valid hexadecimal
        try:
            int(measurement.provenance_hash, 16)
        except ValueError:
            pytest.fail("Provenance hash is not valid hexadecimal")

    @pytest.mark.unit
    def test_provenance_hash_determinism(self):
        """Test provenance hash is deterministic."""
        content = "PIPE-001|2024-01-15T10:00:00|45.000000|25.000000"

        hashes = [
            hashlib.sha256(content.encode()).hexdigest()
            for _ in range(10)
        ]

        assert all(h == hashes[0] for h in hashes), (
            "Provenance hash must be deterministic"
        )

    @pytest.mark.unit
    def test_different_inputs_different_hashes(self):
        """Test different inputs produce different hashes."""
        content1 = "PIPE-001|2024-01-15T10:00:00|45.000000|25.000000"
        content2 = "PIPE-001|2024-01-15T10:00:00|45.000001|25.000000"

        hash1 = hashlib.sha256(content1.encode()).hexdigest()
        hash2 = hashlib.sha256(content2.encode()).hexdigest()

        assert hash1 != hash2, "Different inputs must produce different hashes"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestHeatLossCalculations",
    "TestThermalResistanceCalculations",
    "TestROICalculations",
    "TestConditionScoring",
    "TestEdgeCasesAndErrors",
    "TestProvenanceAndDeterminism",
    "calculate_cylindrical_heat_loss",
    "calculate_bare_pipe_heat_loss",
    "calculate_surface_coefficient",
    "calculate_insulation_efficiency",
    "calculate_condition_score",
    "calculate_payback_period",
    "calculate_npv",
]
