# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Property-Based Invariant Tests

Property-based tests using Hypothesis for testing physical invariants:
- Heat loss >= 0 for hot systems (T_process > T_ambient)
- Heat gain >= 0 for cold systems (T_process < T_ambient)
- Condition score always in [0, 100]
- Payback period positive when savings > 0
- Insulated heat loss < bare surface heat loss
- Thermal resistance always positive
- Efficiency bounded [0, 100]%

Reference:
- ASTM C680 heat loss methodology
- Thermodynamic first and second laws

Author: GL-TestEngineer
Version: 2.0.0
"""

import pytest
import math
from typing import Tuple
from dataclasses import dataclass

from hypothesis import given, strategies as st, settings, assume


# =============================================================================
# HYPOTHESIS CUSTOM STRATEGIES
# =============================================================================

@st.composite
def valid_pipe_parameters(draw) -> dict:
    """Generate physically valid pipe and insulation parameters."""
    pipe_od = draw(st.floats(min_value=0.02, max_value=1.5, allow_nan=False, allow_infinity=False))
    insulation_thickness = draw(st.floats(min_value=0.01, max_value=0.3, allow_nan=False, allow_infinity=False))
    length = draw(st.floats(min_value=0.5, max_value=200, allow_nan=False, allow_infinity=False))

    return {
        "pipe_od_m": pipe_od,
        "insulation_thickness_m": insulation_thickness,
        "length_m": length,
    }


@st.composite
def valid_temperatures_hot(draw) -> Tuple[float, float]:
    """Generate physically valid temperature pair for hot systems."""
    ambient_temp = draw(st.floats(min_value=-40, max_value=45, allow_nan=False, allow_infinity=False))
    # Process temp must be higher than ambient for hot systems
    process_temp = draw(st.floats(min_value=ambient_temp + 20, max_value=600, allow_nan=False, allow_infinity=False))
    return process_temp, ambient_temp


@st.composite
def valid_temperatures_cold(draw) -> Tuple[float, float]:
    """Generate physically valid temperature pair for cold systems."""
    ambient_temp = draw(st.floats(min_value=15, max_value=45, allow_nan=False, allow_infinity=False))
    # Process temp must be lower than ambient for cold systems
    process_temp = draw(st.floats(min_value=-200, max_value=ambient_temp - 20, allow_nan=False, allow_infinity=False))
    return process_temp, ambient_temp


@st.composite
def valid_material_properties(draw) -> dict:
    """Generate valid insulation material properties."""
    return {
        "k_insulation": draw(st.floats(min_value=0.015, max_value=0.15, allow_nan=False, allow_infinity=False)),
        "h_surface": draw(st.floats(min_value=5, max_value=50, allow_nan=False, allow_infinity=False)),
    }


# =============================================================================
# HEAT LOSS CALCULATION FUNCTIONS
# =============================================================================

def calculate_cylindrical_heat_loss(
    pipe_od_m: float,
    insulation_thickness_m: float,
    process_temp_C: float,
    ambient_temp_C: float,
    k_insulation: float,
    h_surface: float,
) -> float:
    """Calculate heat loss per meter for insulated pipe."""
    r_inner = pipe_od_m / 2
    r_outer = r_inner + insulation_thickness_m

    delta_T = process_temp_C - ambient_temp_C

    R_insulation = math.log(r_outer / r_inner) / (2 * math.pi * k_insulation)
    R_surface = 1 / (2 * math.pi * r_outer * h_surface)
    R_total = R_insulation + R_surface

    return delta_T / R_total


def calculate_bare_pipe_heat_loss(
    pipe_od_m: float,
    process_temp_C: float,
    ambient_temp_C: float,
    h_surface: float,
) -> float:
    """Calculate heat loss for bare pipe."""
    r = pipe_od_m / 2
    delta_T = process_temp_C - ambient_temp_C
    return 2 * math.pi * r * h_surface * delta_T


def calculate_insulation_efficiency(
    heat_loss_insulated: float,
    heat_loss_bare: float,
) -> float:
    """Calculate insulation efficiency."""
    if abs(heat_loss_bare) < 1e-10:
        return 0.0
    return (1 - abs(heat_loss_insulated) / abs(heat_loss_bare)) * 100


def calculate_condition_score(
    thermal_ratio: float,
    age_ratio: float,
    damage_factor: float,
    moisture_factor: float,
) -> float:
    """Calculate condition score (0-100)."""
    thermal_score = min(100, max(0, thermal_ratio * 100))
    age_score = max(0, 100 * (1 - age_ratio))
    damage_score = max(0, 100 * (1 - damage_factor))
    moisture_score = max(0, 100 * (1 - moisture_factor))

    overall = (
        0.40 * thermal_score +
        0.20 * age_score +
        0.25 * damage_score +
        0.15 * moisture_score
    )

    return max(0, min(100, overall))


def calculate_payback(investment: float, annual_savings: float) -> float:
    """Calculate simple payback period."""
    if annual_savings <= 0:
        return float('inf')
    return investment / annual_savings


# =============================================================================
# TEST CLASS: HEAT LOSS INVARIANTS
# =============================================================================

class TestHeatLossInvariants:
    """Property-based tests for heat loss invariants."""

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        params=valid_pipe_parameters(),
        temps=valid_temperatures_hot(),
        material=valid_material_properties(),
    )
    def test_heat_loss_positive_for_hot_systems(self, params, temps, material):
        """Property: Heat loss >= 0 when process temp > ambient temp."""
        process_temp, ambient_temp = temps

        q = calculate_cylindrical_heat_loss(
            pipe_od_m=params["pipe_od_m"],
            insulation_thickness_m=params["insulation_thickness_m"],
            process_temp_C=process_temp,
            ambient_temp_C=ambient_temp,
            k_insulation=material["k_insulation"],
            h_surface=material["h_surface"],
        )

        assert q >= 0, (
            f"Heat loss must be >= 0 for hot system: "
            f"T_process={process_temp:.1f}C, T_ambient={ambient_temp:.1f}C, q={q:.2f}"
        )

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        params=valid_pipe_parameters(),
        temps=valid_temperatures_cold(),
        material=valid_material_properties(),
    )
    def test_heat_gain_negative_for_cold_systems(self, params, temps, material):
        """Property: Heat loss < 0 (heat gain) when process temp < ambient temp."""
        process_temp, ambient_temp = temps

        q = calculate_cylindrical_heat_loss(
            pipe_od_m=params["pipe_od_m"],
            insulation_thickness_m=params["insulation_thickness_m"],
            process_temp_C=process_temp,
            ambient_temp_C=ambient_temp,
            k_insulation=material["k_insulation"],
            h_surface=material["h_surface"],
        )

        assert q <= 0, (
            f"Heat loss must be <= 0 (heat gain) for cold system: "
            f"T_process={process_temp:.1f}C, T_ambient={ambient_temp:.1f}C, q={q:.2f}"
        )

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        params=valid_pipe_parameters(),
        temps=valid_temperatures_hot(),
        material=valid_material_properties(),
    )
    def test_insulated_less_than_bare(self, params, temps, material):
        """Property: Insulated heat loss magnitude < bare heat loss magnitude."""
        process_temp, ambient_temp = temps

        q_insulated = calculate_cylindrical_heat_loss(
            pipe_od_m=params["pipe_od_m"],
            insulation_thickness_m=params["insulation_thickness_m"],
            process_temp_C=process_temp,
            ambient_temp_C=ambient_temp,
            k_insulation=material["k_insulation"],
            h_surface=material["h_surface"],
        )

        q_bare = calculate_bare_pipe_heat_loss(
            pipe_od_m=params["pipe_od_m"],
            process_temp_C=process_temp,
            ambient_temp_C=ambient_temp,
            h_surface=material["h_surface"],
        )

        assert abs(q_insulated) < abs(q_bare), (
            f"Insulated |q|={abs(q_insulated):.2f} should be < bare |q|={abs(q_bare):.2f}"
        )

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        pipe_od=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
        process_temp=st.floats(min_value=100, max_value=300, allow_nan=False, allow_infinity=False),
        ambient_temp=st.floats(min_value=15, max_value=35, allow_nan=False, allow_infinity=False),
        k=st.floats(min_value=0.03, max_value=0.08, allow_nan=False, allow_infinity=False),
        h=st.floats(min_value=8, max_value=20, allow_nan=False, allow_infinity=False),
    )
    def test_heat_loss_monotonic_with_thickness(self, pipe_od, process_temp, ambient_temp, k, h):
        """Property: Heat loss decreases monotonically with insulation thickness."""
        assume(process_temp > ambient_temp)

        thicknesses = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
        heat_losses = []

        for t in thicknesses:
            q = calculate_cylindrical_heat_loss(
                pipe_od_m=pipe_od,
                insulation_thickness_m=t,
                process_temp_C=process_temp,
                ambient_temp_C=ambient_temp,
                k_insulation=k,
                h_surface=h,
            )
            heat_losses.append(q)

        # Check monotonic decrease
        for i in range(1, len(heat_losses)):
            assert heat_losses[i] <= heat_losses[i-1], (
                f"Heat loss must decrease with thickness: "
                f"t={thicknesses[i]}m, q={heat_losses[i]:.2f} > "
                f"t={thicknesses[i-1]}m, q={heat_losses[i-1]:.2f}"
            )

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        params=valid_pipe_parameters(),
        material=valid_material_properties(),
    )
    def test_heat_loss_proportional_to_delta_t(self, params, material):
        """Property: Heat loss proportional to temperature difference."""
        ambient_temp = 25.0
        delta_ts = [50, 100, 150, 200]
        heat_losses = []

        for dt in delta_ts:
            q = calculate_cylindrical_heat_loss(
                pipe_od_m=params["pipe_od_m"],
                insulation_thickness_m=params["insulation_thickness_m"],
                process_temp_C=ambient_temp + dt,
                ambient_temp_C=ambient_temp,
                k_insulation=material["k_insulation"],
                h_surface=material["h_surface"],
            )
            heat_losses.append(q)

        # Check proportionality: q1/dT1 ~= q2/dT2
        ratios = [q / dt for q, dt in zip(heat_losses, delta_ts)]

        for i in range(1, len(ratios)):
            relative_diff = abs(ratios[i] - ratios[0]) / ratios[0]
            assert relative_diff < 0.01, (
                f"Heat loss not proportional to dT: ratio variance {relative_diff:.4f}"
            )


# =============================================================================
# TEST CLASS: THERMAL RESISTANCE INVARIANTS
# =============================================================================

class TestThermalResistanceInvariants:
    """Property-based tests for thermal resistance invariants."""

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        params=valid_pipe_parameters(),
        material=valid_material_properties(),
    )
    def test_thermal_resistance_always_positive(self, params, material):
        """Property: Thermal resistance must always be positive."""
        r_inner = params["pipe_od_m"] / 2
        r_outer = r_inner + params["insulation_thickness_m"]

        R_insulation = math.log(r_outer / r_inner) / (2 * math.pi * material["k_insulation"])
        R_surface = 1 / (2 * math.pi * r_outer * material["h_surface"])

        assert R_insulation > 0, f"R_insulation must be > 0, got {R_insulation}"
        assert R_surface > 0, f"R_surface must be > 0, got {R_surface}"

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        params=valid_pipe_parameters(),
        material=valid_material_properties(),
    )
    def test_total_resistance_greater_than_parts(self, params, material):
        """Property: Total resistance > individual resistances."""
        r_inner = params["pipe_od_m"] / 2
        r_outer = r_inner + params["insulation_thickness_m"]

        R_insulation = math.log(r_outer / r_inner) / (2 * math.pi * material["k_insulation"])
        R_surface = 1 / (2 * math.pi * r_outer * material["h_surface"])
        R_total = R_insulation + R_surface

        assert R_total > R_insulation
        assert R_total > R_surface

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        pipe_od=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
        k=st.floats(min_value=0.03, max_value=0.08, allow_nan=False, allow_infinity=False),
    )
    def test_resistance_increases_with_thickness(self, pipe_od, k):
        """Property: Thermal resistance increases with insulation thickness."""
        r_inner = pipe_od / 2
        thicknesses = [0.01, 0.025, 0.05, 0.1, 0.15]
        resistances = []

        for t in thicknesses:
            r_outer = r_inner + t
            R = math.log(r_outer / r_inner) / (2 * math.pi * k)
            resistances.append(R)

        for i in range(1, len(resistances)):
            assert resistances[i] > resistances[i-1], (
                f"R must increase with thickness: "
                f"t={thicknesses[i]}m, R={resistances[i]:.4f} <= "
                f"t={thicknesses[i-1]}m, R={resistances[i-1]:.4f}"
            )


# =============================================================================
# TEST CLASS: EFFICIENCY INVARIANTS
# =============================================================================

class TestEfficiencyInvariants:
    """Property-based tests for efficiency invariants."""

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        q_insulated=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        q_bare_extra=st.floats(min_value=0.01, max_value=4000, allow_nan=False, allow_infinity=False),
    )
    def test_efficiency_bounded_0_to_100(self, q_insulated, q_bare_extra):
        """Property: Efficiency always in [0, 100]%."""
        q_bare = q_insulated + q_bare_extra  # Ensure q_bare > q_insulated

        efficiency = calculate_insulation_efficiency(q_insulated, q_bare)

        assert 0 <= efficiency <= 100, (
            f"Efficiency {efficiency}% out of bounds [0, 100]"
        )

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        q_bare=st.floats(min_value=100, max_value=5000, allow_nan=False, allow_infinity=False),
    )
    def test_efficiency_100_when_no_loss(self, q_bare):
        """Property: Efficiency = 100% when insulated loss = 0."""
        q_insulated = 0.0

        efficiency = calculate_insulation_efficiency(q_insulated, q_bare)

        assert efficiency == 100.0, f"Efficiency should be 100% when q=0, got {efficiency}%"

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        q=st.floats(min_value=100, max_value=5000, allow_nan=False, allow_infinity=False),
    )
    def test_efficiency_0_when_no_improvement(self, q):
        """Property: Efficiency = 0% when insulated loss = bare loss."""
        efficiency = calculate_insulation_efficiency(q, q)

        assert efficiency == 0.0, f"Efficiency should be 0% when no improvement, got {efficiency}%"

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        pipe_od=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
        process_temp=st.floats(min_value=100, max_value=300, allow_nan=False, allow_infinity=False),
        ambient_temp=st.floats(min_value=15, max_value=35, allow_nan=False, allow_infinity=False),
        k=st.floats(min_value=0.03, max_value=0.08, allow_nan=False, allow_infinity=False),
        h=st.floats(min_value=8, max_value=20, allow_nan=False, allow_infinity=False),
    )
    def test_efficiency_increases_with_thickness(self, pipe_od, process_temp, ambient_temp, k, h):
        """Property: Efficiency increases with insulation thickness."""
        assume(process_temp > ambient_temp)

        q_bare = calculate_bare_pipe_heat_loss(
            pipe_od_m=pipe_od,
            process_temp_C=process_temp,
            ambient_temp_C=ambient_temp,
            h_surface=h,
        )

        thicknesses = [0.01, 0.025, 0.05, 0.1, 0.15]
        efficiencies = []

        for t in thicknesses:
            q = calculate_cylindrical_heat_loss(
                pipe_od_m=pipe_od,
                insulation_thickness_m=t,
                process_temp_C=process_temp,
                ambient_temp_C=ambient_temp,
                k_insulation=k,
                h_surface=h,
            )
            eff = calculate_insulation_efficiency(q, q_bare)
            efficiencies.append(eff)

        for i in range(1, len(efficiencies)):
            assert efficiencies[i] >= efficiencies[i-1], (
                f"Efficiency must increase with thickness"
            )


# =============================================================================
# TEST CLASS: CONDITION SCORE INVARIANTS
# =============================================================================

class TestConditionScoreInvariants:
    """Property-based tests for condition score invariants."""

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        thermal_ratio=st.floats(min_value=0, max_value=1.5, allow_nan=False, allow_infinity=False),
        age_ratio=st.floats(min_value=0, max_value=2.0, allow_nan=False, allow_infinity=False),
        damage_factor=st.floats(min_value=0, max_value=1.0, allow_nan=False, allow_infinity=False),
        moisture_factor=st.floats(min_value=0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_condition_score_bounded_0_to_100(self, thermal_ratio, age_ratio, damage_factor, moisture_factor):
        """Property: Condition score always in [0, 100]."""
        score = calculate_condition_score(
            thermal_ratio=thermal_ratio,
            age_ratio=age_ratio,
            damage_factor=damage_factor,
            moisture_factor=moisture_factor,
        )

        assert 0 <= score <= 100, f"Score {score} out of bounds [0, 100]"

    @pytest.mark.property
    def test_perfect_condition_scores_100(self):
        """Property: Perfect condition scores 100."""
        score = calculate_condition_score(
            thermal_ratio=1.0,
            age_ratio=0.0,
            damage_factor=0.0,
            moisture_factor=0.0,
        )

        assert score == 100.0, f"Perfect condition should score 100, got {score}"

    @pytest.mark.property
    def test_worst_condition_scores_near_0(self):
        """Property: Worst condition scores near 0."""
        score = calculate_condition_score(
            thermal_ratio=0.0,
            age_ratio=2.0,  # Double expected life
            damage_factor=1.0,
            moisture_factor=1.0,
        )

        assert score == 0.0, f"Worst condition should score 0, got {score}"

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        thermal_ratio=st.floats(min_value=0.8, max_value=1.0, allow_nan=False, allow_infinity=False),
        age_ratio=st.floats(min_value=0.1, max_value=0.3, allow_nan=False, allow_infinity=False),
        moisture_factor=st.floats(min_value=0.05, max_value=0.2, allow_nan=False, allow_infinity=False),
        base_damage=st.floats(min_value=0.05, max_value=0.15, allow_nan=False, allow_infinity=False),
        damage_increase=st.floats(min_value=0.2, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_score_decreases_with_degradation(self, thermal_ratio, age_ratio, moisture_factor, base_damage, damage_increase):
        """Property: Score decreases as conditions degrade."""
        base_score = calculate_condition_score(
            thermal_ratio=thermal_ratio,
            age_ratio=age_ratio,
            damage_factor=base_damage,
            moisture_factor=moisture_factor,
        )

        # Increase damage
        degraded_score = calculate_condition_score(
            thermal_ratio=thermal_ratio,
            age_ratio=age_ratio,
            damage_factor=base_damage + damage_increase,  # More damage
            moisture_factor=moisture_factor,
        )

        assert degraded_score <= base_score, (
            f"Score should decrease with degradation: "
            f"base={base_score}, degraded={degraded_score}"
        )


# =============================================================================
# TEST CLASS: PAYBACK INVARIANTS
# =============================================================================

class TestPaybackInvariants:
    """Property-based tests for payback period invariants."""

    @pytest.mark.property
    @settings(max_examples=100)
    @given(
        investment=st.floats(min_value=1000, max_value=100000, allow_nan=False, allow_infinity=False),
        savings=st.floats(min_value=100, max_value=50000, allow_nan=False, allow_infinity=False),
    )
    def test_payback_positive_when_savings_positive(self, investment, savings):
        """Property: Payback > 0 when savings > 0."""
        payback = calculate_payback(investment, savings)

        assert payback > 0, f"Payback must be > 0: investment={investment}, savings={savings}"

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        investment=st.floats(min_value=1000, max_value=100000, allow_nan=False, allow_infinity=False),
    )
    def test_payback_infinite_when_no_savings(self, investment):
        """Property: Payback = inf when savings = 0."""
        savings = 0.0

        payback = calculate_payback(investment, savings)

        assert payback == float('inf'), f"Payback should be inf when savings=0"

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        investment=st.floats(min_value=1000, max_value=100000, allow_nan=False, allow_infinity=False),
        savings=st.floats(min_value=100, max_value=5000, allow_nan=False, allow_infinity=False),
    )
    def test_payback_infinite_when_negative_savings(self, investment, savings):
        """Property: Payback = inf when savings < 0."""
        payback = calculate_payback(investment, -savings)

        assert payback == float('inf'), f"Payback should be inf when savings<0"

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        base_investment=st.floats(min_value=5000, max_value=20000, allow_nan=False, allow_infinity=False),
        savings=st.floats(min_value=2000, max_value=10000, allow_nan=False, allow_infinity=False),
    )
    def test_payback_proportional_to_investment(self, base_investment, savings):
        """Property: Payback proportional to investment."""
        investments = [base_investment, base_investment * 2, base_investment * 3, base_investment * 4]
        paybacks = [calculate_payback(inv, savings) for inv in investments]

        # Check proportionality
        for i in range(1, len(paybacks)):
            expected_ratio = investments[i] / investments[0]
            actual_ratio = paybacks[i] / paybacks[0]

            assert abs(expected_ratio - actual_ratio) < 0.001, (
                f"Payback not proportional to investment"
            )

    @pytest.mark.property
    @settings(max_examples=50)
    @given(
        investment=st.floats(min_value=20000, max_value=80000, allow_nan=False, allow_infinity=False),
        base_savings=st.floats(min_value=2000, max_value=10000, allow_nan=False, allow_infinity=False),
    )
    def test_payback_inversely_proportional_to_savings(self, investment, base_savings):
        """Property: Payback inversely proportional to savings."""
        savings_list = [base_savings, base_savings * 2, base_savings * 4, base_savings * 5]
        paybacks = [calculate_payback(investment, s) for s in savings_list]

        # Check inverse proportionality: payback1 * savings1 = payback2 * savings2
        product = paybacks[0] * savings_list[0]

        for i in range(1, len(paybacks)):
            current_product = paybacks[i] * savings_list[i]
            relative_diff = abs(current_product - product) / product

            assert relative_diff < 0.001, (
                f"Payback not inversely proportional to savings"
            )


# =============================================================================
# TEST CLASS: DETERMINISM INVARIANTS
# =============================================================================

class TestDeterminismInvariants:
    """Property-based tests for calculation determinism."""

    @pytest.mark.property
    @settings(max_examples=20)
    @given(
        pipe_od=st.floats(min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False),
        insulation_thickness=st.floats(min_value=0.02, max_value=0.15, allow_nan=False, allow_infinity=False),
        process_temp=st.floats(min_value=100, max_value=300, allow_nan=False, allow_infinity=False),
        ambient_temp=st.floats(min_value=10, max_value=40, allow_nan=False, allow_infinity=False),
        k_insulation=st.floats(min_value=0.02, max_value=0.1, allow_nan=False, allow_infinity=False),
        h_surface=st.floats(min_value=5, max_value=25, allow_nan=False, allow_infinity=False),
    )
    def test_heat_loss_deterministic(self, pipe_od, insulation_thickness, process_temp, ambient_temp, k_insulation, h_surface):
        """Property: Heat loss calculation is deterministic."""
        params = {
            "pipe_od_m": pipe_od,
            "insulation_thickness_m": insulation_thickness,
            "process_temp_C": process_temp,
            "ambient_temp_C": ambient_temp,
            "k_insulation": k_insulation,
            "h_surface": h_surface,
        }

        results = [calculate_cylindrical_heat_loss(**params) for _ in range(20)]

        assert all(r == results[0] for r in results), "Heat loss must be deterministic"

    @pytest.mark.property
    @settings(max_examples=20)
    @given(
        thermal_ratio=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
        age_ratio=st.floats(min_value=0.1, max_value=0.8, allow_nan=False, allow_infinity=False),
        damage_factor=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        moisture_factor=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_condition_score_deterministic(self, thermal_ratio, age_ratio, damage_factor, moisture_factor):
        """Property: Condition score is deterministic."""
        params = {
            "thermal_ratio": thermal_ratio,
            "age_ratio": age_ratio,
            "damage_factor": damage_factor,
            "moisture_factor": moisture_factor,
        }

        results = [calculate_condition_score(**params) for _ in range(20)]

        assert all(r == results[0] for r in results), "Condition score must be deterministic"

    @pytest.mark.property
    @settings(max_examples=20)
    @given(
        investment=st.floats(min_value=10000, max_value=100000, allow_nan=False, allow_infinity=False),
        savings=st.floats(min_value=1000, max_value=50000, allow_nan=False, allow_infinity=False),
    )
    def test_payback_deterministic(self, investment, savings):
        """Property: Payback calculation is deterministic."""
        results = [calculate_payback(investment, savings) for _ in range(20)]

        assert all(r == results[0] for r in results), "Payback must be deterministic"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestHeatLossInvariants",
    "TestThermalResistanceInvariants",
    "TestEfficiencyInvariants",
    "TestConditionScoreInvariants",
    "TestPaybackInvariants",
    "TestDeterminismInvariants",
    "valid_pipe_parameters",
    "valid_temperatures_hot",
    "valid_temperatures_cold",
    "valid_material_properties",
]
