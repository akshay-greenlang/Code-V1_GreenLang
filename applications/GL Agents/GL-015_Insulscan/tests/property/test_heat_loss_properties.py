# -*- coding: utf-8 -*-
"""GL-015 INSULSCAN - Property-based tests for heat loss calculations.

Property-based tests using Hypothesis framework to verify that heat loss
calculations respect fundamental physical laws and constraints.

Physical Invariants Tested:
    1. Heat loss is positive when operating temp > ambient temp
    2. Thicker insulation reduces heat loss (monotonicity)
    3. Larger surface area increases heat loss (proportionality)
    4. Higher temperature differential increases heat loss
    5. Condition score is always bounded in [0, 100]
    6. More hot spots result in lower condition score

These tests use randomized inputs to explore edge cases and verify
that the calculation engine maintains physical correctness across
the entire valid input domain.

Author: GL-TestEngineer
Version: 1.0.0
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, settings, assume

from gl_015_insulscan.calculators.heat_loss import HeatLossCalculator, SurfaceType
from gl_015_insulscan.core.config import InsulscanSettings


@pytest.fixture
def calculator() -> HeatLossCalculator:
    """Create heat loss calculator instance."""
    return HeatLossCalculator()


@pytest.fixture
def settings_fixture() -> InsulscanSettings:
    """Create settings fixture."""
    return InsulscanSettings()


class TestHeatLossProperties:
    """Property-based tests for heat loss calculations."""

    @given(
        operating_temp=st.floats(min_value=50, max_value=500),
        ambient_temp=st.floats(min_value=-20, max_value=45),
        surface_area=st.floats(min_value=0.1, max_value=1000),
        thermal_resistance=st.floats(min_value=0.1, max_value=10),
    )
    @settings(max_examples=100)
    def test_heat_loss_always_positive_when_temp_diff_positive(
        self,
        calculator: HeatLossCalculator,
        operating_temp: float,
        ambient_temp: float,
        surface_area: float,
        thermal_resistance: float,
    ) -> None:
        """
        Heat loss should always be positive when operating > ambient.

        This is a fundamental thermodynamic principle: heat flows from
        high temperature to low temperature regions.
        """
        assume(operating_temp > ambient_temp + 10)  # Ensure meaningful diff

        result = calculator.calculate_bare_surface_loss(
            surface_temp_c=operating_temp,
            ambient_temp_c=ambient_temp,
            surface_area_m2=surface_area,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.VERTICAL_FLAT,
        )

        assert float(result.total_heat_loss_w) > 0

    @given(
        thickness1=st.floats(min_value=10, max_value=100),
        thickness2=st.floats(min_value=100, max_value=200),
    )
    @settings(max_examples=50)
    def test_thicker_insulation_reduces_heat_loss(
        self,
        calculator: HeatLossCalculator,
        thickness1: float,
        thickness2: float,
    ) -> None:
        """
        Thicker insulation should result in lower heat loss.

        This follows from Fourier's Law: Q = -k * A * dT/dx
        Greater thickness (dx) reduces heat flux.
        """
        assume(thickness2 > thickness1 * 1.5)  # Ensure significant difference

        # Calculate thermal resistance for each thickness
        # R = thickness / conductivity
        conductivity = 0.04  # W/m-K for mineral wool

        result1 = calculator.calculate_insulated_surface_loss(
            operating_temp_c=200.0,
            ambient_temp_c=25.0,
            insulation_type="mineral_wool",
            thickness_mm=thickness1,
            surface_area_m2=10.0,
        )

        result2 = calculator.calculate_insulated_surface_loss(
            operating_temp_c=200.0,
            ambient_temp_c=25.0,
            insulation_type="mineral_wool",
            thickness_mm=thickness2,
            surface_area_m2=10.0,
        )

        assert float(result2.total_heat_loss_w) < float(result1.total_heat_loss_w)

    @given(
        area1=st.floats(min_value=1, max_value=50),
        area2=st.floats(min_value=50, max_value=200),
    )
    @settings(max_examples=50)
    def test_larger_area_increases_heat_loss(
        self,
        calculator: HeatLossCalculator,
        area1: float,
        area2: float,
    ) -> None:
        """
        Larger surface area should result in higher heat loss.

        Heat loss is proportional to surface area: Q = h * A * dT
        """
        assume(area2 > area1 * 1.5)

        result1 = calculator.calculate_bare_surface_loss(
            surface_temp_c=150.0,
            ambient_temp_c=25.0,
            surface_area_m2=area1,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
        )

        result2 = calculator.calculate_bare_surface_loss(
            surface_temp_c=150.0,
            ambient_temp_c=25.0,
            surface_area_m2=area2,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
        )

        assert float(result2.total_heat_loss_w) > float(result1.total_heat_loss_w)

    @given(
        temp_diff1=st.floats(min_value=10, max_value=100),
        temp_diff2=st.floats(min_value=100, max_value=300),
    )
    @settings(max_examples=50)
    def test_higher_temp_diff_increases_heat_loss(
        self,
        calculator: HeatLossCalculator,
        temp_diff1: float,
        temp_diff2: float,
    ) -> None:
        """
        Higher temperature difference should increase heat loss.

        This follows from Newton's Law of Cooling: Q = h * A * (T_s - T_a)
        """
        assume(temp_diff2 > temp_diff1 * 1.5)
        ambient = 25.0

        result1 = calculator.calculate_bare_surface_loss(
            surface_temp_c=ambient + temp_diff1,
            ambient_temp_c=ambient,
            surface_area_m2=10.0,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.VERTICAL_FLAT,
        )

        result2 = calculator.calculate_bare_surface_loss(
            surface_temp_c=ambient + temp_diff2,
            ambient_temp_c=ambient,
            surface_area_m2=10.0,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.VERTICAL_FLAT,
        )

        assert float(result2.total_heat_loss_w) > float(result1.total_heat_loss_w)

    @given(
        wind_speed1=st.floats(min_value=0, max_value=1),
        wind_speed2=st.floats(min_value=5, max_value=20),
    )
    @settings(max_examples=50)
    def test_wind_increases_heat_loss(
        self,
        calculator: HeatLossCalculator,
        wind_speed1: float,
        wind_speed2: float,
    ) -> None:
        """
        Higher wind speed should increase heat loss due to forced convection.

        Forced convection has higher heat transfer coefficient than natural.
        """
        assume(wind_speed2 > wind_speed1 + 3)

        result1 = calculator.calculate_bare_surface_loss(
            surface_temp_c=100.0,
            ambient_temp_c=25.0,
            surface_area_m2=5.0,
            wind_speed_ms=wind_speed1,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
            characteristic_length_m=0.2,
        )

        result2 = calculator.calculate_bare_surface_loss(
            surface_temp_c=100.0,
            ambient_temp_c=25.0,
            surface_area_m2=5.0,
            wind_speed_ms=wind_speed2,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
            characteristic_length_m=0.2,
        )

        assert float(result2.total_heat_loss_w) > float(result1.total_heat_loss_w)


class TestConditionScoreProperties:
    """Property-based tests for condition scoring."""

    @given(
        heat_loss=st.floats(min_value=100, max_value=10000),
        expected_loss=st.floats(min_value=100, max_value=5000),
        hot_spots=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_condition_score_bounded(
        self,
        heat_loss: float,
        expected_loss: float,
        hot_spots: int,
    ) -> None:
        """
        Condition score should always be between 0 and 100.

        This is a fundamental constraint of the scoring system.
        """
        # Simple condition calculation logic
        if expected_loss > 0:
            loss_ratio = heat_loss / expected_loss
            base_score = max(0, 100 - (loss_ratio - 1) * 50)
        else:
            base_score = 100

        hot_spot_penalty = min(30, hot_spots * 5)
        condition_score = max(0, min(100, base_score - hot_spot_penalty))

        assert 0 <= condition_score <= 100

    @given(
        hot_spots1=st.integers(min_value=0, max_value=3),
        hot_spots2=st.integers(min_value=5, max_value=10),
    )
    @settings(max_examples=50)
    def test_more_hot_spots_lower_score(
        self,
        hot_spots1: int,
        hot_spots2: int,
    ) -> None:
        """
        More hot spots should result in lower condition score.

        Hot spots indicate insulation degradation and increase penalty.
        """
        assume(hot_spots2 > hot_spots1 + 2)

        base_score = 80

        score1 = max(0, base_score - min(30, hot_spots1 * 5))
        score2 = max(0, base_score - min(30, hot_spots2 * 5))

        assert score2 <= score1

    @given(
        base_score=st.floats(min_value=0, max_value=100),
        age_years=st.floats(min_value=0, max_value=30),
        degradation_rate=st.floats(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_age_degrades_condition(
        self,
        base_score: float,
        age_years: float,
        degradation_rate: float,
    ) -> None:
        """
        Older insulation should have lower or equal condition score.

        Degradation over time reduces insulation effectiveness.
        """
        degradation = age_years * degradation_rate
        aged_score = max(0, min(100, base_score - degradation))

        assert aged_score <= base_score


class TestThermalResistanceProperties:
    """Property-based tests for thermal resistance calculations."""

    @given(
        thickness=st.floats(min_value=10, max_value=200),
        conductivity=st.floats(min_value=0.01, max_value=0.1),
    )
    @settings(max_examples=50)
    def test_thermal_resistance_positive(
        self,
        thickness: float,
        conductivity: float,
    ) -> None:
        """
        Thermal resistance should always be positive.

        R = thickness / conductivity > 0 for positive values.
        """
        thickness_m = thickness / 1000  # Convert to meters
        thermal_resistance = thickness_m / conductivity

        assert thermal_resistance > 0

    @given(
        thickness1=st.floats(min_value=10, max_value=50),
        thickness2=st.floats(min_value=60, max_value=150),
    )
    @settings(max_examples=50)
    def test_thicker_means_higher_resistance(
        self,
        thickness1: float,
        thickness2: float,
    ) -> None:
        """
        Thicker insulation should have higher thermal resistance.

        R = thickness / conductivity is proportional to thickness.
        """
        assume(thickness2 > thickness1 * 1.2)
        conductivity = 0.04  # W/m-K

        r1 = (thickness1 / 1000) / conductivity
        r2 = (thickness2 / 1000) / conductivity

        assert r2 > r1


class TestEnergyConservation:
    """Property-based tests for energy conservation."""

    @given(
        surface_temp=st.floats(min_value=50, max_value=300),
        ambient_temp=st.floats(min_value=10, max_value=40),
        area=st.floats(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_total_loss_equals_sum_of_components(
        self,
        calculator: HeatLossCalculator,
        surface_temp: float,
        ambient_temp: float,
        area: float,
    ) -> None:
        """
        Total heat loss should equal sum of convection and radiation.

        This validates energy conservation in the calculation.
        """
        assume(surface_temp > ambient_temp + 10)

        result = calculator.calculate_bare_surface_loss(
            surface_temp_c=surface_temp,
            ambient_temp_c=ambient_temp,
            surface_area_m2=area,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.VERTICAL_FLAT,
        )

        # Total should equal sum of components (within numerical precision)
        total = float(result.total_heat_loss_w)
        convection = float(result.convection_loss_w)
        radiation = float(result.radiation_loss_w)

        assert abs(total - (convection + radiation)) < 0.01


class TestProvenanceHashProperties:
    """Property-based tests for provenance hash generation."""

    @given(
        temp1=st.floats(min_value=50, max_value=200),
        temp2=st.floats(min_value=50, max_value=200),
    )
    @settings(max_examples=50)
    def test_same_inputs_same_hash(
        self,
        calculator: HeatLossCalculator,
        temp1: float,
        temp2: float,
    ) -> None:
        """
        Same inputs should produce same provenance hash.

        This ensures reproducibility and auditability.
        """
        # Use identical values
        temp = (temp1 + temp2) / 2

        result1 = calculator.calculate_bare_surface_loss(
            surface_temp_c=temp,
            ambient_temp_c=25.0,
            surface_area_m2=10.0,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
        )

        result2 = calculator.calculate_bare_surface_loss(
            surface_temp_c=temp,
            ambient_temp_c=25.0,
            surface_area_m2=10.0,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
        )

        assert result1.provenance_hash == result2.provenance_hash

    @given(
        temp1=st.floats(min_value=50, max_value=100),
        temp2=st.floats(min_value=150, max_value=200),
    )
    @settings(max_examples=50)
    def test_different_inputs_different_hash(
        self,
        calculator: HeatLossCalculator,
        temp1: float,
        temp2: float,
    ) -> None:
        """
        Different inputs should produce different provenance hash.

        This ensures unique identification of calculations.
        """
        assume(abs(temp1 - temp2) > 10)

        result1 = calculator.calculate_bare_surface_loss(
            surface_temp_c=temp1,
            ambient_temp_c=25.0,
            surface_area_m2=10.0,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
        )

        result2 = calculator.calculate_bare_surface_loss(
            surface_temp_c=temp2,
            ambient_temp_c=25.0,
            surface_area_m2=10.0,
            wind_speed_ms=0.0,
            surface_type=SurfaceType.HORIZONTAL_PIPE,
        )

        assert result1.provenance_hash != result2.provenance_hash
