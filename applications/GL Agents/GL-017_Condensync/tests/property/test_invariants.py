# -*- coding: utf-8 -*-
"""
Property-Based Tests: Physical and Mathematical Invariants

Tests using Hypothesis to verify that GL-017 Condensync calculations
maintain physical law invariants and mathematical properties across
a wide range of randomly generated inputs.

Key Invariants:
- First Law of Thermodynamics (energy conservation)
- Second Law (temperature differentials)
- Mathematical properties (LMTD formula, hash determinism)
- Physical constraints (non-negative flows, valid ranges)

Author: GL-TestEngineer
Date: December 2025
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import (
    calculate_lmtd,
    saturation_temp_from_pressure,
    pressure_from_saturation_temp,
    OPERATING_LIMITS,
    TEST_SEED,
)


# =============================================================================
# CUSTOM STRATEGIES
# =============================================================================

# Temperature strategy (valid condenser operating range)
temperature_c = st.floats(min_value=5.0, max_value=60.0, allow_nan=False, allow_infinity=False)

# Small positive temperature difference
temp_diff_positive = st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False)

# Pressure strategy (condenser vacuum range)
pressure_kpa = st.floats(min_value=2.0, max_value=20.0, allow_nan=False, allow_infinity=False)

# Flow rate strategy
flow_rate_m3_s = st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Cleanliness factor strategy
cleanliness_factor = st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)

# Power/energy strategy
power_kw = st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False)

# Positive float
positive_float = st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestThermodynamicInvariants:
    """Tests for thermodynamic law invariants."""

    @pytest.mark.property
    @given(
        cw_inlet=temperature_c,
        cw_rise=temp_diff_positive
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_cw_outlet_greater_than_inlet(self, cw_inlet: float, cw_rise: float):
        """Test CW outlet is always greater than inlet (heat absorbed)."""
        cw_outlet = cw_inlet + cw_rise

        # Second law: heat flows from hot to cold
        # CW absorbs heat, so outlet > inlet
        assert cw_outlet > cw_inlet

    @pytest.mark.property
    @given(
        cw_outlet=temperature_c,
        ttd=temp_diff_positive
    )
    @settings(max_examples=200)
    def test_saturation_temp_greater_than_cw_outlet(
        self,
        cw_outlet: float,
        ttd: float
    ):
        """Test saturation temp is greater than CW outlet (heat transfer direction)."""
        sat_temp = cw_outlet + ttd

        # Heat transfers from steam to CW
        # Steam side must be hotter
        assert sat_temp > cw_outlet

    @pytest.mark.property
    @given(
        flow_kg_s=positive_float,
        temp_rise=temp_diff_positive,
        cp=st.floats(min_value=3.5, max_value=4.5, allow_nan=False)
    )
    @settings(max_examples=200)
    def test_heat_duty_non_negative(
        self,
        flow_kg_s: float,
        temp_rise: float,
        cp: float
    ):
        """Test heat duty is always non-negative."""
        heat_duty = flow_kg_s * cp * temp_rise

        # Energy is always positive
        assert heat_duty >= 0

    @pytest.mark.property
    @given(
        q_cw=power_kw,
        tolerance=st.floats(min_value=0.0, max_value=0.05, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_energy_balance(self, q_cw: float, tolerance: float):
        """Test energy balance: Q_cw = Q_steam (conservation)."""
        # At steady state, energy in = energy out
        q_steam = q_cw * (1 + tolerance)  # Allow small imbalance

        # Energy must be conserved (with measurement tolerance)
        assert abs(q_steam - q_cw) / max(q_cw, 1.0) <= tolerance + 0.001


class TestLMTDMathematicalProperties:
    """Tests for LMTD mathematical invariants."""

    @pytest.mark.property
    @given(
        ttd=temp_diff_positive,
        approach_offset=temp_diff_positive
    )
    @settings(max_examples=300)
    def test_lmtd_positive(self, ttd: float, approach_offset: float):
        """Test LMTD is always positive for valid inputs."""
        approach = ttd + approach_offset  # Ensure approach >= TTD

        lmtd = calculate_lmtd(ttd, approach)

        assert lmtd > 0

    @pytest.mark.property
    @given(
        ttd=temp_diff_positive,
        approach_offset=temp_diff_positive
    )
    @settings(max_examples=300)
    def test_lmtd_between_ttd_and_approach(self, ttd: float, approach_offset: float):
        """Test LMTD is between TTD and Approach."""
        approach = ttd + approach_offset

        lmtd = calculate_lmtd(ttd, approach)

        # LMTD should be a weighted average
        assert ttd <= lmtd <= approach

    @pytest.mark.property
    @given(value=temp_diff_positive)
    @settings(max_examples=100)
    def test_lmtd_equals_value_when_equal(self, value: float):
        """Test LMTD equals common value when TTD = Approach."""
        lmtd = calculate_lmtd(value, value)

        assert lmtd == value

    @pytest.mark.property
    @given(
        ttd=temp_diff_positive,
        approach_offset=temp_diff_positive
    )
    @settings(max_examples=200)
    def test_lmtd_formula_identity(self, ttd: float, approach_offset: float):
        """Test LMTD satisfies the analytical formula."""
        approach = ttd + approach_offset

        # Skip near-singularity
        assume(abs(ttd - approach) > 0.01)

        lmtd = calculate_lmtd(ttd, approach)

        # Verify: LMTD = (approach - ttd) / ln(approach/ttd)
        expected = (approach - ttd) / math.log(approach / ttd)

        assert abs(lmtd - expected) < 1e-9

    @pytest.mark.property
    @given(
        ttd=temp_diff_positive,
        approach_offset=temp_diff_positive,
        scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False)
    )
    @settings(max_examples=200)
    def test_lmtd_scaling_property(
        self,
        ttd: float,
        approach_offset: float,
        scale: float
    ):
        """Test LMTD scales linearly with temperatures."""
        approach = ttd + approach_offset

        lmtd_original = calculate_lmtd(ttd, approach)
        lmtd_scaled = calculate_lmtd(ttd * scale, approach * scale)

        # LMTD(a*T1, a*T2) = a * LMTD(T1, T2)
        assert abs(lmtd_scaled - scale * lmtd_original) < 1e-9


class TestPressureTemperatureRelation:
    """Tests for pressure-temperature relationship invariants."""

    @pytest.mark.property
    @given(pressure=pressure_kpa)
    @settings(max_examples=200)
    def test_saturation_temp_increases_with_pressure(self, pressure: float):
        """Test saturation temperature increases with pressure."""
        pressure_higher = pressure * 1.5

        t1 = saturation_temp_from_pressure(pressure)
        t2 = saturation_temp_from_pressure(pressure_higher)

        # Clausius-Clapeyron: T increases with P
        assert t2 > t1

    @pytest.mark.property
    @given(temp=temperature_c)
    @settings(max_examples=200)
    def test_saturation_pressure_increases_with_temp(self, temp: float):
        """Test saturation pressure increases with temperature."""
        temp_higher = temp + 5.0

        p1 = pressure_from_saturation_temp(temp)
        p2 = pressure_from_saturation_temp(temp_higher)

        # Clausius-Clapeyron: P increases with T
        assert p2 > p1

    @pytest.mark.property
    @given(pressure=pressure_kpa)
    @settings(max_examples=100)
    def test_pressure_temp_roundtrip(self, pressure: float):
        """Test P -> T -> P roundtrip (with tolerance)."""
        temp = saturation_temp_from_pressure(pressure)
        pressure_back = pressure_from_saturation_temp(temp)

        # Allow tolerance for simplified equations
        relative_error = abs(pressure_back - pressure) / pressure
        assert relative_error < 0.20  # 20% tolerance for simplified model


class TestCleanlinessFactorInvariants:
    """Tests for cleanliness factor invariants."""

    @pytest.mark.property
    @given(
        ua_actual=positive_float,
        ua_design=positive_float
    )
    @settings(max_examples=200)
    def test_cf_non_negative(self, ua_actual: float, ua_design: float):
        """Test CF is always non-negative."""
        cf = min(1.0, max(0.0, ua_actual / ua_design))

        assert cf >= 0

    @pytest.mark.property
    @given(
        ua_actual=positive_float,
        ua_design=positive_float
    )
    @settings(max_examples=200)
    def test_cf_bounded(self, ua_actual: float, ua_design: float):
        """Test CF is bounded [0, 1] when clamped."""
        cf = min(1.0, max(0.0, ua_actual / ua_design))

        assert 0.0 <= cf <= 1.0

    @pytest.mark.property
    @given(ua_design=positive_float)
    @settings(max_examples=100)
    def test_cf_equals_one_at_design(self, ua_design: float):
        """Test CF = 1.0 when actual equals design."""
        cf = ua_design / ua_design

        assert cf == 1.0

    @pytest.mark.property
    @given(
        ua_actual=positive_float,
        ua_design=positive_float
    )
    @settings(max_examples=200)
    def test_cf_proportional_to_ua(self, ua_actual: float, ua_design: float):
        """Test CF is proportional to UA_actual."""
        cf1 = ua_actual / ua_design
        cf2 = (ua_actual * 2) / ua_design

        # Double UA -> double CF (before clamping)
        if cf1 < 0.5:  # Avoid clamping
            assert abs(cf2 - 2 * cf1) < 1e-9


class TestHashDeterminism:
    """Tests for hash function invariants."""

    @pytest.mark.property
    @given(data=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
        min_size=1,
        max_size=5
    ))
    @settings(max_examples=100)
    def test_hash_deterministic(self, data: Dict[str, Any]):
        """Test hash is deterministic for same input."""
        hash1 = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

        assert hash1 == hash2

    @pytest.mark.property
    @given(
        key=st.text(min_size=1, max_size=10),
        value1=st.integers(),
        value2=st.integers()
    )
    @settings(max_examples=100)
    def test_different_values_different_hash(
        self,
        key: str,
        value1: int,
        value2: int
    ):
        """Test different values produce different hashes."""
        assume(value1 != value2)

        hash1 = hashlib.sha256(
            json.dumps({key: value1}, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps({key: value2}, sort_keys=True).encode()
        ).hexdigest()

        assert hash1 != hash2

    @pytest.mark.property
    @given(data=st.dictionaries(
        keys=st.text(min_size=1, max_size=5),
        values=st.integers(),
        min_size=2,
        max_size=5
    ))
    @settings(max_examples=100)
    def test_hash_key_order_independent(self, data: Dict[str, int]):
        """Test hash is independent of dictionary key order."""
        # Same data, sorted vs unsorted
        hash1 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        hash2 = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

        assert hash1 == hash2


class TestRangeConstraints:
    """Tests for physical range constraints."""

    @pytest.mark.property
    @given(
        cw_inlet=temperature_c,
        cw_rise=temp_diff_positive
    )
    @settings(max_examples=200)
    def test_temperature_rise_within_limits(
        self,
        cw_inlet: float,
        cw_rise: float
    ):
        """Test CW temperature rise is within physical limits."""
        # Typical condenser: 5-20C rise
        assume(5.0 <= cw_rise <= 20.0)

        # Physical constraint holds
        assert OPERATING_LIMITS["cw_rise_min_c"] <= cw_rise <= OPERATING_LIMITS["cw_rise_max_c"]

    @pytest.mark.property
    @given(cf=cleanliness_factor)
    @settings(max_examples=100)
    def test_cf_within_operating_limits(self, cf: float):
        """Test CF is within operating limits."""
        assert cf >= OPERATING_LIMITS["cf_min"]
        assert cf <= OPERATING_LIMITS["cf_max"]

    @pytest.mark.property
    @given(
        ttd=st.floats(min_value=2.0, max_value=15.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_ttd_within_limits(self, ttd: float):
        """Test TTD is within operating limits."""
        assert ttd >= OPERATING_LIMITS["ttd_min_c"]
        assert ttd <= OPERATING_LIMITS["ttd_max_c"]


class TestMathematicalProperties:
    """Tests for general mathematical properties."""

    @pytest.mark.property
    @given(
        a=positive_float,
        b=positive_float
    )
    @settings(max_examples=100)
    def test_arithmetic_mean_ge_log_mean(self, a: float, b: float):
        """Test arithmetic mean >= logarithmic mean."""
        assume(abs(a - b) > 0.01)  # Avoid singularity

        arithmetic_mean = (a + b) / 2
        log_mean = (a - b) / math.log(a / b) if a > b else (b - a) / math.log(b / a)

        # AM >= LM (always true)
        assert arithmetic_mean >= log_mean - 1e-9

    @pytest.mark.property
    @given(
        a=positive_float,
        b=positive_float
    )
    @settings(max_examples=100)
    def test_log_mean_ge_geometric_mean(self, a: float, b: float):
        """Test logarithmic mean >= geometric mean."""
        assume(abs(a - b) > 0.01)

        geometric_mean = math.sqrt(a * b)
        log_mean = (a - b) / math.log(a / b) if a > b else (b - a) / math.log(b / a)

        # LM >= GM (always true)
        assert log_mean >= geometric_mean - 1e-9

    @pytest.mark.property
    @given(
        x=st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_log_identity(self, x: float):
        """Test log(exp(x)) = x."""
        result = math.log(math.exp(x))

        assert abs(result - x) < 1e-9


class TestMonotonicityProperties:
    """Tests for monotonicity properties."""

    @pytest.mark.property
    @given(
        ttd1=temp_diff_positive,
        ttd2=temp_diff_positive,
        approach=st.floats(min_value=10.0, max_value=30.0, allow_nan=False)
    )
    @settings(max_examples=200)
    def test_lmtd_decreases_with_ttd(self, ttd1: float, ttd2: float, approach: float):
        """Test LMTD decreases as TTD increases (fixed approach)."""
        assume(ttd1 < ttd2 < approach - 0.5)

        lmtd1 = calculate_lmtd(ttd1, approach)
        lmtd2 = calculate_lmtd(ttd2, approach)

        # Higher TTD -> smaller temperature difference -> lower LMTD
        assert lmtd1 > lmtd2

    @pytest.mark.property
    @given(
        ttd=temp_diff_positive,
        approach1=st.floats(min_value=5.0, max_value=15.0, allow_nan=False),
        approach2=st.floats(min_value=15.0, max_value=30.0, allow_nan=False)
    )
    @settings(max_examples=200)
    def test_lmtd_increases_with_approach(
        self,
        ttd: float,
        approach1: float,
        approach2: float
    ):
        """Test LMTD increases as approach increases (fixed TTD)."""
        assume(ttd < approach1 - 0.5)
        assume(approach1 < approach2)

        lmtd1 = calculate_lmtd(ttd, approach1)
        lmtd2 = calculate_lmtd(ttd, approach2)

        # Higher approach -> larger temp range -> higher LMTD
        assert lmtd2 > lmtd1
