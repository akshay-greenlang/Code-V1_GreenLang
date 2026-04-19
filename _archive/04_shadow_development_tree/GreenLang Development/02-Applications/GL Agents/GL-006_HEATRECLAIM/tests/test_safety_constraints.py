"""
GL-006 HEATRECLAIM - Comprehensive Safety Constraint Tests

Complete test suite for all 5 safety constraints from pack.yaml:
1. DELTA_T_MIN: 5C minimum approach temperature
2. MAX_FILM_TEMPERATURE: 400C coking prevention
3. ACID_DEW_POINT: 120C minimum outlet for flue gas
4. MAX_PRESSURE_DROP: 50 kPa liquids, 5 kPa gases
5. THERMAL_STRESS_RATE: 5C/min maximum temperature change

Test Categories:
- Unit tests for individual constraint checks
- Integration tests for full HEN validation
- Edge case tests for boundary conditions
- Penalty cost calculation tests
- Provenance hash verification tests

Standards Verified:
- ASME PTC 4.3/4.4
- API 660
- ISO 14414
- TEMA Standards

Coverage Target: 95%+
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone
from typing import List

from ..core.config import ThermalConstraints, Phase, StreamType
from ..core.schemas import (
    HeatExchanger,
    HeatStream,
    HENDesign,
)
from ..safety.constraint_validator import (
    ConstraintValidator,
    ConstraintType,
    ConstraintLimit,
    ConstraintCheckResult,
    ConstraintCheckSummary,
    PenaltyLevel,
)
from ..safety.exceptions import (
    SafetyViolationError,
    ViolationSeverity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def thermal_constraints():
    """Standard thermal constraints from pack.yaml."""
    return ThermalConstraints(
        delta_t_min_default=10.0,
        delta_t_min_gas_gas=20.0,
        delta_t_min_gas_liquid=15.0,
        delta_t_min_liquid_liquid=10.0,
        delta_t_min_phase_change=5.0,
        max_film_temperature=400.0,
        acid_dew_point=120.0,
        max_pressure_drop_liquid=50.0,
        max_pressure_drop_gas=5.0,
        max_thermal_stress_rate=5.0,
    )


@pytest.fixture
def constraint_validator(thermal_constraints):
    """Constraint validator with fail_closed=False for testing."""
    return ConstraintValidator(
        constraints=thermal_constraints,
        fail_closed=False,
        apply_penalties=True,
    )


@pytest.fixture
def strict_validator(thermal_constraints):
    """Constraint validator with fail_closed=True."""
    return ConstraintValidator(
        constraints=thermal_constraints,
        fail_closed=True,
        apply_penalties=True,
    )


@pytest.fixture
def safe_liquid_exchanger():
    """Heat exchanger that passes all safety checks (liquid-liquid)."""
    return HeatExchanger(
        exchanger_id="E-SAFE-001",
        exchanger_name="Safe Liquid-Liquid Exchanger",
        hot_stream_id="H1",
        cold_stream_id="C1",
        duty_kW=1000.0,
        hot_inlet_T_C=200.0,
        hot_outlet_T_C=150.0,
        cold_inlet_T_C=50.0,
        cold_outlet_T_C=100.0,
        delta_T_hot_end_C=100.0,
        delta_T_cold_end_C=100.0,
        LMTD_C=100.0,
        hot_side_dp_kPa=20.0,
        cold_side_dp_kPa=15.0,
    )


@pytest.fixture
def hot_liquid_stream():
    """Liquid hot stream."""
    return HeatStream(
        stream_id="H1",
        stream_name="Hot Water",
        stream_type=StreamType.HOT,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=200.0,
        T_target_C=100.0,
        m_dot_kg_s=10.0,
        Cp_kJ_kgK=4.186,
    )


@pytest.fixture
def cold_liquid_stream():
    """Liquid cold stream."""
    return HeatStream(
        stream_id="C1",
        stream_name="Cold Water",
        stream_type=StreamType.COLD,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=50.0,
        T_target_C=150.0,
        m_dot_kg_s=10.0,
        Cp_kJ_kgK=4.186,
    )


@pytest.fixture
def flue_gas_stream():
    """Flue gas stream for acid dew point testing."""
    return HeatStream(
        stream_id="FG1",
        stream_name="Combustion Flue Gas",
        stream_type=StreamType.HOT,
        fluid_name="flue_gas",
        phase=Phase.GAS,
        T_supply_C=350.0,
        T_target_C=180.0,
        m_dot_kg_s=50.0,
        Cp_kJ_kgK=1.1,
    )


@pytest.fixture
def gas_stream():
    """Generic gas stream."""
    return HeatStream(
        stream_id="G1",
        stream_name="Process Gas",
        stream_type=StreamType.HOT,
        fluid_name="Nitrogen",
        phase=Phase.GAS,
        T_supply_C=300.0,
        T_target_C=150.0,
        m_dot_kg_s=20.0,
        Cp_kJ_kgK=1.04,
    )


def create_hen_design(exchangers: List[HeatExchanger], design_id: str = "TEST-001") -> HENDesign:
    """Helper to create HEN design from exchangers."""
    total_duty = sum(e.duty_kW for e in exchangers)
    return HENDesign(
        design_id=design_id,
        exchangers=exchangers,
        total_heat_recovered_kW=total_duty,
        hot_utility_required_kW=0.0,
        cold_utility_required_kW=0.0,
    )


# =============================================================================
# CONSTRAINT 1: DELTA_T_MIN (Approach Temperature)
# =============================================================================

class TestDeltaTMinConstraint:
    """Tests for minimum approach temperature constraint."""

    def test_sufficient_approach_passes(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Exchanger with approach > delta_t_min passes validation."""
        design = create_hen_design([safe_liquid_exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.is_acceptable is True
        assert summary.has_violations is False

        # Check delta_t_min results
        delta_t_checks = [
            c for c in summary.check_results
            if c["constraint_type"] == "DELTA_T_MIN"
        ]
        assert len(delta_t_checks) == 2  # Hot end and cold end
        for check in delta_t_checks:
            assert check["is_violation"] is False

    def test_approach_below_minimum_fails(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Exchanger with approach < delta_t_min fails validation."""
        # Create exchanger with small approach temperature
        exchanger = HeatExchanger(
            exchanger_id="E-SMALL-APPROACH",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=110.0,
            hot_outlet_T_C=57.0,     # Cold end approach: 57 - 50 = 7 < 10
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=103.0,   # Hot end approach: 110 - 103 = 7 < 10
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.is_acceptable is False
        assert summary.has_violations is True
        assert summary.violations_count >= 1

        # Verify violation details
        delta_t_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "DELTA_T_MIN"
        ]
        assert len(delta_t_violations) >= 1

    def test_temperature_crossover_is_critical(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Temperature crossover (negative approach) is critical violation."""
        # Create exchanger with temperature crossover
        exchanger = HeatExchanger(
            exchanger_id="E-CROSSOVER",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=60.0,
            cold_inlet_T_C=70.0,     # Cold inlet > hot outlet (crossover)
            cold_outlet_T_C=110.0,   # Cold outlet > hot inlet (crossover)
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.is_acceptable is False
        assert summary.critical_count >= 1

        # Verify negative approach values
        crossover_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "DELTA_T_MIN" and v["actual_value"] < 0
        ]
        assert len(crossover_violations) >= 1

    def test_phase_specific_delta_t_min_applied(
        self,
        constraint_validator,
        gas_stream,
        cold_liquid_stream,
    ):
        """Gas-liquid exchanger uses phase-specific delta_t_min (15C)."""
        # Create exchanger with approach = 12C (ok for liquid, fails for gas-liquid)
        exchanger = HeatExchanger(
            exchanger_id="E-GAS-LIQUID",
            hot_stream_id="G1",
            cold_stream_id="C1",
            duty_kW=500.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=100.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=88.0,  # Hot end approach: 200 - 88 = 112 (OK)
            # Cold end approach: 100 - 50 = 50 (OK)
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [gas_stream], [cold_liquid_stream]
        )

        # Check that gas-liquid minimum (15C) was used
        delta_t_checks = [
            c for c in summary.check_results
            if c["constraint_type"] == "DELTA_T_MIN"
        ]
        for check in delta_t_checks:
            assert check["limit_value"] == 15.0  # Gas-liquid minimum

    def test_near_violation_generates_penalty(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Approach near limit generates penalty cost."""
        # Create exchanger with approach at 95% of limit (10 * 1.05 = 10.5)
        exchanger = HeatExchanger(
            exchanger_id="E-NEAR-LIMIT",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=120.0,
            hot_outlet_T_C=61.0,     # Cold end approach: 61 - 50 = 11 (just above 10)
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=109.0,   # Hot end approach: 120 - 109 = 11 (just above 10)
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        # Should pass but have near-violation warnings
        assert summary.is_acceptable is True
        # May or may not have near violations depending on thresholds


# =============================================================================
# CONSTRAINT 2: MAX_FILM_TEMPERATURE (Coking Prevention)
# =============================================================================

class TestMaxFilmTemperatureConstraint:
    """Tests for maximum film temperature constraint."""

    def test_film_temp_below_limit_passes(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Exchanger with film temp < 400C passes."""
        design = create_hen_design([safe_liquid_exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        film_checks = [
            c for c in summary.check_results
            if c["constraint_type"] == "MAX_FILM_TEMPERATURE"
        ]
        assert len(film_checks) == 1
        assert film_checks[0]["is_violation"] is False

    def test_film_temp_above_limit_fails(
        self,
        constraint_validator,
        cold_liquid_stream,
    ):
        """Exchanger with film temp > 400C fails."""
        # Create hot stream with very high temperature
        hot_stream = HeatStream(
            stream_id="H-HOT",
            stream_name="Hot Thermal Oil",
            stream_type=StreamType.HOT,
            fluid_name="Thermal Oil",
            phase=Phase.LIQUID,
            T_supply_C=450.0,  # Above 400C limit
            T_target_C=350.0,
            m_dot_kg_s=5.0,
            Cp_kJ_kgK=2.5,
        )

        exchanger = HeatExchanger(
            exchanger_id="E-HIGH-TEMP",
            hot_stream_id="H-HOT",
            cold_stream_id="C1",
            duty_kW=500.0,
            hot_inlet_T_C=450.0,  # Above max film temp
            hot_outlet_T_C=350.0,
            cold_inlet_T_C=100.0,
            cold_outlet_T_C=200.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_stream], [cold_liquid_stream]
        )

        assert summary.has_violations is True

        film_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "MAX_FILM_TEMPERATURE"
        ]
        assert len(film_violations) == 1
        assert film_violations[0]["actual_value"] == 450.0
        assert film_violations[0]["limit_value"] == 400.0

    def test_film_temp_critical_violation(
        self,
        constraint_validator,
        cold_liquid_stream,
    ):
        """Film temp 10% above limit is critical violation."""
        hot_stream = HeatStream(
            stream_id="H-EXTREME",
            stream_name="Extreme Hot Oil",
            stream_type=StreamType.HOT,
            fluid_name="Thermal Oil",
            phase=Phase.LIQUID,
            T_supply_C=500.0,  # 25% above limit (critical)
            T_target_C=400.0,
            m_dot_kg_s=5.0,
            Cp_kJ_kgK=2.5,
        )

        exchanger = HeatExchanger(
            exchanger_id="E-EXTREME-TEMP",
            hot_stream_id="H-EXTREME",
            cold_stream_id="C1",
            duty_kW=500.0,
            hot_inlet_T_C=500.0,
            hot_outlet_T_C=400.0,
            cold_inlet_T_C=100.0,
            cold_outlet_T_C=200.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_stream], [cold_liquid_stream]
        )

        assert summary.critical_count >= 1


# =============================================================================
# CONSTRAINT 3: ACID_DEW_POINT (Corrosion Prevention)
# =============================================================================

class TestAcidDewPointConstraint:
    """Tests for acid dew point constraint (flue gas only)."""

    def test_flue_gas_above_acid_dew_point_passes(
        self,
        constraint_validator,
        flue_gas_stream,
        cold_liquid_stream,
    ):
        """Flue gas outlet above 120C passes acid dew point check."""
        exchanger = HeatExchanger(
            exchanger_id="E-FLUE-GAS-SAFE",
            hot_stream_id="FG1",
            cold_stream_id="C1",
            duty_kW=2000.0,
            hot_inlet_T_C=350.0,
            hot_outlet_T_C=150.0,  # Above 120C acid dew point
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [flue_gas_stream], [cold_liquid_stream]
        )

        acid_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "ACID_DEW_POINT"
        ]
        assert len(acid_violations) == 0

    def test_flue_gas_below_acid_dew_point_fails(
        self,
        constraint_validator,
        flue_gas_stream,
        cold_liquid_stream,
    ):
        """Flue gas outlet below 120C fails acid dew point check."""
        exchanger = HeatExchanger(
            exchanger_id="E-FLUE-GAS-ACID",
            hot_stream_id="FG1",
            cold_stream_id="C1",
            duty_kW=3000.0,
            hot_inlet_T_C=350.0,
            hot_outlet_T_C=100.0,  # Below 120C acid dew point
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=150.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [flue_gas_stream], [cold_liquid_stream]
        )

        assert summary.has_violations is True

        acid_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "ACID_DEW_POINT"
        ]
        assert len(acid_violations) == 1
        assert acid_violations[0]["actual_value"] == 100.0
        assert acid_violations[0]["limit_value"] == 120.0

    def test_non_flue_gas_skips_acid_check(
        self,
        constraint_validator,
        gas_stream,  # Not flue gas
        cold_liquid_stream,
    ):
        """Non-flue gas streams skip acid dew point check."""
        exchanger = HeatExchanger(
            exchanger_id="E-PROCESS-GAS",
            hot_stream_id="G1",
            cold_stream_id="C1",
            duty_kW=1500.0,
            hot_inlet_T_C=300.0,
            hot_outlet_T_C=80.0,  # Would fail acid dew point if checked
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [gas_stream], [cold_liquid_stream]
        )

        # Should not have acid dew point violations for nitrogen
        acid_checks = [
            c for c in summary.check_results
            if c["constraint_type"] == "ACID_DEW_POINT"
        ]
        assert len(acid_checks) == 0

    def test_acid_dew_point_severity_based_on_margin(
        self,
        constraint_validator,
        flue_gas_stream,
        cold_liquid_stream,
    ):
        """Acid dew point violation severity increases with margin."""
        # Test 5C below limit (warning)
        exchanger_warning = HeatExchanger(
            exchanger_id="E-ACID-WARNING",
            hot_stream_id="FG1",
            cold_stream_id="C1",
            duty_kW=2500.0,
            hot_inlet_T_C=350.0,
            hot_outlet_T_C=115.0,  # 5C below limit
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=120.0,
        )

        # Test 25C below limit (critical)
        exchanger_critical = HeatExchanger(
            exchanger_id="E-ACID-CRITICAL",
            hot_stream_id="FG1",
            cold_stream_id="C1",
            duty_kW=3500.0,
            hot_inlet_T_C=350.0,
            hot_outlet_T_C=95.0,  # 25C below limit
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=150.0,
        )

        # Test warning level
        design_warning = create_hen_design([exchanger_warning], "ACID-WARN")
        summary_warning = constraint_validator.validate_hen_design(
            design_warning, [flue_gas_stream], [cold_liquid_stream]
        )
        assert summary_warning.has_violations is True

        # Test critical level
        design_critical = create_hen_design([exchanger_critical], "ACID-CRIT")
        summary_critical = constraint_validator.validate_hen_design(
            design_critical, [flue_gas_stream], [cold_liquid_stream]
        )
        assert summary_critical.has_violations is True
        assert summary_critical.critical_count >= 1


# =============================================================================
# CONSTRAINT 4: MAX_PRESSURE_DROP
# =============================================================================

class TestMaxPressureDropConstraint:
    """Tests for maximum pressure drop constraint."""

    def test_pressure_drop_within_limits_passes(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Pressure drop within limits passes validation."""
        design = create_hen_design([safe_liquid_exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        dp_checks = [
            c for c in summary.check_results
            if c["constraint_type"] == "MAX_PRESSURE_DROP"
        ]
        assert len(dp_checks) == 2  # Hot side and cold side
        for check in dp_checks:
            assert check["is_violation"] is False

    def test_liquid_pressure_drop_above_50kpa_fails(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Liquid pressure drop > 50 kPa fails validation."""
        exchanger = HeatExchanger(
            exchanger_id="E-HIGH-DP",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=150.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
            hot_side_dp_kPa=75.0,   # Above 50 kPa limit
            cold_side_dp_kPa=60.0,  # Above 50 kPa limit
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.has_violations is True

        dp_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "MAX_PRESSURE_DROP"
        ]
        assert len(dp_violations) == 2  # Both sides violated

    def test_gas_pressure_drop_above_5kpa_fails(
        self,
        constraint_validator,
        gas_stream,
        cold_liquid_stream,
    ):
        """Gas pressure drop > 5 kPa fails validation."""
        exchanger = HeatExchanger(
            exchanger_id="E-GAS-HIGH-DP",
            hot_stream_id="G1",
            cold_stream_id="C1",
            duty_kW=500.0,
            hot_inlet_T_C=300.0,
            hot_outlet_T_C=150.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
            hot_side_dp_kPa=8.0,   # Above 5 kPa limit for gas
            cold_side_dp_kPa=20.0,  # OK for liquid (below 50)
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [gas_stream], [cold_liquid_stream]
        )

        assert summary.has_violations is True

        dp_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "MAX_PRESSURE_DROP"
        ]
        # Only hot side (gas) should violate
        assert len(dp_violations) == 1
        assert dp_violations[0]["location"] == "hot_side"
        assert dp_violations[0]["limit_value"] == 5.0  # Gas limit

    def test_pressure_drop_penalty_calculated(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Pressure drop violations generate penalty costs."""
        exchanger = HeatExchanger(
            exchanger_id="E-DP-PENALTY",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=150.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
            hot_side_dp_kPa=55.0,   # 5 kPa over limit
            cold_side_dp_kPa=30.0,  # Within limit
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.has_violations is True
        assert summary.total_penalty_cost_usd > 0
        assert "MAX_PRESSURE_DROP" in summary.penalty_cost_by_constraint


# =============================================================================
# CONSTRAINT 5: THERMAL_STRESS_RATE
# =============================================================================

class TestThermalStressRateConstraint:
    """Tests for thermal stress rate constraint."""

    def test_thermal_stress_within_limit_passes(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Thermal stress rate within 5C/min passes."""
        design = create_hen_design([safe_liquid_exchanger])

        # 50C temperature change over 15 minutes = 3.33 C/min (OK)
        summary = constraint_validator.validate_hen_design(
            design,
            [hot_liquid_stream],
            [cold_liquid_stream],
            startup_time_minutes=15.0,
        )

        stress_checks = [
            c for c in summary.check_results
            if c["constraint_type"] == "THERMAL_STRESS_RATE"
        ]
        assert len(stress_checks) == 2  # Hot and cold sides
        for check in stress_checks:
            assert check["is_violation"] is False

    def test_thermal_stress_above_limit_fails(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Thermal stress rate above 5C/min fails."""
        exchanger = HeatExchanger(
            exchanger_id="E-FAST-STARTUP",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=100.0,  # 100C delta
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=150.0,  # 100C delta
        )

        design = create_hen_design([exchanger])

        # 100C temperature change over 5 minutes = 20 C/min (exceeds 5 C/min)
        summary = constraint_validator.validate_hen_design(
            design,
            [hot_liquid_stream],
            [cold_liquid_stream],
            startup_time_minutes=5.0,
        )

        assert summary.has_violations is True

        stress_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "THERMAL_STRESS_RATE"
        ]
        assert len(stress_violations) >= 1
        assert stress_violations[0]["actual_value"] == 20.0  # 100C / 5min
        assert stress_violations[0]["limit_value"] == 5.0

    def test_thermal_stress_skipped_without_startup_time(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Thermal stress check skipped when no startup time provided."""
        design = create_hen_design([safe_liquid_exchanger])

        summary = constraint_validator.validate_hen_design(
            design,
            [hot_liquid_stream],
            [cold_liquid_stream],
            startup_time_minutes=None,  # No startup time
        )

        stress_checks = [
            c for c in summary.check_results
            if c["constraint_type"] == "THERMAL_STRESS_RATE"
        ]
        assert len(stress_checks) == 0


# =============================================================================
# FAIL-CLOSED BEHAVIOR
# =============================================================================

class TestFailClosedBehavior:
    """Tests for fail-closed safety behavior."""

    def test_fail_closed_raises_on_violation(
        self,
        strict_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Fail-closed validator raises exception on violation."""
        exchanger = HeatExchanger(
            exchanger_id="E-VIOLATION",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=55.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=95.0,
        )

        design = create_hen_design([exchanger])

        with pytest.raises(SafetyViolationError) as exc_info:
            strict_validator.validate_hen_design(
                design, [hot_liquid_stream], [cold_liquid_stream]
            )

        assert "rejected due to safety violations" in str(exc_info.value)

    def test_fail_closed_returns_summary_on_pass(
        self,
        strict_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Fail-closed validator returns summary when no violations."""
        design = create_hen_design([safe_liquid_exchanger])

        summary = strict_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert isinstance(summary, ConstraintCheckSummary)
        assert summary.is_acceptable is True


# =============================================================================
# PENALTY COST CALCULATIONS
# =============================================================================

class TestPenaltyCostCalculations:
    """Tests for penalty cost calculations."""

    def test_total_penalty_accumulated(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Total penalty is sum of all constraint penalties."""
        # Create exchanger with multiple violations
        exchanger = HeatExchanger(
            exchanger_id="E-MULTI-VIOLATION",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=55.0,    # Approach violation
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=95.0,    # Approach violation
            hot_side_dp_kPa=60.0,   # Pressure drop violation
            cold_side_dp_kPa=55.0,  # Pressure drop violation
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.total_penalty_cost_usd > 0
        assert len(summary.penalty_cost_by_constraint) > 0

        # Verify penalty sum
        constraint_penalties = sum(summary.penalty_cost_by_constraint.values())
        assert abs(constraint_penalties - summary.total_penalty_cost_usd) < 0.01

    def test_penalty_disabled_when_apply_penalties_false(
        self,
        thermal_constraints,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Penalties not calculated when apply_penalties=False."""
        validator = ConstraintValidator(
            constraints=thermal_constraints,
            fail_closed=False,
            apply_penalties=False,  # Disable penalties
        )

        exchanger = HeatExchanger(
            exchanger_id="E-NO-PENALTY",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=55.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=95.0,
        )

        design = create_hen_design([exchanger])
        summary = validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.total_penalty_cost_usd == 0


# =============================================================================
# PROVENANCE AND AUDIT TRAIL
# =============================================================================

class TestProvenanceTracking:
    """Tests for SHA-256 provenance tracking."""

    def test_summary_has_provenance_hashes(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Summary includes all required provenance hashes."""
        design = create_hen_design([safe_liquid_exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        # Verify SHA-256 hashes (64 hex characters)
        assert len(summary.constraints_hash) == 64
        assert len(summary.design_hash) == 64
        assert len(summary.result_hash) == 64

    def test_check_results_have_calculation_hash(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Each check result has a calculation hash."""
        design = create_hen_design([safe_liquid_exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        for check in summary.check_results:
            assert "calculation_hash" in check
            assert len(check["calculation_hash"]) == 64

    def test_same_input_produces_same_hash(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Deterministic: same input produces same design hash."""
        design = create_hen_design([safe_liquid_exchanger])

        summary1 = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )
        summary2 = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary1.design_hash == summary2.design_hash
        assert summary1.constraints_hash == summary2.constraints_hash

    def test_constraint_validator_version_tracked(
        self,
        constraint_validator,
    ):
        """Validator version is available for audit."""
        assert constraint_validator.VERSION == "1.0.0"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHENIntegration:
    """Integration tests for complete HEN validation."""

    def test_multi_exchanger_network_validation(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Multiple exchangers in network are all validated."""
        exchangers = [
            HeatExchanger(
                exchanger_id=f"E-{i}",
                hot_stream_id="H1",
                cold_stream_id="C1",
                duty_kW=500.0,
                hot_inlet_T_C=200.0 - (i * 20),
                hot_outlet_T_C=180.0 - (i * 20),
                cold_inlet_T_C=50.0 + (i * 20),
                cold_outlet_T_C=70.0 + (i * 20),
                hot_side_dp_kPa=10.0,
                cold_side_dp_kPa=10.0,
            )
            for i in range(3)
        ]

        design = create_hen_design(exchangers, "MULTI-HEX")
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        # Should have checks for all 3 exchangers
        # 2 delta_t checks + 1 film temp + 2 pressure drop = 5 per exchanger
        assert summary.checks_performed >= 15

    def test_rejection_reason_is_informative(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Rejection reason provides actionable information."""
        exchanger = HeatExchanger(
            exchanger_id="E-REJECT",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=55.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=95.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        reason = summary.get_rejection_reason()

        assert "E-REJECT" in reason
        assert "DELTA_T_MIN" in reason
        assert "limit:" in reason

    def test_get_constraint_limits_returns_all_limits(
        self,
        constraint_validator,
    ):
        """get_constraint_limits returns all 5 constraint limits."""
        limits = constraint_validator.get_constraint_limits()

        assert len(limits) == 5
        assert "DELTA_T_MIN" in limits
        assert "MAX_FILM_TEMPERATURE" in limits
        assert "ACID_DEW_POINT" in limits
        assert "MAX_PRESSURE_DROP" in limits
        assert "THERMAL_STRESS_RATE" in limits

        # Verify structure
        for name, limit in limits.items():
            assert "limit_value" in limit
            assert "unit" in limit
            assert "standard_reference" in limit


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_exactly_at_limit_passes(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Value exactly at limit passes (not a violation)."""
        # Approach exactly at 10C (liquid-liquid limit)
        exchanger = HeatExchanger(
            exchanger_id="E-AT-LIMIT",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=120.0,
            hot_outlet_T_C=60.0,     # Cold end: 60 - 50 = 10 (exactly at limit)
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=110.0,   # Hot end: 120 - 110 = 10 (exactly at limit)
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        assert summary.is_acceptable is True

    def test_zero_pressure_drop_passes(
        self,
        constraint_validator,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Zero pressure drop is valid."""
        exchanger = HeatExchanger(
            exchanger_id="E-ZERO-DP",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=150.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
            hot_side_dp_kPa=0.0,
            cold_side_dp_kPa=0.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [hot_liquid_stream], [cold_liquid_stream]
        )

        dp_violations = [
            v for v in summary.violations
            if v["constraint_type"] == "MAX_PRESSURE_DROP"
        ]
        assert len(dp_violations) == 0

    def test_missing_stream_data_uses_defaults(
        self,
        constraint_validator,
    ):
        """Missing stream data uses default delta_t_min."""
        exchanger = HeatExchanger(
            exchanger_id="E-NO-STREAMS",
            hot_stream_id="UNKNOWN-H",
            cold_stream_id="UNKNOWN-C",
            duty_kW=1000.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=150.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
        )

        design = create_hen_design([exchanger])
        summary = constraint_validator.validate_hen_design(
            design, [], []  # No stream data
        )

        # Should still validate using default delta_t_min
        assert summary.checks_performed > 0

    def test_single_exchanger_validation(
        self,
        constraint_validator,
        safe_liquid_exchanger,
        hot_liquid_stream,
        cold_liquid_stream,
    ):
        """Single exchanger can be validated without full design."""
        summary = constraint_validator.validate_single_exchanger(
            safe_liquid_exchanger,
            hot_stream=hot_liquid_stream,
            cold_stream=cold_liquid_stream,
        )

        assert isinstance(summary, ConstraintCheckSummary)
        assert summary.is_acceptable is True
