"""
Tests for GL-006 HEATRECLAIM SafetyValidator.

Comprehensive test suite for safety constraint enforcement including:
- Approach temperature validation
- Film temperature checks
- Acid dew point prevention
- Pressure drop limits
- Fail-closed behavior

Standards Verified:
- ASME PTC 4.3/4.4
- API 660
- ISO 14414
"""

import pytest
from datetime import datetime, timezone

from ..core.config import ThermalConstraints, Phase, StreamType
from ..core.schemas import (
    HeatExchanger,
    HeatStream,
    HENDesign,
)
from ..safety import (
    SafetyValidator,
    SafetyValidationResult,
    SafetyViolationError,
    ApproachTemperatureViolation,
    FilmTemperatureViolation,
    AcidDewPointViolation,
    PressureDropViolation,
)
from ..safety.safety_validator import ValidationMode
from ..safety.exceptions import ViolationSeverity


@pytest.fixture
def default_constraints():
    """Default thermal constraints for testing."""
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
def safe_exchanger():
    """Heat exchanger that passes all safety checks."""
    return HeatExchanger(
        exchanger_id="E-101",
        exchanger_name="Safe Exchanger",
        hot_stream_id="H1",
        cold_stream_id="C1",
        duty_kW=1000.0,
        hot_inlet_T_C=200.0,
        hot_outlet_T_C=150.0,
        cold_inlet_T_C=50.0,
        cold_outlet_T_C=100.0,
        delta_T_hot_end_C=100.0,  # 200 - 100 = 100
        delta_T_cold_end_C=100.0,  # 150 - 50 = 100
        LMTD_C=100.0,
        hot_side_dp_kPa=20.0,
        cold_side_dp_kPa=15.0,
    )


@pytest.fixture
def hot_stream_liquid():
    """Liquid hot stream."""
    return HeatStream(
        stream_id="H1",
        stream_name="Hot Liquid",
        stream_type=StreamType.HOT,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=200.0,
        T_target_C=100.0,
        m_dot_kg_s=10.0,
        Cp_kJ_kgK=4.186,
    )


@pytest.fixture
def cold_stream_liquid():
    """Liquid cold stream."""
    return HeatStream(
        stream_id="C1",
        stream_name="Cold Liquid",
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
        stream_name="Flue Gas",
        stream_type=StreamType.HOT,
        fluid_name="flue_gas",
        phase=Phase.GAS,
        T_supply_C=300.0,
        T_target_C=150.0,
        m_dot_kg_s=50.0,
        Cp_kJ_kgK=1.1,
    )


class TestSafetyValidatorInit:
    """Tests for SafetyValidator initialization."""

    def test_init_with_default_constraints(self, default_constraints):
        """Validator initializes with default constraints."""
        validator = SafetyValidator(default_constraints)
        assert validator.constraints == default_constraints
        assert validator.mode == ValidationMode.STRICT
        assert validator.fail_closed is True

    def test_init_with_relaxed_mode(self, default_constraints):
        """Validator can be initialized in relaxed mode."""
        validator = SafetyValidator(
            default_constraints,
            mode=ValidationMode.RELAXED,
            fail_closed=False,
        )
        assert validator.mode == ValidationMode.RELAXED
        assert validator.fail_closed is False

    def test_constraints_hash_is_deterministic(self, default_constraints):
        """Constraints hash is deterministic for same input."""
        validator1 = SafetyValidator(default_constraints)
        validator2 = SafetyValidator(default_constraints)
        assert validator1._constraints_hash == validator2._constraints_hash


class TestApproachTemperatureValidation:
    """Tests for approach temperature constraint."""

    def test_safe_exchanger_passes(
        self,
        default_constraints,
        safe_exchanger,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Exchanger with sufficient approach temperature passes."""
        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[safe_exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )

        assert result.is_safe is True
        assert result.validation_passed is True
        assert result.total_violations == 0

    def test_approach_violation_detected(
        self,
        default_constraints,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Exchanger with insufficient approach temperature fails."""
        # Create exchanger with small approach temperature
        exchanger = HeatExchanger(
            exchanger_id="E-102",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=55.0,     # 55 - 50 = 5 < 10 required
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=95.0,    # 100 - 95 = 5 < 10 required
        )

        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )

        assert result.is_safe is True  # Not critical
        assert result.validation_passed is False
        assert result.total_violations >= 1
        assert any(
            v.constraint_tag == "DELTA_T_MIN" for v in result.violations
        )

    def test_temperature_crossover_is_critical(
        self,
        default_constraints,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Temperature crossover (negative approach) is critical violation."""
        # Create exchanger with temperature crossover
        exchanger = HeatExchanger(
            exchanger_id="E-103",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=60.0,
            cold_inlet_T_C=70.0,     # Cold inlet > hot outlet = crossover
            cold_outlet_T_C=110.0,   # Cold outlet > hot inlet = crossover
        )

        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )

        assert result.is_safe is False  # Critical violation
        assert result.critical_violations >= 1


class TestFilmTemperatureValidation:
    """Tests for film temperature constraint."""

    def test_high_film_temperature_violation(
        self,
        default_constraints,
        cold_stream_liquid,
    ):
        """Film temperature above limit raises violation."""
        # Create exchanger with high inlet temperature
        hot_stream = HeatStream(
            stream_id="H2",
            stream_name="Hot Oil",
            stream_type=StreamType.HOT,
            fluid_name="Thermal Oil",
            phase=Phase.LIQUID,
            T_supply_C=450.0,  # Above 400C limit
            T_target_C=350.0,
            m_dot_kg_s=5.0,
            Cp_kJ_kgK=2.5,
        )

        exchanger = HeatExchanger(
            exchanger_id="E-104",
            hot_stream_id="H2",
            cold_stream_id="C1",
            duty_kW=500.0,
            hot_inlet_T_C=450.0,  # Above max film temp
            hot_outlet_T_C=350.0,
            cold_inlet_T_C=100.0,
            cold_outlet_T_C=200.0,
        )

        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=500.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [hot_stream], [cold_stream_liquid]
        )

        assert result.total_violations >= 1
        film_violations = [
            v for v in result.violations
            if v.constraint_tag == "MAX_FILM_TEMPERATURE"
        ]
        assert len(film_violations) >= 1


class TestAcidDewPointValidation:
    """Tests for acid dew point constraint."""

    def test_flue_gas_below_acid_dew_point(
        self,
        default_constraints,
        flue_gas_stream,
        cold_stream_liquid,
    ):
        """Flue gas outlet below acid dew point raises violation."""
        # Create exchanger with outlet below acid dew point
        exchanger = HeatExchanger(
            exchanger_id="E-105",
            hot_stream_id="FG1",
            cold_stream_id="C1",
            duty_kW=2000.0,
            hot_inlet_T_C=300.0,
            hot_outlet_T_C=100.0,  # Below 120C acid dew point
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=150.0,
        )

        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=2000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [flue_gas_stream], [cold_stream_liquid]
        )

        assert result.total_violations >= 1
        acid_violations = [
            v for v in result.violations
            if v.constraint_tag == "ACID_DEW_POINT"
        ]
        assert len(acid_violations) >= 1

    def test_flue_gas_above_acid_dew_point_passes(
        self,
        default_constraints,
        flue_gas_stream,
        cold_stream_liquid,
    ):
        """Flue gas outlet above acid dew point passes."""
        # Create exchanger with outlet above acid dew point
        exchanger = HeatExchanger(
            exchanger_id="E-106",
            hot_stream_id="FG1",
            cold_stream_id="C1",
            duty_kW=1500.0,
            hot_inlet_T_C=300.0,
            hot_outlet_T_C=150.0,  # Above 120C acid dew point
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
        )

        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=1500.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [flue_gas_stream], [cold_stream_liquid]
        )

        acid_violations = [
            v for v in result.violations
            if v.constraint_tag == "ACID_DEW_POINT"
        ]
        assert len(acid_violations) == 0


class TestPressureDropValidation:
    """Tests for pressure drop constraint."""

    def test_excessive_pressure_drop_violation(
        self,
        default_constraints,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Excessive pressure drop raises violation."""
        exchanger = HeatExchanger(
            exchanger_id="E-107",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=150.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=100.0,
            hot_side_dp_kPa=75.0,   # Above 50 kPa limit for liquid
            cold_side_dp_kPa=60.0,  # Above 50 kPa limit for liquid
        )

        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )

        assert result.total_violations >= 2  # Both sides violated
        dp_violations = [
            v for v in result.violations
            if v.constraint_tag == "MAX_PRESSURE_DROP"
        ]
        assert len(dp_violations) >= 2


class TestFailClosedBehavior:
    """Tests for fail-closed safety behavior."""

    def test_fail_closed_raises_exception(
        self,
        default_constraints,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Fail-closed mode raises exception on violation."""
        exchanger = HeatExchanger(
            exchanger_id="E-108",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=55.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=95.0,
        )

        validator = SafetyValidator(
            default_constraints,
            mode=ValidationMode.STRICT,
            fail_closed=True,
        )
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        with pytest.raises(SafetyViolationError):
            validator.validate_design(
                design, [hot_stream_liquid], [cold_stream_liquid]
            )

    def test_fail_open_returns_result(
        self,
        default_constraints,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Fail-open mode returns result without exception."""
        exchanger = HeatExchanger(
            exchanger_id="E-109",
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=55.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=95.0,
        )

        validator = SafetyValidator(
            default_constraints,
            mode=ValidationMode.STRICT,
            fail_closed=False,
        )
        design = HENDesign(
            design_id="test-design",
            exchangers=[exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )

        assert isinstance(result, SafetyValidationResult)
        assert result.validation_passed is False


class TestProvenanceTracking:
    """Tests for SHA-256 provenance tracking."""

    def test_result_has_provenance_hashes(
        self,
        default_constraints,
        safe_exchanger,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Validation result includes provenance hashes."""
        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[safe_exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )

        assert result.constraints_hash is not None
        assert len(result.constraints_hash) == 64  # SHA-256 hex length
        assert result.design_hash is not None
        assert len(result.design_hash) == 64
        assert result.result_hash is not None
        assert len(result.result_hash) == 64

    def test_same_input_same_hash(
        self,
        default_constraints,
        safe_exchanger,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Same input produces same result hash (determinism)."""
        validator = SafetyValidator(default_constraints, fail_closed=False)
        design = HENDesign(
            design_id="test-design",
            exchangers=[safe_exchanger],
            total_heat_recovered_kW=1000.0,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        result1 = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )
        result2 = validator.validate_design(
            design, [hot_stream_liquid], [cold_stream_liquid]
        )

        assert result1.design_hash == result2.design_hash
        assert result1.constraints_hash == result2.constraints_hash


class TestSingleExchangerValidation:
    """Tests for single exchanger validation convenience method."""

    def test_validate_single_exchanger(
        self,
        default_constraints,
        safe_exchanger,
        hot_stream_liquid,
        cold_stream_liquid,
    ):
        """Single exchanger can be validated without full design."""
        validator = SafetyValidator(default_constraints, fail_closed=True)

        # Should not raise even with fail_closed=True
        result = validator.validate_single_exchanger(
            safe_exchanger,
            hot_stream=hot_stream_liquid,
            cold_stream=cold_stream_liquid,
        )

        assert result.is_safe is True
        assert result.validation_passed is True
