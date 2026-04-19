"""
GL-017 CONDENSYNC Agent - Schema Tests

Unit tests for all input/output schemas and data models.
Tests cover Pydantic validation, serialization, and edge cases.

Coverage targets:
    - All input schema fields
    - All output schema fields
    - Enum conversions
    - Default value handling
    - Field constraints
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    CondenserInput,
    CondenserOutput,
    CondenserStatus,
    CleanlinessResult,
    TubeFoulingResult,
    VacuumSystemResult,
    AirIngresResult,
    CoolingTowerResult,
    PerformanceResult,
    OptimizationRecommendation,
    Alert,
    AlertSeverity,
    CleaningStatus,
    CoolingTowerInput,
    VacuumSystemInput,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_condenser_input():
    """Create valid condenser input."""
    return CondenserInput(
        condenser_id="C-001",
        load_pct=85.0,
        exhaust_steam_flow_lb_hr=450000.0,
        exhaust_steam_pressure_psia=1.2,
        condenser_vacuum_inhga=1.5,
        saturation_temperature_f=101.0,
        hotwell_temperature_f=100.5,
        cw_inlet_temperature_f=75.0,
        cw_outlet_temperature_f=95.0,
        cw_inlet_flow_gpm=90000.0,
    )


@pytest.fixture
def cleanliness_result():
    """Create cleanliness result fixture."""
    return CleanlinessResult(
        cleanliness_factor=0.82,
        design_cleanliness=0.85,
        cleanliness_ratio=0.965,
        u_actual_btu_hr_ft2_f=450.0,
        u_clean_btu_hr_ft2_f=550.0,
        u_design_btu_hr_ft2_f=467.5,
        lmtd_f=18.5,
        heat_duty_btu_hr=450_000_000.0,
        surface_area_ft2=150000.0,
    )


@pytest.fixture
def tube_fouling_result():
    """Create tube fouling result fixture."""
    return TubeFoulingResult(
        fouling_detected=True,
        fouling_severity="light",
        current_backpressure_inhga=1.65,
        expected_backpressure_inhga=1.5,
        backpressure_penalty_inhg=0.15,
        backpressure_deviation_pct=10.0,
    )


@pytest.fixture
def vacuum_system_result():
    """Create vacuum system result fixture."""
    return VacuumSystemResult(
        vacuum_normal=True,
        current_vacuum_inhga=1.52,
        expected_vacuum_inhga=1.5,
        vacuum_deviation_inhg=0.02,
    )


@pytest.fixture
def air_ingress_result():
    """Create air ingress result fixture."""
    return AirIngresResult(
        air_ingress_detected=False,
        estimated_air_ingress_scfm=2.5,
    )


@pytest.fixture
def performance_result():
    """Create performance result fixture."""
    return PerformanceResult(
        actual_duty_btu_hr=450_000_000.0,
        design_duty_btu_hr=500_000_000.0,
        duty_ratio_pct=90.0,
        actual_backpressure_inhga=1.55,
        expected_backpressure_inhga=1.5,
        backpressure_deviation_inhg=0.05,
        backpressure_deviation_pct=3.3,
        ttd_actual_f=5.5,
        ttd_design_f=5.0,
        ttd_deviation_f=0.5,
    )


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_condenser_status_values(self):
        """Test CondenserStatus enum values."""
        expected = {"offline", "standby", "warming", "normal", "degraded", "alarm", "trip"}
        actual = {e.value for e in CondenserStatus}
        assert actual == expected

    def test_cleaning_status_values(self):
        """Test CleaningStatus enum values."""
        expected = {"not_required", "recommended", "required", "urgent", "in_progress"}
        actual = {e.value for e in CleaningStatus}
        assert actual == expected

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        expected = {"info", "warning", "alarm", "critical"}
        actual = {e.value for e in AlertSeverity}
        assert actual == expected


# =============================================================================
# CONDENSER INPUT TESTS
# =============================================================================

class TestCondenserInput:
    """Test CondenserInput schema."""

    def test_valid_input(self, valid_condenser_input):
        """Test valid input creation."""
        assert valid_condenser_input.condenser_id == "C-001"
        assert valid_condenser_input.load_pct == 85.0
        assert valid_condenser_input.exhaust_steam_flow_lb_hr == 450000.0

    def test_required_fields(self):
        """Test required field validation."""
        with pytest.raises(ValidationError):
            CondenserInput(
                condenser_id="C-001",
                # Missing required fields
            )

    def test_timestamp_default(self, valid_condenser_input):
        """Test timestamp has default value."""
        assert valid_condenser_input.timestamp is not None
        assert isinstance(valid_condenser_input.timestamp, datetime)

    def test_operating_status_default(self, valid_condenser_input):
        """Test operating status default."""
        assert valid_condenser_input.operating_status == CondenserStatus.NORMAL

    def test_load_pct_validation(self):
        """Test load percentage validation."""
        # Valid range
        input_data = CondenserInput(
            condenser_id="C-001",
            load_pct=100.0,
            exhaust_steam_flow_lb_hr=500000.0,
            exhaust_steam_pressure_psia=1.2,
            condenser_vacuum_inhga=1.5,
            saturation_temperature_f=101.0,
            hotwell_temperature_f=100.5,
            cw_inlet_temperature_f=75.0,
            cw_outlet_temperature_f=95.0,
            cw_inlet_flow_gpm=90000.0,
        )
        assert input_data.load_pct == 100.0

        # Above max (120%)
        with pytest.raises(ValidationError):
            CondenserInput(
                condenser_id="C-001",
                load_pct=150.0,  # Invalid
                exhaust_steam_flow_lb_hr=500000.0,
                exhaust_steam_pressure_psia=1.2,
                condenser_vacuum_inhga=1.5,
                saturation_temperature_f=101.0,
                hotwell_temperature_f=100.5,
                cw_inlet_temperature_f=75.0,
                cw_outlet_temperature_f=95.0,
                cw_inlet_flow_gpm=90000.0,
            )

    def test_vacuum_validation(self):
        """Test vacuum value validation."""
        # Valid
        input_data = CondenserInput(
            condenser_id="C-001",
            load_pct=85.0,
            exhaust_steam_flow_lb_hr=450000.0,
            exhaust_steam_pressure_psia=1.2,
            condenser_vacuum_inhga=2.0,
            saturation_temperature_f=101.0,
            hotwell_temperature_f=100.5,
            cw_inlet_temperature_f=75.0,
            cw_outlet_temperature_f=95.0,
            cw_inlet_flow_gpm=90000.0,
        )
        assert input_data.condenser_vacuum_inhga == 2.0

        # Invalid - zero or negative
        with pytest.raises(ValidationError):
            CondenserInput(
                condenser_id="C-001",
                load_pct=85.0,
                exhaust_steam_flow_lb_hr=450000.0,
                exhaust_steam_pressure_psia=1.2,
                condenser_vacuum_inhga=0.0,  # Invalid
                saturation_temperature_f=101.0,
                hotwell_temperature_f=100.5,
                cw_inlet_temperature_f=75.0,
                cw_outlet_temperature_f=95.0,
                cw_inlet_flow_gpm=90000.0,
            )

    def test_optional_fields(self, valid_condenser_input):
        """Test optional fields have None defaults."""
        assert valid_condenser_input.wet_bulb_temperature_f is None
        assert valid_condenser_input.cw_conductivity_umhos is None
        assert valid_condenser_input.air_removal_scfm is None
        assert valid_condenser_input.condensate_dissolved_o2_ppb is None

    def test_barometric_pressure_default(self, valid_condenser_input):
        """Test barometric pressure has default."""
        assert valid_condenser_input.barometric_pressure_inhg == 29.92

    @pytest.mark.parametrize("steam_pressure,valid", [
        (1.0, True),
        (0.5, True),
        (15.0, True),
        (0.0, False),
        (-1.0, False),
        (25.0, False),
    ])
    def test_steam_pressure_validation(self, steam_pressure, valid):
        """Test steam pressure validation."""
        base_kwargs = {
            "condenser_id": "C-001",
            "load_pct": 85.0,
            "exhaust_steam_flow_lb_hr": 450000.0,
            "condenser_vacuum_inhga": 1.5,
            "saturation_temperature_f": 101.0,
            "hotwell_temperature_f": 100.5,
            "cw_inlet_temperature_f": 75.0,
            "cw_outlet_temperature_f": 95.0,
            "cw_inlet_flow_gpm": 90000.0,
        }
        if valid:
            input_data = CondenserInput(
                exhaust_steam_pressure_psia=steam_pressure,
                **base_kwargs
            )
            assert input_data.exhaust_steam_pressure_psia == steam_pressure
        else:
            with pytest.raises(ValidationError):
                CondenserInput(
                    exhaust_steam_pressure_psia=steam_pressure,
                    **base_kwargs
                )


# =============================================================================
# COOLING TOWER INPUT TESTS
# =============================================================================

class TestCoolingTowerInput:
    """Test CoolingTowerInput schema."""

    def test_valid_input(self):
        """Test valid cooling tower input."""
        input_data = CoolingTowerInput(
            tower_id="CT-001",
            hot_water_temp_f=105.0,
            cold_water_temp_f=85.0,
            wet_bulb_temp_f=78.0,
            circulation_flow_gpm=100000.0,
        )
        assert input_data.tower_id == "CT-001"
        assert input_data.hot_water_temp_f == 105.0

    def test_fan_speed_default(self):
        """Test fan speed has default."""
        input_data = CoolingTowerInput(
            tower_id="CT-001",
            hot_water_temp_f=105.0,
            cold_water_temp_f=85.0,
            wet_bulb_temp_f=78.0,
            circulation_flow_gpm=100000.0,
        )
        assert input_data.fan_speed_pct == 100.0

    def test_fans_operating_default(self):
        """Test fans operating has default."""
        input_data = CoolingTowerInput(
            tower_id="CT-001",
            hot_water_temp_f=105.0,
            cold_water_temp_f=85.0,
            wet_bulb_temp_f=78.0,
            circulation_flow_gpm=100000.0,
        )
        assert input_data.fans_operating == 1


# =============================================================================
# VACUUM SYSTEM INPUT TESTS
# =============================================================================

class TestVacuumSystemInput:
    """Test VacuumSystemInput schema."""

    def test_valid_input(self):
        """Test valid vacuum system input."""
        input_data = VacuumSystemInput(
            condenser_vacuum_inhga=1.5,
            motive_steam_pressure_psig=150.0,
        )
        assert input_data.condenser_vacuum_inhga == 1.5
        assert input_data.motive_steam_pressure_psig == 150.0

    def test_optional_stage_pressures(self):
        """Test optional stage pressure fields."""
        input_data = VacuumSystemInput(
            condenser_vacuum_inhga=1.5,
            motive_steam_pressure_psig=150.0,
            first_stage_suction_inhga=1.6,
            second_stage_suction_inhga=3.0,
        )
        assert input_data.first_stage_suction_inhga == 1.6
        assert input_data.second_stage_suction_inhga == 3.0


# =============================================================================
# RESULT SCHEMA TESTS
# =============================================================================

class TestCleanlinessResult:
    """Test CleanlinessResult schema."""

    def test_valid_result(self, cleanliness_result):
        """Test valid cleanliness result."""
        assert cleanliness_result.cleanliness_factor == 0.82
        assert cleanliness_result.design_cleanliness == 0.85

    def test_cleaning_status_default(self, cleanliness_result):
        """Test cleaning status default."""
        assert cleanliness_result.cleaning_status == CleaningStatus.NOT_REQUIRED

    def test_calculation_method_default(self, cleanliness_result):
        """Test calculation method default."""
        assert cleanliness_result.calculation_method == "HEI_STANDARD"

    def test_formula_reference_default(self, cleanliness_result):
        """Test formula reference default."""
        assert "HEI Standards" in cleanliness_result.formula_reference

    @pytest.mark.parametrize("cf,valid", [
        (0.85, True),
        (0.0, True),
        (1.0, True),
        (1.2, True),  # Max allowed
        (-0.1, False),
        (1.3, False),
    ])
    def test_cleanliness_factor_range(self, cf, valid):
        """Test cleanliness factor validation."""
        kwargs = {
            "design_cleanliness": 0.85,
            "cleanliness_ratio": 1.0,
            "u_actual_btu_hr_ft2_f": 450.0,
            "u_clean_btu_hr_ft2_f": 550.0,
            "u_design_btu_hr_ft2_f": 467.5,
            "lmtd_f": 18.5,
            "heat_duty_btu_hr": 450_000_000.0,
            "surface_area_ft2": 150000.0,
        }
        if valid:
            result = CleanlinessResult(cleanliness_factor=cf, **kwargs)
            assert result.cleanliness_factor == cf
        else:
            with pytest.raises(ValidationError):
                CleanlinessResult(cleanliness_factor=cf, **kwargs)


class TestTubeFoulingResult:
    """Test TubeFoulingResult schema."""

    def test_valid_result(self, tube_fouling_result):
        """Test valid tube fouling result."""
        assert tube_fouling_result.fouling_detected is True
        assert tube_fouling_result.fouling_severity == "light"

    def test_default_values(self, tube_fouling_result):
        """Test default values."""
        assert tube_fouling_result.fouling_trend == "stable"
        assert tube_fouling_result.heat_rate_penalty_btu_kwh == 0.0
        assert tube_fouling_result.cleaning_recommended is False


class TestVacuumSystemResult:
    """Test VacuumSystemResult schema."""

    def test_valid_result(self, vacuum_system_result):
        """Test valid vacuum system result."""
        assert vacuum_system_result.vacuum_normal is True
        assert vacuum_system_result.current_vacuum_inhga == 1.52

    def test_default_values(self, vacuum_system_result):
        """Test default values."""
        assert vacuum_system_result.air_removal_capacity_pct == 100.0
        assert vacuum_system_result.air_ingress_excessive is False
        assert vacuum_system_result.maintenance_required is False


class TestAirIngresResult:
    """Test AirIngresResult schema."""

    def test_valid_result(self, air_ingress_result):
        """Test valid air ingress result."""
        assert air_ingress_result.air_ingress_detected is False
        assert air_ingress_result.estimated_air_ingress_scfm == 2.5

    def test_default_values(self, air_ingress_result):
        """Test default values."""
        assert air_ingress_result.ingress_severity == "none"
        assert air_ingress_result.subcooling_observed_f == 0.0
        assert air_ingress_result.probable_leak_locations == []
        assert air_ingress_result.confidence_pct == 0.0


class TestCoolingTowerResult:
    """Test CoolingTowerResult schema."""

    def test_valid_result(self):
        """Test valid cooling tower result."""
        result = CoolingTowerResult(
            thermal_efficiency_pct=85.0,
            approach_f=8.0,
            range_f=20.0,
            liquid_to_gas_ratio=1.2,
            cycles_of_concentration=5.0,
            evaporation_rate_gpm=800.0,
            blowdown_rate_gpm=200.0,
            makeup_required_gpm=1010.0,
            optimal_cycles=5.5,
            optimal_blowdown_gpm=180.0,
        )
        assert result.thermal_efficiency_pct == 85.0
        assert result.cycles_of_concentration == 5.0

    def test_default_values(self):
        """Test default values."""
        result = CoolingTowerResult(
            thermal_efficiency_pct=85.0,
            approach_f=8.0,
            range_f=20.0,
            liquid_to_gas_ratio=1.2,
            cycles_of_concentration=5.0,
            evaporation_rate_gpm=800.0,
            blowdown_rate_gpm=200.0,
            makeup_required_gpm=1010.0,
            optimal_cycles=5.5,
            optimal_blowdown_gpm=180.0,
        )
        assert result.chemistry_compliant is True
        assert result.chemistry_deviations == []
        assert result.scaling_potential == "low"
        assert result.corrosion_potential == "low"


class TestPerformanceResult:
    """Test PerformanceResult schema."""

    def test_valid_result(self, performance_result):
        """Test valid performance result."""
        assert performance_result.actual_duty_btu_hr == 450_000_000.0
        assert performance_result.backpressure_deviation_pct == 3.3

    def test_default_values(self, performance_result):
        """Test default values."""
        assert performance_result.heat_rate_impact_btu_kwh == 0.0
        assert performance_result.capacity_impact_mw == 0.0
        assert performance_result.degradation_source == "none"
        assert performance_result.degradation_breakdown == {}


# =============================================================================
# RECOMMENDATION AND ALERT TESTS
# =============================================================================

class TestOptimizationRecommendation:
    """Test OptimizationRecommendation schema."""

    def test_valid_recommendation(self):
        """Test valid recommendation."""
        rec = OptimizationRecommendation(
            category="fouling",
            title="Clean Condenser Tubes",
            description="Tube cleaning recommended based on cleanliness factor",
        )
        assert rec.category == "fouling"
        assert rec.title == "Clean Condenser Tubes"

    def test_auto_generated_id(self):
        """Test recommendation_id is auto-generated."""
        rec = OptimizationRecommendation(
            category="fouling",
            title="Clean Condenser Tubes",
            description="Test description",
        )
        assert rec.recommendation_id is not None
        assert len(rec.recommendation_id) == 8

    def test_default_priority(self):
        """Test default priority."""
        rec = OptimizationRecommendation(
            category="fouling",
            title="Test",
            description="Test",
        )
        assert rec.priority == AlertSeverity.INFO

    def test_default_difficulty(self):
        """Test default implementation difficulty."""
        rec = OptimizationRecommendation(
            category="fouling",
            title="Test",
            description="Test",
        )
        assert rec.implementation_difficulty == "low"
        assert rec.requires_outage is False


class TestAlert:
    """Test Alert schema."""

    def test_valid_alert(self):
        """Test valid alert creation."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category="vacuum",
            title="Low Vacuum Warning",
            description="Vacuum approaching trip point",
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.category == "vacuum"

    def test_auto_generated_fields(self):
        """Test auto-generated fields."""
        alert = Alert(
            severity=AlertSeverity.ALARM,
            category="fouling",
            title="Test Alert",
            description="Test",
        )
        assert alert.alert_id is not None
        assert alert.timestamp is not None

    def test_optional_fields(self):
        """Test optional fields have None defaults."""
        alert = Alert(
            severity=AlertSeverity.INFO,
            category="test",
            title="Test",
            description="Test",
        )
        assert alert.value is None
        assert alert.threshold is None
        assert alert.unit is None
        assert alert.recommended_action is None


# =============================================================================
# CONDENSER OUTPUT TESTS
# =============================================================================

class TestCondenserOutput:
    """Test CondenserOutput schema."""

    def test_valid_output(
        self,
        cleanliness_result,
        tube_fouling_result,
        vacuum_system_result,
        air_ingress_result,
        performance_result,
    ):
        """Test valid output creation."""
        output = CondenserOutput(
            condenser_id="C-001",
            cleanliness=cleanliness_result,
            tube_fouling=tube_fouling_result,
            vacuum_system=vacuum_system_result,
            air_ingress=air_ingress_result,
            performance=performance_result,
        )
        assert output.condenser_id == "C-001"
        assert output.status == "success"

    def test_auto_generated_fields(
        self,
        cleanliness_result,
        tube_fouling_result,
        vacuum_system_result,
        air_ingress_result,
        performance_result,
    ):
        """Test auto-generated fields."""
        output = CondenserOutput(
            condenser_id="C-001",
            cleanliness=cleanliness_result,
            tube_fouling=tube_fouling_result,
            vacuum_system=vacuum_system_result,
            air_ingress=air_ingress_result,
            performance=performance_result,
        )
        assert output.request_id is not None
        assert output.timestamp is not None

    def test_optional_cooling_tower(
        self,
        cleanliness_result,
        tube_fouling_result,
        vacuum_system_result,
        air_ingress_result,
        performance_result,
    ):
        """Test cooling tower result is optional."""
        output = CondenserOutput(
            condenser_id="C-001",
            cleanliness=cleanliness_result,
            tube_fouling=tube_fouling_result,
            vacuum_system=vacuum_system_result,
            air_ingress=air_ingress_result,
            performance=performance_result,
        )
        assert output.cooling_tower is None

    def test_default_collections(
        self,
        cleanliness_result,
        tube_fouling_result,
        vacuum_system_result,
        air_ingress_result,
        performance_result,
    ):
        """Test default empty collections."""
        output = CondenserOutput(
            condenser_id="C-001",
            cleanliness=cleanliness_result,
            tube_fouling=tube_fouling_result,
            vacuum_system=vacuum_system_result,
            air_ingress=air_ingress_result,
            performance=performance_result,
        )
        assert output.recommendations == []
        assert output.alerts == []
        assert output.kpis == {}
        assert output.metadata == {}

    def test_serialization(
        self,
        cleanliness_result,
        tube_fouling_result,
        vacuum_system_result,
        air_ingress_result,
        performance_result,
    ):
        """Test output can be serialized."""
        output = CondenserOutput(
            condenser_id="C-001",
            cleanliness=cleanliness_result,
            tube_fouling=tube_fouling_result,
            vacuum_system=vacuum_system_result,
            air_ingress=air_ingress_result,
            performance=performance_result,
        )
        output_dict = output.dict()
        assert "condenser_id" in output_dict
        assert "cleanliness" in output_dict
        assert "performance" in output_dict

        json_str = output.json()
        assert "C-001" in json_str
