"""
Unit tests for GL-009 THERMALIQ Agent Schemas

Tests all Pydantic schema validation for input/output models.
Validates field constraints, enum handling, and data integrity.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    # Enums
    ThermalFluidType,
    DegradationLevel,
    SafetyStatus,
    HeaterType,
    FlowRegime,
    ValidationStatus,
    OptimizationStatus,
    # Input models
    ThermalFluidInput,
    FluidLabAnalysis,
    ExpansionTankData,
    # Output models
    FluidProperties,
    ExergyAnalysis,
    DegradationAnalysis,
    HeatTransferAnalysis,
    ExpansionTankSizing,
    SafetyAnalysis,
    OptimizationRecommendation,
    ThermalFluidOutput,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestThermalFluidTypeEnum:
    """Tests for ThermalFluidType enum."""

    def test_therminol_66_value(self):
        """Test Therminol 66 enum value."""
        assert ThermalFluidType.THERMINOL_66.value == "therminol_66"

    def test_dowtherm_a_value(self):
        """Test Dowtherm A enum value."""
        assert ThermalFluidType.DOWTHERM_A.value == "dowtherm_a"

    def test_all_fluids_have_values(self):
        """Test all thermal fluids have string values."""
        for fluid in ThermalFluidType:
            assert isinstance(fluid.value, str)
            assert len(fluid.value) > 0

    @pytest.mark.parametrize("fluid_type", list(ThermalFluidType))
    def test_fluid_types_are_lowercase(self, fluid_type):
        """Test all fluid type values are lowercase."""
        assert fluid_type.value == fluid_type.value.lower()


class TestDegradationLevelEnum:
    """Tests for DegradationLevel enum."""

    def test_degradation_levels_ordered(self):
        """Test degradation levels follow severity order."""
        levels = [
            DegradationLevel.EXCELLENT,
            DegradationLevel.GOOD,
            DegradationLevel.FAIR,
            DegradationLevel.POOR,
            DegradationLevel.CRITICAL,
        ]
        assert len(levels) == 5

    def test_excellent_value(self):
        """Test excellent degradation level."""
        assert DegradationLevel.EXCELLENT.value == "excellent"

    def test_critical_value(self):
        """Test critical degradation level."""
        assert DegradationLevel.CRITICAL.value == "critical"


class TestSafetyStatusEnum:
    """Tests for SafetyStatus enum."""

    def test_normal_status(self):
        """Test normal safety status."""
        assert SafetyStatus.NORMAL.value == "normal"

    def test_emergency_shutdown_status(self):
        """Test emergency shutdown status."""
        assert SafetyStatus.EMERGENCY_SHUTDOWN.value == "emergency_shutdown"

    def test_all_safety_statuses(self):
        """Test all safety status values exist."""
        statuses = {
            SafetyStatus.NORMAL,
            SafetyStatus.WARNING,
            SafetyStatus.ALARM,
            SafetyStatus.TRIP,
            SafetyStatus.EMERGENCY_SHUTDOWN,
        }
        assert len(statuses) == 5


class TestFlowRegimeEnum:
    """Tests for FlowRegime enum."""

    def test_laminar_regime(self):
        """Test laminar flow regime."""
        assert FlowRegime.LAMINAR.value == "laminar"

    def test_turbulent_regime(self):
        """Test turbulent flow regime."""
        assert FlowRegime.TURBULENT.value == "turbulent"

    def test_transitional_regime(self):
        """Test transitional flow regime."""
        assert FlowRegime.TRANSITIONAL.value == "transitional"


# =============================================================================
# INPUT MODEL TESTS
# =============================================================================

class TestThermalFluidInput:
    """Tests for ThermalFluidInput schema."""

    @pytest.fixture
    def valid_input_data(self):
        """Create valid input data fixture."""
        return {
            "system_id": "TF-001",
            "fluid_type": ThermalFluidType.THERMINOL_66,
            "bulk_temperature_f": 550.0,
            "flow_rate_gpm": 250.0,
        }

    def test_valid_input_creation(self, valid_input_data):
        """Test creating valid input."""
        input_model = ThermalFluidInput(**valid_input_data)

        assert input_model.system_id == "TF-001"
        assert input_model.fluid_type == ThermalFluidType.THERMINOL_66
        assert input_model.bulk_temperature_f == 550.0
        assert input_model.flow_rate_gpm == 250.0

    def test_timestamp_default(self, valid_input_data):
        """Test timestamp has default value."""
        input_model = ThermalFluidInput(**valid_input_data)

        assert input_model.timestamp is not None
        assert isinstance(input_model.timestamp, datetime)

    def test_system_id_required(self):
        """Test system_id is required."""
        with pytest.raises(ValidationError) as exc_info:
            ThermalFluidInput(
                fluid_type=ThermalFluidType.THERMINOL_66,
                bulk_temperature_f=550.0,
                flow_rate_gpm=250.0,
            )
        assert "system_id" in str(exc_info.value)

    def test_system_id_min_length(self):
        """Test system_id minimum length validation."""
        with pytest.raises(ValidationError):
            ThermalFluidInput(
                system_id="",  # Empty string
                fluid_type=ThermalFluidType.THERMINOL_66,
                bulk_temperature_f=550.0,
                flow_rate_gpm=250.0,
            )

    def test_system_id_max_length(self):
        """Test system_id maximum length validation."""
        with pytest.raises(ValidationError):
            ThermalFluidInput(
                system_id="x" * 51,  # Too long
                fluid_type=ThermalFluidType.THERMINOL_66,
                bulk_temperature_f=550.0,
                flow_rate_gpm=250.0,
            )

    def test_bulk_temperature_range_validation(self, valid_input_data):
        """Test bulk temperature range validation."""
        # Test valid lower bound
        valid_input_data["bulk_temperature_f"] = 0.0
        input_model = ThermalFluidInput(**valid_input_data)
        assert input_model.bulk_temperature_f == 0.0

        # Test valid upper bound
        valid_input_data["bulk_temperature_f"] = 800.0
        input_model = ThermalFluidInput(**valid_input_data)
        assert input_model.bulk_temperature_f == 800.0

    def test_bulk_temperature_below_range(self, valid_input_data):
        """Test bulk temperature below valid range."""
        valid_input_data["bulk_temperature_f"] = -1.0
        with pytest.raises(ValidationError):
            ThermalFluidInput(**valid_input_data)

    def test_bulk_temperature_above_range(self, valid_input_data):
        """Test bulk temperature above valid range."""
        valid_input_data["bulk_temperature_f"] = 801.0
        with pytest.raises(ValidationError):
            ThermalFluidInput(**valid_input_data)

    def test_flow_rate_must_be_positive(self, valid_input_data):
        """Test flow rate must be positive."""
        valid_input_data["flow_rate_gpm"] = 0
        with pytest.raises(ValidationError):
            ThermalFluidInput(**valid_input_data)

        valid_input_data["flow_rate_gpm"] = -100
        with pytest.raises(ValidationError):
            ThermalFluidInput(**valid_input_data)

    def test_optional_fields_default_none(self, valid_input_data):
        """Test optional fields default to None."""
        input_model = ThermalFluidInput(**valid_input_data)

        assert input_model.fluid_charge_gallons is None
        assert input_model.fluid_age_months is None
        assert input_model.inlet_temperature_f is None
        assert input_model.outlet_temperature_f is None
        assert input_model.heater_duty_btu_hr is None

    def test_expansion_tank_level_range(self, valid_input_data):
        """Test expansion tank level percentage range."""
        valid_input_data["expansion_tank_level_pct"] = 50.0
        input_model = ThermalFluidInput(**valid_input_data)
        assert input_model.expansion_tank_level_pct == 50.0

        # Below range
        valid_input_data["expansion_tank_level_pct"] = -1.0
        with pytest.raises(ValidationError):
            ThermalFluidInput(**valid_input_data)

        # Above range
        valid_input_data["expansion_tank_level_pct"] = 101.0
        with pytest.raises(ValidationError):
            ThermalFluidInput(**valid_input_data)

    def test_heater_type_default(self, valid_input_data):
        """Test heater type has default value."""
        input_model = ThermalFluidInput(**valid_input_data)
        assert input_model.heater_type == HeaterType.FIRED_HEATER

    def test_pump_discharge_pressure_default(self, valid_input_data):
        """Test pump discharge pressure has default."""
        input_model = ThermalFluidInput(**valid_input_data)
        assert input_model.pump_discharge_pressure_psig == 50.0

    @pytest.mark.parametrize("temp_f,expected", [
        (550.0, 550.0),
        (600.0, 600.0),
        (100.0, 100.0),
    ])
    def test_various_temperature_values(self, valid_input_data, temp_f, expected):
        """Test various temperature values."""
        valid_input_data["bulk_temperature_f"] = temp_f
        input_model = ThermalFluidInput(**valid_input_data)
        assert input_model.bulk_temperature_f == expected


class TestFluidLabAnalysis:
    """Tests for FluidLabAnalysis schema."""

    def test_default_sample_id_generation(self):
        """Test sample ID is auto-generated."""
        lab = FluidLabAnalysis()
        assert lab.sample_id is not None
        assert len(lab.sample_id) == 8

    def test_sample_date_default(self):
        """Test sample date has default."""
        lab = FluidLabAnalysis()
        assert lab.sample_date is not None

    def test_viscosity_must_be_positive(self):
        """Test viscosity must be positive if provided."""
        with pytest.raises(ValidationError):
            FluidLabAnalysis(viscosity_cst_100f=-5.0)

    def test_flash_point_must_be_positive(self):
        """Test flash point must be positive if provided."""
        with pytest.raises(ValidationError):
            FluidLabAnalysis(flash_point_f=-100)

    def test_acid_number_non_negative(self):
        """Test total acid number must be non-negative."""
        with pytest.raises(ValidationError):
            FluidLabAnalysis(total_acid_number_mg_koh_g=-0.1)

    def test_carbon_residue_percentage_range(self):
        """Test carbon residue percentage range."""
        lab = FluidLabAnalysis(carbon_residue_pct=0.5)
        assert lab.carbon_residue_pct == 0.5

        with pytest.raises(ValidationError):
            FluidLabAnalysis(carbon_residue_pct=-1.0)

        with pytest.raises(ValidationError):
            FluidLabAnalysis(carbon_residue_pct=101.0)

    def test_moisture_ppm_non_negative(self):
        """Test moisture ppm must be non-negative."""
        lab = FluidLabAnalysis(moisture_ppm=500)
        assert lab.moisture_ppm == 500

        with pytest.raises(ValidationError):
            FluidLabAnalysis(moisture_ppm=-100)

    def test_color_astm_range(self):
        """Test ASTM color rating range."""
        lab = FluidLabAnalysis(color_astm=4.0)
        assert lab.color_astm == 4.0

        with pytest.raises(ValidationError):
            FluidLabAnalysis(color_astm=-1.0)

        with pytest.raises(ValidationError):
            FluidLabAnalysis(color_astm=9.0)


class TestExpansionTankData:
    """Tests for ExpansionTankData schema."""

    @pytest.fixture
    def valid_tank_data(self):
        """Create valid tank data fixture."""
        return {
            "tank_id": "ET-001",
            "total_volume_gallons": 1000.0,
            "current_level_pct": 50.0,
            "current_temperature_f": 250.0,
            "system_volume_gallons": 5000.0,
            "max_operating_temp_f": 600.0,
        }

    def test_valid_tank_data(self, valid_tank_data):
        """Test valid expansion tank data."""
        tank = ExpansionTankData(**valid_tank_data)

        assert tank.tank_id == "ET-001"
        assert tank.total_volume_gallons == 1000.0
        assert tank.current_level_pct == 50.0

    def test_volume_must_be_positive(self, valid_tank_data):
        """Test volume must be positive."""
        valid_tank_data["total_volume_gallons"] = 0
        with pytest.raises(ValidationError):
            ExpansionTankData(**valid_tank_data)

    def test_level_percentage_range(self, valid_tank_data):
        """Test level percentage range validation."""
        valid_tank_data["current_level_pct"] = 0
        tank = ExpansionTankData(**valid_tank_data)
        assert tank.current_level_pct == 0

        valid_tank_data["current_level_pct"] = 100
        tank = ExpansionTankData(**valid_tank_data)
        assert tank.current_level_pct == 100

        valid_tank_data["current_level_pct"] = -1
        with pytest.raises(ValidationError):
            ExpansionTankData(**valid_tank_data)


# =============================================================================
# OUTPUT MODEL TESTS
# =============================================================================

class TestFluidProperties:
    """Tests for FluidProperties output schema."""

    @pytest.fixture
    def valid_properties(self):
        """Create valid fluid properties fixture."""
        return {
            "temperature_f": 550.0,
            "density_lb_ft3": 52.0,
            "specific_heat_btu_lb_f": 0.55,
            "thermal_conductivity_btu_hr_ft_f": 0.065,
            "kinematic_viscosity_cst": 1.5,
            "dynamic_viscosity_cp": 1.2,
            "prandtl_number": 15.0,
            "vapor_pressure_psia": 0.5,
            "flash_point_f": 340.0,
            "auto_ignition_temp_f": 750.0,
            "max_film_temp_f": 705.0,
            "max_bulk_temp_f": 650.0,
        }

    def test_valid_properties_creation(self, valid_properties):
        """Test creating valid fluid properties."""
        props = FluidProperties(**valid_properties)

        assert props.temperature_f == 550.0
        assert props.density_lb_ft3 == 52.0
        assert props.prandtl_number == 15.0

    def test_all_required_fields(self, valid_properties):
        """Test all required fields are present."""
        # Remove a required field
        del valid_properties["density_lb_ft3"]

        with pytest.raises(ValidationError):
            FluidProperties(**valid_properties)


class TestExergyAnalysis:
    """Tests for ExergyAnalysis output schema."""

    @pytest.fixture
    def valid_exergy(self):
        """Create valid exergy analysis fixture."""
        return {
            "exergy_efficiency_pct": 45.0,
            "first_law_efficiency_pct": 85.0,
            "exergy_input_btu_hr": 5000000.0,
            "exergy_output_btu_hr": 2250000.0,
            "exergy_destruction_btu_hr": 2750000.0,
            "carnot_efficiency_pct": 55.0,
            "log_mean_temp_ratio": 1.8,
        }

    def test_valid_exergy_analysis(self, valid_exergy):
        """Test creating valid exergy analysis."""
        analysis = ExergyAnalysis(**valid_exergy)

        assert analysis.exergy_efficiency_pct == 45.0
        assert analysis.carnot_efficiency_pct == 55.0

    def test_efficiency_range_validation(self, valid_exergy):
        """Test efficiency percentage range validation."""
        # Valid range
        valid_exergy["exergy_efficiency_pct"] = 0.0
        analysis = ExergyAnalysis(**valid_exergy)
        assert analysis.exergy_efficiency_pct == 0.0

        valid_exergy["exergy_efficiency_pct"] = 100.0
        analysis = ExergyAnalysis(**valid_exergy)
        assert analysis.exergy_efficiency_pct == 100.0

        # Invalid range
        valid_exergy["exergy_efficiency_pct"] = -1.0
        with pytest.raises(ValidationError):
            ExergyAnalysis(**valid_exergy)

        valid_exergy["exergy_efficiency_pct"] = 101.0
        with pytest.raises(ValidationError):
            ExergyAnalysis(**valid_exergy)

    def test_exergy_values_non_negative(self, valid_exergy):
        """Test exergy values must be non-negative."""
        valid_exergy["exergy_input_btu_hr"] = -1000
        with pytest.raises(ValidationError):
            ExergyAnalysis(**valid_exergy)

    def test_log_mean_temp_ratio_positive(self, valid_exergy):
        """Test log mean temp ratio must be positive."""
        valid_exergy["log_mean_temp_ratio"] = 0
        with pytest.raises(ValidationError):
            ExergyAnalysis(**valid_exergy)

    def test_default_values(self, valid_exergy):
        """Test default values are set correctly."""
        analysis = ExergyAnalysis(**valid_exergy)

        assert analysis.reference_temperature_f == 77.0
        assert analysis.calculation_method == "SECOND_LAW_AVAILABILITY"


class TestDegradationAnalysis:
    """Tests for DegradationAnalysis output schema."""

    @pytest.fixture
    def valid_degradation(self):
        """Create valid degradation analysis fixture."""
        return {
            "degradation_level": DegradationLevel.GOOD,
            "remaining_life_pct": 75.0,
            "degradation_score": 25.0,
        }

    def test_valid_degradation_analysis(self, valid_degradation):
        """Test creating valid degradation analysis."""
        analysis = DegradationAnalysis(**valid_degradation)

        assert analysis.degradation_level == DegradationLevel.GOOD
        assert analysis.remaining_life_pct == 75.0

    def test_remaining_life_range(self, valid_degradation):
        """Test remaining life percentage range."""
        valid_degradation["remaining_life_pct"] = 0.0
        analysis = DegradationAnalysis(**valid_degradation)
        assert analysis.remaining_life_pct == 0.0

        valid_degradation["remaining_life_pct"] = 100.0
        analysis = DegradationAnalysis(**valid_degradation)
        assert analysis.remaining_life_pct == 100.0

        valid_degradation["remaining_life_pct"] = -1.0
        with pytest.raises(ValidationError):
            DegradationAnalysis(**valid_degradation)

    def test_degradation_score_range(self, valid_degradation):
        """Test degradation score range."""
        valid_degradation["degradation_score"] = 0.0
        analysis = DegradationAnalysis(**valid_degradation)
        assert analysis.degradation_score == 0.0

        valid_degradation["degradation_score"] = 100.0
        analysis = DegradationAnalysis(**valid_degradation)
        assert analysis.degradation_score == 100.0

    def test_default_status_values(self, valid_degradation):
        """Test default status values."""
        analysis = DegradationAnalysis(**valid_degradation)

        assert analysis.viscosity_status == ValidationStatus.VALID
        assert analysis.replacement_recommended == False


class TestHeatTransferAnalysis:
    """Tests for HeatTransferAnalysis output schema."""

    @pytest.fixture
    def valid_heat_transfer(self):
        """Create valid heat transfer analysis fixture."""
        return {
            "reynolds_number": 50000.0,
            "flow_regime": FlowRegime.TURBULENT,
            "film_coefficient_btu_hr_ft2_f": 150.0,
            "nusselt_number": 200.0,
            "correlation_used": "Gnielinski",
        }

    def test_valid_heat_transfer_analysis(self, valid_heat_transfer):
        """Test creating valid heat transfer analysis."""
        analysis = HeatTransferAnalysis(**valid_heat_transfer)

        assert analysis.reynolds_number == 50000.0
        assert analysis.flow_regime == FlowRegime.TURBULENT

    def test_reynolds_number_non_negative(self, valid_heat_transfer):
        """Test Reynolds number must be non-negative."""
        valid_heat_transfer["reynolds_number"] = -1000
        with pytest.raises(ValidationError):
            HeatTransferAnalysis(**valid_heat_transfer)

    def test_film_coefficient_positive(self, valid_heat_transfer):
        """Test film coefficient must be positive."""
        valid_heat_transfer["film_coefficient_btu_hr_ft2_f"] = 0
        with pytest.raises(ValidationError):
            HeatTransferAnalysis(**valid_heat_transfer)

    def test_nusselt_number_positive(self, valid_heat_transfer):
        """Test Nusselt number must be positive."""
        valid_heat_transfer["nusselt_number"] = 0
        with pytest.raises(ValidationError):
            HeatTransferAnalysis(**valid_heat_transfer)


class TestExpansionTankSizing:
    """Tests for ExpansionTankSizing output schema."""

    @pytest.fixture
    def valid_sizing(self):
        """Create valid expansion tank sizing fixture."""
        return {
            "required_volume_gallons": 800.0,
            "actual_volume_gallons": 1000.0,
            "sizing_adequate": True,
            "thermal_expansion_pct": 18.0,
            "expansion_volume_gallons": 720.0,
            "cold_level_pct": 25.0,
            "hot_level_pct": 75.0,
            "required_npsh_ft": 10.0,
            "available_npsh_ft": 25.0,
            "npsh_margin_ft": 15.0,
        }

    def test_valid_sizing(self, valid_sizing):
        """Test creating valid sizing analysis."""
        sizing = ExpansionTankSizing(**valid_sizing)

        assert sizing.sizing_adequate == True
        assert sizing.npsh_margin_ft == 15.0

    def test_volume_positive(self, valid_sizing):
        """Test volumes must be positive."""
        valid_sizing["required_volume_gallons"] = 0
        with pytest.raises(ValidationError):
            ExpansionTankSizing(**valid_sizing)

    def test_level_percentages_range(self, valid_sizing):
        """Test level percentages are within range."""
        valid_sizing["cold_level_pct"] = -1
        with pytest.raises(ValidationError):
            ExpansionTankSizing(**valid_sizing)

        valid_sizing["cold_level_pct"] = 25.0
        valid_sizing["hot_level_pct"] = 101
        with pytest.raises(ValidationError):
            ExpansionTankSizing(**valid_sizing)


class TestSafetyAnalysis:
    """Tests for SafetyAnalysis output schema."""

    @pytest.fixture
    def valid_safety(self):
        """Create valid safety analysis fixture."""
        return {
            "safety_status": SafetyStatus.NORMAL,
            "film_temp_margin_f": 100.0,
            "bulk_temp_margin_f": 75.0,
            "flash_point_margin_f": 150.0,
            "auto_ignition_margin_f": 200.0,
        }

    def test_valid_safety_analysis(self, valid_safety):
        """Test creating valid safety analysis."""
        analysis = SafetyAnalysis(**valid_safety)

        assert analysis.safety_status == SafetyStatus.NORMAL
        assert analysis.film_temp_margin_f == 100.0

    def test_default_values(self, valid_safety):
        """Test default values are set."""
        analysis = SafetyAnalysis(**valid_safety)

        assert analysis.minimum_flow_met == True
        assert analysis.npsh_adequate == True
        assert analysis.active_alarms == []
        assert analysis.active_trips == []


class TestOptimizationRecommendation:
    """Tests for OptimizationRecommendation schema."""

    @pytest.fixture
    def valid_recommendation(self):
        """Create valid recommendation fixture."""
        return {
            "category": "efficiency",
            "title": "Improve Heat Recovery",
            "description": "Add economizer to improve efficiency",
        }

    def test_valid_recommendation(self, valid_recommendation):
        """Test creating valid recommendation."""
        rec = OptimizationRecommendation(**valid_recommendation)

        assert rec.category == "efficiency"
        assert rec.title == "Improve Heat Recovery"

    def test_priority_range(self, valid_recommendation):
        """Test priority range validation."""
        valid_recommendation["priority"] = 1
        rec = OptimizationRecommendation(**valid_recommendation)
        assert rec.priority == 1

        valid_recommendation["priority"] = 5
        rec = OptimizationRecommendation(**valid_recommendation)
        assert rec.priority == 5

        valid_recommendation["priority"] = 0
        with pytest.raises(ValidationError):
            OptimizationRecommendation(**valid_recommendation)

        valid_recommendation["priority"] = 6
        with pytest.raises(ValidationError):
            OptimizationRecommendation(**valid_recommendation)

    def test_default_priority(self, valid_recommendation):
        """Test default priority value."""
        rec = OptimizationRecommendation(**valid_recommendation)
        assert rec.priority == 2

    def test_recommendation_id_auto_generated(self, valid_recommendation):
        """Test recommendation ID is auto-generated."""
        rec = OptimizationRecommendation(**valid_recommendation)
        assert rec.recommendation_id is not None
        assert len(rec.recommendation_id) == 8


class TestThermalFluidOutput:
    """Tests for ThermalFluidOutput schema."""

    @pytest.fixture
    def valid_output(self):
        """Create valid output fixture."""
        return {
            "system_id": "TF-001",
            "fluid_properties": FluidProperties(
                temperature_f=550.0,
                density_lb_ft3=52.0,
                specific_heat_btu_lb_f=0.55,
                thermal_conductivity_btu_hr_ft_f=0.065,
                kinematic_viscosity_cst=1.5,
                dynamic_viscosity_cp=1.2,
                prandtl_number=15.0,
                vapor_pressure_psia=0.5,
                flash_point_f=340.0,
                auto_ignition_temp_f=750.0,
                max_film_temp_f=705.0,
                max_bulk_temp_f=650.0,
            ),
            "safety_analysis": SafetyAnalysis(
                safety_status=SafetyStatus.NORMAL,
                film_temp_margin_f=100.0,
                bulk_temp_margin_f=75.0,
                flash_point_margin_f=150.0,
                auto_ignition_margin_f=200.0,
            ),
        }

    def test_valid_output(self, valid_output):
        """Test creating valid output."""
        output = ThermalFluidOutput(**valid_output)

        assert output.system_id == "TF-001"
        assert output.status == "success"

    def test_request_id_auto_generated(self, valid_output):
        """Test request ID is auto-generated."""
        output = ThermalFluidOutput(**valid_output)
        assert output.request_id is not None

    def test_timestamp_auto_generated(self, valid_output):
        """Test timestamp is auto-generated."""
        output = ThermalFluidOutput(**valid_output)
        assert output.timestamp is not None

    def test_default_status_values(self, valid_output):
        """Test default status values."""
        output = ThermalFluidOutput(**valid_output)

        assert output.status == "success"
        assert output.overall_status == OptimizationStatus.OPTIMAL
        assert output.processing_time_ms == 0.0

    def test_empty_collections_default(self, valid_output):
        """Test empty collections have defaults."""
        output = ThermalFluidOutput(**valid_output)

        assert output.recommendations == []
        assert output.kpis == {}
        assert output.alerts == []
        assert output.warnings == []


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSchemaIntegration:
    """Integration tests for schema interactions."""

    def test_input_to_output_flow(self):
        """Test input schema can feed into output schema workflow."""
        # Create input
        input_data = ThermalFluidInput(
            system_id="TF-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=550.0,
            flow_rate_gpm=250.0,
        )

        # Verify input is valid
        assert input_data.system_id == "TF-001"

        # Create mock output using input system_id
        output = ThermalFluidOutput(
            system_id=input_data.system_id,
            fluid_properties=FluidProperties(
                temperature_f=input_data.bulk_temperature_f,
                density_lb_ft3=52.0,
                specific_heat_btu_lb_f=0.55,
                thermal_conductivity_btu_hr_ft_f=0.065,
                kinematic_viscosity_cst=1.5,
                dynamic_viscosity_cp=1.2,
                prandtl_number=15.0,
                vapor_pressure_psia=0.5,
                flash_point_f=340.0,
                auto_ignition_temp_f=750.0,
                max_film_temp_f=705.0,
                max_bulk_temp_f=650.0,
            ),
            safety_analysis=SafetyAnalysis(
                safety_status=SafetyStatus.NORMAL,
                film_temp_margin_f=100.0,
                bulk_temp_margin_f=75.0,
                flash_point_margin_f=150.0,
                auto_ignition_margin_f=200.0,
            ),
        )

        # Verify consistency
        assert output.system_id == input_data.system_id
        assert output.fluid_properties.temperature_f == input_data.bulk_temperature_f

    def test_enum_serialization(self):
        """Test enum values serialize correctly."""
        input_data = ThermalFluidInput(
            system_id="TF-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=550.0,
            flow_rate_gpm=250.0,
        )

        # Check enum serialization with use_enum_values=True
        data_dict = input_data.dict()
        assert data_dict["fluid_type"] == "therminol_66"
        assert data_dict["heater_type"] == "fired_heater"
