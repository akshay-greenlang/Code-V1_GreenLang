"""
GL-015 INSULSCAN - Surface Temperature Calculator Tests

Unit tests for SurfaceTemperatureCalculator including OSHA compliance
checking, burn risk assessment, and minimum thickness calculations.

Coverage target: 85%+
"""

import pytest

from greenlang.agents.process_heat.gl_015_insulation_analysis.surface_temperature import (
    SurfaceTemperatureCalculator,
    BurnRiskAssessment,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    InsulationAnalysisConfig,
    SafetyConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.materials import (
    InsulationMaterialDatabase,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    PipeGeometry,
    InsulationLayer,
    JacketingSpec,
    GeometryType,
    JacketingType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analysis_config():
    """Create analysis configuration."""
    return InsulationAnalysisConfig(
        facility_id="TEST-FACILITY",
        safety=SafetyConfig(
            max_touch_temperature_c=60.0,
            max_touch_temperature_f=140.0,
            warning_threshold_c=50.0,
            alarm_threshold_c=55.0,
        ),
    )


@pytest.fixture
def surface_temp_calc(analysis_config):
    """Create surface temperature calculator."""
    return SurfaceTemperatureCalculator(config=analysis_config)


@pytest.fixture
def high_temp_pipe():
    """Create high temperature pipe that may exceed OSHA limits."""
    return InsulationInput(
        item_name="High Temp Pipe",
        operating_temperature_f=600.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        ),
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=1.0,  # May not be enough
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def well_insulated_pipe():
    """Create well-insulated pipe that should be compliant."""
    return InsulationInput(
        item_name="Well Insulated Pipe",
        operating_temperature_f=400.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        ),
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="mineral_wool_8pcf",
                thickness_in=3.0,
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def bare_hot_pipe():
    """Create bare hot pipe (definitely non-compliant)."""
    return InsulationInput(
        item_name="Bare Hot Pipe",
        operating_temperature_f=300.0,
        ambient_temperature_f=77.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        ),
        insulation_layers=[],  # No insulation
    )


# =============================================================================
# CALCULATOR INITIALIZATION TESTS
# =============================================================================

class TestSurfaceTemperatureCalculatorInit:
    """Tests for calculator initialization."""

    def test_calculator_initialization(self, surface_temp_calc):
        """Test calculator initializes correctly."""
        calc = surface_temp_calc

        assert calc.config is not None
        assert calc.safety is not None
        assert calc.material_db is not None
        assert calc.heat_loss_calc is not None
        assert calc.calculation_count == 0

    def test_osha_limits(self, surface_temp_calc):
        """Test OSHA limit constants."""
        assert surface_temp_calc.OSHA_LIMIT_C == 60.0
        assert surface_temp_calc.OSHA_LIMIT_F == 140.0


# =============================================================================
# COMPLIANCE CHECK TESTS
# =============================================================================

class TestComplianceCheck:
    """Tests for OSHA compliance checking."""

    def test_compliant_result(self, surface_temp_calc, well_insulated_pipe):
        """Test compliant pipe returns is_compliant=True."""
        result = surface_temp_calc.check_compliance(well_insulated_pipe)

        assert result is not None
        assert result.is_compliant is True
        assert result.margin_f > 0
        assert result.calculated_surface_temp_f < result.osha_limit_temp_f

    def test_non_compliant_bare_pipe(self, surface_temp_calc, bare_hot_pipe):
        """Test bare hot pipe is non-compliant."""
        result = surface_temp_calc.check_compliance(bare_hot_pipe)

        assert result.is_compliant is False
        assert result.margin_f < 0
        assert result.calculated_surface_temp_f > result.osha_limit_temp_f

    def test_surface_temperature_calculated(self, surface_temp_calc, well_insulated_pipe):
        """Test surface temperature is calculated."""
        result = surface_temp_calc.check_compliance(well_insulated_pipe)

        assert result.calculated_surface_temp_f is not None
        assert result.calculated_surface_temp_c is not None
        # F and C should be consistent
        expected_c = (result.calculated_surface_temp_f - 32) * 5 / 9
        assert abs(result.calculated_surface_temp_c - expected_c) < 0.1

    def test_temperature_units_consistency(self, surface_temp_calc, well_insulated_pipe):
        """Test F and C values are consistent."""
        result = surface_temp_calc.check_compliance(well_insulated_pipe)

        # Convert F to C and compare
        calc_c = (result.calculated_surface_temp_f - 32) * 5 / 9
        assert abs(result.calculated_surface_temp_c - calc_c) < 0.5

        limit_c = (result.osha_limit_temp_f - 32) * 5 / 9
        assert abs(result.osha_limit_temp_c - limit_c) < 0.5


# =============================================================================
# MARGIN CALCULATION TESTS
# =============================================================================

class TestMarginCalculation:
    """Tests for margin calculation."""

    def test_positive_margin_compliant(self, surface_temp_calc, well_insulated_pipe):
        """Test positive margin for compliant surface."""
        result = surface_temp_calc.check_compliance(well_insulated_pipe)

        assert result.margin_f > 0
        assert result.margin_c > 0

    def test_negative_margin_non_compliant(self, surface_temp_calc, bare_hot_pipe):
        """Test negative margin for non-compliant surface."""
        result = surface_temp_calc.check_compliance(bare_hot_pipe)

        assert result.margin_f < 0
        assert result.margin_c < 0

    def test_margin_calculation_formula(self, surface_temp_calc, well_insulated_pipe):
        """Test margin is calculated correctly."""
        result = surface_temp_calc.check_compliance(well_insulated_pipe)

        expected_margin_f = result.osha_limit_temp_f - result.calculated_surface_temp_f
        assert abs(result.margin_f - expected_margin_f) < 0.1


# =============================================================================
# MINIMUM THICKNESS CALCULATION TESTS
# =============================================================================

class TestMinimumThicknessCalculation:
    """Tests for minimum thickness for compliance."""

    def test_minimum_thickness_calculated(self, surface_temp_calc, high_temp_pipe):
        """Test minimum thickness is calculated for non-compliant."""
        result = surface_temp_calc.check_compliance(high_temp_pipe)

        if not result.is_compliant:
            assert result.minimum_thickness_for_compliance_in is not None
            assert result.minimum_thickness_for_compliance_in > 0

    def test_additional_thickness_needed(self, surface_temp_calc, high_temp_pipe):
        """Test additional thickness needed is calculated."""
        result = surface_temp_calc.check_compliance(high_temp_pipe)

        if not result.is_compliant:
            assert result.additional_thickness_needed_in >= 0

    def test_current_thickness_tracked(self, surface_temp_calc, high_temp_pipe):
        """Test current thickness is tracked."""
        result = surface_temp_calc.check_compliance(high_temp_pipe)

        expected_thickness = sum(
            layer.thickness_in for layer in high_temp_pipe.insulation_layers
        )
        assert result.current_thickness_in == expected_thickness

    def test_calculate_thickness_for_target(self, surface_temp_calc, bare_hot_pipe):
        """Test calculating thickness for target temperature."""
        thickness = surface_temp_calc.calculate_thickness_for_target_temperature(
            input_data=bare_hot_pipe,
            target_surface_temp_f=120.0,
        )

        assert thickness > 0

    def test_higher_temp_needs_more_insulation(self, surface_temp_calc):
        """Test higher operating temp needs more insulation."""
        low_temp_pipe = InsulationInput(
            operating_temperature_f=300.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
        )

        high_temp_pipe = InsulationInput(
            operating_temperature_f=600.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
        )

        low_thickness = surface_temp_calc.calculate_thickness_for_target_temperature(
            low_temp_pipe, 130.0
        )
        high_thickness = surface_temp_calc.calculate_thickness_for_target_temperature(
            high_temp_pipe, 130.0
        )

        assert high_thickness > low_thickness


# =============================================================================
# BURN RISK ASSESSMENT TESTS
# =============================================================================

class TestBurnRiskAssessment:
    """Tests for burn risk assessment."""

    def test_no_burn_risk_cold_surface(self, surface_temp_calc):
        """Test no burn risk for cold surface."""
        risk = surface_temp_calc._assess_burn_risk(
            surface_temp_c=30.0,
            jacketing_material="aluminum",
        )

        assert risk.risk_level == "none"
        assert risk.time_to_injury_sec is None

    def test_low_burn_risk(self, surface_temp_calc):
        """Test low burn risk for warm surface."""
        risk = surface_temp_calc._assess_burn_risk(
            surface_temp_c=49.0,
            jacketing_material="aluminum",
        )

        assert risk.risk_level in ["none", "low"]

    def test_high_burn_risk(self, surface_temp_calc):
        """Test high burn risk for hot surface."""
        risk = surface_temp_calc._assess_burn_risk(
            surface_temp_c=65.0,
            jacketing_material="aluminum",
        )

        assert risk.risk_level in ["high", "extreme"]
        assert risk.time_to_injury_sec is not None
        assert risk.time_to_injury_sec < 10  # Quick injury

    def test_extreme_burn_risk(self, surface_temp_calc):
        """Test extreme burn risk for very hot surface."""
        risk = surface_temp_calc._assess_burn_risk(
            surface_temp_c=80.0,
            jacketing_material="aluminum",
        )

        assert risk.risk_level == "extreme"
        assert risk.time_to_injury_sec < 1

    def test_metal_vs_non_metal_surface(self, surface_temp_calc):
        """Test metal surfaces have shorter time to injury."""
        metal_risk = surface_temp_calc._assess_burn_risk(
            surface_temp_c=55.0,
            jacketing_material="aluminum",
        )

        non_metal_risk = surface_temp_calc._assess_burn_risk(
            surface_temp_c=55.0,
            jacketing_material="pvc",
        )

        # Metal conducts heat faster, shorter time to injury
        if metal_risk.time_to_injury_sec and non_metal_risk.time_to_injury_sec:
            assert metal_risk.time_to_injury_sec < non_metal_risk.time_to_injury_sec

    def test_protection_recommendations(self, surface_temp_calc):
        """Test protection recommendations are provided."""
        risk = surface_temp_calc._assess_burn_risk(
            surface_temp_c=65.0,
            jacketing_material="aluminum",
        )

        assert len(risk.recommended_protection) > 0

    def test_risk_result_in_compliance_check(self, surface_temp_calc, high_temp_pipe):
        """Test burn risk is included in compliance result."""
        result = surface_temp_calc.check_compliance(high_temp_pipe)

        assert result.contact_burn_risk is not None
        assert result.contact_burn_risk in ["none", "low", "medium", "high", "extreme"]


# =============================================================================
# PERSONNEL PROTECTION TESTS
# =============================================================================

class TestPersonnelProtection:
    """Tests for personnel protection requirements."""

    def test_protection_required_hot_surface(self, surface_temp_calc, bare_hot_pipe):
        """Test protection required for hot surface."""
        result = surface_temp_calc.check_compliance(bare_hot_pipe)

        assert result.personnel_protection_required is True

    def test_protection_recommendations(self, surface_temp_calc, bare_hot_pipe):
        """Test protection recommendations are provided."""
        result = surface_temp_calc.check_compliance(bare_hot_pipe)

        assert len(result.recommended_protection) > 0

    def test_no_protection_compliant_surface(self, surface_temp_calc, well_insulated_pipe):
        """Test protection may not be required for compliant surface."""
        result = surface_temp_calc.check_compliance(well_insulated_pipe)

        # If compliant and cool, protection may not be required
        if result.calculated_surface_temp_f < 100:
            assert result.personnel_protection_required is False


# =============================================================================
# COMPLIANCE REPORT TESTS
# =============================================================================

class TestComplianceReport:
    """Tests for compliance report generation."""

    def test_generate_compliance_report(self, surface_temp_calc, high_temp_pipe):
        """Test compliance report generation."""
        report = surface_temp_calc.generate_compliance_report(high_temp_pipe)

        assert report is not None
        assert "item_id" in report
        assert "surface_temperature" in report
        assert "compliance" in report
        assert "burn_risk" in report
        assert "personnel_protection" in report

    def test_report_includes_corrective_action(self, surface_temp_calc, bare_hot_pipe):
        """Test report includes corrective action for non-compliant."""
        report = surface_temp_calc.generate_compliance_report(bare_hot_pipe)

        assert report["compliance"]["is_compliant"] is False
        assert report["corrective_action"] is not None
        assert "minimum_thickness_in" in report["corrective_action"]

    def test_report_compliant_no_corrective_action(self, surface_temp_calc, well_insulated_pipe):
        """Test compliant report has no corrective action."""
        report = surface_temp_calc.generate_compliance_report(well_insulated_pipe)

        assert report["compliance"]["is_compliant"] is True
        assert report["corrective_action"] is None


# =============================================================================
# SURFACE TEMPERATURE CALCULATION TESTS
# =============================================================================

class TestSurfaceTemperatureCalculation:
    """Tests for surface temperature calculation."""

    def test_calculate_surface_temperature(self, surface_temp_calc, well_insulated_pipe):
        """Test direct surface temperature calculation."""
        surface_temp = surface_temp_calc.calculate_surface_temperature(well_insulated_pipe)

        assert surface_temp is not None
        assert surface_temp > well_insulated_pipe.ambient_temperature_f
        assert surface_temp < well_insulated_pipe.operating_temperature_f

    def test_bare_surface_temp_equals_operating(self, surface_temp_calc, bare_hot_pipe):
        """Test bare surface approaches operating temperature."""
        surface_temp = surface_temp_calc.calculate_surface_temperature(bare_hot_pipe)

        # Surface temp should be close to operating temp for bare surface
        # (with some heat loss to ambient)
        assert surface_temp > bare_hot_pipe.ambient_temperature_f
        assert surface_temp <= bare_hot_pipe.operating_temperature_f


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_high_temperature(self, surface_temp_calc):
        """Test very high operating temperature."""
        input_data = InsulationInput(
            operating_temperature_f=1200.0,
            ambient_temperature_f=77.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            insulation_layers=[
                InsulationLayer(
                    layer_number=1,
                    material_id="calcium_silicate_8pcf",
                    thickness_in=4.0,
                ),
            ],
        )

        result = surface_temp_calc.check_compliance(input_data)
        assert result is not None

    def test_low_ambient_temperature(self, surface_temp_calc):
        """Test low ambient temperature."""
        input_data = InsulationInput(
            operating_temperature_f=300.0,
            ambient_temperature_f=0.0,  # Freezing
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            insulation_layers=[
                InsulationLayer(
                    layer_number=1,
                    material_id="mineral_wool_8pcf",
                    thickness_in=2.0,
                ),
            ],
        )

        result = surface_temp_calc.check_compliance(input_data)
        assert result is not None

    def test_calculation_counter(self, surface_temp_calc, well_insulated_pipe):
        """Test calculation counter increments."""
        initial_count = surface_temp_calc.calculation_count

        surface_temp_calc.check_compliance(well_insulated_pipe)
        assert surface_temp_calc.calculation_count == initial_count + 1


# =============================================================================
# BURN THRESHOLD TABLE TESTS
# =============================================================================

class TestBurnThresholdTables:
    """Tests for burn threshold lookup tables."""

    def test_metal_thresholds_exist(self, surface_temp_calc):
        """Test metal burn thresholds are defined."""
        assert hasattr(surface_temp_calc, "BURN_THRESHOLDS_METAL")
        assert len(surface_temp_calc.BURN_THRESHOLDS_METAL) > 0

    def test_non_metal_thresholds_exist(self, surface_temp_calc):
        """Test non-metal burn thresholds are defined."""
        assert hasattr(surface_temp_calc, "BURN_THRESHOLDS_NON_METAL")
        assert len(surface_temp_calc.BURN_THRESHOLDS_NON_METAL) > 0

    def test_thresholds_are_ordered(self, surface_temp_calc):
        """Test thresholds are ordered correctly."""
        metal = surface_temp_calc.BURN_THRESHOLDS_METAL
        temps = sorted(metal.keys())
        times = [metal[t] for t in temps]

        # Higher temperature should mean shorter time to injury
        for i in range(len(times) - 1):
            assert times[i] > times[i + 1]

    def test_known_threshold_values(self, surface_temp_calc):
        """Test known threshold values are correct."""
        metal = surface_temp_calc.BURN_THRESHOLDS_METAL

        # 60C should be 1 second for metal
        assert metal[60.0] == 1.0

        # 48C should be 60 seconds for metal
        assert metal[48.0] == 60.0
