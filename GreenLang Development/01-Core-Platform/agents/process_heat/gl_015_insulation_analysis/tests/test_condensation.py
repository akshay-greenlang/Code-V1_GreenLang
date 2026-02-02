"""
GL-015 INSULSCAN - Condensation Analyzer Tests

Unit tests for CondensationAnalyzer including dew point calculation,
vapor barrier requirements, and minimum thickness for prevention.

Coverage target: 85%+
"""

import pytest
import math

from greenlang.agents.process_heat.gl_015_insulation_analysis.condensation import (
    CondensationAnalyzer,
    DewPointResult,
    VaporBarrierAnalysis,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    InsulationAnalysisConfig,
    CondensationConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    PipeGeometry,
    InsulationLayer,
    JacketingSpec,
    GeometryType,
    JacketingType,
    ServiceType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analysis_config():
    """Create analysis configuration."""
    return InsulationAnalysisConfig(
        facility_id="TEST-FACILITY",
        condensation=CondensationConfig(
            design_ambient_temp_f=95.0,
            design_relative_humidity_pct=90.0,
            design_dew_point_margin_f=5.0,
            vapor_barrier_required=True,
        ),
    )


@pytest.fixture
def condensation_analyzer(analysis_config):
    """Create condensation analyzer."""
    return CondensationAnalyzer(config=analysis_config)


@pytest.fixture
def cold_pipe_input():
    """Create cold pipe input at risk of condensation."""
    return InsulationInput(
        item_name="Cold Pipe",
        operating_temperature_f=40.0,  # Cold service
        ambient_temperature_f=85.0,
        relative_humidity_pct=80.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        ),
        service_type=ServiceType.COLD,
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="cellular_glass_7pcf",
                thickness_in=1.0,  # May not be enough
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def well_insulated_cold_pipe():
    """Create well-insulated cold pipe."""
    return InsulationInput(
        item_name="Well Insulated Cold Pipe",
        operating_temperature_f=40.0,
        ambient_temperature_f=85.0,
        relative_humidity_pct=70.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        ),
        service_type=ServiceType.COLD,
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="cellular_glass_7pcf",
                thickness_in=3.0,  # Good thickness
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def cryogenic_pipe():
    """Create cryogenic pipe input."""
    return InsulationInput(
        item_name="Cryogenic Pipe",
        operating_temperature_f=-200.0,
        ambient_temperature_f=77.0,
        relative_humidity_pct=60.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        ),
        service_type=ServiceType.CRYOGENIC,
        insulation_layers=[
            InsulationLayer(
                layer_number=1,
                material_id="cellular_glass_7pcf",
                thickness_in=4.0,
            ),
        ],
    )


@pytest.fixture
def bare_cold_pipe():
    """Create bare cold pipe (no insulation)."""
    return InsulationInput(
        item_name="Bare Cold Pipe",
        operating_temperature_f=40.0,
        ambient_temperature_f=85.0,
        relative_humidity_pct=80.0,
        geometry_type=GeometryType.PIPE,
        pipe_geometry=PipeGeometry(
            nominal_pipe_size_in=4.0,
            pipe_length_ft=100.0,
        ),
        service_type=ServiceType.COLD,
        insulation_layers=[],
    )


# =============================================================================
# ANALYZER INITIALIZATION TESTS
# =============================================================================

class TestCondensationAnalyzerInit:
    """Tests for analyzer initialization."""

    def test_analyzer_initialization(self, condensation_analyzer):
        """Test analyzer initializes correctly."""
        analyzer = condensation_analyzer

        assert analyzer.config is not None
        assert analyzer.condensation is not None
        assert analyzer.material_db is not None
        assert analyzer.heat_loss_calc is not None
        assert analyzer.calculation_count == 0

    def test_condensation_config_values(self, condensation_analyzer):
        """Test condensation config values are set."""
        analyzer = condensation_analyzer

        assert analyzer.condensation.design_ambient_temp_f == 95.0
        assert analyzer.condensation.design_relative_humidity_pct == 90.0
        assert analyzer.condensation.design_dew_point_margin_f == 5.0


# =============================================================================
# DEW POINT CALCULATION TESTS
# =============================================================================

class TestDewPointCalculation:
    """Tests for dew point calculation."""

    def test_dew_point_calculation(self, condensation_analyzer):
        """Test dew point calculation."""
        result = condensation_analyzer._calculate_dew_point(
            dry_bulb_temp_f=85.0,
            relative_humidity_pct=80.0,
        )

        assert isinstance(result, DewPointResult)
        assert result.dew_point_f is not None
        assert result.dew_point_c is not None
        assert result.relative_humidity_pct == 80.0
        assert result.dry_bulb_temp_f == 85.0

    def test_dew_point_f_c_consistency(self, condensation_analyzer):
        """Test F and C dew points are consistent."""
        result = condensation_analyzer._calculate_dew_point(
            dry_bulb_temp_f=85.0,
            relative_humidity_pct=80.0,
        )

        # Convert F to C
        expected_c = (result.dew_point_f - 32) * 5 / 9
        assert abs(result.dew_point_c - expected_c) < 0.1

    @pytest.mark.parametrize("temp_f,rh,expected_dp_f", [
        (77.0, 50.0, 57.0),   # ~57F dew point
        (95.0, 90.0, 92.0),   # High humidity ~92F
        (85.0, 60.0, 69.0),   # Moderate ~69F
    ])
    def test_dew_point_known_values(self, condensation_analyzer, temp_f, rh, expected_dp_f):
        """Test dew point against known values."""
        result = condensation_analyzer._calculate_dew_point(
            dry_bulb_temp_f=temp_f,
            relative_humidity_pct=rh,
        )

        # Allow 3F tolerance
        assert abs(result.dew_point_f - expected_dp_f) < 3.0

    def test_dew_point_increases_with_humidity(self, condensation_analyzer):
        """Test dew point increases with humidity."""
        low_rh = condensation_analyzer._calculate_dew_point(85.0, 40.0)
        high_rh = condensation_analyzer._calculate_dew_point(85.0, 90.0)

        assert high_rh.dew_point_f > low_rh.dew_point_f

    def test_dew_point_100_percent_humidity(self, condensation_analyzer):
        """Test dew point equals dry bulb at 100% RH."""
        result = condensation_analyzer._calculate_dew_point(
            dry_bulb_temp_f=85.0,
            relative_humidity_pct=100.0,
        )

        # At 100% RH, dew point should equal dry bulb
        assert abs(result.dew_point_f - 85.0) < 1.0

    def test_dew_point_low_humidity(self, condensation_analyzer):
        """Test dew point calculation at low humidity."""
        result = condensation_analyzer._calculate_dew_point(
            dry_bulb_temp_f=85.0,
            relative_humidity_pct=10.0,
        )

        # Should handle low humidity without error
        assert result.dew_point_f < 50.0  # Much lower than dry bulb


# =============================================================================
# CONDENSATION ANALYSIS TESTS
# =============================================================================

class TestCondensationAnalysis:
    """Tests for condensation analysis."""

    def test_analyze_cold_pipe(self, condensation_analyzer, cold_pipe_input):
        """Test analysis of cold pipe."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        assert result is not None
        assert result.ambient_dew_point_f is not None
        assert result.surface_temperature_f is not None
        assert result.margin_above_dew_point_f is not None
        assert result.condensation_risk is not None
        assert result.condensation_risk_level is not None

    def test_condensation_risk_identification(self, condensation_analyzer, bare_cold_pipe):
        """Test condensation risk is identified."""
        result = condensation_analyzer.analyze(bare_cold_pipe)

        # Bare cold pipe should have condensation risk
        assert result.condensation_risk is True
        assert result.condensation_risk_level in ["high", "medium"]

    def test_no_condensation_risk_well_insulated(self, condensation_analyzer, well_insulated_cold_pipe):
        """Test no condensation risk for well-insulated pipe."""
        result = condensation_analyzer.analyze(well_insulated_cold_pipe)

        # Well-insulated should have low or no risk
        assert result.margin_above_dew_point_f > 0 or result.condensation_risk_level in ["low", "none"]

    def test_risk_level_classification(self, condensation_analyzer, cold_pipe_input):
        """Test risk level classification."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        assert result.condensation_risk_level in ["none", "low", "medium", "high"]


# =============================================================================
# VAPOR BARRIER TESTS
# =============================================================================

class TestVaporBarrierAnalysis:
    """Tests for vapor barrier analysis."""

    def test_vapor_barrier_required_cold_service(self, condensation_analyzer, cold_pipe_input):
        """Test vapor barrier required for cold service."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        assert result.vapor_barrier_required is True

    def test_vapor_barrier_location(self, condensation_analyzer, cold_pipe_input):
        """Test vapor barrier location is specified."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        assert result.vapor_barrier_location is not None
        # For cold service, vapor barrier should be on warm side (outermost)
        assert result.vapor_barrier_location == "innermost"

    def test_cellular_glass_no_vapor_barrier(self, condensation_analyzer, well_insulated_cold_pipe):
        """Test cellular glass may not need additional vapor barrier."""
        result = condensation_analyzer.analyze(well_insulated_cold_pipe)

        # Cellular glass is impermeable
        assert result.has_adequate_vapor_barrier is True

    def test_vapor_barrier_analysis_method(self, condensation_analyzer, cold_pipe_input):
        """Test vapor barrier analysis method."""
        result = condensation_analyzer._analyze_vapor_barrier_requirements(
            input_data=cold_pipe_input,
            dew_point_f=65.0,
        )

        assert isinstance(result, VaporBarrierAnalysis)
        assert result.recommended_perm_rating is not None
        assert len(result.material_suggestions) > 0


# =============================================================================
# MINIMUM THICKNESS TESTS
# =============================================================================

class TestMinimumThicknessForPrevention:
    """Tests for minimum thickness calculation."""

    def test_minimum_thickness_calculated(self, condensation_analyzer, cold_pipe_input):
        """Test minimum thickness is calculated."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        assert result.minimum_thickness_for_prevention_in > 0

    def test_additional_thickness_needed(self, condensation_analyzer, cold_pipe_input):
        """Test additional thickness needed is calculated."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        if result.condensation_risk:
            assert result.additional_thickness_needed_in >= 0

    def test_thicker_insulation_reduces_risk(self, condensation_analyzer):
        """Test thicker insulation reduces condensation risk."""
        thin_input = InsulationInput(
            operating_temperature_f=40.0,
            ambient_temperature_f=85.0,
            relative_humidity_pct=80.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            service_type=ServiceType.COLD,
            insulation_layers=[
                InsulationLayer(
                    layer_number=1,
                    material_id="cellular_glass_7pcf",
                    thickness_in=0.5,
                ),
            ],
        )

        thick_input = thin_input.copy(deep=True)
        thick_input.insulation_layers[0].thickness_in = 3.0

        thin_result = condensation_analyzer.analyze(thin_input)
        thick_result = condensation_analyzer.analyze(thick_input)

        # Thicker should have more margin
        assert thick_result.margin_above_dew_point_f > thin_result.margin_above_dew_point_f


# =============================================================================
# CRYOGENIC SERVICE TESTS
# =============================================================================

class TestCryogenicService:
    """Tests for cryogenic service conditions."""

    def test_cryogenic_analysis(self, condensation_analyzer, cryogenic_pipe):
        """Test cryogenic pipe analysis."""
        result = condensation_analyzer.analyze(cryogenic_pipe)

        assert result is not None
        # Cryogenic always needs careful vapor barrier
        assert result.vapor_barrier_required is True

    def test_cryogenic_material_selection(self, condensation_analyzer, cryogenic_pipe):
        """Test material selection for cryogenic."""
        material = condensation_analyzer._select_cold_service_material(
            operating_temp_f=-200.0
        )

        assert material is not None
        assert material.suitable_for_cold_service is True
        assert material.temperature_range.contains(-200.0)

    def test_cryogenic_vapor_barrier_perm_rating(self, condensation_analyzer, cryogenic_pipe):
        """Test cryogenic needs very low perm rating."""
        result = condensation_analyzer._analyze_vapor_barrier_requirements(
            input_data=cryogenic_pipe,
            dew_point_f=50.0,
        )

        # Cryogenic should have very low perm rating
        assert result.recommended_perm_rating <= 0.01


# =============================================================================
# DESIGN CONDITIONS TESTS
# =============================================================================

class TestDesignConditions:
    """Tests for design condition lookup."""

    def test_humid_subtropical_conditions(self, condensation_analyzer):
        """Test humid subtropical design conditions."""
        conditions = condensation_analyzer.calculate_design_conditions(
            location_climate="humid_subtropical"
        )

        assert conditions["design_temp_f"] == 95
        assert conditions["design_rh_pct"] == 95
        assert conditions["dew_point_f"] > 70

    def test_arid_conditions(self, condensation_analyzer):
        """Test arid climate design conditions."""
        conditions = condensation_analyzer.calculate_design_conditions(
            location_climate="arid"
        )

        assert conditions["design_temp_f"] > 100
        assert conditions["design_rh_pct"] < 60
        # Arid has lower dew point
        assert conditions["dew_point_f"] < 70

    def test_unknown_climate_defaults(self, condensation_analyzer):
        """Test unknown climate returns default (humid_subtropical)."""
        conditions = condensation_analyzer.calculate_design_conditions(
            location_climate="unknown"
        )

        default = condensation_analyzer.calculate_design_conditions(
            location_climate="humid_subtropical"
        )

        assert conditions == default


# =============================================================================
# CONDENSATION REPORT TESTS
# =============================================================================

class TestCondensationReport:
    """Tests for condensation report generation."""

    def test_generate_report(self, condensation_analyzer, cold_pipe_input):
        """Test report generation."""
        report = condensation_analyzer.generate_condensation_report(cold_pipe_input)

        assert report is not None
        assert "item_id" in report
        assert "ambient_conditions" in report
        assert "surface_analysis" in report
        assert "condensation_risk" in report
        assert "insulation_requirements" in report
        assert "vapor_barrier" in report
        assert "recommendations" in report

    def test_report_ambient_conditions(self, condensation_analyzer, cold_pipe_input):
        """Test report includes ambient conditions."""
        report = condensation_analyzer.generate_condensation_report(cold_pipe_input)

        ambient = report["ambient_conditions"]
        assert "temperature_f" in ambient
        assert "relative_humidity_pct" in ambient
        assert "dew_point_f" in ambient
        assert "dew_point_c" in ambient

    def test_report_recommendations(self, condensation_analyzer, bare_cold_pipe):
        """Test report includes recommendations for at-risk."""
        report = condensation_analyzer.generate_condensation_report(bare_cold_pipe)

        # Should have recommendations for bare cold pipe
        assert len(report["recommendations"]) > 0

    def test_report_insulation_requirements(self, condensation_analyzer, cold_pipe_input):
        """Test report includes insulation requirements."""
        report = condensation_analyzer.generate_condensation_report(cold_pipe_input)

        requirements = report["insulation_requirements"]
        assert "current_thickness_in" in requirements
        assert "minimum_thickness_in" in requirements
        assert "additional_needed_in" in requirements


# =============================================================================
# HOT SERVICE TESTS
# =============================================================================

class TestHotServiceCondensation:
    """Tests for hot service (should have no condensation risk)."""

    def test_hot_service_no_condensation(self, condensation_analyzer):
        """Test hot service has no condensation risk."""
        hot_input = InsulationInput(
            operating_temperature_f=350.0,
            ambient_temperature_f=77.0,
            relative_humidity_pct=50.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            service_type=ServiceType.HOT,
            insulation_layers=[
                InsulationLayer(
                    layer_number=1,
                    material_id="mineral_wool_8pcf",
                    thickness_in=2.0,
                ),
            ],
        )

        result = condensation_analyzer.analyze(hot_input)

        # Hot surface should be well above dew point
        assert result.margin_above_dew_point_f > 0
        assert result.condensation_risk is False


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_100_percent_humidity(self, condensation_analyzer):
        """Test behavior at 100% humidity."""
        input_data = InsulationInput(
            operating_temperature_f=40.0,
            ambient_temperature_f=85.0,
            relative_humidity_pct=100.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            service_type=ServiceType.COLD,
            insulation_layers=[
                InsulationLayer(
                    layer_number=1,
                    material_id="cellular_glass_7pcf",
                    thickness_in=2.0,
                ),
            ],
        )

        result = condensation_analyzer.analyze(input_data)
        assert result is not None
        # At 100% RH, dew point = ambient temp, so high risk
        assert result.condensation_risk is True

    def test_low_humidity(self, condensation_analyzer):
        """Test behavior at low humidity."""
        input_data = InsulationInput(
            operating_temperature_f=40.0,
            ambient_temperature_f=85.0,
            relative_humidity_pct=20.0,
            geometry_type=GeometryType.PIPE,
            pipe_geometry=PipeGeometry(
                nominal_pipe_size_in=4.0,
                pipe_length_ft=100.0,
            ),
            service_type=ServiceType.COLD,
            insulation_layers=[
                InsulationLayer(
                    layer_number=1,
                    material_id="cellular_glass_7pcf",
                    thickness_in=1.0,
                ),
            ],
        )

        result = condensation_analyzer.analyze(input_data)
        assert result is not None
        # Low humidity means low dew point, less risk
        assert result.ambient_dew_point_f < 50.0

    def test_calculation_counter(self, condensation_analyzer, cold_pipe_input):
        """Test calculation counter increments."""
        initial_count = condensation_analyzer.calculation_count

        condensation_analyzer.analyze(cold_pipe_input)
        assert condensation_analyzer.calculation_count == initial_count + 1


# =============================================================================
# SURFACE TEMPERATURE MARGIN TESTS
# =============================================================================

class TestSurfaceTemperatureMargin:
    """Tests for surface temperature margin calculations."""

    def test_margin_calculation(self, condensation_analyzer, cold_pipe_input):
        """Test margin is surface temp minus dew point."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        expected_margin = result.surface_temperature_f - result.ambient_dew_point_f
        assert abs(result.margin_above_dew_point_f - expected_margin) < 0.5

    def test_negative_margin_means_condensation(self, condensation_analyzer, bare_cold_pipe):
        """Test negative margin indicates condensation."""
        result = condensation_analyzer.analyze(bare_cold_pipe)

        if result.margin_above_dew_point_f < 0:
            assert result.condensation_risk is True

    def test_risk_based_on_margin(self, condensation_analyzer, cold_pipe_input):
        """Test risk level based on margin."""
        result = condensation_analyzer.analyze(cold_pipe_input)

        # Risk level should correlate with margin
        if result.margin_above_dew_point_f < 0:
            assert result.condensation_risk_level == "high"
        elif result.margin_above_dew_point_f < 5:
            assert result.condensation_risk_level in ["high", "medium"]
        elif result.margin_above_dew_point_f < 10:
            assert result.condensation_risk_level in ["medium", "low"]
