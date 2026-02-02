"""
GL-015 INSULSCAN - IR Thermography Survey Tests

Unit tests for IRThermographySurvey including hot spot detection,
anomaly classification, heat loss quantification, and ROI analysis.

Coverage target: 85%+
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_015_insulation_analysis.ir_survey import (
    IRThermographySurvey,
    ThermalImageData,
    AnomalyDetection,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    InsulationAnalysisConfig,
    IRSurveyConfig,
)
from greenlang.agents.process_heat.gl_015_insulation_analysis.schemas import (
    InsulationInput,
    PipeGeometry,
    InsulationLayer,
    JacketingSpec,
    GeometryType,
    JacketingType,
    IRHotSpot,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analysis_config():
    """Create analysis configuration."""
    return InsulationAnalysisConfig(
        facility_id="TEST-FACILITY",
        ir_survey=IRSurveyConfig(
            emissivity_default=0.95,
            hot_spot_threshold_delta_f=15.0,
            damaged_insulation_threshold_pct=25.0,
            missing_insulation_threshold_pct=100.0,
        ),
    )


@pytest.fixture
def ir_survey(analysis_config):
    """Create IR thermography survey instance."""
    return IRThermographySurvey(config=analysis_config)


@pytest.fixture
def normal_thermal_image():
    """Create normal thermal image data (no anomalies)."""
    return ThermalImageData(
        image_id="IMG-001",
        location="PIPE-001",
        timestamp=datetime.now(timezone.utc),
        min_temp_f=90.0,
        max_temp_f=105.0,
        avg_temp_f=98.0,
        spot_temps={
            "SP1": 95.0,
            "SP2": 98.0,
            "SP3": 102.0,
        },
        ambient_temp_f=77.0,
        emissivity=0.95,
        distance_ft=6.0,
        camera_model="FLIR T560",
    )


@pytest.fixture
def hot_spot_thermal_image():
    """Create thermal image with hot spot anomaly."""
    return ThermalImageData(
        image_id="IMG-002",
        location="PIPE-002",
        timestamp=datetime.now(timezone.utc),
        min_temp_f=90.0,
        max_temp_f=180.0,  # Hot spot
        avg_temp_f=120.0,
        spot_temps={
            "SP1": 95.0,
            "SP2": 180.0,  # Hot spot
            "SP3": 100.0,
        },
        ambient_temp_f=77.0,
        emissivity=0.95,
        distance_ft=6.0,
    )


@pytest.fixture
def baseline_input():
    """Create baseline input for comparison."""
    return InsulationInput(
        item_name="Test Pipe",
        operating_temperature_f=350.0,
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
                thickness_in=2.0,
            ),
        ],
        jacketing=JacketingSpec(
            jacketing_type=JacketingType.ALUMINUM,
            emissivity=0.10,
        ),
    )


@pytest.fixture
def multiple_thermal_images():
    """Create multiple thermal images for survey."""
    return [
        ThermalImageData(
            image_id="IMG-001",
            location="PIPE-001",
            timestamp=datetime.now(timezone.utc),
            min_temp_f=90.0,
            max_temp_f=105.0,
            avg_temp_f=98.0,
            spot_temps={"SP1": 95.0, "SP2": 98.0},
            ambient_temp_f=77.0,
            emissivity=0.95,
            distance_ft=6.0,
        ),
        ThermalImageData(
            image_id="IMG-002",
            location="PIPE-002",
            timestamp=datetime.now(timezone.utc),
            min_temp_f=90.0,
            max_temp_f=200.0,  # Hot spot
            avg_temp_f=130.0,
            spot_temps={"SP1": 200.0, "SP2": 95.0},  # Hot spot at SP1
            ambient_temp_f=77.0,
            emissivity=0.95,
            distance_ft=6.0,
        ),
        ThermalImageData(
            image_id="IMG-003",
            location="PIPE-003",
            timestamp=datetime.now(timezone.utc),
            min_temp_f=88.0,
            max_temp_f=102.0,
            avg_temp_f=95.0,
            spot_temps={"SP1": 90.0, "SP2": 95.0, "SP3": 100.0},
            ambient_temp_f=77.0,
            emissivity=0.95,
            distance_ft=6.0,
        ),
    ]


# =============================================================================
# SURVEY INITIALIZATION TESTS
# =============================================================================

class TestIRSurveyInitialization:
    """Tests for survey initialization."""

    def test_survey_initialization(self, ir_survey):
        """Test survey initializes correctly."""
        assert ir_survey.config is not None
        assert ir_survey.ir_config is not None
        assert ir_survey.material_db is not None
        assert ir_survey.heat_loss_calc is not None
        assert ir_survey.survey_count == 0

    def test_ir_config_values(self, ir_survey):
        """Test IR config values are set."""
        assert ir_survey.ir_config.emissivity_default == 0.95
        assert ir_survey.ir_config.hot_spot_threshold_delta_f == 15.0


# =============================================================================
# SINGLE IMAGE ANALYSIS TESTS
# =============================================================================

class TestSingleImageAnalysis:
    """Tests for single image analysis."""

    def test_analyze_normal_image(self, ir_survey, normal_thermal_image, baseline_input):
        """Test analysis of normal image (no hot spots)."""
        hot_spots = ir_survey.analyze_single_image(
            image=normal_thermal_image,
            baseline_input=baseline_input,
        )

        # Normal image should have few or no hot spots
        assert isinstance(hot_spots, list)

    def test_analyze_hot_spot_image(self, ir_survey, hot_spot_thermal_image, baseline_input):
        """Test analysis of image with hot spot."""
        hot_spots = ir_survey.analyze_single_image(
            image=hot_spot_thermal_image,
            baseline_input=baseline_input,
        )

        # Should detect the hot spot
        assert len(hot_spots) > 0

        # Check hot spot properties
        for hs in hot_spots:
            assert isinstance(hs, IRHotSpot)
            assert hs.measured_temperature_f is not None
            assert hs.delta_t_f is not None
            assert hs.severity is not None

    def test_hot_spot_severity(self, ir_survey, hot_spot_thermal_image, baseline_input):
        """Test hot spot severity classification."""
        hot_spots = ir_survey.analyze_single_image(
            image=hot_spot_thermal_image,
            baseline_input=baseline_input,
        )

        for hs in hot_spots:
            assert hs.severity in ["low", "medium", "high", "critical"]


# =============================================================================
# FULL SURVEY ANALYSIS TESTS
# =============================================================================

class TestFullSurveyAnalysis:
    """Tests for full survey analysis."""

    def test_analyze_survey(self, ir_survey, multiple_thermal_images, baseline_input):
        """Test full survey analysis."""
        baselines = {
            "PIPE-001": baseline_input,
            "PIPE-002": baseline_input,
            "PIPE-003": baseline_input,
        }

        result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs=baselines,
        )

        assert result is not None
        assert result.survey_id is not None
        assert result.survey_date is not None
        assert result.items_surveyed == 3

    def test_survey_identifies_anomalies(self, ir_survey, multiple_thermal_images, baseline_input):
        """Test survey identifies anomalies."""
        baselines = {
            "PIPE-001": baseline_input,
            "PIPE-002": baseline_input,
            "PIPE-003": baseline_input,
        }

        result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs=baselines,
        )

        # Should identify hot spots from PIPE-002
        assert result.total_anomalies > 0

    def test_survey_totals(self, ir_survey, multiple_thermal_images, baseline_input):
        """Test survey totals are calculated."""
        baselines = {
            "PIPE-001": baseline_input,
            "PIPE-002": baseline_input,
            "PIPE-003": baseline_input,
        }

        result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs=baselines,
        )

        assert result.total_excess_heat_loss_btu_hr >= 0
        assert result.annual_excess_energy_cost_usd >= 0

    def test_survey_missing_baseline(self, ir_survey, multiple_thermal_images, baseline_input):
        """Test survey handles missing baseline gracefully."""
        # Only provide baseline for some locations
        baselines = {
            "PIPE-001": baseline_input,
            # Missing PIPE-002 and PIPE-003
        }

        result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs=baselines,
        )

        # Should still complete without error
        assert result is not None


# =============================================================================
# ANOMALY CLASSIFICATION TESTS
# =============================================================================

class TestAnomalyClassification:
    """Tests for anomaly classification."""

    def test_classify_hot_spot(self, ir_survey, baseline_input):
        """Test classification of hot spot anomaly."""
        anomaly = ir_survey._classify_anomaly(
            location="PIPE-001/SP1",
            measured_temp_f=150.0,
            expected_temp_f=100.0,
            delta_t_f=50.0,
            baseline_input=baseline_input,
        )

        assert anomaly is not None
        assert isinstance(anomaly, AnomalyDetection)
        assert anomaly.anomaly_type in ["hot_spot", "damaged", "missing"]

    def test_classify_missing_insulation(self, ir_survey, baseline_input):
        """Test classification of missing insulation."""
        anomaly = ir_survey._classify_anomaly(
            location="PIPE-001/SP1",
            measured_temp_f=300.0,  # Near operating temp
            expected_temp_f=100.0,
            delta_t_f=200.0,  # Very high delta
            baseline_input=baseline_input,
        )

        assert anomaly is not None
        # Very high temp suggests missing insulation
        assert anomaly.anomaly_type == "missing"

    def test_classify_damaged_insulation(self, ir_survey, baseline_input):
        """Test classification of damaged insulation."""
        anomaly = ir_survey._classify_anomaly(
            location="PIPE-001/SP1",
            measured_temp_f=140.0,  # Elevated but not bare
            expected_temp_f=100.0,
            delta_t_f=40.0,
            baseline_input=baseline_input,
        )

        assert anomaly is not None
        # Moderate elevation suggests damage
        assert anomaly.anomaly_type in ["hot_spot", "damaged"]

    def test_no_anomaly_below_threshold(self, ir_survey, baseline_input):
        """Test no anomaly when below threshold."""
        anomaly = ir_survey._classify_anomaly(
            location="PIPE-001/SP1",
            measured_temp_f=105.0,
            expected_temp_f=100.0,
            delta_t_f=5.0,  # Below 15F threshold
            baseline_input=baseline_input,
        )

        # Should return None for small delta-T
        assert anomaly is None

    def test_severity_levels(self, ir_survey, baseline_input):
        """Test severity level classification."""
        # Low severity
        low = ir_survey._classify_anomaly(
            location="TEST", measured_temp_f=120.0, expected_temp_f=100.0,
            delta_t_f=20.0, baseline_input=baseline_input,
        )
        assert low is not None
        assert low.severity in ["low", "medium"]

        # High severity
        high = ir_survey._classify_anomaly(
            location="TEST", measured_temp_f=200.0, expected_temp_f=100.0,
            delta_t_f=100.0, baseline_input=baseline_input,
        )
        assert high is not None
        assert high.severity in ["high", "critical"]


# =============================================================================
# EXCESS HEAT LOSS ESTIMATION TESTS
# =============================================================================

class TestExcessHeatLossEstimation:
    """Tests for excess heat loss estimation."""

    def test_estimate_excess_heat_loss(self, ir_survey, baseline_input):
        """Test excess heat loss estimation."""
        excess = ir_survey._estimate_excess_heat_loss(
            baseline_input=baseline_input,
            measured_temp_f=150.0,
            expected_temp_f=100.0,
        )

        assert excess >= 0

    def test_higher_temp_more_excess(self, ir_survey, baseline_input):
        """Test higher measured temp means more excess heat loss."""
        moderate_excess = ir_survey._estimate_excess_heat_loss(
            baseline_input=baseline_input,
            measured_temp_f=130.0,
            expected_temp_f=100.0,
        )

        high_excess = ir_survey._estimate_excess_heat_loss(
            baseline_input=baseline_input,
            measured_temp_f=200.0,
            expected_temp_f=100.0,
        )

        assert high_excess > moderate_excess


# =============================================================================
# COST CALCULATION TESTS
# =============================================================================

class TestCostCalculations:
    """Tests for cost calculations."""

    def test_annual_energy_cost(self, ir_survey):
        """Test annual energy cost calculation."""
        # 1000 BTU/hr excess
        annual_cost = ir_survey._calculate_annual_energy_cost(1000.0)

        # Should be positive
        assert annual_cost > 0

        # Calculated: 1000 * 8760 / 1,000,000 * energy_cost
        expected = 1000 * 8760 / 1_000_000 * ir_survey.economic.energy_cost_per_mmbtu
        assert abs(annual_cost - expected) < 0.01

    def test_repair_cost_estimation(self, ir_survey):
        """Test repair cost estimation."""
        hot_spots = [
            IRHotSpot(
                location_description="Test 1",
                measured_temperature_f=150.0,
                expected_temperature_f=100.0,
                delta_t_f=50.0,
                severity="high",
            ),
            IRHotSpot(
                location_description="Test 2",
                measured_temperature_f=130.0,
                expected_temperature_f=100.0,
                delta_t_f=30.0,
                severity="medium",
            ),
        ]

        cost = ir_survey._estimate_repair_cost(hot_spots)

        assert cost > 0

    def test_repair_cost_by_severity(self, ir_survey):
        """Test repair cost scales with severity."""
        low_severity = [
            IRHotSpot(
                location_description="Test",
                measured_temperature_f=120.0,
                expected_temperature_f=100.0,
                delta_t_f=20.0,
                severity="low",
            ),
        ]

        high_severity = [
            IRHotSpot(
                location_description="Test",
                measured_temperature_f=200.0,
                expected_temperature_f=100.0,
                delta_t_f=100.0,
                severity="critical",
            ),
        ]

        low_cost = ir_survey._estimate_repair_cost(low_severity)
        high_cost = ir_survey._estimate_repair_cost(high_severity)

        assert high_cost > low_cost


# =============================================================================
# ROI CALCULATION TESTS
# =============================================================================

class TestROICalculation:
    """Tests for ROI calculations."""

    def test_calculate_roi(self, ir_survey):
        """Test ROI calculation for repairs."""
        hot_spots = [
            IRHotSpot(
                location_description="Test",
                measured_temperature_f=200.0,
                expected_temperature_f=100.0,
                delta_t_f=100.0,
                severity="high",
                estimated_heat_loss_btu_hr=5000.0,
            ),
        ]

        roi = ir_survey.calculate_roi_for_repairs(hot_spots)

        assert roi is not None
        assert "total_repair_cost_usd" in roi
        assert "annual_energy_savings_usd" in roi
        assert "simple_payback_years" in roi
        assert "npv_10_years_usd" in roi
        assert "roi_10_years_pct" in roi

    def test_roi_positive_for_good_investment(self, ir_survey):
        """Test ROI is positive for worthwhile repair."""
        hot_spots = [
            IRHotSpot(
                location_description="Test",
                measured_temperature_f=300.0,  # Very hot - lots of savings
                expected_temperature_f=100.0,
                delta_t_f=200.0,
                severity="critical",
                estimated_heat_loss_btu_hr=50000.0,  # Large excess
            ),
        ]

        roi = ir_survey.calculate_roi_for_repairs(hot_spots)

        # Should have positive NPV for large savings
        assert roi["npv_10_years_usd"] > 0


# =============================================================================
# SURVEY REPORT TESTS
# =============================================================================

class TestSurveyReport:
    """Tests for survey report generation."""

    def test_generate_report(self, ir_survey, multiple_thermal_images, baseline_input):
        """Test report generation."""
        baselines = {
            "PIPE-001": baseline_input,
            "PIPE-002": baseline_input,
            "PIPE-003": baseline_input,
        }

        survey_result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs=baselines,
        )

        report = ir_survey.generate_survey_report(survey_result)

        assert report is not None
        assert "survey_id" in report
        assert "survey_date" in report
        assert "conditions" in report
        assert "summary" in report
        assert "heat_loss_impact" in report
        assert "repair_economics" in report
        assert "hot_spots" in report
        assert "recommendations" in report

    def test_report_summary(self, ir_survey, multiple_thermal_images, baseline_input):
        """Test report summary section."""
        baselines = {
            "PIPE-001": baseline_input,
            "PIPE-002": baseline_input,
            "PIPE-003": baseline_input,
        }

        survey_result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs=baselines,
        )

        report = ir_survey.generate_survey_report(survey_result)

        summary = report["summary"]
        assert "items_surveyed" in summary
        assert "total_anomalies" in summary
        assert "critical_repairs" in summary

    def test_report_hot_spots_sorted(self, ir_survey, multiple_thermal_images, baseline_input):
        """Test hot spots are sorted by severity."""
        baselines = {
            "PIPE-001": baseline_input,
            "PIPE-002": baseline_input,
            "PIPE-003": baseline_input,
        }

        survey_result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs=baselines,
        )

        report = ir_survey.generate_survey_report(survey_result)

        hot_spots = report["hot_spots"]
        if len(hot_spots) > 1:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            for i in range(len(hot_spots) - 1):
                s1 = severity_order.get(hot_spots[i]["severity"], 4)
                s2 = severity_order.get(hot_spots[i+1]["severity"], 4)
                assert s1 <= s2  # Should be sorted


# =============================================================================
# THERMAL IMAGE DATA TESTS
# =============================================================================

class TestThermalImageData:
    """Tests for ThermalImageData dataclass."""

    def test_thermal_image_creation(self, normal_thermal_image):
        """Test thermal image data creation."""
        img = normal_thermal_image

        assert img.image_id == "IMG-001"
        assert img.location == "PIPE-001"
        assert img.min_temp_f == 90.0
        assert img.max_temp_f == 105.0
        assert img.avg_temp_f == 98.0
        assert len(img.spot_temps) == 3
        assert img.emissivity == 0.95

    def test_thermal_image_with_camera_info(self, normal_thermal_image):
        """Test thermal image with camera info."""
        assert normal_thermal_image.camera_model == "FLIR T560"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_survey(self, ir_survey, baseline_input):
        """Test survey with no images."""
        result = ir_survey.analyze_survey(
            thermal_images=[],
            baseline_inputs={"PIPE-001": baseline_input},
        )

        assert result is not None
        assert result.total_anomalies == 0

    def test_survey_counter(self, ir_survey, normal_thermal_image, baseline_input):
        """Test survey counter increments."""
        initial_count = ir_survey.survey_count

        ir_survey.analyze_survey(
            thermal_images=[normal_thermal_image],
            baseline_inputs={"PIPE-001": baseline_input},
        )

        assert ir_survey.survey_count == initial_count + 1

    def test_all_images_have_no_baseline(self, ir_survey, multiple_thermal_images):
        """Test survey when no baselines match."""
        result = ir_survey.analyze_survey(
            thermal_images=multiple_thermal_images,
            baseline_inputs={},  # No baselines
        )

        # Should complete but find no anomalies (can't compare)
        assert result is not None
        assert result.total_anomalies == 0
