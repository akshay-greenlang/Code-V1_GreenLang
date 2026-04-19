"""
GL-017 CONDENSYNC Agent - Air Ingress Detector Tests

Unit tests for AirIngressDetector including multi-indicator analysis,
leak source identification, and trend detection.

Coverage targets:
    - Dissolved oxygen scoring
    - Subcooling scoring
    - Vacuum deviation scoring
    - Combined severity determination
    - Leak location identification
    - Leak survey analysis
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_017_condenser_optimization.air_ingress import (
    AirIngressDetector,
    AirIngressConstants,
    AirIngressReading,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    AirIngresConfig,
    VacuumSystemConfig,
    PerformanceConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    AirIngresResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def air_ingress_config():
    """Create default air ingress configuration."""
    return AirIngresConfig()


@pytest.fixture
def vacuum_config():
    """Create default vacuum system configuration."""
    return VacuumSystemConfig()


@pytest.fixture
def performance_config():
    """Create default performance configuration."""
    return PerformanceConfig()


@pytest.fixture
def detector(air_ingress_config, vacuum_config, performance_config):
    """Create AirIngressDetector instance."""
    return AirIngressDetector(
        air_ingress_config, vacuum_config, performance_config
    )


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestAirIngressConstants:
    """Test AirIngressConstants values."""

    def test_target_do(self):
        """Test target dissolved oxygen value."""
        assert AirIngressConstants.TARGET_DO_PPB == 7.0

    def test_subcooling_thresholds(self):
        """Test subcooling threshold values."""
        assert AirIngressConstants.NORMAL_SUBCOOLING_F == 1.0
        assert AirIngressConstants.WARNING_SUBCOOLING_F == 3.0
        assert AirIngressConstants.ALARM_SUBCOOLING_F == 5.0

    def test_heat_rate_penalty(self):
        """Test heat rate penalty value."""
        assert AirIngressConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_F == 5.0

    def test_common_leak_sources(self):
        """Test common leak sources exist."""
        assert "turbine_shaft_seals" in AirIngressConstants.COMMON_LEAK_SOURCES
        assert "expansion_joints" in AirIngressConstants.COMMON_LEAK_SOURCES

    def test_leak_source_ranges(self):
        """Test leak source ranges are tuples."""
        for source, range_tuple in AirIngressConstants.COMMON_LEAK_SOURCES.items():
            assert isinstance(range_tuple, tuple)
            assert len(range_tuple) == 2
            assert range_tuple[0] < range_tuple[1]


# =============================================================================
# DETECTOR INITIALIZATION TESTS
# =============================================================================

class TestDetectorInitialization:
    """Test detector initialization."""

    def test_basic_initialization(
        self, air_ingress_config, vacuum_config, performance_config
    ):
        """Test detector initializes correctly."""
        detector = AirIngressDetector(
            air_ingress_config, vacuum_config, performance_config
        )
        assert detector is not None

    def test_history_empty(self, detector):
        """Test history is empty on initialization."""
        assert len(detector._history) == 0


# =============================================================================
# DISSOLVED OXYGEN SCORING TESTS
# =============================================================================

class TestDissolvedOxygenScoring:
    """Test dissolved oxygen indicator scoring."""

    def test_score_no_do(self, detector):
        """Test score is zero when DO not provided."""
        score = detector._score_dissolved_oxygen(None)
        assert score == 0.0

    def test_score_excellent_do(self, detector):
        """Test score for excellent DO."""
        score = detector._score_dissolved_oxygen(5.0)
        assert score == 0.0

    def test_score_warning_do(self, detector):
        """Test score for warning level DO."""
        score = detector._score_dissolved_oxygen(15.0)
        assert 0 < score < 0.5

    def test_score_alarm_do(self, detector):
        """Test score for alarm level DO."""
        score = detector._score_dissolved_oxygen(30.0)
        assert 0.5 <= score < 1.0

    def test_score_severe_do(self, detector):
        """Test score for severe DO."""
        score = detector._score_dissolved_oxygen(100.0)
        assert score >= 0.8

    @pytest.mark.parametrize("do,expected_range", [
        (5.0, (0.0, 0.1)),
        (10.0, (0.0, 0.3)),
        (20.0, (0.3, 0.6)),
        (40.0, (0.6, 0.9)),
        (80.0, (0.8, 1.0)),
    ])
    def test_score_ranges(self, detector, do, expected_range):
        """Test DO scores fall in expected ranges."""
        score = detector._score_dissolved_oxygen(do)
        assert expected_range[0] <= score <= expected_range[1]


# =============================================================================
# SUBCOOLING SCORING TESTS
# =============================================================================

class TestSubcoolingScoring:
    """Test subcooling indicator scoring."""

    def test_score_no_subcooling(self, detector):
        """Test score is zero when subcooling not provided."""
        score = detector._score_subcooling(None)
        assert score == 0.0

    def test_score_normal_subcooling(self, detector):
        """Test score for normal subcooling."""
        score = detector._score_subcooling(0.5)
        assert score == 0.0

    def test_score_warning_subcooling(self, detector):
        """Test score for warning subcooling."""
        score = detector._score_subcooling(2.5)
        assert 0 < score < 0.5

    def test_score_alarm_subcooling(self, detector):
        """Test score for alarm subcooling."""
        score = detector._score_subcooling(4.0)
        assert 0.5 <= score < 0.8

    def test_score_severe_subcooling(self, detector):
        """Test score for severe subcooling."""
        score = detector._score_subcooling(7.0)
        assert score >= 0.8

    @pytest.mark.parametrize("sc,expected_range", [
        (0.5, (0.0, 0.1)),
        (2.0, (0.1, 0.4)),
        (3.5, (0.4, 0.7)),
        (5.0, (0.6, 0.9)),
        (8.0, (0.8, 1.0)),
    ])
    def test_score_ranges(self, detector, sc, expected_range):
        """Test subcooling scores fall in expected ranges."""
        score = detector._score_subcooling(sc)
        assert expected_range[0] <= score <= expected_range[1]


# =============================================================================
# VACUUM DEVIATION SCORING TESTS
# =============================================================================

class TestVacuumDeviationScoring:
    """Test vacuum deviation indicator scoring."""

    def test_score_no_deviation(self, detector):
        """Test score for no vacuum deviation."""
        score = detector._score_vacuum_deviation(1.5, 1.5)
        assert score == 0.0

    def test_score_better_vacuum(self, detector):
        """Test score for better than expected vacuum."""
        score = detector._score_vacuum_deviation(1.4, 1.5)
        assert score == 0.0

    def test_score_small_deviation(self, detector):
        """Test score for small vacuum deviation."""
        score = detector._score_vacuum_deviation(1.6, 1.5)
        assert 0 < score < 0.3

    def test_score_moderate_deviation(self, detector):
        """Test score for moderate vacuum deviation."""
        score = detector._score_vacuum_deviation(1.9, 1.5)
        assert 0.3 <= score < 0.7

    def test_score_large_deviation(self, detector):
        """Test score for large vacuum deviation."""
        score = detector._score_vacuum_deviation(2.5, 1.5)
        assert score >= 0.7


# =============================================================================
# AIR REMOVAL RATE SCORING TESTS
# =============================================================================

class TestAirRemovalRateScoring:
    """Test air removal rate indicator scoring."""

    def test_score_no_measurement(self, detector):
        """Test score when air removal not measured."""
        score = detector._score_air_removal_rate(None)
        assert score == 0.0

    def test_score_normal_rate(self, detector):
        """Test score for normal air removal rate."""
        # Design capacity = 50 SCFM, normal = 25 SCFM
        score = detector._score_air_removal_rate(20.0)
        assert score == 0.0

    def test_score_elevated_rate(self, detector):
        """Test score for elevated air removal rate."""
        score = detector._score_air_removal_rate(35.0)  # 70% of capacity
        assert 0 < score < 0.5

    def test_score_high_rate(self, detector):
        """Test score for high air removal rate."""
        score = detector._score_air_removal_rate(45.0)  # 90% of capacity
        assert score >= 0.5


# =============================================================================
# SEVERITY DETERMINATION TESTS
# =============================================================================

class TestSeverityDetermination:
    """Test severity determination logic."""

    def test_severity_none(self, detector):
        """Test none severity for low score."""
        severity = detector._determine_severity(0.2)
        assert severity == "none"

    def test_severity_minor(self, detector):
        """Test minor severity."""
        severity = detector._determine_severity(0.4)
        assert severity == "minor"

    def test_severity_moderate(self, detector):
        """Test moderate severity."""
        severity = detector._determine_severity(0.6)
        assert severity == "moderate"

    def test_severity_severe(self, detector):
        """Test severe severity."""
        severity = detector._determine_severity(0.8)
        assert severity == "severe"

    @pytest.mark.parametrize("score,expected", [
        (0.1, "none"),
        (0.35, "minor"),
        (0.55, "moderate"),
        (0.75, "severe"),
        (0.9, "severe"),
    ])
    def test_severity_thresholds(self, detector, score, expected):
        """Test severity at various thresholds."""
        severity = detector._determine_severity(score)
        assert severity == expected


# =============================================================================
# AIR INGRESS DETECTION TESTS
# =============================================================================

class TestAirIngressDetection:
    """Test main detection method."""

    def test_detect_no_ingress(self, detector):
        """Test detection with no air ingress indicators."""
        result = detector.detect_air_ingress(
            dissolved_o2_ppb=5.0,
            subcooling_f=0.5,
            condenser_vacuum_inhga=1.5,
            expected_vacuum_inhga=1.5,
        )

        assert isinstance(result, AirIngresResult)
        assert result.air_ingress_detected is False
        assert result.ingress_severity == "none"

    def test_detect_moderate_ingress(self, detector):
        """Test detection with moderate air ingress."""
        result = detector.detect_air_ingress(
            dissolved_o2_ppb=30.0,
            subcooling_f=3.5,
            condenser_vacuum_inhga=1.8,
            expected_vacuum_inhga=1.5,
        )

        assert result.air_ingress_detected is True
        assert result.ingress_severity in ["minor", "moderate"]

    def test_detect_severe_ingress(self, detector):
        """Test detection with severe air ingress."""
        result = detector.detect_air_ingress(
            dissolved_o2_ppb=80.0,
            subcooling_f=6.0,
            condenser_vacuum_inhga=2.5,
            expected_vacuum_inhga=1.5,
            air_removal_scfm=45.0,
        )

        assert result.air_ingress_detected is True
        assert result.ingress_severity in ["moderate", "severe"]

    def test_result_components(self, detector):
        """Test all result components are populated."""
        result = detector.detect_air_ingress(
            dissolved_o2_ppb=25.0,
            subcooling_f=3.0,
            condenser_vacuum_inhga=1.7,
        )

        assert result.air_ingress_detected is not None
        assert result.ingress_severity is not None
        assert result.estimated_air_ingress_scfm is not None
        assert result.confidence_pct is not None

    def test_subcooling_calculated_from_temps(self, detector):
        """Test subcooling is calculated from temperatures."""
        result = detector.detect_air_ingress(
            condenser_vacuum_inhga=1.6,
            saturation_temp_f=101.0,
            hotwell_temp_f=98.0,  # 3F subcooling
        )

        assert result.subcooling_observed_f == 3.0


# =============================================================================
# LEAK LOCATION IDENTIFICATION TESTS
# =============================================================================

class TestLeakLocationIdentification:
    """Test leak location identification."""

    def test_no_locations_for_low_ingress(self, detector):
        """Test no locations identified for low ingress."""
        locations = detector._identify_probable_locations(
            dissolved_o2_ppb=5.0,
            subcooling_f=0.5,
            estimated_ingress=0.5,
            air_removal_scfm=5.0,
        )

        assert len(locations) == 0

    def test_major_sources_for_high_ingress(self, detector):
        """Test major sources identified for high ingress."""
        locations = detector._identify_probable_locations(
            dissolved_o2_ppb=50.0,
            subcooling_f=5.0,
            estimated_ingress=15.0,
            air_removal_scfm=40.0,
        )

        assert len(locations) > 0
        # Should include major leak sources
        assert any("shaft" in loc or "hood" in loc for loc in locations)

    def test_pump_seals_for_high_do(self, detector):
        """Test pump seals identified for high DO."""
        locations = detector._identify_probable_locations(
            dissolved_o2_ppb=80.0,  # Very high DO
            subcooling_f=2.0,
            estimated_ingress=5.0,
            air_removal_scfm=None,
        )

        assert "condensate_pump_seals" in locations

    def test_top_shell_for_high_subcooling(self, detector):
        """Test top shell identified for high subcooling."""
        locations = detector._identify_probable_locations(
            dissolved_o2_ppb=20.0,
            subcooling_f=5.0,  # High subcooling
            estimated_ingress=5.0,
            air_removal_scfm=None,
        )

        assert "condenser_shell_top" in locations

    def test_max_five_locations(self, detector):
        """Test maximum 5 locations returned."""
        locations = detector._identify_probable_locations(
            dissolved_o2_ppb=100.0,
            subcooling_f=8.0,
            estimated_ingress=20.0,
            air_removal_scfm=50.0,
        )

        assert len(locations) <= 5


# =============================================================================
# CONFIDENCE CALCULATION TESTS
# =============================================================================

class TestConfidenceCalculation:
    """Test confidence calculation."""

    def test_base_confidence(self, detector):
        """Test base confidence with vacuum only."""
        confidence = detector._calculate_confidence(
            has_do=False,
            has_subcooling=False,
            has_air_rate=False,
        )

        assert confidence == 50.0

    def test_confidence_with_do(self, detector):
        """Test confidence with DO measurement."""
        confidence = detector._calculate_confidence(
            has_do=True,
            has_subcooling=False,
            has_air_rate=False,
        )

        assert confidence == 70.0

    def test_confidence_with_all_indicators(self, detector):
        """Test confidence with all indicators."""
        confidence = detector._calculate_confidence(
            has_do=True,
            has_subcooling=True,
            has_air_rate=True,
        )

        assert confidence == 100.0


# =============================================================================
# HEAT RATE IMPACT TESTS
# =============================================================================

class TestHeatRateImpact:
    """Test heat rate impact calculation."""

    def test_no_impact_normal(self, detector):
        """Test no impact for normal conditions."""
        impact = detector._calculate_heat_rate_impact(
            subcooling_f=0.5,
            estimated_ingress=2.0,
        )

        assert impact == 0.0

    def test_impact_from_subcooling(self, detector):
        """Test impact from subcooling."""
        impact = detector._calculate_heat_rate_impact(
            subcooling_f=3.0,  # 2F above normal
            estimated_ingress=0.0,
        )

        # 2F * 5 BTU/kWh/F = 10 BTU/kWh
        expected = 2.0 * AirIngressConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_F
        assert impact == pytest.approx(expected, rel=0.1)

    def test_impact_from_air_blanket(self, detector):
        """Test additional impact from air blanket."""
        impact = detector._calculate_heat_rate_impact(
            subcooling_f=1.0,  # No subcooling penalty
            estimated_ingress=10.0,  # High ingress
        )

        # Should have air blanket penalty
        assert impact > 0


# =============================================================================
# DO IMPACT ASSESSMENT TESTS
# =============================================================================

class TestDOImpactAssessment:
    """Test dissolved oxygen impact assessment."""

    def test_impact_unknown(self, detector):
        """Test unknown impact when DO not measured."""
        impact = detector._assess_do_impact(None)
        assert impact == "unknown"

    def test_impact_none(self, detector):
        """Test no impact for low DO."""
        impact = detector._assess_do_impact(5.0)
        assert impact == "none"

    def test_impact_minor(self, detector):
        """Test minor corrosion risk."""
        impact = detector._assess_do_impact(15.0)
        assert impact == "minor_corrosion_risk"

    def test_impact_moderate(self, detector):
        """Test moderate corrosion risk."""
        impact = detector._assess_do_impact(35.0)
        assert impact == "moderate_corrosion_risk"

    def test_impact_severe(self, detector):
        """Test severe corrosion risk."""
        impact = detector._assess_do_impact(80.0)
        assert impact == "severe_corrosion_risk"


# =============================================================================
# TEST METHOD RECOMMENDATION TESTS
# =============================================================================

class TestTestMethodRecommendation:
    """Test leak test method recommendation."""

    def test_no_test_for_none(self, detector):
        """Test no test method for none severity."""
        method = detector._recommend_test_method("none", [])
        assert method is None

    def test_ultrasonic_for_minor(self, detector):
        """Test ultrasonic for minor severity."""
        method = detector._recommend_test_method("minor", [])
        assert "ultrasonic" in method.lower()

    def test_helium_for_moderate(self, detector):
        """Test helium tracer for moderate severity."""
        method = detector._recommend_test_method("moderate", [])
        assert "helium" in method.lower()

    def test_comprehensive_for_severe(self, detector):
        """Test comprehensive method for severe severity."""
        method = detector._recommend_test_method("severe", [])
        assert "helium" in method.lower()
        assert "mass_spectrometer" in method.lower()


# =============================================================================
# LEAK SURVEY ANALYSIS TESTS
# =============================================================================

class TestLeakSurveyAnalysis:
    """Test leak survey result analysis."""

    def test_analyze_clean_survey(self, detector):
        """Test analysis of clean survey."""
        result = detector.analyze_leak_survey_results(
            zone_readings={
                "LP_turbine_exhaust": 0.05,
                "condenser_shell_top": 0.03,
            }
        )

        assert result["leaks_identified"] == 0

    def test_analyze_survey_with_leaks(self, detector):
        """Test analysis with identified leaks."""
        result = detector.analyze_leak_survey_results(
            zone_readings={
                "LP_turbine_exhaust": 0.5,  # Elevated
                "condenser_shell_top": 0.3,  # Elevated
                "instrument_connections": 0.05,  # Normal
            },
            baseline_readings={
                "LP_turbine_exhaust": 0.05,
                "condenser_shell_top": 0.05,
                "instrument_connections": 0.05,
            }
        )

        assert result["leaks_identified"] == 2

    def test_leak_prioritization(self, detector):
        """Test leaks are prioritized by contribution."""
        result = detector.analyze_leak_survey_results(
            zone_readings={
                "LP_turbine_exhaust": 1.0,
                "condenser_shell_top": 0.5,
            }
        )

        # Larger leak should be first
        if result["leaks_identified"] > 0:
            first_leak = result["leak_details"][0]["zone"]
            assert first_leak == "LP_turbine_exhaust"


# =============================================================================
# HISTORY TESTS
# =============================================================================

class TestAirIngressHistory:
    """Test air ingress history management."""

    def test_record_reading(self, detector):
        """Test recording readings."""
        detector._record_reading(25.0, 2.5, 1.6, 30.0)

        assert len(detector._history) == 1

    def test_get_trend_data(self, detector):
        """Test retrieving trend data."""
        for i in range(5):
            detector._record_reading(
                20.0 + i,  # Increasing DO
                2.0 + (i * 0.2),  # Increasing subcooling
                1.5 + (i * 0.05),  # Increasing vacuum
                25.0 + i,  # Increasing air removal
            )

        trends = detector.get_trend_data(hours=24)

        assert "dissolved_o2" in trends
        assert "subcooling" in trends
        assert "vacuum" in trends
        assert "air_removal" in trends


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Test calculation counting."""

    def test_count_increments(self, detector):
        """Test calculation count increments."""
        initial = detector.calculation_count

        detector.detect_air_ingress(
            condenser_vacuum_inhga=1.5,
        )

        assert detector.calculation_count == initial + 1
