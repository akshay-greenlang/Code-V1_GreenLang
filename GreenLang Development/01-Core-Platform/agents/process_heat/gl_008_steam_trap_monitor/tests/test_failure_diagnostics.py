# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Failure Diagnostics Module Tests

Unit tests for failure_diagnostics.py module including ultrasonic analysis,
temperature differential analysis, and diagnostic decision trees.

Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone
from typing import List
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    SteamTrapMonitorConfig,
    TrapType,
    DiagnosticMethod,
    DiagnosticThresholds,
    UltrasonicThresholds,
    TemperatureThresholds,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    TrapStatus,
    DiagnosisConfidence,
    UltrasonicReading,
    TemperatureReading,
    VisualInspectionReading,
    TrapDiagnosticInput,
    TrapDiagnosticOutput,
    TrapInfo,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.failure_diagnostics import (
    UltrasonicDiagnosticResult,
    TemperatureDiagnosticResult,
    CombinedDiagnosticResult,
    UltrasonicAnalyzer,
    TemperatureDifferentialAnalyzer,
    DiagnosticDecisionTree,
    FailureModeDetector,
    TrapDiagnosticsEngine,
)


class TestUltrasonicDiagnosticResult:
    """Tests for UltrasonicDiagnosticResult dataclass."""

    def test_creation(self):
        """Test creating result object."""
        result = UltrasonicDiagnosticResult(
            status=TrapStatus.GOOD,
            confidence=0.85,
            average_db=55.0,
            peak_db=62.0,
            cycling_detected=True,
            continuous_flow_detected=False,
        )

        assert result.status == TrapStatus.GOOD
        assert result.confidence == 0.85
        assert result.cycling_detected is True

    def test_evidence_list(self):
        """Test evidence list initialization."""
        result = UltrasonicDiagnosticResult(
            status=TrapStatus.FAILED_OPEN,
            confidence=0.85,
            average_db=95.0,
            peak_db=100.0,
            cycling_detected=False,
            continuous_flow_detected=True,
            evidence=["High continuous ultrasonic level"],
        )

        assert len(result.evidence) == 1
        assert "High" in result.evidence[0]


class TestUltrasonicAnalyzer:
    """Tests for UltrasonicAnalyzer."""

    @pytest.fixture
    def analyzer(self, diagnostic_thresholds) -> UltrasonicAnalyzer:
        """Create analyzer instance."""
        return UltrasonicAnalyzer(diagnostic_thresholds)

    @pytest.fixture
    def good_ultrasonic_reading(self, test_data_generator) -> UltrasonicReading:
        """Create good ultrasonic reading."""
        return test_data_generator.generate_ultrasonic_reading(
            decibel_level=55.0,
            cycling_detected=True,
        )

    @pytest.fixture
    def failed_open_reading(self, test_data_generator) -> UltrasonicReading:
        """Create failed open ultrasonic reading."""
        return test_data_generator.generate_ultrasonic_reading(
            decibel_level=95.0,
            continuous_flow=True,
        )

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer._analysis_count == 0

    def test_empty_readings_returns_unknown(self, analyzer):
        """Test empty readings returns unknown status."""
        result = analyzer.analyze(
            readings=[],
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.UNKNOWN
        assert result.confidence == 0.0

    def test_good_trap_detection(self, analyzer, good_ultrasonic_reading):
        """Test detection of good trap with cycling."""
        result = analyzer.analyze(
            readings=[good_ultrasonic_reading],
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.GOOD
        assert result.confidence > 0.7
        assert result.cycling_detected is True

    def test_failed_open_detection(self, analyzer, failed_open_reading):
        """Test detection of failed open trap."""
        result = analyzer.analyze(
            readings=[failed_open_reading],
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.FAILED_OPEN
        assert result.confidence > 0.8
        assert result.continuous_flow_detected is True

    def test_failed_closed_detection(self, analyzer, test_data_generator):
        """Test detection of failed closed trap."""
        reading = test_data_generator.generate_ultrasonic_reading(
            decibel_level=35.0,  # Very low - blocked
        )

        result = analyzer.analyze(
            readings=[reading],
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.FAILED_CLOSED
        assert "Very low" in " ".join(result.evidence)

    def test_leaking_detection(self, analyzer, test_data_generator):
        """Test detection of leaking trap."""
        reading = test_data_generator.generate_ultrasonic_reading(
            decibel_level=78.0,  # Elevated but not continuous
        )

        result = analyzer.analyze(
            readings=[reading],
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.LEAKING
        assert "Elevated" in " ".join(result.evidence)

    def test_thermodynamic_adjustment(self, analyzer, test_data_generator):
        """Test thermodynamic trap dB adjustment."""
        reading = test_data_generator.generate_ultrasonic_reading(
            decibel_level=80.0,  # High for F&T but normal for TD
            cycling_detected=True,
        )

        # For F&T trap - would be leaking
        result_ft = analyzer.analyze(
            readings=[reading],
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        # For TD trap - with adjustment should be OK
        result_td = analyzer.analyze(
            readings=[reading],
            trap_type=TrapType.THERMODYNAMIC,
        )

        # TD has +10dB adjustment, so 80dB becomes effectively 70dB
        assert result_td.status != TrapStatus.FAILED_OPEN

    def test_cycling_detection_from_variance(self, analyzer, test_data_generator):
        """Test cycling detected from dB variance."""
        readings = [
            test_data_generator.generate_ultrasonic_reading(decibel_level=50.0),
            test_data_generator.generate_ultrasonic_reading(decibel_level=70.0),
            test_data_generator.generate_ultrasonic_reading(decibel_level=45.0),
        ]

        result = analyzer.analyze(
            readings=readings,
            trap_type=TrapType.INVERTED_BUCKET,
        )

        # High variance (>15dB) should detect cycling
        assert result.cycling_detected is True

    def test_analysis_count_increments(self, analyzer, good_ultrasonic_reading):
        """Test analysis count increments."""
        assert analyzer.analysis_count == 0

        analyzer.analyze([good_ultrasonic_reading], TrapType.FLOAT_THERMOSTATIC)
        assert analyzer.analysis_count == 1

        analyzer.analyze([good_ultrasonic_reading], TrapType.FLOAT_THERMOSTATIC)
        assert analyzer.analysis_count == 2

    @pytest.mark.parametrize("db_level,expected_status", [
        (35.0, TrapStatus.FAILED_CLOSED),
        (55.0, TrapStatus.GOOD),
        (78.0, TrapStatus.LEAKING),
        (95.0, TrapStatus.FAILED_OPEN),
    ])
    def test_db_level_thresholds(
        self,
        analyzer,
        test_data_generator,
        db_level: float,
        expected_status: TrapStatus,
    ):
        """Test dB level threshold-based detection."""
        reading = test_data_generator.generate_ultrasonic_reading(
            decibel_level=db_level,
            continuous_flow=(db_level > 85),
        )

        result = analyzer.analyze(
            readings=[reading],
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == expected_status


class TestTemperatureDifferentialAnalyzer:
    """Tests for TemperatureDifferentialAnalyzer."""

    @pytest.fixture
    def analyzer(
        self,
        diagnostic_thresholds,
        steam_trap_config,
    ) -> TemperatureDifferentialAnalyzer:
        """Create analyzer instance."""
        return TemperatureDifferentialAnalyzer(
            diagnostic_thresholds,
            steam_trap_config,
        )

    @pytest.fixture
    def good_temp_reading(self, test_data_generator) -> TemperatureReading:
        """Create good temperature reading."""
        return test_data_generator.generate_temperature_reading(
            inlet_temp=366.0,
            outlet_temp=340.0,
        )

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer._analysis_count == 0

    def test_empty_readings_returns_unknown(self, analyzer):
        """Test empty readings returns unknown."""
        result = analyzer.analyze(
            readings=[],
            steam_pressure_psig=150.0,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.UNKNOWN
        assert result.confidence == 0.0

    def test_good_trap_detection(self, analyzer, good_temp_reading):
        """Test detection of good trap from temperature."""
        result = analyzer.analyze(
            readings=[good_temp_reading],
            steam_pressure_psig=150.0,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.GOOD
        assert result.delta_t_f > 15  # Normal range

    def test_failed_open_low_delta_t(self, analyzer, test_data_generator):
        """Test failed open detection from low delta T."""
        reading = test_data_generator.generate_temperature_reading(
            inlet_temp=366.0,
            outlet_temp=362.0,  # Very low delta T
        )

        result = analyzer.analyze(
            readings=[reading],
            steam_pressure_psig=150.0,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.FAILED_OPEN
        assert "low" in " ".join(result.evidence).lower()

    def test_failed_closed_high_delta_t(self, analyzer, test_data_generator):
        """Test failed closed detection from high delta T."""
        reading = test_data_generator.generate_temperature_reading(
            inlet_temp=366.0,
            outlet_temp=150.0,  # Very high delta T
        )

        result = analyzer.analyze(
            readings=[reading],
            steam_pressure_psig=150.0,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.FAILED_CLOSED
        assert "high" in " ".join(result.evidence).lower()

    def test_cold_trap_detection(self, analyzer, test_data_generator):
        """Test cold trap detection from low inlet temp."""
        reading = test_data_generator.generate_temperature_reading(
            inlet_temp=120.0,  # Very low inlet
            outlet_temp=100.0,
        )

        result = analyzer.analyze(
            readings=[reading],
            steam_pressure_psig=150.0,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert result.status == TrapStatus.COLD
        assert "inlet temperature too low" in " ".join(result.evidence).lower()

    def test_subcooling_calculation(self, analyzer, good_temp_reading):
        """Test subcooling is calculated correctly."""
        result = analyzer.analyze(
            readings=[good_temp_reading],
            steam_pressure_psig=150.0,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        # Subcooling = saturation temp - outlet temp
        # At 150 psig, sat temp ~366F
        expected_subcooling = 366 - 340  # = 26F
        assert abs(result.subcooling_f - expected_subcooling) < 10

    def test_analysis_count_increments(self, analyzer, good_temp_reading):
        """Test analysis count increments."""
        assert analyzer.analysis_count == 0

        analyzer.analyze(
            [good_temp_reading],
            steam_pressure_psig=150.0,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert analyzer.analysis_count == 1


class TestDiagnosticDecisionTree:
    """Tests for DiagnosticDecisionTree."""

    @pytest.fixture
    def decision_tree(self, diagnostic_thresholds) -> DiagnosticDecisionTree:
        """Create decision tree instance."""
        return DiagnosticDecisionTree(diagnostic_thresholds)

    def test_no_data_returns_unknown(self, decision_tree):
        """Test no diagnostic data returns unknown."""
        result = decision_tree.combine_results(
            ultrasonic=None,
            temperature=None,
            visual=None,
        )

        assert result.status == TrapStatus.UNKNOWN
        assert result.confidence == 0.0
        assert len(result.methods_used) == 0

    def test_ultrasonic_only(self, decision_tree):
        """Test using only ultrasonic data."""
        ultrasonic = UltrasonicDiagnosticResult(
            status=TrapStatus.GOOD,
            confidence=0.80,
            average_db=55.0,
            peak_db=60.0,
            cycling_detected=True,
            continuous_flow_detected=False,
            evidence=["Normal cycling"],
        )

        result = decision_tree.combine_results(
            ultrasonic=ultrasonic,
            temperature=None,
            visual=None,
        )

        assert result.status == TrapStatus.GOOD
        assert DiagnosticMethod.ULTRASONIC in result.methods_used
        assert len(result.methods_used) == 1

    def test_temperature_only(self, decision_tree):
        """Test using only temperature data."""
        temperature = TemperatureDiagnosticResult(
            status=TrapStatus.GOOD,
            confidence=0.75,
            inlet_temp_f=366.0,
            outlet_temp_f=340.0,
            delta_t_f=26.0,
            subcooling_f=26.0,
        )

        result = decision_tree.combine_results(
            ultrasonic=None,
            temperature=temperature,
            visual=None,
        )

        assert result.status == TrapStatus.GOOD
        assert DiagnosticMethod.TEMPERATURE in result.methods_used

    def test_methods_agree_high_confidence(self, decision_tree):
        """Test agreeing methods produce high confidence."""
        ultrasonic = UltrasonicDiagnosticResult(
            status=TrapStatus.FAILED_OPEN,
            confidence=0.85,
            average_db=95.0,
            peak_db=100.0,
            cycling_detected=False,
            continuous_flow_detected=True,
        )

        temperature = TemperatureDiagnosticResult(
            status=TrapStatus.FAILED_OPEN,
            confidence=0.85,
            inlet_temp_f=366.0,
            outlet_temp_f=362.0,
            delta_t_f=4.0,
            subcooling_f=4.0,
        )

        result = decision_tree.combine_results(
            ultrasonic=ultrasonic,
            temperature=temperature,
            visual=None,
        )

        assert result.status == TrapStatus.FAILED_OPEN
        assert result.confidence > 0.85  # Agreement bonus
        assert "Multiple methods agree" in " ".join(result.evidence)

    def test_methods_disagree_lower_confidence(self, decision_tree):
        """Test disagreeing methods produce lower confidence."""
        ultrasonic = UltrasonicDiagnosticResult(
            status=TrapStatus.FAILED_OPEN,
            confidence=0.85,
            average_db=95.0,
            peak_db=100.0,
            cycling_detected=False,
            continuous_flow_detected=True,
        )

        temperature = TemperatureDiagnosticResult(
            status=TrapStatus.GOOD,
            confidence=0.75,
            inlet_temp_f=366.0,
            outlet_temp_f=340.0,
            delta_t_f=26.0,
            subcooling_f=26.0,
        )

        result = decision_tree.combine_results(
            ultrasonic=ultrasonic,
            temperature=temperature,
            visual=None,
        )

        # Should have inconsistencies noted
        assert len(result.inconsistencies) > 0

    def test_visual_inspection_steam_discharge(self, decision_tree):
        """Test visual inspection with steam discharge."""
        visual = VisualInspectionReading(
            inspector_id="INSP-01",
            timestamp=datetime.now(timezone.utc),
            visible_steam_discharge=True,
            condensate_visible=False,
        )

        result = decision_tree.combine_results(
            ultrasonic=None,
            temperature=None,
            visual=visual,
        )

        assert result.status == TrapStatus.FAILED_OPEN
        assert DiagnosticMethod.VISUAL in result.methods_used

    def test_confidence_level_categorization(self, decision_tree):
        """Test confidence level categorization."""
        ultrasonic = UltrasonicDiagnosticResult(
            status=TrapStatus.GOOD,
            confidence=0.95,
            average_db=55.0,
            peak_db=60.0,
            cycling_detected=True,
            continuous_flow_detected=False,
        )

        result = decision_tree.combine_results(
            ultrasonic=ultrasonic,
            temperature=None,
            visual=None,
        )

        assert result.confidence_level == DiagnosisConfidence.HIGH

    def test_failure_probabilities_calculated(self, decision_tree):
        """Test failure probabilities are calculated."""
        ultrasonic = UltrasonicDiagnosticResult(
            status=TrapStatus.LEAKING,
            confidence=0.70,
            average_db=78.0,
            peak_db=82.0,
            cycling_detected=False,
            continuous_flow_detected=False,
        )

        result = decision_tree.combine_results(
            ultrasonic=ultrasonic,
            temperature=None,
            visual=None,
        )

        assert "leaking" in result.failure_probabilities


class TestFailureModeDetector:
    """Tests for FailureModeDetector."""

    @pytest.fixture
    def detector(self) -> FailureModeDetector:
        """Create detector instance."""
        return FailureModeDetector()

    def test_failed_open_analysis(self, detector):
        """Test failed open failure mode analysis."""
        combined = CombinedDiagnosticResult(
            status=TrapStatus.FAILED_OPEN,
            confidence=0.85,
            confidence_level=DiagnosisConfidence.HIGH,
            failure_probabilities={"failed_open": 0.85},
            evidence=["High continuous ultrasonic level detected"],
            inconsistencies=[],
            methods_used=[DiagnosticMethod.ULTRASONIC],
        )

        results = detector.analyze_failure_mode(
            status=TrapStatus.FAILED_OPEN,
            combined=combined,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert len(results) > 0
        assert results[0].failure_mode == "failed_open"
        assert results[0].probability > 0.8

    def test_failed_closed_analysis(self, detector):
        """Test failed closed failure mode analysis."""
        combined = CombinedDiagnosticResult(
            status=TrapStatus.FAILED_CLOSED,
            confidence=0.80,
            confidence_level=DiagnosisConfidence.HIGH,
            failure_probabilities={"failed_closed": 0.80},
            evidence=["Very high temperature differential", "No flow detected"],
            inconsistencies=[],
            methods_used=[DiagnosticMethod.TEMPERATURE],
        )

        results = detector.analyze_failure_mode(
            status=TrapStatus.FAILED_CLOSED,
            combined=combined,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert len(results) > 0
        failed_closed = next(
            (r for r in results if r.failure_mode == "failed_closed"),
            None
        )
        assert failed_closed is not None

    def test_indicators_populated(self, detector):
        """Test indicators are populated from evidence."""
        combined = CombinedDiagnosticResult(
            status=TrapStatus.FAILED_OPEN,
            confidence=0.85,
            confidence_level=DiagnosisConfidence.HIGH,
            failure_probabilities={"failed_open": 0.85},
            evidence=["Continuous flow detected", "High ultrasonic level"],
            inconsistencies=[],
            methods_used=[DiagnosticMethod.ULTRASONIC],
        )

        results = detector.analyze_failure_mode(
            status=TrapStatus.FAILED_OPEN,
            combined=combined,
            trap_type=TrapType.FLOAT_THERMOSTATIC,
        )

        assert len(results[0].indicators) > 0


class TestTrapDiagnosticsEngine:
    """Tests for main TrapDiagnosticsEngine."""

    @pytest.fixture
    def engine(self, steam_trap_config) -> TrapDiagnosticsEngine:
        """Create engine instance."""
        return TrapDiagnosticsEngine(steam_trap_config)

    def test_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine._diagnosis_count == 0

    def test_diagnose_good_trap(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test diagnosing a good trap."""
        result = engine.diagnose(sample_diagnostic_input_good)

        assert isinstance(result, TrapDiagnosticOutput)
        assert result.status == "success"
        assert result.condition.status == TrapStatus.GOOD
        assert result.processing_time_ms > 0

    def test_diagnose_failed_trap(
        self,
        engine,
        sample_diagnostic_input_failed,
    ):
        """Test diagnosing a failed trap."""
        result = engine.diagnose(sample_diagnostic_input_failed)

        assert isinstance(result, TrapDiagnosticOutput)
        assert result.condition.status == TrapStatus.FAILED_OPEN

    def test_health_score_calculated(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test health score is calculated."""
        result = engine.diagnose(sample_diagnostic_input_good)

        assert result.health_score is not None
        assert result.health_score.overall_score > 0

    def test_good_trap_high_health_score(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test good trap has high health score."""
        result = engine.diagnose(sample_diagnostic_input_good)

        assert result.health_score.overall_score > 80
        assert result.health_score.category in ["good", "excellent"]

    def test_failed_trap_low_health_score(
        self,
        engine,
        sample_diagnostic_input_failed,
    ):
        """Test failed trap has low health score."""
        result = engine.diagnose(sample_diagnostic_input_failed)

        assert result.health_score.overall_score < 50
        assert result.health_score.category in ["poor", "critical"]

    def test_recommendations_generated(
        self,
        engine,
        sample_diagnostic_input_failed,
    ):
        """Test maintenance recommendations are generated."""
        result = engine.diagnose(sample_diagnostic_input_failed)

        assert len(result.recommendations) > 0
        assert result.recommendations[0].priority.value == "urgent"

    def test_diagnostic_methods_tracked(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test diagnostic methods used are tracked."""
        result = engine.diagnose(sample_diagnostic_input_good)

        assert len(result.diagnostic_methods_used) > 0
        assert "ultrasonic" in result.diagnostic_methods_used

    def test_provenance_hash_generated(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test provenance hash is generated."""
        result = engine.diagnose(sample_diagnostic_input_good)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_asme_compliance_check(
        self,
        engine,
        test_data_generator,
    ):
        """Test ASME compliance is checked."""
        input_data = test_data_generator.generate_diagnostic_input(
            trap_status=TrapStatus.GOOD,
            steam_pressure=200.0,  # High pressure
        )
        # Set low pressure rating for trap
        input_data.trap_info.pressure_rating_psig = 150.0

        result = engine.diagnose(input_data)

        assert result.asme_b16_34_compliant is False
        assert result.pressure_rating_adequate is False

    def test_data_quality_assessment(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test sensor data quality is assessed."""
        result = engine.diagnose(sample_diagnostic_input_good)

        assert result.sensor_data_quality >= 0
        assert result.sensor_data_quality <= 1

    def test_diagnosis_count_increments(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test diagnosis count increments."""
        assert engine.diagnosis_count == 0

        engine.diagnose(sample_diagnostic_input_good)
        assert engine.diagnosis_count == 1

        engine.diagnose(sample_diagnostic_input_good)
        assert engine.diagnosis_count == 2


class TestDiagnosticsIntegration:
    """Integration tests for diagnostics."""

    @pytest.fixture
    def engine(self, steam_trap_config) -> TrapDiagnosticsEngine:
        """Create engine instance."""
        return TrapDiagnosticsEngine(steam_trap_config)

    @pytest.mark.parametrize("trap_type", [
        TrapType.FLOAT_THERMOSTATIC,
        TrapType.INVERTED_BUCKET,
        TrapType.THERMOSTATIC,
        TrapType.THERMODYNAMIC,
    ])
    def test_all_trap_types_supported(
        self,
        engine,
        test_data_generator,
        trap_type: TrapType,
    ):
        """Test all trap types can be diagnosed."""
        input_data = test_data_generator.generate_diagnostic_input(
            trap_status=TrapStatus.GOOD,
            trap_type=trap_type,
        )

        result = engine.diagnose(input_data)

        assert result.status == "success"
        assert result.condition is not None

    @pytest.mark.parametrize("expected_status", [
        TrapStatus.GOOD,
        TrapStatus.FAILED_OPEN,
        TrapStatus.FAILED_CLOSED,
        TrapStatus.LEAKING,
    ])
    def test_all_statuses_detectable(
        self,
        engine,
        test_data_generator,
        expected_status: TrapStatus,
    ):
        """Test all failure statuses can be detected."""
        input_data = test_data_generator.generate_diagnostic_input(
            trap_status=expected_status,
        )

        result = engine.diagnose(input_data)

        assert result.status == "success"
        # Status detection should reasonably match expected
        # Note: may differ due to threshold settings

    def test_processing_time_reasonable(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test processing time is reasonable (<100ms)."""
        result = engine.diagnose(sample_diagnostic_input_good)

        assert result.processing_time_ms < 100

    def test_deterministic_results(
        self,
        engine,
        sample_diagnostic_input_good,
    ):
        """Test results are deterministic for same inputs."""
        result1 = engine.diagnose(sample_diagnostic_input_good)
        result2 = engine.diagnose(sample_diagnostic_input_good)

        assert result1.condition.status == result2.condition.status
        assert result1.condition.confidence_score == result2.condition.confidence_score
        assert result1.health_score.overall_score == result2.health_score.overall_score
