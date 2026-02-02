"""
GL-016 WATERGUARD Agent - Feedwater Analyzer Tests

Unit tests for FeedwaterAnalyzer covering:
- pH analysis per ASME guidelines
- Dissolved oxygen control
- Hardness monitoring
- Corrosion product transport (iron, copper)
- Oxygen scavenger effectiveness
- Provenance tracking

Author: GL-TestEngineer
Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    FeedwaterInput,
    FeedwaterOutput,
    WaterQualityStatus,
    ChemicalType,
    BoilerPressureClass,
)

from greenlang.agents.process_heat.gl_016_water_treatment.feedwater import (
    FeedwaterAnalyzer,
    FeedwaterConstants,
    calculate_scavenger_requirement,
)


class TestFeedwaterAnalyzerInitialization:
    """Test FeedwaterAnalyzer initialization."""

    def test_default_initialization(self):
        """Test analyzer initializes with defaults."""
        analyzer = FeedwaterAnalyzer()
        assert analyzer.pressure_class == BoilerPressureClass.MEDIUM_PRESSURE
        assert analyzer.scavenger_type == ChemicalType.SULFITE
        assert analyzer.limits is not None
        assert analyzer.scavenger_config is not None

    def test_initialization_with_pressure_class(self):
        """Test analyzer initializes with specified pressure class."""
        analyzer = FeedwaterAnalyzer(
            pressure_class=BoilerPressureClass.HIGH_PRESSURE,
        )
        assert analyzer.pressure_class == BoilerPressureClass.HIGH_PRESSURE
        # High pressure has stricter limits
        assert analyzer.limits.dissolved_oxygen_max_ppb == 5

    def test_initialization_with_scavenger_type(self):
        """Test analyzer initializes with specified scavenger type."""
        analyzer = FeedwaterAnalyzer(
            scavenger_type=ChemicalType.HYDRAZINE,
        )
        assert analyzer.scavenger_type == ChemicalType.HYDRAZINE
        assert analyzer.scavenger_config is not None


class TestFeedwaterAnalysis:
    """Test feedwater analysis methods."""

    @pytest.fixture
    def analyzer(self):
        """Create standard feedwater analyzer."""
        return FeedwaterAnalyzer(
            pressure_class=BoilerPressureClass.MEDIUM_PRESSURE,
            scavenger_type=ChemicalType.SULFITE,
        )

    def test_analyze_excellent_feedwater(self, analyzer, feedwater_input_excellent):
        """Test analysis of excellent quality feedwater."""
        result = analyzer.analyze(feedwater_input_excellent)

        assert isinstance(result, FeedwaterOutput)
        assert result.sample_id == feedwater_input_excellent.sample_id
        assert result.overall_status in [WaterQualityStatus.EXCELLENT, WaterQualityStatus.GOOD]
        assert result.oxygen_control_adequate == True
        assert result.iron_transport_concern == False
        assert result.copper_transport_concern == False
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256
        assert result.processing_time_ms > 0

    def test_analyze_warning_feedwater(self, analyzer, feedwater_input_warning):
        """Test analysis of warning quality feedwater."""
        result = analyzer.analyze(feedwater_input_warning)

        assert result.overall_status in [WaterQualityStatus.WARNING, WaterQualityStatus.OUT_OF_SPEC]
        assert len(result.recommendations) > 0

    def test_analyze_critical_feedwater(self, analyzer, feedwater_input_critical):
        """Test analysis of critical quality feedwater."""
        result = analyzer.analyze(feedwater_input_critical)

        assert result.overall_status in [WaterQualityStatus.CRITICAL, WaterQualityStatus.OUT_OF_SPEC]
        assert result.iron_transport_concern == True
        assert result.copper_transport_concern == True
        assert len(result.recommendations) > 0


class TestPHAnalysis:
    """Test pH analysis methods."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    @pytest.mark.parametrize("ph,expected_status", [
        (9.0, WaterQualityStatus.EXCELLENT),  # Target pH
        (9.3, WaterQualityStatus.GOOD),
        (8.6, WaterQualityStatus.ACCEPTABLE),
        (8.4, WaterQualityStatus.OUT_OF_SPEC),  # Below min
        (9.7, WaterQualityStatus.OUT_OF_SPEC),  # Above max
        (7.5, WaterQualityStatus.CRITICAL),  # Way below
    ])
    def test_ph_status_classification(self, analyzer, ph, expected_status):
        """Test pH status classification."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=ph,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
        )
        result = analyzer._analyze_ph(input_data)
        assert result.status == expected_status

    def test_ph_result_fields(self, analyzer, feedwater_input_excellent):
        """Test pH result has all required fields."""
        result = analyzer._analyze_ph(feedwater_input_excellent)

        assert result.parameter == "pH"
        assert result.value == feedwater_input_excellent.ph
        assert result.unit == "pH units"
        assert result.min_limit is not None
        assert result.max_limit is not None
        assert result.target_value is not None
        assert result.deviation_pct is not None


class TestDissolvedOxygenAnalysis:
    """Test dissolved oxygen analysis."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    @pytest.mark.parametrize("do_ppb,expected_status", [
        (2.0, WaterQualityStatus.EXCELLENT),  # Well below limit
        (3.5, WaterQualityStatus.EXCELLENT),  # At 50% of limit
        (5.0, WaterQualityStatus.GOOD),
        (6.5, WaterQualityStatus.WARNING),
        (8.0, WaterQualityStatus.OUT_OF_SPEC),  # Above 7 ppb limit
        (15.0, WaterQualityStatus.CRITICAL),  # Way above
    ])
    def test_do_status_classification(self, analyzer, do_ppb, expected_status):
        """Test dissolved oxygen status classification."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=do_ppb,
        )
        result = analyzer._analyze_dissolved_oxygen(input_data)
        assert result.status == expected_status

    def test_do_limit_by_pressure_class(self):
        """Test DO limits vary by pressure class."""
        mp_analyzer = FeedwaterAnalyzer(pressure_class=BoilerPressureClass.MEDIUM_PRESSURE)
        hp_analyzer = FeedwaterAnalyzer(pressure_class=BoilerPressureClass.HIGH_PRESSURE)

        assert mp_analyzer.limits.dissolved_oxygen_max_ppb == 7
        assert hp_analyzer.limits.dissolved_oxygen_max_ppb == 5


class TestHardnessAnalysis:
    """Test total hardness analysis."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    @pytest.mark.parametrize("hardness,expected_status", [
        (0.02, WaterQualityStatus.EXCELLENT),  # Very low
        (0.05, WaterQualityStatus.GOOD),
        (0.08, WaterQualityStatus.ACCEPTABLE),
        (0.15, WaterQualityStatus.OUT_OF_SPEC),  # Above 0.1 ppm limit
        (0.5, WaterQualityStatus.CRITICAL),  # Hardness breakthrough
    ])
    def test_hardness_status_classification(self, analyzer, hardness, expected_status):
        """Test hardness status classification."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            total_hardness_ppm=hardness,
        )
        result = analyzer._analyze_hardness(input_data)
        assert result.status == expected_status


class TestIronAnalysis:
    """Test iron (corrosion product) analysis."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    @pytest.mark.parametrize("iron,expected_status", [
        (5.0, WaterQualityStatus.EXCELLENT),
        (10.0, WaterQualityStatus.EXCELLENT),
        (15.0, WaterQualityStatus.GOOD),
        (25.0, WaterQualityStatus.OUT_OF_SPEC),  # Above 20 ppb limit
        (70.0, WaterQualityStatus.CRITICAL),  # Way above
    ])
    def test_iron_status_classification(self, analyzer, iron, expected_status):
        """Test iron status classification."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            iron_ppb=iron,
        )
        result = analyzer._analyze_iron(input_data)
        assert result.status == expected_status

    def test_iron_transport_concern_evaluation(self, analyzer):
        """Test iron transport concern evaluation."""
        # Below limit - no concern
        input_low = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            iron_ppb=15.0,
        )
        assert analyzer._evaluate_iron_transport(input_low) == False

        # Above limit - concern
        input_high = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            iron_ppb=30.0,
        )
        assert analyzer._evaluate_iron_transport(input_high) == True


class TestCopperAnalysis:
    """Test copper (corrosion product) analysis."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    @pytest.mark.parametrize("copper,expected_status", [
        (3.0, WaterQualityStatus.EXCELLENT),
        (7.5, WaterQualityStatus.EXCELLENT),
        (12.0, WaterQualityStatus.GOOD),
        (20.0, WaterQualityStatus.OUT_OF_SPEC),  # Above 15 ppb limit
        (50.0, WaterQualityStatus.CRITICAL),
    ])
    def test_copper_status_classification(self, analyzer, copper, expected_status):
        """Test copper status classification."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            copper_ppb=copper,
        )
        result = analyzer._analyze_copper(input_data)
        assert result.status == expected_status


class TestOxygenScavengerAnalysis:
    """Test oxygen scavenger residual analysis."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer(scavenger_type=ChemicalType.SULFITE)

    @pytest.mark.parametrize("residual,expected_status", [
        (30.0, WaterQualityStatus.EXCELLENT),  # At target
        (25.0, WaterQualityStatus.GOOD),
        (20.0, WaterQualityStatus.ACCEPTABLE),  # At minimum
        (15.0, WaterQualityStatus.OUT_OF_SPEC),  # Below minimum
        (5.0, WaterQualityStatus.CRITICAL),  # Very low
    ])
    def test_scavenger_residual_classification(self, analyzer, residual, expected_status):
        """Test scavenger residual status classification."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            oxygen_scavenger_residual_ppm=residual,
            oxygen_scavenger_type=ChemicalType.SULFITE,
        )
        result = analyzer._analyze_scavenger_residual(input_data)
        assert result.status == expected_status


class TestOxygenControlEvaluation:
    """Test oxygen control evaluation."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    def test_oxygen_control_adequate(self, analyzer):
        """Test adequate oxygen control evaluation."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,  # Within limit
            oxygen_scavenger_residual_ppm=30.0,  # Good residual
        )
        is_adequate, adjustment = analyzer._evaluate_oxygen_control(input_data)
        assert is_adequate == True
        assert adjustment is None

    def test_oxygen_control_inadequate_low_residual(self, analyzer):
        """Test inadequate oxygen control due to low residual."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            oxygen_scavenger_residual_ppm=10.0,  # Below minimum
        )
        is_adequate, adjustment = analyzer._evaluate_oxygen_control(input_data)
        assert is_adequate == False
        assert adjustment is not None
        assert adjustment > 0  # Need to increase dose

    def test_oxygen_control_inadequate_high_do(self, analyzer):
        """Test inadequate oxygen control due to high DO."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=15.0,  # Above limit
            oxygen_scavenger_residual_ppm=30.0,
        )
        is_adequate, adjustment = analyzer._evaluate_oxygen_control(input_data)
        assert is_adequate == False


class TestOverallStatusDetermination:
    """Test overall status determination."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    def test_overall_status_excellent(self, analyzer, feedwater_input_excellent):
        """Test overall status is excellent for good feedwater."""
        result = analyzer.analyze(feedwater_input_excellent)
        assert result.overall_status in [WaterQualityStatus.EXCELLENT, WaterQualityStatus.GOOD]

    def test_overall_status_worst_parameter(self, analyzer):
        """Test overall status is based on worst parameter."""
        from greenlang.agents.process_heat.gl_016_water_treatment.schemas import WaterQualityResult

        results = [
            WaterQualityResult(
                parameter="pH",
                value=9.0,
                unit="pH units",
                status=WaterQualityStatus.EXCELLENT,
            ),
            WaterQualityResult(
                parameter="DO",
                value=10.0,
                unit="ppb",
                status=WaterQualityStatus.CRITICAL,  # Worst
            ),
            WaterQualityResult(
                parameter="Iron",
                value=10.0,
                unit="ppb",
                status=WaterQualityStatus.GOOD,
            ),
        ]
        overall = analyzer._determine_overall_status(results)
        assert overall == WaterQualityStatus.CRITICAL


class TestRecommendationGeneration:
    """Test recommendation generation."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    def test_recommendations_for_high_do(self, analyzer):
        """Test recommendations generated for high dissolved oxygen."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=15.0,  # High
            oxygen_scavenger_residual_ppm=15.0,  # Low
        )
        result = analyzer.analyze(input_data)

        # Should have recommendations about DO and scavenger
        assert len(result.recommendations) > 0
        rec_text = " ".join(result.recommendations).lower()
        assert "deaerator" in rec_text or "scavenger" in rec_text

    def test_recommendations_for_iron_transport(self, analyzer):
        """Test recommendations generated for iron transport."""
        input_data = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
            iron_ppb=50.0,  # High
        )
        result = analyzer.analyze(input_data)

        assert result.iron_transport_concern == True
        rec_text = " ".join(result.recommendations).lower()
        assert "iron" in rec_text or "corrosion" in rec_text


class TestCalculateScavengerRequirement:
    """Test scavenger requirement calculation function."""

    def test_sulfite_requirement(self):
        """Test sulfite scavenger requirement calculation."""
        # 5 ppb O2 with 50% excess
        dose = calculate_scavenger_requirement(
            dissolved_oxygen_ppb=5.0,
            scavenger_type=ChemicalType.SULFITE,
            excess_factor=1.5,
        )
        # Expected: 5/1000 * 7.9 * 1.5 = 0.059 ppm
        assert dose == pytest.approx(0.06, rel=0.1)

    def test_hydrazine_requirement(self):
        """Test hydrazine scavenger requirement calculation."""
        dose = calculate_scavenger_requirement(
            dissolved_oxygen_ppb=5.0,
            scavenger_type=ChemicalType.HYDRAZINE,
            excess_factor=2.0,  # 100% excess
        )
        # Expected: 5/1000 * 1.0 * 2.0 = 0.01 ppm
        assert dose == pytest.approx(0.01, rel=0.1)

    def test_unknown_scavenger_defaults_to_sulfite(self):
        """Test unknown scavenger defaults to sulfite ratio."""
        dose = calculate_scavenger_requirement(
            dissolved_oxygen_ppb=5.0,
            scavenger_type=ChemicalType.POLYMER,  # Not a scavenger
            excess_factor=1.5,
        )
        # Should use default sulfite ratio
        assert dose > 0


class TestProvenanceTracking:
    """Test provenance hash calculation."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    def test_provenance_hash_generated(self, analyzer, feedwater_input_excellent):
        """Test provenance hash is generated."""
        result = analyzer.analyze(feedwater_input_excellent)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_reproducible(self, analyzer, feedwater_input_excellent):
        """Test same input produces same provenance hash."""
        hash1 = analyzer._calculate_provenance_hash(feedwater_input_excellent)
        hash2 = analyzer._calculate_provenance_hash(feedwater_input_excellent)
        assert hash1 == hash2

    def test_provenance_hash_different_inputs(self, analyzer):
        """Test different inputs produce different hashes."""
        input1 = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.0,
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
        )
        input2 = FeedwaterInput(
            sample_point="da_outlet",
            ph=9.1,  # Different pH
            specific_conductivity_umho=0.5,
            dissolved_oxygen_ppb=5.0,
        )
        hash1 = analyzer._calculate_provenance_hash(input1)
        hash2 = analyzer._calculate_provenance_hash(input2)
        assert hash1 != hash2


class TestFeedwaterConstants:
    """Test feedwater constants values."""

    def test_do_limits_defined(self):
        """Test DO limits are defined for all pressure classes."""
        assert "low_pressure" in FeedwaterConstants.DO_LIMITS
        assert "medium_pressure" in FeedwaterConstants.DO_LIMITS
        assert "high_pressure" in FeedwaterConstants.DO_LIMITS
        assert "supercritical" in FeedwaterConstants.DO_LIMITS

    def test_do_limits_stricter_with_pressure(self):
        """Test DO limits decrease with pressure."""
        limits = FeedwaterConstants.DO_LIMITS
        assert limits["low_pressure"] >= limits["medium_pressure"]
        assert limits["medium_pressure"] >= limits["high_pressure"]
        assert limits["high_pressure"] >= limits["supercritical"]

    def test_iron_limits_defined(self):
        """Test iron limits are defined."""
        assert len(FeedwaterConstants.IRON_LIMITS) == 4

    def test_copper_limits_defined(self):
        """Test copper limits are defined."""
        assert len(FeedwaterConstants.COPPER_LIMITS) == 4

    def test_scavenger_ratios_defined(self):
        """Test scavenger stoichiometric ratios are defined."""
        assert FeedwaterConstants.SULFITE_RATIO > 0
        assert FeedwaterConstants.HYDRAZINE_RATIO > 0


class TestPerformance:
    """Performance tests for feedwater analyzer."""

    @pytest.fixture
    def analyzer(self):
        return FeedwaterAnalyzer()

    @pytest.mark.performance
    def test_analysis_performance(self, analyzer, feedwater_input_excellent):
        """Test analysis completes within performance target."""
        import time
        start = time.perf_counter()
        result = analyzer.analyze(feedwater_input_excellent)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50  # Should complete in < 50ms
        assert result.processing_time_ms < 50

    @pytest.mark.performance
    def test_batch_analysis_performance(self, analyzer, feedwater_input_excellent):
        """Test batch analysis maintains throughput."""
        import time
        num_samples = 100

        start = time.perf_counter()
        for _ in range(num_samples):
            analyzer.analyze(feedwater_input_excellent)
        elapsed_s = time.perf_counter() - start

        throughput = num_samples / elapsed_s
        assert throughput > 100  # At least 100 samples/second


class TestComplianceWithASME:
    """Test ASME compliance of feedwater analysis."""

    @pytest.mark.compliance
    def test_dissolved_oxygen_limits_per_asme(self):
        """Test DO limits match ASME guidelines."""
        # ASME specifies 7 ppb for up to 900 psig, 5 ppb for higher
        mp_analyzer = FeedwaterAnalyzer(pressure_class=BoilerPressureClass.MEDIUM_PRESSURE)
        hp_analyzer = FeedwaterAnalyzer(pressure_class=BoilerPressureClass.HIGH_PRESSURE)

        assert mp_analyzer.limits.dissolved_oxygen_max_ppb == 7
        assert hp_analyzer.limits.dissolved_oxygen_max_ppb == 5

    @pytest.mark.compliance
    def test_hardness_limits_per_asme(self):
        """Test hardness limits match ASME guidelines."""
        mp_analyzer = FeedwaterAnalyzer(pressure_class=BoilerPressureClass.MEDIUM_PRESSURE)
        hp_analyzer = FeedwaterAnalyzer(pressure_class=BoilerPressureClass.HIGH_PRESSURE)

        # ASME specifies stricter limits at higher pressure
        assert mp_analyzer.limits.total_hardness_max_ppm == 0.1
        assert hp_analyzer.limits.total_hardness_max_ppm == 0.05
