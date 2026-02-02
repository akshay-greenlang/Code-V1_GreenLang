"""
GL-016 WATERGUARD Agent - Condensate Analyzer Tests

Unit tests for CondensateAnalyzer covering:
- pH analysis for condensate corrosion
- Iron/copper corrosion product monitoring
- Contamination detection (hardness, oil)
- Amine treatment effectiveness
- Corrosion rate estimation
- Provenance tracking

Author: GL-TestEngineer
Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    CondensateInput,
    CondensateOutput,
    WaterQualityStatus,
    ChemicalType,
)

from greenlang.agents.process_heat.gl_016_water_treatment.condensate import (
    CondensateAnalyzer,
    CondensateConstants,
    calculate_amine_requirement,
)


class TestCondensateAnalyzerInitialization:
    """Test CondensateAnalyzer initialization."""

    def test_default_initialization(self):
        """Test analyzer initializes with defaults."""
        analyzer = CondensateAnalyzer()
        assert analyzer.amine_type is None
        assert analyzer.amine_config is None

    def test_initialization_with_amine(self):
        """Test analyzer initializes with amine type."""
        analyzer = CondensateAnalyzer(amine_type=ChemicalType.MORPHOLINE)
        assert analyzer.amine_type == ChemicalType.MORPHOLINE
        assert analyzer.amine_config is not None


class TestCondensateAnalysis:
    """Test condensate analysis methods."""

    @pytest.fixture
    def analyzer(self):
        """Create standard condensate analyzer with amine treatment."""
        return CondensateAnalyzer(amine_type=ChemicalType.MORPHOLINE)

    @pytest.fixture
    def analyzer_no_amine(self):
        """Create condensate analyzer without amine treatment."""
        return CondensateAnalyzer()

    def test_analyze_excellent_condensate(self, analyzer, condensate_input_excellent):
        """Test analysis of excellent quality condensate."""
        result = analyzer.analyze(condensate_input_excellent)

        assert isinstance(result, CondensateOutput)
        assert result.sample_id == condensate_input_excellent.sample_id
        assert result.overall_status in [WaterQualityStatus.EXCELLENT, WaterQualityStatus.GOOD]
        assert result.contamination_detected == False
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert result.processing_time_ms > 0

    def test_analyze_contaminated_condensate(self, analyzer, condensate_input_contaminated):
        """Test analysis of contaminated condensate."""
        result = analyzer.analyze(condensate_input_contaminated)

        assert result.overall_status in [WaterQualityStatus.WARNING, WaterQualityStatus.CRITICAL, WaterQualityStatus.OUT_OF_SPEC]
        assert result.contamination_detected == True
        assert result.contamination_source is not None
        assert len(result.recommendations) > 0

    def test_analyze_corrosion_indicators(self, analyzer, condensate_input_corrosion):
        """Test analysis of condensate with corrosion indicators."""
        result = analyzer.analyze(condensate_input_corrosion)

        assert result.overall_status in [WaterQualityStatus.WARNING, WaterQualityStatus.OUT_OF_SPEC]
        assert result.corrosion_rate_mpy is not None
        assert result.corrosion_rate_mpy > 0


class TestPHAnalysis:
    """Test pH analysis methods."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer(amine_type=ChemicalType.MORPHOLINE)

    @pytest.mark.parametrize("ph,expected_status", [
        (8.7, WaterQualityStatus.EXCELLENT),  # Target pH
        (8.5, WaterQualityStatus.GOOD),
        (8.2, WaterQualityStatus.ACCEPTABLE),
        (7.8, WaterQualityStatus.WARNING),  # Low - carbonic acid risk
        (7.0, WaterQualityStatus.CRITICAL),  # Very low
        (9.5, WaterQualityStatus.WARNING),  # High pH
    ])
    def test_ph_status_classification(self, analyzer, ph, expected_status):
        """Test pH status classification."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=ph,
            specific_conductivity_umho=1.0,
            iron_ppb=50.0,
        )
        result = analyzer._analyze_ph(input_data)
        assert result.status == expected_status

    def test_ph_result_fields(self, analyzer, condensate_input_excellent):
        """Test pH result has all required fields."""
        result = analyzer._analyze_ph(condensate_input_excellent)

        assert result.parameter == "pH"
        assert result.value == condensate_input_excellent.ph
        assert result.unit == "pH units"
        assert result.min_limit is not None
        assert result.max_limit is not None


class TestIronAnalysis:
    """Test iron (corrosion product) analysis."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer()

    @pytest.mark.parametrize("iron,expected_status", [
        (10.0, WaterQualityStatus.EXCELLENT),
        (30.0, WaterQualityStatus.GOOD),
        (75.0, WaterQualityStatus.WARNING),
        (150.0, WaterQualityStatus.OUT_OF_SPEC),
        (300.0, WaterQualityStatus.CRITICAL),
    ])
    def test_iron_status_classification(self, analyzer, iron, expected_status):
        """Test iron status classification."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=8.5,
            specific_conductivity_umho=1.0,
            iron_ppb=iron,
        )
        result = analyzer._analyze_iron(input_data)
        assert result.status == expected_status


class TestCopperAnalysis:
    """Test copper (corrosion product) analysis."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer()

    @pytest.mark.parametrize("copper,expected_status", [
        (3.0, WaterQualityStatus.EXCELLENT),
        (8.0, WaterQualityStatus.GOOD),
        (15.0, WaterQualityStatus.WARNING),
        (30.0, WaterQualityStatus.OUT_OF_SPEC),
        (60.0, WaterQualityStatus.CRITICAL),
    ])
    def test_copper_status_classification(self, analyzer, copper, expected_status):
        """Test copper status classification."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=8.5,
            specific_conductivity_umho=1.0,
            iron_ppb=50.0,
            copper_ppb=copper,
        )
        result = analyzer._analyze_copper(input_data)
        assert result.status == expected_status


class TestContaminationDetection:
    """Test contamination detection."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer()

    def test_no_contamination_detected(self, analyzer, condensate_input_excellent):
        """Test no contamination in clean condensate."""
        is_contaminated, source = analyzer._detect_contamination(condensate_input_excellent)
        assert is_contaminated == False
        assert source is None

    def test_hardness_contamination(self, analyzer):
        """Test hardness contamination detection."""
        input_data = CondensateInput(
            sample_point="process_return",
            ph=8.5,
            specific_conductivity_umho=2.0,
            iron_ppb=50.0,
            hardness_ppm=5.0,  # Contamination
        )
        is_contaminated, source = analyzer._detect_contamination(input_data)
        assert is_contaminated == True
        assert "hardness" in source.lower() or "cooling" in source.lower()

    def test_oil_contamination(self, analyzer):
        """Test oil contamination detection."""
        input_data = CondensateInput(
            sample_point="process_return",
            ph=8.5,
            specific_conductivity_umho=2.0,
            iron_ppb=50.0,
            oil_ppm=2.0,  # Oil contamination
        )
        is_contaminated, source = analyzer._detect_contamination(input_data)
        assert is_contaminated == True
        assert "oil" in source.lower()

    def test_high_conductivity_contamination(self, analyzer):
        """Test high conductivity contamination detection."""
        input_data = CondensateInput(
            sample_point="process_return",
            ph=8.5,
            specific_conductivity_umho=50.0,  # Very high
            iron_ppb=50.0,
        )
        is_contaminated, source = analyzer._detect_contamination(input_data)
        assert is_contaminated == True


class TestAmineAnalysis:
    """Test amine treatment analysis."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer(amine_type=ChemicalType.MORPHOLINE)

    def test_amine_residual_adequate(self, analyzer):
        """Test adequate amine residual detection."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=8.7,
            specific_conductivity_umho=1.0,
            iron_ppb=30.0,
            amine_residual_ppm=5.0,  # At target
            amine_type=ChemicalType.MORPHOLINE,
        )
        result = analyzer._analyze_amine_residual(input_data)
        assert result.status in [WaterQualityStatus.EXCELLENT, WaterQualityStatus.GOOD]

    def test_amine_residual_low(self, analyzer):
        """Test low amine residual detection."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=8.0,
            specific_conductivity_umho=1.0,
            iron_ppb=100.0,
            amine_residual_ppm=1.0,  # Low
            amine_type=ChemicalType.MORPHOLINE,
        )
        result = analyzer._analyze_amine_residual(input_data)
        assert result.status in [WaterQualityStatus.WARNING, WaterQualityStatus.OUT_OF_SPEC]


class TestCorrosionRateEstimation:
    """Test corrosion rate estimation."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer()

    def test_corrosion_rate_from_iron(self, analyzer):
        """Test corrosion rate estimation from iron levels."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=7.5,  # Low pH
            specific_conductivity_umho=1.5,
            iron_ppb=200.0,  # High iron
        )
        rate = analyzer._estimate_corrosion_rate(input_data)
        assert rate is not None
        assert rate > 0

    def test_low_iron_low_corrosion_rate(self, analyzer):
        """Test low iron indicates low corrosion rate."""
        low_iron_input = CondensateInput(
            sample_point="main_return",
            ph=8.7,
            specific_conductivity_umho=0.5,
            iron_ppb=20.0,
        )
        high_iron_input = CondensateInput(
            sample_point="main_return",
            ph=7.5,
            specific_conductivity_umho=1.5,
            iron_ppb=200.0,
        )
        low_rate = analyzer._estimate_corrosion_rate(low_iron_input)
        high_rate = analyzer._estimate_corrosion_rate(high_iron_input)
        assert low_rate < high_rate


class TestCondensateQualityEvaluation:
    """Test overall condensate quality evaluation."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer()

    def test_quality_evaluation_for_return(self, analyzer):
        """Test quality evaluation for condensate return decision."""
        # Good quality - should return
        good_input = CondensateInput(
            sample_point="main_return",
            ph=8.7,
            specific_conductivity_umho=0.5,
            iron_ppb=30.0,
        )
        result = analyzer.analyze(good_input)
        assert result.suitable_for_return == True

        # Poor quality - should not return
        poor_input = CondensateInput(
            sample_point="process_return",
            ph=7.0,
            specific_conductivity_umho=10.0,
            iron_ppb=300.0,
            hardness_ppm=5.0,
        )
        result = analyzer.analyze(poor_input)
        assert result.suitable_for_return == False


class TestRecommendationGeneration:
    """Test recommendation generation."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer(amine_type=ChemicalType.MORPHOLINE)

    def test_recommendations_for_low_ph(self, analyzer):
        """Test recommendations for low pH condensate."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=7.2,  # Low
            specific_conductivity_umho=1.0,
            iron_ppb=150.0,
            amine_residual_ppm=2.0,
        )
        result = analyzer.analyze(input_data)

        rec_text = " ".join(result.recommendations).lower()
        assert "amine" in rec_text or "ph" in rec_text

    def test_recommendations_for_high_iron(self, analyzer):
        """Test recommendations for high iron levels."""
        input_data = CondensateInput(
            sample_point="main_return",
            ph=8.5,
            specific_conductivity_umho=1.0,
            iron_ppb=200.0,
        )
        result = analyzer.analyze(input_data)

        rec_text = " ".join(result.recommendations).lower()
        assert "iron" in rec_text or "corrosion" in rec_text

    def test_recommendations_for_contamination(self, analyzer):
        """Test recommendations for contaminated condensate."""
        input_data = CondensateInput(
            sample_point="process_return",
            ph=8.0,
            specific_conductivity_umho=5.0,
            iron_ppb=100.0,
            hardness_ppm=3.0,
        )
        result = analyzer.analyze(input_data)

        rec_text = " ".join(result.recommendations).lower()
        assert "contamination" in rec_text or "leak" in rec_text or "divert" in rec_text


class TestCalculateAmineRequirement:
    """Test amine requirement calculation function."""

    def test_morpholine_requirement(self):
        """Test morpholine amine requirement calculation."""
        dose = calculate_amine_requirement(
            target_ph=8.7,
            current_ph=8.0,
            amine_type=ChemicalType.MORPHOLINE,
        )
        assert dose > 0

    def test_cyclohexylamine_requirement(self):
        """Test cyclohexylamine requirement calculation."""
        dose = calculate_amine_requirement(
            target_ph=8.7,
            current_ph=8.0,
            amine_type=ChemicalType.CYCLOHEXYLAMINE,
        )
        assert dose > 0

    def test_no_dose_when_ph_adequate(self):
        """Test no additional dose when pH is adequate."""
        dose = calculate_amine_requirement(
            target_ph=8.7,
            current_ph=8.6,  # Close to target
            amine_type=ChemicalType.MORPHOLINE,
        )
        # Should be minimal or zero
        assert dose < 1.0


class TestProvenanceTracking:
    """Test provenance hash calculation."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer()

    def test_provenance_hash_generated(self, analyzer, condensate_input_excellent):
        """Test provenance hash is generated."""
        result = analyzer.analyze(condensate_input_excellent)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_reproducible(self, analyzer, condensate_input_excellent):
        """Test same input produces same provenance hash."""
        hash1 = analyzer._calculate_provenance_hash(condensate_input_excellent)
        hash2 = analyzer._calculate_provenance_hash(condensate_input_excellent)
        assert hash1 == hash2


class TestCondensateConstants:
    """Test condensate constants values."""

    def test_ph_limits_defined(self):
        """Test pH limits are defined."""
        assert CondensateConstants.PH_MIN >= 7.0
        assert CondensateConstants.PH_MAX <= 10.0
        assert CondensateConstants.PH_TARGET > CondensateConstants.PH_MIN

    def test_iron_limits_defined(self):
        """Test iron limits are defined."""
        assert CondensateConstants.IRON_EXCELLENT_PPB > 0
        assert CondensateConstants.IRON_GOOD_PPB > CondensateConstants.IRON_EXCELLENT_PPB
        assert CondensateConstants.IRON_WARNING_PPB > CondensateConstants.IRON_GOOD_PPB
        assert CondensateConstants.IRON_CRITICAL_PPB > CondensateConstants.IRON_WARNING_PPB

    def test_copper_limits_defined(self):
        """Test copper limits are defined."""
        assert CondensateConstants.COPPER_EXCELLENT_PPB > 0
        assert CondensateConstants.COPPER_CRITICAL_PPB > CondensateConstants.COPPER_EXCELLENT_PPB

    def test_contamination_limits_defined(self):
        """Test contamination limits are defined."""
        assert CondensateConstants.HARDNESS_MAX_PPM > 0
        assert CondensateConstants.OIL_MAX_PPM > 0
        assert CondensateConstants.CONDUCTIVITY_MAX_UMHO > 0


class TestPerformance:
    """Performance tests for condensate analyzer."""

    @pytest.fixture
    def analyzer(self):
        return CondensateAnalyzer()

    @pytest.mark.performance
    def test_analysis_performance(self, analyzer, condensate_input_excellent):
        """Test analysis completes within performance target."""
        import time
        start = time.perf_counter()
        result = analyzer.analyze(condensate_input_excellent)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50  # Should complete in < 50ms
        assert result.processing_time_ms < 50

    @pytest.mark.performance
    def test_batch_analysis_performance(self, analyzer, condensate_input_excellent):
        """Test batch analysis maintains throughput."""
        import time
        num_samples = 100

        start = time.perf_counter()
        for _ in range(num_samples):
            analyzer.analyze(condensate_input_excellent)
        elapsed_s = time.perf_counter() - start

        throughput = num_samples / elapsed_s
        assert throughput > 100  # At least 100 samples/second
