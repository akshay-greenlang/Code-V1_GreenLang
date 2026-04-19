"""
GL-016 WATERGUARD Agent - Deaerator Analyzer Tests

Unit tests for DeaeratorAnalyzer covering:
- Oxygen removal efficiency calculation
- CO2 removal efficiency
- Steam consumption analysis
- Vent rate optimization
- Corrosion potential assessment
- Provenance tracking

Author: GL-TestEngineer
Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    DeaerationInput,
    DeaerationOutput,
    WaterQualityStatus,
)

from greenlang.agents.process_heat.gl_016_water_treatment.deaeration import (
    DeaeratorAnalyzer,
    DeaerationConstants,
    calculate_deaerator_capacity,
)


class TestDeaeratorAnalyzerInitialization:
    """Test DeaeratorAnalyzer initialization."""

    def test_default_initialization(self):
        """Test analyzer initializes with defaults."""
        analyzer = DeaeratorAnalyzer()
        assert analyzer.o2_limit_ppb == 7.0

    def test_custom_o2_limit(self):
        """Test analyzer with custom O2 limit."""
        analyzer = DeaeratorAnalyzer(o2_limit_ppb=5.0)
        assert analyzer.o2_limit_ppb == 5.0


class TestDeaerationAnalysis:
    """Test deaerator analysis methods."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    def test_analyze_returns_output(self, analyzer, deaeration_input_excellent):
        """Test analyze returns DeaerationOutput."""
        result = analyzer.analyze(deaeration_input_excellent)
        assert isinstance(result, DeaerationOutput)

    def test_analyze_excellent_performance(self, analyzer, deaeration_input_excellent):
        """Test analysis of excellent performance deaerator."""
        result = analyzer.analyze(deaeration_input_excellent)

        assert result.oxygen_removal_efficiency_pct > 99.0
        assert result.performance_status in [WaterQualityStatus.EXCELLENT, WaterQualityStatus.GOOD]
        assert result.outlet_o2_within_limit == True
        assert result.provenance_hash is not None
        assert result.processing_time_ms > 0

    def test_analyze_poor_performance(self, analyzer, deaeration_input_poor):
        """Test analysis of poor performance deaerator."""
        result = analyzer.analyze(deaeration_input_poor)

        assert result.outlet_o2_within_limit == False
        assert result.performance_status in [WaterQualityStatus.WARNING, WaterQualityStatus.OUT_OF_SPEC, WaterQualityStatus.CRITICAL]
        assert len(result.recommendations) > 0


class TestOxygenRemovalEfficiency:
    """Test oxygen removal efficiency calculations."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    @pytest.mark.parametrize("inlet_o2,outlet_o2,expected_eff", [
        (8000.0, 5.0, 99.94),   # Excellent
        (8000.0, 40.0, 99.5),   # Good
        (8000.0, 80.0, 99.0),   # Minimum acceptable
        (8000.0, 400.0, 95.0),  # Poor
        (5000.0, 5.0, 99.9),    # Lower inlet, excellent removal
    ])
    def test_o2_removal_efficiency(self, analyzer, inlet_o2, outlet_o2, expected_eff):
        """Test O2 removal efficiency calculation."""
        efficiency = analyzer._calculate_o2_removal_efficiency(inlet_o2, outlet_o2)
        assert efficiency == pytest.approx(expected_eff, rel=0.01)

    def test_zero_inlet_o2(self, analyzer):
        """Test zero inlet O2 returns 0% efficiency."""
        efficiency = analyzer._calculate_o2_removal_efficiency(0.0, 5.0)
        assert efficiency == 0.0

    def test_negative_inlet_o2(self, analyzer):
        """Test negative inlet O2 returns 0% efficiency."""
        efficiency = analyzer._calculate_o2_removal_efficiency(-100.0, 5.0)
        assert efficiency == 0.0


class TestCO2RemovalEfficiency:
    """Test CO2 removal efficiency calculations."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    @pytest.mark.parametrize("inlet_co2,outlet_co2,expected_eff", [
        (10.0, 0.5, 95.0),
        (20.0, 2.0, 90.0),
        (15.0, 1.5, 90.0),
    ])
    def test_co2_removal_efficiency(self, analyzer, inlet_co2, outlet_co2, expected_eff):
        """Test CO2 removal efficiency calculation."""
        efficiency = analyzer._calculate_co2_removal_efficiency(inlet_co2, outlet_co2)
        assert efficiency == pytest.approx(expected_eff, rel=0.01)

    def test_co2_efficiency_zero_inlet(self, analyzer):
        """Test CO2 efficiency with zero inlet."""
        efficiency = analyzer._calculate_co2_removal_efficiency(0.0, 0.5)
        assert efficiency == 0.0


class TestPerformanceStatusDetermination:
    """Test performance status determination."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    @pytest.mark.parametrize("o2_eff,outlet_o2,expected_status", [
        (99.95, 3.0, WaterQualityStatus.EXCELLENT),
        (99.9, 5.0, WaterQualityStatus.EXCELLENT),
        (99.7, 5.0, WaterQualityStatus.GOOD),
        (99.0, 7.0, WaterQualityStatus.ACCEPTABLE),
        (98.5, 10.0, WaterQualityStatus.OUT_OF_SPEC),
        (95.0, 25.0, WaterQualityStatus.CRITICAL),
    ])
    def test_performance_status(self, analyzer, o2_eff, outlet_o2, expected_status):
        """Test performance status determination."""
        status = analyzer._determine_performance_status(o2_eff, outlet_o2)
        assert status == expected_status


class TestSaturationTemperature:
    """Test saturation temperature lookup."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    @pytest.mark.parametrize("pressure,expected_temp", [
        (0, 212.0),
        (5, 227.1),
        (10, 240.1),
        (15, 250.3),
    ])
    def test_saturation_temp_at_known_pressure(self, analyzer, pressure, expected_temp):
        """Test saturation temperature at known pressures."""
        temp = analyzer._get_saturation_temperature(pressure)
        assert temp == pytest.approx(expected_temp, rel=0.01)

    def test_saturation_temp_interpolation(self, analyzer):
        """Test saturation temperature interpolation."""
        temp = analyzer._get_saturation_temperature(7.5)
        # Should be between 5 psig (227.1) and 10 psig (240.1)
        assert 227.0 < temp < 241.0

    def test_saturation_temp_below_minimum(self, analyzer):
        """Test saturation temperature below minimum pressure."""
        temp = analyzer._get_saturation_temperature(-5.0)
        assert temp == DeaerationConstants.SATURATION_TEMPS[0]

    def test_saturation_temp_above_maximum(self, analyzer):
        """Test saturation temperature above maximum pressure."""
        temp = analyzer._get_saturation_temperature(30.0)
        assert temp == DeaerationConstants.SATURATION_TEMPS[25]


class TestSteamRequirementCalculation:
    """Test theoretical steam requirement calculation."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    def test_steam_requirement_positive(self, analyzer):
        """Test steam requirement is positive."""
        steam = analyzer._calculate_theoretical_steam(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=150.0,
            saturation_temp_f=227.0,
        )
        assert steam > 0

    def test_steam_requirement_scales_with_flow(self, analyzer):
        """Test steam requirement scales with water flow."""
        steam_low = analyzer._calculate_theoretical_steam(
            water_flow_lb_hr=25000.0,
            inlet_temp_f=150.0,
            saturation_temp_f=227.0,
        )
        steam_high = analyzer._calculate_theoretical_steam(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=150.0,
            saturation_temp_f=227.0,
        )
        assert steam_high == pytest.approx(steam_low * 2, rel=0.01)

    def test_steam_requirement_scales_with_temp_rise(self, analyzer):
        """Test steam requirement scales with temperature rise."""
        steam_small_rise = analyzer._calculate_theoretical_steam(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=200.0,  # Small temp rise
            saturation_temp_f=227.0,
        )
        steam_large_rise = analyzer._calculate_theoretical_steam(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=100.0,  # Large temp rise
            saturation_temp_f=227.0,
        )
        assert steam_large_rise > steam_small_rise

    def test_steam_requirement_zero_temp_rise(self, analyzer):
        """Test zero steam when no temperature rise needed."""
        steam = analyzer._calculate_theoretical_steam(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=230.0,  # Above saturation
            saturation_temp_f=227.0,
        )
        assert steam == 0


class TestVentRateCalculation:
    """Test vent rate calculation."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    def test_vent_rate_positive(self, analyzer):
        """Test vent rate is positive."""
        vent = analyzer._calculate_recommended_vent_rate(
            water_flow_lb_hr=50000.0,
            inlet_o2_ppb=8000.0,
        )
        assert vent > 0

    def test_vent_rate_increases_with_o2(self, analyzer):
        """Test vent rate increases with higher inlet O2."""
        vent_low_o2 = analyzer._calculate_recommended_vent_rate(
            water_flow_lb_hr=50000.0,
            inlet_o2_ppb=200.0,  # Low O2
        )
        vent_high_o2 = analyzer._calculate_recommended_vent_rate(
            water_flow_lb_hr=50000.0,
            inlet_o2_ppb=2000.0,  # High O2
        )
        assert vent_high_o2 > vent_low_o2


class TestVentRateEvaluation:
    """Test vent rate adequacy evaluation."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    def test_vent_rate_adequate(self, analyzer):
        """Test adequate vent rate evaluation."""
        status = analyzer._evaluate_vent_rate(30.0, 25.0)
        assert status == "adequate"

    def test_vent_rate_insufficient(self, analyzer):
        """Test insufficient vent rate evaluation."""
        status = analyzer._evaluate_vent_rate(10.0, 50.0)
        assert status == "insufficient"

    def test_vent_rate_excessive(self, analyzer):
        """Test excessive vent rate evaluation."""
        status = analyzer._evaluate_vent_rate(150.0, 30.0)
        assert status == "excessive"

    def test_vent_rate_unknown(self, analyzer):
        """Test unknown vent rate when None."""
        status = analyzer._evaluate_vent_rate(None, 30.0)
        assert status == "unknown"


class TestCorrosionPotentialAssessment:
    """Test corrosion potential assessment."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    def test_low_corrosion_potential(self, analyzer):
        """Test low corrosion potential assessment."""
        potential = analyzer._assess_corrosion_potential(
            outlet_o2_ppb=3.0,  # Low
            outlet_co2_ppm=0.5,
        )
        assert potential == "low"

    def test_medium_corrosion_potential(self, analyzer):
        """Test medium corrosion potential assessment."""
        potential = analyzer._assess_corrosion_potential(
            outlet_o2_ppb=10.0,  # Above limit
            outlet_co2_ppm=3.0,
        )
        assert potential == "medium"

    def test_high_corrosion_potential(self, analyzer):
        """Test high corrosion potential assessment."""
        potential = analyzer._assess_corrosion_potential(
            outlet_o2_ppb=20.0,  # Very high
            outlet_co2_ppm=8.0,  # Very high
        )
        assert potential == "high"


class TestRecommendationGeneration:
    """Test recommendation generation."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    def test_recommendations_for_high_o2(self, analyzer):
        """Test recommendations for high outlet O2."""
        input_data = DeaerationInput(
            deaerator_pressure_psig=5.0,
            inlet_water_temperature_f=150.0,
            inlet_dissolved_oxygen_ppb=8000.0,
            outlet_dissolved_oxygen_ppb=15.0,  # Above limit
            total_flow_lb_hr=50000.0,
        )
        result = analyzer.analyze(input_data)

        rec_text = " ".join(result.recommendations).lower()
        assert "o2" in rec_text or "oxygen" in rec_text

    def test_recommendations_for_subcooling(self, analyzer):
        """Test recommendations for subcooling."""
        input_data = DeaerationInput(
            deaerator_pressure_psig=5.0,
            deaerator_temperature_f=215.0,  # Below saturation (227)
            inlet_water_temperature_f=150.0,
            inlet_dissolved_oxygen_ppb=8000.0,
            outlet_dissolved_oxygen_ppb=5.0,
            total_flow_lb_hr=50000.0,
        )
        result = analyzer.analyze(input_data)

        rec_text = " ".join(result.recommendations).lower()
        assert "subcooling" in rec_text or "pressure" in rec_text or "steam" in rec_text

    def test_recommendations_for_low_pressure(self, analyzer):
        """Test recommendations for low DA pressure."""
        input_data = DeaerationInput(
            deaerator_pressure_psig=2.0,  # Low
            inlet_water_temperature_f=150.0,
            inlet_dissolved_oxygen_ppb=8000.0,
            outlet_dissolved_oxygen_ppb=5.0,
            total_flow_lb_hr=50000.0,
        )
        result = analyzer.analyze(input_data)

        rec_text = " ".join(result.recommendations).lower()
        assert "pressure" in rec_text


class TestCalculateDeaeratorCapacity:
    """Test deaerator capacity calculation function."""

    def test_capacity_calculation(self):
        """Test deaerator capacity calculation."""
        result = calculate_deaerator_capacity(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=150.0,
            da_pressure_psig=5.0,
        )

        assert "saturation_temp_f" in result
        assert "temp_rise_f" in result
        assert "heat_duty_btu_hr" in result
        assert "steam_required_lb_hr" in result
        assert "storage_volume_gal" in result

    def test_capacity_saturation_temp(self):
        """Test saturation temperature in capacity calculation."""
        result = calculate_deaerator_capacity(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=150.0,
            da_pressure_psig=5.0,
        )
        assert result["saturation_temp_f"] == pytest.approx(227.1, rel=0.01)

    def test_capacity_temp_rise(self):
        """Test temperature rise calculation."""
        result = calculate_deaerator_capacity(
            water_flow_lb_hr=50000.0,
            inlet_temp_f=150.0,
            da_pressure_psig=5.0,
        )
        expected_rise = 227.1 - 150.0
        assert result["temp_rise_f"] == pytest.approx(expected_rise, rel=0.01)


class TestDeaerationConstants:
    """Test deaeration constants values."""

    def test_saturation_temps_defined(self):
        """Test saturation temperatures are defined."""
        assert len(DeaerationConstants.SATURATION_TEMPS) > 0
        assert 0 in DeaerationConstants.SATURATION_TEMPS
        assert 5 in DeaerationConstants.SATURATION_TEMPS
        assert 10 in DeaerationConstants.SATURATION_TEMPS

    def test_o2_removal_targets(self):
        """Test O2 removal targets are defined."""
        assert DeaerationConstants.O2_REMOVAL_EXCELLENT >= 99.9
        assert DeaerationConstants.O2_REMOVAL_GOOD >= 99.5
        assert DeaerationConstants.O2_REMOVAL_MIN >= 99.0

    def test_o2_limits(self):
        """Test O2 limits are defined."""
        assert DeaerationConstants.OUTLET_O2_EXCELLENT <= 7
        assert DeaerationConstants.OUTLET_O2_LIMIT == 7
        assert DeaerationConstants.OUTLET_O2_ACTION > DeaerationConstants.OUTLET_O2_LIMIT

    def test_vent_rate_limits(self):
        """Test vent rate limits are defined."""
        assert DeaerationConstants.MIN_VENT_RATE_PCT > 0
        assert DeaerationConstants.MAX_VENT_RATE_PCT > DeaerationConstants.MIN_VENT_RATE_PCT
        assert DeaerationConstants.TYPICAL_VENT_RATE_PCT > DeaerationConstants.MIN_VENT_RATE_PCT


class TestProvenanceTracking:
    """Test provenance hash calculation."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    def test_provenance_hash_generated(self, analyzer, deaeration_input_excellent):
        """Test provenance hash is generated."""
        result = analyzer.analyze(deaeration_input_excellent)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_reproducible(self, analyzer, deaeration_input_excellent):
        """Test same input produces same provenance hash."""
        hash1 = analyzer._calculate_provenance_hash(deaeration_input_excellent)
        hash2 = analyzer._calculate_provenance_hash(deaeration_input_excellent)
        assert hash1 == hash2


class TestPerformance:
    """Performance tests for deaerator analyzer."""

    @pytest.fixture
    def analyzer(self):
        return DeaeratorAnalyzer()

    @pytest.mark.performance
    def test_analysis_performance(self, analyzer, deaeration_input_excellent):
        """Test analysis completes within performance target."""
        import time
        start = time.perf_counter()
        result = analyzer.analyze(deaeration_input_excellent)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50  # Should complete in < 50ms
        assert result.processing_time_ms < 50

    @pytest.mark.performance
    def test_batch_analysis_performance(self, analyzer, deaeration_input_excellent):
        """Test batch analysis maintains throughput."""
        import time
        num_samples = 100

        start = time.perf_counter()
        for _ in range(num_samples):
            analyzer.analyze(deaeration_input_excellent)
        elapsed_s = time.perf_counter() - start

        throughput = num_samples / elapsed_s
        assert throughput > 100  # At least 100 samples/second
