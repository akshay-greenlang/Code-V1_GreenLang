"""
GL-003 Unified Steam System Optimizer - Quality Module Tests

Unit tests for steam quality monitoring per ASME standards.
Target: 85%+ coverage of quality.py

Tests:
    - Dryness fraction calculations
    - TDS/conductivity monitoring
    - Carryover analysis
    - ASME quality limit validation
    - Quality trend analysis
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch


from greenlang.agents.process_heat.gl_003_unified_steam.quality import (
    SteamQualityMonitor,
    QualityLimitCalculator,
    DrynessFractionCalculator,
    CarryoverAnalyzer,
    ASMEQualityLimits,
)
from greenlang.agents.process_heat.gl_003_unified_steam.config import (
    QualityMonitoringConfig,
    SteamQualityStandard,
)
from greenlang.agents.process_heat.gl_003_unified_steam.schemas import (
    SteamQualityReading,
    SteamQualityAnalysis,
    ValidationStatus,
)


# =============================================================================
# ASME QUALITY LIMITS TESTS
# =============================================================================

class TestASMEQualityLimits:
    """Test suite for ASME quality limit constants."""

    def test_tds_limits_by_pressure(self, asme_quality_limits):
        """Test TDS limits vary correctly by pressure."""
        # HP has strictest limits
        assert ASMEQualityLimits.MAX_TDS_PPM_HP == asme_quality_limits["max_tds_hp"]
        assert ASMEQualityLimits.MAX_TDS_PPM_MP == asme_quality_limits["max_tds_mp"]
        assert ASMEQualityLimits.MAX_TDS_PPM_LP == asme_quality_limits["max_tds_lp"]

    def test_dryness_limits(self, asme_quality_limits):
        """Test dryness fraction limits."""
        assert ASMEQualityLimits.MIN_DRYNESS_FRACTION == asme_quality_limits["min_dryness_fraction"]

    def test_conductivity_limit(self, asme_quality_limits):
        """Test cation conductivity limit."""
        assert ASMEQualityLimits.MAX_CATION_CONDUCTIVITY_US_CM == asme_quality_limits["max_cation_conductivity"]

    def test_silica_limit(self, asme_quality_limits):
        """Test silica limit."""
        assert ASMEQualityLimits.MAX_SILICA_PPM == asme_quality_limits["max_silica"]

    def test_dissolved_oxygen_limit(self, asme_quality_limits):
        """Test dissolved oxygen limit."""
        assert ASMEQualityLimits.MAX_DISSOLVED_O2_PPB == asme_quality_limits["max_dissolved_o2"]


# =============================================================================
# DRYNESS FRACTION CALCULATOR TESTS
# =============================================================================

class TestDrynessFractionCalculator:
    """Test suite for DrynessFractionCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create dryness fraction calculator."""
        return DrynessFractionCalculator()

    def test_calculate_from_throttling_calorimeter(self, calculator):
        """Test dryness calculation from throttling calorimeter."""
        # Standard throttling calorimeter test
        dryness = calculator.calculate_from_throttling_calorimeter(
            inlet_pressure_psig=150.0,
            outlet_pressure_psig=0.0,  # Atmospheric
            outlet_temperature_f=250.0,  # Superheated at atmospheric
        )

        # Expected dryness > 0.95 for typical steam
        assert 0.95 < dryness <= 1.0

    def test_calculate_from_separating_calorimeter(self, calculator):
        """Test dryness calculation from separating calorimeter."""
        dryness = calculator.calculate_from_separating_calorimeter(
            mass_steam_collected_lb=9.5,
            mass_water_collected_lb=0.5,
        )

        assert dryness == pytest.approx(0.95, rel=0.01)

    def test_calculate_from_tds_method(self, calculator):
        """Test dryness calculation from TDS method."""
        # If steam TDS = 50 ppm and boiler TDS = 2500 ppm
        # Carryover = 50/2500 = 2%
        # Dryness = 1 - 0.02 = 0.98
        dryness = calculator.calculate_from_tds(
            steam_tds_ppm=50.0,
            boiler_tds_ppm=2500.0,
        )

        assert dryness == pytest.approx(0.98, rel=0.01)

    def test_calculate_from_temperature_drop(self, calculator):
        """Test dryness calculation from temperature drop method."""
        # At 150 psig, saturation = 365.9F
        # If steam temp = 364F, it's slightly wet
        dryness = calculator.calculate_from_temperature_drop(
            pressure_psig=150.0,
            measured_temperature_f=364.0,
            saturation_temperature_f=365.9,
        )

        assert 0.95 < dryness <= 1.0

    @pytest.mark.parametrize("steam_tds,boiler_tds,expected_dryness", [
        (25.0, 2500.0, 0.99),   # 1% carryover -> 99% dryness
        (50.0, 2500.0, 0.98),   # 2% carryover -> 98% dryness
        (125.0, 2500.0, 0.95),  # 5% carryover -> 95% dryness
        (250.0, 2500.0, 0.90),  # 10% carryover -> 90% dryness
    ])
    def test_tds_dryness_relationship(
        self, calculator, steam_tds, boiler_tds, expected_dryness
    ):
        """Test TDS-dryness relationship accuracy."""
        dryness = calculator.calculate_from_tds(steam_tds, boiler_tds)
        assert dryness == pytest.approx(expected_dryness, rel=0.01)

    def test_invalid_tds_values(self, calculator):
        """Test error handling for invalid TDS values."""
        with pytest.raises(ValueError):
            calculator.calculate_from_tds(
                steam_tds_ppm=2600.0,  # Higher than boiler TDS
                boiler_tds_ppm=2500.0,
            )

    def test_dryness_bounds(self, calculator):
        """Test dryness fraction stays within bounds."""
        # Very small carryover
        dryness = calculator.calculate_from_tds(1.0, 2500.0)
        assert dryness <= 1.0

        # High carryover
        dryness = calculator.calculate_from_tds(500.0, 2500.0)
        assert dryness >= 0.0


# =============================================================================
# CARRYOVER ANALYZER TESTS
# =============================================================================

class TestCarryoverAnalyzer:
    """Test suite for CarryoverAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create carryover analyzer."""
        return CarryoverAnalyzer()

    def test_calculate_carryover_percentage(self, analyzer):
        """Test carryover percentage calculation."""
        carryover = analyzer.calculate_carryover_pct(
            steam_tds_ppm=50.0,
            boiler_tds_ppm=2500.0,
        )
        assert carryover == pytest.approx(2.0, rel=0.01)

    def test_identify_carryover_cause_high_tds(self, analyzer):
        """Test carryover cause identification for high TDS."""
        causes = analyzer.identify_carryover_causes(
            steam_tds_ppm=100.0,
            boiler_tds_ppm=2500.0,
            steam_load_pct=60.0,
            drum_level_pct=50.0,
            blowdown_rate_pct=3.0,
        )

        assert len(causes) > 0
        assert any("TDS" in cause for cause in causes)

    def test_identify_carryover_cause_high_drum_level(self, analyzer):
        """Test carryover cause for high drum level."""
        causes = analyzer.identify_carryover_causes(
            steam_tds_ppm=50.0,
            boiler_tds_ppm=2500.0,
            steam_load_pct=60.0,
            drum_level_pct=80.0,  # High level
            blowdown_rate_pct=3.0,
        )

        assert any("drum level" in cause.lower() for cause in causes)

    def test_identify_carryover_cause_high_load(self, analyzer):
        """Test carryover cause for high load."""
        causes = analyzer.identify_carryover_causes(
            steam_tds_ppm=50.0,
            boiler_tds_ppm=2500.0,
            steam_load_pct=95.0,  # Very high load
            drum_level_pct=50.0,
            blowdown_rate_pct=3.0,
        )

        assert any("load" in cause.lower() for cause in causes)

    def test_carryover_recommendations(self, analyzer):
        """Test carryover mitigation recommendations."""
        recommendations = analyzer.get_mitigation_recommendations(
            carryover_pct=5.0,
            steam_tds_ppm=100.0,
            boiler_tds_ppm=2000.0,
        )

        assert len(recommendations) > 0
        # Should recommend increasing blowdown for high TDS
        assert any("blowdown" in rec.lower() for rec in recommendations)


# =============================================================================
# QUALITY LIMIT CALCULATOR TESTS
# =============================================================================

class TestQualityLimitCalculator:
    """Test suite for QualityLimitCalculator."""

    @pytest.fixture
    def calculator(self, quality_config):
        """Create quality limit calculator."""
        return QualityLimitCalculator(config=quality_config)

    def test_get_tds_limit_hp(self, calculator):
        """Test TDS limit for high pressure steam."""
        limit = calculator.get_tds_limit(pressure_psig=600.0)
        assert limit == 2500.0  # HP limit

    def test_get_tds_limit_mp(self, calculator):
        """Test TDS limit for medium pressure steam."""
        limit = calculator.get_tds_limit(pressure_psig=150.0)
        assert limit == 3000.0  # MP limit

    def test_get_tds_limit_lp(self, calculator):
        """Test TDS limit for low pressure steam."""
        limit = calculator.get_tds_limit(pressure_psig=15.0)
        assert limit == 3500.0  # LP limit

    def test_check_tds_within_limit(self, calculator):
        """Test TDS check when within limits."""
        result = calculator.check_tds(
            tds_ppm=20.0,
            pressure_psig=150.0,
        )
        assert result["status"] == ValidationStatus.VALID

    def test_check_tds_warning(self, calculator):
        """Test TDS check when in warning zone."""
        # Warning at 80% of limit (3000 * 0.8 = 2400)
        result = calculator.check_tds(
            tds_ppm=2500.0,  # Between 80-95% of limit
            pressure_psig=150.0,
        )
        assert result["status"] == ValidationStatus.WARNING

    def test_check_tds_exceeded(self, calculator):
        """Test TDS check when limit exceeded."""
        result = calculator.check_tds(
            tds_ppm=3500.0,  # Above MP limit
            pressure_psig=150.0,
        )
        assert result["status"] == ValidationStatus.INVALID

    def test_check_dryness_valid(self, calculator):
        """Test dryness check when valid."""
        result = calculator.check_dryness(dryness_fraction=0.98)
        assert result["status"] == ValidationStatus.VALID

    def test_check_dryness_warning(self, calculator):
        """Test dryness check in warning zone."""
        # Warning threshold at 80% of range between min and target
        result = calculator.check_dryness(dryness_fraction=0.96)
        assert result["status"] == ValidationStatus.WARNING

    def test_check_dryness_invalid(self, calculator):
        """Test dryness check when below minimum."""
        result = calculator.check_dryness(dryness_fraction=0.90)
        assert result["status"] == ValidationStatus.INVALID

    def test_check_conductivity_valid(self, calculator):
        """Test conductivity check when valid."""
        result = calculator.check_conductivity(conductivity_us_cm=0.15)
        assert result["status"] == ValidationStatus.VALID

    def test_check_conductivity_exceeded(self, calculator):
        """Test conductivity check when exceeded."""
        result = calculator.check_conductivity(conductivity_us_cm=0.5)
        assert result["status"] == ValidationStatus.INVALID

    def test_check_silica_valid(self, calculator):
        """Test silica check when valid."""
        result = calculator.check_silica(silica_ppm=0.01)
        assert result["status"] == ValidationStatus.VALID

    def test_check_silica_exceeded(self, calculator):
        """Test silica check when exceeded."""
        result = calculator.check_silica(silica_ppm=0.05)
        assert result["status"] == ValidationStatus.INVALID


# =============================================================================
# STEAM QUALITY MONITOR TESTS
# =============================================================================

class TestSteamQualityMonitor:
    """Test suite for SteamQualityMonitor."""

    @pytest.fixture
    def monitor(self, quality_config):
        """Create steam quality monitor."""
        return SteamQualityMonitor(config=quality_config)

    def test_initialization(self, monitor, quality_config):
        """Test monitor initialization."""
        assert monitor.config == quality_config
        assert monitor.limit_calc is not None
        assert monitor.dryness_calc is not None

    def test_analyze_quality_good(
        self, monitor, steam_quality_reading_good
    ):
        """Test quality analysis for good steam."""
        analysis = monitor.analyze_quality(steam_quality_reading_good)

        assert isinstance(analysis, SteamQualityAnalysis)
        assert analysis.overall_status == ValidationStatus.VALID
        assert analysis.dryness_status == ValidationStatus.VALID
        assert analysis.tds_status == ValidationStatus.VALID
        assert len(analysis.limits_exceeded) == 0

    def test_analyze_quality_poor(
        self, monitor, steam_quality_reading_poor
    ):
        """Test quality analysis for poor steam."""
        analysis = monitor.analyze_quality(steam_quality_reading_poor)

        assert analysis.overall_status == ValidationStatus.INVALID
        assert len(analysis.limits_exceeded) > 0
        assert len(analysis.recommendations) > 0

    def test_analyze_quality_with_boiler_tds(
        self, monitor, steam_quality_reading_good
    ):
        """Test quality analysis includes carryover when boiler TDS provided."""
        analysis = monitor.analyze_quality(
            steam_quality_reading_good,
            boiler_water_tds_ppm=2500.0,
        )

        assert analysis.carryover_pct is not None
        assert analysis.carryover_pct < 5.0  # Good steam should have low carryover

    def test_analyze_quality_provenance(
        self, monitor, steam_quality_reading_good
    ):
        """Test quality analysis includes provenance hash."""
        analysis = monitor.analyze_quality(steam_quality_reading_good)

        assert analysis.provenance_hash is not None
        assert len(analysis.provenance_hash) == 64  # SHA-256

    def test_analyze_quality_deterministic(
        self, monitor, steam_quality_reading_good
    ):
        """Test quality analysis is deterministic."""
        analysis1 = monitor.analyze_quality(steam_quality_reading_good)
        analysis2 = monitor.analyze_quality(steam_quality_reading_good)

        assert analysis1.dryness_status == analysis2.dryness_status
        assert analysis1.tds_status == analysis2.tds_status

    def test_calculate_quality_score(
        self, monitor, steam_quality_reading_good
    ):
        """Test quality score calculation."""
        score = monitor.calculate_quality_score(steam_quality_reading_good)

        # Good quality steam should have high score (>90)
        assert 90 < score <= 100

    def test_calculate_quality_score_poor(
        self, monitor, steam_quality_reading_poor
    ):
        """Test quality score for poor steam."""
        score = monitor.calculate_quality_score(steam_quality_reading_poor)

        # Poor quality should have lower score
        assert score < 90

    def test_trend_analysis(self, monitor):
        """Test quality trend analysis."""
        # Add series of readings with declining dryness
        readings = []
        for i in range(10):
            reading = SteamQualityReading(
                location_id="TEST",
                pressure_psig=150.0,
                temperature_f=366.0,
                dryness_fraction=0.99 - i * 0.005,  # Declining dryness
            )
            readings.append(reading)
            monitor.analyze_quality(reading)

        trend = monitor.get_dryness_trend("TEST")
        assert trend == "declining"

    def test_quality_alert_generation(self, monitor, steam_quality_reading_poor):
        """Test alert generation for poor quality."""
        analysis = monitor.analyze_quality(steam_quality_reading_poor)

        assert len(analysis.alerts) > 0
        # Should have alert for dryness below minimum
        assert any("dryness" in alert.lower() for alert in analysis.alerts)


# =============================================================================
# COMPLIANCE TESTS
# =============================================================================

class TestASMECompliance:
    """Compliance tests for ASME steam quality standards."""

    @pytest.fixture
    def monitor(self, quality_config):
        """Create monitor with ASME config."""
        config = quality_config
        config.standard = SteamQualityStandard.ASME
        return SteamQualityMonitor(config=config)

    @pytest.mark.compliance
    def test_asme_dryness_minimum(self, monitor):
        """Test ASME minimum dryness requirement (95%)."""
        reading = SteamQualityReading(
            location_id="TEST",
            pressure_psig=150.0,
            temperature_f=366.0,
            dryness_fraction=0.94,  # Below ASME 95%
        )

        analysis = monitor.analyze_quality(reading)

        assert analysis.dryness_status == ValidationStatus.INVALID
        assert any("ASME" in str(limit) or "dryness" in str(limit).lower()
                   for limit in analysis.limits_exceeded)

    @pytest.mark.compliance
    def test_asme_tds_limits(self, monitor, asme_quality_limits):
        """Test ASME TDS limits by pressure range."""
        # Test HP limit
        reading_hp = SteamQualityReading(
            location_id="TEST-HP",
            pressure_psig=600.0,
            temperature_f=750.0,
            tds_ppm=3000.0,  # Above HP limit (2500)
        )
        analysis_hp = monitor.analyze_quality(reading_hp)
        assert analysis_hp.tds_status == ValidationStatus.INVALID

    @pytest.mark.compliance
    def test_asme_conductivity_limit(self, monitor):
        """Test ASME cation conductivity limit (0.3 uS/cm)."""
        reading = SteamQualityReading(
            location_id="TEST",
            pressure_psig=150.0,
            temperature_f=366.0,
            cation_conductivity_us_cm=0.4,  # Above 0.3
        )

        analysis = monitor.analyze_quality(reading)

        assert analysis.conductivity_status == ValidationStatus.INVALID

    @pytest.mark.compliance
    def test_asme_silica_limit(self, monitor):
        """Test ASME silica limit (0.02 ppm)."""
        reading = SteamQualityReading(
            location_id="TEST",
            pressure_psig=150.0,
            temperature_f=366.0,
            silica_ppm=0.03,  # Above 0.02
        )

        analysis = monitor.analyze_quality(reading)

        assert analysis.silica_status == ValidationStatus.INVALID


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestQualityPerformance:
    """Performance tests for quality module."""

    @pytest.fixture
    def monitor(self, quality_config):
        """Create steam quality monitor."""
        return SteamQualityMonitor(config=quality_config)

    @pytest.mark.performance
    def test_analysis_speed(self, monitor, steam_quality_reading_good):
        """Test quality analysis speed (<5ms)."""
        import time
        start = time.time()

        for _ in range(100):
            monitor.analyze_quality(steam_quality_reading_good)

        elapsed = time.time() - start
        # 100 analyses in <500ms = <5ms each
        assert elapsed < 0.5

    @pytest.mark.performance
    def test_batch_analysis_throughput(self, monitor):
        """Test batch analysis throughput (>1000/sec)."""
        import time

        readings = [
            SteamQualityReading(
                location_id=f"LOC-{i}",
                pressure_psig=150.0,
                temperature_f=366.0,
                dryness_fraction=0.98,
            )
            for i in range(1000)
        ]

        start = time.time()
        for reading in readings:
            monitor.analyze_quality(reading)
        elapsed = time.time() - start

        throughput = 1000 / elapsed
        assert throughput >= 1000  # At least 1000/sec
