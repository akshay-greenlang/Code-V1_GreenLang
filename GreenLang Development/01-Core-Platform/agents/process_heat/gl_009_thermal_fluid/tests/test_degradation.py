"""
Unit tests for GL-009 THERMALIQ Agent Degradation Monitoring

Tests fluid degradation analysis including viscosity changes, flash point
degradation, acid number, carbon residue, and remaining life estimation.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from greenlang.agents.process_heat.gl_009_thermal_fluid.degradation import (
    DegradationMonitor,
    FluidDegradationLimits,
    ASTM_TAN_LIMITS,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.config import DegradationThresholds
from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ThermalFluidType,
    DegradationLevel,
    DegradationAnalysis,
    FluidLabAnalysis,
    ValidationStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def degradation_monitor():
    """Create degradation monitor instance."""
    return DegradationMonitor(fluid_type=ThermalFluidType.THERMINOL_66)


@pytest.fixture
def degradation_monitor_with_thresholds():
    """Create monitor with custom thresholds."""
    thresholds = DegradationThresholds(
        viscosity_warning_pct=8.0,
        viscosity_critical_pct=20.0,
        flash_point_warning_drop_f=25.0,
        flash_point_critical_drop_f=40.0,
        acid_number_warning=0.15,
        acid_number_critical=0.4,
    )
    return DegradationMonitor(
        fluid_type=ThermalFluidType.THERMINOL_66,
        thresholds=thresholds,
    )


@pytest.fixture
def excellent_fluid_sample():
    """Create lab analysis for excellent condition fluid."""
    return FluidLabAnalysis(
        sample_id="SAMPLE-001",
        sample_date=datetime.now(timezone.utc),
        viscosity_cst_100f=28.0,  # Nominal for Therminol 66
        viscosity_cst_210f=3.5,
        flash_point_f=340.0,  # Nominal
        total_acid_number_mg_koh_g=0.01,
        carbon_residue_pct=0.01,
        moisture_ppm=100,
        color_astm=1.0,
        low_boilers_pct=0.1,
        high_boilers_pct=0.2,
        particulate_ppm=10,
    )


@pytest.fixture
def degraded_fluid_sample():
    """Create lab analysis for degraded condition fluid."""
    return FluidLabAnalysis(
        sample_id="SAMPLE-002",
        sample_date=datetime.now(timezone.utc),
        viscosity_cst_100f=35.0,  # 25% increase
        viscosity_cst_210f=4.5,
        flash_point_f=290.0,  # 50F drop
        total_acid_number_mg_koh_g=0.45,  # High
        carbon_residue_pct=0.25,  # Elevated
        moisture_ppm=800,  # High
        color_astm=6.0,  # Dark
        low_boilers_pct=2.5,  # High
        high_boilers_pct=8.0,  # Very high
        particulate_ppm=150,  # Elevated
    )


@pytest.fixture
def critical_fluid_sample():
    """Create lab analysis for critical condition fluid."""
    return FluidLabAnalysis(
        sample_id="SAMPLE-003",
        sample_date=datetime.now(timezone.utc),
        viscosity_cst_100f=42.0,  # 50% increase
        viscosity_cst_210f=5.5,
        flash_point_f=250.0,  # 90F drop
        total_acid_number_mg_koh_g=0.8,  # Very high
        carbon_residue_pct=0.5,  # High
        moisture_ppm=1500,  # Very high
        color_astm=7.5,  # Very dark
        low_boilers_pct=5.0,  # Critical
        high_boilers_pct=15.0,  # Critical
        particulate_ppm=500,  # Very high
    )


@pytest.fixture
def baseline_analysis():
    """Create baseline (new fluid) analysis."""
    return FluidLabAnalysis(
        sample_id="BASELINE-001",
        sample_date=datetime.now(timezone.utc) - timedelta(days=365),
        viscosity_cst_100f=28.0,
        viscosity_cst_210f=3.5,
        flash_point_f=340.0,
        total_acid_number_mg_koh_g=0.005,
        carbon_residue_pct=0.005,
        moisture_ppm=50,
        color_astm=0.5,
        low_boilers_pct=0.05,
        high_boilers_pct=0.1,
    )


# =============================================================================
# MONITOR INITIALIZATION TESTS
# =============================================================================

class TestDegradationMonitorInit:
    """Tests for DegradationMonitor initialization."""

    def test_default_initialization(self, degradation_monitor):
        """Test monitor initializes with defaults."""
        assert degradation_monitor.fluid_type == ThermalFluidType.THERMINOL_66
        assert degradation_monitor._calculation_count == 0

    def test_custom_thresholds(self, degradation_monitor_with_thresholds):
        """Test monitor with custom thresholds."""
        assert degradation_monitor_with_thresholds.thresholds.viscosity_warning_pct == 8.0
        assert degradation_monitor_with_thresholds.thresholds.acid_number_warning == 0.15

    def test_fluid_limits_loaded(self, degradation_monitor):
        """Test fluid-specific limits are loaded."""
        limits = degradation_monitor.get_fluid_limits()

        assert isinstance(limits, FluidDegradationLimits)
        assert limits.nominal_viscosity_cst_100f > 0
        assert limits.nominal_flash_point_f > 0

    @pytest.mark.parametrize("fluid_type", [
        ThermalFluidType.THERMINOL_66,
        ThermalFluidType.THERMINOL_VP1,
        ThermalFluidType.DOWTHERM_A,
        ThermalFluidType.SYLTHERM_800,
    ])
    def test_all_fluids_have_limits(self, fluid_type):
        """Test all fluids have degradation limits defined."""
        monitor = DegradationMonitor(fluid_type=fluid_type)
        limits = monitor.get_fluid_limits()

        assert limits is not None
        assert limits.nominal_viscosity_cst_100f > 0


# =============================================================================
# DEGRADATION ANALYSIS TESTS
# =============================================================================

class TestDegradationAnalysis:
    """Tests for degradation analysis."""

    def test_analyze_excellent_fluid(
        self, degradation_monitor, excellent_fluid_sample, baseline_analysis
    ):
        """Test analysis of excellent condition fluid."""
        result = degradation_monitor.analyze(excellent_fluid_sample, baseline_analysis)

        assert isinstance(result, DegradationAnalysis)
        assert result.degradation_level == DegradationLevel.EXCELLENT
        assert result.remaining_life_pct > 80
        assert result.replacement_recommended == False

    def test_analyze_degraded_fluid(
        self, degradation_monitor, degraded_fluid_sample, baseline_analysis
    ):
        """Test analysis of degraded condition fluid."""
        result = degradation_monitor.analyze(degraded_fluid_sample, baseline_analysis)

        assert result.degradation_level in [DegradationLevel.FAIR, DegradationLevel.POOR]
        assert result.remaining_life_pct < 60
        assert len(result.issues) > 0

    def test_analyze_critical_fluid(
        self, degradation_monitor, critical_fluid_sample, baseline_analysis
    ):
        """Test analysis of critical condition fluid."""
        result = degradation_monitor.analyze(critical_fluid_sample, baseline_analysis)

        assert result.degradation_level == DegradationLevel.CRITICAL
        assert result.remaining_life_pct < 20
        assert result.replacement_recommended == True
        assert len(result.issues) > 0

    def test_analyze_without_baseline(self, degradation_monitor, excellent_fluid_sample):
        """Test analysis without baseline uses nominal values."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        # Should still work using nominal values
        assert isinstance(result, DegradationAnalysis)
        assert result.degradation_level is not None

    def test_degradation_score_range(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test degradation score is in valid range."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert 0 <= result.degradation_score <= 100

    def test_remaining_life_range(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test remaining life is in valid range."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert 0 <= result.remaining_life_pct <= 100


# =============================================================================
# VISCOSITY ANALYSIS TESTS
# =============================================================================

class TestViscosityAnalysis:
    """Tests for viscosity analysis."""

    def test_viscosity_status_valid(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test viscosity status is valid for good fluid."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert result.viscosity_status == ValidationStatus.VALID

    def test_viscosity_status_warning(self, degradation_monitor, baseline_analysis):
        """Test viscosity status warning for moderate change."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=31.0,  # ~10% increase
            flash_point_f=340.0,
        )

        result = degradation_monitor.analyze(sample, baseline_analysis)

        assert result.viscosity_status in [ValidationStatus.WARNING, ValidationStatus.VALID]

    def test_viscosity_status_invalid(self, degradation_monitor, baseline_analysis):
        """Test viscosity status invalid for large change."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=42.0,  # 50% increase
            flash_point_f=340.0,
        )

        result = degradation_monitor.analyze(sample, baseline_analysis)

        assert result.viscosity_status == ValidationStatus.INVALID
        assert any("viscosity" in issue.lower() for issue in result.issues)

    def test_viscosity_change_calculation(
        self, degradation_monitor, degraded_fluid_sample, baseline_analysis
    ):
        """Test viscosity change calculation."""
        result = degradation_monitor.analyze(degraded_fluid_sample, baseline_analysis)

        # Should have viscosity change in details
        assert "viscosity_change_pct" in result.details
        assert result.details["viscosity_change_pct"] > 0

    @pytest.mark.parametrize("viscosity,expected_status", [
        (28.0, ValidationStatus.VALID),  # Nominal
        (31.0, ValidationStatus.VALID),  # +10%
        (35.0, ValidationStatus.WARNING),  # +25%
        (42.0, ValidationStatus.INVALID),  # +50%
    ])
    def test_viscosity_thresholds(
        self, degradation_monitor, viscosity, expected_status, baseline_analysis
    ):
        """Test viscosity threshold levels."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=viscosity,
            flash_point_f=340.0,
            total_acid_number_mg_koh_g=0.01,
        )

        result = degradation_monitor.analyze(sample, baseline_analysis)

        assert result.viscosity_status == expected_status


# =============================================================================
# FLASH POINT ANALYSIS TESTS
# =============================================================================

class TestFlashPointAnalysis:
    """Tests for flash point analysis."""

    def test_flash_point_status_valid(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test flash point status is valid for good fluid."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert result.flash_point_status == ValidationStatus.VALID

    def test_flash_point_status_warning(self, degradation_monitor, baseline_analysis):
        """Test flash point status warning for moderate drop."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=310.0,  # 30F drop
        )

        result = degradation_monitor.analyze(sample, baseline_analysis)

        assert result.flash_point_status == ValidationStatus.WARNING

    def test_flash_point_status_invalid(self, degradation_monitor, baseline_analysis):
        """Test flash point status invalid for large drop."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=280.0,  # 60F drop
        )

        result = degradation_monitor.analyze(sample, baseline_analysis)

        assert result.flash_point_status == ValidationStatus.INVALID
        assert any("flash point" in issue.lower() for issue in result.issues)

    @pytest.mark.parametrize("flash_point,expected_status", [
        (340.0, ValidationStatus.VALID),  # Nominal
        (320.0, ValidationStatus.VALID),  # -20F
        (305.0, ValidationStatus.WARNING),  # -35F
        (280.0, ValidationStatus.INVALID),  # -60F
    ])
    def test_flash_point_thresholds(
        self, degradation_monitor, flash_point, expected_status, baseline_analysis
    ):
        """Test flash point threshold levels."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=flash_point,
            total_acid_number_mg_koh_g=0.01,
        )

        result = degradation_monitor.analyze(sample, baseline_analysis)

        assert result.flash_point_status == expected_status


# =============================================================================
# ACID NUMBER ANALYSIS TESTS
# =============================================================================

class TestAcidNumberAnalysis:
    """Tests for acid number (TAN) analysis."""

    def test_acid_number_status_valid(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test acid number status is valid for good fluid."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert result.acid_number_status == ValidationStatus.VALID

    def test_acid_number_status_warning(self, degradation_monitor):
        """Test acid number status warning."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            total_acid_number_mg_koh_g=0.25,  # Above warning
        )

        result = degradation_monitor.analyze(sample)

        assert result.acid_number_status == ValidationStatus.WARNING

    def test_acid_number_status_invalid(self, degradation_monitor):
        """Test acid number status invalid."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            total_acid_number_mg_koh_g=0.6,  # Above critical
        )

        result = degradation_monitor.analyze(sample)

        assert result.acid_number_status == ValidationStatus.INVALID
        assert any("acid" in issue.lower() for issue in result.issues)

    @pytest.mark.parametrize("tan,expected_status", [
        (0.01, ValidationStatus.VALID),
        (0.15, ValidationStatus.VALID),
        (0.25, ValidationStatus.WARNING),
        (0.6, ValidationStatus.INVALID),
    ])
    def test_acid_number_thresholds(
        self, degradation_monitor, tan, expected_status
    ):
        """Test acid number threshold levels."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            total_acid_number_mg_koh_g=tan,
        )

        result = degradation_monitor.analyze(sample)

        assert result.acid_number_status == expected_status


# =============================================================================
# BOILERS ANALYSIS TESTS
# =============================================================================

class TestBoilersAnalysis:
    """Tests for low/high boilers analysis."""

    def test_low_boilers_high_indicates_cracking(self, degradation_monitor):
        """Test high low boilers indicates thermal cracking."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            low_boilers_pct=5.0,  # High
        )

        result = degradation_monitor.analyze(sample)

        assert any("low boilers" in issue.lower() or "cracking" in issue.lower()
                  for issue in result.issues)

    def test_high_boilers_high_indicates_polymerization(self, degradation_monitor):
        """Test high high boilers indicates polymerization."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            high_boilers_pct=10.0,  # High
        )

        result = degradation_monitor.analyze(sample)

        assert any("high boilers" in issue.lower() or "polymer" in issue.lower()
                  for issue in result.issues)


# =============================================================================
# MOISTURE ANALYSIS TESTS
# =============================================================================

class TestMoistureAnalysis:
    """Tests for moisture content analysis."""

    def test_moisture_status_valid(self, degradation_monitor, excellent_fluid_sample):
        """Test moisture status valid for dry fluid."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert result.moisture_status == ValidationStatus.VALID

    def test_moisture_status_warning(self, degradation_monitor):
        """Test moisture status warning for elevated moisture."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            moisture_ppm=600,  # Elevated
        )

        result = degradation_monitor.analyze(sample)

        assert result.moisture_status == ValidationStatus.WARNING

    def test_moisture_status_invalid(self, degradation_monitor):
        """Test moisture status invalid for high moisture."""
        sample = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            moisture_ppm=1200,  # Very high
        )

        result = degradation_monitor.analyze(sample)

        assert result.moisture_status == ValidationStatus.INVALID


# =============================================================================
# REMAINING LIFE ESTIMATION TESTS
# =============================================================================

class TestRemainingLifeEstimation:
    """Tests for remaining life estimation."""

    def test_excellent_fluid_high_remaining_life(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test excellent fluid has high remaining life."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert result.remaining_life_pct >= 80

    def test_degraded_fluid_medium_remaining_life(
        self, degradation_monitor, degraded_fluid_sample
    ):
        """Test degraded fluid has medium remaining life."""
        result = degradation_monitor.analyze(degraded_fluid_sample)

        assert 20 <= result.remaining_life_pct <= 60

    def test_critical_fluid_low_remaining_life(
        self, degradation_monitor, critical_fluid_sample
    ):
        """Test critical fluid has low remaining life."""
        result = degradation_monitor.analyze(critical_fluid_sample)

        assert result.remaining_life_pct < 20

    def test_estimated_months_remaining(
        self, degradation_monitor, degraded_fluid_sample
    ):
        """Test estimated months remaining is calculated."""
        result = degradation_monitor.analyze(degraded_fluid_sample)

        assert "estimated_months_remaining" in result.details
        assert result.details["estimated_months_remaining"] >= 0


# =============================================================================
# DEGRADATION LEVEL TESTS
# =============================================================================

class TestDegradationLevel:
    """Tests for degradation level determination."""

    @pytest.mark.parametrize("score,expected_level", [
        (5.0, DegradationLevel.EXCELLENT),
        (15.0, DegradationLevel.GOOD),
        (35.0, DegradationLevel.FAIR),
        (55.0, DegradationLevel.POOR),
        (85.0, DegradationLevel.CRITICAL),
    ])
    def test_degradation_level_from_score(self, degradation_monitor, score, expected_level):
        """Test degradation level determination from score."""
        level = degradation_monitor._score_to_level(score)

        assert level == expected_level

    def test_degradation_score_calculation(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test degradation score is calculated."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert result.degradation_score >= 0
        assert result.degradation_score <= 100


# =============================================================================
# RECOMMENDATIONS TESTS
# =============================================================================

class TestRecommendations:
    """Tests for degradation recommendations."""

    def test_excellent_fluid_no_immediate_action(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test excellent fluid has no immediate action needed."""
        result = degradation_monitor.analyze(excellent_fluid_sample)

        assert result.replacement_recommended == False
        # Should have routine recommendation
        assert len(result.recommendations) > 0
        assert any("routine" in rec.lower() or "continue" in rec.lower()
                  for rec in result.recommendations)

    def test_degraded_fluid_has_action_recommendations(
        self, degradation_monitor, degraded_fluid_sample
    ):
        """Test degraded fluid has action recommendations."""
        result = degradation_monitor.analyze(degraded_fluid_sample)

        assert len(result.recommendations) > 1
        # Should recommend monitoring or treatment
        assert any("monitor" in rec.lower() or "treat" in rec.lower() or
                  "filter" in rec.lower() or "test" in rec.lower()
                  for rec in result.recommendations)

    def test_critical_fluid_replacement_recommended(
        self, degradation_monitor, critical_fluid_sample
    ):
        """Test critical fluid has replacement recommended."""
        result = degradation_monitor.analyze(critical_fluid_sample)

        assert result.replacement_recommended == True
        # Should explicitly recommend replacement
        assert any("replace" in rec.lower() or "change" in rec.lower()
                  for rec in result.recommendations)


# =============================================================================
# TRENDING ANALYSIS TESTS
# =============================================================================

class TestTrendingAnalysis:
    """Tests for trending analysis over multiple samples."""

    @pytest.fixture
    def sample_history(self):
        """Create sample history for trending."""
        base_date = datetime.now(timezone.utc)
        return [
            FluidLabAnalysis(
                sample_id=f"SAMPLE-{i}",
                sample_date=base_date - timedelta(days=180*i),
                viscosity_cst_100f=28.0 + i * 2,  # Increasing viscosity
                flash_point_f=340.0 - i * 10,  # Decreasing flash point
                total_acid_number_mg_koh_g=0.01 + i * 0.05,  # Increasing TAN
            )
            for i in range(5)
        ]

    def test_analyze_trend(self, degradation_monitor, sample_history):
        """Test trend analysis with multiple samples."""
        trend = degradation_monitor.analyze_trend(sample_history)

        assert "viscosity_trend" in trend
        assert "flash_point_trend" in trend
        assert "acid_number_trend" in trend

    def test_viscosity_increasing_trend(self, degradation_monitor, sample_history):
        """Test detection of increasing viscosity trend."""
        trend = degradation_monitor.analyze_trend(sample_history)

        assert trend["viscosity_trend"] == "increasing"

    def test_flash_point_decreasing_trend(self, degradation_monitor, sample_history):
        """Test detection of decreasing flash point trend."""
        trend = degradation_monitor.analyze_trend(sample_history)

        assert trend["flash_point_trend"] == "decreasing"

    def test_trend_extrapolation(self, degradation_monitor, sample_history):
        """Test trend extrapolation for remaining life."""
        trend = degradation_monitor.analyze_trend(sample_history)

        assert "estimated_replacement_date" in trend or "estimated_months_to_limit" in trend


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Tests for calculation counting."""

    def test_calculation_count_increments(
        self, degradation_monitor, excellent_fluid_sample
    ):
        """Test calculation count increments."""
        assert degradation_monitor.calculation_count == 0

        degradation_monitor.analyze(excellent_fluid_sample)
        assert degradation_monitor.calculation_count == 1

        degradation_monitor.analyze(excellent_fluid_sample)
        assert degradation_monitor.calculation_count == 2


# =============================================================================
# ASTM LIMITS TESTS
# =============================================================================

class TestASTMLimits:
    """Tests for ASTM standard limits."""

    def test_astm_tan_limits_defined(self):
        """Test ASTM TAN limits are defined."""
        assert len(ASTM_TAN_LIMITS) > 0

    def test_astm_limits_reasonable(self):
        """Test ASTM limits are reasonable values."""
        for fluid_type, limit in ASTM_TAN_LIMITS.items():
            assert 0.1 <= limit <= 1.0  # Typical TAN limits


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for degradation monitoring."""

    def test_full_analysis_workflow(
        self, degradation_monitor, degraded_fluid_sample, baseline_analysis
    ):
        """Test complete analysis workflow."""
        # Perform analysis
        result = degradation_monitor.analyze(degraded_fluid_sample, baseline_analysis)

        # Verify all outputs present
        assert result.degradation_level is not None
        assert result.remaining_life_pct >= 0
        assert result.degradation_score >= 0
        assert result.viscosity_status is not None
        assert result.flash_point_status is not None
        assert result.acid_number_status is not None
        assert result.moisture_status is not None
        assert len(result.issues) >= 0
        assert len(result.recommendations) >= 0

    def test_deterministic_results(
        self, degradation_monitor, degraded_fluid_sample, baseline_analysis
    ):
        """Test results are deterministic."""
        result1 = degradation_monitor.analyze(degraded_fluid_sample, baseline_analysis)
        result2 = degradation_monitor.analyze(degraded_fluid_sample, baseline_analysis)

        assert result1.degradation_level == result2.degradation_level
        assert result1.degradation_score == result2.degradation_score
        assert result1.remaining_life_pct == result2.remaining_life_pct
