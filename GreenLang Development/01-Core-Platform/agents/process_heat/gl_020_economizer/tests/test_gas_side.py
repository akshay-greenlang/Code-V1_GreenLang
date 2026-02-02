"""
Unit tests for GL-020 ECONOPULSE Gas-Side Fouling Analyzer

Tests gas-side fouling detection through pressure drop and heat transfer analysis.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units

Zero-Hallucination: All calculations use deterministic formulas.
"""

import pytest
from datetime import datetime, timezone, timedelta

from ..gas_side import (
    GasSideFoulingAnalyzer,
    GasSideFoulingInput,
    GasSideFoulingResult,
    create_gas_side_fouling_analyzer,
    ASH_THERMAL_CONDUCTIVITY,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analyzer():
    """Create default gas-side fouling analyzer."""
    return GasSideFoulingAnalyzer()


@pytest.fixture
def analyzer_custom():
    """Create analyzer with custom ash conductivity."""
    return GasSideFoulingAnalyzer(ash_thermal_conductivity=0.15)


@pytest.fixture
def clean_input():
    """Input representing clean economizer condition."""
    return GasSideFoulingInput(
        current_dp_in_wc=2.0,
        design_dp_in_wc=2.0,
        current_gas_flow_lb_hr=100000.0,
        design_gas_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
        gas_outlet_temp_f=350.0,
        current_u_btu_hr_ft2_f=10.0,
        clean_u_btu_hr_ft2_f=10.0,
        design_u_btu_hr_ft2_f=10.0,
    )


@pytest.fixture
def fouled_input():
    """Input representing fouled economizer condition."""
    return GasSideFoulingInput(
        current_dp_in_wc=3.0,
        design_dp_in_wc=2.0,
        current_gas_flow_lb_hr=100000.0,
        design_gas_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
        gas_outlet_temp_f=400.0,  # Higher outlet temp
        current_u_btu_hr_ft2_f=7.5,
        clean_u_btu_hr_ft2_f=10.0,
        design_u_btu_hr_ft2_f=10.0,
    )


@pytest.fixture
def severely_fouled_input():
    """Input representing severely fouled condition."""
    return GasSideFoulingInput(
        current_dp_in_wc=4.0,
        design_dp_in_wc=2.0,
        current_gas_flow_lb_hr=100000.0,
        design_gas_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
        gas_outlet_temp_f=450.0,
        current_u_btu_hr_ft2_f=5.0,
        clean_u_btu_hr_ft2_f=10.0,
        design_u_btu_hr_ft2_f=10.0,
    )


@pytest.fixture
def trending_data():
    """Historical data for trend analysis."""
    base_time = datetime.now(timezone.utc)
    dp_history = [
        (base_time - timedelta(hours=72), 2.0),
        (base_time - timedelta(hours=48), 2.2),
        (base_time - timedelta(hours=24), 2.4),
        (base_time, 2.6),
    ]
    u_history = [
        (base_time - timedelta(hours=72), 10.0),
        (base_time - timedelta(hours=48), 9.5),
        (base_time - timedelta(hours=24), 9.0),
        (base_time, 8.5),
    ]
    return dp_history, u_history


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestGasSideFoulingAnalyzerInit:
    """Test analyzer initialization."""

    def test_default_initialization(self, analyzer):
        """Test default analyzer initialization."""
        assert analyzer.ash_thermal_conductivity == ASH_THERMAL_CONDUCTIVITY

    def test_custom_thermal_conductivity(self, analyzer_custom):
        """Test custom ash thermal conductivity."""
        assert analyzer_custom.ash_thermal_conductivity == 0.15

    def test_factory_function(self):
        """Test factory function creates analyzer."""
        analyzer = create_gas_side_fouling_analyzer(ash_thermal_conductivity=0.12)
        assert isinstance(analyzer, GasSideFoulingAnalyzer)
        assert analyzer.ash_thermal_conductivity == 0.12


# =============================================================================
# PRESSURE DROP ANALYSIS TESTS
# =============================================================================

class TestPressureDropAnalysis:
    """Test pressure drop analysis methods."""

    def test_clean_condition(self, analyzer):
        """Test DP analysis at clean condition."""
        corrected_dp, dp_ratio, dp_deviation = analyzer.analyze_pressure_drop(
            current_dp=2.0,
            design_dp=2.0,
            flow_ratio=1.0,
        )

        assert corrected_dp == pytest.approx(2.0, rel=0.01)
        assert dp_ratio == pytest.approx(1.0, rel=0.01)
        assert dp_deviation == pytest.approx(0.0, rel=0.01)

    def test_fouled_condition(self, analyzer):
        """Test DP analysis at fouled condition."""
        corrected_dp, dp_ratio, dp_deviation = analyzer.analyze_pressure_drop(
            current_dp=3.0,
            design_dp=2.0,
            flow_ratio=1.0,
        )

        assert corrected_dp == pytest.approx(3.0, rel=0.01)
        assert dp_ratio == pytest.approx(1.5, rel=0.01)
        assert dp_deviation == pytest.approx(50.0, rel=0.01)

    def test_flow_correction(self, analyzer):
        """Test DP correction for different flow rates."""
        # At 80% flow, DP should be 64% of full flow DP (DP ~ flow^2)
        corrected_dp, dp_ratio, dp_deviation = analyzer.analyze_pressure_drop(
            current_dp=1.28,  # 2.0 * 0.8^2
            design_dp=2.0,
            flow_ratio=0.8,
        )

        # Corrected DP should be back to design (1.28 / 0.64 = 2.0)
        assert corrected_dp == pytest.approx(2.0, rel=0.01)
        assert dp_ratio == pytest.approx(1.0, rel=0.01)

    def test_high_flow_correction(self, analyzer):
        """Test DP correction at high flow."""
        # At 120% flow, measured DP is higher due to velocity
        corrected_dp, dp_ratio, dp_deviation = analyzer.analyze_pressure_drop(
            current_dp=2.88,  # 2.0 * 1.2^2
            design_dp=2.0,
            flow_ratio=1.2,
        )

        # Corrected DP should be back to design
        assert corrected_dp == pytest.approx(2.0, rel=0.01)
        assert dp_ratio == pytest.approx(1.0, rel=0.01)

    def test_low_flow_ratio_handling(self, analyzer):
        """Test handling of very low flow ratio."""
        corrected_dp, dp_ratio, dp_deviation = analyzer.analyze_pressure_drop(
            current_dp=0.1,
            design_dp=2.0,
            flow_ratio=0.0,  # Edge case
        )

        # Should handle gracefully
        assert corrected_dp > 0
        assert dp_ratio > 0


# =============================================================================
# HEAT TRANSFER DEGRADATION TESTS
# =============================================================================

class TestHeatTransferDegradation:
    """Test heat transfer degradation analysis."""

    def test_clean_condition(self, analyzer):
        """Test U-value analysis at clean condition."""
        fouling_resistance, u_degradation = analyzer.analyze_heat_transfer_degradation(
            current_u=10.0,
            clean_u=10.0,
        )

        assert fouling_resistance == pytest.approx(0.0, abs=0.0001)
        assert u_degradation == pytest.approx(0.0, rel=0.01)

    def test_fouled_condition(self, analyzer):
        """Test U-value degradation with fouling."""
        fouling_resistance, u_degradation = analyzer.analyze_heat_transfer_degradation(
            current_u=8.0,
            clean_u=10.0,
        )

        # U degradation = (10 - 8) / 10 * 100 = 20%
        assert u_degradation == pytest.approx(20.0, rel=0.01)

        # Fouling resistance = 1/8 - 1/10 = 0.025
        expected_rf = (1.0 / 8.0) - (1.0 / 10.0)
        assert fouling_resistance == pytest.approx(expected_rf, rel=0.01)

    def test_severe_fouling(self, analyzer):
        """Test severe U-value degradation."""
        fouling_resistance, u_degradation = analyzer.analyze_heat_transfer_degradation(
            current_u=5.0,
            clean_u=10.0,
        )

        assert u_degradation == pytest.approx(50.0, rel=0.01)
        assert fouling_resistance > 0.05

    def test_fouling_resistance_formula(self, analyzer):
        """Test fouling resistance formula: Rf = 1/U_actual - 1/U_clean."""
        current_u = 7.5
        clean_u = 10.0

        fouling_resistance, _ = analyzer.analyze_heat_transfer_degradation(
            current_u=current_u,
            clean_u=clean_u,
        )

        expected_rf = (1.0 / current_u) - (1.0 / clean_u)
        assert fouling_resistance == pytest.approx(expected_rf, rel=0.001)


# =============================================================================
# FOULING THICKNESS ESTIMATION TESTS
# =============================================================================

class TestFoulingThicknessEstimation:
    """Test fouling layer thickness estimation."""

    def test_thickness_from_resistance(self, analyzer):
        """Test thickness calculation from fouling resistance."""
        fouling_resistance = 0.01  # hr-ft2-F/BTU

        thickness = analyzer.estimate_fouling_thickness(fouling_resistance)

        # thickness (in) = Rf * k * 12
        expected_thickness = fouling_resistance * ASH_THERMAL_CONDUCTIVITY * 12
        assert thickness == pytest.approx(expected_thickness, rel=0.01)

    def test_zero_fouling_resistance(self, analyzer):
        """Test zero fouling resistance gives zero thickness."""
        thickness = analyzer.estimate_fouling_thickness(0.0)

        assert thickness == 0.0

    def test_custom_thermal_conductivity(self, analyzer_custom):
        """Test thickness with custom thermal conductivity."""
        fouling_resistance = 0.01

        thickness = analyzer_custom.estimate_fouling_thickness(fouling_resistance)

        expected_thickness = fouling_resistance * 0.15 * 12
        assert thickness == pytest.approx(expected_thickness, rel=0.01)

    def test_thickness_increases_with_resistance(self, analyzer):
        """Test thickness increases with fouling resistance."""
        thickness_low = analyzer.estimate_fouling_thickness(0.005)
        thickness_high = analyzer.estimate_fouling_thickness(0.020)

        assert thickness_high > thickness_low
        assert thickness_high / thickness_low == pytest.approx(4.0, rel=0.01)


# =============================================================================
# FOULING SOURCE DIFFERENTIATION TESTS
# =============================================================================

class TestFoulingSourceDifferentiation:
    """Test differentiation between gas and water-side fouling."""

    def test_gas_side_fouling_pattern(self, analyzer, trending_data):
        """Test detection of gas-side fouling pattern."""
        dp_history, u_history = trending_data

        fouling_type, confidence, explanation = analyzer.differentiate_fouling_source(
            dp_trend=dp_history,  # Increasing DP
            u_trend=u_history,    # Decreasing U
        )

        assert fouling_type == "gas_side"
        assert confidence > 0.5
        assert "gas" in explanation.lower()

    def test_water_side_fouling_pattern(self, analyzer):
        """Test detection of water-side fouling pattern."""
        base_time = datetime.now(timezone.utc)

        # Stable DP, decreasing U suggests water-side
        dp_history = [
            (base_time - timedelta(hours=72), 2.0),
            (base_time - timedelta(hours=48), 2.0),
            (base_time - timedelta(hours=24), 2.0),
            (base_time, 2.0),
        ]
        u_history = [
            (base_time - timedelta(hours=72), 10.0),
            (base_time - timedelta(hours=48), 9.0),
            (base_time - timedelta(hours=24), 8.0),
            (base_time, 7.0),
        ]

        fouling_type, confidence, explanation = analyzer.differentiate_fouling_source(
            dp_trend=dp_history,
            u_trend=u_history,
        )

        assert fouling_type == "water_side"
        assert confidence > 0.3

    def test_no_fouling_pattern(self, analyzer):
        """Test detection of no fouling pattern."""
        base_time = datetime.now(timezone.utc)

        # Stable DP and U
        dp_history = [
            (base_time - timedelta(hours=72), 2.0),
            (base_time - timedelta(hours=48), 2.0),
            (base_time - timedelta(hours=24), 2.0),
            (base_time, 2.0),
        ]
        u_history = [
            (base_time - timedelta(hours=72), 10.0),
            (base_time - timedelta(hours=48), 10.0),
            (base_time - timedelta(hours=24), 10.0),
            (base_time, 10.0),
        ]

        fouling_type, confidence, explanation = analyzer.differentiate_fouling_source(
            dp_trend=dp_history,
            u_trend=u_history,
        )

        assert fouling_type == "none"

    def test_insufficient_data(self, analyzer):
        """Test handling of insufficient data."""
        base_time = datetime.now(timezone.utc)

        dp_history = [(base_time, 2.0)]  # Only 1 point
        u_history = [(base_time, 10.0)]

        fouling_type, confidence, explanation = analyzer.differentiate_fouling_source(
            dp_trend=dp_history,
            u_trend=u_history,
        )

        assert fouling_type == "unknown"
        assert confidence == 0.0
        assert "insufficient" in explanation.lower()


# =============================================================================
# FOULING SEVERITY TESTS
# =============================================================================

class TestFoulingSeverity:
    """Test fouling severity determination."""

    def test_no_fouling(self, analyzer):
        """Test no fouling severity."""
        severity = analyzer.determine_fouling_severity(dp_ratio=1.0, u_degradation_pct=0.0)
        assert severity == "none"

    def test_light_fouling(self, analyzer):
        """Test light fouling severity."""
        severity = analyzer.determine_fouling_severity(dp_ratio=1.15, u_degradation_pct=7.0)
        assert severity == "light"

    def test_moderate_fouling(self, analyzer):
        """Test moderate fouling severity."""
        severity = analyzer.determine_fouling_severity(dp_ratio=1.35, u_degradation_pct=12.0)
        assert severity == "moderate"

    def test_severe_fouling(self, analyzer):
        """Test severe fouling severity."""
        severity = analyzer.determine_fouling_severity(dp_ratio=1.55, u_degradation_pct=22.0)
        assert severity == "severe"

    def test_critical_fouling(self, analyzer):
        """Test critical fouling severity."""
        severity = analyzer.determine_fouling_severity(dp_ratio=2.0, u_degradation_pct=35.0)
        assert severity == "critical"

    def test_severity_based_on_worse_indicator(self, analyzer):
        """Test severity is based on the worse indicator."""
        # High DP but low U degradation
        severity = analyzer.determine_fouling_severity(dp_ratio=1.6, u_degradation_pct=5.0)
        assert severity == "severe"  # Based on DP

        # Low DP but high U degradation
        severity = analyzer.determine_fouling_severity(dp_ratio=1.05, u_degradation_pct=25.0)
        assert severity == "severe"  # Based on U


# =============================================================================
# CLEANING STATUS TESTS
# =============================================================================

class TestCleaningStatus:
    """Test cleaning status determination."""

    def test_not_required(self, analyzer):
        """Test cleaning not required status."""
        status, soot_blow, hours = analyzer.determine_cleaning_status(
            dp_ratio=1.0,
            u_degradation_pct=0.0,
        )

        assert status == "not_required"
        assert soot_blow is False

    def test_monitor_status(self, analyzer):
        """Test monitor cleaning status."""
        status, soot_blow, hours = analyzer.determine_cleaning_status(
            dp_ratio=1.25,
            u_degradation_pct=6.0,
        )

        assert status == "monitor"
        assert soot_blow is True

    def test_recommended_status(self, analyzer):
        """Test recommended cleaning status."""
        status, soot_blow, hours = analyzer.determine_cleaning_status(
            dp_ratio=1.45,
            u_degradation_pct=16.0,
        )

        assert status == "recommended"
        assert soot_blow is True

    def test_required_status(self, analyzer):
        """Test required cleaning status."""
        status, soot_blow, hours = analyzer.determine_cleaning_status(
            dp_ratio=1.65,
            u_degradation_pct=26.0,
        )

        assert status == "required"

    def test_urgent_status(self, analyzer):
        """Test urgent cleaning status."""
        status, soot_blow, hours = analyzer.determine_cleaning_status(
            dp_ratio=1.85,
            u_degradation_pct=40.0,
        )

        assert status == "urgent"


# =============================================================================
# EFFICIENCY IMPACT TESTS
# =============================================================================

class TestEfficiencyImpact:
    """Test efficiency and fuel waste impact calculations."""

    def test_no_degradation(self, analyzer):
        """Test zero efficiency loss with no degradation."""
        eff_loss, fuel_waste = analyzer.calculate_efficiency_impact(
            u_degradation_pct=0.0,
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=350.0,
        )

        assert eff_loss == pytest.approx(0.0, abs=0.1)

    def test_efficiency_loss_with_degradation(self, analyzer):
        """Test efficiency loss with U degradation."""
        eff_loss, fuel_waste = analyzer.calculate_efficiency_impact(
            u_degradation_pct=20.0,
            gas_inlet_temp_f=600.0,
            gas_outlet_temp_f=400.0,  # Higher outlet temp
            design_outlet_temp_f=350.0,
        )

        # 50F higher outlet temp -> ~1.25% efficiency loss (50/40)
        assert eff_loss > 0
        assert fuel_waste > 0


# =============================================================================
# COMPLETE ANALYSIS TESTS
# =============================================================================

class TestCompleteAnalysis:
    """Test complete gas-side fouling analysis."""

    def test_clean_analysis(self, analyzer, clean_input):
        """Test analysis at clean condition."""
        result = analyzer.analyze(clean_input)

        assert isinstance(result, GasSideFoulingResult)
        assert result.fouling_detected is False
        assert result.fouling_severity == "none"
        assert result.dp_ratio == pytest.approx(1.0, rel=0.01)
        assert result.cleaning_status == "not_required"

    def test_fouled_analysis(self, analyzer, fouled_input):
        """Test analysis at fouled condition."""
        result = analyzer.analyze(fouled_input)

        assert result.fouling_detected is True
        assert result.fouling_severity in ["light", "moderate", "severe"]
        assert result.dp_ratio > 1.0
        assert result.soot_blow_recommended is True

    def test_severely_fouled_analysis(self, analyzer, severely_fouled_input):
        """Test analysis at severely fouled condition."""
        result = analyzer.analyze(severely_fouled_input)

        assert result.fouling_detected is True
        assert result.fouling_severity in ["severe", "critical"]
        assert result.dp_ratio > 1.5
        assert result.cleaning_status in ["required", "urgent"]

    def test_provenance_hash_included(self, analyzer, clean_input):
        """Test provenance hash is included in result."""
        result = analyzer.analyze(clean_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16

    def test_calculation_method_recorded(self, analyzer, clean_input):
        """Test calculation method is recorded."""
        result = analyzer.analyze(clean_input)

        assert result.calculation_method == "ASME_PTC_4.3"


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestParameterized:
    """Parameterized tests for various conditions."""

    @pytest.mark.parametrize("dp_ratio,expected_severity", [
        (1.0, "none"),
        (1.15, "light"),
        (1.35, "moderate"),
        (1.55, "severe"),
        (1.85, "critical"),
    ])
    def test_severity_by_dp_ratio(self, analyzer, dp_ratio, expected_severity):
        """Test severity determination by DP ratio."""
        severity = analyzer.determine_fouling_severity(dp_ratio, u_degradation_pct=0.0)
        assert severity == expected_severity

    @pytest.mark.parametrize("u_degradation,expected_severity", [
        (0.0, "none"),
        (7.0, "light"),
        (12.0, "moderate"),
        (22.0, "severe"),
        (35.0, "critical"),
    ])
    def test_severity_by_u_degradation(self, analyzer, u_degradation, expected_severity):
        """Test severity determination by U degradation."""
        severity = analyzer.determine_fouling_severity(dp_ratio=1.0, u_degradation_pct=u_degradation)
        assert severity == expected_severity

    @pytest.mark.parametrize("flow_ratio", [0.6, 0.8, 1.0, 1.2, 1.4])
    def test_flow_correction_range(self, analyzer, flow_ratio):
        """Test flow correction across range."""
        # DP at the flow ratio
        measured_dp = 2.0 * (flow_ratio ** 2)

        corrected_dp, dp_ratio, _ = analyzer.analyze_pressure_drop(
            current_dp=measured_dp,
            design_dp=2.0,
            flow_ratio=flow_ratio,
        )

        # Corrected should be back to design
        assert corrected_dp == pytest.approx(2.0, rel=0.01)
        assert dp_ratio == pytest.approx(1.0, rel=0.01)
