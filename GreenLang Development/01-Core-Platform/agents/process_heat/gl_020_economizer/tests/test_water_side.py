"""
Unit tests for GL-020 ECONOPULSE Water-Side Fouling Analyzer

Tests water-side scaling and fouling analysis including chemistry compliance.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - ASME Guidelines for Feedwater Chemistry

Zero-Hallucination: All calculations use deterministic formulas.
"""

import pytest
from datetime import datetime, timezone, timedelta

from ..water_side import (
    WaterSideFoulingAnalyzer,
    WaterSideFoulingInput,
    WaterSideFoulingResult,
    WaterChemistryData,
    create_water_side_fouling_analyzer,
    SCALE_THERMAL_CONDUCTIVITIES,
    DEFAULT_SCALE_CONDUCTIVITY,
    CHEMISTRY_LIMITS_HP,
    CHEMISTRY_LIMITS_MP,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analyzer():
    """Create default water-side fouling analyzer."""
    return WaterSideFoulingAnalyzer()


@pytest.fixture
def clean_input():
    """Input representing clean economizer condition."""
    return WaterSideFoulingInput(
        current_dp_psi=5.0,
        design_dp_psi=5.0,
        current_water_flow_lb_hr=80000.0,
        design_water_flow_lb_hr=80000.0,
        water_inlet_temp_f=250.0,
        water_outlet_temp_f=350.0,
        design_fouling_factor=0.001,
    )


@pytest.fixture
def fouled_input():
    """Input representing fouled condition with chemistry issues."""
    return WaterSideFoulingInput(
        current_dp_psi=6.5,
        design_dp_psi=5.0,
        current_water_flow_lb_hr=80000.0,
        design_water_flow_lb_hr=80000.0,
        water_inlet_temp_f=250.0,
        water_outlet_temp_f=340.0,
        design_fouling_factor=0.001,
        chemistry=WaterChemistryData(
            ph=8.5,
            hardness_ppm=1.0,
            silica_ppm=0.05,
            iron_ppm=0.03,
        ),
        pressure_class="high",
    )


@pytest.fixture
def compliant_chemistry():
    """Water chemistry within ASME limits."""
    return WaterChemistryData(
        ph=9.2,
        hardness_ppm=0.3,
        silica_ppm=0.01,
        iron_ppm=0.005,
        copper_ppm=0.002,
        oxygen_ppb=5.0,
    )


@pytest.fixture
def noncompliant_chemistry():
    """Water chemistry exceeding ASME limits."""
    return WaterChemistryData(
        ph=8.0,  # Too low
        hardness_ppm=2.0,  # Too high
        silica_ppm=0.1,  # Too high
        iron_ppm=0.05,  # Too high
        copper_ppm=0.02,  # Too high
        oxygen_ppb=20.0,  # Too high
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestWaterSideFoulingAnalyzerInit:
    """Test analyzer initialization."""

    def test_default_initialization(self, analyzer):
        """Test default analyzer initialization."""
        assert analyzer.scale_conductivity == DEFAULT_SCALE_CONDUCTIVITY

    def test_custom_scale_conductivity(self):
        """Test custom scale thermal conductivity."""
        analyzer = WaterSideFoulingAnalyzer(scale_conductivity=1.5)
        assert analyzer.scale_conductivity == 1.5

    def test_factory_function(self):
        """Test factory function creates analyzer."""
        analyzer = create_water_side_fouling_analyzer(scale_conductivity=1.0)
        assert isinstance(analyzer, WaterSideFoulingAnalyzer)
        assert analyzer.scale_conductivity == 1.0


# =============================================================================
# PRESSURE DROP ANALYSIS TESTS
# =============================================================================

class TestWaterPressureDropAnalysis:
    """Test water-side pressure drop analysis."""

    def test_clean_condition(self, analyzer):
        """Test DP analysis at clean condition."""
        corrected_dp, dp_ratio = analyzer.analyze_pressure_drop(
            current_dp=5.0,
            design_dp=5.0,
            flow_ratio=1.0,
        )

        assert corrected_dp == pytest.approx(5.0, rel=0.01)
        assert dp_ratio == pytest.approx(1.0, rel=0.01)

    def test_fouled_condition(self, analyzer):
        """Test DP analysis at fouled condition."""
        corrected_dp, dp_ratio = analyzer.analyze_pressure_drop(
            current_dp=7.0,
            design_dp=5.0,
            flow_ratio=1.0,
        )

        assert corrected_dp == pytest.approx(7.0, rel=0.01)
        assert dp_ratio == pytest.approx(1.4, rel=0.01)

    def test_flow_correction_turbulent(self, analyzer):
        """Test DP correction for turbulent flow (n=1.85)."""
        # At 80% flow with turbulent exponent 1.85
        flow_ratio = 0.8
        expected_dp_factor = flow_ratio ** 1.85

        corrected_dp, dp_ratio = analyzer.analyze_pressure_drop(
            current_dp=5.0 * expected_dp_factor,
            design_dp=5.0,
            flow_ratio=flow_ratio,
        )

        assert corrected_dp == pytest.approx(5.0, rel=0.02)
        assert dp_ratio == pytest.approx(1.0, rel=0.02)

    def test_low_flow_handling(self, analyzer):
        """Test handling of very low flow ratio."""
        corrected_dp, dp_ratio = analyzer.analyze_pressure_drop(
            current_dp=1.0,
            design_dp=5.0,
            flow_ratio=0.0,  # Edge case
        )

        assert corrected_dp > 0
        assert dp_ratio > 0


# =============================================================================
# FOULING FACTOR CALCULATION TESTS
# =============================================================================

class TestFoulingFactorCalculation:
    """Test fouling factor calculation."""

    def test_clean_condition(self, analyzer):
        """Test fouling factor at clean condition."""
        fouling_factor = analyzer.calculate_fouling_factor(
            current_ua=100000.0,
            design_ua=100000.0,
            heat_transfer_area_ft2=5000.0,
        )

        assert fouling_factor == pytest.approx(0.0, abs=0.0001)

    def test_fouled_condition(self, analyzer):
        """Test fouling factor with degraded UA."""
        fouling_factor = analyzer.calculate_fouling_factor(
            current_ua=80000.0,
            design_ua=100000.0,
            heat_transfer_area_ft2=5000.0,
        )

        # Rf = A * (1/UA_actual - 1/UA_design)
        expected_rf = 5000.0 * (1.0/80000.0 - 1.0/100000.0)
        assert fouling_factor == pytest.approx(expected_rf, rel=0.01)

    def test_zero_current_ua(self, analyzer):
        """Test handling of zero current UA."""
        fouling_factor = analyzer.calculate_fouling_factor(
            current_ua=0.0,
            design_ua=100000.0,
            heat_transfer_area_ft2=5000.0,
        )

        assert fouling_factor == 0.0


# =============================================================================
# SCALE THICKNESS ESTIMATION TESTS
# =============================================================================

class TestScaleThicknessEstimation:
    """Test scale thickness estimation."""

    def test_thickness_calculation(self, analyzer):
        """Test scale thickness calculation."""
        fouling_factor = 0.001

        thickness = analyzer.estimate_scale_thickness(fouling_factor, "mixed")

        # thickness (mils) = Rf * k * 12000
        expected_thickness = fouling_factor * SCALE_THERMAL_CONDUCTIVITIES["mixed"] * 12000
        assert thickness == pytest.approx(expected_thickness, rel=0.01)

    def test_different_scale_types(self, analyzer):
        """Test thickness with different scale compositions."""
        fouling_factor = 0.001

        thickness_caco3 = analyzer.estimate_scale_thickness(fouling_factor, "calcium_carbonate")
        thickness_silica = analyzer.estimate_scale_thickness(fouling_factor, "silica")

        # CaCO3 is more conductive, so thicker layer for same resistance
        assert thickness_caco3 > thickness_silica

    def test_zero_fouling_factor(self, analyzer):
        """Test zero fouling factor gives zero thickness."""
        thickness = analyzer.estimate_scale_thickness(0.0, "mixed")

        assert thickness == 0.0


# =============================================================================
# WATER CHEMISTRY ANALYSIS TESTS
# =============================================================================

class TestWaterChemistryAnalysis:
    """Test water chemistry compliance analysis."""

    def test_compliant_chemistry_high_pressure(self, analyzer, compliant_chemistry):
        """Test compliant chemistry for high-pressure system."""
        compliant, deviations, scale_type = analyzer.analyze_water_chemistry(
            compliant_chemistry,
            pressure_class="high",
        )

        assert compliant is True
        assert len(deviations) == 0
        assert scale_type is None

    def test_noncompliant_chemistry(self, analyzer, noncompliant_chemistry):
        """Test non-compliant chemistry detection."""
        compliant, deviations, scale_type = analyzer.analyze_water_chemistry(
            noncompliant_chemistry,
            pressure_class="high",
        )

        assert compliant is False
        assert len(deviations) > 0

    def test_ph_low_detection(self, analyzer):
        """Test low pH detection."""
        chemistry = WaterChemistryData(ph=8.5)  # Below 9.0

        compliant, deviations, _ = analyzer.analyze_water_chemistry(
            chemistry,
            pressure_class="high",
        )

        assert compliant is False
        assert any("pH low" in d for d in deviations)

    def test_ph_high_detection(self, analyzer):
        """Test high pH detection."""
        chemistry = WaterChemistryData(ph=10.0)  # Above 9.5

        compliant, deviations, _ = analyzer.analyze_water_chemistry(
            chemistry,
            pressure_class="high",
        )

        assert compliant is False
        assert any("pH high" in d for d in deviations)

    def test_hardness_detection(self, analyzer):
        """Test high hardness detection."""
        chemistry = WaterChemistryData(
            ph=9.2,
            hardness_ppm=1.0,  # Above 0.5 limit
        )

        compliant, deviations, scale_type = analyzer.analyze_water_chemistry(
            chemistry,
            pressure_class="high",
        )

        assert compliant is False
        assert any("hardness" in d.lower() for d in deviations)
        assert "calcium_carbonate" in scale_type

    def test_silica_detection(self, analyzer):
        """Test high silica detection."""
        chemistry = WaterChemistryData(
            ph=9.2,
            silica_ppm=0.1,  # Above 0.02 limit
        )

        compliant, deviations, scale_type = analyzer.analyze_water_chemistry(
            chemistry,
            pressure_class="high",
        )

        assert compliant is False
        assert any("silica" in d.lower() for d in deviations)
        assert "silica" in scale_type

    def test_iron_detection(self, analyzer):
        """Test high iron detection."""
        chemistry = WaterChemistryData(
            ph=9.2,
            iron_ppm=0.05,  # Above 0.01 limit
        )

        compliant, deviations, scale_type = analyzer.analyze_water_chemistry(
            chemistry,
            pressure_class="high",
        )

        assert compliant is False
        assert "iron_oxide" in scale_type

    def test_medium_pressure_limits(self, analyzer):
        """Test medium pressure system has higher limits."""
        chemistry = WaterChemistryData(
            ph=9.0,
            hardness_ppm=1.5,  # Above HP limit but below MP limit
        )

        # Should fail HP limits
        compliant_hp, _, _ = analyzer.analyze_water_chemistry(chemistry, "high")
        assert compliant_hp is False

        # Should pass MP limits
        compliant_mp, _, _ = analyzer.analyze_water_chemistry(chemistry, "medium")
        assert compliant_mp is True

    def test_mixed_scale_composition(self, analyzer):
        """Test mixed scale composition detection."""
        chemistry = WaterChemistryData(
            ph=9.2,
            hardness_ppm=1.0,
            silica_ppm=0.1,
            iron_ppm=0.05,
        )

        _, _, scale_type = analyzer.analyze_water_chemistry(
            chemistry,
            pressure_class="high",
        )

        assert scale_type == "mixed"


# =============================================================================
# FOULING SEVERITY TESTS
# =============================================================================

class TestWaterSideFoulingSeverity:
    """Test water-side fouling severity determination."""

    def test_no_fouling(self, analyzer):
        """Test no fouling severity."""
        severity = analyzer.determine_fouling_severity(
            dp_ratio=1.0,
            fouling_factor_ratio=1.0,
            chemistry_deviations=0,
        )
        assert severity == "none"

    def test_light_fouling(self, analyzer):
        """Test light fouling severity."""
        severity = analyzer.determine_fouling_severity(
            dp_ratio=1.15,
            fouling_factor_ratio=1.3,
            chemistry_deviations=0,
        )
        assert severity == "light"

    def test_moderate_fouling(self, analyzer):
        """Test moderate fouling severity."""
        severity = analyzer.determine_fouling_severity(
            dp_ratio=1.25,
            fouling_factor_ratio=1.6,
            chemistry_deviations=1,
        )
        assert severity == "moderate"

    def test_severe_fouling(self, analyzer):
        """Test severe fouling severity."""
        severity = analyzer.determine_fouling_severity(
            dp_ratio=1.45,
            fouling_factor_ratio=2.5,
            chemistry_deviations=2,
        )
        assert severity in ["severe", "critical"]

    def test_chemistry_adds_severity(self, analyzer):
        """Test chemistry deviations add to severity."""
        severity_no_chem = analyzer.determine_fouling_severity(
            dp_ratio=1.2,
            fouling_factor_ratio=1.4,
            chemistry_deviations=0,
        )
        severity_with_chem = analyzer.determine_fouling_severity(
            dp_ratio=1.2,
            fouling_factor_ratio=1.4,
            chemistry_deviations=3,
        )

        severity_order = ["none", "light", "moderate", "severe", "critical"]
        assert severity_order.index(severity_with_chem) >= severity_order.index(severity_no_chem)


# =============================================================================
# FOULING TYPE DETERMINATION TESTS
# =============================================================================

class TestFoulingTypeDetermination:
    """Test fouling type determination."""

    def test_no_fouling(self, analyzer):
        """Test no fouling type."""
        fouling_type = analyzer.determine_fouling_type(
            dp_ratio=1.0,
            fouling_factor_ratio=1.0,
            chemistry_deviations=[],
        )
        assert fouling_type == "none"

    def test_scale_from_chemistry(self, analyzer):
        """Test scale type from chemistry deviations."""
        fouling_type = analyzer.determine_fouling_type(
            dp_ratio=1.3,
            fouling_factor_ratio=1.5,
            chemistry_deviations=["Hardness high: 1.0 > 0.5 ppm (scale risk)"],
        )
        assert fouling_type == "scale"

    def test_corrosion_from_chemistry(self, analyzer):
        """Test corrosion type from chemistry deviations."""
        fouling_type = analyzer.determine_fouling_type(
            dp_ratio=1.2,
            fouling_factor_ratio=1.3,
            chemistry_deviations=["pH low: 8.0 < 9.0 (corrosion risk)"],
        )
        assert fouling_type == "corrosion"

    def test_deposit_from_chemistry(self, analyzer):
        """Test deposit type from chemistry deviations."""
        fouling_type = analyzer.determine_fouling_type(
            dp_ratio=1.3,
            fouling_factor_ratio=1.2,
            chemistry_deviations=["Iron high: 0.05 > 0.01 ppm"],
        )
        assert fouling_type == "deposit"


# =============================================================================
# CLEANING METHOD RECOMMENDATION TESTS
# =============================================================================

class TestCleaningRecommendation:
    """Test cleaning method recommendations."""

    def test_no_cleaning_light_fouling(self, analyzer):
        """Test no cleaning recommended for light fouling."""
        method = analyzer.recommend_cleaning_method(
            fouling_type="scale",
            scale_composition="calcium_carbonate",
            severity="light",
        )
        assert method is None

    def test_acid_cleaning_for_caco3(self, analyzer):
        """Test acid cleaning recommended for CaCO3 scale."""
        method = analyzer.recommend_cleaning_method(
            fouling_type="scale",
            scale_composition="calcium_carbonate",
            severity="moderate",
        )
        assert "acid" in method.lower() or "HCl" in method

    def test_alkaline_cleaning_for_silica(self, analyzer):
        """Test alkaline cleaning for silica scale."""
        method = analyzer.recommend_cleaning_method(
            fouling_type="scale",
            scale_composition="silica",
            severity="moderate",
        )
        assert "alkaline" in method.lower() or "NaOH" in method

    def test_citric_acid_for_iron(self, analyzer):
        """Test citric acid for iron oxide deposits."""
        method = analyzer.recommend_cleaning_method(
            fouling_type="deposit",
            scale_composition="iron_oxide",
            severity="moderate",
        )
        assert "citric" in method.lower()


# =============================================================================
# COMPLETE ANALYSIS TESTS
# =============================================================================

class TestCompleteAnalysis:
    """Test complete water-side fouling analysis."""

    def test_clean_analysis(self, analyzer, clean_input):
        """Test analysis at clean condition."""
        result = analyzer.analyze(clean_input)

        assert isinstance(result, WaterSideFoulingResult)
        assert result.fouling_detected is False
        assert result.fouling_severity == "none"
        assert result.dp_ratio == pytest.approx(1.0, rel=0.01)
        assert result.chemistry_compliant is True

    def test_fouled_analysis(self, analyzer, fouled_input):
        """Test analysis at fouled condition with chemistry issues."""
        result = analyzer.analyze(fouled_input)

        assert result.fouling_detected is True
        assert result.dp_ratio > 1.0
        assert result.chemistry_compliant is False
        assert len(result.chemistry_deviations) > 0

    def test_provenance_hash_included(self, analyzer, clean_input):
        """Test provenance hash is included."""
        result = analyzer.analyze(clean_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16

    def test_calculation_method_recorded(self, analyzer, clean_input):
        """Test calculation method is recorded."""
        result = analyzer.analyze(clean_input)

        assert result.calculation_method == "ASME_PTC_4.1"


# =============================================================================
# CONSTANT VERIFICATION TESTS
# =============================================================================

class TestConstants:
    """Test constant values are reasonable."""

    def test_scale_conductivities_exist(self):
        """Test scale thermal conductivities are defined."""
        assert "calcium_carbonate" in SCALE_THERMAL_CONDUCTIVITIES
        assert "calcium_sulfate" in SCALE_THERMAL_CONDUCTIVITIES
        assert "silica" in SCALE_THERMAL_CONDUCTIVITIES
        assert "iron_oxide" in SCALE_THERMAL_CONDUCTIVITIES
        assert "mixed" in SCALE_THERMAL_CONDUCTIVITIES

    def test_chemistry_limits_reasonable(self):
        """Test chemistry limits are reasonable."""
        # High pressure limits
        assert CHEMISTRY_LIMITS_HP["hardness_ppm"] < 1.0
        assert CHEMISTRY_LIMITS_HP["silica_ppm"] < 0.1
        assert 9.0 <= CHEMISTRY_LIMITS_HP["ph_min"] < CHEMISTRY_LIMITS_HP["ph_max"] <= 10.0

        # Medium pressure limits are higher
        assert CHEMISTRY_LIMITS_MP["hardness_ppm"] > CHEMISTRY_LIMITS_HP["hardness_ppm"]
        assert CHEMISTRY_LIMITS_MP["silica_ppm"] > CHEMISTRY_LIMITS_HP["silica_ppm"]


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestParameterized:
    """Parameterized tests for various conditions."""

    @pytest.mark.parametrize("dp_ratio,expected_severity", [
        (1.0, "none"),
        (1.15, "light"),
        (1.25, "moderate"),
        (1.45, "severe"),
        (1.65, "critical"),
    ])
    def test_severity_by_dp_ratio(self, analyzer, dp_ratio, expected_severity):
        """Test severity by DP ratio."""
        severity = analyzer.determine_fouling_severity(
            dp_ratio=dp_ratio,
            fouling_factor_ratio=1.0,
            chemistry_deviations=0,
        )
        assert severity == expected_severity

    @pytest.mark.parametrize("scale_type,conductivity_range", [
        ("calcium_carbonate", (1.0, 2.0)),
        ("silica", (0.5, 1.0)),
        ("iron_oxide", (2.0, 3.0)),
        ("mixed", (1.0, 1.5)),
    ])
    def test_scale_conductivities_in_range(self, scale_type, conductivity_range):
        """Test scale conductivities are in expected ranges."""
        k = SCALE_THERMAL_CONDUCTIVITIES[scale_type]
        assert conductivity_range[0] <= k <= conductivity_range[1]
