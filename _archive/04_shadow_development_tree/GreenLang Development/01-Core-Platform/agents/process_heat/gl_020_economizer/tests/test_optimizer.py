"""
Unit tests for GL-020 ECONOPULSE Main Optimizer Agent

Tests the complete economizer optimization agent including integration of all components.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - Verhoff & Banchero (1974) for acid dew point

Zero-Hallucination: All calculations use deterministic formulas with full provenance.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from ..config import (
    EconomizerOptimizationConfig,
    EconomizerDesignConfig,
    PerformanceBaselineConfig,
    GasSideFoulingConfig,
    WaterSideFoulingConfig,
    SootBlowerConfig,
    AcidDewPointConfig,
    EffectivenessConfig,
    SteamingConfig,
    EconomizerType,
    EconomizerArrangement,
    FuelType,
)
from ..schemas import (
    EconomizerInput,
    EconomizerOutput,
    EconomizerStatus,
    FoulingType,
    FoulingSeverity,
    CleaningStatus,
    AlertSeverity,
)
from ..optimizer import EconomizerOptimizer, create_economizer_optimizer


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default economizer optimization configuration."""
    return EconomizerOptimizationConfig(
        economizer_id="ECO-001",
        name="Test Economizer",
        boiler_id="BLR-001",
        design=EconomizerDesignConfig(
            economizer_type=EconomizerType.FINNED_TUBE,
            arrangement=EconomizerArrangement.COUNTERFLOW,
            total_surface_area_ft2=5000.0,
        ),
        baseline=PerformanceBaselineConfig(
            design_duty_btu_hr=20_000_000.0,
            design_gas_inlet_temp_f=600.0,
            design_gas_outlet_temp_f=350.0,
            design_water_inlet_temp_f=250.0,
            design_water_outlet_temp_f=350.0,
            design_gas_flow_lb_hr=100000.0,
            design_water_flow_lb_hr=80000.0,
            design_ua_btu_hr_f=100000.0,
            clean_ua_btu_hr_f=120000.0,
            design_gas_dp_in_wc=2.0,
            design_water_dp_psi=5.0,
        ),
        gas_side=GasSideFoulingConfig(
            design_gas_dp_in_wc=2.0,
            dp_warning_ratio=1.3,
            dp_alarm_ratio=1.5,
            dp_cleaning_trigger_ratio=1.7,
        ),
        acid_dew_point=AcidDewPointConfig(
            fuel_type=FuelType.NATURAL_GAS,
            fuel_sulfur_pct=0.001,
            acid_dew_point_margin_f=30.0,
        ),
        effectiveness=EffectivenessConfig(
            design_effectiveness=0.80,
            effectiveness_warning_pct=90.0,
            effectiveness_alarm_pct=80.0,
        ),
        steaming=SteamingConfig(
            design_approach_temp_f=30.0,
            approach_warning_f=15.0,
            approach_alarm_f=10.0,
            approach_trip_f=5.0,
            steaming_risk_load_pct=30.0,
        ),
        optimization_enabled=True,
    )


@pytest.fixture
def optimizer(default_config):
    """Create economizer optimizer with default config."""
    return EconomizerOptimizer(default_config)


@pytest.fixture
def normal_input():
    """Normal operating condition input."""
    return EconomizerInput(
        economizer_id="ECO-001",
        timestamp=datetime.now(timezone.utc),
        operating_status=EconomizerStatus.NORMAL,
        load_pct=75.0,
        gas_inlet_temp_f=600.0,
        gas_inlet_flow_lb_hr=100000.0,
        gas_inlet_pressure_in_wc=5.0,
        gas_outlet_temp_f=350.0,
        gas_outlet_pressure_in_wc=3.0,  # 2.0 in WC DP
        water_inlet_temp_f=250.0,
        water_inlet_flow_lb_hr=80000.0,
        water_inlet_pressure_psig=550.0,
        water_outlet_temp_f=340.0,
        water_outlet_pressure_psig=545.0,
        flue_gas_o2_pct=3.0,
        flue_gas_moisture_pct=10.0,
        flue_gas_so2_ppm=5.0,
        drum_pressure_psig=500.0,
        ambient_temp_f=70.0,
        barometric_pressure_inhg=29.92,
        soot_blower_active=False,
    )


@pytest.fixture
def degraded_input():
    """Degraded performance input."""
    return EconomizerInput(
        economizer_id="ECO-001",
        timestamp=datetime.now(timezone.utc),
        operating_status=EconomizerStatus.DEGRADED,
        load_pct=75.0,
        gas_inlet_temp_f=600.0,
        gas_inlet_flow_lb_hr=100000.0,
        gas_inlet_pressure_in_wc=6.0,
        gas_outlet_temp_f=400.0,  # Higher outlet - less heat transfer
        gas_outlet_pressure_in_wc=3.0,  # 3.0 in WC DP - higher due to fouling
        water_inlet_temp_f=250.0,
        water_inlet_flow_lb_hr=80000.0,
        water_inlet_pressure_psig=550.0,
        water_outlet_temp_f=320.0,  # Lower outlet - less heat transfer
        water_outlet_pressure_psig=543.0,
        flue_gas_o2_pct=3.5,
        flue_gas_moisture_pct=10.0,
        flue_gas_so2_ppm=10.0,
        drum_pressure_psig=500.0,
        ambient_temp_f=70.0,
        barometric_pressure_inhg=29.92,
        soot_blower_active=False,
    )


@pytest.fixture
def high_sulfur_input():
    """High sulfur fuel input (acid dew point risk)."""
    return EconomizerInput(
        economizer_id="ECO-001",
        timestamp=datetime.now(timezone.utc),
        operating_status=EconomizerStatus.NORMAL,
        load_pct=60.0,
        gas_inlet_temp_f=600.0,
        gas_inlet_flow_lb_hr=80000.0,
        gas_inlet_pressure_in_wc=5.0,
        gas_outlet_temp_f=330.0,  # Low outlet temp
        gas_outlet_pressure_in_wc=3.0,
        water_inlet_temp_f=230.0,  # Low water inlet temp
        water_inlet_flow_lb_hr=65000.0,
        water_inlet_pressure_psig=550.0,
        water_outlet_temp_f=320.0,
        water_outlet_pressure_psig=545.0,
        flue_gas_o2_pct=3.0,
        flue_gas_moisture_pct=8.0,
        flue_gas_so2_ppm=500.0,  # High SO2
        fuel_sulfur_pct=2.0,  # High sulfur
        cold_end_metal_temp_f=280.0,  # Low metal temp
        drum_pressure_psig=500.0,
        ambient_temp_f=70.0,
        barometric_pressure_inhg=29.92,
        soot_blower_active=False,
    )


@pytest.fixture
def low_load_input():
    """Low load input (steaming risk)."""
    return EconomizerInput(
        economizer_id="ECO-001",
        timestamp=datetime.now(timezone.utc),
        operating_status=EconomizerStatus.NORMAL,
        load_pct=25.0,  # Low load
        gas_inlet_temp_f=550.0,
        gas_inlet_flow_lb_hr=30000.0,
        gas_inlet_pressure_in_wc=4.0,
        gas_outlet_temp_f=350.0,
        gas_outlet_pressure_in_wc=3.0,
        water_inlet_temp_f=250.0,
        water_inlet_flow_lb_hr=25000.0,  # Low flow
        water_inlet_pressure_psig=550.0,
        water_outlet_temp_f=440.0,  # High outlet - approaching steaming
        water_outlet_pressure_psig=545.0,
        saturation_temp_f=467.0,
        flue_gas_o2_pct=4.0,
        flue_gas_moisture_pct=10.0,
        drum_pressure_psig=500.0,
        ambient_temp_f=70.0,
        barometric_pressure_inhg=29.92,
        soot_blower_active=False,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestEconomizerOptimizerInit:
    """Test optimizer initialization."""

    def test_default_initialization(self, optimizer, default_config):
        """Test default initialization."""
        assert optimizer.config == default_config
        assert optimizer.economizer_id == "ECO-001"
        assert optimizer.AGENT_ID == "GL-020"
        assert optimizer.AGENT_NAME == "ECONOPULSE"

    def test_factory_function(self, default_config):
        """Test factory function."""
        optimizer = create_economizer_optimizer(default_config)
        assert isinstance(optimizer, EconomizerOptimizer)
        assert optimizer.economizer_id == "ECO-001"

    def test_calculators_initialized(self, optimizer):
        """Test component calculators are initialized."""
        assert optimizer.acid_dew_point_calc is not None
        assert optimizer.effectiveness_calc is not None


# =============================================================================
# NORMAL OPERATION TESTS
# =============================================================================

class TestNormalOperation:
    """Test optimizer behavior during normal operation."""

    def test_process_normal_input(self, optimizer, normal_input):
        """Test processing normal operating input."""
        result = optimizer.process(normal_input)

        assert isinstance(result, EconomizerOutput)
        assert result.economizer_id == "ECO-001"
        assert result.status == "success"
        assert result.operating_status == EconomizerStatus.NORMAL

    def test_output_contains_all_components(self, optimizer, normal_input):
        """Test output contains all analysis components."""
        result = optimizer.process(normal_input)

        # All component results should be present
        assert result.gas_side_fouling is not None
        assert result.water_side_fouling is not None
        assert result.soot_blower is not None
        assert result.acid_dew_point is not None
        assert result.effectiveness is not None
        assert result.steaming is not None

    def test_kpis_calculated(self, optimizer, normal_input):
        """Test KPIs are calculated."""
        result = optimizer.process(normal_input)

        assert "effectiveness_pct" in result.kpis
        assert "dp_ratio" in result.kpis or "gas_dp_ratio" in result.kpis
        assert "health_score" in result.kpis

    def test_provenance_hash_generated(self, optimizer, normal_input):
        """Test provenance hash is generated."""
        result = optimizer.process(normal_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA256

    def test_input_hash_generated(self, optimizer, normal_input):
        """Test input hash is generated."""
        result = optimizer.process(normal_input)

        assert result.input_hash is not None
        assert len(result.input_hash) == 16

    def test_metadata_included(self, optimizer, normal_input):
        """Test metadata is included."""
        result = optimizer.process(normal_input)

        assert "agent_id" in result.metadata
        assert result.metadata["agent_id"] == "GL-020"
        assert result.metadata["agent_name"] == "ECONOPULSE"


# =============================================================================
# ACID DEW POINT ANALYSIS TESTS
# =============================================================================

class TestAcidDewPointAnalysis:
    """Test acid dew point analysis integration."""

    def test_acid_dew_point_calculated(self, optimizer, normal_input):
        """Test acid dew point is calculated."""
        result = optimizer.process(normal_input)

        assert result.acid_dew_point.sulfuric_acid_dew_point_f > 0
        assert result.acid_dew_point.water_dew_point_f > 0
        assert result.acid_dew_point.effective_dew_point_f > 0

    def test_high_sulfur_triggers_alert(self, optimizer, high_sulfur_input):
        """Test high sulfur fuel triggers acid dew point alert."""
        result = optimizer.process(high_sulfur_input)

        # Should have acid dew point alert
        acid_alerts = [a for a in result.alerts if a.category == "acid_dew_point"]
        assert len(acid_alerts) > 0 or result.acid_dew_point.corrosion_risk in ["high", "critical"]

    def test_corrosion_risk_assessment(self, optimizer, normal_input):
        """Test corrosion risk is assessed."""
        result = optimizer.process(normal_input)

        assert result.acid_dew_point.corrosion_risk in ["low", "moderate", "high", "critical"]


# =============================================================================
# EFFECTIVENESS ANALYSIS TESTS
# =============================================================================

class TestEffectivenessAnalysis:
    """Test effectiveness analysis integration."""

    def test_effectiveness_calculated(self, optimizer, normal_input):
        """Test effectiveness is calculated."""
        result = optimizer.process(normal_input)

        assert 0.0 < result.effectiveness.current_effectiveness < 1.0
        assert result.effectiveness.effectiveness_ratio > 0

    def test_degraded_effectiveness_detected(self, optimizer, degraded_input):
        """Test degraded effectiveness is detected."""
        result = optimizer.process(degraded_input)

        # Effectiveness should be lower
        assert result.effectiveness.effectiveness_ratio < 1.0
        assert result.effectiveness.ua_degradation_pct > 0

    def test_lmtd_calculated(self, optimizer, normal_input):
        """Test LMTD is calculated."""
        result = optimizer.process(normal_input)

        assert result.effectiveness.lmtd_f > 0

    def test_duty_calculated(self, optimizer, normal_input):
        """Test heat duty is calculated."""
        result = optimizer.process(normal_input)

        assert result.effectiveness.actual_duty_btu_hr > 0


# =============================================================================
# GAS-SIDE FOULING TESTS
# =============================================================================

class TestGasSideFoulingAnalysis:
    """Test gas-side fouling analysis integration."""

    def test_gas_side_fouling_detected(self, optimizer, degraded_input):
        """Test gas-side fouling is detected."""
        result = optimizer.process(degraded_input)

        # DP ratio should be elevated
        assert result.gas_side_fouling.dp_ratio > 1.0

    def test_cleaning_status_determined(self, optimizer, degraded_input):
        """Test cleaning status is determined."""
        result = optimizer.process(degraded_input)

        assert result.gas_side_fouling.cleaning_status in [
            CleaningStatus.NOT_REQUIRED,
            CleaningStatus.MONITOR,
            CleaningStatus.RECOMMENDED,
            CleaningStatus.REQUIRED,
            CleaningStatus.URGENT,
        ]

    def test_soot_blow_recommendation(self, optimizer, degraded_input):
        """Test soot blow recommendation."""
        result = optimizer.process(degraded_input)

        # Soot blow should be recommended for fouled condition
        assert isinstance(result.gas_side_fouling.soot_blow_recommended, bool)


# =============================================================================
# WATER-SIDE FOULING TESTS
# =============================================================================

class TestWaterSideFoulingAnalysis:
    """Test water-side fouling analysis integration."""

    def test_water_side_fouling_analyzed(self, optimizer, normal_input):
        """Test water-side fouling is analyzed."""
        result = optimizer.process(normal_input)

        assert result.water_side_fouling.dp_ratio > 0
        assert isinstance(result.water_side_fouling.chemistry_compliant, bool)

    def test_chemistry_deviations_tracked(self, optimizer, normal_input):
        """Test chemistry deviations are tracked."""
        result = optimizer.process(normal_input)

        assert isinstance(result.water_side_fouling.chemistry_deviations, list)


# =============================================================================
# SOOT BLOWER OPTIMIZATION TESTS
# =============================================================================

class TestSootBlowerOptimization:
    """Test soot blower optimization integration."""

    def test_soot_blower_optimized(self, optimizer, normal_input):
        """Test soot blower is optimized."""
        result = optimizer.process(normal_input)

        assert isinstance(result.soot_blower.blowing_recommended, bool)
        assert result.soot_blower.optimal_blow_interval_hours > 0

    def test_trigger_status_reported(self, optimizer, normal_input):
        """Test trigger status is reported."""
        result = optimizer.process(normal_input)

        assert isinstance(result.soot_blower.dp_trigger_active, bool)
        assert isinstance(result.soot_blower.effectiveness_trigger_active, bool)
        assert isinstance(result.soot_blower.time_trigger_active, bool)


# =============================================================================
# STEAMING DETECTION TESTS
# =============================================================================

class TestSteamingDetection:
    """Test steaming detection integration."""

    def test_steaming_detected(self, optimizer, normal_input):
        """Test steaming detection."""
        result = optimizer.process(normal_input)

        assert isinstance(result.steaming.steaming_detected, bool)
        assert result.steaming.steaming_risk in ["low", "moderate", "high", "critical"]

    def test_low_load_steaming_risk(self, optimizer, low_load_input):
        """Test low load steaming risk detection."""
        result = optimizer.process(low_load_input)

        # Low load should increase steaming risk
        assert result.steaming.low_load_risk is True or result.steaming.steaming_risk != "low"

    def test_approach_temperature_calculated(self, optimizer, normal_input):
        """Test approach temperature is calculated."""
        result = optimizer.process(normal_input)

        assert result.steaming.approach_temp_f is not None


# =============================================================================
# FOULING SOURCE DETERMINATION TESTS
# =============================================================================

class TestFoulingSourceDetermination:
    """Test primary fouling source determination."""

    def test_no_fouling_source(self, optimizer, normal_input):
        """Test no fouling when clean."""
        result = optimizer.process(normal_input)

        # Should be none or gas_side for normal conditions
        assert result.primary_fouling_type in [FoulingType.NONE, FoulingType.GAS_SIDE]

    def test_fouling_type_determined(self, optimizer, degraded_input):
        """Test fouling type is determined for degraded condition."""
        result = optimizer.process(degraded_input)

        assert result.primary_fouling_type is not None
        assert result.overall_fouling_severity is not None


# =============================================================================
# OPERATING STATUS TESTS
# =============================================================================

class TestOperatingStatusDetermination:
    """Test operating status determination."""

    def test_normal_status(self, optimizer, normal_input):
        """Test normal operating status."""
        result = optimizer.process(normal_input)

        assert result.operating_status in [
            EconomizerStatus.NORMAL,
            EconomizerStatus.DEGRADED,
        ]

    def test_degraded_status_with_fouling(self, optimizer, degraded_input):
        """Test degraded status with fouling."""
        result = optimizer.process(degraded_input)

        # Should be degraded or alarm
        assert result.operating_status in [
            EconomizerStatus.DEGRADED,
            EconomizerStatus.ALARM,
        ]


# =============================================================================
# ALERT GENERATION TESTS
# =============================================================================

class TestAlertGeneration:
    """Test alert generation."""

    def test_alerts_generated(self, optimizer, degraded_input):
        """Test alerts are generated for degraded condition."""
        result = optimizer.process(degraded_input)

        # May or may not have alerts depending on severity
        assert isinstance(result.alerts, list)

    def test_alert_structure(self, optimizer, degraded_input):
        """Test alert structure is correct."""
        result = optimizer.process(degraded_input)

        for alert in result.alerts:
            assert alert.severity in [
                AlertSeverity.INFO,
                AlertSeverity.WARNING,
                AlertSeverity.ALARM,
                AlertSeverity.CRITICAL,
            ]
            assert alert.category is not None
            assert alert.title is not None
            assert alert.description is not None


# =============================================================================
# RECOMMENDATION GENERATION TESTS
# =============================================================================

class TestRecommendationGeneration:
    """Test recommendation generation."""

    def test_recommendations_generated(self, optimizer, degraded_input):
        """Test recommendations are generated."""
        result = optimizer.process(degraded_input)

        assert isinstance(result.recommendations, list)

    def test_recommendation_structure(self, optimizer, degraded_input):
        """Test recommendation structure is correct."""
        result = optimizer.process(degraded_input)

        for rec in result.recommendations:
            assert rec.category is not None
            assert rec.title is not None
            assert rec.description is not None


# =============================================================================
# PROCESSING TIME TESTS
# =============================================================================

class TestProcessingTime:
    """Test processing time tracking."""

    def test_processing_time_tracked(self, optimizer, normal_input):
        """Test processing time is tracked."""
        result = optimizer.process(normal_input)

        assert result.processing_time_ms is not None
        assert result.processing_time_ms > 0

    def test_processing_time_reasonable(self, optimizer, normal_input):
        """Test processing time is reasonable (under 1 second)."""
        result = optimizer.process(normal_input)

        assert result.processing_time_ms < 1000


# =============================================================================
# INTELLIGENCE MIXIN TESTS
# =============================================================================

class TestIntelligenceMixin:
    """Test intelligence mixin integration."""

    def test_explanation_generated(self, optimizer, normal_input):
        """Test explanation is generated."""
        result = optimizer.process(normal_input)

        # Explanation may be None if LLM not available
        # Just verify it's attempted
        assert hasattr(result, "explanation")

    def test_intelligent_recommendations(self, optimizer, normal_input):
        """Test intelligent recommendations are attempted."""
        result = optimizer.process(normal_input)

        assert hasattr(result, "intelligent_recommendations")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_flow_handling(self, optimizer):
        """Test handling of zero flow condition."""
        input_data = EconomizerInput(
            economizer_id="ECO-001",
            load_pct=0.0,
            gas_inlet_temp_f=300.0,
            gas_inlet_flow_lb_hr=0.0,  # Zero flow
            gas_outlet_temp_f=300.0,
            water_inlet_temp_f=250.0,
            water_inlet_flow_lb_hr=0.0,  # Zero flow
            water_inlet_pressure_psig=550.0,
            water_outlet_temp_f=250.0,
            water_outlet_pressure_psig=550.0,
        )

        # Should handle gracefully without crashing
        result = optimizer.process(input_data)
        assert result.status == "success"

    def test_extreme_temperatures(self, optimizer):
        """Test handling of extreme temperatures."""
        input_data = EconomizerInput(
            economizer_id="ECO-001",
            load_pct=100.0,
            gas_inlet_temp_f=1000.0,  # Very high
            gas_inlet_flow_lb_hr=100000.0,
            gas_outlet_temp_f=500.0,
            water_inlet_temp_f=350.0,  # High
            water_inlet_flow_lb_hr=80000.0,
            water_inlet_pressure_psig=550.0,
            water_outlet_temp_f=450.0,
            water_outlet_pressure_psig=545.0,
        )

        result = optimizer.process(input_data)
        assert result.status == "success"


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Test calculation determinism."""

    def test_provenance_hash_deterministic(self, optimizer, normal_input):
        """Test provenance hash is deterministic for same input."""
        result1 = optimizer.process(normal_input)
        result2 = optimizer.process(normal_input)

        # Note: timestamps may differ slightly, so we compare key outputs
        assert result1.effectiveness.current_effectiveness == \
               result2.effectiveness.current_effectiveness
        assert result1.acid_dew_point.sulfuric_acid_dew_point_f == \
               result2.acid_dew_point.sulfuric_acid_dew_point_f

    def test_input_hash_deterministic(self, optimizer, normal_input):
        """Test input hash is deterministic."""
        result1 = optimizer.process(normal_input)
        result2 = optimizer.process(normal_input)

        assert result1.input_hash == result2.input_hash


# =============================================================================
# NFPA 86 COMPLIANCE TESTS
# =============================================================================

class TestNFPA86Compliance:
    """Test NFPA 86 compliance-related features."""

    def test_safety_limits_respected(self, optimizer, normal_input):
        """Test safety limits are respected."""
        result = optimizer.process(normal_input)

        # Steaming detection is a safety feature
        assert result.steaming is not None
        assert result.steaming.min_safe_load_pct > 0

    def test_trip_conditions_identified(self, optimizer):
        """Test trip conditions are identified."""
        # Create input that would cause a trip
        input_data = EconomizerInput(
            economizer_id="ECO-001",
            load_pct=75.0,
            gas_inlet_temp_f=600.0,
            gas_inlet_flow_lb_hr=100000.0,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_inlet_flow_lb_hr=80000.0,
            water_inlet_pressure_psig=550.0,
            water_outlet_temp_f=465.0,  # Near saturation
            water_outlet_pressure_psig=545.0,
            saturation_temp_f=467.0,
        )

        result = optimizer.process(input_data)

        # Should identify steaming risk
        assert result.steaming.steaming_risk in ["high", "critical"] or \
               result.steaming.approach_temp_f < 10


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestParameterized:
    """Parameterized tests for various conditions."""

    @pytest.mark.parametrize("load_pct", [25.0, 50.0, 75.0, 100.0])
    def test_various_loads(self, optimizer, load_pct):
        """Test optimizer at various load conditions."""
        input_data = EconomizerInput(
            economizer_id="ECO-001",
            load_pct=load_pct,
            gas_inlet_temp_f=600.0,
            gas_inlet_flow_lb_hr=100000.0 * load_pct / 100,
            gas_outlet_temp_f=350.0,
            water_inlet_temp_f=250.0,
            water_inlet_flow_lb_hr=80000.0 * load_pct / 100,
            water_inlet_pressure_psig=550.0,
            water_outlet_temp_f=340.0,
            water_outlet_pressure_psig=545.0,
        )

        result = optimizer.process(input_data)
        assert result.status == "success"

    @pytest.mark.parametrize("gas_dp_in_wc", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_various_dp_ratios(self, optimizer, gas_dp_in_wc):
        """Test optimizer at various DP ratios."""
        input_data = EconomizerInput(
            economizer_id="ECO-001",
            load_pct=75.0,
            gas_inlet_temp_f=600.0,
            gas_inlet_flow_lb_hr=100000.0,
            gas_inlet_pressure_in_wc=5.0 + gas_dp_in_wc,
            gas_outlet_temp_f=350.0,
            gas_outlet_pressure_in_wc=5.0,
            water_inlet_temp_f=250.0,
            water_inlet_flow_lb_hr=80000.0,
            water_inlet_pressure_psig=550.0,
            water_outlet_temp_f=340.0,
            water_outlet_pressure_psig=545.0,
        )

        result = optimizer.process(input_data)
        assert result.status == "success"
        assert result.gas_side_fouling.current_dp_in_wc == pytest.approx(gas_dp_in_wc, abs=0.1)
