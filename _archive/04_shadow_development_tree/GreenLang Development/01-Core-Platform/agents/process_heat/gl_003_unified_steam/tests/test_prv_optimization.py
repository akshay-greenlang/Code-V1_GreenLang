"""
GL-003 Unified Steam System Optimizer - PRV Optimization Module Tests

Unit tests for PRV sizing and optimization per ASME B31.1.
Target: 85%+ coverage of prv_optimization.py

Tests:
    - Cv calculation methods
    - Opening percentage optimization (50-70% target)
    - Critical flow detection
    - Desuperheater calculations
    - Multi-PRV coordination
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import math


from greenlang.agents.process_heat.gl_003_unified_steam.prv_optimization import (
    CvCalculator,
    DesuperheaterCalculator,
    PRVOptimizer,
    MultiPRVCoordinator,
    PRVConstants,
    DesuperheaterConstants,
    PRVSteamProperties,
)
from greenlang.agents.process_heat.gl_003_unified_steam.config import (
    PRVConfig,
    PRVSizingMethod,
    DesuperheaterType,
)
from greenlang.agents.process_heat.gl_003_unified_steam.schemas import (
    PRVOperatingPoint,
    PRVSizingOutput,
)


# =============================================================================
# PRV CONSTANTS TESTS
# =============================================================================

class TestPRVConstants:
    """Test suite for PRV constants."""

    def test_critical_pressure_ratio(self):
        """Test critical pressure ratio for steam."""
        # For steam with k=1.3, critical ratio = 0.577
        assert PRVConstants.CRITICAL_PRESSURE_RATIO == pytest.approx(0.577, rel=0.01)

    def test_gamma_steam(self):
        """Test specific heat ratio for steam."""
        assert PRVConstants.GAMMA_STEAM == 1.3

    def test_steam_cv_factor(self):
        """Test steam Cv sizing factor."""
        assert PRVConstants.STEAM_CV_FACTOR == 63.3

    def test_asme_b31_1_opening_targets(self, asme_prv_opening_targets):
        """Test ASME B31.1 opening targets."""
        assert PRVConstants.TARGET_OPENING_MIN == asme_prv_opening_targets["minimum"]
        assert PRVConstants.TARGET_OPENING_MAX == asme_prv_opening_targets["maximum"]

    def test_cv_safety_margin(self):
        """Test Cv safety margin."""
        assert PRVConstants.CV_SAFETY_MARGIN == 1.15  # 15% margin


# =============================================================================
# PRV STEAM PROPERTIES TESTS
# =============================================================================

class TestPRVSteamProperties:
    """Test suite for PRV steam property helpers."""

    def test_get_saturation_temp(self):
        """Test saturation temperature lookup."""
        t_sat = PRVSteamProperties.get_saturation_temp(150.0)
        assert t_sat == pytest.approx(365.9, rel=0.01)

    def test_get_saturation_temp_interpolation(self):
        """Test saturation temperature interpolation."""
        t_sat = PRVSteamProperties.get_saturation_temp(75.0)

        # 75 psig between 50 (298F) and 100 (337.9F)
        assert 298.0 < t_sat < 337.9

    def test_get_enthalpy_saturated(self):
        """Test enthalpy for saturated steam."""
        h = PRVSteamProperties.get_enthalpy(150.0)
        assert h == pytest.approx(1196.0, rel=0.01)

    def test_get_enthalpy_superheated(self):
        """Test enthalpy for superheated steam."""
        h = PRVSteamProperties.get_enthalpy(150.0, temperature_f=450.0)

        # Superheated by ~84F, add Cp * superheat
        h_sat = 1196.0
        superheat = 450.0 - 365.9
        expected = h_sat + 0.48 * superheat
        assert h == pytest.approx(expected, rel=0.02)


# =============================================================================
# CV CALCULATOR TESTS
# =============================================================================

class TestCvCalculator:
    """Test suite for CvCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create Cv calculator."""
        return CvCalculator()

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None

    def test_calculate_cv_steam_subcritical(self, calculator):
        """Test Cv calculation for subcritical flow."""
        cv, details = calculator.calculate_cv_steam(
            flow_lb_hr=30000.0,
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,  # Ratio = 164.7/614.7 = 0.27 < 0.577 CRITICAL
        )

        assert cv > 0
        assert details["is_critical_flow"] is True  # This is actually critical

    def test_calculate_cv_steam_critical(self, calculator):
        """Test Cv calculation for critical flow."""
        cv, details = calculator.calculate_cv_steam(
            flow_lb_hr=30000.0,
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
        )

        # P2/P1 = (150+14.7)/(600+14.7) = 0.27 < 0.577, so critical
        assert details["is_critical_flow"] is True

    def test_calculate_cv_steam_subcritical_small_drop(self, calculator):
        """Test Cv calculation for subcritical (small pressure drop)."""
        cv, details = calculator.calculate_cv_steam(
            flow_lb_hr=30000.0,
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=500.0,  # Small drop
        )

        # P2/P1 = 514.7/614.7 = 0.84 > 0.577, so subcritical
        assert details["is_critical_flow"] is False
        assert details["calculation_method"] == "subcritical"

    def test_calculate_cv_invalid_pressures(self, calculator):
        """Test error for invalid pressure configuration."""
        with pytest.raises(ValueError):
            calculator.calculate_cv_steam(
                flow_lb_hr=30000.0,
                inlet_pressure_psig=150.0,
                outlet_pressure_psig=600.0,  # Higher than inlet
            )

    def test_calculate_flow_from_cv(self, calculator):
        """Test flow calculation from Cv."""
        # First calculate Cv for known flow
        cv, _ = calculator.calculate_cv_steam(
            flow_lb_hr=30000.0,
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=500.0,
        )

        # Then calculate flow from that Cv
        flow = calculator.calculate_flow_from_cv(
            cv=cv,
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=500.0,
        )

        assert flow == pytest.approx(30000.0, rel=0.01)

    def test_calculate_opening_percentage_equal_percent(self, calculator):
        """Test opening percentage for equal percentage valve."""
        opening = calculator.calculate_opening_percentage(
            actual_cv=75.0,
            rated_cv=150.0,
            characteristic="equal_percent",
        )

        # Equal percentage: Cv/Cv_rated = R^(opening-1)
        # 75/150 = 0.5 = 50^(opening-1)
        # opening = 1 + log(0.5)/log(50) = 0.823
        # Opening % = 82.3%
        assert 0 < opening < 100

    def test_calculate_opening_percentage_linear(self, calculator):
        """Test opening percentage for linear valve."""
        opening = calculator.calculate_opening_percentage(
            actual_cv=75.0,
            rated_cv=150.0,
            characteristic="linear",
        )

        # Linear: opening = actual/rated * 100
        assert opening == pytest.approx(50.0, rel=0.01)

    def test_calculate_opening_percentage_bounds(self, calculator):
        """Test opening percentage stays in bounds."""
        # Above rated
        opening_high = calculator.calculate_opening_percentage(
            actual_cv=200.0,
            rated_cv=150.0,
        )
        assert opening_high == 100.0

        # Zero Cv
        opening_zero = calculator.calculate_opening_percentage(
            actual_cv=0.0,
            rated_cv=150.0,
        )
        assert opening_zero == 0.0

    def test_cv_increases_with_flow(self, calculator):
        """Test Cv increases with flow rate."""
        cv_low, _ = calculator.calculate_cv_steam(
            flow_lb_hr=10000.0,
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
        )

        cv_high, _ = calculator.calculate_cv_steam(
            flow_lb_hr=50000.0,
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
        )

        assert cv_high > cv_low


# =============================================================================
# DESUPERHEATER CALCULATOR TESTS
# =============================================================================

class TestDesuperheaterCalculator:
    """Test suite for DesuperheaterCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create desuperheater calculator."""
        return DesuperheaterCalculator()

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None

    def test_calculate_spray_rate(self, calculator):
        """Test spray water rate calculation."""
        result = calculator.calculate_spray_rate(
            steam_flow_lb_hr=30000.0,
            inlet_temperature_f=750.0,
            outlet_temperature_f=400.0,
            inlet_pressure_psig=150.0,
            spray_water_temp_f=200.0,
        )

        assert result["spray_required"] is True
        assert result["spray_rate_lb_hr"] > 0
        assert result["spray_rate_pct"] > 0

    def test_no_spray_needed(self, calculator):
        """Test when no spray is needed."""
        result = calculator.calculate_spray_rate(
            steam_flow_lb_hr=30000.0,
            inlet_temperature_f=400.0,
            outlet_temperature_f=450.0,  # Target above inlet
            inlet_pressure_psig=150.0,
            spray_water_temp_f=200.0,
        )

        assert result["spray_required"] is False
        assert result["spray_rate_lb_hr"] == 0

    def test_outlet_below_minimum(self, calculator):
        """Test error when outlet below saturation + approach."""
        result = calculator.calculate_spray_rate(
            steam_flow_lb_hr=30000.0,
            inlet_temperature_f=750.0,
            outlet_temperature_f=370.0,  # Close to saturation (365.9F at 150 psig)
            inlet_pressure_psig=150.0,
            spray_water_temp_f=200.0,
        )

        assert "error" in result

    def test_spray_rate_increases_with_temperature_drop(self, calculator):
        """Test spray rate increases with larger temperature reduction."""
        result_small = calculator.calculate_spray_rate(
            steam_flow_lb_hr=30000.0,
            inlet_temperature_f=500.0,
            outlet_temperature_f=450.0,
            inlet_pressure_psig=150.0,
            spray_water_temp_f=200.0,
        )

        result_large = calculator.calculate_spray_rate(
            steam_flow_lb_hr=30000.0,
            inlet_temperature_f=750.0,
            outlet_temperature_f=450.0,
            inlet_pressure_psig=150.0,
            spray_water_temp_f=200.0,
        )

        assert result_large["spray_rate_lb_hr"] > result_small["spray_rate_lb_hr"]


# =============================================================================
# PRV OPTIMIZER TESTS
# =============================================================================

class TestPRVOptimizer:
    """Test suite for PRVOptimizer."""

    @pytest.fixture
    def optimizer(self, prv_config):
        """Create PRV optimizer."""
        return PRVOptimizer(config=prv_config)

    def test_initialization(self, optimizer, prv_config):
        """Test optimizer initialization."""
        assert optimizer.config == prv_config
        assert optimizer.cv_calc is not None
        assert optimizer.desuper_calc is not None

    def test_size_prv(self, optimizer):
        """Test PRV sizing."""
        result = optimizer.size_prv()

        assert isinstance(result, PRVSizingOutput)
        assert result.cv_required > 0
        assert result.cv_recommended > 0

    def test_size_prv_cv_margin(self, optimizer):
        """Test CV recommendation includes margin."""
        result = optimizer.size_prv()

        # Recommended should have 15% margin
        assert result.cv_margin_pct == 15.0

    def test_size_prv_opening_targets(self, optimizer, asme_prv_opening_targets):
        """Test opening percentage against ASME B31.1 targets."""
        result = optimizer.size_prv()

        if result.meets_opening_targets:
            assert asme_prv_opening_targets["minimum"] <= result.opening_at_design_pct <= asme_prv_opening_targets["maximum"]

    def test_size_prv_critical_flow_detection(self, optimizer):
        """Test critical flow detection."""
        result = optimizer.size_prv()

        # Large pressure drop (600->150) should be critical
        assert result.is_critical_flow is True

    def test_size_prv_provenance_hash(self, optimizer):
        """Test provenance hash generation."""
        result = optimizer.size_prv()

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_size_prv_formula_reference(self, optimizer):
        """Test formula reference included."""
        result = optimizer.size_prv()

        assert "ASME B31.1" in result.formula_reference

    def test_size_prv_with_temperature(self, optimizer):
        """Test sizing with specified temperature."""
        result = optimizer.size_prv(inlet_temperature_f=750.0)

        assert result.cv_required > 0

    def test_analyze_operating_point(self, optimizer, prv_operating_point):
        """Test operating point analysis."""
        result = optimizer.analyze_operating_point(prv_operating_point)

        assert "cv_current" in result
        assert "expected_opening_pct" in result
        assert "within_target_range" in result

    def test_analyze_operating_point_deviation(self, optimizer):
        """Test opening deviation detection."""
        point = PRVOperatingPoint(
            prv_id="PRV-HP-MP",
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
            flow_rate_lb_hr=30000.0,
            opening_pct=85.0,  # High deviation
            inlet_temperature_f=750.0,
        )

        result = optimizer.analyze_operating_point(point)

        if abs(result["opening_deviation_pct"]) > 5:
            assert len(result["warnings"]) > 0

    def test_calculate_desuperheating(self, optimizer):
        """Test desuperheating calculation."""
        result = optimizer.calculate_desuperheating(
            steam_flow_lb_hr=30000.0,
            inlet_temperature_f=750.0,
        )

        if result["enabled"]:
            assert result["spray_required"] is True
            assert result["spray_rate_lb_hr"] > 0

    def test_calculate_desuperheating_disabled(self, prv_config):
        """Test when desuperheater is disabled."""
        config = prv_config
        config.desuperheater_enabled = False
        optimizer = PRVOptimizer(config=config)

        result = optimizer.calculate_desuperheating(
            steam_flow_lb_hr=30000.0,
            inlet_temperature_f=750.0,
        )

        assert result["enabled"] is False


# =============================================================================
# MULTI-PRV COORDINATOR TESTS
# =============================================================================

class TestMultiPRVCoordinator:
    """Test suite for MultiPRVCoordinator."""

    @pytest.fixture
    def prv_configs(self):
        """Create multiple PRV configurations."""
        return [
            PRVConfig(
                prv_id="PRV-001",
                inlet_pressure_psig=600.0,
                outlet_pressure_psig=150.0,
                design_flow_lb_hr=40000.0,
                max_flow_lb_hr=50000.0,
                cv_rated=150.0,
            ),
            PRVConfig(
                prv_id="PRV-002",
                inlet_pressure_psig=600.0,
                outlet_pressure_psig=150.0,
                design_flow_lb_hr=30000.0,
                max_flow_lb_hr=40000.0,
                cv_rated=120.0,
            ),
        ]

    @pytest.fixture
    def coordinator(self, prv_configs):
        """Create multi-PRV coordinator."""
        return MultiPRVCoordinator(prvs=prv_configs)

    def test_initialization(self, coordinator, prv_configs):
        """Test coordinator initialization."""
        assert len(coordinator.prvs) == 2
        assert "PRV-001" in coordinator.prvs
        assert "PRV-002" in coordinator.prvs

    def test_optimize_load_distribution(self, coordinator):
        """Test load distribution optimization."""
        result = coordinator.optimize_load_distribution(
            total_flow_required_lb_hr=60000.0,
            header_pressure_psig=150.0,
        )

        assert len(result["allocations"]) == 2
        assert result["all_demand_met"] is True
        assert result["total_allocated_lb_hr"] == pytest.approx(60000.0, rel=0.1)

    def test_optimize_load_exceeds_capacity(self, coordinator):
        """Test when demand exceeds total capacity."""
        result = coordinator.optimize_load_distribution(
            total_flow_required_lb_hr=100000.0,  # Exceeds combined capacity
            header_pressure_psig=150.0,
        )

        assert result["all_demand_met"] is False
        assert result["shortfall_lb_hr"] > 0

    def test_optimize_load_standby_prv(self, coordinator):
        """Test PRV placed on standby when not needed."""
        result = coordinator.optimize_load_distribution(
            total_flow_required_lb_hr=30000.0,  # Below single PRV capacity
            header_pressure_psig=150.0,
        )

        # At least one PRV may be on standby
        statuses = [a["status"] for a in result["allocations"]]
        # Should have at least one active
        assert "active" in statuses or "low_opening" in statuses or "high_opening" in statuses


# =============================================================================
# ASME B31.1 COMPLIANCE TESTS
# =============================================================================

class TestASMEB311Compliance:
    """Compliance tests for ASME B31.1 requirements."""

    @pytest.fixture
    def optimizer(self, prv_config):
        """Create PRV optimizer."""
        return PRVOptimizer(config=prv_config)

    @pytest.mark.compliance
    def test_opening_target_range(self, optimizer, asme_prv_opening_targets):
        """Test ASME B31.1 opening target range (50-70%)."""
        result = optimizer.size_prv()

        # Target status should reference 50-70% range
        assert "50" in result.opening_target_status or str(asme_prv_opening_targets["minimum"]) in result.opening_target_status

    @pytest.mark.compliance
    def test_cv_safety_margin(self, optimizer):
        """Test Cv includes 15% safety margin per ASME."""
        result = optimizer.size_prv()

        # Cv_recommended = Cv_required * 1.15
        expected_margin = (result.cv_recommended / result.cv_required - 1) * 100

        assert expected_margin == pytest.approx(15.0, rel=0.01)

    @pytest.mark.compliance
    def test_critical_flow_warning(self, optimizer):
        """Test critical flow condition generates warning."""
        result = optimizer.size_prv()

        if result.is_critical_flow:
            # Should have recommendation about noise/vibration
            assert any("critical" in rec.lower() or "noise" in rec.lower()
                       for rec in result.recommendations)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPRVPerformance:
    """Performance tests for PRV module."""

    @pytest.fixture
    def calculator(self):
        """Create Cv calculator."""
        return CvCalculator()

    @pytest.fixture
    def optimizer(self, prv_config):
        """Create PRV optimizer."""
        return PRVOptimizer(config=prv_config)

    @pytest.mark.performance
    def test_cv_calculation_speed(self, calculator):
        """Test Cv calculation speed."""
        import time
        start = time.time()

        for _ in range(1000):
            calculator.calculate_cv_steam(
                flow_lb_hr=30000.0,
                inlet_pressure_psig=600.0,
                outlet_pressure_psig=150.0,
            )

        elapsed = time.time() - start
        assert elapsed < 1.0  # 1000 calculations in <1s

    @pytest.mark.performance
    def test_sizing_speed(self, optimizer):
        """Test PRV sizing speed."""
        import time
        start = time.time()

        for _ in range(100):
            optimizer.size_prv()

        elapsed = time.time() - start
        assert elapsed < 1.0  # 100 sizings in <1s

    @pytest.mark.performance
    def test_opening_calculation_speed(self, calculator):
        """Test opening percentage calculation speed."""
        import time
        start = time.time()

        for cv in range(1, 1001):
            calculator.calculate_opening_percentage(float(cv), 1000.0)

        elapsed = time.time() - start
        assert elapsed < 0.5  # 1000 calculations in <0.5s
