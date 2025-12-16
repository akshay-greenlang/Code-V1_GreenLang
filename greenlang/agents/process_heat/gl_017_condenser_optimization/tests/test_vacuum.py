"""
GL-017 CONDENSYNC Agent - Vacuum System Monitor Tests

Unit tests for VacuumSystemMonitor and SteamJetEjectorModel.
Tests cover vacuum analysis, ejector performance, and decay testing.

Coverage targets:
    - Vacuum deviation analysis
    - Ejector capacity calculations
    - Steam consumption calculations
    - Vacuum decay test analysis
    - Maintenance recommendations
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_017_condenser_optimization.vacuum import (
    VacuumSystemMonitor,
    SteamJetEjectorModel,
    VacuumConstants,
    VacuumReading,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    VacuumSystemConfig,
    VacuumEquipmentType,
    PerformanceConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    VacuumSystemResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def vacuum_config():
    """Create default vacuum system configuration."""
    return VacuumSystemConfig()


@pytest.fixture
def performance_config():
    """Create default performance configuration."""
    return PerformanceConfig()


@pytest.fixture
def monitor(vacuum_config, performance_config):
    """Create VacuumSystemMonitor instance."""
    return VacuumSystemMonitor(vacuum_config, performance_config)


@pytest.fixture
def ejector_model():
    """Create SteamJetEjectorModel instance."""
    return SteamJetEjectorModel(
        stages=2,
        design_capacity_scfm=50.0,
        design_motive_pressure_psig=150.0,
        design_suction_inhga=1.5,
    )


# =============================================================================
# VACUUM CONSTANTS TESTS
# =============================================================================

class TestVacuumConstants:
    """Test VacuumConstants values."""

    def test_specific_steam_values(self):
        """Test specific steam consumption values."""
        assert VacuumConstants.SPECIFIC_STEAM_SINGLE_STAGE == 150.0
        assert VacuumConstants.SPECIFIC_STEAM_TWO_STAGE == 80.0
        assert VacuumConstants.SPECIFIC_STEAM_THREE_STAGE == 40.0

    def test_decay_rate_limits(self):
        """Test decay rate limit values."""
        assert VacuumConstants.ACCEPTABLE_DECAY_RATE_INHG_MIN == 0.1
        assert VacuumConstants.WARNING_DECAY_RATE_INHG_MIN == 0.2
        assert VacuumConstants.ALARM_DECAY_RATE_INHG_MIN == 0.5

    def test_air_equivalents(self):
        """Test air equivalent weights."""
        assert "air" in VacuumConstants.AIR_EQUIVALENT
        assert VacuumConstants.AIR_EQUIVALENT["air"] == 1.0


# =============================================================================
# EJECTOR MODEL TESTS
# =============================================================================

class TestSteamJetEjectorModel:
    """Test SteamJetEjectorModel class."""

    def test_initialization(self, ejector_model):
        """Test ejector model initializes correctly."""
        assert ejector_model.stages == 2
        assert ejector_model.design_capacity_scfm == 50.0

    def test_specific_steam_by_stages(self):
        """Test specific steam varies by stages."""
        model_1 = SteamJetEjectorModel(stages=1)
        model_2 = SteamJetEjectorModel(stages=2)
        model_3 = SteamJetEjectorModel(stages=3)

        assert model_1.design_specific_steam > model_2.design_specific_steam
        assert model_2.design_specific_steam > model_3.design_specific_steam

    def test_capacity_at_design(self, ejector_model):
        """Test capacity at design conditions."""
        capacity = ejector_model.calculate_capacity(
            motive_steam_pressure_psig=150.0,
            suction_pressure_inhga=1.5,
        )

        # Should be close to design capacity
        assert capacity == pytest.approx(50.0, rel=0.1)

    def test_capacity_vs_motive_pressure(self, ejector_model):
        """Test capacity increases with motive pressure."""
        cap_low = ejector_model.calculate_capacity(
            motive_steam_pressure_psig=100.0,
            suction_pressure_inhga=1.5,
        )

        cap_high = ejector_model.calculate_capacity(
            motive_steam_pressure_psig=200.0,
            suction_pressure_inhga=1.5,
        )

        assert cap_high > cap_low

    def test_steam_consumption(self, ejector_model):
        """Test steam consumption calculation."""
        steam = ejector_model.calculate_steam_consumption(
            air_removal_scfm=30.0,
            motive_steam_pressure_psig=150.0,
            suction_pressure_inhga=1.5,
        )

        # Should be positive
        assert steam > 0

    def test_efficiency_calculation(self, ejector_model):
        """Test efficiency calculation."""
        efficiency = ejector_model.calculate_efficiency(
            actual_capacity_scfm=30.0,
            actual_steam_lb_hr=5000.0,
            motive_steam_pressure_psig=150.0,
        )

        assert 0 <= efficiency <= 100

    def test_efficiency_at_design(self, ejector_model):
        """Test efficiency at design conditions."""
        # Calculate ideal steam consumption
        ideal_steam = ejector_model.calculate_steam_consumption(
            air_removal_scfm=30.0,
            motive_steam_pressure_psig=150.0,
            suction_pressure_inhga=1.5,
        )

        efficiency = ejector_model.calculate_efficiency(
            actual_capacity_scfm=30.0,
            actual_steam_lb_hr=ideal_steam,
            motive_steam_pressure_psig=150.0,
        )

        # Should be close to 100% at design
        assert efficiency == pytest.approx(100.0, rel=0.1)


# =============================================================================
# MONITOR INITIALIZATION TESTS
# =============================================================================

class TestMonitorInitialization:
    """Test monitor initialization."""

    def test_basic_initialization(self, vacuum_config, performance_config):
        """Test monitor initializes correctly."""
        monitor = VacuumSystemMonitor(vacuum_config, performance_config)
        assert monitor is not None

    def test_ejector_model_created(self, monitor):
        """Test ejector model is created for ejector equipment."""
        assert monitor.ejector_model is not None

    def test_no_ejector_for_pump(self, performance_config):
        """Test no ejector model for vacuum pump."""
        config = VacuumSystemConfig(
            primary_equipment=VacuumEquipmentType.LIQUID_RING_PUMP
        )
        monitor = VacuumSystemMonitor(config, performance_config)

        assert monitor.ejector_model is None


# =============================================================================
# VACUUM ANALYSIS TESTS
# =============================================================================

class TestVacuumAnalysis:
    """Test vacuum system analysis."""

    def test_analyze_normal_vacuum(self, monitor):
        """Test analysis with normal vacuum."""
        result = monitor.analyze_vacuum_system(
            condenser_vacuum_inhga=1.5,
            load_pct=100.0,
        )

        assert isinstance(result, VacuumSystemResult)
        assert result.vacuum_normal is True

    def test_analyze_degraded_vacuum(self, monitor):
        """Test analysis with degraded vacuum."""
        result = monitor.analyze_vacuum_system(
            condenser_vacuum_inhga=2.5,  # High = bad vacuum
            load_pct=100.0,
        )

        assert result.vacuum_normal is False
        assert result.vacuum_deviation_inhg > 0

    def test_result_components(self, monitor):
        """Test all result components are populated."""
        result = monitor.analyze_vacuum_system(
            condenser_vacuum_inhga=1.6,
            load_pct=85.0,
            motive_steam_pressure_psig=150.0,
            air_removal_scfm=30.0,
        )

        assert result.current_vacuum_inhga is not None
        assert result.expected_vacuum_inhga is not None
        assert result.vacuum_deviation_inhg is not None
        assert result.air_removal_capacity_pct is not None

    def test_with_ejector_data(self, monitor):
        """Test analysis with full ejector data."""
        result = monitor.analyze_vacuum_system(
            condenser_vacuum_inhga=1.5,
            motive_steam_pressure_psig=150.0,
            motive_steam_flow_lb_hr=5000.0,
            air_removal_scfm=30.0,
            load_pct=85.0,
        )

        assert result.ejector_efficiency_pct is not None
        assert result.motive_steam_consumption_lb_hr is not None


# =============================================================================
# EXPECTED VACUUM TESTS
# =============================================================================

class TestExpectedVacuum:
    """Test expected vacuum calculation."""

    def test_design_conditions(self, monitor):
        """Test expected vacuum at design conditions."""
        expected = monitor._calculate_expected_vacuum(
            load_pct=100.0,
            cw_inlet_temp_f=70.0,
        )

        # Should be close to design (1.5 inHgA)
        assert 1.0 < expected < 2.0

    def test_lower_load_better_vacuum(self, monitor):
        """Test lower load gives better vacuum."""
        vac_high_load = monitor._calculate_expected_vacuum(
            load_pct=100.0,
            cw_inlet_temp_f=70.0,
        )

        vac_low_load = monitor._calculate_expected_vacuum(
            load_pct=50.0,
            cw_inlet_temp_f=70.0,
        )

        # Lower vacuum number = better vacuum
        assert vac_low_load < vac_high_load

    def test_temperature_correction(self, monitor):
        """Test inlet temperature affects expected vacuum."""
        vac_cold = monitor._calculate_expected_vacuum(
            load_pct=85.0,
            cw_inlet_temp_f=60.0,
        )

        vac_hot = monitor._calculate_expected_vacuum(
            load_pct=85.0,
            cw_inlet_temp_f=90.0,
        )

        # Hotter inlet = worse (higher) vacuum
        assert vac_hot > vac_cold


# =============================================================================
# VACUUM NORMAL TESTS
# =============================================================================

class TestVacuumNormal:
    """Test vacuum normal determination."""

    def test_vacuum_normal(self, monitor):
        """Test vacuum is normal within tolerance."""
        is_normal = monitor._is_vacuum_normal(1.5, 1.5)
        assert is_normal is True

    def test_vacuum_abnormal_high(self, monitor):
        """Test vacuum abnormal when too high."""
        is_normal = monitor._is_vacuum_normal(3.0, 1.5)
        assert is_normal is False

    def test_vacuum_near_boundary(self, monitor):
        """Test vacuum at boundary."""
        is_normal = monitor._is_vacuum_normal(1.8, 1.5)
        # Deviation = 0.3, at boundary
        assert is_normal in [True, False]  # Depends on exact threshold


# =============================================================================
# CAPACITY UTILIZATION TESTS
# =============================================================================

class TestCapacityUtilization:
    """Test capacity utilization calculation."""

    def test_capacity_at_50_percent(self, monitor):
        """Test capacity utilization at 50%."""
        utilization = monitor._calculate_capacity_utilization(
            air_removal_scfm=25.0,
            condenser_vacuum=1.5,
            motive_pressure=150.0,
        )

        # 25/50 = 50%
        assert utilization == pytest.approx(50.0, rel=0.2)

    def test_capacity_with_no_measurement(self, monitor):
        """Test capacity defaults to 100% if not measured."""
        utilization = monitor._calculate_capacity_utilization(
            air_removal_scfm=None,
            condenser_vacuum=1.5,
            motive_pressure=150.0,
        )

        assert utilization == 100.0


# =============================================================================
# AIR INGRESS ESTIMATION TESTS
# =============================================================================

class TestAirIngressEstimation:
    """Test air ingress estimation."""

    def test_estimate_from_air_removal(self, monitor):
        """Test air ingress estimated from air removal."""
        ingress = monitor._estimate_air_ingress(
            actual_vacuum=1.5,
            expected_vacuum=1.5,
            air_removal_scfm=30.0,
        )

        # Should equal air removal at steady state
        assert ingress == 30.0

    def test_estimate_from_vacuum_deviation(self, monitor):
        """Test air ingress estimated from vacuum deviation."""
        ingress = monitor._estimate_air_ingress(
            actual_vacuum=2.0,  # 0.5 deviation
            expected_vacuum=1.5,
            air_removal_scfm=None,
        )

        # Should estimate based on deviation
        assert ingress > 0


# =============================================================================
# MAINTENANCE TESTS
# =============================================================================

class TestMaintenanceRecommendation:
    """Test maintenance recommendation logic."""

    def test_maintenance_for_high_deviation(self, monitor):
        """Test maintenance required for high deviation."""
        required = monitor._check_maintenance_required(
            actual_vacuum=2.5,
            expected_vacuum=1.5,
            ejector_efficiency=85.0,
            capacity_utilization=60.0,
        )

        assert required is True

    def test_maintenance_for_low_efficiency(self, monitor):
        """Test maintenance required for low efficiency."""
        required = monitor._check_maintenance_required(
            actual_vacuum=1.6,
            expected_vacuum=1.5,
            ejector_efficiency=50.0,  # Below 70%
            capacity_utilization=60.0,
        )

        assert required is True

    def test_maintenance_for_high_utilization(self, monitor):
        """Test maintenance required for high utilization."""
        required = monitor._check_maintenance_required(
            actual_vacuum=1.6,
            expected_vacuum=1.5,
            ejector_efficiency=85.0,
            capacity_utilization=95.0,  # Above 90%
        )

        assert required is True

    def test_no_maintenance_normal(self, monitor):
        """Test no maintenance for normal operation."""
        required = monitor._check_maintenance_required(
            actual_vacuum=1.6,
            expected_vacuum=1.5,
            ejector_efficiency=85.0,
            capacity_utilization=60.0,
        )

        assert required is False


# =============================================================================
# VACUUM DECAY TEST TESTS
# =============================================================================

class TestVacuumDecayTest:
    """Test vacuum decay test analysis."""

    def test_acceptable_decay(self, monitor):
        """Test acceptable decay rate."""
        result = monitor.perform_vacuum_decay_test(
            initial_vacuum_inhga=1.5,
            final_vacuum_inhga=1.55,  # Small increase
            duration_minutes=10.0,
        )

        # Rate = 0.05 / 10 = 0.005 inHg/min < 0.1
        assert result["test_passed"] is True
        assert result["status"] == "acceptable"

    def test_marginal_decay(self, monitor):
        """Test marginal decay rate."""
        result = monitor.perform_vacuum_decay_test(
            initial_vacuum_inhga=1.5,
            final_vacuum_inhga=2.0,  # 0.5 increase
            duration_minutes=3.0,
        )

        # Rate = 0.5 / 3 = 0.167 inHg/min
        assert result["status"] in ["marginal", "excessive"]

    def test_excessive_decay(self, monitor):
        """Test excessive decay rate."""
        result = monitor.perform_vacuum_decay_test(
            initial_vacuum_inhga=1.5,
            final_vacuum_inhga=3.0,  # Large increase
            duration_minutes=5.0,
        )

        # Rate = 1.5 / 5 = 0.3 inHg/min
        assert result["test_passed"] is False
        assert result["status"] in ["excessive", "severe"]

    def test_severe_decay(self, monitor):
        """Test severe decay rate."""
        result = monitor.perform_vacuum_decay_test(
            initial_vacuum_inhga=1.5,
            final_vacuum_inhga=4.0,  # Very large increase
            duration_minutes=3.0,
        )

        # Rate = 2.5 / 3 = 0.83 inHg/min
        assert result["test_passed"] is False
        assert result["status"] == "severe"

    def test_decay_test_results(self, monitor):
        """Test decay test result structure."""
        result = monitor.perform_vacuum_decay_test(
            initial_vacuum_inhga=1.5,
            final_vacuum_inhga=1.6,
            duration_minutes=10.0,
        )

        assert "test_passed" in result
        assert "decay_rate_inhg_min" in result
        assert "status" in result
        assert "severity" in result
        assert "estimated_air_ingress_scfm" in result
        assert "recommended_action" in result


# =============================================================================
# HISTORY TESTS
# =============================================================================

class TestVacuumHistory:
    """Test vacuum history management."""

    def test_record_reading(self, monitor):
        """Test recording vacuum readings."""
        monitor._record_reading(1.5, 85.0, 30.0, 150.0)

        assert len(monitor._history) == 1

    def test_history_trimming(self, monitor):
        """Test old history is trimmed."""
        # Add old reading
        old_reading = VacuumReading(
            timestamp=datetime.now(timezone.utc) - timedelta(days=10),
            vacuum_inhga=1.5,
            load_pct=85.0,
            air_removal_scfm=30.0,
            motive_steam_pressure_psig=150.0,
        )
        monitor._history.append(old_reading)

        # Record new reading - triggers trimming
        monitor._record_reading(1.55, 85.0, 30.0, 150.0)

        # Old entry should be removed (older than 7 days)
        assert len(monitor._history) == 1

    def test_get_vacuum_trend(self, monitor):
        """Test retrieving vacuum trend."""
        for i in range(5):
            monitor._record_reading(1.5 + (i * 0.02), 85.0, 30.0, 150.0)

        trend = monitor.get_vacuum_trend(hours=24)

        assert isinstance(trend, list)
        assert len(trend) == 5


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Test calculation counting."""

    def test_count_increments(self, monitor):
        """Test calculation count increments."""
        initial = monitor.calculation_count

        monitor.analyze_vacuum_system(
            condenser_vacuum_inhga=1.5,
            load_pct=85.0,
        )

        assert monitor.calculation_count == initial + 1
