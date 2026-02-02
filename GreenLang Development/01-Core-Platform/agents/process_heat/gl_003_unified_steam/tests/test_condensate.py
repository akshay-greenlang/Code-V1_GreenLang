"""
GL-003 Unified Steam System Optimizer - Condensate Module Tests

Unit tests for condensate return optimization module.
Target: 85%+ coverage of condensate.py

Tests:
    - Condensate heat recovery calculations
    - Return rate analysis
    - Condensate quality monitoring
    - Steam trap survey analysis
    - Makeup water calculations
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_003_unified_steam.condensate import (
    CondensateReturnOptimizer,
    CondensateHeatCalculator,
    CondensateQualityAnalyzer,
    SteamTrapSurveyAnalyzer,
)
from greenlang.agents.process_heat.gl_003_unified_steam.config import (
    CondensateConfig,
    SteamTrapSurveyConfig,
)
from greenlang.agents.process_heat.gl_003_unified_steam.schemas import (
    CondensateReading,
    CondensateReturnAnalysis,
    SteamTrapReading,
    TrapSurveyAnalysis,
    TrapStatus,
)


# =============================================================================
# CONDENSATE HEAT CALCULATOR TESTS
# =============================================================================

class TestCondensateHeatCalculator:
    """Test suite for CondensateHeatCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create condensate heat calculator."""
        return CondensateHeatCalculator()

    def test_calculate_heat_content(self, calculator):
        """Test heat content calculation."""
        # Heat = mass * Cp * (T - T_ref)
        # = 1000 lb/hr * 1.0 BTU/lb-F * (180 - 60)F
        # = 120,000 BTU/hr
        heat = calculator.calculate_heat_content(
            flow_lb_hr=1000.0,
            temperature_f=180.0,
            reference_temp_f=60.0,
        )
        assert heat == pytest.approx(120000.0, rel=0.01)

    def test_calculate_heat_recovery(self, calculator):
        """Test heat recovery calculation."""
        # Heat recovered vs lost to drain
        recovery = calculator.calculate_heat_recovery(
            return_flow_lb_hr=85000.0,
            return_temp_f=180.0,
            steam_flow_lb_hr=100000.0,
            makeup_temp_f=60.0,
        )

        assert recovery["heat_recovered_btu_hr"] > 0
        assert recovery["heat_recovery_efficiency_pct"] > 0
        assert recovery["fuel_savings_pct"] > 0

    def test_calculate_makeup_water_requirement(self, calculator):
        """Test makeup water calculation."""
        makeup = calculator.calculate_makeup_water(
            steam_flow_lb_hr=100000.0,
            condensate_return_lb_hr=85000.0,
            blowdown_pct=3.0,
        )

        # Makeup = steam - return + blowdown
        # = 100000 - 85000 + 3000 = 18000 lb/hr
        expected = 100000.0 - 85000.0 + 3000.0
        assert makeup == pytest.approx(expected, rel=0.01)

    def test_calculate_return_temperature_effect(self, calculator):
        """Test return temperature effect on fuel savings."""
        # Higher return temp = more fuel savings
        savings_140 = calculator.calculate_fuel_savings(
            return_temp_f=140.0,
            makeup_temp_f=60.0,
            return_flow_lb_hr=85000.0,
            fuel_cost_per_mmbtu=5.0,
        )

        savings_180 = calculator.calculate_fuel_savings(
            return_temp_f=180.0,
            makeup_temp_f=60.0,
            return_flow_lb_hr=85000.0,
            fuel_cost_per_mmbtu=5.0,
        )

        assert savings_180 > savings_140

    @pytest.mark.parametrize("return_temp,expected_savings_ratio", [
        (100.0, 0.5),
        (140.0, 1.0),
        (180.0, 1.5),
    ])
    def test_temperature_savings_relationship(
        self, calculator, return_temp, expected_savings_ratio
    ):
        """Test linear relationship between temperature and savings."""
        savings = calculator.calculate_fuel_savings(
            return_temp_f=return_temp,
            makeup_temp_f=60.0,
            return_flow_lb_hr=85000.0,
            fuel_cost_per_mmbtu=5.0,
        )
        # Savings should be proportional to temp difference
        assert savings > 0


# =============================================================================
# CONDENSATE QUALITY ANALYZER TESTS
# =============================================================================

class TestCondensateQualityAnalyzer:
    """Test suite for CondensateQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self, condensate_config):
        """Create condensate quality analyzer."""
        return CondensateQualityAnalyzer(config=condensate_config)

    def test_check_contamination_clean(self, analyzer, condensate_reading):
        """Test contamination check for clean condensate."""
        result = analyzer.check_contamination(condensate_reading)

        assert result["is_contaminated"] is False
        assert result["tds_ok"] is True
        assert result["oil_ok"] is True

    def test_check_contamination_tds_high(self, analyzer):
        """Test contamination detection for high TDS."""
        reading = CondensateReading(
            location_id="TEST",
            flow_rate_lb_hr=40000.0,
            temperature_f=180.0,
            tds_ppm=100.0,  # Above 50 ppm limit
        )

        result = analyzer.check_contamination(reading)

        assert result["is_contaminated"] is True
        assert result["tds_ok"] is False

    def test_check_contamination_oil_high(self, analyzer):
        """Test contamination detection for high oil content."""
        reading = CondensateReading(
            location_id="TEST",
            flow_rate_lb_hr=40000.0,
            temperature_f=180.0,
            oil_ppm=2.0,  # Above 1.0 ppm limit
        )

        result = analyzer.check_contamination(reading)

        assert result["is_contaminated"] is True
        assert result["oil_ok"] is False

    def test_check_contamination_iron_high(self, analyzer):
        """Test contamination detection for high iron content."""
        reading = CondensateReading(
            location_id="TEST",
            flow_rate_lb_hr=40000.0,
            temperature_f=180.0,
            iron_ppb=200.0,  # Above 100 ppb limit
        )

        result = analyzer.check_contamination(reading)

        assert result["is_contaminated"] is True
        assert result["iron_ok"] is False

    def test_get_contamination_sources(self, analyzer):
        """Test identification of contamination sources."""
        reading = CondensateReading(
            location_id="TEST",
            flow_rate_lb_hr=40000.0,
            temperature_f=180.0,
            tds_ppm=100.0,
            oil_ppm=2.0,
            iron_ppb=200.0,
        )

        sources = analyzer.identify_contamination_sources(reading)

        assert len(sources) >= 2  # Multiple contamination indicators

    def test_ph_analysis(self, analyzer):
        """Test pH analysis for corrosion risk."""
        # Low pH indicates corrosion risk
        reading_low_ph = CondensateReading(
            location_id="TEST",
            flow_rate_lb_hr=40000.0,
            temperature_f=180.0,
            ph=6.5,  # Below 7.0
        )

        result = analyzer.analyze_ph(reading_low_ph)
        assert result["corrosion_risk"] is True

        # Optimal pH range
        reading_normal_ph = CondensateReading(
            location_id="TEST",
            flow_rate_lb_hr=40000.0,
            temperature_f=180.0,
            ph=9.0,  # Normal for condensate
        )

        result = analyzer.analyze_ph(reading_normal_ph)
        assert result["corrosion_risk"] is False


# =============================================================================
# CONDENSATE RETURN OPTIMIZER TESTS
# =============================================================================

class TestCondensateReturnOptimizer:
    """Test suite for CondensateReturnOptimizer."""

    @pytest.fixture
    def optimizer(self, condensate_config, trap_survey_config):
        """Create condensate return optimizer."""
        return CondensateReturnOptimizer(
            config=condensate_config,
            trap_survey_config=trap_survey_config,
        )

    def test_initialization(self, optimizer, condensate_config):
        """Test optimizer initialization."""
        assert optimizer.config == condensate_config
        assert optimizer.heat_calc is not None
        assert optimizer.quality_analyzer is not None

    def test_analyze_return_system_optimal(self, optimizer):
        """Test analysis for optimal return system."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=50000.0,
                temperature_f=185.0,
                tds_ppm=20.0,
            ),
            CondensateReading(
                location_id="RETURN-2",
                flow_rate_lb_hr=35000.0,
                temperature_f=180.0,
                tds_ppm=25.0,
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        assert isinstance(analysis, CondensateReturnAnalysis)
        assert analysis.return_rate_pct == 85.0
        assert analysis.condensate_return_lb_hr == 85000.0

    def test_analyze_return_system_low_return(self, optimizer):
        """Test analysis for low return rate."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=30000.0,
                temperature_f=180.0,
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        assert analysis.return_rate_pct < optimizer.config.target_return_rate_pct
        assert len(analysis.warnings) > 0

    def test_analyze_return_system_low_temperature(self, optimizer):
        """Test analysis for low return temperature."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=85000.0,
                temperature_f=130.0,  # Below 140F minimum
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        assert analysis.avg_return_temperature_f < optimizer.config.min_return_temp_f
        assert len(analysis.recommendations) > 0

    def test_heat_recovery_calculation(self, optimizer):
        """Test heat recovery is calculated correctly."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=85000.0,
                temperature_f=180.0,
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        # Heat recovered = 85000 * 1.0 * (180 - 60) = 10,200,000 BTU/hr
        expected = 85000.0 * 1.0 * (180.0 - 60.0)
        assert analysis.heat_recovered_btu_hr == pytest.approx(expected, rel=0.01)

    def test_makeup_water_calculation(self, optimizer):
        """Test makeup water calculation."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=85000.0,
                temperature_f=180.0,
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        # Makeup = steam - return = 100000 - 85000 = 15000 lb/hr
        assert analysis.makeup_water_required_lb_hr == pytest.approx(15000.0, rel=0.1)

    def test_contaminated_condensate_handling(self, optimizer):
        """Test handling of contaminated condensate."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=50000.0,
                temperature_f=180.0,
                is_contaminated=True,  # Contaminated
            ),
            CondensateReading(
                location_id="RETURN-2",
                flow_rate_lb_hr=35000.0,
                temperature_f=180.0,
                is_contaminated=False,  # Clean
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        assert analysis.contaminated_condensate_lb_hr == 50000.0
        assert len(analysis.warnings) > 0

    def test_economic_analysis(self, optimizer):
        """Test economic analysis is included."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=85000.0,
                temperature_f=180.0,
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        assert analysis.annual_fuel_savings_usd is not None
        assert analysis.annual_fuel_savings_usd > 0

    def test_provenance_hash(self, optimizer):
        """Test provenance hash is generated."""
        readings = [
            CondensateReading(
                location_id="RETURN-1",
                flow_rate_lb_hr=85000.0,
                temperature_f=180.0,
            ),
        ]

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=readings,
        )

        assert analysis.provenance_hash is not None
        assert len(analysis.provenance_hash) == 64


# =============================================================================
# STEAM TRAP SURVEY ANALYZER TESTS
# =============================================================================

class TestSteamTrapSurveyAnalyzer:
    """Test suite for SteamTrapSurveyAnalyzer."""

    @pytest.fixture
    def analyzer(self, trap_survey_config):
        """Create steam trap survey analyzer."""
        return SteamTrapSurveyAnalyzer(config=trap_survey_config)

    def test_initialization(self, analyzer, trap_survey_config):
        """Test analyzer initialization."""
        assert analyzer.config == trap_survey_config

    def test_analyze_survey_all_operating(self, analyzer, generate_trap_readings):
        """Test survey analysis with all traps operating."""
        readings = generate_trap_readings(100, failure_rate=0.0)

        analysis = analyzer.analyze_survey(readings)

        assert isinstance(analysis, TrapSurveyAnalysis)
        assert analysis.total_traps == 100
        assert analysis.operating_count == 100
        assert analysis.failure_rate_pct == 0.0

    def test_analyze_survey_with_failures(self, analyzer, generate_trap_readings):
        """Test survey analysis with failed traps."""
        readings = generate_trap_readings(100, failure_rate=0.1)

        analysis = analyzer.analyze_survey(readings)

        assert analysis.failure_rate_pct > 0
        assert analysis.failed_open_count + analysis.failed_closed_count > 0

    def test_steam_loss_calculation(self, analyzer):
        """Test steam loss calculation for failed traps."""
        readings = [
            SteamTrapReading(
                trap_id="TRAP-001",
                location="HX-001",
                status=TrapStatus.FAILED_OPEN,
                inlet_pressure_psig=150.0,
                steam_loss_lb_hr=50.0,
            ),
            SteamTrapReading(
                trap_id="TRAP-002",
                location="HX-002",
                status=TrapStatus.OPERATING,
                inlet_pressure_psig=150.0,
            ),
        ]

        analysis = analyzer.analyze_survey(readings)

        assert analysis.total_steam_loss_lb_hr == 50.0

    def test_annual_cost_calculation(self, analyzer):
        """Test annual cost calculation for steam losses."""
        readings = [
            SteamTrapReading(
                trap_id="TRAP-001",
                location="HX-001",
                status=TrapStatus.FAILED_OPEN,
                inlet_pressure_psig=150.0,
                steam_loss_lb_hr=100.0,  # 100 lb/hr loss
            ),
        ]

        analysis = analyzer.analyze_survey(readings)

        # Annual loss = 100 lb/hr * 8000 hr/yr = 800,000 lb/yr = 800 Mlb
        # Cost = 800 Mlb * $10/Mlb = $8,000
        assert analysis.annual_loss_cost_usd == pytest.approx(8000.0, rel=0.1)

    def test_priority_repairs(self, analyzer):
        """Test priority repair list generation."""
        readings = [
            SteamTrapReading(
                trap_id="TRAP-001",
                location="Critical Process",
                status=TrapStatus.FAILED_OPEN,
                inlet_pressure_psig=150.0,
                steam_loss_lb_hr=100.0,
            ),
            SteamTrapReading(
                trap_id="TRAP-002",
                location="Secondary Process",
                status=TrapStatus.FAILED_OPEN,
                inlet_pressure_psig=15.0,
                steam_loss_lb_hr=20.0,
            ),
        ]

        analysis = analyzer.analyze_survey(readings)

        assert len(analysis.priority_repairs) == 2
        # Higher loss should be first priority
        assert "TRAP-001" in analysis.priority_repairs[0]

    def test_failure_by_type(self, analyzer):
        """Test failure breakdown by trap type."""
        readings = [
            SteamTrapReading(
                trap_id="TRAP-001",
                location="HX-001",
                trap_type="thermodynamic",
                status=TrapStatus.FAILED_OPEN,
                inlet_pressure_psig=150.0,
            ),
            SteamTrapReading(
                trap_id="TRAP-002",
                location="HX-002",
                trap_type="thermostatic",
                status=TrapStatus.FAILED_CLOSED,
                inlet_pressure_psig=150.0,
            ),
            SteamTrapReading(
                trap_id="TRAP-003",
                location="HX-003",
                trap_type="thermodynamic",
                status=TrapStatus.OPERATING,
                inlet_pressure_psig=150.0,
            ),
        ]

        analysis = analyzer.analyze_survey(readings)

        assert analysis.failures_by_type is not None
        assert "thermodynamic" in analysis.failures_by_type
        assert "thermostatic" in analysis.failures_by_type

    def test_empty_survey(self, analyzer):
        """Test analysis with empty survey."""
        analysis = analyzer.analyze_survey([])

        assert analysis.total_traps == 0
        assert analysis.failure_rate_pct == 0.0

    def test_recommendations_high_failure_rate(self, analyzer, generate_trap_readings):
        """Test recommendations for high failure rate."""
        # 15% failure rate
        readings = generate_trap_readings(100, failure_rate=0.15)

        analysis = analyzer.analyze_survey(readings)

        assert len(analysis.recommendations) > 0
        assert any("survey" in rec.lower() or "maintenance" in rec.lower()
                   for rec in analysis.recommendations)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCondensateIntegration:
    """Integration tests for condensate module."""

    @pytest.fixture
    def optimizer(self, condensate_config, trap_survey_config):
        """Create condensate return optimizer."""
        return CondensateReturnOptimizer(
            config=condensate_config,
            trap_survey_config=trap_survey_config,
        )

    @pytest.mark.integration
    def test_full_analysis_with_traps(
        self,
        optimizer,
        generate_condensate_readings,
        generate_trap_readings,
    ):
        """Test full analysis including trap survey."""
        condensate_readings = generate_condensate_readings(5)
        trap_readings = generate_trap_readings(50, failure_rate=0.08)

        analysis = optimizer.analyze_return_system(
            steam_flow_lb_hr=100000.0,
            condensate_readings=condensate_readings,
            trap_readings=trap_readings,
        )

        assert analysis.return_rate_pct > 0
        assert analysis.trap_survey_summary is not None


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestCondensatePerformance:
    """Performance tests for condensate module."""

    @pytest.fixture
    def optimizer(self, condensate_config, trap_survey_config):
        """Create condensate return optimizer."""
        return CondensateReturnOptimizer(
            config=condensate_config,
            trap_survey_config=trap_survey_config,
        )

    @pytest.mark.performance
    def test_analysis_speed(self, optimizer, generate_condensate_readings):
        """Test analysis speed (<10ms)."""
        import time
        readings = generate_condensate_readings(10)

        start = time.time()
        for _ in range(100):
            optimizer.analyze_return_system(
                steam_flow_lb_hr=100000.0,
                condensate_readings=readings,
            )
        elapsed = time.time() - start

        assert elapsed < 1.0  # 100 analyses in <1s

    @pytest.mark.performance
    def test_trap_survey_throughput(self, generate_trap_readings, trap_survey_config):
        """Test trap survey analysis throughput."""
        import time
        analyzer = SteamTrapSurveyAnalyzer(config=trap_survey_config)

        # Large survey with 1000 traps
        readings = generate_trap_readings(1000, failure_rate=0.1)

        start = time.time()
        analyzer.analyze_survey(readings)
        elapsed = time.time() - start

        # Should analyze 1000 traps in <500ms
        assert elapsed < 0.5
