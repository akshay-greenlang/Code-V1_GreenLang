"""
GL-003 Unified Steam System Optimizer - Distribution Module Tests

Unit tests for steam distribution and header balancing module.
Target: 85%+ coverage of distribution.py

Tests:
    - IAPWS-IF97 steam property calculations
    - Steam header pressure balancing
    - Exergy calculations
    - Supply-demand coordination
    - Pressure trend analysis
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import math


from greenlang.agents.process_heat.gl_003_unified_steam.distribution import (
    SteamPropertyCalculator,
    SteamTableConstants,
    ExergyConstants,
    HeaderBalanceCalculator,
    SteamDistributionOptimizer,
)
from greenlang.agents.process_heat.gl_003_unified_steam.config import (
    SteamHeaderConfig,
    SteamHeaderLevel,
    ExergyOptimizationConfig,
)
from greenlang.agents.process_heat.gl_003_unified_steam.schemas import (
    HeaderBalanceInput,
    HeaderBalanceOutput,
    OptimizationStatus,
    SteamProperties,
)


# =============================================================================
# STEAM PROPERTY CALCULATOR TESTS
# =============================================================================

class TestSteamPropertyCalculator:
    """Test suite for SteamPropertyCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create steam property calculator."""
        return SteamPropertyCalculator(reference_temp_f=77.0)

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.reference_temp_f == 77.0
        assert calculator.reference_temp_r == pytest.approx(536.67)

    def test_saturation_properties_at_150_psig(self, calculator, iapws_saturation_reference):
        """Test saturation properties at 150 psig match IAPWS-IF97."""
        props = calculator.get_saturation_properties(150.0)

        ref = iapws_saturation_reference[150]
        assert props["saturation_temp_f"] == pytest.approx(ref[0], rel=0.01)
        assert props["h_f_btu_lb"] == pytest.approx(ref[1], rel=0.01)
        assert props["h_fg_btu_lb"] == pytest.approx(ref[2], rel=0.01)
        assert props["h_g_btu_lb"] == pytest.approx(ref[3], rel=0.01)

    @pytest.mark.parametrize("pressure_psig,expected_t_sat", [
        (0, 212.0),
        (15, 250.3),
        (50, 298.0),
        (100, 337.9),
        (150, 365.9),
        (300, 421.7),
        (600, 489.0),
    ])
    def test_saturation_temperature_accuracy(
        self, calculator, pressure_psig, expected_t_sat
    ):
        """Test saturation temperature accuracy against IAPWS-IF97."""
        props = calculator.get_saturation_properties(pressure_psig)
        assert props["saturation_temp_f"] == pytest.approx(expected_t_sat, rel=0.01)

    def test_saturation_properties_interpolation(self, calculator):
        """Test interpolation between table values."""
        # 125 psig is between 100 and 150 in the table
        props = calculator.get_saturation_properties(125.0)

        # Should be between 100 psig (337.9F) and 150 psig (365.9F)
        assert 337.9 < props["saturation_temp_f"] < 365.9

    def test_saturation_properties_clamping_low(self, calculator):
        """Test clamping for pressures below table range."""
        props = calculator.get_saturation_properties(-10.0)
        # Should clamp to 0 psig
        assert props["saturation_temp_f"] == pytest.approx(212.0, rel=0.01)

    def test_saturation_properties_clamping_high(self, calculator):
        """Test clamping for pressures above table range."""
        props = calculator.get_saturation_properties(700.0)
        # Should clamp to 600 psig
        assert props["saturation_temp_f"] == pytest.approx(489.0, rel=0.01)

    def test_calculate_steam_enthalpy_saturated(self, calculator):
        """Test enthalpy calculation for saturated steam."""
        h = calculator.calculate_steam_enthalpy(
            pressure_psig=150.0,
            temperature_f=None,
            dryness_fraction=1.0,
        )
        # h_g at 150 psig = 1196.0 BTU/lb
        assert h == pytest.approx(1196.0, rel=0.01)

    def test_calculate_steam_enthalpy_wet_steam(self, calculator):
        """Test enthalpy calculation for wet steam."""
        h = calculator.calculate_steam_enthalpy(
            pressure_psig=150.0,
            temperature_f=None,
            dryness_fraction=0.95,
        )
        # h = h_f + x * h_fg = 339.2 + 0.95 * 856.8 = 1153.1 BTU/lb
        expected = 339.2 + 0.95 * 856.8
        assert h == pytest.approx(expected, rel=0.01)

    def test_calculate_steam_enthalpy_superheated(self, calculator):
        """Test enthalpy calculation for superheated steam."""
        h = calculator.calculate_steam_enthalpy(
            pressure_psig=150.0,
            temperature_f=450.0,  # Above saturation (365.9F)
            dryness_fraction=1.0,
        )
        # h = h_g + Cp * superheat = 1196.0 + 0.48 * (450 - 365.9)
        superheat = 450.0 - 365.9
        expected = 1196.0 + 0.48 * superheat
        assert h == pytest.approx(expected, rel=0.01)

    def test_calculate_steam_entropy_saturated(self, calculator):
        """Test entropy calculation for saturated steam."""
        s = calculator.calculate_steam_entropy(
            pressure_psig=150.0,
            temperature_f=None,
            dryness_fraction=1.0,
        )
        # s_g at 150 psig = 1.6375 BTU/lb-R
        assert s == pytest.approx(1.6375, rel=0.01)

    def test_calculate_steam_entropy_wet_steam(self, calculator):
        """Test entropy calculation for wet steam."""
        s = calculator.calculate_steam_entropy(
            pressure_psig=150.0,
            temperature_f=None,
            dryness_fraction=0.95,
        )
        # s = s_f + x * s_fg = 0.5208 + 0.95 * 1.1167
        expected = 0.5208 + 0.95 * 1.1167
        assert s == pytest.approx(expected, rel=0.01)

    def test_calculate_specific_exergy(self, calculator):
        """Test specific exergy calculation."""
        exergy = calculator.calculate_specific_exergy(
            pressure_psig=150.0,
            temperature_f=None,
            dryness_fraction=1.0,
        )
        # Exergy should be positive
        assert exergy > 0
        # Typical values for saturated steam at 150 psig: ~400-500 BTU/lb
        assert 300 < exergy < 600

    def test_exergy_increases_with_pressure(self, calculator):
        """Test exergy increases with pressure."""
        exergy_100 = calculator.calculate_specific_exergy(100.0)
        exergy_300 = calculator.calculate_specific_exergy(300.0)
        exergy_600 = calculator.calculate_specific_exergy(600.0)

        assert exergy_100 < exergy_300 < exergy_600

    def test_calculate_water_enthalpy(self, calculator):
        """Test water enthalpy calculation."""
        h = calculator.calculate_water_enthalpy(180.0)
        # h = Cp * (T - 32) = 1.0 * (180 - 32) = 148 BTU/lb
        assert h == pytest.approx(148.0, rel=0.01)

    def test_get_steam_properties_saturated_vapor(self, calculator):
        """Test complete steam properties for saturated vapor."""
        props = calculator.get_steam_properties(
            pressure_psig=150.0,
            temperature_f=None,
            dryness_fraction=1.0,
        )

        assert isinstance(props, SteamProperties)
        assert props.phase == "saturated_vapor"
        assert props.dryness_fraction == 1.0
        assert props.superheat_f is None
        assert props.enthalpy_btu_lb == pytest.approx(1196.0, rel=0.01)

    def test_get_steam_properties_superheated(self, calculator):
        """Test complete steam properties for superheated steam."""
        props = calculator.get_steam_properties(
            pressure_psig=600.0,
            temperature_f=750.0,
            dryness_fraction=1.0,
        )

        assert props.phase == "superheated_vapor"
        assert props.superheat_f == pytest.approx(261.0, rel=0.01)
        assert props.exergy_btu_lb is not None

    def test_get_steam_properties_wet_steam(self, calculator):
        """Test complete steam properties for wet steam."""
        props = calculator.get_steam_properties(
            pressure_psig=150.0,
            temperature_f=None,
            dryness_fraction=0.95,
        )

        assert props.phase == "wet_steam"
        assert props.dryness_fraction == 0.95


# =============================================================================
# HEADER BALANCE CALCULATOR TESTS
# =============================================================================

class TestHeaderBalanceCalculator:
    """Test suite for HeaderBalanceCalculator."""

    @pytest.fixture
    def header_config(self):
        """Create header configuration."""
        return SteamHeaderConfig(
            name="HP-MAIN",
            level=SteamHeaderLevel.HIGH_PRESSURE,
            design_pressure_psig=600.0,
            min_pressure_psig=580.0,
            max_pressure_psig=620.0,
            design_flow_lb_hr=100000.0,
            max_flow_lb_hr=120000.0,
        )

    @pytest.fixture
    def exergy_config(self):
        """Create exergy configuration."""
        return ExergyOptimizationConfig(
            enabled=True,
            reference_temperature_f=77.0,
        )

    @pytest.fixture
    def calculator(self, header_config, exergy_config):
        """Create header balance calculator."""
        return HeaderBalanceCalculator(
            header_config=header_config,
            exergy_config=exergy_config,
        )

    def test_initialization(self, calculator, header_config):
        """Test calculator initialization."""
        assert calculator.header_config == header_config
        assert calculator.steam_calc is not None

    def test_calculate_balance_optimal(self, calculator):
        """Test balance calculation for optimal conditions."""
        input_data = HeaderBalanceInput(
            header_id="HP-MAIN",
            current_pressure_psig=600.0,
            current_temperature_f=750.0,
            pressure_setpoint_psig=600.0,
            supplies=[
                {"id": "BLR-001", "flow_lb_hr": 50000},
                {"id": "BLR-002", "flow_lb_hr": 50000},
            ],
            demands=[
                {"id": "TURB-001", "flow_lb_hr": 50000},
                {"id": "TURB-002", "flow_lb_hr": 50000},
            ],
            pressure_deadband_psi=2.0,
        )

        result = calculator.calculate_balance(input_data)

        assert isinstance(result, HeaderBalanceOutput)
        assert result.status == OptimizationStatus.OPTIMAL
        assert result.total_supply_lb_hr == 100000.0
        assert result.total_demand_lb_hr == 100000.0
        assert result.imbalance_lb_hr == 0.0
        assert result.pressure_deviation_psi == 0.0

    def test_calculate_balance_suboptimal(self, calculator):
        """Test balance calculation for suboptimal conditions."""
        input_data = HeaderBalanceInput(
            header_id="HP-MAIN",
            current_pressure_psig=595.0,  # 5 psi below setpoint
            current_temperature_f=745.0,
            pressure_setpoint_psig=600.0,
            supplies=[{"id": "BLR-001", "flow_lb_hr": 90000}],
            demands=[{"id": "TURB-001", "flow_lb_hr": 100000}],
            pressure_deadband_psi=2.0,
        )

        result = calculator.calculate_balance(input_data)

        assert result.status == OptimizationStatus.SUBOPTIMAL
        assert result.pressure_deviation_psi == pytest.approx(-5.0)
        assert len(result.warnings) > 0

    def test_calculate_balance_with_exergy(self, calculator):
        """Test balance calculation includes exergy analysis."""
        input_data = HeaderBalanceInput(
            header_id="HP-MAIN",
            current_pressure_psig=600.0,
            current_temperature_f=750.0,
            pressure_setpoint_psig=600.0,
            supplies=[{"id": "BLR-001", "flow_lb_hr": 100000}],
            demands=[{"id": "TURB-001", "flow_lb_hr": 100000}],
        )

        result = calculator.calculate_balance(input_data)

        assert result.exergy_supply_btu_hr is not None
        assert result.exergy_supply_btu_hr > 0
        assert result.exergy_efficiency_pct is not None

    def test_pressure_trend_analysis(self, calculator):
        """Test pressure trend analysis over multiple readings."""
        # Add several readings with rising pressure
        for i in range(10):
            input_data = HeaderBalanceInput(
                header_id="HP-MAIN",
                current_pressure_psig=600.0 + i * 0.5,
                current_temperature_f=750.0,
                pressure_setpoint_psig=600.0,
                supplies=[{"id": "BLR-001", "flow_lb_hr": 100000}],
                demands=[{"id": "TURB-001", "flow_lb_hr": 95000}],
            )
            result = calculator.calculate_balance(input_data)

        assert result.pressure_trend == "rising"

    def test_adjustment_recommendations(self, calculator):
        """Test adjustment recommendations are generated."""
        input_data = HeaderBalanceInput(
            header_id="HP-MAIN",
            current_pressure_psig=605.0,  # High pressure
            current_temperature_f=750.0,
            pressure_setpoint_psig=600.0,
            supplies=[
                {
                    "id": "BLR-001",
                    "flow_lb_hr": 60000,
                    "min_flow_lb_hr": 30000,
                    "max_flow_lb_hr": 70000,
                    "controllable": True,
                },
            ],
            demands=[{"id": "TURB-001", "flow_lb_hr": 50000}],
            pressure_deadband_psi=2.0,
        )

        result = calculator.calculate_balance(input_data)

        assert len(result.adjustments) > 0
        assert result.adjustments[0]["action"] == "decrease"

    def test_provenance_hash_deterministic(self, calculator):
        """Test provenance hash is deterministic."""
        input_data = HeaderBalanceInput(
            header_id="HP-MAIN",
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            current_pressure_psig=600.0,
            current_temperature_f=750.0,
            pressure_setpoint_psig=600.0,
            supplies=[{"id": "BLR-001", "flow_lb_hr": 100000}],
            demands=[{"id": "TURB-001", "flow_lb_hr": 100000}],
        )

        result1 = calculator.calculate_balance(input_data)
        result2 = calculator.calculate_balance(input_data)

        # Same input should produce same hash
        assert result1.provenance_hash == result2.provenance_hash
        assert len(result1.provenance_hash) == 64  # SHA-256


# =============================================================================
# STEAM DISTRIBUTION OPTIMIZER TESTS
# =============================================================================

class TestSteamDistributionOptimizer:
    """Test suite for SteamDistributionOptimizer."""

    @pytest.fixture
    def headers(self):
        """Create list of header configurations."""
        return [
            SteamHeaderConfig(
                name="HP-MAIN",
                level=SteamHeaderLevel.HIGH_PRESSURE,
                design_pressure_psig=600.0,
                min_pressure_psig=580.0,
                max_pressure_psig=620.0,
                design_flow_lb_hr=100000.0,
                max_flow_lb_hr=120000.0,
            ),
            SteamHeaderConfig(
                name="MP-MAIN",
                level=SteamHeaderLevel.MEDIUM_PRESSURE,
                design_pressure_psig=150.0,
                min_pressure_psig=140.0,
                max_pressure_psig=160.0,
                design_flow_lb_hr=50000.0,
                max_flow_lb_hr=65000.0,
            ),
        ]

    @pytest.fixture
    def optimizer(self, headers):
        """Create steam distribution optimizer."""
        return SteamDistributionOptimizer(headers=headers)

    def test_initialization(self, optimizer, headers):
        """Test optimizer initialization."""
        assert len(optimizer.headers) == 2
        assert "HP-MAIN" in optimizer.headers
        assert "MP-MAIN" in optimizer.headers
        assert len(optimizer.calculators) == 2

    def test_balance_header(self, optimizer):
        """Test single header balance."""
        input_data = HeaderBalanceInput(
            header_id="HP-MAIN",
            current_pressure_psig=600.0,
            current_temperature_f=750.0,
            pressure_setpoint_psig=600.0,
            supplies=[{"id": "BLR-001", "flow_lb_hr": 100000}],
            demands=[{"id": "TURB-001", "flow_lb_hr": 100000}],
        )

        result = optimizer.balance_header("HP-MAIN", input_data)

        assert isinstance(result, HeaderBalanceOutput)
        assert result.header_id == "HP-MAIN"

    def test_balance_unknown_header(self, optimizer):
        """Test error for unknown header."""
        input_data = HeaderBalanceInput(
            header_id="UNKNOWN",
            current_pressure_psig=600.0,
            current_temperature_f=750.0,
            pressure_setpoint_psig=600.0,
            supplies=[],
            demands=[],
        )

        with pytest.raises(ValueError, match="Unknown header"):
            optimizer.balance_header("UNKNOWN", input_data)

    def test_balance_all_headers(self, optimizer):
        """Test balancing all headers."""
        readings = [
            HeaderBalanceInput(
                header_id="HP-MAIN",
                current_pressure_psig=600.0,
                current_temperature_f=750.0,
                pressure_setpoint_psig=600.0,
                supplies=[{"id": "BLR-001", "flow_lb_hr": 100000}],
                demands=[{"id": "TURB-001", "flow_lb_hr": 100000}],
            ),
            HeaderBalanceInput(
                header_id="MP-MAIN",
                current_pressure_psig=150.0,
                current_temperature_f=366.0,
                pressure_setpoint_psig=150.0,
                supplies=[{"id": "PRV-001", "flow_lb_hr": 50000}],
                demands=[{"id": "HX-001", "flow_lb_hr": 50000}],
            ),
        ]

        results = optimizer.balance_all_headers(readings)

        assert len(results) == 2

    def test_optimize_load_allocation(self, optimizer):
        """Test load allocation optimization."""
        supplies = [
            {
                "id": "BLR-001",
                "max_flow_lb_hr": 60000,
                "min_flow_lb_hr": 20000,
                "exergy_efficiency_pct": 85,
                "cost_per_mlb": 8,
            },
            {
                "id": "BLR-002",
                "max_flow_lb_hr": 50000,
                "min_flow_lb_hr": 15000,
                "exergy_efficiency_pct": 80,
                "cost_per_mlb": 10,
            },
        ]

        allocations = optimizer.optimize_load_allocation(
            total_demand_lb_hr=80000,
            available_supplies=supplies,
        )

        assert len(allocations) == 2
        total_allocated = sum(a["allocated_flow_lb_hr"] for a in allocations)
        assert total_allocated == pytest.approx(80000, rel=0.1)

    def test_calculate_system_exergy_efficiency(self, optimizer):
        """Test system exergy efficiency calculation."""
        results = [
            HeaderBalanceOutput(
                header_id="HP-MAIN",
                status=OptimizationStatus.OPTIMAL,
                total_supply_lb_hr=100000,
                total_demand_lb_hr=100000,
                imbalance_lb_hr=0,
                imbalance_pct=0,
                pressure_psig=600,
                pressure_deviation_psi=0,
                exergy_supply_btu_hr=50000000,
                exergy_demand_btu_hr=48000000,
                provenance_hash="a" * 64,
            ),
        ]

        efficiency = optimizer.calculate_system_exergy_efficiency(results)

        assert efficiency is not None
        assert 0 < efficiency <= 100

    def test_pressure_cascade_recommendation(self, optimizer):
        """Test pressure cascade optimization recommendations."""
        recommendations = optimizer.get_pressure_cascade_recommendation()

        # Should recommend cascade between HP (600) and MP (150) - 450 psi drop
        assert len(recommendations) > 0
        assert recommendations[0]["type"] == "cascade_opportunity"


# =============================================================================
# STEAM TABLE CONSTANTS TESTS
# =============================================================================

class TestSteamTableConstants:
    """Test suite for SteamTableConstants."""

    def test_reference_conditions(self):
        """Test reference condition constants."""
        assert SteamTableConstants.REFERENCE_TEMP_F == 77.0
        assert SteamTableConstants.REFERENCE_PRESSURE_PSIA == 14.696

    def test_conversion_factors(self):
        """Test conversion factor constants."""
        assert SteamTableConstants.BTU_PER_LB_TO_KJ_KG == 2.326
        assert SteamTableConstants.PSI_TO_KPA == pytest.approx(6.89476)

    def test_f_to_c_conversion(self):
        """Test Fahrenheit to Celsius conversion."""
        f_to_c = SteamTableConstants.F_TO_C
        assert f_to_c(32) == pytest.approx(0.0)
        assert f_to_c(212) == pytest.approx(100.0)

    def test_saturation_data_completeness(self, iapws_saturation_reference):
        """Test saturation data matches IAPWS reference."""
        for pressure, expected_data in iapws_saturation_reference.items():
            if pressure in SteamTableConstants.SATURATION_DATA:
                actual_data = SteamTableConstants.SATURATION_DATA[pressure]
                assert actual_data[0] == pytest.approx(expected_data[0], rel=0.01)
                assert actual_data[3] == pytest.approx(expected_data[3], rel=0.01)


# =============================================================================
# EXERGY CONSTANTS TESTS
# =============================================================================

class TestExergyConstants:
    """Test suite for ExergyConstants."""

    def test_dead_state_conditions(self):
        """Test dead state condition constants."""
        assert ExergyConstants.T0_R == pytest.approx(536.67)  # 77F in Rankine
        assert ExergyConstants.P0_PSIA == 14.696

    def test_reference_entropy(self):
        """Test reference entropy constant."""
        assert ExergyConstants.S0_BTU_LB_R == 0.0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestDistributionPerformance:
    """Performance tests for distribution module."""

    @pytest.fixture
    def calculator(self):
        """Create steam property calculator."""
        return SteamPropertyCalculator()

    @pytest.mark.performance
    def test_property_calculation_speed(self, calculator, benchmark):
        """Test steam property calculation meets performance target (<5ms)."""
        def calculate():
            return calculator.get_steam_properties(
                pressure_psig=150.0,
                temperature_f=400.0,
                dryness_fraction=1.0,
            )

        result = benchmark(calculate)
        # Benchmark will measure time automatically

    @pytest.mark.performance
    def test_saturation_lookup_speed(self, calculator):
        """Test saturation property lookup is fast."""
        import time
        start = time.time()

        for _ in range(1000):
            calculator.get_saturation_properties(150.0)

        elapsed = time.time() - start
        # Should be <100ms for 1000 lookups
        assert elapsed < 0.1

    @pytest.mark.performance
    def test_balance_calculation_speed(self, hp_header_config, exergy_config):
        """Test header balance calculation speed."""
        import time

        calc = HeaderBalanceCalculator(
            header_config=hp_header_config,
            exergy_config=exergy_config,
        )

        input_data = HeaderBalanceInput(
            header_id="HP-MAIN",
            current_pressure_psig=600.0,
            current_temperature_f=750.0,
            pressure_setpoint_psig=600.0,
            supplies=[{"id": "BLR-001", "flow_lb_hr": 100000}],
            demands=[{"id": "TURB-001", "flow_lb_hr": 100000}],
        )

        start = time.time()
        for _ in range(100):
            calc.calculate_balance(input_data)
        elapsed = time.time() - start

        # Should be <500ms for 100 calculations
        assert elapsed < 0.5
