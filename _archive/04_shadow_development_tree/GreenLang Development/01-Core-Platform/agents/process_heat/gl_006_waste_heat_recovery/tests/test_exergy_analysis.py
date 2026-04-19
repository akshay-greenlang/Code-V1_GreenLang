"""
GL-006 WasteHeatRecovery Agent - Exergy Analysis Tests

Comprehensive unit tests for the ExergyAnalyzer class.
Tests second law efficiency, exergy destruction, and thermoeconomics.

Coverage Target: 85%+
"""

import pytest
import math
from datetime import datetime
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_006_waste_heat_recovery.exergy_analysis import (
    ExergyAnalyzer,
    ExergyStream,
    ProcessComponent,
    ComponentType,
    ComponentExergyResult,
    SystemExergyResult,
    DestructionCategory,
    calculate_exergy_efficiency_comparison,
    estimate_improvement_payback,
    DEFAULT_DEAD_STATE_TEMP_F,
    DEFAULT_DEAD_STATE_PRESSURE_PSIA,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analyzer():
    """Create ExergyAnalyzer instance for testing."""
    return ExergyAnalyzer(
        dead_state_temp_f=77.0,
        dead_state_pressure_psia=14.696,
        fuel_exergy_cost_usd_per_mmbtu=8.0,
    )


@pytest.fixture
def hot_inlet_stream():
    """Create hot inlet stream for testing."""
    return ExergyStream(
        name="Hot_In",
        temp_f=800.0,
        pressure_psia=15.0,
        mass_flow_lb_hr=10000.0,
        specific_heat_btu_lb_f=0.25,
        is_inlet=True,
    )


@pytest.fixture
def hot_outlet_stream():
    """Create hot outlet stream for testing."""
    return ExergyStream(
        name="Hot_Out",
        temp_f=300.0,
        pressure_psia=14.7,
        mass_flow_lb_hr=10000.0,
        specific_heat_btu_lb_f=0.25,
        is_inlet=False,
    )


@pytest.fixture
def cold_inlet_stream():
    """Create cold inlet stream for testing."""
    return ExergyStream(
        name="Cold_In",
        temp_f=100.0,
        pressure_psia=50.0,
        mass_flow_lb_hr=15000.0,
        specific_heat_btu_lb_f=1.0,
        is_inlet=True,
    )


@pytest.fixture
def cold_outlet_stream():
    """Create cold outlet stream for testing."""
    return ExergyStream(
        name="Cold_Out",
        temp_f=250.0,
        pressure_psia=48.0,
        mass_flow_lb_hr=15000.0,
        specific_heat_btu_lb_f=1.0,
        is_inlet=False,
    )


@pytest.fixture
def heat_exchanger_component(hot_inlet_stream, hot_outlet_stream):
    """Create heat exchanger component."""
    return ProcessComponent(
        name="Economizer",
        component_type=ComponentType.HEAT_EXCHANGER,
        inlet_streams=["Hot_In"],
        outlet_streams=["Hot_Out"],
        heat_transfer_btu_hr=-500000,  # Heat removed from hot stream
        heat_transfer_temp_f=550.0,
        capital_cost_usd=75000.0,
        operating_hours_yr=8000,
        maintenance_cost_usd_yr=3000.0,
    )


@pytest.fixture
def sample_system_streams():
    """Create sample streams for system analysis."""
    return [
        ExergyStream(
            name="Exhaust_In",
            temp_f=800.0,
            pressure_psia=15.0,
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.25,
            is_inlet=True,
        ),
        ExergyStream(
            name="Exhaust_Out",
            temp_f=350.0,
            pressure_psia=14.7,
            mass_flow_lb_hr=10000.0,
            specific_heat_btu_lb_f=0.25,
            is_inlet=False,
        ),
        ExergyStream(
            name="Water_In",
            temp_f=100.0,
            pressure_psia=60.0,
            mass_flow_lb_hr=8000.0,
            specific_heat_btu_lb_f=1.0,
            is_inlet=True,
        ),
        ExergyStream(
            name="Water_Out",
            temp_f=280.0,
            pressure_psia=55.0,
            mass_flow_lb_hr=8000.0,
            specific_heat_btu_lb_f=1.0,
            is_inlet=False,
        ),
    ]


@pytest.fixture
def sample_components():
    """Create sample components for system analysis."""
    return [
        ProcessComponent(
            name="Economizer_1",
            component_type=ComponentType.HEAT_EXCHANGER,
            inlet_streams=["Exhaust_In", "Water_In"],
            outlet_streams=["Exhaust_Out", "Water_Out"],
            capital_cost_usd=80000.0,
        ),
    ]


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestExergyAnalyzerInitialization:
    """Test ExergyAnalyzer initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test analyzer initializes with defaults."""
        analyzer = ExergyAnalyzer()

        assert analyzer.dead_state_temp_f == DEFAULT_DEAD_STATE_TEMP_F
        assert analyzer.dead_state_pressure == DEFAULT_DEAD_STATE_PRESSURE_PSIA
        assert analyzer.dead_state_temp_r == DEFAULT_DEAD_STATE_TEMP_F + 459.67

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test analyzer with custom parameters."""
        analyzer = ExergyAnalyzer(
            dead_state_temp_f=80.0,
            dead_state_pressure_psia=14.5,
            fuel_exergy_cost_usd_per_mmbtu=10.0,
        )

        assert analyzer.dead_state_temp_f == 80.0
        assert analyzer.dead_state_pressure == 14.5
        assert analyzer.fuel_exergy_cost == 10.0


# =============================================================================
# PHYSICAL EXERGY TESTS
# =============================================================================

class TestPhysicalExergy:
    """Test physical exergy calculations."""

    @pytest.mark.unit
    def test_physical_exergy_above_dead_state(self, analyzer):
        """Test physical exergy for stream above dead state."""
        specific_ex, exergy_rate = analyzer.calculate_physical_exergy(
            temp_f=200.0,
            pressure_psia=14.696,
            mass_flow_lb_hr=1000.0,
            cp_btu_lb_f=0.24,
        )

        # Stream above dead state should have positive exergy
        assert specific_ex > 0
        assert exergy_rate > 0

    @pytest.mark.unit
    def test_physical_exergy_at_dead_state(self, analyzer):
        """Test physical exergy at dead state is zero."""
        specific_ex, exergy_rate = analyzer.calculate_physical_exergy(
            temp_f=77.0,  # Dead state temp
            pressure_psia=14.696,  # Dead state pressure
            mass_flow_lb_hr=1000.0,
            cp_btu_lb_f=0.24,
        )

        # At dead state, exergy should be approximately zero
        assert abs(specific_ex) < 1.0  # Allow small numerical tolerance
        assert abs(exergy_rate) < 1000.0

    @pytest.mark.unit
    def test_physical_exergy_below_dead_state(self, analyzer):
        """Test physical exergy for stream below dead state temp."""
        specific_ex, exergy_rate = analyzer.calculate_physical_exergy(
            temp_f=32.0,  # Below dead state
            pressure_psia=14.696,
            mass_flow_lb_hr=1000.0,
            cp_btu_lb_f=1.0,
        )

        # Stream below dead state still has exergy (can be used for cooling)
        # The specific exergy formula gives positive value for temps both above and below T0
        assert specific_ex != 0

    @pytest.mark.unit
    def test_physical_exergy_high_pressure(self, analyzer):
        """Test physical exergy includes pressure contribution."""
        # At same temperature but different pressures
        _, exergy_low_p = analyzer.calculate_physical_exergy(
            temp_f=200.0,
            pressure_psia=14.696,
            mass_flow_lb_hr=1000.0,
            cp_btu_lb_f=0.24,
        )

        _, exergy_high_p = analyzer.calculate_physical_exergy(
            temp_f=200.0,
            pressure_psia=100.0,  # Higher pressure
            mass_flow_lb_hr=1000.0,
            cp_btu_lb_f=0.24,
        )

        # Higher pressure should give more exergy
        assert exergy_high_p > exergy_low_p


# =============================================================================
# HEAT EXERGY TESTS
# =============================================================================

class TestHeatExergy:
    """Test heat exergy calculations."""

    @pytest.mark.unit
    def test_heat_exergy_high_temp(self, analyzer):
        """Test heat exergy at high temperature."""
        exergy = analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1000000.0,
            temperature_f=500.0,
        )

        # High temperature heat has significant exergy
        # Carnot factor = 1 - T0/T
        temp_r = 500.0 + 459.67
        t0_r = analyzer.dead_state_temp_r
        expected_carnot = 1.0 - t0_r / temp_r

        expected_exergy = 1000000.0 * expected_carnot
        assert exergy == pytest.approx(expected_exergy, rel=0.01)

    @pytest.mark.unit
    def test_heat_exergy_near_dead_state(self, analyzer):
        """Test heat exergy near dead state temperature."""
        exergy = analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1000000.0,
            temperature_f=100.0,  # Close to dead state
        )

        # Low temperature heat has low exergy
        # Carnot factor is small
        assert exergy < 1000000.0 * 0.10  # Less than 10% of heat

    @pytest.mark.unit
    def test_heat_exergy_at_dead_state(self, analyzer):
        """Test heat exergy at dead state is zero."""
        exergy = analyzer.calculate_heat_exergy(
            heat_rate_btu_hr=1000000.0,
            temperature_f=77.0,  # Dead state
        )

        # At dead state, exergy of heat is zero
        assert exergy == pytest.approx(0.0, abs=100)


# =============================================================================
# CARNOT EFFICIENCY TESTS
# =============================================================================

class TestCarnotEfficiency:
    """Test Carnot efficiency calculation."""

    @pytest.mark.unit
    def test_carnot_efficiency_formula(self, analyzer):
        """Test Carnot efficiency formula."""
        efficiency = analyzer.calculate_carnot_efficiency(
            temp_hot_f=500.0,
            temp_cold_f=100.0,
        )

        # eta_carnot = 1 - Tc/Th (in absolute temperatures)
        t_hot_r = 500.0 + 459.67
        t_cold_r = 100.0 + 459.67
        expected = 1.0 - t_cold_r / t_hot_r

        assert efficiency == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_carnot_efficiency_range(self, analyzer):
        """Test Carnot efficiency is between 0 and 1."""
        efficiency = analyzer.calculate_carnot_efficiency(
            temp_hot_f=1000.0,
            temp_cold_f=100.0,
        )

        assert 0.0 < efficiency < 1.0

    @pytest.mark.unit
    def test_carnot_efficiency_equal_temps(self, analyzer):
        """Test Carnot efficiency with equal temperatures."""
        efficiency = analyzer.calculate_carnot_efficiency(
            temp_hot_f=200.0,
            temp_cold_f=200.0,
        )

        assert efficiency == 0.0

    @pytest.mark.unit
    def test_carnot_efficiency_inverted_temps(self, analyzer):
        """Test Carnot efficiency with inverted temperatures."""
        efficiency = analyzer.calculate_carnot_efficiency(
            temp_hot_f=100.0,
            temp_cold_f=200.0,  # Hot < Cold
        )

        assert efficiency == 0.0


# =============================================================================
# COMPONENT ANALYSIS TESTS
# =============================================================================

class TestComponentAnalysis:
    """Test component-level exergy analysis."""

    @pytest.mark.unit
    def test_component_exergy_result(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test component exergy analysis returns result."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert len(result.component_results) == 1
        assert isinstance(result.component_results[0], ComponentExergyResult)

    @pytest.mark.unit
    def test_component_exergetic_efficiency(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test component exergetic efficiency is calculated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        component = result.component_results[0]

        # Efficiency should be between 0 and 100%
        assert 0 <= component.exergetic_efficiency_pct <= 100

    @pytest.mark.unit
    def test_component_exergy_destruction(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test component exergy destruction is calculated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        component = result.component_results[0]

        # Destruction should be non-negative
        assert component.exergy_destruction_btu_hr >= 0

    @pytest.mark.unit
    def test_component_improvement_potential(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test component improvement potential is calculated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        component = result.component_results[0]

        # Improvement potential should be non-negative
        assert component.improvement_potential_btu_hr >= 0


# =============================================================================
# SYSTEM ANALYSIS TESTS
# =============================================================================

class TestSystemAnalysis:
    """Test system-level exergy analysis."""

    @pytest.mark.unit
    def test_system_analysis_returns_result(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test system analysis returns valid result."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert isinstance(result, SystemExergyResult)
        assert result.analysis_id is not None

    @pytest.mark.unit
    def test_system_second_law_efficiency(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test system second law efficiency is calculated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        # Second law efficiency should be between 0 and 100%
        assert 0 <= result.second_law_efficiency_pct <= 100

    @pytest.mark.unit
    def test_system_exergy_totals(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test system exergy totals are calculated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert result.total_exergy_input_btu_hr >= 0
        assert result.total_exergy_output_btu_hr >= 0
        assert result.total_exergy_destruction_btu_hr >= 0

    @pytest.mark.unit
    def test_system_destruction_ranking(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test destruction ranking is generated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert len(result.destruction_ranking) == len(sample_components)
        assert result.destruction_ranking[0]["rank"] == 1

    @pytest.mark.unit
    def test_system_improvement_ranking(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test improvement ranking is generated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert len(result.improvement_ranking) == len(sample_components)


# =============================================================================
# THERMOECONOMIC TESTS
# =============================================================================

class TestThermoeconomics:
    """Test thermoeconomic analysis."""

    @pytest.mark.unit
    def test_destruction_cost_calculated(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test exergy destruction cost is calculated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        # Total destruction cost should be non-negative
        assert result.total_exergy_destruction_cost_usd_yr >= 0

    @pytest.mark.unit
    def test_exergoeconomic_factor(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test exergoeconomic factor is calculated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        component = result.component_results[0]

        # Exergoeconomic factor should be between 0 and 100%
        assert 0 <= component.exergoeconomic_factor_pct <= 100


# =============================================================================
# AVOIDABLE/UNAVOIDABLE DESTRUCTION TESTS
# =============================================================================

class TestDestructionCategories:
    """Test avoidable/unavoidable destruction categorization."""

    @pytest.mark.unit
    def test_destruction_split(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test destruction is split into avoidable and unavoidable."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        component = result.component_results[0]

        # Sum should equal total destruction
        total = (
            component.avoidable_destruction_btu_hr +
            component.unavoidable_destruction_btu_hr
        )
        assert total == pytest.approx(
            component.exergy_destruction_btu_hr, rel=0.01
        )

    @pytest.mark.unit
    def test_unavoidable_fraction_by_type(self, analyzer):
        """Test unavoidable fraction varies by component type."""
        # Heat exchanger
        hx_fraction = analyzer._get_unavoidable_fraction(ComponentType.HEAT_EXCHANGER)
        assert 0 < hx_fraction < 1

        # Valve (more irreversible)
        valve_fraction = analyzer._get_unavoidable_fraction(ComponentType.VALVE)
        assert valve_fraction > hx_fraction


# =============================================================================
# RECOMMENDATION TESTS
# =============================================================================

class TestRecommendations:
    """Test recommendation generation."""

    @pytest.mark.unit
    def test_system_recommendations_generated(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test system recommendations are generated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert isinstance(result.system_recommendations, list)
        assert len(result.system_recommendations) > 0

    @pytest.mark.unit
    def test_component_recommendations_generated(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test component recommendations are generated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        # Components with low efficiency should have recommendations
        for component in result.component_results:
            assert isinstance(component.recommendations, list)


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Test provenance tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test provenance hash is generated."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    @pytest.mark.unit
    def test_calculation_method_recorded(
        self, analyzer, sample_system_streams, sample_components
    ):
        """Test calculation method is recorded."""
        result = analyzer.analyze_system(sample_system_streams, sample_components)

        assert result.calculation_method is not None
        assert "exergy" in result.calculation_method.lower()


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.unit
    def test_efficiency_comparison(self):
        """Test efficiency comparison function."""
        result = calculate_exergy_efficiency_comparison(
            first_law_efficiency=0.40,
            hot_temp_f=1000.0,
            cold_temp_f=100.0,
        )

        assert "first_law_efficiency_pct" in result
        assert "carnot_efficiency_pct" in result
        assert "second_law_efficiency_pct" in result

        # Second law efficiency = first law / Carnot
        assert result["second_law_efficiency_pct"] > result["first_law_efficiency_pct"]

    @pytest.mark.unit
    def test_improvement_payback(self):
        """Test improvement payback estimation."""
        result = estimate_improvement_payback(
            current_destruction_btu_hr=1000000.0,
            improvement_fraction=0.30,  # 30% reduction
            fuel_cost_usd_per_mmbtu=8.0,
            retrofit_cost_usd=50000.0,
            operating_hours_yr=8000,
        )

        assert "exergy_reduction_btu_hr" in result
        assert "annual_savings_usd" in result
        assert "simple_payback_years" in result
        assert "recommendation" in result

        # Verify calculation
        expected_reduction = 1000000.0 * 0.30
        assert result["exergy_reduction_btu_hr"] == pytest.approx(
            expected_reduction, rel=0.01
        )


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_empty_streams(self, analyzer):
        """Test with empty streams list."""
        result = analyzer.analyze_system([], [])

        assert result.total_exergy_input_btu_hr == 0
        assert result.total_exergy_destruction_btu_hr == 0

    @pytest.mark.unit
    def test_component_without_streams(self, analyzer):
        """Test component with no matching streams."""
        streams = [
            ExergyStream(
                name="Stream_A",
                temp_f=200.0,
                mass_flow_lb_hr=1000.0,
            ),
        ]

        components = [
            ProcessComponent(
                name="Component_1",
                component_type=ComponentType.HEAT_EXCHANGER,
                inlet_streams=["Stream_X"],  # Non-existent stream
                outlet_streams=["Stream_Y"],
            ),
        ]

        result = analyzer.analyze_system(streams, components)

        # Should still produce result, just with zero exergy flows
        assert result is not None

    @pytest.mark.unit
    def test_very_high_temperature(self, analyzer):
        """Test with very high temperature stream."""
        specific_ex, exergy_rate = analyzer.calculate_physical_exergy(
            temp_f=2000.0,  # Very high
            pressure_psia=14.696,
            mass_flow_lb_hr=1000.0,
            cp_btu_lb_f=0.24,
        )

        # Should produce valid positive exergy
        assert specific_ex > 0
        assert exergy_rate > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for exergy analysis."""

    @pytest.mark.performance
    def test_analysis_speed_small_system(self, analyzer):
        """Test analysis speed with small system."""
        import time

        streams = [
            ExergyStream(
                name=f"Stream_{i}",
                temp_f=100.0 + i * 50,
                mass_flow_lb_hr=5000.0 + i * 1000,
            )
            for i in range(10)
        ]

        components = [
            ProcessComponent(
                name=f"Component_{i}",
                component_type=ComponentType.HEAT_EXCHANGER,
                inlet_streams=[f"Stream_{i}"],
                outlet_streams=[f"Stream_{i+1}" if i < 9 else "Stream_0"],
            )
            for i in range(5)
        ]

        start = time.time()
        result = analyzer.analyze_system(streams, components)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete in under 1 second

    @pytest.mark.performance
    @pytest.mark.slow
    def test_analysis_speed_large_system(self, analyzer):
        """Test analysis speed with larger system."""
        import time

        streams = [
            ExergyStream(
                name=f"Stream_{i}",
                temp_f=80.0 + (i % 20) * 40,
                mass_flow_lb_hr=3000.0 + i * 100,
            )
            for i in range(50)
        ]

        components = [
            ProcessComponent(
                name=f"Component_{i}",
                component_type=ComponentType.HEAT_EXCHANGER,
                inlet_streams=[f"Stream_{i*2}"],
                outlet_streams=[f"Stream_{i*2+1}"],
            )
            for i in range(25)
        ]

        start = time.time()
        result = analyzer.analyze_system(streams, components)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds
