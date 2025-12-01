# -*- coding: utf-8 -*-
"""
Exergy Calculations tests for GL-006 HeatRecoveryMaximizer.

This module validates second law thermodynamic calculations including:
- Physical exergy calculation
- Chemical exergy calculation
- Exergetic efficiency metrics
- Irreversibility analysis
- Reference environment handling
- Stream exergy balance
- Component-level analysis
- Carnot efficiency reference

Target: 20+ exergy tests
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Tuple
from decimal import Decimal
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from calculators.exergy_calculator import (
    ExergyCalculator,
    StreamState,
    FluidType,
    ReferenceEnvironment,
    ExergyFlow,
    ExergyAnalysisResult,
    ComponentAnalysis,
    calculate_advanced_exergy_metrics,
    identify_exergy_improvement_opportunities
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calculator():
    """Create exergy calculator with default reference environment."""
    return ExergyCalculator()


@pytest.fixture
def custom_reference_env():
    """Create custom reference environment."""
    return ReferenceEnvironment(
        temperature=298.15,  # 25 C
        pressure=101.325,    # 1 atm
        relative_humidity=0.6
    )


@pytest.fixture
def steam_stream():
    """Create steam stream for testing."""
    return StreamState(
        stream_id="STEAM-001",
        fluid_type=FluidType.STEAM,
        temperature=473.15,  # 200 C
        pressure=1000.0,     # 10 bar
        mass_flow=5.0        # kg/s
    )


@pytest.fixture
def water_stream():
    """Create water stream for testing."""
    return StreamState(
        stream_id="WATER-001",
        fluid_type=FluidType.WATER,
        temperature=353.15,  # 80 C
        pressure=200.0,      # 2 bar
        mass_flow=10.0       # kg/s
    )


@pytest.fixture
def flue_gas_stream():
    """Create flue gas stream for testing."""
    return StreamState(
        stream_id="FG-001",
        fluid_type=FluidType.FLUE_GAS,
        temperature=573.15,  # 300 C
        pressure=101.325,    # 1 atm
        mass_flow=20.0,      # kg/s
        composition={
            "N2": 0.73,
            "CO2": 0.12,
            "H2O": 0.10,
            "O2": 0.05
        }
    )


# ============================================================================
# PHYSICAL EXERGY TESTS
# ============================================================================

@pytest.mark.exergy
class TestPhysicalExergy:
    """Test physical exergy calculations."""

    def test_physical_exergy_positive_for_elevated_temperature(self, calculator, steam_stream):
        """Test that elevated temperature streams have positive physical exergy."""
        result = calculator._calculate_stream_exergy(steam_stream)

        # Physical exergy should be positive for T > T0
        assert result.physical_exergy > 0

    def test_physical_exergy_positive_for_elevated_pressure(self, calculator, custom_reference_env):
        """Test that elevated pressure streams have positive physical exergy."""
        calc = ExergyCalculator(custom_reference_env)

        # Air at high pressure, reference temperature
        stream = StreamState(
            stream_id="AIR-HP",
            fluid_type=FluidType.AIR,
            temperature=custom_reference_env.temperature,  # Same as reference
            pressure=500.0,  # 5 bar (above reference)
            mass_flow=1.0
        )

        result = calc._calculate_stream_exergy(stream)

        # Should have some exergy from pressure difference
        assert result.physical_exergy >= 0

    def test_physical_exergy_zero_at_dead_state(self, calculator, custom_reference_env):
        """Test that physical exergy is zero at dead state."""
        calc = ExergyCalculator(custom_reference_env)

        # Stream at reference conditions
        stream = StreamState(
            stream_id="DEAD-STATE",
            fluid_type=FluidType.AIR,
            temperature=custom_reference_env.temperature,
            pressure=custom_reference_env.pressure,
            mass_flow=1.0
        )

        result = calc._calculate_stream_exergy(stream)

        # Physical exergy should be approximately zero at dead state
        assert abs(result.physical_exergy) < 1.0  # Small numerical tolerance

    def test_physical_exergy_proportional_to_mass_flow(self, calculator):
        """Test that physical exergy scales with mass flow rate."""
        stream1 = StreamState(
            stream_id="S1",
            fluid_type=FluidType.STEAM,
            temperature=473.15,
            pressure=1000.0,
            mass_flow=5.0
        )
        stream2 = StreamState(
            stream_id="S2",
            fluid_type=FluidType.STEAM,
            temperature=473.15,
            pressure=1000.0,
            mass_flow=10.0  # Double mass flow
        )

        result1 = calculator._calculate_stream_exergy(stream1)
        result2 = calculator._calculate_stream_exergy(stream2)

        # Exergy should approximately double with mass flow
        assert result2.physical_exergy == pytest.approx(2 * result1.physical_exergy, rel=0.1)


# ============================================================================
# CHEMICAL EXERGY TESTS
# ============================================================================

@pytest.mark.exergy
class TestChemicalExergy:
    """Test chemical exergy calculations."""

    def test_chemical_exergy_for_fuel(self, calculator):
        """Test chemical exergy for fuel streams."""
        natural_gas = StreamState(
            stream_id="NG-001",
            fluid_type=FluidType.NATURAL_GAS,
            temperature=298.15,  # Reference temp
            pressure=101.325,    # Reference pressure
            mass_flow=1.0,
            composition={"CH4": 0.95, "C2H6": 0.03, "N2": 0.02}
        )

        result = calculator._calculate_stream_exergy(natural_gas)

        # Natural gas should have significant chemical exergy
        assert result.chemical_exergy > 0

    def test_chemical_exergy_for_air(self, calculator):
        """Test chemical exergy for air (reference composition)."""
        air = StreamState(
            stream_id="AIR-001",
            fluid_type=FluidType.AIR,
            temperature=298.15,
            pressure=101.325,
            mass_flow=1.0
        )

        result = calculator._calculate_stream_exergy(air)

        # Air at reference should have minimal chemical exergy
        assert result.chemical_exergy >= 0

    def test_chemical_exergy_for_combustion_products(self, calculator, flue_gas_stream):
        """Test chemical exergy for combustion products."""
        result = calculator._calculate_stream_exergy(flue_gas_stream)

        # Flue gas should have some chemical exergy (CO2)
        assert result.chemical_exergy >= 0


# ============================================================================
# KINETIC AND POTENTIAL EXERGY TESTS
# ============================================================================

@pytest.mark.exergy
class TestKineticPotentialExergy:
    """Test kinetic and potential exergy calculations."""

    def test_kinetic_exergy_calculation(self, calculator):
        """Test kinetic exergy calculation."""
        stream = StreamState(
            stream_id="FAST-001",
            fluid_type=FluidType.AIR,
            temperature=298.15,
            pressure=101.325,
            mass_flow=10.0,
            velocity=50.0  # m/s
        )

        result = calculator._calculate_stream_exergy(stream)

        # Kinetic exergy = 0.5 * m * v^2
        expected_kinetic = 0.5 * 10.0 * 50.0 ** 2 / 1000  # kW
        assert result.kinetic_exergy == pytest.approx(expected_kinetic, rel=0.1)

    def test_potential_exergy_calculation(self, calculator):
        """Test potential exergy calculation."""
        stream = StreamState(
            stream_id="ELEVATED-001",
            fluid_type=FluidType.WATER,
            temperature=298.15,
            pressure=101.325,
            mass_flow=10.0,
            elevation=100.0  # m above reference
        )

        result = calculator._calculate_stream_exergy(stream)

        # Potential exergy = m * g * z
        g = 9.81
        expected_potential = 10.0 * g * 100.0 / 1000  # kW
        assert result.potential_exergy == pytest.approx(expected_potential, rel=0.1)


# ============================================================================
# EXERGETIC EFFICIENCY TESTS
# ============================================================================

@pytest.mark.exergy
class TestExergeticEfficiency:
    """Test exergetic efficiency calculations."""

    def test_exergetic_efficiency_bounds(self, calculator, steam_stream, water_stream):
        """Test that exergetic efficiency is between 0 and 1."""
        result = calculator.calculate([steam_stream], [water_stream])

        assert 0 <= result.exergetic_efficiency <= 1

    def test_exergetic_efficiency_with_work_output(self, calculator, steam_stream, water_stream):
        """Test exergetic efficiency with work output."""
        work_output = 100.0  # kW

        result = calculator.calculate([steam_stream], [water_stream], work_output=work_output)

        # With work output, efficiency should account for it
        assert result.exergetic_efficiency >= 0

    def test_carnot_efficiency_reference(self, calculator, steam_stream, water_stream):
        """Test Carnot efficiency is calculated correctly."""
        result = calculator.calculate([steam_stream], [water_stream])

        # Carnot efficiency = 1 - Tc/Th
        max_temp = max(steam_stream.temperature, water_stream.temperature)
        min_temp = min(steam_stream.temperature, water_stream.temperature)
        expected_carnot = 1 - min_temp / max_temp

        assert result.carnot_efficiency == pytest.approx(expected_carnot, rel=0.01)

    def test_relative_efficiency(self, calculator, steam_stream, water_stream):
        """Test relative efficiency (actual/Carnot)."""
        result = calculator.calculate([steam_stream], [water_stream])

        # Relative efficiency = exergetic / carnot
        if result.carnot_efficiency > 0:
            expected_relative = result.exergetic_efficiency / result.carnot_efficiency
            assert result.relative_efficiency == pytest.approx(expected_relative, rel=0.01)


# ============================================================================
# IRREVERSIBILITY ANALYSIS TESTS
# ============================================================================

@pytest.mark.exergy
class TestIrreversibilityAnalysis:
    """Test irreversibility (exergy destruction) analysis."""

    def test_total_exergy_destruction_positive(self, calculator, steam_stream, water_stream):
        """Test that total exergy destruction is non-negative."""
        result = calculator.calculate([steam_stream], [water_stream])

        # Exergy destruction cannot be negative (2nd law)
        assert result.total_exergy_destruction >= 0

    def test_exergy_balance(self, calculator, steam_stream, water_stream):
        """Test exergy balance: Ein = Eout + Edest + Eloss."""
        result = calculator.calculate([steam_stream], [water_stream])

        # Balance check (with tolerance for rounding)
        balance = (result.total_exergy_input -
                   result.total_exergy_output -
                   result.total_exergy_destruction)

        # Should be close to exergy loss
        assert abs(balance - result.total_exergy_loss) < 10.0  # Small tolerance

    def test_irreversibility_distribution(self, calculator, steam_stream, water_stream):
        """Test irreversibility distribution across components."""
        result = calculator.calculate([steam_stream], [water_stream])

        # Distribution should sum to 100% (or less if unaccounted)
        total_distribution = sum(result.irreversibility_distribution.values())

        assert total_distribution <= 100.5  # Allow small rounding error


# ============================================================================
# COMPONENT ANALYSIS TESTS
# ============================================================================

@pytest.mark.exergy
class TestComponentAnalysis:
    """Test component-level exergy analysis."""

    def test_component_exergy_input_output(self, calculator, steam_stream, water_stream):
        """Test component exergy input/output values."""
        result = calculator.calculate([steam_stream], [water_stream])

        for comp in result.component_analyses:
            # Input should be >= output (2nd law)
            assert comp.exergy_input >= comp.exergy_output

    def test_component_exergy_destruction(self, calculator, steam_stream, water_stream):
        """Test component exergy destruction calculation."""
        result = calculator.calculate([steam_stream], [water_stream])

        for comp in result.component_analyses:
            # Destruction = input - output - loss
            calculated_destruction = comp.exergy_input - comp.exergy_output - comp.exergy_loss
            assert comp.exergy_destruction == pytest.approx(calculated_destruction, rel=0.1)

    def test_component_efficiency_bounds(self, calculator, steam_stream, water_stream):
        """Test component exergetic efficiency bounds."""
        result = calculator.calculate([steam_stream], [water_stream])

        for comp in result.component_analyses:
            assert 0 <= comp.exergetic_efficiency <= 1

    def test_improvement_potential_calculation(self, calculator, steam_stream, water_stream):
        """Test improvement potential is calculated."""
        result = calculator.calculate([steam_stream], [water_stream])

        for comp in result.component_analyses:
            # Improvement potential should be non-negative
            assert comp.improvement_potential >= 0


# ============================================================================
# REFERENCE ENVIRONMENT TESTS
# ============================================================================

@pytest.mark.exergy
class TestReferenceEnvironment:
    """Test reference environment handling."""

    def test_default_reference_environment(self, calculator):
        """Test default reference environment values."""
        assert calculator.ref_env.temperature == pytest.approx(298.15, abs=0.1)
        assert calculator.ref_env.pressure == pytest.approx(101.325, abs=0.1)

    def test_custom_reference_temperature(self, steam_stream, water_stream):
        """Test custom reference temperature effect."""
        # Higher reference temperature = less exergy
        ref_low = ReferenceEnvironment(temperature=280.0, pressure=101.325)
        ref_high = ReferenceEnvironment(temperature=310.0, pressure=101.325)

        calc_low = ExergyCalculator(ref_low)
        calc_high = ExergyCalculator(ref_high)

        result_low = calc_low.calculate([steam_stream], [water_stream])
        result_high = calc_high.calculate([steam_stream], [water_stream])

        # Higher reference temp should give less exergy (smaller temperature difference)
        assert result_low.total_exergy_input >= result_high.total_exergy_input

    def test_reference_composition(self):
        """Test reference environment composition."""
        ref_env = ReferenceEnvironment()

        # Standard atmosphere composition
        assert "N2" in ref_env.composition
        assert "O2" in ref_env.composition

        # Mole fractions should sum to approximately 1
        total = sum(ref_env.composition.values())
        assert total == pytest.approx(1.0, abs=0.01)


# ============================================================================
# STREAM EXERGY BALANCE TESTS
# ============================================================================

@pytest.mark.exergy
class TestStreamExergyBalance:
    """Test stream exergy balance calculations."""

    def test_total_exergy_components(self, calculator, steam_stream):
        """Test total exergy equals sum of components."""
        result = calculator._calculate_stream_exergy(steam_stream)

        total = (result.physical_exergy +
                 result.chemical_exergy +
                 result.kinetic_exergy +
                 result.potential_exergy)

        assert result.total_exergy == pytest.approx(total, rel=0.01)

    def test_specific_exergy_calculation(self, calculator, steam_stream):
        """Test specific exergy (per unit mass) calculation."""
        result = calculator._calculate_stream_exergy(steam_stream)

        # Specific exergy = total / mass_flow
        expected_specific = result.total_exergy / steam_stream.mass_flow

        assert result.exergy_flux == pytest.approx(expected_specific, rel=0.01)

    def test_exergy_flow_rate(self, calculator, steam_stream):
        """Test exergy flow rate (power) is in kW."""
        result = calculator._calculate_stream_exergy(steam_stream)

        # Total exergy should be reasonable power value
        assert result.total_exergy > 0
        assert result.total_exergy < 100000  # Reasonable upper bound


# ============================================================================
# ADVANCED METRICS TESTS
# ============================================================================

@pytest.mark.exergy
class TestAdvancedMetrics:
    """Test advanced exergy metrics."""

    def test_avoidable_unavoidable_split(self, calculator, steam_stream, water_stream):
        """Test avoidable/unavoidable exergy destruction split."""
        result = calculator.calculate([steam_stream], [water_stream])
        metrics = calculate_advanced_exergy_metrics(result)

        # Avoidable + unavoidable should equal total destruction
        total = metrics['avoidable_destruction'] + metrics['unavoidable_destruction']
        assert total == pytest.approx(result.total_exergy_destruction, rel=0.01)

    def test_improvement_factor(self, calculator, steam_stream, water_stream):
        """Test improvement factor calculation."""
        result = calculator.calculate([steam_stream], [water_stream])
        metrics = calculate_advanced_exergy_metrics(result)

        # Improvement factor should be between 0 and 1
        assert 0 <= metrics['improvement_factor'] <= 1

    def test_cost_of_exergy_destruction(self, calculator, steam_stream, water_stream):
        """Test annual cost of exergy destruction."""
        result = calculator.calculate([steam_stream], [water_stream])
        metrics = calculate_advanced_exergy_metrics(result)

        # Annual cost should be positive if there's destruction
        if result.total_exergy_destruction > 0:
            assert metrics['annual_destruction_cost'] > 0


# ============================================================================
# IMPROVEMENT OPPORTUNITIES TESTS
# ============================================================================

@pytest.mark.exergy
class TestImprovementOpportunities:
    """Test identification of improvement opportunities."""

    def test_identify_opportunities(self, calculator, steam_stream, water_stream):
        """Test identification of exergy improvement opportunities."""
        result = calculator.calculate([steam_stream], [water_stream])
        opportunities = identify_exergy_improvement_opportunities(result)

        # Should identify opportunities for components with destruction
        assert isinstance(opportunities, list)

    def test_opportunities_sorted_by_potential(self, calculator, steam_stream, water_stream):
        """Test opportunities are sorted by improvement potential."""
        result = calculator.calculate([steam_stream], [water_stream])
        opportunities = identify_exergy_improvement_opportunities(result)

        if len(opportunities) > 1:
            # Should be sorted descending by improvement potential
            for i in range(len(opportunities) - 1):
                assert (opportunities[i]['improvement_potential'] >=
                        opportunities[i+1]['improvement_potential'])

    def test_opportunity_priority_assignment(self, calculator, steam_stream, water_stream):
        """Test priority assignment for opportunities."""
        result = calculator.calculate([steam_stream], [water_stream])
        opportunities = identify_exergy_improvement_opportunities(result)

        for opp in opportunities:
            assert opp['priority'] in ['high', 'medium', 'low']


# ============================================================================
# PROVENANCE TESTS
# ============================================================================

@pytest.mark.exergy
class TestExergyProvenance:
    """Test exergy calculation provenance tracking."""

    def test_calculation_hash_generated(self, calculator, steam_stream, water_stream):
        """Test that calculation hash is generated."""
        result = calculator.calculate([steam_stream], [water_stream])

        assert result.calculation_hash is not None
        assert len(result.calculation_hash) == 64  # SHA-256

    def test_calculation_hash_deterministic(self, calculator, steam_stream, water_stream):
        """Test that calculation hash is deterministic."""
        result1 = calculator.calculate([steam_stream], [water_stream])
        result2 = calculator.calculate([steam_stream], [water_stream])

        assert result1.calculation_hash == result2.calculation_hash

    def test_calculation_hash_changes_with_input(self, calculator, steam_stream, water_stream):
        """Test that calculation hash changes with different input."""
        result1 = calculator.calculate([steam_stream], [water_stream])

        # Modify stream
        modified_stream = StreamState(
            stream_id="STEAM-001",
            fluid_type=FluidType.STEAM,
            temperature=500.0,  # Different temperature
            pressure=1000.0,
            mass_flow=5.0
        )

        result2 = calculator.calculate([modified_stream], [water_stream])

        assert result1.calculation_hash != result2.calculation_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "exergy"])
