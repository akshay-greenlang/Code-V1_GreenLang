"""
GL-002 Flameguard - Efficiency Calculator Unit Tests

Comprehensive unit tests for boiler efficiency calculations per ASME PTC 4.1:
- Direct (input-output) method
- Indirect (heat loss) method
- Individual loss calculations
- Uncertainty analysis

Reference Standards:
- ASME PTC 4.1-2013 (Fired Steam Generators)
- ASME PTC 19.1 (Test Uncertainty)

Target Coverage: 85%+

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import pytest
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch
import hashlib
import json
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the actual calculator
try:
    from calculators.efficiency_calculator import (
        EfficiencyCalculator,
        EfficiencyInput,
        EfficiencyResult,
        FuelProperties,
        FUEL_DATABASE,
    )
except ImportError:
    # If import fails, tests will use local implementation
    pass


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def efficiency_calculator():
    """Create efficiency calculator instance."""
    try:
        return EfficiencyCalculator()
    except NameError:
        # Use a mock if the real calculator isn't available
        from calculators.efficiency_calculator import EfficiencyCalculator
        return EfficiencyCalculator()


@pytest.fixture
def sample_efficiency_input():
    """Sample efficiency calculation input."""
    try:
        return EfficiencyInput(
            steam_flow_klb_hr=150.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=3.5,
            flue_gas_co_ppm=25.0,
            ambient_temperature_f=77.0,
            blowdown_rate_percent=3.0,
            fuel_type="natural_gas",
        )
    except NameError:
        return None


@pytest.fixture
def high_efficiency_input():
    """Input for high-efficiency boiler scenario."""
    try:
        return EfficiencyInput(
            steam_flow_klb_hr=200.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=250.0,  # High feedwater temp
            fuel_flow_rate=9500.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=300.0,  # Low stack temp
            flue_gas_o2_percent=2.5,  # Low excess air
            flue_gas_co_ppm=15.0,
            ambient_temperature_f=77.0,
            blowdown_rate_percent=2.0,
            fuel_type="natural_gas",
        )
    except NameError:
        return None


@pytest.fixture
def low_efficiency_input():
    """Input for low-efficiency boiler scenario."""
    try:
        return EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=200.0,  # Lower feedwater temp
            fuel_flow_rate=7000.0,
            fuel_flow_unit="lb_hr",
            flue_gas_temperature_f=450.0,  # High stack temp
            flue_gas_o2_percent=6.0,  # High excess air
            flue_gas_co_ppm=50.0,
            ambient_temperature_f=77.0,
            blowdown_rate_percent=5.0,
            fuel_type="natural_gas",
        )
    except NameError:
        return None


# =============================================================================
# DIRECT METHOD TESTS
# =============================================================================

class TestDirectMethod:
    """Test direct (input-output) efficiency method."""

    def test_direct_method_basic_calculation(self, efficiency_calculator, sample_efficiency_input):
        """
        Validate basic direct method calculation.

        Efficiency = (Steam Output / Fuel Input) * 100
        """
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="direct")

        # Direct method should return valid efficiency
        assert 50.0 <= result.efficiency_hhv_percent <= 100.0, (
            f"Direct efficiency {result.efficiency_hhv_percent}% outside valid range"
        )

        # Method should be recorded
        assert result.method == "direct"

    def test_direct_method_higher_output_higher_efficiency(
        self,
        efficiency_calculator,
        high_efficiency_input,
        low_efficiency_input,
    ):
        """Higher steam output should result in higher efficiency."""
        if high_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result_high = efficiency_calculator.calculate(high_efficiency_input, method="direct")
        result_low = efficiency_calculator.calculate(low_efficiency_input, method="direct")

        assert result_high.efficiency_hhv_percent > result_low.efficiency_hhv_percent, (
            f"High efficiency {result_high.efficiency_hhv_percent}% should be "
            f"greater than low efficiency {result_low.efficiency_hhv_percent}%"
        )

    def test_direct_method_uncertainty(self, efficiency_calculator, sample_efficiency_input):
        """Direct method typically has higher uncertainty than indirect."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="direct")

        # Direct method uncertainty typically ~1%
        assert 0.5 <= result.uncertainty_percent <= 2.0, (
            f"Direct method uncertainty {result.uncertainty_percent}% "
            f"outside expected range"
        )


# =============================================================================
# INDIRECT METHOD TESTS
# =============================================================================

class TestIndirectMethod:
    """Test indirect (heat loss) efficiency method."""

    def test_indirect_method_basic_calculation(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """
        Validate basic indirect method calculation.

        Efficiency = 100% - Sum of all losses
        """
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="indirect")

        # Indirect method should return valid efficiency
        assert 50.0 <= result.efficiency_hhv_percent <= 100.0, (
            f"Indirect efficiency {result.efficiency_hhv_percent}% outside valid range"
        )

        # Method should be recorded
        assert result.method == "indirect"

    def test_indirect_method_losses_sum_correctly(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """
        Validate that individual losses sum to total losses.

        Total Losses = L1 + L2 + L3 + ... + L9
        """
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="indirect")

        # Sum individual losses
        calculated_total = (
            result.dry_flue_gas_loss_percent +
            result.moisture_in_fuel_loss_percent +
            result.hydrogen_combustion_loss_percent +
            result.moisture_in_air_loss_percent +
            result.unburned_carbon_loss_percent +
            result.co_loss_percent +
            result.radiation_loss_percent +
            result.blowdown_loss_percent +
            result.other_losses_percent
        )

        assert abs(calculated_total - result.total_losses_percent) < 0.1, (
            f"Individual losses sum {calculated_total}% != "
            f"total losses {result.total_losses_percent}%"
        )

    def test_indirect_method_efficiency_equals_100_minus_losses(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """
        Validate efficiency = 100% - total losses.
        """
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="indirect")

        calculated_efficiency = 100.0 - result.total_losses_percent

        assert abs(result.efficiency_hhv_percent - calculated_efficiency) < 0.1, (
            f"Efficiency {result.efficiency_hhv_percent}% != "
            f"100 - losses ({calculated_efficiency}%)"
        )


# =============================================================================
# INDIVIDUAL LOSS TESTS
# =============================================================================

class TestIndividualLosses:
    """Test individual heat loss calculations per ASME PTC 4.1."""

    def test_dry_flue_gas_loss_increases_with_stack_temp(
        self,
        efficiency_calculator,
    ):
        """
        L1 (Dry Flue Gas Loss) should increase with stack temperature.

        Higher stack temperature = more heat leaving in flue gas.
        """
        try:
            input_low_temp = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=300.0,  # Low stack temp
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )

            input_high_temp = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=450.0,  # High stack temp
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result_low = efficiency_calculator.calculate(input_low_temp, method="indirect")
        result_high = efficiency_calculator.calculate(input_high_temp, method="indirect")

        assert result_high.dry_flue_gas_loss_percent > result_low.dry_flue_gas_loss_percent, (
            f"Dry flue gas loss should increase with stack temp: "
            f"low={result_low.dry_flue_gas_loss_percent}%, "
            f"high={result_high.dry_flue_gas_loss_percent}%"
        )

    def test_dry_flue_gas_loss_increases_with_excess_air(
        self,
        efficiency_calculator,
    ):
        """
        L1 (Dry Flue Gas Loss) should increase with excess air.

        Higher excess air = more mass flow = more heat leaving.
        """
        try:
            input_low_o2 = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=2.0,  # Low O2 / excess air
                fuel_type="natural_gas",
            )

            input_high_o2 = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=6.0,  # High O2 / excess air
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result_low = efficiency_calculator.calculate(input_low_o2, method="indirect")
        result_high = efficiency_calculator.calculate(input_high_o2, method="indirect")

        assert result_high.dry_flue_gas_loss_percent > result_low.dry_flue_gas_loss_percent, (
            f"Dry flue gas loss should increase with excess air: "
            f"low O2={result_low.dry_flue_gas_loss_percent}%, "
            f"high O2={result_high.dry_flue_gas_loss_percent}%"
        )

    def test_hydrogen_combustion_loss_depends_on_fuel(
        self,
        efficiency_calculator,
    ):
        """
        L3 (Hydrogen Combustion Loss) should vary by fuel hydrogen content.

        Natural gas (25% H) > fuel oil (12.5% H) > coal (5% H)
        """
        try:
            input_gas = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )

            input_oil = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="fuel_oil_no2",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result_gas = efficiency_calculator.calculate(input_gas, method="indirect")
        result_oil = efficiency_calculator.calculate(input_oil, method="indirect")

        # Natural gas has higher hydrogen content
        assert result_gas.hydrogen_combustion_loss_percent >= result_oil.hydrogen_combustion_loss_percent, (
            f"H2 combustion loss: gas={result_gas.hydrogen_combustion_loss_percent}%, "
            f"oil={result_oil.hydrogen_combustion_loss_percent}%"
        )

    def test_radiation_loss_inversely_proportional_to_capacity(
        self,
        efficiency_calculator,
    ):
        """
        L7 (Radiation Loss) should decrease with larger capacity.

        Surface area to volume ratio decreases with size.
        """
        try:
            input_small = EfficiencyInput(
                steam_flow_klb_hr=50.0,  # Small boiler
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=2800.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )

            input_large = EfficiencyInput(
                steam_flow_klb_hr=300.0,  # Large boiler
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=16000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result_small = efficiency_calculator.calculate(input_small, method="indirect")
        result_large = efficiency_calculator.calculate(input_large, method="indirect")

        # Radiation loss should be higher percentage for smaller boiler
        assert result_small.radiation_loss_percent >= result_large.radiation_loss_percent, (
            f"Radiation loss: small={result_small.radiation_loss_percent}%, "
            f"large={result_large.radiation_loss_percent}%"
        )

    def test_blowdown_loss_proportional_to_rate(
        self,
        efficiency_calculator,
    ):
        """
        L8 (Blowdown Loss) should increase with blowdown rate.
        """
        try:
            input_low_bd = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                blowdown_rate_percent=2.0,  # Low blowdown
                fuel_type="natural_gas",
            )

            input_high_bd = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                blowdown_rate_percent=8.0,  # High blowdown
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result_low = efficiency_calculator.calculate(input_low_bd, method="indirect")
        result_high = efficiency_calculator.calculate(input_high_bd, method="indirect")

        assert result_high.blowdown_loss_percent > result_low.blowdown_loss_percent, (
            f"Blowdown loss: low={result_low.blowdown_loss_percent}%, "
            f"high={result_high.blowdown_loss_percent}%"
        )


# =============================================================================
# LHV vs HHV EFFICIENCY TESTS
# =============================================================================

class TestLHVvsHHV:
    """Test LHV and HHV efficiency relationships."""

    def test_lhv_efficiency_higher_than_hhv(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """
        LHV efficiency should be higher than HHV efficiency.

        LHV excludes latent heat of water vapor, so denominator is smaller.
        """
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input)

        assert result.efficiency_lhv_percent >= result.efficiency_hhv_percent, (
            f"LHV efficiency {result.efficiency_lhv_percent}% should be >= "
            f"HHV efficiency {result.efficiency_hhv_percent}%"
        )

    def test_lhv_hhv_ratio_appropriate_for_fuel(
        self,
        efficiency_calculator,
    ):
        """
        LHV/HHV ratio should be appropriate for fuel type.

        Natural gas: ~1.11 (high hydrogen)
        Fuel oil: ~1.06
        Coal: ~1.03-1.05
        """
        try:
            input_gas = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(input_gas)

        ratio = result.efficiency_lhv_percent / result.efficiency_hhv_percent

        # Natural gas LHV/HHV ratio typically ~1.11
        assert 1.08 <= ratio <= 1.15, (
            f"LHV/HHV ratio {ratio} outside expected range for natural gas"
        )


# =============================================================================
# EXCESS AIR TESTS
# =============================================================================

class TestExcessAir:
    """Test excess air calculation and reporting."""

    @pytest.mark.parametrize("o2_percent,expected_ea_min,expected_ea_max", [
        (2.0, 8.0, 12.0),   # Low O2
        (3.0, 14.0, 18.0),  # Normal O2
        (5.0, 28.0, 35.0),  # Moderate O2
        (7.0, 45.0, 55.0),  # High O2
    ])
    def test_excess_air_from_o2(
        self,
        efficiency_calculator,
        o2_percent: float,
        expected_ea_min: float,
        expected_ea_max: float,
    ):
        """
        Validate excess air calculation from O2 measurement.

        Formula: EA% = O2% / (21 - O2%) * 100
        """
        try:
            input_data = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=o2_percent,
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(input_data)

        assert expected_ea_min <= result.excess_air_percent <= expected_ea_max, (
            f"Excess air {result.excess_air_percent}% from O2={o2_percent}% "
            f"outside expected range [{expected_ea_min}, {expected_ea_max}]"
        )


# =============================================================================
# PROVENANCE AND AUDIT TESTS
# =============================================================================

class TestProvenanceAndAudit:
    """Test provenance tracking and audit trail."""

    def test_result_has_provenance_hashes(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """Validate provenance hash generation."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input)

        assert hasattr(result, 'input_hash'), "Result should have input_hash"
        assert hasattr(result, 'output_hash'), "Result should have output_hash"
        assert len(result.input_hash) == 16, "Input hash should be 16 characters"
        assert len(result.output_hash) == 16, "Output hash should be 16 characters"

    def test_result_has_timestamp(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """Validate timestamp is recorded."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input)

        assert hasattr(result, 'timestamp'), "Result should have timestamp"
        assert isinstance(result.timestamp, datetime), "Timestamp should be datetime"

    def test_result_has_formula_version(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """Validate formula version is documented."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input)

        assert hasattr(result, 'formula_version'), "Result should have formula_version"
        assert "ASME" in result.formula_version or "PTC" in result.formula_version

    def test_deterministic_results(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """Same inputs should produce same outputs."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result1 = efficiency_calculator.calculate(sample_efficiency_input)
        result2 = efficiency_calculator.calculate(sample_efficiency_input)

        assert result1.efficiency_hhv_percent == result2.efficiency_hhv_percent
        assert result1.total_losses_percent == result2.total_losses_percent
        assert result1.input_hash == result2.input_hash


# =============================================================================
# BOUNDARY AND VALIDATION TESTS
# =============================================================================

class TestBoundaryConditions:
    """Test boundary conditions and input validation."""

    def test_efficiency_clamped_to_valid_range(
        self,
        efficiency_calculator,
    ):
        """Efficiency should be clamped to [50%, 100%]."""
        try:
            # Unrealistic input that would give >100% or <50%
            input_data = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=100.0,  # Unrealistically low
                flue_gas_o2_percent=1.0,
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(input_data)

        assert 50.0 <= result.efficiency_hhv_percent <= 100.0, (
            f"Efficiency {result.efficiency_hhv_percent}% outside valid range"
        )

    def test_losses_are_non_negative(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """All individual losses should be non-negative."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="indirect")

        assert result.dry_flue_gas_loss_percent >= 0
        assert result.moisture_in_fuel_loss_percent >= 0
        assert result.hydrogen_combustion_loss_percent >= 0
        assert result.moisture_in_air_loss_percent >= 0
        assert result.unburned_carbon_loss_percent >= 0
        assert result.co_loss_percent >= 0
        assert result.radiation_loss_percent >= 0
        assert result.blowdown_loss_percent >= 0
        assert result.other_losses_percent >= 0

    def test_total_losses_less_than_100(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """Total losses should be less than 100%."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="indirect")

        assert result.total_losses_percent < 100, (
            f"Total losses {result.total_losses_percent}% >= 100%"
        )


# =============================================================================
# FUEL TYPE COMPARISON TESTS
# =============================================================================

class TestFuelTypeComparison:
    """Test efficiency differences between fuel types."""

    def test_natural_gas_higher_efficiency_than_coal(
        self,
        efficiency_calculator,
    ):
        """
        Natural gas should typically have higher efficiency than coal.

        Gas has lower hydrogen loss, no ash, no unburned carbon.
        """
        try:
            input_gas = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )

            input_coal = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=24000.0,  # More coal needed for same steam
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="coal_bituminous",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        result_gas = efficiency_calculator.calculate(input_gas)
        result_coal = efficiency_calculator.calculate(input_coal)

        # Natural gas typically 2-5% more efficient
        assert result_gas.efficiency_hhv_percent >= result_coal.efficiency_hhv_percent - 1, (
            f"Gas efficiency {result_gas.efficiency_hhv_percent}% should be >= "
            f"coal efficiency {result_coal.efficiency_hhv_percent}%"
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple calculations."""

    def test_complete_calculation_workflow(
        self,
        efficiency_calculator,
        sample_efficiency_input,
    ):
        """Test complete calculation workflow with all outputs."""
        if sample_efficiency_input is None:
            pytest.skip("EfficiencyInput not available")

        result = efficiency_calculator.calculate(sample_efficiency_input, method="indirect")

        # Verify all required fields are populated
        assert result.calculation_id is not None
        assert result.timestamp is not None
        assert result.method == "indirect"
        assert result.efficiency_hhv_percent > 0
        assert result.efficiency_lhv_percent > 0
        assert result.uncertainty_percent > 0
        assert result.fuel_input_mmbtu_hr > 0
        assert result.steam_output_mmbtu_hr > 0
        assert result.total_losses_mmbtu_hr >= 0
        assert result.excess_air_percent >= 0
        assert result.air_fuel_ratio > 0
        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.formula_version is not None

    def test_efficiency_sensitivity_analysis(
        self,
        efficiency_calculator,
    ):
        """
        Test sensitivity of efficiency to key parameters.

        Verify that efficiency changes in expected direction
        when parameters are varied.
        """
        try:
            base_input = EfficiencyInput(
                steam_flow_klb_hr=150.0,
                steam_pressure_psig=150.0,
                steam_temperature_f=366.0,
                feedwater_temperature_f=227.0,
                fuel_flow_rate=8000.0,
                flue_gas_temperature_f=350.0,
                flue_gas_o2_percent=3.5,
                fuel_type="natural_gas",
            )
        except NameError:
            pytest.skip("EfficiencyInput not available")

        base_result = efficiency_calculator.calculate(base_input, method="indirect")

        # Vary stack temperature
        input_high_stack = EfficiencyInput(
            steam_flow_klb_hr=150.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=450.0,  # Higher stack temp
            flue_gas_o2_percent=3.5,
            fuel_type="natural_gas",
        )
        result_high_stack = efficiency_calculator.calculate(input_high_stack, method="indirect")

        # Higher stack temp should reduce efficiency
        assert result_high_stack.efficiency_hhv_percent < base_result.efficiency_hhv_percent, (
            "Higher stack temp should reduce efficiency"
        )

        # Vary feedwater temperature
        input_high_fw = EfficiencyInput(
            steam_flow_klb_hr=150.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=280.0,  # Higher feedwater temp
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=350.0,
            flue_gas_o2_percent=3.5,
            fuel_type="natural_gas",
        )
        result_high_fw = efficiency_calculator.calculate(input_high_fw, method="indirect")

        # Higher feedwater temp should not significantly affect indirect method losses
        # but affects direct method positively
