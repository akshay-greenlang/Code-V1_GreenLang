"""
Unit Tests: Calculator Modules

Tests all calculator modules for ThermalCommand agent including:
- Thermal efficiency calculator
- Heat distribution calculator
- Energy balance calculator
- Emissions compliance calculator
- KPI calculator
- Provenance hash calculator

Reference: GL-001 Specification Section 11.1
Target Coverage: 85%+
"""

import pytest
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Calculator Classes (Simulated Production Code)
# =============================================================================

class CalculationError(Exception):
    """Raised when a calculation fails."""
    pass


class ThermalEfficiencyCalculator:
    """Calculates thermal efficiency per ASME PTC 4.1."""

    # Heat capacity of water (kJ/kg.K)
    CP_WATER = 4.186

    # Reference temperature for enthalpy calculations (Celsius)
    REFERENCE_TEMP = 25.0

    @classmethod
    def calculate_boiler_efficiency_direct(
        cls,
        steam_output_kwh: float,
        fuel_input_kwh: float
    ) -> float:
        """Calculate boiler efficiency using direct method.

        Efficiency = Energy Output / Energy Input
        """
        if fuel_input_kwh <= 0:
            raise CalculationError("Fuel input must be positive")

        efficiency = steam_output_kwh / fuel_input_kwh

        if efficiency < 0 or efficiency > 1:
            raise CalculationError(f"Calculated efficiency {efficiency} is invalid")

        return efficiency

    @classmethod
    def calculate_boiler_efficiency_indirect(
        cls,
        dry_flue_gas_loss: float,
        moisture_loss: float,
        combustible_loss: float,
        radiation_loss: float,
        blowdown_loss: float = 0.0
    ) -> float:
        """Calculate boiler efficiency using indirect (loss) method per ASME PTC 4.1.

        Efficiency = 1 - Sum of all losses
        """
        total_losses = (
            dry_flue_gas_loss +
            moisture_loss +
            combustible_loss +
            radiation_loss +
            blowdown_loss
        )

        if total_losses < 0:
            raise CalculationError("Total losses cannot be negative")

        if total_losses > 1:
            raise CalculationError("Total losses exceed 100%")

        return 1.0 - total_losses

    @classmethod
    def calculate_heat_rate(
        cls,
        fuel_input_mj: float,
        power_output_kwh: float
    ) -> float:
        """Calculate heat rate (MJ/kWh)."""
        if power_output_kwh <= 0:
            raise CalculationError("Power output must be positive")

        return fuel_input_mj / power_output_kwh


class HeatDistributionCalculator:
    """Calculates heat distribution across network."""

    # Heat transfer coefficient typical values
    DEFAULT_U_VALUE = 500  # W/(m2.K) for steam pipes

    @classmethod
    def calculate_heat_loss(
        cls,
        pipe_length: float,  # meters
        pipe_diameter: float,  # meters
        temp_fluid: float,  # Celsius
        temp_ambient: float,  # Celsius
        insulation_thickness: float = 0.05,  # meters
        u_value: float = DEFAULT_U_VALUE
    ) -> float:
        """Calculate heat loss from pipe in kW."""
        if pipe_length <= 0 or pipe_diameter <= 0:
            raise CalculationError("Pipe dimensions must be positive")

        # Calculate surface area
        surface_area = math.pi * pipe_diameter * pipe_length

        # Temperature difference
        delta_t = temp_fluid - temp_ambient

        # Heat loss (W)
        heat_loss_w = u_value * surface_area * delta_t

        # Apply insulation factor (simplified)
        insulation_factor = 1 / (1 + insulation_thickness * 10)

        return (heat_loss_w * insulation_factor) / 1000  # Convert to kW

    @classmethod
    def calculate_distribution_efficiency(
        cls,
        heat_produced: float,
        heat_delivered: float
    ) -> float:
        """Calculate distribution network efficiency."""
        if heat_produced <= 0:
            raise CalculationError("Heat produced must be positive")

        if heat_delivered > heat_produced:
            raise CalculationError("Heat delivered cannot exceed heat produced")

        return heat_delivered / heat_produced

    @classmethod
    def calculate_flow_allocation(
        cls,
        total_demand: float,
        consumer_demands: List[float],
        consumer_priorities: List[int]
    ) -> List[float]:
        """Allocate flow based on demands and priorities.

        Higher priority (lower number) gets served first.
        """
        if len(consumer_demands) != len(consumer_priorities):
            raise CalculationError("Demands and priorities must have same length")

        if not consumer_demands:
            return []

        # Sort by priority
        indexed_demands = list(enumerate(zip(consumer_demands, consumer_priorities)))
        indexed_demands.sort(key=lambda x: x[1][1])  # Sort by priority

        allocations = [0.0] * len(consumer_demands)
        remaining = total_demand

        for original_idx, (demand, priority) in indexed_demands:
            allocation = min(demand, remaining)
            allocations[original_idx] = allocation
            remaining -= allocation

            if remaining <= 0:
                break

        return allocations


class EnergyBalanceCalculator:
    """Calculates energy balance for thermal systems."""

    @classmethod
    def calculate_energy_balance(
        cls,
        energy_inputs: Dict[str, float],  # kWh
        energy_outputs: Dict[str, float],  # kWh
        energy_stored: float = 0.0,  # kWh
        tolerance: float = 0.05  # 5% tolerance
    ) -> Tuple[float, bool]:
        """Calculate energy balance and check if within tolerance.

        Returns: (imbalance_fraction, is_balanced)
        """
        total_in = sum(energy_inputs.values())
        total_out = sum(energy_outputs.values())

        if total_in <= 0:
            return (0.0, True) if total_out == 0 and energy_stored == 0 else (float('inf'), False)

        imbalance = total_in - total_out - energy_stored
        imbalance_fraction = abs(imbalance) / total_in

        return (imbalance_fraction, imbalance_fraction <= tolerance)

    @classmethod
    def calculate_heat_recovery_potential(
        cls,
        exhaust_temp: float,  # Celsius
        ambient_temp: float,  # Celsius
        exhaust_flow_rate: float,  # kg/s
        recovery_efficiency: float = 0.7
    ) -> float:
        """Calculate heat recovery potential in kW."""
        if exhaust_flow_rate <= 0:
            return 0.0

        # Specific heat of air (kJ/kg.K)
        cp_air = 1.005

        # Temperature difference available
        delta_t = exhaust_temp - ambient_temp

        if delta_t <= 0:
            return 0.0

        # Maximum recoverable heat
        max_recovery = exhaust_flow_rate * cp_air * delta_t

        return max_recovery * recovery_efficiency


class EmissionsComplianceCalculator:
    """Calculates emissions and compliance metrics."""

    # Emission factors (kg CO2 per unit)
    EMISSION_FACTORS = {
        'natural_gas': 2.0,  # kg CO2 per m3
        'diesel': 2.68,  # kg CO2 per liter
        'coal': 2.86,  # kg CO2 per kg
        'biomass': 0.0,  # Considered carbon neutral
    }

    @classmethod
    def calculate_co2_emissions(
        cls,
        fuel_type: str,
        fuel_quantity: float,
        custom_factor: Optional[float] = None
    ) -> float:
        """Calculate CO2 emissions in kg."""
        if fuel_quantity < 0:
            raise CalculationError("Fuel quantity cannot be negative")

        if custom_factor is not None:
            emission_factor = custom_factor
        elif fuel_type in cls.EMISSION_FACTORS:
            emission_factor = cls.EMISSION_FACTORS[fuel_type]
        else:
            raise CalculationError(f"Unknown fuel type: {fuel_type}")

        return fuel_quantity * emission_factor

    @classmethod
    def calculate_emission_intensity(
        cls,
        total_emissions_kg: float,
        energy_output_mwh: float
    ) -> float:
        """Calculate emission intensity in kg CO2/MWh."""
        if energy_output_mwh <= 0:
            raise CalculationError("Energy output must be positive")

        return total_emissions_kg / energy_output_mwh

    @classmethod
    def check_compliance(
        cls,
        emission_intensity: float,
        limit: float
    ) -> Tuple[bool, float]:
        """Check if emission intensity is within compliance limit.

        Returns: (is_compliant, margin_percent)
        """
        if limit <= 0:
            raise CalculationError("Limit must be positive")

        is_compliant = emission_intensity <= limit
        margin_percent = ((limit - emission_intensity) / limit) * 100

        return (is_compliant, margin_percent)


class KPICalculator:
    """Calculates Key Performance Indicators for thermal systems."""

    @classmethod
    def calculate_availability(
        cls,
        operating_hours: float,
        total_hours: float
    ) -> float:
        """Calculate equipment availability."""
        if total_hours <= 0:
            raise CalculationError("Total hours must be positive")

        if operating_hours < 0:
            raise CalculationError("Operating hours cannot be negative")

        if operating_hours > total_hours:
            raise CalculationError("Operating hours cannot exceed total hours")

        return operating_hours / total_hours

    @classmethod
    def calculate_capacity_factor(
        cls,
        actual_output: float,
        max_possible_output: float
    ) -> float:
        """Calculate capacity factor."""
        if max_possible_output <= 0:
            raise CalculationError("Max output must be positive")

        if actual_output < 0:
            raise CalculationError("Actual output cannot be negative")

        factor = actual_output / max_possible_output

        if factor > 1:
            raise CalculationError("Capacity factor cannot exceed 1")

        return factor

    @classmethod
    def calculate_specific_energy_consumption(
        cls,
        energy_consumed: float,
        production_output: float
    ) -> float:
        """Calculate specific energy consumption (energy per unit product)."""
        if production_output <= 0:
            raise CalculationError("Production output must be positive")

        return energy_consumed / production_output

    @classmethod
    def calculate_overall_efficiency(
        cls,
        efficiencies: List[float]
    ) -> float:
        """Calculate overall efficiency (product of individual efficiencies)."""
        if not efficiencies:
            raise CalculationError("Efficiency list cannot be empty")

        for eff in efficiencies:
            if eff < 0 or eff > 1:
                raise CalculationError(f"Invalid efficiency value: {eff}")

        overall = 1.0
        for eff in efficiencies:
            overall *= eff

        return overall


class ProvenanceCalculator:
    """Calculates SHA-256 provenance hashes for audit trails."""

    @staticmethod
    def calculate_hash(data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        if isinstance(data, dict):
            # Sort keys for deterministic hashing
            data_str = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(list(data), default=str)
        else:
            data_str = str(data)

        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

    @staticmethod
    def calculate_chain_hash(previous_hash: str, current_data: Any) -> str:
        """Calculate chained hash for sequential provenance."""
        current_hash = ProvenanceCalculator.calculate_hash(current_data)
        combined = f"{previous_hash}:{current_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    @staticmethod
    def verify_hash(data: Any, expected_hash: str) -> bool:
        """Verify that data matches expected provenance hash."""
        actual_hash = ProvenanceCalculator.calculate_hash(data)
        return actual_hash == expected_hash


# =============================================================================
# Test Classes
# =============================================================================

class TestThermalEfficiencyCalculator:
    """Test suite for thermal efficiency calculator."""

    @pytest.mark.parametrize("steam_output,fuel_input,expected_efficiency", [
        (80, 100, 0.80),
        (90, 100, 0.90),
        (100, 100, 1.00),
        (50, 100, 0.50),
    ])
    def test_direct_efficiency_calculation(self, steam_output, fuel_input, expected_efficiency):
        """Test direct method efficiency calculation."""
        result = ThermalEfficiencyCalculator.calculate_boiler_efficiency_direct(
            steam_output, fuel_input
        )
        assert pytest.approx(result, rel=1e-6) == expected_efficiency

    def test_direct_efficiency_zero_fuel_fails(self):
        """Test that zero fuel input raises error."""
        with pytest.raises(CalculationError) as exc_info:
            ThermalEfficiencyCalculator.calculate_boiler_efficiency_direct(100, 0)
        assert "positive" in str(exc_info.value).lower()

    def test_direct_efficiency_negative_fuel_fails(self):
        """Test that negative fuel input raises error."""
        with pytest.raises(CalculationError):
            ThermalEfficiencyCalculator.calculate_boiler_efficiency_direct(100, -10)

    @pytest.mark.parametrize("losses,expected_efficiency", [
        ((0.05, 0.03, 0.02, 0.02, 0.0), 0.88),  # Total 12% losses = 88% eff
        ((0.10, 0.05, 0.03, 0.02, 0.0), 0.80),  # Total 20% losses = 80% eff
        ((0.0, 0.0, 0.0, 0.0, 0.0), 1.00),  # No losses = 100% eff
    ])
    def test_indirect_efficiency_calculation(self, losses, expected_efficiency):
        """Test indirect method efficiency calculation."""
        result = ThermalEfficiencyCalculator.calculate_boiler_efficiency_indirect(*losses)
        assert pytest.approx(result, rel=1e-6) == expected_efficiency

    def test_indirect_efficiency_excess_losses_fails(self):
        """Test that losses exceeding 100% raises error."""
        with pytest.raises(CalculationError) as exc_info:
            ThermalEfficiencyCalculator.calculate_boiler_efficiency_indirect(
                0.5, 0.3, 0.2, 0.1, 0.0  # Total 110%
            )
        assert "exceed" in str(exc_info.value).lower()

    def test_heat_rate_calculation(self):
        """Test heat rate calculation."""
        result = ThermalEfficiencyCalculator.calculate_heat_rate(100, 25)  # 100 MJ / 25 kWh
        assert pytest.approx(result, rel=1e-6) == 4.0


class TestHeatDistributionCalculator:
    """Test suite for heat distribution calculator."""

    def test_heat_loss_calculation(self):
        """Test pipe heat loss calculation."""
        result = HeatDistributionCalculator.calculate_heat_loss(
            pipe_length=100,
            pipe_diameter=0.1,
            temp_fluid=200,
            temp_ambient=20
        )
        assert result > 0

    def test_heat_loss_zero_length_fails(self):
        """Test that zero pipe length raises error."""
        with pytest.raises(CalculationError):
            HeatDistributionCalculator.calculate_heat_loss(0, 0.1, 200, 20)

    def test_heat_loss_increases_with_temperature_difference(self):
        """Test that heat loss increases with temperature difference."""
        loss_small = HeatDistributionCalculator.calculate_heat_loss(100, 0.1, 100, 20)
        loss_large = HeatDistributionCalculator.calculate_heat_loss(100, 0.1, 300, 20)
        assert loss_large > loss_small

    def test_heat_loss_decreases_with_insulation(self):
        """Test that heat loss decreases with more insulation."""
        loss_thin = HeatDistributionCalculator.calculate_heat_loss(
            100, 0.1, 200, 20, insulation_thickness=0.02
        )
        loss_thick = HeatDistributionCalculator.calculate_heat_loss(
            100, 0.1, 200, 20, insulation_thickness=0.10
        )
        assert loss_thick < loss_thin

    @pytest.mark.parametrize("produced,delivered,expected_efficiency", [
        (1000, 950, 0.95),
        (1000, 800, 0.80),
        (1000, 1000, 1.00),
    ])
    def test_distribution_efficiency(self, produced, delivered, expected_efficiency):
        """Test distribution efficiency calculation."""
        result = HeatDistributionCalculator.calculate_distribution_efficiency(
            produced, delivered
        )
        assert pytest.approx(result, rel=1e-6) == expected_efficiency

    def test_distribution_efficiency_exceeds_100_fails(self):
        """Test that delivered > produced raises error."""
        with pytest.raises(CalculationError):
            HeatDistributionCalculator.calculate_distribution_efficiency(100, 110)

    def test_flow_allocation_priority_based(self):
        """Test flow allocation respects priorities."""
        demands = [100, 200, 150]
        priorities = [2, 1, 3]  # Second consumer has highest priority

        allocations = HeatDistributionCalculator.calculate_flow_allocation(
            total_demand=250,
            consumer_demands=demands,
            consumer_priorities=priorities
        )

        # Priority 1 (index 1) should get full 200
        assert allocations[1] == 200
        # Priority 2 (index 0) should get remaining 50
        assert allocations[0] == 50
        # Priority 3 (index 2) should get 0
        assert allocations[2] == 0

    def test_flow_allocation_sufficient_supply(self):
        """Test flow allocation when supply meets demand."""
        demands = [100, 200, 150]
        priorities = [1, 2, 3]

        allocations = HeatDistributionCalculator.calculate_flow_allocation(
            total_demand=500,  # More than total demand (450)
            consumer_demands=demands,
            consumer_priorities=priorities
        )

        assert allocations == demands


class TestEnergyBalanceCalculator:
    """Test suite for energy balance calculator."""

    def test_energy_balance_perfect(self):
        """Test perfect energy balance."""
        imbalance, is_balanced = EnergyBalanceCalculator.calculate_energy_balance(
            energy_inputs={"boiler_1": 500, "boiler_2": 500},
            energy_outputs={"consumer_1": 700, "consumer_2": 300},
            energy_stored=0
        )
        assert is_balanced == True
        assert imbalance == 0

    def test_energy_balance_with_storage(self):
        """Test energy balance with storage."""
        imbalance, is_balanced = EnergyBalanceCalculator.calculate_energy_balance(
            energy_inputs={"source": 1000},
            energy_outputs={"consumer": 800},
            energy_stored=200
        )
        assert is_balanced == True

    def test_energy_balance_imbalanced(self):
        """Test energy balance exceeding tolerance."""
        imbalance, is_balanced = EnergyBalanceCalculator.calculate_energy_balance(
            energy_inputs={"source": 1000},
            energy_outputs={"consumer": 800},
            energy_stored=0,
            tolerance=0.05
        )
        assert is_balanced == False
        assert pytest.approx(imbalance, rel=1e-6) == 0.2  # 20% imbalance

    def test_heat_recovery_potential(self):
        """Test heat recovery potential calculation."""
        result = EnergyBalanceCalculator.calculate_heat_recovery_potential(
            exhaust_temp=200,
            ambient_temp=20,
            exhaust_flow_rate=10,
            recovery_efficiency=0.7
        )
        # Expected: 10 kg/s * 1.005 kJ/kg.K * 180 K * 0.7 = 1266.3 kW
        assert pytest.approx(result, rel=0.01) == 1266.3

    def test_heat_recovery_no_potential(self):
        """Test heat recovery when exhaust is cold."""
        result = EnergyBalanceCalculator.calculate_heat_recovery_potential(
            exhaust_temp=20,
            ambient_temp=25,
            exhaust_flow_rate=10
        )
        assert result == 0.0


class TestEmissionsComplianceCalculator:
    """Test suite for emissions compliance calculator."""

    @pytest.mark.parametrize("fuel_type,quantity,expected_co2", [
        ('natural_gas', 100, 200),  # 100 m3 * 2.0 = 200 kg
        ('diesel', 100, 268),  # 100 L * 2.68 = 268 kg
        ('coal', 100, 286),  # 100 kg * 2.86 = 286 kg
        ('biomass', 100, 0),  # Carbon neutral
    ])
    def test_co2_emissions_by_fuel_type(self, fuel_type, quantity, expected_co2):
        """Test CO2 emissions for different fuel types."""
        result = EmissionsComplianceCalculator.calculate_co2_emissions(fuel_type, quantity)
        assert pytest.approx(result, rel=1e-6) == expected_co2

    def test_co2_emissions_custom_factor(self):
        """Test CO2 emissions with custom emission factor."""
        result = EmissionsComplianceCalculator.calculate_co2_emissions(
            'any_fuel', 100, custom_factor=3.0
        )
        assert result == 300

    def test_co2_emissions_unknown_fuel_fails(self):
        """Test that unknown fuel type raises error."""
        with pytest.raises(CalculationError) as exc_info:
            EmissionsComplianceCalculator.calculate_co2_emissions('unknown_fuel', 100)
        assert "Unknown fuel" in str(exc_info.value)

    def test_co2_emissions_negative_quantity_fails(self):
        """Test that negative quantity raises error."""
        with pytest.raises(CalculationError):
            EmissionsComplianceCalculator.calculate_co2_emissions('natural_gas', -100)

    def test_emission_intensity(self):
        """Test emission intensity calculation."""
        result = EmissionsComplianceCalculator.calculate_emission_intensity(400, 1)  # 400 kg / 1 MWh
        assert result == 400

    def test_compliance_check_passing(self):
        """Test compliance check that passes."""
        is_compliant, margin = EmissionsComplianceCalculator.check_compliance(300, 400)
        assert is_compliant == True
        assert pytest.approx(margin, rel=1e-6) == 25.0  # 25% margin

    def test_compliance_check_failing(self):
        """Test compliance check that fails."""
        is_compliant, margin = EmissionsComplianceCalculator.check_compliance(500, 400)
        assert is_compliant == False
        assert margin < 0  # Negative margin


class TestKPICalculator:
    """Test suite for KPI calculator."""

    @pytest.mark.parametrize("operating,total,expected", [
        (8760, 8760, 1.0),  # 100% availability
        (8000, 8760, 0.9132),  # ~91% availability
        (0, 8760, 0.0),  # 0% availability
    ])
    def test_availability_calculation(self, operating, total, expected):
        """Test availability calculation."""
        result = KPICalculator.calculate_availability(operating, total)
        assert pytest.approx(result, rel=0.001) == expected

    def test_availability_exceeds_total_fails(self):
        """Test that operating hours exceeding total fails."""
        with pytest.raises(CalculationError):
            KPICalculator.calculate_availability(9000, 8760)

    @pytest.mark.parametrize("actual,max_output,expected", [
        (800, 1000, 0.8),
        (1000, 1000, 1.0),
        (500, 1000, 0.5),
    ])
    def test_capacity_factor(self, actual, max_output, expected):
        """Test capacity factor calculation."""
        result = KPICalculator.calculate_capacity_factor(actual, max_output)
        assert pytest.approx(result, rel=1e-6) == expected

    def test_specific_energy_consumption(self):
        """Test specific energy consumption calculation."""
        result = KPICalculator.calculate_specific_energy_consumption(
            energy_consumed=1000,  # kWh
            production_output=500  # units
        )
        assert result == 2.0  # 2 kWh per unit

    @pytest.mark.parametrize("efficiencies,expected_overall", [
        ([0.9, 0.9, 0.9], 0.729),  # 0.9^3
        ([0.8, 0.9], 0.72),  # 0.8 * 0.9
        ([1.0, 1.0, 1.0], 1.0),  # Perfect
        ([0.5], 0.5),  # Single efficiency
    ])
    def test_overall_efficiency(self, efficiencies, expected_overall):
        """Test overall efficiency calculation."""
        result = KPICalculator.calculate_overall_efficiency(efficiencies)
        assert pytest.approx(result, rel=1e-6) == expected_overall

    def test_overall_efficiency_empty_list_fails(self):
        """Test that empty efficiency list raises error."""
        with pytest.raises(CalculationError):
            KPICalculator.calculate_overall_efficiency([])

    def test_overall_efficiency_invalid_value_fails(self):
        """Test that invalid efficiency value raises error."""
        with pytest.raises(CalculationError):
            KPICalculator.calculate_overall_efficiency([0.9, 1.5, 0.8])


class TestProvenanceCalculator:
    """Test suite for provenance calculator."""

    def test_hash_length(self):
        """Test that hash is 64 characters (SHA-256)."""
        result = ProvenanceCalculator.calculate_hash({"test": "data"})
        assert len(result) == 64

    def test_hash_deterministic(self):
        """Test that same input produces same hash."""
        data = {"temperature": 450.0, "pressure": 15.0}

        hash1 = ProvenanceCalculator.calculate_hash(data)
        hash2 = ProvenanceCalculator.calculate_hash(data)

        assert hash1 == hash2

    def test_hash_different_data_different_hash(self):
        """Test that different data produces different hash."""
        data1 = {"temperature": 450.0}
        data2 = {"temperature": 451.0}

        hash1 = ProvenanceCalculator.calculate_hash(data1)
        hash2 = ProvenanceCalculator.calculate_hash(data2)

        assert hash1 != hash2

    def test_hash_dict_key_order_independent(self):
        """Test that dict key order doesn't affect hash."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "b": 2, "a": 1}

        hash1 = ProvenanceCalculator.calculate_hash(data1)
        hash2 = ProvenanceCalculator.calculate_hash(data2)

        assert hash1 == hash2

    def test_hash_list_data(self):
        """Test hashing of list data."""
        data = [1, 2, 3, 4, 5]
        result = ProvenanceCalculator.calculate_hash(data)
        assert len(result) == 64

    def test_hash_string_data(self):
        """Test hashing of string data."""
        result = ProvenanceCalculator.calculate_hash("test string")
        assert len(result) == 64

    def test_chain_hash(self):
        """Test chained hash calculation."""
        previous = "a" * 64
        current_data = {"value": 100}

        result = ProvenanceCalculator.calculate_chain_hash(previous, current_data)

        assert len(result) == 64
        assert result != previous

    def test_verify_hash_correct(self):
        """Test hash verification with correct hash."""
        data = {"sensor": "SENSOR_001", "value": 450.0}
        correct_hash = ProvenanceCalculator.calculate_hash(data)

        assert ProvenanceCalculator.verify_hash(data, correct_hash) == True

    def test_verify_hash_incorrect(self):
        """Test hash verification with incorrect hash."""
        data = {"sensor": "SENSOR_001", "value": 450.0}
        wrong_hash = "b" * 64

        assert ProvenanceCalculator.verify_hash(data, wrong_hash) == False

    def test_hash_with_datetime(self):
        """Test hashing data containing datetime."""
        data = {
            "timestamp": datetime(2025, 1, 15, 10, 30, 0),
            "value": 100
        }
        result = ProvenanceCalculator.calculate_hash(data)
        assert len(result) == 64

    def test_hash_nested_dict(self):
        """Test hashing nested dictionaries."""
        data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        result = ProvenanceCalculator.calculate_hash(data)
        assert len(result) == 64


class TestCalculatorIntegration:
    """Integration tests for calculators working together."""

    def test_efficiency_chain_calculation(self):
        """Test calculating overall system efficiency."""
        # Boiler efficiency
        boiler_eff = ThermalEfficiencyCalculator.calculate_boiler_efficiency_direct(850, 1000)

        # Distribution efficiency
        dist_eff = HeatDistributionCalculator.calculate_distribution_efficiency(850, 800)

        # Overall efficiency
        overall = KPICalculator.calculate_overall_efficiency([boiler_eff, dist_eff])

        assert pytest.approx(overall, rel=0.01) == 0.8  # ~80%

    def test_emissions_with_efficiency(self):
        """Test emissions calculation accounting for efficiency."""
        # Calculate fuel needed for 1 MWh output at 85% efficiency
        fuel_input_mwh = 1.0 / 0.85

        # Convert to natural gas volume (assuming 10 kWh/m3)
        fuel_m3 = (fuel_input_mwh * 1000) / 10

        # Calculate emissions
        emissions = EmissionsComplianceCalculator.calculate_co2_emissions('natural_gas', fuel_m3)

        # Calculate intensity
        intensity = EmissionsComplianceCalculator.calculate_emission_intensity(emissions, 1.0)

        # Check compliance (limit 400 kg/MWh)
        is_compliant, margin = EmissionsComplianceCalculator.check_compliance(intensity, 400)

        assert is_compliant == True

    def test_full_audit_trail_with_provenance(self):
        """Test creating audit trail with provenance hashes."""
        # Input data
        input_data = {"boiler_1_fuel": 100, "boiler_2_fuel": 80}
        input_hash = ProvenanceCalculator.calculate_hash(input_data)

        # Calculation result
        total_emissions = sum(
            EmissionsComplianceCalculator.calculate_co2_emissions('natural_gas', v)
            for v in input_data.values()
        )

        result_data = {
            "total_emissions": total_emissions,
            "input_hash": input_hash
        }

        # Chain the results
        result_hash = ProvenanceCalculator.calculate_chain_hash(input_hash, result_data)

        # Verify the chain
        assert len(result_hash) == 64
        assert ProvenanceCalculator.verify_hash(input_data, input_hash)
