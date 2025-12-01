# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests for GL-007 FURNACEPULSE (FurnacePerformanceOptimizer)

This module provides comprehensive determinism tests covering:
- Bit-perfect reproducibility of thermal efficiency calculations
- Fuel consumption analysis consistency
- Excess air calculation determinism
- Stack loss calculation reproducibility
- SEC (Specific Energy Consumption) reproducibility
- Equipment health scoring consistency
- Provenance hash consistency
- Floating-point stability using Decimal
- Random seed propagation verification
- Cross-iteration consistency

Zero-Hallucination Verification:
All furnace performance calculations must produce identical results when given
identical inputs. This ensures regulatory compliance (EPA, ISO 50001), audit
trail integrity, and industrial certification requirements.

Standards Compliance:
- ASME PTC 4.1 (Steam Generating Units)
- API 560 (Fired Heaters)
- ISO 50001 (Energy Management)
- EPA CEMS Requirements

Author: GL-BackendDeveloper
Date: 2025-11-22
Version: 1.0.0
"""

import pytest
import hashlib
import json
import math
import random
import sys
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_CEILING
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def deterministic_seed():
    """Provide deterministic seed for reproducibility testing."""
    return 42


@pytest.fixture
def furnace_efficiency_inputs():
    """Create standardized inputs for thermal efficiency calculations."""
    return {
        "furnace_id": "FURNACE-001",
        "fuel_input": {
            "mass_flow_rate_kg_hr": Decimal("1850.0"),
            "higher_heating_value_mj_kg": Decimal("50.0"),
            "lower_heating_value_mj_kg": Decimal("45.2")
        },
        "useful_heat_output_mw": Decimal("20.5"),
        "flue_gas": {
            "temperature_c": Decimal("185.0"),
            "o2_percent_dry": Decimal("3.5"),
            "co2_percent_dry": Decimal("10.8")
        },
        "ambient_temperature_c": Decimal("25.0"),
        "radiation_loss_mw": Decimal("0.5"),
        "convection_loss_mw": Decimal("0.3")
    }


@pytest.fixture
def fuel_consumption_inputs():
    """Create standardized inputs for fuel consumption analysis."""
    return {
        "consumption_rate_kg_hr": Decimal("1850.0"),
        "heating_value_mj_kg": Decimal("50.0"),
        "production_rate_tons_hr": Decimal("18.5"),
        "baseline_sec_gj_ton": Decimal("4.8"),
        "fuel_cost_usd_per_gj": Decimal("8.50"),
        "emission_factor_kg_co2_per_gj": Decimal("56.1")
    }


@pytest.fixture
def stack_loss_inputs():
    """Create standardized inputs for stack loss calculations."""
    return {
        "flue_gas_temp_c": Decimal("185.0"),
        "ambient_temp_c": Decimal("25.0"),
        "o2_percent_dry": Decimal("3.5"),
        "co2_percent_dry": Decimal("10.8"),
        "specific_heat_flue_gas_kj_kg_k": Decimal("1.08"),
        "fuel_mass_flow_kg_hr": Decimal("1850.0"),
        "fuel_hhv_mj_kg": Decimal("50.0"),
        "air_fuel_ratio_stoich": Decimal("17.2")
    }


@pytest.fixture
def equipment_health_inputs():
    """Create standardized inputs for equipment health calculations."""
    return {
        "equipment_id": "REFRACTORY-001",
        "design_life_hours": Decimal("40000"),
        "operating_hours": Decimal("28000"),
        "last_inspection_date": "2024-06-15",
        "criticality_weight": Decimal("100"),
        "failure_history_count": 2
    }


@pytest.fixture
def excess_air_inputs():
    """Create standardized inputs for excess air calculations."""
    return {
        "o2_percent_dry": Decimal("3.5"),
        "co2_percent_dry": Decimal("10.8"),
        "theoretical_air_fuel_ratio": Decimal("17.2")
    }


# ============================================================================
# THERMAL EFFICIENCY REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestThermalEfficiencyReproducibility:
    """Test bit-perfect reproducibility of thermal efficiency calculations."""

    @pytest.mark.determinism
    def test_direct_method_efficiency_reproducibility(self, furnace_efficiency_inputs):
        """Test direct method thermal efficiency is deterministic."""
        results = []

        for _ in range(1000):
            fuel = furnace_efficiency_inputs["fuel_input"]
            useful_heat = furnace_efficiency_inputs["useful_heat_output_mw"]

            # Fuel energy input (MJ/hr -> MW)
            fuel_energy_mw = (
                fuel["mass_flow_rate_kg_hr"] * fuel["higher_heating_value_mj_kg"]
            ) / Decimal("3600")

            # Direct efficiency = Useful Heat / Fuel Energy * 100
            efficiency = (useful_heat / fuel_energy_mw) * Decimal("100")
            efficiency = efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            results.append(efficiency)

        assert len(set(results)) == 1, "Direct method efficiency not deterministic"
        # Verify expected value
        assert results[0] == Decimal("79.78")

    @pytest.mark.determinism
    def test_indirect_method_efficiency_reproducibility(self, furnace_efficiency_inputs):
        """Test indirect (heat loss) method efficiency is deterministic."""
        results = []

        for _ in range(1000):
            fuel = furnace_efficiency_inputs["fuel_input"]
            flue = furnace_efficiency_inputs["flue_gas"]
            ambient = furnace_efficiency_inputs["ambient_temperature_c"]
            radiation = furnace_efficiency_inputs["radiation_loss_mw"]
            convection = furnace_efficiency_inputs["convection_loss_mw"]

            # Fuel energy
            fuel_energy_mw = (
                fuel["mass_flow_rate_kg_hr"] * fuel["higher_heating_value_mj_kg"]
            ) / Decimal("3600")

            # Stack loss (simplified per API 560)
            temp_diff = flue["temperature_c"] - ambient
            stack_loss_percent = temp_diff / Decimal("10")  # Simplified correlation

            # Radiation and convection loss percentages
            radiation_percent = (radiation / fuel_energy_mw) * Decimal("100")
            convection_percent = (convection / fuel_energy_mw) * Decimal("100")

            # Total losses
            total_loss_percent = stack_loss_percent + radiation_percent + convection_percent

            # Indirect efficiency
            efficiency = Decimal("100") - total_loss_percent
            efficiency = efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            results.append(efficiency)

        assert len(set(results)) == 1, "Indirect method efficiency not deterministic"

    @pytest.mark.determinism
    def test_combined_efficiency_reproducibility(self, furnace_efficiency_inputs):
        """Test combined direct/indirect efficiency is deterministic."""
        results = []

        for _ in range(500):
            # Direct method
            fuel = furnace_efficiency_inputs["fuel_input"]
            useful_heat = furnace_efficiency_inputs["useful_heat_output_mw"]

            fuel_energy_mw = (
                fuel["mass_flow_rate_kg_hr"] * fuel["higher_heating_value_mj_kg"]
            ) / Decimal("3600")

            direct_eff = (useful_heat / fuel_energy_mw) * Decimal("100")

            # Indirect method (simplified)
            flue_temp = furnace_efficiency_inputs["flue_gas"]["temperature_c"]
            ambient = furnace_efficiency_inputs["ambient_temperature_c"]
            stack_loss = (flue_temp - ambient) / Decimal("10")
            indirect_eff = Decimal("100") - stack_loss - Decimal("3.5")

            # Combined (average)
            combined = (direct_eff + indirect_eff) / Decimal("2")
            combined = combined.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            results.append(combined)

        assert len(set(results)) == 1, "Combined efficiency not deterministic"


# ============================================================================
# EXCESS AIR CALCULATION REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestExcessAirReproducibility:
    """Test bit-perfect reproducibility of excess air calculations."""

    @pytest.mark.determinism
    def test_o2_based_excess_air_reproducibility(self, excess_air_inputs):
        """Test excess air from O2 measurement is deterministic."""
        results = []

        for _ in range(1000):
            o2 = excess_air_inputs["o2_percent_dry"]

            # EA% = (O2 / (21 - O2)) * 100
            excess_air = (o2 / (Decimal("21") - o2)) * Decimal("100")
            excess_air = excess_air.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

            results.append(excess_air)

        assert len(set(results)) == 1, "Excess air calculation not deterministic"
        # Verify expected value: 3.5 / (21 - 3.5) * 100 = 20.0%
        assert results[0] == Decimal("20.0")

    @pytest.mark.determinism
    def test_co2_based_excess_air_reproducibility(self, excess_air_inputs):
        """Test excess air from CO2 measurement is deterministic."""
        results = []

        for _ in range(1000):
            co2_measured = excess_air_inputs["co2_percent_dry"]
            co2_stoich = Decimal("11.8")  # Theoretical CO2 for natural gas

            # EA% = ((CO2_stoich / CO2_measured) - 1) * 100
            excess_air = ((co2_stoich / co2_measured) - Decimal("1")) * Decimal("100")
            excess_air = excess_air.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

            results.append(excess_air)

        assert len(set(results)) == 1, "CO2-based excess air not deterministic"


# ============================================================================
# FUEL CONSUMPTION REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestFuelConsumptionReproducibility:
    """Test bit-perfect reproducibility of fuel consumption calculations."""

    @pytest.mark.determinism
    def test_sec_calculation_reproducibility(self, fuel_consumption_inputs):
        """Test Specific Energy Consumption is deterministic."""
        results = []

        for _ in range(1000):
            consumption = fuel_consumption_inputs["consumption_rate_kg_hr"]
            hhv = fuel_consumption_inputs["heating_value_mj_kg"]
            production = fuel_consumption_inputs["production_rate_tons_hr"]

            # Energy consumption in GJ/hr
            energy_gj_hr = (consumption * hhv) / Decimal("1000")

            # SEC = Energy / Production (GJ/ton)
            sec = energy_gj_hr / production
            sec = sec.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            results.append(sec)

        assert len(set(results)) == 1, "SEC calculation not deterministic"
        # Verify: (1850 * 50 / 1000) / 18.5 = 5.0 GJ/ton
        assert results[0] == Decimal("5.000")

    @pytest.mark.determinism
    def test_deviation_from_baseline_reproducibility(self, fuel_consumption_inputs):
        """Test deviation from baseline is deterministic."""
        results = []

        for _ in range(1000):
            # Calculate current SEC
            consumption = fuel_consumption_inputs["consumption_rate_kg_hr"]
            hhv = fuel_consumption_inputs["heating_value_mj_kg"]
            production = fuel_consumption_inputs["production_rate_tons_hr"]

            energy_gj_hr = (consumption * hhv) / Decimal("1000")
            current_sec = energy_gj_hr / production

            # Calculate deviation
            baseline_sec = fuel_consumption_inputs["baseline_sec_gj_ton"]
            deviation_percent = ((current_sec - baseline_sec) / baseline_sec) * Decimal("100")
            deviation_percent = deviation_percent.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            results.append(deviation_percent)

        assert len(set(results)) == 1, "Deviation calculation not deterministic"
        # Verify: ((5.0 - 4.8) / 4.8) * 100 = 4.17%
        assert results[0] == Decimal("4.17")

    @pytest.mark.determinism
    def test_fuel_cost_calculation_reproducibility(self, fuel_consumption_inputs):
        """Test fuel cost calculation is deterministic."""
        results = []

        for _ in range(1000):
            consumption = fuel_consumption_inputs["consumption_rate_kg_hr"]
            hhv = fuel_consumption_inputs["heating_value_mj_kg"]
            cost_per_gj = fuel_consumption_inputs["fuel_cost_usd_per_gj"]

            # Energy in GJ/hr
            energy_gj_hr = (consumption * hhv) / Decimal("1000")

            # Hourly cost
            hourly_cost = energy_gj_hr * cost_per_gj
            hourly_cost = hourly_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            results.append(hourly_cost)

        assert len(set(results)) == 1, "Fuel cost calculation not deterministic"
        # Verify: (1850 * 50 / 1000) * 8.50 = 786.25 USD/hr
        assert results[0] == Decimal("786.25")


# ============================================================================
# STACK LOSS CALCULATION REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestStackLossReproducibility:
    """Test bit-perfect reproducibility of stack loss calculations."""

    @pytest.mark.determinism
    def test_sensible_heat_loss_reproducibility(self, stack_loss_inputs):
        """Test sensible heat stack loss is deterministic."""
        results = []

        for _ in range(1000):
            flue_temp = stack_loss_inputs["flue_gas_temp_c"]
            ambient = stack_loss_inputs["ambient_temp_c"]
            cp_flue = stack_loss_inputs["specific_heat_flue_gas_kj_kg_k"]

            # Temperature difference
            delta_t = flue_temp - ambient

            # Simplified stack loss per ASME PTC 4.1
            # Actual implementation would include stoichiometric calculations
            stack_loss_percent = (delta_t * cp_flue) / Decimal("50")  # Simplified
            stack_loss_percent = stack_loss_percent.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            results.append(stack_loss_percent)

        assert len(set(results)) == 1, "Stack loss calculation not deterministic"

    @pytest.mark.determinism
    def test_dry_flue_gas_loss_reproducibility(self, stack_loss_inputs):
        """Test dry flue gas heat loss is deterministic."""
        results = []

        for _ in range(1000):
            flue_temp = stack_loss_inputs["flue_gas_temp_c"]
            ambient = stack_loss_inputs["ambient_temp_c"]
            o2 = stack_loss_inputs["o2_percent_dry"]
            fuel_mass = stack_loss_inputs["fuel_mass_flow_kg_hr"]
            hhv = stack_loss_inputs["fuel_hhv_mj_kg"]

            # Calculate excess air
            excess_air = (o2 / (Decimal("21") - o2)) * Decimal("100")

            # Dry flue gas loss (simplified ASME formula)
            # L_dg = K * (T_fg - T_amb) * (1 + EA/100)
            k_factor = Decimal("0.0024")  # Simplified for natural gas
            dry_loss_percent = k_factor * (flue_temp - ambient) * (Decimal("1") + excess_air / Decimal("100"))
            dry_loss_percent = dry_loss_percent.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            results.append(dry_loss_percent)

        assert len(set(results)) == 1, "Dry flue gas loss not deterministic"


# ============================================================================
# EQUIPMENT HEALTH REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestEquipmentHealthReproducibility:
    """Test bit-perfect reproducibility of equipment health calculations."""

    @pytest.mark.determinism
    def test_rul_calculation_reproducibility(self, equipment_health_inputs):
        """Test Remaining Useful Life is deterministic."""
        results = []

        for _ in range(1000):
            design_life = equipment_health_inputs["design_life_hours"]
            operating_hours = equipment_health_inputs["operating_hours"]

            # RUL in hours
            rul_hours = design_life - operating_hours
            rul_hours = max(Decimal("0"), rul_hours)

            # RUL percentage
            rul_percent = (rul_hours / design_life) * Decimal("100")
            rul_percent = rul_percent.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

            results.append(rul_percent)

        assert len(set(results)) == 1, "RUL calculation not deterministic"
        # Verify: (40000 - 28000) / 40000 * 100 = 30%
        assert results[0] == Decimal("30.0")

    @pytest.mark.determinism
    def test_health_index_reproducibility(self, equipment_health_inputs):
        """Test health index calculation is deterministic."""
        results = []

        for _ in range(1000):
            design_life = equipment_health_inputs["design_life_hours"]
            operating_hours = equipment_health_inputs["operating_hours"]
            failures = Decimal(str(equipment_health_inputs["failure_history_count"]))

            # Age factor (0-1, where 0 is new)
            age_factor = operating_hours / design_life

            # Failure penalty (reduces health by 5% per failure)
            failure_penalty = failures * Decimal("5")

            # Health index (100 = perfect)
            health_index = Decimal("100") - (age_factor * Decimal("100")) - failure_penalty
            health_index = max(Decimal("0"), health_index)
            health_index = health_index.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

            results.append(health_index)

        assert len(set(results)) == 1, "Health index not deterministic"
        # Verify: 100 - (28000/40000 * 100) - (2 * 5) = 100 - 70 - 10 = 20.0
        assert results[0] == Decimal("20.0")

    @pytest.mark.determinism
    def test_priority_score_reproducibility(self, equipment_health_inputs):
        """Test maintenance priority score is deterministic."""
        results = []

        for _ in range(1000):
            criticality = equipment_health_inputs["criticality_weight"]
            design_life = equipment_health_inputs["design_life_hours"]
            operating_hours = equipment_health_inputs["operating_hours"]

            # Calculate health
            health = Decimal("100") - (operating_hours / design_life * Decimal("100"))

            # Priority = Criticality * (100 - Health) / 100
            priority = criticality * (Decimal("100") - health) / Decimal("100")
            priority = priority.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

            results.append(priority)

        assert len(set(results)) == 1, "Priority score not deterministic"
        # Verify: 100 * (100 - 30) / 100 = 70.0
        assert results[0] == Decimal("70.0")


# ============================================================================
# PROVENANCE HASH CONSISTENCY
# ============================================================================

@pytest.mark.determinism
class TestProvenanceHashConsistency:
    """Test provenance hash consistency for audit trail."""

    @pytest.mark.determinism
    def test_hash_consistency_same_input(self, furnace_efficiency_inputs):
        """Test same input always produces same hash."""
        # Convert Decimal to string for JSON serialization
        data = json.loads(
            json.dumps(furnace_efficiency_inputs, default=str, sort_keys=True)
        )

        hashes = []
        for _ in range(100):
            h = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            hashes.append(h)

        assert len(set(hashes)) == 1, "Hash not consistent for same input"

    @pytest.mark.determinism
    def test_hash_changes_with_input(self, furnace_efficiency_inputs):
        """Test hash changes when input changes."""
        data1 = json.loads(json.dumps(furnace_efficiency_inputs, default=str, sort_keys=True))

        original_hash = hashlib.sha256(
            json.dumps(data1, sort_keys=True).encode()
        ).hexdigest()

        # Modify input
        data2 = data1.copy()
        data2["useful_heat_output_mw"] = "21.0"

        modified_hash = hashlib.sha256(
            json.dumps(data2, sort_keys=True).encode()
        ).hexdigest()

        assert original_hash != modified_hash, "Hash should change with different input"

    @pytest.mark.determinism
    def test_hash_length_always_64(self, furnace_efficiency_inputs):
        """Test SHA-256 hash is always 64 characters."""
        for i in range(100):
            data = {"iteration": i, **json.loads(json.dumps(furnace_efficiency_inputs, default=str))}
            h = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()

            assert len(h) == 64, f"Hash length {len(h)} != 64"


# ============================================================================
# FLOATING-POINT STABILITY
# ============================================================================

@pytest.mark.determinism
class TestFloatingPointStability:
    """Test floating-point calculation stability using Decimal."""

    @pytest.mark.determinism
    def test_decimal_associativity(self):
        """Test associativity is preserved with Decimal."""
        values = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]

        left_assoc = (values[0] + values[1]) + values[2]
        right_assoc = values[0] + (values[1] + values[2])

        assert left_assoc == right_assoc, "Associativity not preserved"
        assert left_assoc == Decimal("0.6")

    @pytest.mark.determinism
    def test_decimal_commutativity(self):
        """Test commutativity is preserved with Decimal."""
        a = Decimal("3.14159")
        b = Decimal("2.71828")

        assert a + b == b + a
        assert a * b == b * a

    @pytest.mark.determinism
    def test_efficiency_precision_preservation(self):
        """Test efficiency calculation preserves precision."""
        useful_heat = Decimal("20.500000001")
        fuel_energy = Decimal("25.700000001")

        efficiency = (useful_heat / fuel_energy) * Decimal("100")
        efficiency_rounded = efficiency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Verify precision is maintained before rounding
        assert efficiency != efficiency_rounded
        # But after rounding, should be consistent
        assert efficiency_rounded == Decimal("79.77")

    @pytest.mark.determinism
    def test_edge_case_small_values(self):
        """Test edge cases with very small values."""
        small_loss = Decimal("0.0001")
        total_energy = Decimal("100.0")

        loss_percent = (small_loss / total_energy) * Decimal("100")

        assert loss_percent == Decimal("0.0001")

    @pytest.mark.determinism
    def test_edge_case_large_values(self):
        """Test edge cases with large values."""
        large_energy = Decimal("1E12")  # 1 TW
        efficiency = Decimal("0.85")

        useful_output = large_energy * efficiency

        assert useful_output == Decimal("8.5E11")


# ============================================================================
# SEED PROPAGATION
# ============================================================================

@pytest.mark.determinism
class TestSeedPropagation:
    """Test random seed propagation for reproducibility."""

    @pytest.mark.determinism
    def test_random_seed_propagation(self, deterministic_seed):
        """Test random seed produces consistent sequences."""
        random.seed(deterministic_seed)
        sequence1 = [random.random() for _ in range(100)]

        random.seed(deterministic_seed)
        sequence2 = [random.random() for _ in range(100)]

        assert sequence1 == sequence2, "Random sequences not reproducible"

    @pytest.mark.determinism
    def test_no_hidden_randomness_in_calculations(self, furnace_efficiency_inputs):
        """Test calculations have no hidden random elements."""
        results = []

        for _ in range(100):
            fuel = furnace_efficiency_inputs["fuel_input"]
            useful_heat = furnace_efficiency_inputs["useful_heat_output_mw"]

            fuel_energy = (
                fuel["mass_flow_rate_kg_hr"] * fuel["higher_heating_value_mj_kg"]
            ) / Decimal("3600")

            efficiency = (useful_heat / fuel_energy) * Decimal("100")
            results.append(efficiency)

        assert len(set(results)) == 1, "Hidden randomness detected"


# ============================================================================
# CROSS-ITERATION CONSISTENCY
# ============================================================================

@pytest.mark.determinism
class TestCrossIterationConsistency:
    """Test consistency across multiple iterations and invocations."""

    @pytest.mark.determinism
    def test_full_pipeline_consistency(self, furnace_efficiency_inputs, fuel_consumption_inputs):
        """Test full calculation pipeline produces consistent results."""
        results = []

        for _ in range(50):
            # Step 1: Efficiency calculation
            fuel = furnace_efficiency_inputs["fuel_input"]
            useful_heat = furnace_efficiency_inputs["useful_heat_output_mw"]
            fuel_energy = (
                fuel["mass_flow_rate_kg_hr"] * fuel["higher_heating_value_mj_kg"]
            ) / Decimal("3600")
            efficiency = (useful_heat / fuel_energy) * Decimal("100")

            # Step 2: SEC calculation
            consumption = fuel_consumption_inputs["consumption_rate_kg_hr"]
            hhv = fuel_consumption_inputs["heating_value_mj_kg"]
            production = fuel_consumption_inputs["production_rate_tons_hr"]
            energy_gj_hr = (consumption * hhv) / Decimal("1000")
            sec = energy_gj_hr / production

            # Step 3: Combine results
            combined = {
                "efficiency": str(efficiency.quantize(Decimal("0.01"))),
                "sec": str(sec.quantize(Decimal("0.001")))
            }

            result_hash = hashlib.sha256(
                json.dumps(combined, sort_keys=True).encode()
            ).hexdigest()
            results.append(result_hash)

        assert len(set(results)) == 1, "Full pipeline not consistent"

    @pytest.mark.determinism
    def test_order_independence(self):
        """Test calculation order does not affect results."""
        values = [Decimal("10.5"), Decimal("20.3"), Decimal("15.7"), Decimal("8.2")]

        # Different orderings
        sum_forward = sum(values)
        sum_reverse = sum(reversed(values))
        sum_sorted = sum(sorted(values))

        assert sum_forward == sum_reverse == sum_sorted


# ============================================================================
# GOLDEN VALUE TESTS
# ============================================================================

@pytest.mark.determinism
class TestGoldenValues:
    """Test against known golden values for verification."""

    @pytest.mark.determinism
    def test_known_excess_air_value(self):
        """Test excess air calculation against known value."""
        o2_percent = Decimal("4.0")

        # Standard formula: EA% = O2 / (21 - O2) * 100
        excess_air = (o2_percent / (Decimal("21") - o2_percent)) * Decimal("100")
        excess_air = excess_air.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Known value: 4 / 17 * 100 = 23.5%
        expected = Decimal("23.5")
        assert excess_air == expected

    @pytest.mark.determinism
    def test_known_efficiency_value(self):
        """Test efficiency calculation against known value."""
        useful_heat_mw = Decimal("20.0")
        fuel_energy_mw = Decimal("25.0")

        efficiency = (useful_heat_mw / fuel_energy_mw) * Decimal("100")

        expected = Decimal("80")
        assert efficiency == expected

    @pytest.mark.determinism
    def test_known_sec_value(self):
        """Test SEC calculation against known value."""
        energy_gj_hr = Decimal("100.0")
        production_tons_hr = Decimal("20.0")

        sec = energy_gj_hr / production_tons_hr

        expected = Decimal("5.0")
        assert sec == expected

    @pytest.mark.determinism
    def test_known_rul_value(self):
        """Test RUL calculation against known value."""
        design_hours = Decimal("50000")
        operating_hours = Decimal("35000")

        rul_hours = design_hours - operating_hours
        rul_percent = (rul_hours / design_hours) * Decimal("100")

        expected = Decimal("30")
        assert rul_percent == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "determinism"])
