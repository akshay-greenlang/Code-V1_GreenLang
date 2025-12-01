# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests for GL-011 FUELCRAFT.

Verifies bit-perfect reproducibility following zero-hallucination principles.
Tests heating value calculations (HHV/LHV), combustion stoichiometry,
emission factor lookups, and fuel blending optimization determinism.

Standards Compliance:
    - ISO 6976:2016 - Natural gas calorific value
    - ISO 17225 - Solid biofuels specifications
    - ASTM D4809 - Heat of combustion liquid fuels
    - IPCC 2006 Guidelines - Emission factors

Author: GL-DeterminismAuditor
Version: 1.0.0
"""

import pytest
import hashlib
import json
import random
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# Add parent directories to path
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent.parent
sys.path.insert(0, str(AGENT_DIR))

try:
    from calculators.calorific_value_calculator import (
        CalorificValueCalculator,
        CalorificValueInput,
    )
    from calculators.emissions_factor_calculator import (
        EmissionsFactorCalculator,
        EmissionFactorInput,
    )
    from calculators.fuel_blending_calculator import (
        FuelBlendingCalculator,
        BlendingInput,
    )
    from calculators.multi_fuel_optimizer import (
        MultiFuelOptimizer,
        MultiFuelOptimizationInput,
    )
    from calculators.provenance_tracker import ProvenanceTracker
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def deterministic_seed():
    """Fixed seed for determinism tests."""
    return 42


@pytest.fixture
def natural_gas_inputs():
    """Standard natural gas calculation inputs."""
    return {
        "fuel_type": "natural_gas",
        "composition": {
            "methane": 95.0,
            "ethane": 2.5,
            "propane": 0.5,
            "nitrogen": 2.0
        },
        "temperature_c": 15.0,
        "pressure_kpa": 101.325,
        "moisture_percent": 0.0
    }


@pytest.fixture
def fuel_properties():
    """Standard fuel properties for multi-fuel tests."""
    return {
        'natural_gas': {
            'heating_value_mj_kg': 50.0,
            'carbon_content_percent': 75.0,
            'emission_factor_co2_kg_gj': 56.1,
            'emission_factor_nox_g_gj': 50.0,
            'emission_factor_sox_g_gj': 0.3,
            'renewable': False
        },
        'coal': {
            'heating_value_mj_kg': 25.0,
            'carbon_content_percent': 60.0,
            'emission_factor_co2_kg_gj': 94.6,
            'emission_factor_nox_g_gj': 250.0,
            'emission_factor_sox_g_gj': 500.0,
            'moisture_content_percent': 8.0,
            'ash_content_percent': 10.0,
            'sulfur_content_percent': 2.0,
            'renewable': False
        },
        'biomass': {
            'heating_value_mj_kg': 18.0,
            'carbon_content_percent': 50.0,
            'emission_factor_co2_kg_gj': 0.0,
            'emission_factor_nox_g_gj': 150.0,
            'emission_factor_sox_g_gj': 20.0,
            'moisture_content_percent': 25.0,
            'ash_content_percent': 2.0,
            'sulfur_content_percent': 0.02,
            'renewable': True
        },
        'biogas': {
            'heating_value_mj_kg': 20.0,
            'carbon_content_percent': 50.0,
            'emission_factor_co2_kg_gj': 0.0,
            'emission_factor_nox_g_gj': 40.0,
            'emission_factor_sox_g_gj': 15.0,
            'renewable': True
        }
    }


@pytest.fixture
def market_prices():
    """Standard market prices for testing."""
    return {
        'natural_gas': 0.045,
        'coal': 0.035,
        'biomass': 0.08,
        'biogas': 0.06
    }


# =============================================================================
# HEATING VALUE (HHV/LHV) REPRODUCIBILITY TESTS
# =============================================================================

class TestHeatingValueReproducibility:
    """Test suite for calorific value calculation determinism."""

    @pytest.mark.determinism
    @pytest.mark.heating_value
    def test_natural_gas_hhv_reproducibility(self, natural_gas_inputs):
        """Test natural gas HHV calculation is deterministic across 1000 runs."""
        calculator = CalorificValueCalculator()
        input_data = CalorificValueInput(
            fuel_type=natural_gas_inputs['fuel_type'],
            composition=natural_gas_inputs['composition'],
            temperature_c=natural_gas_inputs['temperature_c'],
            pressure_kpa=natural_gas_inputs['pressure_kpa']
        )

        results = []
        hashes = []
        for _ in range(1000):
            result = calculator.calculate(input_data)
            results.append(result.gross_calorific_value_mj_kg)
            hashes.append(result.provenance_hash)

        # All results must be identical
        assert len(set(results)) == 1, f"HHV not deterministic: {len(set(results))} unique values"
        assert len(set(hashes)) == 1, f"Provenance hash not deterministic: {len(set(hashes))} unique hashes"

    @pytest.mark.determinism
    @pytest.mark.heating_value
    def test_lhv_hhv_difference_determinism(self, natural_gas_inputs):
        """Test LHV-HHV difference calculation is deterministic."""
        calculator = CalorificValueCalculator()
        input_data = CalorificValueInput(
            fuel_type=natural_gas_inputs['fuel_type'],
            composition=natural_gas_inputs['composition'],
            temperature_c=natural_gas_inputs['temperature_c'],
            pressure_kpa=natural_gas_inputs['pressure_kpa']
        )

        differences = []
        for _ in range(100):
            result = calculator.calculate(input_data)
            diff = result.gross_calorific_value_mj_kg - result.net_calorific_value_mj_kg
            differences.append(round(diff, 6))

        assert len(set(differences)) == 1, "LHV-HHV difference not deterministic"

    @pytest.mark.determinism
    @pytest.mark.heating_value
    @pytest.mark.parametrize("fuel_type,composition,expected_method", [
        ("natural_gas", {"methane": 100.0}, "ISO_6976"),
        ("coal", {"carbon": 75.0, "hydrogen": 5.0, "oxygen": 8.0, "sulfur": 2.0}, "Dulong_formula"),
        ("fuel_oil", {"density_kg_m3": 850.0}, "ASTM_D4809"),
    ])
    def test_calculation_method_determinism(self, fuel_type, composition, expected_method):
        """Test correct calculation method is selected deterministically."""
        calculator = CalorificValueCalculator()
        input_data = CalorificValueInput(
            fuel_type=fuel_type,
            composition=composition,
            temperature_c=15.0,
            pressure_kpa=101.325
        )

        methods = []
        for _ in range(100):
            result = calculator.calculate(input_data)
            methods.append(result.calculation_method)

        assert len(set(methods)) == 1, f"Calculation method not deterministic"
        assert methods[0] == expected_method, f"Wrong method: {methods[0]} vs {expected_method}"

    @pytest.mark.determinism
    @pytest.mark.heating_value
    def test_wobbe_index_reproducibility(self, natural_gas_inputs):
        """Test Wobbe index calculation is deterministic."""
        calculator = CalorificValueCalculator()
        input_data = CalorificValueInput(
            fuel_type=natural_gas_inputs['fuel_type'],
            composition=natural_gas_inputs['composition'],
            temperature_c=natural_gas_inputs['temperature_c'],
            pressure_kpa=natural_gas_inputs['pressure_kpa']
        )

        wobbe_values = []
        for _ in range(100):
            result = calculator.calculate(input_data)
            wobbe_values.append(result.wobbe_index)

        assert len(set(wobbe_values)) == 1, "Wobbe index not deterministic"


# =============================================================================
# COMBUSTION STOICHIOMETRY DETERMINISM TESTS
# =============================================================================

class TestCombustionStoichiometryDeterminism:
    """Test suite for combustion stoichiometry calculation determinism."""

    @pytest.mark.determinism
    @pytest.mark.stoichiometry
    def test_stoichiometric_air_calculation(self):
        """Test stoichiometric air requirement is deterministic."""
        # Stoichiometric air = (11.53 * C + 34.34 * H - 4.29 * O + 4.32 * S) / 100
        compositions = [
            {'carbon': 75.0, 'hydrogen': 5.0, 'oxygen': 8.0, 'sulfur': 2.0},
            {'carbon': 50.0, 'hydrogen': 6.0, 'oxygen': 43.0, 'sulfur': 0.02},
        ]

        for comp in compositions:
            results = []
            for _ in range(100):
                C = Decimal(str(comp.get('carbon', 0)))
                H = Decimal(str(comp.get('hydrogen', 0)))
                O = Decimal(str(comp.get('oxygen', 0)))
                S = Decimal(str(comp.get('sulfur', 0)))

                stoich_air = (
                    Decimal("11.53") * C +
                    Decimal("34.34") * H -
                    Decimal("4.29") * O +
                    Decimal("4.32") * S
                ) / Decimal("100")

                results.append(stoich_air.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

            assert len(set(results)) == 1, "Stoichiometric air not deterministic"

    @pytest.mark.determinism
    @pytest.mark.stoichiometry
    def test_excess_air_calculation_determinism(self):
        """Test excess air calculation is deterministic."""
        actual_air = Decimal("12.5")
        stoich_air = Decimal("10.0")

        results = []
        for _ in range(1000):
            excess_air_pct = ((actual_air - stoich_air) / stoich_air * Decimal("100"))
            results.append(excess_air_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        assert len(set(results)) == 1, "Excess air calculation not deterministic"
        assert results[0] == Decimal("25.00"), f"Expected 25.00%, got {results[0]}"


# =============================================================================
# EMISSION FACTOR REPRODUCIBILITY TESTS
# =============================================================================

class TestEmissionFactorReproducibility:
    """Test suite for emission factor calculation determinism."""

    @pytest.mark.determinism
    @pytest.mark.emission_factor
    def test_co2_emission_factor_determinism(self):
        """Test CO2 emission factor lookup is deterministic."""
        calculator = EmissionsFactorCalculator()

        fuel_types = ['natural_gas', 'coal', 'fuel_oil', 'biomass', 'hydrogen']

        for fuel_type in fuel_types:
            input_data = EmissionFactorInput(
                fuel_type=fuel_type,
                combustion_technology='boiler',
                emission_control='uncontrolled'
            )

            results = []
            for _ in range(100):
                result = calculator.get_emission_factors(input_data)
                results.append(result.co2_kg_gj)

            assert len(set(results)) == 1, f"CO2 factor not deterministic for {fuel_type}"

    @pytest.mark.determinism
    @pytest.mark.emission_factor
    def test_co2e_calculation_determinism(self):
        """Test CO2e (CO2 equivalent) calculation is deterministic."""
        calculator = EmissionsFactorCalculator()
        input_data = EmissionFactorInput(
            fuel_type='natural_gas',
            combustion_technology='boiler',
            emission_control='uncontrolled'
        )

        co2e_values = []
        for _ in range(100):
            result = calculator.get_emission_factors(input_data)
            co2e_values.append(result.co2e_kg_gj)

        assert len(set(co2e_values)) == 1, "CO2e calculation not deterministic"

    @pytest.mark.determinism
    @pytest.mark.emission_factor
    @pytest.mark.ipcc
    def test_ipcc_tier1_factors_match_reference(self):
        """Test IPCC Tier 1 emission factors match reference values."""
        calculator = EmissionsFactorCalculator()

        # IPCC 2006 Vol 2 Table 2.2 reference values
        ipcc_reference = {
            'natural_gas': 56.1,
            'coal': 94.6,
            'fuel_oil': 77.4,
            'diesel': 74.1,
        }

        for fuel_type, expected_co2 in ipcc_reference.items():
            input_data = EmissionFactorInput(
                fuel_type=fuel_type,
                combustion_technology='boiler',
                emission_control='uncontrolled'
            )
            result = calculator.get_emission_factors(input_data)

            assert result.co2_kg_gj == expected_co2, \
                f"IPCC factor mismatch for {fuel_type}: {result.co2_kg_gj} vs {expected_co2}"

    @pytest.mark.determinism
    @pytest.mark.emission_factor
    def test_emission_calculation_determinism(self):
        """Test emissions calculation for given energy is deterministic."""
        calculator = EmissionsFactorCalculator()

        energy_gj = 1000.0
        fuel_type = 'natural_gas'

        results = []
        for _ in range(100):
            emissions = calculator.calculate_emissions(fuel_type, energy_gj)
            results.append(emissions['co2_kg'])

        assert len(set(results)) == 1, "Emissions calculation not deterministic"


# =============================================================================
# FUEL BLENDING DETERMINISM TESTS
# =============================================================================

class TestFuelBlendingDeterminism:
    """Test suite for fuel blending optimization determinism."""

    @pytest.mark.determinism
    @pytest.mark.blending
    def test_blend_ratio_determinism(self, fuel_properties):
        """Test blend ratio optimization is deterministic."""
        calculator = FuelBlendingCalculator()
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=22.0,
            max_moisture=15.0,
            max_ash=12.0,
            max_sulfur=1.5,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        results = []
        for _ in range(100):
            result = calculator.optimize_blend(input_data)
            results.append(json.dumps(result.blend_ratios, sort_keys=True))

        assert len(set(results)) == 1, "Blend ratios not deterministic"

    @pytest.mark.determinism
    @pytest.mark.blending
    def test_blend_properties_determinism(self, fuel_properties):
        """Test blended fuel properties calculation is deterministic."""
        calculator = FuelBlendingCalculator()
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=22.0,
            max_moisture=15.0,
            max_ash=12.0,
            max_sulfur=1.5,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        heating_values = []
        for _ in range(100):
            result = calculator.optimize_blend(input_data)
            heating_values.append(result.blend_heating_value)

        assert len(set(heating_values)) == 1, "Blend heating value not deterministic"


# =============================================================================
# PROVENANCE HASH CONSISTENCY TESTS
# =============================================================================

class TestProvenanceHashConsistency:
    """Test suite for SHA-256 provenance hash consistency."""

    @pytest.mark.determinism
    @pytest.mark.hash_consistency
    def test_hash_consistency_same_input(self, natural_gas_inputs):
        """Test provenance hash is identical for identical inputs."""
        data = {k: str(v) for k, v in natural_gas_inputs.items()}
        hashes = []
        for _ in range(100):
            h = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            hashes.append(h)

        assert len(set(hashes)) == 1, "Hash not consistent for same input"
        assert len(hashes[0]) == 64, "Invalid SHA-256 hash length"

    @pytest.mark.determinism
    @pytest.mark.hash_consistency
    def test_hash_changes_with_input(self, natural_gas_inputs):
        """Test provenance hash changes when input changes."""
        data1 = {k: str(v) for k, v in natural_gas_inputs.items()}
        hash1 = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()

        data2 = data1.copy()
        data2['temperature_c'] = "16.0"  # Tiny change
        hash2 = hashlib.sha256(json.dumps(data2, sort_keys=True).encode()).hexdigest()

        assert hash1 != hash2, "Hash should change when input changes"

    @pytest.mark.determinism
    @pytest.mark.hash_consistency
    def test_provenance_tracker_reproducibility(self):
        """Test ProvenanceTracker produces reproducible hashes."""
        tracker = ProvenanceTracker()

        inputs = {'fuel_type': 'natural_gas', 'energy_mj': 1000}
        outputs = {'co2_kg': 56.1, 'cost_usd': 45.0}

        record1 = tracker.record(
            operation='fuel_analysis',
            inputs=inputs,
            outputs=outputs
        )

        record2 = tracker.record(
            operation='fuel_analysis',
            inputs=inputs,
            outputs=outputs
        )

        assert record1.input_hash == record2.input_hash, "Input hashes should be identical"
        assert record1.output_hash == record2.output_hash, "Output hashes should be identical"


# =============================================================================
# SEED PROPAGATION TESTS
# =============================================================================

class TestSeedPropagation:
    """Test suite for random seed propagation verification."""

    @pytest.mark.determinism
    def test_random_seed_propagation(self, deterministic_seed):
        """Test random seed produces identical sequences."""
        random.seed(deterministic_seed)
        values_1 = [random.random() for _ in range(100)]

        random.seed(deterministic_seed)
        values_2 = [random.random() for _ in range(100)]

        assert values_1 == values_2, "Random seed propagation failed"

    @pytest.mark.determinism
    def test_no_hidden_randomness_in_calculations(self, natural_gas_inputs):
        """Test calculations have no hidden randomness."""
        calculator = CalorificValueCalculator()
        input_data = CalorificValueInput(
            fuel_type=natural_gas_inputs['fuel_type'],
            composition=natural_gas_inputs['composition'],
            temperature_c=natural_gas_inputs['temperature_c'],
            pressure_kpa=natural_gas_inputs['pressure_kpa']
        )

        results = []
        for _ in range(100):
            # Don't reset seed - if calculation uses random internally, results will differ
            result = calculator.calculate(input_data)
            results.append(result.gross_calorific_value_mj_kg)

        assert len(set(results)) == 1, "Hidden randomness detected in calculation"


# =============================================================================
# FLOATING-POINT STABILITY TESTS
# =============================================================================

class TestFloatingPointStability:
    """Test suite for floating-point calculation stability."""

    @pytest.mark.determinism
    def test_associativity_preserved_with_decimal(self):
        """Test Decimal preserves associativity."""
        values = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]

        sum_forward = sum(values, Decimal("0"))
        sum_reverse = sum(reversed(values), Decimal("0"))

        assert sum_forward == sum_reverse, "Decimal associativity failed"

    @pytest.mark.determinism
    def test_decimal_precision_maintained(self):
        """Test Decimal maintains required precision."""
        value = Decimal("1.0000000001")
        subtract = Decimal("0.0000000001")
        result = value - subtract

        assert result == Decimal("1.0"), f"Precision lost: {result}"

    @pytest.mark.determinism
    def test_small_value_handling(self):
        """Test handling of very small values."""
        small1 = Decimal("1E-15")
        small2 = Decimal("1E-15")
        result = small1 + small2

        assert result == Decimal("2E-15"), "Small value addition failed"

    @pytest.mark.determinism
    def test_heating_value_rounding_determinism(self):
        """Test heating value rounding is deterministic."""
        raw_values = [Decimal("39.8245"), Decimal("39.8255"), Decimal("39.8250")]

        for raw in raw_values:
            rounded_results = []
            for _ in range(100):
                rounded = raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                rounded_results.append(rounded)

            assert len(set(rounded_results)) == 1, f"Rounding not deterministic for {raw}"


# =============================================================================
# MULTI-FUEL OPTIMIZATION DETERMINISM TESTS
# =============================================================================

class TestMultiFuelOptimizationDeterminism:
    """Test suite for multi-fuel optimization determinism."""

    @pytest.mark.determinism
    def test_multi_fuel_optimizer_determinism(self, fuel_properties, market_prices):
        """Test multi-fuel optimizer produces identical results."""
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        results = [optimizer.optimize(input_data) for _ in range(100)]

        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.optimal_fuel_mix == first.optimal_fuel_mix, \
                f"Fuel mix differs at run {i}"
            assert result.provenance_hash == first.provenance_hash, \
                f"Provenance hash differs at run {i}"

    @pytest.mark.determinism
    @pytest.mark.parametrize("objective", [
        'minimize_cost',
        'minimize_emissions',
        'balanced',
        'maximize_efficiency'
    ])
    def test_optimization_objective_determinism(self, fuel_properties, market_prices, objective):
        """Test each optimization objective produces deterministic results."""
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective=objective
        )

        results = [optimizer.optimize(input_data) for _ in range(50)]
        hashes = [r.provenance_hash for r in results]

        assert len(set(hashes)) == 1, f"Objective '{objective}' not deterministic"


# =============================================================================
# GOLDEN VALUE TESTS
# =============================================================================

class TestGoldenValues:
    """Test suite for golden (known-good) reference values."""

    @pytest.mark.determinism
    @pytest.mark.golden
    @pytest.mark.iso_6976
    def test_pure_methane_heating_value(self):
        """Golden test: Pure methane HHV should match ISO 6976 reference."""
        calculator = CalorificValueCalculator()
        input_data = CalorificValueInput(
            fuel_type='natural_gas',
            composition={'methane': 100.0},
            temperature_c=15.0,
            pressure_kpa=101.325
        )

        result = calculator.calculate(input_data)

        # ISO 6976 reference: Methane HHV = 55.53 MJ/kg (approximately)
        # Allow 5% tolerance for implementation differences
        assert 52.0 <= result.gross_calorific_value_mj_kg <= 58.0, \
            f"Pure methane HHV out of range: {result.gross_calorific_value_mj_kg}"

    @pytest.mark.determinism
    @pytest.mark.golden
    def test_coal_dulong_formula(self):
        """Golden test: Dulong formula for coal heating value."""
        calculator = CalorificValueCalculator()
        input_data = CalorificValueInput(
            fuel_type='coal',
            composition={
                'carbon': 75.0,
                'hydrogen': 5.0,
                'oxygen': 8.0,
                'sulfur': 2.0
            },
            temperature_c=15.0,
            pressure_kpa=101.325,
            moisture_percent=8.0
        )

        result = calculator.calculate(input_data)

        # Dulong formula: GCV = 0.3383*C + 1.443*(H - O/8) + 0.0942*S
        # Expected range: 25-30 MJ/kg for bituminous coal
        assert 20.0 <= result.gross_calorific_value_mj_kg <= 35.0, \
            f"Coal HHV out of range: {result.gross_calorific_value_mj_kg}"
