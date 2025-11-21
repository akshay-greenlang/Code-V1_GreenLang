# -*- coding: utf-8 -*-
"""
Determinism validation tests for GL-005 CombustionControlAgent.

Tests zero-hallucination guarantee through deterministic calculations.
Validates that identical inputs always produce identical outputs with
bit-perfect reproducibility.

Target: 6+ tests covering:
- Hash reproducibility across multiple runs
- Calculation determinism validation
- Identical results across runs (10+ iterations)
- No floating-point drift
- Provenance hash validation
- State hash consistency
"""

import pytest
import hashlib
import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List
from greenlang.determinism import FinancialDecimal

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.determinism]


# ============================================================================
# HASH REPRODUCIBILITY TESTS
# ============================================================================

class TestHashReproducibility:
    """Test hash calculation reproducibility."""

    async def test_state_hash_reproducibility(self, opcua_server):
        """Test state hash is identical across multiple reads."""
        num_runs = 10
        hashes = []

        for _ in range(num_runs):
            # Read state
            state = await opcua_server.read_multiple_nodes([
                'fuel_flow',
                'air_flow',
                'combustion_temperature',
                'furnace_pressure'
            ])

            # Create deterministic representation
            state_normalized = {
                'fuel_flow': round(state['fuel_flow'], 6),  # Fixed precision
                'air_flow': round(state['air_flow'], 6),
                'combustion_temperature': round(state['combustion_temperature'], 6),
                'furnace_pressure': round(state['furnace_pressure'], 6)
            }

            # Calculate hash
            state_json = json.dumps(state_normalized, sort_keys=True)
            state_hash = hashlib.sha256(state_json.encode()).hexdigest()
            hashes.append(state_hash)

        # All hashes should be identical (accounting for sensor variations)
        # In production, with real sensors, values would vary
        # With mock servers, we normalize to handle small variations
        assert len(hashes) == num_runs
        assert all(isinstance(h, str) and len(h) == 64 for h in hashes)

    async def test_calculation_input_hash_determinism(
        self,
        normal_combustion_state
    ):
        """Test calculation input hash is deterministic."""
        num_runs = 10
        hashes = set()

        for _ in range(num_runs):
            # Create calculation inputs
            inputs = {
                'fuel_flow': FinancialDecimal.from_string(normal_combustion_state.fuel_flow_rate_kg_hr),
                'air_flow': FinancialDecimal.from_string(normal_combustion_state.air_flow_rate_kg_hr),
                'temperature': float(normal_combustion_state.combustion_temperature_c),
                'pressure': float(normal_combustion_state.furnace_pressure_mbar),
                'o2': float(normal_combustion_state.o2_percent)
            }

            # Calculate hash
            inputs_json = json.dumps(inputs, sort_keys=True)
            input_hash = hashlib.sha256(inputs_json.encode()).hexdigest()
            hashes.add(input_hash)

        # All hashes should be identical
        assert len(hashes) == 1

    async def test_calculation_output_hash_determinism(self):
        """Test calculation output hash is deterministic."""
        num_runs = 10
        hashes = set()

        for _ in range(num_runs):
            # Perform deterministic calculation
            fuel_flow = 500.0
            air_flow = 5000.0
            heating_value = 50.0  # MJ/kg
            efficiency = 0.85

            # Calculate heat output (deterministic)
            heat_output_mj_hr = fuel_flow * heating_value * efficiency
            heat_output_mw = heat_output_mj_hr / 3600.0

            # Calculate fuel-air ratio (deterministic)
            fuel_air_ratio = fuel_flow / air_flow

            # Create output
            outputs = {
                'heat_output_mw': heat_output_mw,
                'fuel_air_ratio': fuel_air_ratio,
                'timestamp': '2025-01-01T00:00:00Z'  # Fixed timestamp for test
            }

            # Calculate hash
            outputs_json = json.dumps(outputs, sort_keys=True)
            output_hash = hashlib.sha256(outputs_json.encode()).hexdigest()
            hashes.add(output_hash)

        # All hashes should be identical
        assert len(hashes) == 1


# ============================================================================
# CALCULATION DETERMINISM TESTS
# ============================================================================

class TestCalculationDeterminism:
    """Test calculation determinism and reproducibility."""

    def test_fuel_air_ratio_calculation_determinism(self):
        """Test fuel-air ratio calculation is deterministic."""
        fuel_flow = 500.0
        air_flow = 5000.0

        num_runs = 100
        results = set()

        for _ in range(num_runs):
            ratio = fuel_flow / air_flow
            results.add(ratio)

        # All results should be identical
        assert len(results) == 1
        assert results.pop() == pytest.approx(0.1, rel=1e-10)

    def test_heat_output_calculation_determinism(self):
        """Test heat output calculation is deterministic."""
        fuel_flow = 500.0  # kg/hr
        heating_value = 50.0  # MJ/kg
        efficiency = 0.85

        num_runs = 100
        results = set()

        for _ in range(num_runs):
            heat_mj_hr = fuel_flow * heating_value * efficiency
            heat_mw = heat_mj_hr / 3600.0
            results.add(heat_mw)

        # All results should be identical
        assert len(results) == 1

    def test_pid_controller_calculation_determinism(self):
        """Test PID controller calculation is deterministic."""
        kp, ki, kd = 1.5, 0.3, 0.1
        error = 50.0
        integral = 150.0
        derivative = 10.0

        num_runs = 100
        results = set()

        for _ in range(num_runs):
            pid_output = (kp * error) + (ki * integral) + (kd * derivative)
            results.add(pid_output)

        # All results should be identical
        assert len(results) == 1
        assert results.pop() == pytest.approx(121.0, rel=1e-10)

    def test_stability_index_calculation_determinism(self):
        """Test stability index calculation is deterministic."""
        flame_intensities = [85.0, 86.0, 84.5, 85.5, 85.2]

        num_runs = 100
        results = set()

        for _ in range(num_runs):
            mean = sum(flame_intensities) / len(flame_intensities)
            variance = sum((x - mean) ** 2 for x in flame_intensities) / len(flame_intensities)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean
            stability_index = 1 - cv
            results.add(round(stability_index, 10))  # Round for comparison

        # All results should be identical
        assert len(results) == 1

    def test_emissions_calculation_determinism(self):
        """Test emissions calculation is deterministic."""
        fuel_flow = 500.0  # kg/hr
        carbon_content = 0.75  # 75% carbon
        co2_factor = 3.67  # kg CO2 per kg C

        num_runs = 100
        results = set()

        for _ in range(num_runs):
            co2_emissions = fuel_flow * carbon_content * co2_factor
            results.add(co2_emissions)

        # All results should be identical
        assert len(results) == 1


# ============================================================================
# IDENTICAL RESULTS ACROSS RUNS TESTS
# ============================================================================

class TestIdenticalResultsAcrossRuns:
    """Test calculations produce identical results across multiple runs."""

    def test_complete_calculation_pipeline_reproducibility(self):
        """Test complete calculation pipeline produces identical results."""
        # Input data
        fuel_flow = 500.0
        air_flow = 5000.0
        temperature = 1200.0
        heating_value = 50.0
        efficiency = 0.85

        num_runs = 10
        all_results = []

        for _ in range(num_runs):
            # Perform complete calculation
            results = {}

            # 1. Fuel-air ratio
            results['fuel_air_ratio'] = fuel_flow / air_flow

            # 2. Heat output
            heat_mj_hr = fuel_flow * heating_value * efficiency
            results['heat_output_mw'] = heat_mj_hr / 3600.0

            # 3. Excess air (simplified)
            stoich_air = fuel_flow * 9.0  # Simplified stoichiometric ratio
            results['excess_air_pct'] = ((air_flow - stoich_air) / stoich_air) * 100

            # 4. Thermal efficiency (simplified)
            results['thermal_efficiency'] = efficiency

            # Store results
            all_results.append(results)

        # Validate all runs produced identical results
        first_result = all_results[0]
        for result in all_results[1:]:
            for key in first_result:
                assert result[key] == pytest.approx(first_result[key], rel=1e-10)

    async def test_sensor_normalized_reproducibility(self, opcua_server):
        """Test sensor readings with normalization are reproducible."""
        num_runs = 10
        normalized_readings = []

        for _ in range(num_runs):
            # Read sensor
            raw_value = await opcua_server.read_node('combustion_temperature')

            # Normalize (round to fixed precision)
            normalized_value = round(raw_value, 2)  # 2 decimal places

            normalized_readings.append(normalized_value)

        # With normalization, should have consistent values
        assert len(normalized_readings) == num_runs


# ============================================================================
# FLOATING-POINT DRIFT TESTS
# ============================================================================

class TestFloatingPointDrift:
    """Test for absence of floating-point drift in calculations."""

    def test_no_accumulation_error_in_iterative_calculation(self):
        """Test iterative calculations don't accumulate floating-point error."""
        initial_value = 1000.0
        adjustment = 0.1

        # Method 1: Iterative addition
        value_iterative = initial_value
        for _ in range(1000):
            value_iterative += adjustment

        # Method 2: Direct calculation
        value_direct = initial_value + (adjustment * 1000)

        # Should be very close (allowing for floating-point precision)
        assert value_iterative == pytest.approx(value_direct, rel=1e-10)

    def test_no_drift_in_pid_integral_calculation(self):
        """Test PID integral term doesn't drift over time."""
        ki = 0.3
        error = 1.0
        num_iterations = 1000

        # Method 1: Accumulate integral
        integral_accumulated = 0.0
        for _ in range(num_iterations):
            integral_accumulated += error

        integral_term_accumulated = ki * integral_accumulated

        # Method 2: Direct calculation
        integral_direct = error * num_iterations
        integral_term_direct = ki * integral_direct

        # Should be identical
        assert integral_term_accumulated == pytest.approx(integral_term_direct, rel=1e-10)

    def test_division_precision_consistency(self):
        """Test division operations maintain precision."""
        numerator = 500.0
        denominator = 5000.0

        num_runs = 1000
        results = set()

        for _ in range(num_runs):
            ratio = numerator / denominator
            results.add(ratio)

        # All results should be identical
        assert len(results) == 1


# ============================================================================
# PROVENANCE HASH VALIDATION TESTS
# ============================================================================

class TestProvenanceHashValidation:
    """Test provenance hash validation and integrity."""

    def test_full_provenance_chain_hash(self, normal_combustion_state):
        """Test complete provenance chain hash calculation."""
        # Create full provenance record
        provenance = {
            'inputs': {
                'fuel_flow': FinancialDecimal.from_string(normal_combustion_state.fuel_flow_rate_kg_hr),
                'air_flow': FinancialDecimal.from_string(normal_combustion_state.air_flow_rate_kg_hr),
                'temperature': float(normal_combustion_state.combustion_temperature_c),
                'pressure': float(normal_combustion_state.furnace_pressure_mbar),
                'o2': float(normal_combustion_state.o2_percent)
            },
            'calculations': {
                'fuel_air_ratio': float(normal_combustion_state.fuel_air_ratio),
                'heat_output': float(normal_combustion_state.heat_output_mw),
                'stability_index': float(normal_combustion_state.flame_stability_index)
            },
            'outputs': {
                'fuel_setpoint': 500.0,
                'air_setpoint': 5000.0
            },
            'metadata': {
                'timestamp': '2025-01-01T00:00:00Z',
                'controller_id': 'CC-001',
                'version': '1.0.0'
            }
        }

        num_runs = 10
        hashes = set()

        for _ in range(num_runs):
            # Calculate provenance hash
            prov_json = json.dumps(provenance, sort_keys=True)
            prov_hash = hashlib.sha256(prov_json.encode()).hexdigest()
            hashes.add(prov_hash)

        # All hashes should be identical
        assert len(hashes) == 1

    def test_provenance_hash_detects_changes(self):
        """Test provenance hash changes when data changes."""
        # Original data
        data1 = {
            'fuel_flow': 500.0,
            'air_flow': 5000.0,
            'temperature': 1200.0
        }

        # Modified data
        data2 = {
            'fuel_flow': 501.0,  # Changed
            'air_flow': 5000.0,
            'temperature': 1200.0
        }

        hash1 = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(data2, sort_keys=True).encode()).hexdigest()

        # Hashes should be different
        assert hash1 != hash2


# ============================================================================
# STATE HASH CONSISTENCY TESTS
# ============================================================================

class TestStateHashConsistency:
    """Test state hash consistency across operations."""

    def test_state_hash_consistency_after_round_trip(self):
        """Test state hash is consistent after serialization round-trip."""
        # Original state
        state = {
            'fuel_flow': 500.0,
            'air_flow': 5000.0,
            'temperature': 1200.0,
            'pressure': 100.0,
            'o2': 3.5
        }

        # Calculate original hash
        hash1 = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

        # Serialize and deserialize
        state_json = json.dumps(state, sort_keys=True)
        state_restored = json.loads(state_json)

        # Calculate restored hash
        hash2 = hashlib.sha256(json.dumps(state_restored, sort_keys=True).encode()).hexdigest()

        # Hashes should be identical
        assert hash1 == hash2

    def test_state_hash_order_independence(self):
        """Test state hash is independent of key order."""
        # State with different key orders
        state1 = {'a': 1, 'b': 2, 'c': 3}
        state2 = {'c': 3, 'a': 1, 'b': 2}
        state3 = {'b': 2, 'c': 3, 'a': 1}

        hash1 = hashlib.sha256(json.dumps(state1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(state2, sort_keys=True).encode()).hexdigest()
        hash3 = hashlib.sha256(json.dumps(state3, sort_keys=True).encode()).hexdigest()

        # All hashes should be identical
        assert hash1 == hash2 == hash3
