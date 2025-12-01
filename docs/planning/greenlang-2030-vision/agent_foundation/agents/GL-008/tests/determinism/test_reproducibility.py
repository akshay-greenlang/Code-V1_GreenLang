# -*- coding: utf-8 -*-
"""
Reproducibility validation tests for GL-008 TRAPCATCHER SteamTrapInspector.

This module validates bit-perfect reproducibility across multiple executions,
different runs, and various environmental conditions. Ensures that identical
inputs always produce identical outputs for regulatory compliance.

Reproducibility Requirements:
- All calculations must be bit-perfect reproducible
- Provenance hashes must be identical for identical inputs
- Results must be consistent across different execution orders
- Random seeds must be enforced for any stochastic components
"""

import pytest
import hashlib
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import sys
from pathlib import Path
import copy

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools import SteamTrapTools
from config import TrapInspectorConfig, TrapType, FailureMode


@pytest.fixture
def tools():
    """Create SteamTrapTools instance for testing."""
    return SteamTrapTools()


@pytest.fixture
def deterministic_config():
    """Create deterministic test configuration."""
    return TrapInspectorConfig(
        agent_id="GL-008-REPRO-TEST",
        enable_llm_classification=False,
        cache_ttl_seconds=60,
        max_concurrent_inspections=5,
        llm_temperature=0.0,
        llm_seed=42
    )


@pytest.fixture
def reference_acoustic_data():
    """Create reference acoustic data for reproducibility tests."""
    np.random.seed(42)
    return {
        'trap_id': 'TRAP-REPRO-ACOUSTIC',
        'signal': (np.random.randn(10000) * 0.2).tolist(),
        'sampling_rate_hz': 250000
    }


@pytest.fixture
def reference_thermal_data():
    """Create reference thermal data for reproducibility tests."""
    return {
        'trap_id': 'TRAP-REPRO-THERMAL',
        'temperature_upstream_c': 150.0,
        'temperature_downstream_c': 130.0,
        'ambient_temp_c': 20.0
    }


@pytest.mark.determinism
class TestAcousticReproducibility:
    """Test reproducibility of acoustic analysis."""

    def test_acoustic_identical_inputs_identical_outputs(self, tools, reference_acoustic_data):
        """Test that identical acoustic inputs produce identical outputs."""
        results = [
            tools.analyze_acoustic_signature(copy.deepcopy(reference_acoustic_data))
            for _ in range(10)
        ]

        first = results[0]
        for result in results[1:]:
            assert result.failure_probability == first.failure_probability, \
                "Failure probability differs across runs"
            assert result.confidence_score == first.confidence_score, \
                "Confidence score differs across runs"
            assert result.signal_strength_db == first.signal_strength_db, \
                "Signal strength differs across runs"
            assert result.frequency_peak_hz == first.frequency_peak_hz, \
                "Frequency peak differs across runs"
            assert result.provenance_hash == first.provenance_hash, \
                "Provenance hash differs across runs"

    def test_acoustic_hash_determinism(self, tools, reference_acoustic_data):
        """Test that provenance hash is deterministic."""
        hashes = []
        for _ in range(20):
            result = tools.analyze_acoustic_signature(copy.deepcopy(reference_acoustic_data))
            hashes.append(result.provenance_hash)

        # All hashes must be identical
        assert len(set(hashes)) == 1, \
            f"Expected 1 unique hash, got {len(set(hashes))}"

    def test_acoustic_different_inputs_different_hashes(self, tools):
        """Test that different inputs produce different provenance hashes."""
        np.random.seed(42)
        hashes = set()

        for i in range(50):
            np.random.seed(i)  # Different seed for each signal
            acoustic_data = {
                'trap_id': f'TRAP-DIFF-{i:03d}',
                'signal': (np.random.randn(10000) * 0.2).tolist(),
                'sampling_rate_hz': 250000
            }
            result = tools.analyze_acoustic_signature(acoustic_data)
            hashes.add(result.provenance_hash)

        # All hashes should be unique (no collisions)
        assert len(hashes) == 50, \
            f"Expected 50 unique hashes, got {len(hashes)}"

    def test_acoustic_bit_level_reproducibility(self, tools, reference_acoustic_data):
        """Test bit-level reproducibility using Python's repr."""
        results = [
            tools.analyze_acoustic_signature(copy.deepcopy(reference_acoustic_data))
            for _ in range(5)
        ]

        # Convert to comparable format
        first_repr = {
            'failure_prob': repr(results[0].failure_probability),
            'confidence': repr(results[0].confidence_score),
            'signal_db': repr(results[0].signal_strength_db),
            'freq_hz': repr(results[0].frequency_peak_hz)
        }

        for result in results[1:]:
            result_repr = {
                'failure_prob': repr(result.failure_probability),
                'confidence': repr(result.confidence_score),
                'signal_db': repr(result.signal_strength_db),
                'freq_hz': repr(result.frequency_peak_hz)
            }
            assert result_repr == first_repr, \
                "Bit-level representation differs"


@pytest.mark.determinism
class TestThermalReproducibility:
    """Test reproducibility of thermal analysis."""

    def test_thermal_identical_inputs_identical_outputs(self, tools, reference_thermal_data):
        """Test that identical thermal inputs produce identical outputs."""
        results = [
            tools.analyze_thermal_pattern(copy.deepcopy(reference_thermal_data))
            for _ in range(10)
        ]

        first = results[0]
        for result in results[1:]:
            assert result.trap_health_score == first.trap_health_score
            assert result.temperature_differential_c == first.temperature_differential_c
            assert result.condensate_pooling_detected == first.condensate_pooling_detected
            assert result.provenance_hash == first.provenance_hash

    def test_thermal_hash_collision_resistance(self, tools):
        """Test thermal hash collision resistance."""
        hashes = set()

        for upstream in range(100, 200, 5):
            for downstream in range(50, 150, 10):
                thermal_data = {
                    'trap_id': f'TRAP-THERMAL-{upstream}-{downstream}',
                    'temperature_upstream_c': float(upstream),
                    'temperature_downstream_c': float(downstream),
                    'ambient_temp_c': 20.0
                }
                result = tools.analyze_thermal_pattern(thermal_data)
                hashes.add(result.provenance_hash)

        # All hashes should be unique
        expected_count = len(range(100, 200, 5)) * len(range(50, 150, 10))
        assert len(hashes) == expected_count, \
            f"Hash collisions detected: {expected_count - len(hashes)} collisions"


@pytest.mark.determinism
class TestEnergyLossReproducibility:
    """Test reproducibility of energy loss calculations."""

    def test_energy_loss_identical_inputs_identical_outputs(self, tools):
        """Test that identical energy loss inputs produce identical outputs."""
        trap_data = {
            'trap_id': 'TRAP-ENERGY-REPRO',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'steam_cost_usd_per_1000lb': 8.50,
            'operating_hours_yr': 8760,
            'failure_severity': 1.0
        }

        results = [
            tools.calculate_energy_loss(copy.deepcopy(trap_data), FailureMode.FAILED_OPEN)
            for _ in range(10)
        ]

        first = results[0]
        for result in results[1:]:
            assert result.steam_loss_lb_hr == first.steam_loss_lb_hr
            assert result.steam_loss_kg_hr == first.steam_loss_kg_hr
            assert result.energy_loss_gj_yr == first.energy_loss_gj_yr
            assert result.cost_loss_usd_yr == first.cost_loss_usd_yr
            assert result.co2_emissions_kg_yr == first.co2_emissions_kg_yr
            assert result.provenance_hash == first.provenance_hash

    def test_energy_loss_napier_equation_reproducibility(self, tools):
        """Test Napier equation calculation reproducibility."""
        # Known reference value: W = 24.24 * P * D^2 * C
        # P=100, D=0.125, C=0.7: W = 24.24 * 100 * 0.015625 * 0.7 = 26.5125 lb/hr
        expected_steam_loss = 26.5125

        trap_data = {
            'trap_id': 'TRAP-NAPIER-REPRO',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        # Run 20 times and verify consistency
        results = [
            tools.calculate_energy_loss(copy.deepcopy(trap_data), FailureMode.FAILED_OPEN)
            for _ in range(20)
        ]

        for result in results:
            assert abs(result.steam_loss_lb_hr - expected_steam_loss) < 0.01, \
                f"Napier equation result {result.steam_loss_lb_hr} deviates from expected {expected_steam_loss}"


@pytest.mark.determinism
class TestDiagnosisReproducibility:
    """Test reproducibility of failure diagnosis."""

    def test_diagnosis_reproducibility_with_both_inputs(self, tools, reference_acoustic_data, reference_thermal_data):
        """Test diagnosis reproducibility with both acoustic and thermal inputs."""
        # Get consistent acoustic and thermal results first
        acoustic_result = tools.analyze_acoustic_signature(reference_acoustic_data)
        thermal_result = tools.analyze_thermal_pattern(reference_thermal_data)

        sensor_data = {
            'trap_id': 'TRAP-DIAG-REPRO',
            'pressure_upstream_psig': 100.0
        }

        diagnoses = [
            tools.diagnose_trap_failure(
                copy.deepcopy(sensor_data),
                acoustic_result,
                thermal_result
            )
            for _ in range(10)
        ]

        first = diagnoses[0]
        for diagnosis in diagnoses[1:]:
            assert diagnosis.failure_mode == first.failure_mode
            assert diagnosis.confidence == first.confidence
            assert diagnosis.failure_severity == first.failure_severity
            assert diagnosis.urgency_hours == first.urgency_hours


@pytest.mark.determinism
class TestRULReproducibility:
    """Test reproducibility of RUL predictions."""

    def test_rul_identical_inputs_identical_outputs(self, tools):
        """Test that identical RUL inputs produce identical outputs."""
        condition_data = {
            'trap_id': 'TRAP-RUL-REPRO',
            'current_age_days': 1000,
            'degradation_rate': 0.1,
            'current_health_score': 70,
            'historical_failures': [1800, 2000, 2200, 1900]
        }

        results = [
            tools.predict_remaining_useful_life(copy.deepcopy(condition_data))
            for _ in range(10)
        ]

        first = results[0]
        for result in results[1:]:
            assert result.rul_days == first.rul_days
            assert result.rul_confidence_lower == first.rul_confidence_lower
            assert result.rul_confidence_upper == first.rul_confidence_upper
            assert result.provenance_hash == first.provenance_hash

    def test_rul_weibull_reproducibility(self, tools):
        """Test Weibull distribution calculation reproducibility."""
        condition_data = {
            'trap_id': 'TRAP-WEIBULL-REPRO',
            'current_age_days': 500,
            'current_health_score': 80,
            'weibull_beta': 2.5,
            'weibull_eta': 2000
        }

        results = [
            tools.predict_remaining_useful_life(copy.deepcopy(condition_data))
            for _ in range(10)
        ]

        rul_values = [r.rul_days for r in results]
        assert len(set(rul_values)) == 1, \
            f"RUL values differ: {set(rul_values)}"


@pytest.mark.determinism
class TestFleetPrioritizationReproducibility:
    """Test reproducibility of fleet prioritization."""

    def test_prioritization_ordering_reproducibility(self, tools):
        """Test that fleet prioritization produces consistent ordering."""
        fleet = [
            {
                'trap_id': f'TRAP-PRI-{i:03d}',
                'failure_mode': [FailureMode.NORMAL, FailureMode.FAILED_OPEN,
                               FailureMode.LEAKING, FailureMode.FAILED_CLOSED][i % 4],
                'energy_loss_usd_yr': max(0, 15000 - i * 500),
                'process_criticality': 10 - (i % 5),
                'current_age_years': 1 + (i % 12),
                'health_score': max(20, 90 - i * 3)
            }
            for i in range(20)
        ]

        results = [
            tools.prioritize_maintenance(copy.deepcopy(fleet))
            for _ in range(5)
        ]

        # Compare ordering
        first_order = [t['trap_id'] for t in results[0].priority_list]
        for result in results[1:]:
            result_order = [t['trap_id'] for t in result.priority_list]
            assert result_order == first_order, \
                "Priority ordering differs across runs"

    def test_prioritization_scores_reproducibility(self, tools):
        """Test that priority scores are reproducible."""
        fleet = [
            {
                'trap_id': f'TRAP-SCORE-{i:03d}',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 10000 - i * 100,
                'process_criticality': 8,
                'current_age_years': 5,
                'health_score': 50
            }
            for i in range(10)
        ]

        results = [
            tools.prioritize_maintenance(copy.deepcopy(fleet))
            for _ in range(5)
        ]

        first_scores = {t['trap_id']: t['priority_score'] for t in results[0].priority_list}
        for result in results[1:]:
            result_scores = {t['trap_id']: t['priority_score'] for t in result.priority_list}
            assert result_scores == first_scores, \
                "Priority scores differ across runs"


@pytest.mark.determinism
class TestCrossRunReproducibility:
    """Test reproducibility across different execution contexts."""

    def test_execution_order_independence(self, tools, reference_acoustic_data, reference_thermal_data):
        """Test that execution order doesn't affect results."""
        # Run 1: acoustic then thermal
        acoustic_1 = tools.analyze_acoustic_signature(copy.deepcopy(reference_acoustic_data))
        thermal_1 = tools.analyze_thermal_pattern(copy.deepcopy(reference_thermal_data))

        # Run 2: thermal then acoustic
        thermal_2 = tools.analyze_thermal_pattern(copy.deepcopy(reference_thermal_data))
        acoustic_2 = tools.analyze_acoustic_signature(copy.deepcopy(reference_acoustic_data))

        # Results should be identical regardless of order
        assert acoustic_1.provenance_hash == acoustic_2.provenance_hash
        assert thermal_1.provenance_hash == thermal_2.provenance_hash

    def test_interleaved_execution_reproducibility(self, tools):
        """Test reproducibility with interleaved execution of different operations."""
        np.random.seed(42)

        # Create test data
        acoustic_data = {
            'trap_id': 'TRAP-INTERLEAVE',
            'signal': (np.random.randn(10000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        }
        thermal_data = {
            'trap_id': 'TRAP-INTERLEAVE',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0
        }
        energy_data = {
            'trap_id': 'TRAP-INTERLEAVE',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        # Run operations in different orders
        results_order1 = {
            'acoustic': tools.analyze_acoustic_signature(copy.deepcopy(acoustic_data)),
            'thermal': tools.analyze_thermal_pattern(copy.deepcopy(thermal_data)),
            'energy': tools.calculate_energy_loss(copy.deepcopy(energy_data), FailureMode.FAILED_OPEN)
        }

        results_order2 = {
            'energy': tools.calculate_energy_loss(copy.deepcopy(energy_data), FailureMode.FAILED_OPEN),
            'acoustic': tools.analyze_acoustic_signature(copy.deepcopy(acoustic_data)),
            'thermal': tools.analyze_thermal_pattern(copy.deepcopy(thermal_data))
        }

        # Compare results
        assert results_order1['acoustic'].provenance_hash == results_order2['acoustic'].provenance_hash
        assert results_order1['thermal'].provenance_hash == results_order2['thermal'].provenance_hash
        assert results_order1['energy'].provenance_hash == results_order2['energy'].provenance_hash


@pytest.mark.determinism
class TestDataSerializationReproducibility:
    """Test reproducibility of data serialization."""

    def test_json_serialization_determinism(self, tools):
        """Test that JSON serialization is deterministic."""
        data1 = {'trap_id': 'TEST', 'pressure': 100.0, 'temperature': 150.5}
        data2 = {'trap_id': 'TEST', 'pressure': 100.0, 'temperature': 150.5}

        json1 = json.dumps(data1, sort_keys=True)
        json2 = json.dumps(data2, sort_keys=True)

        assert json1 == json2

        hash1 = hashlib.sha256(json1.encode()).hexdigest()
        hash2 = hashlib.sha256(json2.encode()).hexdigest()

        assert hash1 == hash2

    def test_numpy_array_serialization_determinism(self, tools):
        """Test that numpy arrays serialize deterministically."""
        np.random.seed(42)
        array1 = np.random.randn(1000)
        np.random.seed(42)
        array2 = np.random.randn(1000)

        list1 = array1.tolist()
        list2 = array2.tolist()

        assert list1 == list2

        json1 = json.dumps(list1)
        json2 = json.dumps(list2)

        assert json1 == json2


@pytest.mark.determinism
class TestFloatingPointReproducibility:
    """Test floating point calculation reproducibility."""

    def test_floating_point_precision_preservation(self, tools):
        """Test that floating point precision is preserved."""
        trap_data = {
            'trap_id': 'TRAP-FLOAT-PREC',
            'orifice_diameter_in': 0.333333333333333,  # Repeating decimal
            'steam_pressure_psig': 100.123456789,
            'failure_severity': 0.999999999
        }

        results = [
            tools.calculate_energy_loss(copy.deepcopy(trap_data), FailureMode.FAILED_OPEN)
            for _ in range(10)
        ]

        # All results must be bit-perfect identical
        for result in results[1:]:
            assert result.steam_loss_lb_hr == results[0].steam_loss_lb_hr, \
                "Floating point precision not preserved"

    def test_transcendental_function_reproducibility(self, tools):
        """Test reproducibility of calculations involving transcendental functions."""
        # Signal with transcendental function components
        t = np.linspace(0, 1.0, 10000)
        signal = (np.sin(2 * np.pi * 30000 * t) * np.exp(-t) * np.log(t + 1)).tolist()

        acoustic_data = {
            'trap_id': 'TRAP-TRANSCENDENTAL',
            'signal': signal,
            'sampling_rate_hz': 250000
        }

        results = [
            tools.analyze_acoustic_signature(copy.deepcopy(acoustic_data))
            for _ in range(5)
        ]

        for result in results[1:]:
            assert result.signal_strength_db == results[0].signal_strength_db
            assert result.frequency_peak_hz == results[0].frequency_peak_hz


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "determinism"])
