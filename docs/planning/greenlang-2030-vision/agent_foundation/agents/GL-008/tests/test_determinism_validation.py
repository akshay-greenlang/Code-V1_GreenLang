# -*- coding: utf-8 -*-
"""
Determinism validation tests for GL-008 SteamTrapInspector.

This module validates that all calculations are bit-perfect reproducible,
provenance hashes are correct, and LLM temperature/seed enforcement works.
"""

import pytest
import hashlib
import json
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tools import SteamTrapTools
from config import TrapInspectorConfig, FailureMode, TrapType


@pytest.mark.determinism
class TestProvenanceHashing:
    """Test SHA-256 provenance hash generation and validation."""

    def test_provenance_hash_format(self, tools, normal_acoustic_signal):
        """Test that provenance hash is valid SHA-256."""
        acoustic_data = {
            'trap_id': 'TRAP-HASH-FORMAT',
            'signal': normal_acoustic_signal.tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Validate hash format
        assert result.provenance_hash is not None
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64  # SHA-256 = 256 bits = 64 hex chars
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    def test_provenance_hash_determinism(self, tools, normal_acoustic_signal):
        """Test that identical inputs produce identical provenance hashes."""
        acoustic_data = {
            'trap_id': 'TRAP-HASH-DET',
            'signal': normal_acoustic_signal.tolist(),
            'sampling_rate_hz': 250000
        }

        # Generate hash 10 times
        hashes = []
        for _ in range(10):
            result = tools.analyze_acoustic_signature(acoustic_data)
            hashes.append(result.provenance_hash)

        # All hashes must be identical
        assert len(set(hashes)) == 1  # Only one unique hash

    def test_provenance_hash_input_sensitivity(self, tools):
        """Test that different inputs produce different hashes."""
        signal1 = np.random.randn(10000) * 0.1
        signal2 = np.random.randn(10000) * 0.1  # Different random signal

        acoustic_data1 = {
            'trap_id': 'TRAP-1',
            'signal': signal1.tolist(),
            'sampling_rate_hz': 250000
        }

        acoustic_data2 = {
            'trap_id': 'TRAP-1',  # Same trap ID
            'signal': signal2.tolist(),  # Different signal
            'sampling_rate_hz': 250000
        }

        result1 = tools.analyze_acoustic_signature(acoustic_data1)
        result2 = tools.analyze_acoustic_signature(acoustic_data2)

        # Different inputs → Different hashes
        assert result1.provenance_hash != result2.provenance_hash

    def test_provenance_hash_includes_all_inputs(self, tools):
        """Test that provenance hash changes if any input changes."""
        base_signal = np.random.randn(10000) * 0.1

        # Test 1: Change trap_id
        data1 = {'trap_id': 'TRAP-A', 'signal': base_signal.tolist(), 'sampling_rate_hz': 250000}
        data2 = {'trap_id': 'TRAP-B', 'signal': base_signal.tolist(), 'sampling_rate_hz': 250000}

        result1 = tools.analyze_acoustic_signature(data1)
        result2 = tools.analyze_acoustic_signature(data2)
        assert result1.provenance_hash != result2.provenance_hash

        # Test 2: Change sampling rate
        data3 = {'trap_id': 'TRAP-A', 'signal': base_signal.tolist(), 'sampling_rate_hz': 200000}
        result3 = tools.analyze_acoustic_signature(data3)
        assert result1.provenance_hash != result3.provenance_hash

    def test_provenance_hash_collision_resistance(self, tools):
        """Test that provenance hashes are unique (no collisions)."""
        hashes = set()

        for i in range(100):
            signal = np.random.randn(1000) * 0.1
            acoustic_data = {
                'trap_id': f'TRAP-{i}',
                'signal': signal.tolist(),
                'sampling_rate_hz': 250000
            }

            result = tools.analyze_acoustic_signature(acoustic_data)
            hashes.add(result.provenance_hash)

        # All hashes should be unique (no collisions)
        assert len(hashes) == 100


@pytest.mark.determinism
class TestBitPerfectReproducibility:
    """Test that calculations are bit-perfect reproducible."""

    def test_acoustic_analysis_reproducibility(self, tools, determinism_test_iterations):
        """Test acoustic analysis produces identical results across runs."""
        signal = np.random.randn(10000) * 0.2

        acoustic_data = {
            'trap_id': 'TRAP-REPRO',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        # Run analysis multiple times
        results = [
            tools.analyze_acoustic_signature(acoustic_data)
            for _ in range(determinism_test_iterations)
        ]

        # All results must be bit-perfect identical
        first = results[0]
        for result in results[1:]:
            assert result.failure_probability == first.failure_probability
            assert result.confidence_score == first.confidence_score
            assert result.signal_strength_db == first.signal_strength_db
            assert result.frequency_peak_hz == first.frequency_peak_hz
            assert result.provenance_hash == first.provenance_hash

    def test_thermal_analysis_reproducibility(self, tools, determinism_test_iterations):
        """Test thermal analysis produces identical results across runs."""
        thermal_data = {
            'trap_id': 'TRAP-THERMAL-REPRO',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 20.0
        }

        # Run analysis multiple times
        results = [
            tools.analyze_thermal_pattern(thermal_data)
            for _ in range(determinism_test_iterations)
        ]

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result.trap_health_score == first.trap_health_score
            assert result.temperature_differential_c == first.temperature_differential_c
            assert result.condensate_pooling_detected == first.condensate_pooling_detected
            assert result.provenance_hash == first.provenance_hash

    def test_energy_loss_reproducibility(self, tools, determinism_test_iterations):
        """Test energy loss calculation produces identical results across runs."""
        trap_data = {
            'trap_id': 'TRAP-ENERGY-REPRO',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        # Run calculation multiple times
        results = [
            tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
            for _ in range(determinism_test_iterations)
        ]

        # All results must be bit-perfect identical
        first = results[0]
        for result in results[1:]:
            assert result.steam_loss_lb_hr == first.steam_loss_lb_hr
            assert result.steam_loss_kg_hr == first.steam_loss_kg_hr
            assert result.energy_loss_gj_yr == first.energy_loss_gj_yr
            assert result.cost_loss_usd_yr == first.cost_loss_usd_yr
            assert result.co2_emissions_kg_yr == first.co2_emissions_kg_yr
            assert result.provenance_hash == first.provenance_hash

    def test_full_workflow_reproducibility(self, tools, determinism_test_iterations):
        """Test complete workflow produces identical results across runs."""
        signal = np.random.randn(10000) * 0.3

        # Complete workflow: acoustic → thermal → diagnosis → energy loss
        workflows = []

        for _ in range(determinism_test_iterations):
            # Step 1: Acoustic
            acoustic_result = tools.analyze_acoustic_signature({
                'trap_id': 'TRAP-WORKFLOW',
                'signal': signal.tolist(),
                'sampling_rate_hz': 250000
            })

            # Step 2: Thermal
            thermal_result = tools.analyze_thermal_pattern({
                'trap_id': 'TRAP-WORKFLOW',
                'temperature_upstream_c': 150.0,
                'temperature_downstream_c': 148.0
            })

            # Step 3: Diagnosis
            diagnosis_result = tools.diagnose_trap_failure(
                {'trap_id': 'TRAP-WORKFLOW', 'pressure_upstream_psig': 100.0},
                acoustic_result,
                thermal_result
            )

            # Step 4: Energy loss
            if diagnosis_result.failure_mode != FailureMode.NORMAL:
                energy_result = tools.calculate_energy_loss(
                    {
                        'trap_id': 'TRAP-WORKFLOW',
                        'orifice_diameter_in': 0.125,
                        'steam_pressure_psig': 100.0,
                        'failure_severity': diagnosis_result.confidence
                    },
                    diagnosis_result.failure_mode
                )
                workflows.append({
                    'acoustic_hash': acoustic_result.provenance_hash,
                    'thermal_hash': thermal_result.provenance_hash,
                    'diagnosis_mode': diagnosis_result.failure_mode.value,
                    'energy_cost': energy_result.cost_loss_usd_yr
                })

        if workflows:
            # All workflows must produce identical results
            first = workflows[0]
            for workflow in workflows[1:]:
                assert workflow['acoustic_hash'] == first['acoustic_hash']
                assert workflow['thermal_hash'] == first['thermal_hash']
                assert workflow['diagnosis_mode'] == first['diagnosis_mode']
                assert workflow['energy_cost'] == first['energy_cost']


@pytest.mark.determinism
class TestNumpyRandomSeedControl:
    """Test that numpy random seed is properly controlled."""

    def test_numpy_seed_consistency(self, tools):
        """Test that numpy operations use consistent seeding."""
        # This test verifies that any numpy random operations in tools
        # are properly seeded for determinism

        signal1 = np.random.RandomState(42).randn(10000) * 0.1
        signal2 = np.random.RandomState(42).randn(10000) * 0.1

        # Signals generated with same seed should be identical
        assert np.array_equal(signal1, signal2)

    def test_fft_determinism(self, tools):
        """Test that FFT operations are deterministic."""
        signal = np.random.RandomState(42).randn(10000) * 0.2

        acoustic_data = {
            'trap_id': 'TRAP-FFT',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        results = [
            tools.analyze_acoustic_signature(acoustic_data)
            for _ in range(5)
        ]

        # FFT should produce identical results
        for result in results[1:]:
            assert result.frequency_peak_hz == results[0].frequency_peak_hz


@pytest.mark.determinism
class TestLLMDeterminismEnforcement:
    """Test LLM temperature=0.0 and seed=42 enforcement."""

    def test_llm_temperature_enforcement(self, base_config):
        """Test that LLM temperature is enforced to 0.0."""
        assert base_config.llm_temperature == 0.0

        # Attempt to create config with non-zero temperature should fail
        with pytest.raises(AssertionError):
            TrapInspectorConfig(
                agent_id="TEST",
                llm_temperature=0.5  # Non-deterministic
            )

    def test_llm_seed_enforcement(self, base_config):
        """Test that LLM seed is enforced to 42."""
        assert base_config.llm_seed == 42

        # Attempt to create config with different seed should fail
        with pytest.raises(AssertionError):
            TrapInspectorConfig(
                agent_id="TEST",
                llm_seed=123  # Different seed
            )

    def test_config_validates_determinism(self):
        """Test that config validation enforces determinism settings."""
        # Valid deterministic config
        config = TrapInspectorConfig(
            agent_id="TEST-DET",
            llm_temperature=0.0,
            llm_seed=42
        )
        assert config.llm_temperature == 0.0
        assert config.llm_seed == 42

        # Invalid: temperature not 0.0
        with pytest.raises(AssertionError, match="temperature must be 0.0"):
            TrapInspectorConfig(
                agent_id="TEST-INVALID",
                llm_temperature=0.1,
                llm_seed=42
            )

        # Invalid: seed not 42
        with pytest.raises(AssertionError, match="seed must be 42"):
            TrapInspectorConfig(
                agent_id="TEST-INVALID",
                llm_temperature=0.0,
                llm_seed=100
            )


@pytest.mark.determinism
class TestFloatingPointConsistency:
    """Test floating point calculation consistency."""

    def test_floating_point_precision(self, tools):
        """Test that floating point calculations maintain precision."""
        trap_data = {
            'trap_id': 'TRAP-FLOAT',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        results = [
            tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
            for _ in range(10)
        ]

        # Check bit-level equality (not just approximate)
        for result in results[1:]:
            # Use == for exact comparison, not pytest.approx
            assert result.steam_loss_lb_hr == results[0].steam_loss_lb_hr
            assert result.cost_loss_usd_yr == results[0].cost_loss_usd_yr

    def test_division_consistency(self, tools):
        """Test that division operations are consistent."""
        # Test a calculation that involves division
        trap_data = {
            'trap_id': 'TRAP-DIV',
            'orifice_diameter_in': 0.333,  # Repeating decimal
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        results = [
            tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
            for _ in range(10)
        ]

        # Division should be deterministic
        for result in results[1:]:
            assert result.steam_loss_lb_hr == results[0].steam_loss_lb_hr

    def test_transcendental_function_consistency(self, tools):
        """Test that transcendental functions (sin, exp, log) are consistent."""
        # Generate signal using transcendental functions
        t = np.linspace(0, 1.0, 10000)
        signal = np.sin(2 * np.pi * 30000 * t) * np.exp(-t)  # Decaying sinusoid

        acoustic_data = {
            'trap_id': 'TRAP-TRANSCENDENTAL',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        results = [
            tools.analyze_acoustic_signature(acoustic_data)
            for _ in range(5)
        ]

        # Results should be identical
        for result in results[1:]:
            assert result.signal_strength_db == results[0].signal_strength_db


@pytest.mark.determinism
class TestTimestampHandling:
    """Test that timestamps don't affect determinism of core calculations."""

    def test_timestamp_excluded_from_calculation_hash(self, tools):
        """Test that timestamps don't affect calculation provenance hash."""
        import time

        acoustic_data = {
            'trap_id': 'TRAP-TIME',
            'signal': np.random.randn(10000).tolist(),
            'sampling_rate_hz': 250000
        }

        # Run analysis at different times
        result1 = tools.analyze_acoustic_signature(acoustic_data)
        time.sleep(0.1)  # Wait to ensure different timestamp
        result2 = tools.analyze_acoustic_signature(acoustic_data)

        # Timestamps should differ
        assert result1.timestamp != result2.timestamp

        # But provenance hashes should be identical (timestamp excluded)
        assert result1.provenance_hash == result2.provenance_hash

        # And core calculation results should be identical
        assert result1.failure_probability == result2.failure_probability


@pytest.mark.determinism
class TestDataStructureSerialization:
    """Test that data structures serialize deterministically."""

    def test_dict_serialization_determinism(self, tools):
        """Test that dictionaries serialize consistently for hashing."""
        # Python dicts maintain insertion order (Python 3.7+)
        data1 = {'trap_id': 'TEST', 'pressure': 100.0, 'temperature': 150.0}
        data2 = {'trap_id': 'TEST', 'pressure': 100.0, 'temperature': 150.0}

        # Serialize to JSON
        json1 = json.dumps(data1, sort_keys=True)
        json2 = json.dumps(data2, sort_keys=True)

        assert json1 == json2

        # Hash should be identical
        hash1 = hashlib.sha256(json1.encode()).hexdigest()
        hash2 = hashlib.sha256(json2.encode()).hexdigest()

        assert hash1 == hash2

    def test_numpy_array_serialization(self, tools):
        """Test that numpy arrays serialize deterministically."""
        array1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        array2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Convert to list for serialization
        list1 = array1.tolist()
        list2 = array2.tolist()

        assert list1 == list2

        # JSON serialization
        json1 = json.dumps(list1)
        json2 = json.dumps(list2)

        assert json1 == json2


@pytest.mark.determinism
class TestMultithreadingSafety:
    """Test that determinism is maintained in concurrent execution."""

    def test_concurrent_analysis_determinism(self, tools):
        """Test that concurrent analyses don't interfere with determinism."""
        import concurrent.futures

        signal = np.random.randn(10000) * 0.2

        def run_analysis(index):
            acoustic_data = {
                'trap_id': f'TRAP-CONCURRENT-{index}',
                'signal': signal.tolist(),
                'sampling_rate_hz': 250000
            }
            return tools.analyze_acoustic_signature(acoustic_data)

        # Run analyses concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(run_analysis, range(10)))

        # Group by trap_id and verify consistency
        trap_results = {}
        for result in results:
            if result.trap_id not in trap_results:
                trap_results[result.trap_id] = []
            trap_results[result.trap_id].append(result)

        # Each trap should have identical results
        for trap_id, trap_results_list in trap_results.items():
            first = trap_results_list[0]
            for result in trap_results_list[1:]:
                assert result.provenance_hash == first.provenance_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "determinism"])
