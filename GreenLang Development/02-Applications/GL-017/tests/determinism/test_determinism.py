# -*- coding: utf-8 -*-
"""
Determinism tests for GL-017 CONDENSYNC.

Tests calculation reproducibility, same-input-same-output guarantees,
and provenance consistency across multiple executions.

Author: GL-017 Test Engineering Team
Target: 100% reproducibility for all calculations
"""

import pytest
import sys
import hashlib
import json
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Mock Classes for Determinism Testing
# ============================================================================

@dataclass
class CondenserConditions:
    """Condenser conditions for determinism testing."""
    vacuum_pressure_mbar: float = 50.0
    steam_saturation_temp_c: float = 33.0
    hotwell_temperature_c: float = 32.5
    cooling_water_inlet_temp_c: float = 25.0
    cooling_water_outlet_temp_c: float = 32.0
    cooling_water_flow_rate_m3_hr: float = 45000.0
    heat_duty_mw: float = 180.0
    surface_area_m2: float = 17500.0
    cleanliness_factor: float = 0.85


class DeterministicCalculator:
    """Calculator with deterministic outputs for testing."""

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    def calculate_lmtd(self, t_hot_in: float, t_hot_out: float,
                       t_cold_in: float, t_cold_out: float) -> Dict[str, Any]:
        """Calculate LMTD deterministically."""
        import math

        delta_t1 = t_hot_in - t_cold_out
        delta_t2 = t_hot_out - t_cold_in

        if delta_t1 <= 0 or delta_t2 <= 0:
            lmtd = Decimal('0.0')
        elif abs(delta_t1 - delta_t2) < 0.001:
            lmtd = Decimal(str(delta_t1))
        else:
            lmtd = Decimal(str((delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)))

        result = {
            'lmtd_c': float(lmtd),
            'delta_t1_c': delta_t1,
            'delta_t2_c': delta_t2,
            'version': self.version
        }

        # Generate deterministic provenance hash
        result['provenance_hash'] = self._generate_provenance_hash(result)

        return result

    def calculate_overall_htc(self, heat_duty_mw: float, lmtd_c: float,
                              surface_area_m2: float) -> Dict[str, Any]:
        """Calculate overall HTC deterministically."""
        if lmtd_c > 0 and surface_area_m2 > 0:
            heat_duty_w = Decimal(str(heat_duty_mw)) * Decimal('1000000')
            overall_htc = heat_duty_w / (Decimal(str(surface_area_m2)) * Decimal(str(lmtd_c)))
        else:
            overall_htc = Decimal('0.0')

        result = {
            'overall_htc_w_m2k': float(overall_htc),
            'heat_duty_mw': heat_duty_mw,
            'lmtd_c': lmtd_c,
            'surface_area_m2': surface_area_m2,
            'version': self.version
        }

        result['provenance_hash'] = self._generate_provenance_hash(result)

        return result

    def calculate_vacuum_efficiency(self, current_vacuum_mbar: float,
                                    design_vacuum_mbar: float) -> Dict[str, Any]:
        """Calculate vacuum efficiency deterministically."""
        if design_vacuum_mbar > 0:
            efficiency = Decimal(str(design_vacuum_mbar)) / Decimal(str(current_vacuum_mbar)) * Decimal('100')
            efficiency = min(efficiency, Decimal('100.0'))
        else:
            efficiency = Decimal('0.0')

        deviation = Decimal(str(current_vacuum_mbar)) - Decimal(str(design_vacuum_mbar))
        heat_rate_penalty = deviation * Decimal('12.5') if deviation > 0 else Decimal('0.0')

        result = {
            'vacuum_efficiency_percent': float(efficiency),
            'deviation_mbar': float(deviation),
            'heat_rate_penalty_kj_kwh': float(heat_rate_penalty),
            'version': self.version
        }

        result['provenance_hash'] = self._generate_provenance_hash(result)

        return result

    def calculate_fouling_rate(self, cleanliness_factor: float,
                               tds_ppm: float, velocity_m_s: float) -> Dict[str, Any]:
        """Calculate fouling rate deterministically."""
        base_rate = Decimal('0.01')
        tds_factor = Decimal('1.0') + (Decimal(str(tds_ppm)) - Decimal('1000')) / Decimal('5000')
        velocity_factor = Decimal('2.0') / Decimal(str(velocity_m_s)) if velocity_m_s > 0 else Decimal('2.0')

        fouling_rate = base_rate * tds_factor * velocity_factor

        result = {
            'fouling_rate_per_1000hr': float(fouling_rate),
            'base_rate': float(base_rate),
            'tds_factor': float(tds_factor),
            'velocity_factor': float(velocity_factor),
            'version': self.version
        }

        result['provenance_hash'] = self._generate_provenance_hash(result)

        return result

    def _generate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Generate deterministic provenance hash."""
        # Exclude hash from hashing
        hashable_data = {k: v for k, v in data.items() if k != 'provenance_hash'}

        # Sort keys for deterministic ordering
        sorted_data = json.dumps(hashable_data, sort_keys=True, default=str)

        # Generate SHA-256 hash
        return hashlib.sha256(sorted_data.encode()).hexdigest()


class ProvenanceTracker:
    """Provenance tracker with deterministic hash generation."""

    def __init__(self, calculation_id: str, calculation_type: str, version: str):
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.version = version
        self.steps = []
        self.inputs = {}

    def record_inputs(self, inputs: Dict[str, Any]):
        """Record calculation inputs."""
        self.inputs = inputs

    def record_step(self, operation: str, inputs: Dict[str, Any],
                    output: Any, formula: str):
        """Record calculation step."""
        self.steps.append({
            'operation': operation,
            'inputs': inputs,
            'output': output,
            'formula': formula
        })

    def get_provenance_hash(self) -> str:
        """Generate deterministic provenance hash."""
        provenance_data = {
            'calculation_id': self.calculation_id,
            'calculation_type': self.calculation_type,
            'version': self.version,
            'inputs': self.inputs,
            'steps': self.steps
        }

        # Sort and serialize deterministically
        sorted_data = json.dumps(provenance_data, sort_keys=True, default=str)

        return hashlib.sha256(sorted_data.encode()).hexdigest()


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def calculator():
    """Create deterministic calculator."""
    return DeterministicCalculator(version="1.0.0-test")


@pytest.fixture
def standard_conditions():
    """Standard condenser conditions."""
    return CondenserConditions()


@pytest.fixture
def lmtd_inputs():
    """Standard LMTD calculation inputs."""
    return {
        't_hot_in': 33.0,
        't_hot_out': 32.5,
        't_cold_in': 25.0,
        't_cold_out': 32.0
    }


@pytest.fixture
def htc_inputs():
    """Standard HTC calculation inputs."""
    return {
        'heat_duty_mw': 180.0,
        'lmtd_c': 10.5,
        'surface_area_m2': 17500.0
    }


@pytest.fixture
def vacuum_inputs():
    """Standard vacuum calculation inputs."""
    return {
        'current_vacuum_mbar': 50.0,
        'design_vacuum_mbar': 45.0
    }


@pytest.fixture
def fouling_inputs():
    """Standard fouling calculation inputs."""
    return {
        'cleanliness_factor': 0.85,
        'tds_ppm': 1500.0,
        'velocity_m_s': 2.0
    }


# ============================================================================
# Same Input Same Output Tests
# ============================================================================

class TestSameInputSameOutput:
    """Tests for same-input-same-output guarantee."""

    @pytest.mark.determinism
    def test_lmtd_same_input_same_output(self, calculator, lmtd_inputs):
        """Test LMTD produces same output for same input."""
        results = [calculator.calculate_lmtd(**lmtd_inputs) for _ in range(10)]

        # All LMTD values should be identical
        lmtd_values = [r['lmtd_c'] for r in results]
        assert all(v == lmtd_values[0] for v in lmtd_values)

    @pytest.mark.determinism
    def test_htc_same_input_same_output(self, calculator, htc_inputs):
        """Test HTC produces same output for same input."""
        results = [calculator.calculate_overall_htc(**htc_inputs) for _ in range(10)]

        htc_values = [r['overall_htc_w_m2k'] for r in results]
        assert all(v == htc_values[0] for v in htc_values)

    @pytest.mark.determinism
    def test_vacuum_efficiency_same_input_same_output(self, calculator, vacuum_inputs):
        """Test vacuum efficiency produces same output for same input."""
        results = [calculator.calculate_vacuum_efficiency(**vacuum_inputs) for _ in range(10)]

        efficiency_values = [r['vacuum_efficiency_percent'] for r in results]
        assert all(v == efficiency_values[0] for v in efficiency_values)

    @pytest.mark.determinism
    def test_fouling_rate_same_input_same_output(self, calculator, fouling_inputs):
        """Test fouling rate produces same output for same input."""
        results = [calculator.calculate_fouling_rate(**fouling_inputs) for _ in range(10)]

        rate_values = [r['fouling_rate_per_1000hr'] for r in results]
        assert all(v == rate_values[0] for v in rate_values)

    @pytest.mark.determinism
    def test_multiple_calculations_same_sequence(self, calculator, lmtd_inputs, htc_inputs):
        """Test sequence of calculations produces same results."""
        def run_calculation_sequence():
            lmtd_result = calculator.calculate_lmtd(**lmtd_inputs)
            htc_result = calculator.calculate_overall_htc(**htc_inputs)
            return lmtd_result['lmtd_c'], htc_result['overall_htc_w_m2k']

        results = [run_calculation_sequence() for _ in range(10)]

        assert all(r == results[0] for r in results)

    @pytest.mark.determinism
    def test_decimal_precision_maintained(self, calculator):
        """Test decimal precision is maintained across calculations."""
        inputs = {
            't_hot_in': 33.123456789,
            't_hot_out': 32.987654321,
            't_cold_in': 25.111111111,
            't_cold_out': 31.999999999
        }

        results = [calculator.calculate_lmtd(**inputs) for _ in range(10)]

        # Check all values match exactly
        for i in range(1, len(results)):
            assert results[i]['lmtd_c'] == results[0]['lmtd_c']
            assert results[i]['delta_t1_c'] == results[0]['delta_t1_c']
            assert results[i]['delta_t2_c'] == results[0]['delta_t2_c']


# ============================================================================
# Provenance Consistency Tests
# ============================================================================

class TestProvenanceConsistency:
    """Tests for provenance hash consistency."""

    @pytest.mark.determinism
    def test_lmtd_provenance_hash_consistent(self, calculator, lmtd_inputs):
        """Test LMTD provenance hash is consistent."""
        results = [calculator.calculate_lmtd(**lmtd_inputs) for _ in range(10)]

        hashes = [r['provenance_hash'] for r in results]
        assert all(h == hashes[0] for h in hashes)

    @pytest.mark.determinism
    def test_htc_provenance_hash_consistent(self, calculator, htc_inputs):
        """Test HTC provenance hash is consistent."""
        results = [calculator.calculate_overall_htc(**htc_inputs) for _ in range(10)]

        hashes = [r['provenance_hash'] for r in results]
        assert all(h == hashes[0] for h in hashes)

    @pytest.mark.determinism
    def test_provenance_hash_length(self, calculator, lmtd_inputs):
        """Test provenance hash is correct length (SHA-256)."""
        result = calculator.calculate_lmtd(**lmtd_inputs)

        assert len(result['provenance_hash']) == 64

    @pytest.mark.determinism
    def test_different_inputs_different_hashes(self, calculator):
        """Test different inputs produce different hashes."""
        inputs1 = {'t_hot_in': 33.0, 't_hot_out': 32.5, 't_cold_in': 25.0, 't_cold_out': 32.0}
        inputs2 = {'t_hot_in': 35.0, 't_hot_out': 34.0, 't_cold_in': 26.0, 't_cold_out': 33.0}

        result1 = calculator.calculate_lmtd(**inputs1)
        result2 = calculator.calculate_lmtd(**inputs2)

        assert result1['provenance_hash'] != result2['provenance_hash']

    @pytest.mark.determinism
    def test_provenance_tracker_hash_consistent(self):
        """Test provenance tracker produces consistent hashes."""
        hashes = []

        for _ in range(10):
            tracker = ProvenanceTracker(
                calculation_id='TEST-001',
                calculation_type='test_calculation',
                version='1.0.0'
            )
            tracker.record_inputs({'x': 100, 'y': 200})
            tracker.record_step(
                operation='add',
                inputs={'x': 100, 'y': 200},
                output=300,
                formula='x + y'
            )
            hashes.append(tracker.get_provenance_hash())

        assert all(h == hashes[0] for h in hashes)

    @pytest.mark.determinism
    def test_provenance_tracker_step_order_matters(self):
        """Test provenance hash changes with step order."""
        # First order
        tracker1 = ProvenanceTracker('TEST-001', 'test', '1.0.0')
        tracker1.record_step('step1', {'a': 1}, 10, 'a * 10')
        tracker1.record_step('step2', {'b': 2}, 20, 'b * 10')
        hash1 = tracker1.get_provenance_hash()

        # Reverse order
        tracker2 = ProvenanceTracker('TEST-001', 'test', '1.0.0')
        tracker2.record_step('step2', {'b': 2}, 20, 'b * 10')
        tracker2.record_step('step1', {'a': 1}, 10, 'a * 10')
        hash2 = tracker2.get_provenance_hash()

        assert hash1 != hash2


# ============================================================================
# Calculation Reproducibility Tests
# ============================================================================

class TestCalculationReproducibility:
    """Tests for calculation reproducibility across sessions."""

    @pytest.mark.determinism
    def test_lmtd_reproducible_across_instances(self, lmtd_inputs):
        """Test LMTD is reproducible across calculator instances."""
        results = []
        for _ in range(5):
            calc = DeterministicCalculator(version="1.0.0")
            results.append(calc.calculate_lmtd(**lmtd_inputs))

        lmtd_values = [r['lmtd_c'] for r in results]
        assert all(v == lmtd_values[0] for v in lmtd_values)

    @pytest.mark.determinism
    def test_htc_reproducible_across_instances(self, htc_inputs):
        """Test HTC is reproducible across calculator instances."""
        results = []
        for _ in range(5):
            calc = DeterministicCalculator(version="1.0.0")
            results.append(calc.calculate_overall_htc(**htc_inputs))

        htc_values = [r['overall_htc_w_m2k'] for r in results]
        assert all(v == htc_values[0] for v in htc_values)

    @pytest.mark.determinism
    def test_calculation_order_independence(self, calculator, lmtd_inputs, htc_inputs, vacuum_inputs):
        """Test calculations are independent of execution order."""
        # Order 1: LMTD -> HTC -> Vacuum
        lmtd1 = calculator.calculate_lmtd(**lmtd_inputs)
        htc1 = calculator.calculate_overall_htc(**htc_inputs)
        vacuum1 = calculator.calculate_vacuum_efficiency(**vacuum_inputs)

        # Order 2: Vacuum -> HTC -> LMTD
        vacuum2 = calculator.calculate_vacuum_efficiency(**vacuum_inputs)
        htc2 = calculator.calculate_overall_htc(**htc_inputs)
        lmtd2 = calculator.calculate_lmtd(**lmtd_inputs)

        assert lmtd1['lmtd_c'] == lmtd2['lmtd_c']
        assert htc1['overall_htc_w_m2k'] == htc2['overall_htc_w_m2k']
        assert vacuum1['vacuum_efficiency_percent'] == vacuum2['vacuum_efficiency_percent']


# ============================================================================
# Floating Point Handling Tests
# ============================================================================

class TestFloatingPointHandling:
    """Tests for floating point handling in deterministic calculations."""

    @pytest.mark.determinism
    def test_small_values_deterministic(self, calculator):
        """Test small values are handled deterministically."""
        inputs = {
            't_hot_in': 0.001,
            't_hot_out': 0.0005,
            't_cold_in': 0.0001,
            't_cold_out': 0.0008
        }

        results = [calculator.calculate_lmtd(**inputs) for _ in range(10)]

        lmtd_values = [r['lmtd_c'] for r in results]
        assert all(v == lmtd_values[0] for v in lmtd_values)

    @pytest.mark.determinism
    def test_large_values_deterministic(self, calculator):
        """Test large values are handled deterministically."""
        inputs = {
            'heat_duty_mw': 1000000.0,
            'lmtd_c': 100.0,
            'surface_area_m2': 100000.0
        }

        results = [calculator.calculate_overall_htc(**inputs) for _ in range(10)]

        htc_values = [r['overall_htc_w_m2k'] for r in results]
        assert all(v == htc_values[0] for v in htc_values)

    @pytest.mark.determinism
    def test_edge_case_values_deterministic(self, calculator):
        """Test edge case values are handled deterministically."""
        # Nearly equal delta T values (special LMTD case)
        inputs = {
            't_hot_in': 33.0,
            't_hot_out': 32.5,
            't_cold_in': 25.5,
            't_cold_out': 32.0
        }

        results = [calculator.calculate_lmtd(**inputs) for _ in range(10)]

        lmtd_values = [r['lmtd_c'] for r in results]
        assert all(v == lmtd_values[0] for v in lmtd_values)


# ============================================================================
# Version Sensitivity Tests
# ============================================================================

class TestVersionSensitivity:
    """Tests for version-sensitive determinism."""

    @pytest.mark.determinism
    def test_same_version_same_results(self, lmtd_inputs):
        """Test same version produces same results."""
        calc1 = DeterministicCalculator(version="1.0.0")
        calc2 = DeterministicCalculator(version="1.0.0")

        result1 = calc1.calculate_lmtd(**lmtd_inputs)
        result2 = calc2.calculate_lmtd(**lmtd_inputs)

        assert result1['lmtd_c'] == result2['lmtd_c']
        assert result1['provenance_hash'] == result2['provenance_hash']

    @pytest.mark.determinism
    def test_different_version_different_hash(self, lmtd_inputs):
        """Test different versions produce different hashes."""
        calc1 = DeterministicCalculator(version="1.0.0")
        calc2 = DeterministicCalculator(version="1.0.1")

        result1 = calc1.calculate_lmtd(**lmtd_inputs)
        result2 = calc2.calculate_lmtd(**lmtd_inputs)

        # Values should be same (algorithm unchanged)
        assert result1['lmtd_c'] == result2['lmtd_c']

        # But hashes should differ (version is part of provenance)
        assert result1['provenance_hash'] != result2['provenance_hash']


# ============================================================================
# Compliance Tests
# ============================================================================

class TestComplianceDeterminism:
    """Tests for compliance-related determinism requirements."""

    @pytest.mark.determinism
    @pytest.mark.compliance
    def test_audit_trail_reproducibility(self, calculator, lmtd_inputs):
        """Test audit trail can be reproduced."""
        result = calculator.calculate_lmtd(**lmtd_inputs)

        # Store audit record
        audit_record = {
            'inputs': lmtd_inputs,
            'outputs': result,
            'provenance_hash': result['provenance_hash']
        }

        # Reproduce calculation
        reproduced = calculator.calculate_lmtd(**lmtd_inputs)

        # Verify reproducibility
        assert reproduced['lmtd_c'] == audit_record['outputs']['lmtd_c']
        assert reproduced['provenance_hash'] == audit_record['provenance_hash']

    @pytest.mark.determinism
    @pytest.mark.compliance
    def test_hash_verification(self, calculator, htc_inputs):
        """Test provenance hash can be verified."""
        result = calculator.calculate_overall_htc(**htc_inputs)
        stored_hash = result['provenance_hash']

        # Recalculate hash manually
        hashable_data = {k: v for k, v in result.items() if k != 'provenance_hash'}
        sorted_data = json.dumps(hashable_data, sort_keys=True, default=str)
        recalculated_hash = hashlib.sha256(sorted_data.encode()).hexdigest()

        assert stored_hash == recalculated_hash

    @pytest.mark.determinism
    @pytest.mark.compliance
    def test_bit_perfect_reproducibility(self, calculator, vacuum_inputs):
        """Test calculations are bit-perfect reproducible."""
        results = [calculator.calculate_vacuum_efficiency(**vacuum_inputs) for _ in range(100)]

        # All values must be identical (bit-perfect)
        reference = results[0]
        for result in results[1:]:
            assert result['vacuum_efficiency_percent'] == reference['vacuum_efficiency_percent']
            assert result['deviation_mbar'] == reference['deviation_mbar']
            assert result['heat_rate_penalty_kj_kwh'] == reference['heat_rate_penalty_kj_kwh']
            assert result['provenance_hash'] == reference['provenance_hash']
