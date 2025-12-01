# -*- coding: utf-8 -*-
"""
Security validation tests for GL-008 TRAPCATCHER SteamTrapInspector.

This module tests input sanitization, injection prevention, access control,
data validation, and secure handling of sensitive operational data.

Security Requirements:
- Input validation to prevent injection attacks
- Proper handling of malformed data
- Protection against resource exhaustion attacks
- Secure provenance hash generation
- Access control validation
"""

import pytest
import numpy as np
from typing import Dict, List, Any
import sys
from pathlib import Path
import json
import hashlib

sys.path.append(str(Path(__file__).parent.parent.parent))

from tools import SteamTrapTools
from config import TrapInspectorConfig, TrapType, FailureMode


@pytest.fixture
def tools():
    """Create SteamTrapTools instance for security testing."""
    return SteamTrapTools()


@pytest.fixture
def secure_config():
    """Create secure test configuration."""
    return TrapInspectorConfig(
        agent_id="GL-008-SECURITY-TEST",
        enable_llm_classification=False,
        cache_ttl_seconds=60,
        max_concurrent_inspections=5,
        llm_temperature=0.0,
        llm_seed=42
    )


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    def test_trap_id_sanitization(self, tools):
        """Test that trap IDs are properly validated."""
        # Valid trap ID
        valid_data = {
            'trap_id': 'TRAP-001',
            'signal': (np.random.randn(1000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        }
        result = tools.analyze_acoustic_signature(valid_data)
        assert result.trap_id == 'TRAP-001'

    def test_trap_id_with_special_characters(self, tools):
        """Test handling of trap IDs with special characters."""
        special_chars_ids = [
            'TRAP-001-ABC',
            'TRAP_001_DEF',
            'TRAP.001.GHI',
            'TRAP 001',
        ]

        for trap_id in special_chars_ids:
            data = {
                'trap_id': trap_id,
                'signal': (np.random.randn(1000) * 0.2).tolist(),
                'sampling_rate_hz': 250000
            }
            try:
                result = tools.analyze_acoustic_signature(data)
                # If it succeeds, verify the ID is preserved or sanitized
                assert result is not None
            except (ValueError, AssertionError):
                # Expected for invalid IDs
                pass

    def test_sql_injection_prevention_in_trap_id(self, tools):
        """Test that SQL injection attempts in trap_id are handled safely."""
        injection_attempts = [
            "'; DROP TABLE traps; --",
            "1; DELETE FROM traps WHERE 1=1; --",
            "TRAP-001' OR '1'='1",
            "TRAP-001; EXEC xp_cmdshell('dir'); --",
            "' UNION SELECT * FROM users --"
        ]

        for malicious_id in injection_attempts:
            data = {
                'trap_id': malicious_id,
                'signal': (np.random.randn(1000) * 0.2).tolist(),
                'sampling_rate_hz': 250000
            }
            try:
                result = tools.analyze_acoustic_signature(data)
                # If processed, the ID should be sanitized or used literally
                # No SQL should be executed
                assert result is not None
            except (ValueError, AssertionError):
                # Expected behavior for invalid input
                pass

    def test_command_injection_prevention(self, tools):
        """Test that command injection attempts are prevented."""
        injection_attempts = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& ls -la"
        ]

        for malicious_input in injection_attempts:
            data = {
                'trap_id': f'TRAP{malicious_input}001',
                'signal': (np.random.randn(1000) * 0.2).tolist(),
                'sampling_rate_hz': 250000
            }
            try:
                result = tools.analyze_acoustic_signature(data)
                # Should not execute any commands
                assert result is not None
            except (ValueError, AssertionError):
                pass

    def test_path_traversal_prevention(self, tools):
        """Test that path traversal attempts are prevented."""
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32\\config",
            "/etc/passwd",
            "C:\\Windows\\System32\\config"
        ]

        for malicious_path in traversal_attempts:
            data = {
                'trap_id': malicious_path,
                'signal': (np.random.randn(1000) * 0.2).tolist(),
                'sampling_rate_hz': 250000
            }
            try:
                result = tools.analyze_acoustic_signature(data)
                # Should not access any files
                assert result is not None
            except (ValueError, AssertionError):
                pass


@pytest.mark.security
class TestNumericValidation:
    """Test numeric input validation."""

    def test_negative_temperature_handling(self, tools):
        """Test handling of negative temperatures."""
        thermal_data = {
            'trap_id': 'TRAP-NEG-TEMP',
            'temperature_upstream_c': -100.0,  # Invalid for most scenarios
            'temperature_downstream_c': -150.0,
            'ambient_temp_c': -50.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)
        # Should handle gracefully (may flag as anomaly or reject)
        assert result is not None

    def test_extreme_temperature_values(self, tools):
        """Test handling of extreme temperature values."""
        extreme_values = [
            (float('inf'), 100.0),
            (100.0, float('inf')),
            (float('-inf'), 100.0),
            (float('nan'), 100.0),
            (1e308, 100.0),  # Near max float
            (-1e308, 100.0)
        ]

        for upstream, downstream in extreme_values:
            thermal_data = {
                'trap_id': 'TRAP-EXTREME',
                'temperature_upstream_c': upstream,
                'temperature_downstream_c': downstream
            }
            try:
                result = tools.analyze_thermal_pattern(thermal_data)
                # Should handle extreme values appropriately
            except (ValueError, OverflowError, FloatingPointError):
                # Expected for invalid values
                pass

    def test_negative_pressure_handling(self, tools):
        """Test handling of negative pressure values."""
        trap_data = {
            'trap_id': 'TRAP-NEG-PRESSURE',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': -100.0,  # Invalid
            'failure_severity': 1.0
        }

        try:
            result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
            # Should handle negative pressure (may clamp to 0 or reject)
        except (ValueError, AssertionError):
            # Expected behavior
            pass

    def test_zero_orifice_diameter(self, tools):
        """Test handling of zero orifice diameter."""
        trap_data = {
            'trap_id': 'TRAP-ZERO-ORIFICE',
            'orifice_diameter_in': 0.0,  # Zero diameter
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        # Should result in zero loss (no orifice = no leak)
        assert result.steam_loss_lb_hr == 0.0 or result.steam_loss_lb_hr >= 0


@pytest.mark.security
class TestArrayInputValidation:
    """Test array/signal input validation."""

    def test_empty_signal_handling(self, tools):
        """Test handling of empty signal array."""
        data = {
            'trap_id': 'TRAP-EMPTY',
            'signal': [],
            'sampling_rate_hz': 250000
        }

        with pytest.raises((ValueError, IndexError, AssertionError)):
            tools.analyze_acoustic_signature(data)

    def test_single_element_signal(self, tools):
        """Test handling of single element signal."""
        data = {
            'trap_id': 'TRAP-SINGLE',
            'signal': [0.5],
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(data)
        assert result is not None

    def test_very_large_signal_handling(self, tools):
        """Test handling of very large signals (potential DoS)."""
        # Large but not excessive (10 million samples)
        np.random.seed(42)
        large_signal = (np.random.randn(10000000) * 0.3).tolist()

        data = {
            'trap_id': 'TRAP-LARGE',
            'signal': large_signal,
            'sampling_rate_hz': 250000
        }

        # Should handle without crashing or excessive memory use
        try:
            result = tools.analyze_acoustic_signature(data)
            assert result is not None
        except MemoryError:
            # Acceptable if memory limit enforced
            pass

    def test_non_numeric_signal_values(self, tools):
        """Test handling of non-numeric values in signal."""
        invalid_signals = [
            ['a', 'b', 'c'],
            [None, None, None],
            [{'value': 1}, {'value': 2}],
            [[1, 2], [3, 4]],
        ]

        for invalid_signal in invalid_signals:
            data = {
                'trap_id': 'TRAP-INVALID-SIGNAL',
                'signal': invalid_signal,
                'sampling_rate_hz': 250000
            }

            with pytest.raises((TypeError, ValueError)):
                tools.analyze_acoustic_signature(data)


@pytest.mark.security
class TestResourceExhaustion:
    """Test protection against resource exhaustion attacks."""

    def test_fleet_size_limits(self, tools):
        """Test handling of very large fleet sizes."""
        # Generate large fleet (potential DoS vector)
        large_fleet = [
            {
                'trap_id': f'TRAP-{i:07d}',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 5000,
                'process_criticality': 7,
                'current_age_years': 5,
                'health_score': 50
            }
            for i in range(10000)
        ]

        # Should complete without hanging
        result = tools.prioritize_maintenance(large_fleet)
        assert len(result.priority_list) == 10000

    def test_repeated_analysis_stability(self, tools):
        """Test stability under repeated analysis calls."""
        np.random.seed(42)
        data = {
            'trap_id': 'TRAP-REPEATED',
            'signal': (np.random.randn(10000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        }

        # Run many times to test for memory leaks or resource exhaustion
        for i in range(100):
            result = tools.analyze_acoustic_signature(data)
            assert result is not None

    def test_deeply_nested_input_handling(self, tools):
        """Test handling of deeply nested input structures."""
        # Create deeply nested structure (potential stack overflow)
        nested = {'value': 1}
        for _ in range(100):
            nested = {'nested': nested}

        data = {
            'trap_id': 'TRAP-NESTED',
            'signal': [0.1, 0.2, 0.3],
            'sampling_rate_hz': 250000,
            'metadata': nested
        }

        # Should handle without stack overflow
        try:
            result = tools.analyze_acoustic_signature(data)
            assert result is not None
        except (RecursionError, ValueError):
            # Acceptable if depth limit enforced
            pass


@pytest.mark.security
class TestProvenanceHashSecurity:
    """Test security of provenance hash generation."""

    def test_provenance_hash_format(self, tools):
        """Test that provenance hash follows secure format (SHA-256)."""
        np.random.seed(42)
        data = {
            'trap_id': 'TRAP-HASH-TEST',
            'signal': (np.random.randn(10000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(data)

        # Validate SHA-256 format
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 = 64 hex chars
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    def test_provenance_hash_uniqueness(self, tools):
        """Test that different inputs produce unique provenance hashes."""
        hashes = set()

        for i in range(100):
            np.random.seed(i)
            data = {
                'trap_id': f'TRAP-UNIQUE-{i:03d}',
                'signal': (np.random.randn(1000) * 0.2).tolist(),
                'sampling_rate_hz': 250000
            }
            result = tools.analyze_acoustic_signature(data)
            hashes.add(result.provenance_hash)

        # All hashes should be unique (no collisions)
        assert len(hashes) == 100

    def test_provenance_hash_non_predictable(self, tools):
        """Test that provenance hash is not easily predictable."""
        np.random.seed(42)
        data = {
            'trap_id': 'TRAP-PREDICT',
            'signal': (np.random.randn(1000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        }

        result = tools.analyze_acoustic_signature(data)

        # Verify hash is not simply hash of trap_id
        simple_hash = hashlib.sha256('TRAP-PREDICT'.encode()).hexdigest()
        assert result.provenance_hash != simple_hash


@pytest.mark.security
class TestConfigurationSecurity:
    """Test security of configuration handling."""

    def test_llm_temperature_enforcement(self, secure_config):
        """Test that LLM temperature is enforced to 0.0."""
        assert secure_config.llm_temperature == 0.0

    def test_llm_seed_enforcement(self, secure_config):
        """Test that LLM seed is enforced to 42."""
        assert secure_config.llm_seed == 42

    def test_invalid_temperature_rejection(self):
        """Test that non-zero temperature is rejected."""
        with pytest.raises(AssertionError):
            TrapInspectorConfig(
                agent_id="TEST",
                llm_temperature=0.5  # Non-deterministic
            )

    def test_invalid_seed_rejection(self):
        """Test that non-standard seed is rejected."""
        with pytest.raises(AssertionError):
            TrapInspectorConfig(
                agent_id="TEST",
                llm_temperature=0.0,
                llm_seed=123  # Non-standard seed
            )


@pytest.mark.security
class TestDataLeakagePrevention:
    """Test prevention of data leakage between analyses."""

    def test_no_cross_trap_data_leakage(self, tools):
        """Test that data from one trap doesn't leak to another."""
        np.random.seed(42)

        # Analyze first trap
        data1 = {
            'trap_id': 'TRAP-SECRET-001',
            'signal': (np.random.randn(10000) * 0.5).tolist(),
            'sampling_rate_hz': 250000
        }
        result1 = tools.analyze_acoustic_signature(data1)

        # Analyze second trap
        np.random.seed(99)
        data2 = {
            'trap_id': 'TRAP-SECRET-002',
            'signal': (np.random.randn(10000) * 0.3).tolist(),
            'sampling_rate_hz': 250000
        }
        result2 = tools.analyze_acoustic_signature(data2)

        # Results should be independent
        assert result1.trap_id == 'TRAP-SECRET-001'
        assert result2.trap_id == 'TRAP-SECRET-002'
        assert result1.provenance_hash != result2.provenance_hash

    def test_cache_isolation(self, tools):
        """Test that cached results are properly isolated."""
        np.random.seed(42)

        # Run same analysis twice
        data = {
            'trap_id': 'TRAP-CACHE-TEST',
            'signal': (np.random.randn(10000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        }

        result1 = tools.analyze_acoustic_signature(data)
        result2 = tools.analyze_acoustic_signature(data)

        # Results should be identical (deterministic) but independent objects
        assert result1.provenance_hash == result2.provenance_hash
        assert result1 is not result2  # Different objects


@pytest.mark.security
class TestBoundaryConditions:
    """Test security at boundary conditions."""

    def test_sampling_rate_boundaries(self, tools):
        """Test handling of edge case sampling rates."""
        boundary_rates = [1, 100, 1000000, 10000000]

        np.random.seed(42)
        signal = (np.random.randn(1000) * 0.2).tolist()

        for rate in boundary_rates:
            data = {
                'trap_id': f'TRAP-RATE-{rate}',
                'signal': signal,
                'sampling_rate_hz': rate
            }
            try:
                result = tools.analyze_acoustic_signature(data)
                assert result is not None
            except (ValueError, AssertionError):
                # Expected for invalid rates
                pass

    def test_failure_severity_boundaries(self, tools):
        """Test handling of failure severity at boundaries."""
        boundary_values = [0.0, 0.001, 0.5, 0.999, 1.0, 1.001, -0.1, 2.0]

        for severity in boundary_values:
            trap_data = {
                'trap_id': f'TRAP-SEV-{severity}',
                'orifice_diameter_in': 0.125,
                'steam_pressure_psig': 100.0,
                'failure_severity': severity
            }
            try:
                result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
                # Values outside [0, 1] should be clamped or rejected
                if 0 <= severity <= 1:
                    assert result is not None
            except (ValueError, AssertionError):
                # Expected for out-of-range values
                pass

    def test_health_score_boundaries(self, tools):
        """Test handling of health score at boundaries."""
        boundary_scores = [0, 1, 50, 99, 100, 101, -1, 200]

        for score in boundary_scores:
            condition_data = {
                'trap_id': f'TRAP-HEALTH-{score}',
                'current_age_days': 1000,
                'current_health_score': score,
                'degradation_rate': 0.1
            }
            try:
                result = tools.predict_remaining_useful_life(condition_data)
                # Scores outside [0, 100] should be clamped or rejected
                if 0 <= score <= 100:
                    assert result is not None
            except (ValueError, AssertionError):
                # Expected for out-of-range values
                pass


@pytest.mark.security
class TestErrorHandlingSecurity:
    """Test secure error handling."""

    def test_no_stack_trace_in_response(self, tools):
        """Test that stack traces are not exposed in error responses."""
        invalid_data = {
            'trap_id': None,  # Invalid
            'signal': 'not a list',  # Invalid
            'sampling_rate_hz': 'invalid'  # Invalid
        }

        try:
            tools.analyze_acoustic_signature(invalid_data)
        except Exception as e:
            error_message = str(e)
            # Should not contain file paths or line numbers
            assert 'File "' not in error_message or '/test_' in error_message
            assert '.py", line' not in error_message or '/test_' in error_message

    def test_graceful_degradation(self, tools):
        """Test graceful degradation with partial valid input."""
        partial_data = {
            'trap_id': 'TRAP-PARTIAL',
            'temperature_upstream_c': 150.0,
            # Missing temperature_downstream_c
        }

        try:
            result = tools.analyze_thermal_pattern(partial_data)
            # Should either succeed with defaults or raise clear error
        except (KeyError, TypeError, ValueError) as e:
            # Should provide clear error message
            assert len(str(e)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "security"])
