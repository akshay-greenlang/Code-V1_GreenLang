# -*- coding: utf-8 -*-
"""
Tests for security validation.

Tests security aspects of GL-011 FUELCRAFT:
- Input sanitization
- SQL injection prevention
- Path traversal prevention
- Data validation boundaries
- Authentication token handling
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.multi_fuel_optimizer import MultiFuelOptimizer, MultiFuelOptimizationInput
from calculators.cost_optimization_calculator import CostOptimizationCalculator, CostOptimizationInput
from config import FuelManagementConfig, FuelSpecification


class TestInputSanitization:
    """Test suite for input sanitization."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return MultiFuelOptimizer()

    @pytest.fixture
    def valid_fuel_properties(self):
        """Valid fuel properties for testing."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            }
        }

    def test_sql_injection_in_fuel_name(self, optimizer, valid_fuel_properties):
        """Test SQL injection attempts in fuel names are rejected."""
        malicious_names = [
            "'; DROP TABLE fuels; --",
            "natural_gas'; DELETE FROM prices WHERE '1'='1",
            "1 OR 1=1",
            "UNION SELECT * FROM users--",
        ]

        for malicious_name in malicious_names:
            # Fuel name with SQL injection should be rejected or sanitized
            with pytest.raises((ValueError, KeyError)):
                input_data = MultiFuelOptimizationInput(
                    energy_demand_mw=100,
                    available_fuels=[malicious_name],
                    fuel_properties={malicious_name: valid_fuel_properties['natural_gas']},
                    market_prices={malicious_name: 0.045},
                    emission_limits={},
                    constraints={},
                    optimization_objective='balanced'
                )
                optimizer.optimize(input_data)

    def test_path_traversal_prevention(self):
        """Test path traversal attempts are prevented."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        for path in malicious_paths:
            # Config should reject path traversal attempts
            with pytest.raises((ValueError, OSError)):
                config = FuelManagementConfig(
                    config_file_path=path
                )

    def test_numeric_overflow_prevention(self, optimizer, valid_fuel_properties):
        """Test numeric overflow is prevented."""
        overflow_values = [
            float('inf'),
            -float('inf'),
            1e308,  # Near max float
            -1e308,
        ]

        for value in overflow_values:
            with pytest.raises(ValueError):
                input_data = MultiFuelOptimizationInput(
                    energy_demand_mw=value,
                    available_fuels=['natural_gas'],
                    fuel_properties=valid_fuel_properties,
                    market_prices={'natural_gas': 0.045},
                    emission_limits={},
                    constraints={},
                    optimization_objective='balanced'
                )
                optimizer.optimize(input_data)

    def test_negative_value_rejection(self, optimizer, valid_fuel_properties):
        """Test negative values are rejected where inappropriate."""
        with pytest.raises(ValueError):
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=-100,  # Negative demand
                available_fuels=['natural_gas'],
                fuel_properties=valid_fuel_properties,
                market_prices={'natural_gas': 0.045},
                emission_limits={},
                constraints={},
                optimization_objective='balanced'
            )
            optimizer.optimize(input_data)

    def test_nan_value_rejection(self, optimizer, valid_fuel_properties):
        """Test NaN values are rejected."""
        import math

        with pytest.raises(ValueError):
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=float('nan'),
                available_fuels=['natural_gas'],
                fuel_properties=valid_fuel_properties,
                market_prices={'natural_gas': 0.045},
                emission_limits={},
                constraints={},
                optimization_objective='balanced'
            )
            optimizer.optimize(input_data)


class TestDataBoundaryValidation:
    """Test suite for data boundary validation."""

    def test_fuel_heating_value_bounds(self):
        """Test heating value is within physical bounds."""
        # Hydrogen has highest heating value at ~120 MJ/kg
        # Values above 150 MJ/kg are physically impossible
        with pytest.raises(ValueError):
            FuelSpecification(
                fuel_id="impossible_fuel",
                fuel_name="Impossible Fuel",
                fuel_type="solid",
                heating_value_mj_kg=200.0,  # Physically impossible
                emission_factor_co2_kg_gj=0.0
            )

    def test_emission_factor_non_negative(self):
        """Test emission factors cannot be negative."""
        with pytest.raises(ValueError):
            FuelSpecification(
                fuel_id="bad_fuel",
                fuel_name="Bad Fuel",
                fuel_type="solid",
                heating_value_mj_kg=25.0,
                emission_factor_co2_kg_gj=-10.0  # Negative emission impossible
            )

    def test_percentage_bounds(self):
        """Test percentage values are 0-100."""
        with pytest.raises(ValueError):
            FuelSpecification(
                fuel_id="bad_fuel",
                fuel_name="Bad Fuel",
                fuel_type="solid",
                heating_value_mj_kg=25.0,
                emission_factor_co2_kg_gj=90.0,
                carbon_content_percent=150.0  # Over 100%
            )

    def test_string_length_limits(self):
        """Test string fields have length limits."""
        # Very long string should be rejected
        long_string = "A" * 10000

        with pytest.raises(ValueError):
            FuelSpecification(
                fuel_id=long_string,  # Too long
                fuel_name="Normal Name",
                fuel_type="solid",
                heating_value_mj_kg=25.0,
                emission_factor_co2_kg_gj=90.0
            )


class TestAuthenticationSecurity:
    """Test suite for authentication security."""

    def test_api_key_not_logged(self, caplog):
        """Test API keys are not logged in plain text."""
        import logging

        # Create config with API key
        config = FuelManagementConfig(
            api_key="super_secret_key_12345"
        )

        # Check logs don't contain the key
        for record in caplog.records:
            assert "super_secret_key_12345" not in record.message

    def test_credentials_not_in_repr(self):
        """Test credentials are masked in repr."""
        config = FuelManagementConfig(
            api_key="secret_api_key",
            database_password="db_password_123"
        )

        repr_str = repr(config)
        assert "secret_api_key" not in repr_str
        assert "db_password_123" not in repr_str

    def test_credentials_not_in_str(self):
        """Test credentials are masked in str output."""
        config = FuelManagementConfig(
            api_key="secret_api_key",
            database_password="db_password_123"
        )

        str_output = str(config)
        assert "secret_api_key" not in str_output
        assert "db_password_123" not in str_output


class TestProvenanceIntegrity:
    """Test suite for provenance integrity."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return MultiFuelOptimizer()

    @pytest.fixture
    def fuel_properties(self):
        """Standard fuel properties."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            }
        }

    def test_provenance_hash_format(self, optimizer, fuel_properties):
        """Test provenance hash is valid SHA-256."""
        import re

        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        # SHA-256 is 64 hex characters
        assert len(result.provenance_hash) == 64
        assert re.match(r'^[a-f0-9]{64}$', result.provenance_hash)

    def test_provenance_tamper_detection(self, optimizer, fuel_properties):
        """Test that tampered data is detectable via hash mismatch."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result1 = optimizer.optimize(input_data)

        # Modify input slightly
        input_data2 = MultiFuelOptimizationInput(
            energy_demand_mw=100.001,  # Tiny change
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result2 = optimizer.optimize(input_data2)

        # Hashes should differ
        assert result1.provenance_hash != result2.provenance_hash


class TestConcurrencySafety:
    """Test suite for concurrency safety."""

    def test_thread_safe_optimization(self):
        """Test optimizer is thread-safe."""
        import threading
        import time

        optimizer = MultiFuelOptimizer()
        fuel_properties = {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            }
        }

        results = []
        errors = []

        def run_optimization(demand):
            try:
                input_data = MultiFuelOptimizationInput(
                    energy_demand_mw=demand,
                    available_fuels=['natural_gas'],
                    fuel_properties=fuel_properties,
                    market_prices={'natural_gas': 0.045},
                    emission_limits={},
                    constraints={},
                    optimization_objective='balanced'
                )
                result = optimizer.optimize(input_data)
                results.append((demand, result))
            except Exception as e:
                errors.append((demand, str(e)))

        # Run concurrent optimizations
        threads = []
        for i in range(10):
            t = threading.Thread(target=run_optimization, args=(100 + i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        assert len(results) == 10

    def test_no_race_conditions_in_cache(self):
        """Test cache has no race conditions."""
        import threading
        from fuel_management_orchestrator import ThreadSafeCache

        cache = ThreadSafeCache(max_size=100)
        errors = []

        def write_read_cycle(thread_id):
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    value = f"value_{thread_id}_{i}"
                    cache.set(key, value)
                    retrieved = cache.get(key)
                    if retrieved != value:
                        errors.append(f"Mismatch: expected {value}, got {retrieved}")
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=write_read_cycle, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
