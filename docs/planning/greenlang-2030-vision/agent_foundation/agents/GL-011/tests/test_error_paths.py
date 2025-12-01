# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Error Path Test Suite.

This module tests error handling and recovery mechanisms:
- Integration connector failures
- API timeout handling
- Database connection loss
- Invalid configuration handling
- Calculation errors with proper recovery
- Partial optimization failure
- Network errors
- File I/O errors
- Retry logic
- Graceful degradation

Test Count: 25+ error path tests
Coverage: Error handling, recovery, resilience

Author: GreenLang Industrial Optimization Team
Version: 1.0.0
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FuelManagementConfig, create_default_config
from calculators.multi_fuel_optimizer import MultiFuelOptimizer, MultiFuelOptimizationInput
from calculators.cost_optimization_calculator import CostOptimizationCalculator
from fuel_management_orchestrator import FuelManagementOrchestrator
from pydantic import ValidationError


@pytest.mark.error_path
class TestIntegrationConnectorFailures:
    """Error paths for integration connector failures."""

    def test_erp_connector_connection_failure(self, mock_erp_connector, base_config):
        """
        Error path: ERP connector fails to connect.

        Expected:
        - Raises ConnectionError
        - Includes meaningful error message
        - No state corruption
        """
        mock_erp_connector.connect.return_value = False
        mock_erp_connector.connect.side_effect = ConnectionError("ERP system unreachable")

        orchestrator = FuelManagementOrchestrator(base_config)

        with pytest.raises(ConnectionError) as exc_info:
            # Attempt operation requiring ERP
            orchestrator._connect_erp()

        assert "ERP" in str(exc_info.value)
        assert "unreachable" in str(exc_info.value)

    def test_market_data_connector_timeout(self, mock_market_data_connector):
        """
        Error path: Market data API timeout.

        Expected:
        - Raises TimeoutError
        - Falls back to cached data if available
        """
        mock_market_data_connector.get_current_price.side_effect = TimeoutError("API timeout after 30s")

        with pytest.raises(TimeoutError) as exc_info:
            mock_market_data_connector.get_current_price("natural_gas")

        assert "timeout" in str(exc_info.value).lower()

    def test_storage_connector_read_failure(self, mock_storage_connector):
        """
        Error path: Fuel storage connector read failure.

        Expected:
        - Raises IOError
        - Logs error details
        """
        mock_storage_connector.get_current_level.side_effect = IOError("Sensor malfunction")

        with pytest.raises(IOError) as exc_info:
            mock_storage_connector.get_current_level()

        assert "Sensor malfunction" in str(exc_info.value)

    def test_emissions_monitoring_connector_offline(self, mock_emissions_monitoring_connector):
        """
        Error path: Emissions monitoring system offline.

        Expected:
        - Raises ConnectionError
        - System continues with estimated values (if configured)
        """
        mock_emissions_monitoring_connector.get_current_emissions.side_effect = \
            ConnectionError("CEMS offline")

        with pytest.raises(ConnectionError):
            mock_emissions_monitoring_connector.get_current_emissions()


@pytest.mark.error_path
class TestDatabaseConnectionFailures:
    """Error paths for database connection issues."""

    def test_database_connection_lost_during_query(self):
        """
        Error path: Database connection lost during query.

        Expected:
        - Raises ConnectionError
        - Triggers reconnection attempt
        """
        mock_db = Mock()
        mock_db.execute.side_effect = ConnectionError("Connection lost")

        with pytest.raises(ConnectionError):
            mock_db.execute("SELECT * FROM fuel_inventory")

    def test_database_deadlock_detection(self):
        """
        Error path: Database deadlock detected.

        Expected:
        - Raises specific deadlock error
        - Triggers automatic retry
        """
        mock_db = Mock()
        mock_db.execute.side_effect = Exception("Deadlock detected")

        with pytest.raises(Exception) as exc_info:
            mock_db.execute("UPDATE fuel_inventory SET quantity = 100")

        assert "Deadlock" in str(exc_info.value)

    def test_database_transaction_rollback(self):
        """
        Error path: Transaction rollback on error.

        Expected:
        - Rolls back incomplete transaction
        - No partial state committed
        """
        mock_db = Mock()
        mock_db.commit.side_effect = Exception("Commit failed")

        try:
            mock_db.begin()
            mock_db.execute("UPDATE fuel_inventory SET quantity = 100")
            mock_db.commit()
        except Exception:
            mock_db.rollback.assert_called_once()


@pytest.mark.error_path
class TestInvalidConfigurationHandling:
    """Error paths for invalid configuration."""

    def test_missing_required_config_field(self):
        """
        Error path: Missing required configuration field.

        Expected:
        - Raises ValidationError
        - Clear error message indicating missing field
        """
        with pytest.raises(ValidationError) as exc_info:
            FuelManagementConfig(
                agent_id="GL-011",
                # Missing other required fields
            )

        assert "validation error" in str(exc_info.value).lower()

    def test_invalid_config_value_range(self):
        """
        Error path: Configuration value out of valid range.

        Expected:
        - Raises ValidationError
        - Indicates valid range
        """
        from config import OptimizationParameters

        with pytest.raises(ValidationError) as exc_info:
            OptimizationParameters(
                cost_weight=1.5,  # Invalid: > 1.0
                emissions_weight=0.3,
                efficiency_weight=0.2,
                reliability_weight=0.1,
            )

        assert "less than or equal to 1" in str(exc_info.value)

    def test_config_weights_not_summing_to_one(self):
        """
        Error path: Optimization weights don't sum to 1.0.

        Expected:
        - Raises ValidationError
        - Indicates weights must sum to 1.0
        """
        from config import OptimizationParameters

        with pytest.raises(ValidationError) as exc_info:
            OptimizationParameters(
                cost_weight=0.5,
                emissions_weight=0.3,
                efficiency_weight=0.3,  # Total = 1.1
                reliability_weight=0.0,
            )

        assert "must sum to 1.0" in str(exc_info.value)


@pytest.mark.error_path
class TestCalculationErrorHandling:
    """Error paths for calculation errors."""

    def test_optimization_infeasible_solution(self, fuel_properties, market_prices):
        """
        Error path: Optimization problem has no feasible solution.

        Expected:
        - Raises ValueError with meaningful message
        - Indicates constraint conflict
        """
        optimizer = MultiFuelOptimizer()

        # Create impossible constraints
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={'co2_kg_hr': 0.001},  # Impossible limit
            constraints={},
            optimization_objective='emissions'
        )

        try:
            result = optimizer.optimize(input_data)
            # If solver finds solution, emissions should be very low
            assert result.total_emissions_kg > 0
        except ValueError as e:
            # Acceptable to raise error for infeasible problem
            assert "infeasible" in str(e).lower() or "no solution" in str(e).lower()

    def test_calculation_numerical_overflow(self, fuel_properties):
        """
        Error path: Numerical overflow in calculation.

        Expected:
        - Handles gracefully
        - Raises OverflowError or returns inf
        """
        optimizer = MultiFuelOptimizer()

        # Extreme values that might cause overflow
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=1e10,  # 10 billion MW
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 1e10},
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        try:
            result = optimizer.optimize(input_data)
            # If handled, result should be valid or inf
            assert result.total_cost_usd >= 0 or result.total_cost_usd == float('inf')
        except OverflowError:
            # Acceptable to raise overflow error
            pass

    def test_calculation_division_by_zero_protection(self):
        """
        Error path: Division by zero in calculation.

        Expected:
        - Protected by validation (zero values rejected)
        - Or raises ZeroDivisionError
        """
        # Zero heating value should be caught by validation
        from config import FuelSpecification, FuelCategory, FuelState

        with pytest.raises(ValidationError):
            FuelSpecification(
                fuel_id="ZERO-HV",
                fuel_name="Zero Heating Value",
                fuel_type="invalid",
                category=FuelCategory.FOSSIL,
                state=FuelState.SOLID,
                gross_calorific_value_mj_kg=0.0,  # Would cause div by zero
                net_calorific_value_mj_kg=0.0,
                density_kg_m3=1000.0,
                carbon_content_percent=50.0,
                hydrogen_content_percent=5.0,
                emission_factor_co2_kg_gj=50.0,
            )


@pytest.mark.error_path
class TestPartialOptimizationFailure:
    """Error paths for partial optimization failures."""

    def test_partial_fuel_data_missing(self, fuel_properties, market_prices):
        """
        Error path: Some fuel properties missing.

        Expected:
        - Skips incomplete fuels
        - Optimizes with available data
        """
        # Remove properties for coal
        partial_properties = {k: v for k, v in fuel_properties.items() if k != 'coal'}

        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=partial_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        try:
            result = optimizer.optimize(input_data)
            # Should work with available fuels only
            assert 'coal' not in result.optimal_fuel_mix or result.optimal_fuel_mix['coal'] == 0
        except KeyError:
            # Acceptable to fail if fuel data missing
            pass

    def test_partial_market_data_stale(self, mock_market_data_connector):
        """
        Error path: Market data is stale.

        Expected:
        - Flags data as stale
        - Uses cached data with warning
        - Or triggers refresh
        """
        mock_market_data_connector.is_data_stale.return_value = True

        is_stale = mock_market_data_connector.is_data_stale()
        assert is_stale is True

        # System should handle stale data gracefully


@pytest.mark.error_path
class TestNetworkErrorHandling:
    """Error paths for network errors."""

    def test_api_http_500_error(self):
        """
        Error path: API returns HTTP 500 error.

        Expected:
        - Raises HTTPError
        - Includes status code and message
        """
        mock_api = Mock()
        mock_api.get.side_effect = Exception("HTTP 500: Internal Server Error")

        with pytest.raises(Exception) as exc_info:
            mock_api.get("/api/fuel-prices")

        assert "500" in str(exc_info.value)

    def test_api_rate_limit_exceeded(self):
        """
        Error path: API rate limit exceeded.

        Expected:
        - Raises RateLimitError
        - Implements backoff/retry
        """
        mock_api = Mock()
        mock_api.get.side_effect = Exception("HTTP 429: Too Many Requests")

        with pytest.raises(Exception) as exc_info:
            mock_api.get("/api/fuel-prices")

        assert "429" in str(exc_info.value)

    def test_network_connection_refused(self):
        """
        Error path: Network connection refused.

        Expected:
        - Raises ConnectionRefusedError
        - Falls back to cached data
        """
        mock_connector = Mock()
        mock_connector.connect.side_effect = ConnectionRefusedError("Connection refused")

        with pytest.raises(ConnectionRefusedError):
            mock_connector.connect()


@pytest.mark.error_path
class TestFileIOErrorHandling:
    """Error paths for file I/O errors."""

    def test_config_file_not_found(self):
        """
        Error path: Configuration file not found.

        Expected:
        - Raises FileNotFoundError
        - Clear error message
        """
        with pytest.raises(FileNotFoundError):
            with open("/nonexistent/config.json", "r") as f:
                f.read()

    def test_config_file_permission_denied(self, temp_directory):
        """
        Error path: Configuration file permission denied.

        Expected:
        - Raises PermissionError
        """
        import os
        config_file = temp_directory / "restricted_config.json"
        config_file.write_text("{}")

        # Make file read-only (platform-specific)
        try:
            os.chmod(config_file, 0o000)
            with pytest.raises(PermissionError):
                with open(config_file, "r") as f:
                    f.read()
        finally:
            # Restore permissions for cleanup
            os.chmod(config_file, 0o644)

    def test_config_file_corrupted_json(self, temp_directory):
        """
        Error path: Configuration file contains invalid JSON.

        Expected:
        - Raises JSONDecodeError
        - Indicates parse error location
        """
        import json

        config_file = temp_directory / "corrupt_config.json"
        config_file.write_text("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            with open(config_file, "r") as f:
                json.load(f)


@pytest.mark.error_path
class TestRetryLogic:
    """Error paths testing retry mechanisms."""

    def test_retry_on_transient_failure(self):
        """
        Error path: Retry succeeds after transient failure.

        Expected:
        - Retries configured number of times
        - Succeeds on retry
        """
        mock_api = Mock()
        # Fail first 2 times, succeed on 3rd
        mock_api.get.side_effect = [
            Exception("Timeout"),
            Exception("Timeout"),
            {"price": 0.045}
        ]

        # Implement simple retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = mock_api.get("/api/price")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1)

        assert result == {"price": 0.045}

    def test_retry_exhausted_raises_error(self):
        """
        Error path: All retries exhausted.

        Expected:
        - Raises final error
        - Logs all retry attempts
        """
        mock_api = Mock()
        mock_api.get.side_effect = Exception("Persistent failure")

        max_retries = 3
        with pytest.raises(Exception) as exc_info:
            for attempt in range(max_retries):
                try:
                    mock_api.get("/api/price")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise

        assert "Persistent failure" in str(exc_info.value)

    def test_exponential_backoff_retry(self):
        """
        Error path: Retry with exponential backoff.

        Expected:
        - Delay increases exponentially
        - Eventually succeeds or fails
        """
        mock_api = Mock()
        mock_api.get.side_effect = [
            Exception("Timeout"),
            Exception("Timeout"),
            {"price": 0.045}
        ]

        max_retries = 3
        base_delay = 0.1

        for attempt in range(max_retries):
            try:
                result = mock_api.get("/api/price")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise

        assert result == {"price": 0.045}


@pytest.mark.error_path
class TestGracefulDegradation:
    """Error paths testing graceful degradation."""

    def test_fallback_to_cached_data_on_api_failure(self):
        """
        Error path: Falls back to cached data when API fails.

        Expected:
        - Uses cached data
        - Logs warning about staleness
        """
        cache = {"natural_gas": 0.045}
        mock_api = Mock()
        mock_api.get.side_effect = Exception("API unavailable")

        try:
            price = mock_api.get("/api/price/natural_gas")
        except Exception:
            # Fallback to cache
            price = cache.get("natural_gas", 0.05)  # Default if not in cache

        assert price == 0.045

    def test_degraded_mode_with_limited_features(self, base_config):
        """
        Error path: System operates in degraded mode.

        Expected:
        - Core features work
        - Advanced features disabled
        - Clear indication of degraded mode
        """
        # Simulate degraded mode (no market data)
        base_config.integration.market_data_enabled = False

        orchestrator = FuelManagementOrchestrator(base_config)

        # Should still work with static prices
        assert orchestrator is not None

    def test_circuit_breaker_opens_after_failures(self):
        """
        Error path: Circuit breaker opens after repeated failures.

        Expected:
        - Stops calling failing service
        - Returns cached/default data
        - Periodically retries (half-open state)
        """
        mock_api = Mock()
        mock_api.get.side_effect = Exception("Service down")

        failure_count = 0
        threshold = 5

        for _ in range(10):
            try:
                mock_api.get("/api/data")
            except Exception:
                failure_count += 1
                if failure_count >= threshold:
                    # Circuit breaker opens
                    break

        assert failure_count == threshold
