# -*- coding: utf-8 -*-
"""
Unit tests for utility functions.

Tests cover:
- Unit conversion utilities
- Network utilities
- Performance tracking
- File operations
- Path handling
- Logging utilities
- Helper functions
"""

import os
import tempfile
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, mock_open, Mock
from typing import Dict, Any
import requests

from greenlang.utils.unit_converter import UnitConverter
from greenlang.utils.net import (
    NetworkPolicy, http_get, http_post, download_file,
    policy_allow, add_allowed_domain, add_blocked_domain,
    get_network_audit_log, reset_network_policy
)
from greenlang.utils.performance_tracker import PerformanceTracker


class TestUnitConverter:
    """Test unit conversion utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = UnitConverter()

    def test_energy_conversion_basic(self):
        """Test basic energy conversions."""
        converter = UnitConverter()

        # kWh to MMBtu
        result = converter.convert_energy(1000, "kWh", "MMBtu")
        assert abs(result - 3.412) < 0.001

        # MWh to kWh
        result = converter.convert_energy(1, "MWh", "kWh")
        assert result == 1000

        # Same unit conversion
        result = converter.convert_energy(100, "kWh", "kWh")
        assert result == 100

    def test_energy_conversion_invalid_units(self):
        """Test energy conversion with invalid units."""
        converter = UnitConverter()

        with pytest.raises(ValueError, match="Unknown energy unit"):
            converter.convert_energy(100, "invalid_unit", "kWh")

        with pytest.raises(ValueError, match="Unknown energy unit"):
            converter.convert_energy(100, "kWh", "invalid_unit")

    def test_area_conversion(self):
        """Test area conversions."""
        converter = UnitConverter()

        # sqm to sqft
        result = converter.convert_area(100, "sqm", "sqft")
        assert abs(result - 1076.4) < 0.1

        # sqft to sqm
        result = converter.convert_area(1076.4, "sqft", "sqm")
        assert abs(result - 100) < 0.1

        # Same unit
        result = converter.convert_area(500, "sqft", "sqft")
        assert result == 500

    def test_area_conversion_invalid_units(self):
        """Test area conversion with invalid units."""
        converter = UnitConverter()

        with pytest.raises(ValueError, match="Unknown area unit"):
            converter.convert_area(100, "invalid_area", "sqft")

    def test_mass_conversion(self):
        """Test mass conversions."""
        converter = UnitConverter()

        # kg to lbs
        result = converter.convert_mass(100, "kg", "lbs")
        assert abs(result - 220.46) < 0.1

        # tons to kg
        result = converter.convert_mass(1, "ton", "kg")
        assert result == 1000

        # Same unit
        result = converter.convert_mass(50, "kg", "kg")
        assert result == 50

    def test_volume_conversion(self):
        """Test volume conversions."""
        converter = UnitConverter()

        # gallons to liters
        result = converter.convert_volume(1, "gallon", "liter")
        assert abs(result - 3.785) < 0.01

        # liters to ml
        result = converter.convert_volume(1, "liter", "ml")
        assert result == 1000

        # Same unit
        result = converter.convert_volume(100, "liter", "liter")
        assert result == 100

    def test_fuel_to_energy_conversion(self):
        """Test fuel to energy conversions."""
        converter = UnitConverter()

        # Natural gas therms to MMBtu
        result = converter.convert_fuel_to_energy(10, "therms", "natural_gas", "MMBtu")
        assert abs(result - 1.0) < 0.01

        # Diesel gallons to MMBtu
        result = converter.convert_fuel_to_energy(1, "gallon", "diesel", "MMBtu")
        assert abs(result - 0.138) < 0.001

        # Unknown fuel type - should fallback to energy conversion
        result = converter.convert_fuel_to_energy(1, "kWh", "unknown_fuel", "MMBtu")
        assert abs(result - 0.003412) < 0.000001

    def test_fuel_to_energy_invalid_unit(self):
        """Test fuel to energy conversion with invalid units."""
        converter = UnitConverter()

        with pytest.raises(ValueError, match="Unknown unit"):
            converter.convert_fuel_to_energy(1, "invalid_unit", "natural_gas", "MMBtu")

    def test_emissions_conversion(self):
        """Test emissions conversions."""
        converter = UnitConverter()

        # kg to tons
        result = converter.convert_emissions(1000, "kg", "tons")
        assert result == 1

        # tCO2e to kgCO2e
        result = converter.convert_emissions(1, "tCO2e", "kgCO2e")
        assert result == 1000

        # Same unit
        result = converter.convert_emissions(500, "kg", "kg")
        assert result == 500

    def test_normalize_unit_name(self):
        """Test unit name normalization."""
        converter = UnitConverter()

        # Test normalization
        assert converter.normalize_unit_name("square_feet") == "sqft"
        assert converter.normalize_unit_name("kilowatt_hour") == "kWh"
        assert converter.normalize_unit_name("million_btu") == "MMBtu"
        assert converter.normalize_unit_name("cubic_meters") == "m3"

        # Test case insensitive
        assert converter.normalize_unit_name("SQUARE_FEET") == "sqft"
        assert converter.normalize_unit_name("Kilowatt_Hour") == "kWh"

        # Test with spaces and hyphens
        assert converter.normalize_unit_name("square feet") == "sqft"
        assert converter.normalize_unit_name("kilowatt-hour") == "kWh"

    def test_get_conversion_factor(self):
        """Test getting conversion factors."""
        converter = UnitConverter()

        # Energy conversion factor
        factor = converter.get_conversion_factor("kWh", "MMBtu", "energy")
        assert abs(factor - 0.003412) < 0.000001

        # Area conversion factor
        factor = converter.get_conversion_factor("sqm", "sqft", "area")
        assert abs(factor - 10.764) < 0.001

        # Mass conversion factor
        factor = converter.get_conversion_factor("kg", "lbs", "mass")
        assert abs(factor - 2.204) < 0.01

        # Volume conversion factor
        factor = converter.get_conversion_factor("gallon", "liter", "volume")
        assert abs(factor - 3.785) < 0.01

    def test_get_conversion_factor_invalid_type(self):
        """Test conversion factor with invalid type."""
        converter = UnitConverter()

        with pytest.raises(ValueError, match="Unknown conversion type"):
            converter.get_conversion_factor("kg", "lbs", "invalid_type")


class TestNetworkUtilities:
    """Test network utilities and policy enforcement."""

    def test_network_policy_check_url_allowed(self):
        """Test URL checking for allowed domains."""
        policy = NetworkPolicy()

        # Add test domain
        policy.allowed_domains.append("example.com")

        assert policy.check_url("https://example.com/api", "test") is True
        assert policy.check_url("http://sub.example.com/data", "test") is True

    def test_network_policy_check_url_blocked(self):
        """Test URL checking for blocked domains."""
        policy = NetworkPolicy()

        # Add to blocklist
        policy.blocked_domains.append("malicious.com")

        assert policy.check_url("https://malicious.com/bad", "test") is False
        assert policy.check_url("http://sub.malicious.com/evil", "test") is False

    def test_network_policy_check_url_not_in_allowlist(self):
        """Test URL checking for domains not in allowlist."""
        policy = NetworkPolicy()

        # Clear allowlist
        policy.allowed_domains.clear()

        assert policy.check_url("https://unknown.com/api", "test") is False

    def test_network_policy_audit_log(self):
        """Test network policy audit logging."""
        policy = NetworkPolicy()

        # Clear allowlist and blocklist
        policy.allowed_domains.clear()
        policy.blocked_domains.clear()

        # Add specific domains
        policy.allowed_domains.append("allowed.com")
        policy.blocked_domains.append("blocked.com")

        # Test allowed access
        policy.check_url("https://allowed.com/api", "test_allowed")

        # Test blocked access
        policy.check_url("https://blocked.com/bad", "test_blocked")

        # Test denied access
        policy.check_url("https://unknown.com/api", "test_denied")

        # Check audit log
        audit_log = policy.get_audit_log()
        assert len(audit_log) == 3

        # Check entries
        allowed_entry = next(e for e in audit_log if e["action"] == "allowed")
        assert allowed_entry["domain"] == "allowed.com"
        assert allowed_entry["tag"] == "test_allowed"

        blocked_entry = next(e for e in audit_log if e["action"] == "blocked")
        assert blocked_entry["domain"] == "blocked.com"
        assert blocked_entry["tag"] == "test_blocked"

        denied_entry = next(e for e in audit_log if e["action"] == "denied")
        assert denied_entry["domain"] == "unknown.com"
        assert denied_entry["tag"] == "test_denied"

    def test_policy_allow_success(self):
        """Test policy_allow function with allowed URL."""
        with patch('greenlang.utils.net._network_policy') as mock_policy:
            mock_policy.check_url.return_value = True

            # Should not raise
            policy_allow("https://allowed.com/api", "test")
            mock_policy.check_url.assert_called_once_with("https://allowed.com/api", "test")

    def test_policy_allow_failure(self):
        """Test policy_allow function with denied URL."""
        with patch('greenlang.utils.net._network_policy') as mock_policy:
            mock_policy.check_url.return_value = False

            with pytest.raises(RuntimeError, match="Network access denied by policy"):
                policy_allow("https://denied.com/api", "test")

    @patch('greenlang.utils.net.requests.get')
    def test_http_get_success(self, mock_get):
        """Test successful HTTP GET request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch('greenlang.utils.net.policy_allow'):
            response = http_get("https://example.com/api", tag="test")

            assert response == mock_response
            mock_get.assert_called_once_with("https://example.com/api", timeout=30)

    @patch('greenlang.utils.net.requests.get')
    def test_http_get_policy_denied(self, mock_get):
        """Test HTTP GET with policy denial."""
        with patch('greenlang.utils.net.policy_allow', side_effect=RuntimeError("Denied")):
            with pytest.raises(RuntimeError, match="Denied"):
                http_get("https://denied.com/api", tag="test")

            # Should not make the request
            mock_get.assert_not_called()

    @patch('greenlang.utils.net.requests.post')
    def test_http_post_success(self, mock_post):
        """Test successful HTTP POST request."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        test_data = {"key": "value"}

        with patch('greenlang.utils.net.policy_allow'):
            response = http_post("https://example.com/api", data=test_data, tag="test")

            assert response == mock_response
            mock_post.assert_called_once_with("https://example.com/api", data=test_data, timeout=30)

    @patch('greenlang.utils.net.requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        # Mock response with content
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = Path(temp_dir) / "test_file.txt"

            with patch('greenlang.utils.net.policy_allow'):
                result_path = download_file("https://example.com/file.txt", dest_path, tag="test")

                assert result_path == dest_path
                assert dest_path.exists()

                # Check content
                with open(dest_path, 'rb') as f:
                    content = f.read()
                assert content == b"chunk1chunk2"

    def test_add_allowed_domain(self):
        """Test adding domain to allowlist."""
        reset_network_policy()  # Reset to clean state

        add_allowed_domain("new-domain.com")

        from greenlang.utils.net import _network_policy
        assert "new-domain.com" in _network_policy.allowed_domains

    def test_add_blocked_domain(self):
        """Test adding domain to blocklist."""
        reset_network_policy()  # Reset to clean state

        add_blocked_domain("bad-domain.com")

        from greenlang.utils.net import _network_policy
        assert "bad-domain.com" in _network_policy.blocked_domains

    def test_get_network_audit_log(self):
        """Test getting network audit log."""
        reset_network_policy()  # Reset to clean state

        # Make some network checks to generate log entries
        from greenlang.utils.net import _network_policy
        _network_policy.check_url("https://greenlang.io/test", "test")

        audit_log = get_network_audit_log()
        assert len(audit_log) >= 1
        assert isinstance(audit_log, list)

    def test_reset_network_policy(self):
        """Test resetting network policy."""
        # Modify policy
        add_allowed_domain("test.com")
        add_blocked_domain("bad.com")

        # Reset
        reset_network_policy()

        from greenlang.utils.net import _network_policy
        # Should not have custom domains
        assert "test.com" not in _network_policy.allowed_domains
        assert "bad.com" not in _network_policy.blocked_domains

        # Should have empty defaults (secure by default)
        assert len(_network_policy.allowed_domains) == 0


class TestPerformanceTracker:
    """Test performance tracking utilities."""

    def test_performance_tracker_creation(self):
        """Test creating performance tracker."""
        tracker = PerformanceTracker("test_operation")

        assert tracker.agent_id == "test_operation"
        assert tracker.metrics == []
        assert tracker.active_metrics == {}

    def test_performance_tracker_timing(self):
        """Test performance tracker timing."""
        tracker = PerformanceTracker("test_timing")

        # Start tracking
        metric = tracker.start_tracking("test_phase")

        # Simulate some work
        import time
        time.sleep(0.01)

        # Stop tracking
        result_metric = tracker.stop_tracking("test_phase")

        assert result_metric is not None
        assert result_metric.duration > 0
        assert "test_phase" in [m.name for m in tracker.metrics]

    def test_performance_tracker_multiple_phases(self):
        """Test tracking multiple phases."""
        tracker = PerformanceTracker("multi_phase_test")

        # Track multiple phases
        tracker.start_tracking("phase1")
        import time
        time.sleep(0.005)
        tracker.stop_tracking("phase1")

        tracker.start_tracking("phase2")
        time.sleep(0.005)
        tracker.stop_tracking("phase2")

        assert len(tracker.metrics) == 2
        metric_names = [m.name for m in tracker.metrics]
        assert "phase1" in metric_names
        assert "phase2" in metric_names

    def test_performance_tracker_context_manager(self):
        """Test performance tracker as context manager."""
        tracker = PerformanceTracker("context_test")

        with tracker.track("work") as metric:
            import time
            time.sleep(0.01)

        # Should have metrics after context
        assert len(tracker.metrics) > 0
        assert "work" in [m.name for m in tracker.metrics]
        assert metric.duration > 0

    def test_performance_tracker_memory_tracking(self):
        """Test memory usage tracking if available."""
        tracker = PerformanceTracker("memory_test")

        # Track an operation to ensure memory is tracked
        tracker.start_tracking("test_memory")
        tracker.stop_tracking("test_memory")

        # Should have at least one metric
        assert len(tracker.metrics) > 0
        assert tracker.metrics[0].memory_start is not None

    def test_performance_tracker_error_handling(self):
        """Test error handling in performance tracker."""
        tracker = PerformanceTracker("error_test")

        # Try to stop a phase that wasn't started
        result = tracker.stop_tracking("non_existent_phase")
        assert result is None

        # Should not crash the tracker
        assert tracker.agent_id == "error_test"