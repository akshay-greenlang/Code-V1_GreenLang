# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-004: Assumptions Registry Agent

Tests cover:
    - Assumption CRUD operations
    - Version control and history
    - Scenario management
    - Change logging and audit trail
    - Validation rules
    - Dependency tracking
    - Assumption inheritance
    - Sensitivity analysis

Author: GreenLang Team
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict

# Import directly from module to avoid __init__.py import chain issues
from greenlang.agents.foundation.assumptions_registry import (
    AssumptionsRegistryAgent,
    Assumption,
    AssumptionCategory,
    AssumptionDataType,
    AssumptionMetadata,
    ChangeType,
    Scenario,
    ScenarioType,
    ValidationRule,
    ValidationSeverity,
)
from greenlang.agents.base import AgentConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry():
    """Create a fresh Assumptions Registry agent."""
    return AssumptionsRegistryAgent()


@pytest.fixture
def sample_assumption_data():
    """Sample assumption data for testing."""
    return {
        "assumption_id": "ef.diesel.us.2024",
        "name": "US Diesel Emission Factor 2024",
        "description": "CO2 emission factor for diesel fuel in the United States",
        "category": "emission_factor",
        "data_type": "float",
        "unit": "kgCO2e/gallon",
        "current_value": 10.21,
        "default_value": 10.21,
        "metadata": {
            "source": "EPA",
            "source_url": "https://www.epa.gov/ghgemissions",
            "source_year": 2024,
            "methodology": "default",
            "geographic_scope": "US",
            "uncertainty_pct": 5.0,
            "tags": ["diesel", "fuel", "scope1", "transport"]
        },
        "validation_rules": [
            {
                "rule_id": "ef_positive",
                "description": "Emission factor must be positive",
                "min_value": 0.0,
                "severity": "error"
            },
            {
                "rule_id": "ef_reasonable",
                "description": "Emission factor should be in reasonable range",
                "max_value": 50.0,
                "severity": "warning"
            }
        ]
    }


@pytest.fixture
def sample_scenario_data():
    """Sample scenario data for testing."""
    return {
        "name": "High Carbon Intensity",
        "description": "Scenario with higher emission factors for conservative estimates",
        "scenario_type": "conservative",
        "overrides": {},
        "tags": ["conservative", "high-carbon"]
    }


# =============================================================================
# Assumption CRUD Tests
# =============================================================================


class TestAssumptionCRUD:
    """Tests for assumption create, read, update, delete operations."""

    def test_create_assumption_success(self, registry, sample_assumption_data):
        """Test creating a new assumption."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        assert result.success, f"Failed to create assumption: {result.error}"
        assert result.data["data"]["assumption_id"] == "ef.diesel.us.2024"
        assert result.data["data"]["version"] == 1

    def test_create_assumption_duplicate_fails(self, registry, sample_assumption_data):
        """Test that creating duplicate assumption fails."""
        # Create first
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        # Try to create duplicate
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Duplicate attempt"
        })

        assert not result.success
        assert "already exists" in result.error

    def test_get_assumption_success(self, registry, sample_assumption_data):
        """Test retrieving an assumption."""
        # Create first
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        # Get assumption
        result = registry.run({
            "operation": "get_assumption",
            "assumption_id": "ef.diesel.us.2024"
        })

        assert result.success
        assumption = result.data["assumption"]
        assert assumption["name"] == "US Diesel Emission Factor 2024"
        assert assumption["current_value"] == 10.21

    def test_get_assumption_not_found(self, registry):
        """Test getting non-existent assumption."""
        result = registry.run({
            "operation": "get_assumption",
            "assumption_id": "nonexistent"
        })

        assert not result.success
        assert "not found" in result.error

    def test_update_assumption_value(self, registry, sample_assumption_data):
        """Test updating an assumption value."""
        # Create first
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        # Update value
        result = registry.run({
            "operation": "update_assumption",
            "assumption_id": "ef.diesel.us.2024",
            "value": 10.35,
            "user_id": "test_user",
            "change_reason": "Updated based on 2024 data"
        })

        assert result.success
        assert result.data["data"]["old_value"] == 10.21
        assert result.data["data"]["new_value"] == 10.35
        assert result.data["data"]["version"] == 2

    def test_delete_assumption_success(self, registry, sample_assumption_data):
        """Test deleting an assumption."""
        # Create first
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        # Delete
        result = registry.run({
            "operation": "delete_assumption",
            "assumption_id": "ef.diesel.us.2024",
            "user_id": "test_user",
            "change_reason": "No longer needed"
        })

        assert result.success
        assert result.data["data"]["deleted"] is True

        # Verify it's gone
        get_result = registry.run({
            "operation": "get_assumption",
            "assumption_id": "ef.diesel.us.2024"
        })
        assert not get_result.success

    def test_list_assumptions(self, registry, sample_assumption_data):
        """Test listing assumptions."""
        # Create multiple assumptions
        for i in range(3):
            data = sample_assumption_data.copy()
            data["assumption_id"] = f"ef.diesel.us.{2022 + i}"
            data["name"] = f"US Diesel Emission Factor {2022 + i}"
            registry.run({
                "operation": "create_assumption",
                "assumption_data": data,
                "user_id": "test_user",
                "change_reason": "Bulk setup"
            })

        result = registry.run({
            "operation": "list_assumptions"
        })

        assert result.success
        assert result.data["data"]["count"] == 3

    def test_list_assumptions_with_filter(self, registry):
        """Test listing assumptions with category filter."""
        # Create assumptions in different categories
        registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "ef.test",
                "name": "Test EF",
                "description": "Test emission factor",
                "category": "emission_factor",
                "data_type": "float",
                "current_value": 1.0,
                "metadata": {"source": "test"}
            },
            "user_id": "test",
            "change_reason": "test"
        })

        registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "econ.test",
                "name": "Test Economic",
                "description": "Test economic assumption",
                "category": "economic",
                "data_type": "float",
                "current_value": 100.0,
                "metadata": {"source": "test"}
            },
            "user_id": "test",
            "change_reason": "test"
        })

        # Filter by emission_factor
        result = registry.run({
            "operation": "list_assumptions",
            "filters": {"category": "emission_factor"}
        })

        assert result.success
        assert result.data["data"]["count"] == 1
        assert result.data["assumptions"][0]["assumption_id"] == "ef.test"


# =============================================================================
# Version Control Tests
# =============================================================================


class TestVersionControl:
    """Tests for version control functionality."""

    def test_version_history_created(self, registry, sample_assumption_data):
        """Test that version history is maintained."""
        # Create assumption
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        # Update multiple times
        for i in range(3):
            registry.run({
                "operation": "update_assumption",
                "assumption_id": "ef.diesel.us.2024",
                "value": 10.21 + (i + 1) * 0.1,
                "user_id": "test_user",
                "change_reason": f"Update {i + 1}"
            })

        # Get assumption and check versions
        result = registry.run({
            "operation": "get_assumption",
            "assumption_id": "ef.diesel.us.2024"
        })

        assert result.success
        assumption = result.data["assumption"]
        assert len(assumption["versions"]) == 4  # Initial + 3 updates
        assert assumption["versions"][-1]["version_number"] == 4

    def test_version_provenance_hash(self, registry, sample_assumption_data):
        """Test that each version has a provenance hash."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        result = registry.run({
            "operation": "get_assumption",
            "assumption_id": "ef.diesel.us.2024"
        })

        version = result.data["assumption"]["versions"][0]
        assert version["provenance_hash"] != ""
        assert len(version["provenance_hash"]) == 64  # SHA-256

    def test_version_tracks_previous(self, registry, sample_assumption_data):
        """Test that versions track their parent."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Initial setup"
        })

        registry.run({
            "operation": "update_assumption",
            "assumption_id": "ef.diesel.us.2024",
            "value": 10.30,
            "user_id": "test_user",
            "change_reason": "Update"
        })

        result = registry.run({
            "operation": "get_assumption",
            "assumption_id": "ef.diesel.us.2024"
        })

        versions = result.data["assumption"]["versions"]
        assert versions[1]["parent_version_id"] == versions[0]["version_id"]


# =============================================================================
# Scenario Management Tests
# =============================================================================


class TestScenarioManagement:
    """Tests for scenario management functionality."""

    def test_default_scenarios_created(self, registry):
        """Test that default scenarios are created on init."""
        result = registry.run({
            "operation": "list_scenarios"
        })

        assert result.success
        scenarios = result.data["scenarios"]

        # Check for baseline, conservative, optimistic
        scenario_types = [s["scenario_type"] for s in scenarios]
        assert "baseline" in scenario_types
        assert "conservative" in scenario_types
        assert "optimistic" in scenario_types

    def test_create_custom_scenario(self, registry, sample_scenario_data):
        """Test creating a custom scenario."""
        result = registry.run({
            "operation": "create_scenario",
            "scenario_data": sample_scenario_data,
            "user_id": "test_user"
        })

        assert result.success
        assert result.data["scenario"]["name"] == "High Carbon Intensity"

    def test_scenario_with_overrides(self, registry, sample_assumption_data):
        """Test scenario with assumption overrides."""
        # Create assumption
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        # Create scenario with override
        result = registry.run({
            "operation": "create_scenario",
            "scenario_data": {
                "name": "Conservative EF",
                "description": "Higher emission factors",
                "scenario_type": "conservative",
                "overrides": {
                    "ef.diesel.us.2024": 11.50  # Higher than baseline
                }
            },
            "user_id": "test_user"
        })

        assert result.success
        scenario_id = result.data["scenario"]["scenario_id"]

        # Get value with scenario
        value_result = registry.run({
            "operation": "get_value",
            "assumption_id": "ef.diesel.us.2024",
            "scenario_id": scenario_id,
            "user_id": "test_user"
        })

        assert value_result.success
        assert value_result.data["data"]["value"] == 11.50
        assert "scenario:Conservative EF" in value_result.data["data"]["value_source"]

    def test_scenario_invalid_override_fails(self, registry):
        """Test that scenario with invalid assumption reference fails."""
        result = registry.run({
            "operation": "create_scenario",
            "scenario_data": {
                "name": "Bad Scenario",
                "description": "Has invalid override",
                "scenario_type": "custom",
                "overrides": {
                    "nonexistent.assumption": 999
                }
            },
            "user_id": "test_user"
        })

        assert not result.success
        assert "unknown assumption" in result.error

    def test_delete_baseline_scenario_fails(self, registry):
        """Test that baseline scenario cannot be deleted."""
        # Get baseline scenario
        result = registry.run({
            "operation": "list_scenarios",
            "filters": {"scenario_type": "baseline"}
        })

        baseline = result.data["scenarios"][0]

        # Try to delete
        delete_result = registry.run({
            "operation": "delete_scenario",
            "scenario_id": baseline["scenario_id"],
            "user_id": "test_user"
        })

        assert not delete_result.success
        assert "Cannot delete baseline" in delete_result.error


# =============================================================================
# Change Logging Tests
# =============================================================================


class TestChangeLogging:
    """Tests for change logging and audit trail."""

    def test_create_logs_change(self, registry, sample_assumption_data):
        """Test that creating assumption logs the change."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "analyst_1",
            "change_reason": "Q1 2024 setup"
        })

        result = registry.run({
            "operation": "get_change_log",
            "filters": {"assumption_id": "ef.diesel.us.2024"}
        })

        assert result.success
        assert result.data["data"]["count"] >= 1

        log_entry = result.data["change_log"][0]
        assert log_entry["change_type"] == "create"
        assert log_entry["user_id"] == "analyst_1"
        assert log_entry["change_reason"] == "Q1 2024 setup"

    def test_update_logs_change_with_values(self, registry, sample_assumption_data):
        """Test that update logs old and new values."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        registry.run({
            "operation": "update_assumption",
            "assumption_id": "ef.diesel.us.2024",
            "value": 10.50,
            "user_id": "analyst_2",
            "change_reason": "Updated per EPA 2024 revision"
        })

        result = registry.run({
            "operation": "get_change_log",
            "filters": {"change_type": "update"}
        })

        assert result.success
        update_log = result.data["change_log"][0]
        assert update_log["old_value"] == 10.21
        assert update_log["new_value"] == 10.50
        assert update_log["user_id"] == "analyst_2"

    def test_change_log_has_provenance_hash(self, registry, sample_assumption_data):
        """Test that change log entries have provenance hashes."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        result = registry.run({
            "operation": "get_change_log"
        })

        log_entry = result.data["change_log"][0]
        assert log_entry["provenance_hash"] != ""
        assert len(log_entry["provenance_hash"]) == 64

    def test_change_log_filtering(self, registry, sample_assumption_data):
        """Test filtering change log by various criteria."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "user_a",
            "change_reason": "Setup"
        })

        registry.run({
            "operation": "update_assumption",
            "assumption_id": "ef.diesel.us.2024",
            "value": 10.30,
            "user_id": "user_b",
            "change_reason": "Update"
        })

        # Filter by user
        result = registry.run({
            "operation": "get_change_log",
            "filters": {"user_id": "user_b"}
        })

        assert result.success
        assert all(e["user_id"] == "user_b" for e in result.data["change_log"])


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for assumption validation."""

    def test_validation_min_value(self, registry):
        """Test validation rejects values below minimum."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "test.min",
                "name": "Min Test",
                "description": "Test minimum validation",
                "category": "custom",
                "data_type": "float",
                "current_value": -5.0,  # Negative value
                "metadata": {"source": "test"},
                "validation_rules": [
                    {
                        "rule_id": "positive",
                        "description": "Must be positive",
                        "min_value": 0.0,
                        "severity": "error"
                    }
                ]
            },
            "user_id": "test",
            "change_reason": "test"
        })

        assert not result.success
        assert "Validation failed" in result.error

    def test_validation_max_value(self, registry):
        """Test validation rejects values above maximum."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "test.max",
                "name": "Max Test",
                "description": "Test maximum validation",
                "category": "custom",
                "data_type": "percentage",
                "current_value": 150.0,  # Over 100%
                "metadata": {"source": "test"},
                "validation_rules": [
                    {
                        "rule_id": "percentage",
                        "description": "Must be 0-100",
                        "min_value": 0.0,
                        "max_value": 100.0,
                        "severity": "error"
                    }
                ]
            },
            "user_id": "test",
            "change_reason": "test"
        })

        assert not result.success
        assert "Validation failed" in result.error

    def test_validation_allowed_values(self, registry):
        """Test validation enforces allowed values."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "test.allowed",
                "name": "Allowed Values Test",
                "description": "Test allowed values",
                "category": "custom",
                "data_type": "string",
                "current_value": "invalid",
                "metadata": {"source": "test"},
                "validation_rules": [
                    {
                        "rule_id": "allowed",
                        "description": "Must be valid option",
                        "allowed_values": ["low", "medium", "high"],
                        "severity": "error"
                    }
                ]
            },
            "user_id": "test",
            "change_reason": "test"
        })

        assert not result.success
        assert "Validation failed" in result.error

    def test_validation_warning_allows_save(self, registry):
        """Test that warnings don't block save."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "test.warning",
                "name": "Warning Test",
                "description": "Test warning validation",
                "category": "custom",
                "data_type": "float",
                "current_value": 999.0,  # High value
                "metadata": {"source": "test"},
                "validation_rules": [
                    {
                        "rule_id": "warn_high",
                        "description": "Warn if over 100",
                        "max_value": 100.0,
                        "severity": "warning"  # Warning, not error
                    }
                ]
            },
            "user_id": "test",
            "change_reason": "test"
        })

        # Should succeed despite warning
        assert result.success

    def test_custom_validator(self, registry):
        """Test custom validator function."""
        # Register custom validator
        registry.register_custom_validator(
            "is_even",
            lambda v: isinstance(v, int) and v % 2 == 0
        )

        # Test with odd value (should fail)
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "test.custom",
                "name": "Custom Validator Test",
                "description": "Test custom validation",
                "category": "custom",
                "data_type": "integer",
                "current_value": 5,  # Odd number
                "metadata": {"source": "test"},
                "validation_rules": [
                    {
                        "rule_id": "even_check",
                        "description": "Must be even",
                        "custom_validator": "is_even",
                        "severity": "error"
                    }
                ]
            },
            "user_id": "test",
            "change_reason": "test"
        })

        assert not result.success

    def test_validate_assumption_operation(self, registry, sample_assumption_data):
        """Test explicit validation operation."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        # Validate current value
        result = registry.run({
            "operation": "validate_assumption",
            "assumption_id": "ef.diesel.us.2024"
        })

        assert result.success
        assert result.data["validation_result"]["is_valid"] is True

        # Validate specific value
        result_invalid = registry.run({
            "operation": "validate_assumption",
            "assumption_id": "ef.diesel.us.2024",
            "value": -10.0  # Negative, should fail
        })

        assert result_invalid.success  # Operation succeeded
        assert result_invalid.data["validation_result"]["is_valid"] is False


# =============================================================================
# Dependency Tracking Tests
# =============================================================================


class TestDependencyTracking:
    """Tests for dependency tracking functionality."""

    def test_dependency_graph_created(self, registry, sample_assumption_data):
        """Test that dependency graph is created."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        result = registry.run({
            "operation": "get_dependencies",
            "assumption_id": "ef.diesel.us.2024"
        })

        assert result.success
        assert "ef.diesel.us.2024" in result.data["dependencies"]

    def test_register_calculation_dependency(self, registry, sample_assumption_data):
        """Test registering calculation dependencies."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        # Register that a calculation uses this assumption
        registry.register_calculation_dependency(
            "calc_scope1_transport",
            ["ef.diesel.us.2024"]
        )

        # Get assumption and check used_by
        result = registry.run({
            "operation": "get_assumption",
            "assumption_id": "ef.diesel.us.2024"
        })

        assert "calc_scope1_transport" in result.data["assumption"]["used_by"]

    def test_delete_fails_when_in_use(self, registry, sample_assumption_data):
        """Test that delete fails when assumption is used."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        # Register usage
        registry.register_calculation_dependency(
            "calc_scope1",
            ["ef.diesel.us.2024"]
        )

        # Try to delete
        result = registry.run({
            "operation": "delete_assumption",
            "assumption_id": "ef.diesel.us.2024",
            "user_id": "test_user",
            "change_reason": "Attempt delete"
        })

        assert not result.success
        assert "Cannot delete" in result.error

    def test_get_assumptions_for_calculation(self, registry):
        """Test getting all assumptions for a calculation."""
        # Create multiple assumptions
        for fuel in ["diesel", "gasoline", "natural_gas"]:
            registry.run({
                "operation": "create_assumption",
                "assumption_data": {
                    "assumption_id": f"ef.{fuel}.us.2024",
                    "name": f"US {fuel.title()} EF 2024",
                    "description": f"Emission factor for {fuel}",
                    "category": "emission_factor",
                    "data_type": "float",
                    "current_value": 10.0 + len(fuel),
                    "metadata": {"source": "EPA"}
                },
                "user_id": "test",
                "change_reason": "Setup"
            })

        # Register calculation dependencies
        registry.register_calculation_dependency(
            "calc_fleet_emissions",
            ["ef.diesel.us.2024", "ef.gasoline.us.2024"]
        )

        # Get assumptions for calculation
        assumptions = registry.get_assumptions_for_calculation("calc_fleet_emissions")

        assert "ef.diesel.us.2024" in assumptions
        assert "ef.gasoline.us.2024" in assumptions
        assert "ef.natural_gas.us.2024" not in assumptions


# =============================================================================
# Sensitivity Analysis Tests
# =============================================================================


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis functionality."""

    def test_sensitivity_analysis_basic(self, registry, sample_assumption_data):
        """Test basic sensitivity analysis."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        result = registry.run({
            "operation": "get_sensitivity_analysis",
            "assumption_id": "ef.diesel.us.2024"
        })

        assert result.success
        data = result.data["data"]
        assert data["baseline_value"] == 10.21
        assert "scenario_values" in data

    def test_sensitivity_analysis_with_scenarios(self, registry, sample_assumption_data):
        """Test sensitivity analysis shows scenario variations."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        # Create scenarios with overrides
        registry.run({
            "operation": "create_scenario",
            "scenario_data": {
                "name": "High EF",
                "description": "High emission factors",
                "scenario_type": "conservative",
                "overrides": {"ef.diesel.us.2024": 12.0}
            },
            "user_id": "test"
        })

        registry.run({
            "operation": "create_scenario",
            "scenario_data": {
                "name": "Low EF",
                "description": "Low emission factors",
                "scenario_type": "optimistic",
                "overrides": {"ef.diesel.us.2024": 9.0}
            },
            "user_id": "test"
        })

        result = registry.run({
            "operation": "get_sensitivity_analysis",
            "assumption_id": "ef.diesel.us.2024"
        })

        data = result.data["data"]
        assert data["min_value"] == 9.0
        assert data["max_value"] == 12.0
        assert data["range"] == 3.0


# =============================================================================
# Export/Import Tests
# =============================================================================


class TestExportImport:
    """Tests for export and import functionality."""

    def test_export_assumptions(self, registry, sample_assumption_data):
        """Test exporting assumptions."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        result = registry.run({
            "operation": "export_assumptions",
            "user_id": "export_user"
        })

        assert result.success
        export_data = result.data["data"]
        assert "assumptions" in export_data
        assert "scenarios" in export_data
        assert "change_log" in export_data
        assert "export_hash" in export_data
        assert len(export_data["assumptions"]) >= 1

    def test_import_assumptions(self, registry, sample_assumption_data):
        """Test importing assumptions."""
        # Create in first registry
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        # Export
        export_result = registry.run({
            "operation": "export_assumptions",
            "user_id": "export_user"
        })

        # Create new registry and import
        new_registry = AssumptionsRegistryAgent()

        import_result = new_registry.run({
            "operation": "import_assumptions",
            "assumption_data": export_result.data["data"],
            "user_id": "import_user",
            "change_reason": "Imported from export"
        })

        assert import_result.success
        assert import_result.data["data"]["imported_count"] >= 1


# =============================================================================
# Convenience Method Tests
# =============================================================================


class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_get_assumption_value_method(self, registry, sample_assumption_data):
        """Test get_assumption_value convenience method."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        value = registry.get_assumption_value("ef.diesel.us.2024")
        assert value == 10.21

    def test_set_assumption_value_method(self, registry, sample_assumption_data):
        """Test set_assumption_value convenience method."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        success = registry.set_assumption_value(
            "ef.diesel.us.2024",
            10.50,
            "analyst",
            "Updated value"
        )

        assert success

        # Verify update
        value = registry.get_assumption_value("ef.diesel.us.2024")
        assert value == 10.50

    def test_get_stats(self, registry, sample_assumption_data):
        """Test get_stats method."""
        registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        stats = registry.get_stats()

        assert stats["total_assumptions"] >= 1
        assert stats["total_scenarios"] >= 3  # Default scenarios
        assert "assumptions_by_category" in stats
        assert "scenarios_by_type" in stats


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_required_fields(self, registry):
        """Test that missing required fields are handled."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "test",
                # Missing name, current_value, metadata
            },
            "user_id": "test",
            "change_reason": "test"
        })

        assert not result.success

    def test_invalid_operation(self, registry):
        """Test handling of invalid operation."""
        result = registry.run({
            "operation": "nonexistent_operation"
        })

        assert not result.success
        assert "Invalid operation" in result.error

    def test_invalid_data_type(self, registry):
        """Test handling of invalid data type."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "test.invalid_type",
                "name": "Invalid Type Test",
                "description": "Test",
                "category": "custom",
                "data_type": "nonexistent_type",  # Invalid
                "current_value": 1.0,
                "metadata": {"source": "test"}
            },
            "user_id": "test",
            "change_reason": "test"
        })

        assert not result.success

    def test_assumption_id_format_validation(self, registry):
        """Test assumption ID format validation."""
        # Too short
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "ab",  # Too short
                "name": "Test",
                "description": "Test",
                "category": "custom",
                "data_type": "float",
                "current_value": 1.0,
                "metadata": {"source": "test"}
            },
            "user_id": "test",
            "change_reason": "test"
        })

        assert not result.success

    def test_processing_time_tracked(self, registry, sample_assumption_data):
        """Test that processing time is tracked."""
        result = registry.run({
            "operation": "create_assumption",
            "assumption_data": sample_assumption_data,
            "user_id": "test_user",
            "change_reason": "Setup"
        })

        assert result.success
        # Processing time is tracked in the output model, may be 0 if very fast
        assert "processing_time_ms" in result.data
        # On fast systems this can be 0.0, so just verify the field exists
        assert isinstance(result.data["processing_time_ms"], (int, float))


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_assumption_lifecycle(self, registry):
        """Test complete assumption lifecycle."""
        # 1. Create assumption
        create_result = registry.run({
            "operation": "create_assumption",
            "assumption_data": {
                "assumption_id": "ef.lifecycle.test",
                "name": "Lifecycle Test EF",
                "description": "Testing full lifecycle",
                "category": "emission_factor",
                "data_type": "float",
                "unit": "kgCO2e/unit",
                "current_value": 1.0,
                "metadata": {
                    "source": "Test",
                    "source_year": 2024
                },
                "validation_rules": [
                    {
                        "rule_id": "positive",
                        "description": "Must be positive",
                        "min_value": 0.0,
                        "severity": "error"
                    }
                ]
            },
            "user_id": "analyst",
            "change_reason": "Initial creation"
        })
        assert create_result.success

        # 2. Update value multiple times
        for i in range(3):
            registry.run({
                "operation": "update_assumption",
                "assumption_id": "ef.lifecycle.test",
                "value": 1.0 + (i + 1) * 0.5,
                "user_id": "analyst",
                "change_reason": f"Update iteration {i + 1}"
            })

        # 3. Create scenario with override
        scenario_result = registry.run({
            "operation": "create_scenario",
            "scenario_data": {
                "name": "High Scenario",
                "description": "Higher values",
                "scenario_type": "conservative",
                "overrides": {"ef.lifecycle.test": 5.0}
            },
            "user_id": "analyst"
        })
        assert scenario_result.success
        scenario_id = scenario_result.data["scenario"]["scenario_id"]

        # 4. Get value with and without scenario
        baseline_result = registry.run({
            "operation": "get_value",
            "assumption_id": "ef.lifecycle.test",
            "user_id": "analyst"
        })
        assert baseline_result.data["data"]["value"] == 2.5  # Last update

        scenario_value_result = registry.run({
            "operation": "get_value",
            "assumption_id": "ef.lifecycle.test",
            "scenario_id": scenario_id,
            "user_id": "analyst"
        })
        assert scenario_value_result.data["data"]["value"] == 5.0

        # 5. Check change log
        log_result = registry.run({
            "operation": "get_change_log",
            "filters": {"assumption_id": "ef.lifecycle.test"}
        })
        assert log_result.data["data"]["count"] >= 4  # Create + 3 updates

        # 6. Get sensitivity analysis
        sensitivity_result = registry.run({
            "operation": "get_sensitivity_analysis",
            "assumption_id": "ef.lifecycle.test"
        })
        assert sensitivity_result.success

        # 7. Validate current state
        validate_result = registry.run({
            "operation": "validate_assumption",
            "assumption_id": "ef.lifecycle.test"
        })
        assert validate_result.data["validation_result"]["is_valid"]

        # 8. Get stats
        stats = registry.get_stats()
        assert stats["total_assumptions"] >= 1

    def test_multi_assumption_scenario(self, registry):
        """Test scenario with multiple assumption overrides."""
        # Create multiple related assumptions
        fuels = [
            ("diesel", 10.21),
            ("gasoline", 8.78),
            ("natural_gas", 5.30),
        ]

        for fuel, value in fuels:
            registry.run({
                "operation": "create_assumption",
                "assumption_data": {
                    "assumption_id": f"ef.{fuel}.us.2024",
                    "name": f"US {fuel.title()} EF 2024",
                    "description": f"Emission factor for {fuel}",
                    "category": "emission_factor",
                    "data_type": "float",
                    "unit": "kgCO2e/gallon",
                    "current_value": value,
                    "metadata": {"source": "EPA"}
                },
                "user_id": "test",
                "change_reason": "Setup"
            })

        # Create scenario with all overrides
        scenario_result = registry.run({
            "operation": "create_scenario",
            "scenario_data": {
                "name": "2030 Projection",
                "description": "Projected values for 2030",
                "scenario_type": "custom",
                "overrides": {
                    "ef.diesel.us.2024": 8.0,
                    "ef.gasoline.us.2024": 7.0,
                    "ef.natural_gas.us.2024": 4.0,
                },
                "tags": ["2030", "projection"]
            },
            "user_id": "test"
        })

        scenario_id = scenario_result.data["scenario"]["scenario_id"]

        # Verify all overrides work
        for fuel, _ in fuels:
            result = registry.run({
                "operation": "get_value",
                "assumption_id": f"ef.{fuel}.us.2024",
                "scenario_id": scenario_id,
                "user_id": "test"
            })
            assert result.success
            # All 2030 values should be lower
            assert result.data["data"]["value"] < 9.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
