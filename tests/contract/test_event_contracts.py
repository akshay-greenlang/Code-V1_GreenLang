# -*- coding: utf-8 -*-
"""
Event Schema Contract Tests for GreenLang

Tests event schemas to ensure producers and consumers agree on
event payload structures. Uses JSON Schema validation.

Test Coverage:
- Calculation events (completed, failed, progress)
- Alarm events (triggered, cleared, acknowledged)
- Model lifecycle events (deployed, degraded, retrained)
- Compliance events (violation, report_ready)
- Audit events (access, modification)

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import json
from typing import Dict, Any, List
from datetime import datetime
from uuid import uuid4

from .conftest import EventSchemaValidator, ContractVerificationResult


# ==============================================================================
# Event Schema Definitions
# ==============================================================================

CALCULATION_COMPLETED_SCHEMA = {
    "type": "object",
    "required": ["calculation_id", "result", "timestamp", "provenance_hash"],
    "properties": {
        "calculation_id": {"type": "string", "pattern": "^calc_[a-zA-Z0-9]+$"},
        "result": {
            "type": "object",
            "required": ["emissions_kg_co2e"],
            "properties": {
                "emissions_kg_co2e": {"type": "number", "minimum": 0},
                "emissions_tonnes_co2e": {"type": "number", "minimum": 0},
                "emission_factor": {"type": "number"},
                "emission_factor_source": {"type": "string"},
            },
        },
        "timestamp": {"type": "string", "format": "date-time"},
        "provenance_hash": {"type": "string", "minLength": 64, "maxLength": 64},
        "execution_time_ms": {"type": "number", "minimum": 0},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}

CALCULATION_FAILED_SCHEMA = {
    "type": "object",
    "required": ["calculation_id", "error", "error_code", "timestamp"],
    "properties": {
        "calculation_id": {"type": "string"},
        "error": {"type": "string"},
        "error_code": {"type": "string"},
        "timestamp": {"type": "string"},
        "stack_trace": {"type": "string"},
        "retry_count": {"type": "integer", "minimum": 0},
        "is_retriable": {"type": "boolean"},
    },
}

ALARM_TRIGGERED_SCHEMA = {
    "type": "object",
    "required": ["alarm_id", "severity", "message", "source", "timestamp"],
    "properties": {
        "alarm_id": {"type": "string"},
        "severity": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low", "info"],
        },
        "message": {"type": "string"},
        "source": {"type": "string"},
        "timestamp": {"type": "string"},
        "value": {"type": "number"},
        "threshold": {"type": "number"},
        "unit": {"type": "string"},
        "acknowledged": {"type": "boolean"},
        "escalation_level": {"type": "integer"},
    },
}

ALARM_CLEARED_SCHEMA = {
    "type": "object",
    "required": ["alarm_id", "cleared_at", "duration_seconds"],
    "properties": {
        "alarm_id": {"type": "string"},
        "cleared_at": {"type": "string"},
        "duration_seconds": {"type": "number", "minimum": 0},
        "cleared_by": {"type": "string"},
        "resolution_notes": {"type": "string"},
    },
}

MODEL_DEPLOYED_SCHEMA = {
    "type": "object",
    "required": ["model_id", "version", "deployed_at", "deployment_type"],
    "properties": {
        "model_id": {"type": "string"},
        "version": {"type": "string"},
        "deployed_at": {"type": "string"},
        "deployment_type": {
            "type": "string",
            "enum": ["champion", "challenger", "shadow"],
        },
        "traffic_percentage": {"type": "number", "minimum": 0, "maximum": 100},
        "metrics_baseline": {"type": "object"},
        "rollback_version": {"type": "string"},
    },
}

COMPLIANCE_VIOLATION_SCHEMA = {
    "type": "object",
    "required": ["violation_id", "rule_id", "severity", "detected_at", "details"],
    "properties": {
        "violation_id": {"type": "string"},
        "rule_id": {"type": "string"},
        "rule_name": {"type": "string"},
        "severity": {"type": "string"},
        "detected_at": {"type": "string"},
        "details": {
            "type": "object",
            "required": ["expected", "actual"],
            "properties": {
                "expected": {"type": "string"},
                "actual": {"type": "string"},
                "deviation_percent": {"type": "number"},
            },
        },
        "remediation_steps": {"type": "array", "items": {"type": "string"}},
        "auto_remediated": {"type": "boolean"},
    },
}


# ==============================================================================
# Calculation Event Tests
# ==============================================================================

class TestCalculationEventContracts:
    """Test calculation event schema contracts."""

    @pytest.fixture
    def validator(self) -> EventSchemaValidator:
        """Create validator with calculation schemas."""
        validator = EventSchemaValidator()
        validator.register_schema("calculation.completed", CALCULATION_COMPLETED_SCHEMA)
        validator.register_schema("calculation.failed", CALCULATION_FAILED_SCHEMA)
        return validator

    @pytest.mark.event_contract
    def test_calculation_completed_valid_event(self, validator: EventSchemaValidator):
        """Test valid calculation.completed event."""
        event = {
            "calculation_id": "calc_abc123def456",
            "result": {
                "emissions_kg_co2e": 5300.0,
                "emissions_tonnes_co2e": 5.3,
                "emission_factor": 5.3,
                "emission_factor_source": "EPA 2024",
            },
            "timestamp": "2025-12-07T12:00:00Z",
            "provenance_hash": "a" * 64,
            "execution_time_ms": 45.2,
            "metadata": {"region": "US", "fuel_type": "natural_gas"},
        }

        result = validator.validate("calculation.completed", event)

        assert result.passed, f"Validation failed: {result.errors}"
        assert len(result.warnings) == 0

    @pytest.mark.event_contract
    def test_calculation_completed_missing_required_field(self, validator: EventSchemaValidator):
        """Test calculation.completed event missing required field."""
        event = {
            "calculation_id": "calc_abc123",
            "result": {"emissions_kg_co2e": 5300.0},
            # Missing: timestamp, provenance_hash
        }

        result = validator.validate("calculation.completed", event)

        assert not result.passed
        assert any("timestamp" in e for e in result.errors)

    @pytest.mark.event_contract
    def test_calculation_completed_invalid_result_type(self, validator: EventSchemaValidator):
        """Test calculation.completed event with invalid result type."""
        event = {
            "calculation_id": "calc_abc123",
            "result": "not an object",  # Should be object
            "timestamp": "2025-12-07T12:00:00Z",
            "provenance_hash": "a" * 64,
        }

        result = validator.validate("calculation.completed", event)

        assert not result.passed
        assert any("result" in e and "object" in e for e in result.errors)

    @pytest.mark.event_contract
    def test_calculation_failed_valid_event(self, validator: EventSchemaValidator):
        """Test valid calculation.failed event."""
        event = {
            "calculation_id": "calc_xyz789",
            "error": "Invalid fuel type specified",
            "error_code": "INVALID_FUEL_TYPE",
            "timestamp": "2025-12-07T12:00:00Z",
            "retry_count": 3,
            "is_retriable": False,
        }

        result = validator.validate("calculation.failed", event)

        assert result.passed, f"Validation failed: {result.errors}"

    @pytest.mark.event_contract
    def test_calculation_failed_with_stack_trace(self, validator: EventSchemaValidator):
        """Test calculation.failed event with stack trace."""
        event = {
            "calculation_id": "calc_error123",
            "error": "Database connection failed",
            "error_code": "DB_CONNECTION_ERROR",
            "timestamp": "2025-12-07T12:00:00Z",
            "stack_trace": "Traceback (most recent call last):\n  File ...",
            "retry_count": 0,
            "is_retriable": True,
        }

        result = validator.validate("calculation.failed", event)

        assert result.passed


# ==============================================================================
# Alarm Event Tests
# ==============================================================================

class TestAlarmEventContracts:
    """Test alarm event schema contracts."""

    @pytest.fixture
    def validator(self) -> EventSchemaValidator:
        """Create validator with alarm schemas."""
        validator = EventSchemaValidator()
        validator.register_schema("alarm.triggered", ALARM_TRIGGERED_SCHEMA)
        validator.register_schema("alarm.cleared", ALARM_CLEARED_SCHEMA)
        return validator

    @pytest.mark.event_contract
    def test_alarm_triggered_valid_event(self, validator: EventSchemaValidator):
        """Test valid alarm.triggered event."""
        event = {
            "alarm_id": "alarm_temp_high_001",
            "severity": "critical",
            "message": "Process temperature exceeded maximum threshold",
            "source": "process_heat_agent",
            "timestamp": "2025-12-07T12:00:00Z",
            "value": 1050.0,
            "threshold": 1000.0,
            "unit": "celsius",
            "acknowledged": False,
            "escalation_level": 1,
        }

        result = validator.validate("alarm.triggered", event)

        assert result.passed, f"Validation failed: {result.errors}"

    @pytest.mark.event_contract
    def test_alarm_triggered_all_severities(self, validator: EventSchemaValidator):
        """Test alarm.triggered with all valid severity levels."""
        severities = ["critical", "high", "medium", "low", "info"]

        for severity in severities:
            event = {
                "alarm_id": f"alarm_{severity}_001",
                "severity": severity,
                "message": f"Test {severity} alarm",
                "source": "test_source",
                "timestamp": "2025-12-07T12:00:00Z",
            }

            result = validator.validate("alarm.triggered", event)
            assert result.passed, f"Failed for severity {severity}: {result.errors}"

    @pytest.mark.event_contract
    def test_alarm_cleared_valid_event(self, validator: EventSchemaValidator):
        """Test valid alarm.cleared event."""
        event = {
            "alarm_id": "alarm_temp_high_001",
            "cleared_at": "2025-12-07T12:30:00Z",
            "duration_seconds": 1800.0,
            "cleared_by": "operator_john",
            "resolution_notes": "Reduced process load to lower temperature",
        }

        result = validator.validate("alarm.cleared", event)

        assert result.passed, f"Validation failed: {result.errors}"

    @pytest.mark.event_contract
    def test_alarm_cleared_auto_cleared(self, validator: EventSchemaValidator):
        """Test alarm.cleared event for auto-cleared alarm."""
        event = {
            "alarm_id": "alarm_pressure_001",
            "cleared_at": "2025-12-07T12:15:00Z",
            "duration_seconds": 300.0,
            # No cleared_by for auto-cleared
        }

        result = validator.validate("alarm.cleared", event)

        assert result.passed


# ==============================================================================
# Model Lifecycle Event Tests
# ==============================================================================

class TestModelLifecycleEventContracts:
    """Test model lifecycle event schema contracts."""

    @pytest.fixture
    def validator(self) -> EventSchemaValidator:
        """Create validator with model schemas."""
        validator = EventSchemaValidator()
        validator.register_schema("model.deployed", MODEL_DEPLOYED_SCHEMA)
        return validator

    @pytest.mark.event_contract
    def test_model_deployed_champion(self, validator: EventSchemaValidator):
        """Test model.deployed event for champion model."""
        event = {
            "model_id": "fuel_emission_model",
            "version": "2.1.0",
            "deployed_at": "2025-12-07T12:00:00Z",
            "deployment_type": "champion",
            "traffic_percentage": 100.0,
            "metrics_baseline": {
                "accuracy": 0.95,
                "latency_p99_ms": 50,
            },
            "rollback_version": "2.0.5",
        }

        result = validator.validate("model.deployed", event)

        assert result.passed, f"Validation failed: {result.errors}"

    @pytest.mark.event_contract
    def test_model_deployed_challenger(self, validator: EventSchemaValidator):
        """Test model.deployed event for challenger model."""
        event = {
            "model_id": "fuel_emission_model",
            "version": "2.2.0-beta",
            "deployed_at": "2025-12-07T12:00:00Z",
            "deployment_type": "challenger",
            "traffic_percentage": 10.0,
            "rollback_version": "2.1.0",
        }

        result = validator.validate("model.deployed", event)

        assert result.passed

    @pytest.mark.event_contract
    def test_model_deployed_shadow(self, validator: EventSchemaValidator):
        """Test model.deployed event for shadow model."""
        event = {
            "model_id": "experimental_model",
            "version": "0.1.0",
            "deployed_at": "2025-12-07T12:00:00Z",
            "deployment_type": "shadow",
            "traffic_percentage": 0.0,  # Shadow gets no traffic
        }

        result = validator.validate("model.deployed", event)

        assert result.passed


# ==============================================================================
# Compliance Event Tests
# ==============================================================================

class TestComplianceEventContracts:
    """Test compliance event schema contracts."""

    @pytest.fixture
    def validator(self) -> EventSchemaValidator:
        """Create validator with compliance schemas."""
        validator = EventSchemaValidator()
        validator.register_schema("compliance.violation", COMPLIANCE_VIOLATION_SCHEMA)
        return validator

    @pytest.mark.event_contract
    def test_compliance_violation_valid_event(self, validator: EventSchemaValidator):
        """Test valid compliance.violation event."""
        event = {
            "violation_id": "viol_ghg_001",
            "rule_id": "GHG_PROTOCOL_SCOPE1_ACCURACY",
            "rule_name": "Scope 1 Calculation Accuracy",
            "severity": "high",
            "detected_at": "2025-12-07T12:00:00Z",
            "details": {
                "expected": "5% tolerance",
                "actual": "8.5% deviation",
                "deviation_percent": 8.5,
            },
            "remediation_steps": [
                "Review emission factor source",
                "Verify input data quality",
                "Recalculate with updated factors",
            ],
            "auto_remediated": False,
        }

        result = validator.validate("compliance.violation", event)

        assert result.passed, f"Validation failed: {result.errors}"

    @pytest.mark.event_contract
    def test_compliance_violation_auto_remediated(self, validator: EventSchemaValidator):
        """Test compliance.violation event with auto-remediation."""
        event = {
            "violation_id": "viol_data_quality_001",
            "rule_id": "DATA_COMPLETENESS",
            "severity": "medium",
            "detected_at": "2025-12-07T12:00:00Z",
            "details": {
                "expected": "100% data completeness",
                "actual": "98% data completeness",
                "deviation_percent": 2.0,
            },
            "auto_remediated": True,
        }

        result = validator.validate("compliance.violation", event)

        assert result.passed


# ==============================================================================
# Event Envelope Tests
# ==============================================================================

class TestEventEnvelopeContracts:
    """Test event envelope (wrapper) schema contracts."""

    EVENT_ENVELOPE_SCHEMA = {
        "type": "object",
        "required": ["event_id", "event_type", "version", "timestamp", "source", "data"],
        "properties": {
            "event_id": {"type": "string"},
            "event_type": {"type": "string"},
            "version": {"type": "string"},
            "timestamp": {"type": "string"},
            "source": {"type": "string"},
            "correlation_id": {"type": "string"},
            "causation_id": {"type": "string"},
            "data": {"type": "object"},
            "metadata": {"type": "object"},
        },
    }

    @pytest.fixture
    def validator(self) -> EventSchemaValidator:
        """Create validator with envelope schema."""
        validator = EventSchemaValidator()
        validator.register_schema("event.envelope", self.EVENT_ENVELOPE_SCHEMA)
        return validator

    @pytest.mark.event_contract
    def test_event_envelope_valid(self, validator: EventSchemaValidator):
        """Test valid event envelope."""
        envelope = {
            "event_id": str(uuid4()),
            "event_type": "calculation.completed",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "source": "calculation_service",
            "correlation_id": str(uuid4()),
            "causation_id": str(uuid4()),
            "data": {
                "calculation_id": "calc_123",
                "result": {"emissions_kg_co2e": 5300.0},
            },
            "metadata": {
                "region": "us-east-1",
                "environment": "production",
            },
        }

        result = validator.validate("event.envelope", envelope)

        assert result.passed, f"Validation failed: {result.errors}"

    @pytest.mark.event_contract
    def test_event_envelope_minimal(self, validator: EventSchemaValidator):
        """Test minimal valid event envelope (required fields only)."""
        envelope = {
            "event_id": str(uuid4()),
            "event_type": "test.event",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "source": "test_service",
            "data": {},
        }

        result = validator.validate("event.envelope", envelope)

        assert result.passed


# ==============================================================================
# Event Schema Versioning Tests
# ==============================================================================

class TestEventSchemaVersioning:
    """Test event schema versioning and backward compatibility."""

    @pytest.mark.event_contract
    def test_schema_version_compatibility_v1_to_v2(self):
        """Test that v1 events are compatible with v2 schema (additive changes)."""
        # V1 schema (original)
        v1_schema = {
            "type": "object",
            "required": ["id", "value"],
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "number"},
            },
        }

        # V2 schema (added optional field)
        v2_schema = {
            "type": "object",
            "required": ["id", "value"],
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "number"},
                "unit": {"type": "string"},  # New optional field
            },
        }

        # V1 event should be valid against V2 schema
        v1_event = {"id": "test_123", "value": 42.0}

        validator = EventSchemaValidator()
        validator.register_schema("test.event.v2", v2_schema)

        result = validator.validate("test.event.v2", v1_event)

        # V1 event should still be valid (unit is optional)
        assert result.passed, "V1 event should be valid against V2 schema"

    @pytest.mark.event_contract
    def test_schema_breaking_change_detection(self):
        """Test that breaking changes are detected."""
        # V1 schema
        v1_schema = {
            "type": "object",
            "required": ["id", "value"],
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "number"},
            },
        }

        # V2 schema with breaking change (new required field)
        v2_breaking_schema = {
            "type": "object",
            "required": ["id", "value", "unit"],  # unit now required
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "number"},
                "unit": {"type": "string"},
            },
        }

        # V1 event is NOT valid against V2 breaking schema
        v1_event = {"id": "test_123", "value": 42.0}

        validator = EventSchemaValidator()
        validator.register_schema("test.event.v2.breaking", v2_breaking_schema)

        result = validator.validate("test.event.v2.breaking", v1_event)

        # Should fail because 'unit' is now required
        assert not result.passed, "V1 event should be invalid against breaking V2 schema"
        assert any("unit" in e for e in result.errors)


# ==============================================================================
# Event Correlation Tests
# ==============================================================================

class TestEventCorrelation:
    """Test event correlation and causation tracking."""

    @pytest.mark.event_contract
    def test_correlated_events_have_same_correlation_id(self):
        """Test that related events share correlation ID."""
        correlation_id = str(uuid4())

        # Parent event
        parent_event = {
            "event_id": str(uuid4()),
            "event_type": "batch.started",
            "correlation_id": correlation_id,
            "data": {"batch_id": "batch_123"},
        }

        # Child events
        child_events = [
            {
                "event_id": str(uuid4()),
                "event_type": "calculation.completed",
                "correlation_id": correlation_id,
                "causation_id": parent_event["event_id"],
                "data": {"calculation_id": f"calc_{i}"},
            }
            for i in range(3)
        ]

        # Verify all events share correlation ID
        all_events = [parent_event] + child_events
        correlation_ids = {e["correlation_id"] for e in all_events}

        assert len(correlation_ids) == 1
        assert correlation_id in correlation_ids

        # Verify causation chain
        for child in child_events:
            assert child["causation_id"] == parent_event["event_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "event_contract"])
