# -*- coding: utf-8 -*-
"""
Tests for AuditLogger - Security Audit Logging
"""

import json
import tempfile
from pathlib import Path

import pytest

from greenlang.security.audit_logger import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
    configure_audit_logger,
    get_audit_logger,
)


class TestAuditLogger:
    """Test AuditLogger functionality."""

    def test_create_audit_logger(self, tmp_path):
        """Test creating audit logger with custom path."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        assert logger.log_file == log_file
        assert logger.service_name == "greenlang"
        assert log_file.parent.exists()

    def test_log_auth_success(self, tmp_path):
        """Test logging successful authentication."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_auth_success(
            user_id="user123",
            username="john.doe",
            ip_address="192.168.1.100",
            session_id="sess123",
        )

        # Read and parse log
        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "auth.success"
        assert log_entry["user_id"] == "user123"
        assert log_entry["username"] == "john.doe"
        assert log_entry["ip_address"] == "192.168.1.100"
        assert log_entry["result"] == "success"

    def test_log_auth_failure(self, tmp_path):
        """Test logging failed authentication."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_auth_failure(
            username="attacker",
            ip_address="10.0.0.1",
            reason="Invalid credentials",
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "auth.failure"
        assert log_entry["severity"] == "warning"
        assert log_entry["username"] == "attacker"
        assert log_entry["reason"] == "Invalid credentials"

    def test_log_authz_allowed(self, tmp_path):
        """Test logging authorization allowed."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_authz_decision(
            user_id="user123",
            resource_type="agent",
            resource_id="fuel_agent",
            action="execute",
            allowed=True,
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "authz.allowed"
        assert log_entry["result"] == "allowed"
        assert log_entry["resource_type"] == "agent"

    def test_log_authz_denied(self, tmp_path):
        """Test logging authorization denied."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_authz_decision(
            user_id="user123",
            resource_type="agent",
            resource_id="fuel_agent",
            action="execute",
            allowed=False,
            reason="Insufficient permissions",
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "authz.denied"
        assert log_entry["severity"] == "warning"
        assert log_entry["result"] == "denied"

    def test_log_config_change(self, tmp_path):
        """Test logging configuration change."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_config_change(
            user_id="admin",
            config_key="rate_limit",
            old_value="100",
            new_value="200",
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "config.changed"
        assert log_entry["resource_id"] == "rate_limit"
        assert log_entry["details"]["old_value"] == "100"
        assert log_entry["details"]["new_value"] == "200"

    def test_log_data_access(self, tmp_path):
        """Test logging data access."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_data_access(
            user_id="user123",
            data_type="emission_factors",
            data_id="grid_factor_US",
            operation="read",
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "data.read"
        assert log_entry["resource_type"] == "emission_factors"
        assert log_entry["action"] == "read_data"

    def test_log_agent_execution_success(self, tmp_path):
        """Test logging successful agent execution."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_agent_execution(
            agent_name="FuelAgent",
            user_id="user123",
            result="success",
            execution_time_ms=1234.56,
            details={"emissions_kg_co2": 123.45},
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "agent.completed"
        assert log_entry["severity"] == "info"
        assert log_entry["resource_name"] == "FuelAgent"
        assert log_entry["details"]["execution_time_ms"] == 1234.56

    def test_log_agent_execution_failure(self, tmp_path):
        """Test logging failed agent execution."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_agent_execution(
            agent_name="FuelAgent",
            user_id="user123",
            result="failure",
            details={"error": "Invalid input"},
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "agent.failed"
        assert log_entry["severity"] == "error"

    def test_log_security_violation(self, tmp_path):
        """Test logging security violation."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        logger.log_security_violation(
            violation_type="rate_limit_exceeded",
            description="User exceeded rate limit",
            user_id="user123",
            ip_address="192.168.1.100",
            details={"requests_per_minute": 150},
        )

        log_content = log_file.read_text()
        log_entry = json.loads(log_content.strip())

        assert log_entry["event_type"] == "security.violation"
        assert log_entry["severity"] == "critical"
        assert log_entry["details"]["violation_type"] == "rate_limit_exceeded"

    def test_multiple_log_entries(self, tmp_path):
        """Test multiple log entries in JSONL format."""
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        # Log multiple events
        logger.log_auth_success(user_id="user1", username="user1")
        logger.log_auth_success(user_id="user2", username="user2")
        logger.log_auth_failure(username="hacker")

        # Read all lines
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

        # Parse each line
        entries = [json.loads(line) for line in lines]
        assert entries[0]["user_id"] == "user1"
        assert entries[1]["user_id"] == "user2"
        assert entries[2]["username"] == "hacker"

    def test_global_audit_logger(self, tmp_path):
        """Test global audit logger instance."""
        log_file = tmp_path / "audit.jsonl"

        # Configure global logger
        logger = configure_audit_logger(log_file=log_file)
        assert logger is not None

        # Get global logger (should be same instance)
        logger2 = get_audit_logger()
        assert logger2 is logger

        # Test logging
        logger2.log_auth_success(user_id="test")
        assert log_file.exists()

    def test_audit_event_model(self):
        """Test AuditEvent pydantic model."""
        event = AuditEvent(
            timestamp="2025-11-07T10:00:00Z",
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            action="authenticate",
            result="success",
            user_id="user123",
        )

        assert event.event_type == "auth.success"
        assert event.severity == "info"
        assert event.user_id == "user123"

        # Test JSON serialization
        json_str = event.model_dump_json()
        assert "auth.success" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
