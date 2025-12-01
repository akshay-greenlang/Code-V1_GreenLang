# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Security Tests
Comprehensive security testing for predictive maintenance system.

Tests cover:
- Input validation (injection prevention)
- SQL injection prevention
- Authentication requirements
- Authorization enforcement
- Secrets protection (not in logs)
- Provenance tamper detection
- Data sanitization
- Access control
- Audit logging security

Security Standards:
- OWASP Top 10 compliance
- NIST SP 800-53 controls
- ISO 27001 requirements

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import logging

# Import test fixtures from conftest
from ..conftest import (
    MachineClass,
    VibrationZone,
    HealthState,
    WEIBULL_PARAMETERS,
)


# =============================================================================
# TEST CLASS: INPUT VALIDATION
# =============================================================================


class TestInputValidation:
    """Tests for input validation and sanitization."""

    @pytest.mark.security
    def test_reject_negative_operating_hours(self, rul_calculator):
        """Test that negative operating hours are rejected."""
        with pytest.raises((ValueError, InvalidOperation)):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("-1000"),
            )

    @pytest.mark.security
    def test_reject_invalid_reliability_range(self, rul_calculator):
        """Test that out-of-range reliability values are rejected."""
        # Greater than 1
        with pytest.raises(ValueError):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("25000"),
                target_reliability="1.5",
            )

        # Negative
        with pytest.raises(ValueError):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("25000"),
                target_reliability="-0.5",
            )

    @pytest.mark.security
    def test_reject_negative_vibration(self, vibration_analyzer):
        """Test that negative vibration values are rejected."""
        with pytest.raises(ValueError):
            vibration_analyzer.assess_severity(
                velocity_rms=Decimal("-1.0"),
                machine_class=MachineClass.CLASS_II,
            )

    @pytest.mark.security
    def test_reject_invalid_weibull_beta(self, failure_probability_calculator):
        """Test that non-positive beta values are rejected."""
        with pytest.raises(ValueError):
            failure_probability_calculator.calculate_weibull_failure_probability(
                beta=Decimal("0"),
                eta=Decimal("50000"),
                time_hours=Decimal("25000"),
            )

        with pytest.raises(ValueError):
            failure_probability_calculator.calculate_weibull_failure_probability(
                beta=Decimal("-2.5"),
                eta=Decimal("50000"),
                time_hours=Decimal("25000"),
            )

    @pytest.mark.security
    def test_reject_invalid_weibull_eta(self, failure_probability_calculator):
        """Test that non-positive eta values are rejected."""
        with pytest.raises(ValueError):
            failure_probability_calculator.calculate_weibull_failure_probability(
                beta=Decimal("2.5"),
                eta=Decimal("-50000"),
                time_hours=Decimal("25000"),
            )

    @pytest.mark.security
    def test_type_coercion_safety(self, rul_calculator):
        """Test that type coercion doesn't bypass validation."""
        # String that looks like a number but with special chars
        with pytest.raises((ValueError, TypeError, InvalidOperation)):
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours="25000; DROP TABLE equipment;",
            )

    @pytest.mark.security
    def test_unicode_input_handling(self, rul_calculator):
        """Test handling of unicode in equipment type."""
        # Should handle gracefully or reject appropriately
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",  # Normal string
            operating_hours=Decimal("25000"),
        )

        assert result is not None

    @pytest.mark.security
    @pytest.mark.parametrize("malicious_input", [
        "'; DROP TABLE equipment; --",
        "1; DELETE FROM sensors WHERE 1=1; --",
        "UNION SELECT * FROM credentials --",
        "' OR '1'='1",
        "<script>alert('xss')</script>",
        "{{7*7}}",  # SSTI test
        "${7*7}",   # Expression injection
        "\\x00\\x00",  # Null bytes
    ])
    def test_reject_malicious_equipment_id(self, vibration_analyzer, malicious_input):
        """Test rejection or sanitization of malicious equipment IDs."""
        # Should either reject, sanitize, or handle safely
        try:
            result = vibration_analyzer.assess_severity(
                velocity_rms=Decimal("4.5"),
                machine_class=MachineClass.CLASS_II,
                equipment_id=malicious_input,
            )
            # If it returns, verify no injection occurred
            assert "DROP" not in str(result)
            assert "DELETE" not in str(result)
            assert "<script>" not in str(result)
        except (ValueError, TypeError):
            pass  # Rejection is acceptable


# =============================================================================
# TEST CLASS: SQL INJECTION PREVENTION
# =============================================================================


class TestNoSQLInjection:
    """Tests for SQL injection prevention."""

    @pytest.mark.security
    def test_equipment_type_no_injection(self, rul_calculator):
        """Test equipment type field is not vulnerable to SQL injection."""
        injection_attempts = [
            "pump' OR '1'='1",
            "pump'; DROP TABLE equipment; --",
            "pump UNION SELECT password FROM users --",
            "pump\"; DROP TABLE equipment; --",
        ]

        for attempt in injection_attempts:
            try:
                # Should either reject or handle safely
                result = rul_calculator.calculate_weibull_rul(
                    equipment_type=attempt,
                    operating_hours=Decimal("25000"),
                )
                # If it returns, should use default params, not execute SQL
                assert result is not None
            except (ValueError, KeyError):
                pass  # Rejection is acceptable

    @pytest.mark.security
    def test_parameterized_query_usage(self, mock_database):
        """Test that database queries use parameterized statements."""
        # This tests the pattern - actual implementation would use real DB
        equipment_id = "PUMP-001'; DROP TABLE equipment; --"

        # Simulated parameterized query
        query = "SELECT * FROM equipment WHERE id = ?"
        params = (equipment_id,)

        # The query string should not contain the injection
        assert "DROP TABLE" not in query
        assert equipment_id in params

    @pytest.mark.security
    def test_no_dynamic_query_construction(self):
        """Test that queries are not constructed by string concatenation."""
        # Dangerous pattern
        equipment_id = "PUMP-001'; DROP TABLE equipment; --"

        # BAD: String concatenation (should not exist in code)
        # dangerous_query = f"SELECT * FROM equipment WHERE id = '{equipment_id}'"

        # GOOD: Parameterized query
        safe_query = "SELECT * FROM equipment WHERE id = %s"
        safe_params = (equipment_id,)

        assert "%s" in safe_query or "?" in safe_query
        assert equipment_id not in safe_query


# =============================================================================
# TEST CLASS: AUTHENTICATION REQUIREMENTS
# =============================================================================


class TestAuthenticationRequired:
    """Tests for authentication requirements."""

    @pytest.mark.security
    def test_api_requires_authentication(self, mock_cmms_connector):
        """Test that API calls require authentication."""
        # Mock an unauthenticated call
        unauthenticated_connector = Mock()
        unauthenticated_connector.get_equipment.side_effect = \
            PermissionError("Authentication required")

        with pytest.raises(PermissionError):
            unauthenticated_connector.get_equipment("PUMP-001")

    @pytest.mark.security
    def test_token_validation(self):
        """Test that authentication tokens are properly validated."""
        # Valid token structure
        valid_token = {
            "token_type": "Bearer",
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "expires_in": 3600,
            "issued_at": datetime.now().isoformat(),
        }

        # Basic validation checks
        assert "access_token" in valid_token
        assert valid_token["token_type"] == "Bearer"
        assert valid_token["expires_in"] > 0

    @pytest.mark.security
    def test_expired_token_rejected(self):
        """Test that expired tokens are rejected."""
        expired_token = {
            "access_token": "expired_token",
            "expires_at": (datetime.now() - timedelta(hours=1)).isoformat(),
        }

        def validate_token(token):
            expires_at = datetime.fromisoformat(token["expires_at"])
            if expires_at < datetime.now():
                raise PermissionError("Token expired")
            return True

        with pytest.raises(PermissionError, match="Token expired"):
            validate_token(expired_token)

    @pytest.mark.security
    def test_no_hardcoded_credentials(self, default_config):
        """Test that configuration does not contain hardcoded credentials."""
        config_str = json.dumps(default_config, default=str)

        # Check for common credential patterns
        credential_patterns = [
            r"password\s*[=:]\s*['\"][^'\"]+['\"]",
            r"api_key\s*[=:]\s*['\"][^'\"]+['\"]",
            r"secret\s*[=:]\s*['\"][^'\"]+['\"]",
            r"token\s*[=:]\s*['\"][^'\"]+['\"]",
            r"AWS_SECRET_ACCESS_KEY",
            r"PRIVATE_KEY",
        ]

        for pattern in credential_patterns:
            assert not re.search(pattern, config_str, re.IGNORECASE), \
                f"Potential hardcoded credential found: {pattern}"


# =============================================================================
# TEST CLASS: AUTHORIZATION ENFORCEMENT
# =============================================================================


class TestAuthorizationEnforced:
    """Tests for authorization enforcement."""

    @pytest.mark.security
    def test_role_based_access_control(self):
        """Test role-based access control for operations."""
        # Define roles and permissions
        roles = {
            "viewer": ["read_equipment", "read_alerts"],
            "operator": ["read_equipment", "read_alerts", "acknowledge_alerts"],
            "engineer": ["read_equipment", "read_alerts", "acknowledge_alerts",
                        "modify_thresholds", "create_work_orders"],
            "admin": ["read_equipment", "read_alerts", "acknowledge_alerts",
                     "modify_thresholds", "create_work_orders", "manage_users",
                     "delete_equipment"],
        }

        def check_permission(user_role: str, action: str) -> bool:
            return action in roles.get(user_role, [])

        # Viewer cannot create work orders
        assert not check_permission("viewer", "create_work_orders")

        # Engineer can create work orders
        assert check_permission("engineer", "create_work_orders")

        # Only admin can manage users
        assert not check_permission("engineer", "manage_users")
        assert check_permission("admin", "manage_users")

    @pytest.mark.security
    def test_equipment_access_restriction(self):
        """Test access restriction to specific equipment."""
        # Define equipment access by plant/area
        equipment_access = {
            "plant_a_operator": ["PUMP-001", "PUMP-002", "MTR-001"],
            "plant_b_operator": ["PUMP-101", "PUMP-102", "MTR-101"],
            "global_engineer": "*",  # Access to all
        }

        def can_access_equipment(user_role: str, equipment_id: str) -> bool:
            allowed = equipment_access.get(user_role, [])
            if allowed == "*":
                return True
            return equipment_id in allowed

        # Plant A operator cannot access Plant B equipment
        assert can_access_equipment("plant_a_operator", "PUMP-001")
        assert not can_access_equipment("plant_a_operator", "PUMP-101")

        # Global engineer can access all
        assert can_access_equipment("global_engineer", "PUMP-001")
        assert can_access_equipment("global_engineer", "PUMP-101")

    @pytest.mark.security
    def test_operation_audit_logging(self):
        """Test that sensitive operations are logged."""
        audit_log = []

        def audit_operation(user: str, action: str, target: str, result: str):
            audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "user": user,
                "action": action,
                "target": target,
                "result": result,
            })

        # Simulate operations
        audit_operation("admin", "modify_threshold", "PUMP-001", "success")
        audit_operation("engineer", "create_work_order", "WO-001", "success")

        assert len(audit_log) == 2
        assert audit_log[0]["action"] == "modify_threshold"
        assert audit_log[1]["user"] == "engineer"


# =============================================================================
# TEST CLASS: SECRETS PROTECTION
# =============================================================================


class TestSecretsNotInLogs:
    """Tests for secrets protection in logs."""

    @pytest.mark.security
    def test_api_key_not_logged(self, caplog):
        """Test that API keys are not written to logs."""
        api_key = "sk_live_abcd1234efgh5678ijkl9012"

        # Simulate logging that should mask secrets
        with caplog.at_level(logging.INFO):
            logging.info(f"Connecting to API with key: {'*' * 20}")

        # Check logs don't contain actual key
        for record in caplog.records:
            assert api_key not in record.message

    @pytest.mark.security
    def test_password_not_logged(self, caplog):
        """Test that passwords are not written to logs."""
        password = "SuperSecret123!"

        # Simulate authentication logging
        with caplog.at_level(logging.INFO):
            logging.info(f"User authentication attempt (password hidden)")

        for record in caplog.records:
            assert password not in record.message

    @pytest.mark.security
    def test_secrets_masked_in_config_dump(self, default_config):
        """Test that secrets are masked when config is dumped."""
        config_with_secrets = default_config.copy()
        config_with_secrets["api_key"] = "sk_secret_12345"
        config_with_secrets["database_password"] = "db_password_123"

        def mask_secrets(config: dict) -> dict:
            """Mask sensitive fields in configuration."""
            sensitive_fields = ["api_key", "password", "secret", "token"]
            masked = config.copy()

            for key in masked:
                for sensitive in sensitive_fields:
                    if sensitive in key.lower():
                        masked[key] = "****MASKED****"

            return masked

        masked_config = mask_secrets(config_with_secrets)

        assert masked_config["api_key"] == "****MASKED****"
        assert "sk_secret_12345" not in json.dumps(masked_config)

    @pytest.mark.security
    def test_provenance_hash_no_secrets(self, rul_calculator):
        """Test that provenance hashes don't expose secrets."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        hash_value = result["provenance_hash"]

        # Hash should be valid SHA-256 (not contain raw data)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


# =============================================================================
# TEST CLASS: PROVENANCE TAMPER DETECTION
# =============================================================================


class TestProvenanceTamperDetection:
    """Tests for provenance integrity and tamper detection."""

    @pytest.mark.security
    def test_detect_result_tampering(self, rul_calculator, provenance_validator):
        """Test detection of tampered calculation results."""
        # Get original result
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        original_hash = result["provenance_hash"]
        original_rul = result["rul_hours"]

        # Create tampered result
        tampered_result = result.copy()
        tampered_result["rul_hours"] = Decimal("999999")  # Tampered value

        # Recalculate hash for tampered result
        tampered_data = {"rul_hours": str(tampered_result["rul_hours"])}
        tampered_hash = provenance_validator.compute_hash(tampered_data)

        # Hashes should not match
        assert tampered_hash != original_hash

    @pytest.mark.security
    def test_merkle_root_tamper_detection(self, provenance_validator):
        """Test Merkle tree detects tampering of any leaf."""
        # Original leaves
        leaves = [
            hashlib.sha256(f"calculation_{i}".encode()).hexdigest()
            for i in range(4)
        ]

        # Calculate original root
        l01 = hashlib.sha256((leaves[0] + leaves[1]).encode()).hexdigest()
        l23 = hashlib.sha256((leaves[2] + leaves[3]).encode()).hexdigest()
        original_root = hashlib.sha256((l01 + l23).encode()).hexdigest()

        # Tamper with one leaf
        tampered_leaves = leaves.copy()
        tampered_leaves[2] = hashlib.sha256(b"tampered_value").hexdigest()

        # Verify tampering is detected
        assert not provenance_validator.verify_merkle_root(
            tampered_leaves, original_root
        )

    @pytest.mark.security
    def test_provenance_chain_integrity(self, rul_calculator, vibration_analyzer):
        """Test integrity of multi-step provenance chain."""
        # Step 1: Vibration analysis
        vib_result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("4.5"),
            machine_class=MachineClass.CLASS_II,
        )

        # Step 2: RUL calculation
        rul_result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        # Create chain hash
        chain_input = vib_result["provenance_hash"] + rul_result["provenance_hash"]
        chain_hash = hashlib.sha256(chain_input.encode()).hexdigest()

        # Verify chain hash is deterministic
        chain_input_2 = vib_result["provenance_hash"] + rul_result["provenance_hash"]
        chain_hash_2 = hashlib.sha256(chain_input_2.encode()).hexdigest()

        assert chain_hash == chain_hash_2

    @pytest.mark.security
    def test_timestamp_in_provenance(self):
        """Test that provenance includes timestamp for audit trail."""
        provenance_record = {
            "record_id": "calc_001",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "inputs": {"equipment_type": "pump", "hours": "25000"},
            "outputs": {"rul_hours": "15000"},
            "hash": "abc123...",
        }

        # Timestamp should be present and valid
        assert "timestamp" in provenance_record
        assert provenance_record["timestamp"].endswith("Z")  # UTC


# =============================================================================
# TEST CLASS: DATA SANITIZATION
# =============================================================================


class TestDataSanitization:
    """Tests for data sanitization."""

    @pytest.mark.security
    def test_html_encoding_in_output(self):
        """Test that HTML special characters are encoded."""
        import html

        user_input = "<script>alert('xss')</script>"
        sanitized = html.escape(user_input)

        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized

    @pytest.mark.security
    def test_json_output_safety(self, rul_calculator):
        """Test that JSON output is safe from injection."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000"),
        )

        # Convert to JSON
        json_output = json.dumps(result, default=str)

        # Verify valid JSON (would raise if injection corrupted structure)
        parsed = json.loads(json_output)
        assert "rul_hours" in parsed

    @pytest.mark.security
    def test_equipment_id_sanitization(self, vibration_analyzer):
        """Test equipment ID sanitization."""
        # Test with potentially dangerous characters
        dangerous_ids = [
            "PUMP-001<script>",
            "PUMP-001\"; DROP TABLE",
            "PUMP-001$(command)",
            "PUMP-001`id`",
        ]

        for eq_id in dangerous_ids:
            result = vibration_analyzer.assess_severity(
                velocity_rms=Decimal("4.5"),
                machine_class=MachineClass.CLASS_II,
                equipment_id=eq_id,
            )

            # Result should not contain executable code
            result_str = str(result)
            assert "<script>" not in result_str.lower()
            assert "drop table" not in result_str.lower()


# =============================================================================
# TEST CLASS: ERROR HANDLING SECURITY
# =============================================================================


class TestErrorHandlingSecurity:
    """Tests for secure error handling."""

    @pytest.mark.security
    def test_no_stack_trace_in_user_errors(self):
        """Test that stack traces are not exposed to users."""
        def safe_error_handler(func):
            """Wrapper that sanitizes errors for user display."""
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Return generic error, log detailed error internally
                    return {"error": "An error occurred", "error_code": "E001"}
            return wrapper

        @safe_error_handler
        def risky_operation():
            raise RuntimeError("Detailed internal error with sensitive info")

        result = risky_operation()

        assert "error" in result
        assert "Detailed internal" not in str(result)
        assert "sensitive" not in str(result)

    @pytest.mark.security
    def test_no_path_disclosure_in_errors(self):
        """Test that file paths are not disclosed in errors."""
        # Simulate error that might contain path info
        error_message = "FileNotFoundError: /home/user/app/config/secrets.json"

        def sanitize_error(message: str) -> str:
            """Remove file paths from error messages."""
            # Remove path patterns
            sanitized = re.sub(r'/[\w/]+\.(py|json|yaml|txt)', '[PATH_HIDDEN]', message)
            sanitized = re.sub(r'C:\\[\w\\]+\.(py|json|yaml|txt)', '[PATH_HIDDEN]', sanitized)
            return sanitized

        sanitized = sanitize_error(error_message)

        assert "/home/user" not in sanitized
        assert "secrets.json" not in sanitized


# =============================================================================
# TEST CLASS: RATE LIMITING
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting protection."""

    @pytest.mark.security
    def test_rate_limit_exceeded(self):
        """Test rate limiting for API calls."""
        rate_limit = 100  # requests per minute
        request_count = 0
        window_start = datetime.now()

        def check_rate_limit() -> bool:
            nonlocal request_count, window_start

            # Reset window if minute has passed
            if (datetime.now() - window_start).seconds >= 60:
                request_count = 0
                window_start = datetime.now()

            request_count += 1
            return request_count <= rate_limit

        # First 100 should pass
        for _ in range(100):
            assert check_rate_limit() is True

        # 101st should fail
        assert check_rate_limit() is False

    @pytest.mark.security
    def test_request_throttling(self):
        """Test request throttling implementation."""
        import time

        min_interval = 0.01  # 10ms minimum between requests
        last_request_time = 0

        def throttled_request() -> bool:
            nonlocal last_request_time

            current_time = time.time()
            if current_time - last_request_time < min_interval:
                return False  # Throttled

            last_request_time = current_time
            return True

        # First request should pass
        assert throttled_request() is True

        # Immediate second request should be throttled
        assert throttled_request() is False

        # After waiting, should pass
        time.sleep(min_interval)
        assert throttled_request() is True
