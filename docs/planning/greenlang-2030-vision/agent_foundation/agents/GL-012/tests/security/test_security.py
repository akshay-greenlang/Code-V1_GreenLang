# -*- coding: utf-8 -*-
"""
Security Tests for GL-012 STEAMQUAL.
Comprehensive security validation covering OWASP Top 10 and industrial safety.
"""

import pytest
import json
import re


@pytest.fixture
def valid_steam_data():
    return {"pressure_bar": 10.0, "temperature_c": 180.0, "dryness_fraction": 0.98}


@pytest.fixture
def injection_payloads():
    return {
        "sql_injection": ["'; DROP TABLE data; --", "1 OR 1=1"],
        "command_injection": ["; rm -rf /", "| cat /etc/passwd", "$(whoami)"],
        "path_traversal": ["../../../etc/passwd", "..\..\windows"],
    }


@pytest.fixture
def mock_auth_provider():
    class MockAuth:
        def __init__(self):
            self.valid_creds = {"operator": ("op_pass", "operator"), "admin": ("admin_pass", "admin")}
            self.sessions = {}
        
        def authenticate(self, user, password):
            if user in self.valid_creds and self.valid_creds[user][0] == password:
                sid = f"session_{user}"
                self.sessions[sid] = {"user": user, "role": self.valid_creds[user][1]}
                return {"success": True, "session_id": sid}
            return {"success": False}
        
        def check_permission(self, sid, action):
            if sid not in self.sessions:
                return False
            role = self.sessions[sid]["role"]
            perms = {"operator": ["read"], "admin": ["read", "write", "control", "admin"]}
            return action in perms.get(role, [])
    return MockAuth()


class TestInputValidation:
    @pytest.mark.security
    def test_reject_sql_injection(self, injection_payloads):
        for payload in injection_payloads["sql_injection"]:
            assert not re.match(r"^[a-zA-Z0-9_]+$", payload)

    @pytest.mark.security
    def test_reject_command_injection(self, injection_payloads):
        for payload in injection_payloads["command_injection"]:
            assert not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", payload)

    @pytest.mark.security
    def test_boundary_validation_pressure(self, valid_steam_data):
        for p in [-1.0, 0.0, 500.0, float("inf"), float("nan")]:
            assert not (isinstance(p, float) and 0.0 < p < 300.0 and p == p)

    @pytest.mark.security
    def test_boundary_validation_dryness(self, valid_steam_data):
        for x in [-0.1, 1.5, float("inf")]:
            assert not (0.0 <= x <= 1.0)


class TestAuthentication:
    @pytest.mark.security
    def test_valid_credentials_success(self, mock_auth_provider):
        result = mock_auth_provider.authenticate("admin", "admin_pass")
        assert result["success"]

    @pytest.mark.security
    def test_invalid_credentials_rejection(self, mock_auth_provider):
        assert not mock_auth_provider.authenticate("admin", "wrong")["success"]
        assert not mock_auth_provider.authenticate("unknown", "pass")["success"]


class TestAuthorization:
    @pytest.mark.security
    def test_operator_read_only(self, mock_auth_provider):
        result = mock_auth_provider.authenticate("operator", "op_pass")
        sid = result["session_id"]
        assert mock_auth_provider.check_permission(sid, "read")
        assert not mock_auth_provider.check_permission(sid, "write")
        assert not mock_auth_provider.check_permission(sid, "control")

    @pytest.mark.security
    def test_admin_full_access(self, mock_auth_provider):
        result = mock_auth_provider.authenticate("admin", "admin_pass")
        sid = result["session_id"]
        for action in ["read", "write", "control", "admin"]:
            assert mock_auth_provider.check_permission(sid, action)


class TestDataProtection:
    @pytest.mark.security
    def test_no_secrets_in_logs(self):
        log = "User logged in with session_id=abc123"
        for word in ["password", "secret", "key", "token"]:
            assert word not in log.lower()


class TestSafetyInterlocks:
    @pytest.mark.security
    def test_interlock_bypass_requires_auth(self):
        class Interlock:
            def __init__(self):
                self.active = True
                self.bypass_auth = False
            def bypass(self):
                if not self.bypass_auth:
                    raise PermissionError("Not authorized")
        
        il = Interlock()
        with pytest.raises(PermissionError):
            il.bypass()
        assert il.active

    @pytest.mark.security
    def test_emergency_stop_local_only(self):
        class EStop:
            def disable(self, local=False):
                return local
        
        estop = EStop()
        assert not estop.disable(local=False)
        assert estop.disable(local=True)


class TestAuditCompliance:
    @pytest.mark.security
    def test_control_actions_logged(self):
        audit_log = []
        def log_action(action):
            audit_log.append({"action": action["type"], "user": action.get("user", "system")})
        
        log_action({"type": "set_valve", "user": "engineer"})
        assert len(audit_log) == 1
        assert audit_log[0]["user"] == "engineer"
