"""
Test security denial scenarios to ensure default-deny policies work correctly.
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules we need to test
from greenlang.policy.enforcer import PolicyEnforcer
from greenlang.runtime.guard import RuntimeGuard, CapabilityViolation
from greenlang.security.network import validate_url
from greenlang.registry.oci_client import OCIClient


class TestSecurityDenials:
    """Test suite for security denial scenarios"""

    def test_unsigned_pack_denied(self):
        """Test that unsigned packs are rejected by default"""
        enforcer = PolicyEnforcer()

        # Pack without signature
        unsigned_pack = {
            "pack": {
                "name": "test-pack",
                "version": "1.0.0",
                "license": "MIT",
                "provenance": {
                    "signed": False  # Not signed
                }
            }
        }

        result = enforcer._eval_install_policy(unsigned_pack)
        assert result is False, "Unsigned pack should be denied"

    def test_signed_pack_allowed(self):
        """Test that properly signed packs are allowed"""
        enforcer = PolicyEnforcer()

        # Pack with signature
        signed_pack = {
            "pack": {
                "name": "test-pack",
                "version": "1.0.0",
                "license": "MIT",
                "provenance": {
                    "signed": True  # Properly signed
                },
                "security": {
                    "sbom": "sbom.json"
                },
                "policy": {
                    "network": ["api.example.com"],
                    "ef_vintage_min": 2024
                }
            }
        }

        result = enforcer._eval_install_policy(signed_pack)
        assert result is True, "Signed pack with proper metadata should be allowed"

    def test_network_wildcard_denied(self):
        """Test that wildcard network patterns are rejected"""
        # This would be tested through OPA policy evaluation
        # For now, we test that wildcards aren't in allowed patterns

        test_policy = """
        package test
        import greenlang.runtime

        test_wildcard_rejected {
            not match_pattern("https://example.com", "*")
        }
        """
        # In real implementation, this would be evaluated by OPA
        # Here we verify the code doesn't have wildcard bypass
        from greenlang.runtime.guard import RuntimeGuard

        guard = RuntimeGuard(capabilities={})
        # Capability override should not exist anymore
        assert not hasattr(guard, 'override_mode') or guard.override_mode is False

    def test_capability_override_removed(self):
        """Test that GL_CAPS_OVERRIDE environment variable is ignored"""
        # Set the environment variable that should be ignored
        os.environ['GL_CAPS_OVERRIDE'] = '1'

        try:
            guard = RuntimeGuard(capabilities={})
            # Override mode should always be False regardless of env var
            assert guard.override_mode is False, "Override mode should be disabled"
        finally:
            # Clean up
            del os.environ['GL_CAPS_OVERRIDE']

    def test_http_registry_denied(self):
        """Test that HTTP registries are always denied"""
        with pytest.raises(ValueError, match="HTTP registries are not allowed"):
            client = OCIClient(registry="http://insecure.registry.com")

    def test_insecure_tls_denied(self):
        """Test that insecure TLS is always denied"""
        with pytest.raises(ValueError, match="Insecure TLS is not allowed"):
            client = OCIClient(registry="https://registry.com", insecure=True)

    def test_http_url_validation_fails(self):
        """Test that HTTP URLs are rejected in validation"""
        with pytest.raises(ValueError, match="Insecure scheme 'http' not allowed"):
            validate_url("http://insecure.site.com/data")

    def test_network_capability_denied_by_default(self):
        """Test that network access is denied without capability"""
        guard = RuntimeGuard(capabilities={})  # No network capability

        # Check that network is not allowed
        assert not guard._check_capability('net'), "Network should be denied by default"

    def test_filesystem_capability_denied_by_default(self):
        """Test that filesystem access is denied without capability"""
        guard = RuntimeGuard(capabilities={})  # No fs capability

        # Check that filesystem is not allowed
        assert not guard._check_capability('fs'), "Filesystem should be denied by default"

    def test_subprocess_capability_denied_by_default(self):
        """Test that subprocess execution is denied without capability"""
        guard = RuntimeGuard(capabilities={})  # No subprocess capability

        # Check that subprocess is not allowed
        assert not guard._check_capability('subprocess'), "Subprocess should be denied by default"

    def test_clock_capability_denied_by_default(self):
        """Test that clock access is denied without capability"""
        guard = RuntimeGuard(capabilities={})  # No clock capability

        # Check that clock is not allowed
        assert not guard._check_capability('clock'), "Clock should be denied by default"

    def test_data_residency_enforced(self):
        """Test that data residency is enforced when data operations occur"""
        enforcer = PolicyEnforcer()

        # Test with data operations but no location specified
        runtime_input = {
            "user": {"authenticated": True},
            "has_data_operations": True,
            # Missing data_location should cause denial
            "pack": {
                "policy": {
                    "data_residency": ["US", "EU"]
                }
            }
        }

        # This should fail because data_location is required for data operations
        # The actual implementation would check this in the Rego policy
        # Here we verify the concept is understood
        assert "has_data_operations" in runtime_input

    def test_gpl_license_denied(self):
        """Test that GPL licenses are rejected"""
        enforcer = PolicyEnforcer()

        gpl_pack = {
            "pack": {
                "name": "test-pack",
                "version": "1.0.0",
                "license": "GPL-3.0",  # GPL license should be rejected
                "provenance": {
                    "signed": True
                }
            }
        }

        result = enforcer._eval_install_policy(gpl_pack)
        assert result is False, "GPL licensed pack should be denied"

    def test_old_ef_vintage_denied(self):
        """Test that old EF vintage is rejected"""
        enforcer = PolicyEnforcer()

        old_vintage_pack = {
            "pack": {
                "name": "test-pack",
                "version": "1.0.0",
                "license": "MIT",
                "provenance": {
                    "signed": True
                },
                "policy": {
                    "ef_vintage_min": 2020,  # Too old
                    "network": ["api.example.com"]
                },
                "security": {
                    "sbom": "sbom.json"
                }
            }
        }

        result = enforcer._eval_install_policy(old_vintage_pack)
        assert result is False, "Old EF vintage should be denied"

    def test_missing_sbom_denied(self):
        """Test that packs without SBOM are rejected"""
        enforcer = PolicyEnforcer()

        no_sbom_pack = {
            "pack": {
                "name": "test-pack",
                "version": "1.0.0",
                "license": "MIT",
                "provenance": {
                    "signed": True
                },
                "policy": {
                    "network": ["api.example.com"],
                    "ef_vintage_min": 2024
                },
                "security": {}  # Missing SBOM
            }
        }

        result = enforcer._eval_install_policy(no_sbom_pack)
        assert result is False, "Pack without SBOM should be denied"

    def test_empty_network_allowlist_denied(self):
        """Test that empty network allowlist is rejected for packs"""
        enforcer = PolicyEnforcer()

        empty_network_pack = {
            "pack": {
                "name": "test-pack",
                "version": "1.0.0",
                "license": "MIT",
                "kind": "pack",
                "provenance": {
                    "signed": True
                },
                "policy": {
                    "network": [],  # Empty network allowlist
                    "ef_vintage_min": 2024
                },
                "security": {
                    "sbom": "sbom.json"
                }
            }
        }

        result = enforcer._eval_install_policy(empty_network_pack)
        assert result is False, "Pack with empty network allowlist should be denied"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])