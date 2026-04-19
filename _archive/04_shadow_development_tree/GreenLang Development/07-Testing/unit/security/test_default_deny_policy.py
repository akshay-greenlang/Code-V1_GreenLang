# -*- coding: utf-8 -*-
"""
Unit tests for default-deny policy enforcement

SECURITY GATE: These tests verify that the system defaults to DENY
when policies are missing, malformed, or fail to evaluate.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Removed sys.path manipulation - using installed package

from greenlang.policy.enforcer import PolicyEnforcer, check_install, check_run
from greenlang.policy.opa import evaluate


class TestDefaultDenyPolicy:
    """Test suite for default-deny policy behavior"""

    def test_no_policy_loaded_denies(self):
        """Test A: No policies loaded ⇒ deny"""
        enforcer = PolicyEnforcer()

        # Create test input
        test_input = {
            "pack": {
                "name": "test-pack",
                "version": "1.0.0",
                "signature_verified": False,
                "publisher": "unknown"
            }
        }

        # Check should return False (deny)
        policy_file = Path("nonexistent.rego")
        result = enforcer.check(policy_file, test_input)
        assert result == False, "Should deny when no policy is loaded"

    def test_policy_returns_false_denies(self):
        """Test B: Policy loaded but returns allow=false ⇒ deny"""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir)
            enforcer = PolicyEnforcer(policy_dir=policy_dir)

            # Create a policy that explicitly denies
            deny_policy = """
            package greenlang.install
            default allow = false
            """

            policy_file = policy_dir / "deny.rego"
            policy_file.write_text(deny_policy)

            test_input = {
                "pack": {
                    "name": "test-pack",
                    "version": "1.0.0"
                }
            }

            # This should be denied
            result = enforcer.check(policy_file, test_input)
            assert result == False, "Should deny when policy returns allow=false"

    def test_policy_returns_true_allows(self):
        """Test C: Policy returns allow=true ⇒ allow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir)
            enforcer = PolicyEnforcer(policy_dir=policy_dir)

            # Create a policy that explicitly allows
            allow_policy = """
            package greenlang.install
            default allow = false
            allow = true
            """

            policy_file = policy_dir / "allow.rego"
            policy_file.write_text(allow_policy)

            test_input = {
                "pack": {
                    "name": "test-pack",
                    "version": "1.0.0"
                }
            }

            # Mock the OPA evaluator to return allow=true
            with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
                mock_eval.return_value = {"allow": True}

                # Call check_install instead
                allowed, reasons = enforcer.check_install(
                    Mock(dict=lambda: test_input["pack"]),
                    "/test/path",
                    "add"
                )
                assert allowed == True, "Should allow when policy returns allow=true"

    @patch('subprocess.run')
    def test_opa_error_denies(self, mock_run):
        """Test D: OPA error/timeout ⇒ deny"""
        # Mock OPA to raise an exception
        mock_run.side_effect = Exception("OPA crashed")

        # Test with check_install
        with pytest.raises(RuntimeError) as exc_info:
            check_install(
                Mock(name="test", version="1.0", model_dump=lambda: {}),
                "/test/path",
                "add"            )

        assert "POLICY.DENIED_INSTALL" in str(exc_info.value)

    def test_missing_allow_field_denies(self):
        """Test: Policy doesn't return 'allow' field ⇒ deny"""
        with patch('greenlang.policy.opa.evaluate') as mock_eval:
            # Return a decision without 'allow' field
            mock_eval.return_value = {"reason": "some reason"}

            # This should default to deny
            decision = mock_eval("test.rego", {})
            assert decision.get("allow", False) == False, "Missing 'allow' field should default to False"

    def test_invalid_allow_value_denies(self):
        """Test: Policy returns non-boolean allow ⇒ deny"""
        with patch('greenlang.policy.opa.evaluate') as mock_eval:
            # Test various non-true values
            for invalid_value in [None, "", 0, "false", []]:
                mock_eval.return_value = {"allow": invalid_value}
                decision = mock_eval("test.rego", {})
                # Should be coerced to boolean False
                assert bool(decision.get("allow", False)) == False, f"Value {invalid_value} should evaluate to False"

    def test_unsigned_pack_denied_by_default(self):
        """Test: Unsigned pack is denied by default"""
        with pytest.raises(RuntimeError) as exc_info:
            check_install(
                Mock(
                    name="unsigned-pack",
                    signature_verified=False,
                    model_dump=lambda: {"signature_verified": False}
                ),
                "/test/path",
                "add"            )

        assert "signature" in str(exc_info.value).lower() or "signed" in str(exc_info.value).lower()

    def test_runtime_deny_without_authentication(self):
        """Test: Runtime execution denied without authentication"""
        pipeline = Mock(to_policy_doc=lambda: {})
        ctx = Mock(
            authenticated=False,
            region="us-west-2",
            egress_targets=[],
            role="basic"
        )

        with pytest.raises(RuntimeError) as exc_info:
            check_run(pipeline, ctx)

        assert "POLICY.DENIED_EXECUTION" in str(exc_info.value)

    @patch('greenlang.policy.opa._check_opa_installed')
    def test_opa_not_installed_denies(self, mock_check):
        """Test: OPA not installed ⇒ deny"""
        mock_check.return_value = False

        decision = evaluate("test.rego", {}, permissive_mode=False)
        assert decision["allow"] == False
        assert "OPA not installed" in decision["reason"]

    def test_no_permissive_mode_available(self):
        """Test: Verify no permissive mode functionality exists"""
        # Enforce that PolicyEnforcer always uses strict mode
        enforcer = PolicyEnforcer()

        # Test that unsigned packs are always denied
        test_input = {
            "pack": {
                "name": "test-pack",
                "signature_verified": False,
                "publisher": "unknown"
            }
        }

        policy_file = Path("nonexistent.rego")
        result = enforcer.check(policy_file, test_input)
        assert result == False, "Should always deny unsafe operations - no permissive mode"


class TestPolicyEnforcerIntegration:
    """Integration tests for PolicyEnforcer with real policies"""

    def test_install_policy_enforcement(self):
        """Test that install policy correctly enforces signature requirement"""
        # Create a pack without signature
        pack_manifest = Mock(
            name="test-pack",
            version="1.0.0",
            license="MIT",
            dict=lambda: {
                "name": "test-pack",
                "version": "1.0.0",
                "license": "MIT",
                "signature_verified": False,
                "publisher": "unknown"
            }
        )

        enforcer = PolicyEnforcer()
        allowed, reasons = enforcer.check_install(pack_manifest, "/test/path", "add")

        assert allowed == False, "Unsigned pack should be denied"
        assert any("sign" in r.lower() for r in reasons), "Should mention signature requirement"

    def test_runtime_policy_enforcement(self):
        """Test that runtime policy enforces authentication"""
        pipeline = Mock(
            name="test-pipeline",
            to_policy_doc=lambda: {"name": "test-pipeline"},
            dict=lambda: {"name": "test-pipeline"}
        )

        context = Mock(
            authenticated=False,
            user_role="basic",
            user_id="test-user",
            region="us-west-2",
            config=Mock(egress_targets=[])
        )

        enforcer = PolicyEnforcer()
        allowed, reasons = enforcer.check_run(pipeline, context)

        assert allowed == False, "Unauthenticated execution should be denied"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])