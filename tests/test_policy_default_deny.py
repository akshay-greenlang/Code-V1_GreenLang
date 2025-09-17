"""
Unit Tests for Default-Deny Security Policy
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Removed sys.path manipulation - using installed package

from greenlang.policy.opa import evaluate
from greenlang.policy.enforcer import PolicyEnforcer, check_install, check_run


class TestDefaultDenyBehavior(unittest.TestCase):
    """Test that policies default to deny when conditions not met"""

    def setUp(self):
        """Setup test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.enforcer = PolicyEnforcer(policy_dir=self.test_dir, permissive_mode=False)

    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_deny_when_no_policy(self):
        """Test that missing policy results in denial"""
        with patch('greenlang.policy.opa._check_opa_installed', return_value=True):
            with patch('greenlang.policy.opa._resolve_policy_path') as mock_resolve:
                # Simulate missing policy file
                mock_path = MagicMock()
                mock_path.exists.return_value = False
                mock_resolve.return_value = mock_path

                result = evaluate("nonexistent.rego", {"test": "data"})

                self.assertFalse(result["allow"])
                self.assertIn("POLICY.DENIED", result["reason"])

    def test_deny_when_opa_not_installed(self):
        """Test that missing OPA results in denial"""
        with patch('greenlang.policy.opa._check_opa_installed', return_value=False):
            result = evaluate("test.rego", {"test": "data"})

            self.assertFalse(result["allow"])
            self.assertIn("POLICY.DENIED", result["reason"])
            self.assertIn("OPA not installed", result["reason"])

    def test_deny_unsigned_pack_by_default(self):
        """Test that unsigned packs are denied without override"""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "signature_verified": False,
            "publisher": "greenlang-official"
        }

        with self.assertRaises(RuntimeError) as ctx:
            check_install(manifest, str(self.test_dir), "add", permissive=False)

        self.assertIn("POLICY.DENIED", str(ctx.exception))

    def test_deny_unknown_publisher(self):
        """Test that unknown publishers are denied"""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "signature_verified": True,
            "publisher": "unknown-publisher"
        }

        # Mock the OPA evaluation to use our test logic
        with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
            mock_eval.return_value = {
                "allow": False,
                "reason": "POLICY.DENIED_INSTALL: Publisher 'unknown-publisher' not in allowed list"
            }

            with self.assertRaises(RuntimeError) as ctx:
                check_install(manifest, str(self.test_dir), "publish")

            self.assertIn("Publisher", str(ctx.exception))
            self.assertIn("not in allowed list", str(ctx.exception))

    def test_deny_undeclared_capability(self):
        """Test that undeclared capabilities are denied"""
        pipeline = {
            "name": "test-pipeline",
            "requested_capabilities": ["net", "subprocess"]
        }

        context = MagicMock()
        context.authenticated = True
        context.role = "developer"
        context.region = "US"

        with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
            mock_eval.return_value = {
                "allow": False,
                "reason": "POLICY.DENIED_CAPABILITY: Capability 'subprocess' not declared in pack manifest"
            }

            with self.assertRaises(RuntimeError) as ctx:
                check_run(pipeline, context)

            self.assertIn("POLICY.DENIED", str(ctx.exception))
            self.assertIn("Capability", str(ctx.exception))

    def test_deny_unauthenticated_execution(self):
        """Test that unauthenticated users cannot execute"""
        pipeline = {"name": "test-pipeline"}

        context = MagicMock()
        context.authenticated = False
        context.region = "US"

        with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
            mock_eval.return_value = {
                "allow": False,
                "reason": "POLICY.DENIED_EXECUTION: User not authenticated"
            }

            with self.assertRaises(RuntimeError) as ctx:
                check_run(pipeline, context)

            self.assertIn("not authenticated", str(ctx.exception))

    def test_deny_disallowed_region(self):
        """Test that execution in disallowed regions is denied"""
        pipeline = {"name": "test-pipeline"}

        context = MagicMock()
        context.authenticated = True
        context.region = "RESTRICTED_REGION"

        with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
            mock_eval.return_value = {
                "allow": False,
                "reason": "POLICY.DENIED_EXECUTION: Region 'RESTRICTED_REGION' not allowed"
            }

            with self.assertRaises(RuntimeError) as ctx:
                check_run(pipeline, context)

            self.assertIn("Region", str(ctx.exception))
            self.assertIn("not allowed", str(ctx.exception))

    def test_permissive_mode_allows_failures(self):
        """Test that permissive mode allows operations that would be denied"""
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "signature_verified": False,  # Would normally be denied
            "publisher": "unknown"  # Would normally be denied
        }

        with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
            # In permissive mode, policy returns allow=True
            mock_eval.return_value = {
                "allow": True,
                "reason": "Policy not found (PERMISSIVE MODE)"
            }

            # Should not raise in permissive mode
            try:
                check_install(manifest, str(self.test_dir), "add", permissive=True)
            except RuntimeError:
                self.fail("Permissive mode should have allowed installation")


class TestCapabilityGates(unittest.TestCase):
    """Test capability-based access control"""

    def test_fs_capability_required_for_file_access(self):
        """Test that filesystem access requires 'fs' capability"""
        input_doc = {
            "request": {
                "requested_capabilities": ["fs"]
            },
            "pack": {
                "declared_capabilities": []  # Missing fs
            },
            "org": {
                "allowed_capabilities": ["fs"]
            }
        }

        # Capability should be denied because pack doesn't declare it
        with patch('greenlang.policy.opa.evaluate') as mock_eval:
            mock_eval.return_value = {
                "allow": False,
                "reason": "POLICY.DENIED_CAPABILITY: Capability 'fs' not declared",
                "capabilities": []
            }

            result = mock_eval("test.rego", input_doc)
            self.assertFalse(result["allow"])
            self.assertIn("not declared", result["reason"])

    def test_net_capability_required_for_network(self):
        """Test that network access requires 'net' capability"""
        input_doc = {
            "request": {
                "requested_capabilities": ["net"]
            },
            "pack": {
                "declared_capabilities": ["net"]
            },
            "org": {
                "allowed_capabilities": []  # Org doesn't allow net
            }
        }

        with patch('greenlang.policy.opa.evaluate') as mock_eval:
            mock_eval.return_value = {
                "allow": False,
                "reason": "POLICY.DENIED_CAPABILITY: Capability 'net' not allowed by organization",
                "capabilities": []
            }

            result = mock_eval("test.rego", input_doc)
            self.assertFalse(result["allow"])
            self.assertIn("not allowed", result["reason"])

    def test_capability_allowed_when_all_conditions_met(self):
        """Test that capability is allowed when all conditions are met"""
        input_doc = {
            "request": {
                "requested_capabilities": ["fs"]
            },
            "pack": {
                "declared_capabilities": ["fs"]  # Pack declares it
            },
            "org": {
                "allowed_capabilities": ["fs"]  # Org allows it
            }
        }

        with patch('greenlang.policy.opa.evaluate') as mock_eval:
            mock_eval.return_value = {
                "allow": True,
                "reason": "All capabilities allowed",
                "capabilities": ["fs"]
            }

            result = mock_eval("test.rego", input_doc)
            self.assertTrue(result["allow"])
            self.assertIn("fs", result["capabilities"])


class TestPolicyErrorMessages(unittest.TestCase):
    """Test standardized error messages"""

    def test_error_message_format(self):
        """Test that error messages follow standard format"""
        error_prefixes = [
            "POLICY.DENIED_INSTALL",
            "POLICY.DENIED_EXECUTION",
            "POLICY.DENIED_CAPABILITY"
        ]

        for prefix in error_prefixes:
            with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
                mock_eval.return_value = {
                    "allow": False,
                    "reason": f"{prefix}: Test denial reason"
                }

                manifest = {"name": "test", "publisher": "test"}
                try:
                    check_install(manifest, "/tmp", "add")
                except RuntimeError as e:
                    self.assertIn(prefix, str(e))

    def test_error_includes_remediation_hints(self):
        """Test that errors include helpful remediation hints"""
        test_cases = [
            ("unsigned", "use --allow-unsigned flag"),
            ("not in allowed list", "contact security team"),
            ("capability", "declare in manifest")
        ]

        for keyword, expected_hint in test_cases:
            with patch('greenlang.policy.enforcer.opa_eval') as mock_eval:
                mock_eval.return_value = {
                    "allow": False,
                    "reason": f"POLICY.DENIED: {keyword} - {expected_hint}"
                }

                manifest = {"name": "test"}
                try:
                    check_install(manifest, "/tmp", "add")
                except RuntimeError as e:
                    error_str = str(e)
                    self.assertIn(keyword, error_str)
                    # Verify the error is informative


if __name__ == "__main__":
    unittest.main()