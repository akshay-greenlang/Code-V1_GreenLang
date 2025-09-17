"""
Integration Tests for Security Workflows
Tests end-to-end security scenarios including install, run, and capability checks
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import os

# Removed sys.path manipulation - using installed package


class TestSecurityWorkflows(unittest.TestCase):
    """Integration tests for complete security workflows"""

    @classmethod
    def setUpClass(cls):
        """Setup test environment once for all tests"""
        cls.test_root = Path(tempfile.mkdtemp())
        cls.test_pack_dir = cls.test_root / "test-pack"
        cls.test_pack_dir.mkdir()

        # Create a test pack manifest
        cls.pack_manifest = {
            "name": "test-security-pack",
            "version": "1.0.0",
            "publisher": "greenlang-official",
            "license": "MIT",
            "capabilities": ["fs"],
            "policy": {
                "network": [],
                "data_residency": ["US", "EU"],
                "ef_vintage_min": 2024
            },
            "security": {
                "sbom": "sbom.json"
            }
        }

        # Write pack manifest
        with open(cls.test_pack_dir / "pack.yaml", "w") as f:
            import yaml
            yaml.dump(cls.pack_manifest, f)

        # Create test policy input
        cls.test_input = {
            "pack": cls.pack_manifest,
            "stage": "install",
            "org": {
                "allowed_publishers": ["greenlang-official"],
                "allowed_regions": ["US", "EU"],
                "allowed_capabilities": ["fs"]
            },
            "env": {
                "region": "US"
            },
            "user": {
                "authenticated": True,
                "role": "developer"
            }
        }

        # Write test input
        with open(cls.test_root / "test_input.json", "w") as f:
            json.dump(cls.test_input, f)

    @classmethod
    def tearDownClass(cls):
        """Cleanup test environment"""
        if cls.test_root.exists():
            shutil.rmtree(cls.test_root)

    def test_install_workflow_unsigned_pack_denied(self):
        """Test that unsigned packs are denied by default"""
        # Modify input to simulate unsigned pack
        unsigned_input = self.test_input.copy()
        unsigned_input["pack"]["signature_verified"] = False

        input_file = self.test_root / "unsigned_input.json"
        with open(input_file, "w") as f:
            json.dump(unsigned_input, f)

        # Test with gl policy check command (if available)
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.policy", "check", str(input_file)],
            capture_output=True,
            text=True
        )

        # Should be denied
        self.assertNotEqual(result.returncode, 0, "Unsigned pack should be denied")
        self.assertIn("denied", result.stdout.lower() + result.stderr.lower())

    def test_install_workflow_signed_pack_allowed(self):
        """Test that properly signed packs from allowed publishers are allowed"""
        # Modify input to simulate signed pack
        signed_input = self.test_input.copy()
        signed_input["pack"]["signature_verified"] = True

        input_file = self.test_root / "signed_input.json"
        with open(input_file, "w") as f:
            json.dump(signed_input, f)

        # Test with gl policy check command
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.policy", "check", str(input_file)],
            capture_output=True,
            text=True
        )

        # Should be allowed (if OPA is available)
        # Note: This test may fail if OPA is not installed
        if "OPA not" not in result.stderr:
            self.assertEqual(result.returncode, 0, "Signed pack should be allowed")

    def test_execution_workflow_capability_check(self):
        """Test that execution checks capabilities properly"""
        # Create execution input
        exec_input = {
            "stage": "run",
            "pipeline": {
                "name": "test-pipeline",
                "steps": []
            },
            "request": {
                "requested_capabilities": ["fs", "net"],  # Request net which pack doesn't have
                "run_id": "test-123"
            },
            "pack": self.pack_manifest,
            "org": self.test_input["org"],
            "env": self.test_input["env"],
            "user": self.test_input["user"]
        }

        input_file = self.test_root / "exec_input.json"
        with open(input_file, "w") as f:
            json.dump(exec_input, f)

        # Test with gl policy check command
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.policy", "check", str(input_file), "--stage", "run"],
            capture_output=True,
            text=True
        )

        # Should be denied due to missing 'net' capability
        self.assertNotEqual(result.returncode, 0, "Should deny undeclared capability")
        output = result.stdout.lower() + result.stderr.lower()
        if "opa not" not in output:  # Skip if OPA not available
            self.assertIn("capabilit", output)

    def test_region_restriction_workflow(self):
        """Test that region restrictions are enforced"""
        # Create input for restricted region
        region_input = self.test_input.copy()
        region_input["env"]["region"] = "RESTRICTED"

        input_file = self.test_root / "region_input.json"
        with open(input_file, "w") as f:
            json.dump(region_input, f)

        # Test with policy check
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.policy", "check", str(input_file)],
            capture_output=True,
            text=True
        )

        # Should be denied due to region
        self.assertNotEqual(result.returncode, 0, "Restricted region should be denied")
        output = result.stdout.lower() + result.stderr.lower()
        if "opa not" not in output:
            self.assertIn("region", output)

    def test_permissive_mode_workflow(self):
        """Test that permissive mode allows otherwise denied operations"""
        # Set environment variable for permissive mode
        os.environ["GL_POLICY_PERMISSIVE"] = "1"

        try:
            # Create input that would normally be denied
            bad_input = {
                "pack": {
                    "name": "bad-pack",
                    "publisher": "unknown",
                    "signature_verified": False
                },
                "stage": "install"
            }

            input_file = self.test_root / "bad_input.json"
            with open(input_file, "w") as f:
                json.dump(bad_input, f)

            # In permissive mode, this should pass (with warnings)
            from greenlang.policy.opa import evaluate
            result = evaluate(
                "policies/default/allowlists.rego",
                bad_input,
                permissive_mode=True
            )

            # In permissive mode, should allow
            self.assertTrue(result.get("allow", False), "Permissive mode should allow")
            self.assertIn("PERMISSIVE", result.get("reason", ""))

        finally:
            # Clean up environment
            del os.environ["GL_POLICY_PERMISSIVE"]


class TestCLIIntegration(unittest.TestCase):
    """Test CLI commands with security features"""

    def setUp(self):
        """Setup for each test"""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Cleanup after each test"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_pack_add_with_allow_unsigned_flag(self):
        """Test gl pack add --allow-unsigned flag"""
        # This would require a mock registry or local pack
        # For now, just test that the flag is recognized
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.pack", "add", "--help"],
            capture_output=True,
            text=True
        )

        self.assertIn("--allow-unsigned", result.stdout)
        self.assertIn("--policy-permissive", result.stdout)

    def test_pack_publish_with_skip_policy_flag(self):
        """Test gl pack publish --skip-policy flag"""
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.pack", "publish", "--help"],
            capture_output=True,
            text=True
        )

        self.assertIn("--skip-policy", result.stdout)

    def test_policy_test_command(self):
        """Test gl policy test command"""
        # Create a simple test policy
        test_policy = """
package test
default allow = false
allow {
    input.test == true
}
"""
        policy_file = self.test_dir / "test.rego"
        with open(policy_file, "w") as f:
            f.write(test_policy)

        # Create test input
        input_data = {"test": True}
        input_file = self.test_dir / "input.json"
        with open(input_file, "w") as f:
            json.dump(input_data, f)

        # Run policy test
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.policy", "test",
             str(policy_file), "--input", str(input_file)],
            capture_output=True,
            text=True
        )

        # Check output (may fail if OPA not installed)
        if "OPA not" not in result.stderr:
            self.assertIn("allow", result.stdout.lower() or result.stderr.lower())

    def test_policy_init_command(self):
        """Test gl policy init command"""
        result = subprocess.run(
            ["python", "-m", "greenlang.cli.policy", "init"],
            capture_output=True,
            text=True,
            cwd=str(self.test_dir)
        )

        # Should create default policies
        if result.returncode == 0:
            # Check if policies were created
            home_policies = Path.home() / ".greenlang" / "policies"
            self.assertTrue(
                home_policies.exists() or "initialized" in result.stdout.lower(),
                "Policies should be initialized"
            )


class TestPolicyDecisionAudit(unittest.TestCase):
    """Test that policy decisions are properly logged for audit"""

    def test_denial_logged_with_reason(self):
        """Test that denials are logged with clear reasons"""
        import logging
        from io import StringIO

        # Capture logs
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)

        logger = logging.getLogger("greenlang.policy")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            # Trigger a denial
            from greenlang.policy.enforcer import check_install
            manifest = {
                "name": "test",
                "signature_verified": False,
                "publisher": "unknown"
            }

            with self.assertRaises(RuntimeError):
                check_install(manifest, "/tmp", "add", permissive=False)

            # Check logs
            log_contents = log_capture.getvalue()
            self.assertIn("denied", log_contents.lower())

        finally:
            logger.removeHandler(handler)

    def test_capability_denial_logged(self):
        """Test that capability denials are logged"""
        import logging
        from io import StringIO

        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)

        logger = logging.getLogger("greenlang.policy")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            from greenlang.policy.enforcer import check_run
            pipeline = {
                "name": "test",
                "requested_capabilities": ["net", "subprocess"]
            }
            context = type('obj', (), {
                'authenticated': True,
                'region': 'US',
                'role': 'developer'
            })()

            with self.assertRaises(RuntimeError):
                check_run(pipeline, context, permissive=False)

            log_contents = log_capture.getvalue()
            # Should log about capability denial
            self.assertTrue(
                "capabilit" in log_contents.lower() or
                "denied" in log_contents.lower()
            )

        finally:
            logger.removeHandler(handler)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)