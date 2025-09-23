"""
Comprehensive tests for GreenLang Capabilities System

Tests cover:
- Capability declaration and validation
- Runtime guard enforcement
- Network/FS/Subprocess/Clock controls
- Policy enforcement
- Audit logging
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Removed sys.path manipulation - using installed package

from greenlang.packs.manifest import (
    PackManifest, Capabilities, NetCapability,
    FsCapability, ClockCapability, SubprocessCapability
)
from greenlang.packs.installer import PackInstaller
from greenlang.runtime.guard import RuntimeGuard, CapabilityViolation
from greenlang.policy.enforcer import PolicyEnforcer


class TestManifestCapabilities(unittest.TestCase):
    """Test capability declarations in pack manifest"""

    def test_default_deny_all(self):
        """Test that all capabilities default to deny"""
        manifest = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack",
            license="MIT",
            contents={"pipelines": ["test.yaml"]}
        )

        # No capabilities specified means deny all
        self.assertIsNone(manifest.capabilities)

    def test_network_capability_declaration(self):
        """Test network capability with allowlist"""
        manifest = PackManifest(
            name="test-pack",
            version="1.0.0",
            kind="pack",
            license="MIT",
            contents={"pipelines": ["test.yaml"]},
            capabilities=Capabilities(
                net=NetCapability(
                    allow=True,
                    outbound={"allowlist": [
                        "https://api.example.com/*",
                        "https://*.climatenza.com/*"
                    ]}
                )
            )
        )

        self.assertTrue(manifest.capabilities.net.allow)
        self.assertEqual(len(manifest.capabilities.net.outbound["allowlist"]), 2)

    def test_filesystem_capability_validation(self):
        """Test filesystem capability path validation"""
        # Valid paths
        caps = Capabilities(
            fs=FsCapability(
                allow=True,
                read={"allowlist": ["${INPUT_DIR}/**", "${PACK_DATA_DIR}/**"]},
                write={"allowlist": ["${RUN_TMP}/**"]}
            )
        )
        self.assertTrue(caps.fs.allow)

        # Invalid path with traversal should raise
        with self.assertRaises(ValueError) as ctx:
            FsCapability(
                allow=True,
                read={"allowlist": ["../../../etc/passwd"]}
            )
        self.assertIn("Path traversal", str(ctx.exception))

    def test_subprocess_capability_validation(self):
        """Test subprocess capability binary validation"""
        # Valid absolute paths
        caps = Capabilities(
            subprocess=SubprocessCapability(
                allow=True,
                allowlist=["/usr/bin/exiftool", "/usr/local/bin/ffmpeg"]
            )
        )
        self.assertTrue(caps.subprocess.allow)

        # Invalid relative path should raise
        with self.assertRaises(ValueError) as ctx:
            SubprocessCapability(
                allow=True,
                allowlist=["exiftool", "../bin/dangerous"]
            )
        self.assertIn("must be absolute", str(ctx.exception))


class TestRuntimeGuard(unittest.TestCase):
    """Test runtime guard enforcement"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="gl_test_")
        os.environ['GL_INPUT_DIR'] = str(Path(self.temp_dir) / "input")
        os.environ['GL_PACK_DATA_DIR'] = str(Path(self.temp_dir) / "pack")
        os.environ['GL_RUN_TMP'] = str(Path(self.temp_dir) / "tmp")

        # Create directories
        Path(os.environ['GL_INPUT_DIR']).mkdir(parents=True)
        Path(os.environ['GL_PACK_DATA_DIR']).mkdir(parents=True)
        Path(os.environ['GL_RUN_TMP']).mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_network_deny_by_default(self):
        """Test that network access is denied by default"""
        guard = RuntimeGuard(capabilities={})

        # Socket creation should fail
        with self.assertRaises(CapabilityViolation) as ctx:
            import socket
            sock = socket.socket()

        self.assertEqual(ctx.exception.capability, 'net')

    def test_network_allowlist_enforcement(self):
        """Test network domain allowlist"""
        guard = RuntimeGuard(capabilities={
            "net": {
                "allow": True,
                "outbound": {
                    "allowlist": ["api.example.com", "*.test.com"]
                }
            }
        })

        # Mock socket to test domain checking
        original_socket = guard.originals['socket']

        with patch.object(original_socket, 'connect') as mock_connect:
            sock = socket.socket()

            # Allowed domain should work
            sock.connect(("api.example.com", 443))
            mock_connect.assert_called_once()

            # Disallowed domain should fail
            with self.assertRaises(CapabilityViolation):
                sock.connect(("evil.com", 443))

    def test_filesystem_read_denied(self):
        """Test filesystem read is denied by default"""
        guard = RuntimeGuard(capabilities={})

        # Reading should fail
        with self.assertRaises(CapabilityViolation) as ctx:
            with open("/etc/passwd", "r") as f:
                f.read()

        self.assertEqual(ctx.exception.capability, 'fs')

    def test_filesystem_allowed_paths(self):
        """Test filesystem access to allowed paths"""
        guard = RuntimeGuard(capabilities={
            "fs": {
                "allow": True,
                "read": {
                    "allowlist": ["${INPUT_DIR}/**"]
                },
                "write": {
                    "allowlist": ["${RUN_TMP}/**"]
                }
            }
        })

        # Write to allowed path should work
        test_file = Path(os.environ['GL_RUN_TMP']) / "test.txt"
        with open(test_file, "w") as f:
            f.write("test")

        # Read from allowed path should work
        input_file = Path(os.environ['GL_INPUT_DIR']) / "input.txt"
        input_file.write_text("input data")
        with open(input_file, "r") as f:
            data = f.read()
        self.assertEqual(data, "input data")

        # Write to disallowed path should fail
        with self.assertRaises(CapabilityViolation):
            with open("/tmp/evil.txt", "w") as f:
                f.write("should fail")

    def test_subprocess_denied(self):
        """Test subprocess execution is denied by default"""
        guard = RuntimeGuard(capabilities={})

        # Subprocess should fail
        import subprocess
        with self.assertRaises(CapabilityViolation) as ctx:
            subprocess.run(["ls"])

        self.assertEqual(ctx.exception.capability, 'subprocess')

    def test_subprocess_allowlist(self):
        """Test subprocess binary allowlist"""
        guard = RuntimeGuard(capabilities={
            "subprocess": {
                "allow": True,
                "allowlist": ["/bin/ls", "/usr/bin/echo"]
            }
        })

        # Mock to avoid actual execution
        with patch('subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock()

            # Allowed binary should work
            subprocess.run(["/bin/ls", "-la"])
            mock_popen.assert_called()

            # Disallowed binary should fail
            with self.assertRaises(CapabilityViolation):
                subprocess.run(["/bin/rm", "-rf", "/"])

    def test_clock_frozen_mode(self):
        """Test frozen clock mode for determinism"""
        frozen_time = 1234567890.0
        os.environ['GL_FROZEN_TIME'] = str(frozen_time)

        guard = RuntimeGuard(capabilities={})

        # Time should be frozen
        t1 = time.time()
        time.sleep(0.1)  # Would normally advance
        t2 = time.time()

        self.assertEqual(t1, frozen_time)
        self.assertEqual(t2, frozen_time)

    def test_clock_real_time(self):
        """Test real-time clock when allowed"""
        guard = RuntimeGuard(capabilities={
            "clock": {"allow": True}
        })

        # Time should advance normally
        t1 = time.time()
        time.sleep(0.01)
        t2 = time.time()

        self.assertGreater(t2, t1)

    def test_audit_logging(self):
        """Test audit log generation"""
        guard = RuntimeGuard(capabilities={})

        # Trigger some violations
        try:
            with open("/etc/passwd", "r") as f:
                pass
        except CapabilityViolation:
            pass

        try:
            import socket
            socket.socket()
        except CapabilityViolation:
            pass

        # Check audit log
        audit_log = guard.get_audit_log()
        self.assertGreater(len(audit_log), 0)

        # Check for expected events
        capabilities_audited = [event['capability'] for event in audit_log]
        self.assertIn('fs', capabilities_audited)
        self.assertIn('net', capabilities_audited)


class TestCapabilityInstaller(unittest.TestCase):
    """Test pack installer capability validation"""

    def setUp(self):
        """Set up test packs"""
        self.temp_dir = tempfile.mkdtemp(prefix="gl_pack_test_")
        self.installer = PackInstaller(Path(self.temp_dir))

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_dangerous_capabilities(self):
        """Test validation catches dangerous capability requests"""
        # Create a pack with dangerous capabilities
        pack_dir = Path(self.temp_dir) / "dangerous-pack"
        pack_dir.mkdir()

        manifest = {
            "name": "dangerous-pack",
            "version": "1.0.0",
            "license": "MIT",
            "contents": {"pipelines": ["pipeline.yaml"]},
            "capabilities": {
                "fs": {
                    "allow": True,
                    "write": {
                        "allowlist": ["/etc/**", "/usr/bin/**"]
                    }
                },
                "subprocess": {
                    "allow": True,
                    "allowlist": ["/bin/sh", "/bin/bash"]
                }
            }
        }

        manifest_path = pack_dir / "pack.yaml"
        import yaml
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)

        # Create dummy pipeline file
        (pack_dir / "pipeline.yaml").touch()

        # Validate should find issues
        is_valid, issues = self.installer.validate_manifest(manifest_path)

        self.assertFalse(is_valid)
        self.assertTrue(any("system configuration" in issue.lower() for issue in issues))
        self.assertTrue(any("dangerous binary" in issue.lower() for issue in issues))

    def test_lint_capabilities_report(self):
        """Test capability lint report generation"""
        # Create a pack with mixed capabilities
        pack_dir = Path(self.temp_dir) / "test-pack"
        pack_dir.mkdir()

        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "MIT",
            "contents": {"pipelines": ["pipeline.yaml"]},
            "capabilities": {
                "net": {
                    "allow": True,
                    "outbound": {
                        "allowlist": ["https://api.example.com/*"]
                    }
                },
                "fs": {"allow": False},
                "clock": {"allow": False},
                "subprocess": {"allow": False}
            }
        }

        manifest_path = pack_dir / "pack.yaml"
        import yaml
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)

        (pack_dir / "pipeline.yaml").touch()

        # Generate lint report
        report = self.installer.lint_capabilities(manifest_path)

        self.assertIn("test-pack", report)
        self.assertIn("NETWORK: Enabled", report)
        self.assertIn("FILESYSTEM: Disabled", report)
        self.assertIn("CLOCK: Frozen/deterministic", report)
        self.assertIn("SUBPROCESS: Disabled", report)


class TestPolicyEnforcement(unittest.TestCase):
    """Test OPA policy enforcement for capabilities"""

    def setUp(self):
        """Set up policy enforcer"""
        self.policy_dir = tempfile.mkdtemp(prefix="gl_policy_test_")
        self.enforcer = PolicyEnforcer(
            policy_dir=Path(self.policy_dir)
        )

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.policy_dir, ignore_errors=True)

    def test_check_capability_policy(self):
        """Test capability policy checking"""
        # Define organization policy
        org_policy = {
            "net": {
                "allow": True,
                "allowed_domains": ["*.company.com", "api.partner.com"]
            },
            "fs": {"allow": True},
            "subprocess": {
                "allow": False
            },
            "clock": {"allow": False}
        }

        # Pack requesting allowed capabilities
        pack_caps = {
            "net": {
                "allow": True,
                "outbound": {
                    "allowlist": ["api.company.com"]
                }
            },
            "fs": {"allow": True}
        }

        allowed, reasons = self.enforcer.check_capability_policy(pack_caps, org_policy)
        self.assertTrue(allowed)
        self.assertEqual(len(reasons), 0)

        # Pack requesting denied capability
        pack_caps_denied = {
            "subprocess": {"allow": True, "allowlist": ["/bin/ls"]}
        }

        allowed, reasons = self.enforcer.check_capability_policy(pack_caps_denied, org_policy)
        self.assertFalse(allowed)
        self.assertTrue(any("subprocess" in r for r in reasons))

        # Pack requesting domain not in org allowlist
        pack_caps_bad_domain = {
            "net": {
                "allow": True,
                "outbound": {
                    "allowlist": ["https://evil.com"]
                }
            }
        }

        allowed, reasons = self.enforcer.check_capability_policy(pack_caps_bad_domain, org_policy)
        self.assertFalse(allowed)
        self.assertTrue(any("evil.com" in r for r in reasons))


class TestCapabilityOverride(unittest.TestCase):
    """Test capability override for development"""

    def test_override_mode(self):
        """Test capability override mode for development"""
        # Enable override mode
        os.environ['GL_CAPS_OVERRIDE'] = '1'

        guard = RuntimeGuard(capabilities={})

        # Everything should be allowed in override mode
        # Note: We're testing that no exceptions are raised

        # Network should be allowed
        import socket
        sock = socket.socket()  # Should not raise

        # Filesystem should be allowed
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"test")  # Should not raise

        # Cleanup
        del os.environ['GL_CAPS_OVERRIDE']


class TestEndToEndCapabilities(unittest.TestCase):
    """End-to-end integration tests"""

    def test_pack_with_capabilities_execution(self):
        """Test executing a pack with capabilities"""
        from greenlang.runtime.executor import PipelineExecutor, ExecutionContext
        from greenlang.sdk.pipeline_spec import PipelineSpec, StepSpec

        # Create pipeline with capability requirements
        pipeline = PipelineSpec(
            name="test-pipeline",
            version="1.0.0",
            steps=[
                StepSpec(
                    name="fetch-data",
                    agent="TestAgent",
                    action="fetch",
                    inputs={"url": "https://api.example.com/data"}
                )
            ]
        )

        # Create context with capabilities
        context = ExecutionContext(
            run_id="test-run-001",
            pipeline_name="test-pipeline",
            capabilities={
                "net": {
                    "allow": True,
                    "outbound": {
                        "allowlist": ["https://api.example.com/*"]
                    }
                }
            }
        )

        executor = PipelineExecutor()

        # Mock the agent execution
        with patch.object(executor, '_execute_in_guarded_worker') as mock_worker:
            mock_worker.return_value = {
                "output": {"data": "test"},
                "stdout": "",
                "stderr": "",
                "worker_status": "success"
            }

            # Execute step
            result = executor.execute_step(pipeline.steps[0], context)

            # Verify guarded worker was used
            mock_worker.assert_called_once()

            # Check result
            self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()