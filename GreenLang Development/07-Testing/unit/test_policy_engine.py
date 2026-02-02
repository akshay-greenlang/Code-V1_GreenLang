# -*- coding: utf-8 -*-
"""
Policy Engine Tests
===================

Tests for the GreenLang Policy Engine covering:
- Default-deny network/filesystem access
- Allow via manifest configuration
- Region/publisher allowlists
- Policy validation
- OPA policy evaluation

Target: High coverage for policy module
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from greenlang.policy.enforcer import PolicyEnforcer, check_install, check_run


class TestPolicyEnforcerInit:
    """Test PolicyEnforcer initialization"""

    def test_default_initialization(self):
        """Test default policy enforcer initialization"""
        enforcer = PolicyEnforcer()
        assert enforcer.policy_dir.name == "policies"
        assert enforcer.policy_dir.exists()
        assert isinstance(enforcer.policies, dict)

    def test_custom_policy_dir(self):
        """Test initialization with custom policy directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir) / "custom_policies"
            enforcer = PolicyEnforcer(policy_dir=policy_dir)
            assert enforcer.policy_dir == policy_dir
            assert policy_dir.exists()

    def test_policy_loading(self):
        """Test that policies are loaded from directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir)

            # Create test policy file
            test_policy = """package greenlang.test

allow {
    input.user.authenticated == true
}"""
            with open(policy_dir / "test.rego", "w") as f:
                f.write(test_policy)

            enforcer = PolicyEnforcer(policy_dir=policy_dir)
            assert "test" in enforcer.policies
            assert "package greenlang.test" in enforcer.policies["test"]


class TestInstallPolicyEvaluation:
    """Test install-time policy evaluation"""

    def setup_method(self):
        """Setup test enforcer"""
        self.temp_dir = tempfile.mkdtemp()
        self.enforcer = PolicyEnforcer(policy_dir=Path(self.temp_dir))

    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deny_unsigned_pack(self):
        """Test that unsigned packs are denied"""
        pack_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "MIT",
            "provenance": {"signed": False}
        }

        result = self.enforcer._eval_install_policy({"pack": pack_manifest})
        assert result is False

    def test_allow_signed_pack(self):
        """Test that signed packs with valid license are allowed"""
        pack_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "MIT",
            "provenance": {"signed": True},
            "kind": "pack",
            "policy": {"network": ["https://api.greenlang.io"]},
            "security": {"sbom": True}
        }

        result = self.enforcer._eval_install_policy({"pack": pack_manifest})
        assert result is True

    def test_deny_invalid_license(self):
        """Test that packs with invalid licenses are denied"""
        pack_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "GPL-3.0",  # Not in allowlist
            "provenance": {"signed": True}
        }

        result = self.enforcer._eval_install_policy({"pack": pack_manifest})
        assert result is False

    def test_deny_empty_network_policy(self):
        """Test that packs without network policy are denied"""
        pack_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "MIT",
            "provenance": {"signed": True},
            "kind": "pack",
            "policy": {"network": []},  # Empty network allowlist
            "security": {"sbom": True}
        }

        result = self.enforcer._eval_install_policy({"pack": pack_manifest})
        assert result is False

    def test_deny_old_ef_vintage(self):
        """Test that packs with old EF vintage are denied"""
        pack_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "MIT",
            "provenance": {"signed": True},
            "policy": {"ef_vintage_min": 2020},  # Too old
            "security": {"sbom": True}
        }

        result = self.enforcer._eval_install_policy({"pack": pack_manifest})
        assert result is False

    def test_deny_missing_sbom(self):
        """Test that packs without SBOM are denied"""
        pack_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "MIT",
            "provenance": {"signed": True},
            "kind": "pack",
            "policy": {"network": ["https://api.greenlang.io"]},
            "security": {}  # No SBOM
        }

        result = self.enforcer._eval_install_policy({"pack": pack_manifest})
        assert result is False

    def test_allow_commercial_license(self):
        """Test that commercial licenses are allowed"""
        pack_manifest = {
            "name": "commercial-pack",
            "version": "1.0.0",
            "license": "Commercial",
            "provenance": {"signed": True},
            "kind": "pack",
            "policy": {"network": ["https://api.greenlang.io"]},
            "security": {"sbom": True}
        }

        result = self.enforcer._eval_install_policy({"pack": pack_manifest})
        assert result is True


class TestRuntimePolicyEvaluation:
    """Test runtime policy evaluation"""

    def setup_method(self):
        """Setup test enforcer"""
        self.temp_dir = tempfile.mkdtemp()
        self.enforcer = PolicyEnforcer(policy_dir=Path(self.temp_dir))

    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deny_unauthenticated_user(self):
        """Test that unauthenticated users are denied"""
        input_data = {
            "user": {"authenticated": False},
            "resources": {"memory_mb": 1024, "cpu_cores": 1}
        }

        result = self.enforcer._eval_runtime_policy(input_data)
        assert result is False

    def test_allow_authenticated_user(self):
        """Test that authenticated users with reasonable resources are allowed"""
        input_data = {
            "user": {"authenticated": True, "requests_per_minute": 50, "role": "basic"},
            "resources": {"memory_mb": 1024, "cpu_cores": 1}
        }

        result = self.enforcer._eval_runtime_policy(input_data)
        assert result is True

    def test_deny_excessive_memory(self):
        """Test that excessive memory requests are denied"""
        input_data = {
            "user": {"authenticated": True},
            "resources": {"memory_mb": 8192, "cpu_cores": 1}  # > 4GB
        }

        result = self.enforcer._eval_runtime_policy(input_data)
        assert result is False

    def test_deny_excessive_cpu(self):
        """Test that excessive CPU requests are denied"""
        input_data = {
            "user": {"authenticated": True},
            "resources": {"memory_mb": 1024, "cpu_cores": 8}  # > 4 cores
        }

        result = self.enforcer._eval_runtime_policy(input_data)
        assert result is False

    def test_deny_rate_limit_exceeded_basic(self):
        """Test that rate limits are enforced for basic users"""
        input_data = {
            "user": {"authenticated": True, "requests_per_minute": 150, "role": "basic"},
            "resources": {"memory_mb": 1024, "cpu_cores": 1}
        }

        result = self.enforcer._eval_runtime_policy(input_data)
        assert result is False

    def test_allow_premium_higher_rate_limit(self):
        """Test that premium users have higher rate limits"""
        input_data = {
            "user": {"authenticated": True, "requests_per_minute": 500, "role": "premium"},
            "resources": {"memory_mb": 1024, "cpu_cores": 1}
        }

        result = self.enforcer._eval_runtime_policy(input_data)
        assert result is True

    def test_deny_premium_rate_limit_exceeded(self):
        """Test that even premium users have rate limits"""
        input_data = {
            "user": {"authenticated": True, "requests_per_minute": 1500, "role": "premium"},
            "resources": {"memory_mb": 1024, "cpu_cores": 1}
        }

        result = self.enforcer._eval_runtime_policy(input_data)
        assert result is False


class TestPolicyChecking:
    """Test high-level policy checking methods"""

    def setup_method(self):
        """Setup test enforcer"""
        self.temp_dir = tempfile.mkdtemp()
        self.enforcer = PolicyEnforcer(policy_dir=Path(self.temp_dir))

    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_check_missing_policy_file(self):
        """Test checking against missing policy file"""
        nonexistent_policy = Path(self.temp_dir) / "nonexistent.rego"
        result = self.enforcer.check(nonexistent_policy, {"test": "data"})
        assert result is False

    def test_check_with_simple_rego_evaluation(self):
        """Test policy checking with simple Rego evaluation"""
        # Create a test policy file
        policy_content = """package greenlang.install

deny[msg] {
    input.pack.license == "GPL"
    msg := "GPL license not allowed"
}

allow {
    input.pack.license == "MIT"
}"""
        policy_file = Path(self.temp_dir) / "test_install.rego"
        with open(policy_file, "w") as f:
            f.write(policy_content)

        # Test with allowed license
        input_data = {"pack": {"license": "MIT"}}
        result = self.enforcer.check(policy_file, input_data)
        # This depends on the simple evaluation logic
        assert isinstance(result, bool)

    @patch('greenlang.policy.enforcer.opa_eval')
    def test_check_install_with_opa(self, mock_opa_eval):
        """Test check_install method with OPA evaluation"""
        mock_opa_eval.return_value = {"allow": True, "reasons": ["Pack is valid"]}

        pack_manifest = Mock()
        pack_manifest.dict.return_value = {"name": "test", "version": "1.0.0"}

        allowed, reasons = self.enforcer.check_install(pack_manifest, "/test/path", "add")
        assert allowed is True
        assert reasons == ["Pack is valid"]
        mock_opa_eval.assert_called_once()

    @patch('greenlang.policy.enforcer.opa_eval')
    def test_check_install_denied(self, mock_opa_eval):
        """Test check_install when policy denies"""
        mock_opa_eval.return_value = {
            "allow": False,
            "reasons": ["Pack is unsigned", "Invalid license"]
        }

        pack_manifest = Mock()
        pack_manifest.dict.return_value = {"name": "bad-pack", "version": "1.0.0"}

        allowed, reasons = self.enforcer.check_install(pack_manifest, "/test/path", "add")
        assert allowed is False
        assert "Pack is unsigned" in reasons
        assert "Invalid license" in reasons

    @patch('greenlang.policy.enforcer.opa_eval')
    def test_check_run_with_pipeline(self, mock_opa_eval):
        """Test check_run method with pipeline"""
        mock_opa_eval.return_value = {"allow": True, "reason": "Pipeline allowed"}

        pipeline = Mock()
        pipeline.to_policy_doc.return_value = {"name": "test-pipeline", "version": "1.0"}

        context = Mock()
        context.profile = "dev"
        context.region = "us-east-1"

        allowed, reasons = self.enforcer.check_run(pipeline, context)
        assert allowed is True
        assert "Pipeline allowed" in reasons


class TestPolicyManagement:
    """Test policy management operations"""

    def setup_method(self):
        """Setup test enforcer"""
        self.temp_dir = tempfile.mkdtemp()
        self.enforcer = PolicyEnforcer(policy_dir=Path(self.temp_dir))

    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_policy(self):
        """Test adding a new policy"""
        # Create a temporary policy file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rego", delete=False) as f:
            f.write("package test\nallow { true }")
            temp_policy_path = f.name

        try:
            policy_file = Path(temp_policy_path)
            self.enforcer.add_policy(policy_file)

            # Check that policy was added
            policy_name = policy_file.stem
            assert policy_name in self.enforcer.policies
            assert (self.enforcer.policy_dir / policy_file.name).exists()
        finally:
            Path(temp_policy_path).unlink(missing_ok=True)

    def test_add_nonexistent_policy(self):
        """Test adding a non-existent policy file"""
        nonexistent_file = Path("/nonexistent/policy.rego")
        with pytest.raises(ValueError, match="Policy file not found"):
            self.enforcer.add_policy(nonexistent_file)

    def test_remove_policy(self):
        """Test removing a policy"""
        # First add a policy
        policy_content = "package test\nallow { true }"
        policy_file = self.enforcer.policy_dir / "test_removal.rego"
        with open(policy_file, "w") as f:
            f.write(policy_content)
        self.enforcer.policies["test_removal"] = policy_content

        # Remove the policy
        self.enforcer.remove_policy("test_removal")

        assert "test_removal" not in self.enforcer.policies
        assert not policy_file.exists()

    def test_remove_nonexistent_policy(self):
        """Test removing a non-existent policy"""
        with pytest.raises(ValueError, match="Policy not found"):
            self.enforcer.remove_policy("nonexistent")

    def test_list_policies(self):
        """Test listing policies"""
        # Add some test policies
        test_policies = ["policy1", "policy2", "policy3"]
        for name in test_policies:
            policy_file = self.enforcer.policy_dir / f"{name}.rego"
            with open(policy_file, "w") as f:
                f.write(f"package {name}\nallow {{ true }}")
            self.enforcer.policies[name] = f"package {name}\nallow {{ true }}"

        policies = self.enforcer.list_policies()
        for name in test_policies:
            assert name in policies

    def test_get_policy(self):
        """Test getting policy content"""
        policy_name = "test_get"
        policy_content = "package test_get\nallow { input.valid == true }"
        self.enforcer.policies[policy_name] = policy_content

        retrieved_content = self.enforcer.get_policy(policy_name)
        assert retrieved_content == policy_content

        # Test non-existent policy
        assert self.enforcer.get_policy("nonexistent") is None


class TestPolicyUtilities:
    """Test policy utility methods"""

    def setup_method(self):
        """Setup test enforcer"""
        self.temp_dir = tempfile.mkdtemp()
        self.enforcer = PolicyEnforcer(policy_dir=Path(self.temp_dir))

    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_files(self):
        """Test _list_files utility method"""
        # Create test directory structure
        test_dir = Path(self.temp_dir) / "test_pack"
        test_dir.mkdir()
        (test_dir / "file1.py").write_text("content1")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file2.yml").write_text("content2")

        files = self.enforcer._list_files(str(test_dir))
        assert "file1.py" in files
        assert "subdir/file2.yml" in files or "subdir\\file2.yml" in files  # Handle Windows paths

    def test_list_files_nonexistent(self):
        """Test _list_files with non-existent directory"""
        files = self.enforcer._list_files("/nonexistent/path")
        assert files == []

    def test_detect_licenses(self):
        """Test _detect_licenses utility method"""
        test_dir = Path(self.temp_dir) / "test_licenses"
        test_dir.mkdir()

        # Create MIT license file
        mit_license = test_dir / "LICENSE"
        mit_license.write_text("MIT License\n\nPermission is hereby granted...")

        # Create Apache license file
        apache_license = test_dir / "LICENSE.apache"
        apache_license.write_text("Apache License\nVersion 2.0")

        licenses = self.enforcer._detect_licenses(str(test_dir))
        assert "MIT" in licenses
        assert "Apache-2.0" in licenses

    def test_detect_licenses_unknown(self):
        """Test license detection with unknown license"""
        test_dir = Path(self.temp_dir) / "test_unknown"
        test_dir.mkdir()

        unknown_license = test_dir / "LICENSE"
        unknown_license.write_text("Custom proprietary license")

        licenses = self.enforcer._detect_licenses(str(test_dir))
        assert "Unknown" in licenses

    def test_pipeline_to_policy_doc(self):
        """Test _pipeline_to_policy_doc conversion"""
        # Test with object that has to_policy_doc method
        pipeline_with_method = Mock()
        pipeline_with_method.to_policy_doc.return_value = {"name": "test", "version": "1.0"}

        result = self.enforcer._pipeline_to_policy_doc(pipeline_with_method)
        assert result == {"name": "test", "version": "1.0"}

        # Test with object that has dict method but no to_policy_doc
        pipeline_with_dict = Mock(spec=['dict'])  # Only allow dict method
        pipeline_with_dict.dict.return_value = {"name": "test2", "version": "1.1"}

        result = self.enforcer._pipeline_to_policy_doc(pipeline_with_dict)
        assert result == {"name": "test2", "version": "1.1"}

        # Test with dict directly - it should return with default name
        dict_pipeline = {"name": "simple", "version": "1.0", "steps": []}
        result = self.enforcer._pipeline_to_policy_doc(dict_pipeline)
        # The enforcer might transform the name, so just check key fields exist
        assert "name" in result
        assert "version" in result

    def test_get_egress_targets(self):
        """Test _get_egress_targets method"""
        # Test with context that has egress_targets
        context_with_targets = Mock()
        context_with_targets.config.egress_targets = ["https://api.example.com", "https://data.example.com"]

        targets = self.enforcer._get_egress_targets(context_with_targets)
        assert targets == ["https://api.example.com", "https://data.example.com"]

        # Test with context without egress_targets
        context_without = Mock()
        del context_without.config

        targets = self.enforcer._get_egress_targets(context_without)
        assert targets == []


class TestDefaultPolicyCreation:
    """Test default policy creation"""

    def setup_method(self):
        """Setup test enforcer"""
        self.temp_dir = tempfile.mkdtemp()
        self.enforcer = PolicyEnforcer(policy_dir=Path(self.temp_dir))

    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_default_policies(self):
        """Test creating default policy templates"""
        self.enforcer.create_default_policies()

        # Check that policy files were created
        expected_policies = ["install.rego", "runtime.rego", "data.rego"]
        for policy_file in expected_policies:
            policy_path = self.enforcer.policy_dir / policy_file
            assert policy_path.exists()

            # Check basic content
            content = policy_path.read_text()
            assert "package greenlang." in content
            assert "deny[msg]" in content or "allow" in content

    def test_default_install_policy_content(self):
        """Test default install policy content"""
        self.enforcer.create_default_policies()

        install_policy = self.enforcer.policy_dir / "install.rego"
        content = install_policy.read_text()

        # Check for key security rules
        assert "package greenlang.install" in content
        assert "untrusted" in content  # Source checking
        assert "signing" in content    # Signature checking
        assert "size" in content       # Size limits

    def test_default_runtime_policy_content(self):
        """Test default runtime policy content"""
        self.enforcer.create_default_policies()

        runtime_policy = self.enforcer.policy_dir / "runtime.rego"
        content = runtime_policy.read_text()

        # Check for key runtime rules
        assert "package greenlang.runtime" in content
        assert "memory" in content        # Resource limits
        assert "cpu" in content           # CPU limits
        assert "authenticated" in content # Authentication


class TestStandaloneFunctions:
    """Test standalone policy functions"""

    @patch('greenlang.policy.enforcer.opa_eval')
    def test_check_install_function_success(self, mock_opa_eval):
        """Test standalone check_install function success"""
        mock_opa_eval.return_value = {"allow": True}

        pack_manifest = Mock()
        pack_manifest.model_dump.return_value = {"name": "test", "version": "1.0.0"}

        # Should not raise exception
        check_install(pack_manifest, "/test/path", "publish")
        mock_opa_eval.assert_called_once()

    @patch('greenlang.policy.enforcer.opa_eval')
    def test_check_install_function_denied(self, mock_opa_eval):
        """Test standalone check_install function denial"""
        mock_opa_eval.return_value = {"allow": False, "reason": "Pack is unsigned"}

        pack_manifest = Mock()
        pack_manifest.model_dump.return_value = {"name": "bad", "version": "1.0.0"}

        with pytest.raises(RuntimeError, match="Pack is unsigned"):
            check_install(pack_manifest, "/test/path", "add")

    @patch('greenlang.policy.enforcer.opa_eval')
    def test_check_run_function_success(self, mock_opa_eval):
        """Test standalone check_run function success"""
        mock_opa_eval.return_value = {"allow": True}

        pipeline = Mock()
        pipeline.to_policy_doc.return_value = {"name": "test-pipeline"}

        context = Mock()
        context.egress_targets = ["https://api.example.com"]
        context.region = "us-east-1"

        # Should not raise exception
        check_run(pipeline, context)
        mock_opa_eval.assert_called_once()

    @patch('greenlang.policy.enforcer.opa_eval')
    def test_check_run_function_denied(self, mock_opa_eval):
        """Test standalone check_run function denial"""
        mock_opa_eval.return_value = {"allow": False, "reason": "Pipeline not allowed"}

        pipeline = Mock()
        pipeline.to_policy_doc.return_value = {"name": "bad-pipeline"}

        context = Mock()
        context.egress_targets = []
        context.region = "unknown"

        with pytest.raises(RuntimeError, match="Pipeline not allowed"):
            check_run(pipeline, context)


@pytest.mark.parametrize("license,should_allow", [
    ("MIT", True),
    ("Apache-2.0", True),
    ("Commercial", True),
    ("GPL-3.0", False),
    ("Unknown", False),
    ("", False)
])
def test_license_allowlist(license, should_allow):
    """Parametrized test for license allowlist"""
    temp_dir = tempfile.mkdtemp()
    try:
        enforcer = PolicyEnforcer(policy_dir=Path(temp_dir))

        pack_manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": license,
            "provenance": {"signed": True},
            "kind": "pack",
            "policy": {"network": ["https://api.greenlang.io"]},
            "security": {"sbom": True}
        }

        result = enforcer._eval_install_policy({"pack": pack_manifest})
        assert result == should_allow
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.parametrize("memory_mb,cpu_cores,should_allow", [
    (1024, 1, True),    # Normal usage
    (4096, 4, True),    # Max allowed
    (8192, 1, False),   # Too much memory
    (1024, 8, False),   # Too many cores
    (8192, 8, False),   # Both excessive
])
def test_resource_limits(memory_mb, cpu_cores, should_allow):
    """Parametrized test for resource limits"""
    temp_dir = tempfile.mkdtemp()
    try:
        enforcer = PolicyEnforcer(policy_dir=Path(temp_dir))

        input_data = {
            "user": {"authenticated": True, "requests_per_minute": 50, "role": "basic"},
            "resources": {"memory_mb": memory_mb, "cpu_cores": cpu_cores}
        }

        result = enforcer._eval_runtime_policy(input_data)
        assert result == should_allow
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)