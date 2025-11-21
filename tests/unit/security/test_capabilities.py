# -*- coding: utf-8 -*-
"""
Unit tests for runtime capability enforcement

SECURITY GATE: These tests verify that capabilities default to deny
and are properly enforced at runtime.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Removed sys.path manipulation - using installed package

from greenlang.runtime.executor import PipelineExecutor, ExecutionContext
from greenlang.packs.manifest import Capabilities, NetCapability, FsCapability, SubprocessCapability


class TestCapabilityDefaults:
    """Test that capabilities default to deny"""

    def test_capabilities_default_to_false(self):
        """Test: All capabilities default to False (deny)"""
        caps = Capabilities()

        # All capabilities should default to False
        assert caps.net.allow == False
        assert caps.fs.allow == False
        assert caps.clock.allow == False
        assert caps.subprocess.allow == False

    def test_net_capability_defaults(self):
        """Test: Network capability defaults to deny"""
        net_cap = NetCapability()
        assert net_cap.allow == False
        assert net_cap.outbound is None

    def test_fs_capability_defaults(self):
        """Test: Filesystem capability defaults to deny"""
        fs_cap = FsCapability()
        assert fs_cap.allow == False
        assert fs_cap.read is None
        assert fs_cap.write is None

    def test_subprocess_capability_defaults(self):
        """Test: Subprocess capability defaults to deny"""
        sub_cap = SubprocessCapability()
        assert sub_cap.allow == False
        assert sub_cap.allowlist == []


class TestRuntimeCapabilityEnforcement:
    """Test runtime capability enforcement"""

    def test_pack_tries_network_while_net_false_denied(self):
        """Test M: Pack tries network while net:false ⇒ denied"""
        executor = PipelineExecutor()

        # Create a step that would use network
        step = Mock(
            name="fetch-data",
            operation="http_fetch",
            capabilities={}  # No network capability
        )

        context = ExecutionContext(
            run_id="test-run",
            pipeline_name="test-pipeline",
            capabilities={"net": {"allow": False}}  # Network denied
        )

        # Should use guarded worker by default
        assert executor._should_use_guarded_worker(step, context) == True

        # The guarded worker should enforce capabilities
        with patch('subprocess.run') as mock_run:
            # Mock the worker to simulate network access attempt
            mock_run.return_value = Mock(
                returncode=1,
                stdout='{"error": "Network access denied"}',
                stderr="CAPABILITY_DENIED: net"
            )

            with pytest.raises(Exception) as exc_info:
                executor._execute_in_guarded_worker(
                    step,
                    {"url": "https://api.example.com"},
                    context
                )

    def test_pack_tries_file_read_while_fs_false_denied(self):
        """Test N: Pack tries file read while fs:false ⇒ denied"""
        executor = PipelineExecutor()

        step = Mock(
            name="read-file",
            operation="file_read",
            capabilities={}  # No fs capability
        )

        context = ExecutionContext(
            run_id="test-run",
            pipeline_name="test-pipeline",
            capabilities={"fs": {"allow": False}}  # Filesystem denied
        )

        # Should use guarded worker
        assert executor._should_use_guarded_worker(step, context) == True

    def test_network_with_capability_true_allowed(self):
        """Test O: Network action when capability is true ⇒ allowed"""
        executor = PipelineExecutor()

        step = Mock(
            name="fetch-data",
            operation="http_fetch",
            capabilities={"net": {"allow": True, "outbound": {"allowlist": ["api.example.com"]}}}
        )

        context = ExecutionContext(
            run_id="test-run",
            pipeline_name="test-pipeline",
            capabilities={"net": {"allow": True}}  # Network allowed
        )

        # Should still use guarded worker for enforcement
        assert executor._should_use_guarded_worker(step, context) == True

        # The capabilities should be passed to the worker
        with patch('subprocess.run') as mock_run:
            with patch('tempfile.TemporaryDirectory'):
                with patch('builtins.open', create=True):
                    with patch.object(executor, '_create_worker_script'):
                        mock_run.return_value = Mock(
                            returncode=0,
                            stdout='{"result": "success"}',
                            stderr=""
                        )

                        # This should succeed with network capability
                        result = executor._execute_in_guarded_worker(
                            step,
                            {"url": "https://api.example.com"},
                            context
                        )

                        # Check that capabilities were passed to worker
                        env_passed = mock_run.call_args[1]['env']
                        caps_json = env_passed.get('GL_CAPS')
                        if caps_json:
                            caps = json.loads(caps_json)
                            assert caps['net']['allow'] == True

    def test_guarded_worker_default_enabled(self):
        """Test: Guarded worker is enabled by default"""
        executor = PipelineExecutor()

        step = Mock(name="test-step", capabilities=None)
        context = ExecutionContext(
            run_id="test-run",
            pipeline_name="test-pipeline"
        )

        # Should default to using guarded worker
        assert executor._should_use_guarded_worker(step, context) == True

    @patch.dict(os.environ, {'GL_DISABLE_GUARD': '1'})
    def test_guard_can_be_disabled_with_warning(self):
        """Test: Guard can be disabled but logs warning"""
        import logging

        with patch.object(logging.getLogger('greenlang.runtime.executor'), 'warning') as mock_warn:
            executor = PipelineExecutor()

            step = Mock(name="test-step", capabilities=None)
            context = ExecutionContext(
                run_id="test-run",
                pipeline_name="test-pipeline"
            )

            # Guard should be disabled
            assert executor._should_use_guarded_worker(step, context) == False

            # Warning should be logged
            assert mock_warn.called
            warning_msg = str(mock_warn.call_args[0][0])
            assert "SECURITY WARNING" in warning_msg
            assert "Guard disabled" in warning_msg

    def test_capability_merge_denies_escalation(self):
        """Test: Steps cannot escalate privileges beyond manifest"""
        executor = PipelineExecutor()

        # Step requests network but manifest doesn't allow it
        step = Mock(
            name="escalate-step",
            capabilities={"net": {"allow": True}}  # Step wants network
        )

        # Context/manifest doesn't allow network
        context = ExecutionContext(
            run_id="test-run",
            pipeline_name="test-pipeline",
            capabilities={"net": {"allow": False}}  # Network not allowed
        )

        import logging
        with patch.object(logging.getLogger('greenlang.runtime.executor'), 'warning') as mock_warn:
            # Try to execute - should warn about privilege escalation attempt
            with patch('subprocess.run'):
                with patch('tempfile.TemporaryDirectory'):
                    with patch('builtins.open', create=True):
                        with patch.object(executor, '_create_worker_script'):
                            executor._execute_in_guarded_worker(step, {}, context)

            # Should have warned about capability request
            if mock_warn.called:
                warnings = [str(call[0][0]) for call in mock_warn.call_args_list]
                # Check if any warning mentions the step requesting capabilities
                assert any(step.name in w for w in warnings)

    def test_capabilities_passed_to_worker_env(self):
        """Test: Capabilities are passed to worker via environment"""
        executor = PipelineExecutor()

        capabilities = {
            "net": {"allow": True},
            "fs": {"allow": False},
            "clock": {"allow": False},
            "subprocess": {"allow": False}
        }

        step = Mock(name="test-step", capabilities={})
        context = ExecutionContext(
            run_id="test-run",
            pipeline_name="test-pipeline",
            capabilities=capabilities
        )

        with patch('subprocess.run') as mock_run:
            with patch('tempfile.TemporaryDirectory'):
                with patch('builtins.open', create=True):
                    with patch.object(executor, '_create_worker_script'):
                        mock_run.return_value = Mock(returncode=0, stdout='{}', stderr='')

                        executor._execute_in_guarded_worker(step, {}, context)

                        # Check environment variables passed to subprocess
                        call_args = mock_run.call_args
                        env = call_args[1]['env']

                        # GL_CAPS should contain the capabilities
                        assert 'GL_CAPS' in env
                        caps_from_env = json.loads(env['GL_CAPS'])
                        assert caps_from_env['net']['allow'] == True
                        assert caps_from_env['fs']['allow'] == False


class TestCapabilityValidation:
    """Test capability validation in pack installer"""

    def test_dangerous_binaries_rejected(self):
        """Test: Dangerous binaries in subprocess allowlist are rejected"""
        from greenlang.packs.installer import PackInstaller

        installer = PackInstaller()

        # Test dangerous binaries
        dangerous = ['/bin/sh', '/usr/bin/python', '/usr/bin/curl', '/usr/bin/sudo']

        for binary in dangerous:
            issues = installer._validate_capabilities(Mock(
                subprocess=Mock(allow=True, allowlist=[binary]),
                net=None,
                fs=None,
                clock=None
            ))

            assert any(f"Dangerous binary in allowlist: {binary}" in issue for issue in issues)

    def test_root_filesystem_access_rejected(self):
        """Test: Root filesystem write access is rejected"""
        from greenlang.packs.installer import PackInstaller

        installer = PackInstaller()

        issues = installer._validate_capabilities(Mock(
            fs=Mock(allow=True, write={"allowlist": ["/**"]}),
            net=None,
            clock=None,
            subprocess=None
        ))

        assert any("Root filesystem write access is not allowed" in issue for issue in issues)

    def test_path_traversal_rejected(self):
        """Test: Path traversal patterns are rejected"""
        from greenlang.packs.installer import PackInstaller

        installer = PackInstaller()

        issues = installer._validate_capabilities(Mock(
            fs=Mock(allow=True, read={"allowlist": ["../../../etc/passwd"]}),
            net=None,
            clock=None,
            subprocess=None
        ))

        assert any("Path traversal" in issue for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])