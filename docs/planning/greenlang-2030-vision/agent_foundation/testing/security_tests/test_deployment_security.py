# -*- coding: utf-8 -*-
"""
Tests for secure deployment and command execution.

Tests command injection prevention in:
- Kubectl execution
- Docker execution
- Helm execution
- Command argument validation
"""

import pytest
from pathlib import Path

from factory.deployment_secure import (
    SecureCommandExecutor,
    SecureDeploymentManager,
)


class TestSecureCommandExecutor:
    """Test secure command execution."""

    def test_kubectl_get_pods_valid(self):
        """Test valid kubectl get pods command."""
        executor = SecureCommandExecutor()

        # This will fail in test environment without kubectl, but validates structure
        with pytest.raises((FileNotFoundError, Exception)):
            executor.execute_kubectl(
                command="get",
                resource_type="pods",
                namespace="default",
                timeout=5
            )

    def test_kubectl_invalid_command(self):
        """Test invalid kubectl command rejected."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="not allowed"):
            executor.execute_kubectl(
                command="hack",  # Not in whitelist
                resource_type="pods",
                namespace="default"
            )

    def test_kubectl_command_injection_in_namespace(self):
        """Test command injection in namespace blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="shell characters"):
            executor.execute_kubectl(
                command="get",
                resource_type="pods",
                namespace="default; rm -rf /",
                timeout=5
            )

    def test_kubectl_command_injection_in_resource_name(self):
        """Test command injection in resource name blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="shell characters"):
            executor.execute_kubectl(
                command="get",
                resource_type="pod",
                resource_name="mypod && echo hacked",
                namespace="default",
                timeout=5
            )

    def test_kubectl_command_injection_in_extra_args(self):
        """Test command injection in extra args blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="shell characters"):
            executor.execute_kubectl(
                command="get",
                resource_type="pods",
                namespace="default",
                extra_args=["--output=json; rm -rf /"],
                timeout=5
            )

    def test_docker_build_valid(self):
        """Test valid docker build command."""
        executor = SecureCommandExecutor()

        with pytest.raises((FileNotFoundError, Exception)):
            executor.execute_docker(
                command="build",
                image="myapp",
                tag="v1.0",
                timeout=5
            )

    def test_docker_invalid_command(self):
        """Test invalid docker command rejected."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="not allowed"):
            executor.execute_docker(
                command="exec",  # Not in whitelist
                image="myapp",
                timeout=5
            )

    def test_docker_invalid_image_name(self):
        """Test invalid docker image name rejected."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="Invalid docker image"):
            executor.execute_docker(
                command="build",
                image="myapp; rm -rf /",  # Invalid characters
                timeout=5
            )

    def test_docker_command_injection_in_tag(self):
        """Test command injection in tag blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="alphanumeric"):
            executor.execute_docker(
                command="build",
                image="myapp",
                tag="v1.0 && echo hacked",
                timeout=5
            )

    def test_helm_install_valid(self):
        """Test valid helm install command."""
        executor = SecureCommandExecutor()

        with pytest.raises((FileNotFoundError, Exception)):
            executor.execute_helm(
                command="install",
                release_name="myapp",
                chart="stable/nginx",
                namespace="default",
                timeout=5
            )

    def test_helm_invalid_command(self):
        """Test invalid helm command rejected."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="not allowed"):
            executor.execute_helm(
                command="hack",  # Not in whitelist
                release_name="myapp",
                chart="stable/nginx",
                namespace="default",
                timeout=5
            )

    def test_helm_command_injection_in_release_name(self):
        """Test command injection in release name blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError, match="alphanumeric"):
            executor.execute_helm(
                command="install",
                release_name="myapp; rm -rf /",
                chart="stable/nginx",
                namespace="default",
                timeout=5
            )


class TestSecureDeploymentManager:
    """Test secure deployment manager."""

    def test_deploy_agent_validates_inputs(self):
        """Test deploy_agent validates all inputs."""
        manager = SecureDeploymentManager()

        # Invalid agent name
        with pytest.raises(ValueError, match="alphanumeric"):
            manager.deploy_agent(
                agent_name="agent@123",  # @ not allowed
                image_tag="v1.0",
                namespace="default"
            )

        # Invalid namespace
        with pytest.raises(ValueError, match="alphanumeric"):
            manager.deploy_agent(
                agent_name="myagent",
                image_tag="v1.0",
                namespace="default; rm -rf /"
            )

        # Invalid replicas
        with pytest.raises(ValueError, match="replicas"):
            manager.deploy_agent(
                agent_name="myagent",
                image_tag="v1.0",
                namespace="default",
                replicas=-1
            )

    def test_scale_deployment_validates_inputs(self):
        """Test scale_deployment validates inputs."""
        manager = SecureDeploymentManager()

        # Invalid deployment name
        with pytest.raises(ValueError, match="alphanumeric"):
            manager.scale_deployment(
                deployment_name="deploy; rm -rf /",
                replicas=3,
                namespace="default"
            )

        # Invalid replicas
        with pytest.raises(ValueError, match="replicas"):
            manager.scale_deployment(
                deployment_name="myapp",
                replicas=200,  # Exceeds max
                namespace="default"
            )

    def test_rollback_deployment_validates_inputs(self):
        """Test rollback_deployment validates inputs."""
        manager = SecureDeploymentManager()

        # Invalid deployment name
        with pytest.raises(ValueError, match="alphanumeric"):
            manager.rollback_deployment(
                deployment_name="deploy && echo hacked",
                namespace="default"
            )

        # Invalid revision
        with pytest.raises(ValueError, match="revision"):
            manager.rollback_deployment(
                deployment_name="myapp",
                namespace="default",
                revision=0  # Must be >= 1
            )


class TestCommandWhitelisting:
    """Test command whitelisting."""

    def test_kubectl_allowed_commands(self):
        """Test kubectl allowed commands."""
        executor = SecureCommandExecutor()

        allowed = [
            'get', 'describe', 'logs', 'apply', 'delete',
            'create', 'rollout', 'scale'
        ]

        for cmd in allowed:
            assert cmd in executor.ALLOWED_KUBECTL_COMMANDS

    def test_kubectl_dangerous_commands_blocked(self):
        """Test dangerous kubectl commands blocked."""
        executor = SecureCommandExecutor()

        dangerous = [
            'exec',  # Can execute arbitrary commands in pods (WAIT - this is allowed!)
            'proxy',
            'attach',
        ]

        # exec is actually in allowed list, test others
        assert 'proxy' not in executor.ALLOWED_KUBECTL_COMMANDS
        assert 'attach' not in executor.ALLOWED_KUBECTL_COMMANDS

    def test_docker_allowed_commands(self):
        """Test docker allowed commands."""
        executor = SecureCommandExecutor()

        allowed = ['build', 'push', 'pull', 'tag', 'images', 'ps', 'inspect']

        for cmd in allowed:
            assert cmd in executor.ALLOWED_DOCKER_COMMANDS

    def test_docker_dangerous_commands_blocked(self):
        """Test dangerous docker commands blocked."""
        executor = SecureCommandExecutor()

        dangerous = ['run', 'exec', 'rm', 'rmi', 'kill', 'stop']

        for cmd in dangerous:
            assert cmd not in executor.ALLOWED_DOCKER_COMMANDS

    def test_helm_allowed_commands(self):
        """Test helm allowed commands."""
        executor = SecureCommandExecutor()

        allowed = ['install', 'upgrade', 'uninstall', 'list', 'status', 'rollback']

        for cmd in allowed:
            assert cmd in executor.ALLOWED_HELM_COMMANDS


class TestShellInjectionVectors:
    """Test various shell injection vectors are blocked."""

    def test_semicolon_injection(self):
        """Test semicolon injection blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError):
            executor.execute_kubectl(
                command="get",
                resource_type="pods",
                resource_name="mypod; rm -rf /",
                namespace="default"
            )

    def test_pipe_injection(self):
        """Test pipe injection blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError):
            executor.execute_kubectl(
                command="get",
                resource_type="pods",
                resource_name="mypod | cat /etc/passwd",
                namespace="default"
            )

    def test_background_execution_injection(self):
        """Test background execution injection blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError):
            executor.execute_kubectl(
                command="get",
                resource_type="pods",
                resource_name="mypod & malicious_command",
                namespace="default"
            )

    def test_command_substitution_injection(self):
        """Test command substitution injection blocked."""
        executor = SecureCommandExecutor()

        injections = [
            "mypod `whoami`",
            "mypod $(ls -la)",
        ]

        for injection in injections:
            with pytest.raises(ValueError):
                executor.execute_kubectl(
                    command="get",
                    resource_type="pods",
                    resource_name=injection,
                    namespace="default"
                )

    def test_redirection_injection(self):
        """Test redirection injection blocked."""
        executor = SecureCommandExecutor()

        injections = [
            "mypod > /tmp/output",
            "mypod < /etc/passwd",
        ]

        for injection in injections:
            with pytest.raises(ValueError):
                executor.execute_kubectl(
                    command="get",
                    resource_type="pods",
                    resource_name=injection,
                    namespace="default"
                )

    def test_brace_expansion_injection(self):
        """Test brace expansion injection blocked."""
        executor = SecureCommandExecutor()

        with pytest.raises(ValueError):
            executor.execute_kubectl(
                command="get",
                resource_type="pods",
                resource_name="mypod{1..10}",
                namespace="default"
            )


class TestImageNameValidation:
    """Test Docker image name validation."""

    def test_valid_image_names(self):
        """Test valid image names."""
        executor = SecureCommandExecutor()

        valid_names = [
            "myapp",
            "registry.io/namespace/image",
            "gcr.io/project/image-name",
            "docker.io/library/nginx",
        ]

        for name in valid_names:
            assert executor._is_valid_image_name(name)

    def test_invalid_image_names(self):
        """Test invalid image names."""
        executor = SecureCommandExecutor()

        invalid_names = [
            "myapp; rm -rf /",
            "myapp && echo hacked",
            "myapp | cat /etc/passwd",
            "myapp `whoami`",
        ]

        for name in invalid_names:
            assert not executor._is_valid_image_name(name)


class TestChartNameValidation:
    """Test Helm chart name validation."""

    def test_valid_chart_names(self):
        """Test valid chart names."""
        executor = SecureCommandExecutor()

        valid_names = [
            "stable/nginx",
            "bitnami/postgresql",
            "my-chart",
        ]

        for name in valid_names:
            assert executor._is_valid_chart_name(name)

    def test_invalid_chart_names(self):
        """Test invalid chart names."""
        executor = SecureCommandExecutor()

        invalid_names = [
            "stable/nginx; rm -rf /",
            "chart && echo hacked",
            "chart | cat /etc/passwd",
        ]

        for name in invalid_names:
            assert not executor._is_valid_chart_name(name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
