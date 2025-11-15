"""
Secure Deployment - Enhanced subprocess execution with input validation.

This module provides secure command execution for deployment operations,
preventing command injection attacks.

Example:
    >>> from factory.deployment_secure import SecureCommandExecutor
    >>> executor = SecureCommandExecutor()
    >>> result = executor.execute_kubectl("get", "pods", namespace="default")
"""

import subprocess
import shlex
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
from pydantic import BaseModel, Field, validator

from security.input_validation import InputValidator, SafeCommandInput

logger = logging.getLogger(__name__)


class SecureCommandExecutor:
    """
    Secure command executor with input validation.

    All commands are validated against whitelist.
    All arguments are validated for injection patterns.
    Commands are executed with shell=False for safety.

    Example:
        >>> executor = SecureCommandExecutor()
        >>> result = executor.execute_kubectl("get", "pods")
    """

    # Whitelisted commands
    ALLOWED_KUBECTL_COMMANDS = {
        'get', 'describe', 'logs', 'apply', 'delete', 'create',
        'rollout', 'scale', 'exec', 'port-forward'
    }

    ALLOWED_DOCKER_COMMANDS = {
        'build', 'push', 'pull', 'tag', 'images', 'ps', 'inspect'
    }

    ALLOWED_HELM_COMMANDS = {
        'install', 'upgrade', 'uninstall', 'list', 'status', 'rollback'
    }

    def __init__(self):
        """Initialize secure command executor."""
        self.validator = InputValidator()

    def execute_kubectl(
        self,
        command: str,
        resource_type: Optional[str] = None,
        resource_name: Optional[str] = None,
        namespace: str = "default",
        extra_args: Optional[List[str]] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute kubectl command with validation.

        Args:
            command: kubectl subcommand (get, apply, delete, etc.)
            resource_type: Resource type (pod, deployment, service, etc.)
            resource_name: Resource name
            namespace: Kubernetes namespace
            extra_args: Additional arguments
            timeout: Command timeout in seconds

        Returns:
            Dict with stdout, stderr, return_code

        Raises:
            ValueError: If validation fails
            subprocess.TimeoutExpired: If command times out
            subprocess.CalledProcessError: If command fails

        Example:
            >>> executor.execute_kubectl("get", "pods", namespace="default")
        """
        # Validate command against whitelist
        if command not in self.ALLOWED_KUBECTL_COMMANDS:
            raise ValueError(
                f"kubectl command '{command}' not allowed. Allowed: {self.ALLOWED_KUBECTL_COMMANDS}"
            )

        # Build arguments
        args = ["kubectl", command]

        # Add resource type and name
        if resource_type:
            # Validate resource type (alphanumeric only)
            validated_type = self.validator.validate_alphanumeric(
                resource_type, "resource_type", max_length=50
            )
            args.append(validated_type)

        if resource_name:
            # Validate resource name
            validated_name = self.validator.validate_alphanumeric(
                resource_name, "resource_name", max_length=253
            )
            args.append(validated_name)

        # Add namespace
        validated_namespace = self.validator.validate_alphanumeric(
            namespace, "namespace", max_length=63
        )
        args.extend(["-n", validated_namespace])

        # Add extra arguments (validated)
        if extra_args:
            for arg in extra_args:
                self.validator.validate_no_command_injection(arg, "extra_arg")
                args.append(arg)

        # Execute command
        return self._execute_command(args, timeout=timeout)

    def execute_docker(
        self,
        command: str,
        image: Optional[str] = None,
        tag: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Execute docker command with validation.

        Args:
            command: docker subcommand
            image: Docker image name
            tag: Image tag
            extra_args: Additional arguments
            timeout: Command timeout

        Returns:
            Dict with stdout, stderr, return_code

        Example:
            >>> executor.execute_docker("build", "myapp", tag="v1.0")
        """
        # Validate command
        if command not in self.ALLOWED_DOCKER_COMMANDS:
            raise ValueError(
                f"docker command '{command}' not allowed. Allowed: {self.ALLOWED_DOCKER_COMMANDS}"
            )

        # Build arguments
        args = ["docker", command]

        if image:
            # Validate image name (allow alphanumeric, hyphen, underscore, slash, dot)
            if not self._is_valid_image_name(image):
                raise ValueError(f"Invalid docker image name: {image}")
            args.append(image)

        if tag:
            # Validate tag
            validated_tag = self.validator.validate_alphanumeric(tag, "tag", max_length=128)
            args[-1] = f"{args[-1]}:{validated_tag}"

        # Add extra arguments
        if extra_args:
            for arg in extra_args:
                self.validator.validate_no_command_injection(arg, "extra_arg")
                args.append(arg)

        # Execute command
        return self._execute_command(args, timeout=timeout)

    def execute_helm(
        self,
        command: str,
        release_name: Optional[str] = None,
        chart: Optional[str] = None,
        namespace: str = "default",
        values_file: Optional[Path] = None,
        extra_args: Optional[List[str]] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Execute helm command with validation.

        Args:
            command: helm subcommand
            release_name: Release name
            chart: Chart name or path
            namespace: Kubernetes namespace
            values_file: Path to values file
            extra_args: Additional arguments
            timeout: Command timeout

        Returns:
            Dict with stdout, stderr, return_code

        Example:
            >>> executor.execute_helm("install", "myapp", "stable/nginx")
        """
        # Validate command
        if command not in self.ALLOWED_HELM_COMMANDS:
            raise ValueError(
                f"helm command '{command}' not allowed. Allowed: {self.ALLOWED_HELM_COMMANDS}"
            )

        # Build arguments
        args = ["helm", command]

        if release_name:
            validated_release = self.validator.validate_alphanumeric(
                release_name, "release_name", max_length=253
            )
            args.append(validated_release)

        if chart:
            # Validate chart name/path
            if "/" in chart:
                # It's a path, validate it
                validated_chart = str(self.validator.validate_path(
                    chart, must_exist=True, allow_relative=True
                ))
            else:
                # It's a chart name, validate alphanumeric with slash
                if not self._is_valid_chart_name(chart):
                    raise ValueError(f"Invalid chart name: {chart}")
                validated_chart = chart
            args.append(validated_chart)

        # Add namespace
        validated_namespace = self.validator.validate_alphanumeric(
            namespace, "namespace", max_length=63
        )
        args.extend(["-n", validated_namespace])

        # Add values file
        if values_file:
            validated_path = self.validator.validate_path(
                str(values_file),
                must_exist=True,
                allowed_extensions=['.yaml', '.yml']
            )
            args.extend(["-f", str(validated_path)])

        # Add extra arguments
        if extra_args:
            for arg in extra_args:
                self.validator.validate_no_command_injection(arg, "extra_arg")
                args.append(arg)

        # Execute command
        return self._execute_command(args, timeout=timeout)

    def _execute_command(
        self,
        args: List[str],
        timeout: int = 60,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute command with security measures.

        Args:
            args: Command arguments (already validated)
            timeout: Timeout in seconds
            cwd: Working directory
            env: Environment variables

        Returns:
            Dict with execution results
        """
        logger.info(
            f"Executing command: {' '.join(args[:3])}...",
            extra={"command": args[0], "args_count": len(args)}
        )

        try:
            # Execute with shell=False for security
            result = subprocess.run(
                args,
                shell=False,  # CRITICAL: Never use shell=True
                check=False,  # Don't raise on non-zero exit
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env
            )

            logger.info(
                f"Command completed: {args[0]}",
                extra={
                    "command": args[0],
                    "return_code": result.returncode,
                    "stdout_length": len(result.stdout),
                    "stderr_length": len(result.stderr)
                }
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": args[0]
            }

        except subprocess.TimeoutExpired as e:
            logger.error(
                f"Command timeout: {args[0]}",
                extra={"command": args[0], "timeout": timeout}
            )
            raise

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Command failed: {args[0]}",
                extra={
                    "command": args[0],
                    "return_code": e.returncode,
                    "stderr": e.stderr
                }
            )
            raise

        except Exception as e:
            logger.error(
                f"Unexpected error executing command: {args[0]}",
                extra={"command": args[0], "error": str(e)}
            )
            raise

    def _is_valid_image_name(self, image: str) -> bool:
        """
        Validate docker image name format.

        Allows: alphanumeric, hyphen, underscore, slash, dot
        Example: registry.io/namespace/image-name
        """
        import re
        pattern = re.compile(r'^[a-zA-Z0-9._/-]+$')
        return bool(pattern.match(image)) and len(image) <= 255

    def _is_valid_chart_name(self, chart: str) -> bool:
        """
        Validate helm chart name format.

        Allows: alphanumeric, hyphen, slash
        Example: stable/nginx
        """
        import re
        pattern = re.compile(r'^[a-zA-Z0-9-/]+$')
        return bool(pattern.match(chart)) and len(chart) <= 255


class SecureDeploymentManager:
    """
    High-level deployment manager with security.

    Example:
        >>> manager = SecureDeploymentManager()
        >>> await manager.deploy_to_kubernetes("myapp", "v1.0", namespace="prod")
    """

    def __init__(self):
        """Initialize deployment manager."""
        self.executor = SecureCommandExecutor()
        self.validator = InputValidator()

    def deploy_agent(
        self,
        agent_name: str,
        image_tag: str,
        namespace: str = "greenlang-agents",
        replicas: int = 1,
        manifest_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Deploy agent to Kubernetes.

        Args:
            agent_name: Agent name
            image_tag: Docker image tag
            namespace: K8s namespace
            replicas: Number of replicas
            manifest_path: Path to K8s manifest

        Returns:
            Deployment result

        Example:
            >>> manager.deploy_agent("carbon-agent", "v1.0", namespace="prod")
        """
        # Validate inputs
        validated_name = self.validator.validate_alphanumeric(
            agent_name, "agent_name", max_length=253
        )
        validated_tag = self.validator.validate_alphanumeric(
            image_tag, "image_tag", max_length=128
        )
        validated_namespace = self.validator.validate_alphanumeric(
            namespace, "namespace", max_length=63
        )
        validated_replicas = self.validator.validate_integer(
            replicas, "replicas", min_value=1, max_value=100
        )

        # If manifest provided, apply it
        if manifest_path:
            validated_path = self.validator.validate_path(
                str(manifest_path),
                must_exist=True,
                allowed_extensions=['.yaml', '.yml']
            )

            result = self.executor.execute_kubectl(
                "apply",
                extra_args=["-f", str(validated_path)],
                namespace=validated_namespace
            )

            return result

        # Otherwise, create deployment imperatively (less preferred)
        logger.warning("Creating deployment imperatively without manifest")

        # Create deployment
        image_name = f"greenlang/{validated_name}:{validated_tag}"

        result = self.executor.execute_kubectl(
            "create",
            "deployment",
            validated_name,
            namespace=validated_namespace,
            extra_args=[f"--image={image_name}", f"--replicas={validated_replicas}"]
        )

        return result

    def rollback_deployment(
        self,
        deployment_name: str,
        namespace: str = "greenlang-agents",
        revision: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rollback deployment to previous revision.

        Args:
            deployment_name: Deployment name
            namespace: K8s namespace
            revision: Specific revision to rollback to

        Returns:
            Rollback result
        """
        # Validate inputs
        validated_name = self.validator.validate_alphanumeric(
            deployment_name, "deployment_name", max_length=253
        )
        validated_namespace = self.validator.validate_alphanumeric(
            namespace, "namespace", max_length=63
        )

        # Build arguments
        extra_args = []
        if revision is not None:
            validated_revision = self.validator.validate_integer(
                revision, "revision", min_value=1
            )
            extra_args.append(f"--to-revision={validated_revision}")

        # Execute rollback
        result = self.executor.execute_kubectl(
            "rollout",
            "undo",
            f"deployment/{validated_name}",
            namespace=validated_namespace,
            extra_args=extra_args
        )

        return result

    def scale_deployment(
        self,
        deployment_name: str,
        replicas: int,
        namespace: str = "greenlang-agents"
    ) -> Dict[str, Any]:
        """
        Scale deployment to specified replicas.

        Args:
            deployment_name: Deployment name
            replicas: Target replica count
            namespace: K8s namespace

        Returns:
            Scale result
        """
        # Validate inputs
        validated_name = self.validator.validate_alphanumeric(
            deployment_name, "deployment_name", max_length=253
        )
        validated_replicas = self.validator.validate_integer(
            replicas, "replicas", min_value=0, max_value=100
        )
        validated_namespace = self.validator.validate_alphanumeric(
            namespace, "namespace", max_length=63
        )

        # Execute scale
        result = self.executor.execute_kubectl(
            "scale",
            "deployment",
            validated_name,
            namespace=validated_namespace,
            extra_args=[f"--replicas={validated_replicas}"]
        )

        return result
