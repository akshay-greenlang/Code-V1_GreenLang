"""
Resource Limits - Agent Factory Sandbox (INFRA-010) - v1 Scope

Defines resource limit dataclasses and an enforcer for sandbox
executions.  Provides environment-specific defaults for dev, staging,
and production deployments.

v1 enforces limits via ``resource.setrlimit`` on Linux with best-effort
fallback on other platforms.  cgroup-based enforcement is not included
here -- K8s pod resource limits handle that at the container level for
dedicated-mode agents.

Classes:
    - ResourceLimits: Frozen dataclass of execution resource limits.
    - ResourceLimitEnforcer: Validates and applies resource limits.
    - ResourceLimitValidationError: Raised on invalid limit values.

Example:
    >>> limits = ResourceLimits.for_environment("prod")
    >>> enforcer = ResourceLimitEnforcer(limits)
    >>> enforcer.validate()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource Limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceLimits:
    """Frozen resource limits for a sandboxed execution.

    Attributes:
        cpu_limit_cores: Maximum CPU cores the execution may use.
        memory_limit_mb: Maximum memory in megabytes.
        disk_io_limit_mbps: Maximum disk I/O throughput in MB/s.
        max_execution_seconds: Maximum wall-clock execution time in seconds.
        temp_dir_size_mb: Maximum temp directory size in megabytes.
        max_open_files: Maximum number of open file descriptors.
        max_processes: Maximum number of child processes.
        network_enabled: Whether network access is allowed.
    """

    cpu_limit_cores: float = 4.0
    memory_limit_mb: int = 2048
    disk_io_limit_mbps: int = 200
    max_execution_seconds: float = 600.0
    temp_dir_size_mb: int = 1024
    max_open_files: int = 1024
    max_processes: int = 64
    network_enabled: bool = False

    # ------------------------------------------------------------------
    # Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def for_environment(cls, environment: str) -> ResourceLimits:
        """Create resource limits appropriate for the given environment.

        Args:
            environment: One of 'dev', 'staging', 'prod'.

        Returns:
            ResourceLimits with environment-specific defaults.
        """
        env = environment.strip().lower()
        profiles = _ENVIRONMENT_PROFILES.get(env)
        if profiles is None:
            logger.warning(
                "Unknown environment '%s', falling back to dev limits", environment,
            )
            profiles = _ENVIRONMENT_PROFILES["dev"]
        return cls(**profiles)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResourceLimits:
        """Create resource limits from a dictionary.

        Unknown keys are silently ignored.

        Args:
            data: Dictionary of limit fields.

        Returns:
            ResourceLimits instance.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise limits to a dictionary.

        Returns:
            Dictionary of all limit fields.
        """
        return {
            "cpu_limit_cores": self.cpu_limit_cores,
            "memory_limit_mb": self.memory_limit_mb,
            "disk_io_limit_mbps": self.disk_io_limit_mbps,
            "max_execution_seconds": self.max_execution_seconds,
            "temp_dir_size_mb": self.temp_dir_size_mb,
            "max_open_files": self.max_open_files,
            "max_processes": self.max_processes,
            "network_enabled": self.network_enabled,
        }


# ---------------------------------------------------------------------------
# Environment Profiles
# ---------------------------------------------------------------------------

_ENVIRONMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "dev": {
        "cpu_limit_cores": 2.0,
        "memory_limit_mb": 1024,
        "disk_io_limit_mbps": 100,
        "max_execution_seconds": 300.0,
        "temp_dir_size_mb": 512,
        "max_open_files": 512,
        "max_processes": 32,
        "network_enabled": True,
    },
    "staging": {
        "cpu_limit_cores": 4.0,
        "memory_limit_mb": 2048,
        "disk_io_limit_mbps": 200,
        "max_execution_seconds": 600.0,
        "temp_dir_size_mb": 1024,
        "max_open_files": 1024,
        "max_processes": 64,
        "network_enabled": False,
    },
    "prod": {
        "cpu_limit_cores": 4.0,
        "memory_limit_mb": 2048,
        "disk_io_limit_mbps": 200,
        "max_execution_seconds": 600.0,
        "temp_dir_size_mb": 1024,
        "max_open_files": 1024,
        "max_processes": 64,
        "network_enabled": False,
    },
}


# ---------------------------------------------------------------------------
# Validation Errors
# ---------------------------------------------------------------------------


class ResourceLimitValidationError(ValueError):
    """Raised when resource limits fail validation.

    Attributes:
        violations: List of human-readable violation descriptions.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            f"Resource limit validation failed: {'; '.join(violations)}"
        )


# ---------------------------------------------------------------------------
# Resource Limit Enforcer
# ---------------------------------------------------------------------------


class ResourceLimitEnforcer:
    """Validates and applies resource limits for sandbox executions.

    Ensures that requested limits are within acceptable ranges and
    applies them to the execution environment where possible.

    Attributes:
        limits: The resource limits to enforce.
    """

    # Absolute maximums to prevent misuse
    _MAX_CPU_CORES: float = 16.0
    _MAX_MEMORY_MB: int = 16384
    _MAX_EXECUTION_S: float = 3600.0
    _MAX_TEMP_DIR_MB: int = 10240

    def __init__(self, limits: ResourceLimits) -> None:
        """Initialize the enforcer with the given limits.

        Args:
            limits: Resource limits to validate and enforce.
        """
        self.limits = limits

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate that all limits are within acceptable ranges.

        Raises:
            ResourceLimitValidationError: If any limits are out of range.
        """
        violations: list[str] = []

        if self.limits.cpu_limit_cores <= 0:
            violations.append("cpu_limit_cores must be > 0")
        elif self.limits.cpu_limit_cores > self._MAX_CPU_CORES:
            violations.append(
                f"cpu_limit_cores {self.limits.cpu_limit_cores} exceeds "
                f"maximum {self._MAX_CPU_CORES}"
            )

        if self.limits.memory_limit_mb <= 0:
            violations.append("memory_limit_mb must be > 0")
        elif self.limits.memory_limit_mb > self._MAX_MEMORY_MB:
            violations.append(
                f"memory_limit_mb {self.limits.memory_limit_mb} exceeds "
                f"maximum {self._MAX_MEMORY_MB}"
            )

        if self.limits.max_execution_seconds <= 0:
            violations.append("max_execution_seconds must be > 0")
        elif self.limits.max_execution_seconds > self._MAX_EXECUTION_S:
            violations.append(
                f"max_execution_seconds {self.limits.max_execution_seconds} exceeds "
                f"maximum {self._MAX_EXECUTION_S}"
            )

        if self.limits.temp_dir_size_mb <= 0:
            violations.append("temp_dir_size_mb must be > 0")
        elif self.limits.temp_dir_size_mb > self._MAX_TEMP_DIR_MB:
            violations.append(
                f"temp_dir_size_mb {self.limits.temp_dir_size_mb} exceeds "
                f"maximum {self._MAX_TEMP_DIR_MB}"
            )

        if self.limits.max_open_files <= 0:
            violations.append("max_open_files must be > 0")

        if self.limits.max_processes <= 0:
            violations.append("max_processes must be > 0")

        if violations:
            raise ResourceLimitValidationError(violations)

        logger.debug(
            "ResourceLimits validated: cpu=%.1f cores, mem=%dMB, time=%.0fs",
            self.limits.cpu_limit_cores,
            self.limits.memory_limit_mb,
            self.limits.max_execution_seconds,
        )

    # ------------------------------------------------------------------
    # Application (platform-specific)
    # ------------------------------------------------------------------

    def apply_to_environment(self, env: Dict[str, str]) -> Dict[str, str]:
        """Add resource limit variables to the subprocess environment.

        Injects GL_SANDBOX_* variables that can be read by the sandboxed
        process for self-enforcement.

        Args:
            env: Existing environment dictionary (mutated in place).

        Returns:
            The modified environment dictionary.
        """
        env["GL_SANDBOX_CPU_CORES"] = str(self.limits.cpu_limit_cores)
        env["GL_SANDBOX_MEMORY_MB"] = str(self.limits.memory_limit_mb)
        env["GL_SANDBOX_MAX_EXEC_S"] = str(self.limits.max_execution_seconds)
        env["GL_SANDBOX_TEMP_SIZE_MB"] = str(self.limits.temp_dir_size_mb)
        env["GL_SANDBOX_MAX_FILES"] = str(self.limits.max_open_files)
        env["GL_SANDBOX_MAX_PROCS"] = str(self.limits.max_processes)
        env["GL_SANDBOX_NETWORK"] = "1" if self.limits.network_enabled else "0"
        return env

    def set_process_limits(self) -> None:
        """Apply resource limits via ``resource.setrlimit`` (Linux/macOS).

        This is typically called inside the sandboxed subprocess before
        handing control to the agent.  It is a **best-effort** mechanism:

        - On Linux: ``RLIMIT_AS``, ``RLIMIT_CPU``, ``RLIMIT_NOFILE``,
          and ``RLIMIT_NPROC`` are set.
        - On macOS: ``RLIMIT_AS`` and ``RLIMIT_NPROC`` may not be
          available; individual failures are logged and skipped.
        - On Windows: The ``resource`` module does not exist; a warning
          is logged and no limits are applied.  Rely on K8s container
          limits for production enforcement.

        Note:
            cgroup-based memory limits are **not** applied here.  In
            dedicated mode, K8s pod resource limits enforce memory and
            CPU ceilings at the container level.
        """
        try:
            import resource as resource_mod
        except ImportError:
            logger.warning(
                "resource module not available (Windows?). "
                "Process limits not enforced -- rely on K8s pod limits "
                "for production."
            )
            return

        applied: list[str] = []

        # Memory limit (soft, hard)
        mem_bytes = self.limits.memory_limit_mb * 1024 * 1024
        try:
            resource_mod.setrlimit(
                resource_mod.RLIMIT_AS, (mem_bytes, mem_bytes)
            )
            applied.append(f"mem={self.limits.memory_limit_mb}MB")
        except (ValueError, AttributeError, OSError) as exc:
            logger.debug("RLIMIT_AS not available: %s", exc)

        # CPU time limit
        cpu_seconds = int(self.limits.max_execution_seconds)
        try:
            resource_mod.setrlimit(
                resource_mod.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 10)
            )
            applied.append(f"cpu={cpu_seconds}s")
        except (ValueError, AttributeError, OSError) as exc:
            logger.debug("RLIMIT_CPU not available: %s", exc)

        # Open files limit
        try:
            resource_mod.setrlimit(
                resource_mod.RLIMIT_NOFILE,
                (self.limits.max_open_files, self.limits.max_open_files),
            )
            applied.append(f"files={self.limits.max_open_files}")
        except (ValueError, AttributeError, OSError) as exc:
            logger.debug("RLIMIT_NOFILE not available: %s", exc)

        # Process limit
        try:
            resource_mod.setrlimit(
                resource_mod.RLIMIT_NPROC,
                (self.limits.max_processes, self.limits.max_processes),
            )
            applied.append(f"procs={self.limits.max_processes}")
        except (ValueError, AttributeError, OSError) as exc:
            logger.debug("RLIMIT_NPROC not available: %s", exc)

        if applied:
            logger.info("Process limits applied: %s", ", ".join(applied))
        else:
            logger.warning(
                "No resource limits could be applied on this platform."
            )


__all__ = [
    "ResourceLimitEnforcer",
    "ResourceLimitValidationError",
    "ResourceLimits",
]
