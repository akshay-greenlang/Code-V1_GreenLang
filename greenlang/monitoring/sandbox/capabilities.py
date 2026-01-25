# -*- coding: utf-8 -*-
"""
Sandbox Capabilities
====================

Defines the capability system for sandboxed execution.
Implements default deny policy with explicit capability grants.
"""

import os
import logging
from enum import Enum, auto
from typing import Set, List, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class Capability(Enum):
    """
    Available capabilities that can be granted to sandboxed code.
    Default policy: DENY ALL - capabilities must be explicitly granted.
    """

    # File system operations
    FILE_READ = auto()  # Read files from allowed paths
    FILE_WRITE = auto()  # Write files to allowed paths
    FILE_CREATE = auto()  # Create new files
    FILE_DELETE = auto()  # Delete existing files

    # Network operations
    NETWORK_HTTP = auto()  # HTTP/HTTPS requests
    NETWORK_SOCKET = auto()  # Raw socket operations
    NETWORK_DNS = auto()  # DNS lookups

    # Process execution
    EXEC_SUBPROCESS = auto()  # Execute subprocesses
    EXEC_SHELL = auto()  # Shell command execution

    # System information
    SYS_ENV_READ = auto()  # Read environment variables
    SYS_ENV_WRITE = auto()  # Modify environment variables
    SYS_INFO = auto()  # Access system information

    # Resource access
    MEMORY_UNLIMITED = auto()  # Unlimited memory usage
    CPU_INTENSIVE = auto()  # CPU-intensive operations

    # External integrations
    CLOUD_AWS = auto()  # AWS services access
    CLOUD_GCP = auto()  # Google Cloud services access
    CLOUD_AZURE = auto()  # Azure services access

    # Database operations
    DB_READ = auto()  # Database read operations
    DB_WRITE = auto()  # Database write operations
    DB_ADMIN = auto()  # Database administration

    # Cryptographic operations
    CRYPTO_SIGN = auto()  # Digital signing
    CRYPTO_ENCRYPT = auto()  # Encryption operations
    CRYPTO_RANDOM = auto()  # Secure random generation


@dataclass
class CapabilityPolicy:
    """
    Defines a capability policy with allowed capabilities and restrictions.
    """

    # Explicitly granted capabilities
    allowed_capabilities: Set[Capability] = field(default_factory=set)

    # Path restrictions for file operations
    allowed_read_paths: List[str] = field(default_factory=list)
    allowed_write_paths: List[str] = field(default_factory=list)

    # Network restrictions
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_cpu_time_seconds: Optional[int] = None
    max_file_size_mb: Optional[int] = None

    # Environment variable restrictions
    allowed_env_vars: List[str] = field(default_factory=list)
    blocked_env_vars: List[str] = field(
        default_factory=lambda: [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "AZURE_CLIENT_SECRET",
        ]
    )

    def __post_init__(self):
        """Validate policy configuration"""
        # Ensure file paths are normalized
        self.allowed_read_paths = [os.path.normpath(p) for p in self.allowed_read_paths]
        self.allowed_write_paths = [
            os.path.normpath(p) for p in self.allowed_write_paths
        ]

    def has_capability(self, capability: Capability) -> bool:
        """Check if policy grants a specific capability"""
        return capability in self.allowed_capabilities

    def can_read_path(self, path: str) -> bool:
        """Check if policy allows reading from a specific path"""
        if not self.has_capability(Capability.FILE_READ):
            return False

        if not self.allowed_read_paths:
            return False

        normalized_path = os.path.normpath(path)
        for allowed_path in self.allowed_read_paths:
            if normalized_path.startswith(allowed_path):
                return True
        return False

    def can_write_path(self, path: str) -> bool:
        """Check if policy allows writing to a specific path"""
        if not self.has_capability(Capability.FILE_WRITE):
            return False

        if not self.allowed_write_paths:
            return False

        normalized_path = os.path.normpath(path)
        for allowed_path in self.allowed_write_paths:
            if normalized_path.startswith(allowed_path):
                return True
        return False

    def can_access_host(self, host: str) -> bool:
        """Check if policy allows network access to a specific host"""
        if not self.has_capability(Capability.NETWORK_HTTP):
            return False

        if not self.allowed_hosts:
            return False

        return host in self.allowed_hosts or "*" in self.allowed_hosts

    def can_access_env_var(self, var_name: str) -> bool:
        """Check if policy allows access to an environment variable"""
        if var_name in self.blocked_env_vars:
            return False

        if not self.allowed_env_vars:
            return self.has_capability(Capability.SYS_ENV_READ)

        return var_name in self.allowed_env_vars


class CapabilityValidator:
    """
    Validates and enforces capability policies during execution.
    """

    def __init__(self, policy: CapabilityPolicy):
        self.policy = policy
        self.violations: List[Dict[str, Any]] = []

    def check_capability(
        self, capability: Capability, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a capability is allowed and log violations.

        Args:
            capability: The capability to check
            context: Additional context for the capability check

        Returns:
            True if capability is allowed, False otherwise
        """
        context = context or {}

        # Basic capability check
        if not self.policy.has_capability(capability):
            self._log_violation(capability, "Capability not granted", context)
            return False

        # Additional context-specific checks
        if capability == Capability.FILE_READ and "path" in context:
            if not self.policy.can_read_path(context["path"]):
                self._log_violation(
                    capability, f"Path not allowed: {context['path']}", context
                )
                return False

        elif capability == Capability.FILE_WRITE and "path" in context:
            if not self.policy.can_write_path(context["path"]):
                self._log_violation(
                    capability, f"Path not allowed: {context['path']}", context
                )
                return False

        elif capability == Capability.NETWORK_HTTP and "host" in context:
            if not self.policy.can_access_host(context["host"]):
                self._log_violation(
                    capability, f"Host not allowed: {context['host']}", context
                )
                return False

        elif capability == Capability.SYS_ENV_READ and "var_name" in context:
            if not self.policy.can_access_env_var(context["var_name"]):
                self._log_violation(
                    capability,
                    f"Environment variable not allowed: {context['var_name']}",
                    context,
                )
                return False

        return True

    def _log_violation(
        self, capability: Capability, reason: str, context: Dict[str, Any]
    ):
        """Log a capability violation"""
        violation = {
            "capability": capability.name,
            "reason": reason,
            "context": context,
            "timestamp": (
                logger.handlers[0].formatter.formatTime(
                    logger.makeRecord(logger.name, logging.INFO, "", 0, "", (), None)
                )
                if logger.handlers
                else None
            ),
        }
        self.violations.append(violation)
        logger.warning(f"Capability violation: {capability.name} - {reason}")

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get all recorded violations"""
        return self.violations.copy()

    def clear_violations(self):
        """Clear all recorded violations"""
        self.violations.clear()


# Predefined policies for common use cases
DEFAULT_POLICY = CapabilityPolicy()  # Empty - denies everything

BASIC_COMPUTE_POLICY = CapabilityPolicy(
    allowed_capabilities={Capability.MEMORY_UNLIMITED, Capability.CRYPTO_RANDOM}
)

DATA_PROCESSING_POLICY = CapabilityPolicy(
    allowed_capabilities={
        Capability.FILE_READ,
        Capability.MEMORY_UNLIMITED,
        Capability.CPU_INTENSIVE,
        Capability.CRYPTO_RANDOM,
    },
    allowed_read_paths=["/tmp", "/var/tmp"],
    max_memory_mb=1024,
    max_cpu_time_seconds=300,
)

NETWORK_CLIENT_POLICY = CapabilityPolicy(
    allowed_capabilities={
        Capability.NETWORK_HTTP,
        Capability.NETWORK_DNS,
        Capability.MEMORY_UNLIMITED,
        Capability.CRYPTO_RANDOM,
    },
    allowed_hosts=["*"],  # Allow all hosts (be careful with this in production)
    max_memory_mb=512,
)

FULL_ACCESS_POLICY = CapabilityPolicy(
    allowed_capabilities=set(Capability),  # All capabilities
    allowed_read_paths=["/"],
    allowed_write_paths=["/tmp", "/var/tmp"],
    allowed_hosts=["*"],
    allowed_env_vars=["*"],
)
